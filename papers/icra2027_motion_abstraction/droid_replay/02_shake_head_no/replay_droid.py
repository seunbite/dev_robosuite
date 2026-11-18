#!/usr/bin/env python3
"""Go to a dumped cue pose, optionally play the rest, on a DROID FR3.

    # current joints → first gesture pose only
    python replay_droid.py --traj 02_shake_head_no_llm.npz --pose --slow 0.1

    # pose then play the cue
    python replay_droid.py --traj 02_shake_head_no_llm.npz --pose --play --slow 0.1

``--slow 1`` ≈ 30 °/s on the approach and dump-clock playback.
``--slow 0.1`` is 10× slower. Skips the baked sim-home prefix in the npz.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

_FR3_Q_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
_FR3_Q_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
_HOME = np.array(
    [0.0, np.pi / 16.0, 0.0, -np.pi / 2.0 - np.pi / 3.0, 0.0, np.pi - 0.2, np.pi / 4],
    dtype=np.float64,
)
# Approach rate at --slow 1.
_QDOT_AT_SLOW_1 = np.deg2rad(30.0)


def _clip_q(q: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(q, dtype=np.float64).reshape(-1)[:7], _FR3_Q_LO, _FR3_Q_HI)


def _load(path: Path) -> dict:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        pose_frame = data.get("pose_frame")
        return {
            "hz": float(data["hz"]),
            "joints_rad": np.asarray(data["joints_rad"], dtype=np.float64),
            "gripper": np.asarray(data["gripper"], dtype=np.float64),
            "cue": data.get("cue", path.stem),
            "robot": data.get("robot", "FR3"),
            "hardware": data.get("hardware") or {},
            "pose_frame": None if pose_frame is None else int(pose_frame),
        }
    z = np.load(path, allow_pickle=True)
    hz = float(np.asarray(z["hz"]).reshape(-1)[0])
    robot = "FR3"
    if "robot" in z.files:
        robot = str(np.asarray(z["robot"]).reshape(-1)[0])
    pose_frame = None
    if "pose_frame" in z.files:
        pose_frame = int(np.asarray(z["pose_frame"]).reshape(-1)[0])
    return {
        "hz": hz,
        "joints_rad": np.asarray(z["joints_rad"], dtype=np.float64),
        "gripper": np.asarray(z["gripper"], dtype=np.float64),
        "cue": path.stem,
        "robot": robot,
        "hardware": {},
        "pose_frame": pose_frame,
    }


def _first_pose_index(q: np.ndarray, stored: int | None) -> int:
    if stored is not None and 0 <= int(stored) < len(q):
        return int(stored)
    if len(q) < 4:
        return 0
    home_err = float(np.max(np.abs(q[0, :7] - _HOME[: q.shape[1]])))
    if home_err > np.deg2rad(20.0):
        return 0
    dq = np.max(np.abs(np.diff(q[:, :7], axis=0)), axis=1)
    still = np.deg2rad(0.4)
    moving = dq > still
    i = 0
    while i < len(moving) and not moving[i]:
        i += 1
    if i >= len(moving):
        return 0
    while i < len(moving) and moving[i]:
        i += 1
    return min(i, len(q) - 1)


def _joints_from_obs(obs: dict) -> np.ndarray:
    if obs is None:
        raise RuntimeError("empty observation")
    for key in ("joint_positions", "joint_position"):
        if key in obs:
            return np.asarray(obs[key], dtype=np.float64).reshape(-1)[:7]
    rs = obs.get("robot_state") or obs.get("observation") or {}
    for key in ("joint_positions", "joint_position"):
        if key in rs:
            return np.asarray(rs[key], dtype=np.float64).reshape(-1)[:7]
    raise RuntimeError(f"could not find 7-DoF joints in obs keys={list(obs)}")


def _build_env(control_hz: float):
    from droid.robot_env import RobotEnv  # noqa: WPS433

    kwargs = dict(
        action_space="joint_position",
        gripper_action_space="position",
    )
    try:
        return RobotEnv(control_hz=int(round(control_hz)), **kwargs)
    except TypeError:
        return RobotEnv(**kwargs)


def _current_q(env) -> np.ndarray:
    obs = env.get_observation() if hasattr(env, "get_observation") else env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    return _clip_q(_joints_from_obs(obs))


def _lerp_to(env, q_goal: np.ndarray, gripper: float, *, slow: float, hz: float) -> None:
    q0 = _current_q(env)
    q_goal = _clip_q(q_goal)
    max_dq = float(np.max(np.abs(q_goal - q0)))
    qdot = _QDOT_AT_SLOW_1 * max(float(slow), 1e-4)
    duration = max(max_dq / max(qdot, 1e-6), 0.5)
    n = max(2, int(round(duration * hz)))
    dt = 1.0 / hz
    print(
        f"pose goto max_|Δq|={np.degrees(max_dq):.1f}°  "
        f"{np.degrees(qdot):.1f}°/s  {duration:.1f}s  ({n} steps)"
    )
    for i in range(1, n + 1):
        u = i / n
        t0 = time.time()
        q = _clip_q((1.0 - u) * q0 + u * q_goal)
        env.step(np.concatenate([q, [gripper]]))
        leftover = dt - (time.time() - t0)
        if leftover > 0:
            time.sleep(leftover)


def replay(
    traj: dict,
    *,
    slow: float,
    do_pose: bool,
    do_play: bool,
    dry_run: bool,
) -> None:
    q = traj["joints_rad"]
    g = traj["gripper"]
    if q.ndim != 2 or q.shape[1] < 7:
        raise SystemExit(f"expected (N, >=7) joints, got {q.shape}")
    hz_src = float(traj["hz"])
    pose_i = _first_pose_index(q, traj.get("pose_frame"))
    q_pose = q[pose_i, :7]
    g0 = float(g[pose_i]) if pose_i < len(g) else (float(g[0]) if len(g) else 0.0)
    play_n = q.shape[0] - pose_i
    play_s = play_n / max(hz_src * float(slow), 1e-6)
    print(
        f"cue={traj['cue']} robot={traj.get('robot')} frames={q.shape[0]} "
        f"pose_frame={pose_i}/{q.shape[0]} dump_hz={hz_src} slow={slow}"
    )
    print(f"  q_pose_deg={np.round(np.degrees(q_pose), 1).tolist()}")
    print(f"  play remainder {play_n} frames → {play_s:.1f}s at this --slow")
    if dry_run:
        print("dry-run: not contacting the robot")
        return
    if not do_pose and not do_play:
        raise SystemExit("pass --pose and/or --play")

    env = _build_env(hz_src)
    if do_pose or do_play:
        _lerp_to(env, q_pose, g0, slow=slow, hz=hz_src)
        print("at first cue pose")
    if not do_play:
        print("pose only (--play to continue)")
        return
    dt = 1.0 / max(hz_src * float(slow), 1e-6)
    print("playing cue")
    for i in range(pose_i, q.shape[0]):
        t0 = time.time()
        gi = float(g[i]) if i < len(g) else g0
        env.step(np.concatenate([_clip_q(q[i, :7]), [gi]]))
        leftover = dt - (time.time() - t0)
        if leftover > 0:
            time.sleep(leftover)
    print("done")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--traj", type=Path, required=True)
    p.add_argument("--pose", action="store_true", help="current joints → first cue pose")
    p.add_argument("--play", action="store_true", help="after pose, play the rest of the npz")
    p.add_argument("--slow", type=float, default=0.1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not args.pose and not args.play and not args.dry_run:
        raise SystemExit("need --pose and/or --play (or --dry-run)")
    traj = _load(args.traj)
    do_pose = bool(args.pose or args.play)
    replay(
        traj,
        slow=args.slow,
        do_pose=do_pose,
        do_play=bool(args.play),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
