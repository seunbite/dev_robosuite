#!/usr/bin/env python3
"""Go to a dumped cue pose, optionally play the rest, on a DROID FR3.

    python replay_droid.py --traj 02_shake_head_no_llm.npz --pose --slow 0.1
    python replay_droid.py --traj 02_shake_head_no_llm.npz --pose --play --slow 0.1

Dumps store q[0] as the first gesture pose (no sim-home). ``--pose`` uses the
same smoothstep joint lerp as the GIF approach. No lift / side / back detour.
``--play`` after ``--pose`` only nudges if needed, then plays.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from droid_traj import HOME as _HOME
from droid_traj import cue_start_index, strip_to_cue

_FR3_Q_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
_FR3_Q_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
_QDOT_AT_SLOW_1 = np.deg2rad(30.0)
_AT_POSE_DEG = 4.0
_NUDGE_DEG = 12.0


def _clip_q(q: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(q, dtype=np.float64).reshape(-1)[:7], _FR3_Q_LO, _FR3_Q_HI)


def _far(a: np.ndarray, b: np.ndarray, deg: float = 6.0) -> bool:
    return float(np.max(np.abs(_clip_q(a) - _clip_q(b)))) > np.deg2rad(deg)


def _max_abs_dq(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(_clip_q(a) - _clip_q(b))))


def _smoothstep01(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def labeled_pose_path(q0: np.ndarray, q_goal: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Current joints → cue pose. Same single segment the GIF approach used."""
    q0 = _clip_q(q0)
    goal = _clip_q(q_goal)
    if not _far(q0, goal, _AT_POSE_DEG):
        return []
    return [("cue pose", goal)]


def plan_pose_path(q0: np.ndarray, q_goal: np.ndarray) -> list[np.ndarray]:
    return [q for _name, q in labeled_pose_path(q0, q_goal)]


def _load(path: Path) -> dict:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        stored = data.get("pose_frame")
        q, g, _ee, pose_frame = strip_to_cue(
            np.asarray(data["joints_rad"], dtype=np.float64),
            np.asarray(data["gripper"], dtype=np.float64),
            None,
            None if stored is None else int(stored),
        )
        return {
            "hz": float(data["hz"]),
            "joints_rad": q,
            "gripper": g,
            "cue": data.get("cue", path.stem),
            "robot": data.get("robot", "FR3"),
            "hardware": data.get("hardware") or {},
            "pose_frame": pose_frame,
        }
    z = np.load(path, allow_pickle=True)
    hz = float(np.asarray(z["hz"]).reshape(-1)[0])
    robot = "FR3"
    if "robot" in z.files:
        robot = str(np.asarray(z["robot"]).reshape(-1)[0])
    stored = None
    if "pose_frame" in z.files:
        stored = int(np.asarray(z["pose_frame"]).reshape(-1)[0])
    q, g, _ee, pose_frame = strip_to_cue(
        np.asarray(z["joints_rad"], dtype=np.float64),
        np.asarray(z["gripper"], dtype=np.float64) if "gripper" in z.files else None,
        None,
        stored,
    )
    return {
        "hz": hz,
        "joints_rad": q,
        "gripper": g,
        "cue": path.stem,
        "robot": robot,
        "hardware": {},
        "pose_frame": pose_frame,
    }


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


def _lerp_to(
    env,
    q_goal: np.ndarray,
    gripper: float,
    *,
    slow: float,
    hz: float,
    q0: np.ndarray | None = None,
    min_s: float = 0.4,
) -> np.ndarray:
    q0 = _clip_q(q0 if q0 is not None else _current_q(env))
    q_goal = _clip_q(q_goal)
    max_dq = _max_abs_dq(q0, q_goal)
    qdot = _QDOT_AT_SLOW_1 * max(float(slow), 1e-4)
    duration = max(max_dq / max(qdot, 1e-6), min_s)
    n = max(2, int(round(duration * hz)))
    dt = 1.0 / hz
    print(
        f"  smoothstep max_|Δq|={np.degrees(max_dq):.1f}°  "
        f"{np.degrees(qdot):.1f}°/s  {duration:.1f}s  ({n} steps)"
    )
    q = q0
    for i in range(1, n + 1):
        u = _smoothstep01(i / n)
        t0 = time.time()
        q = _clip_q((1.0 - u) * q0 + u * q_goal)
        env.step(np.concatenate([q, [gripper]]))
        leftover = dt - (time.time() - t0)
        if leftover > 0:
            time.sleep(leftover)
    return q


def _goto_pose(env, q_goal: np.ndarray, gripper: float, *, slow: float, hz: float) -> np.ndarray:
    q0 = _current_q(env)
    goal = _clip_q(q_goal)
    max_dq = _max_abs_dq(q0, goal)
    max_deg = float(np.degrees(max_dq))
    print(f"  start_deg {np.round(np.degrees(q0), 1).tolist()}")
    print(f"  goal_deg  {np.round(np.degrees(goal), 1).tolist()}")
    if max_deg <= _AT_POSE_DEG:
        print(f"already at cue pose (max |Δq|={max_deg:.1f}°); skip approach")
        return goal
    if max_deg <= _NUDGE_DEG:
        print(f"pose path (nudge → cue)  max |Δq|={max_deg:.1f}°")
        _lerp_to(env, goal, gripper, slow=slow, hz=hz, q0=q0, min_s=0.15)
        return goal
    print(f"pose path (current → cue, GIF lerp)  max |Δq|={max_deg:.1f}°")
    _lerp_to(env, goal, gripper, slow=slow, hz=hz, q0=q0)
    return goal


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
    pose_i = int(traj.get("pose_frame") or 0)
    pose_i = cue_start_index(q, pose_i)
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
        max_deg = float(np.degrees(_max_abs_dq(_HOME, q_pose)))
        print("dry-run from sim-home (real robot uses current joints):")
        print(f"  start_deg {np.round(np.degrees(_clip_q(_HOME)), 1).tolist()}")
        print(f"  goal_deg  {np.round(np.degrees(_clip_q(q_pose)), 1).tolist()}")
        if max_deg <= _AT_POSE_DEG:
            print(f"  already at cue (max |Δq|={max_deg:.1f}°); --play would start immediately")
        elif max_deg <= _NUDGE_DEG:
            print(f"  nudge only (max |Δq|={max_deg:.1f}°)")
        else:
            print(f"  one smoothstep lerp current → cue (max |Δq|={max_deg:.1f}°)")
        print("dry-run: not contacting the robot")
        return
    if not do_pose and not do_play:
        raise SystemExit("pass --pose and/or --play")

    env = _build_env(hz_src)
    if do_pose or do_play:
        _goto_pose(env, q_pose, g0, slow=slow, hz=hz_src)
        print("at cue pose (npz start)")
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
    p.add_argument("--pose", action="store_true", help="smoothstep lerp current joints to npz[0]")
    p.add_argument("--play", action="store_true", help="nudge to cue if needed, then play the npz")
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
