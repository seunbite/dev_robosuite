#!/usr/bin/env python3
"""Check that DROID RobotEnv talks to the FR3, then nudge one joint ~2°.

Copy next to replay_droid.py on the robot laptop.

    python check_droid_env.py              # import + obs + tiny nudge
    python check_droid_env.py --check-only # no motion
    python check_droid_env.py --slow 0.1   # default: 10x slower nudge and return
"""
from __future__ import annotations

import argparse
import time
import traceback

import numpy as np

_Q_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
_Q_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def _clip(q: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(q, dtype=np.float64).reshape(-1)[:7], _Q_LO, _Q_HI)


def _unwrap_obs(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    return obs


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


def _gripper_from_obs(obs: dict) -> float:
    rs = obs if "gripper_position" in obs else (obs.get("robot_state") or obs.get("observation") or {})
    g = rs.get("gripper_position", obs.get("gripper_position", 0.0))
    return float(np.asarray(g).reshape(-1)[0])


def _read_obs(env):
    if hasattr(env, "get_observation"):
        return _unwrap_obs(env.get_observation())
    if hasattr(env, "get_state"):
        return _unwrap_obs(env.get_state())
    return _unwrap_obs(env.reset())


def _step(env, q: np.ndarray, gripper: float) -> None:
    env.step(np.concatenate([_clip(q), [float(gripper)]]))


def _lerp(env, q0, q1, gripper: float, seconds: float, hz: float) -> None:
    n = max(1, int(round(seconds * hz)))
    dt = 1.0 / hz
    q0 = _clip(q0)
    q1 = _clip(q1)
    for i in range(1, n + 1):
        u = i / n
        t0 = time.time()
        _step(env, (1.0 - u) * q0 + u * q1, gripper)
        leftover = dt - (time.time() - t0)
        if leftover > 0:
            time.sleep(leftover)


def check_import() -> None:
    print("[1] import droid.robot_env.RobotEnv")
    try:
        from droid.robot_env import RobotEnv  # noqa: F401
    except Exception:
        traceback.print_exc()
        raise SystemExit("FAIL: droid is not importable in this env")
    print("    ok")


def build_env(hz: float):
    from droid.robot_env import RobotEnv

    print("[2] RobotEnv(action_space=joint_position)")
    kwargs = dict(action_space="joint_position", gripper_action_space="position")
    try:
        env = RobotEnv(control_hz=int(round(hz)), **kwargs)
    except TypeError:
        env = RobotEnv(**kwargs)
    print("    ok", type(env))
    return env


def check_obs(env) -> tuple[np.ndarray, float]:
    print("[3] read observation")
    obs = _read_obs(env)
    if not isinstance(obs, dict):
        raise SystemExit(f"FAIL: obs type {type(obs)}")
    print("    keys", sorted(obs.keys())[:24])
    q = _joints_from_obs(obs)
    g = _gripper_from_obs(obs)
    print("    q_rad", np.round(q, 4).tolist())
    print("    q_deg", np.round(np.degrees(q), 2).tolist())
    print("    gripper", g)
    return q, g


def nudge(
    env,
    q0: np.ndarray,
    gripper: float,
    *,
    joint: int,
    deg: float,
    hz: float,
    slow: float,
) -> None:
    seconds = 2.0 / slow
    print(
        f"[4] nudge joint {joint} by {deg} deg and back "
        f"(slow={slow:g}, {seconds:g}s each way)"
    )
    delta = np.zeros(7, dtype=np.float64)
    delta[int(joint)] = np.deg2rad(float(deg))
    q1 = _clip(q0 + delta)
    _lerp(env, q0, q1, gripper, seconds=seconds, hz=hz)
    _lerp(env, q1, q0, gripper, seconds=seconds, hz=hz)
    q_end = _joints_from_obs(_read_obs(env))
    print("    back q_deg", np.round(np.degrees(q_end), 2).tolist())
    print("    ok")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--joint", type=int, default=6, help="0-6, default wrist")
    p.add_argument("--deg", type=float, default=2.0)
    p.add_argument("--hz", type=float, default=15.0)
    p.add_argument("--slow", type=float, default=0.1, help="speed multiplier; 0.1 is 10x slower")
    args = p.parse_args()
    if not (0 <= args.joint <= 6):
        raise SystemExit("--joint must be 0..6")
    if args.slow <= 0:
        raise SystemExit("--slow must be > 0")
    check_import()
    env = build_env(args.hz)
    q0, g = check_obs(env)
    if args.check_only:
        print("check-only: no motion")
        return
    nudge(env, q0, g, joint=args.joint, deg=args.deg, hz=args.hz, slow=args.slow)
    print("ready for replay_droid.py")


if __name__ == "__main__":
    main()
