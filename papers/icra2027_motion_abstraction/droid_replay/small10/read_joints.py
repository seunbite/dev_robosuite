#!/usr/bin/env python3
"""Read current FR3 joints. Does not command motion.

    python read_joints.py
    python read_joints.py --write
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_DEFAULT_OUT = Path(__file__).resolve().with_name("neutral_joints.json")


def _unwrap(obs):
    return obs[0] if isinstance(obs, tuple) else obs


def _joints(obs: dict) -> np.ndarray:
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


def _gripper(obs: dict) -> float:
    rs = obs if "gripper_position" in obs else (obs.get("robot_state") or obs.get("observation") or {})
    g = rs.get("gripper_position", obs.get("gripper_position", 0.0))
    return float(np.asarray(g).reshape(-1)[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--write",
        nargs="?",
        const=str(_DEFAULT_OUT),
        default=None,
        metavar="PATH",
        help=f"save JSON (default: {_DEFAULT_OUT.name})",
    )
    args = p.parse_args()

    from droid.robot_env import RobotEnv

    kwargs = dict(action_space="joint_position", gripper_action_space="position")
    try:
        env = RobotEnv(control_hz=15, **kwargs)
    except TypeError:
        env = RobotEnv(**kwargs)

    if hasattr(env, "get_observation"):
        obs = _unwrap(env.get_observation())
    elif hasattr(env, "get_state"):
        obs = _unwrap(env.get_state())
    else:
        raise SystemExit("no get_observation/get_state; refusing reset() so the arm does not move")

    q = _joints(obs)
    g = _gripper(obs)
    q_deg = np.degrees(q)
    print("q_rad", np.round(q, 6).tolist())
    print("q_deg", np.round(q_deg, 2).tolist())
    print("gripper", g)
    print("copy", " ".join(f"{x:.2f}" for x in q_deg))

    if args.write is not None:
        out = Path(args.write).expanduser()
        if not out.is_absolute():
            out = Path.cwd() / out
        payload = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "q_rad": q.tolist(),
            "q_deg": q_deg.tolist(),
            "gripper": g,
        }
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print("wrote", out)



if __name__ == "__main__":
    main()
