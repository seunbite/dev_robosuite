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


def _nested_get(obs: dict, *keys):
    blobs = [obs]
    for nest in ("robot_state", "observation", "state"):
        inner = obs.get(nest)
        if isinstance(inner, dict):
            blobs.append(inner)
    for blob in blobs:
        for key in keys:
            if key in blob:
                return blob[key]
    return None


def _R_from_euler_xyz(rpy: np.ndarray) -> np.ndarray:
    rx, ry, rz = [float(v) for v in np.asarray(rpy, dtype=np.float64).reshape(-1)[:3]]
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz_m @ ry_m @ rx_m


def _axis_clock(axis: np.ndarray) -> tuple[float, float]:
    """Elevation from the floor plane, and yaw of the floor projection (world +X = 0°)."""
    u = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(u))
    if n < 1e-9:
        return float("nan"), float("nan")
    u = u / n
    elev = float(np.degrees(np.arcsin(np.clip(u[2], -1.0, 1.0))))
    xy = u[:2]
    nxy = float(np.linalg.norm(xy))
    yaw = float(np.degrees(np.arctan2(xy[1], xy[0]))) if nxy > 1e-9 else float("nan")
    return elev, yaw


def _print_tool_clock(obs: dict, q: np.ndarray) -> None:
    """Wrist/flange orientation about the tool axis vs the floor.

    DROID ``cartesian_position`` is Franka flange: xyz at the gripper-mount
    origin (sim ``right_hand``) and euler of flange axes (sim ``link7``).
    Sim DroidRobotiq85Gripper is +45° about tool Z to cancel Panda
    ``right_hand`` -45°, so the 2F-85 opening follows flange Y.
    """
    q7 = float(np.degrees(q[6]))
    print(f"    q7 (wrist about tool) {q7:.2f} deg")
    print("    sim gripper clock vs flange  +45 deg about tool Z (cancels Panda right_hand -45)")
    cart = _nested_get(
        obs,
        "cartesian_position",
        "cartesian_pose",
        "ee_position",
        "ee_pose",
        "tcp_pose",
    )
    if cart is None:
        print("    cartesian_position: not in obs (cannot print live flange clock)")
        return
    cart = np.asarray(cart, dtype=np.float64).reshape(-1)
    if cart.size < 6:
        print(f"    cartesian_position size={cart.size}, expected 6 (xyz+rpy)")
        return
    xyz, rpy = cart[:3], cart[3:6]
    print("    cartesian xyz_m", np.round(xyz, 4).tolist())
    print("    cartesian rpy_rad", np.round(rpy, 4).tolist())
    print("    cartesian rpy_deg", np.round(np.degrees(rpy), 2).tolist())
    R = _R_from_euler_xyz(rpy)
    tool = R[:, 2]
    t_elev, _ = _axis_clock(tool)
    print(f"    tool axis (flange Z) world {np.round(tool, 3).tolist()}  elev_from_horiz {t_elev:.1f} deg")
    for name, col in (("flange X", 0), ("flange Y", 1)):
        elev, yaw = _axis_clock(R[:, col])
        print(
            f"    {name}  world {np.round(R[:, col], 3).tolist()}  "
            f"elev_from_horiz {elev:+.1f} deg  yaw_in_floor {yaw:+.1f} deg"
        )
    # 2F-85 opening follows flange Y after the sim +45° / Panda -45° cancel.
    y_elev, y_yaw = _axis_clock(R[:, 1])
    print(
        f"    jaw-span (flange Y)  "
        f"elev {y_elev:+.1f} deg  floor-yaw {y_yaw:+.1f} deg"
    )
    print("    floor-yaw 0=world +X, ±90=±Y (frontview left-right). 45° = diagonal on the table.")


def _read_obs(env):
    if hasattr(env, "get_observation"):
        return _unwrap_obs(env.get_observation())
    if hasattr(env, "get_state"):
        return _unwrap_obs(env.get_state())
    return _unwrap_obs(env.reset())


def _step(env, q: np.ndarray, gripper: float) -> None:
    action = np.concatenate([_clip(q), [float(gripper)]])
    if hasattr(env, "update_robot"):
        env.update_robot(
            action,
            action_space="joint_position",
            gripper_action_space="position",
            blocking=True,
        )
        return
    env.step(action)


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

    print("[2] RobotEnv(action_space=joint_position, do_reset=False)")
    kwargs = dict(
        action_space="joint_position",
        gripper_action_space="position",
        do_reset=False,
    )
    try:
        env = RobotEnv(control_hz=int(round(hz)), **kwargs)
    except TypeError:
        kwargs.pop("do_reset", None)
        try:
            env = RobotEnv(control_hz=int(round(hz)), **kwargs)
        except TypeError:
            env = RobotEnv(**kwargs)
    print("    ok", type(env), "(no DROID home yank)")
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
    _print_tool_clock(obs, q)
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
