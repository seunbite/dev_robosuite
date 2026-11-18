#!/usr/bin/env python3
"""Trace the ±15 cm EE cube on a DROID FR3 (9 edges).

    # plan on a machine with robosuite/mujoco (this Mac), then copy the npz
    python replay_cube.py --speed 0.1 --x -15 --y 10 --save cube.npz --dry-run

    # robot laptop
    python replay_cube.py --speed 0.1 --play --traj cube.npz

    # same machine has both DROID and robosuite
    python replay_cube.py --speed 0.1 --x -15 --y 10 --play

``--speed 1`` ≈ 15 cm/s along each 30 cm edge and 30 °/s approach.
Default cube is sim-reset EE raised 40 cm in Z (center ≈ 0.457, 0, 1.411).
``--x --y --z`` are extra centimetre offsets from that. Gallery: +X forward, +Y left, +Z up.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

_FR3_Q_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
_FR3_Q_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
_EE_NEUTRAL = np.array([0.457, 0.0, 1.011], dtype=np.float64)
_CUBE_LIFT_Z_CM = 40.0
_HALF_M = 0.15
_HZ = 15
_SAMPLES = 18
_QDOT_AT_SPEED_1 = np.deg2rad(30.0)
_V_AT_SPEED_1 = 0.15  # m/s along cube edges


def _clip_q(q: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(q, dtype=np.float64).reshape(-1)[:7], _FR3_Q_LO, _FR3_Q_HI)


def cube_center(*, x_cm: float = 0.0, y_cm: float = 0.0, z_cm: float = 0.0) -> np.ndarray:
    return _EE_NEUTRAL + np.array(
        [x_cm, y_cm, z_cm + _CUBE_LIFT_Z_CM], dtype=np.float64
    ) * 0.01


def cube_corners(center: np.ndarray, half_m: float = _HALF_M) -> dict[str, list[np.ndarray]]:
    c = np.asarray(center, dtype=np.float64)
    h = float(half_m)
    xf, xb = c[0] + h, c[0] - h
    ym, yp = c[1] - h, c[1] + h
    zm, zp = c[2] - h, c[2] + h
    front = [
        np.array([xf, ym, zm]),
        np.array([xf, yp, zm]),
        np.array([xf, yp, zp]),
        np.array([xf, ym, zp]),
    ]
    back = [
        np.array([xb, ym, zm]),
        np.array([xb, yp, zm]),
        np.array([xb, yp, zp]),
        np.array([xb, ym, zp]),
    ]
    return {"front": front, "back": back}


def _edge_points(a: np.ndarray, b: np.ndarray, n: int, *, include_start: bool) -> list[np.ndarray]:
    i0 = 0 if include_start else 1
    return [(1.0 - k / n) * a + (k / n) * b for k in range(i0, n + 1)]


def nine_edge_path(cube: dict[str, list[np.ndarray]], n: int) -> list[tuple[str, np.ndarray]]:
    """Front 4 → one −X 30 cm edge from the front start/end corner → back 4."""
    front, back = cube["front"], cube["back"]
    path: list[tuple[str, np.ndarray]] = []
    for i in range(4):
        path.extend(
            ("front", p)
            for p in _edge_points(front[i], front[(i + 1) % 4], n, include_start=(i == 0))
        )
    path.extend(("depth", p) for p in _edge_points(front[0], back[0], n, include_start=False))
    for i in range(4):
        path.extend(
            ("back", p) for p in _edge_points(back[i], back[(i + 1) % 4], n, include_start=False)
        )
    return path


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

    kwargs = dict(action_space="joint_position", gripper_action_space="position")
    try:
        return RobotEnv(control_hz=int(round(control_hz)), **kwargs)
    except TypeError:
        return RobotEnv(**kwargs)


def _current_q(env) -> np.ndarray:
    obs = env.get_observation() if hasattr(env, "get_observation") else env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    return _clip_q(_joints_from_obs(obs))


def _lerp_q(env, q0: np.ndarray, q1: np.ndarray, gripper: float, *, speed: float, hz: float) -> np.ndarray:
    q0, q1 = _clip_q(q0), _clip_q(q1)
    max_dq = float(np.max(np.abs(q1 - q0)))
    qdot = _QDOT_AT_SPEED_1 * max(float(speed), 1e-4)
    duration = max(max_dq / max(qdot, 1e-6), 0.4)
    n = max(2, int(round(duration * hz)))
    dt = 1.0 / hz
    print(f"  approach max_|Δq|={np.degrees(max_dq):.1f}°  {np.degrees(qdot):.1f}°/s  {duration:.1f}s")
    q = q0
    for i in range(1, n + 1):
        t0 = time.time()
        q = _clip_q((1.0 - i / n) * q0 + (i / n) * q1)
        env.step(np.concatenate([q, [gripper]]))
        leftover = dt - (time.time() - t0)
        if leftover > 0:
            time.sleep(leftover)
    return q


def plan_cube_joints(
    center: np.ndarray,
    *,
    half_m: float = _HALF_M,
    samples: int = _SAMPLES,
) -> dict:
    import sys
    from pathlib import Path as _P

    import mujoco

    _here = str(_P(__file__).resolve().parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from render_fr3_neutral import _env

    cube = cube_corners(center, half_m)
    path = nine_edge_path(cube, samples)
    env = _env()
    robot = env.robots[0]
    names = list(robot.robot_model.joints)[:7]
    dof_ids = np.array([int(env.sim.model.jnt_dofadr[env.sim.model.joint_name2id(n)]) for n in names])
    arm = "right" if "right" in robot.eef_site_id else next(iter(robot.eef_site_id))
    sid = int(robot.eef_site_id[arm])
    q_bias = np.asarray(robot.robot_model.init_qpos, dtype=np.float64).reshape(-1)[:7]
    robot.set_robot_joint_positions(_clip_q(q_bias))
    env.sim.forward()
    mj_model = env.sim.model._model if hasattr(env.sim.model, "_model") else env.sim.model
    mj_data = env.sim.data._data if hasattr(env.sim.data, "_data") else env.sim.data

    qs: list[np.ndarray] = []
    xyz: list[np.ndarray] = []
    tags: list[str] = []
    errs: list[float] = []
    for tag, target in path:
        q = np.asarray(mj_data.qpos[dof_ids], dtype=np.float64).copy()
        jacp = np.zeros((3, mj_model.nv))
        jacr = np.zeros((3, mj_model.nv))
        err = 1.0
        for _ in range(60):
            mujoco.mj_jacSite(mj_model, mj_data, jacp, jacr, sid)
            j = jacp[:, dof_ids]
            xpos = np.asarray(mj_data.site_xpos[sid], dtype=np.float64)
            err_v = target - xpos
            err = float(np.linalg.norm(err_v))
            if err < 0.006:
                break
            jj = j @ j.T + 2e-3 * np.eye(3)
            dq = j.T @ np.linalg.solve(jj, err_v)
            q = _clip_q(q + np.clip(dq, -0.12, 0.12) + 0.04 * (q_bias - q))
            robot.set_robot_joint_positions(q)
            env.sim.forward()
        qs.append(_clip_q(q))
        xyz.append(np.asarray(mj_data.site_xpos[sid], dtype=np.float64).copy())
        tags.append(tag)
        errs.append(err)
    return {
        "joints_rad": np.stack(qs, axis=0),
        "ee_xyz": np.stack(xyz, axis=0),
        "tags": np.array(tags),
        "center": np.asarray(center, dtype=np.float64),
        "half_m": np.array([half_m], dtype=np.float64),
        "hz": np.array([_HZ], dtype=np.float64),
        "ik_err_m": np.asarray(errs, dtype=np.float64),
        "samples": np.array([samples], dtype=np.int32),
    }


def _load_traj(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {
        "joints_rad": np.asarray(z["joints_rad"], dtype=np.float64),
        "ee_xyz": np.asarray(z["ee_xyz"], dtype=np.float64) if "ee_xyz" in z.files else None,
        "tags": z["tags"] if "tags" in z.files else None,
        "center": np.asarray(z["center"], dtype=np.float64) if "center" in z.files else _EE_NEUTRAL,
        "ik_err_m": np.asarray(z["ik_err_m"], dtype=np.float64) if "ik_err_m" in z.files else None,
        "samples": int(np.asarray(z["samples"]).reshape(-1)[0]) if "samples" in z.files else _SAMPLES,
    }


def _print_plan(traj: dict, *, x_cm: float, y_cm: float, z_cm: float) -> None:
    q = traj["joints_rad"]
    c = np.asarray(traj["center"]).reshape(-1)[:3]
    print(f"cube center {np.round(c, 3).tolist()} m  offset cm x={x_cm} y={y_cm} z={z_cm}")
    print(f"  frames={len(q)}  q0_deg {np.round(np.degrees(q[0]), 1).tolist()}")
    print(f"  qN_deg {np.round(np.degrees(q[-1]), 1).tolist()}")
    if traj.get("ee_xyz") is not None:
        xyz = traj["ee_xyz"]
        print(f"  EE start {np.round(xyz[0], 3).tolist()}  end {np.round(xyz[-1], 3).tolist()}")
        print(
            "  EE span cm "
            f"Δx={100*(xyz[:, 0].max()-xyz[:, 0].min()):.1f} "
            f"Δy={100*(xyz[:, 1].max()-xyz[:, 1].min()):.1f} "
            f"Δz={100*(xyz[:, 2].max()-xyz[:, 2].min()):.1f}"
        )
    if traj.get("ik_err_m") is not None:
        e = traj["ik_err_m"]
        print(f"  IK err median {100*np.median(e):.1f} cm  max {100*np.max(e):.1f} cm")


def _resolve_traj(args) -> dict:
    if args.traj is not None:
        traj = _load_traj(args.traj)
        print("loaded", args.traj)
        return traj
    print("planning IK in sim (robosuite)…")
    center = cube_center(x_cm=args.x, y_cm=args.y, z_cm=args.z)
    return plan_cube_joints(center, half_m=_HALF_M, samples=int(args.samples))


def _n_per_edge(traj: dict) -> int:
    if traj.get("samples"):
        return int(traj["samples"])
    n = len(traj["joints_rad"])
    return max(1, int(round((n - 1) / 9)))


def play_traj(traj: dict, *, speed: float, hz: float) -> None:
    q = traj["joints_rad"]
    v = _V_AT_SPEED_1 * max(float(speed), 1e-4)
    ds = 2.0 * _HALF_M / max(_n_per_edge(traj), 1)
    dt = max(ds / v, 1.0 / (hz * 20.0))
    n_edge = _n_per_edge(traj)
    print(
        f"play {len(q)} steps  ~9×{n_edge}  "
        f"edge {100 * _HALF_M * 2:.0f} cm  {100 * v:.1f} cm/s  dt={dt:.3f}s"
    )
    env = _build_env(hz)
    g = 0.0
    q0 = _current_q(env)
    print(f"  robot start_deg {np.round(np.degrees(q0), 1).tolist()}")
    _lerp_q(env, q0, q[0], g, speed=speed, hz=hz)
    print("tracing 9 edges")
    for i in range(len(q)):
        t0 = time.time()
        env.step(np.concatenate([_clip_q(q[i]), [g]]))
        leftover = dt - (time.time() - t0)
        if leftover > 0:
            time.sleep(leftover)
    print("done")


def main() -> None:
    p = argparse.ArgumentParser(description="DROID FR3: 9-edge ±15 cm cube")
    p.add_argument("--speed", type=float, required=True, help="1 ≈ 15 cm/s on edges, 30 °/s approach")
    p.add_argument("--x", type=float, default=0.0, help="cube translate X, cm (gallery forward)")
    p.add_argument("--y", type=float, default=0.0, help="cube translate Y, cm (+ left)")
    p.add_argument("--z", type=float, default=0.0, help="cube translate Z, cm")
    p.add_argument("--samples", type=int, default=_SAMPLES)
    p.add_argument("--traj", type=Path, help="npz from --save; skip IK")
    p.add_argument("--save", type=Path, help="write planned joints npz")
    p.add_argument("--play", action="store_true", help="command the real FR3")
    p.add_argument("--dry-run", action="store_true", help="plan / print only")
    args = p.parse_args()
    if not args.play and not args.dry_run and args.save is None:
        raise SystemExit("pass --play and/or --dry-run (and optional --save)")

    traj = _resolve_traj(args)
    _print_plan(traj, x_cm=args.x, y_cm=args.y, z_cm=args.z)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save,
            joints_rad=traj["joints_rad"],
            ee_xyz=traj.get("ee_xyz"),
            tags=traj.get("tags"),
            center=traj["center"],
            hz=np.array([_HZ]),
            ik_err_m=traj.get("ik_err_m"),
            samples=np.array([traj.get("samples", args.samples)]),
            offset_cm=np.array([args.x, args.y, args.z], dtype=np.float64),
        )
        print("wrote", args.save)
    if args.dry_run and not args.play:
        print("dry-run: not contacting the robot")
        return
    play_traj(traj, speed=args.speed, hz=_HZ)


if __name__ == "__main__":
    main()
