#!/usr/bin/env python3
"""Interactive MuJoCo window for the FR3 EmptySpace dump poses.

Drag to orbit. Scroll to zoom. Close the window to quit.

    python adhoc/generation/robotarm/view_fr3.py
    python adhoc/generation/robotarm/view_fr3.py --traj .../02_shake_head_no_llm.npz
    python adhoc/generation/robotarm/view_fr3.py --home
    python adhoc/generation/robotarm/view_fr3.py --pose
    python adhoc/generation/robotarm/view_fr3.py --play
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from replay_droid import _QDOT_AT_SLOW_1, labeled_pose_path, plan_pose_path  # noqa: E402
from droid_traj import cue_start_index, strip_to_cue  # noqa: E402
_REPO = _HERE.parents[2]
_DEFAULT_TRAJ = (
    _REPO
    / "papers/icra2027_motion_abstraction/droid_replay/small10"
    / "02_shake_head_no/llm/02_shake_head_no_llm.npz"
)

_HOME = np.array(
    [0.0, np.pi / 16.0, 0.0, -np.pi / 2.0 - np.pi / 3.0, 0.0, np.pi - 0.2, np.pi / 4],
    dtype=np.float64,
)


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


def _load_traj(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    stored = None
    if "pose_frame" in z.files:
        stored = int(np.asarray(z["pose_frame"]).reshape(-1)[0])
    q, _g, _ee, pose_frame = strip_to_cue(
        np.asarray(z["joints_rad"], dtype=np.float64),
        np.asarray(z["gripper"], dtype=np.float64) if "gripper" in z.files else None,
        None,
        stored,
    )
    return {
        "joints_rad": q,
        "hz": float(np.asarray(z["hz"]).reshape(-1)[0]),
        "pose_frame": pose_frame,
        "path": path,
    }


def _make_env():
    robot = "FR3"
    arm_cfg = suite.load_part_controller_config(default_controller="JOINT_POSITION")
    env = suite.make(
        env_name="EmptySpace",
        robots=robot,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        control_freq=20,
        controller_configs=refactor_composite_controller_config(
            arm_cfg, robot, ["right", "left"]
        ),
        horizon=10000,
    )
    env.reset()
    env.sim.model.opt.gravity[:] = [0.0, 0.0, 0.0]
    return env


def _set_q(env, q: np.ndarray) -> None:
    env.robots[0].set_robot_joint_positions(np.asarray(q, dtype=np.float64).reshape(-1)[:7])
    env.sim.forward()


def _cam_pos(env, name: str) -> np.ndarray:
    cid = env.sim.model.camera_name2id(name)
    return np.asarray(env.sim.data.cam_xpos[cid], dtype=np.float64).copy()


def _grip(env):
    robot = env.robots[0]
    arm = "right" if "right" in robot.eef_site_id else next(iter(robot.eef_site_id))
    sid = robot.eef_site_id[arm]
    pos = np.asarray(env.sim.data.site_xpos[sid], dtype=np.float64).copy()
    mat = np.asarray(env.sim.data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
    # Grip site: local +Z is fingertips / approach.
    approach = mat[:, 2]
    return pos, approach


def _print_state(env, *, label: str) -> None:
    pos, approach = _grip(env)
    q = np.asarray(env.robots[0]._joint_positions, dtype=np.float64)[:7]
    front = _cam_pos(env, "frontview")
    print(f"\n=== {label} ===", flush=True)
    print(f"q_deg        {np.round(np.degrees(q), 1).tolist()}", flush=True)
    print(f"EE xyz (m)   {np.round(pos, 3).tolist()}", flush=True)
    print(f"EE approach   {np.round(approach, 3).tolist()}  (grip +Z, fingertips)", flush=True)
    print(f"frontview    {np.round(front, 3).tolist()}  (GIF camera, +X side of table)", flush=True)
    world_axes = {"+x (front/camera)": [1, 0, 0], "+y (left)": [0, 1, 0], "+z (up)": [0, 0, 1]}
    for name, ax in world_axes.items():
        print(f"  approach·{name:18s} {float(np.dot(approach, ax)):+.2f}", flush=True)


def _mark(handle, env) -> None:
    import mujoco

    pos, approach = _grip(env)
    front = _cam_pos(env, "frontview")
    handle.add_marker(
        pos=front,
        size=np.array([0.04, 0.04, 0.04]),
        rgba=np.array([1.0, 0.15, 0.1, 0.9]),
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
    )
    tip = pos + 0.35 * approach
    handle.add_marker(
        pos=0.5 * (pos + tip),
        size=np.array([0.012, 0.012, 0.18]),
        mat=_arrow_mat(tip - pos),
        rgba=np.array([0.15, 0.85, 1.0, 1.0]),
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    )


def _arrow_mat(direction: np.ndarray) -> np.ndarray:
    d = np.asarray(direction, dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return np.eye(3)
    z = d / n
    tmp = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(tmp, z)
    x /= max(float(np.linalg.norm(x)), 1e-9)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def _safe_add_marker(self, marker):
    import mujoco

    if self.scn.ngeom >= self.scn.maxgeom:
        return
    g = self.scn.geoms[self.scn.ngeom]
    g.dataid = -1
    g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
    g.objid = -1
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    if hasattr(g, "texuniform"):
        g.texuniform = 0
    if hasattr(g, "texrepeat"):
        g.texrepeat[0] = 1
        g.texrepeat[1] = 1
    g.emission = 0
    g.specular = 0.5
    g.shininess = 0.5
    g.reflectance = 0
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size[:] = np.ones(3) * 0.04
    g.mat[:] = np.eye(3)
    g.rgba[:] = np.ones(4)
    for key, value in marker.items():
        if key in ("texid", "label"):
            continue
        if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
            setattr(g, key, value)
        elif isinstance(value, (tuple, list, np.ndarray)) and hasattr(g, key):
            attr = getattr(g, key)
            attr[:] = np.asarray(value).reshape(attr.shape)
    self.scn.ngeom += 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--traj", type=Path, default=_DEFAULT_TRAJ)
    p.add_argument("--home", action="store_true", help="sim reset pose, not the cue pose")
    p.add_argument("--pose", action="store_true", help="lerp reset → first cue pose (replay_droid --pose)")
    p.add_argument("--frame", type=int, default=None, help="absolute dump frame")
    p.add_argument("--play", action="store_true", help="loop the dump after the pose frame")
    p.add_argument("--slow", type=float, default=1.0)
    p.add_argument("--print-only", action="store_true")
    args = p.parse_args()

    env = _make_env()
    q_seq = None
    pose_i = 0
    hz = 15.0
    pose_lerp = None
    if args.home:
        q = np.asarray(env.robots[0].robot_model.init_qpos, dtype=np.float64).reshape(-1)[:7]
        _set_q(env, q)
        _print_state(env, label="sim home / reset")
    else:
        if not args.traj.exists():
            raise SystemExit(f"missing traj: {args.traj}")
        traj = _load_traj(args.traj)
        q_seq = traj["joints_rad"]
        hz = traj["hz"]
        pose_i = args.frame if args.frame is not None else cue_start_index(q_seq, traj["pose_frame"])
        q_goal = q_seq[pose_i, :7]
        if args.pose:
            q0 = np.asarray(env.robots[0].robot_model.init_qpos, dtype=np.float64).reshape(-1)[:7]
            wps = plan_pose_path(q0, q_goal)
            labels = [n for n, _ in labeled_pose_path(q0, q_goal)]
            pose_lerp = []
            cur = q0
            for wp in wps:
                max_dq = float(np.max(np.abs(wp - cur)))
                qdot = _QDOT_AT_SLOW_1 * max(float(args.slow), 1e-4)
                duration = max(max_dq / max(qdot, 1e-6), 0.4)
                n = max(2, int(round(duration * hz)))
                for i in range(1, n + 1):
                    u = i / n
                    pose_lerp.append((1.0 - u) * cur + u * wp)
                cur = wp
            _set_q(env, q0)
            _print_state(env, label="--pose start (reset)")
            print(f"pose path {' → '.join(labels)}  {len(wps)} waypoints, {len(pose_lerp)} steps", flush=True)
        else:
            _set_q(env, q_goal)
            _print_state(env, label=f"{args.traj.name} frame {pose_i}/{len(q_seq)}")

    if args.print_only:
        return

    import mujoco_viewer

    print(
        "\nMuJoCo window: left-drag orbit, right-drag pan, scroll zoom. Close window to quit.",
        flush=True,
    )
    handle = mujoco_viewer.MujocoViewer(
        env.sim.model._model,
        env.sim.data._data,
        title="FR3 pose",
        width=1280,
        height=800,
    )
    handle.cam.lookat[:] = [0.4, 0.0, 0.9]
    handle.cam.distance = 2.4
    handle.cam.azimuth = 140.0
    handle.cam.elevation = -20.0
    lerp_i = 0
    play_i = pose_i
    dt_pose = 1.0 / hz
    dt_play = 1.0 / max(hz * float(args.slow), 1e-6)
    try:
        while handle.is_alive:
            t0 = time.time()
            sleep_dt = 0.03
            if pose_lerp is not None and lerp_i < len(pose_lerp):
                _set_q(env, pose_lerp[lerp_i])
                lerp_i += 1
                sleep_dt = dt_pose
            elif args.play and q_seq is not None:
                _set_q(env, q_seq[play_i, :7])
                play_i += 1
                if play_i >= len(q_seq):
                    play_i = pose_i
                sleep_dt = dt_play
            else:
                env.sim.forward()
            handle.render()
            leftover = sleep_dt - (time.time() - t0)
            if leftover > 0:
                time.sleep(leftover)
    finally:
        if handle.is_alive:
            handle.close()


if __name__ == "__main__":
    main()
