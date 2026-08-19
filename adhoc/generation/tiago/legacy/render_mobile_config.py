#!/usr/bin/env python3
"""Render a mobile manipulator config JSON to a GIF using TIAGo in robosuite.

This is the canonical Tiago / TIAGo mobile renderer. All shot renders,
generation outputs, and pilot GIFs should go through ``render_config`` here
(EE path steps use incremental Cartesian deltas for smooth strokes).

Usage:
  python render_mobile_config.py --config path/to/config.json --out path/to/output.gif
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import zlib

import fire
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import robosuite.utils.transform_utils as T
from robosuite.utils.ik_utils import IKSolver

local_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if local_root not in sys.path:
    sys.path.insert(0, local_root)
_robotarm_dir = os.path.join(local_root, "robotarm")
if _robotarm_dir not in sys.path:
    sys.path.insert(0, _robotarm_dir)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_POSE_BANK_ENTRIES: list | None = None
_POSE_BANK_BOUNDS: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

from adhoc.generation.joint_motion_schema import (  # noqa: E402
    canonical_joint_keyword,
    tiago_preprocess_movement_joints,
)
from motion_speed import (  # noqa: E402
    V_REF_M_PER_S,
    beat_n_frames,
    clamp_speed,
    path_n_frames,
)

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)

# ── qpos map ─────────────────────────────────────────────────────────────
QI_FWD = 0
QI_SIDE = 1
QI_YAW = 2
QI_TORSO = 3
QI_HEAD_PAN = 4
QI_HEAD_TILT = 5
QI_R_ARM = slice(6, 12)
QI_L_ARM = slice(18, 24)

FPS = 20
W, H = 512, 512
CAM = "frontview"
# Roomba-reference linear speed (was 0.12 — felt too slow vs other embodiments).
EE_PATH_M_PER_S_AT_SPEED_1 = V_REF_M_PER_S
PATH_IK_MAX_ITERS = 1
PATH_IK_POS_TOL_M = 0.005
PATH_IK_MAX_DQ = 0.06
PATH_IK_INTEGRATION_DT = 0.05
PATH_IK_DAMPING = 0.05
PATH_JAC_SUBSTEPS = 6
PATH_JAC_TRACK_ITERS = 40
PATH_JAC_POS_TOL_M = 0.004
PATH_MAX_EE_STEP_M = 0.028
PATH_DLS_MAX_ITERS = 120
PATH_DLS_LAMBDA = 0.05
PATH_DLS_MAX_DQ = 0.10
PATH_DLS_POS_TOL_M = 0.004
POSE_TRIM_STEP_M = 0.03
POSE_TRIM_MAX_DQ_RAD = 0.14

CART_AXES = {"x": 0, "y": 1, "z": 2}

_R_ARM_JOINT_NAMES = [f"robot0_arm_right_{i}_joint" for i in range(1, 7)]
_L_ARM_JOINT_NAMES = [f"robot0_arm_left_{i}_joint" for i in range(1, 7)]
_ARM_SIDE_META = {
    "right": (_R_ARM_JOINT_NAMES, "gripper0_right_grip_site"),
    "left": (_L_ARM_JOINT_NAMES, "gripper0_left_grip_site"),
}
_TIAGO_ARM_IK_CACHE: dict[int, dict[str, IKSolver]] = {}


def _line_distance_meters(distance: float) -> float:
    """Configs use meters when |d|<=1.5; legacy large values are scaled down."""
    if abs(distance) <= 1.5:
        return float(distance)
    return float(np.sign(distance) * min(0.45, abs(distance) * 0.008))


def _arc_radius_meters(radius: float) -> float:
    if radius <= 1.5:
        return float(np.clip(radius, 0.02, 0.35))
    return float(np.clip(radius * 0.01, 0.02, 0.35))


def _left_arm_path_line_mirror(axis: str) -> bool:
    """Mirror lateral line distance for bilateral (side: both) paths."""
    return str(axis).lower() == "y"


def _left_arm_path_sweep_mirror(plane: str) -> bool:
    """Whether to negate arc sweep on the left arm for bilateral paths.

    Only ``xy`` (horizontal) arcs need opposite sweeps so the hands open/close
    symmetrically. For ``yz`` / ``xz``, the same sweep keeps a shared vertical
    direction (e.g. ``depict_big_bowl`` scoop). Negating ``yz`` made one hand
    rise while the other fell.
    """
    return str(plane).lower() == "xy"


def _bilateral_left_line_distance(
    axis: str,
    distance_m: float,
    *,
    ee_start_r: np.ndarray | None = None,
    ee_start_l: np.ndarray | None = None,
) -> float:
    """Left-arm line distance for bilateral paths.

    Default: mirror lateral (y) so a wave opens symmetrically. If the hands are
    already clasped near the midline, keep the same sign so y=50 palms-together
    stays closed instead of being pulled apart.
    """
    if (
        str(axis).lower() == "y"
        and ee_start_r is not None
        and ee_start_l is not None
        and abs(float(ee_start_r[1]) - float(ee_start_l[1])) < 0.12
    ):
        return float(distance_m)
    return -distance_m if _left_arm_path_line_mirror(axis) else distance_m


def _bilateral_left_sweep_rad(plane: str, sweep_rad: float) -> float:
    return -sweep_rad if _left_arm_path_sweep_mirror(plane) else sweep_rad


def _normalize_path_plane(plane: str) -> str:
    p = str(plane).lower().strip()
    return "xz" if p == "zx" else p


def _path_plane_basis(plane: str) -> tuple[np.ndarray, np.ndarray]:
    p = _normalize_path_plane(plane)
    if p == "xy":
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    if p == "yz":
        return np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    return np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])


def _path_axis_unit(axis: str) -> np.ndarray:
    v = np.zeros(3, dtype=np.float64)
    v[CART_AXES[str(axis).lower()]] = 1.0
    return v


def _path_line_target(start_pos: np.ndarray, axis: str, distance_m: float, pp: float) -> np.ndarray:
    end_pos = start_pos + _path_axis_unit(axis) * float(distance_m)
    t = float(np.clip(pp, 0.0, 1.0))
    return (start_pos * (1.0 - t) + end_pos * t).astype(np.float64)


def _signed_sweep_rad(sweep_deg: float) -> float:
    """Signed arc angle in radians. +180 and −180 travel opposite ways."""
    s = float(sweep_deg)
    if abs(s) >= 360.0:
        return math.copysign(2.0 * math.pi * (abs(s) / 360.0), s)
    return math.radians(s)


def _path_arc_target(
    start_pos: np.ndarray,
    *,
    plane: str,
    radius_m: float,
    sweep_deg: float,
    pp: float,
) -> np.ndarray:
    """EE arc: center along −e1; sign of sweep selects travel sense (no cw/ccw).

    On plane=yz, −sweep bulges downward (bowl ∪); +sweep bulges upward.
    """
    e1, e2 = _path_plane_basis(plane)
    center = start_pos - float(radius_m) * e1
    theta = _signed_sweep_rad(sweep_deg) * float(pp)
    return (center + float(radius_m) * (math.cos(theta) * e1 + math.sin(theta) * e2)).astype(np.float64)


def _tiago_arm_ik(env, side: str) -> IKSolver:
    cache = _TIAGO_ARM_IK_CACHE.setdefault(id(env), {})
    if side not in cache:
        joint_names, site_name = _ARM_SIDE_META[side]
        robot_config = {
            "joint_names": joint_names,
            "end_effector_sites": [site_name],
            "nullspace_gains": [1.0] * len(joint_names),
        }
        cache[side] = IKSolver(
            model=env.sim.model._model,
            data=env.sim.data._data,
            robot_config=robot_config,
            damping=PATH_IK_DAMPING,
            integration_dt=PATH_IK_INTEGRATION_DT,
            max_dq=PATH_IK_MAX_DQ,
            input_type="keyboard",
            debug=False,
            input_action_repr="absolute",
            input_rotation_repr="axis_angle",
        )
    return cache[side]


def _eef_body_pos(env, side: str) -> np.ndarray:
    """Hand body position — same frame as Jacobian path tracking."""
    ee_body = R_EE_BODY if side == "right" else L_EE_BODY
    rid = env.sim.model.body_name2id(ee_body)
    return env.sim.data.body_xpos[rid].copy()


def _eef_grip_site_pos(env, side: str) -> np.ndarray:
    _, site_name = _ARM_SIDE_META[side]
    sid = env.sim.model.site_name2id(site_name)
    return env.sim.data.site_xpos[sid].copy()


def _eef_hold_quat_wxyz(env, side: str) -> np.ndarray:
    _, site_name = _ARM_SIDE_META[side]
    sid = env.sim.model.site_name2id(site_name)
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(env.sim.data.site_xmat[sid], dtype=np.float64).ravel())
    if float(np.linalg.norm(quat)) < 1e-8:
        quat[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat


def _site_hold_quat_wxyz(mujoco_data, site_id: int) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(mujoco_data.site(site_id).xmat, dtype=np.float64).ravel())
    if float(np.linalg.norm(quat)) < 1e-8:
        quat[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat


def _arm_side_kinematics(side: str) -> tuple[str, list[int]]:
    if side == "left":
        return L_EE_BODY, L_ARM_QVEL
    return R_EE_BODY, R_ARM_QVEL


def _apply_arm_cartesian_delta(
    env,
    q_frame: np.ndarray,
    *,
    side: str,
    delta: np.ndarray,
    substeps: int = PATH_JAC_SUBSTEPS,
) -> None:
    """Apply one world-frame EE delta via closed-loop Jacobian micro-steps."""
    delta = np.asarray(delta, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(delta)) < 1e-9:
        return
    ee_body, arm_qvel = _arm_side_kinematics(side)
    rid = env.sim.model.body_name2id(ee_body)
    env.sim.data.qpos[:] = q_frame
    env.sim.forward()
    goal = env.sim.data.body_xpos[rid].copy() + delta
    micro_cap = PATH_MAX_EE_STEP_M / max(1, int(substeps))
    for _ in range(max(1, int(substeps))):
        env.sim.data.qpos[:] = q_frame
        env.sim.forward()
        cur = env.sim.data.body_xpos[rid].copy()
        rem = goal - cur
        rem_norm = float(np.linalg.norm(rem))
        if rem_norm < PATH_JAC_POS_TOL_M:
            break
        step = rem
        if rem_norm > micro_cap:
            step = rem * (micro_cap / rem_norm)
        _apply_world_xyz_delta_via_jacobian(
            env,
            q_frame,
            ee_body=ee_body,
            arm_qvel=arm_qvel,
            dx=float(step[0]),
            dy=float(step[1]),
            dz=float(step[2]),
        )


def _dls_track_arm_to_cartesian_target(
    env,
    q_frame: np.ndarray,
    *,
    side: str,
    target_pos: np.ndarray,
    max_iters: int = PATH_DLS_MAX_ITERS,
    pos_tol_m: float = PATH_DLS_POS_TOL_M,
    damping: float = PATH_DLS_LAMBDA,
    max_dq: float = PATH_DLS_MAX_DQ,
) -> None:
    """6-DOF damped least-squares IK on hand body — smooth cumulative EE paths."""
    ee_body, arm_qvel = _arm_side_kinematics(side)
    rid = env.sim.model.body_name2id(ee_body)
    q_idxs = [int(arm_qvel[i]) for i in range(6)]
    target = np.asarray(target_pos, dtype=np.float64).reshape(3)
    diag = (float(damping) ** 2) * np.eye(3, dtype=np.float64)
    for _ in range(max_iters):
        env.sim.data.qpos[:] = q_frame
        env.sim.forward()
        err = target - env.sim.data.body_xpos[rid].copy()
        if float(np.linalg.norm(err)) < pos_tol_m:
            break
        jacp = env.sim.data.get_body_jacp(ee_body).reshape(3, -1)[:, q_idxs]
        dq = jacp.T @ np.linalg.solve(jacp @ jacp.T + diag, err)
        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > max_dq:
            dq *= max_dq / dq_norm
        for i, qi in enumerate(q_idxs):
            q_frame[qi] = float(q_frame[qi]) + float(dq[i])


def _track_arm_to_cartesian_target(
    env,
    q_frame: np.ndarray,
    *,
    side: str,
    target_pos: np.ndarray,
    max_iters: int = PATH_JAC_TRACK_ITERS,
    pos_tol_m: float = PATH_JAC_POS_TOL_M,
) -> None:
    """Jacobian steps toward an absolute Cartesian target (for per-frame base resets)."""
    ee_body, arm_qvel = _arm_side_kinematics(side)
    rid = env.sim.model.body_name2id(ee_body)
    target = np.asarray(target_pos, dtype=np.float64).reshape(3)
    for _ in range(max_iters):
        env.sim.data.qpos[:] = q_frame
        env.sim.forward()
        cur = env.sim.data.body_xpos[rid].copy()
        d = target - cur
        if float(np.linalg.norm(d)) < pos_tol_m:
            break
        step = d
        step_norm = float(np.linalg.norm(step))
        if step_norm > PATH_MAX_EE_STEP_M:
            step = step * (PATH_MAX_EE_STEP_M / step_norm)
        _apply_world_xyz_delta_via_jacobian(
            env,
            q_frame,
            ee_body=ee_body,
            arm_qvel=arm_qvel,
            dx=float(step[0]),
            dy=float(step[1]),
            dz=float(step[2]),
        )


def _path_arc_progress_delta(
    start_pos: np.ndarray,
    *,
    plane: str,
    radius_m: float,
    sweep_deg: float,
    pp: float,
    pp_prev: float,
) -> np.ndarray:
    cur = _path_arc_target(
        start_pos, plane=plane, radius_m=radius_m, sweep_deg=sweep_deg, pp=pp,
    )
    prev = _path_arc_target(
        start_pos, plane=plane, radius_m=radius_m, sweep_deg=sweep_deg, pp=pp_prev,
    )
    return (cur - prev).astype(np.float64)


def _path_line_progress_delta(
    start_pos: np.ndarray,
    *,
    axis: str,
    distance_m,
    pp: float,
    pp_prev: float,
    reps: int = 1,
) -> np.ndarray:
    # Absolute axis offsets from path start (span-aware), then incremental delta.
    off = _path_line_axis_offset_m(distance_m, pp, reps=reps)
    off_prev = _path_line_axis_offset_m(distance_m, pp_prev, reps=reps)
    return (_path_axis_unit(axis) * (off - off_prev)).astype(np.float64)


def _follow_path_arc_progress(
    env,
    q_frame: np.ndarray,
    *,
    start_pos: np.ndarray,
    plane: str,
    radius_m: float,
    sweep_deg: float,
    pp: float,
    pp_prev: float,
    side: str,
) -> None:
    delta = _path_arc_progress_delta(
        start_pos,
        plane=plane,
        radius_m=radius_m,
        sweep_deg=sweep_deg,
        pp=pp,
        pp_prev=pp_prev,
    )
    if float(np.linalg.norm(delta)) < 1e-9:
        return
    env.sim.data.qpos[:] = q_frame
    env.sim.forward()
    cur = _eef_body_pos(env, side)
    _dls_track_arm_to_cartesian_target(
        env,
        q_frame,
        side=side,
        target_pos=cur + delta,
    )


def _follow_path_line_progress(
    env,
    q_frame: np.ndarray,
    *,
    start_pos: np.ndarray,
    axis: str,
    distance_m,
    pp: float,
    pp_prev: float,
    side: str,
    reps: int = 1,
) -> None:
    delta = _path_line_progress_delta(
        start_pos,
        axis=axis,
        distance_m=distance_m,
        pp=pp,
        pp_prev=pp_prev,
        reps=reps,
    )
    if float(np.linalg.norm(delta)) < 1e-9:
        return
    env.sim.data.qpos[:] = q_frame
    env.sim.forward()
    cur = _eef_body_pos(env, side)
    _dls_track_arm_to_cartesian_target(
        env,
        q_frame,
        side=side,
        target_pos=cur + delta,
    )


def _path_pp_for_frame(
    fi: int,
    total_frames: int,
    reps: int,
    *,
    full_circle: bool = False,
    linear: bool = False,
) -> float:
    if full_circle and reps == 1:
        # Avoid pp=1 (same as start) on closed loops; span [0, (N-1)/N).
        t_global = fi / max(1, total_frames)
    else:
        t_global = fi / max(1, total_frames - 1)
    if reps == 1:
        pp = t_global
    elif linear:
        # Arc paths: same-direction laps (360° × reps), not ping-pong reversal.
        pp = t_global * reps
    else:
        cycle_t = (t_global * reps) % 1.0
        pp = _pingpong(cycle_t)
    if linear:
        return float(pp)
    return pp * pp * (3.0 - 2.0 * pp)


def _path_distance_stroke_m(distance) -> float:
    """One-way stroke length (m) for timing; supports scalar or [lo, hi] span."""
    if isinstance(distance, (list, tuple)) and len(distance) >= 2:
        lo = _line_distance_meters(float(distance[0]))
        hi = _line_distance_meters(float(distance[-1]))
        return abs(hi - lo)
    return abs(_line_distance_meters(float(distance if distance is not None else 0.0)))


def _path_line_axis_offset_m(distance, pp: float, *, reps: int) -> float:
    """World-axis offset (m) from path start for progress ``pp`` ∈ [0, 1].

    - scalar + rep=1: one-way 0→d
    - scalar + rep≥2: ``pp`` is already a ping-pong 0→1→0 from ``_path_pp_for_frame``
    - [lo, hi]: sine about midpoint (no snap to ``lo`` on frame 0), ``pp`` ping-pong
      is remapped so one 0→1→0 beat ≈ one lo→hi→lo set when reps≥2.
    """
    if isinstance(distance, (list, tuple)) and len(distance) >= 2:
        lo = _line_distance_meters(float(distance[0]))
        hi = _line_distance_meters(float(distance[-1]))
        mid = 0.5 * (lo + hi)
        amp = 0.5 * (hi - lo)
        # pp ping-pong 0→1→0 ≡ sin phase 0→π→2π over one beat; for reps, the
        # outer loop already tiles beats via n_frames — use cos-like from mid.
        return mid + amp * math.sin(math.pi * float(pp))
    dist_m = _line_distance_meters(float(distance if distance is not None else 0.0))
    return dist_m * float(pp)


def _path_frame_count(path: dict, *, shape: str, duration: float, speed: float) -> int:
    """Motion frames for a path stroke (hold_time is appended separately).

    Zero-length lines (``hold(t=…)`` rewrite) must return 0 so the last pose
    freezes instead of running filler IK frames that drift / cut the gesture.
    Uses shared Roomba-scale path timing (line length / arc ``r·θ``).
    """
    spd = clamp_speed(speed)
    if shape == "line":
        # One-way |hi−lo| (or |d|). Oscillating cycle = round-trip 2× that.
        stroke_m = _path_distance_stroke_m(path.get("distance", 0.0))
        if stroke_m < 1e-6:
            return 0
        reps = max(1, int(path.get("repetition", 1) or 1))
        oscillating = reps >= 2 or (
            isinstance(path.get("distance"), (list, tuple))
            and len(path.get("distance")) >= 2
        )
        if oscillating:
            n = path_n_frames(
                fps=FPS,
                speed=spd,
                oscillating=True,
                rep=reps,
                stroke_m=2.0 * stroke_m,
            )
        else:
            n = path_n_frames(
                fps=FPS,
                speed=spd,
                oscillating=False,
                rep=1,
                stroke_m=stroke_m,
            )
        if duration and duration > 0:
            n = max(n, int(duration * FPS))
        return max(n, 4 if oscillating else 1)
    radius_m = _arc_radius_meters(float(path.get("radius", 0.12)))
    sweep_deg = float(path.get("sweep", path.get("degrees", 360)))
    reps = max(1, int(path.get("repetition", 1) or 1))
    # Full circles: time one lap of circumference (same linear speed as a line).
    if abs(sweep_deg) >= 360:
        sweep_for_len = 360.0 * max(1, int(round(abs(sweep_deg) / 360.0)))
    else:
        sweep_for_len = abs(sweep_deg)
    # rep multiplies path length at constant linear speed (not beat timing).
    n = path_n_frames(
        fps=FPS,
        speed=spd,
        oscillating=False,
        rep=1,
        sweep_deg=sweep_for_len * reps,
        radius_m=radius_m,
    )
    if duration and duration > 0:
        n = max(n, int(duration * FPS))
    return max(n, 4 if reps >= 2 else 1)


# ── Presets (must match build_prompt19_mobile_from_cues.py) ──────────────
TORSO_MAP = {"low": 0.05, "mid": 0.18, "high": 0.34}

ARM_PRESETS = {
    "up":         [-90, -90, 0, 17, 0, 0],
    "back":       [-90,   0, 0, 17, 0, 0],
    "down+back":  [-90,  90, 0, 17, 0, 0],
    "right":      [  0,   0, 0, 17, 0, 0],
    "down+right": [  0,  90, 0, 17, 0, 0],
    "front":      [ 90,   0, 0, 17, 0, 0],
    "down+front": [ 90,  90, 0, 17, 0, 0],
    "fold":       [ 50,  70, 160, 100, 0, 0],
    "still":      [  0,  90, 0, 17, 0, 0],
}
ARM_ALIAS_TO_PRESET = {
    "front": "front",
    "back": "back",
    "in": "right",
    "out": "back",
    "up": "up",
    "down": "down+right",
    "still": "still",
}
DISPLAY_DIR_TO_SOURCE = {
    "front": "front",
    "back": "back",
    "in": "right",
    "out": "left",
    "up": "up",
    "down": "down",
}

GRIPPER_MAP = {"horizontal": 1.57, "vertical": 0.0}

# Same joint numbers on L/R → mirrored cartesian hang (Tiago URDF axes).
# Do NOT use [0, -0.3, …] for the right arm — that is a raised/abducted pose.
L_ARM_REST = [0, 1.57, 0, 0.3, 0, 0]
R_ARM_REST = [0, 1.57, 0, 0.3, 0, 0]

HEAD_PRESETS = {
    "center": [0.0,  0.0],
    "left":   [-0.8, 0.0],
    "right":  [0.8,  0.0],
    "up":     [0.0,  0.5],
    "down":   [0.0, -0.6],
}

JOINT_QPOS_MAP = {
    ("shoulder", "pitch"): 6,
    ("shoulder", "roll"):  7,
    ("elbow", "pitch"):    8,
    ("elbow", "roll"):     9,
    ("wrist", "pitch"):    10,
    ("wrist", "roll"):     11,
    ("torso", "height"):   3,
    ("head", "pan"):       4,
    ("head", "tilt"):      5,
}

L_JOINT_QPOS_MAP = {
    ("shoulder", "pitch"): 18,
    ("shoulder", "roll"):  19,
    ("elbow", "pitch"):    20,
    ("elbow", "roll"):     21,
    ("wrist", "pitch"):    22,
    ("wrist", "roll"):     23,
}

# ── Cartesian (Jacobian IK) constants ─────────────────────────────────────
R_EE_BODY = "robot0_right_hand"
L_EE_BODY = "robot0_left_hand"
# Forearm proximal link (~ elbow); used only for frontal pose-bank rejects.
R_ELBOW_BODY = "robot0_arm_right_4_link"
TORSO_REF_BODY = "robot0_torso_lift_link"
HEAD_REF_BODY = "robot0_head"
R_ARM_QVEL = list(range(6, 12))
L_ARM_QVEL = list(range(18, 24))

FOLD_RAD = np.array([np.deg2rad(d) for d in ARM_PRESETS["fold"]])

_JOINT_LOCAL_DOFS = {
    "shoulder": [0, 1],
    "elbow":    [2, 3],
    "wrist":    [4, 5],
    "arm":      [0, 1, 2, 3, 4, 5],
}

# movement ``shoulder|elbow|wrist`` + axis x|y|z with |degrees|>1 → local joint ° (not world Jacobian).
_ARM_AXIS_LOCAL_IDX = {"x": 0, "y": 1, "z": 2}


def _bilateral_mirror_sign_for_axis(axis: str) -> float:
    """Mirror convention: same sign on x/z, opposite on y (lateral)."""
    return -1.0 if str(axis).lower() == "y" else 1.0


def _movement_joint_qpos_targets(jname: str, axis: str, side: str) -> list[tuple[int, float]]:
    """Resolve qpos indices (+ bilateral mirror sign) for a movement joint key."""
    ax = str(axis).lower()
    mirror_sign = _bilateral_mirror_sign_for_axis(ax)
    if jname in _JOINT_LOCAL_DOFS and ax in _ARM_AXIS_LOCAL_IDX:
        local_dofs = _JOINT_LOCAL_DOFS[jname]
        slot = min(_ARM_AXIS_LOCAL_IDX[ax], len(local_dofs) - 1)
        abs_local = int(local_dofs[slot])
        if side == "both":
            return [
                (6 + abs_local, 1.0),
                (18 + abs_local, mirror_sign),
            ]
        if side == "left":
            return [(18 + abs_local, 1.0)]
        return [(6 + abs_local, 1.0)]

    if side == "both":
        qr = JOINT_QPOS_MAP.get((jname, axis))
        ql = L_JOINT_QPOS_MAP.get((jname, axis))
        out: list[tuple[int, float]] = []
        if qr is not None:
            out.append((qr, 1.0))
        if ql is not None:
            out.append((ql, mirror_sign))
        return out
    if side == "left":
        qi = L_JOINT_QPOS_MAP.get((jname, axis))
        return [(qi, 1.0)] if qi is not None else []
    qi = JOINT_QPOS_MAP.get((jname, axis))
    return [(qi, 1.0)] if qi is not None else []

# Legacy per-local-DOF signs (kept for reference); motion-axis rule above is authoritative.
_BILATERAL_MIRROR_SIGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float64)


_JAC_MIN_USEFUL = 0.08  # below this, joint-space inverse is ill-posed; expand search to all arm DOFs
_WRIST_BODY = {
    "right": "robot0_arm_right_6_link",
    "left": "robot0_arm_left_6_link",
}
_FINGERTIP_GEOMS = {
    "right": (
        "gripper0_right_left_fingertip_collision",
        "gripper0_right_right_fingertip_collision",
    ),
    "left": (
        "gripper0_left_left_fingertip_collision",
        "gripper0_left_right_fingertip_collision",
    ),
}


def _arm_side_from_ee_body(ee_body: str) -> str:
    return "left" if "left" in str(ee_body) else "right"


def _fingertip_pad_mid_world(env, side: str = "right") -> np.ndarray | None:
    """World midpoint of the two fingertip collision geoms for ``side``."""
    names = _FINGERTIP_GEOMS.get(side)
    if not names:
        return None
    try:
        g0 = env.sim.model.geom_name2id(names[0])
        g1 = env.sim.model.geom_name2id(names[1])
    except Exception:
        return None
    return np.asarray(
        0.5 * (env.sim.data.geom_xpos[g0] + env.sim.data.geom_xpos[g1]),
        dtype=np.float64,
    )


def _translational_jac_at_point(env, ee_body: str, point_world: np.ndarray) -> np.ndarray:
    """3×nv translational Jacobian of ``ee_body`` at a world point (MuJoCo)."""
    model = env.sim.model
    data = env.sim.data
    m = model._model if hasattr(model, "_model") else model
    d = data._data if hasattr(data, "_data") else data
    bid = int(m.body(ee_body).id)
    jacp = np.zeros((3, int(m.nv)), dtype=np.float64)
    jacr = np.zeros((3, int(m.nv)), dtype=np.float64)
    pt = np.asarray(point_world, dtype=np.float64).reshape(3, 1)
    mujoco.mj_jac(m, d, jacp, jacr, pt, bid)
    return jacp


def _find_best_dof_for_axis(
    env,
    q,
    ee_body,
    arm_qvel,
    axis_idx,
    joint_group="elbow",
    *,
    restrict_to_group: bool = False,
):
    """Pick DOF in *joint_group* best aligned with world *axis_idx*.

    Shoulder/elbow use EE-body translational Jacobian. Wrist uses the fingertip-pad
    midpoint (plus a short offset along the pad span) so ``move(wrist, z, …)``
    scores the DOF that actually bobs the fingers in world height — EE-origin jac
    for wrist DOFs is near-zero and collapses to a weakly coupled roll.

    Score = |jac[axis]|² / ||jac||² (axis purity), tie-break on |jac[axis]|.
    If the best coupling in that group is tiny (near kinematic singularities), scan
    all six arm joints — unless *restrict_to_group*.
    """
    local_dofs = list(_JOINT_LOCAL_DOFS.get(joint_group, _JOINT_LOCAL_DOFS["elbow"]))
    env.sim.data.qpos[:] = q
    env.sim.forward()
    side = _arm_side_from_ee_body(ee_body)
    jacp = env.sim.data.get_body_jacp(ee_body).reshape(3, -1)
    if joint_group == "wrist":
        pad = _fingertip_pad_mid_world(env, side)
        if pad is not None:
            names = _FINGERTIP_GEOMS.get(side)
            span = np.zeros(3, dtype=np.float64)
            if names:
                try:
                    g0 = env.sim.data.geom_xpos[env.sim.model.geom_name2id(names[0])]
                    g1 = env.sim.data.geom_xpos[env.sim.model.geom_name2id(names[1])]
                    span = np.asarray(g1 - g0, dtype=np.float64)
                    n = float(np.linalg.norm(span))
                    if n > 1e-9:
                        span = span / n
                except Exception:
                    span = np.zeros(3, dtype=np.float64)
            # Offset along pad span so wrist revolutes get a lever (G1 palm-offset analog).
            jacp = _translational_jac_at_point(env, ee_body, pad + 0.06 * span)

    def _score(li: int) -> tuple[float, float]:
        col = np.asarray(jacp[:, arm_qvel[li]], dtype=np.float64)
        val = float(col[axis_idx])
        frac = float(val * val) / (float(np.dot(col, col)) + 1e-12)
        return frac, abs(val)

    best_local = local_dofs[0]
    best_key = (-1.0, -1.0)
    best_val = 0.0
    for li in local_dofs:
        frac, ab = _score(li)
        key = (frac, ab)
        if key > best_key:
            best_key = key
            best_local = li
            best_val = float(jacp[axis_idx, arm_qvel[li]])
    if not restrict_to_group and abs(best_val) < _JAC_MIN_USEFUL:
        best_local, best_val = 0, 0.0
        best_key = (-1.0, -1.0)
        for li in range(6):
            frac, ab = _score(li)
            key = (frac, ab)
            if key > best_key:
                best_key = key
                best_local = li
                best_val = float(jacp[axis_idx, arm_qvel[li]])
    return best_local, best_val


def _apply_wrist_world_axis_degrees(
    env,
    q_frame: np.ndarray,
    *,
    base_q: np.ndarray,
    side: str,
    axis_idx: int,
    delta_deg: float,
    damping: float = 0.05,
    max_dq: float = np.deg2rad(25.0),
) -> None:
    """Drive fingertip pad along world axis using wrist DOFs (+ elbow if needed).

    Single-DOF wrist swings follow an arc (large lateral Δx for ``z``). Solving the
    2-wrist-DOF pad Jacobian toward a pure-axis tip target keeps air-quotes-style
    ``move(wrist, z, …)`` vertically readable.

    At some poses (e.g. front + grip=vert) wrist jac on world ``x`` is near-zero, so
    wrist-only IK thrashes orientation for ~cm tip motion. Recruit elbow DOFs when
    residual axis error stays large so beckon-style ``move(wrist, x, ±deg)`` reads.
    """
    ee_body, arm_qvel = (R_EE_BODY, R_ARM_QVEL) if side == "right" else (L_EE_BODY, L_ARM_QVEL)
    wrist_locals = list(_JOINT_LOCAL_DOFS["wrist"])
    elbow_locals = list(_JOINT_LOCAL_DOFS["elbow"])
    wrist_idxs = [int(arm_qvel[li]) for li in wrist_locals]
    elbow_idxs = [int(arm_qvel[li]) for li in elbow_locals]
    # Reset wrist (and elbow — may assist below) to base; keep shoulder from pose.
    for qi in wrist_idxs + elbow_idxs:
        q_frame[qi] = float(base_q[qi])
    env.sim.data.qpos[:] = q_frame
    env.sim.forward()
    pad0 = _fingertip_pad_mid_world(env, side)
    if pad0 is None:
        pad0 = np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id(ee_body)], float)
    wrist_body = _WRIST_BODY[side]
    try:
        wpos = np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id(wrist_body)], float)
        lever = float(np.linalg.norm(pad0 - wpos))
    except Exception:
        lever = 0.10
    lever = float(np.clip(lever, 0.05, 0.18))
    # Tip arc length for the requested wrist swing.
    # Small-angle lever*θ under-reads on pulled-back frontview (beckon ±35° ≈ 6cm
    # tip → nearly invisible). Scale up and floor so gestures stay readable.
    tip = float(np.deg2rad(delta_deg)) * lever * 2.5
    if abs(delta_deg) >= 15.0:
        tip = float(np.sign(tip) * max(abs(tip), 0.12))
    disp = np.zeros(3, dtype=np.float64)
    disp[axis_idx] = tip
    target = pad0 + disp
    target_axis = float(disp[axis_idx])
    diag = (float(damping) ** 2) * np.eye(3, dtype=np.float64)

    def _ik_step(q_idxs: list[int], iters: int) -> float:
        last_axis_err = abs(target_axis)
        for _ in range(iters):
            env.sim.data.qpos[:] = q_frame
            env.sim.forward()
            pad = _fingertip_pad_mid_world(env, side)
            if pad is None:
                pad = np.asarray(
                    env.sim.data.body_xpos[env.sim.model.body_name2id(ee_body)], float
                )
            err = target - pad
            last_axis_err = abs(float(err[axis_idx]))
            if float(np.linalg.norm(err)) < 1e-4:
                break
            jacp = _translational_jac_at_point(env, ee_body, pad)[:, q_idxs]
            dq = jacp.T @ np.linalg.solve(jacp @ jacp.T + diag, err)
            n = float(np.linalg.norm(dq))
            if n > max_dq:
                dq *= max_dq / n
            for i, qi in enumerate(q_idxs):
                q_frame[qi] = float(q_frame[qi]) + float(dq[i])
        return last_axis_err

    axis_err = _ik_step(wrist_idxs, 8)
    # Weak wrist coupling on this axis → pull elbow so tip travels along world axis.
    if abs(target_axis) > 1e-4 and axis_err > 0.35 * abs(target_axis):
        _ik_step(elbow_idxs + wrist_idxs, 10)


_OVERLAP_THRESHOLD = 0.08  # 8 cm

_MIN_JAC = 0.04  # minimum |∂x/∂q|; must match plausible EE motion scale (0.01 caused ~25 rad swings)


def _safe_jacobian_div(displacement: float, jac_val: float) -> float:
    """Map Cartesian displacement (m) to a *joint* delta (rad or slide m)."""
    if jac_val == 0.0:
        return 0.0
    lim = max(abs(jac_val), _MIN_JAC)
    return displacement / (np.sign(jac_val) * lim)


def _clamp_arm_deltas(base_q: np.ndarray, q_frame: np.ndarray, max_abs_delta: float) -> None:
    """After kinematic assembly, prevent any single arm DOF from moving more than max_abs_delta."""
    for sl in (slice(6, 12), slice(18, 24)):
        for i in range(sl.start, sl.stop):
            d = float(q_frame[i] - base_q[i])
            if abs(d) > max_abs_delta:
                q_frame[i] = base_q[i] + float(np.clip(d, -max_abs_delta, max_abs_delta))


def _gripper_qpos_keep(from_q: np.ndarray, into_q: np.ndarray) -> None:
    """Do not animate mimic gripper joints from keyframed poses (reduces spurious finger jitter)."""
    into_q[12:18] = from_q[12:18]
    into_q[24:30] = from_q[24:30]


def _apply_bilateral_mirror_line_frame(
    env,
    q_frame: np.ndarray,
    path_anchor_q: np.ndarray,
    *,
    axis: str,
    distance_m: float,
    dist_l: float,
    pp: float,
) -> None:
    """Bimanual mirror line: each arm moves from anchor via its own DOF (no sequential IK)."""
    ax_idx = CART_AXES[str(axis).lower()]
    disp_r = float(distance_m) * float(pp)
    disp_l = float(dist_l) * float(pp)
    q_frame[:] = path_anchor_q
    for disp, ee_body, arm_qvel, q_off in (
        (disp_r, R_EE_BODY, R_ARM_QVEL, 6),
        (disp_l, L_EE_BODY, L_ARM_QVEL, 18),
    ):
        if abs(disp) < 1e-9:
            continue
        li, jv = _find_best_dof_for_axis(
            env, path_anchor_q, ee_body, arm_qvel, ax_idx, "elbow", restrict_to_group=True
        )
        qi = q_off + li
        q_frame[qi] = path_anchor_q[qi] + _safe_jacobian_div(disp, jv)


def _apply_bilateral_mirror_line_delta(
    env,
    q_frame: np.ndarray,
    *,
    axis: str,
    distance_m: float,
    dist_l: float,
    pp: float,
    pp_prev: float,
) -> None:
    """Bimanual mirror line: apply only the progress increment since the previous frame."""
    ax_idx = CART_AXES[str(axis).lower()]
    disp_r = float(distance_m) * float(pp - pp_prev)
    disp_l = float(dist_l) * float(pp - pp_prev)
    for disp, ee_body, arm_qvel, q_off in (
        (disp_r, R_EE_BODY, R_ARM_QVEL, 6),
        (disp_l, L_EE_BODY, L_ARM_QVEL, 18),
    ):
        if abs(disp) < 1e-9:
            continue
        li, jv = _find_best_dof_for_axis(
            env, q_frame, ee_body, arm_qvel, ax_idx, "elbow", restrict_to_group=True
        )
        qi = q_off + li
        q_frame[qi] = float(q_frame[qi]) + _safe_jacobian_div(disp, jv)


def _track_bilateral_cartesian_targets(
    env,
    q_frame: np.ndarray,
    *,
    target_r: np.ndarray,
    target_l: np.ndarray,
    passes: int = 2,
) -> None:
    """Alternate right/left IK passes so neither arm drifts when tracking the other."""
    for _ in range(max(1, int(passes))):
        _track_arm_to_cartesian_target(env, q_frame, side="right", target_pos=target_r)
        _track_arm_to_cartesian_target(env, q_frame, side="left", target_pos=target_l)


def _stagger_if_overlapping(env, q_frame):
    """If both EEs are closer than threshold, stagger left arm forward+up."""
    env.sim.data.qpos[:] = q_frame
    env.sim.forward()
    r_id = env.sim.model.body_name2id(R_EE_BODY)
    l_id = env.sim.model.body_name2id(L_EE_BODY)
    r_pos = env.sim.data.body_xpos[r_id]
    l_pos = env.sim.data.body_xpos[l_id]
    dist = np.linalg.norm(r_pos - l_pos)
    if dist < _OVERLAP_THRESHOLD:
        gap = _OVERLAP_THRESHOLD - dist
        q_frame[18 + 0] += np.deg2rad(5) * (gap / _OVERLAP_THRESHOLD)   # shoulder pitch fwd
        q_frame[18 + 2] += np.deg2rad(-3) * (gap / _OVERLAP_THRESHOLD)  # elbow pitch up


# Empty-arena frontview default is pos=(1.6, 0, 1.45), fovy=45. Pull the camera
# further back (+x) and widen FOV so forward EE strokes are not clipped.
# NOTE: robosuite hard-reset reloads the XML model, wiping cam_pos/fovy — framing
# must be re-applied after every env.reset() (see render_config).
_CAM_FRONT_BASE = (1.6, 0.0, 1.45)
# Continuum (pull-in): 1/2→1.55/64, 3/4→1.40/60, 4/5→1.28/55, 6/7→1.22/53.
_CAM_POS_SCALE_DEFAULT = 1.22
_CAM_FOVY_DEFAULT = 53.0


def _apply_frontview_framing(
    env,
    *,
    cam_pos_scale: float | None = None,
    cam_fovy: float | None = None,
) -> None:
    if cam_pos_scale is None:
        cam_pos_scale = float(getattr(env, "_tiago_cam_pos_scale", _CAM_POS_SCALE_DEFAULT))
    if cam_fovy is None:
        cam_fovy = float(getattr(env, "_tiago_cam_fovy", _CAM_FOVY_DEFAULT))
    try:
        for cname in ["sideview", "birdview"]:
            cid = env.sim.model.camera_name2id(cname)
            env.sim.model.cam_fovy[cid] = 55.0
        cid = env.sim.model.camera_name2id(CAM)
        bx, by, bz = _CAM_FRONT_BASE
        # Stay on the front axis (looking at the robot); scale pulls the camera
        # further "back" from the subject so arm tips stay in frame.
        env.sim.model.cam_pos[cid] = (
            float(bx) * float(cam_pos_scale),
            float(by),
            float(bz) * (0.55 + 0.45 * float(cam_pos_scale)),
        )
        env.sim.model.cam_fovy[cid] = float(cam_fovy)
        try:
            env.sim.forward()
        except Exception:
            pass
    except Exception:
        pass


def _disable_debug_overlays(env) -> None:
    """Hide MuJoCo rangefinder / tendon debug draws in offscreen captures.

    Tiago's lidar sites otherwise paint long green rays through the grippers
    whenever ``mjVIS_RANGEFINDER`` is left on (robosuite default).
    """
    try:
        import mujoco

        ctx = getattr(env.sim, "_render_context_offscreen", None)
        if ctx is None or not hasattr(ctx, "vopt"):
            return
        for flag_name in ("mjVIS_RANGEFINDER", "mjVIS_TENDON"):
            flag = getattr(mujoco.mjtVisFlag, flag_name, None)
            if flag is not None:
                ctx.vopt.flags[flag] = 0
    except Exception:
        pass


def _make_env(
    *,
    cam_pos_scale: float = _CAM_POS_SCALE_DEFAULT,
    cam_fovy: float = _CAM_FOVY_DEFAULT,
):
    arm_cfg = suite.load_part_controller_config(default_controller="OSC_POSE")
    ctrl = refactor_composite_controller_config(arm_cfg, "Tiago", ["right", "left"])
    env = suite.make(
        env_name="EmptySpace",
        robots="Tiago",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=CAM,
        camera_heights=H,
        camera_widths=W,
        control_freq=FPS,
        controller_configs=ctrl,
    )
    # Persist so render_config can re-apply after hard reset reloads the XML cams.
    env._tiago_cam_pos_scale = float(cam_pos_scale)
    env._tiago_cam_fovy = float(cam_fovy)
    env.reset()
    _disable_debug_overlays(env)
    _apply_frontview_framing(env, cam_pos_scale=cam_pos_scale, cam_fovy=cam_fovy)
    return env


def _default_qpos(env):
    q = env.sim.data.qpos.copy()
    q[:3] = 0
    q[QI_TORSO] = TORSO_MAP["mid"]
    q[QI_HEAD_PAN] = 0
    q[QI_HEAD_TILT] = 0
    # Bilateral hang: identical joint rows (URDF axes already mirror).
    q[QI_R_ARM] = R_ARM_REST[:]
    q[QI_L_ARM] = L_ARM_REST[:]
    return q


def _arm_rad(arm_position: str, gripper: str = "vertical") -> list[float]:
    preset_key = ARM_ALIAS_TO_PRESET.get(arm_position, arm_position)
    arm_deg = ARM_PRESETS.get(preset_key, ARM_PRESETS["down+right"])
    arm_rad = [np.deg2rad(d) for d in arm_deg]
    if arm_deg[2] == 0:
        arm_rad[2] = GRIPPER_MAP.get(gripper, 0.0)
    return arm_rad


def _mirror_offset(l_rad: list[float], arm_position: str) -> None:
    """Deprecated no-op.

    Historical: nudged left shoulder to avoid overlap. That broke
    ``left_arm=mirror`` symmetry (joint angles must match right). Kept as a
    stub so call sites stay harmless.
    """
    del l_rad, arm_position


def _remirror_left_arm_joints(q: np.ndarray) -> None:
    """Copy right-arm joint angles onto the left arm (Tiago URDF mirror)."""
    q[QI_L_ARM] = list(q[QI_R_ARM])


def _pose_bank_json_path() -> str:
    return os.environ.get(
        "MOBILE_POSE_BANK_JSON",
        os.path.join(_REPO_ROOT, "data", "seed", "tiago", "mobile_pose_bank_729.json"),
    )


def _get_pose_bank_entries() -> list:
    global _POSE_BANK_ENTRIES
    if _POSE_BANK_ENTRIES is not None:
        return _POSE_BANK_ENTRIES
    p = _pose_bank_json_path()
    if not os.path.isfile(p):
        _POSE_BANK_ENTRIES = []
        return _POSE_BANK_ENTRIES
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    poses = data.get("poses") if isinstance(data, dict) else None
    _POSE_BANK_ENTRIES = poses if isinstance(poses, list) else []
    _load_pose_bank_bounds()
    return _POSE_BANK_ENTRIES


def _load_pose_bank_bounds() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-axis (lo, hi, body_center) in meters for body-centered 0–100 percentiles."""
    global _POSE_BANK_BOUNDS
    if _POSE_BANK_BOUNDS is not None:
        return _POSE_BANK_BOUNDS
    p = _pose_bank_json_path()
    if not os.path.isfile(p):
        _POSE_BANK_BOUNDS = None
        return None
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("meta") if isinstance(data, dict) else None
    wb = (meta or {}).get("world_pad_mid_bounds_m") if isinstance(meta, dict) else None
    if not isinstance(wb, dict):
        _POSE_BANK_BOUNDS = None
        return None
    try:
        lo = np.array([float(wb["x"][0]), float(wb["y"][0]), float(wb["z"][0])], dtype=np.float64)
        hi = np.array([float(wb["x"][1]), float(wb["y"][1]), float(wb["z"][1])], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        _POSE_BANK_BOUNDS = None
        return None
    wc = (meta or {}).get("world_body_center_m") if isinstance(meta, dict) else None
    if isinstance(wc, dict):
        try:
            center = np.array([float(wc["x"]), float(wc["y"]), float(wc["z"])], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            center = 0.5 * (lo + hi)
    else:
        # Legacy banks: linear mid of lo/hi (same as old 0–100 encoding).
        center = 0.5 * (lo + hi)
    from adhoc.generation.tiago.legacy import render_mobile_manip_schema as sch

    center = sch._clamp_center_into_bounds(lo, hi, center)
    _POSE_BANK_BOUNDS = (lo, hi, center)
    return _POSE_BANK_BOUNDS


def pose_xyz_resolution_mode() -> str:
    """``nearest``: bank L1 top1 only (legacy). ``trim``: top1 + off-screen EE path trim (default)."""
    mode = os.environ.get("MOBILE_POSE_XYZ_MODE", "trim").strip().lower()
    return mode if mode in ("nearest", "trim") else "trim"


def _torso_forward_world_unit(sim) -> np.ndarray:
    """World-unit vector pointing from torso out the chest (+X local of torso_lift)."""
    tid = sim.model.body_name2id(TORSO_REF_BODY)
    R = sim.data.body_xmat[tid].reshape(3, 3)
    v = np.asarray(R[:, 0], dtype=float).copy()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=float)


def _gesture_ee_body(pose: dict) -> str:
    """End-effector body for the gesturing arm (pose-bank r_arm_rad drives the active side)."""
    left = pose.get("left_arm", "still")
    arm = str(pose.get("arm_position", "down+right")).strip().lower()
    if left == "still" and arm != "still":
        return R_EE_BODY
    return L_EE_BODY


def _hand_head_x_gap_ok(sim, ee_body: str = R_EE_BODY) -> bool:
    """True when hand_x - head_x >= gap_min (default -0.1 m; negative allows hand behind head)."""
    gap_min = float(os.environ.get("MOBILE_POSE_HEAD_HAND_X_GAP", "-0.1"))
    try:
        ee = sim.data.body_xpos[sim.model.body_name2id(ee_body)]
        head = sim.data.body_xpos[sim.model.body_name2id(HEAD_REF_BODY)]
    except Exception:
        return True
    hand_x = float(ee[0])
    head_x = float(head[0])
    return hand_x - head_x >= gap_min


def _hand_forward_vs_torso_head_ok(sim, ee_body: str = R_EE_BODY) -> bool:
    """Deprecated alias — use _hand_head_x_gap_ok."""
    return _hand_head_x_gap_ok(sim, ee_body)


def _right_hand_forward_vs_torso_head_ok(sim) -> bool:
    """Deprecated alias — prefer _hand_forward_vs_torso_head_ok with _gesture_ee_body(pose)."""
    return _hand_forward_vs_torso_head_ok(sim, R_EE_BODY)


def _right_elbow_forward_vs_torso_head_ok(sim) -> bool:
    """Deprecated alias — use hand (EE) check for pose-bank filtering."""
    return _right_hand_forward_vs_torso_head_ok(sim)


def _apply_pose_nominal(q, pose: dict) -> None:
    """Arm presets only (used to label dir/orient for pose-bank lookup)."""
    from adhoc.generation.tiago.mobile_pose_normalize import normalize_pose_dict  # noqa: WPS433

    pose = normalize_pose_dict(pose)
    torso = pose.get("torso_height", "mid")
    q[QI_TORSO] = TORSO_MAP.get(torso, 0.18)

    arm = pose.get("arm_position", "down+right")
    grip = pose.get("gripper_orientation", "vertical")
    q[QI_R_ARM] = _arm_rad(arm, grip)

    left = pose.get("left_arm", "still")
    if left == "still":
        q[QI_L_ARM] = L_ARM_REST[:]
    elif left == "mirror":
        q[QI_L_ARM] = list(q[QI_R_ARM])
    else:
        q[QI_L_ARM] = _arm_rad(left, grip)

    head = pose.get("head", "center")
    hp = HEAD_PRESETS.get(head, [0, 0])
    q[QI_HEAD_PAN] = hp[0]
    q[QI_HEAD_TILT] = hp[1]


def _apply_bank_r_arm(q, pose: dict, r_arm_rad: list[float]) -> None:
    """Apply pose template with a pose-bank r_arm_rad entry (matches tile renderer)."""
    from adhoc.generation.tiago.mobile_pose_normalize import normalize_pose_dict  # noqa: WPS433

    pose = normalize_pose_dict(pose)
    torso = pose.get("torso_height", "mid")
    q[QI_TORSO] = TORSO_MAP.get(torso, 0.18)
    q[QI_R_ARM] = list(r_arm_rad)

    arm = pose.get("arm_position", "down+right")
    grip = pose.get("gripper_orientation", "vertical")
    left = pose.get("left_arm", "still")
    if left == "still" and str(arm).strip().lower() == "still":
        q[QI_L_ARM] = L_ARM_REST[:]
    elif left == "still":
        q[QI_L_ARM] = L_ARM_REST[:]
    elif left == "mirror":
        q[QI_L_ARM] = list(r_arm_rad)
    else:
        q[QI_L_ARM] = _arm_rad(left, grip)

    head = pose.get("head", "center")
    hp = HEAD_PRESETS.get(head, [0, 0])
    q[QI_HEAD_PAN] = hp[0]
    q[QI_HEAD_TILT] = hp[1]


def pose_has_explicit_xyz(pose: dict) -> bool:
    """True when config sets any of x/y/z (numeric or low/med/high)."""
    return any(k in pose for k in ("x", "y", "z"))


def pose_xyz_axes_specified(pose: dict) -> tuple[str, ...]:
    """Axes explicitly present with a real value (missing / null keys are not forced)."""
    out: list[str] = []
    for a in ("x", "y", "z"):
        if a not in pose:
            continue
        v = pose[a]
        if isinstance(v, (int, float)):
            out.append(a)
        elif isinstance(v, str) and v.strip().lower() in ("low", "med", "high"):
            out.append(a)
    return tuple(out)


def pose_xyz_rank_mode(pose: dict, *, ignore_xyz: bool = False) -> str:
    """How to rank pose-bank pool: random | partial | full | tile_idx."""
    del ignore_xyz
    axes = pose_xyz_axes_specified(pose)
    if not axes:
        return "tile_idx"
    if len(axes) == 3:
        return "full"
    return "partial"


def pose_use_xyz_distance(pose: dict, *, ignore_xyz: bool = False) -> bool:
    """Whether pose-bank ranking uses xyz error (partial or full)."""
    return pose_xyz_rank_mode_for_motion(pose, ignore_xyz=ignore_xyz) in ("partial", "full")


def pose_xyz_rank_mode_for_motion(pose: dict, *, ignore_xyz: bool = False) -> str:
    """Rank mode for motion pick (random when no xyz and not ignore_xyz pool)."""
    if ignore_xyz:
        return pose_xyz_rank_mode(pose)
    axes = pose_xyz_axes_specified(pose)
    if not axes:
        return "random"
    if len(axes) == 3:
        return "full"
    return "partial"


def pose_target_xyz_tuple(pose: dict) -> tuple[float | None, float | None, float | None]:
    """Resolved targets per axis; ``None`` when that axis is omitted from pose."""
    out: list[float | None] = []
    for axis in "xyz":
        if axis in pose:
            out.append(_resolve_level(pose[axis], axis))
        else:
            out.append(None)
    return tuple(out)


def _pose_rank_seed(pose: dict) -> int:
    keys = ("arm_position", "gripper_orientation", "elbow_bended", "x", "y", "z", "head")
    blob = "|".join(f"{k}={pose.get(k)!r}" for k in keys)
    return zlib.adler32(blob.encode("utf-8"))


def _pose_dist_fn(pose: dict, *, ignore_xyz: bool = False):
    """Return sum-of-|Δ| callable when pose specifies xyz axes, else ``None``."""
    del ignore_xyz
    axes = pose_xyz_axes_specified(pose)
    if not axes:
        return None
    targets = {a: _resolve_level(pose[a], a) for a in axes}

    def _dist(e: dict) -> float:
        return sum(abs(float(e.get(axis, 50)) - target) for axis, target in targets.items())

    return _dist


def _slice_pool(pool: list[dict], top_k: int | None) -> list[dict]:
    if top_k is None or top_k <= 0:
        return pool
    return pool[:top_k]


def _rank_filtered_pool(
    pool: list[dict],
    pose: dict,
    *,
    ignore_xyz: bool,
    top_k: int | None,
) -> list[dict]:
    """Rank filtered candidates: random / partial xyz / full xyz / tile_idx."""
    mode = (
        pose_xyz_rank_mode(pose)
        if ignore_xyz
        else pose_xyz_rank_mode_for_motion(pose, ignore_xyz=False)
    )
    if mode == "tile_idx":
        ranked = sorted(pool, key=_pose_pool_sort_key)
    elif mode == "random":
        ranked = list(pool)
        random.Random(_pose_rank_seed(pose)).shuffle(ranked)
    else:
        dist_fn = _pose_dist_fn(pose, ignore_xyz=ignore_xyz)
        ranked = sorted(pool, key=dist_fn)
    return _slice_pool(ranked, top_k)


def resolve_pose_bank_topk(*, ignore_xyz: bool, topk: int | None) -> int | None:
    """Effective top_k for pose-bank candidate listing.

    - ``ignore_xyz=False``: always 1 (xyz-ranked motion pick).
    - ``ignore_xyz=True``, ``topk=None``: all matching tiles (0 → no slice).
    - ``ignore_xyz=True``, ``topk=N``: top N tiles after filter/sort.
    """
    if not ignore_xyz:
        return 1
    if topk is None:
        return 0
    return topk


def _pose_pool_sort_key(e: dict) -> tuple:
    """Stable ordering when xyz distance is ignored."""
    tile_idx = e.get("tile_idx")
    if tile_idx is not None and int(tile_idx) >= 0:
        return (0, int(tile_idx))
    return (
        1,
        float(e.get("x", 50)),
        float(e.get("y", 50)),
        float(e.get("z", 50)),
    )


def _resolve_level(val: str | float, axis: str) -> float:
    """Map categorical levels (low, med, high) to body-centered percentiles (med≈50)."""
    if isinstance(val, (int, float)):
        return float(val)
    v = str(val).lower()
    # Body-centered scale: 50 = torso; low/high ≈ quarter / three-quarter toward extrema.
    if v == "low":
        return 25.0
    if v == "med":
        return 50.0
    if v == "high":
        return 75.0
    return 50.0


_PLAUSIBLE_POSES: list[dict] | None = None


def _get_plausible_poses() -> list[dict]:
    global _PLAUSIBLE_POSES
    if _PLAUSIBLE_POSES is not None:
        return _PLAUSIBLE_POSES
    p = os.path.join(_REPO_ROOT, "data", "seed", "tiago", "plausible_poses_metadata.jsonl")
    if not os.path.isfile(p):
        _PLAUSIBLE_POSES = []
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    _PLAUSIBLE_POSES = out
    return out


def rank_pose_bank_topk(
    env,
    pose: dict,
    *,
    top_k: int | None = None,
    ignore_xyz: bool = False,
) -> dict:
    """Rank pose-bank entries: filter dir+orient (+ elbow_bended), then sort."""
    return list_pose_bank_candidates(env, pose, top_k=top_k, ignore_xyz=ignore_xyz)


def list_pose_bank_candidates(
    env,
    pose: dict,
    *,
    top_k: int | None = None,
    plausible_only: bool = False,
    relax_elbow_if_empty: bool = False,
    ignore_xyz: bool = False,
) -> dict:
    """List pose-bank candidates after dir+orient (+ elbow_bended) filtering."""
    top_k = resolve_pose_bank_topk(ignore_xyz=ignore_xyz, topk=top_k)
    arm = pose.get("arm_position", "down+right")
    grip_raw = str(pose.get("gripper_orientation", "vertical")).lower().replace("_", "").replace(" ", "")
    # Palm semantics are matched on humanoid via normals; Google bank is horiz/vert only.
    _palm_to_hv = {
        "palmup": "horizontal",
        "palmdown": "horizontal",
        "palmfront": "vertical",
        "palmback": "vertical",
        "palmin": "vertical",
        "palmout": "vertical",
    }
    _hv = {
        "horiz": "horizontal",
        "horizontal": "horizontal",
        "open": "horizontal",
        "vert": "vertical",
        "vertical": "vertical",
        "close": "vertical",
        "closed": "vertical",
        "pinch": "vertical",
    }
    parts = [p for p in re.split(r"[|/+,]", grip_raw) if p] or [grip_raw]
    # Compounds like vert|palmin: Google bank only has hv — prefer explicit hv token.
    hv = next((_hv[p] for p in parts if p in _hv), None)
    palm = next((p for p in parts if p in _palm_to_hv), None)
    if hv is not None:
        grip = hv
    elif palm is not None:
        grip = _palm_to_hv[palm]
    else:
        grip = grip_raw if grip_raw in ("horizontal", "vertical") else "vertical"
    arm_label = str(arm).strip().lower()
    if arm_label == "still":
        return {
            "arm_position": "still",
            "target_xyz": None,
            "bank_dir": None,
            "bank_orient": None,
            "n_pool": 0,
            "n_hand_ok": 0,
            "entries": [],
        }

    target_xyz = pose_target_xyz_tuple(pose)
    xyz_rank_mode = pose_xyz_rank_mode(pose, ignore_xyz=ignore_xyz)
    dist_fn = _pose_dist_fn(pose, ignore_xyz=ignore_xyz)

    # 1. Try plausible set first
    plausible = _get_plausible_poses()
    desired_dir = DISPLAY_DIR_TO_SOURCE.get(arm_label, arm_label)
    cand = [
        e for e in plausible
        if e.get("arm_position") == arm_label and e.get("gripper_orientation") == grip
    ]
    base_cand = list(cand)

    # Optional: Filter by elbow_bended
    eb_target = pose.get("elbow_bended")
    elbow_relaxed = False
    if eb_target is not None:
        eb_target = bool(eb_target)
        eb_cand = [e for e in cand if e.get("elbow_bended") == eb_target]
        if eb_cand:
            cand = eb_cand
        elif relax_elbow_if_empty and base_cand:
            cand = base_cand
            elbow_relaxed = True
        else:
            cand = eb_cand

    def _entry_from_plausible(e: dict, rank: int) -> dict:
        return {
            "rank": rank,
            "dist": dist_fn(e) if dist_fn else None,
            "x": float(e.get("x", 50)),
            "y": float(e.get("y", 50)),
            "z": float(e.get("z", 50)),
            "r_arm_rad": [float(x) for x in e["r_arm_rad"]],
            "selected": rank == 1,
            "head_hand_gap_ok": True,
            "source": "plausible_set",
            "tile_idx": int(e.get("tile_idx", -1)),
            "elbow_bended": bool(e.get("elbow_bended", False)),
        }

    # Optional: pin exact plausible tile
    tile_idx = pose.get("plausible_tile_idx")
    if tile_idx is not None and cand:
        pinned = [e for e in cand if int(e.get("tile_idx", -1)) == int(tile_idx)]
        if pinned:
            e = pinned[0]
            return {
                "arm_position": arm_label,
                "gripper_orientation": str(grip),
                "target_xyz": target_xyz,
                "elbow_bended": eb_target,
                "n_pool": len(cand),
                "entries": [{
                    "rank": 1,
                    "dist": dist_fn(e) if dist_fn else None,
                    "x": float(e.get("x", 50)),
                    "y": float(e.get("y", 50)),
                    "z": float(e.get("z", 50)),
                    "r_arm_rad": [float(x) for x in e["r_arm_rad"]],
                    "selected": True,
                    "head_hand_gap_ok": True,
                    "source": "plausible_set",
                    "tile_idx": int(tile_idx),
                    "elbow_bended": bool(e.get("elbow_bended", False)),
                }],
                "source": "plausible_set",
            }

    if cand:
        ranked_raw = _rank_filtered_pool(cand, pose, ignore_xyz=ignore_xyz, top_k=top_k)
        entries = [_entry_from_plausible(e, i) for i, e in enumerate(ranked_raw, start=1)]
        return {
            "arm_position": arm_label,
            "gripper_orientation": str(grip),
            "target_xyz": target_xyz,
            "elbow_bended": eb_target,
            "elbow_bended_relaxed": elbow_relaxed,
            "n_pool": len(cand),
            "entries": entries,
            "source": "plausible_set",
            "ignore_xyz": ignore_xyz,
            "xyz_rank_mode": xyz_rank_mode,
        }

    if plausible_only:
        return {
            "arm_position": arm_label,
            "gripper_orientation": str(grip),
            "target_xyz": target_xyz,
            "elbow_bended": eb_target,
            "elbow_bended_relaxed": elbow_relaxed,
            "n_pool": 0,
            "n_hand_ok": 0,
            "entries": [],
            "source": "plausible_set",
        }

    # 2. Fallback to full bank if no plausible matches
    bank = _get_pose_bank_entries()
    qn = _default_qpos(env)
    _apply_pose_nominal(qn, pose)
    env.sim.forward()
    from adhoc.generation.tiago.legacy import render_mobile_manip_schema as sch

    d = desired_dir if desired_dir else sch._dir_six_way_from_anchor(env, qn, cam_name=sch.CAM)
    o = sch._gripper_horizontal_vertical_by_reach_dir(env.sim, d)

    cand = [e for e in bank if e.get("dir") == d and e.get("orient") == o]
    if eb_target is not None:
        cand = [
            e for e in cand
            if (abs(float(e.get("joint_deg", [0] * 6)[3])) >= 15) == eb_target
        ]

    ee_body = _gesture_ee_body(pose)

    def _head_hand_gap_ok(e: dict) -> bool:
        qc = _default_qpos(env)
        _apply_bank_r_arm(qc, pose, e["r_arm_rad"])
        env.sim.data.qpos[:] = qc
        env.sim.forward()
        return _hand_head_x_gap_ok(env.sim, ee_body)

    hand_ok = [e for e in cand if _head_hand_gap_ok(e)]
    ranked_raw = _rank_filtered_pool(hand_ok, pose, ignore_xyz=ignore_xyz, top_k=top_k)

    entries: list[dict] = []
    for i, e in enumerate(ranked_raw, start=1):
        entries.append(
            {
                "rank": i,
                "dist": dist_fn(e) if dist_fn else None,
                "x": float(e.get("x", 50)),
                "y": float(e.get("y", 50)),
                "z": float(e.get("z", 50)),
                "r_arm_rad": [float(x) for x in e["r_arm_rad"]],
                "selected": i == 1,
                "head_hand_gap_ok": True,
                "source": "full_bank",
                "elbow_bended": abs(float(e.get("joint_deg", [0] * 6)[3])) >= 15,
            }
        )
    return {
        "arm_position": arm_label,
        "gripper_orientation": str(grip),
        "target_xyz": target_xyz,
        "elbow_bended": eb_target,
        "bank_dir": d,
        "bank_orient": o,
        "n_pool": len(cand),
        "n_hand_ok": len(hand_ok),
        "n_ranked": len(entries),
        "entries": entries,
        "fallback": "nominal" if not entries and not bank else None,
        "source": "full_bank",
        "ignore_xyz": ignore_xyz,
        "xyz_rank_mode": xyz_rank_mode,
    }


def _gesture_arm_side(pose: dict) -> str:
    left = str(pose.get("left_arm", "still")).strip().lower()
    arm = str(pose.get("arm_position", "down+right")).strip().lower()
    if left != "still" and arm == "still":
        return "left"
    return "right"


def _pose_pct_to_world_m(axis: str, pct: float) -> float:
    """Body-centered inverse: percentile → world pad-mid coordinate (meters)."""
    bounds = _load_pose_bank_bounds()
    if bounds is None:
        return 0.0
    lo, hi, center = bounds
    idx = CART_AXES[axis]
    from adhoc.generation.tiago.legacy import render_mobile_manip_schema as sch

    return sch._pct_0100_to_world_axis(float(pct), float(lo[idx]), float(hi[idx]), float(center[idx]))


def _pose_pct_delta_to_meters(axis: str, from_pct: float, to_pct: float) -> float:
    """Meters between two percentiles on one axis (asymmetric spans around body center)."""
    return _pose_pct_to_world_m(axis, to_pct) - _pose_pct_to_world_m(axis, from_pct)


def _pose_xyz_trim_spec_from_ee(env, q: np.ndarray, pose: dict, side: str) -> list[dict]:
    """Trim using *measured* EE → body-centered percentile targets (not bank labels).

    Bank x/y/z labels drift with torso height / IK, so Δpct from the bank entry often
    moves the wrong way. y=50 must land on the robot body center in world meters.
    """
    env.sim.data.qpos[:] = q
    env.sim.forward()
    cur = np.asarray(_eef_body_pos(env, side), dtype=float)
    out: list[dict] = []
    for axis in pose_xyz_axes_specified(pose):
        target_pct = _resolve_level(pose[axis], axis)
        target_m = _pose_pct_to_world_m(axis, target_pct)
        idx = CART_AXES[axis]
        distance_m = float(target_m - cur[idx])
        if abs(distance_m) < 1e-4:
            continue
        out.append(
            {
                "axis": axis,
                "target_pct": target_pct,
                "target_m": target_m,
                "distance_m": distance_m,
            }
        )
    return out


def _apply_pose_xyz_trim_instant(
    env,
    q: np.ndarray,
    pose: dict,
    bank_entry: dict | None = None,
) -> None:
    """Apply per-axis EE line correction at pp=1 — not emitted as render frames."""
    del bank_entry  # kept for call-site compat; trim uses measured EE, not bank %
    side = _gesture_arm_side(pose)
    trim_spec = _pose_xyz_trim_spec_from_ee(env, q, pose, side)
    if not trim_spec:
        return
    env.sim.data.qpos[:] = q
    env.sim.forward()
    ee_start = _eef_body_pos(env, side)
    for spec in trim_spec:
        axis = str(spec["axis"])
        distance_m = float(spec["distance_m"])
        n_sub = max(1, int(math.ceil(abs(distance_m) / POSE_TRIM_STEP_M)))
        q_step_anchor = q.copy()
        for si in range(1, n_sub + 1):
            pp_prev = (si - 1) / n_sub
            pp = si / n_sub
            delta = _path_line_progress_delta(
                ee_start, axis=axis, distance_m=distance_m, pp=pp, pp_prev=pp_prev,
            )
            if float(np.linalg.norm(delta)) < 1e-9:
                continue
            env.sim.data.qpos[:] = q
            env.sim.forward()
            cur = _eef_body_pos(env, side)
            _dls_track_arm_to_cartesian_target(
                env, q, side=side, target_pos=cur + delta,
            )
            _clamp_arm_deltas(q_step_anchor, q, POSE_TRIM_MAX_DQ_RAD)
            q_step_anchor = q.copy()
        env.sim.data.qpos[:] = q
        env.sim.forward()
        ee_start = _eef_body_pos(env, side)
        # Second pass from *actual* residual (clamp/IK may undershoot).
        idx = CART_AXES[axis]
        residual = float(spec["target_m"] - ee_start[idx])
        if abs(residual) > 1e-3:
            n_sub = max(1, int(math.ceil(abs(residual) / POSE_TRIM_STEP_M)))
            q_step_anchor = q.copy()
            for si in range(1, n_sub + 1):
                pp_prev = (si - 1) / n_sub
                pp = si / n_sub
                delta = _path_line_progress_delta(
                    ee_start, axis=axis, distance_m=residual, pp=pp, pp_prev=pp_prev,
                )
                if float(np.linalg.norm(delta)) < 1e-9:
                    continue
                env.sim.data.qpos[:] = q
                env.sim.forward()
                cur = _eef_body_pos(env, side)
                _dls_track_arm_to_cartesian_target(
                    env, q, side=side, target_pos=cur + delta,
                )
                _clamp_arm_deltas(q_step_anchor, q, POSE_TRIM_MAX_DQ_RAD)
                q_step_anchor = q.copy()
            env.sim.data.qpos[:] = q
            env.sim.forward()
            ee_start = _eef_body_pos(env, side)


def _resolve_right_arm_rad_from_bank(env, pose: dict, *, ignore_xyz: bool = False) -> list[float]:
    arm = pose.get("arm_position", "down+right")
    grip = pose.get("gripper_orientation", "vertical")
    if str(arm).strip().lower() == "still":
        return R_ARM_REST[:]
    pinned = pose.get("pinned_r_arm_rad")
    if pinned is not None:
        return [float(x) for x in pinned]
    nominal = _arm_rad(arm, grip)
    if not _get_pose_bank_entries():
        return nominal

    ranked = rank_pose_bank_topk(env, pose, top_k=1, ignore_xyz=ignore_xyz)
    if ranked.get("entries"):
        return ranked["entries"][0]["r_arm_rad"]

    qn = _default_qpos(env)
    _apply_pose_nominal(qn, pose)
    env.sim.data.qpos[:] = qn
    env.sim.forward()
    return nominal


def _apply_pose(env, q, pose: dict, *, ignore_xyz: bool = False) -> None:
    from adhoc.generation.tiago.mobile_pose_normalize import normalize_pose_dict  # noqa: WPS433

    pose = normalize_pose_dict(pose)
    torso = pose.get("torso_height", "mid")
    q[QI_TORSO] = TORSO_MAP.get(torso, 0.18)

    arm = pose.get("arm_position", "down+right")
    grip = pose.get("gripper_orientation", "vertical")
    r_res = _resolve_right_arm_rad_from_bank(env, pose, ignore_xyz=ignore_xyz)
    q[QI_R_ARM] = r_res

    left = pose.get("left_arm", "still")
    if left == "still" and str(arm).strip().lower() == "still":
        # Explicit bilateral still mode: keep both arms at rest.
        q[QI_R_ARM] = R_ARM_REST[:]
        q[QI_L_ARM] = L_ARM_REST[:]
    elif left == "still":
        q[QI_L_ARM] = L_ARM_REST[:]
    elif left == "mirror":
        q[QI_L_ARM] = list(r_res)
    else:
        q[QI_L_ARM] = _arm_rad(left, grip)

    head = pose.get("head", "center")
    hp = HEAD_PRESETS.get(head, [0, 0])
    q[QI_HEAD_PAN] = hp[0]
    q[QI_HEAD_TILT] = hp[1]

    if (
        not ignore_xyz
        and pose_xyz_resolution_mode() == "trim"
        and pose_xyz_axes_specified(pose)
        and str(arm).strip().lower() != "still"
        and _get_pose_bank_entries()
    ):
        ranked = rank_pose_bank_topk(env, pose, top_k=1, ignore_xyz=False)
        if ranked.get("entries"):
            _apply_pose_xyz_trim_instant(env, q, pose, ranked["entries"][0])
            if left == "mirror":
                # Exact joint copy after xyz trim (mirror = same joint angles).
                _remirror_left_arm_joints(q)


def _q_from_pose(
    env,
    pose: dict,
    *,
    base_x: float,
    base_y: float,
    base_yaw: float,
    q_seed: np.ndarray | None = None,
    ignore_xyz: bool = False,
) -> np.ndarray:
    q_tgt = (q_seed.copy() if q_seed is not None else _default_qpos(env))
    _apply_pose(env, q_tgt, pose, ignore_xyz=ignore_xyz)
    q_tgt[QI_FWD] = base_x
    q_tgt[QI_SIDE] = base_y
    q_tgt[QI_YAW] = base_yaw
    return q_tgt


def _pose_transition_frames(speed: float) -> int:
    """Manipulator-style: transition lasts 1/speed seconds at FPS (Roomba clamp)."""
    return max(1, beat_n_frames(fps=FPS, speed=speed, rep=1))


def _capture(env, q, camera=CAM):
    """Set pose and render. Pure kinematics: no env.step — stepping with zeros fights OSC
    + contact solver after manual qpos writes (looks like violent jitter / collision pops).
    """
    # Offscreen context can recreate / reset vopt between frames — re-apply every capture.
    _disable_debug_overlays(env)
    env.sim.data.qpos[:] = q
    try:
        env.sim.data.qvel[:] = 0.0
    except Exception:
        pass
    env.sim.forward()
    frame = env.sim.render(camera_name=camera, width=W, height=H, depth=False)
    # Copy buffer immediately — MuJoCo may reuse the underlying array.
    return Image.fromarray(np.flipud(frame).copy())


def _pingpong(t01: float) -> float:
    """Map [0,1] to triangle wave [0→1→0]."""
    return 1.0 - abs(2.0 * t01 - 1.0)


def _movement_beat_frames(speed: float) -> int:
    """Manipulator-style: one direction beat lasts 1/speed seconds (Roomba clamp)."""
    return max(1, beat_n_frames(fps=FPS, speed=speed, rep=1))


def _movement_degrees_are_cartesian_meters(lo: float, hi: float) -> bool:
    """x/y/z on arm joints use meters when |value| <= 1 (EE deltas); larger values are degrees."""
    return max(abs(float(lo)), abs(float(hi))) <= 1.0


def _smoothstep01(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def _movement_pp_frames(reps: int, speed: float, joints_work: list[dict]) -> list[float]:
    """Build per-frame pp values via discrete beats (manipulator-style).

    - repetition=1 + degrees [lo,hi]: one-way lo→hi, hold at hi (bows, transitions).
    - repetition≥2 + degrees [lo,hi]: smooth ping-pong lo→hi→lo per rep (nods, waves).
    """
    beat_frames = _movement_beat_frames(speed)
    has_range = any(
        isinstance(j.get("degrees"), (list, tuple)) and len(j.get("degrees")) >= 2
        for j in joints_work
    )
    reps = max(1, int(reps))
    pp_frames: list[float] = []

    def _append_ramp(forward: bool) -> None:
        for fi in range(beat_frames):
            t = _smoothstep01((fi + 1) / beat_frames)
            pp_frames.append(t if forward else 1.0 - t)

    if has_range and reps >= 2:
        n = max(2, beat_frames * 2 * reps)
        for fi in range(n):
            t = fi / max(1, n - 1)
            phase = 2.0 * math.pi * reps * t
            pp_frames.append(0.5 * (1.0 - math.cos(phase)))
    elif has_range:
        _append_ramp(True)
    else:
        for _ in range(reps):
            _append_ramp(True)
    return pp_frames


def _resample_pp_frames_to_duration(pp_frames: list[float], duration_s: float) -> list[float]:
    """Upsample pp to match step ``duration`` so motion is not frozen between sparse beats."""
    target = max(1, int(float(duration_s) * FPS))
    if not pp_frames or target <= len(pp_frames):
        return pp_frames
    src = pp_frames
    if len(src) == 1:
        return src * target
    out: list[float] = []
    for fi in range(target):
        u = fi / max(1, target - 1)
        src_f = u * (len(src) - 1)
        i0 = int(src_f)
        i1 = min(i0 + 1, len(src) - 1)
        t = src_f - i0
        out.append(float(src[i0] * (1.0 - t) + src[i1] * t))
    return out


def _snapshots_for_base_path(
    base_x: float,
    base_y: float,
    base_yaw: float,
    path: dict,
) -> tuple[list[tuple[float, float, float]], float, float, float]:
    """Discretize a base ``path`` (line or arc) into world (x, y, yaw) samples at ``FPS``."""
    snapshots: list[tuple[float, float, float]] = []
    shape = path.get("shape", "line")
    speed = float(path.get("speed", 1.0))
    bx, by, yaw = base_x, base_y, base_yaw

    if shape == "line":
        x_dist = float(path.get("x", 0.0))
        y_dist = float(path.get("y", 0.0))
        total_dist = math.sqrt(x_dist**2 + y_dist**2)
        n_frames = path_n_frames(
            fps=FPS, speed=speed, oscillating=False, rep=1, stroke_m=total_dist
        )
        n_frames = max(1, n_frames)
        for fi in range(n_frames):
            t = fi / max(1, n_frames - 1)
            snapshots.append((bx + x_dist * t, by + y_dist * t, yaw))
        bx += x_dist
        by += y_dist
    elif shape == "arc":
        radius = float(path.get("radius", 0.0))
        degrees = float(path.get("degrees", 90))
        total_rad = math.radians(degrees)
        n_frames = path_n_frames(
            fps=FPS,
            speed=speed,
            oscillating=False,
            rep=1,
            sweep_deg=abs(degrees),
            radius_m=abs(radius),
        )
        n_frames = max(1, n_frames)
        cx, cy = bx, by
        cur_yaw = yaw
        for fi in range(n_frames):
            t = fi / max(1, n_frames - 1)
            angle = total_rad * t
            if radius == 0:
                nx, ny = cx, cy
            else:
                nx = cx + radius * math.sin(angle)
                ny = cy + radius * (1 - math.cos(angle))
            snapshots.append((nx, ny, cur_yaw + angle))
        yaw = cur_yaw + total_rad
        if radius != 0:
            bx = cx + radius * math.sin(total_rad)
            by = cy + radius * (1 - math.cos(total_rad))
        else:
            bx, by = cx, cy
    else:
        return [], bx, by, yaw

    return snapshots, bx, by, yaw


def _plane_uv(plane: str, xyz: np.ndarray) -> tuple[float, float]:
    p = plane.lower()
    if p == "xy":
        return float(xyz[0]), float(xyz[1])
    if p == "xz":
        return float(xyz[0]), float(xyz[2])
    if p == "yz":
        return float(xyz[1]), float(xyz[2])
    raise ValueError(f"Unknown plane {plane!r} (expected xy, xz, yz)")


def _set_plane_uv(plane: str, dest: np.ndarray, u: float, v: float) -> None:
    p = plane.lower()
    if p == "xy":
        dest[0], dest[1] = u, v
    elif p == "xz":
        dest[0], dest[2] = u, v
    elif p == "yz":
        dest[1], dest[2] = u, v
    else:
        raise ValueError(f"Unknown plane {plane!r}")


def _apply_world_xyz_delta_via_jacobian(
    env,
    q_frame: np.ndarray,
    *,
    ee_body: str,
    arm_qvel: list[int],
    dx: float,
    dy: float,
    dz: float,
) -> None:
    for axis, disp in (("x", dx), ("y", dy), ("z", dz)):
        if abs(disp) < 1e-9:
            continue
        ax_idx = CART_AXES[axis]
        li, jv = _find_best_dof_for_axis(env, q_frame, ee_body, arm_qvel, ax_idx, "elbow")
        qi = int(arm_qvel[li])
        dq = _safe_jacobian_div(float(disp), jv)
        q_frame[qi] = float(q_frame[qi]) + dq


def _apply_arm_cartesian_arc_step(
    env,
    q_frame: np.ndarray,
    ee_start: np.ndarray,
    *,
    plane: str,
    radius_m: float,
    sweep_rad: float,
    pp: float,
    side: str,
    max_iters: int = PATH_IK_MAX_ITERS,
    pos_tol_m: float = PATH_IK_POS_TOL_M,
) -> None:
    """One EE sample on a circular arc (world-fixed plane) using Jacobian steps toward the target."""
    plane = plane.lower()
    if radius_m <= 1e-9 or abs(sweep_rad) < 1e-9:
        return
    eu, ev = _plane_uv(plane, ee_start)
    cu, cv = eu + radius_m, ev
    theta0 = math.atan2(ev - cv, eu - cu)
    ang = theta0 + sweep_rad * pp
    tu = cu + radius_m * math.cos(ang)
    tv = cv + radius_m * math.sin(ang)
    target = np.array(ee_start, dtype=np.float64, copy=True)
    _set_plane_uv(plane, target, tu, tv)

    ee_body = R_EE_BODY if side == "right" else L_EE_BODY
    arm_qvel = R_ARM_QVEL if side == "right" else L_ARM_QVEL
    rid = env.sim.model.body_name2id(ee_body)
    for _ in range(max_iters):
        env.sim.data.qpos[:] = q_frame
        env.sim.forward()
        cur = env.sim.data.body_xpos[rid].copy()
        d = target - cur
        if float(np.linalg.norm(d)) < pos_tol_m:
            break
        _apply_world_xyz_delta_via_jacobian(
            env, q_frame, ee_body=ee_body, arm_qvel=arm_qvel, dx=d[0], dy=d[1], dz=d[2]
        )


def _apply_arm_cartesian_line_step(
    env,
    q_frame: np.ndarray,
    ee_start: np.ndarray,
    *,
    axis: str,
    distance_m: float,
    pp: float,
    side: str,
    max_iters: int = PATH_IK_MAX_ITERS,
    pos_tol_m: float = PATH_IK_POS_TOL_M,
) -> None:
    """One EE sample on a straight world-fixed line using Jacobian increments."""
    axis = str(axis).lower()
    if axis not in CART_AXES or abs(distance_m) <= 1e-9:
        return
    target = np.array(ee_start, dtype=np.float64, copy=True)
    target[CART_AXES[axis]] += float(distance_m) * float(pp)

    ee_body = R_EE_BODY if side == "right" else L_EE_BODY
    arm_qvel = R_ARM_QVEL if side == "right" else L_ARM_QVEL
    rid = env.sim.model.body_name2id(ee_body)
    for _ in range(max_iters):
        env.sim.data.qpos[:] = q_frame
        env.sim.forward()
        cur = env.sim.data.body_xpos[rid].copy()
        d = target - cur
        if float(np.linalg.norm(d)) < pos_tol_m:
            break
        _apply_world_xyz_delta_via_jacobian(
            env, q_frame, ee_body=ee_body, arm_qvel=arm_qvel, dx=d[0], dy=d[1], dz=d[2]
        )


def _sample_base_snap(snaps: list[tuple[float, float, float]], fi: int, total_frames: int):
    """Pick a base (x,y,yaw) sample for movement frame ``fi`` (smooth index mapping)."""
    if not snaps:
        return None
    if len(snaps) == 1:
        return snaps[0]
    u = fi / max(1, total_frames - 1)
    idx = min(max(int(round(u * (len(snaps) - 1))), 0), len(snaps) - 1)
    return snaps[idx]


def tiago_trajectory_track_policy(config: dict) -> dict[str, bool]:
    """
    Which projected trails to draw for Tiago: only tracks for body parts that are
    intentionally driven in this config (movement / pose_to_pose), not passive motion.
    """
    show_head = False
    show_torso = False
    show_ee = False
    for step in config.get("movements") or []:
        st = step.get("type")
        params = step.get("parameters") or {}
        if st == "movement":
            mv = params.get("movement") or {}
            joints_t = tiago_preprocess_movement_joints(list(mv.get("joints") or []))
            for jspec in joints_t:
                jc = canonical_joint_keyword(jspec.get("joint"))
                jname = str(jspec.get("joint", "")).lower()
                axis = str(jspec.get("axis", "")).lower()
                mshape = str(jspec.get("_motion_shape", "line")).lower()
                if jc == "base":
                    continue
                if mshape == "arc" and axis in CART_AXES:
                    show_ee = True
                elif axis in CART_AXES and jname in _JOINT_LOCAL_DOFS:
                    show_ee = True
                elif axis in CART_AXES:
                    show_ee = True
                elif jname == "head":
                    show_head = True
                elif jname == "torso":
                    show_torso = True
                elif jname in ("shoulder", "elbow", "wrist"):
                    show_ee = True
        elif st == "pose_to_pose":
            sp = params.get("start_pose") or {}
            ep = params.get("end_pose") or {}
            if sp.get("head") != ep.get("head"):
                show_head = True
            if sp.get("torso_height") != ep.get("torso_height"):
                show_torso = True
            for k in ("arm_position", "gripper_orientation", "left_arm"):
                if sp.get(k) != ep.get(k):
                    show_ee = True
                    break
        elif st == "path":
            path = (params.get("path") or {})
            mode = str(path.get("mode", "")).lower().strip()
            is_ee = mode == "ee" or (
                mode != "base"
                and path.get("axis") in CART_AXES
                and path.get("distance") is not None
            )
            if is_ee:
                show_ee = True
            else:
                show_torso = True
    return {"show_head": show_head, "show_torso": show_torso, "show_ee": show_ee}


def render_config(
    config: dict,
    env=None,
    camera=CAM,
    *,
    overlay_progress_bar: bool = True,
    progress_bar_style: str = "typed",
    ignore_xyz: bool = False,
    cam_pos_scale: float | None = None,
    cam_fovy: float | None = None,
) -> list[Image.Image]:
    """Render a mobile-manip config dict to a list of PIL frames."""
    own_env = env is None
    if own_env:
        env = _make_env(
            cam_pos_scale=_CAM_POS_SCALE_DEFAULT if cam_pos_scale is None else cam_pos_scale,
            cam_fovy=_CAM_FOVY_DEFAULT if cam_fovy is None else cam_fovy,
        )
    elif cam_pos_scale is not None or cam_fovy is not None:
        if cam_pos_scale is not None:
            env._tiago_cam_pos_scale = float(cam_pos_scale)
        if cam_fovy is not None:
            env._tiago_cam_fovy = float(cam_fovy)
    env.reset()
    # hard_reset reloads XML → wipe cam_pos/fovy; re-apply pull-back framing.
    _apply_frontview_framing(env)
    _disable_debug_overlays(env)
    # Stale IKSolver entries reference pre-reset mujoco data and cause flickery EE paths.
    _TIAGO_ARM_IK_CACHE.pop(id(env), None)

    from adhoc.generation.render_progress_overlay import (
        overlay_simple_progress_bar_on_frames,
        overlay_typed_progress_bar_on_frames,
    )

    movements = config.get("movements", [])
    q = _default_qpos(env)
    frames: list[Image.Image] = []
    step_spans: list[dict[str, int | str]] = []

    def _push_span(step_type: str, span_start: int) -> None:
        if len(frames) > span_start:
            step_spans.append({"type": step_type, "start": span_start, "end": len(frames)})

    # Track base position for path steps
    base_x, base_y, base_yaw = 0.0, 0.0, 0.0
    # Sticky: once a pose sets left_arm=mirror, later movement/path stay bilateral
    # until a pose explicitly sets left_arm to still / an independent arm dir.
    arms_mirrored = False

    for step_idx, step in enumerate(movements):
        stype = step.get("type")
        params = step.get("parameters", {})
        duration = step.get("duration", 1.0)
        n_hold_frames = max(1, int(duration * FPS))
        span_start = len(frames)

        if stype == "hold":
            hold_s = float(duration or params.get("t") or params.get("hold_time") or 1.0)
            n_hold = max(1, int(hold_s * FPS))
            for _ in range(n_hold):
                frames.append(_capture(env, q, camera))
            _push_span("hold", span_start)
            continue

        if stype == "pose":
            pose = params.get("pose", {})
            left_raw = str(pose.get("left_arm", "")).strip().lower()
            if left_raw == "mirror":
                arms_mirrored = True
            elif left_raw:
                # Explicit still / front / in / … — end sticky mirror mode.
                arms_mirrored = False
            # Omitted / empty left_arm (e.g. hold pose:{}): keep prior arms_mirrored.
            speed = float(params.get("speed", 1.0))
            # Empty pose after path/movement = hold end configuration (VLM sometimes emits pose:{}).
            hold_only = not pose or not str(pose.get("arm_position", "")).strip()
            if hold_only:
                q_end = q.copy()
            else:
                # Ensure sticky mirror is written onto the pose used for IK/bank.
                pose_apply = dict(pose)
                if arms_mirrored and str(pose_apply.get("left_arm", "")).strip().lower() != "mirror":
                    pose_apply["left_arm"] = "mirror"
                q_end = _q_from_pose(
                    env,
                    pose_apply,
                    base_x=base_x,
                    base_y=base_y,
                    base_yaw=base_yaw,
                    q_seed=q,
                    ignore_xyz=ignore_xyz,
                )
            if step_idx > 0 and not hold_only:
                # movement/path/pose → pose: joint-space ramp (manipulator execute_cue style)
                start_q = q.copy()
                n_trans = _pose_transition_frames(speed)
                for fi in range(n_trans):
                    t = (fi + 1) / n_trans
                    q_interp = start_q + (q_end - start_q) * t
                    frames.append(_capture(env, q_interp, camera))
            q = q_end.copy()
            for _ in range(n_hold_frames):
                frames.append(_capture(env, q, camera))
            _push_span("pose", span_start)

        elif stype == "movement":
            mv = params.get("movement", {})
            raw_joints = list(mv.get("joints") or [])
            joints = [
                j
                for j in tiago_preprocess_movement_joints(raw_joints)
                if canonical_joint_keyword(j.get("joint")) != "base"
            ]
            if not joints:
                raise ValueError(
                    "movement step has no arm/head/torso joints; "
                    "use type: path with path.mode='base' for locomotion"
                )

            reps = int(mv.get("repetition", 1))
            mv_speed = float(mv.get("speed", 1.0))

            joints_work = joints

            pp_frames = _movement_pp_frames(reps, mv_speed, joints_work)
            pp_frames = _resample_pp_frames_to_duration(pp_frames, float(duration))
            total_frames = len(pp_frames)

            base_q = q.copy()

            cart_elbow: dict[int, dict] = {}
            world_deg_info: dict[int, dict] = {}
            joint_mode: list[str] = []
            for ji, jspec in enumerate(joints_work):
                axis = jspec.get("axis", "")
                jgroup = jspec.get("joint", "")
                mshape = str(jspec.get("_motion_shape") or "line").lower()
                deg = jspec.get("degrees", [0, 0])
                if isinstance(deg, (int, float)):
                    lo_d, hi_d = 0.0, float(deg)
                elif isinstance(deg, (list, tuple)) and len(deg) >= 2:
                    lo_d, hi_d = float(deg[0]), float(deg[1])
                elif isinstance(deg, (list, tuple)) and len(deg) == 1:
                    lo_d, hi_d = 0.0, float(deg[0])
                else:
                    lo_d, hi_d = 0.0, 0.0
                cart_meters = _movement_degrees_are_cartesian_meters(lo_d, hi_d)
                cart_ok = (
                    axis in CART_AXES
                    and jgroup in _JOINT_LOCAL_DOFS
                    and mshape == "line"
                    and cart_meters
                )
                if cart_ok:
                    joint_mode.append("cartesian")
                    side = jspec.get("side", "right")
                    if (
                        arms_mirrored
                        and side == "right"
                        and jgroup in ("shoulder", "elbow", "wrist")
                    ):
                        side = "both"
                    ax_idx = CART_AXES[str(axis).lower()]
                    info: dict[str, tuple] = {}
                    if side == "both":
                        li, jv = _find_best_dof_for_axis(
                            env, base_q, R_EE_BODY, R_ARM_QVEL, ax_idx, jgroup,
                            restrict_to_group=True,
                        )
                        info["_bilateral"] = (li, jv)
                    else:
                        if side == "right":
                            li, jv = _find_best_dof_for_axis(
                                env, base_q, R_EE_BODY, R_ARM_QVEL, ax_idx, jgroup,
                                restrict_to_group=True,
                            )
                            info["right"] = (6 + li, jv)
                        if side == "left":
                            li, jv = _find_best_dof_for_axis(
                                env, base_q, L_EE_BODY, L_ARM_QVEL, ax_idx, jgroup,
                                restrict_to_group=True,
                            )
                            info["left"] = (18 + li, jv)
                    cart_elbow[ji] = info
                elif axis in CART_AXES and jgroup in _JOINT_LOCAL_DOFS and mshape == "arc":
                    joint_mode.append("cart_arc")
                    cart_elbow[ji] = {}
                else:
                    joint_mode.append("joint")
                    # Degree-space arm moves: pick DOF most aligned with world axis;
                    # +deg → +world (forward / left / up).
                    if axis in CART_AXES and jgroup in ("shoulder", "elbow", "wrist"):
                        side = jspec.get("side", "right")
                        if (
                            arms_mirrored
                            and side == "right"
                            and jgroup in ("shoulder", "elbow", "wrist")
                        ):
                            side = "both"
                        ax_idx = CART_AXES[str(axis).lower()]
                        info_d: dict[str, tuple] = {}
                        if side == "both":
                            li, jv = _find_best_dof_for_axis(
                                env, base_q, R_EE_BODY, R_ARM_QVEL, ax_idx, jgroup,
                                restrict_to_group=True,
                            )
                            info_d["_bilateral"] = (li, jv)
                        else:
                            if side == "right":
                                li, jv = _find_best_dof_for_axis(
                                    env, base_q, R_EE_BODY, R_ARM_QVEL, ax_idx, jgroup,
                                    restrict_to_group=True,
                                )
                                info_d["right"] = (6 + li, jv)
                            if side == "left":
                                li, jv = _find_best_dof_for_axis(
                                    env, base_q, L_EE_BODY, L_ARM_QVEL, ax_idx, jgroup,
                                    restrict_to_group=True,
                                )
                                info_d["left"] = (18 + li, jv)
                        world_deg_info[ji] = info_d

            # Sticky mirror promotes local side→both without mutating joints_work;
            # detect bilateral from cart_elbow / arms_mirrored, not raw side field.
            has_bilateral_cart = any(
                joint_mode[ji] == "cartesian"
                and (
                    "_bilateral" in cart_elbow.get(ji, {})
                    or joints_work[ji].get("side") == "both"
                    or (
                        arms_mirrored
                        and joints_work[ji].get("side", "right") == "right"
                        and joints_work[ji].get("joint") in ("shoulder", "elbow", "wrist")
                    )
                )
                for ji in range(len(joints_work))
            )

            q_seed = base_q.copy()
            q_seed[QI_FWD] = base_x
            q_seed[QI_SIDE] = base_y
            q_seed[QI_YAW] = base_yaw
            env.sim.data.qpos[:] = q_seed
            env.sim.forward()
            arc_seed = {
                "right": _eef_grip_site_pos(env, "right"),
                "left": _eef_grip_site_pos(env, "left"),
            }

            q_prev_mov = base_q.copy()
            pp_prev = 0.0

            for fi in range(total_frames):
                pp = pp_frames[min(fi, len(pp_frames) - 1)]

                q_frame = q_prev_mov.copy() if fi > 0 else base_q.copy()
                q_frame[QI_FWD] = base_x
                q_frame[QI_SIDE] = base_y
                q_frame[QI_YAW] = base_yaw

                for ji, jspec in enumerate(joints_work):
                    jname = jspec.get("joint", "")
                    axis = jspec.get("axis", "")
                    side = jspec.get("side", "right")
                    if (
                        arms_mirrored
                        and side == "right"
                        and jname in ("shoulder", "elbow", "wrist")
                    ):
                        side = "both"
                    deg = jspec.get("degrees", [0, 0])
                    if isinstance(deg, (int, float)):
                        lo, hi = 0.0, float(deg)
                    elif isinstance(deg, (list, tuple)) and len(deg) >= 2:
                        lo, hi = float(deg[0]), float(deg[1])
                    elif isinstance(deg, (list, tuple)) and len(deg) == 1:
                        lo, hi = 0.0, float(deg[0])
                    else:
                        lo, hi = 0.0, 0.0

                    if joint_mode[ji] == "cartesian":
                        elbow_info = cart_elbow[ji]
                        bilateral = elbow_info.get("_bilateral")
                        if bilateral is not None:
                            li, jv_r = bilateral
                            qi_r = 6 + li
                            lo_rad = _safe_jacobian_div(lo, jv_r)
                            hi_rad = _safe_jacobian_div(hi, jv_r)
                            dq = lo_rad + (hi_rad - lo_rad) * pp
                            q_frame[qi_r] = base_q[qi_r] + dq
                            qi_l = 18 + li
                            m = _bilateral_mirror_sign_for_axis(axis)
                            q_frame[qi_l] = base_q[qi_l] + m * dq
                        else:
                            for _arm_label, (qi, jac_val) in elbow_info.items():
                                lo_rad = _safe_jacobian_div(lo, jac_val)
                                hi_rad = _safe_jacobian_div(hi, jac_val)
                                q_frame[qi] = base_q[qi] + lo_rad + (hi_rad - lo_rad) * pp
                    elif joint_mode[ji] == "cart_arc":
                        # cart_arc applies incremental EE deltas — carry prior frame qpos.
                        if fi == 0:
                            q_frame[:] = base_q
                            q_frame[QI_FWD] = base_x
                            q_frame[QI_SIDE] = base_y
                            q_frame[QI_YAW] = base_yaw
                        plane = _normalize_path_plane(str(jspec.get("plane", "xy")))
                        radius_m = _arc_radius_meters(float(jspec.get("radius", 0.12)))
                        sweep_deg = float(jspec.get("sweep", jspec.get("degrees", 90)))
                        sweep_deg_l = (
                            -sweep_deg
                            if arms_mirrored and side == "both" and _left_arm_path_sweep_mirror(plane)
                            else sweep_deg
                        )
                        if side == "both":
                            _follow_path_arc_progress(
                                env,
                                q_frame,
                                start_pos=arc_seed["right"],
                                plane=plane,
                                radius_m=radius_m,
                                sweep_deg=sweep_deg,
                                pp=pp,
                                pp_prev=pp_prev,
                                side="right",
                            )
                            if arms_mirrored:
                                # Joint-space remirror (not independent left EE IK).
                                _remirror_left_arm_joints(q_frame)
                            else:
                                _follow_path_arc_progress(
                                    env,
                                    q_frame,
                                    start_pos=arc_seed["left"],
                                    plane=plane,
                                    radius_m=radius_m,
                                    sweep_deg=sweep_deg_l,
                                    pp=pp,
                                    pp_prev=pp_prev,
                                    side="left",
                                )
                        else:
                            arm = "right" if side == "right" else "left"
                            _follow_path_arc_progress(
                                env,
                                q_frame,
                                start_pos=arc_seed[arm],
                                plane=plane,
                                radius_m=radius_m,
                                sweep_deg=sweep_deg,
                                pp=pp,
                                pp_prev=pp_prev,
                                side=arm,
                            )
                    else:
                        delta_deg = lo + (hi - lo) * pp
                        if ji in world_deg_info and jname == "wrist":
                            # Wrist world-axis degrees: 2-DOF pad IK → pure tip motion on axis
                            # (single revolute arcs contaminate e.g. z with large Δx).
                            ax_idx = CART_AXES[str(axis).lower()]
                            info_d = world_deg_info[ji]
                            bilateral = info_d.get("_bilateral")
                            if bilateral is not None:
                                _apply_wrist_world_axis_degrees(
                                    env,
                                    q_frame,
                                    base_q=base_q,
                                    side="right",
                                    axis_idx=ax_idx,
                                    delta_deg=delta_deg,
                                )
                                # Mirror: same tip Δ on left (world axis; y flips).
                                m = _bilateral_mirror_sign_for_axis(axis)
                                _apply_wrist_world_axis_degrees(
                                    env,
                                    q_frame,
                                    base_q=base_q,
                                    side="left",
                                    axis_idx=ax_idx,
                                    delta_deg=m * delta_deg,
                                )
                            else:
                                for arm_label in info_d:
                                    _apply_wrist_world_axis_degrees(
                                        env,
                                        q_frame,
                                        base_q=base_q,
                                        side=str(arm_label),
                                        axis_idx=ax_idx,
                                        delta_deg=delta_deg,
                                    )
                        elif ji in world_deg_info:
                            # World-aligned joint degrees: +deg → +world axis on EE.
                            from adhoc.generation.world_axis_motion import (
                                align_offset_to_world_axis,
                            )

                            info_d = world_deg_info[ji]
                            bilateral = info_d.get("_bilateral")
                            if bilateral is not None:
                                li, jv_r = bilateral
                                dq = align_offset_to_world_axis(np.deg2rad(delta_deg), jv_r)
                                q_frame[6 + li] = base_q[6 + li] + dq
                                m = _bilateral_mirror_sign_for_axis(axis)
                                q_frame[18 + li] = base_q[18 + li] + m * dq
                            else:
                                for _arm_label, (qi, jv) in info_d.items():
                                    dq = align_offset_to_world_axis(
                                        np.deg2rad(delta_deg), jv
                                    )
                                    q_frame[qi] = base_q[qi] + dq
                        elif jname == "torso" and axis == "height":
                            # Categorical height (low|mid|high): lerp current → target.
                            # Without this, missing degrees defaulted to [0,0] and the
                            # torso snapped to 0 m (looked like a mid-clip cut).
                            level = str(jspec.get("height", "")).lower()
                            if level == "tall":
                                level = "high"
                            if level in TORSO_MAP:
                                target = float(TORSO_MAP[level])
                                start = float(base_q[QI_TORSO])
                                for qi, _m in _movement_joint_qpos_targets(
                                    jname, axis, side
                                ):
                                    q_frame[qi] = start + (target - start) * pp
                            else:
                                for qi, _m in _movement_joint_qpos_targets(
                                    jname, axis, side
                                ):
                                    q_frame[qi] = lo + (hi - lo) * pp
                        else:
                            use_relative = jname in ("shoulder", "elbow", "wrist", "head")
                            for qi, mirror in _movement_joint_qpos_targets(jname, axis, side):
                                if use_relative:
                                    val = base_q[qi] + mirror * np.deg2rad(delta_deg)
                                else:
                                    val = np.deg2rad(delta_deg)
                                q_frame[qi] = val

                if has_bilateral_cart:
                    _stagger_if_overlapping(env, q_frame)
                _gripper_qpos_keep(base_q, q_frame)
                # World-axis wrist/elbow IK targets are absolute from base_q each
                # frame; clamping vs previous frame crushed ±35° beckon swings
                # (esp. after elbow assist) into a few degrees of crawl.
                if world_deg_info:
                    _clamp_arm_deltas(base_q, q_frame, np.deg2rad(55))
                else:
                    _clamp_arm_deltas(q_prev_mov, q_frame, np.deg2rad(12))
                q_prev_mov = q_frame.copy()
                pp_prev = pp
                frames.append(_capture(env, q_frame, camera))
            # empty pp_frames → no loop body; keep last pose
            q = q_prev_mov.copy()
            hold_time = float(mv.get("hold_time", 0.0) or 0.0)
            for _ in range(max(0, int(hold_time * FPS))):
                frames.append(_capture(env, q, camera))
            _push_span("movement", span_start)

        elif stype == "pose_to_pose":
            n_frames = max(1, int(duration * FPS))
            start_pose = params.get("start_pose", {})
            end_pose = params.get("end_pose", {})
            q_start = _default_qpos(env)
            _apply_pose(env, q_start, start_pose)
            q_start[QI_FWD] = base_x
            q_start[QI_SIDE] = base_y
            q_start[QI_YAW] = base_yaw
            q_end = _default_qpos(env)
            _apply_pose(env, q_end, end_pose)
            q_end[QI_FWD] = base_x
            q_end[QI_SIDE] = base_y
            q_end[QI_YAW] = base_yaw

            for fi in range(n_frames):
                t = fi / max(1, n_frames - 1)
                q_interp = q_start + (q_end - q_start) * t
                frames.append(_capture(env, q_interp, camera))
            q = q_end.copy()
            _push_span("movement", span_start)

        elif stype == "path":
            path = params.get("path", {})
            shape = str(path.get("shape", "line")).lower()
            mode = str(path.get("mode", "")).lower().strip()
            if mode not in ("ee", "base"):
                mode = ""

            # EE path line/arc (manipulator-style): axis+distance or plane+radius+sweep
            if (
                mode == "ee"
                or (
                    not mode
                    and (
                        (shape == "line" and path.get("axis") in CART_AXES and path.get("distance") is not None)
                        or (shape == "arc" and path.get("plane") is not None and path.get("radius") is not None)
                    )
                )
            ):
                speed = float(path.get("speed", 1.0))
                reps = max(1, int(path.get("repetition", 1) or 1))
                side = str(path.get("side", "right"))
                if side not in ("right", "left", "both"):
                    side = "right"
                # Pose left_arm=mirror → subsequent EE paths drive both arms
                # (same rule as movement joints above).
                if arms_mirrored and side == "right":
                    side = "both"

                env.sim.data.qpos[:] = q
                env.sim.forward()
                mirror_bilateral = side == "both" and arms_mirrored
                total_frames = _path_frame_count(path, shape=shape, duration=duration, speed=speed)
                path_anchor_q = q.copy()
                full_circle = shape == "arc" and abs(float(path.get("sweep", path.get("degrees", 360)))) >= 360

                if side == "both":
                    ee_start_r = _eef_body_pos(env, "right")
                    ee_start_l = _eef_body_pos(env, "left")
                else:
                    ee_start_r = _eef_body_pos(env, side)

                plane = _normalize_path_plane(str(path.get("plane", "xy")))
                sweep_deg = float(path.get("sweep", path.get("degrees", 360)))
                sweep_deg_l = (
                    -sweep_deg
                    if mirror_bilateral and _left_arm_path_sweep_mirror(plane)
                    else sweep_deg
                )

                pp_prev = 0.0
                q_carry = path_anchor_q.copy()
                for fi in range(total_frames):
                    q_frame = q_carry.copy()
                    pp = _path_pp_for_frame(
                        fi,
                        total_frames,
                        reps,
                        full_circle=full_circle,
                        linear=(shape == "arc"),
                    )
                    if shape == "line":
                        axis = str(path.get("axis", "x"))
                        distance_raw = path.get("distance", 0.0)
                        if side == "both":
                            if isinstance(distance_raw, (list, tuple)) and len(distance_raw) >= 2:
                                dist_r = distance_raw
                                dist_l = (
                                    [
                                        _bilateral_left_line_distance(
                                            axis,
                                            _line_distance_meters(float(distance_raw[0])),
                                            ee_start_r=ee_start_r,
                                            ee_start_l=ee_start_l,
                                        ),
                                        _bilateral_left_line_distance(
                                            axis,
                                            _line_distance_meters(float(distance_raw[-1])),
                                            ee_start_r=ee_start_r,
                                            ee_start_l=ee_start_l,
                                        ),
                                    ]
                                    if mirror_bilateral
                                    else list(distance_raw)
                                )
                            else:
                                distance_m = _line_distance_meters(float(distance_raw))
                                dist_r = distance_m
                                dist_l = (
                                    _bilateral_left_line_distance(
                                        axis,
                                        distance_m,
                                        ee_start_r=ee_start_r,
                                        ee_start_l=ee_start_l,
                                    )
                                    if mirror_bilateral
                                    else distance_m
                                )
                            _follow_path_line_progress(
                                env,
                                q_frame,
                                start_pos=ee_start_r,
                                axis=axis,
                                distance_m=dist_r,
                                pp=pp,
                                pp_prev=pp_prev,
                                side="right",
                                reps=reps,
                            )
                            if mirror_bilateral:
                                # Remirror after right IK; independent left EE IK
                                # diverges elbow/wrist configs (e.g. depict_big_bowl).
                                pass
                            else:
                                _follow_path_line_progress(
                                    env,
                                    q_frame,
                                    start_pos=ee_start_l,
                                    axis=axis,
                                    distance_m=dist_l,
                                    pp=pp,
                                    pp_prev=pp_prev,
                                    side="left",
                                    reps=reps,
                                )
                        else:
                            _follow_path_line_progress(
                                env,
                                q_frame,
                                start_pos=ee_start_r,
                                axis=axis,
                                distance_m=distance_raw,
                                pp=pp,
                                pp_prev=pp_prev,
                                side=side,
                                reps=reps,
                            )
                    else:
                        radius_m = _arc_radius_meters(float(path.get("radius", 0.12)))
                        if side == "both":
                            _follow_path_arc_progress(
                                env,
                                q_frame,
                                start_pos=ee_start_r,
                                plane=plane,
                                radius_m=radius_m,
                                sweep_deg=sweep_deg,
                                pp=pp,
                                pp_prev=pp_prev,
                                side="right",
                            )
                            if not mirror_bilateral:
                                _follow_path_arc_progress(
                                    env,
                                    q_frame,
                                    start_pos=ee_start_l,
                                    plane=plane,
                                    radius_m=radius_m,
                                    sweep_deg=sweep_deg_l,
                                    pp=pp,
                                    pp_prev=pp_prev,
                                    side="left",
                                )
                        else:
                            _follow_path_arc_progress(
                                env,
                                q_frame,
                                start_pos=ee_start_r,
                                plane=plane,
                                radius_m=radius_m,
                                sweep_deg=sweep_deg,
                                pp=pp,
                                pp_prev=pp_prev,
                                side=side,
                            )

                    _gripper_qpos_keep(path_anchor_q, q_frame)
                    _clamp_arm_deltas(q_carry, q_frame, np.deg2rad(10))
                    if mirror_bilateral:
                        _remirror_left_arm_joints(q_frame)
                    frames.append(_capture(env, q_frame, camera))
                    q_carry = q_frame.copy()
                    pp_prev = pp
                # distance=0 hold-only paths can have total_frames=0 (no loop body).
                q = q_carry.copy()

                hold_time = float(path.get("hold_time", 0.0))
                n_hold = max(0, int(hold_time * FPS))
                for _ in range(n_hold):
                    frames.append(_capture(env, q, camera))
                _push_span("path", span_start)
            else:
                # legacy base path
                snaps, base_x, base_y, base_yaw = _snapshots_for_base_path(base_x, base_y, base_yaw, path)
                for nx, ny, nyaw in snaps:
                    qc = q.copy()
                    qc[QI_FWD] = nx
                    qc[QI_SIDE] = ny
                    qc[QI_YAW] = nyaw
                    frames.append(_capture(env, qc, camera))
                q = qc.copy()
                _push_span("path", span_start)

    if own_env:
        env.close()

    if overlay_progress_bar and frames:
        if progress_bar_style == "typed" and step_spans:
            frames = overlay_typed_progress_bar_on_frames(frames, step_spans)
        else:
            frames = overlay_simple_progress_bar_on_frames(frames)

    return frames


def _right_ee_state(env):
    rid = env.sim.model.body_name2id(R_EE_BODY)
    return (
        env.sim.data.body_xpos[rid].copy(),
        env.sim.data.body_xmat[rid].reshape(3, 3).copy(),
    )


def _tiago_head_torso_state(env):
    """World pose for head (prefer end of kinematic chain) and torso lift link."""
    model = env.sim.model
    head_p = head_r = None
    torso_p = torso_r = None
    for i in range(model.nbody):
        name = model.body_id2name(i) or ""
        low = name.lower()
        if head_p is None and ("head_2" in low or low.endswith("head_2_link")):
            head_p = env.sim.data.body_xpos[i].copy()
            head_r = env.sim.data.body_xmat[i].reshape(3, 3).copy()
        if torso_p is None and "torso_lift" in low and "link" in low:
            torso_p = env.sim.data.body_xpos[i].copy()
            torso_r = env.sim.data.body_xmat[i].reshape(3, 3).copy()
    if head_p is None:
        for i in range(model.nbody):
            name = (model.body_id2name(i) or "").lower()
            if "head" in name and "link" in name:
                head_p = env.sim.data.body_xpos[i].copy()
                head_r = env.sim.data.body_xmat[i].reshape(3, 3).copy()
                break
    if torso_p is None:
        for i in range(model.nbody):
            name = (model.body_id2name(i) or "").lower()
            if "torso" in name and "lift" in name:
                torso_p = env.sim.data.body_xpos[i].copy()
                torso_r = env.sim.data.body_xmat[i].reshape(3, 3).copy()
                break
    return (head_p, head_r), (torso_p, torso_r)


def render_config_with_trajectory(
    config: dict,
    camera: str = CAM,
    *,
    overlay_progress_bar: bool = True,
    progress_bar_style: str = "typed",
    cam_pos_scale: float = _CAM_POS_SCALE_DEFAULT,
    cam_fovy: float = _CAM_FOVY_DEFAULT,
):
    """Render TIAGo frames; record right EE + head + torso for trajectory overlays."""
    env = _make_env(cam_pos_scale=cam_pos_scale, cam_fovy=cam_fovy)
    trajectory: list[dict] = []
    orig_capture = _capture

    def _capture_tracked(env, q, cam=camera):
        img = orig_capture(env, q, cam)
        pos, rot = _right_ee_state(env)
        (hp, hr), (tp, tr) = _tiago_head_torso_state(env)
        entry = {"pos": pos, "rot": rot}
        if hp is not None:
            entry["track_head"] = {"pos": hp, "rot": hr}
        if tp is not None:
            entry["track_torso"] = {"pos": tp, "rot": tr}
        trajectory.append(entry)
        return img

    mod = sys.modules[__name__]
    setattr(mod, "_capture", _capture_tracked)
    try:
        frames = render_config(
            config,
            env=env,
            camera=camera,
            overlay_progress_bar=overlay_progress_bar,
            progress_bar_style=progress_bar_style,
        )
        cam_id = env.sim.model.camera_name2id(camera)
        cam_pos = env.sim.data.cam_xpos[cam_id].copy()
        cam_rot = env.sim.data.cam_xmat[cam_id].reshape(3, 3).copy()
        fovy = float(env.sim.model.cam_fovy[cam_id])
    finally:
        setattr(mod, "_capture", orig_capture)
        env.close()
    return frames, trajectory, cam_pos, cam_rot, fovy


def main(
    config: str,
    out: str = "mobile_config.gif",
    camera: str = CAM,
):
    """Render a mobile manipulator config JSON to a GIF."""
    with open(config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if isinstance(cfg, list):
        cfg = cfg[0]

    env = _make_env()
    try:
        frames = render_config(cfg, env=env, camera=camera)
    finally:
        env.close()

    if not frames:
        print("No frames rendered.")
        return

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=50, loop=0)
    print(f"Saved {len(frames)} frames → {out}")


if __name__ == "__main__":
    fire.Fire(main)
