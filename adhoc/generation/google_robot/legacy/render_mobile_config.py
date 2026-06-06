#!/usr/bin/env python3
"""Render a mobile manipulator config JSON to a GIF using TIAGo in robosuite.

Usage:
  python render_mobile_config.py --config path/to/config.json --out path/to/output.gif
"""
from __future__ import annotations

import json
import math
import os
import sys

import fire
import numpy as np
from PIL import Image, ImageDraw, ImageFont

local_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if local_root not in sys.path:
    sys.path.insert(0, local_root)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_POSE_BANK_ENTRIES: list | None = None

from adhoc.generation.joint_motion_schema import (  # noqa: E402
    canonical_joint_keyword,
    tiago_preprocess_movement_joints,
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
EE_PATH_M_PER_S_AT_SPEED_1 = 0.12

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
CART_AXES = {"x": 0, "y": 1, "z": 2}


FOLD_RAD = np.array([np.deg2rad(d) for d in ARM_PRESETS["fold"]])

_JOINT_LOCAL_DOFS = {
    "shoulder": [0, 1],
    "elbow":    [2, 3],
    "wrist":    [4, 5],
    "arm":      [0, 1, 2, 3, 4, 5],
}

# side=="both": right-arm delta drives left arm with mirrored sign per local DOF (symmetric bimanual).
_BILATERAL_MIRROR_SIGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float64)


_JAC_MIN_USEFUL = 0.08  # below this, joint-space inverse is ill-posed; expand search to all arm DOFs


def _find_best_dof_for_axis(env, q, ee_body, arm_qvel, axis_idx, joint_group="elbow"):
    """Use Jacobian to find which DOF in *joint_group* moves EE most along *axis_idx*.

    If the best coupling in that group is tiny (near kinematic singularities), scan all six
    arm joints — otherwise Cartesian ``displacement / jac`` explodes (violent spins).
    """
    local_dofs = list(_JOINT_LOCAL_DOFS.get(joint_group, _JOINT_LOCAL_DOFS["elbow"]))
    env.sim.data.qpos[:] = q
    env.sim.forward()
    jacp = env.sim.data.get_body_jacp(ee_body).reshape(3, -1)
    best_local, best_val = local_dofs[0], 0.0
    for li in local_dofs:
        val = jacp[axis_idx, arm_qvel[li]]
        if abs(val) > abs(best_val):
            best_val = val
            best_local = li
    if abs(best_val) < _JAC_MIN_USEFUL:
        best_local, best_val = 0, 0.0
        for li in range(6):
            val = jacp[axis_idx, arm_qvel[li]]
            if abs(val) > abs(best_val):
                best_val = val
                best_local = li
    return best_local, best_val


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


def _make_env():
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
    env.reset()
    try:
        for cname in ["sideview", "birdview"]:
            cid = env.sim.model.camera_name2id(cname)
            env.sim.model.cam_fovy[cid] = 55.0
        # frontview: pull slightly back from the scene origin and widen FOV (same 512×512 aspect).
        cid = env.sim.model.camera_name2id(CAM)
        p = env.sim.model.cam_pos[cid].copy()
        pn = float(np.linalg.norm(p))
        if pn > 1e-6:
            env.sim.model.cam_pos[cid] = p * 1.09
        env.sim.model.cam_fovy[cid] = 58.0
    except Exception:
        pass
    return env


def _default_qpos(env):
    q = env.sim.data.qpos.copy()
    q[:3] = 0
    q[QI_TORSO] = TORSO_MAP["mid"]
    q[QI_HEAD_PAN] = 0
    q[QI_HEAD_TILT] = 0
    q[QI_R_ARM] = [0, -0.3, 0, 0.3, 0, 0]
    q[QI_L_ARM] = [0, 1.57, 0, 0.3, 0, 0]
    return q


def _arm_rad(arm_position: str, gripper: str = "vertical") -> list[float]:
    preset_key = ARM_ALIAS_TO_PRESET.get(arm_position, arm_position)
    arm_deg = ARM_PRESETS.get(preset_key, ARM_PRESETS["down+right"])
    arm_rad = [np.deg2rad(d) for d in arm_deg]
    if arm_deg[2] == 0:
        arm_rad[2] = GRIPPER_MAP.get(gripper, 0.0)
    return arm_rad


def _mirror_offset(l_rad: list[float], arm_position: str) -> None:
    """Offset left arm slightly when mirroring to prevent overlap.

    Shifts shoulder_pitch forward and shoulder_roll inward so the left arm
    sits in front of the right arm, creating natural interleaving.
    Only applies a meaningful offset when the preset has high inward roll
    (arms likely to collide near the torso center).
    """
    arm_deg = ARM_PRESETS.get(arm_position, ARM_PRESETS["down+right"])
    roll_deg = abs(arm_deg[1])
    if roll_deg >= 30:
        l_rad[0] += np.deg2rad(12)
        l_rad[1] += np.deg2rad(8)


def _pose_bank_json_path() -> str:
    return os.environ.get(
        "MOBILE_POSE_BANK_JSON",
        os.path.join(_REPO_ROOT, "data", "seed", "google_robot", "mobile_pose_bank_729.json"),
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
    return _POSE_BANK_ENTRIES


def _torso_forward_world_unit(sim) -> np.ndarray:
    """World-unit vector pointing from torso out the chest (+X local of torso_lift)."""
    tid = sim.model.body_name2id(TORSO_REF_BODY)
    R = sim.data.body_xmat[tid].reshape(3, 3)
    v = np.asarray(R[:, 0], dtype=float).copy()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=float)


def _right_elbow_forward_vs_torso_head_ok(sim) -> bool:
    """False if right elbow (arm_right_4) is posterior to torso/head along chest +X.

    Tune with env: MOBILE_POSE_ELBOW_MARGIN_TORSO, MOBILE_POSE_ELBOW_MARGIN_HEAD,
    MOBILE_POSE_CHEST_FWD_PAD_M (virtual chest plane offset forward from torso_lift).
    """
    margin_torso = float(os.environ.get("MOBILE_POSE_ELBOW_MARGIN_TORSO", "0.055"))
    margin_head = float(os.environ.get("MOBILE_POSE_ELBOW_MARGIN_HEAD", "0.06"))
    chest_fwd_pad = float(os.environ.get("MOBILE_POSE_CHEST_FWD_PAD_M", "0.06"))

    fwd = _torso_forward_world_unit(sim)
    try:
        eid = sim.model.body_name2id(R_ELBOW_BODY)
        tid = sim.model.body_name2id(TORSO_REF_BODY)
        hid = sim.model.body_name2id(HEAD_REF_BODY)
    except Exception:
        return True

    p_e = sim.data.body_xpos[eid]
    p_t = sim.data.body_xpos[tid]
    p_h = sim.data.body_xpos[hid]
    # Virtual chest/front plane: torso origin shifted slightly along outward normal.
    p_front = np.asarray(p_t, dtype=float) + chest_fwd_pad * fwd

    # Too far behind chest plane (+ margin slack).
    if float(np.dot(p_e - p_front, fwd)) < -margin_torso:
        return False
    # Not posterior to head (horizontal depth along same fwd).
    if float(np.dot(p_e - p_h, fwd)) < -margin_head:
        return False
    return True


def _apply_pose_nominal(q, pose: dict) -> None:
    """Arm presets only (used to label dir/orient for pose-bank lookup)."""
    torso = pose.get("torso_height", "mid")
    q[QI_TORSO] = TORSO_MAP.get(torso, 0.18)

    arm = pose.get("arm_position", "down+right")
    grip = pose.get("gripper_orientation", "vertical")
    q[QI_R_ARM] = _arm_rad(arm, grip)

    left = pose.get("left_arm", "still")
    if left == "still":
        q[QI_L_ARM] = L_ARM_REST[:]
    elif left == "mirror":
        l_rad = _arm_rad(arm, grip)
        _mirror_offset(l_rad, arm)
        q[QI_L_ARM] = l_rad
    else:
        q[QI_L_ARM] = _arm_rad(left, grip)

    head = pose.get("head", "center")
    hp = HEAD_PRESETS.get(head, [0, 0])
    q[QI_HEAD_PAN] = hp[0]
    q[QI_HEAD_TILT] = hp[1]


def _resolve_right_arm_rad_from_bank(env, pose: dict) -> list[float]:
    bank = _get_pose_bank_entries()
    arm = pose.get("arm_position", "down+right")
    grip = pose.get("gripper_orientation", "vertical")
    if str(arm).strip().lower() == "still":
        return R_ARM_REST[:]
    nominal = _arm_rad(arm, grip)
    if not bank:
        return nominal

    tx = float(pose.get("x", 50))
    ty = float(pose.get("y", 50))
    tz = float(pose.get("z", 50))

    qn = _default_qpos(env)
    _apply_pose_nominal(qn, pose)
    env.sim.data.qpos[:] = qn
    env.sim.forward()
    from adhoc.generation.google_robot.legacy import render_mobile_manip_schema as sch

    arm_label = str(pose.get("arm_position", "")).strip().lower()
    desired_dir = DISPLAY_DIR_TO_SOURCE.get(arm_label)
    d = desired_dir if desired_dir else sch._dir_six_way_from_anchor(env, qn, cam_name=sch.CAM)
    o = sch._gripper_horizontal_vertical_by_reach_dir(env.sim, d)
    if d == "?" or o == "?":
        return nominal

    cand = [e for e in bank if e.get("dir") == d and e.get("orient") == o]
    if not cand:
        return nominal

    def _dist2(e: dict) -> float:
        return (
            (float(e.get("x", 50)) - tx) ** 2
            + (float(e.get("y", 50)) - ty) ** 2
            + (float(e.get("z", 50)) - tz) ** 2
        )

    def _elbow_acceptable_bank_entry(e: dict) -> bool:
        qc = np.asarray(qn, dtype=float).copy()
        qc[QI_R_ARM] = np.asarray(e["r_arm_rad"], dtype=float)
        env.sim.data.qpos[:] = qc
        env.sim.forward()
        return _right_elbow_forward_vs_torso_head_ok(env.sim)

    elbow_ok = [e for e in cand if _elbow_acceptable_bank_entry(e)]
    cand_pick = elbow_ok if elbow_ok else []

    # No bank pose keeps elbow anterior: fall back to nominal preset (still dir/orient-consistent-ish).
    if not cand_pick:
        env.sim.data.qpos[:] = qn
        env.sim.forward()
        return nominal

    best = min(cand_pick, key=_dist2)
    return [float(x) for x in best["r_arm_rad"]]


def _apply_pose(env, q, pose: dict) -> None:
    torso = pose.get("torso_height", "mid")
    q[QI_TORSO] = TORSO_MAP.get(torso, 0.18)

    arm = pose.get("arm_position", "down+right")
    grip = pose.get("gripper_orientation", "vertical")
    r_res = _resolve_right_arm_rad_from_bank(env, pose)
    q[QI_R_ARM] = r_res

    left = pose.get("left_arm", "still")
    if left == "still" and str(arm).strip().lower() == "still":
        # Explicit bilateral still mode: keep both arms at rest.
        q[QI_R_ARM] = R_ARM_REST[:]
        q[QI_L_ARM] = L_ARM_REST[:]
    elif left == "still":
        q[QI_L_ARM] = L_ARM_REST[:]
    elif left == "mirror":
        l_rad = list(r_res)
        _mirror_offset(l_rad, arm)
        q[QI_L_ARM] = l_rad
    else:
        q[QI_L_ARM] = _arm_rad(left, grip)

    head = pose.get("head", "center")
    hp = HEAD_PRESETS.get(head, [0, 0])
    q[QI_HEAD_PAN] = hp[0]
    q[QI_HEAD_TILT] = hp[1]


def _capture(env, q, camera=CAM):
    """Set pose and render. Pure kinematics: no env.step — stepping with zeros fights OSC
    + contact solver after manual qpos writes (looks like violent jitter / collision pops).
    """
    env.sim.data.qpos[:] = q
    try:
        env.sim.data.qvel[:] = 0.0
    except Exception:
        pass
    env.sim.forward()
    frame = env.sim.render(camera_name=camera, width=W, height=H, depth=False)
    return Image.fromarray(np.flipud(frame))


def _pingpong(t01: float) -> float:
    """Map [0,1] to triangle wave [0→1→0]."""
    return 1.0 - abs(2.0 * t01 - 1.0)


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
        duration = max(0.5, total_dist / max(0.1, speed * 0.5))
        n_frames = max(1, int(duration * FPS))
        for fi in range(n_frames):
            t = fi / max(1, n_frames - 1)
            snapshots.append((bx + x_dist * t, by + y_dist * t, yaw))
        bx += x_dist
        by += y_dist
    elif shape == "arc":
        radius = float(path.get("radius", 0.0))
        degrees = float(path.get("degrees", 90))
        total_rad = math.radians(degrees)
        if radius == 0:
            duration = max(0.5, abs(total_rad) / max(0.1, speed * 1.0))
        else:
            arc_len = abs(radius * total_rad)
            duration = max(0.5, arc_len / max(0.1, speed * 0.5))
        n_frames = max(1, int(duration * FPS))
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
    max_iters: int = 12,
    pos_tol_m: float = 0.003,
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
    max_iters: int = 12,
    pos_tol_m: float = 0.003,
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
    return {"show_head": show_head, "show_torso": show_torso, "show_ee": show_ee}


def render_config(
    config: dict,
    env=None,
    camera=CAM,
    *,
    overlay_progress_bar: bool = True,
    progress_bar_style: str = "typed",
) -> list[Image.Image]:
    """Render a mobile-manip config dict to a list of PIL frames."""
    own_env = env is None
    if own_env:
        env = _make_env()
    env.reset()

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

    for step in movements:
        stype = step.get("type")
        params = step.get("parameters", {})
        duration = step.get("duration", 1.0)
        n_frames = max(1, int(duration * FPS))
        span_start = len(frames)

        if stype == "pose":
            pose = params.get("pose", {})
            _apply_pose(env, q, pose)
            q[QI_FWD] = base_x
            q[QI_SIDE] = base_y
            q[QI_YAW] = base_yaw
            for _ in range(n_frames):
                frames.append(_capture(env, q, camera))
            _push_span("pose", span_start)

        elif stype == "movement":
            mv = params.get("movement", {})
            raw_joints = list(mv.get("joints") or [])
            joints = tiago_preprocess_movement_joints(raw_joints)

            base_specs = [
                j for j in joints if canonical_joint_keyword(j.get("joint")) == "base" and isinstance(j.get("path"), dict)
            ]
            arm_joints = [j for j in joints if j not in base_specs]

            if base_specs and not arm_joints:
                bx, by, yaw = base_x, base_y, base_yaw
                for bspec in base_specs:
                    snaps, bx, by, yaw = _snapshots_for_base_path(bx, by, yaw, bspec["path"])
                    for nx, ny, nyaw in snaps:
                        qc = q.copy()
                        qc[QI_FWD] = nx
                        qc[QI_SIDE] = ny
                        qc[QI_YAW] = nyaw
                        frames.append(_capture(env, qc, camera))
                base_x, base_y, base_yaw = bx, by, yaw
                _push_span("path", span_start)
                continue

            reps = mv.get("repetition", 1)
            total_cycles = reps
            base_snapshots: list[tuple[float, float, float]] = []
            bx_prog, by_prog, yaw_prog = base_x, base_y, base_yaw
            for bspec in base_specs:
                seg, bx_prog, by_prog, yaw_prog = _snapshots_for_base_path(
                    bx_prog, by_prog, yaw_prog, bspec["path"]
                )
                base_snapshots.extend(seg)
            nb = len(base_snapshots)

            joints_work = arm_joints

            duration_frames = max(1, int(duration * FPS))
            total_frames = max(duration_frames, nb) if nb else duration_frames

            base_q = q.copy()

            cart_elbow: dict[int, dict] = {}
            joint_mode: list[str] = []
            for ji, jspec in enumerate(joints_work):
                axis = jspec.get("axis", "")
                jgroup = jspec.get("joint", "")
                mshape = str(jspec.get("_motion_shape") or "line").lower()
                cart_ok = axis in CART_AXES and jgroup in _JOINT_LOCAL_DOFS and mshape == "line"
                if cart_ok:
                    joint_mode.append("cartesian")
                    side = jspec.get("side", "right")
                    ax_idx = CART_AXES[str(axis).lower()]
                    info: dict[str, tuple] = {}
                    if side == "both":
                        li, jv = _find_best_dof_for_axis(
                            env, base_q, R_EE_BODY, R_ARM_QVEL, ax_idx, jgroup
                        )
                        info["_bilateral"] = (li, jv)
                    else:
                        if side == "right":
                            li, jv = _find_best_dof_for_axis(
                                env, base_q, R_EE_BODY, R_ARM_QVEL, ax_idx, jgroup
                            )
                            info["right"] = (6 + li, jv)
                        if side == "left":
                            li, jv = _find_best_dof_for_axis(
                                env, base_q, L_EE_BODY, L_ARM_QVEL, ax_idx, jgroup
                            )
                            sign = -1.0 if ax_idx == 1 else 1.0
                            info["left"] = (18 + li, jv * sign)
                    cart_elbow[ji] = info
                elif axis in CART_AXES and jgroup in _JOINT_LOCAL_DOFS and mshape == "arc":
                    joint_mode.append("cart_arc")
                    cart_elbow[ji] = {}
                else:
                    joint_mode.append("joint")

            has_bilateral_cart = any(
                joint_mode[ji] == "cartesian" and joints_work[ji].get("side") == "both"
                for ji in range(len(joints_work))
            )

            q_seed = base_q.copy()
            sm0 = _sample_base_snap(base_snapshots, 0, total_frames)
            if sm0 is not None:
                q_seed[QI_FWD], q_seed[QI_SIDE], q_seed[QI_YAW] = sm0
            else:
                q_seed[QI_FWD] = base_x
                q_seed[QI_SIDE] = base_y
                q_seed[QI_YAW] = base_yaw
            env.sim.data.qpos[:] = q_seed
            env.sim.forward()
            arc_seed = {
                "right": env.sim.data.body_xpos[env.sim.model.body_name2id(R_EE_BODY)].copy(),
                "left": env.sim.data.body_xpos[env.sim.model.body_name2id(L_EE_BODY)].copy(),
            }

            for fi in range(total_frames):
                t_global = fi / max(1, total_frames - 1)
                if total_cycles == 1:
                    pp = t_global
                else:
                    cycle_t = (t_global * total_cycles) % 1.0
                    pp = _pingpong(cycle_t)

                q_frame = base_q.copy()
                sm = _sample_base_snap(base_snapshots, fi, total_frames)
                if sm is not None:
                    q_frame[QI_FWD], q_frame[QI_SIDE], q_frame[QI_YAW] = sm
                else:
                    q_frame[QI_FWD] = base_x
                    q_frame[QI_SIDE] = base_y
                    q_frame[QI_YAW] = base_yaw

                for ji, jspec in enumerate(joints_work):
                    jname = jspec.get("joint", "")
                    axis = jspec.get("axis", "")
                    side = jspec.get("side", "right")
                    deg = jspec.get("degrees", [0, 0])
                    if isinstance(deg, (int, float)):
                        lo, hi = 0.0, float(deg)
                    else:
                        lo, hi = float(deg[0]), float(deg[1])

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
                            m = float(_BILATERAL_MIRROR_SIGN[li])
                            q_frame[qi_l] = base_q[qi_l] + m * dq
                        else:
                            for _arm_label, (qi, jac_val) in elbow_info.items():
                                lo_rad = _safe_jacobian_div(lo, jac_val)
                                hi_rad = _safe_jacobian_div(hi, jac_val)
                                q_frame[qi] = base_q[qi] + lo_rad + (hi_rad - lo_rad) * pp
                    elif joint_mode[ji] == "cart_arc":
                        if side == "both":
                            continue
                        plane = str(jspec.get("plane", "xy")).lower()
                        radius_m = float(jspec.get("radius", 0.12))
                        sweep_deg = float(jspec.get("sweep", jspec.get("degrees", 90)))
                        sweep_rad = math.radians(sweep_deg)
                        ee0 = arc_seed["right" if side == "right" else "left"]
                        _apply_arm_cartesian_arc_step(
                            env,
                            q_frame,
                            ee0,
                            plane=plane,
                            radius_m=radius_m,
                            sweep_rad=sweep_rad,
                            pp=pp,
                            side=str(side),
                        )
                    else:
                        if side == "both":
                            qr = JOINT_QPOS_MAP.get((jname, axis))
                            ql = L_JOINT_QPOS_MAP.get((jname, axis))
                            if jname == "torso" and axis == "height" and qr is not None:
                                q_frame[qr] = lo + (hi - lo) * pp
                            elif qr is not None and ql is not None and 6 <= qr < 12 and 18 <= ql < 24:
                                val_r = np.deg2rad(lo + (hi - lo) * pp)
                                dq = val_r - base_q[qr]
                                i = int(qr - 6)
                                q_frame[qr] = base_q[qr] + dq
                                q_frame[ql] = base_q[ql] + float(_BILATERAL_MIRROR_SIGN[i]) * dq
                            else:
                                for qi in (qr, ql):
                                    if qi is None:
                                        continue
                                    if jname == "torso" and axis == "height":
                                        val = lo + (hi - lo) * pp
                                    else:
                                        val = np.deg2rad(lo + (hi - lo) * pp)
                                    q_frame[qi] = val
                        elif side == "left":
                            qis = [L_JOINT_QPOS_MAP.get((jname, axis))]
                            for qi in qis:
                                if qi is None:
                                    continue
                                if jname == "torso" and axis == "height":
                                    val = lo + (hi - lo) * pp
                                else:
                                    val = np.deg2rad(lo + (hi - lo) * pp)
                                q_frame[qi] = val
                        else:
                            qis = [JOINT_QPOS_MAP.get((jname, axis))]
                            for qi in qis:
                                if qi is None:
                                    continue
                                if jname == "torso" and axis == "height":
                                    val = lo + (hi - lo) * pp
                                else:
                                    val = np.deg2rad(lo + (hi - lo) * pp)
                                q_frame[qi] = val

                if has_bilateral_cart:
                    _stagger_if_overlapping(env, q_frame)
                _gripper_qpos_keep(base_q, q_frame)
                _clamp_arm_deltas(base_q, q_frame, np.deg2rad(95))
                frames.append(_capture(env, q_frame, camera))
            q = q_frame.copy()
            if base_specs:
                base_x, base_y, base_yaw = bx_prog, by_prog, yaw_prog
            _push_span("movement", span_start)

        elif stype == "pose_to_pose":
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
                if side not in ("right", "left"):
                    side = "right"

                env.sim.data.qpos[:] = q
                env.sim.forward()
                rid = env.sim.model.body_name2id(R_EE_BODY if side == "right" else L_EE_BODY)
                ee_start = env.sim.data.body_xpos[rid].copy()
                total_frames = max(1, int(duration * FPS))
                if shape == "line":
                    distance_m = float(path.get("distance", 0.0))
                    total_frames = max(
                        total_frames,
                        max(12, int(abs(distance_m) / max(0.01, speed * EE_PATH_M_PER_S_AT_SPEED_1) * FPS)),
                    )
                else:
                    radius_m = float(path.get("radius", 0.12))
                    sweep_deg = float(path.get("sweep", path.get("degrees", 360)))
                    arc_len = abs(radius_m * math.radians(sweep_deg))
                    total_frames = max(
                        total_frames,
                        max(12, int(arc_len / max(0.01, speed * EE_PATH_M_PER_S_AT_SPEED_1) * FPS)),
                        max(24, int(abs(sweep_deg) / 12.0)),
                    )

                path_anchor_q = q.copy()
                q_frame = q.copy()
                sweep_rad = math.radians(float(path.get("sweep", path.get("degrees", 360))))
                for fi in range(total_frames):
                    t_global = fi / max(1, total_frames - 1)
                    if reps == 1:
                        pp = t_global
                    else:
                        cycle_t = (t_global * reps) % 1.0
                        pp = _pingpong(cycle_t)

                    q_before = q_frame.copy()
                    if shape == "line":
                        _apply_arm_cartesian_line_step(
                            env,
                            q_frame,
                            ee_start,
                            axis=str(path.get("axis", "x")),
                            distance_m=float(path.get("distance", 0.0)),
                            pp=pp,
                            side=side,
                        )
                    else:
                        _apply_arm_cartesian_arc_step(
                            env,
                            q_frame,
                            ee_start,
                            plane=str(path.get("plane", "xy")),
                            radius_m=float(path.get("radius", 0.12)),
                            sweep_rad=sweep_rad,
                            pp=pp,
                            side=side,
                        )
                    _gripper_qpos_keep(path_anchor_q, q_frame)
                    _clamp_arm_deltas(q_before, q_frame, np.deg2rad(35))
                    frames.append(_capture(env, q_frame, camera))
                q = q_frame.copy()

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
):
    """Render TIAGo frames; record right EE + head + torso for trajectory overlays."""
    env = _make_env()
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
