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
}

GRIPPER_MAP = {"horizontal": 1.57, "vertical": 0.0}

L_ARM_REST = [0, 1.57, 0, 0.3, 0, 0]

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


def _find_best_dof_for_axis(env, q, ee_body, arm_qvel, axis_idx, joint_group="elbow"):
    """Use Jacobian to find which DOF in *joint_group* moves EE most along *axis_idx*.

    Returns (local_idx, jac_value).
    """
    local_dofs = _JOINT_LOCAL_DOFS.get(joint_group, _JOINT_LOCAL_DOFS["elbow"])
    env.sim.data.qpos[:] = q
    env.sim.forward()
    jacp = env.sim.data.get_body_jacp(ee_body).reshape(3, -1)
    best_local, best_val = local_dofs[0], 0.0
    for li in local_dofs:
        val = jacp[axis_idx, arm_qvel[li]]
        if abs(val) > abs(best_val):
            best_val = val
            best_local = li
    return best_local, best_val


_OVERLAP_THRESHOLD = 0.08  # 8 cm


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
        for cname in [CAM, "sideview", "birdview"]:
            cid = env.sim.model.camera_name2id(cname)
            env.sim.model.cam_fovy[cid] = 55.0
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
    arm_deg = ARM_PRESETS.get(arm_position, ARM_PRESETS["down+right"])
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


def _apply_pose(q, pose: dict):
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


def _capture(env, q, camera=CAM):
    env.sim.data.qpos[:] = q
    env.sim.forward()
    env.step(np.zeros(env.action_dim))
    frame = env.sim.render(camera_name=camera, width=W, height=H, depth=False)
    return Image.fromarray(np.flipud(frame))


def _pingpong(t01: float) -> float:
    """Map [0,1] to triangle wave [0→1→0]."""
    return 1.0 - abs(2.0 * t01 - 1.0)


def render_config(config: dict, env=None, camera=CAM) -> list[Image.Image]:
    """Render a mobile-manip config dict to a list of PIL frames."""
    own_env = env is None
    if own_env:
        env = _make_env()
    env.reset()

    movements = config.get("movements", [])
    q = _default_qpos(env)
    frames: list[Image.Image] = []

    # Track base position for path steps
    base_x, base_y, base_yaw = 0.0, 0.0, 0.0

    for step in movements:
        stype = step.get("type")
        params = step.get("parameters", {})
        duration = step.get("duration", 1.0)
        n_frames = max(1, int(duration * FPS))

        if stype == "pose":
            pose = params.get("pose", {})
            _apply_pose(q, pose)
            q[QI_FWD] = base_x
            q[QI_SIDE] = base_y
            q[QI_YAW] = base_yaw
            for _ in range(n_frames):
                frames.append(_capture(env, q, camera))

        elif stype == "movement":
            mv = params.get("movement", {})
            joints = mv.get("joints", [])
            reps = mv.get("repetition", 1)
            speed = mv.get("speed", 1.0)

            base_q = q.copy()
            total_cycles = reps
            total_frames = n_frames

            # Pre-resolve Cartesian joints: Jacobian picks best DOF in the specified joint group
            cart_elbow: dict[int, dict] = {}
            joint_mode: list[str] = []
            for ji, jspec in enumerate(joints):
                axis = jspec.get("axis", "")
                jgroup = jspec.get("joint", "")
                if axis in CART_AXES and jgroup in _JOINT_LOCAL_DOFS:
                    joint_mode.append("cartesian")
                    side = jspec.get("side", "right")
                    ax_idx = CART_AXES[axis]
                    info: dict[str, tuple] = {}
                    if side in ("right", "both"):
                        li, jv = _find_best_dof_for_axis(env, base_q, R_EE_BODY, R_ARM_QVEL, ax_idx, jgroup)
                        info["right"] = (6 + li, jv)
                    if side in ("left", "both"):
                        li, jv = _find_best_dof_for_axis(env, base_q, L_EE_BODY, L_ARM_QVEL, ax_idx, jgroup)
                        sign = -1.0 if ax_idx == 1 else 1.0
                        info["left"] = (18 + li, jv * sign)
                    cart_elbow[ji] = info
                else:
                    joint_mode.append("joint")

            has_bilateral_cart = any(
                joint_mode[ji] == "cartesian" and joints[ji].get("side") == "both"
                for ji in range(len(joints))
            )

            for fi in range(total_frames):
                t_global = fi / max(1, total_frames - 1)
                if total_cycles == 1:
                    pp = t_global
                else:
                    cycle_t = (t_global * total_cycles) % 1.0
                    pp = _pingpong(cycle_t)

                q_frame = base_q.copy()
                q_frame[QI_FWD] = base_x
                q_frame[QI_SIDE] = base_y
                q_frame[QI_YAW] = base_yaw

                for ji, jspec in enumerate(joints):
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
                        _MIN_JAC = 0.01
                        for arm_label, (qi, jac_val) in elbow_info.items():
                            if abs(jac_val) >= _MIN_JAC:
                                lo_rad = lo / jac_val
                                hi_rad = hi / jac_val
                            else:
                                lo_rad = lo * 10.0
                                hi_rad = hi * 10.0
                            q_frame[qi] = base_q[qi] + lo_rad + (hi_rad - lo_rad) * pp
                    else:
                        if side == "both":
                            qis = [JOINT_QPOS_MAP.get((jname, axis)),
                                   L_JOINT_QPOS_MAP.get((jname, axis))]
                        elif side == "left":
                            qis = [L_JOINT_QPOS_MAP.get((jname, axis))]
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
                frames.append(_capture(env, q_frame, camera))
            q = q_frame.copy()

        elif stype == "pose_to_pose":
            start_pose = params.get("start_pose", {})
            end_pose = params.get("end_pose", {})
            q_start = _default_qpos(env)
            _apply_pose(q_start, start_pose)
            q_start[QI_FWD] = base_x
            q_start[QI_SIDE] = base_y
            q_start[QI_YAW] = base_yaw
            q_end = _default_qpos(env)
            _apply_pose(q_end, end_pose)
            q_end[QI_FWD] = base_x
            q_end[QI_SIDE] = base_y
            q_end[QI_YAW] = base_yaw

            for fi in range(n_frames):
                t = fi / max(1, n_frames - 1)
                q_interp = q_start + (q_end - q_start) * t
                frames.append(_capture(env, q_interp, camera))
            q = q_end.copy()

        elif stype == "path":
            path = params.get("path", {})
            shape = path.get("shape", "line")
            speed = path.get("speed", 1.0)

            if shape == "line":
                x_dist = path.get("x", 0.0)
                y_dist = path.get("y", 0.0)
                total_dist = math.sqrt(x_dist**2 + y_dist**2)
                duration = max(0.5, total_dist / max(0.1, speed * 0.5))
                n_frames = max(1, int(duration * FPS))

                for fi in range(n_frames):
                    t = fi / max(1, n_frames - 1)
                    q[QI_FWD] = base_x + x_dist * t
                    q[QI_SIDE] = base_y + y_dist * t
                    q[QI_YAW] = base_yaw
                    frames.append(_capture(env, q, camera))

                base_x += x_dist
                base_y += y_dist

            elif shape == "arc":
                radius = path.get("radius", 0.0)
                degrees = path.get("degrees", 90)
                total_rad = math.radians(degrees)
                if radius == 0:
                    duration = max(0.5, abs(total_rad) / max(0.1, speed * 1.0))
                else:
                    arc_len = abs(radius * total_rad)
                    duration = max(0.5, arc_len / max(0.1, speed * 0.5))
                n_frames = max(1, int(duration * FPS))

                for fi in range(n_frames):
                    t = fi / max(1, n_frames - 1)
                    angle = total_rad * t
                    if radius == 0:
                        q[QI_FWD] = base_x
                        q[QI_SIDE] = base_y
                    else:
                        q[QI_FWD] = base_x + radius * math.sin(angle)
                        q[QI_SIDE] = base_y + radius * (1 - math.cos(angle))
                    q[QI_YAW] = base_yaw + angle
                    frames.append(_capture(env, q, camera))

                base_yaw += total_rad
                if radius != 0:
                    base_x += radius * math.sin(total_rad)
                    base_y += radius * (1 - math.cos(total_rad))

    if own_env:
        env.close()

    return frames


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
