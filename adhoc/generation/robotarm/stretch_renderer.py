#!/usr/bin/env python3
"""Hello Robot Stretch 3 Motion Code renderer (mujoco_menagerie MJCF)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from embodiment_renderers import (  # noqa: E402
    _progress_overlay,
    _roomba_path_n_frames,
    _roomba_pingpong,
    _roomba_span,
    _save_media,
)
from motion_speed import beat_n_frames  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_STRETCH_SCENE = (
    _REPO / "data/seed/stretch/hello_robot_stretch_3/motion_code_scene.xml"
)
_EE_BODY = "link_gripper_finger_left"

# Model arm extends along local −y; rotate base so arm points +x (toward viewer).
_BASE_YAW0 = math.pi / 2.0

_LIFT = {"low": 0.28, "mid": 0.55, "high": 0.90, "tall": 0.90, "short": 0.28}
_HEAD = {
    "center": (0.0, 0.0),
    "left": (0.75, 0.0),
    "right": (-0.75, 0.0),
    "up": (0.0, -0.45),
    "down": (0.0, 0.40),
}
# dir → (lift, arm_ext, wrist_yaw, wrist_pitch, wrist_roll)
# Horizontal facing: out / front / in are exact 90° wrist_yaw steps
# (out = −π/2, front = 0, in = +π/2). Keep the same reach so only yaw differs.
_DIR = {
    "still": (0.50, 0.06, 0.0, 0.0, 0.0),
    "front": (0.58, 0.36, 0.0, 0.0, 0.0),
    "back": (0.50, 0.10, math.pi, -0.35, 0.0),
    "in": (0.58, 0.36, math.pi / 2.0, 0.0, 0.0),
    "out": (0.58, 0.36, -math.pi / 2.0, 0.0, 0.0),
    "up": (0.95, 0.30, 0.0, 0.45, 0.0),
    "down": (0.30, 0.35, 0.0, -1.20, 0.0),
}


def _qadr(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(name)
    return int(model.jnt_qposadr[jid])


def _pose_from_dict(pose: dict[str, Any]) -> dict[str, float]:
    dir_ = str(pose.get("arm_position", pose.get("dir", "still"))).lower()
    lift, arm, wy, wp, wr = _DIR.get(dir_, _DIR["still"])
    torso = str(pose.get("torso_height", pose.get("torso", ""))).lower()
    if torso in _LIFT:
        lift = _LIFT[torso]
    head = str(pose.get("head", "center")).lower()
    hpan, htilt = _HEAD.get(head, _HEAD["center"])
    grip = str(pose.get("gripper_orientation", "vertical")).lower()
    if grip in ("horizontal", "horiz", "open"):
        wr = 1.55
    if pose.get("elbow_bended") is True or str(pose.get("elbow", "")).upper() in (
        "T",
        "TRUE",
    ):
        wp = min(wp, -0.6)
    elif pose.get("elbow_bended") is False or str(pose.get("elbow", "")).upper() in (
        "F",
        "FALSE",
    ):
        wp = max(wp, -0.15)
    # Percentiles: z→lift, x→arm, y→wrist_yaw
    if pose.get("z") is not None:
        z = float(pose["z"]) / 100.0
        lift = 0.20 + z * 0.85
    if pose.get("x") is not None:
        x = float(pose["x"]) / 100.0
        arm = 0.04 + x * 0.48
    if pose.get("y") is not None:
        y = float(pose["y"]) / 100.0
        wy = wy + (y - 0.5) * 1.6
    return {
        "lift": float(np.clip(lift, 0.0, 1.1)),
        "arm": float(np.clip(arm, 0.0, 0.52)),
        "wrist_yaw": float(np.clip(wy, -math.pi / 2.0, 4.42)),
        "wrist_pitch": float(np.clip(wp, -1.57, 0.56)),
        "wrist_roll": float(wr),
        "head_pan": float(hpan),
        "head_tilt": float(htilt),
        "grip": 0.0,
    }


def _apply(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    x: float,
    y: float,
    yaw: float,
    joints: dict[str, float],
) -> None:
    data.qpos[:] = 0
    data.qvel[:] = 0
    # Freejoint: z≈0 (wheels on floor). Yaw includes model→viewer offset.
    yaw_w = yaw + _BASE_YAW0
    data.qpos[0:3] = (x, y, 0.0)
    data.qpos[3:7] = (math.cos(yaw_w / 2), 0, 0, math.sin(yaw_w / 2))
    data.qpos[_qadr(model, "joint_lift")] = joints["lift"]
    seg = joints["arm"] / 4.0
    for n in ("joint_arm_l0", "joint_arm_l1", "joint_arm_l2", "joint_arm_l3"):
        data.qpos[_qadr(model, n)] = seg
    data.qpos[_qadr(model, "joint_wrist_yaw")] = joints["wrist_yaw"]
    data.qpos[_qadr(model, "joint_wrist_pitch")] = joints["wrist_pitch"]
    data.qpos[_qadr(model, "joint_wrist_roll")] = joints["wrist_roll"]
    data.qpos[_qadr(model, "joint_gripper_slide")] = joints.get("grip", 0.0)
    data.qpos[_qadr(model, "joint_head_pan")] = joints["head_pan"]
    data.qpos[_qadr(model, "joint_head_tilt")] = joints["head_tilt"]
    mujoco.mj_forward(model, data)


def _ee(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _EE_BODY)
    return np.array(data.xpos[bid], dtype=float)


# Soft clip: allow full −90° out (stock MJCF lower was −1.39 ≈ −80°).
_JLO = {"lift": 0.0, "arm": 0.0, "wrist_yaw": -math.pi / 2.0, "wrist_pitch": -1.57, "wrist_roll": -3.14}
_JHI = {"lift": 1.1, "arm": 0.52, "wrist_yaw": 4.42, "wrist_pitch": 0.56, "wrist_roll": 3.14}


def _clamp_joints(j: dict[str, float]) -> dict[str, float]:
    out = dict(j)
    for k, lo in _JLO.items():
        if k in out:
            out[k] = float(np.clip(out[k], lo, _JHI[k]))
    return out


def _ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    target: np.ndarray,
    x: float,
    y: float,
    yaw: float,
    joints: dict[str, float],
    iters: int = 12,
    max_step: dict[str, float] | None = None,
) -> dict[str, float]:
    """Damped finite-diff IK on lift / arm / wrist_yaw (no huge joint snaps)."""
    keys = ("lift", "arm", "wrist_yaw")
    step_lim = max_step or {"lift": 0.06, "arm": 0.04, "wrist_yaw": 0.25}
    j = _clamp_joints(joints)
    seed = dict(j)
    _apply(model, data, x=x, y=y, yaw=yaw, joints=j)
    best_j = dict(j)
    best_err = float(np.linalg.norm(target - _ee(model, data)))
    damp = 1e-2
    for _ in range(iters):
        _apply(model, data, x=x, y=y, yaw=yaw, joints=j)
        err = target - _ee(model, data)
        err_n = float(np.linalg.norm(err))
        if err_n < 0.012:
            break
        jac = np.zeros((3, 3))
        eps = 1e-3
        base = _ee(model, data)
        for col, k in enumerate(keys):
            j2 = dict(j)
            j2[k] = float(np.clip(j2[k] + eps, _JLO[k], _JHI[k]))
            _apply(model, data, x=x, y=y, yaw=yaw, joints=j2)
            jac[:, col] = (_ee(model, data) - base) / eps
        # Damped least squares: (JᵀJ + λI)⁻¹ Jᵀ e
        try:
            jtj = jac.T @ jac + damp * np.eye(3)
            dq = np.linalg.solve(jtj, jac.T @ err)
        except np.linalg.LinAlgError:
            break
        cand = dict(j)
        for col, k in enumerate(keys):
            d = float(np.clip(0.5 * dq[col], -step_lim[k], step_lim[k]))
            # Also cap total drift from the frame seed (stops singularity flips).
            total_lim = step_lim[k] * 3.0
            cand[k] = float(
                np.clip(
                    np.clip(j[k] + d, seed[k] - total_lim, seed[k] + total_lim),
                    _JLO[k],
                    _JHI[k],
                )
            )
        _apply(model, data, x=x, y=y, yaw=yaw, joints=cand)
        cand_err = float(np.linalg.norm(target - _ee(model, data)))
        if cand_err < best_err - 1e-4:
            j, best_j, best_err = cand, dict(cand), cand_err
        else:
            damp *= 2.0
            if damp > 1.0:
                break
    return best_j


def _arc_joints_analytic(
    seed: dict[str, float],
    *,
    plane: str,
    radius: float,
    ang: float,
) -> dict[str, float]:
    """Parametric joint-space circle (sin/cos). Avoid linear yaw+=ang (hits limits → ping-pong)."""
    j = dict(seed)
    # Keep radii small — Stretch wrist/telescope travel is limited.
    r = float(np.clip(abs(radius), 0.0, 0.10))
    s, c = math.sin(ang), math.cos(ang)
    if plane == "xy":
        # Horizontal circle ≈ wrist yaw ↔ slight telescope (arm reach chord).
        amp_yaw = 0.35 + 2.2 * r
        amp_arm = 0.55 * r
        j["wrist_yaw"] = seed["wrist_yaw"] + amp_yaw * s
        j["arm"] = seed["arm"] + amp_arm * c
    elif plane == "xz":
        # Reach × lift ellipse in the sagittal plane.
        amp_arm = max(0.04, r)
        amp_lift = max(0.04, r)
        j["arm"] = seed["arm"] + amp_arm * s
        j["lift"] = seed["lift"] + amp_lift * c
        j["wrist_pitch"] = seed["wrist_pitch"] + 0.15 * s
    else:  # yz — signed bowl-ish: lateral yaw + vertical lift (sin)
        # Matches EE arc convention: −sweep dips down (bowl ∪), +sweep rises.
        # Scale with r (no large lift floor) so rub_eye r≈0.02 stays a small circle.
        amp_yaw = 0.12 + 4.0 * r
        amp_lift = max(0.02, 2.2 * r)
        j["wrist_yaw"] = seed["wrist_yaw"] + amp_yaw * (c - 1.0)
        j["lift"] = seed["lift"] + amp_lift * s
    return _clamp_joints(j)


def _line_joints_analytic(
    seed: dict[str, float],
    *,
    axis: str,
    dist: float,
    frac: float,
) -> dict[str, float]:
    """Joint-space EE line. frac in [0,1] (or ping-pong) along signed dist."""
    j = dict(seed)
    span = float(np.clip(dist, -0.22, 0.22))
    delta = span * float(frac)
    if axis == "x":
        j["arm"] = seed["arm"] + delta
    elif axis == "y":
        # Lateral ≈ wrist yaw only (do not telescope — z/x own those DOFs).
        lever = max(0.22, float(seed["arm"]) + 0.18)
        j["wrist_yaw"] = seed["wrist_yaw"] + delta / lever
    else:  # z — mast lift only (same DOF as move(shoulder, z, …))
        j["lift"] = seed["lift"] + delta
    return _clamp_joints(j)


def _lerp_joints(a: dict[str, float], b: dict[str, float], t: float) -> dict[str, float]:
    return {k: a[k] + (b[k] - a[k]) * t for k in a}


def _stretch_states(
    row: dict[str, Any],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    fps: int,
) -> list[dict[str, Any]]:
    x = y = yaw = 0.0
    joints = _pose_from_dict({"dir": "still", "torso": "mid", "head": "center"})
    frames: list[dict[str, Any]] = []

    def snap() -> dict[str, Any]:
        return {"x": x, "y": y, "yaw": yaw, "joints": dict(joints)}

    # No rest-pose warmup: leading pose(...) must already be the first GIF frame
    # (e.g. torso=high / z=80 starts high — never show the lift-up approach).

    def _apply_one(step, x, y, yaw, joints, *, force_n: int | None = None):
        """Apply one movement step; return (frames, x, y, yaw, joints).

        ``force_n``: emit exactly this many motion frames (used for ``order=together``
        so a short lift-bob is sampled as densely as a long base arc). Skips
        trailing settle/hold so the forced window stays pure motion.
        """
        local_frames: list[dict[str, Any]] = []

        def snap() -> dict[str, Any]:
            return {"x": x, "y": y, "yaw": yaw, "joints": dict(joints)}

        kind = step.get("type")
        params = step.get("parameters") or {}
        if kind == "hold":
            hold = float(
                step.get("duration")
                or params.get("t")
                or params.get("hold_time")
                or 1.0
            )
            local_frames.extend([snap()] * max(1, round(hold * fps)))
            return local_frames, x, y, yaw, joints
        if kind == "pose":
            target = _pose_from_dict(params.get("pose") or {})
            # Setup pose before any recorded content: snap (no approach lerp).
            # Later pose→pose transitions still animate (bowl, raise, etc.).
            if not frames:
                joints = dict(target)
                local_frames.extend([snap()] * max(1, fps // 6))
                return local_frames, x, y, yaw, joints
            n = max(4, round(float(step.get("duration") or 0.6) * fps))
            if force_n is not None:
                n = max(2, int(force_n))
            a0 = dict(joints)
            for i in range(n):
                joints = _lerp_joints(a0, target, (i + 1) / n)
                local_frames.append(snap())
            return local_frames, x, y, yaw, joints
        if kind == "movement":
            mv = params.get("movement") or {}
            j0 = (mv.get("joints") or [{}])[0]
            joint = str(j0.get("joint", "wrist")).lower()
            axis = str(j0.get("axis", "y")).lower()
            rep = max(1, int(mv.get("repetition", 1)))
            speed = float(mv.get("speed", 1.0))
            n = beat_n_frames(fps=fps, speed=speed, rep=rep)
            if force_n is not None:
                n = max(2, int(force_n))
            else:
                # Periodic sine with i∈[0,n]: need ≥4 samples/cycle. With FRAME_SCALE
                # 0.5 + high spd, n=2 samples phase 0 and 1 → both sin=0 (frozen bob).
                deg_raw = j0.get("degrees")
                if isinstance(deg_raw, (list, tuple)) and len(deg_raw) >= 2:
                    n = max(n, 4 * rep)
            base = dict(joints)
            if joint == "torso" and "height" in j0:
                # Categorical level: smooth one-shot lerp base→target.
                # (Old sin(πt) went target→base mid-step, then snapped — looked cut.)
                level = str(j0.get("height", "high")).lower()
                target_lift = _LIFT.get(level, 0.90)
                n_level = max(8, round((max(1, rep) / speed) * fps))
                for i in range(n_level):
                    u = (i + 1) / n_level
                    s = u * u * (3.0 - 2.0 * u)  # smoothstep
                    joints = dict(base)
                    joints["lift"] = float(
                        np.clip(
                            base["lift"] + (target_lift - base["lift"]) * s,
                            0.0,
                            1.1,
                        )
                    )
                    local_frames.append(snap())
                joints = dict(base)
                joints["lift"] = float(np.clip(target_lift, 0.0, 1.1))
            elif joint == "torso" and axis == "height":
                # Numeric lift oscillation in meters (degrees field reused).
                lo, hi = _roomba_span(j0.get("degrees", [0.18, 0.34]), 0.2)
                lo = float(np.clip(lo, 0.0, 1.1))
                hi = float(np.clip(hi, 0.0, 1.1))
                for i in range(n):
                    # i/n (not i/(n-1)): avoid sampling both ends of a period at sin=0.
                    phase = (i / max(n, 1)) * rep
                    mid, amp = 0.5 * (lo + hi), 0.5 * (hi - lo)
                    joints = dict(base)
                    joints["lift"] = float(
                        np.clip(mid + amp * math.sin(2 * math.pi * phase), 0.0, 1.1)
                    )
                    local_frames.append(snap())
                if force_n is None:
                    # Soft settle to span center (no hard cut).
                    end = dict(joints)
                    settle_to = dict(base)
                    settle_to["lift"] = float(np.clip(0.5 * (lo + hi), 0.0, 1.1))
                    for i in range(max(2, fps // 6)):
                        joints = _lerp_joints(end, settle_to, (i + 1) / max(2, fps // 6))
                        local_frames.append(snap())
            elif joint == "head":
                lo, hi = _roomba_span(j0.get("degrees", [-20, 20]), 20.0)
                for i in range(n):
                    phase = (i / max(n, 1)) * rep
                    mid, amp = 0.5 * (lo + hi), 0.5 * (hi - lo)
                    off = math.radians(mid + amp * math.sin(2 * math.pi * phase))
                    joints = dict(base)
                    if axis == "tilt":
                        joints["head_tilt"] = base["head_tilt"] + off
                    else:
                        joints["head_pan"] = base["head_pan"] + off
                    local_frames.append(snap())
                if force_n is None:
                    # Soft return to pre-move pose (avoids one-frame snap).
                    end = dict(joints)
                    for i in range(max(2, fps // 6)):
                        joints = _lerp_joints(end, base, (i + 1) / max(2, fps // 6))
                        local_frames.append(snap())
            else:
                # World-axis joint map (same convention as other embodiments):
                #   shoulder/x → arm telescope, shoulder/z → mast lift
                #   wrist/x → roll, wrist/y → yaw (lateral), wrist/z → pitch (vertical)
                # Bug: wrist+z used to fall through to wrist_yaw → sideways air-quotes.
                key = "wrist_yaw"
                scale = math.pi / 180.0
                if joint in ("shoulder", "arm"):
                    if axis in ("z", "height"):
                        key = "lift"
                        # degrees → meters of mast lift. ±20° should read as a clear
                        # bob (~±20cm); 0.004 was too subtle next to base motion.
                        scale = 0.010
                    else:
                        key = "arm"
                        scale = 0.004  # degrees → meters of telescope
                elif joint == "elbow":
                    key = "wrist_pitch"
                elif joint == "wrist":
                    if axis in ("z", "pitch"):
                        key = "wrist_pitch"
                    elif axis in ("x", "roll"):
                        key = "wrist_roll"
                    else:
                        # y / yaw / default → lateral
                        key = "wrist_yaw"
                lo, hi = _roomba_span(j0.get("degrees", [-20, 20]), 20.0)
                for i in range(n):
                    phase = (i / max(n, 1)) * rep
                    mid, amp = 0.5 * (lo + hi), 0.5 * (hi - lo)
                    deg = mid + amp * math.sin(2 * math.pi * phase)
                    joints = dict(base)
                    joints[key] = base[key] + deg * scale
                    local_frames.append(snap())
                if force_n is None:
                    end = dict(joints)
                    for i in range(max(2, fps // 6)):
                        joints = _lerp_joints(end, base, (i + 1) / max(2, fps // 6))
                        local_frames.append(snap())
            if force_n is None:
                hold = float(mv.get("hold_time", 0.0))
                local_frames.extend([snap()] * max(0, round(hold * fps)))
            return local_frames, x, y, yaw, joints
        if kind != "path":
            return local_frames, x, y, yaw, joints
        path = params.get("path") or params
        mode = str(path.get("mode", "ee")).lower()
        shape = str(path.get("shape", "line")).lower()
        rep = max(1, int(path.get("repetition", 1)))
        speed = float(path.get("speed", 1.0))

        def _path_n(**kwargs: Any) -> int:
            n_raw = _roomba_path_n_frames(fps=fps, speed=speed, rep=rep, **kwargs)
            if force_n is not None:
                return max(2, int(force_n))
            return n_raw

        if mode == "base":
            if shape == "line":
                x_raw, y_raw = path.get("x", 0.0), path.get("y", 0.0)
                if "x" not in path and "y" not in path:
                    x_raw = path.get("distance", 0.15)
                x_lo, x_hi = _roomba_span(x_raw)
                y_lo, y_hi = _roomba_span(y_raw)
                x0, y0 = x, y
                x_osc = isinstance(x_raw, list)
                y_osc = isinstance(y_raw, list)
                osc = x_osc or y_osc
                if osc:
                    # Same as Roomba: cycle length 2·|hi−lo|; spd = linear speed; rep scales time.
                    cx = 2.0 * abs(x_hi - x_lo) if x_osc else 0.0
                    cy = 2.0 * abs(y_hi - y_lo) if y_osc else 0.0
                    cycle_m = math.hypot(cx, cy) if (cx > 0 and cy > 0) else (cx or cy)
                    n = _path_n(oscillating=True, stroke_m=cycle_m)
                    if force_n is None:
                        n = max(n, 4 * rep)
                    for i in range(n):
                        t = i / max(n - 1, 1)
                        phase = t * rep
                        xi = (
                            _roomba_pingpong(phase, x_lo, x_hi)
                            if x_osc
                            else x_lo + (x_hi - x_lo) * t
                        )
                        yi = (
                            _roomba_pingpong(phase, y_lo, y_hi)
                            if y_osc
                            else y_lo + (y_hi - y_lo) * t
                        )
                        x, y = x0 + xi, y0 + yi
                        local_frames.append(snap())
                else:
                    stroke = math.hypot(x_hi * rep, y_hi * rep)
                    n = _path_n(oscillating=False, stroke_m=stroke)
                    for i in range(max(n, 0)):
                        t = i / max(n - 1, 1) if n > 1 else 1.0
                        x, y = x0 + (x_hi * rep) * t, y0 + (y_hi * rep) * t
                        local_frames.append(snap())
            else:
                deg_raw = path.get("degrees", path.get("sweep", 90.0))
                radius = float(path.get("radius", 0.0))
                x0, y0, yaw0 = x, y, yaw
                if isinstance(deg_raw, list):
                    d0, d1 = _roomba_span(deg_raw, 90.0)
                    cycle_deg = 2.0 * abs(d1 - d0)
                    n = _path_n(
                        oscillating=True,
                        sweep_deg=cycle_deg,
                        radius_m=radius,
                    )
                    if force_n is None:
                        n = max(n, 4 * rep)
                    for i in range(n):
                        phase = (i / max(n, 1)) * rep
                        mid, amp = 0.5 * (d0 + d1), 0.5 * (d1 - d0)
                        theta = math.radians(mid + amp * math.sin(2 * math.pi * phase))
                        yaw = yaw0 + theta
                        if radius > 1e-5:
                            x = x0 + radius * (math.sin(yaw) - math.sin(yaw0))
                            y = y0 - radius * (math.cos(yaw) - math.cos(yaw0))
                        local_frames.append(snap())
                else:
                    total = float(deg_raw) * rep
                    n = _path_n(
                        oscillating=False, sweep_deg=total, radius_m=radius
                    )
                    # Match Roomba: short high-spd in-place turns can round to n=0/1
                    # under FRAME_SCALE and vanish from the GIF / trajectory.
                    if abs(total) > 1e-6 and force_n is None:
                        n = max(n, 4)
                    for i in range(n):
                        t = i / max(n - 1, 1) if n > 1 else 1.0
                        theta = math.radians(total * t)
                        yaw = yaw0 + theta
                        if radius > 1e-5:
                            x = x0 + radius * (math.sin(yaw0 + theta) - math.sin(yaw0))
                            y = y0 - radius * (math.cos(yaw0 + theta) - math.cos(yaw0))
                        local_frames.append(snap())
        else:
            _apply(model, data, x=x, y=y, yaw=yaw, joints=joints)
            p0 = _ee(model, data).copy()
            if shape == "line":
                axis = str(path.get("axis", "x")).lower()
                dist_raw = path.get("distance", 0.1)
                # Oscillating span e.g. dist=[+0.15,-0.15] (regen fist_pump).
                if isinstance(dist_raw, (list, tuple)) and len(dist_raw) >= 2:
                    d_lo, d_hi = _roomba_span(dist_raw, 0.1)
                    d_lo = float(np.clip(d_lo, -0.22, 0.22))
                    d_hi = float(np.clip(d_hi, -0.22, 0.22))
                    cycle_m = 2.0 * abs(d_hi - d_lo)
                    n = _path_n(oscillating=True, stroke_m=cycle_m)
                    if force_n is None:
                        n = max(n, 4 * rep)
                    seed_line = dict(joints)
                    for i in range(n):
                        phase = (i / max(n, 1)) * rep
                        dist_i = _roomba_pingpong(phase, d_lo, d_hi)
                        # seed + dist*frac with frac=1 → absolute signed offset.
                        joints = _line_joints_analytic(
                            seed_line, axis=axis, dist=dist_i, frac=1.0
                        )
                        local_frames.append(snap())
                else:
                    dist = float(dist_raw)
                    dist_cap = float(np.clip(dist, -0.22, 0.22))
                    n = _path_n(
                        oscillating=False, stroke_m=abs(dist_cap) * max(rep, 1)
                    )
                    # A one-way line must retain both its setup and its endpoint.
                    # At HZ=8, dist=0.22 and spd=4 used to round to one frame;
                    # because t becomes 1 for that sole frame, the GIF showed only
                    # the fully extended arm and looked frozen. Four samples keep
                    # the reach visually legible without changing its duration model.
                    if force_n is None:
                        n = max(n, 4)
                    seed_line = dict(joints)
                    for i in range(n):
                        t = i / max(n - 1, 1)
                        if rep > 1:
                            u = (t * rep) % 2
                            frac = u if u <= 1 else 2 - u
                        else:
                            frac = t
                        # Analytic only — Cartesian IK on Stretch y/z lines snaps.
                        joints = _line_joints_analytic(
                            seed_line, axis=axis, dist=dist_cap, frac=frac
                        )
                        local_frames.append(snap())
            else:
                plane = str(path.get("plane", "xy")).lower()
                # Cap radius — large r + linear yaw looked like ping-pong at joint limits.
                radius = float(np.clip(abs(float(path.get("radius", 0.06))), 0.0, 0.10))
                sweep = float(path.get("sweep", path.get("degrees", 90.0)))
                n = _path_n(
                    oscillating=False,
                    sweep_deg=abs(sweep) * rep,
                    radius_m=radius,
                )
                seed_arc = dict(joints)
                for i in range(n):
                    t = i / max(n - 1, 1)
                    # One full parametric circle per 360° of sweep (× rep).
                    ang = math.radians(sweep) * (rep * t)
                    joints = _arc_joints_analytic(
                        seed_arc, plane=plane, radius=radius, ang=ang
                    )
                    local_frames.append(snap())
        if force_n is None:
            hold = float(path.get("hold_time", 0.0))
            local_frames.extend([snap()] * max(0, round(hold * fps)))

        return local_frames, x, y, yaw, joints

    def _merge_together(
        fa: list[dict[str, Any]],
        fb: list[dict[str, Any]],
        seed: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Zip concurrent tracks: base x/y/yaw from the path track, joints from the move track.

        Shorter tracks are time-stretched across the longer one (not padded with a
        frozen last frame), so e.g. lift flaps keep going for the whole base arc.
        Trailing settle/hold near the seed pose is trimmed from each track first.
        """
        if not fa:
            return list(fb)
        if not fb:
            return list(fa)

        def base_delta(f: dict[str, Any]) -> float:
            return (
                abs(float(f["x"]) - float(seed["x"]))
                + abs(float(f["y"]) - float(seed["y"]))
                + abs(float(f["yaw"]) - float(seed["yaw"]))
            )

        def joint_delta(f: dict[str, Any]) -> float:
            sj = seed["joints"]
            return sum(abs(float(f["joints"][k]) - float(sj[k])) for k in sj)

        def trim_tail(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if len(frames) < 3:
                return frames
            jmax = max(joint_delta(f) for f in frames)
            bmax = max(base_delta(f) for f in frames)
            # Drop trailing near-idle frames (soft settle / hold after a move).
            end = len(frames)
            while end > 1:
                f = frames[end - 1]
                idle_j = jmax < 1e-6 or joint_delta(f) <= 0.08 * jmax
                idle_b = bmax < 1e-6 or base_delta(f) <= 0.02 * bmax
                if idle_j and idle_b:
                    end -= 1
                    continue
                break
            return frames[: max(1, end)]

        def sample(frames: list[dict[str, Any]], i: int, n: int) -> dict[str, Any]:
            """Lerp sample — nearest-neighbor stair-steps short lift tracks."""
            if len(frames) == 1:
                return frames[0]
            t = i / max(n - 1, 1)
            pos = t * (len(frames) - 1)
            i0 = int(math.floor(pos))
            i1 = min(i0 + 1, len(frames) - 1)
            u = float(pos - i0)
            a, b = frames[i0], frames[i1]
            return {
                "x": float(a["x"]) + (float(b["x"]) - float(a["x"])) * u,
                "y": float(a["y"]) + (float(b["y"]) - float(a["y"])) * u,
                "yaw": float(a["yaw"]) + (float(b["yaw"]) - float(a["yaw"])) * u,
                "joints": {
                    k: float(a["joints"][k])
                    + (float(b["joints"][k]) - float(a["joints"][k])) * u
                    for k in a["joints"]
                },
            }

        fa = trim_tail(fa)
        fb = trim_tail(fb)

        a_base = max(base_delta(f) for f in fa)
        b_base = max(base_delta(f) for f in fb)
        a_j = max(joint_delta(f) for f in fa)
        b_j = max(joint_delta(f) for f in fb)
        base_a = a_base >= b_base
        joints_a = a_j >= b_j

        n = max(len(fa), len(fb))
        out: list[dict[str, Any]] = []
        for i in range(n):
            a = sample(fa, i, n)
            b = sample(fb, i, n)
            base_src = a if base_a else b
            joint_src = a if joints_a else b
            out.append(
                {
                    "x": base_src["x"],
                    "y": base_src["y"],
                    "yaw": base_src["yaw"],
                    "joints": dict(joint_src["joints"]),
                }
            )
        return out

    movements = list(row.get("movements") or [])
    i = 0
    while i < len(movements):
        step = movements[i]
        order = str(step.get("order") or "sequential").lower()
        if order == "together" and i + 1 < len(movements):
            seed = {"x": x, "y": y, "yaw": yaw, "joints": dict(joints)}
            # Dry-run → match both tracks to the longer content length so a
            # 6-frame lift bob is not stair-stepped across an 84-frame base arc.
            fa0, _, _, _, _ = _apply_one(step, x, y, yaw, dict(joints))
            fb0, _, _, _, _ = _apply_one(movements[i + 1], x, y, yaw, dict(joints))

            def _content_len(fr: list[dict[str, Any]]) -> int:
                if len(fr) < 3:
                    return len(fr)

                def jdelta(f: dict[str, Any]) -> float:
                    sj = seed["joints"]
                    return sum(abs(float(f["joints"][k]) - float(sj[k])) for k in sj)

                def bdelta(f: dict[str, Any]) -> float:
                    return (
                        abs(float(f["x"]) - float(seed["x"]))
                        + abs(float(f["y"]) - float(seed["y"]))
                        + abs(float(f["yaw"]) - float(seed["yaw"]))
                    )

                jmax = max(jdelta(f) for f in fr)
                bmax = max(bdelta(f) for f in fr)
                end = len(fr)
                while end > 1:
                    f = fr[end - 1]
                    idle_j = jmax < 1e-6 or jdelta(f) <= 0.08 * jmax
                    idle_b = bmax < 1e-6 or bdelta(f) <= 0.02 * bmax
                    if idle_j and idle_b:
                        end -= 1
                        continue
                    break
                return max(1, end)

            n_target = max(_content_len(fa0), _content_len(fb0), 2)
            fa, _, _, _, _ = _apply_one(step, x, y, yaw, dict(joints), force_n=n_target)
            fb, _, _, _, _ = _apply_one(
                movements[i + 1], x, y, yaw, dict(joints), force_n=n_target
            )
            merged = _merge_together(fa, fb, seed)
            frames.extend(merged)
            if merged:
                last = merged[-1]
                x, y, yaw = last["x"], last["y"], last["yaw"]
                joints = dict(last["joints"])
            i += 2
            continue
        chunk, x, y, yaw, joints = _apply_one(step, x, y, yaw, joints)
        frames.extend(chunk)
        i += 1

    frames.extend([snap()] * max(1, fps // 3))
    return frames


def render_stretch(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    force: bool = False,
    fps: int = 12,
) -> dict[str, Path]:
    """Render Hello Robot Stretch 3 Motion Code GIFs."""
    if not _STRETCH_SCENE.is_file():
        raise FileNotFoundError(_STRETCH_SCENE)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(_STRETCH_SCENE))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=480)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.15, 0.0, 0.65)
    camera.distance = 2.5
    camera.azimuth = 145
    camera.elevation = -18
    results: dict[str, Path] = {}
    for row in rows:
        idx, cue = int(row["idx"]), str(row["cue"])
        gif = out_dir / f"emb_g{idx:02d}_{cue}.gif"
        if gif.is_file() and not force:
            results[cue] = gif
            continue
        states = _stretch_states(row, model, data, fps=fps)
        xs = [s["x"] for s in states]
        ys = [s["y"] for s in states]
        camera.lookat[:] = (
            0.5 * (min(xs) + max(xs)) + 0.12,
            0.5 * (min(ys) + max(ys)),
            0.62,
        )
        span = max(abs(max(xs) - min(xs)), abs(max(ys) - min(ys)), 0.3)
        camera.distance = max(2.3, 1.8 + 2.0 * span)
        frames: list[Image.Image] = []
        for i, st in enumerate(states):
            _apply(
                model,
                data,
                x=st["x"],
                y=st["y"],
                yaw=st["yaw"],
                joints=st["joints"],
            )
            renderer.update_scene(data, camera)
            frames.append(_progress_overlay(renderer.render(), i, len(states)))
        _save_media(frames, gif, fps=fps)
        results[cue] = gif
        print(f"[render/stretch] {cue} -> {gif.name} ({len(frames)}f)", flush=True)
    renderer.close()
    return results
