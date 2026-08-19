#!/usr/bin/env python3
"""Render Motion Code configs for Unitree G1 and a planar Roomba proxy.

Stretch lives in ``stretch_renderer.py`` (Hello Robot Stretch 3 MJCF).
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from itertools import product
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from motion_speed import (
    V_REF_M_PER_S as _ROOMBA_LINE_M_PER_UNIT,
    _frame_scale as _motion_frame_scale,
    beat_n_frames as _motion_beat_n_frames,
    path_n_frames as _motion_path_n_frames,
)

_REPO = Path(__file__).resolve().parents[3]
_WORKSPACE = _REPO.parent
_G1_XML = (
    _WORKSPACE
    / "unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1.xml"
)
_G1_FEASIBLE_POSES = (
    _REPO / "data/seed/unitree_g1/g1_feasible_poses_metadata.json"
)
_G1_FEASIBLE_POSE_CACHE: list[dict[str, Any]] | None = None


def _save_media(
    frames: list[Image.Image],
    gif_path: Path,
    *,
    fps: int = 12,
    write_mp4: bool = True,
) -> None:
    from adhoc.generation.gif_encode import save_gif_frames

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    save_gif_frames(frames, gif_path, duration_ms=max(1, round(1000 / fps)))
    ffmpeg = shutil.which("ffmpeg")
    if write_mp4 and ffmpeg:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(gif_path),
                "-movflags",
                "+faststart",
                "-pix_fmt",
                "yuv420p",
                str(gif_path.with_suffix(".mp4")),
            ],
            check=False,
        )


def _progress_overlay(image: np.ndarray, index: int, total: int) -> Image.Image:
    from PIL import ImageDraw

    out = Image.fromarray(image)
    draw = ImageDraw.Draw(out)
    width, height = out.size
    margin = 10
    y0 = height - 14
    draw.rounded_rectangle((margin, y0, width - margin, y0 + 5), radius=2, fill=(45, 48, 55))
    frac = (index + 1) / max(total, 1)
    draw.rounded_rectangle(
        (margin, y0, margin + (width - 2 * margin) * frac, y0 + 5),
        radius=2,
        fill=(66, 133, 244),
    )
    return out


def _joint_qpos_map(model: mujoco.MjModel) -> dict[str, int]:
    out: dict[str, int] = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name:
            out[name] = int(model.jnt_qposadr[jid])
    return out


def _set_joint(q: np.ndarray, joint_map: dict[str, int], name: str, value: float) -> None:
    if name in joint_map:
        q[joint_map[name]] = value


def _g1_home(model: mujoco.MjModel, joint_map: dict[str, int]) -> np.ndarray:
    q = np.zeros(model.nq, dtype=float)
    q[0:3] = (0.0, 0.0, 0.80)
    q[3] = 1.0
    for side in ("left", "right"):
        _set_joint(q, joint_map, f"{side}_hip_pitch_joint", -0.1)
        _set_joint(q, joint_map, f"{side}_knee_joint", 0.3)
        _set_joint(q, joint_map, f"{side}_ankle_pitch_joint", -0.2)
        _set_joint(q, joint_map, f"{side}_shoulder_pitch_joint", 0.35)
        _set_joint(q, joint_map, f"{side}_elbow_joint", 0.87)
    _set_joint(q, joint_map, "left_shoulder_roll_joint", 0.18)
    _set_joint(q, joint_map, "right_shoulder_roll_joint", -0.18)
    return q


def _g1_pose_target(
    q: np.ndarray,
    joint_map: dict[str, int],
    pose: dict[str, Any],
) -> np.ndarray:
    target = q.copy()
    direction = str(pose.get("arm_position", "front")).lower()
    elbow_bent = bool(pose.get("elbow_bended", True))
    pitch, roll, yaw = _G1_DIR_ARM.get(direction, _G1_DIR_ARM["front"])
    _set_joint(target, joint_map, "right_shoulder_pitch_joint", pitch)
    _set_joint(target, joint_map, "right_shoulder_roll_joint", roll)
    _set_joint(target, joint_map, "right_shoulder_yaw_joint", yaw)
    _set_joint(target, joint_map, "right_elbow_joint", 1.35 if elbow_bent else 0.20)
    grip = str(pose.get("gripper_orientation", "vertical")).lower()
    # Legacy categorical grips. Explicit palmup/down/out/in are resolved in
    # world coordinates after xyz IK by _g1_apply_palm_orientation.
    wrist_roll = 1.40 if grip in ("horizontal", "horiz", "open") else 0.0
    _set_joint(target, joint_map, "right_wrist_roll_joint", wrist_roll)

    left_arm = str(pose.get("left_arm", "still")).lower()
    if left_arm == "mirror":
        _set_joint(target, joint_map, "left_shoulder_pitch_joint", pitch)
        _set_joint(target, joint_map, "left_shoulder_roll_joint", -roll)
        _set_joint(target, joint_map, "left_shoulder_yaw_joint", -yaw)
        _set_joint(target, joint_map, "left_elbow_joint", 1.35 if elbow_bent else 0.20)
        _set_joint(target, joint_map, "left_wrist_roll_joint", -wrist_roll)

    head = str(pose.get("head", "center")).lower()
    # Prefer synthetic neck joints (added in scene XML); fall back to waist if absent.
    pan_j = "head_pan_joint" if "head_pan_joint" in joint_map else "waist_yaw_joint"
    tilt_j = "head_tilt_joint" if "head_tilt_joint" in joint_map else "waist_pitch_joint"
    if head in ("left", "right"):
        _set_joint(target, joint_map, pan_j, 0.45 if head == "left" else -0.45)
    elif head in ("up", "down"):
        _set_joint(target, joint_map, tilt_j, -0.35 if head == "up" else 0.35)

    torso = str(pose.get("torso_height", "mid")).lower()
    knee = {"low": 0.65, "mid": 0.30, "high": 0.08}.get(torso, 0.30)
    hip = {"low": -0.30, "mid": -0.10, "high": -0.03}.get(torso, -0.10)
    ankle = -0.55 * knee
    for side in ("left", "right"):
        _set_joint(target, joint_map, f"{side}_hip_pitch_joint", hip)
        _set_joint(target, joint_map, f"{side}_knee_joint", knee)
        _set_joint(target, joint_map, f"{side}_ankle_pitch_joint", ankle)
    target[2] = {"low": 0.73, "mid": 0.80, "high": 0.84}.get(torso, 0.80)
    return target


# EE percentiles: linear min→0 … max→100 over observed palm−torso deltas
# (G1 lattice hard-pass). Same mapping for x/y/z — not torso-centered piecewise.
_G1_EE_AXIS_SPAN_NEG_M = {"x": 0.35, "y": 0.25, "z": 0.30}  # legacy; unused by pct
_G1_EE_AXIS_SPAN_POS_M = {"x": 0.35, "y": 0.25, "z": 0.40}  # legacy; unused by pct
_G1_EE_DELTA_MIN_M = {
    "x": -0.27657775501872955,
    "y": -0.5882680916453166,
    "z": -0.20275679201675068,
}
_G1_EE_DELTA_MAX_M = {
    "x": 0.4514609652423486,
    "y": 0.16788961541898695,
    "z": 0.5791485632285343,
}
# Back-compat aliases
_G1_EE_Z_DELTA_MIN_M = _G1_EE_DELTA_MIN_M["z"]
_G1_EE_Z_DELTA_MAX_M = _G1_EE_DELTA_MAX_M["z"]

_G1_ARM_JOINTS_RIGHT = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
_G1_EE_BODY = "right_wrist_yaw_link"
_G1_EE_BODY_LEFT = "left_wrist_yaw_link"


def _g1_feasible_pose_rows() -> list[dict[str, Any]]:
    global _G1_FEASIBLE_POSE_CACHE
    if _G1_FEASIBLE_POSE_CACHE is None:
        payload = json.loads(_G1_FEASIBLE_POSES.read_text(encoding="utf-8"))
        rows = payload.get("poses") or []
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"No G1 feasible poses in {_G1_FEASIBLE_POSES}")
        _G1_FEASIBLE_POSE_CACHE = rows
    return _G1_FEASIBLE_POSE_CACHE


def _g1_palm_direction_key(row: dict[str, Any]) -> str:
    """Dominant world-axis key of ``palm_normal_world`` (e.g. ``y+``)."""
    palm = np.asarray(row.get("palm_normal_world", (0.0, 0.0, 1.0)), dtype=float)
    norm = float(np.linalg.norm(palm)) + 1e-12
    palm = palm / norm
    axis = int(np.argmax(np.abs(palm)))
    sign = "+" if palm[axis] >= 0.0 else "-"
    return f"{'xyz'[axis]}{sign}"


# Per dir|orient: palm_normal ± axis → palm semantic (opposites implied).
# Empirically matches the feasible-set tile grouping; hand_span labels agree.
_G1_CAT_PALM_AXIS: dict[str, dict[str, str]] = {
    "front|horizontal": {"y-": "palmup", "y+": "palmdown"},
    "front|vertical": {"z+": "palmin", "z-": "palmout"},
    "back|horizontal": {"y+": "palmup", "y-": "palmdown"},
    "back|vertical": {"z+": "palmout", "z-": "palmin"},
    "out|horizontal": {"x-": "palmup", "x+": "palmdown"},
    "out|vertical": {"z-": "palmback", "z+": "palmfront"},
    "in|horizontal": {"x+": "palmup", "x-": "palmdown"},
    "in|vertical": {"z-": "palmfront", "z+": "palmback"},
    "up|horizontal": {"y+": "palmfront", "y-": "palmback"},
    "up|vertical": {"x-": "palmin", "x+": "palmout"},
    "down|horizontal": {"y-": "palmfront", "y+": "palmback"},
    "down|vertical": {"x+": "palmin", "x-": "palmout"},
}

# World palm-pad facing for wrist IK (wrist local +Y / hand_span).
_G1_SEMANTIC_TO_VISUAL_VEC = {
    "palmfront": np.array([1.0, 0.0, 0.0]),
    "palmback": np.array([-1.0, 0.0, 0.0]),
    "palmin": np.array([0.0, 1.0, 0.0]),
    "palmout": np.array([0.0, -1.0, 0.0]),
    "palmup": np.array([0.0, 0.0, 1.0]),
    "palmdown": np.array([0.0, 0.0, -1.0]),
}


def _g1_category_palm_semantic(row: dict[str, Any]) -> str | None:
    """Palm label from dir|orient × ``palm_normal_world`` ± axis."""
    cat = f"{row.get('direction')}|{row.get('gripper_orientation')}"
    axis_map = _G1_CAT_PALM_AXIS.get(cat)
    if not axis_map:
        return None
    return axis_map.get(_g1_palm_direction_key(row))


def _g1_category_valid_palms(
    direction: str | None,
    bank_orient: str | None,
) -> set[str] | None:
    """Two palm tokens allowed for ``dir|orient``; ``None`` if category unknown."""
    if not direction or bank_orient not in ("horizontal", "vertical"):
        return None
    axis_map = _G1_CAT_PALM_AXIS.get(f"{direction}|{bank_orient}")
    if not axis_map:
        return None
    return set(axis_map.values())


def _g1_resolve_palm_for_category(
    direction: str | None,
    bank_orient: str | None,
    palm: str | None,
) -> str | None:
    """Keep palm only when it belongs to that dir|orient pair.

    e.g. ``front|vertical`` allows ``palmin``/``palmout`` only — ``palmfront``
    is dropped so selection/IK fall back to plain front+vertical.
    """
    if not palm:
        return None
    palm = str(palm).lower()
    valid = _g1_category_valid_palms(direction, bank_orient)
    if valid is None:
        return palm  # unknown category: keep legacy behavior
    return palm if palm in valid else None


def _g1_visual_palm_key(row: dict[str, Any]) -> str:
    """Dominant world axis of ``hand_span_world`` (visual palm-pad facing)."""
    span = np.asarray(row.get("hand_span_world", (0.0, 0.0, -1.0)), dtype=float)
    norm = float(np.linalg.norm(span)) + 1e-12
    span = span / norm
    axis = int(np.argmax(np.abs(span)))
    sign = "+" if span[axis] >= 0.0 else "-"
    return f"{'xyz'[axis]}{sign}"


def _g1_visual_palm_semantic(row: dict[str, Any]) -> str:
    """Bank palm bucket: category map via palm_normal; fallback hand_span."""
    cat = _g1_category_palm_semantic(row)
    if cat is not None:
        return cat
    # Legacy global hand_span mapping if category unknown.
    legacy = {
        "x+": "palmfront",
        "x-": "palmback",
        "y+": "palmin",
        "y-": "palmout",
        "z+": "palmup",
        "z-": "palmdown",
    }
    return legacy[_g1_visual_palm_key(row)]


def _g1_desired_palm_normal(grip: str, bank_orient: str | None = None) -> np.ndarray | None:
    """World-axis unit vector for a palm DSL token (visual palm-pad facing)."""
    del bank_orient  # semantics are world-absolute via hand_span
    vec = _G1_SEMANTIC_TO_VISUAL_VEC.get(str(grip or "").lower())
    return None if vec is None else np.asarray(vec, dtype=float)


def _g1_parse_grip(raw: Any) -> tuple[str | None, str | None]:
    """Return ``(bank_orient, palm_semantic)`` from grip / compound grip.

    ``horiz|palmdown`` → ``("horizontal", "palmdown")``: keep horizontal bank
    poses whose visual palm facing (hand_span) is ground (−z), then xyz.
    """
    s = str(raw or "").lower().replace("_", "").replace(" ", "")
    if not s:
        return "vertical", None
    parts = [p for p in re.split(r"[|/+,]", s) if p]
    palm_names = {"palmup", "palmdown", "palmfront", "palmback", "palmin", "palmout"}
    hv_map = {
        "horiz": "horizontal",
        "horizontal": "horizontal",
        "open": "horizontal",
        "vert": "vertical",
        "vertical": "vertical",
        "close": "vertical",
        "closed": "vertical",
        "pinch": "vertical",
    }
    palm = next((p for p in parts if p in palm_names), None)
    hv = next((hv_map[p] for p in parts if p in hv_map), None)
    if hv is None and palm is None:
        if s in hv_map:
            return hv_map[s], None
        if s in palm_names:
            return None, s
        return s, None
    return hv, palm


def _g1_filter_by_palm(
    candidates: list[dict[str, Any]],
    palm: str,
    bank_orient: str | None = None,
    *,
    abs_min: float = 0.7,
) -> list[dict[str, Any]]:
    """Keep poses in the matching palm group for this dir|orient.

    Exact category match only. If the requested palm is absent from the bank
    group (or not one of the two axis buckets for that category), return ``[]``
    so callers keep the broader hv pool and fix palm via wrist IK.
    """
    del bank_orient, abs_min
    if not candidates:
        return []
    palm = str(palm or "").lower()
    if palm not in _G1_SEMANTIC_TO_VISUAL_VEC:
        return candidates
    return [row for row in candidates if _g1_visual_palm_semantic(row) == palm]


def _g1_select_feasible_pose(pose: dict[str, Any]) -> dict[str, Any] | None:
    """Pick from the remain bank with fixed filter tiers, then xyz rank.

    1. **dir + orient (required)** — ``arm_position`` × hv (``front|vertical``, …)
    2. **palm (2nd)** — if a category-valid palm is named (e.g. ``vert|palmin`` on
       front), restrict to that palm group; if palm omitted / invalid for the
       category, skip this tier
    3. **elbow** — if ``elbow=T/F`` is set and matches exist, restrict
    4. **xyz** — L1-nearest on explicitly specified axes
    """
    direction = str(pose.get("arm_position", "front")).lower()
    if direction == "still":
        return None
    candidates = [
        row for row in _g1_feasible_pose_rows()
        if str(row.get("direction", "")).lower() == direction
    ]
    if not candidates:
        return None

    bank_orient, palm_raw = _g1_parse_grip(pose.get("gripper_orientation", "vertical"))
    # Tier 1: hv orient inside dir (required when parseable).
    if bank_orient in ("horizontal", "vertical"):
        filtered = [
            row for row in candidates
            if str(row.get("gripper_orientation", "")).lower() == bank_orient
        ]
        if filtered:
            candidates = filtered
    if not candidates:
        return None

    # Tier 2: palm group only when the token is valid for this dir|orient.
    # e.g. front+vert|palmin → filter palmin; front+vert|palmfront → skip palm.
    palm = _g1_resolve_palm_for_category(direction, bank_orient, palm_raw)
    if palm:
        palm_pool = _g1_filter_by_palm(candidates, palm, bank_orient)
        if palm_pool:
            candidates = palm_pool

    # Tier 3: elbow=T/F (SEW interior; 180°≈straight).
    if "elbow_bended" in pose or "elbow" in pose:
        raw_el = pose.get("elbow_bended", pose.get("elbow"))
        if isinstance(raw_el, str):
            want_bent = raw_el.strip().lower() in ("t", "true", "1", "yes", "bent")
        else:
            want_bent = bool(raw_el)

        def _row_visibly_bent(row: dict[str, Any]) -> bool:
            sew = row.get("sew_interior_deg")
            if isinstance(sew, (int, float)):
                return float(sew) < 140.0
            return bool(row.get("elbow_bended"))

        elbow_filtered = [
            row for row in candidates if _row_visibly_bent(row) == want_bent
        ]
        if elbow_filtered:
            candidates = elbow_filtered

    if not candidates:
        return None

    # Tier 4: xyz L1 on specified axes only.
    targets = {
        axis: float(pose[axis])
        for axis in ("x", "y", "z")
        if isinstance(pose.get(axis), (int, float))
    }
    if targets:
        return min(
            candidates,
            key=lambda row: (
                sum(abs(float(row[axis]) - value) for axis, value in targets.items()),
                int(row.get("pose_id", 0)),
            ),
        )
    return min(
        candidates,
        key=lambda row: (
            sum(abs(float(value)) for value in (row.get("joint_deg") or {}).values()),
            int(row.get("pose_id", 0)),
        ),
    )


def _g1_pose_xyz_mode() -> str:
    """``translate`` (default): bank L1 then EE residual snap. ``nearest``: bank only."""
    mode = os.environ.get("G1_POSE_XYZ_MODE", "translate").strip().lower()
    if mode in ("nearest", "bank", "no_translate"):
        return "nearest"
    return "translate"


def _g1_feasible_pose_target(
    pose: dict[str, Any],
    model: mujoco.MjModel,
    joint_map: dict[str, int],
) -> np.ndarray | None:
    # Motion Code ``bank=F`` skips the discrete feasible bank and falls
    # through to procedural shoulder/elbow + xyz IK + palm IK.
    if pose.get("use_feasible_bank") is False:
        return None
    selected = _g1_select_feasible_pose(pose)
    if selected is None:
        return None
    bank_q = np.asarray(selected.get("qpos"), dtype=float)
    if bank_q.shape != (model.nq,):
        return None
    target = _g1_pose_target(bank_q, joint_map, pose)
    for name in _G1_ARM_JOINTS_RIGHT:
        if name in joint_map:
            target[joint_map[name]] = bank_q[joint_map[name]]
    if str(pose.get("left_arm", "still")).lower() == "mirror":
        _g1_remirror_left_arm(target, joint_map)
    # Optional: snap EE to the Motion Code xyz target (residual parallel translate).
    # Substeps are IK-only and never become GIF frames.
    if _g1_pose_xyz_mode() == "translate":
        target = _g1_apply_ee_xyz_world(target, model, joint_map, pose)
    # Palm compound: wrist IK only when palm is valid for this dir|orient.
    # Invalid tokens (e.g. front + vert|palmfront) skip IK and keep bank wrist.
    direction = str(pose.get("arm_position", "front")).lower()
    bank_orient, palm = _g1_parse_grip(pose.get("gripper_orientation", ""))
    palm = _g1_resolve_palm_for_category(direction, bank_orient, palm)
    if palm:
        target = _g1_apply_palm_orientation(target, model, joint_map, pose)
    return target


def _g1_pct_to_delta_m(pct: float, axis: str) -> float:
    """Map EE percentile → meters relative to torso (linear min→0 … max→100)."""
    lo = _G1_EE_DELTA_MIN_M[axis]
    hi = _G1_EE_DELTA_MAX_M[axis]
    u = max(0.0, min(1.0, float(pct) / 100.0))
    return lo + u * (hi - lo)


def _g1_delta_m_to_pct(delta_m: float, axis: str) -> float:
    """Inverse of ``_g1_pct_to_delta_m`` (clipped to [0, 100])."""
    lo = _G1_EE_DELTA_MIN_M[axis]
    hi = _G1_EE_DELTA_MAX_M[axis]
    if hi <= lo:
        return 50.0
    return float(np.clip(100.0 * (float(delta_m) - lo) / (hi - lo), 0.0, 100.0))


# Categorical shoulder (pitch, roll, yaw) — shared with _g1_pose_target.
_G1_DIR_ARM = {
    "front": (-1.15, -0.15, 0.0),
    "back": (1.05, -0.12, 0.0),
    "in": (-0.35, 0.45, 0.45),
    "out": (-0.20, -1.20, -0.15),
    "up": (-2.45, -0.20, 0.0),
    "down": (0.30, -0.18, 0.0),
    "still": (0.35, -0.18, 0.0),
}


def _g1_remirror_left_arm(q: np.ndarray, joint_map: dict[str, int]) -> None:
    for rname, sign in (
        ("right_shoulder_pitch_joint", 1.0),
        ("right_shoulder_roll_joint", -1.0),
        ("right_shoulder_yaw_joint", -1.0),
        ("right_elbow_joint", 1.0),
        ("right_wrist_roll_joint", -1.0),
        ("right_wrist_pitch_joint", 1.0),
        ("right_wrist_yaw_joint", -1.0),
    ):
        lname = rname.replace("right_", "left_")
        if rname in joint_map and lname in joint_map:
            q[joint_map[lname]] = sign * float(q[joint_map[rname]])


def _g1_dls_ee_to_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q: np.ndarray,
    joint_map: dict[str, int],
    target_pos: np.ndarray,
    *,
    max_iters: int = 40,
    pos_tol_m: float = 0.012,
    damping: float = 0.08,
    max_dq: float = 0.25,
) -> None:
    """Damped least-squares IK: move right wrist body to ``target_pos`` (world)."""
    try:
        bid = model.body(_G1_EE_BODY).id
    except Exception:
        return
    dof_ids: list[int] = []
    qpos_ids: list[int] = []
    for name in _G1_ARM_JOINTS_RIGHT:
        if name not in joint_map:
            continue
        jnt = model.joint(name)
        dof_ids.append(int(jnt.dofadr[0]))
        qpos_ids.append(int(joint_map[name]))
    if not dof_ids:
        return
    target = np.asarray(target_pos, dtype=float).reshape(3)
    diag = (float(damping) ** 2) * np.eye(3, dtype=float)
    jacp = np.zeros((3, model.nv), dtype=float)
    for _ in range(max_iters):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        err = target - np.asarray(data.xpos[bid], dtype=float)
        if float(np.linalg.norm(err)) < pos_tol_m:
            break
        jacp[:] = 0.0
        mujoco.mj_jacBody(model, data, jacp, None, bid)
        j = jacp[:, dof_ids]
        try:
            dq = j.T @ np.linalg.solve(j @ j.T + diag, err)
        except np.linalg.LinAlgError:
            break
        nrm = float(np.linalg.norm(dq))
        if nrm > max_dq:
            dq *= max_dq / nrm
        for qi, dqi in zip(qpos_ids, dq):
            q[qi] = float(q[qi]) + float(dqi)


def _g1_apply_ee_xyz_world(
    q: np.ndarray,
    model: mujoco.MjModel,
    joint_map: dict[str, int],
    pose: dict[str, Any],
) -> np.ndarray:
    """After categorical pose: translate EE in world toward body-centered x/y/z.

    Unspecified axes keep the categorical EE coordinate (never invent defaults).
    ``left_arm=mirror`` remirrors after the right-arm IK.
    """
    axes = [a for a in ("x", "y", "z") if isinstance(pose.get(a), (int, float))]
    if not axes:
        return q

    out = q.copy()
    data = mujoco.MjData(model)
    try:
        tid = model.body("torso_link").id
        rid = model.body(_G1_EE_BODY).id
    except Exception:
        return out

    data.qpos[:] = out
    mujoco.mj_forward(model, data)
    torso = np.asarray(data.xpos[tid], dtype=float)
    cur = np.asarray(data.xpos[rid], dtype=float).copy()
    target = cur.copy()
    for axis in axes:
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        target[idx] = float(torso[idx]) + _g1_pct_to_delta_m(float(pose[axis]), axis)

    # Substep in case of large jumps (match google trim spirit).
    delta = target - cur
    dist = float(np.linalg.norm(delta))
    n_sub = max(1, int(math.ceil(dist / 0.04)))
    for si in range(1, n_sub + 1):
        step_tgt = cur + delta * (si / n_sub)
        _g1_dls_ee_to_target(model, data, out, joint_map, step_tgt)

    if str(pose.get("left_arm", "still")).lower() == "mirror":
        _g1_remirror_left_arm(out, joint_map)
    return out


def _g1_apply_palm_orientation(
    q: np.ndarray,
    model: mujoco.MjModel,
    joint_map: dict[str, int],
    pose: dict[str, Any],
) -> np.ndarray:
    """Orient the primary (right) palm, then remirror left if requested.

    Palm tokens are solved for the right hand only in world coordinates
    (up/down=±z, front/back=±x, in/out=±y). With ``left_arm=mirror``, left-arm
    joints are a signed copy of the finished right arm — so a right ``palmin``
    becomes the contralateral facing on the left (world ``palmout``), without a
    second independent left-hand IK.
    """
    direction = str(pose.get("arm_position", "front")).lower()
    bank_orient, palm = _g1_parse_grip(pose.get("gripper_orientation", ""))
    palm = _g1_resolve_palm_for_category(direction, bank_orient, palm)
    if palm is None:
        return q
    desired = _g1_desired_palm_normal(palm, bank_orient)
    if desired is None:
        return q

    out = q.copy()
    data = mujoco.MjData(model)
    jnames = [
        "right_shoulder_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    bname = "right_wrist_yaw_link"
    if not all(name in joint_map for name in jnames):
        return out
    try:
        bid = model.body(bname).id
        ranges = [
            (float(model.joint(name).range[0]), float(model.joint(name).range[1]))
            for name in jnames
        ]
    except Exception:
        return out

    side_desired = np.asarray(desired, dtype=float).copy()
    data.qpos[:] = out
    mujoco.mj_forward(model, data)
    anchor_pos = np.asarray(data.xpos[bid], dtype=float).copy()
    best_angles = tuple(float(out[joint_map[name]]) for name in jnames)
    best_score = -float("inf")
    # Visual palm-pad facing is wrist local +Y (hand_span). Wrist DOFs plus a
    # limited shoulder-yaw search; penalize EE drift so dir/xyz stay primary.
    grids = [np.linspace(lo, hi, 7) for lo, hi in ranges]
    for raw_angles in product(*grids):
        angles = tuple(float(a) for a in raw_angles)
        for name, angle in zip(jnames, angles):
            out[joint_map[name]] = angle
        data.qpos[:] = out
        mujoco.mj_forward(model, data)
        palm_facing = np.asarray(data.xmat[bid], dtype=float).reshape(3, 3)[:, 1]
        displacement = float(
            np.linalg.norm(np.asarray(data.xpos[bid], dtype=float) - anchor_pos)
        )
        score = float(np.dot(palm_facing, side_desired)) - 2.0 * displacement
        if score > best_score:
            best_score = score
            best_angles = angles
    for scale in (0.20, 0.07):
        local_grids = [
            np.linspace(
                max(lo, angle - scale),
                min(hi, angle + scale),
                5,
            )
            for angle, (lo, hi) in zip(best_angles, ranges)
        ]
        for raw_angles in product(*local_grids):
            angles = tuple(float(a) for a in raw_angles)
            for name, angle in zip(jnames, angles):
                out[joint_map[name]] = angle
            data.qpos[:] = out
            mujoco.mj_forward(model, data)
            palm_facing = np.asarray(data.xmat[bid], dtype=float).reshape(3, 3)[:, 1]
            displacement = float(
                np.linalg.norm(np.asarray(data.xpos[bid], dtype=float) - anchor_pos)
            )
            score = float(np.dot(palm_facing, side_desired)) - 2.0 * displacement
            if score > best_score:
                best_score = score
                best_angles = angles
    for name, angle in zip(jnames, best_angles):
        out[joint_map[name]] = angle

    if str(pose.get("left_arm", "still")).lower() == "mirror":
        _g1_remirror_left_arm(out, joint_map)
    return out


def _g1_apply_xyz_trim(
    q: np.ndarray,
    model: mujoco.MjModel,
    joint_map: dict[str, int],
    pose: dict[str, Any],
) -> np.ndarray:
    """Shim → world EE translate (google-style). Prefer ``_g1_apply_ee_xyz_world``."""
    return _g1_apply_ee_xyz_world(q, model, joint_map, pose)


def _g1_snap_ee_to_body_pct(
    q: np.ndarray,
    model: mujoco.MjModel,
    joint_map: dict[str, int],
    pose: dict[str, Any],
) -> np.ndarray:
    """Shim → world EE translate (google-style)."""
    return _g1_apply_ee_xyz_world(q, model, joint_map, pose)


def _g1_mirror_offset_sign(jname: str) -> float:
    """Mirror sign for a *resolved* joint name (not the abstract DSL axis letter).

    Pitch DOFs (raise/lower) point the same physical way on both arms, so the
    left side takes the identical offset. Roll/yaw DOFs (spread-away-from-torso,
    in/out rotation) are mirrored kinematic chains, so a positive value on the
    right corresponds to a *negative* value on the left for the motion to look
    symmetric — same as the static "out"/"up" pose targets already do.
    """
    lname = str(jname).lower()
    if lname.endswith("_roll_joint") or lname.endswith("_yaw_joint"):
        return -1.0
    return 1.0


def _g1_apply_joint_offset(
    q: np.ndarray,
    base: np.ndarray,
    joint_map: dict[str, int],
    *,
    joint: str,
    axis: str,
    offset_rad: float,
    side: str = "right",
    mirror_left: bool = False,
    model: mujoco.MjModel | None = None,
    resolved_jname: str | None = None,
    world_jac: float | None = None,
) -> None:
    """Apply a signed offset to an arm/head/torso joint.

    For ``shoulder`` / ``elbow`` / ``wrist`` with DSL axis ``x|y|z`` (depth / lateral /
    height), prefer ``resolved_jname`` (Jacobian-best DOF) and scale ``offset_rad`` by
    ``world_jac`` sign so **+config° → +world axis** (+z up, −z down, etc.).
    """
    from adhoc.generation.world_axis_motion import align_offset_to_world_axis

    if resolved_jname is not None:
        jname = resolved_jname
    elif (
        model is not None
        and joint.lower() in ("shoulder", "elbow", "wrist")
        and str(axis).lower() in ("x", "y", "z")
    ):
        jname, world_jac = _g1_resolve_arm_joint_for_world_axis(
            model, base, joint_map, joint=joint, axis=axis, side=side
        )
    else:
        jname = _g1_joint_name(joint, axis, side)

    dq = float(offset_rad)
    if (
        world_jac is not None
        and joint.lower() in ("shoulder", "elbow", "wrist")
        and str(axis).lower() in ("x", "y", "z")
    ):
        dq = align_offset_to_world_axis(dq, world_jac)

    if jname in joint_map:
        q[joint_map[jname]] = float(base[joint_map[jname]]) + dq
    if (
        mirror_left
        and str(side).lower() != "left"
        and joint in ("shoulder", "elbow", "wrist")
    ):
        # Mirror the *same physical DOF* on the left (pitch↔pitch, roll↔−roll, …).
        other = jname.replace("right_", "left_")
        if other in joint_map:
            q[joint_map[other]] = float(base[joint_map[other]]) + _g1_mirror_offset_sign(
                other
            ) * dq


def _g1_resolve_arm_joint_for_world_axis(
    model: mujoco.MjModel,
    q: np.ndarray,
    joint_map: dict[str, int],
    *,
    joint: str,
    axis: str,
    side: str = "right",
) -> tuple[str, float]:
    """Pick arm DOF best aligned with world axis; return ``(joint_name, jac_component)``.

    Shoulder/elbow use EE-body translational Jacobian. Wrist uses a point on the
    palm offset along the visual palm normal (hand_span / local +Y) so
    ``move(wrist, z, …)`` selects the DOF that actually nods the hand in world
    height (e.g. yaw for up|horizontal|palmfront air-quotes) instead of the
    near-zero wrist-origin jac that collapses to a mostly-lateral DOF.
    """
    joint = joint.lower()
    axis = axis.lower()
    candidates = {
        "shoulder": (
            f"{side}_shoulder_pitch_joint",
            f"{side}_shoulder_roll_joint",
            f"{side}_shoulder_yaw_joint",
        ),
        "elbow": (f"{side}_elbow_joint",),
        "wrist": (
            f"{side}_wrist_pitch_joint",
            f"{side}_wrist_roll_joint",
            f"{side}_wrist_yaw_joint",
        ),
    }.get(joint)
    if not candidates or axis not in ("x", "y", "z"):
        return _g1_joint_name(joint, axis, side), 1.0
    ax_idx = {"x": 0, "y": 1, "z": 2}[axis]
    ee_body = _G1_EE_BODY if side == "right" else _G1_EE_BODY_LEFT
    palm_site = "right_palm" if side == "right" else "left_palm"
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    try:
        bid = model.body(ee_body).id
    except Exception:
        return _g1_joint_name(joint, axis, side), 1.0

    jacp = np.zeros((3, model.nv), dtype=float)
    point: np.ndarray | None = None
    if joint == "wrist":
        try:
            sid = int(model.site(palm_site).id)
            palm = np.asarray(data.site_xpos[sid], dtype=float)
            # Visual palm-pad normal = wrist local +Y (hand_span).
            span = np.asarray(data.xmat[bid], dtype=float).reshape(3, 3)[:, 1]
            span = span / (float(np.linalg.norm(span)) + 1e-12)
            point = palm + 0.06 * span
        except Exception:
            point = None
    if point is not None:
        jacr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jac(model, data, jacp, jacr, point, bid)
    else:
        mujoco.mj_jacBody(model, data, jacp, None, bid)

    best_name = None
    best_val = 0.0
    best_score = -1.0
    for jn in candidates:
        if jn not in joint_map:
            continue
        dof = int(model.joint(jn).dofadr[0])
        col = np.asarray(jacp[:, dof], dtype=float)
        val = float(col[ax_idx])
        # Prefer strong *and* axis-aligned motion (penalize dominant off-axis drift).
        score = (abs(val) ** 2) / (float(np.dot(col, col)) + 1e-12)
        if score > best_score:
            best_score = score
            best_val = val
            best_name = jn
    if best_name is None:
        return _g1_joint_name(joint, axis, side), 1.0
    return best_name, best_val


def _g1_joint_name(joint: str, axis: str, side: str = "right") -> str:
    joint = joint.lower()
    axis = axis.lower()
    if joint == "head":
        # Use neck DOFs so head pan/tilt does not bend the torso.
        if axis in ("pan", "y", "z", "yaw"):
            return "head_pan_joint"
        return "head_tilt_joint"
    if joint == "torso":
        return {
            "pan": "waist_yaw_joint",
            "yaw": "waist_yaw_joint",
            "x": "waist_pitch_joint",
            "tilt": "waist_pitch_joint",
            "y": "waist_roll_joint",
            "z": "waist_roll_joint",
        }.get(axis, "waist_pitch_joint")
    if joint == "shoulder":
        # DSL x/y/z = depth / lateral / height (prompt), NOT MuJoCo pitch/yaw/roll letters.
        # Static fallback when Jacobian resolve is unavailable; prefer pitch for height.
        suffix = {"x": "yaw", "y": "roll", "z": "pitch"}.get(axis, "pitch")
        return f"{side}_shoulder_{suffix}_joint"
    if joint == "elbow":
        return f"{side}_elbow_joint"
    if joint == "wrist":
        # Same world-axis convention as shoulder (fallback; runtime uses Jacobian).
        suffix = {"x": "yaw", "y": "roll", "z": "pitch"}.get(axis, "roll")
        return f"{side}_wrist_{suffix}_joint"
    return f"{side}_wrist_roll_joint"


def _interp(a: np.ndarray, b: np.ndarray, n: int) -> list[np.ndarray]:
    if n <= 1:
        return [b.copy()]
    return [a + (b - a) * (i / (n - 1)) for i in range(n)]


def _is_oscillating_degrees(lo: float, hi: float, repetition: int) -> bool:
    """±-style spans oscillate; one-way spans like [0, -45] land and hold."""
    if repetition > 1 and lo * hi < 0:
        return True
    if lo * hi < 0 and abs(abs(lo) - abs(hi)) <= max(1.0, 0.05 * max(abs(lo), abs(hi))):
        return True
    return False


def _signed_span_offset_deg(phase: float, first: float, second: float) -> float:
    """Map cycle phase → degree offset for ordered ± spans like [+20, -20].

    One integer cycle visits ``0 → first → second → 0`` so the listed sign order
    is respected (e.g. +20 up first, then -20 down). ``phase`` may exceed 1 for
    ``repetition > 1``.
    """
    t = phase % 1.0
    # Equal thirds: rise to first, cross to second, return to neutral.
    if t < 1.0 / 3.0:
        u = t * 3.0
        u = 0.5 - 0.5 * math.cos(math.pi * u)
        return first * u
    if t < 2.0 / 3.0:
        u = (t - 1.0 / 3.0) * 3.0
        u = 0.5 - 0.5 * math.cos(math.pi * u)
        return first + (second - first) * u
    u = (t - 2.0 / 3.0) * 3.0
    u = 0.5 - 0.5 * math.cos(math.pi * u)
    return second * (1.0 - u)


def _g1_smoothstep01(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def _g1_movement_beat_frames(speed: float, fps: int) -> int:
    """One directional beat lasts ~1/speed seconds (shared Roomba-scale clamp)."""
    return max(1, _motion_beat_n_frames(fps=fps, speed=speed, rep=1))


def _g1_is_ping_pong_degrees(lo: float, hi: float, repetition: int) -> bool:
    """Multi-rep ± spans ping-pong lo↔hi (fist pump, nod); rep=1 keeps ordered span."""
    return max(1, int(repetition)) > 1 and _is_oscillating_degrees(lo, hi, repetition)


def _g1_movement_pp_frames(
    repetition: int,
    speed: float,
    fps: int,
    *,
    has_range: bool,
    ping_pong: bool,
    ordered_pm: bool,
) -> list[float]:
    """Per-frame progress values for G1 joint movements (mirrors mobile renderer)."""
    beat = _g1_movement_beat_frames(speed, fps)
    reps = max(1, int(repetition))
    pp_frames: list[float] = []

    def _append_ramp(forward: bool) -> None:
        for fi in range(beat):
            t = _g1_smoothstep01((fi + 1) / beat)
            pp_frames.append(t if forward else 1.0 - t)

    if ping_pong:
        n = max(12, beat * 2 * reps)
        for fi in range(n):
            t = fi / max(1, n - 1)
            pp_frames.append(0.5 * (1.0 - math.cos(2.0 * math.pi * reps * t)))
    elif ordered_pm:
        n = max(12, int(beat * 3.3))
        for fi in range(n):
            pp_frames.append(fi / max(1, n - 1))
    elif has_range:
        _append_ramp(True)
    else:
        for _ in range(reps):
            _append_ramp(True)
    return pp_frames


def _g1_resample_pp_frames(pp_frames: list[float], duration_s: float, fps: int) -> list[float]:
    """Upsample sparse beat ramps so motion stays continuous through step duration."""
    target = max(1, int(float(duration_s) * fps))
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


def _apply_torso_height(q: np.ndarray, joint_map: dict[str, int], level: str) -> None:
    level = {"tall": "high"}.get(level, level)
    knee = {"low": 0.65, "mid": 0.30, "high": 0.08}.get(level, 0.30)
    hip = {"low": -0.30, "mid": -0.10, "high": -0.03}.get(level, -0.10)
    ankle = -0.55 * knee
    for side in ("left", "right"):
        _set_joint(q, joint_map, f"{side}_hip_pitch_joint", hip)
        _set_joint(q, joint_map, f"{side}_knee_joint", knee)
        _set_joint(q, joint_map, f"{side}_ankle_pitch_joint", ankle)
    q[2] = {"low": 0.73, "mid": 0.80, "high": 0.84}.get(level, 0.80)


def _g1_collapse_together_movements(
    movements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge ``A & B`` (order=together) joint moves into one multi-joint step.

    Stretch handles ``together`` at frame-merge time; G1 applies offsets from a
    shared base, so collapsing joint lists is enough for synced bows/nods.
    """
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(movements):
        step = movements[i]
        order = str(step.get("order") or "sequential").lower()
        nxt = movements[i + 1] if i + 1 < len(movements) else None
        if (
            order == "together"
            and nxt is not None
            and step.get("type") == "movement"
            and nxt.get("type") == "movement"
        ):
            a = (step.get("parameters") or {}).get("movement") or {}
            b = (nxt.get("parameters") or {}).get("movement") or {}
            mv: dict[str, Any] = {
                "joints": list(a.get("joints") or []) + list(b.get("joints") or []),
                "repetition": max(1, int(a.get("repetition", 1)), int(b.get("repetition", 1))),
                "speed": min(float(a.get("speed", 1.0)), float(b.get("speed", 1.0))),
            }
            hold = a.get("hold_time", b.get("hold_time"))
            if hold is not None:
                mv["hold_time"] = float(hold)
            out.append({"type": "movement", "order": "sequential", "parameters": {"movement": mv}})
            i += 2
            continue
        out.append(step)
        i += 1
    return out


def _g1_motion_qpos(
    row: dict[str, Any],
    model: mujoco.MjModel,
    *,
    fps: int,
) -> list[np.ndarray]:
    joint_map = _joint_qpos_map(model)
    current = _g1_home(model, joint_map)
    seq: list[np.ndarray] = []
    started = False
    # Sticky: pose left_arm=mirror → later movement/path keep remirroring left
    # until a pose explicitly sets left_arm to still / an independent direction.
    mirror_left = False
    for step in _g1_collapse_together_movements(list(row.get("movements") or [])):
        kind = step.get("type")
        params = step.get("parameters") or {}
        duration = float(step.get("duration") or 0.8)
        if kind == "hold":
            # First-class freeze (also accept rewritten zero-path holds).
            hold = float(
                step.get("duration")
                or params.get("t")
                or params.get("hold_time")
                or 1.0
            )
            seq.extend([current.copy()] * max(1, round(hold * fps)))
            continue
        if kind == "pose":
            pose = params.get("pose") or {}
            left_raw = str(pose.get("left_arm", "")).strip().lower()
            if left_raw == "mirror":
                mirror_left = True
            elif left_raw:
                mirror_left = False
            pose_apply = dict(pose)
            if mirror_left and str(pose_apply.get("left_arm", "")).strip().lower() != "mirror":
                pose_apply["left_arm"] = "mirror"
            pinned = pose_apply.get("pinned_qpos")
            if (
                isinstance(pinned, list)
                and len(pinned) == model.nq
                and all(isinstance(value, (int, float)) for value in pinned)
            ):
                # Offline feasible-pose banks can pin a validated whole-body
                # configuration and skip runtime xyz IK entirely.
                target = np.asarray(pinned, dtype=float).copy()
                if _g1_pose_xyz_mode() == "translate":
                    target = _g1_apply_ee_xyz_world(target, model, joint_map, pose_apply)
            else:
                target = _g1_feasible_pose_target(pose_apply, model, joint_map)
                if target is None:
                    categorical = _g1_pose_target(current, joint_map, pose_apply)
                    # Head-only/still poses fall back to the procedural resolver.
                    target = _g1_apply_ee_xyz_world(categorical, model, joint_map, pose_apply)
                    target = _g1_apply_palm_orientation(target, model, joint_map, pose_apply)
            hold_n = max(2, round(duration * fps))
            if not started:
                current = target
                seq.extend([current.copy() for _ in range(hold_n)])
                started = True
            else:
                seq.extend(_interp(current, target, hold_n))
                current = target
            continue
        if not started:
            seq.extend([current.copy() for _ in range(max(1, fps // 2))])
            started = True
        if kind == "movement":
            mv = params.get("movement") or {}
            repetition = max(1, int(mv.get("repetition", 1)))
            speed = max(0.3, float(mv.get("speed", 1.0)))
            base = current.copy()
            specs = mv.get("joints") or []
            oscillate = False
            ping_pong = False
            ordered_pm = False
            has_range = False
            for spec in specs:
                degrees = spec.get("degrees")
                if isinstance(degrees, dict):
                    raw = next(iter(degrees.values()), [0, 10])
                else:
                    raw = degrees
                if isinstance(raw, list) and len(raw) >= 2 and all(isinstance(x, (int, float)) for x in raw[:2]):
                    lo, hi = float(raw[0]), float(raw[-1])
                    has_range = True
                    oscillate = _is_oscillating_degrees(lo, hi, repetition)
                    ping_pong = _g1_is_ping_pong_degrees(lo, hi, repetition)
                    ordered_pm = oscillate and not ping_pong
                    break
            pp_frames = _g1_movement_pp_frames(
                repetition,
                speed,
                fps,
                has_range=has_range,
                ping_pong=ping_pong,
                ordered_pm=ordered_pm,
            )
            motion_duration = float(step.get("duration") or 0.0) or (repetition / speed)
            pp_frames = _g1_resample_pp_frames(pp_frames, motion_duration, fps)
            # Resolve DSL x/y/z → physical DOF + jac sign once at the movement base pose.
            resolved_by_spec: list[tuple[str, float] | None] = []
            for spec in specs:
                joint = str(spec.get("joint", "wrist"))
                axis = str(spec.get("axis", "y"))
                if axis == "height" or "height" in spec:
                    resolved_by_spec.append(None)
                    continue
                side = str(spec.get("side", "right"))
                if joint in ("shoulder", "elbow", "wrist") and axis.lower() in ("x", "y", "z"):
                    resolved_by_spec.append(
                        _g1_resolve_arm_joint_for_world_axis(
                            model, base, joint_map, joint=joint, axis=axis, side=side
                        )
                    )
                else:
                    resolved_by_spec.append(None)
            for pp in pp_frames:
                q = base.copy()
                for spec, resolved in zip(specs, resolved_by_spec):
                    joint = str(spec.get("joint", "wrist"))
                    axis = str(spec.get("axis", "y"))
                    if axis == "height" or "height" in spec:
                        level = str(spec.get("height", "low")).lower()
                        mid = base.copy()
                        _apply_torso_height(mid, joint_map, "mid")
                        dst = base.copy()
                        _apply_torso_height(dst, joint_map, level)
                        for jname in (
                            "left_hip_pitch_joint",
                            "right_hip_pitch_joint",
                            "left_knee_joint",
                            "right_knee_joint",
                            "left_ankle_pitch_joint",
                            "right_ankle_pitch_joint",
                        ):
                            if jname in joint_map:
                                a = float(mid[joint_map[jname]])
                                b = float(dst[joint_map[jname]])
                                q[joint_map[jname]] = a + (b - a) * pp
                        q[2] = float(mid[2]) + (float(dst[2]) - float(mid[2])) * pp
                        continue
                    degrees = spec.get("degrees", [0, 10])
                    if isinstance(degrees, dict):
                        raw = degrees.get(axis, [0, 10])
                    else:
                        raw = degrees
                    if not isinstance(raw, list):
                        raw = [0, raw]
                    first, second = float(raw[0]), float(raw[-1])
                    if ordered_pm:
                        offset_deg = _signed_span_offset_deg(pp, first, second)
                    elif isinstance(raw, list) and len(raw) >= 2:
                        offset_deg = first + (second - first) * pp
                    else:
                        offset_deg = first + (float(raw[-1]) - first) * pp
                    side = str(spec.get("side", "right")).lower()
                    rj = resolved[0] if resolved else None
                    wj = resolved[1] if resolved else None
                    # Sticky pose left_arm=mirror → both arms share the movement
                    # unless this joint spec explicitly targets the left only.
                    apply_mirror = bool(
                        mirror_left
                        and joint in ("shoulder", "elbow", "wrist")
                        and side != "left"
                    )
                    _g1_apply_joint_offset(
                        q,
                        base,
                        joint_map,
                        joint=joint,
                        axis=axis,
                        offset_rad=math.radians(offset_deg),
                        side=side if side in ("right", "left") else "right",
                        mirror_left=apply_mirror,
                        model=model,
                        resolved_jname=rj,
                        world_jac=wj,
                    )
                seq.append(q)
            current = seq[-1].copy()
            hold = float(mv.get("hold_time", 0.0))
            seq.extend([current.copy()] * max(0, round(hold * fps)))
            continue
        if kind == "path":
            path = params.get("path") or params
            shape = str(path.get("shape", "line")).lower()
            repetition = max(1, int(path.get("repetition", 1)))
            speed = max(0.3, float(path.get("speed", 1.0)))
            base = current.copy()
            # duration may be set on the step (rare) or omitted for hold paths
            duration = float(step.get("duration") or 0.0) or None
            if shape in ("arc", "circle"):
                radius = float(path.get("radius", 0.12))
                sweep = float(path.get("sweep", path.get("degrees", 360.0)))
                n = _roomba_path_n_frames(
                    oscillating=False,
                    rep=1,
                    speed=speed,
                    fps=fps,
                    sweep_deg=abs(sweep) * repetition,
                    radius_m=abs(radius),
                )
                plane = str(path.get("plane", "xy")).lower()
                plane_axes = {
                    "xy": (0, 1),
                    "xz": (0, 2),
                    "yz": (1, 2),
                }.get(plane, (0, 1))
                data_ik = mujoco.MjData(model)
                data_ik.qpos[:] = base
                mujoco.mj_forward(model, data_ik)
                try:
                    ee_start = np.asarray(
                        data_ik.xpos[model.body(_G1_EE_BODY).id], dtype=float
                    ).copy()
                except Exception:
                    ee_start = np.zeros(3, dtype=float)
                q_work = base.copy()
                for i in range(n):
                    t = i / max(n - 1, 1)
                    # Signed sweep: +180 / −180 travel opposite ways (no cw/ccw).
                    # Center along −e1; θ = signed. On yz, −sweep bulges down (bowl ∪).
                    angle = math.radians(sweep * repetition) * t
                    target = ee_start.copy()
                    a0, a1 = plane_axes
                    target[a0] += radius * (math.cos(angle) - 1.0)
                    target[a1] += radius * math.sin(angle)
                    _g1_dls_ee_to_target(
                        model,
                        data_ik,
                        q_work,
                        joint_map,
                        target,
                        max_iters=60,
                        pos_tol_m=0.004,
                    )
                    if mirror_left:
                        _g1_remirror_left_arm(q_work, joint_map)
                    seq.append(q_work.copy())
                current = seq[-1].copy()
            else:
                # EE/base straight line. Prompt: +x forward (toward viewer), +y left, +z up.
                # EE lines use world DLS IK (google-style) — never shoulder-pitch hacks
                # (G1 +pitch ≈ retract, so a joint nudge made +dist look backward).
                axis = str(path.get("axis", "x")).lower()
                dist = path.get("distance", path.get("x", 0.0))
                if isinstance(dist, list) and len(dist) >= 2:
                    lo, hi = float(dist[0]), float(dist[-1])
                    oscillate = True
                else:
                    lo, hi = 0.0, float(dist or 0.0)
                    oscillate = repetition > 1
                mode = str(path.get("mode") or path.get("target") or "ee").lower()
                stroke = abs(hi - lo) if oscillate else abs(hi)
                # Pure hold (distance≈0): freeze current qpos — do NOT run EE IK
                # for N filler frames (IK drifts the last pose and looks "cut").
                if stroke < 1e-6:
                    n = 0
                else:
                    stroke_total = (
                        abs(hi - lo) * 2.0 * repetition if oscillate else stroke
                    )
                    if duration:
                        n = max(
                            1,
                            int(round(float(duration) * fps * _motion_frame_scale())),
                        )
                    else:
                        n = _roomba_path_n_frames(
                            oscillating=False,
                            rep=1,
                            speed=speed,
                            fps=fps,
                            stroke_m=stroke_total,
                        )
                ax_idx = {"x": 0, "y": 1, "z": 2}.get(axis, 0)
                if mode == "base":
                    for i in range(n):
                        if oscillate:
                            phase = (i / max(n - 1, 1)) * repetition
                            progress = 0.5 - 0.5 * math.cos(2 * math.pi * phase)
                        else:
                            progress = i / max(n - 1, 1)
                        offset = lo + (hi - lo) * progress
                        q = base.copy()
                        if axis == "x":
                            q[0] = float(base[0]) + offset
                        elif axis == "y":
                            q[1] = float(base[1]) + offset
                        elif axis == "z":
                            q[2] = float(base[2]) + offset
                        seq.append(q)
                    if n > 0:
                        current = seq[-1].copy()
                else:
                    data_ik = mujoco.MjData(model)
                    data_ik.qpos[:] = base
                    mujoco.mj_forward(model, data_ik)
                    try:
                        ee_start = np.asarray(
                            data_ik.xpos[model.body(_G1_EE_BODY).id], dtype=float
                        ).copy()
                    except Exception:
                        ee_start = np.zeros(3, dtype=float)
                    q_work = base.copy()
                    for i in range(n):
                        if oscillate:
                            phase = (i / max(n - 1, 1)) * repetition
                            progress = 0.5 - 0.5 * math.cos(2 * math.pi * phase)
                        else:
                            progress = i / max(n - 1, 1)
                        offset = lo + (hi - lo) * progress
                        target = ee_start.copy()
                        target[ax_idx] = float(ee_start[ax_idx]) + offset
                        _g1_dls_ee_to_target(model, data_ik, q_work, joint_map, target)
                        if mirror_left:
                            _g1_remirror_left_arm(q_work, joint_map)
                        seq.append(q_work.copy())
                    if n > 0:
                        current = seq[-1].copy()
            hold = float(path.get("hold_time", 0.0))
            hold_n = max(0, round(hold * fps))
            if hold_n > 0:
                seq.extend([current.copy()] * hold_n)
            continue
    if not seq:
        seq = [current.copy() for _ in range(max(1, fps))]
    seq.extend([current.copy()] * max(1, fps // 2))
    return seq


def _make_g1_scene_xml() -> Path:
    """Build a render scene with synthetic neck joints (G1 stock MJCF has a fixed head)."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(_G1_XML)
    root = tree.getroot()
    compiler = root.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", str(_G1_XML.parent / "assets"))

    # Outdoor-ish backdrop: sky-blue skybox + dark floor. Avoid pure black
    # (hard to see limbs) and pure white studio (washes the robot out).
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "skybox",
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.55 0.72 0.92",
            "rgb2": "0.78 0.88 0.98",
            "width": "512",
            "height": "512",
        },
    )
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", "0.35 0.35 0.38")
    headlight.set("diffuse", "0.75 0.75 0.75")
    headlight.set("specular", "0.12 0.12 0.12")

    world = root.find("worldbody")
    assert world is not None
    ET.SubElement(
        world,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "4 4 0.1",
            "rgba": "0.22 0.24 0.27 1",
            "pos": "0 0 0",
        },
    )
    ET.SubElement(
        world,
        "light",
        {
            "pos": "1 -2 3",
            "dir": "-0.2 0.3 -1",
            "diffuse": "0.95 0.95 0.95",
            "ambient": "0.12 0.12 0.14",
        },
    )

    # Detach head_link from torso and mount it on pan/tilt neck joints so
    # move(head, pan/tilt, ...) reads as a head gesture, not a torso bend.
    torso = None
    for body in world.iter("body"):
        if body.get("name") == "torso_link":
            torso = body
            break
    if torso is not None:
        for geom in list(torso.findall("geom")):
            if geom.get("mesh") == "head_link":
                torso.remove(geom)
        neck_yaw = ET.SubElement(
            torso,
            "body",
            {"name": "neck_yaw_link", "pos": "0 0 0.38"},
        )
        ET.SubElement(
            neck_yaw,
            "inertial",
            {
                "pos": "0 0 0",
                "mass": "0.05",
                "diaginertia": "1e-4 1e-4 1e-4",
            },
        )
        ET.SubElement(
            neck_yaw,
            "joint",
            {
                "name": "head_pan_joint",
                "type": "hinge",
                "axis": "0 0 1",
                "range": "-1.2 1.2",
                "damping": "0.5",
            },
        )
        neck_pitch = ET.SubElement(neck_yaw, "body", {"name": "neck_pitch_link"})
        ET.SubElement(
            neck_pitch,
            "inertial",
            {
                "pos": "0 0 0.02",
                "mass": "0.8",
                "diaginertia": "0.002 0.002 0.002",
            },
        )
        ET.SubElement(
            neck_pitch,
            "joint",
            {
                "name": "head_tilt_joint",
                "type": "hinge",
                "axis": "0 1 0",
                "range": "-0.7 0.7",
                "damping": "0.5",
            },
        )
        # Keep rest pose of the head mesh identical to stock MJCF
        # (was at 0.0039635 0 -0.044 relative to torso).
        ET.SubElement(
            neck_pitch,
            "geom",
            {
                "class": "visual",
                "pos": "0.0039635 0 -0.424",
                "material": "black",
                "mesh": "head_link",
            },
        )

    fd = tempfile.NamedTemporaryFile(suffix=".xml", delete=False)
    fd.close()
    tree.write(fd.name)
    return Path(fd.name)


def render_unitree_g1(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    force: bool = False,
    fps: int = 20,
) -> dict[str, Path]:
    if not _G1_XML.is_file():
        raise FileNotFoundError(f"Unitree G1 MJCF missing: {_G1_XML}")
    scene_xml = _make_g1_scene_xml()
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_xml))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=480, width=480)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0.0, 0.0, 0.85)
        camera.distance = 2.7
        camera.azimuth = 145
        camera.elevation = -8
        for row in rows:
            idx, cue = int(row["idx"]), str(row["cue"])
            gif = out_dir / f"emb_g{idx:02d}_{cue}.gif"
            if gif.is_file() and not force:
                results[cue] = gif
                continue
            qseq = _g1_motion_qpos(row, model, fps=fps)
            frames: list[Image.Image] = []
            for i, q in enumerate(qseq):
                data.qpos[:] = q
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera)
                frames.append(_progress_overlay(renderer.render(), i, len(qseq)))
            _save_media(frames, gif, fps=fps)
            results[cue] = gif
            print(f"[render/unitree_g1] {cue} -> {gif.name} ({len(frames)}f)", flush=True)
        renderer.close()
    finally:
        scene_xml.unlink(missing_ok=True)
    return results


_ROOMBA_XML = """
<mujoco model="motion_code_roomba">
  <option gravity="0 0 -9.81"/>
  <visual>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.8 0.8 0.8"/>
  </visual>
  <asset>
    <texture name="checker" type="2d" builtin="checker"
             rgb1="0.22 0.26 0.32" rgb2="0.86 0.88 0.92"
             width="512" height="512"/>
    <material name="checker_floor" texture="checker" texrepeat="10 10"
              texuniform="true" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0 -2 4" dir="0 0 -1"/>
    <geom type="plane" size="3 3 0.1" material="checker_floor"/>
    <body name="roomba" pos="0 0 0.09">
      <freejoint/>
      <geom type="cylinder" size="0.22 0.06" rgba="0.10 0.12 0.15 1"/>
      <geom type="box" size="0.08 0.015 0.012" pos="0.15 0 0.067" rgba="0.2 0.65 1 1"/>
      <geom type="cylinder" size="0.045 0.018" pos="0.02 0 0.075" rgba="0.45 0.48 0.52 1"/>
      <geom type="box" size="0.025 0.035 0.035" pos="0 0.205 -0.02" rgba="0.04 0.04 0.04 1"/>
      <geom type="box" size="0.025 0.035 0.035" pos="0 -0.205 -0.02" rgba="0.04 0.04 0.04 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _roomba_span(v: Any, default: float = 0.0) -> tuple[float, float]:
    """Return (lo, hi) for scalar or [lo, hi] motion extents."""
    if isinstance(v, list) and len(v) >= 2:
        return float(v[0]), float(v[-1])
    if isinstance(v, list) and len(v) == 1:
        x = float(v[0])
        return (0.0, x)
    if isinstance(v, (int, float)):
        x = float(v)
        return (0.0, x)
    return (0.0, default)


_ROOMBA_POSE_YAW = {
    "front": 0.0,
    "forward": 0.0,
    "back": math.pi,
    "left": math.pi / 2.0,
    "right": -math.pi / 2.0,
}


def _roomba_pose_yaw(pose: dict[str, Any]) -> float:
    facing = str(pose.get("dir", pose.get("facing", "front"))).lower()
    return _ROOMBA_POSE_YAW.get(facing, 0.0)


def _roomba_shortest_yaw_delta(current: float, target: float) -> float:
    """Signed shortest rotation from current → target."""
    delta = (target - current) % (2.0 * math.pi)
    if delta > math.pi:
        delta -= 2.0 * math.pi
    return delta


def _roomba_pingpong(phase: float, lo: float, hi: float) -> float:
    """Centered oscillation: mid → hi → mid → lo → mid (phase 0/1 at mid).

    Starts and ends at the midpoint so chained steps do not teleport from a
    previous rest pose at the origin (old lo→hi→lo cosine started at ``lo``).
    """
    mid = 0.5 * (lo + hi)
    amp = 0.5 * (hi - lo)
    return mid + amp * math.sin(2.0 * math.pi * phase)


def _roomba_path_n_frames(
    *,
    oscillating: bool,
    rep: int,
    speed: float,
    fps: int,
    stroke_m: float = 0.0,
    sweep_deg: float = 0.0,
    radius_m: float = 0.0,
) -> int:
    """Frame count shared by Roomba/Stretch (and used as the global path scale).

    - Oscillating: ``rep/speed`` seconds.
    - One-way line: length / (spd · 0.15 m/s).
    - One-way arc with r>0: ``r·θ`` / (spd · 0.15) — same linear speed as a line.
    - In-place yaw (r≈0): ``90° ≡ 0.15 m`` (Roomba unit).
    - Zero-length one-way: 0 frames.
    """
    return _motion_path_n_frames(
        fps=fps,
        speed=speed,
        oscillating=oscillating,
        rep=rep,
        stroke_m=stroke_m,
        sweep_deg=sweep_deg,
        radius_m=radius_m,
    )


def _roomba_states(row: dict[str, Any], *, fps: int) -> list[tuple[float, float, float]]:
    x = y = yaw = 0.0
    states = [(x, y, yaw)] * max(1, fps // 2)
    for step in row.get("movements") or []:
        kind = step.get("type")
        params = step.get("parameters") or {}
        if kind == "hold":
            hold = float(
                step.get("duration")
                or params.get("t")
                or params.get("hold_time")
                or 1.0
            )
            states.extend([(x, y, yaw)] * max(1, round(hold * fps)))
            continue
        if kind == "pose":
            pose = params.get("pose") or {}
            target_yaw = _roomba_pose_yaw(pose)
            duration = float(step.get("duration") or 0.6)
            delta = _roomba_shortest_yaw_delta(yaw, target_yaw)
            turn_n = max(4, round(min(0.5, duration) * fps)) if abs(delta) > 1e-3 else 0
            for i in range(turn_n):
                t = (i + 1) / turn_n
                states.append((x, y, yaw + delta * t))
            if turn_n:
                yaw = target_yaw
            hold_n = max(2, round(duration * fps))
            states.extend([(x, y, yaw)] * hold_n)
            continue
        if kind == "movement":
            # Fallback wiggle; prefer path conversion for Roomba.
            mv = params.get("movement") or {}
            rep = max(1, int(mv.get("repetition", 1)))
            speed = float(mv.get("speed", 1.0))
            n = _motion_beat_n_frames(fps=fps, speed=speed, rep=rep)
            amp = math.radians(20)
            for i in range(n):
                p = i / max(n - 1, 1)
                states.append((x, y, yaw + amp * math.sin(p * rep * 2 * math.pi)))
            continue
        if kind != "path":
            continue
        path = params.get("path") or params
        shape = str(path.get("shape", "line")).lower()
        rep = max(1, int(path.get("repetition", 1)))
        speed = float(path.get("speed", 1.0))
        # ``states`` already ends at this path's start pose.  Sampling it again
        # adds an unintended one-frame pause at every ``component + component``
        # boundary, even without a requested hold.
        def sample_indices(n: int) -> range:
            if n <= 0:
                return range(0)
            return range(1, n) if n > 1 else range(1)

        if shape == "line":
            x_raw, y_raw = path.get("x", 0.0), path.get("y", 0.0)
            if "x" not in path and "y" not in path:
                x_raw = path.get("distance", 0.15)
            x_osc = isinstance(x_raw, list)
            y_osc = isinstance(y_raw, list)
            x_lo, x_hi = _roomba_span(x_raw)
            y_lo, y_hi = _roomba_span(y_raw)
            x0, y0 = x, y
            if x_osc or y_osc:
                # One sine cycle mid→hi→mid→lo→mid travels 2·|hi−lo| on that axis.
                # spd = linear speed; rep only repeats → time ∝ rep (not beat timing).
                cx = 2.0 * abs(x_hi - x_lo) if x_osc else 0.0
                cy = 2.0 * abs(y_hi - y_lo) if y_osc else 0.0
                cycle_m = math.hypot(cx, cy) if (cx > 0 and cy > 0) else (cx or cy)
                n = _roomba_path_n_frames(
                    oscillating=True,
                    rep=rep,
                    speed=speed,
                    fps=fps,
                    stroke_m=cycle_m,
                )
                n = max(n, 4 * rep)
                for i in sample_indices(n):
                    t = i / max(n - 1, 1)
                    phase = t * rep
                    xi = (
                        _roomba_pingpong(phase, x_lo, x_hi)
                        if x_osc
                        else (x_lo + (x_hi - x_lo) * t)
                    )
                    yi = (
                        _roomba_pingpong(phase, y_lo, y_hi)
                        if y_osc
                        else (y_lo + (y_hi - y_lo) * t)
                    )
                    states.append((x0 + xi, y0 + yi, yaw))
            else:
                # Scalar line: one-way to target (rep multiplies distance).
                stroke = math.hypot(x_hi * rep, y_hi * rep)
                n = _roomba_path_n_frames(
                    oscillating=False,
                    rep=1,
                    speed=speed,
                    fps=fps,
                    stroke_m=stroke,
                )
                for i in sample_indices(max(n, 0)):
                    t = i / max(n - 1, 1) if n > 1 else 1.0
                    states.append((x0 + (x_hi * rep) * t, y0 + (y_hi * rep) * t, yaw))
            if n > 0:
                x, y, yaw = states[-1]
        else:
            deg_raw = path.get("degrees", path.get("sweep", 90.0))
            radius = float(path.get("radius", 0.0))
            x0, y0, yaw0 = x, y, yaw
            if isinstance(deg_raw, list):
                deg_first, deg_second = _roomba_span(deg_raw, 90.0)
                # One sine/± cycle covers 2·|hi−lo| degrees of yaw travel
                # (mid→hi→mid→lo→mid). spd = linear/angular rate; time ∝ rep.
                cycle_deg = 2.0 * abs(deg_second - deg_first)
                n = _roomba_path_n_frames(
                    oscillating=True,
                    rep=rep,
                    speed=speed,
                    fps=fps,
                    sweep_deg=cycle_deg,
                    radius_m=radius,
                )
                n = max(n, 4 * rep)
                for i in sample_indices(n):
                    phase = (i / max(n - 1, 1)) * rep
                    # One authored [+a, -a] pair is one complete set:
                    # neutral → first sign → second sign → neutral.
                    theta = math.radians(
                        _signed_span_offset_deg(phase, deg_first, deg_second)
                    )
                    if radius <= 1e-5:
                        states.append((x0, y0, yaw0 + theta))
                    else:
                        states.append(
                            (
                                x0 + radius * (math.sin(yaw0 + theta) - math.sin(yaw0)),
                                y0 - radius * (math.cos(yaw0 + theta) - math.cos(yaw0)),
                                yaw0 + theta,
                            )
                        )
            else:
                # Scalar arc: one-way turn (180 rep=1 → half turn and stop; 360 rep=2 → two laps).
                total_deg = float(deg_raw) * rep
                n = _roomba_path_n_frames(
                    oscillating=False,
                    rep=rep,
                    speed=speed,
                    fps=fps,
                    sweep_deg=total_deg,
                    radius_m=radius,
                )
                # Short high-spd in-place turns round to n=1 at FRAME_SCALE=0.5;
                # keep a few samples so the yaw change is visible (not a no-op).
                if abs(total_deg) > 1e-6:
                    n = max(n, 4)
                for i in sample_indices(n):
                    # Match line paths: n==1 must land on the end pose (t=1), not t=0.
                    t = i / max(n - 1, 1) if n > 1 else 1.0
                    theta = math.radians(total_deg * t)
                    if radius <= 1e-5:
                        states.append((x0, y0, yaw0 + theta))
                    else:
                        states.append(
                            (
                                x0 + radius * (math.sin(yaw0 + theta) - math.sin(yaw0)),
                                y0 - radius * (math.cos(yaw0 + theta) - math.cos(yaw0)),
                                yaw0 + theta,
                            )
                        )
            if n > 0:
                x, y, yaw = states[-1]
        hold = float(path.get("hold_time", 0.0))
        states.extend([(x, y, yaw)] * max(0, round(hold * fps)))
    states.extend([(x, y, yaw)] * max(1, fps // 2))
    return states


def render_roomba(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    force: bool = False,
    fps: int = 12,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(_ROOMBA_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=420, width=520)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.0)
    camera.distance = 1.8
    camera.azimuth = 135
    camera.elevation = -50
    results: dict[str, Path] = {}
    for row in rows:
        idx, cue = int(row["idx"]), str(row["cue"])
        gif = out_dir / f"emb_g{idx:02d}_{cue}.gif"
        if gif.is_file() and not force:
            results[cue] = gif
            continue
        states = _roomba_states(row, fps=fps)
        # A foil may reverse an arc. Frame it from the authored trajectory so
        # True/Foil keep exactly the same initial camera and pose composition.
        camera_row = row
        reference_movements = row.get("camera_reference_movements")
        if isinstance(reference_movements, list):
            camera_row = {**row, "movements": reference_movements}
        camera_states = _roomba_states(camera_row, fps=fps)
        # Keep a fixed world camera so base translation reads against the checker floor.
        xs = [s[0] for s in camera_states]
        ys = [s[1] for s in camera_states]
        camera.lookat[:] = (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)), 0.0)
        span = max(abs(max(xs) - min(xs)), abs(max(ys) - min(ys)), 0.4)
        camera.distance = max(1.6, 1.2 + 2.2 * span)
        frames: list[Image.Image] = []
        for i, (x, y, yaw) in enumerate(states):
            data.qpos[:] = 0
            data.qpos[0:3] = (x, y, 0.09)
            data.qpos[3:7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera)
            frames.append(_progress_overlay(renderer.render(), i, len(states)))
        _save_media(frames, gif, fps=fps)
        results[cue] = gif
        print(f"[render/roomba] {cue} -> {gif.name} ({len(frames)}f)", flush=True)
    renderer.close()
    return results


def render_stretch(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    force: bool = False,
    fps: int = 12,
) -> dict[str, Path]:
    """Hello Robot Stretch 3 — see ``stretch_renderer.py``."""
    from stretch_renderer import render_stretch as _render_stretch

    return _render_stretch(rows, out_dir, force=force, fps=fps)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))
