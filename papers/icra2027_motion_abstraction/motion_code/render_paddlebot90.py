#!/usr/bin/env python3
"""Synth90 Motion Programs → Paddlebot (gcr_alpha1) GIFs + gallery.

Recipe:
  - orientation: canvas dir → claw= (paddle= if left_arm=mirror); grip HV → palm edge|flat
  - movement: canvas / auto_exec_move, remapped onto claw/paddle channels
  - location: paddle-body xyz (canvas/humanoid priors + dir-aware fill, clamped)

Outputs:
  data/results/motion_configs/embodiment_motion_code_paddlebot90/
  data/results/render/embodiment_motion_code_paddlebot90/gcr_alpha1/gif/
  motion_code/paddlebot90_gallery.html
"""
from __future__ import annotations

import argparse
import hashlib
import html as html_mod
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "glfw")

MC = Path(__file__).resolve().parent
PAPER = MC.parent
REPO = PAPER.parent.parent
ODDKIN = PAPER / "synthetic_morphology"
ADHOC = REPO / "adhoc/generation/robotarm"
sys.path[:0] = [str(MC), str(ODDKIN), str(PAPER), str(REPO), str(ADHOC)]

from render_synth60_motion60 import cue_indices, semantic_blurb  # noqa: E402
from score_humanlike_cuespec import _na  # noqa: E402
import render_cues30 as rc  # noqa: E402
import math  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

OUT_TAG = "embodiment_motion_code_paddlebot90"
OUT_CFG = REPO / "data/results/motion_configs" / OUT_TAG
OUT_RENDER = REPO / "data/results/render" / OUT_TAG
OUT_HTML = MC / "paddlebot90_gallery.html"
OUT_MANIFEST = OUT_CFG / "manifest.json"
XYZ_JSON = MC / "data" / "cue_xyz_90_humanoid_tiago.json"
RC90_JSON = PAPER / "required_components_90.json"
XYZ_OVERRIDE = OUT_CFG / "paddle_xyz_body_overrides.json"
IK_JOINTS_JSON = OUT_CFG / "paddle_ik_joints.json"

# Canvas GT = required_components_90 (pilot-reviewed). Not score_humanlike REQUIRED
# (those diverge on several dirs, e.g. fan_face_hot front vs in).
_RC90_ROWS = json.loads(RC90_JSON.read_text(encoding="utf-8"))["rows"]
CANVAS: dict[str, dict[str, str]] = {
    r["cue"]: {
        "pose": str(r.get("pose") or "—"),
        "movement": str(r.get("movement") or "—"),
        "path": str(r.get("path") or "—"),
    }
    for r in _RC90_ROWS
}
CUES90 = [r["cue"] for r in _RC90_ROWS]

_HV = {"horiz", "horizontal", "vert", "vertical"}

# Tuned arm-only 6-dir presets (from demo_paddle_6dir.py).
A1_DIRS: dict[str, dict[str, float]] = {
    "claw_front": {"br0": 0.02, "br1": 0.32, "br2": 0.23, "br3": 0.0},
    "claw_back": {"br0": 0.03, "br1": -2.20, "br2": -1.02, "br3": 0.0},
    "claw_in": {"br0": 0.99, "br1": -0.01, "br2": 0.43, "br3": 0.0},
    "claw_out": {"br0": -1.14, "br1": 0.00, "br2": 0.08, "br3": 0.0},
    "claw_up": {"br0": -0.20, "br1": -0.70, "br2": 0.25, "br3": 0.0},  # raised forward (not folded-back)
    "claw_down": {"br0": -0.40, "br1": 0.40, "br2": 0.48, "br3": 0.0},
    "claw_still": {"br0": -0.55, "br1": 0.25, "br2": 0.15, "br3": 0.0},
    "paddle_front": {"bl0": 0.10, "bl1": -0.20, "bl2": 1.00},
    "paddle_back": {"bl0": 1.15, "bl1": -0.95, "bl2": 0.55},
    "paddle_in": {"bl0": -0.99, "bl1": 0.00, "bl2": 0.70},
    "paddle_out": {"bl0": 1.18, "bl1": 0.00, "bl2": 0.50},
    "paddle_up": {"bl0": 0.10, "bl1": -1.15, "bl2": 1.00},
    "paddle_down": {"bl0": 1.15, "bl1": 0.50, "bl2": 0.35},
    "paddle_still": {"bl0": 0.55, "bl1": 0.30, "bl2": 0.35},
}

# Dir-aware paddle xyz priors (claw EE percentiles).
_DIR_XYZ = {
    "front": {"x": 78, "y": 42, "z": 55},
    "back": {"x": 28, "y": 48, "z": 52},
    "in": {"x": 58, "y": 78, "z": 72},
    "out": {"x": 55, "y": 18, "z": 55},
    "up": {"x": 55, "y": 48, "z": 88},
    "down": {"x": 55, "y": 45, "z": 32},
    "still": {"x": 50, "y": 50, "z": 50},
}


def _preset(key: str) -> dict[str, float]:
    if key in A1_DIRS:
        return dict(A1_DIRS[key])
    return dict(rc.A1.get(key, {}))


def _strip_palm_grip(grip_raw: str) -> str:
    parts = [p.strip().lower() for p in re.split(r"[|/]", str(grip_raw or "")) if p.strip()]
    hv = next((p for p in parts if p in _HV), None)
    if hv is None:
        return "vert"
    return "horiz" if hv.startswith("horiz") else "vert"


def _parse_pose_fields(pose_spec: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = re.search(r"dir=([a-zA-Z_|]+)", pose_spec or "")
    if m:
        out["dir"] = m.group(1).split("|")[0].lower()
    m = re.search(r"grip=([^,\)]+)", pose_spec or "")
    if m:
        out["grip_raw"] = m.group(1).strip()
        out["grip"] = _strip_palm_grip(m.group(1))
    m = re.search(r"left_arm=([a-zA-Z_]+)", pose_spec or "")
    if m:
        out["left_arm"] = m.group(1).lower()
    m = re.search(r"head=([a-zA-Z_]+)", pose_spec or "")
    if m:
        out["head"] = m.group(1).lower()
    m = re.search(r"torso=([a-zA-Z_]+)", pose_spec or "")
    if m:
        out["torso"] = m.group(1).lower()
    return out


def _num_from_atom(raw: str) -> int | None:
    s = str(raw or "").strip().lower().replace("≈", "").replace("~", "")
    if not s or s in {"_", "—", "-"}:
        return None
    if "near" in s or "chest" in s:
        return 70
    if "shoulder" in s:
        return 60
    if "mid" in s:
        return 55
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    return int(round(float(m.group(1))))


def _xyz_from_pose_string(pose: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for a in ("x", "y", "z"):
        m = re.search(rf"{a}\s*[=≈]\s*([^,\)]+)", pose or "")
        if not m:
            continue
        v = _num_from_atom(m.group(1))
        if v is not None:
            out[a] = max(0, min(100, v))
    return out


def _xyz_from_db_row(row: Any, key: str) -> dict[str, int]:
    if not isinstance(row, dict):
        return {}
    block = row.get(key)
    out: dict[str, int] = {}
    if isinstance(block, dict):
        for a in ("x", "y", "z"):
            if isinstance(block.get(a), (int, float)):
                out[a] = int(round(float(block[a])))
    # fallback parse *_xyz_str
    s = row.get(f"{key}_xyz_str")
    if isinstance(s, str) and s.strip() not in {"", "—", "-"}:
        for a in ("x", "y", "z"):
            if a in out:
                continue
            m = re.search(rf"{a}\s*=\s*(-?\d+)", s)
            if m:
                out[a] = int(m.group(1))
    return out


def _jitter(cue: str, axis: str, span: int = 4) -> int:
    h = hashlib.md5(f"{cue}:{axis}".encode()).hexdigest()
    return (int(h[:4], 16) % (2 * span + 1)) - span


def select_paddle_xyz(cue: str, fields: dict[str, Any], xyz_db: dict[str, Any]) -> dict[str, int]:
    """Pick paddle-feasible xyz: override → canvas/humanoid → tiago → dir prior, then nudge."""
    arm = fields.get("dir") or "still"
    if "dir" not in fields and fields.get("head"):
        # head-only cues: no EE location required
        return {}

    # Hand-picked body-grid winners take precedence.
    if XYZ_OVERRIDE.is_file():
        try:
            ov = json.loads(XYZ_OVERRIDE.read_text(encoding="utf-8"))
            hit = (ov.get("cues") or {}).get(cue)
            if isinstance(hit, dict) and all(a in hit for a in ("x", "y", "z")):
                return {a: int(max(5, min(95, hit[a]))) for a in ("x", "y", "z")}
        except Exception:
            pass

    priors = dict(_DIR_XYZ.get(arm, _DIR_XYZ["front"]))
    row = (xyz_db.get("cues") or xyz_db).get(cue) or {}
    hum = _xyz_from_db_row(row, "humanoid")
    tia = _xyz_from_db_row(row, "tiago")
    pose_xyz = _xyz_from_pose_string(str(CANVAS[cue].get("pose") or ""))

    out: dict[str, int] = {}
    for a in ("x", "y", "z"):
        # preference order
        for src in (pose_xyz, hum, tia, priors):
            if a in src:
                out[a] = int(src[a])
                break
        else:
            out[a] = int(priors[a])

    # Blend lightly toward dir prior so rematerialized body stays readable
    for a in ("x", "y", "z"):
        out[a] = int(round(0.65 * out[a] + 0.35 * priors[a]))
        out[a] = max(5, min(95, out[a] + _jitter(cue, a)))

    # Dir hard clamps (keep silhouette honest)
    if arm == "up":
        out["z"] = max(out["z"], 75)
    elif arm == "down":
        out["z"] = min(out["z"], 40)
    elif arm == "front":
        out["x"] = max(out["x"], 60)
    elif arm == "back":
        out["x"] = min(out["x"], 40)
    elif arm == "in":
        out["y"] = max(out["y"], 65)
    elif arm == "out":
        out["y"] = min(out["y"], 35)
    return out


def _first_axis(token: str, default: str = "x") -> str:
    for p in re.split(r"[|/]", (token or "").strip().lower()):
        p = p.strip()
        if p in ("x", "y", "z"):
            return p
        if p in ("pan",):
            return "y"
        if p in ("tilt", "roll"):
            return "z"
    return default


def _parse_sign(tok: str) -> str:
    s = (tok or "").strip()
    if "±" in s or "+-" in s or "-+" in s:
        return "+-"
    if "−" in s or s.startswith("-") or "away" in s.lower() or "outward" in s.lower():
        # outward on y is treated as − for claw-right convention downstream when axis=y
        if "outward" in s.lower():
            return "-"
        return "-"
    return "+"


def canvas_move_spec(cue: str) -> dict[str, Any]:
    """Parse canvas movement/path → {kind, axis, sign, rep, joint, plane}."""
    req = CANVAS[cue]
    mv = str(req.get("movement") or "—").strip()
    path = str(req.get("path") or "—").strip()

    if path and not _na(path) and path != "—":
        if "arc" in path:
            plane = "yz"
            m = re.search(r"plane=([a-z|]+)", path)
            if m:
                plane = m.group(1).split("|")[0]
            rep = 2 if "rep" in path else 1
            return {"kind": "arc", "axis": "y", "sign": "+-", "rep": rep, "plane": plane, "joint": "ee"}
        axis = "x"
        m = re.search(r"axis=([xyz])", path)
        if m:
            axis = m.group(1)
        elif re.search(r"(^|[^a-z])y([^a-z]|$)", path):
            axis = "y"
        elif re.search(r"(^|[^a-z])z([^a-z]|$)", path):
            axis = "z"
        sign = "-" if ("−" in path or re.search(r"dist\s*=\s*-", path)) else "+"
        return {"kind": "line", "axis": axis, "sign": sign, "rep": 1, "joint": "ee"}

    if not mv or _na(mv) or mv == "—":
        return {"kind": "hold", "axis": None, "sign": None, "rep": 0, "joint": None}

    # underspec placeholders → hold
    if re.search(r"move\(\s*_\s*,\s*_\s*,\s*_", mv) or "move(wrist, _, _)" in mv.replace(" ", ""):
        return {"kind": "hold", "axis": None, "sign": None, "rep": 0, "joint": None}

    raw = mv.replace(" ", "")
    m = re.match(r"move\(([^,]+),([^,]+),([^,\)]+)", raw)
    if not m:
        # free-text head motions
        if "pan" in mv:
            rep = 2 if "rep" in mv else 1
            sign = _parse_sign(mv)
            return {"kind": "joint", "axis": "y", "sign": sign, "rep": rep if sign == "+-" else (rep if "rep" in mv else 0), "joint": "head"}
        if "tilt" in mv or "roll" in mv:
            return {"kind": "joint", "axis": "z", "sign": "+", "rep": 0, "joint": "head"}
        return {"kind": "hold", "axis": None, "sign": None, "rep": 0, "joint": None}

    joint_raw = m.group(1).strip().lower()
    axis_tok = m.group(2).strip()
    sign_tok = m.group(3).strip()
    if joint_raw in ("wrist", "shoulder", "elbow", "head", "torso"):
        joint = joint_raw
    elif joint_raw == "_":
        joint = "claw"
    else:
        joint = "claw"

    axis = _first_axis(axis_tok, "y" if joint in ("wrist", "head") else "x")
    sign = _parse_sign(sign_tok)
    wants_rep = "rep" in mv
    if sign == "+-":
        rep = 2 if wants_rep else 1  # ± needs at least one cycle
    elif wants_rep:
        rep = 2  # canvas asked for repetition on a directed stroke
    else:
        rep = 0  # one-shot; no looping
    if joint == "head" and sign == "+-" and wants_rep:
        rep = max(rep, 2)
    return {"kind": "joint", "axis": axis, "sign": sign, "rep": rep, "joint": joint}


def canvas_exec_move(cue: str, *, mirror: bool) -> str:
    """Motion-code string faithful to canvas axis / sign / rep."""
    spec = canvas_move_spec(cue)
    kind = spec["kind"]
    if kind == "hold":
        return "hold(t=1.0)"
    if kind == "arc":
        plane = spec.get("plane") or "yz"
        rep = max(int(spec["rep"]), 1)
        return f"path(ee_claw, arc, plane={plane}, r=0.06, sweep=360, rep={rep}, spd=1.6)\n+ hold(t=0.5)"
    if kind == "line":
        axis = spec["axis"] or "x"
        dist = {"x": 0.10, "y": 0.12, "z": 0.08}[axis]
        if spec["sign"] == "-":
            dist = -dist
        return f"path(ee_claw, line, axis={axis}, dist={dist:.2f}, spd=2.0)\n+ hold(t=0.7)"

    joint = spec["joint"] or "claw"
    axis = spec["axis"] or "x"
    sign = spec["sign"] or "+"
    rep = int(spec["rep"] or 0)
    amp = 18 if joint == "wrist" else (22 if joint != "head" else 18)

    if joint == "head":
        ax = "pan" if axis == "y" else "tilt"
        if sign == "+-":
            r = max(rep, 1)
            return f"move(head, {ax}, [-{amp},{amp}], rep={r}, spd=1.6, hold=0.1)"
        if sign == "-":
            return f"move(head, {ax}, [0,-{amp}], rep=1, spd=1.4, hold=0.3)"
        return f"move(head, {ax}, [0,{amp}], rep=1, spd=1.4, hold=0.3)"

    if joint == "torso":
        if axis == "z" or sign == "-":
            return "move(torso, pitch, [0,15], rep=1, spd=1.2, hold=0.4)"
        return "move(torso, pitch, [0,-12], rep=1, spd=1.2, hold=0.4)"

    # claw / wrist / shoulder / elbow → claw (+ paddle if mirror)
    if joint == "wrist":
        ch, ax = "wrist_c", ("roll" if axis in ("z", "x") else "yaw")
    else:
        ch, ax = "claw", ("yaw" if axis == "y" else "pitch")

    if sign == "+-" or rep > 0:
        r = max(rep, 1)
        if sign == "-":
            rng = f"[0,-{amp}]"
        elif sign == "+":
            rng = f"[0,{amp}]"
        else:
            rng = f"[-{amp},{amp}]"
        lines = [f"move({ch}, {ax}, {rng}, rep={r}, spd=2.0, hold=0.06)"]
        if mirror:
            ch2 = "wrist_p" if joint == "wrist" else "paddle"
            lines.append(f"move({ch2}, {ax}, {rng}, rep={r}, spd=2.0, hold=0.06)")
        return "\n+ ".join(lines)
    # one-shot + or −
    rng = f"[0,-{amp}]" if sign == "-" else f"[0,{amp}]"
    lines = [f"move({ch}, {ax}, {rng}, rep=1, spd=2.0, hold=0.15)"]
    if mirror:
        ch2 = "wrist_p" if joint == "wrist" else "paddle"
        lines.append(f"move({ch2}, {ax}, {rng}, rep=1, spd=2.0, hold=0.15)")
    return "\n+ ".join(lines)


def build_paddle_code(cue: str, xyz_db: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    fields = _parse_pose_fields(str(CANVAS[cue].get("pose") or ""))
    arm_dir = fields.get("dir")
    head_only = arm_dir is None and bool(fields.get("head"))
    if head_only:
        arm_dir = "still"
    else:
        arm_dir = arm_dir or "front"

    hv = fields.get("grip") or "vert"
    palm = "edge" if hv == "vert" else "flat"
    left = fields.get("left_arm") or "still"
    mirror = left == "mirror"
    opposed = left == "opposed"

    xyz = select_paddle_xyz(cue, fields, xyz_db)
    move_spec = canvas_move_spec(cue)

    parts = [f"dir=front", "torso=mid"]
    if head_only:
        parts += ["claw=still", "paddle=still", "head=center"]
    else:
        parts += [f"claw={arm_dir}", f"palm={palm}", "grip=open"]
        if mirror:
            parts.append(f"paddle={arm_dir}")
        elif opposed:
            opp = {"front": "back", "back": "front", "in": "out", "out": "in", "up": "up", "down": "down"}.get(
                arm_dir, "front"
            )
            parts.append(f"paddle={opp}")
        else:
            parts.append("paddle=still")
        for a in ("x", "y", "z"):
            if a in xyz:
                parts.append(f"{a}={xyz[a]}")

    adapted = canvas_exec_move(cue, mirror=mirror or opposed)
    chunks = [f"pose({', '.join(parts)})"]
    if adapted:
        chunks.append(adapted)
    if "hold(" not in adapted:
        chunks.append("hold(t=0.8)")
    code = "\n+ ".join(chunks)
    meta = {
        "cue": cue,
        "dir": arm_dir,
        "palm_hv": palm,
        "mirror": mirror,
        "opposed": opposed,
        "head_only": head_only,
        "xyz": xyz,
        "exec_move": adapted,
        "move_spec": move_spec,
        "semantic": semantic_blurb(cue),
        "platform": "gcr_alpha1",
        "ik": bool(_load_ik_joints(cue)),
    }
    return code, meta


def _apply_xyz_offsets(joints: dict[str, float], xyz: dict[str, int], *, gain: float = 1.0) -> None:
    """Drive claw/paddle toward body-percentile xyz on top of a dir silhouette.

    gain scales travel; full-workspace grids need ~1.0 with these coeffs so
    low/high/tucked/reach corners look obviously different.
    """
    if not xyz:
        return
    x = float(xyz.get("x", 55))
    y = float(xyz.get("y", 50))
    z = float(xyz.get("z", 55))
    x_n = (x - 50.0) / 50.0  # −1…+1 ; + = forward
    y_n = (y - 50.0) / 50.0  # + = in
    z_n = (z - 50.0) / 50.0  # + = up
    g = float(gain)

    joints["br0"] = joints.get("br0", 0.0) + y_n * 1.40 * g
    joints["br1"] = joints.get("br1", 0.0) + (-x_n * 1.25 - z_n * 1.60) * g
    joints["br2"] = joints.get("br2", 0.0) + (-x_n * 0.70 - z_n * 0.90) * g
    joints["br3"] = joints.get("br3", 0.0) + y_n * 0.30 * g

    if "bl0" in joints:
        joints["bl0"] = joints.get("bl0", 0.0) - y_n * 1.40 * g
        joints["bl1"] = joints.get("bl1", 0.0) + (-x_n * 1.05 - z_n * 1.35) * g
        joints["bl2"] = joints.get("bl2", 0.0) + (x_n * 0.40 + z_n * 0.50) * g

    joints["trunk3_pitch"] = joints.get("trunk3_pitch", 0.0) - z_n * 0.40 * g
    joints["trunk1_pitch"] = joints.get("trunk1_pitch", 0.0) - x_n * 0.20 * g
    if x_n < -0.25:
        joints["trunk1_yaw"] = joints.get("trunk1_yaw", 0.0) + x_n * 0.30 * g


def _axis_amp(axis: str, sign: str, *, joint: str) -> dict[str, float]:
    """Joint deltas for claw from canvas axis/sign (not always lateral wave)."""
    mag = 0.28 if joint != "wrist" else 0.45
    s = -1.0 if sign == "-" else 1.0
    if axis == "y":
        return {"br0": s * mag}
    if axis == "z":
        return {"br1": -s * mag, "br2": -s * (mag * 0.5)}
    return {"br1": -s * mag, "br2": -s * (mag * 0.35)}


def _load_ik_joints(cue: str) -> dict[str, float] | None:
    candidates = (
        IK_JOINTS_JSON,
        MC / "bank/_paddle_xyz_iter/ik_body/paddle_ik_joints.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            hit = (raw.get("cues") or {}).get(cue)
            if isinstance(hit, dict) and hit:
                return {k: float(v) for k, v in hit.items()}
        except Exception:
            continue
    return None


def _paddle_plan(meta: dict[str, Any]) -> list[dict[str, Any]]:
    arm = meta.get("dir") or "front"
    mirror = bool(meta.get("mirror"))
    opposed = bool(meta.get("opposed"))
    palm = meta.get("palm_hv") or "flat"
    xyz = meta.get("xyz") or {}
    spec = meta.get("move_spec") or {"kind": "hold"}
    cue = str(meta.get("cue") or "")

    ik = _load_ik_joints(cue) if cue else None
    if ik:
        joints = dict(rc.home1())
        joints.update(ik)
        joints.update(rc.A1.get("grip_open", {}))
    elif meta.get("head_only") or arm == "still":
        joints = {**_preset("claw_still"), **_preset("paddle_still"), **rc.A1["grip_open"]}
    else:
        claw_key = f"claw_{arm}" if f"claw_{arm}" in A1_DIRS else "claw_front"
        joints = dict(_preset(claw_key))
        if mirror:
            pad_key = f"paddle_{arm}" if f"paddle_{arm}" in A1_DIRS else "paddle_front"
            joints.update(_preset(pad_key))
        elif opposed:
            opp = {"front": "back", "back": "front", "in": "out", "out": "in"}.get(arm, "front")
            pad_key = f"paddle_{opp}" if f"paddle_{opp}" in A1_DIRS else "paddle_front"
            joints.update(_preset(pad_key))
        else:
            joints.update(_preset("paddle_still"))
        joints.update(rc.A1["grip_open"])
        if mirror or opposed:
            if palm == "edge":
                joints["bl2"] = joints.get("bl2", 0.5) + 0.35
            else:
                joints["bl2"] = joints.get("bl2", 0.9) * 0.7
        _apply_xyz_offsets(joints, xyz)

    # Start AT the pose (no neutral→pose approach).
    plan: list[dict[str, Any]] = [{"op": "snap", "joints": joints, "n": 8}]

    kind = spec.get("kind") or "hold"
    axis = spec.get("axis") or "x"
    sign = spec.get("sign") or "+"
    rep = int(spec.get("rep") or 0)
    joint = spec.get("joint") or "claw"

    if kind == "hold":
        plan.append({"op": "hold", "joints": joints, "n": 18})
    elif kind == "arc":
        r = max(rep, 1)
        plan.append(
            {"op": "osc", "joints": {"br0": 0.35, "br3": 0.40}, "rep": r, "n": 20 + 10 * r, "base_joints": joints}
        )
        plan.append({"op": "hold", "joints": joints, "n": 8})
    elif kind == "line":
        amp = _axis_amp(axis, sign, joint="claw")
        plan.append({"op": "osc", "joints": amp, "rep": 1, "n": 20, "base_joints": joints, "phase": "half"})
        plan.append({"op": "hold", "joints": joints, "n": 8})
    elif kind == "joint" and joint == "head":
        if axis == "y":
            amp = {"neck_yaw": 0.40 if sign != "-" else -0.40}
        else:
            amp = {"neck_pitch": 0.30 if sign != "-" else -0.30}
        if sign == "+-":
            amp = {k: abs(v) for k, v in amp.items()}
            plan.append({"op": "osc", "joints": amp, "rep": max(rep, 1), "n": 28, "base_joints": joints})
            plan.append({"op": "hold", "joints": joints, "n": 8})
        else:
            end = dict(joints)
            for k, v in amp.items():
                end[k] = end.get(k, 0.0) + v
            plan.append({"op": "stroke", "from": joints, "to": end, "n": 12})
            plan.append({"op": "hold", "joints": end, "n": 14})
    elif kind == "joint" and joint == "torso":
        end = dict(joints)
        end["trunk1_pitch"] = end.get("trunk1_pitch", 0.0) + (0.22 if sign != "-" else -0.18)
        plan.append({"op": "stroke", "from": joints, "to": end, "n": 14})
        plan.append({"op": "hold", "joints": end, "n": 12})
    elif kind == "joint":
        # Directed +/− with canvas rep → repeat stroke; ± → bilateral osc.
        if sign == "+-" or rep > 0:
            amp = _axis_amp(axis, "+" if sign != "-" else "-", joint=joint)
            amp = {k: abs(v) for k, v in amp.items()}
            if joint == "wrist":
                amp = {"br3": 0.50}
            if mirror or opposed:
                if "br0" in amp:
                    amp["bl0"] = amp["br0"]
                if "br1" in amp:
                    amp["bl1"] = amp["br1"]
                if "br3" in amp:
                    amp["bl2"] = 0.40
            plan.append(
                {
                    "op": "osc",
                    "joints": amp,
                    "rep": max(rep, 1),
                    "n": 24 + 8 * max(rep, 1),
                    "base_joints": joints,
                }
            )
            plan.append({"op": "hold", "joints": joints, "n": 8})
        else:
            amp = _axis_amp(axis, sign, joint=joint)
            if joint == "wrist":
                amp = {"br3": 0.45 if sign == "+" else -0.45}
            end = dict(joints)
            for k, v in amp.items():
                end[k] = end.get(k, 0.0) + v
            if mirror or opposed:
                if "br0" in amp:
                    end["bl0"] = end.get("bl0", 0.0) + amp["br0"]
                if "br1" in amp:
                    end["bl1"] = end.get("bl1", 0.0) + amp["br1"]
            plan.append({"op": "stroke", "from": joints, "to": end, "n": 16})
            plan.append({"op": "hold", "joints": end, "n": 12})
    else:
        plan.append({"op": "hold", "joints": joints, "n": 18})
    return plan


def _ease(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _execute_plan_snap(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Execute plan starting at pose (no neutral approach)."""
    frames: list[dict[str, Any]] = []
    cur_base = (0.0, 0.0, 0.0)
    cur_joints: dict[str, float] = dict(rc.home1())

    for step in plan:
        op = step["op"]
        if op == "snap":
            cur_joints = dict(rc.home1(step.get("joints")))
            for _ in range(int(step.get("n", 6))):
                frames.append({"base": cur_base, "joints": dict(cur_joints)})
        elif op == "hold":
            cur_joints = dict(rc.home1(step.get("joints") or cur_joints))
            for _ in range(int(step.get("n", 10))):
                frames.append({"base": cur_base, "joints": dict(cur_joints)})
        elif op == "stroke":
            frm = dict(rc.home1(step.get("from") or cur_joints))
            to = dict(rc.home1(step.get("to") or frm))
            n = max(int(step.get("n", 12)), 2)
            for i in range(n):
                t = _ease(i / (n - 1))
                full = dict(frm)
                for k in set(frm) | set(to):
                    a = frm.get(k, 0.0)
                    full[k] = a + (to.get(k, a) - a) * t
                frames.append({"base": cur_base, "joints": full})
            cur_joints = dict(to)
        elif op == "osc":
            amp = step.get("joints") or {}
            root = dict(rc.home1(step.get("base_joints") or cur_joints))
            rep = max(int(step.get("rep", 1)), 1)
            n = max(int(step.get("n", 24)), 2)
            phase = step.get("phase", "full")
            last = None
            for i in range(n):
                t = i / (n - 1)
                s = math.sin(math.pi * t) if phase == "half" else math.sin(2 * math.pi * t * rep)
                j = dict(root)
                for k, a in amp.items():
                    j[k] = root.get(k, 0.0) + float(a) * s
                last = {"base": cur_base, "joints": j}
                frames.append(last)
            if last:
                cur_joints = dict(last["joints"])
        else:
            raise ValueError(op)
    return frames


def build_all(xyz_db: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    idxs = cue_indices()
    configs: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for cue in CUES90:
        idx = int(idxs.get(cue, 0)) or (CUES90.index(cue) + 1)
        code, meta = build_paddle_code(cue, xyz_db)
        row = {
            "idx": idx,
            "cue": cue,
            "description": f"Synth90 paddlebot for {cue}",
            "movements": [],
            "state": "synth90_paddlebot",
            "model": "canvas_orient_move_paddle_xyz",
            "experiment": OUT_TAG,
            "platform": "gcr_alpha1",
            "source_code": code,
            "meta": meta,
            "parse_ok": True,
        }
        configs.append(row)
        manifest.append({"idx": idx, "cue": cue, "paddle_code": code, "paddle_meta": meta})
        pose_line = code.splitlines()[0]
        print(f"[code] {idx:02d} {cue}\n  {pose_line}", flush=True)
    return configs, manifest


def render_all(configs: list[dict[str, Any]], *, force: bool = False) -> int:
    gif_dir = OUT_RENDER / "gcr_alpha1" / "gif"
    gif_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(ODDKIN / "xml" / "gcr_alpha1.xml"))
    lookat, distance, base_z = (0.0, 0.0, 0.40), 2.05, 0.13
    n_done = 0
    for row in configs:
        cue, idx = row["cue"], int(row["idx"])
        out_gif = gif_dir / f"emb_g{idx:02d}_{cue}.gif"
        if out_gif.is_file() and out_gif.stat().st_size > 1000 and not force:
            n_done += 1
            continue
        meta = row.get("meta") or {}
        plan = _paddle_plan(meta)
        frames = _execute_plan_snap(plan)
        rc.render_clip(
            model,
            frames,
            out_gif,
            title=f"{idx}. {cue}",
            subtitle=(row.get("source_code") or "")[:90],
            lookat=lookat,
            distance=distance,
            base_z=base_z,
        )
        n_done += 1
        print(f"[render] {cue} -> {out_gif.name} ({len(frames)}f)", flush=True)
    return n_done


def build_html(configs: list[dict[str, Any]]) -> None:
    gif_dir = OUT_RENDER / "gcr_alpha1" / "gif"
    n_gif = sum(1 for p in gif_dir.glob("emb_g*.gif") if p.is_file() and p.stat().st_size > 1000)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Paddlebot90 Motion Programs</title>",
        "<style>",
        "body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:20px;background:#f4f4f4;color:#222}",
        "h1{font-size:22px;margin:0 0 6px} .sub{color:#555;max-width:1100px;line-height:1.45;margin-bottom:16px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}",
        ".card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:10px}",
        ".card img{width:100%;background:#0d0d0d;border-radius:6px;aspect-ratio:1;object-fit:contain}",
        ".meta{font-size:12px;margin-top:8px} .tok{font-family:ui-monospace,Menlo,monospace;font-size:10.5px;",
        "background:#f0f3f8;padding:6px 7px;border-radius:5px;margin-top:5px;white-space:pre-wrap;word-break:break-word}",
        ".pill{display:inline-block;background:#e8f5e9;border:1px solid #81c784;color:#1b5e20;",
        "border-radius:999px;padding:2px 10px;font-size:12px;margin-right:6px}",
        "</style></head><body>",
        "<h1>Paddlebot90 · canvas orient/move + paddle xyz</h1>",
        f"<p class='sub'><span class='pill'>{n_gif}/{len(configs)} GIFs</span>"
        "Body <code>dir=front</code> = facing; arm dir = canvas <b>claw/paddle</b> "
        "(from <code>required_components_90</code>). "
        "Location = <b>paddle xyz</b> (IK body cues snap to solved joints). "
        "Movement = canvas <b>axis/sign/rep</b> only "
        "(start at pose, no neutral ease-in).</p><div class='grid'>",
    ]
    for row in sorted(configs, key=lambda e: e["idx"]):
        cue, idx = row["cue"], row["idx"]
        gif = gif_dir / f"emb_g{idx:02d}_{cue}.gif"
        rel = os.path.relpath(gif, OUT_HTML.parent) if gif.is_file() else ""
        code = row.get("source_code") or ""
        ik_pill = (
            "<span class='pill' style='background:#e3f2fd;border-color:#64b5f6;color:#0d47a1'>IK</span>"
            if (row.get("meta") or {}).get("ik")
            else ""
        )
        parts.append(
            "<div class='card'>"
            + (f"<img src='{html_mod.escape(rel)}?v=3' alt='{html_mod.escape(cue)}'>" if rel else "<div style='aspect-ratio:1;background:#111'></div>")
            + f"<div class='meta'><b>{idx}. {html_mod.escape(cue)}</b> {ik_pill}</div>"
            + f"<div class='tok'>{html_mod.escape(code)}</div></div>"
        )
    parts.append("</div></body></html>")
    OUT_HTML.write_text("".join(parts), encoding="utf-8")
    print(f"[html] {OUT_HTML}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--codes-only", action="store_true")
    ap.add_argument("--cues", nargs="*", default=None, help="Only render these cue names")
    args = ap.parse_args()

    xyz_db = json.loads(XYZ_JSON.read_text(encoding="utf-8"))
    configs, manifest = build_all(xyz_db)
    OUT_CFG.mkdir(parents=True, exist_ok=True)
    (OUT_CFG / "embodiment_motion_code_all.json").write_text(
        json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # also dump selected xyz table
    xyz_out = {
        "platform": "gcr_alpha1",
        "cues": {r["cue"]: (r.get("meta") or {}).get("xyz") or {} for r in configs},
    }
    (OUT_CFG / "paddle_xyz_90.json").write_text(json.dumps(xyz_out, indent=2), encoding="utf-8")
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[cfg] {len(configs)} programs → {OUT_CFG}", flush=True)

    if not args.codes_only:
        to_render = configs
        if args.cues:
            want = set(args.cues)
            to_render = [r for r in configs if r["cue"] in want]
            print(f"[render-filter] {len(to_render)} cues: {sorted(want)}", flush=True)
        render_all(to_render, force=args.force or bool(args.cues))
    build_html(configs)


if __name__ == "__main__":
    main()
