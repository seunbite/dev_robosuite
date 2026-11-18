#!/usr/bin/env python3
"""Parse Motion Code DSL from gesture_embodiment_table.xlsx → renderable configs.

Adds an Excel sheet, writes JSON, and optionally renders Tiago + Manipulator.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_GOOGLE = _REPO / "adhoc/generation/tiago"
for p in (_REPO, _HERE, _GOOGLE, _REPO / "adhoc/generation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DEFAULT_XLSX = Path(
    "/Users/sb/Library/Application Support/Claude/local-agent-mode-sessions/"
    "2c2cb05d-69f7-4540-9b45-083b53d3c3f6/"
    "6bc2a0ab-1d3f-4800-a82e-32c096149213/"
    "local_f6b1d6a9-bbfc-405f-aa49-c9ffd6b0cb8c/outputs/gesture_embodiment_table.xlsx"
)
OUT_JSON_DIR = _REPO / "data/results/motion_configs/embodiment_motion_code"
OUT_RENDER = _REPO / "data/results/render/embodiment_motion_code"
SHEET_NAME = "Renderable JSON"

PLATFORM_COLS = {
    4: "humanoid",
    5: "tiago",
    6: "single_arm",
    7: "quadruped",
    8: "roomba",
    9: "stretch",  # Hello Robot Stretch 3 (new; after legacy 5 cols)
}
# Display / eval order: stretch sits between single_arm and roomba.
PLATFORMS_ORDER = (
    "humanoid",
    "tiago",
    "single_arm",
    "stretch",
    "quadruped",
    "roomba",
)
PLATFORM_HEADERS = {
    "humanoid": "Humanoid (Unitree G1)",
    "tiago": "Tiago",
    "single_arm": "Single-arm (IIWA)",
    "stretch": "Stretch (Hello Robot)",
    "quadruped": "Quadruped (Unitree Go2)",
    "roomba": "Roomba",
}

GRIP_MAP = {
    "horiz": "horizontal",
    "horizontal": "horizontal",
    "vert": "vertical",
    "vertical": "vertical",
    "close": "vertical",
    "closed": "vertical",
    "open": "horizontal",
    "pinch": "vertical",
    "palmup": "palmup",
    "palm_up": "palmup",
    "palmdown": "palmdown",
    "palm_down": "palmdown",
    "palmout": "palmout",
    "palm_out": "palmout",
    "palmin": "palmin",
    "palm_in": "palmin",
    "palmfront": "palmfront",
    "palm_front": "palmfront",
    "palmback": "palmback",
    "palm_back": "palmback",
}
_PALM_GRIPS = {
    "palmup",
    "palmdown",
    "palmout",
    "palmin",
    "palmfront",
    "palmback",
}


def normalize_gripper_orientation(raw: Any) -> str:
    """Normalize grip tokens.

    Compounds like ``vert|palmin`` keep both parts as ``vertical|palmin`` so the
    humanoid bank can filter by wrist class (vertical/horizontal) first, then
    palm normal. Lone palm / lone hv tokens stay as before.
    """
    s = str(raw or "").lower().replace("_", "").replace(" ", "")
    if not s:
        return "vertical"
    parts = [p for p in re.split(r"[|/+,]", s) if p]
    if not parts:
        return "vertical"
    mapped = [GRIP_MAP.get(p, p) for p in parts]
    palm = next((p for p in mapped if p in _PALM_GRIPS), None)
    hv = next((p for p in mapped if p in ("horizontal", "vertical")), None)
    if hv and palm:
        return f"{hv}|{palm}"
    if palm is not None:
        return palm
    if hv is not None:
        return hv
    return mapped[0]
TORSO_MAP = {"tall": "high", "high": "high", "mid": "mid", "low": "low"}
DIR_MAP = {
    "front": "front",
    "back": "back",
    "in": "in",
    "out": "out",
    "up": "up",
    "down": "down",
    "still": "still",
}


def _strip_noise(s: str) -> str:
    s = s.replace("°", "")
    # Keep ∓ — parsed in _parse_value as opposite-phase ping-pong of ±.
    s = re.sub(r"\s+", " ", s)
    # Some spreadsheet cells wrap the whole Motion Code string in quotes. After
    # top-level splitting, only the first/last primitive retains that stray quote.
    return s.strip().strip("\"'")


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    # Tolerate spreadsheet shorthand typos such as ``+-0.15`` / ``-+0.15``.
    # They carry the same ping-pong meaning as ``±0.15``.
    m = re.fullmatch(r"(?:\+\-|\-\+)\s*(\d+(?:\.\d+)?)", raw)
    if m:
        v = float(m.group(1))
        return [-v, v]
    # bracket range / list
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        items = _split_top_level(inner)
        out = []
        for it in items:
            out.append(_parse_value(it))
        return out
    # ±x  →  [-x, +x]  (ping-pong)
    m = re.fullmatch(r"[±]\s*([+-]?\d+(?:\.\d+)?)", raw)
    if m:
        v = float(m.group(1))
        return [-v, v]
    # ∓x  →  [+x, -x]  (same amplitude, opposite phase — e.g. paired with ± on other leg)
    m = re.fullmatch(r"[∓]\s*([+-]?\d+(?:\.\d+)?)", raw)
    if m:
        v = float(m.group(1))
        return [v, -v]
    # plain number
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", raw):
        return float(raw) if "." in raw else int(raw)
    # placeholder tokens like [orient_to_person]
    if raw.startswith("[") or raw.endswith("]"):
        return raw.strip("[]")
    return raw


def _parse_kv_list(args: str) -> tuple[list[Any], dict[str, Any]]:
    """Parse mixed positional + key=value args.

    Spreadsheet cells often wrap after a value without a trailing comma, e.g.
    ``x=65\\n  left_arm=mirror``. After whitespace collapse that becomes
    ``x=65 left_arm=mirror`` and would otherwise be eaten as a single broken
    ``x`` value — insert the missing commas before bare ``key=`` tokens.
    """
    args = re.sub(r"(\S)\s+([A-Za-z_][\w]*)\s*=", r"\1, \2=", args)
    positionals: list[Any] = []
    kwargs: dict[str, Any] = {}
    for part in _split_top_level(args):
        if "=" in part:
            k, v = part.split("=", 1)
            kwargs[k.strip()] = _parse_value(v.strip())
        else:
            positionals.append(_parse_value(part))
    return positionals, kwargs


def _split_steps(code: str) -> list[str]:
    """Split on top-level ``+`` / ``&`` separators (see ``_split_steps_with_ops``)."""
    return [step for _op, step in _split_steps_with_ops(code)]


def _split_steps_with_ops(code: str) -> list[tuple[str, str]]:
    """Split Motion Code into ``(op, step)`` pairs.

    ``op`` is ``""`` for the first step, else the preceding top-level separator
    (``+`` sequential, ``&`` parallel / ``order=together`` with the previous step).
    """
    code = code.replace("\n", " ")
    out: list[tuple[str, str]] = []
    depth = 0
    cur: list[str] = []
    pending_op = ""
    i = 0
    while i < len(code):
        ch = code[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch in "+&" and depth == 0:
            part = "".join(cur).strip()
            if part:
                out.append((pending_op, part))
            pending_op = ch
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    part = "".join(cur).strip()
    if part:
        out.append((pending_op, part))
    return out


def _parse_call(step: str) -> tuple[str, list[Any], dict[str, Any]] | None:
    step = _strip_noise(step)
    m = re.match(r"^([a-zA-Z_]+)\((.*)\)$", step, flags=re.S)
    if not m:
        return None
    name = m.group(1).strip().lower()
    args = m.group(2).strip()
    pos, kw = _parse_kv_list(args)
    return name, pos, kw


def _boolish(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    s = str(v).strip().upper()
    if s in ("T", "TRUE", "1", "YES"):
        return True
    if s in ("F", "FALSE", "0", "NO"):
        return False
    return None


def _degrees_pair(v: Any) -> list[float]:
    if isinstance(v, list) and len(v) >= 2:
        return [float(v[0]), float(v[1])]
    if isinstance(v, list) and len(v) == 1:
        x = float(v[0])
        return [0.0, x]
    if isinstance(v, (int, float)):
        x = float(v)
        # single signed value → [0, x] one-way; keep sign
        return [0.0, x]
    return [0.0, 0.0]


def pose_to_tiago(kwargs: dict[str, Any], positionals: list[Any]) -> dict[str, Any]:
    pose: dict[str, Any] = {}
    # allow bare keys from kwargs
    if "dir" in kwargs:
        d = str(kwargs["dir"]).lower()
        pose["arm_position"] = DIR_MAP.get(d, d)
    if "grip" in kwargs:
        pose["gripper_orientation"] = normalize_gripper_orientation(kwargs["grip"])
    if "head" in kwargs:
        pose["head"] = str(kwargs["head"]).lower()
    if "torso" in kwargs:
        t = str(kwargs["torso"]).lower()
        pose["torso_height"] = TORSO_MAP.get(t, t)
    for k in ("x", "y", "z"):
        if k in kwargs and isinstance(kwargs[k], (int, float)):
            pose[k] = int(round(float(kwargs[k])))
    if "left_arm" in kwargs:
        pose["left_arm"] = str(kwargs["left_arm"]).lower()
    if "elbow" in kwargs:
        b = _boolish(kwargs["elbow"])
        if b is not None:
            pose["elbow_bended"] = b
    if "bank" in kwargs:
        b = _boolish(kwargs["bank"])
        if b is not None:
            pose["use_feasible_bank"] = b
    # defaults so renderer always has something.
    # No ``dir`` → still (arms down); head-only cues like nod_yes must not
    # invent a raised "front" arm.
    pose.setdefault("arm_position", "still")
    pose.setdefault("gripper_orientation", "vertical")
    pose.setdefault("head", "center")
    pose.setdefault("torso_height", "mid")
    # ignore stance / front_legs etc. (quadruped-only)
    return pose


def pose_to_manip(kwargs: dict[str, Any]) -> dict[str, Any]:
    pose: dict[str, Any] = {}
    if "dir" in kwargs:
        d = DIR_MAP.get(str(kwargs["dir"]).lower(), str(kwargs["dir"]).lower())
        # Motion Code uses body-relative in/out; IIWA bank uses left/right.
        # Keep front/back/up/down unchanged so semantic direction is preserved.
        d = {"in": "left", "out": "right", "still": "front"}.get(d, d)
        pose["dir"] = d
    else:
        pose["dir"] = "front"
    if "grip" in kwargs:
        pose["gripper_orientation"] = normalize_gripper_orientation(kwargs["grip"])
    else:
        pose["gripper_orientation"] = "vertical"
    for k in ("x", "y", "z"):
        if k in kwargs and isinstance(kwargs[k], (int, float)):
            pose[k] = int(round(float(kwargs[k])))
    # Do NOT invent x/y/z — omitted axes stay omitted (renderer must not force them).
    return pose


def pose_to_stretch(kwargs: dict[str, Any], positionals: list[Any]) -> dict[str, Any]:
    """Stretch pose — Tiago-like DSL fields, unimanual (no left_arm)."""
    pose = pose_to_tiago(kwargs, positionals)
    pose.pop("left_arm", None)
    return pose


def convert_step_stretch(step: str) -> dict[str, Any] | None:
    """Stretch Motion Code → JSON. First-class hold; no left_arm."""
    parsed = _parse_call(step)
    if not parsed:
        return None
    name, pos, kw = parsed
    if name == "hold":
        t = float(kw.get("t", pos[0] if pos else 1.0))
        return {
            "type": "hold",
            "duration": t,
            "parameters": {"t": t, "hold_time": t},
        }
    if name == "pose":
        pose = pose_to_stretch(kw, pos)
        return {
            "type": "pose",
            "duration": 0.6,
            "parameters": {"pose": pose},
        }
    # Reuse Tiago move/path parsing, then strip left_arm if a pose sneaks in.
    out = convert_step_tiago(step)
    if out is None:
        return None
    if out.get("type") == "pose":
        pose = (out.get("parameters") or {}).get("pose") or {}
        pose.pop("left_arm", None)
    return out


def convert_step_tiago(step: str) -> dict[str, Any] | None:
    parsed = _parse_call(step)
    if not parsed:
        return None
    name, pos, kw = parsed

    if name == "pose":
        pose = pose_to_tiago(kw, pos)
        dur = 0.6
        return {
            "type": "pose",
            "duration": dur,
            "parameters": {"pose": pose},
        }

    if name == "hold":
        # First-class freeze (not a zero-distance path — that invents a fake path.axis).
        t = float(kw.get("t", pos[0] if pos else 1.0))
        return {
            "type": "hold",
            "duration": t,
            "parameters": {"t": t, "hold_time": t},
        }

    if name in ("move", "movement"):
        # move(joint, axis, degrees, rep=, spd=, hold=)
        joint = str(pos[0]).lower() if pos else str(kw.get("joint", "wrist")).lower()
        axis = str(pos[1]).lower() if len(pos) > 1 else str(kw.get("axis", "y")).lower()
        deg_raw = pos[2] if len(pos) > 2 else kw.get("degrees", kw.get("deg", [0, 10]))
        # quadruped stubs like right_front / front_legs → map to arm-ish joints
        joint_alias = {
            "right_front": "shoulder",
            "left_front": "shoulder",
            "front_legs": "torso",
        }
        joint = joint_alias.get(joint, joint)
        # move(torso, height, low|mid|high, ...)
        if joint == "torso" and (
            axis == "height" or str(deg_raw).lower() in ("low", "mid", "high", "tall")
        ):
            level = str(deg_raw).lower() if isinstance(deg_raw, str) else "low"
            if level == "tall":
                level = "high"
            if level not in ("low", "mid", "high"):
                level = "low"
            rep = int(kw.get("rep", kw.get("repetition", 1)))
            speed = float(kw.get("spd", kw.get("speed", 1.0)))
            hold = kw.get("hold", kw.get("hold_time"))
            mv = {
                "joints": [{"joint": "torso", "axis": "height", "height": level}],
                "repetition": max(1, rep),
                "speed": speed,
            }
            if hold is not None:
                mv["hold_time"] = float(hold)
            return {"type": "movement", "parameters": {"movement": mv}}
        if joint == "torso" and axis in ("x", "y"):
            # torso cartesian wobble → height oscillation as readable proxy
            axis = "height"
            if isinstance(deg_raw, (int, float)):
                deg_raw = [0.18, 0.18 + float(deg_raw) * 0.01]
            elif isinstance(deg_raw, list) and deg_raw and all(isinstance(x, (int, float)) for x in deg_raw):
                if abs(float(deg_raw[0])) > 1 or abs(float(deg_raw[-1])) > 1:
                    deg_raw = [0.18, 0.28]
        degrees = _degrees_pair(deg_raw)
        if joint == "torso" and axis == "height":
            # height uses meters in [low, high]
            if abs(degrees[0]) > 1 or abs(degrees[1]) > 1:
                degrees = [0.18, 0.34]
        rep = int(kw.get("rep", kw.get("repetition", 1)))
        speed = float(kw.get("spd", kw.get("speed", 1.0)))
        hold = kw.get("hold", kw.get("hold_time"))
        mv: dict[str, Any] = {
            "joints": [{"joint": joint, "axis": axis, "degrees": degrees}],
            "repetition": max(1, rep),
            "speed": speed,
        }
        if hold is not None:
            mv["hold_time"] = float(hold)
        return {"type": "movement", "parameters": {"movement": mv}}

    if name == "path":
        # path(mode, shape, ...) — also accept positional axis/distance:
        #   path(ee, line, z, dist=0.3)  /  path(ee, line, z, 0.3, spd=…)
        mode = str(pos[0]).lower() if pos else str(kw.get("mode", "ee")).lower()
        shape = str(pos[1]).lower() if len(pos) > 1 else str(kw.get("shape", "line")).lower()
        path: dict[str, Any] = {"mode": mode, "shape": shape}
        pos_i = 2
        if (
            len(pos) > pos_i
            and isinstance(pos[pos_i], str)
            and str(pos[pos_i]).lower() in ("x", "y", "z")
        ):
            path["axis"] = str(pos[pos_i]).lower()
            pos_i += 1
        if "axis" in kw:
            path["axis"] = str(kw["axis"]).lower()
        if "plane" in kw:
            path["plane"] = str(kw["plane"]).lower()
        if "dist" in kw or "distance" in kw:
            d = kw.get("dist", kw.get("distance"))
            if isinstance(d, list):
                nums = [float(x) for x in d if isinstance(x, (int, float))]
                # Keep [lo, hi] spans (beckon dist=[-0.12, 0.02]); do NOT collapse
                # to the last endpoint — that made strokes ~2cm and looked frozen.
                if len(nums) >= 2:
                    path["distance"] = [nums[0], nums[-1]]
                else:
                    path["distance"] = float(nums[0]) if nums else 0.1
            else:
                try:
                    path["distance"] = float(d)
                except (TypeError, ValueError):
                    path["distance"] = 0.1
        elif len(pos) > pos_i and isinstance(pos[pos_i], (int, float, list)):
            d = pos[pos_i]
            if isinstance(d, list):
                nums = [float(x) for x in d if isinstance(x, (int, float))]
                if len(nums) >= 2:
                    path["distance"] = [nums[0], nums[-1]]
                else:
                    path["distance"] = float(nums[0]) if nums else 0.1
            else:
                path["distance"] = float(d)
        if "r" in kw or "radius" in kw:
            try:
                path["radius"] = float(kw.get("r", kw.get("radius")))
            except (TypeError, ValueError):
                path["radius"] = 0.15
        if "deg" in kw or "degrees" in kw or "sweep" in kw:
            deg = kw.get("deg", kw.get("degrees", kw.get("sweep")))
            if isinstance(deg, list):
                nums = [float(x) for x in deg if isinstance(x, (int, float))]
                # Base arcs: keep [lo, hi] so Stretch/Roomba can ping-pong yaw
                # (deg=±30). EE arcs keep a scalar sweep (last endpoint).
                if mode == "base" and len(nums) >= 2:
                    path["degrees"] = [nums[0], nums[-1]]
                elif nums:
                    path["sweep"] = float(nums[-1])
                else:
                    path["sweep"] = 30.0  # symbolic list e.g. [orient_to_person]
            elif isinstance(deg, (int, float)):
                path["sweep"] = float(deg)
            else:
                # symbolic like orient_to_person → small yaw
                path["sweep"] = 30.0
        if "x" in kw:
            x = kw["x"]
            if isinstance(x, list):
                nums = [float(v) for v in x if isinstance(v, (int, float))]
                path["x"] = float(nums[-1]) if nums else 0.1
            else:
                try:
                    path["x"] = float(x)
                except (TypeError, ValueError):
                    path["x"] = 0.1
        if "y" in kw:
            y = kw["y"]
            if isinstance(y, list):
                nums = [float(v) for v in y if isinstance(v, (int, float))]
                path["y"] = float(nums[-1]) if nums else 0.0
            else:
                try:
                    path["y"] = float(y)
                except (TypeError, ValueError):
                    path["y"] = 0.0
        if "spd" in kw or "speed" in kw:
            path["speed"] = float(kw.get("spd", kw.get("speed")))
        if "hold" in kw or "hold_time" in kw:
            path["hold_time"] = float(kw.get("hold", kw.get("hold_time")))
        if "rep" in kw or "repetition" in kw:
            path["repetition"] = int(kw.get("rep", kw.get("repetition")))
        # line defaults
        if shape == "line" and mode == "ee" and "axis" not in path:
            path["axis"] = "x"
        if shape == "line" and mode == "ee" and "distance" not in path:
            if "x" in path:
                path["distance"] = path.pop("x")
                path["axis"] = "x"
            else:
                path["distance"] = 0.1
        if shape == "arc" and "radius" not in path:
            path["radius"] = 0.15
        if shape == "arc" and "sweep" not in path:
            path["sweep"] = 90.0
        if shape == "arc" and mode == "ee" and "plane" not in path:
            path["plane"] = "xy"
        return {"type": "path", "parameters": {"path": path}}

    return None


def convert_step_manip(step: str) -> dict[str, Any] | None:
    """Manipulator/single-arm variant of convert_step_tiago."""
    parsed = _parse_call(step)
    if not parsed:
        return None
    name, pos, kw = parsed

    if name == "pose":
        pose = pose_to_manip(kw)
        return {
            "type": "pose",
            "parameters": {"pose": pose, "speed": 1.2, "hold_time": 0.1},
        }

    if name == "hold":
        # IIWA execute_cue has no first-class hold type — freeze via zero EE path.
        t = float(kw.get("t", pos[0] if pos else 1.0))
        return {
            "type": "path",
            "parameters": {
                "shape": "line",
                "axis": "x",
                "distance": 0.0,
                "speed": 0.5,
                "hold_time": t,
            },
        }

    if name in ("move", "movement"):
        joint = str(pos[0]).lower() if pos else "wrist"
        axis = str(pos[1]).lower() if len(pos) > 1 else "y"
        deg_raw = pos[2] if len(pos) > 2 else kw.get("degrees", [0, 10])
        degrees = _degrees_pair(deg_raw)
        rep = int(kw.get("rep", 1))
        speed = float(kw.get("spd", kw.get("speed", 1.0)))
        hold = float(kw.get("hold", kw.get("hold_time", 0.0)))
        # manipulator movement uses directions list with degrees dict
        dirs = [
            {"degrees": {axis: degrees[0]}, "speed": speed, "hold_time": 0.0},
            {"degrees": {axis: degrees[1]}, "speed": speed, "hold_time": hold},
        ]
        return {
            "type": "movement",
            "parameters": {
                "repetition": max(1, rep),
                "joint": joint if joint in ("shoulder", "elbow", "wrist") else "wrist",
                "directions": dirs,
            },
        }

    if name == "path":
        mode = str(pos[0]).lower() if pos else "ee"
        shape = str(pos[1]).lower() if len(pos) > 1 else "line"
        # manipulator path usually omits mode in some styles; keep mode parsed.
        # Accept positional axis/distance: path(ee, line, z, dist=0.3)
        params: dict[str, Any] = {"shape": shape}
        pos_i = 2
        if (
            len(pos) > pos_i
            and isinstance(pos[pos_i], str)
            and str(pos[pos_i]).lower() in ("x", "y", "z")
        ):
            params["axis"] = str(pos[pos_i]).lower()
            pos_i += 1
        if "axis" in kw:
            axis = str(kw["axis"]).lower()
            # Human-GT rub_eye_tired contains the spreadsheet typo ``axis=7``;
            # its vertical eye-rubbing path and z=80 setup make the intended
            # Cartesian axis unambiguously z.
            params["axis"] = "z" if axis == "7" else axis
        if "plane" in kw:
            params["plane"] = str(kw["plane"]).lower()
        if "dist" in kw or "distance" in kw:
            d = kw.get("dist", kw.get("distance"))
            # Preserve ± spans as [lo, hi] so the renderer can ping-pong one set per rep.
            if isinstance(d, list) and len(d) >= 2 and all(isinstance(x, (int, float)) for x in d[:2]):
                params["distance"] = [float(d[0]), float(d[-1])]
            else:
                params["distance"] = float(d[0] if isinstance(d, list) and d else d)
        elif len(pos) > pos_i and isinstance(pos[pos_i], (int, float, list)):
            d = pos[pos_i]
            if isinstance(d, list) and len(d) >= 2 and all(isinstance(x, (int, float)) for x in d[:2]):
                params["distance"] = [float(d[0]), float(d[-1])]
            else:
                params["distance"] = float(d[0] if isinstance(d, list) and d else d)
        elif "x" in kw:
            params["axis"] = "x"
            x = kw["x"]
            if isinstance(x, list) and len(x) >= 2 and all(isinstance(v, (int, float)) for v in x[:2]):
                params["distance"] = [float(x[0]), float(x[-1])]
            else:
                params["distance"] = float(x[0] if isinstance(x, list) and x else x)
        if "r" in kw or "radius" in kw:
            try:
                params["radius"] = float(kw.get("r", kw.get("radius")))
            except (TypeError, ValueError):
                params["radius"] = 0.15
        if "deg" in kw or "sweep" in kw:
            deg = kw.get("deg", kw.get("sweep"))
            params["sweep"] = float(deg[1] if isinstance(deg, list) else deg) if not isinstance(deg, str) else 90.0
        if "spd" in kw or "speed" in kw:
            params["speed"] = float(kw.get("spd", kw.get("speed")))
        if "hold" in kw:
            params["hold_time"] = float(kw["hold"])
        if "rep" in kw:
            params["repetition"] = int(kw["rep"])
        if shape == "line" and "axis" not in params:
            params["axis"] = "x"
        if shape == "line" and "distance" not in params:
            params["distance"] = 0.1
        if shape == "arc" and "radius" not in params:
            params["radius"] = 0.15
        if shape == "arc" and "sweep" not in params:
            params["sweep"] = 90.0
        if shape == "arc" and "plane" not in params:
            params["plane"] = "xy"
        # Soft-normalize only scalar distances; keep [lo, hi] spans as-is.
        if isinstance(params.get("distance"), (int, float)) and abs(params["distance"]) < 1:
            params["distance"] = float(params["distance"])
        del mode  # unused but parsed
        return {"type": "path", "parameters": params}

    return None


def _finalize_quadruped_step(step: dict[str, Any]) -> dict[str, Any]:
    """Go2 MJLab path steps need ``velocity``; pose_to_pose keeps duration + zero velocity."""
    if step.get("type") != "path":
        return step
    path_p = (step.get("parameters") or {}).get("path") or {}
    shape = str(path_p.get("shape", "")).lower()
    if shape == "pose_to_pose":
        out = dict(step)
        params = dict(out.get("parameters") or {})
        pp = dict(path_p)
        pp.setdefault("velocity", {"vx": 0.0, "vy": 0.0, "wz": 0.0})
        vel = pp.get("velocity") or {}
        if not any(abs(float(vel.get(k, 0.0))) > 1e-6 for k in ("vx", "vy", "wz")):
            pp.setdefault("gait_blend_xy", 0.0)
            pp.setdefault("gait_blend_z", 0.0)
            # Head tilt/pan bows: free the root so feet plant under gravity
            # instead of floating at a held base height.
            pp.setdefault("dynamic_base", True)
        params["path"] = pp
        out["parameters"] = params
        out.setdefault("duration", 0.6)
        return out
    if path_p.get("velocity"):
        return step
    from dev_locomotion.retrieval import normalize_step

    stripped = dict(step)
    stripped.pop("duration", None)
    return normalize_step(stripped)


def _quad_pose_step(
    *,
    direction: str = "neutral",
    front_height: str = "mid",
    back_height: str = "mid",
    three_legs: bool = False,
    hand: str | None = None,
    duration: float = 0.6,
) -> dict[str, Any]:
    pose: dict[str, Any] = {
        "dir": direction,
        "front_height": front_height,
        "back_height": back_height,
        "three_legs": three_legs,
    }
    if three_legs:
        pose["three_leg_pose"] = {"hand_position": hand or "up"}
    return {"type": "pose", "duration": duration, "parameters": {"pose": pose}}


def _quad_pose_dict(
    direction: str = "neutral",
    front_height: str = "mid",
    back_height: str = "mid",
) -> dict[str, Any]:
    return {
        "dir": direction,
        "front_height": front_height,
        "back_height": back_height,
        "three_legs": False,
    }


def _quad_pose_to_pose_step(
    start: dict[str, Any],
    end: dict[str, Any],
    *,
    duration: float,
    speed: float = 1.0,
    static_feet: bool = False,
) -> dict[str, Any]:
    path: dict[str, Any] = {
        "shape": "pose_to_pose",
        "start_pose": start,
        "end_pose": end,
        "speed": speed,
    }
    if static_feet:
        # No gait stomping; plant on the ground under physics (mjlab dynamic_base).
        path["gait_blend_xy"] = 0.0
        path["gait_blend_z"] = 0.0
        path["dynamic_base"] = True
        path["velocity"] = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
    return {
        "type": "path",
        "duration": duration,
        "parameters": {"path": path},
    }


def _height_from_signed(v: float) -> str:
    """Map signed leg-z / tilt amount → height level.

    Convention for forelegs z (and body tilt):
      + → high (sit-like: front legs more extended)
      − → low (play_bow-like: front legs more flexed / angles close)
    """
    if v > 5:
        return "high"
    if v < -5:
        return "low"
    return "mid"


def _dir_from_signed_pan(v: float) -> str:
    """Map signed pan amount → lateral lean (positive ≈ left side down/lean left)."""
    if v > 5:
        return "left"
    if v < -5:
        return "right"
    return "neutral"


def _quad_neutral_state() -> dict[str, Any]:
    return {
        "dir": "neutral",
        "front_height": "mid",
        "back_height": "mid",
        "three_legs": False,
        "lifted_leg": "none",
    }


def _quad_lifted_leg_from_dir(direction: str, three_legs: bool) -> str:
    """Mirror retrieval._lifted_leg_from_pose: dir=left → FL, else FR when braced."""
    if not three_legs:
        return "none"
    return "FL" if str(direction).lower() == "left" else "FR"


def _quad_pair_z_delta_from_deg(deg: float) -> float:
    """Map signed pair-leg z degrees → soft foot-z delta (meters).

    Sign matches sit / play_bow:
      +deg → extend (foot farther down, negative Δz)
      −deg → flex (foot toward body, positive Δz)
    Amplitude is capped so nod_yes stays a planted joint bob, not a head-slam.
    """
    # ~1.0 mm/deg, hard-cap ±3.5 cm (even |90| stays gentle).
    raw = -float(deg) * 0.0010
    return max(-0.035, min(0.035, raw))


def _quad_pair_height_z_steps(
    *,
    state: dict[str, Any],
    which: str,
    degrees: list[float],
    repetition: int,
    speed: float,
    hold: float,
) -> list[dict[str, Any]]:
    """Pair-leg z → full sit / play_bow silhouettes (same as stance poses).

    User / schema contract for planted pair z (esp. ``front_legs``):
      −deg → play_bow (front low, back high — nose down)
      +deg → sit (front high, back low — haunches down)
      [-a, +a] → alternate those two

    Emit ordinary ``pose`` steps (smoothstep ease under gravity), **not**
    ``pose_to_pose`` + ``dynamic_base`` soft foot deltas — those thrash,
    especially when only the front pair height flips.
    """
    steps: list[dict[str, Any]] = []
    first, second = degrees[0], degrees[-1]
    opposite_osc = first * second < 0 and abs(abs(first) - abs(second)) <= max(
        1.0, 0.05 * max(abs(first), abs(second))
    )
    # Scalar one-way: move(..., z, 20) → [0, 20] via _degrees_pair.
    one_way_from_zero = abs(first) < 1e-9 and abs(second) >= 5.0
    authored_range = abs(first - second) > 1e-6 and not one_way_from_zero

    def _silhouette_from_signed(deg: float) -> tuple[str, str]:
        """Return (front_height, back_height). |deg|<5 → mid/mid."""
        if deg > 5:
            return "high", "low"  # sit
        if deg < -5:
            return "low", "high"  # play_bow
        return "mid", "mid"

    def _pose_dict(fh: str, bh: str) -> dict[str, Any]:
        return {
            "dir": "neutral",
            "front_height": fh,
            "back_height": bh,
            "three_legs": False,
            "front_z_delta": 0.0,
            "back_z_delta": 0.0,
        }

    current_fh = str(state.get("front_height", "mid")).lower()
    current_bh = str(state.get("back_height", "mid")).lower()
    rest = _pose_dict(current_fh, current_bh)

    # Segment duration: match expressive pose holds, never thrash at spd=3.
    seg = max(0.7, 1.0 / max(min(speed, 1.4), 0.2))

    if opposite_osc or (authored_range and first * second < 0):
        # Alternate play_bow ↔ sit in authored order.
        a_fh, a_bh = _silhouette_from_signed(first)
        b_fh, b_bh = _silhouette_from_signed(second)
        cycle = [rest, _pose_dict(a_fh, a_bh), _pose_dict(b_fh, b_bh), rest]
    elif authored_range:
        # e.g. [-15, 0] / [0, 15]: visit signed extremes then return.
        a_fh, a_bh = _silhouette_from_signed(first)
        b_fh, b_bh = _silhouette_from_signed(second)
        cycle = [rest, _pose_dict(a_fh, a_bh), _pose_dict(b_fh, b_bh), rest]
    else:
        target_deg = second if abs(second) >= abs(first) else first
        t_fh, t_bh = _silhouette_from_signed(target_deg)
        if (t_fh, t_bh) == (current_fh, current_bh):
            # Already there — no-op short hold so timing still advances.
            if hold <= 0:
                steps.append(
                    _quad_pose_step(
                        direction="neutral",
                        front_height=current_fh,
                        back_height=current_bh,
                        duration=max(0.3, seg * 0.5),
                    )
                )
            last_fh, last_bh = current_fh, current_bh
            if hold > 0:
                steps.append(
                    _quad_pose_step(
                        direction="neutral",
                        front_height=last_fh,
                        back_height=last_bh,
                        duration=hold,
                    )
                )
            state["front_height"] = last_fh
            state["back_height"] = last_bh
            state["front_z_delta"] = 0.0
            state["back_z_delta"] = 0.0
            state["dir"] = "neutral"
            return steps
        cycle = [rest, _pose_dict(t_fh, t_bh)]

    # which=back keeps the same sit/play_bow pitch contract (YAML: z± shared).
    del which  # silhouette is full-body; pair name only selects this code path.

    for _rep in range(repetition):
        for i in range(len(cycle) - 1):
            end = cycle[i + 1]
            steps.append(
                _quad_pose_step(
                    direction="neutral",
                    front_height=str(end["front_height"]),
                    back_height=str(end["back_height"]),
                    duration=seg,
                )
            )
    last = cycle[-1]
    last_fh = str(last["front_height"])
    last_bh = str(last["back_height"])
    state["dir"] = "neutral"
    state["front_height"] = last_fh
    state["back_height"] = last_bh
    state["front_z_delta"] = 0.0
    state["back_z_delta"] = 0.0
    if hold > 0:
        steps.append(
            _quad_pose_step(
                direction="neutral",
                front_height=last_fh,
                back_height=last_bh,
                duration=hold,
            )
        )
    return steps


def _quad_body_attitude_steps(
    *,
    state: dict[str, Any],
    axis: str,
    degrees: list[float],
    repetition: int,
    speed: float,
    hold: float,
) -> list[dict[str, Any]]:
    """Body pitch/roll via pose_to_pose (Go2 has no neck — do not use ``head``)."""
    steps: list[dict[str, Any]] = []
    seg = max(0.45, 0.65 / max(speed, 0.2))
    first, second = degrees[0], degrees[-1]
    oscillating = first * second < 0 and abs(abs(first) - abs(second)) <= max(
        1.0, 0.05 * max(abs(first), abs(second))
    )
    current = dict(state)

    if axis in ("tilt", "pitch"):
        # Body pitch bow via front_height (held-base attitude).
        def _tilt_pose(height: str) -> dict[str, Any]:
            return {
                **current,
                "dir": "neutral",
                "front_height": height,
                "back_height": str(current.get("back_height", "mid")),
            }

        if oscillating:
            cycle = [
                current,
                _tilt_pose(_height_from_signed(first)),
                _tilt_pose(_height_from_signed(second)),
                current,
            ]
        else:
            target_h = _height_from_signed(second if second != 0 else first)
            cycle = [current, _tilt_pose(target_h)]
    else:  # pan / roll — lateral lean (dir)
        if oscillating:
            cycle = [
                current,
                {**current, "dir": _dir_from_signed_pan(first)},
                {**current, "dir": _dir_from_signed_pan(second)},
                current,
            ]
        else:
            target_d = _dir_from_signed_pan(second if second != 0 else first)
            cycle = [current, {**current, "dir": target_d}]

    for _rep in range(repetition):
        for i in range(len(cycle) - 1):
            steps.append(
                _quad_pose_to_pose_step(
                    cycle[i],
                    cycle[i + 1],
                    duration=seg,
                    speed=speed,
                    static_feet=True,
                )
            )
    last = cycle[-1]
    state["dir"] = str(last["dir"])
    state["front_height"] = str(last["front_height"])
    state["back_height"] = str(last["back_height"])
    if hold > 0:
        steps.append(
            _quad_pose_step(
                direction=str(last["dir"]),
                front_height=str(last["front_height"]),
                back_height=str(last["back_height"]),
                duration=hold,
            )
        )
    return steps


def convert_step_quadruped(
    step: str, *, state: dict[str, Any] | None = None
) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Map the DSL onto the Unitree Go2 locomotion schema.

    Pose attributes:
      - stance: stand (n+n) | sit (h+l) | crouch (l+l) | play_bow (l+h)
        | threelegs_front (stand + lift right-front paw forward)
      - three_legs: none | front  (alias; prefer stance=threelegs_front)
    Move attributes:
      - front_legs / forelegs / back_legs: axis x|z only
          z+: full sit silhouette (front high + back low — haunches down)
          z−: full play_bow silhouette (front low + back high — nose down)
          [-a,+a]: alternate play_bow ↔ sit via ordinary pose steps
      - single leg (left_front/right_front/left_back/right_back): x|y|z
    Path: base line/arc, or single-leg path while three_legs.
    """
    if state is None:
        state = _quad_neutral_state()
    parsed = _parse_call(step)
    if not parsed:
        return None
    name, pos, kw = parsed

    if name == "pose":
        # No neck / no facing-dir attribute — stance + three_legs only.
        if "head" in kw or "dir" in kw:
            return None
        stance = str(kw.get("stance", "stand")).lower().replace("-", "_")
        # Compact GT form: stance=threelegs_front ≡ stand + three_legs=front.
        stance_three_legs = stance in (
            "threelegs_front",
            "three_legs_front",
            "threeleg_front",
        )
        if stance_three_legs:
            stance = "stand"
        # front_height + back_height (n=mid, h=high, l=low)
        stance_heights = {
            "stand": ("mid", "mid"),
            "sit": ("high", "low"),
            "crouch": ("low", "low"),
            "play_bow": ("low", "high"),
        }
        if stance not in stance_heights:
            stance = "stand"
        front_height, back_height = stance_heights[stance]

        tl_raw = kw.get("three_legs", "front" if stance_three_legs else "none")
        if isinstance(tl_raw, bool):
            tl = "front" if tl_raw else "none"
        else:
            tl = str(tl_raw).lower()
        # Only front lift is supported going forward (legacy ``up`` → front).
        if tl in ("up", "true", "1"):
            tl = "front"
        three_legs = False
        hand: str | None = None
        # Internal dir selects which paw to lift in the pose bank (always FR).
        direction = "neutral"
        if tl == "front" or stance_three_legs:
            three_legs = True
            hand = "front"
            direction = "front"
        elif tl not in ("none", "false", "0", ""):
            three_legs = False
            hand = None

        # Optional explicit height overrides still allowed.
        if "front_height" in kw:
            front_height = str(kw["front_height"]).lower()
        if "back_height" in kw:
            back_height = str(kw["back_height"]).lower()

        state["dir"] = direction
        state["front_height"] = front_height
        state["back_height"] = back_height
        state["three_legs"] = three_legs
        state["lifted_leg"] = _quad_lifted_leg_from_dir(direction, three_legs)
        # Expressive stances need a beat to read before follow-on locomotion.
        pose_dur = 1.2 if stance in ("sit", "crouch", "play_bow") or three_legs else 0.6
        return _quad_pose_step(
            direction=direction,
            front_height=front_height,
            back_height=back_height,
            three_legs=three_legs,
            hand=hand,
            duration=pose_dur,
        )

    if name == "hold":
        duration = float(kw.get("t", pos[0] if pos else 1.0))
        return {
            "type": "movement",
            "duration": duration,
            "parameters": {"movement": {"joints": [], "repetition": 1, "speed": 1.0}},
        }

    if name in ("move", "movement"):
        source_joint = str(pos[0]).lower() if pos else str(kw.get("joint", "right_front")).lower()
        # Prefer axis= kw when present (e.g. move(forelegs, axis=z, [-22,22], …)).
        if "axis" in kw:
            axis = str(kw["axis"]).lower()
            raw = (
                pos[1]
                if len(pos) > 1
                else kw.get("degrees", kw.get("deg", [0, 15]))
            )
        elif len(pos) > 1 and isinstance(pos[1], (list, tuple, int, float)):
            axis = str(kw.get("axis", "z")).lower()
            raw = pos[1]
        else:
            axis = str(pos[1]).lower() if len(pos) > 1 else str(kw.get("axis", "z")).lower()
            raw = pos[2] if len(pos) > 2 else kw.get("degrees", kw.get("deg", [0, 15]))
        degrees = _degrees_pair(raw)
        # No neck: head pan/tilt → body attitude proxy (e.g. tilt_head_curious).
        if source_joint == "head":
            if axis in ("pan", "tilt", "pitch", "roll", "yaw"):
                source_joint = "body"
                if axis == "yaw":
                    axis = "pan"
            else:
                return None
        if (
            max(abs(degrees[0]), abs(degrees[1])) <= 1.0
            and source_joint not in ("body", "torso")
        ):
            degrees = [v * 90.0 for v in degrees]
        repetition = max(1, int(kw.get("rep", kw.get("repetition", 1))))
        speed = max(0.2, float(kw.get("spd", kw.get("speed", 1.0))))
        hold = float(kw.get("hold", kw.get("hold_time", 0.0)))
        duration = max(0.5, repetition / speed + hold)

        if source_joint in ("body", "torso") and axis in (
            "pan",
            "tilt",
            "pitch",
            "roll",
        ):
            return _quad_body_attitude_steps(
                state=state,
                axis=axis,
                degrees=degrees,
                repetition=repetition,
                speed=speed,
                hold=hold,
            )

        pair_map = {
            "front_legs": ("FL", "FR"),
            "forelegs": ("FL", "FR"),
            "fore_legs": ("FL", "FR"),
            "back_legs": ("RL", "RR"),
            "rear_legs": ("RL", "RR"),
        }
        single_map = {
            "left_front": "FL",
            "right_front": "FR",
            "left_back": "RL",
            "right_back": "RR",
            "left_rear": "RL",
            "right_rear": "RR",
        }

        def _leg_movement(leg: str) -> dict[str, Any]:
            return {
                "type": "movement",
                "duration": duration,
                "parameters": {
                    "movement": {
                        "leg": leg,
                        "joints": [
                            {
                                "joint": "auto",
                                "axis": axis,
                                "degrees": {axis: degrees},
                            }
                        ],
                        "repetition": repetition,
                        "speed": speed,
                        "hold_time": hold,
                    }
                },
            }

        def _ensure_three_legs_before_paw(leg: str) -> list[dict[str, Any]]:
            """Brace the named paw up before a single-leg move (Motion Code→JSON).

            - ``right_front`` → ``dir=front`` (lift FR; same as threelegs_front)
            - ``left_front`` → ``dir=left`` (lift FL; pose bank ``pose_front_left_lift``)
            Re-inject when switching paws (e.g. FR wave then FL wave) even if
            ``three_legs`` is already true. Renderer unchanged.
            """
            if state is None:
                return []
            # retrieval._lifted_leg_from_pose: only dir=left selects FL.
            direction = "left" if leg == "FL" else "front"
            cur = str(state.get("lifted_leg") or "none")
            if bool(state.get("three_legs")) and cur == leg:
                return []
            hand = "front" if leg in ("FR", "FL") else "back"
            fh = str(state.get("front_height", "mid")).lower()
            bh = str(state.get("back_height", "mid")).lower()
            state["three_legs"] = True
            state["dir"] = direction
            state["lifted_leg"] = leg
            return [
                _quad_pose_step(
                    direction=direction,
                    front_height=fh,
                    back_height=bh,
                    three_legs=True,
                    hand=hand,
                    duration=1.2,
                )
            ]

        if source_joint in pair_map:
            if axis not in ("x", "z"):
                return None
            # Planted pair z: silhouette height (sit/play_bow sign), both feet together.
            if axis == "z":
                which = (
                    "front"
                    if source_joint in ("front_legs", "forelegs", "fore_legs")
                    else "back"
                )
                return _quad_pair_height_z_steps(
                    state=state,
                    which=which,
                    degrees=degrees,
                    repetition=repetition,
                    speed=speed,
                    hold=hold,
                )
            # Pair x: reach via per-leg foot IK (sequential FL/FR or RL/RR).
            return [_leg_movement(leg) for leg in pair_map[source_joint]]

        if source_joint in single_map:
            if axis not in ("x", "y", "z"):
                return None
            leg = single_map[source_joint]
            prelude = _ensure_three_legs_before_paw(leg)
            return [*prelude, _leg_movement(leg)] if prelude else _leg_movement(leg)

        # Legacy hip/thigh/knee → treat as right_front auto.
        if source_joint in ("hip", "thigh", "knee", "auto"):
            if axis not in ("x", "y", "z"):
                return None
            prelude = _ensure_three_legs_before_paw("FR")
            return [*prelude, _leg_movement("FR")] if prelude else _leg_movement("FR")

        return None

    if name == "path":
        converted = convert_step_tiago(step)
        if converted is None:
            return None
        path = converted["parameters"]["path"]
        path.pop("mode", None)
        # Prefer original kw so ±deg lists are not collapsed by google converter.
        parsed_path = _parse_call(step)
        kw = parsed_path[2] if parsed_path else {}
        if "deg" in kw or "degrees" in kw or "sweep" in kw:
            path["degrees"] = _range_or_scalar(
                kw.get("deg", kw.get("degrees", kw.get("sweep")))
            )
        elif "sweep" in path:
            path["degrees"] = path.pop("sweep")
        path.pop("sweep", None)
        if path.get("shape") == "line" and "distance" not in path and "x" in path:
            path["distance"] = path.pop("x")
        if path.get("shape") == "line":
            path.pop("axis", None)

        steps: list[dict[str, Any]] = []
        # Locomotion needs a standing support polygon — rise from crouch/sit/bow first.
        fh = str(state.get("front_height", "mid")).lower()
        bh = str(state.get("back_height", "mid")).lower()
        needs_rise = (
            path.get("shape") in ("line", "arc")
            and (fh != "mid" or bh != "mid" or bool(state.get("three_legs")))
        )
        if needs_rise:
            stand = _quad_pose_dict("neutral", "mid", "mid")
            steps.append(
                _quad_pose_to_pose_step(
                    {
                        "dir": str(state.get("dir", "neutral")),
                        "front_height": fh,
                        "back_height": bh,
                        "three_legs": False,
                    },
                    stand,
                    duration=0.55,
                    speed=max(0.8, float(path.get("speed", 1.0))),
                    static_feet=True,
                )
            )
            state["dir"] = "neutral"
            state["front_height"] = "mid"
            state["back_height"] = "mid"
            state["three_legs"] = False
            state["lifted_leg"] = "none"

        if path.get("shape") == "arc":
            steps.extend(_quad_expand_arc_steps(path, state=state))
        else:
            steps.append(_finalize_quadruped_step(converted))
            hold = float(path.get("hold_time", path.get("hold", 0.0)) or 0.0)
            if hold > 0:
                steps.append(
                    {
                        "type": "movement",
                        "duration": hold,
                        "parameters": {
                            "movement": {"joints": [], "repetition": 1, "speed": 1.0}
                        },
                    }
                )
        return steps if len(steps) != 1 else steps[0]

    return None


def _quad_expand_arc_steps(
    path: dict[str, Any], *, state: dict[str, Any]
) -> list[dict[str, Any]]:
    """Expand arc degrees / ± / rep into forward-walking yaw segments.

    ``deg=±90, rep=2`` → −90, +90, −90, +90 (CW then CCW alternating).
    ``deg=360, rep=2`` → two full forward circles.
    """
    del state  # reserved for future heading carry
    deg_raw = path.get("degrees", 90.0)
    rep = max(1, int(path.get("repetition", path.get("rep", 1)) or 1))
    speed = float(path.get("speed", 1.0))
    radius = float(path.get("radius", 0.0) or 0.0)
    hold = float(path.get("hold_time", path.get("hold", 0.0)) or 0.0)

    segments: list[float] = []
    if isinstance(deg_raw, list):
        nums = [float(x) for x in deg_raw if isinstance(x, (int, float))]
        if len(nums) >= 2 and nums[0] * nums[-1] < 0:
            # ±span: alternate the two signed sweeps each repetition.
            a, b = nums[0], nums[-1]
            for _ in range(rep):
                segments.extend([a, b])
        elif nums:
            for _ in range(rep):
                segments.append(nums[-1])
    else:
        sweep = float(deg_raw)
        for _ in range(rep):
            segments.append(sweep)

    out: list[dict[str, Any]] = []
    for sweep in segments:
        if abs(sweep) < 1e-6:
            continue
        seg_path = {
            "shape": "arc",
            "radius": radius,
            "degrees": sweep,
            "speed": speed,
        }
        out.append(
            _finalize_quadruped_step({"type": "path", "parameters": {"path": seg_path}})
        )
    if hold > 0 and out:
        out.append(
            {
                "type": "movement",
                "duration": hold,
                "parameters": {
                    "movement": {"joints": [], "repetition": 1, "speed": 1.0}
                },
            }
        )
    return out or [
        _finalize_quadruped_step(
            {
                "type": "path",
                "parameters": {
                    "path": {
                        "shape": "arc",
                        "radius": radius,
                        "degrees": 90.0,
                        "speed": speed,
                    }
                },
            }
        )
    ]


ROOMBA_DIR_MAP = {
    "front": "front",
    "forward": "front",
    "facing_person": "front",
    "orient_to_person": "front",
    "face_person": "front",
    "back": "back",
    "left": "left",
    "right": "right",
}


def pose_to_roomba(kwargs: dict[str, Any], positionals: list[Any]) -> dict[str, Any]:
    """Roomba pose: planar facing only (front / back / left / right)."""
    if "dir" in kwargs:
        d = str(kwargs["dir"]).lower()
    elif positionals:
        d = str(positionals[0]).lower()
    else:
        d = "front"
    facing = ROOMBA_DIR_MAP.get(d, d)
    if facing not in ("front", "back", "left", "right"):
        facing = "front"
    return {"dir": facing}


def _roomba_symbolic_facing(deg: Any) -> str | None:
    """Map symbolic arc degrees like [orient_to_person] → pose facing."""
    token = ""
    if isinstance(deg, list) and deg:
        token = str(deg[0]).lower()
    elif isinstance(deg, str):
        token = deg.lower().strip("[]")
    if not token or isinstance(deg, (int, float)):
        return None
    if any(k in token for k in ("orient", "face_person", "face")):
        return "front"
    return ROOMBA_DIR_MAP.get(token) if token in ROOMBA_DIR_MAP else None


def _range_or_scalar(v: Any) -> float | list[float]:
    """Keep ± pairs as [lo, hi]; otherwise a single float."""
    if isinstance(v, list):
        nums = [float(x) for x in v if isinstance(x, (int, float))]
        if len(nums) >= 2:
            return [nums[0], nums[-1]]
        if len(nums) == 1:
            return nums[0]
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def convert_step_roomba(step: str) -> dict[str, Any] | None:
    """Planar Roomba: preserve ± ranges so line/arc can ping-pong with ``rep``."""
    parsed = _parse_call(step)
    if not parsed:
        return None
    name, pos, kw = parsed

    if name == "hold":
        t = float(kw.get("t", pos[0] if pos else 1.0))
        return {
            "type": "path",
            "parameters": {
                "path": {
                    "mode": "base",
                    "shape": "line",
                    "x": 0.0,
                    "speed": 0.5,
                    "hold_time": t,
                }
            },
        }

    if name in ("move", "movement"):
        # Tiny in-place yaw wiggle as a readable proxy for joint moves
        axis = str(pos[1]).lower() if len(pos) > 1 else "pan"
        deg = _degrees_pair(pos[2] if len(pos) > 2 else kw.get("degrees", [0, 15]))
        rep = max(1, int(kw.get("rep", kw.get("repetition", 1))))
        speed = max(0.2, float(kw.get("spd", kw.get("speed", 1.0))))
        return {
            "type": "path",
            "parameters": {
                "path": {
                    "mode": "base",
                    "shape": "arc",
                    "radius": 0.0,
                    "degrees": deg if axis in ("pan", "y", "z", "yaw") else [0.0, abs(deg[-1]) * 0.3],
                    "repetition": rep,
                    "speed": speed,
                    **({"hold_time": float(kw["hold"])} if "hold" in kw else {}),
                }
            },
        }

    if name == "pose":
        pose = pose_to_roomba(kw, pos)
        dur = float(kw.get("t", kw.get("duration", 0.6)))
        return {
            "type": "pose",
            "duration": dur,
            "parameters": {"pose": pose},
        }

    if name != "path":
        return None

    mode = str(pos[0]).lower() if pos else "base"
    shape = str(pos[1]).lower() if len(pos) > 1 else str(kw.get("shape", "line")).lower()
    # Symbolic zero-radius arc → categorical facing pose (e.g. orient_to_person → front).
    if shape == "arc":
        try:
            radius = float(kw.get("r", kw.get("radius", 0.0)))
        except (TypeError, ValueError):
            radius = 0.0
        if abs(radius) <= 1e-5:
            facing = _roomba_symbolic_facing(kw.get("deg", kw.get("degrees", kw.get("sweep"))))
            if facing:
                return {
                    "type": "pose",
                    "duration": 0.5,
                    "parameters": {"pose": {"dir": facing}},
                }

    path: dict[str, Any] = {"mode": mode or "base", "shape": shape}
    if "spd" in kw or "speed" in kw:
        path["speed"] = float(kw.get("spd", kw.get("speed")))
    if "hold" in kw or "hold_time" in kw:
        path["hold_time"] = float(kw.get("hold", kw.get("hold_time")))
    if "rep" in kw or "repetition" in kw:
        path["repetition"] = int(kw.get("rep", kw.get("repetition")))

    if shape == "line":
        if "x" in kw:
            path["x"] = _range_or_scalar(kw["x"])
        if "y" in kw:
            path["y"] = _range_or_scalar(kw["y"])
        if "dist" in kw or "distance" in kw:
            path["distance"] = _range_or_scalar(kw.get("dist", kw.get("distance")))
        # Accept len/d/length as signed distance along heading (+x forward, -x back)
        elif any(k in kw for k in ("len", "d", "length")):
            path["x"] = _range_or_scalar(kw.get("len", kw.get("d", kw.get("length"))))
        if "x" not in path and "y" not in path and "distance" not in path:
            path["x"] = 0.15
    else:
        if "r" in kw or "radius" in kw:
            try:
                path["radius"] = float(kw.get("r", kw.get("radius")))
            except (TypeError, ValueError):
                path["radius"] = 0.0
        else:
            path["radius"] = 0.0
        deg = kw.get("deg", kw.get("degrees", kw.get("sweep", 90.0)))
        path["degrees"] = _range_or_scalar(deg)

    return {"type": "path", "parameters": {"path": path}}


def _step_order(step: str) -> str:
    """Parse order= from a Motion Code call. Default sequential (= in order).

    ``order=together`` on step A means A runs concurrently with the *next* step.
    """
    m = re.search(r"\border\s*=\s*([A-Za-z_]+)", step)
    if not m:
        return "sequential"
    token = m.group(1).lower()
    if token in ("together", "parallel", "with_next", "sync"):
        return "together"
    if token in ("sequential", "seq", "serial", "in_order"):
        return "sequential"
    return "sequential"


def _attach_order(step_json: dict[str, Any] | list[dict[str, Any]], order: str) -> None:
    if isinstance(step_json, list):
        if step_json:
            step_json[0]["order"] = order
            for rest in step_json[1:]:
                rest.setdefault("order", "sequential")
    else:
        step_json["order"] = order


def convert_code(code: str, *, platform: str) -> dict[str, Any]:
    """Convert a Motion Code string into a renderable config fragment."""
    notes: list[str] = []
    if not code or not str(code).strip():
        return {"movements": [], "parse_ok": False, "notes": ["empty"], "source_code": ""}

    steps_raw = _split_steps_with_ops(str(code))
    movements: list[dict[str, Any]] = []
    quad_state = _quad_neutral_state() if platform == "quadruped" else None
    for op, step in steps_raw:
        order = _step_order(step)
        # ``A & B`` ⇒ A runs concurrently with B (same as order=together on A).
        if op == "&" and order == "sequential":
            # Mark the previous authored step as together-with-this.
            if movements:
                prev = movements[-1]
                prev["order"] = "together"
        if platform == "single_arm":
            step_json = convert_step_manip(step)
        elif platform == "quadruped":
            step_json = convert_step_quadruped(step, state=quad_state)
        elif platform == "roomba":
            step_json = convert_step_roomba(step)
        elif platform == "stretch":
            step_json = convert_step_stretch(step)
        else:
            step_json = convert_step_tiago(step)
        if step_json is None:
            notes.append(f"unparsed: {step[:80]}")
            continue
        # Go2 has no neck; head pan/tilt is remapped to body attitude.
        if platform == "quadruped" and re.search(r"\broll_over\b", step):
            notes.append(f"rejected_or_approx: {step[:60]}")
        elif platform == "quadruped" and re.search(r"\bhead\b", step):
            notes.append(f"approx: head→body {step[:50]}")
        _attach_order(step_json, order)
        if isinstance(step_json, list):
            movements.extend(step_json)
        else:
            movements.append(step_json)

    if platform == "quadruped":
        movements = [_finalize_quadruped_step(s) for s in movements]

    # ensure at least a holdable pose
    if not movements:
        if platform == "single_arm":
            movements = [
                {
                    "type": "pose",
                    "parameters": {
                        "pose": {"dir": "front", "gripper_orientation": "vertical", "z": 50},
                        "speed": 1.0,
                        "hold_time": 1.0,
                    },
                }
            ]
        elif platform == "quadruped":
            movements = [
                {
                    "type": "pose",
                    "duration": 1.0,
                    "parameters": {
                        "pose": {
                            "dir": "neutral",
                            "front_height": "mid",
                            "back_height": "mid",
                            "three_legs": False,
                        }
                    },
                }
            ]
        elif platform == "roomba":
            movements = [
                {
                    "type": "pose",
                    "duration": 1.0,
                    "parameters": {"pose": {"dir": "front"}},
                }
            ]
        else:
            movements = [
                {
                    "type": "pose",
                    "duration": 1.0,
                    "parameters": {
                        "pose": {
                            "arm_position": "still",
                            "gripper_orientation": "vertical",
                            "head": "center",
                            "torso_height": "mid",
                        }
                    },
                }
            ]
        notes.append("fallback empty pose")

    # manipulator renderer requires a leading pose step
    if platform == "single_arm" and movements and movements[0].get("type") != "pose":
        movements = [
            {
                "type": "pose",
                "parameters": {
                    "pose": {"dir": "front", "gripper_orientation": "horizontal", "x": 55, "y": 50, "z": 55},
                    "speed": 1.2,
                    "hold_time": 0.1,
                },
            },
            *movements,
        ]
        notes.append("prepended default pose")

    # soften sparse manipulator poses (pose bank often misses in/out without xyz)
    # Only fill axes that are already absent — never overwrite explicit values.
    # Prefer leaving axes omitted when the motion code did not set them; the
    # bank still ranks by dir/orient. We only soften when *no* xyz was given
    # at all (legacy bank coverage), and then only for single_arm.
    if platform == "single_arm":
        for step in movements:
            if step.get("type") != "pose":
                continue
            pose = (step.get("parameters") or {}).get("pose") or {}
            if any(isinstance(pose.get(a), (int, float)) for a in ("x", "y", "z")):
                continue  # at least one axis specified → do not invent the others
            d = str(pose.get("dir", "front")).lower()
            if d in ("left", "back"):
                pose.setdefault("x", 40 if d == "back" else 55)
                pose.setdefault("y", 65 if d == "left" else 50)
                pose.setdefault("z", 60)
            elif d == "right":
                pose.setdefault("x", 55)
                pose.setdefault("y", 25)
                pose.setdefault("z", 55)
            else:
                pose.setdefault("x", 55)
                pose.setdefault("y", 50)
                pose.setdefault("z", 55)
            step["parameters"]["pose"] = pose

    return {
        "movements": movements,
        "parse_ok": len(notes) == 0 or not any(n.startswith("unparsed") for n in notes),
        "notes": notes,
        "source_code": code,
    }


def load_motion_code_rows(xlsx: Path, sheet: str = "Motion Code (Top 5)") -> list[dict[str, Any]]:
    from openpyxl import load_workbook

    wb = load_workbook(xlsx, data_only=True)
    if sheet not in wb.sheetnames:
        raise ValueError(f"Sheet '{sheet}' not found in {xlsx}. Available: {wb.sheetnames}")
    ws = wb[sheet]
    rows: list[dict[str, Any]] = []
    for r in range(3, ws.max_row + 1):
        idx = ws.cell(r, 1).value
        cue = ws.cell(r, 2).value
        cat = ws.cell(r, 3).value
        if cue is None or str(cue).startswith("pose("):
            continue
        item: dict[str, Any] = {
            "idx": int(idx) if idx is not None else r - 2,
            "cue": str(cue).strip(),
            "category": str(cat or "").strip(),
            "codes": {},
        }
        for col, plat in PLATFORM_COLS.items():
            item["codes"][plat] = ws.cell(r, col).value or ""
        rows.append(item)
    return rows


def build_all_configs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    now = datetime.now().isoformat(timespec="seconds")
    for row in rows:
        entry: dict[str, Any] = {
            "idx": row["idx"],
            "cue": row["cue"],
            "category": row["category"],
            "platforms": {},
        }
        for plat, code in row["codes"].items():
            converted = convert_code(str(code), platform=plat)
            domain = {
                "single_arm": "manipulator",
                "tiago": "tiago",
                "stretch": "stretch",
                "humanoid": "unitree_g1",
                "quadruped": "unitree_go2",
                "roomba": "roomba",
            }[plat]
            cfg = {
                "idx": row["idx"],
                "cue": row["cue"],
                "description": f"Motion-code embodiment ({plat}) for {row['cue']}",
                "movements": converted["movements"],
                "state": "motion_code_embodiment",
                "model": "motion_code",
                "time": now,
                "experiment": "embodiment_motion_code",
                "generation_valid": True,
                "platform": plat,
                "domain": domain,
                "parse_ok": converted["parse_ok"],
                "notes": converted["notes"],
                "source_code": converted["source_code"],
            }
            entry["platforms"][plat] = cfg
        out.append(entry)
    return out


def write_json_files(configs: list[dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    master = out_dir / "embodiment_motion_code_all.json"
    master.write_text(json.dumps(configs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # per-platform flat lists (render-ready)
    by_plat: dict[str, list[dict[str, Any]]] = {p: [] for p in PLATFORM_COLS.values()}
    for entry in configs:
        for plat, cfg in entry["platforms"].items():
            slim = {
                k: cfg[k]
                for k in (
                    "idx",
                    "cue",
                    "description",
                    "movements",
                    "state",
                    "model",
                    "time",
                    "experiment",
                    "generation_valid",
                    "platform",
                    "source_code",
                    "parse_ok",
                    "notes",
                )
                if k in cfg
            }
            by_plat[plat].append(slim)
            cue_dir = out_dir / plat
            cue_dir.mkdir(parents=True, exist_ok=True)
            (cue_dir / f"{cfg['cue']}.json").write_text(
                json.dumps([slim], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    for plat, rows in by_plat.items():
        (out_dir / f"result_{plat}.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return master


def add_excel_sheet(
    xlsx: Path,
    configs: list[dict[str, Any]],
    *,
    sheet_name: str = SHEET_NAME,
    note: str | None = None,
) -> None:
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font

    wb = load_workbook(xlsx)
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(sheet_name)

    headers = ["#", "Gesture", "Category"] + [PLATFORM_HEADERS[p] for p in PLATFORM_COLS.values()]
    for c, h in enumerate(headers, start=1):
        cell = ws.cell(1, c, h)
        cell.font = Font(bold=True)
    ws.cell(2, 1, "Note")
    ws.cell(
        2,
        2,
        note
        or (
            "Each cell contains platform-specific renderable JSON movements[]. "
            "Full files: data/results/motion_configs/embodiment_motion_code/"
        ),
    )

    for r_i, entry in enumerate(configs, start=3):
        ws.cell(r_i, 1, entry["idx"])
        ws.cell(r_i, 2, entry["cue"])
        ws.cell(r_i, 3, entry["category"])
        for c_i, plat in enumerate(PLATFORM_COLS.values(), start=4):
            cfg = entry["platforms"][plat]
            payload = {
                "cue": cfg["cue"],
                "idx": cfg["idx"],
                "domain": cfg["domain"],
                "platform": plat,
                "movements": cfg["movements"],
                "notes": cfg.get("notes") or [],
            }
            text = json.dumps(payload, indent=2, ensure_ascii=False)
            cell = ws.cell(r_i, c_i, text)
            cell.alignment = Alignment(wrap_text=True, vertical="top")

    ws.column_dimensions["A"].width = 5
    ws.column_dimensions["B"].width = 22
    ws.column_dimensions["C"].width = 16
    for col in ("D", "E", "F", "G", "H"):
        ws.column_dimensions[col].width = 42
    for r in range(3, 3 + len(configs)):
        ws.row_dimensions[r].height = 120

    wb.save(xlsx)


def render_tiago(
    rows: list[dict[str, Any]],
    *,
    out_root: Path,
    force: bool = False,
    cam_pos: tuple[float, float, float] | None = None,
    cam_quat: tuple[float, float, float, float] | None = None,
) -> dict[str, Path]:
    from correction_lib import _ensure_mp4, _import_tiago_render, _save_gif_frames

    # Default: bank nearest then EE residual snap (trim). Callers may set
    # MOBILE_POSE_XYZ_MODE=nearest for bank hit only (no EE translate).
    os.environ.setdefault("MOBILE_POSE_XYZ_MODE", "trim")
    gif_dir = out_root / "tiago" / "gif"
    gif_dir.mkdir(parents=True, exist_ok=True)
    _make_env, render_config, mod = _import_tiago_render()

    def _frames_not_corrupt(frames: list) -> bool:
        """Reject EGL snow / black / luminance-swing captures only.

        Tiny tip/head strokes in full-body framing often fail motion gates but are
        still healthy renders — use this for best-effort fallback acceptance.
        """
        if not frames or len(frames) < 2:
            return False
        import numpy as np
        from PIL import Image

        means: list[float] = []
        arr0 = None
        for fr in frames:
            arr = np.asarray(
                fr.convert("RGB") if hasattr(fr, "convert") else Image.fromarray(fr).convert("RGB"),
                dtype=np.float32,
            )
            if arr0 is None:
                arr0 = arr
            means.append(float(arr.mean()))
        m = np.asarray(means, dtype=np.float64)
        if float(m.min()) < 80.0 or float(m.std()) > 25.0:
            return False
        assert arr0 is not None
        lap = float(
            np.abs(
                arr0[1:-1, 1:-1]
                - 0.25
                * (
                    arr0[:-2, 1:-1]
                    + arr0[2:, 1:-1]
                    + arr0[1:-1, :-2]
                    + arr0[1:-1, 2:]
                )
            ).mean()
        )
        return lap <= 12.0

    def _frames_look_stable(frames: list) -> bool:
        """Reject intermittent black/corrupt offscreen captures (GIF flicker).

        Mean-luminance checks alone miss colorful static/ghosting (mean still ~120).
        Also require low frame-to-frame change and low mid-clip temporal noise.
        Fully static holds (avg_f2f≈0) are valid for pose+hold cues.

        Progress-bar overlays must be excluded from motion checks: a frozen robot
        with only the scrubber moving looks "alive" (uniq>1, tiny f2f) and was
        incorrectly accepted for movement/path cues (e.g. tiago beckon wrist).
        """
        if not _frames_not_corrupt(frames):
            return False
        import numpy as np
        from PIL import Image

        arrs: list[Any] = []
        for fr in frames:
            arr = np.asarray(
                fr.convert("RGB") if hasattr(fr, "convert") else Image.fromarray(fr).convert("RGB"),
                dtype=np.float32,
            )
            arrs.append(arr)
        # Drop typed/simple progress bar strip (top ~36px @ 512).
        h0 = int(arrs[0].shape[0])
        bar_h = min(36, h0 // 8)
        bodies = [a[bar_h:] for a in arrs]
        body_f2f = [
            float(np.abs(bodies[i] - bodies[i + 1]).mean()) for i in range(len(bodies) - 1)
        ]
        body_avg = float(np.mean(body_f2f)) if body_f2f else 0.0
        body_max = float(np.max(body_f2f)) if body_f2f else 0.0
        body_uniq = len({b.tobytes() for b in bodies})
        # Byte-identical body: OK for pose+hold; reject for movement/path (stale EGL
        # buffer). Progress-bar-only animation used to fake uniq>1 — use body crop.
        if body_uniq <= 1 and body_max < 1e-6:
            if getattr(_frames_look_stable, "_expect_motion", False):
                return False
        # Require real robot motion (not scrubber-only / near-frozen stale captures).
        # Threshold stays modest: full-body framing makes even 10cm tip strokes small
        # in mean f2f; scrubber-only freezes are body_uniq≈1 / body_avg≈0.
        if getattr(_frames_look_stable, "_expect_motion", False):
            if body_uniq < 3 or body_avg < 0.2:
                return False
        return True

    out: dict[str, Path] = {}
    for row in rows:
        cue = str(row["cue"])
        idx = int(row["idx"])
        gif_path = gif_dir / f"emb_g{idx:02d}_{cue}.gif"
        if gif_path.is_file() and not force:
            out[cue] = gif_path
            continue
        frames = None
        last_exc: Exception | None = None
        last_nonzero: list | None = None
        expect_motion = any(
            str(s.get("type") or "") in ("movement", "path")
            for s in (row.get("movements") or [])
        )
        _frames_look_stable._expect_motion = expect_motion  # type: ignore[attr-defined]
        for attempt in range(1, 6):
            env = _make_env()
            if cam_pos is not None:
                env._tiago_cam_pos = tuple(float(v) for v in cam_pos)
            if cam_quat is not None:
                env._tiago_cam_quat = tuple(float(v) for v in cam_quat)
            try:
                # typed progress overlay is fine when capture is healthy; flaky EGL
                # captures already flicker without it, so keep bar off for retries≥2.
                frames = render_config(
                    row,
                    env=env,
                    overlay_progress_bar=(attempt == 1),
                    progress_bar_style="typed",
                    ignore_xyz=False,
                )
            except Exception as exc:
                last_exc = exc
                frames = None
            finally:
                try:
                    env.close()
                except Exception:
                    pass
            if frames and _frames_look_stable(frames):
                break
            # Keep a non-black capture as best-effort fallback. Fast authored
            # motions (e.g. flap rep=4) can fail the f2f stability gate even
            # when the offscreen buffer is healthy.
            if frames:
                try:
                    import numpy as np
                    from PIL import Image

                    means = [
                        float(
                            np.asarray(
                                fr.convert("RGB")
                                if hasattr(fr, "convert")
                                else Image.fromarray(fr).convert("RGB"),
                                dtype=np.float32,
                            ).mean()
                        )
                        for fr in frames
                    ]
                    if min(means) >= 80.0:
                        last_nonzero = frames
                except Exception:
                    last_nonzero = frames
            print(
                f"[render/google] unstable frames for {cue} "
                f"(attempt {attempt}/5); retrying…",
                flush=True,
            )
            frames = None
        if frames is None and last_nonzero is not None:
            # Soft fallback: accept healthy tiny-motion clips; reject EGL snow only.
            # (Full _frames_look_stable wrongly discarded tip/head strokes with
            # body_avg≪0.2 that are still usable — e.g. regen stop_palm_out.)
            if _frames_not_corrupt(last_nonzero):
                frames = last_nonzero
                print(
                    f"[render/google] {cue}: accepting best-effort frames "
                    f"(motion gate failed; not corrupt)",
                    flush=True,
                )
            else:
                print(
                    f"[render/google] {cue}: discarding corrupt fallback "
                    f"(EGL snow / unstable)",
                    flush=True,
                )
        if frames:
            # Defensive copy — shared buffers / lazy arrays can collapse on encode.
            frames = [
                fr.copy() if hasattr(fr, "copy") else fr for fr in frames
            ]
            _save_gif_frames(frames, gif_path, duration_ms=80)
            _ensure_mp4(gif_path, gif_path.with_suffix(".mp4"))
            out[cue] = gif_path
            print(f"[render/google] {cue} -> {gif_path.name} ({len(frames)}f)", flush=True)
        else:
            msg = last_exc if last_exc is not None else "unstable/black frames after retries"
            print(f"[render/google] FAIL {cue}: {msg}", flush=True)
    return out


def render_manipulator(
    rows: list[dict[str, Any]],
    *,
    out_root: Path,
    force: bool = False,
    start_from_neutral: bool = False,
) -> dict[str, Path]:
    robotarm = _REPO / "adhoc/generation/robotarm"
    # tiago/legacy shadows robotarm/legacy — force robotarm first
    sys.path = [p for p in sys.path if Path(p).resolve() != (_GOOGLE / "legacy").resolve()]
    if str(robotarm) in sys.path:
        sys.path.remove(str(robotarm))
    sys.path.insert(0, str(robotarm))
    # drop cached wrong legacy if present
    for mod in list(sys.modules):
        if mod == "legacy" or mod.startswith("legacy."):
            del sys.modules[mod]
    from legacy.motion_generation_core import generate  # noqa: WPS433

    os.environ.setdefault("MANIPULATOR_POSE_XYZ_MODE", "trim")
    gif_dir = out_root / "manipulator" / "gif" / "IIWA"
    gif_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg = out_root / "manipulator" / "configs" / "_tmp.json"
    jpath = str(_REPO / "data/seed/_remainder/closest_poses_results.jsonl")
    out: dict[str, Path] = {}
    for row in rows:
        cue = str(row["cue"])
        idx = int(row["idx"])
        gif_path = gif_dir / f"emb_g{idx:02d}_{cue}.gif"
        if gif_path.is_file() and not force:
            out[cue] = gif_path
            continue
        tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
        tmp_cfg.write_text(json.dumps([row], indent=2, ensure_ascii=False), encoding="utf-8")
        scratch = out_root / "manipulator" / "_scratch" / cue
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            generate(
                robot="IIWA",
                cue=cue,
                cue_idx=idx,
                pose_index=None,
                jsonl_path=jpath,
                config_path=str(tmp_cfg),
                output_dir=str(scratch),
                hz=12,
                top_k=1,
                ignore_xyz=False,
                start_from_neutral=start_from_neutral,
            )
            robot_dir = scratch / "IIWA"
            gifs = sorted(robot_dir.glob("*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)
            if gifs:
                shutil.copy2(gifs[0], gif_path)
                out[cue] = gif_path
                print(f"[render/manip] {cue} -> {gif_path.name}", flush=True)
        except Exception as exc:
            print(f"[render/manip] FAIL {cue}: {exc}", flush=True)
    return out


def render_quadruped(
    rows: list[dict[str, Any]],
    *,
    config_path: Path,
    out_root: Path,
    force: bool = False,
) -> dict[str, Path]:
    """Use the existing Unitree Go2 MJLab config renderer."""
    from adhoc.generation.quadruped.render import run as render_go2

    # The Go2 runner launches one subprocess per cue from its own directory.
    # Absolute paths are therefore required for both generated per-cue configs
    # and output media.
    config_path = config_path.resolve()
    gif_dir = (out_root / "quadruped" / "gif").resolve()
    python_bin = Path("/Users/sb/micromamba/envs/robosuite/bin/python")
    if not python_bin.is_file():
        raise FileNotFoundError(f"MJLab Python missing: {python_bin}")
    render_go2(
        config_json=str(config_path),
        output_dir=str(gif_dir),
        python_bin=str(python_bin),
        auto_generate_if_missing=False,
        do_html=False,
        camera="front",
        skip_existing=not force,
    )
    results: dict[str, Path] = {}
    for row in rows:
        idx, cue = int(row["idx"]), str(row["cue"])
        source = gif_dir / f"lc19_g{idx:02d}_{cue}_mjlab_earth.gif"
        target = gif_dir / f"emb_g{idx:02d}_{cue}.gif"
        if source.is_file():
            shutil.copy2(source, target)
            source_mp4 = source.with_suffix(".mp4")
            if source_mp4.is_file():
                shutil.copy2(source_mp4, target.with_suffix(".mp4"))
            results[cue] = target
    return results


_HTML_PLAT_GIF = {
    "humanoid": ("Humanoid (Unitree G1)", ("humanoid", "gif")),
    "tiago": ("Tiago", ("tiago", "gif")),
    "single_arm": ("Single-arm (IIWA)", ("manipulator", "gif", "IIWA")),
    "stretch": ("Stretch (Hello Robot)", ("stretch", "gif")),
    "quadruped": ("Quadruped (Unitree Go2)", ("quadruped", "gif")),
    "roomba": ("Roomba", ("roomba", "gif")),
}


def build_html(
    configs: list[dict[str, Any]],
    render_root: Path,
    out_html: Path,
    *,
    title: str = "Embodiment — Motion Code renders",
    meta: str | None = None,
    platforms: tuple[str, ...] | list[str] | None = None,
) -> Path:
    import html as html_mod

    plats = list(platforms or PLATFORMS_ORDER)
    n_plat = max(1, len(plats))
    sections = []
    for entry in configs:
        cue = entry["cue"]
        idx = entry["idx"]
        cat = html_mod.escape(entry.get("category", ""))
        cols = []
        for plat in plats:
            label, rel_parts = _HTML_PLAT_GIF[plat]
            gif = render_root.joinpath(*rel_parts) / f"emb_g{idx:02d}_{cue}.gif"
            if gif.is_file():
                rel = os.path.relpath(gif, out_html.parent)
                # Bust browser cache when the GIF file is replaced in-place.
                bust = int(gif.stat().st_mtime)
                media = (
                    f'<img src="{html_mod.escape(rel)}?v={bust}" '
                    f'loading="lazy">'
                )
            else:
                media = '<p class="miss">no gif</p>'
            cfg = (entry.get("platforms") or {}).get(plat) or {}
            source_code = str(cfg.get("source_code") or "").strip() or "(empty)"
            movements_txt = json.dumps(cfg.get("movements") or [], indent=2, ensure_ascii=False)
            cols.append(
                f'<div class="col">'
                f"<h3>{label}</h3>{media}"
                f'<div class="mc"><div class="mc-label">Config</div>'
                f'<pre class="mc-pre">{html_mod.escape(movements_txt)}</pre></div>'
                f'<div class="mc"><div class="mc-label">Motion Code</div>'
                f'<pre class="mc-pre">{html_mod.escape(source_code)}</pre></div>'
                f"</div>"
            )
        sections.append(
            f'<section class="cue"><h2>{idx:02d}. {html_mod.escape(cue)} '
            f'<span class="tag">{cat}</span></h2><div class="grid">{"".join(cols)}</div></section>'
        )
    meta_txt = meta or (
        f"{len(configs)} gestures · {n_plat} platforms side-by-side · Config + Motion Code per platform"
    )
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html_mod.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e8eaed}}
h1{{padding:16px 24px;margin:0;border-bottom:1px solid #333}}
.meta{{padding:10px 24px;color:#9aa0a6;font-size:0.9rem}}
.cue{{margin:12px 16px;padding:12px;border:1px solid #333;border-radius:10px;background:#171a21}}
.grid{{display:grid;grid-template-columns:repeat({n_plat},minmax(0,1fr));gap:10px}}
.col{{background:#0f1115;border-radius:8px;padding:8px;min-width:0}}
.col h3{{margin:0 0 6px;font-size:0.78rem;font-weight:600;color:#c4c7ce;line-height:1.25}}
img{{width:100%;height:auto;aspect-ratio:1;object-fit:cover;border-radius:6px;background:#000;display:block}}
.miss{{color:#f28b82}}
.tag{{font-size:0.75rem;color:#8ab4f8}}
.mc{{margin-top:8px}}
.mc-label{{font-size:0.68rem;color:#9aa0a6;margin-bottom:3px;text-transform:uppercase;letter-spacing:0.04em}}
.mc-pre{{font-size:10px;line-height:1.35;overflow:auto;max-height:140px;background:#0a0c10;padding:6px;border-radius:4px;white-space:pre-wrap;word-break:break-word;margin:0;color:#d7dbe2}}
@media (max-width:1300px){{.grid{{grid-template-columns:repeat({n_plat},minmax(140px,1fr));overflow-x:auto}}}}
</style></head><body>
<h1>{html_mod.escape(title)}</h1>
<p class="meta">{html_mod.escape(meta_txt)}</p>
{"".join(sections)}
</body></html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding="utf-8")
    return out_html


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON_DIR)
    ap.add_argument("--out-render", type=Path, default=OUT_RENDER)
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--force-render", action="store_true")
    ap.add_argument(
        "--start_from_neutral",
        action="store_true",
        help="Start IIWA clips from neutral before approaching the first authored pose.",
    )
    ap.add_argument("--max-cues", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--platforms",
        default="humanoid,tiago,single_arm,stretch,quadruped,roomba",
        help="Comma list to render",
    )
    ap.add_argument(
        "--sheet",
        default="Motion Code (Top 5)",
        help="Sheet name to read motion codes from (e.g. 'CF Code' for contextual framing)",
    )
    ap.add_argument(
        "--renderable-sheet",
        default=None,
        help=(
            "Excel sheet name to write renderable JSON into "
            f"(default: '{SHEET_NAME}'). Use a distinct name to avoid overwriting Top-5."
        ),
    )
    ap.add_argument(
        "--html-out",
        type=Path,
        default=None,
        help="Primary HTML gallery path (default: data/results/html/embodiment_motion_code.html)",
    )
    ap.add_argument(
        "--extra-html-out",
        type=Path,
        action="append",
        default=[],
        help="Additional HTML gallery path(s) to write with the same configs/renders",
    )
    ap.add_argument(
        "--html-title",
        default=None,
        help="HTML <h1>/title override",
    )
    args = ap.parse_args()

    rows = load_motion_code_rows(args.xlsx, sheet=args.sheet)
    if args.max_cues > 0:
        rows = rows[: args.max_cues]
    print(f"Loaded {len(rows)} gestures from {args.xlsx} sheet={args.sheet!r}", flush=True)

    configs = build_all_configs(rows)
    master = write_json_files(configs, args.out_json)
    print(f"JSON: {master}", flush=True)

    renderable_sheet = args.renderable_sheet or SHEET_NAME
    sheet_note = (
        "Each cell contains platform-specific renderable JSON movements[] from "
        f"sheet {args.sheet!r}. Full files: {args.out_json}/"
    )
    add_excel_sheet(
        args.xlsx,
        configs,
        sheet_name=renderable_sheet,
        note=sheet_note,
    )
    print(f"Excel sheet {renderable_sheet!r} written: {args.xlsx}", flush=True)

    if not args.skip_render:
        want = {p.strip() for p in args.platforms.split(",") if p.strip()}
        if "tiago" in want:
            g_rows = [e["platforms"]["tiago"] for e in configs]
            render_tiago(g_rows, out_root=args.out_render, force=args.force_render)
        if "single_arm" in want:
            m_rows = [e["platforms"]["single_arm"] for e in configs]
            render_manipulator(
                m_rows,
                out_root=args.out_render,
                force=args.force_render,
                start_from_neutral=args.start_from_neutral,
            )
        from embodiment_renderers import (
            render_stretch,
            render_roomba,
            render_unitree_g1,
        )

        if "humanoid" in want:
            h_rows = [e["platforms"]["humanoid"] for e in configs]
            render_unitree_g1(
                h_rows,
                args.out_render / "humanoid" / "gif",
                force=args.force_render,
            )
        if "stretch" in want:
            gr_rows = [e["platforms"]["stretch"] for e in configs]
            render_stretch(
                gr_rows,
                args.out_render / "stretch" / "gif",
                force=args.force_render,
            )
        if "quadruped" in want:
            q_rows = [e["platforms"]["quadruped"] for e in configs]
            render_quadruped(
                q_rows,
                config_path=args.out_json / "result_quadruped.json",
                out_root=args.out_render,
                force=args.force_render,
            )
        if "roomba" in want:
            r_rows = [e["platforms"]["roomba"] for e in configs]
            render_roomba(
                r_rows,
                args.out_render / "roomba" / "gif",
                force=args.force_render,
            )

        html_paths = [
            args.html_out
            or (_REPO / "data/results/html/embodiment_motion_code.html")
        ] + list(args.extra_html_out or [])
        title = args.html_title or "Embodiment — Motion Code renders"
        meta = (
            f"{len(configs)} gestures · 6 platforms · source sheet {args.sheet!r} · "
            f"render {args.out_render}"
        )
        for html_path in html_paths:
            out = build_html(
                configs,
                args.out_render,
                html_path,
                title=title,
                meta=meta,
            )
            print(f"HTML: {out}", flush=True)


if __name__ == "__main__":
    main()
