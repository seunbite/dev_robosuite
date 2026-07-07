"""Parse Google Robot mobile pose GT abbreviations and score generated configs."""
from __future__ import annotations

import re
from typing import Any

_ARM = {"f": "front", "b": "back", "i": "in", "o": "out", "u": "up", "d": "down"}
_ORIENT = {"h": "horizontal", "v": "vertical"}
_ELBOW_SUFFIX = {"b": True, "n": False}

_POSE_ANY_RE = re.compile(r"^c\d+$", re.IGNORECASE)
_CODE_RE = re.compile(r"^[fbioud][hv][bn]?$", re.IGNORECASE)

_TOKEN_ALIASES = {
    "ivh": "iv",
    "fvt": "fvn",
    "fht": "fhn",
}


def normalize_pose_token(token: str) -> str:
    t = token.strip().lower()
    return _TOKEN_ALIASES.get(t, t)


def parse_pose_code(token: str) -> dict[str, Any] | None:
    t = normalize_pose_token(token)
    if not t or t in {"n", "none"} or _POSE_ANY_RE.match(t):
        return None
    if not _CODE_RE.match(t):
        return None
    arm = _ARM[t[0]]
    orient = _ORIENT[t[1]]
    out: dict[str, Any] = {
        "arm_position": arm,
        "gripper_orientation": orient,
    }
    if len(t) == 3:
        out["elbow_bended"] = _ELBOW_SUFFIX[t[2]]
    return out


def parse_pose_gt_line(line: str) -> tuple[bool, list[dict[str, Any]], str]:
    """Return (pose_any, pose_options, pose_gt_raw)."""
    raw = " ".join(line.strip().split())
    if not raw or raw.startswith("#"):
        return True, [], ""
    lower = raw.lower()
    if lower in {"n", "none"}:
        return True, [], lower
    tokens = [normalize_pose_token(t) for t in lower.split()]
    if all(_POSE_ANY_RE.match(t) or t in {"n", "none"} for t in tokens):
        return True, [], raw
    options: list[dict[str, Any]] = []
    for tok in tokens:
        if tok in {"n", "none"} or _POSE_ANY_RE.match(tok):
            continue
        parsed = parse_pose_code(tok)
        if parsed is not None:
            options.append(parsed)
    if not options and any(t in {"n", "none"} for t in tokens):
        return True, [], raw
    return False, options, raw


def pose_gt_display(pose_any: bool, options: list[dict[str, Any]], raw: str) -> str:
    if pose_any:
        return raw or "n"
    if not options:
        return raw
    parts: list[str] = []
    for opt in options:
        arm = str(opt["arm_position"])[0]
        orient = str(opt["gripper_orientation"])[0]
        if "elbow_bended" in opt:
            parts.append(f"{arm}{orient}{'b' if opt['elbow_bended'] else 'n'}")
        else:
            parts.append(f"{arm}{orient}")
    return " / ".join(parts)


def _pose_matches_option(pose: dict[str, Any], opt: dict[str, Any]) -> bool:
    arm = str(pose.get("arm_position", "")).strip().lower()
    grip = str(pose.get("gripper_orientation", "")).strip().lower()
    if arm != str(opt.get("arm_position", "")).strip().lower():
        return False
    if grip != str(opt.get("gripper_orientation", "")).strip().lower():
        return False
    eb_target = opt.get("elbow_bended")
    if eb_target is None:
        return True
    eb = pose.get("elbow_bended")
    if eb is None:
        return True
    return bool(eb) == bool(eb_target)


def pose_generation_correct_any_mobile(row: dict[str, Any], groundtruth: str | dict[str, Any]) -> bool | None:
    """Any pose step in config may match any listed GT option (incl. elbow_bended)."""
    if isinstance(groundtruth, dict):
        pose_any = bool(groundtruth.get("pose_any"))
        options = list(groundtruth.get("pose_options") or [])
        if pose_any:
            return True
        if not options:
            return None
    else:
        pose_any, options, _ = parse_pose_gt_line(str(groundtruth or ""))
        if pose_any:
            return True
        if not options:
            return None

    poses: list[dict[str, Any]] = []
    for step in row.get("movements") or []:
        if step.get("type") != "pose":
            continue
        pose = (step.get("parameters") or {}).get("pose") or {}
        if pose.get("arm_position") and pose.get("gripper_orientation"):
            poses.append(pose)
    if not poses:
        return None
    return any(_pose_matches_option(p, opt) for p in poses for opt in options)


def human_gt_pose_ok(groundtruth: str | dict[str, Any]) -> bool:
    if isinstance(groundtruth, dict):
        return bool(groundtruth.get("pose_any"))
    pose_any, _, _ = parse_pose_gt_line(str(groundtruth or ""))
    return pose_any


def mobile_fixed_pose_from_gt_row(gt_row: dict[str, Any]) -> dict[str, Any]:
    """Primary GT option → mobile pose dict for exp7 fixed start (defaults x/y/z=50)."""
    if gt_row.get("pose_any"):
        return {
            "torso_height": "mid",
            "arm_position": "front",
            "gripper_orientation": "horizontal",
            "head": "center",
            "left_arm": "still",
            "x": 50,
            "y": 50,
            "z": 50,
        }
    options = list(gt_row.get("pose_options") or [])
    if not options:
        _, options, _ = parse_pose_gt_line(str(gt_row.get("pose_gt") or gt_row.get("groundtruth") or ""))
    if not options:
        return {
            "torso_height": "mid",
            "arm_position": "front",
            "gripper_orientation": "horizontal",
            "head": "center",
            "left_arm": "still",
            "x": 50,
            "y": 50,
            "z": 50,
        }
    opt = options[0]
    pose: dict[str, Any] = {
        "torso_height": "mid",
        "arm_position": opt["arm_position"],
        "gripper_orientation": opt["gripper_orientation"],
        "head": "center",
        "left_arm": "still",
        "x": 50,
        "y": 50,
        "z": 50,
    }
    if "elbow_bended" in opt:
        pose["elbow_bended"] = opt["elbow_bended"]
    return pose
