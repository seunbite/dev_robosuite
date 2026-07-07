"""Normalize mobile pose fields before render / config save."""
from __future__ import annotations

from typing import Any

ARM_POSITIONS = frozenset({"front", "back", "in", "out", "up", "down"})


def normalize_pose_dict(pose: dict[str, Any]) -> dict[str, Any]:
    """When left_arm duplicates arm_position, use mirror (symmetric bimanual)."""
    if not isinstance(pose, dict):
        return pose
    p = dict(pose)
    arm = str(p.get("arm_position", "")).strip().lower()
    left = str(p.get("left_arm", "still")).strip().lower()
    if left in {"still", "mirror"}:
        return p
    if left == arm and arm in ARM_POSITIONS:
        p["left_arm"] = "mirror"
    return p


def normalize_movements(movements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for step in movements or []:
        if not isinstance(step, dict):
            out.append(step)
            continue
        if step.get("type") != "pose":
            out.append(step)
            continue
        params = dict(step.get("parameters") or {})
        pose = params.get("pose")
        if isinstance(pose, dict):
            params["pose"] = normalize_pose_dict(pose)
        out.append({**step, "parameters": params})
    return out


def normalize_config_row(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        return row
    out = dict(row)
    if isinstance(out.get("movements"), list):
        out["movements"] = normalize_movements(out["movements"])
    gfp = out.get("gt_fixed_first_pose")
    if isinstance(gfp, dict):
        out["gt_fixed_first_pose"] = normalize_pose_dict(gfp)
    return out
