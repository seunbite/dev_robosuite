"""Score Google Robot mobile movement tails against component GT."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
_ROBOTARM = _REPO / "adhoc/generation/robotarm"
if str(_ROBOTARM) not in sys.path:
    sys.path.insert(0, str(_ROBOTARM))

from pilot40_experiment_suite import _parse_gt_poses  # noqa: E402
from score_pilot40_motion_gt_components import (  # noqa: E402
    _tail_matches_component,
    _tail_steps,
)

_ARM_TO_DIR = {
    "front": "front",
    "back": "back",
    "in": "right",
    "out": "left",
    "up": "up",
    "down": "down",
}

_AXIS_TO_CART = {
    "pitch": "x",
    "roll": "y",
    "tilt": "z",
    "pan": "y",
    "height": "z",
}


def mobile_pose_to_dir_grip(pose: dict[str, Any]) -> dict[str, str]:
    arm = str(pose.get("arm_position", "")).strip().lower()
    return {
        "dir": _ARM_TO_DIR.get(arm, arm),
        "gripper_orientation": str(pose.get("gripper_orientation", "")).strip().lower(),
    }


def pose_generation_correct_any_mobile(row: dict[str, Any], groundtruth: str) -> bool | None:
    """Any pose step in config may match any listed GT (dir, grip) pair."""
    poses: list[dict[str, str]] = []
    for step in row.get("movements") or []:
        if step.get("type") != "pose":
            continue
        pose = (step.get("parameters") or {}).get("pose") or {}
        if pose.get("arm_position") and pose.get("gripper_orientation"):
            poses.append(mobile_pose_to_dir_grip(pose))
    if not groundtruth or not poses:
        return None
    targets = _parse_gt_poses(groundtruth.strip())
    if not targets:
        return None
    gen_set = {(p["dir"], p["gripper_orientation"]) for p in poses}
    return any(t in gen_set for t in targets)


def _normalize_mobile_step(step: dict[str, Any]) -> dict[str, Any]:
    t = step.get("type")
    params = step.get("parameters") or {}
    if t == "path":
        path = params.get("path") or {}
        return {
            "type": "path",
            "parameters": {
                "shape": path.get("shape"),
                "axis": path.get("axis"),
                "distance": path.get("distance"),
                "plane": path.get("plane"),
                "hold_time": path.get("hold_time", 0),
            },
        }
    if t == "movement":
        mv = params.get("movement") or {}
        directions: list[dict[str, Any]] = []
        joint_name: str | None = None
        for j in mv.get("joints") or []:
            raw_joint = str(j.get("joint", "")).strip().lower()
            if raw_joint in {"right_arm", "left_arm"}:
                joint_name = joint_name or "shoulder"
            elif raw_joint and not joint_name:
                joint_name = raw_joint
            axis = str(j.get("axis", "")).strip().lower()
            cart = _AXIS_TO_CART.get(axis, axis if axis in "xyz" else "")
            degs = j.get("degrees")
            if cart and isinstance(degs, list) and len(degs) >= 2:
                try:
                    delta = float(degs[1]) - float(degs[0])
                except (TypeError, ValueError):
                    continue
                if delta != 0.0:
                    directions.append({"degrees": {cart: delta}})
            elif cart and isinstance(degs, (int, float)):
                directions.append({"degrees": {cart: float(degs)}})
        return {
            "type": "movement",
            "parameters": {
                "joint": joint_name,
                "repetition": int(mv.get("repetition", 1) or 1),
                "directions": directions,
            },
        }
    return step


def mobile_tail_matches_component(tail: list[dict[str, Any]], comp: dict[str, Any]) -> tuple[bool, str | None]:
    normalized = [_normalize_mobile_step(s) for s in tail]
    return _tail_matches_component(normalized, comp)


def mobile_tail_steps(movements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _tail_steps(movements)
