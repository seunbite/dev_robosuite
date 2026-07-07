"""Score Google Robot mobile movement tails against component GT."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
_ROBOTARM = _REPO / "adhoc/generation/robotarm"
_GOOGLE = _REPO / "adhoc/generation/google_robot"
for p in (_ROBOTARM, _GOOGLE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from parse_pose_gt_mobile import (  # noqa: E402
    human_gt_pose_ok,
    pose_generation_correct_any_mobile,
)
from score_pilot40_motion_gt_components import (  # noqa: E402
    _tail_matches_component,
    _tail_steps,
)

_AXIS_TO_CART = {
    "pitch": "x",
    "roll": "y",
    "tilt": "z",
    "pan": "y",
    "height": "z",
}


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
