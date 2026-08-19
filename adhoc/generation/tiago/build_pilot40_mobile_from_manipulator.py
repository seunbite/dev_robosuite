#!/usr/bin/env python3
"""Build 39-cue Tiago mobile configs from manipulator pilot-40 gt-fixed JSON."""
from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]

MANIP_GT = _REPO / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_gt_fixed_pose_pilot40.json"
CUES_MEANING = _REPO / "data/seed/yml/cues_meaning.yml"
DEFAULT_SHOTS_OUT = _REPO / "data/seed/shots/tiago/shot_configs_pilot40_mobile.json"
DEFAULT_MOTION_OUT = _REPO / "data/results/motion_configs/tiago/motion_configs_pilot40_mobile.json"

# manipulator 6-way dir → mobile arm_position (pose-bank lookup labels)
DIR_TO_ARM: dict[str, str] = {
    "front": "front",
    "back": "back",
    "left": "out",
    "right": "in",
    "up": "up",
    "down": "down",
}

DEG_TO_M = 0.004


def _parse_cue_groups(path: Path) -> dict[str, str]:
    groups: dict[str, str] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current = line.rstrip(":").strip()
            continue
        if current is None or ":" not in line:
            continue
        key = line.strip().split(":", 1)[0].strip()
        if key:
            groups[key] = current
    return groups


def _tail_steps(movements: list) -> list[dict]:
    out: list[dict] = []
    seen_pose = False
    for step in movements or []:
        if step.get("type") == "pose":
            seen_pose = True
            continue
        if seen_pose:
            out.append(step)
    return out


def _meters_from_deg(val: float) -> float:
    return round(float(val) * DEG_TO_M, 3)


def _dominant_axis(degrees: dict[str, Any]) -> tuple[str, float]:
    best_ax, best_v = "x", 0.0
    for ax in ("x", "y", "z"):
        v = degrees.get(ax)
        if isinstance(v, (int, float)) and abs(float(v)) >= abs(best_v):
            best_v = float(v)
            best_ax = ax
    return best_ax, best_v


def _convert_path_step(p: dict[str, Any]) -> dict[str, Any]:
    path = copy.deepcopy(p)
    shape = str(path.get("shape", "line")).lower()
    if shape == "line" and isinstance(path.get("distance"), (int, float)):
        path["distance"] = _meters_from_deg(float(path["distance"]))
    if shape == "arc":
        if isinstance(path.get("radius"), (int, float)):
            path["radius"] = _meters_from_deg(float(path["radius"]))
        if isinstance(path.get("sweep"), (int, float)):
            path["sweep"] = float(path["sweep"])
    path["mode"] = "ee"
    rep = path.get("repetition", p.get("repetition", 1))
    if rep is not None:
        path["repetition"] = int(rep)
    speed = path.get("speed", p.get("speed", 1.0))
    if speed is not None:
        path["speed"] = float(speed)
    hold = path.get("hold_time", p.get("hold_time", 0.0))
    if hold is not None and float(hold or 0) > 0:
        path["hold_time"] = float(hold)
    return {"type": "path", "parameters": {"path": path}}


def _convert_movement_step(p: dict[str, Any]) -> dict[str, Any] | None:
    directions = p.get("directions") or []
    if not directions:
        return None
    rep = int(p.get("repetition", 1) or 1)
    speeds = [float(d.get("speed", p.get("speed", 1.0)) or 1.0) for d in directions]
    holds = [float(d.get("hold_time", 0) or 0) for d in directions]
    speed = sum(speeds) / len(speeds)
    hold_time = max(holds) if holds else 0.0

    if len(directions) >= 2:
        ax0, v0 = _dominant_axis(directions[0].get("degrees") or {})
        ax1, v1 = _dominant_axis(directions[1].get("degrees") or {})
        if ax0 == ax1 and v0 * v1 < 0:
            path = {
                "shape": "line",
                "axis": ax0,
                "distance": _meters_from_deg(abs(v0)),
                "speed": speed,
                "repetition": max(2, rep),
                "mode": "ee",
            }
            if hold_time > 0:
                path["hold_time"] = hold_time
            return {"type": "path", "parameters": {"path": path}}

    ax, val = _dominant_axis(directions[0].get("degrees") or {})
    if val == 0:
        return None
    path = {
        "shape": "line",
        "axis": ax,
        "distance": _meters_from_deg(val),
        "speed": speed,
        "repetition": max(1, rep),
        "mode": "ee",
    }
    if hold_time > 0:
        path["hold_time"] = hold_time
    return {"type": "path", "parameters": {"path": path}}


def _convert_tail_step(step: dict[str, Any]) -> dict[str, Any] | None:
    t = step.get("type")
    p = step.get("parameters") or {}
    if t == "path":
        return _convert_path_step(p)
    if t == "movement":
        return _convert_movement_step(p)
    return None


def _mobile_first_pose(m_pose: dict[str, Any]) -> dict[str, Any]:
    d = str(m_pose.get("dir", "front")).strip().lower()
    return {
        "torso_height": "mid",
        "arm_position": DIR_TO_ARM.get(d, "front"),
        "gripper_orientation": str(m_pose.get("gripper_orientation", "horizontal")),
        "head": "center",
        "x": int(round(float(m_pose.get("x", 50)))),
        "y": int(round(float(m_pose.get("y", 50)))),
        "z": int(round(float(m_pose.get("z", 50)))),
        "left_arm": "still",
    }


def convert_row(row: dict[str, Any], cue_groups: dict[str, str]) -> dict[str, Any]:
    cue = str(row["cue"])
    movements = row.get("movements") or []
    first_m_pose = {}
    for step in movements:
        if step.get("type") == "pose":
            first_m_pose = (step.get("parameters") or {}).get("pose") or {}
            break

    mobile_pose = _mobile_first_pose(first_m_pose)
    mobile_movements: list[dict[str, Any]] = [
        {
            "type": "pose",
            "duration": 0.5,
            "parameters": {"pose": mobile_pose},
        }
    ]
    for step in _tail_steps(movements):
        converted = _convert_tail_step(step)
        if converted:
            mobile_movements.append(converted)

    grp = cue_groups.get(cue, "?")
    reasoning = str(row.get("reasoning") or "").strip()
    if not reasoning:
        reasoning = f"# mapped from manipulator pilot-40 gt_fixed: {cue}"

    return {
        "group": grp,
        "cue": cue,
        "cue_text": str(row.get("description", cue)).split(".")[0][:120],
        "description": f"Pilot-40 manipulator gt_fixed → mobile EE. {row.get('description', '')}".strip(),
        "reasoning": reasoning,
        "movements": mobile_movements,
        "idx": int(row["idx"]),
        "state": "pilot40_mapped",
        "model": "manipulator-gt_fixed-v1",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": {
            "manipulator_config": str(MANIP_GT.relative_to(_REPO)),
            "manipulator_idx": int(row["idx"]),
            "groundtruth": row.get("groundtruth"),
        },
    }


def run(args: argparse.Namespace) -> None:
    manip_rows = json.loads(Path(args.manipulator_json).read_text(encoding="utf-8"))
    cue_groups = _parse_cue_groups(Path(args.cues_meaning))
    converted = [convert_row(r, cue_groups) for r in sorted(manip_rows, key=lambda x: int(x["idx"]))]

    shots_out = Path(args.shots_out)
    motion_out = Path(args.motion_out)
    shots_out.parent.mkdir(parents=True, exist_ok=True)
    motion_out.parent.mkdir(parents=True, exist_ok=True)
    shots_out.write_text(json.dumps(converted, indent=2, ensure_ascii=False), encoding="utf-8")
    motion_out.write_text(json.dumps(converted, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(converted)} cues → {shots_out}")
    print(f"Wrote {len(converted)} cues → {motion_out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manipulator-json", default=str(MANIP_GT))
    p.add_argument("--cues-meaning", default=str(CUES_MEANING))
    p.add_argument("--shots-out", default=str(DEFAULT_SHOTS_OUT))
    p.add_argument("--motion-out", default=str(DEFAULT_MOTION_OUT))
    run(p.parse_args())


if __name__ == "__main__":
    main()
