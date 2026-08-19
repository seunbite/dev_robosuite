#!/usr/bin/env python3
"""Create and render five visually auditable one-factor correction foils.

This is the small, inspect-first precursor to the balanced 100-item primitive
correction benchmark.  It reuses the GT/foil renderer so GT and negative share
the same camera and initial pose.
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from run_dynamic_axis_foil import build_gallery, render_foils  # noqa: E402

MASTER = REPO / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
OUT = REPO / "data/results/evaluation/primitive_correction_pilot"


def cell(master: list[dict], idx: int, platform: str) -> tuple[dict, dict]:
    entry = next(x for x in master if int(x["idx"]) == idx)
    return entry, deepcopy(entry["platforms"][platform])


def main() -> None:
    master = json.loads(MASTER.read_text())
    rows: list[dict] = []

    # Each candidate mutates exactly one semantic field.  The small pilot covers
    # five factors; the 100-item benchmark will stratify all six factors.
    specs = [
        ("humanoid", 3, "movement_axis", "xz", "yz"),
        ("tiago", 3, "pose_xyz", (None, None, 60), (None, None, 25)),
        ("single_arm", 3, "movement_axis", "yz", "xz"),
        ("stretch", 3, "movement_speed", 1.4, 0.20),
        ("roomba", 3, "movement_repetition", 2, 1),
    ]
    for platform, idx, factor, old, new in specs:
        entry, negative = cell(master, idx, platform)
        original = deepcopy(negative)
        movements = negative["movements"]
        if factor == "gripper_orientation":
            movements[0]["parameters"]["pose"]["gripper_orientation"] = new
        elif factor == "pose_xyz":
            movements[0]["parameters"]["pose"]["z"] = new[2]
        elif factor == "movement_axis":
            payload = movements[1]["parameters"]
            if "movement" in payload:
                payload["movement"]["joints"][0]["axis"] = new
            elif "path" in payload:
                payload["path"]["plane"] = new
            elif "plane" in payload:
                payload["plane"] = new
            else:
                payload["directions"][0]["degrees"] = {new: -30.0}
                payload["directions"][1]["degrees"] = {new: 30.0}
        elif factor == "movement_speed":
            payload = movements[1]["parameters"]
            (payload.get("movement") or payload.get("path") or payload)["speed"] = new
        elif factor == "movement_repetition":
            movements[0]["parameters"]["path"]["repetition"] = new
        else:
            raise ValueError(factor)
        rows.append({
            "platform": platform,
            "idx": idx,
            "cue": entry["cue"],
            "category": entry.get("category") or "",
            "factor": factor,
            "old_value": old,
            "new_value": new,
            "old_code": original.get("source_code") or "",
            "new_code": (original.get("source_code") or "") + f"  # NEG {factor}: {old}->{new}",
            "note": f"single-factor negative: {factor} {old} -> {new}",
            "parse_ok": True,
            "movements": movements,
            "original_movements": original["movements"],
            "convert_notes": [],
            "actual_motion_diff": True,
        })

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pilot_manifest.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    render_foils(rows, foil_json_dir=OUT / "motion_configs", foil_render_root=OUT / "render_negative", true_render_root=OUT / "render_gt")
    html = OUT / "gallery_primitive_correction_pilot.html"
    build_gallery(rows, out_html=html, true_root=OUT / "render_gt", foil_root=OUT / "render_negative")
    print(html)


if __name__ == "__main__":
    main()
