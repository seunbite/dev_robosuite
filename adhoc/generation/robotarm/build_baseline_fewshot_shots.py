#!/usr/bin/env python3
"""Materialize the five current ours shots per robot for trajectory baselines."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[3]
SOURCE = _REPO / "data/results/shots/embodiment_unified/shot_cues_yml_5.json"
OUT = _REPO / "data/seed/shots/embodiment_unified/shots.yaml"
PLATFORMS = ("humanoid", "tiago", "stretch", "single_arm", "roomba")


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    robots: dict[str, dict] = {}
    for platform in PLATFORMS:
        entries = source.get(platform) or []
        if len(entries) != 5:
            raise ValueError(f"{platform}: expected exactly 5 ours few-shots, found {len(entries)}")
        rows: dict[str, dict] = {}
        for entry in entries:
            joint = entry.get("joint_angles") or {}
            rows[str(entry["cue"])] = {
                "motion_code": {"source_code": str(entry["motion_code"])},
                "xyzrpy": {
                    "hz": int(entry.get("hz") or 8),
                    "link_names": entry.get("link_names") or [],
                    "values": entry.get("xyzrpy") or [],
                },
                "joint_angles": {
                    "hz": int(entry.get("hz") or 8),
                    "joint_names": joint.get("joint_names") or [],
                    "values": joint.get("joints_rad") or [],
                },
            }
        robots[platform] = rows
    payload = {
        "meta": {
            "description": "Current ours few-shots, frozen for joint-angle and xyzrpy trajectory baselines.",
            "source": str(SOURCE.relative_to(_REPO)),
            "n_per_robot": 5,
            "abstractions": ["motion_code", "xyzrpy", "joint_angles"],
        },
        "robots": robots,
    }
    OUT.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
