#!/usr/bin/env python3
"""Serialize the sophisticated Roomba five-shots for vector baselines."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "data/results/shots/embodiment_unified/roomba_sophisticated_cues_yml_5.json"
OUT = REPO / "data/seed/shots/embodiment_unified/roomba_sophisticated_baseline_shots.yaml"


def main() -> None:
    shots = json.loads(SOURCE.read_text(encoding="utf-8"))
    entries = {}
    for shot in shots:
        joint = shot["joint_angles"]
        entries[shot["cue"]] = {
            "motion_code": {"source_code": shot["motion_code"]},
            "joint_angles": {
                "hz": shot.get("hz", 8),
                "joint_names": joint["joint_names"],
                "values": joint["joints_rad"],
            },
            "xyzrpy": {
                "hz": shot.get("hz", 8),
                "link_names": ["base"],
                "values": shot["xyzrpy"],
            },
        }
    payload = {
        "meta": {"description": "Sophisticated Roomba five-shots for vector baseline pilots"},
        "robots": {"roomba": entries},
    }
    OUT.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
