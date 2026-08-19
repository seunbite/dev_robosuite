#!/usr/bin/env python3
"""Convert sophisticated Roomba Motion Code shots into trajectory sketches."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from drag_draw import load_schema, row_to_vector_program  # noqa: E402

SOURCE = REPO / "data/results/shots/embodiment_unified/roomba_sophisticated_cues_yml_5.json"
OUT = REPO / "data/seed/shots/drag_draw/roomba_sophisticated_fewshots.yaml"


def main() -> None:
    schema = load_schema()
    shots = json.loads(SOURCE.read_text(encoding="utf-8"))
    result = {"roomba": []}
    for shot in shots:
        result["roomba"].append(
            {
                "cue": shot["cue"],
                "shot_id": shot["shot_id"],
                "source_motion_code": shot["motion_code"],
                "drag_draw": row_to_vector_program("roomba", shot, schema),
            }
        )
    OUT.write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True, width=140), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
