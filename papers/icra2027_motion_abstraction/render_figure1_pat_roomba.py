#!/usr/bin/env python3
"""Render the human-authored Roomba pat-stomach motion used in Fig. 1."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ROBOTARM = ROOT / "adhoc/generation/robotarm"
sys.path.insert(0, str(ROBOTARM))

from embodiment_renderers import render_roomba  # noqa: E402
from motion_code_to_config import convert_code  # noqa: E402


CODE = (
    "path(base, arc, r=0, deg=-90, spd=1.5) + "
    "path(base, arc, r=0, deg=+/-10, spd=1.5, rep=4)"
)
OUT = ROOT / "data/results/render/figure1_human_authored_roomba/roomba/gif"


def main() -> None:
    converted = convert_code(CODE, platform="roomba")
    if not converted.get("parse_ok"):
        raise ValueError(f"Could not parse motion program: {converted.get('notes')}")
    row = {
        "idx": 32,
        "cue": "pat_stomach",
        "description": "Turns toward the stomach, then pats with four small yaw beats.",
        "movements": converted["movements"],
        "source_code": CODE,
        "parse_ok": True,
        "platform": "roomba",
    }
    rendered = render_roomba([row], OUT, force=True, fps=12)
    print(rendered["pat_stomach"])


if __name__ == "__main__":
    main()
