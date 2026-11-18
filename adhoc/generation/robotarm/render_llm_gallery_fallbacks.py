#!/usr/bin/env python3
"""Render the four Qwen-authored humanoid cells missing from the 50x5 gallery."""

from __future__ import annotations

import json
from pathlib import Path

from embodiment_renderers import render_unitree_g1
from motion_code_to_config import render_tiago


REPO = Path(__file__).resolve().parents[3]
SOURCE = (
    REPO
    / "data/results/motion_configs/embodiment_motion_code_qwen_qwen32b_humanlike90"
    / "embodiment_motion_code_all.json"
)
OUT = REPO / "data/results/render/embodiment_motion_code_qwen_qwen32b_humanlike90"


def main() -> None:
    entries = json.loads(SOURCE.read_text(encoding="utf-8"))
    by_cue = {entry["cue"]: entry for entry in entries}
    g1_cues = [
        "wait_for_me_backward_palm",
        "frustration_down_throw",
        "clap_applause",
    ]
    tiago_cues = ["clap_applause"]
    g1_rows = [by_cue[cue]["platforms"]["humanoid"] for cue in g1_cues]
    tiago_rows = [by_cue[cue]["platforms"]["tiago"] for cue in tiago_cues]
    render_unitree_g1(g1_rows, OUT / "humanoid" / "gif", force=False)
    render_tiago(tiago_rows, out_root=OUT, force=False)


if __name__ == "__main__":
    main()
