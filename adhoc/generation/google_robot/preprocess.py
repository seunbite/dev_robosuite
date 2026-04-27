#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import fire

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

def run(run_motion_generation: bool = True, cue_group: str = "iconic"):
    print("google_robot preprocess: no simulator cache step.")
    if run_motion_generation:
        from motion_generation import run as motion_run

        motion_run(cue_group=cue_group)


if __name__ == "__main__":
    fire.Fire(run)
