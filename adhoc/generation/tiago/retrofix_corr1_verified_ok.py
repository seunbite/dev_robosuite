#!/usr/bin/env python3
"""Reclassify corr1 records where Q2=n/a (Q1 yes) but action was corrected."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_ROBOTARM = _REPO / "adhoc/generation/robotarm"
for p in (_REPO, _ROBOTARM, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from correction_lib import (  # noqa: E402
    config_snapshot_path,
    correction_cue_dir,
    correction_record_path,
    default_config_path,
    load_correction_record,
    write_config_snapshot,
    _ensure_mp4,
    _gif_first_frame_png,
)
from render_paths import render_media_paths  # noqa: E402
from correction_prompts import correction_indicates_verified_ok  # noqa: E402

MODEL_TAG = "gemini-2.5-pro"
EXP_ID = "1"


def retrofix(*, model_tag: str = MODEL_TAG, exp_id: str = EXP_ID) -> list[str]:
    verify_root = _REPO / "data/results/verify/tiago/correction" / f"exp{exp_id}" / model_tag
    fixed: list[str] = []
    for cue_dir in sorted(verify_root.iterdir()):
        if not cue_dir.is_dir():
            continue
        if not correction_record_path(cue_dir, 1).is_file():
            continue
        rec = load_correction_record(cue_dir, 1)
        if rec.get("action") != "corrected":
            continue
        corr = rec.get("correction") or {}
        if not correction_indicates_verified_ok(corr):
            continue
        cue = str(rec.get("cue", cue_dir.name))
        idx = int((rec.get("input_row") or {}).get("idx", -1))
        input_row = rec.get("input_row")
        if not isinstance(input_row, dict):
            continue

        snap = config_snapshot_path("tiago", exp_id, model_tag, cue, 1)  # type: ignore[arg-type]
        write_config_snapshot(snap, input_row)

        gen_gif, _, _ = render_media_paths("tiago", model_tag, exp_id, idx, cue, "gen")
        corr_gif, corr_mp4, corr_pose = render_media_paths(
            "tiago", model_tag, exp_id, idx, cue, "corr"
        )
        if gen_gif.is_file():
            corr_gif.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(gen_gif, corr_gif)
            _ensure_mp4(corr_gif, corr_mp4)
            _gif_first_frame_png(gen_gif, corr_pose)

        rec["action"] = "verified_ok"
        rec["is_correct"] = True
        rec["output_row"] = input_row
        rec["config_snapshot_json"] = str(snap)
        corr["is_correct"] = True
        if rec.get("media") is None:
            rec["media"] = {}
        if gen_gif.is_file():
            rec["media"]["gif"] = str(corr_gif)
            rec["media"]["mp4"] = str(corr_mp4)

        correction_record_path(cue_dir, 1).write_text(
            json.dumps(rec, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        fixed.append(cue)
        print(f"fixed: {cue}")
    return fixed


if __name__ == "__main__":
    names = retrofix()
    print(f"retrofixed {len(names)} cues")
