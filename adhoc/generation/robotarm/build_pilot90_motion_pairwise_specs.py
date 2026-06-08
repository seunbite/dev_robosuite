#!/usr/bin/env python3
"""Write step-10 pairwise spec JSON for pilot-90 (MP4 paths filled when media exists)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pilot90_experiment_suite import MOTION_PAIRWISE_DIR, manifest90_rows_from_cfg  # noqa: E402
from score_pilot40_motion_gt_components import _build_annotation_map  # noqa: E402

OUT = MOTION_PAIRWISE_DIR / "pairwise_specs_pilot90.json"
CFG = (
    _REPO
    / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json"
)


def main() -> None:
    ann = {int(a["cue_idx"]): a for a in _build_annotation_map()}
    rows = manifest90_rows_from_cfg(json.loads(CFG.read_text(encoding="utf-8")))
    specs: list[dict] = []
    for row in rows:
        idx = int(row["idx"])
        cue = str(row["cue"])
        entry = ann.get(idx) or {}
        if entry.get("always_correct") or (entry.get("component") or {}).get("kind") == "any":
            continue
        if not entry.get("component"):
            continue
        pair_mp4 = MOTION_PAIRWISE_DIR / f"{idx:03d}_{cue}_pair_axis.mp4"
        rel = str(pair_mp4.relative_to(_REPO)) if pair_mp4.is_file() else None
        specs.append(
            {
                "idx": idx,
                "cue": cue,
                "gt_side": "left",
                "left": "gt",
                "right": "axis",
                "pair_mp4": rel,
            }
        )
    MOTION_PAIRWISE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "n": len(specs),
        "n_with_mp4": sum(1 for s in specs if s.get("pair_mp4")),
        "mp4": specs,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(specs)} specs ({payload['n_with_mp4']} with mp4) → {OUT}")


if __name__ == "__main__":
    main()
