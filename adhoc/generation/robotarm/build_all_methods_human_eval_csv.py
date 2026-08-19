#!/usr/bin/env python3
"""Export all 5 × 150 human yes/no cells into one reviewable CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from build_all_methods_human_eval_gallery import (
    BASELINE_EVAL,
    GT_EVAL,
    MOTION_PROGRAM_EVAL,
    PLATFORMS,
    TRAJECTORY_EVAL,
    cue_map,
    trajectory_labels,
    wrong_indices,
)

REPO = Path(__file__).resolve().parents[3]
OUT = (
    REPO
    / "data/results/evaluation/embodiment_cue_id"
    / "all_methods_human_yesno_c1_30.csv"
)
METHODS = (
    ("gt", "GT", GT_EVAL),
    ("joint", "Joint-Space", BASELINE_EVAL),
    ("xyzrpy", "Cartesian-Space", BASELINE_EVAL),
    ("motion_program", "Motion Program", MOTION_PROGRAM_EVAL),
    ("trajectory", "Trajectory Sketching", TRAJECTORY_EVAL),
)


def yes_no(label: str) -> str:
    return "YES" if label == "o" else "NO" if label == "x" else "N/A"


def main() -> None:
    baseline = json.loads(BASELINE_EVAL.read_text(encoding="utf-8"))
    gt_wrong = wrong_indices(GT_EVAL)
    motion_wrong = wrong_indices(MOTION_PROGRAM_EVAL)
    trajectory = trajectory_labels()
    cues = cue_map()

    rows: list[dict[str, str | int]] = []
    for idx in range(1, 31):
        for platform in PLATFORMS:
            labels = {
                "gt": "x" if idx in gt_wrong[platform] else "o",
                "joint": baseline["cell_labels"][platform][f"g{idx:02d}"]["joint"],
                "xyzrpy": baseline["cell_labels"][platform][f"g{idx:02d}"]["xyzrpy"],
                "motion_program": "x" if idx in motion_wrong[platform] else "o",
                "trajectory": trajectory[(platform, idx)],
            }
            for method_id, method_name, source in METHODS:
                original = yes_no(labels[method_id])
                rows.append(
                    {
                        "cue_id": f"c{idx:02d}",
                        "idx": idx,
                        "cue": cues[idx],
                        "platform": platform,
                        "method_id": method_id,
                        "method": method_name,
                        "original_human_yesno": original,
                        # Edit only this column during review; blank means retain original.
                        "review_override_yesno": "",
                        "final_human_yesno": original,
                        "label_source": str(source.relative_to(REPO)),
                    }
                )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts = {
        method_id: sum(
            row["original_human_yesno"] == "YES"
            for row in rows
            if row["method_id"] == method_id
        )
        for method_id, _, _ in METHODS
    }
    print(f"{OUT} rows={len(rows)} yes={counts}")


if __name__ == "__main__":
    main()
