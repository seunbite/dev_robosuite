#!/usr/bin/env python3
"""Validate and render the five 11D Stretch Cartesian few-shot trajectories."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import run_embodiment_baseline_traj as baseline  # noqa: E402


OUT = REPO / "data/results/baseline_traj/ours5/stretch_cartesian_11d_fewshot_validation"


def main() -> None:
    gt = baseline._index_gt()
    shots = baseline._fewshot_artifacts(
        REPO / "data/results/baseline_traj/ours5",
        "stretch",
        gt,
        shots_path=baseline.DEFAULT_FEWSHOT_SHOTS,
    )
    if len(shots) != 5:
        raise RuntimeError(f"expected five Stretch few-shots, got {len(shots)}")

    report = []
    for shot in shots:
        cartesian = shot["stretch_cartesian"]
        widths = {name: len(rows[0]) for name, rows in cartesian.items()}
        lengths = {name: len(rows) for name, rows in cartesian.items()}
        if widths != {"base_xyyaw": 3, "head_rp": 2, "hand_xyzrpy": 6}:
            raise ValueError(f"{shot['cue']}: invalid Cartesian widths {widths}")
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{shot['cue']}: inconsistent Cartesian frame counts {lengths}")

        payload = {
            "cue": shot["cue"],
            "hz": shot.get("hz", baseline.HZ),
            "stretch_cartesian": cartesian,
        }
        gif = OUT / f"{shot['cue']}.gif"
        baseline.render_stretch_traj(payload, "xyzrpy", gif, {})
        report.append(
            {
                "cue": shot["cue"],
                "n_frames": lengths["hand_xyzrpy"],
                "dimensions": {"base": 3, "head": 2, "hand": 6, "total": 11},
                "gif": str(gif.relative_to(REPO)),
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation.json").write_text(
        json.dumps({"status": "pass", "examples": report}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "pass", "examples": len(report), "out": str(OUT)}))


if __name__ == "__main__":
    main()
