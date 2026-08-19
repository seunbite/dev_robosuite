#!/usr/bin/env python3
"""Render Tiago vector programs across elbow-branch continuity thresholds."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[3]
sys.path.insert(0, str(_HERE.parent))

from drag_draw import program_to_xyzrpy  # noqa: E402
from run_embodiment_baseline_traj import RENDERERS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO / "data/results/drag_draw/gemini25pro_c1_5_c26_30/predictions_latest.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "data/results/drag_draw/gemini25pro_c1_5_c26_30/elbow_threshold_sweep",
    )
    parser.add_argument("--indices", default="27,28")
    parser.add_argument("--thresholds", default="0.08,0.06,0.04,0.02")
    parser.add_argument("--elbow-mode", choices=("llm", "previous"), default="llm")
    args = parser.parse_args()

    indices = {int(value) for value in args.indices.split(",")}
    thresholds: list[float | None] = (
        [None] if args.elbow_mode == "previous"
        else [float(value) for value in args.thresholds.split(",")]
    )
    rows = json.loads(args.source.read_text(encoding="utf-8"))
    selected = [
        row for row in rows
        if row.get("platform") == "tiago" and int(row["idx"]) in indices and row.get("parse_ok")
    ]
    selected.sort(key=lambda row: int(row["idx"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sections = []
    for row in selected:
        cards = []
        for threshold in thresholds:
            payload = program_to_xyzrpy(row["program"])
            payload["elbow_selection_mode"] = (
                "previous" if args.elbow_mode == "previous" else "llm"
            )
            if threshold is not None:
                payload["elbow_selection_tolerance_m"] = threshold
            suffix = "previous_elbow" if threshold is None else f"step_{threshold:.2f}"
            target = args.out_dir / f"c{int(row['idx']):02d}_{row['cue']}_{suffix}.gif"
            RENDERERS["tiago"](payload, "xyzrpy", target, {})
            infeasible = payload.get("_ik_infeasible_frames") or []
            status = (
                f"<p class='bad'>infeasible at {len(infeasible)} frame(s); "
                "GIF shows the exact-hand fallback</p>"
                if infeasible else "<p class='good'>hand/fingertip + elbow continuity feasible</p>"
            )
            label = (
                "previous-frame elbow"
                if threshold is None else f"{threshold * 100:.0f} cm elbow / frame"
            )
            cards.append(
                "<article>"
                f"<h3>{label}</h3>"
                f"<img src='{html.escape(target.name)}?v={target.stat().st_mtime_ns}'>"
                f"{status}"
                "</article>"
            )
        sections.append(
            f"<section><h2>c{int(row['idx']):02d} · {html.escape(row['cue'])}</h2>"
            f"<div class='grid'>{''.join(cards)}</div></section>"
        )

    page = """<!doctype html><html><head><meta charset='utf-8'>
<title>Tiago elbow continuity threshold sweep</title>
<style>
body{font-family:system-ui;margin:24px;background:#0d1117;color:#e6edf3}
h1{margin-bottom:4px}.meta{color:#9da7b3}.grid{display:grid;grid-template-columns:repeat(4,minmax(260px,1fr));gap:12px}
section{margin:28px 0}article{background:#161b22;border:1px solid #30363d;padding:10px}h3{font-size:14px;margin:0 0 8px}
img{display:block;width:100%;background:#000}.good{color:#3fb950;font-size:12px}.bad{color:#ff7b72;font-size:12px}
</style></head><body><h1>Tiago elbow continuity threshold sweep</h1>
<p class='meta'>Hand XYZ + fingertip direction are hard feasibility tests. Threshold only chooses among those IK equilibria; it never rate-limits the hand.</p>
""" + "".join(sections) + "</body></html>"
    gallery_name = (
        "gallery_tiago_hand_tip_temporal_elbow.html"
        if args.elbow_mode == "previous"
        else "gallery_tiago_elbow_threshold_sweep.html"
    )
    gallery = args.out_dir / gallery_name
    gallery.write_text(page, encoding="utf-8")
    print(gallery)


if __name__ == "__main__":
    main()
