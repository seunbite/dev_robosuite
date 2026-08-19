#!/usr/bin/env python3
"""Render Tiago hand/elbow/fingertip IK-weight variants for visual comparison."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))

from drag_draw import program_to_xyzrpy  # noqa: E402
from run_embodiment_baseline_traj import RENDERERS  # noqa: E402

SOURCE = REPO / "data/results/drag_draw/gemini25pro_c1_5_c26_30/predictions_rendered.json"
OUT = REPO / "data/results/drag_draw/tiago_hand_elbow_weight_sweep"
WEIGHTS = (
    ("equal_cartesian", 20.0, 20.0, 0.2, 0.0),
    ("elbow_biased", 10.0, 20.0, 0.2, 0.0),
    ("balanced", 10.0, 10.0, 0.5, 0.0),
    ("balanced_wrist_eased", 10.0, 10.0, 0.5, 4.0),
)


def main() -> None:
    rows = sorted(
        (
            row for row in json.loads(SOURCE.read_text(encoding="utf-8"))
            if row["platform"] == "tiago" and int(row["idx"]) in (26, 27, 28)
        ),
        key=lambda row: int(row["idx"]),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    sections = []
    for row in rows:
        cards = []
        for name, hand, elbow, tip, wrist in WEIGHTS:
            payload = program_to_xyzrpy(row["program"])
            payload.update({
                "ik_priority": "hand_elbow",
                "show_authored_target_spheres": True,
                "hand_position_weight": hand,
                "elbow_position_weight": elbow,
                "fingertip_direction_weight": tip,
                "wrist_continuity_weight": wrist,
            })
            target = OUT / f"c{int(row['idx']):02d}_{row['cue']}_{name}.gif"
            if not target.is_file():
                RENDERERS["tiago"](payload, "xyzrpy", target, {})
            cards.append(
                "<article>"
                f"<h3>{html.escape(name.replace('_', ' '))}</h3>"
                f"<p>hand {hand:g} · elbow {elbow:g} · tip-dir {tip:g} · wrist-ease {wrist:g}</p>"
                f"<img src='{html.escape(target.name)}?v={target.stat().st_mtime_ns}'>"
                "</article>"
            )
        sections.append(
            f"<section><h2>c{int(row['idx']):02d} · {html.escape(row['cue'])}</h2>"
            f"<div class='grid'>{''.join(cards)}</div></section>"
        )
    gallery = OUT / "gallery_tiago_hand_elbow_weight_sweep.html"
    gallery.write_text(
        """<!doctype html><html><head><meta charset='utf-8'>
<title>Tiago hand/elbow/fingertip weight sweep</title><style>
body{font-family:system-ui;margin:24px;background:#0d1117;color:#e6edf3}.grid{display:grid;grid-template-columns:repeat(4,minmax(260px,1fr));gap:12px}
section{margin:30px 0}article{background:#161b22;border:1px solid #30363d;padding:10px}h3{margin:0;font-size:14px}p{color:#9da7b3;font-size:12px;margin:5px 0 8px}img{display:block;width:100%;background:#000}
</style></head><body><h1>Tiago hand / elbow / fingertip weight sweep</h1>
<p>cyan = hand target · yellow = fingertip target · magenta = elbow target</p>"""
        + "".join(sections)
        + "</body></html>",
        encoding="utf-8",
    )
    print(gallery)


if __name__ == "__main__":
    main()
