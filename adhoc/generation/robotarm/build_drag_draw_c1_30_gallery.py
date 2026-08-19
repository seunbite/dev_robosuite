#!/usr/bin/env python3
"""Combine c1–25 and c26–30 trajectory-sketch renders without rerendering."""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))

from run_drag_draw_gemini import reference_gif  # noqa: E402

SOURCES = (
    REPO / "data/results/drag_draw/gemini25pro_c1_25/predictions_rendered.json",
    REPO / "data/results/drag_draw/gemini25pro_c1_5_c26_30/predictions_rendered.json",
)
OUT = REPO / "data/results/drag_draw/gemini25pro_c1_25/gallery_trajectory_sketching_c1_30.html"
PLATFORMS = ("humanoid", "tiago", "stretch", "single_arm", "roomba")


def main() -> None:
    rows_by: dict[tuple[int, str], dict] = {}
    for source in SOURCES:
        for row in json.loads(source.read_text(encoding="utf-8")):
            rows_by[(int(row["idx"]), row["platform"])] = row

    sections = []
    for idx in range(1, 31):
        cards, cue = [], ""
        for platform in PLATFORMS:
            row = rows_by.get((idx, platform))
            if not row:
                cards.append(f"<article><h3>{platform}</h3><p class='bad'>missing prediction</p></article>")
                continue
            cue = row["cue"]
            rendered = REPO / str(row.get("render_gif") or "")
            if not row.get("parse_ok") or not rendered.is_file():
                cards.append(f"<article><h3>{platform}</h3><p class='bad'>render unavailable</p></article>")
                continue
            reference = reference_gif(platform, idx, cue)
            reference_html = (
                f"<figure><img src='{html.escape(os.path.relpath(reference, OUT.parent))}'>"
                "<figcaption>human-authored reference</figcaption></figure>"
                if reference else "<figure><div class='missing'>reference unavailable</div></figure>"
            )
            generated = html.escape(os.path.relpath(rendered, OUT.parent))
            program = yaml.safe_dump(row["program"], sort_keys=False, allow_unicode=True, width=100)
            cards.append(
                f"<article><h3>{html.escape(platform)}</h3><div class='pair'>{reference_html}"
                f"<figure><img src='{generated}?v={rendered.stat().st_mtime_ns}'>"
                "<figcaption>Gemini trajectory sketch</figcaption></figure></div>"
                f"<details><summary>trajectory YAML</summary><pre>{html.escape(program)}</pre></details></article>"
            )
        sections.append(
            f"<section><h2>c{idx:02d} · {html.escape(cue)}</h2><div class='grid'>{''.join(cards)}</div></section>"
        )

    OUT.write_text(
        """<!doctype html><html><head><meta charset='utf-8'>
<title>Gemini-2.5-Pro · Trajectory Sketching · c1–30</title><style>
body{font-family:system-ui;margin:24px;background:#0d1117;color:#e6edf3}.meta{color:#9da7b3}
.grid{display:grid;grid-template-columns:repeat(5,minmax(300px,1fr));gap:12px;overflow-x:auto}
section{margin:28px 0}article{background:#161b22;border:1px solid #30363d;padding:10px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:5px}
figure{margin:0}img{display:block;width:100%;background:#000}figcaption{text-align:center;color:#9da7b3;font-size:11px;padding-top:4px}
pre{font-size:10px;max-height:520px;overflow:auto;white-space:pre-wrap}.bad{color:#ff7b72}.missing{padding:60px 10px;color:#9da7b3;text-align:center}
</style></head><body><h1>Gemini-2.5-Pro · Trajectory Sketching</h1>
<p class='meta'>30 cues (c1–30) × 5 embodiments · thinking budget 128 · five-shot prompting</p>"""
        + "".join(sections)
        + "</body></html>",
        encoding="utf-8",
    )
    print(OUT)


if __name__ == "__main__":
    main()
