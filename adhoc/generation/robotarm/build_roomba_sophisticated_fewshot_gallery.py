#!/usr/bin/env python3
"""Render sophisticated Roomba alternatives beside the frozen 5 × 5 few-shots."""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from build_embodiment_unified_shots import _add, enrich_and_render  # noqa: E402

OUT = REPO / "data/results/shots/embodiment_unified"
SOURCE = OUT / "shot_cues_yml_5.json"
SELECTION = OUT / "gallery_5col_cues_yml_selection.json"
SOPHISTICATED_JSON = OUT / "roomba_sophisticated_cues_yml_5.json"
GALLERY = OUT / "gallery_6col_roomba_sophisticated.html"
PLATFORMS = ("humanoid", "tiago", "stretch", "single_arm", "roomba")

# Every trajectory starts with an actual translating S-curve (three connected
# circular arcs), then uses a cue-specific mobile-base gesture.
SOPHISTICATED_CODES = {
    "raising_hand_greeting": (
        "pose(dir=front) + path(base, arc, r=0, deg=[30, -45], rep=1, spd=0.5) + "
        "path(base, arc, r=0, deg=15, rep=1, spd=2, hold=0.5) + "
        "path(base, line, x=0.5, spd=3)"
    ),
    "wave_hi": (
        "pose(dir=front) + path(base, arc, r=0, deg=[-80,80], rep=2, spd=2.0, hold=0.25) + "
        "path(base, line, x=0.5, spd=3) + "
        "path(base, arc, r=0, deg=[-80,80], rep=2, spd=2.0, hold=0.25)"
    ),
    "bow_greeting": (
        "pose(dir=front) + path(base, line, x=0.1, spd=1, hold=0.2) + "
        "path(base, line, x=0.1, spd=1, hold=0.2) + "
        "path(base, line, x=0.1, spd=1, hold=0.2) + "
        "path(base, line, x=0.1, spd=1, hold=0.2) + "
        "path(base, line, x=0.2, spd=0.5) + path(base, line, x=-0.2, spd=0.5, hold=0.5) + "
        "path(base, line, x=-0.06, spd=1.1, hold=0.1) + "
        "path(base, line, x=-0.06, spd=1.1, hold=0.1) + "
        "path(base, line, x=-0.06, spd=1.1, hold=0.1) + "
        "path(base, line, x=-0.06, spd=1.1, hold=0.1)"
    ),
    "blow_kiss": (
        "pose(dir=front) + path(base, arc, r=0, deg=+-20, rep=2, spd=3.0) + "
        "path(base, line, x=0.14, spd=1.6, hold=0.15) + "
        "path(base, line, x=-0.08, spd=1.6, hold=0.1) + "
        "path(base, arc, r=0.5, deg=82, spd=3.0, hold=0.35)"
    ),
    "circle_temple_crazy": (
        "pose(dir=front) + path(base, arc, r=0.08, deg=720, spd=1.2) + "
        "path(base, arc, r=0.10, deg=720, spd=1.8) + "
        "path(base, arc, r=0.24, deg=720, spd=3.2, hold=0.25)"
    ),
}


def _card(entry: dict[str, Any], label: str) -> str:
    code = html.escape(entry.get("motion_code") or "")
    if not entry.get("render_ok"):
        return (
            f"<td class='fail'><div class='plat'>{html.escape(label)}</div>"
            f"<p class='miss'>{html.escape(entry.get('render_error') or 'render failed')}</p>"
            f"<pre class='mc'>{code}</pre></td>"
        )
    gif_path = REPO / entry["render_gif"]
    gif = html.escape(
        f"{os.path.relpath(gif_path, OUT)}?v={gif_path.stat().st_mtime_ns}"
    )
    cue = html.escape(entry["cue"])
    return (
        f"<td><div class='plat'>{html.escape(label)}</div>"
        f"<img src='{gif}' alt='{cue}'/>"
        f"<pre class='mc'>{code}</pre></td>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-render", action="store_true")
    args = parser.parse_args()

    frozen: dict[str, list[dict[str, Any]]] = json.loads(SOURCE.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    cues = [item["cue"] for item in sorted(selection, key=lambda item: item["rank"])]

    candidates: dict[str, list[dict[str, Any]]] = {"roomba": []}
    for cue in cues:
        _add(
            candidates,
            platform="roomba",
            cue=cue,
            motion_code=SOPHISTICATED_CODES[cue],
            source="adhoc/generation/robotarm/build_roomba_sophisticated_fewshot_gallery.py",
            role="sophisticated",
            shot_id=f"roomba__{cue}__sophisticated",
            notes="Translating S-curve approach plus cue-specific mobile-base gesture.",
        )
    candidates = enrich_and_render(candidates, force=args.force_render)
    SOPHISTICATED_JSON.write_text(
        json.dumps(candidates["roomba"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    enhanced = {entry["cue"]: entry for entry in candidates["roomba"]}

    rows = []
    for item in sorted(selection, key=lambda value: value["rank"]):
        cue = item["cue"]
        row = (
            f"<tr><th><div class='rank'>#{item['rank']} · {html.escape(item.get('difficulty') or '')}</div>"
            f"<div class='cue'>{html.escape(cue)}</div>"
            f"<div class='why'>{html.escape(item.get('why') or '')}</div></th>"
        )
        for platform in PLATFORMS:
            entry = next(value for value in frozen[platform] if value["cue"] == cue)
            row += _card(entry, platform.replace("_", " ").title())
        row += _card(enhanced[cue], "Roomba · Sophisticated")
        rows.append(row + "</tr>")

    GALLERY.write_text(
        """<!doctype html><html><head><meta charset='utf-8'>
<title>Few-shot · Roomba sophisticated alternatives</title>
<style>
body{font-family:ui-sans-serif,system-ui,sans-serif;margin:24px;background:#0f1115;color:#e8eaed}
h1{font-size:1.25rem;margin:0 0 6px}.sub{color:#9aa0a6;margin:0 0 18px;font-size:.9rem;max-width:100ch}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #2a2f3a;padding:8px;vertical-align:top;width:14%}
th:first-child{width:12%;text-align:left;background:#161a22}td:last-child{background:#142337;border-color:#3b82f6}
img{width:100%;max-width:220px;background:#000;border-radius:4px;display:block}.plat{font-size:.75rem;color:#9aa0a6;margin-bottom:4px}
.cue{font-family:ui-monospace,monospace;font-size:.95rem;margin:4px 0}.rank{color:#8ab4f8;font-size:.8rem}
.why{color:#9aa0a6;font-size:.75rem;margin-top:6px}.mc{font-size:.65rem;white-space:pre-wrap;color:#bdc1c6;margin:6px 0 0;max-height:7em;overflow:auto}
.fail{outline:1px solid #c5221f}.miss{color:#fca5a5;padding:30px 0;text-align:center}
</style></head><body>
<h1>Few-shot gallery · Roomba sophisticated alternatives</h1>
<p class='sub'>The five frozen baseline columns are unchanged. The added blue column renders a translating S-curve approach followed by a cue-specific Roomba gesture; use it to decide whether to replace the canonical Roomba few-shots.</p>
<table><thead><tr><th>Cue</th><th>Humanoid</th><th>Tiago</th><th>Stretch</th><th>Single Arm</th><th>Roomba</th><th>Roomba · Sophisticated</th></tr></thead>
<tbody>"""
        + "".join(rows)
        + "</tbody></table></body></html>",
        encoding="utf-8",
    )
    print(GALLERY)


if __name__ == "__main__":
    main()
