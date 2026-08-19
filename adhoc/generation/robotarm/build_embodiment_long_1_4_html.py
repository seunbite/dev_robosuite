#!/usr/bin/env python3
"""Build a 30 cue × 5 embodiment × 5 level render/config review page."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
CONFIG_ROOT = _REPO / "data/results/motion_configs/embodiment_long_1_4"
RENDER_ROOT = _REPO / "data/results/render/embodiment_long_1_4"
GT_RENDER_ROOT = _REPO / "data/results/render/embodiment_motion_code_gt"
OUT_HTML = _REPO / "data/results/html/embodiment_long_1_4.html"
LEVELS = ("core", "long1", "long2", "long3", "long4")
PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "quadruped", "roomba")
PLATFORM_LABELS = {
    "humanoid": "Humanoid",
    "tiago": "Tiago",
    "single_arm": "Single-arm",
    "stretch": "Stretch",
    "quadruped": "Quadruped",
    "roomba": "Roomba",
}


def _gif_path(level: str, platform: str, idx: int, cue: str) -> Path:
    root = GT_RENDER_ROOT if level == "core" else RENDER_ROOT / level
    if platform == "single_arm":
        folder = root / "manipulator/gif/IIWA"
    else:
        folder = root / platform / "gif"
    return folder / f"emb_g{idx:02d}_{cue}.gif"


def _load() -> dict[str, list[dict[str, Any]]]:
    return {
        level: json.loads(
            (CONFIG_ROOT / level / "embodiment_motion_code_all.json").read_text(
                encoding="utf-8"
            )
        )
        for level in LEVELS
    }


def build() -> Path:
    data = _load()
    sections = []
    missing = []
    for cue_index, core_entry in enumerate(data["core"]):
        idx, cue = int(core_entry["idx"]), str(core_entry["cue"])
        rows = []
        for platform in PLATFORMS:
            cells = []
            for level in LEVELS:
                cfg = data[level][cue_index]["platforms"][platform]
                gif = _gif_path(level, platform, idx, cue)
                if gif.is_file():
                    rel = os.path.relpath(gif, OUT_HTML.parent)
                    media = (
                        f'<img src="{html.escape(rel)}" loading="lazy" '
                        f'alt="{html.escape(platform + " " + level + " " + cue)}">'
                    )
                else:
                    missing.append((level, platform, idx, cue))
                    media = '<div class="missing">render pending</div>'
                component_range = cfg["main_component_range"]
                added = cfg.get("added_submotions") or []
                added_text = "; ".join(
                    f"{item['position']}: {item['description']}" for item in added
                ) or "none"
                config_text = json.dumps(
                    cfg["movements"], indent=2, ensure_ascii=False
                )
                cells.append(
                    f"""
                    <td data-level="{level}">
                      <h4>{level.upper()}</h4>
                      {media}
                      <p class="range">Core {component_range['start']}–{component_range['end']} · added {len(added)}</p>
                      <p class="desc">{html.escape(added_text)}</p>
                      <details><summary>Motion Code</summary><pre>{html.escape(cfg.get('source_code', ''))}</pre></details>
                      <details><summary>JSON config</summary><pre>{html.escape(config_text)}</pre></details>
                    </td>
                    """
                )
            rows.append(
                f"<tr><th>{html.escape(PLATFORM_LABELS[platform])}</th>{''.join(cells)}</tr>"
            )
        sections.append(
            f"""
            <section class="cue" id="cue-{idx:02d}">
              <h2>{idx:02d}. {html.escape(cue)}</h2>
              <p class="category">{html.escape(str(core_entry.get('category', '')))}</p>
              <div class="table-wrap">
                <table>
                  <thead><tr><th>Embodiment</th>{''.join(f'<th>{level.upper()}</th>' for level in LEVELS)}</tr></thead>
                  <tbody>{''.join(rows)}</tbody>
                </table>
              </div>
            </section>
            """
        )

    nav = " ".join(
        f'<a href="#cue-{int(entry["idx"]):02d}">{int(entry["idx"]):02d}</a>'
        for entry in data["core"]
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Embodiment Core–Long4 review</title>
  <style>
    :root {{ color-scheme: light dark; font-family: ui-sans-serif,system-ui,sans-serif; }}
    body {{ margin:0; padding:24px; background:#111318; color:#e9edf5; }}
    header,.cue {{ max-width:1800px; margin:0 auto 24px; }}
    header {{ position:sticky; top:0; z-index:5; background:#111318; padding:10px 0; border-bottom:1px solid #343946; }}
    nav {{ display:flex; flex-wrap:wrap; gap:7px; }}
    nav a {{ color:#b9c9ff; text-decoration:none; padding:3px 6px; border:1px solid #343946; border-radius:4px; }}
    .summary {{ color:#b7bfce; }}
    .cue {{ border-top:1px solid #343946; padding-top:18px; }}
    h2 {{ margin-bottom:2px; }} .category {{ margin-top:0; color:#9da7b9; }}
    .table-wrap {{ overflow-x:auto; }}
    table {{ width:100%; min-width:1500px; border-collapse:collapse; table-layout:fixed; }}
    th,td {{ border:1px solid #343946; padding:9px; vertical-align:top; }}
    thead th,tbody th {{ background:#1b1f28; }}
    tbody th {{ width:110px; position:sticky; left:0; z-index:2; }}
    td {{ width:260px; background:#161920; }}
    h4 {{ margin:0 0 7px; color:#b9c9ff; }}
    img {{ width:100%; border-radius:6px; background:#fff; }}
    .missing {{ height:160px; display:grid; place-items:center; background:#242934; color:#d7a4a4; border-radius:6px; }}
    .range {{ color:#aab9e8; font-size:12px; }}
    .desc {{ min-height:48px; color:#c9cfda; font-size:12px; line-height:1.4; }}
    details {{ margin-top:7px; }} summary {{ cursor:pointer; font-size:12px; }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:#0d0f14; padding:8px; font-size:10px; max-height:300px; overflow:auto; }}
  </style>
</head>
<body>
  <header>
    <h1>30 cues × 5 embodiments × Core–Long4</h1>
    <p class="summary">750 configs · render-ready review · missing renders: {len(missing)}</p>
    <nav>{nav}</nav>
  </header>
  {''.join(sections)}
</body>
</html>"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(page, encoding="utf-8")
    print("HTML", OUT_HTML)
    print("missing", len(missing))
    return OUT_HTML


if __name__ == "__main__":
    build()
