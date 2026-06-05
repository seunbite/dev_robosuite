#!/usr/bin/env python3
"""Capture-friendly HTML: grid-6 / grid-12 images + VLM prompts (no API results)."""
from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from verify_pose_multitile_gt_gemini import _grid_prompt  # noqa: E402
from verify_pose_pairwise_12_gemini import _configs_by_cue  # noqa: E402

IN_JSON = _REPO / "data/results/verify/pilot20_pose_multitile_gt_gemini.json"
OUT_HTML = _REPO / "data/results/html/manipulator/pose_multitile_gt_capture.html"


def _esc(x: object) -> str:
    return html.escape(str(x) if x is not None else "")


def _rel(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        return os.path.relpath(p, OUT_HTML.parent)
    except ValueError:
        return str(p)


def _tile_labels(tiles: list[dict]) -> list[str]:
    return [
        f"#{t['display_index']}: dir={t['dir']}, grip={t['gripper_orientation']}"
        for t in tiles
    ]


def _prompt_for_row(row: dict, description: str) -> str:
    labels = _tile_labels(row.get("tiles") or [])
    return _grid_prompt(
        cue=row["cue"],
        description=description,
        n_tiles=int(row["grid_n"]),
        cols=int(row["grid_cols"]),
        rows=int(row["grid_rows"]),
        tile_labels=labels,
    )


def main() -> None:
    data = json.loads(IN_JSON.read_text(encoding="utf-8"))
    cfg_by_cue = _configs_by_cue()
    by_cue: dict[str, list[dict]] = {}
    for r in data.get("results", []):
        if r.get("error"):
            continue
        by_cue.setdefault(r["cue"], []).append(r)

    nav: list[str] = []
    sections: list[str] = []

    for cue in sorted(by_cue, key=lambda c: int(by_cue[c][0].get("cue_idx", 0))):
        rows = sorted(by_cue[cue], key=lambda x: x.get("grid_n", 0))
        cue_idx = rows[0].get("cue_idx")
        sid = f"c{cue_idx}_{cue}"
        nav.append(f'<a href="#{sid}">c{cue_idx} {_esc(cue)}</a>')

        cfg = cfg_by_cue.get(cue, {})
        description = cfg.get("description", "")

        grid_blocks: list[str] = []
        for r in rows:
            n = int(r["grid_n"])
            img = _rel(r.get("grid_image"))
            gt_idx = r.get("gt_indices") or []
            tiles_html = "".join(
                f'<li class="{"gt" if t.get("is_gt") else ""}">'
                f'#{t["display_index"]}: dir={_esc(t["dir"])}, grip={_esc(t["gripper_orientation"])}'
                f'{" [GT]" if t.get("is_gt") else ""}</li>'
                for t in r.get("tiles", [])
            )
            prompt = _prompt_for_row(r, description)
            layout = f'{r.get("grid_cols")}×{r.get("grid_rows")}'
            grid_blocks.append(
                f'<div class="grid-block">'
                f'<h3>Grid {n} ({layout})</h3>'
                f'<p class="meta">GT tile index: <code>{_esc(gt_idx)}</code> · random baseline {100/n:.1f}%</p>'
                f'<figure class="shot"><img src="{_esc(img)}" alt="grid {n} { _esc(cue) }"/></figure>'
                f'<details open><summary>Tile labels</summary><ul class="tiles">{tiles_html}</ul></details>'
                f'<details open><summary>Prompt</summary><pre class="prompt">{_esc(prompt)}</pre></details>'
                f"</div>"
            )

        sections.append(
            f'<section id="{sid}" class="cue">'
            f'<header><h2>c{cue_idx} <code>{_esc(cue)}</code></h2>'
            f'<p class="gt-line">Human GT: {_esc(rows[0].get("groundtruth", ""))}</p>'
            f'<p class="desc">{_esc(description)}</p></header>'
            f'<div class="grids">{"".join(grid_blocks)}</div>'
            f"</section>"
        )

    css = """
:root { --bg:#fff; --border:#d8dee9; --text:#1a1a2e; --muted:#64748b; --gt:#166534; }
* { box-sizing: border-box; }
body { font-family: system-ui, -apple-system, sans-serif; margin: 0; background: var(--bg); color: var(--text); line-height: 1.45; }
.top { position: sticky; top: 0; z-index: 2; background: #fff; border-bottom: 1px solid var(--border); padding: 16px 20px; }
.top h1 { margin: 0 0 8px; font-size: 1.25rem; }
.top p { margin: 0; color: var(--muted); font-size: .9rem; }
.nav { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.nav a { font-size: .78rem; padding: 4px 8px; border: 1px solid var(--border); border-radius: 999px; text-decoration: none; color: #334; background: #f8fafc; }
.cue { padding: 28px 20px; border-bottom: 2px solid #eef2f7; page-break-inside: avoid; }
.cue h2 { margin: 0 0 6px; font-size: 1.1rem; }
.gt-line, .desc { margin: 4px 0; font-size: .88rem; color: var(--muted); }
.grids { display: grid; grid-template-columns: 1fr; gap: 24px; margin-top: 16px; }
@media (min-width: 1100px) { .grids { grid-template-columns: 1fr 1fr; } }
.grid-block { border: 1px solid var(--border); border-radius: 12px; padding: 14px; background: #fafbfd; }
.grid-block h3 { margin: 0 0 8px; font-size: 1rem; }
.meta { margin: 0 0 10px; font-size: .82rem; color: var(--muted); }
.shot { margin: 0 0 12px; }
.shot img { width: 100%; max-width: 640px; border: 1px solid #cbd5e1; border-radius: 8px; background: #fff; display: block; }
details { margin-top: 10px; }
summary { cursor: pointer; font-weight: 600; font-size: .85rem; margin-bottom: 6px; }
.tiles { margin: 6px 0 0; padding-left: 18px; font-size: .8rem; font-family: ui-monospace, Menlo, monospace; }
.tiles li.gt { color: var(--gt); font-weight: 600; }
.prompt { white-space: pre-wrap; font-size: .74rem; line-height: 1.4; background: #fff; border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin: 0; max-height: 420px; overflow: auto; }
@media print { .top { position: static; } .nav { display: none; } }
"""

    doc = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pose multitile GT — capture pack (grid 6 / 12)</title>
<style>{css}</style>
</head><body>
<div class="top">
  <h1>Pose multitile GT — capture pack</h1>
  <p>20 cues · grid 6 (3×2) and grid 12 (4×3) · input images + prompts for interactive eval</p>
  <nav class="nav">{''.join(nav)}</nav>
</div>
{''.join(sections)}
</body></html>"""

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(doc, encoding="utf-8")
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
