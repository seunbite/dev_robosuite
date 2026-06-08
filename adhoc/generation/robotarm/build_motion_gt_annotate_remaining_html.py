#!/usr/bin/env python3
"""HTML review page for cues missing movement-component GT annotations."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]

GT_JSON = _REPO / "data/results/verify/pilot40_motion_component_gt.json"
SOURCE_HTML = _REPO / "data/results/html/manipulator/render_manipulator_20260608.html"
POSE_CFG = (
    _REPO
    / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json"
)
OUT_HTML = _REPO / "data/results/html/manipulator/motion_gt_annotate_remaining.html"


def _movement_summary(row: dict[str, Any]) -> str:
    chunks: list[str] = []
    for step in row.get("movements", []):
        t = step.get("type")
        p = step.get("parameters", {})
        if t == "movement":
            joint = p.get("joint", "?")
            rep = p.get("repetition", 1)
            dparts = []
            for d in p.get("directions", []):
                deg = d.get("degrees", {}) or {}
                if deg:
                    dparts.append(",".join(f"{k}:{v}" for k, v in deg.items()))
                ht = d.get("hold_time", 0)
                if ht:
                    dparts.append(f"hold={ht}")
            deg_txt = " | ".join(dparts) if dparts else "-"
            chunks.append(f"movement(joint={joint}, rep={rep}, {deg_txt})")
        elif t == "path":
            shape = p.get("shape", "?")
            axis = p.get("axis") or p.get("plane") or "?"
            ht = p.get("hold_time", 0)
            extra = f", hold={ht}" if ht else ""
            chunks.append(f"path({shape} {axis}{extra})")
        elif t == "pose" and len(chunks) == 0:
            pose = (p.get("pose") or {})
            chunks.append(f"pose({pose.get('dir')},{pose.get('gripper_orientation')})")
    tail = row.get("movements") or []
    tail_only = [s for s in tail if s.get("type") != "pose" or tail.index(s) > 0]
    if not tail_only:
        return "(pose only — no tail)"
    return " → ".join(chunks[1:] if chunks and chunks[0].startswith("pose(") else chunks)


def _parse_source_html(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    articles = text.split("</article>")
    out: list[dict[str, Any]] = []
    for art in articles:
        m = re.search(r"<h3>c(\d+) ([^<]+)</h3>", art)
        if not m:
            continue
        idx, cue = int(m.group(1)), m.group(2).strip()
        img_m = re.search(r'<img src="([^"]+)"', art)
        pre_m = re.search(r"<pre>([\s\S]*?)</pre>", art)
        desc = ""
        if pre_m:
            try:
                row = json.loads(pre_m.group(1))
                desc = row.get("description", "")
            except json.JSONDecodeError:
                pass
        out.append(
            {
                "cue_idx": idx,
                "cue": cue,
                "gif_rel": img_m.group(1) if img_m else "",
                "description": desc,
            }
        )
    return out


def build() -> Path:
    gt = json.loads(GT_JSON.read_text(encoding="utf-8"))
    by_idx = {int(a["cue_idx"]): a for a in gt.get("annotations", [])}
    cfg_by = {int(r["idx"]): r for r in json.loads(POSE_CFG.read_text(encoding="utf-8"))}
    source_rows = _parse_source_html(SOURCE_HTML)

    remaining: list[dict[str, Any]] = []
    for i, row in enumerate(source_rows, 1):
        idx = row["cue_idx"]
        raw = (by_idx.get(idx, {}).get("annotation_raw") or "").strip()
        if raw:
            continue
        cfg = cfg_by.get(idx, {})
        remaining.append(
            {
                "seq": len(remaining) + 1,
                "cue_idx": idx,
                "cue": row["cue"],
                "gif_rel": row["gif_rel"],
                "description": row.get("description") or cfg.get("description", ""),
                "tail_summary": _movement_summary(cfg) if cfg else "(config missing)",
            }
        )

    note = gt.get("groundtruth_note", "")
    cards = []
    for r in remaining:
        cards.append(
            f"""
<article class="card" id="seq{r['seq']}">
  <div class="head">
    <span class="seq">#{r['seq']}</span>
    <h2>c{r['cue_idx']} {r['cue']}</h2>
    <code class="ann-line">{r['cue_idx']} </code>
  </div>
  <p class="desc">{r['description']}</p>
  <p class="tail"><strong>LLM tail:</strong> <code>{r['tail_summary']}</code></p>
  <img src="{r['gif_rel']}" alt="{r['cue']}" loading="lazy"/>
</article>"""
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>Motion GT — remaining {len(remaining)} cues</title>
<style>
body{{font-family:system-ui,sans-serif;background:#f4f6fa;margin:0;padding:20px;max-width:1100px}}
h1{{margin:0 0 8px}}
.summary{{background:#fff;border:1px solid #dce1ea;border-radius:10px;padding:14px;margin-bottom:16px;line-height:1.5}}
.summary code{{background:#eef2ff;padding:2px 6px;border-radius:4px}}
.card{{background:#fff;border:1px solid #dce1ea;border-radius:10px;padding:14px;margin-bottom:14px}}
.head{{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}}
.seq{{background:#1e3a8a;color:#fff;border-radius:6px;padding:2px 8px;font-size:13px;font-weight:700}}
h2{{margin:0;font-size:18px}}
.ann-line{{background:#fff8e1;border:1px dashed #e6b800;padding:4px 8px;border-radius:6px;font-size:14px}}
.desc{{color:#444;font-size:14px;margin:8px 0}}
.tail{{font-size:13px;color:#333;margin:8px 0}}
img{{max-width:100%;border:1px solid #ddd;border-radius:8px;margin-top:8px}}
.legend{{font-size:13px;color:#555}}
</style></head><body>
<h1>Movement GT — annotate remaining ({len(remaining)} cues)</h1>
<div class="summary">
  <p><strong>Order:</strong> same as <code>render_manipulator_20260608.html</code> (pilot90, unannotated only).</p>
  <p><strong>Write annotation</strong> after <code>cue_idx</code> (or copy the yellow <code>ann-line</code> per card).</p>
  <p class="legend">{note}</p>
  <p>Examples: <code>z + non hold</code> · <code>y +- rep</code> · <code>arc yz</code> · <code>x + pause</code> · <code>x + / y + - non hold</code></p>
</div>
{"".join(cards)}
</body></html>"""

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {len(remaining)} cues → {OUT_HTML}")
    return OUT_HTML


if __name__ == "__main__":
    build()
