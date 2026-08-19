#!/usr/bin/env python3
"""Build cue-ID pairwise HTML: GT-made vs Gemini-made, judged by Gemini 2.5 Pro."""
from __future__ import annotations

import argparse
import html as H
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

PLATS = ["humanoid", "tiago", "single_arm", "stretch", "roomba"]
PLAT_LABEL = {
    "humanoid": "Humanoid",
    "tiago": "Tiago",
    "single_arm": "Single Arm",
    "stretch": "Stretch",
    "roomba": "Roomba",
}
PLAT_GIF = {
    "humanoid": ("humanoid", "gif"),
    "tiago": ("tiago", "gif"),
    "single_arm": ("manipulator", "gif", "IIWA"),
    "stretch": ("stretch", "gif"),
    "roomba": ("roomba", "gif"),
}

# Content / render fixes shown as (Fixed) next to the matching source label.
FIXED_MEDIA = {
    ("gem", "humanoid", 13),
    ("gem", "humanoid", 15),
    ("gem", "humanoid", 30),
    ("gt", "tiago", 3),
    # 2026-08-03 prior GT edits (kept)
    ("gt", "tiago", 7),
    ("gt", "roomba", 7),
    ("gt", "roomba", 13),  # in-place yaw n=1 bugfix
    ("gt", "roomba", 6),
    # 2026-08-04 GT fill.md sync + sticky mirror / & parallel
    ("gt", "humanoid", 5),
    ("gt", "humanoid", 16),
    ("gt", "humanoid", 17),
    ("gt", "humanoid", 18),
    ("gt", "humanoid", 22),
    ("gt", "humanoid", 28),
    ("gt", "humanoid", 30),
    ("gt", "tiago", 5),
    ("gt", "tiago", 11),  # beckon: wrist-x invisible @ full-body; EE path
    ("gt", "tiago", 16),
    ("gt", "tiago", 18),
    ("gt", "tiago", 20),
    ("gt", "tiago", 22),
    ("gt", "tiago", 27),
    ("gt", "tiago", 28),
    ("gt", "tiago", 30),
    ("gt", "single_arm", 2),
    ("gt", "single_arm", 3),
    ("gt", "single_arm", 5),
    ("gt", "single_arm", 7),
    ("gt", "single_arm", 8),
    ("gt", "single_arm", 11),
    ("gt", "single_arm", 12),
    ("gt", "single_arm", 15),
    ("gt", "single_arm", 18),
    ("gt", "single_arm", 23),
    ("gt", "single_arm", 27),
    ("gt", "single_arm", 28),
    ("gt", "single_arm", 29),
    ("gt", "stretch", 16),
    ("gt", "stretch", 18),  # wrist z was yaw (sideways)
    ("gt", "stretch", 19),
    ("gt", "stretch", 21),
    ("gt", "stretch", 23),
    ("gt", "stretch", 24),
    ("gt", "stretch", 26),
    ("gt", "stretch", 28),
    ("gt", "stretch", 30),
    ("gt", "roomba", 1),  # ±line: spd=linear speed, time∝rep·distance
    ("gt", "roomba", 5),
    ("gt", "roomba", 11),
    ("gt", "roomba", 12),
    ("gt", "roomba", 16),
    ("gt", "roomba", 19),
    ("gt", "roomba", 22),
    ("gt", "roomba", 23),
    ("gt", "roomba", 27),
    ("gt", "roomba", 28),  # 2026-08-04 fill: x=+0.30→+0.70
    ("gt", "roomba", 29),  # 2026-08-04 fill: wobble→full 360
    ("gt", "roomba", 30),
    ("gt", "single_arm", 28),  # 2026-08-04 fill: lion path rewrite
    ("gt", "single_arm", 30),  # 2026-08-04 fill: bilateral flap
    ("gt", "stretch", 28),  # 2026-08-04 fill: longer base lunge
    ("gt", "stretch", 30),  # 2026-08-04 fill: base arc 360→720
    # 2026-08-04 signed EE arc / downward bowl ∪
    ("gt", "humanoid", 27),
    ("gt", "tiago", 27),
    ("gt", "single_arm", 27),
    ("gt", "stretch", 27),
}
# 2026-08-04 evening: path spd∝distance / time∝rep re-render + fill sync
_REP2_TIMING = [
    ("humanoid", 1), ("humanoid", 2), ("humanoid", 3), ("humanoid", 5),
    ("humanoid", 8), ("humanoid", 11), ("humanoid", 18), ("humanoid", 29),
    ("humanoid", 30),
    ("tiago", 1), ("tiago", 2), ("tiago", 3), ("tiago", 8), ("tiago", 11),
    ("tiago", 18), ("tiago", 29), ("tiago", 30),
    ("single_arm", 1), ("single_arm", 2), ("single_arm", 3), ("single_arm", 8),
    ("single_arm", 11), ("single_arm", 18), ("single_arm", 27), ("single_arm", 29),
    ("single_arm", 30),
    ("stretch", 1), ("stretch", 2), ("stretch", 3), ("stretch", 8),
    ("stretch", 11), ("stretch", 18), ("stretch", 19), ("stretch", 30),
    ("quadruped", 3), ("quadruped", 11), ("quadruped", 30),
    ("roomba", 1), ("roomba", 2), ("roomba", 3), ("roomba", 7), ("roomba", 8),
    ("roomba", 11), ("roomba", 12), ("roomba", 16), ("roomba", 17),
    ("roomba", 18), ("roomba", 22), ("roomba", 23), ("roomba", 29),
]
FIXED_MEDIA.update({("gt", p, i) for p, i in _REP2_TIMING})
_GT_FILL_NOTE = "GT Motion Code fill edit (2026-08-04)"
_REP2_NOTE = "path timing: spd=linear speed, time∝rep·distance (re-render)"
FIXED_NOTE = {
    ("gem", "humanoid", 13): "pose bank 548+729 removed → 250",
    ("gem", "humanoid", 15): "pose bank 1116+1126 removed → 525",
    ("gem", "humanoid", 30): "pose bank 251 removed → 518",
    ("gt", "tiago", 3): "rangefinder debug rays off",
    ("gt", "tiago", 7): "categorical torso height lerp (was snap→0)",
    ("gt", "roomba", 7): "fill edit + path timing re-render",
    ("gt", "roomba", 13): "in-place yaw min frames (45° visible)",
    ("gt", "roomba", 6): "in-place yaw min frames",
    ("gt", "humanoid", 16): "sticky left_arm=mirror on movement",
    ("gt", "humanoid", 18): "sticky left_arm=mirror on movement",
    ("gt", "humanoid", 22): "sticky left_arm=mirror on movement",
    ("gt", "humanoid", 30): "sticky left_arm=mirror on movement",
    ("gt", "tiago", 11): "wrist-x invisible @ full-body cam; elbow curl beckon",
    ("gt", "stretch", 18): "wrist z→pitch (was yaw/sideways)",
    ("gt", "roomba", 1): "±line: spd=linear speed, time∝rep·distance",
    ("gt", "tiago", 16): "sticky left_arm=mirror on movement",
    ("gt", "tiago", 18): "sticky left_arm=mirror on movement",
    ("gt", "tiago", 20): "mirror + EE path both arms",
    ("gt", "tiago", 22): "left_arm=still→mirror + path both arms",
    ("gt", "tiago", 27): "sticky left_arm=mirror on path",
    ("gt", "tiago", 30): "sticky left_arm=mirror on movement",
    ("gt", "humanoid", 28): "path & torso parallel (&)",
    ("gt", "stretch", 28): "ee & base parallel (&)",
    ("gt", "single_arm", 29): "tiny arc r=0.01 had 2 frames (closed circle frozen)",
    ("gt", "single_arm", 8): "±line path started at endpoint (snap); axis=7→z, rep=3",
    ("gt", "humanoid", 27): "fill edit: bowl sweep + approx-prompt rejudge",
    ("gt", "roomba", 28): "fill edit: longer charge (x=+0.70)",
    ("gt", "roomba", 29): "fill edit: small wobble→full circle",
    ("gt", "single_arm", 28): "fill edit: lion line+xy arc",
    ("gt", "single_arm", 30): "fill edit: bilateral wing flap",
    ("gt", "stretch", 28): "fill edit: longer base lunge",
    ("gt", "stretch", 30): "fill edit: base arc 720°",
    ("gt", "humanoid", 27): "signed sweep −180 bowl ∪ (no cw/ccw)",
    ("gt", "tiago", 27): "signed sweep −180 bowl ∪ (no cw/ccw)",
    ("gt", "single_arm", 27): "signed sweep −180 bowl ∪ (no cw/ccw)",
    ("gt", "stretch", 27): "signed sweep −180 bowl ∪ (no cw/ccw)",
}
# Default notes: fill edits keep fill note; new timing-batch cells get timing note.
_FILL_BATCH = {
    ("gt", "humanoid", 5), ("gt", "humanoid", 29),
    ("gt", "tiago", 29),
    ("gt", "single_arm", 27), ("gt", "single_arm", 29),
    ("gt", "roomba", 7), ("gt", "roomba", 8), ("gt", "roomba", 11),
    ("gt", "roomba", 12), ("gt", "roomba", 22), ("gt", "roomba", 23),
}
for _k in FIXED_MEDIA:
    if _k[0] != "gt" or _k in FIXED_NOTE:
        continue
    if _k in _FILL_BATCH:
        FIXED_NOTE[_k] = _GT_FILL_NOTE
    elif (_k[1], _k[2]) in {(p, i) for p, i in _REP2_TIMING}:
        FIXED_NOTE[_k] = _REP2_NOTE
    else:
        FIXED_NOTE[_k] = _GT_FILL_NOTE


def _load_preds(path: Path) -> dict[tuple[int, str, str], dict]:
    out: dict[tuple[int, str, str], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (
            int(r["cue_idx"]),
            str(r["platform"]),
            str(r.get("option_order", "")).upper(),
        )
        out[key] = r
    return out


def _gif_path(source: str, platform: str, idx: int, cue: str) -> Path:
    root = REPO / (
        "data/results/render/embodiment_motion_code_gt"
        if source == "gt"
        else "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro"
    )
    return root.joinpath(*PLAT_GIF[platform]) / f"emb_g{idx:02d}_{cue}.gif"


def _load_source_codes(path: Path) -> dict[tuple[int, str], str]:
    """(cue_idx, platform) → Motion Code source_code string."""
    out: dict[tuple[int, str], str] = {}
    if not path.is_file():
        return out
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in data:
        idx = int(entry["idx"])
        for plat, cfg in (entry.get("platforms") or {}).items():
            code = str((cfg or {}).get("source_code") or "").strip()
            if code:
                out[(idx, str(plat))] = code
    return out


def build(
    *,
    pred_root: Path,
    out: Path,
    foil_map: Path,
    gt_json: Path | None = None,
    gem_json: Path | None = None,
    open_browser: bool = True,
) -> Path:
    html_dir = out.parent.resolve()
    gt_by = _load_preds(pred_root / "gt_pairwise" / "predictions.jsonl")
    gem_by = _load_preds(pred_root / "agent_gemini25pro_pairwise" / "predictions.jsonl")
    foil_all = json.loads(foil_map.read_text(encoding="utf-8"))["platforms"]
    foil_by = {
        plat: {int(x["target_idx"]): x for x in rows}
        for plat, rows in foil_all.items()
    }
    gt_codes = _load_source_codes(
        gt_json
        or (REPO / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json")
    )
    gem_codes = _load_source_codes(
        gem_json
        or (
            REPO
            / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro"
            / "embodiment_motion_code_all.json"
        )
    )

    cue_meta: dict[int, dict[str, str]] = {}
    for r in list(gt_by.values()) + list(gem_by.values()):
        cue_meta[int(r["cue_idx"])] = {
            "cue": r["cue"],
            "category": r.get("category") or "",
        }

    cells: dict[tuple[int, str], dict] = {}
    n_ok_gt = n_ok_gem = n_judged = 0
    for k in set(gt_by) | set(gem_by):
        idx, plat, ord_ = k
        if plat not in PLATS:
            continue
        cell = cells.setdefault((idx, plat), {"orders": {}, "mode": "judged"})
        cell["orders"][ord_] = {"gt": gt_by.get(k), "gem": gem_by.get(k)}
        row = gt_by.get(k) or gem_by.get(k) or {}
        cell["cue"] = row.get("cue") or cue_meta.get(idx, {}).get("cue", "?")
        cell["foil"] = (
            row.get("foil_cue")
            or foil_by.get(plat, {}).get(idx, {}).get("foil", "?")
        )
        cell["category"] = row.get("category") or cue_meta.get(idx, {}).get("category", "")
        if k in gt_by and k in gem_by:
            n_ok_gt += int(bool(gt_by[k].get("correct")))
            n_ok_gem += int(bool(gem_by[k].get("correct")))
            n_judged += 1

    # Ensure stretch cells exist even if judge rows missing.
    for idx, meta in cue_meta.items():
        for plat in PLATS:
            if (idx, plat) in cells:
                continue
            f = foil_by.get(plat, {}).get(idx, {})
            cells[(idx, plat)] = {
                "mode": "media",
                "cue": meta["cue"],
                "foil": f.get("foil", "?"),
                "category": meta["category"],
                "orders": {},
            }

    acc_gt = 100 * n_ok_gt / n_judged if n_judged else 0.0
    acc_gem = 100 * n_ok_gem / n_judged if n_judged else 0.0

    def media_url(p: Path) -> str | None:
        if not p.exists():
            return None
        return (
            f"{Path(os.path.relpath(p.resolve(), html_dir)).as_posix()}"
            f"?v={int(p.stat().st_mtime)}"
        )

    def ord_chip(ord_: str, row, mode: str) -> str:
        if mode == "media" or row is None:
            return f'<span class="ochip na"><b>{H.escape(ord_)}</b></span>'
        ok = bool(row.get("correct"))
        cls = "ok" if ok else "bad"
        mark = "✓" if ok else "✗"
        return f'<span class="ochip {cls}"><b>{H.escape(ord_)}</b> {mark}</span>'

    def source_block(label: str, ab_row, ba_row, mode: str) -> str:
        return (
            f'<div class="src">'
            f'<div class="src-lab">{H.escape(label)}</div>'
            f'<div class="ochips">'
            f'{ord_chip("AB", ab_row, mode)}{ord_chip("BA", ba_row, mode)}'
            f"</div></div>"
        )

    def card(idx: int, plat: str, cell: dict) -> str:
        orders = cell.get("orders") or {}
        ab = orders.get("AB", {})
        ba = orders.get("BA", {})
        mode = "judged" if (ab.get("gt") or ab.get("gem") or ba.get("gt") or ba.get("gem")) else "media"
        cue = cell["cue"]
        foil = cell.get("foil") or "?"
        badges = (
            f'<div class="badges">'
            f'{source_block("GT", ab.get("gt"), ba.get("gt"), mode)}'
            f'{source_block("Gemini", ab.get("gem"), ba.get("gem"), mode)}'
            f"</div>"
        )
        meta = (
            f'<div class="qmeta">'
            f'<span><span class="k">GT</span> = <b>{H.escape(cue)}</b></span>'
            f'<span><span class="k">Neg</span> = <b>{H.escape(str(foil))}</b></span>'
            f"</div>"
        )
        url_h = media_url(_gif_path("gt", plat, idx, cue))
        url_g = media_url(_gif_path("gem", plat, idx, cue))

        def col(url, title, src_key: str) -> str:
            fixed = (src_key, plat, idx) in FIXED_MEDIA
            fix_html = (
                f' <span class="fixed" title="{H.escape(FIXED_NOTE.get((src_key, plat, idx), "re-rendered"))}">(Fixed)</span>'
                if fixed
                else ""
            )
            lab = f'<div class="lab{" fixed-lab" if fixed else ""}">{H.escape(title)}{fix_html}</div>'
            img = (
                f'<img src="{H.escape(url)}" loading="lazy" alt="{H.escape(title)}">'
                if url
                else '<div class="miss">missing</div>'
            )
            code_map = gt_codes if src_key == "gt" else gem_codes
            code = code_map.get((idx, plat), "")
            code_html = (
                f'<pre class="mc" title="Motion Code">{H.escape(code)}</pre>'
                if code
                else '<pre class="mc miss-code">(no motion code)</pre>'
            )
            return f'<div class="col{" fixed-col" if fixed else ""}">{lab}{img}{code_html}</div>'

        duo = (
            f'<div class="duo">'
            f'{col(url_h, "Human GT-made", "gt")}'
            f'{col(url_g, "Gemini-made", "gem")}'
            f"</div>"
        )
        return f'<article class="card">{badges}{meta}{duo}</article>'

    parts = [
        """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cue-ID pairwise · Gemini 2.5 Pro judge</title>
<style>
:root { --bg:#14120e; --ink:#f0e8d8; --muted:#9a907c; --line:#3a3428; --card:#1c1914;
  --ok:#9dcc8a; --bad:#e8a09a; --head:#2a241a; --fix:#7eb8e8; }
* { box-sizing:border-box; }
body { margin:0; color:var(--ink); font-family:"Source Serif 4","Iowan Old Style",Palatino,Georgia,serif;
  background: linear-gradient(180deg, #1a2218 0%, transparent 28%), radial-gradient(900px 420px at 100% 0%, #243040 0%, transparent 55%), var(--bg); }
header { padding:28px 20px 12px; }
h1 { margin:0; font-size:1.65rem; letter-spacing:-0.02em; }
.lead { margin:8px 0 0; color:var(--muted); max-width:95ch; line-height:1.45; font-size:0.95rem; }
.jump { display:flex; flex-wrap:wrap; gap:6px; padding:12px 16px 14px; position:sticky; top:0; z-index:5;
  background:rgba(20,18,14,0.92); backdrop-filter:blur(8px); border-bottom:1px solid var(--line);
  font-family:ui-sans-serif,system-ui,sans-serif; }
.jump a { color:var(--ink); text-decoration:none; border:1px solid var(--line); background:var(--card); padding:4px 8px; border-radius:8px; font-size:0.75rem; }
.jump a:hover { border-color:#8a7a55; }
.jump a.has-fix { border-color:#4a6a8a; color:var(--fix); }
section.cue { padding:6px 12px 28px; border-bottom:1px solid var(--line); }
section.cue h2 { margin:16px 8px 12px; font-size:1.15rem; display:flex; flex-wrap:wrap; gap:10px; align-items:baseline; }
section.cue h2 .idx { color:#c4b28a; font-family:ui-sans-serif,system-ui,sans-serif; }
section.cue h2 .cat { color:var(--muted); font-size:0.8rem; font-weight:500; font-family:ui-sans-serif,system-ui,sans-serif; }
section.cue h2 .fix-tag { font-family:ui-sans-serif,system-ui,sans-serif; font-size:0.7rem; color:var(--fix);
  border:1px solid #4a6a8a; background:#1a2838; padding:2px 7px; border-radius:999px; font-weight:700; }
.grid5 { display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:10px; align-items:start; }
.plat-col { min-width:0; }
.plat-head { position:sticky; top:48px; z-index:2; background:var(--head); border:1px solid var(--line); border-radius:10px;
  padding:8px 10px; margin-bottom:8px; font-family:ui-sans-serif,system-ui,sans-serif; font-size:0.75rem; font-weight:700;
  letter-spacing:0.04em; text-transform:uppercase; color:#d8c9a4; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:8px; }
.badges { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-bottom:8px; font-family:ui-sans-serif,system-ui,sans-serif; }
.src { border:1px solid var(--line); border-radius:8px; padding:6px 4px; background:#221e18; }
.src-lab { text-align:center; font-size:0.62rem; letter-spacing:0.05em; text-transform:uppercase; color:var(--muted); margin-bottom:4px; font-weight:700; }
.ochips { display:flex; gap:4px; }
.ochip { flex:1; text-align:center; padding:5px 2px; border-radius:6px; border:1px solid var(--line); font-size:0.68rem; background:#1a1712; color:var(--muted); }
.ochip b { letter-spacing:0.04em; }
.ochip.ok { background:#1e3a28; border-color:#3d6a4a; color:var(--ok); font-weight:700; }
.ochip.bad { background:#3a1e1e; border-color:#6a3a3a; color:var(--bad); font-weight:700; }
.ochip.na { opacity:0.65; }
.qmeta { display:flex; flex-direction:column; gap:2px; margin-bottom:8px; font-family:ui-sans-serif,system-ui,sans-serif; font-size:0.7rem; line-height:1.3; color:#cfc3a8; }
.qmeta .k { color:var(--muted); font-size:0.62rem; text-transform:uppercase; letter-spacing:0.04em; }
.qmeta b { color:var(--ink); }
.duo { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
.lab { font-family:ui-sans-serif,system-ui,sans-serif; font-size:0.62rem; line-height:1.25; margin-bottom:4px; color:var(--muted); text-transform:uppercase; letter-spacing:0.04em; text-align:center; }
.lab .fixed { color:var(--fix); text-transform:none; letter-spacing:0; font-weight:800; }
.fixed-lab { color:#c5d8ea; }
.fixed-col { outline:1px solid #3a5a7a; border-radius:8px; padding:4px; background:#182028; }
img { width:100%; aspect-ratio:1; object-fit:cover; border-radius:6px; background:#000; display:block; }
.miss { aspect-ratio:1; display:grid; place-items:center; background:#222; color:#866; border-radius:6px; font-size:0.7rem; }
.mc { margin:6px 0 0; padding:6px 7px; max-height:7.2em; overflow:auto; border-radius:6px; border:1px solid var(--line);
  background:#0e0c0a; color:#d7cbb0; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:0.58rem; line-height:1.35; white-space:pre-wrap; word-break:break-word; }
.mc.miss-code { color:#866; font-style:italic; }
.fix-legend { margin-top:10px; font-family:ui-sans-serif,system-ui,sans-serif; font-size:0.8rem; color:#b8cce0; }
.fix-legend code { color:var(--fix); font-weight:700; }
@media (max-width:1200px) { .grid5 { grid-template-columns:repeat(3, minmax(0, 1fr)); } }
@media (max-width:800px) { .grid5 { grid-template-columns:repeat(2, minmax(0, 1fr)); } }
</style></head><body>
"""
    ]
    fix_list = "; ".join(
        f"{src.upper()}·{plat}·{idx:02d} ({FIXED_NOTE[(src, plat, idx)]})"
        for src, plat, idx in sorted(FIXED_MEDIA)
    )
    parts.append(
        f"""<header>
  <h1>Cue-ID pairwise · Gemini 2.5 Pro judge</h1>
  <p class="lead">
    Cue 01→30 · humanoid / tiago / single_arm / stretch / roomba.
    Badges: <b>GT-vid</b> / <b>Gemini-vid</b>, each with AB·BA (green=✓, red=✗) — judge = Gemini 2.5 Pro, easy-neg 2AFC.
    GIFs: <b>Human GT-made</b> | <b>Gemini-made</b>, each with its <b>Motion Code</b> under the clip.
    Judged N={n_judged}: GT-vid <b>{acc_gt:.1f}%</b> · Gem-vid <b>{acc_gem:.1f}%</b>.
  </p>
  <p class="fix-legend">Re-rendered / content fixes marked <code>(Fixed)</code>: {H.escape(fix_list)}</p>
</header>
<nav class="jump">"""
    )
    cues_with_fix = {idx for _, _, idx in FIXED_MEDIA}
    for idx in sorted(cue_meta):
        cls = ' class="has-fix"' if idx in cues_with_fix else ""
        parts.append(f'<a href="#c{idx:02d}"{cls}>{idx:02d}</a>')
    parts.append("</nav>\n")

    for idx in sorted(cue_meta):
        fix_tag = (
            ' <span class="fix-tag">Fixed media</span>' if idx in cues_with_fix else ""
        )
        parts.append(
            f'<section class="cue" id="c{idx:02d}">\n'
            f'  <h2><span class="idx">{idx:02d}</span> {H.escape(cue_meta[idx]["cue"])}'
            f' <span class="cat">{H.escape(cue_meta[idx]["category"])}</span>{fix_tag}</h2>\n'
            f'  <div class="grid5">\n'
        )
        for plat in PLATS:
            cell = cells.get((idx, plat))
            parts.append(
                f'<div class="plat-col"><div class="plat-head">{PLAT_LABEL[plat]}</div>'
            )
            parts.append(
                card(idx, plat, cell)
                if cell
                else '<div class="miss" style="aspect-ratio:auto;padding:16px">—</div>'
            )
            parts.append("</div>")
        parts.append("  </div>\n</section>\n")
    parts.append("</body></html>\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {out}  judged N={n_judged} GT={acc_gt:.1f}% Gem={acc_gem:.1f}%")
    if open_browser:
        subprocess.run(["open", str(out)], check=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pred-root",
        type=Path,
        default=REPO
        / "data/results/evaluation/embodiment_cue_id/results_gemini25pro_judge_pairwise_20260803",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "data/results/html/cueid_gt_vs_gemini25pro_pairwise_by_cue.html",
    )
    ap.add_argument(
        "--foil-map",
        type=Path,
        default=REPO
        / "data/results/evaluation/embodiment_cue_id/pairwise_easy_foil_map.json",
    )
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument(
        "--gt-json",
        type=Path,
        default=REPO
        / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json",
    )
    ap.add_argument(
        "--gem-json",
        type=Path,
        default=REPO
        / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro"
        / "embodiment_motion_code_all.json",
    )
    args = ap.parse_args()
    build(
        pred_root=args.pred_root,
        out=args.out,
        foil_map=args.foil_map,
        gt_json=args.gt_json,
        gem_json=args.gem_json,
        open_browser=not args.no_open,
    )


if __name__ == "__main__":
    main()
