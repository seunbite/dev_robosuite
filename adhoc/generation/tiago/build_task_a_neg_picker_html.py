#!/usr/bin/env python3
"""Task A — interactive HTML to pick neg poses on 12×25 group grids.

Positive set = USER_SELECTION (plausible set). User clicks tiles → neg.
Export JSON for binary classification (pos / neg).
"""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build_plausible_pose_set import USER_SELECTION  # noqa: E402
from generate_pose_group_tiles import (  # noqa: E402
    DEFAULT_GRID_COLS,
    DEFAULT_HEAD_HAND_GAP_MIN_M,
    ORDER_DIR,
    ORDER_GRIP,
    SOURCE_DIR_FOR_DISPLAY,
    _select_xy_grid_balanced,
)
from legacy.render_mobile_config import (  # noqa: E402
    QI_R_ARM,
    _capture,
    _default_qpos,
    _get_pose_bank_entries,
    _make_env,
    _right_hand_forward_vs_torso_head_ok,
)
from pilot40_paths import HTML_EXP_DIR, save_json  # noqa: E402

DEFAULT_TILE_ROOT = _REPO / "data/results/visualize/tiago/expA_neg_picker"
DEFAULT_MONTAGE_ROOT = _REPO / "data/results/visualize/google_pose_groups_12"
DEFAULT_HTML = HTML_EXP_DIR / "expA_neg_picker.html"
DEFAULT_NEG_JSON = _REPO / "data/seed/tiago/task_a_neg_selection.json"


def _all_group_keys() -> list[str]:
    return [f"{d}_{g}" for d in ORDER_DIR for g in ORDER_GRIP]


def _collect_groups(env, *, grid_side: int, head_hand_gap_min: float) -> dict[str, list[dict[str, Any]]]:
    os.environ["MOBILE_POSE_HEAD_HAND_X_GAP"] = str(head_hand_gap_min)
    bank = _get_pose_bank_entries()
    buckets: dict[tuple[str, str], list[dict]] = {}
    for e in bank:
        src_dir = str(e.get("dir"))
        orient = str(e.get("orient"))
        disp_dir = next((k for k, v in SOURCE_DIR_FOR_DISPLAY.items() if v == src_dir), None)
        if disp_dir not in ORDER_DIR or orient not in ORDER_GRIP:
            continue
        q = _default_qpos(env)
        q[QI_R_ARM] = [float(v) for v in e["r_arm_rad"]]
        env.sim.data.qpos[:] = q
        env.sim.forward()
        if not _right_hand_forward_vs_torso_head_ok(env.sim):
            continue
        buckets.setdefault((disp_dir, orient), []).append(e)

    out: dict[str, list[dict[str, Any]]] = {}
    for d in ORDER_DIR:
        for g in ORDER_GRIP:
            key = f"{d}_{g}"
            reps = _select_xy_grid_balanced(buckets.get((d, g), []), grid_side=grid_side)
            tiles: list[dict[str, Any]] = []
            for i, entry in enumerate(reps, start=1):
                jd = entry.get("joint_deg") or []
                tiles.append(
                    {
                        "tile_idx": i,
                        "x": float(entry.get("x", 50)),
                        "y": float(entry.get("y", 50)),
                        "z": float(entry.get("z", 50)),
                        "joint_deg": jd,
                        "r_arm_rad": [float(v) for v in entry["r_arm_rad"]],
                        "elbow_bended": abs(float(jd[3])) >= 15 if len(jd) >= 4 else False,
                    }
                )
            out[key] = tiles
    return out


def _save_tile_png(im, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def _montage_href(html_path: Path, montage_root: Path, group: str) -> str:
    p = montage_root / f"group_{group}.png"
    if not p.is_file():
        return ""
    return Path(os.path.relpath(p.resolve(), html_path.parent.resolve())).as_posix()


def _write_html(
    *,
    html_path: Path,
    tile_root: Path,
    montage_root: Path,
    groups: dict[str, list[dict[str, Any]]],
    pos_selection: dict[str, list[int]],
    neg_selection: dict[str, list[int]] | None,
    tile_href_prefix: str,
) -> None:
    neg_init = neg_selection or {k: [] for k in groups}
    sections: list[str] = []
    for group in _all_group_keys():
        tiles = groups.get(group, [])
        if not tiles:
            continue
        pos_set = set(pos_selection.get(group, []))
        montage = _montage_href(html_path, montage_root, group)
        montage_html = (
            f'<p class="montage"><img src="{html.escape(montage)}" alt="montage" loading="lazy"/></p>'
            if montage
            else ""
        )
        cells: list[str] = []
        for t in tiles:
            idx = int(t["tile_idx"])
            fname = f"tile_{idx:02d}.png"
            rel = f"{tile_href_prefix}/{group}/{fname}"
            cls = "tile"
            if idx in pos_set:
                cls += " pos"
            cells.append(
                f'<button type="button" class="{cls}" data-group="{html.escape(group)}" '
                f'data-idx="{idx}" title="{html.escape(group)} #{idx}">'
                f'<img src="{html.escape(rel)}" alt="#{idx}"/>'
                f'<span class="num">{idx}</span></button>'
            )
        sections.append(
            f'<section class="group" id="{html.escape(group)}">'
            f'<h2>{html.escape(group)} '
            f'<span class="counts" data-group-counts="{html.escape(group)}"></span></h2>'
            f'{montage_html}'
            f'<div class="grid5">{"".join(cells)}</div></section>'
        )

    pos_json = json.dumps(pos_selection, ensure_ascii=False)
    neg_json = json.dumps(neg_init, ensure_ascii=False)
    groups_meta = json.dumps(groups, ensure_ascii=False)

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Task A — neg pose picker (12 groups × 25)</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e8eaed}}
.wrap{{padding:16px 20px;max-width:1400px;margin:0 auto}}
h1{{font-size:1.2rem;margin:0 0 8px}}
.legend{{background:#171a21;border:1px solid #333;border-radius:10px;padding:12px;margin:12px 0;font-size:14px;color:#9aa0a6}}
.legend span{{display:inline-block;margin-right:16px}}
.swatch{{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:middle;margin-right:4px}}
.swatch.pos{{background:#166534;border:2px solid #4ade80}}
.swatch.neg{{background:#7f1d1d;border:2px solid #f87171}}
.swatch.neu{{background:#1e293b;border:2px solid #475569}}
.toolbar{{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}}
button.action{{padding:8px 14px;border-radius:8px;border:1px solid #444;background:#1e293b;color:#e2e8f0;cursor:pointer}}
button.action:hover{{background:#334155}}
#exportBox{{width:100%;min-height:80px;background:#0b0d12;color:#a5b4fc;border:1px solid #333;border-radius:8px;padding:8px;font-family:ui-monospace,monospace;font-size:12px}}
.group{{margin:24px 0;padding:16px;border:1px solid #333;border-radius:12px;background:#171a21}}
.group h2{{margin:0 0 10px;font-size:1rem}}
.counts{{font-size:12px;color:#94a3b8;font-weight:normal}}
.montage img{{max-width:100%;border-radius:8px;margin-bottom:10px;opacity:0.85}}
.grid5{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}}
.tile{{position:relative;padding:0;border:3px solid #334155;border-radius:8px;background:#0f1115;cursor:pointer;overflow:hidden}}
.tile img{{width:100%;height:auto;display:block}}
.tile .num{{position:absolute;top:4px;left:4px;background:rgba(0,0,0,.75);color:#fff;font-size:12px;padding:2px 6px;border-radius:4px}}
.tile.pos{{border-color:#4ade80;cursor:not-allowed;opacity:0.95}}
.tile.neg{{border-color:#f87171;box-shadow:0 0 0 2px #f87171}}
.tile.pos::after{{content:'POS';position:absolute;top:4px;right:4px;background:#166534;color:#4ade80;font-size:10px;font-weight:700;padding:2px 5px;border-radius:4px}}
nav.groups{{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0}}
nav.groups a{{color:#93c5fd;font-size:13px;text-decoration:none;padding:4px 8px;border:1px solid #334155;border-radius:6px}}
</style></head>
<body><div class="wrap">
<h1>Task A — pick <b>neg</b> poses (12 groups × 25 tiles)</h1>
<div class="legend">
  <span><i class="swatch pos"></i> POS = plausible set (fixed)</span>
  <span><i class="swatch neg"></i> NEG = click to toggle</span>
  <span><i class="swatch neu"></i> unlabeled</span>
  <p style="margin:8px 0 0">Click any non-POS tile to mark neg. Export → save as
  <code>data/seed/tiago/task_a_neg_selection.json</code></p>
</div>
<div class="toolbar">
  <button type="button" class="action" id="btnExport">Export JSON</button>
  <button type="button" class="action" id="btnCopy">Copy JSON</button>
  <button type="button" class="action" id="btnClear">Clear all neg</button>
  <span id="stats" style="line-height:36px;color:#94a3b8"></span>
</div>
<textarea id="exportBox" readonly placeholder="Export appears here…"></textarea>
<nav class="groups">{"".join(f'<a href="#{html.escape(g)}">{html.escape(g)}</a>' for g in groups)}</nav>
{"".join(sections)}
</div>
<script>
const POS = {pos_json};
let NEG = {neg_json};
const GROUPS = {groups_meta};

function negSet(g) {{
  if (!NEG[g]) NEG[g] = [];
  return new Set(NEG[g]);
}}

function syncUI() {{
  document.querySelectorAll('.tile').forEach(el => {{
    const g = el.dataset.group;
    const idx = parseInt(el.dataset.idx, 10);
    const isPos = (POS[g] || []).includes(idx);
    const isNeg = negSet(g).has(idx);
    el.classList.toggle('pos', isPos);
    el.classList.toggle('neg', isNeg && !isPos);
  }});
  let nPos = 0, nNeg = 0, nTiles = 0;
  for (const [g, tiles] of Object.entries(GROUPS)) {{
    const pos = (POS[g] || []).length;
    const neg = (NEG[g] || []).length;
    nPos += pos; nNeg += neg; nTiles += tiles.length;
    const el = document.querySelector(`[data-group-counts="${{g}}"]`);
    if (el) el.textContent = `(pos=${{pos}} neg=${{neg}} / ${{tiles.length}})`;
  }}
  document.getElementById('stats').textContent =
    `Total: pos=${{nPos}} neg=${{nNeg}} unlabeled=${{nTiles - nPos - nNeg}}`;
}}

function toggleNeg(g, idx) {{
  if ((POS[g] || []).includes(idx)) return;
  const s = negSet(g);
  if (s.has(idx)) s.delete(idx); else s.add(idx);
  NEG[g] = Array.from(s).sort((a,b)=>a-b);
  syncUI();
}}

document.querySelectorAll('.tile').forEach(el => {{
  el.addEventListener('click', () => toggleNeg(el.dataset.group, parseInt(el.dataset.idx,10)));
}});

function buildPayload() {{
  const rows = [];
  for (const [g, tiles] of Object.entries(GROUPS)) {{
    const pos = new Set(POS[g] || []);
    const neg = new Set(NEG[g] || []);
    for (const t of tiles) {{
      const idx = t.tile_idx;
      let label = 'unlabeled';
      if (pos.has(idx)) label = 'pos';
      else if (neg.has(idx)) label = 'neg';
      rows.push({{group: g, tile_idx: idx, label, ...t}});
    }}
  }}
  return {{
    time: new Date().toISOString(),
    experiment: 'A',
    mode: 'manual_neg_picker',
    pos_selection: POS,
    neg_selection: NEG,
    n_pos: rows.filter(r=>r.label==='pos').length,
    n_neg: rows.filter(r=>r.label==='neg').length,
    rows,
  }};
}}

function showExport() {{
  const payload = buildPayload();
  const txt = JSON.stringify(payload, null, 2);
  document.getElementById('exportBox').value = txt;
  return txt;
}}

document.getElementById('btnExport').onclick = () => showExport();
document.getElementById('btnCopy').onclick = async () => {{
  const txt = showExport();
  try {{ await navigator.clipboard.writeText(txt); alert('Copied to clipboard'); }}
  catch(e) {{ alert('Copy failed — select textarea manually'); }}
}};
document.getElementById('btnClear').onclick = () => {{
  if (!confirm('Clear all neg selections?')) return;
  NEG = {{}};
  for (const g of Object.keys(GROUPS)) NEG[g] = [];
  syncUI();
}};
syncUI();
</script></body></html>"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(doc, encoding="utf-8")
    print(f"[html] wrote {html_path}", flush=True)


def run(args: argparse.Namespace) -> Path:
    tile_root = Path(args.tile_root)
    html_path = Path(args.html_out)
    montage_root = Path(args.montage_root)

    if args.regen_montages or not (montage_root / "group_front_horizontal.png").is_file():
        import subprocess as sp

        print(f"[A] generating montages → {montage_root}", flush=True)
        sp.run(
            [
                sys.executable,
                str(_HERE / "generate_pose_group_tiles.py"),
                "--output-root",
                str(montage_root),
                "--grid-cols",
                str(args.grid_cols),
            ],
            cwd=str(_REPO),
            check=False,
        )

    env = _make_env()
    try:
        groups = _collect_groups(env, grid_side=args.grid_cols, head_hand_gap_min=args.head_hand_gap_min)
        for group, tiles in groups.items():
            for t in tiles:
                idx = int(t["tile_idx"])
                png = tile_root / "tiles" / group / f"tile_{idx:02d}.png"
                if png.is_file() and not args.force_tiles:
                    continue
                q = _default_qpos(env)
                q[QI_R_ARM] = [float(v) for v in t["r_arm_rad"]]
                im = _capture(env, q).convert("RGB")
                from PIL import ImageDraw

                dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, 470, 52], fill=(0, 0, 0))
                dr.text(
                    (8, 6),
                    f"#{idx} x{int(round(t['x']))} y{int(round(t['y']))} z{int(round(t['z']))}",
                    fill="white",
                )
                jd = t.get("joint_deg") or []
                if jd:
                    dr.text(
                        (8, 28),
                        "j=[" + ",".join(str(int(round(float(v)))) for v in jd) + "]",
                        fill="white",
                    )
                _save_tile_png(im, png)
    finally:
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()

    neg_existing = None
    neg_path = Path(args.neg_json)
    if neg_path.is_file():
        neg_existing = json.loads(neg_path.read_text(encoding="utf-8")).get("neg_selection")

    tile_href = Path(os.path.relpath(tile_root.resolve() / "tiles", html_path.parent.resolve())).as_posix()
    _write_html(
        html_path=html_path,
        tile_root=tile_root,
        montage_root=montage_root,
        groups=groups,
        pos_selection=USER_SELECTION,
        neg_selection=neg_existing,
        tile_href_prefix=tile_href,
    )

    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(html_path)], check=False)
    return html_path


def main() -> None:
    p = argparse.ArgumentParser(description="Task A interactive neg pose picker HTML")
    p.add_argument("--tile-root", type=Path, default=DEFAULT_TILE_ROOT)
    p.add_argument("--montage-root", type=Path, default=DEFAULT_MONTAGE_ROOT)
    p.add_argument("--html-out", type=Path, default=DEFAULT_HTML)
    p.add_argument("--neg-json", type=Path, default=DEFAULT_NEG_JSON)
    p.add_argument("--grid-cols", type=int, default=DEFAULT_GRID_COLS)
    p.add_argument("--head-hand-gap-min", type=float, default=DEFAULT_HEAD_HAND_GAP_MIN_M)
    p.add_argument("--regen-montages", action="store_true")
    p.add_argument("--force-tiles", action="store_true")
    p.add_argument("--open", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
