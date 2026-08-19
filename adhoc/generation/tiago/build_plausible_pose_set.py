#!/usr/bin/env python3
"""Build a consolidated gallery of user-selected 'plausible' poses for Tiago."""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from generate_pose_group_tiles import (
    ORDER_DIR,
    ORDER_GRIP,
    SOURCE_DIR_FOR_DISPLAY,
    _select_xy_grid_balanced,
)
from legacy.render_mobile_config import (
    QI_R_ARM,
    _capture,
    _default_qpos,
    _get_pose_bank_entries,
    _make_env,
    _right_hand_forward_vs_torso_head_ok,
)

USER_SELECTION = {
    "front_horizontal": [2, 4, 6, 8, 17, 18, 20],
    "front_vertical": [7, 11, 13, 18, 23],
    "back_horizontal": [1, 19],
    "back_vertical": [15],
    "in_horizontal": [2, 10, 13, 17],
    "in_vertical": [2, 9, 14],
    "out_horizontal": [1, 6, 11],
    "out_vertical": [1, 2, 16],
    "up_horizontal": [1, 2, 10, 15, 16, 22, 24],
    "up_vertical": [1, 2, 11, 14, 18],
    "down_horizontal": [1, 2, 6],
    "down_vertical": [1, 10, 21, 23],
}

OUT_DIR = _REPO / "data/results/visualize/tiago/plausible_set"
OUT_HTML = OUT_DIR / "index.html"

def _render_tile(env, entry: dict[str, Any], group_name: str, tile_idx: int) -> Image.Image:
    q = _default_qpos(env)
    q[QI_R_ARM] = [float(v) for v in entry["r_arm_rad"]]
    im = _capture(env, q).convert("RGB")
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 470, 52], fill=(0, 0, 0))
    label_xyz = (
        f"x{int(round(float(entry.get('x', 50))))} "
        f"y{int(round(float(entry.get('y', 50))))} "
        f"z{int(round(float(entry.get('z', 50))))}"
    )
    jd = entry.get("joint_deg")
    if isinstance(jd, list) and len(jd) == 6:
        label_joint = "j=[" + ",".join(str(int(round(float(v)))) for v in jd) + "]"
    else:
        label_joint = "j=[?]"
    dr.text((8, 6), f"{group_name} #{tile_idx} | {label_xyz}", fill="white")
    dr.text((8, 28), label_joint, fill="white")
    return im

def _get_level_x(val: float) -> str:
    if val < 49: return "low"
    if val < 66: return "med"
    return "high"

def _get_level_y(val: float) -> str:
    if val < 35: return "low"
    if val < 62: return "med"
    return "high"

def _get_level_z(val: float) -> str:
    if val < 39: return "low"
    if val < 67: return "med"
    return "high"

def main():
    os.environ["MOBILE_POSE_HEAD_HAND_X_GAP"] = "-0.1"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tiles").mkdir(parents=True, exist_ok=True)

    env = _make_env()
    bank = _get_pose_bank_entries()
    
    # Group the bank entries exactly as in generate_pose_group_tiles.py
    buckets = {}
    for e in bank:
        src_dir = str(e.get("dir"))
        orient = str(e.get("orient"))
        disp_dir = next((k for k, v in SOURCE_DIR_FOR_DISPLAY.items() if v == src_dir), None)
        if disp_dir in ORDER_DIR and orient in ORDER_GRIP:
            # Re-apply the head-hand gap filter
            q = _default_qpos(env)
            q[QI_R_ARM] = [float(v) for v in e["r_arm_rad"]]
            env.sim.data.qpos[:] = q
            env.sim.forward()
            if _right_hand_forward_vs_torso_head_ok(env.sim):
                buckets.setdefault((disp_dir, orient), []).append(e)

    results = []
    total_count = 0
    
    for group_key, indices in USER_SELECTION.items():
        disp_dir, orient = group_key.split("_")
        group_rows = buckets.get((disp_dir, orient), [])
        # Get the same 25 tiles
        reps = _select_xy_grid_balanced(group_rows, grid_side=5)
        
        print(f"Processing {group_key} (found {len(reps)} tiles in grid)...")
        
        for idx in indices:
            if idx > len(reps):
                print(f"  [warn] index {idx} out of range for {group_key} (max {len(reps)})")
                continue
            
            entry = reps[idx - 1]
            im = _render_tile(env, entry, group_key, idx)
            fname = f"{group_key}_tile_{idx:02d}.png"
            im.save(OUT_DIR / "tiles" / fname)
            
            jd = entry.get("joint_deg", [0]*6)
            eb = abs(float(jd[3])) >= 15
            wb = abs(float(jd[5])) >= 15
            
            results.append({
                "group": group_key,
                "tile_idx": idx,
                "image": f"tiles/{fname}",
                "x": entry.get("x"),
                "y": entry.get("y"),
                "z": entry.get("z"),
                "x_lvl": _get_level_x(float(entry.get("x", 50))),
                "y_lvl": _get_level_y(float(entry.get("y", 50))),
                "z_lvl": _get_level_z(float(entry.get("z", 50))),
                "elbow_bended": eb,
                "wrist_bended": wb,
                "joint_deg": jd,
            })
            total_count += 1

    env.close()

    # Write HTML
    cards = []
    for r in results:
        eb_cls = "tag-true" if r["elbow_bended"] else "tag-false"
        wb_cls = "tag-true" if r["wrist_bended"] else "tag-false"
        
        cards.append(
            f'<article><h3>{r["group"]} #{r["tile_idx"]}</h3>'
            f'<div class="tags">'
            f'<span class="tag {eb_cls}">ElbowBended: {r["elbow_bended"]}</span>'
            f'<span class="tag {wb_cls}">WristBended: {r["wrist_bended"]}</span>'
            f'<span class="tag tag-lvl">X: {r["x_lvl"]}</span>'
            f'<span class="tag tag-lvl">Y: {r["y_lvl"]}</span>'
            f'<span class="tag tag-lvl">Z: {r["z_lvl"]}</span>'
            f'</div>'
            f'<p class="meta">xyz=({r["x"]}, {r["y"]}, {r["z"]})<br/>j={r["joint_deg"]}</p>'
            f'<img src="{r["image"]}" loading="lazy"/></article>'
        )
        
    html_content = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Tiago - Plausible Pose Set</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#0f172a;color:#e2e8f0}}
.wrap{{padding:16px}}h1{{font-size:1.5rem;margin-bottom:8px}}
.summary{{background:#111827;border:1px solid #334155;border-radius:10px;padding:12px;margin:12px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}}
article{{background:#111827;border:1px solid #334155;border-radius:10px;padding:12px}}
img{{width:100%;height:auto;border-radius:6px;margin-top:8px}}
.meta{{font-size:12px;color:#94a3b8;margin:4px 0}}
.tags{{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px}}
.tag{{font-size:10px;padding:2px 6px;border-radius:4px;font-weight:600;text-transform:uppercase}}
.tag-true{{background:#065f46;color:#34d399}}
.tag-false{{background:#7f1d1d;color:#f87171}}
.tag-lvl{{background:#1e293b;color:#94a3b8;border:1px solid #334155}}
</style></head><body><div class="wrap">
<h1>Tiago - Plausible Pose Set</h1>
<div class="summary">
<p>Total {total_count} poses selected by user across 12 groups.</p>
</div>
<section class="grid">{''.join(cards)}</section></div></body></html>"""
    
    OUT_HTML.write_text(html_content, encoding="utf-8")
    print(f"Wrote {total_count} tiles to {OUT_DIR}/tiles/")
    print(f"Wrote gallery to {OUT_HTML}")

if __name__ == "__main__":
    main()
