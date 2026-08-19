#!/usr/bin/env python3
"""Stretch attribute gallery — pose / move / path / hold cards.

Hello Robot Stretch 3: mobile base + telescoping mast + pan/tilt head + one arm.
Expressiveness sits between single_arm and roomba (no second arm).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
_ROBOTARM = _REPO / "adhoc" / "generation" / "robotarm"
for p in (_REPO, _ROBOTARM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from stretch_renderer import render_stretch  # noqa: E402
from motion_code_to_config import _split_steps, convert_step_stretch  # noqa: E402

OUT_JSON = _REPO / "data/results/motion_configs/stretch/attr_gallery.json"
OUT_GIF = _REPO / "data/results/render/stretch_attr_gallery"
OUT_HTML = _REPO / "data/results/html/stretch_attr_gallery.html"


def _from_code(code: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for part in _split_steps(code):
        step = convert_step_stretch(part)
        if step is None:
            raise ValueError(f"unconverted step: {part}")
        steps = step if isinstance(step, list) else [step]
        out.extend(steps)
    return out


def _row(idx: int, section: str, cue: str, code: str, *, note: str) -> dict[str, Any]:
    return {
        "idx": idx,
        "group": section,
        "group_idx": idx,
        "cue": cue,
        "section": section,
        "note": note,
        "source_code": code,
        "movements": _from_code(code),
        "platform": "stretch",
        "parse_ok": True,
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    i = 1

    for d in ("still", "front", "up", "out", "in", "back", "down"):
        code = f"pose(dir={d}, grip=vert, z=55, torso=mid) + hold(t=1.0)"
        rows.append(
            _row(i, "A_pose_dir", f"dir_{d}", code, note=f"arm_position={d} (unimanual)")
        )
        i += 1

    for torso in ("low", "mid", "high"):
        rows.append(
            _row(
                i,
                "B_pose_torso_head",
                f"torso_{torso}",
                f"pose(dir=still, torso={torso}, head=center) + hold(t=1.0)",
                note=f"mast height torso={torso}",
            )
        )
        i += 1
    for head in ("left", "right", "up", "down"):
        rows.append(
            _row(
                i,
                "B_pose_torso_head",
                f"head_{head}",
                f"pose(dir=still, head={head}, torso=mid) + hold(t=1.0)",
                note=f"head look={head} (arms stay still)",
            )
        )
        i += 1

    for z in (40, 70, 90):
        rows.append(
            _row(
                i,
                "C_pose_xyz",
                f"z_{z}",
                f"pose(dir=front, grip=vert, z={z}, torso=mid) + hold(t=1.0)",
                note=f"EE height percentile z={z}",
            )
        )
        i += 1
    rows.append(
        _row(
            i,
            "C_pose_xyz",
            "grip_horiz",
            "pose(dir=front, grip=horiz, z=60) + hold(t=1.0)",
            note="gripper_orientation=horizontal",
        )
    )
    i += 1

    move_specs = [
        (
            "nod_tilt",
            "pose(dir=still, head=center) + move(head, tilt, [-22,22], rep=3, spd=1.5, hold=0.1)",
            "head tilt nod",
        ),
        (
            "shake_pan",
            "pose(dir=still, head=center) + move(head, pan, [-25,25], rep=3, spd=2.0, hold=0.1)",
            "head pan shake",
        ),
        (
            "fist_pump",
            "pose(dir=up, grip=vert, torso=high, z=85, x=15) + move(shoulder, z, [+20,-20], rep=3, spd=3.0, hold=0.1)",
            "shoulder z = mast lift bob (arm not telescope; x=15 keeps reach short)",
        ),
        (
            "shoulder_x_telescope",
            "pose(dir=front, grip=vert, z=60, x=30) + move(shoulder, x, [+25,-25], rep=3, spd=2.0, hold=0.1)",
            "shoulder x = arm telescope only (contrast with shoulder z)",
        ),
        (
            "wrist_wave",
            "pose(dir=front, grip=vert, z=60) + move(wrist, y, [-25,25], rep=3, spd=2.5, hold=0.1)",
            "wrist wave",
        ),
        (
            "torso_rise",
            "pose(dir=still, torso=low) + move(torso, height, high, spd=0.8, hold=0.4)",
            "mast rise low→high",
        ),
    ]
    for cue, code, note in move_specs:
        rows.append(_row(i, "D_move", cue, code, note=note))
        i += 1

    path_specs = [
        (
            "ee_line_z",
            "pose(dir=up, grip=vert, torso=high, z=85, x=15) + path(ee, line, axis=z, dist=0.10, rep=3, spd=2.5, hold=0.1)",
            "EE line z = mast lift bob only (= move shoulder z). Arm length fixed (x=15).",
        ),
        (
            "ee_arc_xy",
            "pose(dir=front, grip=vert, z=60, x=40) + path(ee, arc, plane=xy, r=0.05, sweep=360, rep=2, spd=1.4)",
            "EE arc xy — horizontal circle (yaw ↔ slight telescope)",
        ),
        (
            "ee_arc_xz",
            "pose(dir=front, grip=vert, z=60, x=35) + path(ee, arc, plane=xz, r=0.05, sweep=360, rep=2, spd=1.4)",
            "EE arc xz — reach×lift sagittal circle",
        ),
        (
            "ee_arc_yz",
            "pose(dir=front, grip=vert, z=65, x=40) + path(ee, arc, plane=yz, r=0.05, sweep=360, rep=2, spd=1.4)",
            "EE arc yz — wrist-only circle (arm/lift frozen)",
        ),
        (
            "base_forward",
            "pose(dir=still, torso=mid) + path(base, line, x=+0.30, spd=1.5, hold=0.3)",
            "base approach (roomba-like)",
        ),
        (
            "base_spin",
            "pose(dir=still) + path(base, arc, r=0, deg=90, spd=1.2)",
            "in-place yaw",
        ),
        (
            "together_base_wave",
            "pose(dir=front, grip=horiz, z=60, x=35) + path(base, line, x=+0.22, spd=1.2, order=together) + move(wrist, y, [-25,25], rep=3, spd=2.2, hold=0.1)",
            "order=together on base path → runs with following wrist move",
        ),
    ]
    for cue, code, note in path_specs:
        rows.append(_row(i, "E_path", cue, code, note=note))
        i += 1

    combo_specs = [
        (
            "offer_approach",
            "pose(dir=front, grip=horiz, z=55, torso=mid) + path(base, line, x=+0.22, spd=1.2) + hold(t=1.0)",
            "pose + base approach + hold",
        ),
        (
            "point_hold",
            "pose(dir=front, grip=vert, z=60, elbow=F) + hold(t=1.4)",
            "static point emblem",
        ),
    ]
    for cue, code, note in combo_specs:
        rows.append(_row(i, "F_combo", cue, code, note=note))
        i += 1

    return rows


SECTION_TITLES = {
    "A_pose_dir": "A. `pose` — arm dir (still / front / up / out / in / back / down)",
    "B_pose_torso_head": "B. `pose` — torso mast + head look (no invented arm raise)",
    "C_pose_xyz": "C. `pose` — z percentile + grip",
    "D_move": "D. `move` — head / shoulder / wrist / torso",
    "E_path": "E. `path` — ee strokes + base locomotion",
    "F_combo": "F. Combos — offer / point",
}


def write_json(rows: list[dict[str, Any]]) -> Path:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return OUT_JSON


def write_html(rendered: list[tuple[dict[str, Any], Path]]) -> Path:
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    html_dir = OUT_HTML.parent

    def rel(p: Path) -> str:
        if not p or not p.is_file():
            return ""
        return os.path.relpath(p, html_dir)

    by_section: dict[str, list[tuple[dict[str, Any], Path]]] = {}
    for row, gif in rendered:
        by_section.setdefault(str(row["section"]), []).append((row, gif))

    sections: list[str] = []
    for key, title in SECTION_TITLES.items():
        cards = []
        for row, gif in by_section.get(key, []):
            src = rel(gif)
            img = (
                f'<img src="{src}?v={int(gif.stat().st_mtime)}" loading="lazy" alt="{row["cue"]}">'
                if src
                else '<div class="missing">GIF missing</div>'
            )
            cards.append(
                f"<article>"
                f"<h3>g{int(row['idx']):02d} · {row['cue']}</h3>"
                f"<p class='note'>{row.get('note','')}</p>"
                f"{img}"
                f"<pre class='code'>{row.get('source_code','')}</pre>"
                f"</article>"
            )
        sections.append(
            f"<section id='{key}'><h2>{title}</h2>"
            f"<div class='grid'>{''.join(cards)}</div></section>"
        )

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Stretch attribute gallery</title>
<style>
:root {{ --bg:#e8eef2; --ink:#152028; --muted:#5a6870; --line:#c5d0d6; --card:#f7fafc; --acc:#1f6f8b; }}
body {{ margin:0; font-family:"Source Sans 3","IBM Plex Sans",system-ui,sans-serif; background:
  radial-gradient(1200px 600px at 10% -10%, #d7e8f0 0%, transparent 55%),
  linear-gradient(180deg, #eef3f6 0%, var(--bg) 40%, #e3e9ed 100%);
  color:var(--ink); padding:28px 32px 64px; }}
h1 {{ font-family:"Fraunces","IBM Plex Serif",Georgia,serif; font-weight:600; margin:0 0 8px; letter-spacing:-0.02em; }}
.lead {{ color:var(--muted); max-width:72ch; line-height:1.45; }}
nav {{ display:flex; flex-wrap:wrap; gap:8px; margin:16px 0 24px; }}
nav a {{ color:var(--acc); text-decoration:none; border:1px solid var(--line); background:var(--card); padding:6px 10px; font-size:13px; }}
section {{ margin:32px 0; }}
h2 {{ font-size:1.15rem; border-bottom:2px solid var(--acc); padding-bottom:6px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:14px; }}
article {{ background:var(--card); border:1px solid var(--line); padding:12px; }}
article h3 {{ margin:0 0 6px; font-size:14px; }}
.note {{ margin:0 0 8px; color:var(--muted); font-size:13px; }}
img {{ width:100%; display:block; background:#c8d0d6; }}
.missing {{ padding:36px; text-align:center; color:#888; background:#eee; }}
pre.code {{ background:#1a242b; color:#d5e6ef; padding:8px; font-size:12px; overflow:auto; margin:8px 0 0; }}
code {{ font-family:"IBM Plex Mono",ui-monospace,monospace; }}
table.map {{ border-collapse:collapse; width:min(640px,100%); margin:12px 0 8px; background:var(--card); font-size:13px; }}
table.map th, table.map td {{ border:1px solid var(--line); padding:8px 10px; text-align:left; }}
table.map th {{ background:#dce7ed; }}
</style></head><body>
<h1>Stretch attribute gallery</h1>
<p class="lead">
Hello Robot Stretch 3 from <code>mujoco_menagerie</code> (official MJCF).
Schema sits between <code>single_arm</code> and <code>roomba</code> —
base + lift + telescoping arm + wrist/head, <strong>no</strong> <code>left_arm</code>.
</p>
<nav>
  <a href="#A_pose_dir">A dir</a>
  <a href="#B_pose_torso_head">B torso/head</a>
  <a href="#C_pose_xyz">C xyz/grip</a>
  <a href="#D_move">D move</a>
  <a href="#E_path">E path</a>
  <a href="#F_combo">F combo</a>
</nav>
<table class="map">
<thead><tr><th>component</th><th>attributes</th></tr></thead>
<tbody>
<tr><td><code>pose</code></td><td>dir, grip, torso, head, elbow, x/y/z</td></tr>
<tr><td><code>move</code></td><td>shoulder|elbow|wrist × x|y|z · head pan|tilt · torso height</td></tr>
<tr><td><code>path</code></td><td>ee line/arc · base line/arc</td></tr>
<tr><td><code>hold</code></td><td>t (first-class freeze)</td></tr>
</tbody>
</table>
{''.join(sections)}
</body></html>
"""
    OUT_HTML.write_text(page, encoding="utf-8")
    return OUT_HTML


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rows = build_rows()
    path = write_json(rows)
    print(f"JSON: {path} ({len(rows)} rows)", flush=True)

    OUT_GIF.mkdir(parents=True, exist_ok=True)
    if args.skip_render:
        rendered = [(r, OUT_GIF / f"emb_g{int(r['idx']):02d}_{r['cue']}.gif") for r in rows]
    else:
        if not args.skip_existing and args.force:
            for old in OUT_GIF.glob("emb_g*.gif"):
                old.unlink(missing_ok=True)
        render_stretch(rows, OUT_GIF, force=args.force or not args.skip_existing)
        rendered = [(r, OUT_GIF / f"emb_g{int(r['idx']):02d}_{r['cue']}.gif") for r in rows]

    html = write_html(rendered)
    print(f"HTML: {html}", flush=True)


if __name__ == "__main__":
    main()
