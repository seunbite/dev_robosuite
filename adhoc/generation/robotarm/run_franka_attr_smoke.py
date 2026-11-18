#!/usr/bin/env python3
"""Smoke-test Franka (robosuite Panda) on the IIWA single_arm attribute set.

Renders dir / grip / xyz / move / path / hold, writes an HTML gallery.

    python adhoc/generation/robotarm/run_franka_attr_smoke.py
"""
from __future__ import annotations

import html as html_mod
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_PAPER = _REPO / "papers/icra2027_motion_abstraction"
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from motion_code_to_config import convert_code  # noqa: E402

JSONL = _REPO / "data/seed/_remainder/closest_poses_results.jsonl"
OUT = (
    _REPO
    / "data/results/evaluation/embodiment_cue_id"
    / "results_franka_attr_smoke"
)
HTML_COPY = _PAPER / "franka_attr_smoke" / "gallery.html"

# 6 dirs × vert/horiz, then xyz / move / path / hold.
_DIRS = ("front", "back", "up", "down", "left", "right")
CASES: list[tuple[str, str, str]] = [
    *(
        (
            "dir_x_grip",
            f"{d}_{grip}",
            f"pose(dir={d}, grip={grip}, z=55) + hold(t=0.8)",
        )
        for grip in ("vert", "horiz")
        for d in _DIRS
    ),
    ("xyz", "x20", "pose(dir=front, grip=vert, x=20, y=50, z=55) + hold(t=0.8)"),
    ("xyz", "x50", "pose(dir=front, grip=vert, x=50, y=50, z=55) + hold(t=0.8)"),
    ("xyz", "x80", "pose(dir=front, grip=vert, x=80, y=50, z=55) + hold(t=0.8)"),
    ("xyz", "y20", "pose(dir=front, grip=vert, x=60, y=20, z=55) + hold(t=0.8)"),
    ("xyz", "y50", "pose(dir=front, grip=vert, x=60, y=50, z=55) + hold(t=0.8)"),
    ("xyz", "y80", "pose(dir=front, grip=vert, x=60, y=80, z=55) + hold(t=0.8)"),
    ("xyz", "z30", "pose(dir=front, grip=vert, x=60, y=50, z=30) + hold(t=0.8)"),
    ("xyz", "z55", "pose(dir=front, grip=vert, x=60, y=50, z=55) + hold(t=0.8)"),
    ("xyz", "z85", "pose(dir=front, grip=vert, x=60, y=50, z=85) + hold(t=0.8)"),
    ("move", "move_shoulder_z", "pose(dir=front, grip=vert, z=55) + move(shoulder, z, [0,25], spd=2.0, hold=0.15)"),
    ("move", "move_elbow_y", "pose(dir=front, grip=vert, z=55) + move(elbow, y, [0,20], spd=2.0, hold=0.15)"),
    ("move", "move_wrist_x", "pose(dir=front, grip=vert, x=60, y=50, z=55) + move(wrist, x, [-20,20], rep=2, spd=2.5, hold=0.08)"),
    ("path", "path_line_x", "pose(dir=front, grip=vert, x=50, y=50, z=55) + path(ee, line, axis=x, dist=[-0.10,0.10], spd=0.5, hold=0.3)"),
    ("path", "path_line_y", "pose(dir=front, grip=vert, x=50, y=50, z=55) + path(ee, line, axis=y, dist=[-0.10,0.10], spd=0.5, hold=0.3)"),
    ("path", "path_line_z", "pose(dir=front, grip=vert, x=50, y=50, z=55) + path(ee, line, axis=z, dist=[-0.10,0.10], spd=0.5, hold=0.3)"),
    ("path", "path_arc_yz", "pose(dir=front, grip=vert, x=50, y=50, z=55) + path(ee, arc, plane=yz, r=0.10, sweep=90, spd=0.5)"),
    ("hold", "hold_only", "pose(dir=front, grip=vert, x=60, y=50, z=55) + hold(t=1.2)"),
]


def _render_panda(row: dict[str, Any], gif_path: Path, *, force: bool) -> str:
    """Return '' on success, else error string."""
    if gif_path.is_file() and not force:
        return ""
    robotarm = _HERE
    google_legacy = _REPO / "adhoc/generation/tiago/legacy"
    sys.path = [p for p in sys.path if Path(p).resolve() != google_legacy.resolve()]
    if str(robotarm) in sys.path:
        sys.path.remove(str(robotarm))
    sys.path.insert(0, str(robotarm))
    for mod in list(sys.modules):
        if mod == "legacy" or mod.startswith("legacy."):
            del sys.modules[mod]
    from legacy.motion_generation_core import generate  # noqa: WPS433

    os.environ.setdefault("MANIPULATOR_POSE_XYZ_MODE", "trim")
    os.environ["MOTION_FRAME_SCALE"] = "1.0"
    scratch = OUT / "scratch" / str(row["cue"])
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    tmp_cfg = scratch / "cfg.json"
    tmp_cfg.write_text(json.dumps([row], indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        generate(
            robot="Panda",
            cue=str(row["cue"]),
            cue_idx=int(row["idx"]),
            pose_index=None,
            jsonl_path=str(JSONL),
            config_path=str(tmp_cfg),
            output_dir=str(scratch),
            hz=12,
            top_k=1,
            ignore_xyz=False,
            overlay_progress_bar=False,
        )
        robot_dir = scratch / "Panda"
        gifs = sorted(robot_dir.glob("*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not gifs:
            return "no gif written"
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gifs[0], gif_path)
        return ""
    except Exception as exc:  # noqa: BLE001
        return str(exc)


def write_html(results: list[dict[str, Any]], out_html: Path) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        groups.setdefault(r["group"], []).append(r)
    sections = []
    n_ok = sum(1 for r in results if r["gif"] and not r["error"])
    for group, rows in groups.items():
        cells = []
        for r in rows:
            gif = Path(r["gif"]) if r["gif"] else None
            if gif is not None and gif.is_file():
                rel = os.path.relpath(gif, out_html.parent)
                bust = int(gif.stat().st_mtime)
                media = f'<img src="{html_mod.escape(rel)}?v={bust}" loading="lazy">'
            else:
                media = f'<p class="miss">FAIL<br>{html_mod.escape(r.get("error") or "missing")}</p>'
            cells.append(
                f'<div class="col"><h3>{html_mod.escape(r["slug"])}</h3>{media}'
                f'<pre>{html_mod.escape(r["code"])}</pre>'
                f'<p class="notes">{html_mod.escape(r.get("notes") or "")}</p></div>'
            )
        grid_class = "grid6" if group == "dir_x_grip" else "grid"
        sections.append(
            f'<section><h2>{html_mod.escape(group)}</h2>'
            f'<div class="{grid_class}">{"".join(cells)}</div></section>'
        )
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Franka (Panda) attribute smoke</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:#111;color:#eee}}
h1{{padding:16px 24px;margin:0;border-bottom:1px solid #333}}
.meta{{padding:8px 24px;color:#9aa0a6;max-width:980px}}
section{{margin:16px}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}
.grid6{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px}}
.col{{background:#1a1d24;border-radius:10px;padding:8px}}
.col h3{{margin:0 0 6px;font-size:.8rem}}
img{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;background:#000;display:block}}
pre{{font-size:11px;white-space:pre-wrap;word-break:break-word;background:#0a0c10;padding:6px;border-radius:6px}}
.miss{{color:#f28b82;font-size:.8rem}}
.notes{{font-size:.75rem;color:#9aa0a6}}
@media (max-width:1100px){{.grid,.grid6{{grid-template-columns:1fr 1fr}}}}
</style></head>
<body>
<h1>Franka (Panda) · IIWA single-arm attributes</h1>
<p class="meta">{n_ok}/{len(results)} rendered · MuJoCo / robosuite <code>Panda</code> (Franka).
Same Motion Code schema as IIWA (<code>dir, grip, x/y/z, move(shoulder|elbow|wrist), path(ee), hold</code>).
Panda: bank <code>left/right</code> remapped at lookup; bank <code>vert/horiz</code> swapped.
Pipeline: closest dir+grip pose, then EE IK translate to xyz percentiles (orientation held).</p>
<p class="meta">{html_mod.escape(datetime.now().isoformat(timespec="seconds"))}</p>
{"".join(sections)}
</body></html>
"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--group", action="append", default=None)
    args = ap.parse_args()
    groups = set(args.group) if args.group else None
    OUT.mkdir(parents=True, exist_ok=True)
    gif_dir = OUT / "gif"
    gif_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for i, (group, slug, code) in enumerate(CASES, start=1):
        converted = convert_code(code, platform="single_arm")
        row = {
            "idx": i,
            "cue": slug,
            "description": group,
            "movements": converted.get("movements") or [],
            "parse_ok": bool(converted.get("parse_ok")),
            "source_code": converted.get("source_code") or code,
            "platform": "single_arm",
            "domain": "manipulator",
            "notes": converted.get("notes") or [],
        }
        gif_path = gif_dir / f"{i:02d}_{slug}.gif"
        err = ""
        if not row["parse_ok"] or not row["movements"]:
            err = "parse_fail: " + ",".join(row["notes"] or ["empty"])
        else:
            print(f"[franka] {slug} …", flush=True)
            err = _render_panda(
                row,
                gif_path,
                force=bool(args.force or (groups and group in groups)),
            )
            if err:
                print(f"[franka] FAIL {slug}: {err}", flush=True)
            else:
                print(f"[franka] ok {gif_path.name}", flush=True)
        results.append(
            {
                "group": group,
                "slug": slug,
                "code": code,
                "gif": str(gif_path) if gif_path.is_file() else "",
                "error": err,
                "notes": " ".join(row["notes"]) if row["notes"] else "",
            }
        )
    (OUT / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    html_out = OUT / "gallery.html"
    write_html(results, html_out)
    write_html(results, HTML_COPY)
    n_ok = sum(1 for r in results if r["gif"] and not r["error"])
    print(f"done {n_ok}/{len(results)}\n{html_out}\n{HTML_COPY}", flush=True)


if __name__ == "__main__":
    main()
