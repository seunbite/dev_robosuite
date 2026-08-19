#!/usr/bin/env python3
"""Quadruped attribute gallery — gravity-planted stance / move / path cards.

Schema (no head, no dir):
  pose(stance=stand|sit|crouch|play_bow|threelegs_front)
    stand n+n, sit h+l, crouch l+l, play_bow l+h, threelegs_front = stand + FR lift
  move(front_legs|back_legs, x|z, ...) | move(left_front|right_front|left_back|right_back, x|y|z, ...)
  path(base, line|arc, ...)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
_ROBOTARM = _REPO / "adhoc" / "generation" / "robotarm"
for p in (_REPO, _ROBOTARM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from adhoc.utils.repo_paths import unitree_rl_mjlab_root  # noqa: E402
from adhoc.generation.joint_motion_schema import (  # noqa: E402
    quadruped_movements_for_legacy_path,
)
from motion_code_to_config import (  # noqa: E402
    _finalize_quadruped_step,
    _split_steps,
    convert_step_quadruped,
)

OUT_JSON = _REPO / "data/results/motion_configs/quadruped/attr_gallery.json"
OUT_GIF = _REPO / "data/results/render/quadruped_attr_gallery"
OUT_HTML = _REPO / "data/results/html/quadruped_attr_gallery.html"

# front_height + back_height
STANCE_HEIGHTS = {
    "stand": ("mid", "mid"),  # n + n
    "sit": ("high", "low"),  # h + l
    "crouch": ("low", "low"),  # l + l
    "play_bow": ("low", "high"),  # l + h
}


def _from_code(code: str, *, duration_pose: float = 1.4) -> list[dict[str, Any]]:
    state = {
        "dir": "neutral",
        "front_height": "mid",
        "back_height": "mid",
        "three_legs": False,
    }
    out: list[dict[str, Any]] = []
    for part in _split_steps(code):
        step = convert_step_quadruped(part, state=state)
        if step is None:
            raise ValueError(f"unconverted step: {part}")
        steps = step if isinstance(step, list) else [step]
        for s in steps:
            if s.get("type") == "pose":
                s = dict(s)
                s["duration"] = max(float(s.get("duration") or 0.6), duration_pose)
            out.append(_finalize_quadruped_step(s))
    return out


def _row(
    idx: int,
    section: str,
    cue: str,
    code: str,
    *,
    note: str,
) -> dict[str, Any]:
    return {
        "idx": idx,
        "group": section,
        "group_idx": idx,
        "cue": cue,
        "section": section,
        "note": note,
        "source_code": code,
        "movements": _from_code(code),
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    i = 1

    # A. Stance (four feet planted)
    for stance, (fh, bh) in STANCE_HEIGHTS.items():
        code = f"pose(stance={stance}) + hold(t=1.0)"
        rows.append(
            _row(
                i,
                "A_stance",
                f"stance_{stance}",
                code,
                note=f"stance={stance} → front={fh}, back={bh} (four feet planted)",
            )
        )
        i += 1

    # B. threelegs_front only
    rows.append(
        _row(
            i,
            "B_three_legs",
            "threelegs_front",
            "pose(stance=threelegs_front) + hold(t=1.2)",
            note="stance=threelegs_front → lift FR paw forward",
        )
    )
    i += 1

    # C. move
    move_specs = [
        (
            "move_front_legs_z",
            "pose(stance=stand) + move(front_legs, z, [-20,20], rep=2, spd=1.0)",
            "front_legs z: − = play_bow flex, + = sit extend",
        ),
        (
            "move_front_legs_x",
            "pose(stance=stand) + move(front_legs, x, [0,15], rep=2, spd=1.0)",
            "front_legs x (reach)",
        ),
        (
            "move_back_legs_z",
            "pose(stance=stand) + move(back_legs, z, [0,20], rep=2, spd=1.0)",
            "back_legs z: − = flex, + = extend",
        ),
        (
            "move_RF_y",
            "pose(stance=stand) + move(right_front, y, [0,20], rep=2, spd=1.0)",
            "single leg all axes — y",
        ),
        (
            "move_RF_z",
            "pose(stance=threelegs_front) + move(right_front, z, [0,25], rep=2, spd=1.2)",
            "threelegs_front + single FR z wave",
        ),
    ]
    for cue, code, note in move_specs:
        rows.append(_row(i, "C_move", cue, code, note=note))
        i += 1

    # D. path / combos
    combo_specs = [
        (
            "crouch_charge",
            "pose(stance=crouch) + path(base, line, x=+0.30, spd=2.0)",
            "crouch then base forward",
        ),
        (
            "play_bow_hold",
            "pose(stance=play_bow) + hold(t=1.2)",
            "play_bow l+h, four feet",
        ),
        (
            "play_bow_paw",
            "pose(stance=play_bow) + pose(stance=threelegs_front) + hold(t=1.2)",
            "play_bow then threelegs_front",
        ),
        (
            "sit_hold",
            "pose(stance=sit) + hold(t=1.2)",
            "sit h+l",
        ),
        (
            "three_leg_base_arc",
            "pose(stance=threelegs_front) + path(base, arc, r=0, deg=90, spd=1.0)",
            "threelegs_front + base yaw",
        ),
    ]
    for cue, code, note in combo_specs:
        rows.append(_row(i, "D_combo", cue, code, note=note))
        i += 1

    return rows


def write_json(rows: list[dict[str, Any]]) -> Path:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return OUT_JSON


def render_all(
    rows: list[dict[str, Any]],
    *,
    python_bin: str | None = None,
    skip_existing: bool = False,
    camera: str = "side",
    hold_base: bool = False,
    pd_scale: float = 2.0,
    pose_mode: str = "gravity",
) -> list[tuple[dict[str, Any], Path]]:
    """Render with free-root gravity plant (pose_mode=gravity, pd=2 by default)."""
    py = python_bin or os.environ.get("MOTION_RENDER_PYTHON") or sys.executable
    mj_root = unitree_rl_mjlab_root()
    mjlab = mj_root / "scripts" / "record_go2_locomotion_config_mjlab_gif.py"
    cfg_dir = OUT_GIF / "_per_cue_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    OUT_GIF.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(mj_root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    rendered: list[tuple[dict[str, Any], Path]] = []
    for row in rows:
        cue = str(row["cue"])
        safe = cue.replace("/", "_").replace(" ", "_")
        idx = int(row["idx"])
        cpath = cfg_dir / f"g{idx:02d}_{safe}.json"
        gpath = OUT_GIF / f"g{idx:02d}_{safe}.gif"
        if skip_existing and gpath.is_file():
            rendered.append((row, gpath))
            print(f"[skip] {gpath.name}", flush=True)
            continue
        leg = quadruped_movements_for_legacy_path(list(row.get("movements") or []))
        cpath.write_text(
            json.dumps({"cue": cue, "movements": leg}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        cmd = [
            py,
            str(mjlab),
            "--config",
            str(cpath),
            "--out",
            str(gpath),
            "--earth-gravity",
            "--no-hold-base" if not hold_base else "--hold-base",
            "--pose-mode",
            str(pose_mode),
            "--pd-scale",
            str(pd_scale),
            "--camera",
            camera,
        ]
        print(
            f"[render] {cue} (pose_mode={pose_mode}, pd={pd_scale}) …",
            flush=True,
        )
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(mj_root),
            env=env,
        )
        if r.returncode != 0 or not gpath.is_file():
            print(f"  FAIL rc={r.returncode} {(r.stderr or '')[-350:]}", flush=True)
            rendered.append((row, Path("")))
        else:
            print(f"  OK {gpath.name} ({gpath.stat().st_size // 1024}KB)", flush=True)
            rendered.append((row, gpath))
    return rendered


SECTION_TITLES = {
    "A_stance": "A. Stance — stand n+n · sit h+l · crouch l+l · play_bow l+h",
    "B_three_legs": "B. `threelegs_front` — lift FR paw",
    "C_move": "C. `move` — front/back legs (x|z) · single leg (x|y|z)",
    "D_combo": "D. Combos — path / threelegs_front",
}


def write_html(rendered: list[tuple[dict[str, Any], Path]]) -> Path:
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    html_dir = OUT_HTML.parent

    def rel(p: Path) -> str:
        if not p or not p.is_file():
            return ""
        return os.path.relpath(p, html_dir)

    stance_rows = "".join(
        f"<tr><td><code>{s}</code></td><td>{a[0]}</td><td>{a[1]}</td></tr>"
        for s, a in STANCE_HEIGHTS.items()
    )
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
<title>Quadruped attribute gallery</title>
<style>
:root {{ --bg:#eef1ec; --ink:#1a1f1c; --muted:#5a635c; --line:#c5cec6; --card:#fbfcf9; --acc:#2d5a45; }}
body {{ margin:0; font-family:"IBM Plex Sans",system-ui,sans-serif; background:var(--bg); color:var(--ink); padding:28px 32px 64px; }}
h1 {{ font-family:"IBM Plex Serif",Georgia,serif; margin:0 0 8px; }}
.lead {{ color:var(--muted); max-width:72ch; line-height:1.45; }}
nav {{ display:flex; flex-wrap:wrap; gap:8px; margin:16px 0 24px; }}
nav a {{ color:var(--acc); text-decoration:none; border:1px solid var(--line); background:var(--card); padding:6px 10px; font-size:13px; }}
table.map {{ border-collapse:collapse; width:min(520px,100%); margin:12px 0 24px; background:var(--card); font-size:13px; }}
table.map th, table.map td {{ border:1px solid var(--line); padding:8px 10px; text-align:left; }}
table.map th {{ background:#dde5df; }}
section {{ margin:32px 0; }}
h2 {{ font-size:1.2rem; border-bottom:2px solid var(--acc); padding-bottom:6px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:14px; }}
article {{ background:var(--card); border:1px solid var(--line); padding:12px; }}
article h3 {{ margin:0 0 6px; font-size:14px; }}
.note {{ margin:0 0 8px; color:var(--muted); font-size:13px; }}
img {{ width:100%; display:block; background:#ccc; }}
.missing {{ padding:36px; text-align:center; color:#888; background:#eee; }}
pre.code {{ background:#1e2822; color:#d5ebe0; padding:8px; font-size:12px; overflow:auto; margin:8px 0 0; }}
code {{ font-family:"IBM Plex Mono",ui-monospace,monospace; }}
</style></head><body>
<h1>Quadruped attribute gallery</h1>
<p class="lead">
Gravity + free-root plant (<code>pose_mode=gravity</code>, PD=2). Feet stay on the
floor under contact; <code>threelegs_front</code> still uses the kinematic lift path.
<strong>No <code>head</code>, no <code>dir</code></strong>.
</p>
<nav>
  <a href="#A_stance">A Stance</a>
  <a href="#B_three_legs">B threelegs_front</a>
  <a href="#C_move">C move</a>
  <a href="#D_combo">D combo</a>
</nav>
<h2>Stance → front / back height</h2>
<table class="map">
<thead><tr><th>stance</th><th>front</th><th>back</th></tr></thead>
<tbody>{stance_rows}</tbody>
</table>
<p class="lead" style="margin-top:0">
<code>stance=threelegs_front</code>: lift right-front paw forward (only three-leg mode).
<code>move(front_legs|back_legs, x|z)</code> only; single leg allows x/y/z.
</p>
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
    ap.add_argument("--camera", default="side")
    ap.add_argument("--python-bin", default=None)
    ap.add_argument(
        "--hold-base",
        action="store_true",
        help="Pin root (override). Default is free-root gravity plant.",
    )
    ap.add_argument("--pd-scale", type=float, default=2.0)
    ap.add_argument(
        "--pose-mode",
        default="gravity",
        choices=["gravity", "hold_flat", "hold_attitude", "hold_snap", "expressive"],
    )
    args = ap.parse_args()

    rows = build_rows()
    path = write_json(rows)
    print(f"JSON: {path} ({len(rows)} rows)", flush=True)

    if args.skip_render:
        rendered = [
            (r, OUT_GIF / f"g{int(r['idx']):02d}_{r['cue']}.gif") for r in rows
        ]
    else:
        # Wipe old GIFs so schema rename does not leave stale cards.
        if not args.skip_existing:
            for old in OUT_GIF.glob("g*.gif"):
                old.unlink(missing_ok=True)
        rendered = render_all(
            rows,
            python_bin=args.python_bin,
            skip_existing=args.skip_existing,
            camera=args.camera,
            hold_base=args.hold_base,
            pd_scale=args.pd_scale,
            pose_mode=args.pose_mode,
        )
    html = write_html(rendered)
    print(f"HTML: {html}", flush=True)


if __name__ == "__main__":
    main()
