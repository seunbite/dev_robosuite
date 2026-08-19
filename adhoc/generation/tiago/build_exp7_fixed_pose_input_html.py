#!/usr/bin/env python3
"""Render exp7/exp7.1 VLM input PNGs from gt_tiago pose_gt (no LLM generation)."""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE, _REPO / "adhoc/generation/robotarm"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config_gen_mobile_vlm import tiago_gt_fixed_pose  # noqa: E402
from fixed_pose_render import ensure_fixed_pose_png  # noqa: E402
from pilot40_paths import HTML_EXP_DIR, load_gt_by_cue, manifest90_cue_names  # noqa: E402

POSE_KEYS = (
    "torso_height",
    "arm_position",
    "gripper_orientation",
    "elbow_bended",
    "head",
    "left_arm",
    "x",
    "y",
    "z",
)

CSS = """
body{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e8eaed}
h1{padding:16px 24px;margin:0;font-size:1.25rem;border-bottom:1px solid #333}
.meta{padding:12px 24px;color:#9aa0a6;font-size:0.9rem;border-bottom:1px solid #222}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;padding:20px 24px}
.card{padding:12px;border:1px solid #333;border-radius:10px;background:#171a21}
.card h2{margin:0 0 6px;font-size:0.95rem}
.card .pose_gt{margin:0 0 8px;color:#8ab4f8;font-size:0.8rem;word-break:break-word}
.card img{width:100%;height:auto;border-radius:6px;background:#000;display:block;margin:8px 0}
.card pre{white-space:pre-wrap;background:#0f1115;padding:8px;border-radius:6px;font-size:0.7rem;margin:0;overflow:auto}
.missing{color:#f28b82;font-size:0.8rem}
"""


def _esc(x: object) -> str:
    return html.escape(str(x) if x is not None else "")


def _rel(out_html: Path, asset: Path) -> str:
    if not asset.is_file():
        return ""
    return Path(os.path.relpath(asset.resolve(), out_html.parent.resolve())).as_posix()


def _fmt_pose(pose: dict) -> str:
    return ", ".join(f"{k}={pose[k]}" for k in POSE_KEYS if k in pose)


def build_html(
    *,
    png_dir: Path,
    out_html: Path,
    n_cues: int = 40,
    force_png: bool = False,
) -> Path:
    gt_by = load_gt_by_cue()
    out_html.parent.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    from adhoc.generation.tiago.legacy import _make_env  # noqa: WPS433

    env = _make_env()
    cards: list[str] = []
    n_ok = 0
    try:
        for cue in manifest90_cue_names()[:n_cues]:
            gt_row = gt_by.get(cue)
            if not gt_row:
                cards.append(
                    f'<article class="card"><h2>{_esc(cue)}</h2>'
                    f'<p class="missing">missing from gt_tiago.json</p></article>'
                )
                continue
            pose_gt = str(gt_row.get("pose_gt") or gt_row.get("groundtruth") or "")
            any_tag = ' <span style="color:#fdd663">(pose_any)</span>' if gt_row.get("pose_any") else ""
            fixed = tiago_gt_fixed_pose(gt_row)
            cue_idx = int(gt_row.get("cue_idx", 0))
            try:
                png_path, _ = ensure_fixed_pose_png(
                    fixed,
                    cue=cue,
                    cue_idx=cue_idx,
                    png_dir=png_dir,
                    env=env,
                    force=force_png,
                )
                img_rel = _rel(out_html, png_path)
                img_tag = f'<img src="{_esc(img_rel)}" alt="{_esc(cue)}">' if img_rel else '<p class="missing">PNG missing</p>'
                n_ok += 1
            except Exception as e:
                img_tag = f'<p class="missing">render failed: {_esc(e)}</p>'
            cards.append(
                f'<article class="card">'
                f'<h2>g{cue_idx:02d} · {_esc(cue)}</h2>'
                f'<p class="pose_gt"><b>pose_gt:</b> {_esc(pose_gt)}{any_tag}</p>'
                f'{img_tag}'
                f'<pre>{_esc(_fmt_pose(fixed))}</pre>'
                f"</article>"
            )
    finally:
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()

    body = "\n".join(cards)
    page = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>exp7 fixed-pose input (gt_tiago)</title>
<style>{CSS}</style>
</head><body>
<h1>exp7 / 7.1 VLM input — fixed pose from gt_tiago.json</h1>
<p class="meta">{n_ok}/{n_cues} PNGs · source: gt_tiago pose_gt primary (dir, grip) · PNG dir: {_esc(png_dir)}</p>
<div class="grid">
{body}
</div>
</body></html>
"""
    out_html.write_text(page, encoding="utf-8")
    return out_html


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-cues", type=int, default=40)
    p.add_argument(
        "--png-dir",
        default=str(_REPO / "data/results/render/tiago/fixed_pose_png_tiago_gt"),
    )
    p.add_argument(
        "--out-html",
        default=str(HTML_EXP_DIR / "exp7_fixed_pose_input_tiago_gt.html"),
    )
    p.add_argument("--force-png", action="store_true")
    p.add_argument("--open", action="store_true")
    args = p.parse_args()

    out = build_html(
        png_dir=Path(args.png_dir),
        out_html=Path(args.out_html),
        n_cues=args.n_cues,
        force_png=args.force_png,
    )
    print(f"Wrote {out}")
    if args.open:
        subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    main()
