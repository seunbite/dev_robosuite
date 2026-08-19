#!/usr/bin/env python3
"""Exp1 Tiago review HTML — same layout as manipulator pose generation review."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _REPO / "adhoc/generation/tiago", _REPO / "adhoc/generation/robotarm"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from parse_pose_gt_mobile import pose_generation_correct_any_mobile, pose_gt_display  # noqa: E402
from pilot40_paths import (  # noqa: E402
    RENDER_DIR,
    html_result_path,
    load_config_list,
    load_gt_by_cue,
    result_config_path,
    score_result_path,
)

DEFAULT_MODEL_TAG = "gemini-2.5-pro"


def exp1_render_dir(model_tag: str) -> Path:
    return RENDER_DIR / f"exp1_{model_tag}"


def exp1_media_dir(model_tag: str) -> Path:
    return RENDER_DIR / f"exp1_{model_tag}_media"


def exp1_pose_tiles_dir(model_tag: str) -> Path:
    return exp1_render_dir(model_tag) / "pose_tiles"


def _esc(x: object) -> str:
    import html

    return html.escape(str(x) if x is not None else "")


def _first_mobile_pose(row: dict) -> dict:
    for step in row.get("movements") or []:
        if step.get("type") != "pose":
            continue
        pose = (step.get("parameters") or {}).get("pose") or {}
        if pose.get("arm_position") and pose.get("gripper_orientation"):
            return pose
    return {}


def _config_toggle_html(cfg: dict | None) -> str:
    if not cfg:
        body = "(missing)"
    else:
        body = json.dumps(cfg, indent=2, ensure_ascii=False)
    return (
        '<details class="cfg-toggle">'
        "<summary>Generated config</summary>"
        f'<pre class="cfg-pre">{_esc(body)}</pre>'
        "</details>"
    )


def _badge_html(is_true: bool) -> str:
    cls = "true" if is_true else "false"
    txt = "TRUE" if is_true else "FALSE"
    return f'<span class="badge {cls}">{txt}</span>'


def _media_paths(
    idx: int,
    cue: str,
    out_html: Path,
    *,
    model_tag: str,
    ignore_xyz: bool = False,
) -> tuple[str, str, str]:
    stem = f"mm19_g{int(idx):02d}_{cue}"
    gif = exp1_render_dir(model_tag) / f"{stem}.gif"
    mp4 = exp1_media_dir(model_tag) / "mp4" / f"{stem}.mp4"
    tile = exp1_pose_tiles_dir(model_tag) / f"{stem}_pose_pool.gif"
    gif_rel = os.path.relpath(gif, out_html.parent) if gif.is_file() else ""
    mp4_rel = os.path.relpath(mp4, out_html.parent) if mp4.is_file() else ""
    tile_rel = ""
    if ignore_xyz and tile.is_file():
        tile_rel = os.path.relpath(tile, out_html.parent)
    return gif_rel, mp4_rel, tile_rel


def build(
    *,
    model_tag: str = DEFAULT_MODEL_TAG,
    out_html: Path | None = None,
    ignore_xyz: bool = False,
) -> Path:
    score_path = score_result_path(1, model_tag)
    cfg_path = result_config_path(1, model_tag)
    if not score_path.is_file():
        raise FileNotFoundError(score_path)
    score = json.loads(score_path.read_text(encoding="utf-8"))
    cfg_rows = load_config_list(cfg_path) if cfg_path.is_file() else []
    cfg_by_idx = {int(r["idx"]): r for r in cfg_rows if r.get("idx") is not None}
    cfg_by_cue = {str(r.get("cue", "")): r for r in cfg_rows if r.get("cue")}
    gt_by_cue = load_gt_by_cue()
    out = out_html or html_result_path(1, model_tag)

    cards: list[tuple[bool, str]] = []
    n_true = n_total = n_missing = 0

    for row in score.get("rows") or []:
        idx = int(row.get("cue_idx", 0))
        cue = str(row.get("cue", ""))
        cfg = cfg_by_idx.get(idx) or cfg_by_cue.get(cue) or {}
        gt_row = gt_by_cue.get(cue) or {}
        pose_gt = row.get("pose_gt") or pose_gt_display(
            bool(gt_row.get("pose_any")),
            list(gt_row.get("pose_options") or []),
            str(gt_row.get("pose_gt") or gt_row.get("groundtruth") or ""),
        )
        gen_pose = _first_mobile_pose(cfg)
        is_true = bool(row.get("generation_correct"))
        n_total += 1
        if is_true:
            n_true += 1

        gif_rel, mp4_rel, tile_rel = _media_paths(
            idx, cue, out, model_tag=model_tag, ignore_xyz=ignore_xyz
        )
        if not gif_rel and not mp4_rel:
            n_missing += 1

        gif_html = (
            f'<img src="{_esc(gif_rel)}" loading="lazy" />'
            if gif_rel
            else '<div class="missing">GIF missing</div>'
        )
        mp4_html = (
            f'<video controls preload="metadata" src="{_esc(mp4_rel)}"></video>'
            if mp4_rel
            else '<div class="missing">MP4 missing</div>'
        )
        tile_html = ""
        if tile_rel:
            tile_html = (
                f'<figure class="tile-wide"><figcaption>Pose pool</figcaption>'
                f'<img src="{_esc(tile_rel)}" loading="lazy" /></figure>'
            )

        card = "".join(
            [
                '<section class="card">',
                f'<div class="head"><h2>g{idx} {_esc(cue)}</h2>{_badge_html(is_true)}</div>',
                '<div class="meta">',
                f"<div><b>GT</b>: {_esc(pose_gt)}</div>",
                f"<div><b>Generation</b>: {_esc(gen_pose)}</div>",
                "</div>",
                '<div class="media-grid">',
                f"<figure><figcaption>GIF</figcaption>{gif_html}</figure>",
                f"<figure><figcaption>MP4</figcaption>{mp4_html}</figure>",
                "</div>",
                tile_html,
                _config_toggle_html(cfg if cfg else None),
                "</section>",
            ]
        )
        cards.append((is_true, card))

    true_cards = [c for ok, c in cards if ok]
    false_cards = [c for ok, c in cards if not ok]
    acc = (100.0 * n_true / n_total) if n_total else 0.0

    css = """
body{font-family:system-ui,sans-serif;margin:20px;max-width:1500px}
header{background:#f5f7ff;padding:12px 16px;border-radius:10px;margin-bottom:18px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px}
.card{border:1px solid #ddd;border-radius:10px;padding:10px;background:#fff}
.head{display:flex;justify-content:space-between;align-items:center;gap:8px}
h1{margin:0 0 4px;font-size:1.25rem}
h2{margin:0;font-size:1rem}
.meta{font-size:12px;line-height:1.45;margin:8px 0 10px;color:#222}
.media-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.figure{margin:0}
figcaption{font-size:12px;font-weight:600;margin-bottom:4px}
.tile-wide{margin:10px 0 0}
.tile-wide img{width:100%;height:auto;border:1px solid #ccc;border-radius:6px;background:#111}
img,video{width:100%;height:auto;border:1px solid #ccc;border-radius:6px;background:#111}
.missing{border:1px dashed #bbb;border-radius:6px;padding:20px;text-align:center;font-size:12px;color:#666}
.badge{display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;font-weight:700}
.badge.true{background:#e8f8ee;color:#156f37;border:1px solid #b9ebcc}
.badge.false{background:#ffecec;color:#9c1f1f;border:1px solid #ffc6c6}
.cfg-toggle{margin-top:10px;border:1px solid #e0e0e0;border-radius:8px;background:#fafafa}
.cfg-toggle summary{cursor:pointer;padding:8px 10px;font-size:12px;font-weight:600;color:#333;user-select:none}
.cfg-toggle summary:hover{background:#f0f0f0}
.cfg-pre{margin:0;padding:10px;font-size:11px;line-height:1.4;overflow:auto;max-height:360px;background:#f7f7f7;border-top:1px solid #e0e0e0;white-space:pre-wrap;word-break:break-word}
.section{margin:28px 0 10px;padding:8px 12px;border-radius:8px;font-size:1.05rem;font-weight:700}
.section.true{background:#e8f8ee;color:#156f37;border:1px solid #b9ebcc}
.section.false{background:#ffecec;color:#9c1f1f;border:1px solid #ffc6c6}
"""

    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Exp1 Tiago — {model_tag}</title>"
        f"<style>{css}</style></head><body>"
        "<header>"
        f"<h1>Exp1 Tiago — Pose generation ({model_tag})</h1>"
        f"<div>Generation vs mobile GT: <b>{n_true}/{n_total}</b> ({acc:.1f}%)</div>"
        f"<div>Missing media: {n_missing}</div>"
        f"<div>Config: <code>{_esc(cfg_path)}</code></div>"
        f"<div>Render: <code>{_esc(exp1_render_dir(model_tag))}</code></div>"
        + (
            f"<div>ignore_xyz: <b>on</b> — pose tiles in <code>{_esc(exp1_pose_tiles_dir(model_tag))}</code></div>"
            if ignore_xyz
            else ""
        )
        + "<div>Order: <b>TRUE</b> first, then <b>FALSE</b>.</div>"
        "</header>"
        f"<div class='section true'>TRUE — {len(true_cards)} cues</div>"
        f"<div class='grid'>{''.join(true_cards)}</div>"
        f"<div class='section false'>FALSE — {len(false_cards)} cues</div>"
        f"<div class='grid'>{''.join(false_cards)}</div>"
        "</body></html>"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out


def render_exp1_media(*, model_tag: str = DEFAULT_MODEL_TAG, ignore_xyz: bool = False) -> Path:
    """Render GIFs (+ MP4) from exp1 config JSON into exp1-specific dirs."""
    cfg_path = result_config_path(1, model_tag).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    gif_dir = exp1_render_dir(model_tag)
    media_dir = exp1_media_dir(model_tag)
    gif_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    render_cmd = [
        sys.executable,
        str(_HERE / "render.py"),
        "run",
        f"--config_json={cfg_path}",
        f"--output_dir={gif_dir.resolve()}",
        "--do_html=False",
        "--html_only=False",
        f"--ignore_xyz={ignore_xyz}",
    ]
    subprocess.run(render_cmd, cwd=str(_HERE), check=True)

    from prepare_pilot40_media import run as prepare_media  # noqa: WPS433

    prepare_media(
        argparse.Namespace(
            config_json=str(cfg_path),
            gif_dir=str(gif_dir.resolve()),
            media_dir=str(media_dir.resolve()),
            limit=0,
            render_missing=False,
        )
    )
    return gif_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Build Tiago exp1 review HTML (manipulator-style)")
    p.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    p.add_argument("--out-html", type=Path, default=None)
    p.add_argument("--render", action="store_true", help="Render exp1 GIF/MP4 before building HTML")
    p.add_argument(
        "--ignore-xyz",
        action="store_true",
        help="Ignore x/y/z for pose-bank pick; render all dir+orient+elbow matches as tile grid",
    )
    p.add_argument("--open", action="store_true")
    args = p.parse_args()
    if args.render:
        out_dir = render_exp1_media(model_tag=args.model_tag, ignore_xyz=args.ignore_xyz)
        print(f"Rendered exp1 media → {out_dir}")
    out = build(model_tag=args.model_tag, out_html=args.out_html, ignore_xyz=args.ignore_xyz)
    print(f"Wrote {out}")
    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    main()
