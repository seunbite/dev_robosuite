#!/usr/bin/env python3
"""Build pose-generation review HTML with GT match badges, GIF and MP4."""
from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_ROBOTARM = Path(__file__).resolve().parent
if str(_ROBOTARM) not in sys.path:
    sys.path.insert(0, str(_ROBOTARM))

SUITES = {
    "pilot40": {
        "title": "Pilot40",
        "tsv": REPO / "data/results/verify/pilot40_pose_eval_consolidated_scored.tsv",
        "gif_dir": REPO / "data/results/visualize/pose_generation_pilot40_hz10/IIWA",
        "mp4_dir": REPO / "data/results/visualize/pose_generation_pilot40_hz10/IIWA_mp4",
        "cfg": REPO / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_generation_pose_pilot40.json",
        "out_html": REPO / "data/results/html/manipulator/pose_generation_eval_review_pilot40_generation_rendered.html",
        "use_config_rows": False,
    },
    "pilot90": {
        "title": "Pilot90 (task 1)",
        "tsv": None,
        "gif_dir": REPO / "run/IIWA",
        "mp4_dir": REPO / "run/IIWA_mp4",
        "cfg": REPO
        / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json",
        "consolidated": REPO / "data/results/verify/pilot40_pose_eval_consolidated.json",
        "out_html": REPO / "data/results/html/manipulator/pose_generation_eval_review_pilot90_generation_rendered.html",
        "use_config_rows": True,
    },
}


def _esc(x: object) -> str:
    return html.escape(str(x) if x is not None else "")


def _to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}


def _latest_gif_for_cue(cue: str, pose_id: int | None, *, gif_dir: Path) -> Path | None:
    if pose_id is not None:
        files = sorted(
            gif_dir.glob(f"*_IIWA_{cue}_p{pose_id}.gif"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if files:
            return files[0]
    files = sorted(
        gif_dir.glob(f"*_IIWA_{cue}_*.gif"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def _ensure_mp4(gif_path: Path, *, mp4_dir: Path) -> Path | None:
    mp4_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = mp4_dir / (gif_path.stem + ".mp4")
    if mp4_path.exists() and mp4_path.stat().st_mtime >= gif_path.stat().st_mtime:
        return mp4_path
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(gif_path),
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "fps=20,scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(mp4_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"[warn] mp4 convert failed: {gif_path.name}: {exc.stderr.strip()}")
        return None
    return mp4_path


def _badge_html(is_true: bool) -> str:
    cls = "true" if is_true else "false"
    txt = "TRUE" if is_true else "FALSE"
    return f'<span class="badge {cls}">{txt}</span>'


def _config_toggle_html(cfg: dict | None) -> str:
    if not cfg:
        return '<details class="cfg-toggle"><summary>Generated config</summary><pre class="cfg-pre">(missing)</pre></details>'
    body = _esc(json.dumps(cfg, indent=2, ensure_ascii=False))
    return (
        '<details class="cfg-toggle">'
        "<summary>Generated config</summary>"
        f'<pre class="cfg-pre">{body}</pre>'
        "</details>"
    )


def _first_pose_dict(row: dict) -> dict:
    for step in row.get("movements") or []:
        if step.get("type") != "pose":
            continue
        pose = (step.get("parameters") or {}).get("pose") or {}
        if pose.get("dir") and pose.get("gripper_orientation"):
            return pose
    gfp = row.get("gt_fixed_first_pose") or {}
    return gfp if gfp.get("dir") else {}


def _rendered_pose_dict(row: dict) -> dict:
    """Pose used for MuJoCo render (gt_fixed_first_pose or first pose step)."""
    gfp = row.get("gt_fixed_first_pose") or {}
    if gfp.get("dir"):
        return gfp
    return _first_pose_dict(row)


def _pose_index_from_row(row: dict) -> int | None:
    gfp = row.get("gt_fixed_first_pose") or {}
    if gfp.get("pose_id") is not None:
        return int(gfp["pose_id"])
    for step in row.get("movements") or []:
        if step.get("type") != "pose":
            continue
        pose = (step.get("parameters") or {}).get("pose") or {}
        if pose.get("pose_id") is not None:
            return int(pose["pose_id"])
        break
    if row.get("pose_id") is not None:
        return int(row["pose_id"])
    return None


def build(
    *,
    suite: str = "pilot40",
    config: Path | None = None,
    out_html: Path | None = None,
    gif_dir: Path | None = None,
    title: str | None = None,
) -> Path:
    spec = dict(SUITES[suite])
    gif_dir = gif_dir or spec["gif_dir"]
    mp4_dir: Path = spec["mp4_dir"]
    out_html = out_html or spec["out_html"]
    cfg_path: Path = config or spec["cfg"]
    page_title = title or spec["title"]

    cue_alias = {
        "scratch_head_confused": "self_scratch_head_confused",
    }

    cfg_rows = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_by_cue: dict[str, dict] = {cfg.get("cue", ""): cfg for cfg in cfg_rows}

    rows: list[dict[str, str]] = []
    if spec.get("use_config_rows"):
        from pilot90_experiment_suite import pose_generation_correct_any
        from pilot90_paths import load_gt_by_cue, row_generation_done

        cfg_rows = [r for r in cfg_rows if row_generation_done(r)]
        cfg_by_cue = {cfg.get("cue", ""): cfg for cfg in cfg_rows}
        gt_by_cue = load_gt_by_cue()
        for cfg in sorted(cfg_rows, key=lambda r: int(r.get("idx", 0))):
            cue = str(cfg["cue"])
            ev = gt_by_cue.get(cue) or {}
            gt = str(ev.get("pose_gt") or ev.get("groundtruth") or cfg.get("groundtruth") or "")
            gen_pose = _first_pose_dict(cfg)
            ok = pose_generation_correct_any(cfg, gt)
            rows.append(
                {
                    "cue": cue,
                    "cue_idx": str(cfg.get("idx", "")),
                    "groundtruth": gt,
                    "generation": str(gen_pose),
                    "generation_correct": str(ok is True),
                }
            )
    else:
        with spec["tsv"].open("r", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f, delimiter="\t"))

    card_entries: list[tuple[bool, str]] = []
    n_true = 0
    n_total = 0
    n_missing = 0
    n_pose_mismatch = 0

    for row in rows:
        cue = row["cue"]
        resolved_cue = cue_alias.get(cue, cue)
        cue_idx = row["cue_idx"]
        gt = row.get("groundtruth", "")
        generation = row.get("generation", "")
        try:
            gen_obj = ast.literal_eval(generation) if generation else {}
        except (SyntaxError, ValueError):
            gen_obj = {}
        is_true = _to_bool(row.get("generation_correct", ""))
        n_total += 1
        if is_true:
            n_true += 1

        cfg_row = cfg_by_cue.get(resolved_cue) or cfg_by_cue.get(cue) or {}
        cfg_pose = _rendered_pose_dict(cfg_row)
        rendered_dir = cfg_pose.get("dir")
        rendered_ori = cfg_pose.get("gripper_orientation")
        rendered_pose_id = _pose_index_from_row(cfg_row)
        gif_path = _latest_gif_for_cue(resolved_cue, rendered_pose_id, gif_dir=gif_dir)
        mp4_path = _ensure_mp4(gif_path, mp4_dir=mp4_dir) if gif_path else None
        if not gif_path:
            n_missing += 1

        gen_dir = gen_obj.get("dir")
        gen_ori = gen_obj.get("gripper_orientation")
        pose_match = (
            bool(gen_dir)
            and bool(gen_ori)
            and gen_dir == rendered_dir
            and gen_ori == rendered_ori
        )
        if not pose_match:
            n_pose_mismatch += 1

        gif_rel = os.path.relpath(gif_path, out_html.parent) if gif_path else ""
        mp4_rel = os.path.relpath(mp4_path, out_html.parent) if mp4_path else ""

        media_html = (
            f'<img src="{_esc(gif_rel)}" loading="lazy" />'
            if gif_rel
            else '<div class="missing">GIF missing</div>'
        )
        video_html = (
            f'<video controls preload="metadata" src="{_esc(mp4_rel)}"></video>'
            if mp4_rel
            else '<div class="missing">MP4 missing</div>'
        )
        cfg_obj = cfg_row

        card_html = "".join(
                [
                    '<section class="card">',
                    f'<div class="head"><h2>c{_esc(cue_idx)} {_esc(resolved_cue)}</h2>{_badge_html(is_true)}</div>',
                    '<div class="meta">',
                    f'<div><b>GT</b>: {_esc(gt)}</div>',
                    f'<div><b>Generation</b>: {_esc(generation)}</div>',
                    f"<div><b>Rendered start pose</b>: "
                    f"{{'dir': '{_esc(rendered_dir)}', 'gripper_orientation': '{_esc(rendered_ori)}'"
                    + (
                        f", 'pose_id': {int(rendered_pose_id)}"
                        if rendered_pose_id is not None
                        else ""
                    )
                    + "}</div>",
                    (
                        '<div class="warn">Pose mismatch: generation pose and rendered pose are different.</div>'
                        if not pose_match
                        else '<div class="ok">Pose match: generation pose == rendered pose.</div>'
                    ),
                    "</div>",
                    '<div class="media-grid">',
                    f'<figure><figcaption>GIF</figcaption>{media_html}</figure>',
                    f'<figure><figcaption>MP4</figcaption>{video_html}</figure>',
                    "</div>",
                    _config_toggle_html(cfg_obj),
                    "</section>",
                ]
            )
        card_entries.append((is_true, card_html))

    true_cards = [c for ok, c in card_entries if ok]
    false_cards = [c for ok, c in card_entries if not ok]

    acc = (n_true / n_total * 100.0) if n_total else 0.0
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
figure{margin:0}
figcaption{font-size:12px;font-weight:600;margin-bottom:4px}
img,video{width:100%;height:auto;border:1px solid #ccc;border-radius:6px;background:#111}
.missing{border:1px dashed #bbb;border-radius:6px;padding:20px;text-align:center;font-size:12px;color:#666}
.badge{display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;font-weight:700}
.badge.true{background:#e8f8ee;color:#156f37;border:1px solid #b9ebcc}
.badge.false{background:#ffecec;color:#9c1f1f;border:1px solid #ffc6c6}
.warn{margin-top:4px;color:#a52727;font-weight:700}
.ok{margin-top:4px;color:#1b6d37;font-weight:700}
.cfg-toggle{margin-top:10px;border:1px solid #e0e0e0;border-radius:8px;background:#fafafa}
.cfg-toggle summary{cursor:pointer;padding:8px 10px;font-size:12px;font-weight:600;color:#333;user-select:none}
.cfg-toggle summary:hover{background:#f0f0f0}
.cfg-pre{margin:0;padding:10px;font-size:11px;line-height:1.4;overflow:auto;max-height:360px;background:#f7f7f7;border-top:1px solid #e0e0e0;white-space:pre-wrap;word-break:break-word}
.section{margin:28px 0 10px;padding:8px 12px;border-radius:8px;font-size:1.05rem;font-weight:700}
.section.true{background:#e8f8ee;color:#156f37;border:1px solid #b9ebcc}
.section.false{background:#ffecec;color:#9c1f1f;border:1px solid #ffc6c6}
"""

    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Pose Generation Eval Review ({page_title})</title>"
        f"<style>{css}</style></head><body>"
        "<header>"
        f"<h1>Pose Generation Eval Review ({page_title})</h1>"
        f"<div>Generation vs GT: <b>{n_true}/{n_total}</b> ({acc:.1f}%)</div>"
        f"<div>Missing GIFs: {n_missing}</div>"
        f"<div>Generation-vs-rendered pose mismatch: {n_pose_mismatch}</div>"
        "<div>Each card shows full motion (initial pose + tail movement) as GIF and MP4.</div>"
        "<div>Order: <b>TRUE</b> (generation matches human GT) first, then <b>FALSE</b>.</div>"
        "</header>"
        f"<div class='section true'>TRUE — {len(true_cards)} cues</div>"
        f"<div class='grid'>{''.join(true_cards)}</div>"
        f"<div class='section false'>FALSE — {len(false_cards)} cues</div>"
        f"<div class='grid'>{''.join(false_cards)}</div>"
        "</body></html>"
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_doc, encoding="utf-8")
    return out_html


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pose generation GIF/MP4 review HTML")
    p.add_argument("--suite", choices=tuple(SUITES), default="pilot90")
    p.add_argument("--config", type=Path, default=None, help="Motion config JSON (exp1 result)")
    p.add_argument("--out-html", type=Path, default=None)
    p.add_argument("--gif-dir", type=Path, default=None)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--open", action="store_true", help="Open HTML in default browser")
    args = p.parse_args()
    out = build(
        suite=args.suite,
        config=args.config,
        out_html=args.out_html,
        gif_dir=args.gif_dir,
        title=args.title,
    )
    print(f"Wrote {out}")
    if args.open:
        if sys.platform == "darwin":
            subprocess.run(["open", str(out)], check=False)
        elif os.name == "nt":
            os.startfile(str(out))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(out)], check=False)
