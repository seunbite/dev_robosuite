#!/usr/bin/env python3
"""Render core ablation variants (pose+movement GT-correct) and build comparison HTML."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core_ablation import both_correct_rows, build_variants, _variant_still_gt_valid  # noqa: E402
from pilot90_paths import load_gt_by_cue  # noqa: E402
from pilot90_experiment_suite import pose_generation_correct_any  # noqa: E402

EXP1_CFG = _REPO / "data/results/motion_configs/manipulator/exp/result_exp1_gemini-2.5-pro.json"
OUT_ROOT = _REPO / "data/results/render/manipulator/core_ablation_exp1"
CFG_DIR = OUT_ROOT / "configs"
GIF_DIR = OUT_ROOT / "gif"
OUT_HTML = _REPO / "data/results/html/manipulator/core_ablation_exp1.html"
JSONL = _REPO / "data/seed/_remainder/closest_poses_results.jsonl"
ROBOT = "IIWA"
HZ = 10

VARIANTS = (
    ("original", "Original"),
    ("pose_dir", "Direction"),
    ("pose_gripper", "Gripper"),
    ("movement_axis", "Axis"),
    ("movement_rep", "Repetition"),
)


def _esc(x: object) -> str:
    import html

    return html.escape(str(x) if x is not None else "")


def _rel(path: Path | None, html_path: Path) -> str:
    if not path or not path.is_file():
        return ""
    return os.path.relpath(path, html_path.parent).replace(os.sep, "/")


def _pose_index_from_row(row: dict[str, Any]) -> int | None:
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


def render_variant(cfg: dict[str, Any], tag: str, *, force: bool = False) -> Path | None:
    from legacy.motion_generation_core import generate  # noqa: WPS433

    idx = int(cfg["idx"])
    cue = str(cfg["cue"])
    out_gif = GIF_DIR / f"{idx:03d}_{cue}_{tag}.gif"
    if out_gif.is_file() and not force:
        return out_gif

    cfg_path = CFG_DIR / f"{idx:03d}_{cue}_{tag}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps([cfg], indent=2, ensure_ascii=False), encoding="utf-8")

    pose_id = _pose_index_from_row(cfg)
    scratch = OUT_ROOT / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        gif = generate(
            robot=ROBOT,
            cue=cue,
            cue_idx=idx,
            pose_index=pose_id,
            jsonl_path=str(JSONL),
            config_path=str(cfg_path),
            output_dir=str(scratch),
            hz=HZ,
            top_k=1,
            gif_filename_suffix=f"_{tag}",
        )
    except Exception as exc:
        print(f"[render fail] {cue} {tag}: {exc}", flush=True)
        return None
    if not gif:
        return None
    src = Path(gif)
    if src.is_file():
        out_gif.parent.mkdir(parents=True, exist_ok=True)
        out_gif.write_bytes(src.read_bytes())
        return out_gif
    return None


def _short_grip(g: str) -> str:
    return {"horizontal": "h", "vertical": "v"}.get(str(g).strip(), str(g).strip()[:1])


def _short_pose_gt(pose_gt: str) -> str:
    s = str(pose_gt or "").strip()
    s = re.sub(r"\bhorizontal\b", "h", s)
    s = re.sub(r"\bvertical\b", "v", s)
    return re.sub(r"\s+", " ", s)


def _short_mov_gt(gt_row: dict[str, Any]) -> str:
    return str((gt_row.get("movement_gt") or {}).get("annotation_raw") or "").strip()


def _step_axis_sign(cfg: dict[str, Any], step_idx: int | None, axis: str) -> str:
    if step_idx is None or not axis:
        return ""
    movements = cfg.get("movements") or []
    if step_idx < 0 or step_idx >= len(movements):
        return ""
    dirs = ((movements[step_idx].get("parameters") or {}).get("directions") or [])
    val: float | None = None
    for d in dirs:
        v = (d.get("degrees") or {}).get(axis)
        if isinstance(v, (int, float)):
            val = float(v)
    if val is None:
        return ""
    return "+" if val > 0 else "-"


def _is_still_gt(cfg_obj: dict[str, Any] | None, gt_row: dict[str, Any]) -> bool:
    if not isinstance(cfg_obj, dict):
        return False
    ab = cfg_obj.get("ablation") or {}
    if not ab.get("kind"):
        return False
    if "still_gt_valid" in ab:
        return bool(ab["still_gt_valid"])
    return _variant_still_gt_valid(cfg_obj, gt_row)


def _delta_note(
    vkey: str,
    cfg_obj: dict[str, Any] | None,
    gt_row: dict[str, Any],
    orig_cfg: dict[str, Any] | None,
) -> str:
    pose_gt = _short_pose_gt(str(gt_row.get("pose_gt") or gt_row.get("groundtruth") or ""))
    mov_gt = _short_mov_gt(gt_row)
    if vkey == "original":
        return f"pose {pose_gt}<br>mov {mov_gt}"

    if not isinstance(cfg_obj, dict):
        return ""
    ab = cfg_obj.get("ablation") or {}
    kind = ab.get("kind")
    comp = (gt_row.get("movement_gt") or {}).get("component") or {}

    if kind == "pose_dir":
        d0 = str(ab.get("from", ""))
        d1 = str(ab.get("to", ""))
        if orig_cfg is not None:
            i = ab.get("step_idx")
            if isinstance(i, int):
                p = ((orig_cfg.get("movements") or [])[i].get("parameters") or {}).get("pose") or {}
                g = _short_grip(str(p.get("gripper_orientation", "")))
                return f"({d0},{g}) → ({d1},{g})"
        return f"{d0} → {d1}"

    if kind == "pose_gripper":
        g0 = _short_grip(str(ab.get("from", "")))
        g1 = _short_grip(str(ab.get("to", "")))
        d = ""
        if orig_cfg is not None:
            i = ab.get("step_idx")
            if isinstance(i, int):
                p = ((orig_cfg.get("movements") or [])[i].get("parameters") or {}).get("pose") or {}
                d = str(p.get("dir", ""))
        if d:
            return f"({d},{g0}) → ({d},{g1})"
        return f"{g0} → {g1}"

    if kind == "movement_axis":
        gt_ax = str(ab.get("true_axis") or "")
        neg_ax = str(ab.get("neg_axis") or "")
        gt_sign = str((comp.get("axes") or {}).get(gt_ax) or _step_axis_sign(orig_cfg or {}, ab.get("step_idx"), gt_ax))
        neg_sign = _step_axis_sign(cfg_obj, ab.get("step_idx"), neg_ax)
        if gt_sign in ("+", "-") and neg_sign in ("+", "-"):
            return f"{gt_ax}{gt_sign} → {neg_ax}{neg_sign}"
        if gt_ax and neg_ax:
            return f"{gt_ax} → {neg_ax}"
        return ""

    if kind == "movement_rep":
        gt_rep = str(comp.get("repetition") or "?")
        override = str(ab.get("rep_override") or "?")
        return f"{gt_rep} → {override}"

    return ""


def _config_toggle(cfg: dict[str, Any]) -> str:
    body = json.dumps(cfg, indent=2, ensure_ascii=False)
    return (
        '<details class="cfg-toggle">'
        "<summary>Config</summary>"
        f'<pre class="cfg-pre">{_esc(body)}</pre>'
        "</details>"
    )


def build_html(rows: list[dict[str, Any]], media: dict[str, dict[str, Any]], gt: dict[str, Any]) -> Path:
    cards: list[str] = []
    for row in rows:
        idx = int(row["idx"])
        cue = str(row["cue"])
        key = f"{idx}:{cue}"
        gt_row = gt.get(cue) or {}
        orig_cfg = media.get(key, {}).get("original_cfg")
        if not isinstance(orig_cfg, dict):
            orig_cfg = row
        cells: list[str] = []
        for vkey, label in VARIANTS:
            gif = media.get(key, {}).get(vkey)
            src = _rel(gif, OUT_HTML) if isinstance(gif, Path) else ""
            vid = (
                f'<img src="{_esc(src)}" loading="lazy" alt="{_esc(label)}" />'
                if src
                else '<div class="missing">missing</div>'
            )
            cfg_obj = media.get(key, {}).get(f"{vkey}_cfg")
            delta = _delta_note(vkey, cfg_obj if isinstance(cfg_obj, dict) else None, gt_row, orig_cfg)
            gt_tag = (
                '<div class="gt-tag">(GT)</div>'
                if vkey != "original" and _is_still_gt(cfg_obj if isinstance(cfg_obj, dict) else None, gt_row)
                else ""
            )
            note = f'<div class="note">{gt_tag}<code>{delta}</code></div>' if delta or gt_tag else ""
            toggle = _config_toggle(cfg_obj) if isinstance(cfg_obj, dict) else ""
            cells.append(
                f'<div class="cell"><div class="label">{_esc(label)}</div>{vid}{note}{toggle}</div>'
            )
        pose_gt = _short_pose_gt(str(gt_row.get("pose_gt") or gt_row.get("groundtruth") or ""))
        mov_gt = _short_mov_gt(gt_row)
        cards.append(
            f'<article class="row-card">'
            f"<h2>c{idx} {_esc(cue)}</h2>"
            f'<p class="meta">pose GT: <code>{_esc(pose_gt)}</code> · mov GT: <code>{_esc(mov_gt)}</code></p>'
            f'<div class="row">{"".join(cells)}</div>'
            "</article>"
        )

    css = """
body{font-family:system-ui,sans-serif;margin:20px;max-width:100%;background:#f5f7fb}
header{background:#fff;padding:14px 18px;border-radius:10px;margin-bottom:18px;border:1px solid #dce1ea}
.row-card{background:#fff;border:1px solid #dce1ea;border-radius:10px;padding:14px;margin-bottom:18px}
.row{display:grid;grid-template-columns:repeat(5,minmax(180px,1fr));gap:10px;overflow-x:auto}
.cell{background:#fafafa;border:1px solid #e6e6e6;border-radius:8px;padding:8px}
.label{font-size:12px;font-weight:700;margin-bottom:6px;text-align:center}
img{width:100%;height:auto;border-radius:6px;background:#111;border:1px solid #ccc}
.missing{border:1px dashed #bbb;border-radius:6px;padding:24px 8px;text-align:center;font-size:11px;color:#666}
.meta{font-size:12px;color:#444;margin:4px 0 10px}
.note{font-size:11px;color:#333;margin-top:6px;line-height:1.35}
.gt-tag{font-size:10px;font-weight:700;color:#b45309;margin-bottom:3px}
.note code{background:#eef2f8;padding:2px 5px;border-radius:4px;font-size:10px;white-space:pre-wrap}
.cfg-toggle{margin-top:8px;border:1px solid #e0e0e0;border-radius:6px;background:#fff}
.cfg-toggle summary{cursor:pointer;padding:6px 8px;font-size:11px;font-weight:600}
.cfg-pre{margin:0;padding:8px;font-size:10px;line-height:1.35;overflow:auto;max-height:200px;background:#f7f7f7;border-top:1px solid #e0e0e0;white-space:pre-wrap}
"""
    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Core ablation — pose+movement GT correct</title>"
        f"<style>{css}</style></head><body>"
        "<header>"
        "<h1>Core ablation (manipulator exp1 — pose &amp; movement both GT-correct)</h1>"
        f"<p>{len(rows)} cues · columns: Original → Direction → Gripper → Axis → Repetition</p>"
        "<p>Only GT-matching pose / movement steps are perturbed. "
        "<span class='gt-tag'>(GT)</span> = perturbation still within GT.</p>"
        "</header>"
        f"{''.join(cards)}"
        "</body></html>"
    )
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(doc, encoding="utf-8")
    return OUT_HTML


def main() -> None:
    p = argparse.ArgumentParser(description="Core ablation render + HTML")
    p.add_argument("--config", type=Path, default=EXP1_CFG)
    p.add_argument("--max-cues", type=int, default=0, help="0 = all")
    p.add_argument("--force-render", action="store_true")
    p.add_argument("--html-only", action="store_true")
    p.add_argument("--open", action="store_true")
    args = p.parse_args()

    gt = load_gt_by_cue()
    rows = both_correct_rows(args.config, gt, pose_ok_fn=pose_generation_correct_any)
    if args.max_cues > 0:
        rows = rows[: args.max_cues]
    print(f"[ablation] {len(rows)} cues with pose+movement GT match", flush=True)

    media: dict[str, dict[str, Any]] = {}
    if not args.html_only:
        for row in rows:
            cue = str(row["cue"])
            idx = int(row["idx"])
            key = f"{idx}:{cue}"
            gt_row = gt.get(cue) or {}
            variants = build_variants(row, gt_row)
            media[key] = {}
            for vkey, _label in VARIANTS:
                cfg = variants.get(vkey)
                if cfg is None:
                    media[key][vkey] = None
                    media[key][f"{vkey}_cfg"] = None
                    print(f"[skip] {cue} {vkey}: variant unavailable", flush=True)
                    continue
                gif = render_variant(cfg, vkey, force=args.force_render)
                media[key][vkey] = gif
                media[key][f"{vkey}_cfg"] = cfg
                print(f"[render] {cue} {vkey} -> {gif}", flush=True)

        manifest = OUT_ROOT / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    k: {vk: str(v) if isinstance(v, Path) else v for vk, v in vv.items()}
                    for k, vv in media.items()
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    elif args.html_only:
        for row in rows:
            cue = str(row["cue"])
            idx = int(row["idx"])
            key = f"{idx}:{cue}"
            media[key] = {}
            for vkey, _ in VARIANTS:
                gif = GIF_DIR / f"{idx:03d}_{cue}_{vkey}.gif"
                media[key][vkey] = gif if gif.is_file() else None
                cfg_path = CFG_DIR / f"{idx:03d}_{cue}_{vkey}.json"
                if cfg_path.is_file():
                    media[key][f"{vkey}_cfg"] = json.loads(cfg_path.read_text(encoding="utf-8"))[0]
                else:
                    media[key][f"{vkey}_cfg"] = None

    html = build_html(rows, media, gt)
    print(f"Wrote {html}", flush=True)
    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(html)], check=False)


if __name__ == "__main__":
    main()
