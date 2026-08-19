#!/usr/bin/env python3
"""Side-by-side exp7 vs exp7.1 review HTML (GIF + fixed-pose PNG input)."""
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

from pilot40_paths import HTML_EXP_DIR, load_gt_by_cue, manifest90_cue_names  # noqa: E402
from score_mobile_motion_gt import mobile_tail_steps  # noqa: E402

POSE_KEYS = ("torso_height", "arm_position", "gripper_orientation", "head", "left_arm", "x", "y", "z")

CSS = """
body{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e8eaed}
h1{padding:16px 24px;margin:0;font-size:1.25rem;border-bottom:1px solid #333}
.meta{padding:12px 24px;color:#9aa0a6;font-size:0.9rem;border-bottom:1px solid #222}
.card{margin:20px 24px;padding:14px;border:1px solid #333;border-radius:10px;background:#171a21}
.card h2{margin:0 0 8px;font-size:1rem}
.card p.desc{margin:0 0 8px;color:#9aa0a6;font-size:0.85rem}
.cols{display:grid;grid-template-columns:minmax(180px,240px) 1fr 1fr;gap:12px;align-items:start}
@media(max-width:900px){.cols{grid-template-columns:1fr}}
.col label{display:block;font-size:0.75rem;color:#9aa0a6;margin-bottom:4px;text-transform:uppercase;letter-spacing:.04em}
.col img{width:100%;height:auto;border-radius:6px;background:#000;display:block}
pre{white-space:pre-wrap;background:#0f1115;padding:10px;border-radius:6px;font-size:0.72rem;overflow:auto;margin-top:10px}
.ok{color:#81c995;font-weight:600}.miss{color:#f28b82;font-weight:600}.na{color:#888}
.agree{color:#8ab4f8}.split{color:#fdd663}
.missing{color:#f28b82;font-size:0.8rem}
"""


def _esc(x: object) -> str:
    return html.escape(str(x) if x is not None else "")


def _rel(out_html: Path, asset: Path) -> str:
    if not asset.is_file():
        return ""
    return Path(os.path.relpath(asset.resolve(), out_html.parent.resolve())).as_posix()


def _badge(val: bool | None) -> str:
    if val is True:
        return '<span class="ok">OK</span>'
    if val is False:
        return '<span class="miss">NO</span>'
    return '<span class="na">—</span>'


def _agree_badge(a: bool | None, b: bool | None) -> str:
    if a is None or b is None:
        return '<span class="na">—</span>'
    if a == b:
        return f'<span class="agree">both {"OK" if a else "NO"}</span>'
    return '<span class="split">split</span>'


def _fmt_pose(pose: dict) -> str:
    return ", ".join(f"{k}={pose[k]}" for k in POSE_KEYS if k in pose)


def _stem(row: dict) -> str:
    idx = int(row.get("idx", row.get("cue_idx", 0)))
    cue = str(row["cue"]).replace("/", "_").replace("\\", "_").replace(" ", "_")
    return f"mm19_g{idx:02d}_{cue}"


def _fixed_pose(row: dict) -> dict:
    for step in row.get("movements") or []:
        if step.get("type") == "pose":
            return (step.get("parameters") or {}).get("pose") or {}
    return row.get("gt_fixed_first_pose") or {}


def build_compare_html(
    *,
    cfg7: Path,
    cfg71: Path,
    score7: Path,
    score71: Path,
    gif7_dir: Path,
    gif71_dir: Path,
    pose_png_dir: Path,
    out_html: Path,
    n_cues: int = 40,
) -> Path:
    rows7 = {r["cue"]: r for r in json.loads(cfg7.read_text(encoding="utf-8")) if r.get("cue")}
    rows71 = {r["cue"]: r for r in json.loads(cfg71.read_text(encoding="utf-8")) if r.get("cue")}
    sc7 = {r["cue"]: r for r in json.loads(score7.read_text(encoding="utf-8")).get("rows", [])}
    sc71 = {r["cue"]: r for r in json.loads(score71.read_text(encoding="utf-8")).get("rows", [])}
    gt_by = load_gt_by_cue()

    s7 = json.loads(score7.read_text(encoding="utf-8"))
    s71 = json.loads(score71.read_text(encoding="utf-8"))
    meta = (
        f"exp7 {s7.get('n_correct')}/{s7.get('n')} · "
        f"exp7.1 {s71.get('n_correct')}/{s71.get('n')} · "
        f"gen exp7 {len(rows7)}/{n_cues} · exp7.1 {len(rows71)}/{n_cues}"
    )

    cards: list[str] = []
    both_ok = both_no = split = 0
    for cue in manifest90_cue_names()[:n_cues]:
        r7 = rows7.get(cue)
        r71 = rows71.get(cue)
        if not r7 and not r71:
            cards.append(
                f'<section class="card"><h2>{_esc(cue)} <span class="miss">MISSING</span></h2>'
                f'<p class="desc">generation failed (beckon)</p></section>'
            )
            continue

        row = r71 or r7 or {}
        idx = int(row.get("idx", 0))
        gt_row = gt_by.get(cue, {})
        pose_gt = str(gt_row.get("pose_gt") or gt_row.get("groundtruth") or "")
        fixed = _fixed_pose(row)
        mov_gt = _esc((sc7.get(cue) or sc71.get(cue) or {}).get("annotation_raw", ""))

        m7 = (sc7.get(cue) or {}).get("component_match")
        m71 = (sc71.get(cue) or {}).get("component_match")
        if m7 is not None and m71 is not None:
            if m7 == m71:
                both_ok += int(m7)
                both_no += int(not m7)
            else:
                split += 1

        png = pose_png_dir / f"g{idx:02d}_{cue}.png"
        gif7 = gif7_dir / f"{_stem(row)}.gif" if row else Path()
        gif71 = gif71_dir / f"{_stem(row)}.gif" if row else Path()
        png_rel = _rel(out_html, png)
        gif7_rel = _rel(out_html, gif7) if r7 else ""
        gif71_rel = _rel(out_html, gif71) if r71 else ""

        def _img_col(label: str, rel: str, *, missing: str) -> str:
            body = (
                f'<img src="{_esc(rel)}" loading="lazy" alt="{_esc(label)}">'
                if rel
                else f'<p class="missing">{missing}</p>'
            )
            return f'<div class="col"><label>{_esc(label)}</label>{body}</div>'

        tail7 = mobile_tail_steps((r7 or {}).get("movements") or []) if r7 else []
        tail71 = mobile_tail_steps((r71 or {}).get("movements") or []) if r71 else []

        cards.append(
            f'<section class="card"><h2>g{idx:02d} · {_esc(cue)} '
            f'exp7 {_badge(m7)} · exp7.1 {_badge(m71)} · {_agree_badge(m7, m71)}</h2>'
            f'<p class="desc"><b>pose_gt:</b> {_esc(pose_gt)}</p>'
            f'<p class="desc"><b>fixed pose (mobile):</b> {_esc(_fmt_pose(fixed))}</p>'
            f'<p class="desc"><b>movement_gt:</b> {mov_gt}</p>'
            f'<div class="cols">'
            f'{_img_col("Fixed pose (VLM input)", png_rel, missing="No PNG")}'
            f'{_img_col("exp7 GIF", gif7_rel, missing="No exp7 GIF")}'
            f'{_img_col("exp7.1 GIF", gif71_rel, missing="No exp7.1 GIF")}'
            f'</div>'
            f'<details><summary style="cursor:pointer;color:#9aa0a6;font-size:0.85rem;margin-top:10px">Tail JSON</summary>'
            f'<p class="desc"><b>exp7</b></p><pre>{_esc(json.dumps(tail7, indent=2, ensure_ascii=False))}</pre>'
            f'<p class="desc"><b>exp7.1</b></p><pre>{_esc(json.dumps(tail71, indent=2, ensure_ascii=False))}</pre>'
            f'</details></section>'
        )

    summary = (
        f"agreement: both OK {both_ok} · both NO {both_no} · split {split}"
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'/>"
        f"<title>Exp7 vs 7.1 compare</title><style>{CSS}</style></head><body>"
        f"<h1>Exp7 vs Exp7.1 — side-by-side (pilot{n_cues})</h1>"
        f"<div class='meta'>{_esc(meta)} · {_esc(summary)}</div>"
        f"{''.join(cards)}</body></html>",
        encoding="utf-8",
    )
    print(f"[html] wrote {out_html}")
    return out_html


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="pilot40")
    p.add_argument("--n-cues", type=int, default=40)
    p.add_argument("--open", action="store_true", default=True)
    args = p.parse_args()
    suf = args.suffix
    exp = _REPO / "data/results"
    out = build_compare_html(
        cfg7=exp / f"motion_configs/tiago/exp/result_exp7_{suf}.json",
        cfg71=exp / f"motion_configs/tiago/exp/result_exp7_1_{suf}.json",
        score7=exp / f"verify/tiago/exp/score_exp7_{suf}.json",
        score71=exp / f"verify/tiago/exp/score_exp7_1_{suf}.json",
        gif7_dir=exp / "render/tiago" / suf,
        gif71_dir=exp / "render/tiago" / f"{suf}_exp7_1",
        pose_png_dir=exp / "render/tiago/fixed_pose_png_tiago_gt",
        out_html=HTML_EXP_DIR / f"exp7_vs_7_1_{suf}.html",
        n_cues=args.n_cues,
    )
    if args.open:
        try:
            subprocess.run(["open", str(out)], check=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
