#!/usr/bin/env python3
"""Rebuild manipulator exp1 review HTML: render + corr1 GIFs, config per column."""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE, _REPO / "adhoc/generation/tiago"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build_pose_generation_eval_review_html import (  # noqa: E402
    _badge_html,
    _first_pose_dict,
    _rendered_pose_dict,
)
from correction_lib import (  # noqa: E402
    config_snapshot_path,
    correction_cue_dir,
    correction_record_path,
    default_config_path,
    load_config_list,
    load_correction_record,
)
from correction_prompts import correction_indicates_verified_ok  # noqa: E402
from pilot90_experiment_suite import pose_generation_correct_any  # noqa: E402
from pilot90_paths import load_gt_by_cue, row_generation_done  # noqa: E402
from render_paths import manipulator_cue_render_dir, render_media_paths  # noqa: E402

DEFAULT_MODEL_TAG = "gemini-2.5-pro"
DEFAULT_EXP_ID = "1"
OUT_HTML = _REPO / "data/results/html/manipulator/exp1_gemini-2.5-pro_rendered.html"
VISUAL_LABELS_DIR = _REPO / "data/results/visualize/manipulator/correction/exp1"


def _esc(s: object) -> str:
    return html.escape(str(s) if s is not None else "")


def _rel(out: Path, p: Path | None) -> str:
    if p is None or not p.is_file():
        return ""
    rel = os.path.relpath(p, out.parent).replace("\\", "/")
    return _esc(f"{rel}?v={int(p.stat().st_mtime)}")


def _config_display_row(row: dict | None) -> dict:
    if not row:
        return {}
    keys = ("idx", "cue", "description", "movements", "state")
    return {k: row[k] for k in keys if row.get(k) is not None}


def _display_q_text(text: str | None, *, q_num: int) -> str:
    if not text:
        return ""
    s = str(text).strip()
    for prefix in (
        f"Correction Q{q_num}:",
        f"correction q{q_num}:",
        f"Q{q_num}:",
        f"q{q_num}:",
    ):
        if s.lower().startswith(prefix.lower()):
            return s[len(prefix) :].strip()
    return s


def _config_json_html(row: dict | None, *, summary: str = "Config") -> str:
    payload = _config_display_row(row)
    if not payload:
        return '<div class="cfg-missing">Config missing</div>'
    body = _esc(json.dumps(payload, indent=2, ensure_ascii=False))
    return (
        f'<details class="cfg-toggle">'
        f"<summary>{_esc(summary)}</summary>"
        f"<pre>{body}</pre>"
        f"</details>"
    )


def _corr_badge(action: str | None) -> tuple[str, str]:
    act = action or "pending"
    if act == "verified_ok":
        return "Verified", "badge-verified"
    if act == "corrected":
        return "Edited", "badge-edited"
    if act == "pending":
        return "Pending", "badge-pending"
    return act, "badge-other"


def _effective_action(action: str | None, correction: dict | None) -> str:
    act = action or "pending"
    if act == "corrected" and correction and correction_indicates_verified_ok(correction):
        return "verified_ok"
    return act


def _latest_gif_in(dir_path: Path) -> Path | None:
    iwa = dir_path / "IIWA"
    if not iwa.is_dir():
        return None
    cands = sorted(iwa.glob("*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _render_gif(idx: int, cue: str, model_tag: str, exp_id: str) -> Path | None:
    gif, _, _ = render_media_paths("manipulator", model_tag, exp_id, idx, cue, "gen")
    return gif if gif.is_file() else _latest_gif_in(
        manipulator_cue_render_dir(model_tag, exp_id, cue, "new_render")
    )


def _corr_gif(idx: int, cue: str, model_tag: str, exp_id: str, round_n: int = 1) -> Path | None:
    role = "corr" if round_n == 1 else f"corr_r{round_n}"
    gif, _, _ = render_media_paths("manipulator", model_tag, exp_id, idx, cue, role)
    if gif.is_file():
        return gif
    return _latest_gif_in(manipulator_cue_render_dir(model_tag, exp_id, cue, "correction"))


def _corr1_gif(idx: int, cue: str, model_tag: str, exp_id: str) -> Path | None:
    return _corr_gif(idx, cue, model_tag, exp_id, 1)


def _load_corr_row(model_tag: str, exp_id: str, cue: str) -> dict | None:
    snap = config_snapshot_path("manipulator", exp_id, model_tag, cue, 1)  # type: ignore[arg-type]
    if not snap.is_file():
        return None
    data = json.loads(snap.read_text(encoding="utf-8"))
    return data[0] if isinstance(data, list) else data


def _load_visual_labels(model_tag: str) -> dict[str, dict[str, str]]:
    path = VISUAL_LABELS_DIR / model_tag / "visual_outcome_labels.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): v for k, v in raw.items()}


def _visual_orig_pass(visual_labels: dict[str, dict[str, str]], idx: int) -> bool | None:
    entry = visual_labels.get(str(idx))
    if not entry:
        return None
    sym = str(entry.get("orig", "")).strip().lower()
    if sym == "o":
        return True
    if sym == "x":
        return False
    return None


def _fig(
    out: Path,
    title: str,
    gif: Path | None,
    config_row: dict | None,
    *,
    q1: str | None = None,
    q2: str | None = None,
) -> str:
    gif_r = _rel(out, gif)
    if gif_r:
        body = f'<img src="{gif_r}" loading="lazy" alt="{_esc(title)}">'
    else:
        body = '<div class="missing">GIF missing</div>'
    body += _config_json_html(config_row)
    if q1:
        body += f'<div class="vq"><b>Q1:</b> {_esc(_display_q_text(q1, q_num=1))}</div>'
    if q2 is not None:
        body += f'<div class="vq"><b>Q2:</b> {_esc(_display_q_text(q2, q_num=2))}</div>'
    return f"<figure><figcaption>{_esc(title)}</figcaption>{body}</figure>"


def build(
    *,
    model_tag: str = DEFAULT_MODEL_TAG,
    exp_id: str = DEFAULT_EXP_ID,
    out_html: Path = OUT_HTML,
) -> Path:
    cfg_path = default_config_path("manipulator", model_tag, exp_id)  # type: ignore[arg-type]
    cfg_rows = [r for r in load_config_list(cfg_path) if row_generation_done(r)]
    gt_by_cue = load_gt_by_cue()
    visual_labels = _load_visual_labels(model_tag)

    pass_cards: list[str] = []
    fail_cards: list[str] = []
    n_gt_true = 0
    n_visual_pass = 0
    n_render = 0
    n_corr_done = 0

    for cfg in sorted(cfg_rows, key=lambda r: int(r.get("idx", 0))):
        cue = str(cfg.get("cue", ""))
        idx = int(cfg.get("idx", 0))
        gt_row = gt_by_cue.get(cue) or {}
        gt = str(gt_row.get("pose_gt") or gt_row.get("groundtruth") or "")
        gen_pose = _first_pose_dict(cfg)
        is_true = pose_generation_correct_any(cfg, gt) is True
        if is_true:
            n_gt_true += 1
        visual_pass = _visual_orig_pass(visual_labels, idx)
        if visual_pass is True:
            n_visual_pass += 1
        elif visual_pass is None:
            visual_pass = is_true

        render_gif = _render_gif(idx, cue, model_tag, exp_id)
        if render_gif:
            n_render += 1

        rec_path = correction_record_path(
            correction_cue_dir("manipulator", exp_id, model_tag, cue),  # type: ignore[arg-type]
            1,
        )
        corr_gif = _corr1_gif(idx, cue, model_tag, exp_id)
        action = "pending"
        q1 = q2 = None
        corr_row = None
        if rec_path.is_file() and corr_gif:
            cue_dir = correction_cue_dir("manipulator", exp_id, model_tag, cue)  # type: ignore[arg-type]
            rec = load_correction_record(cue_dir, 1)
            corr = rec.get("correction") or {}
            action = _effective_action(str(rec.get("action") or ""), corr)
            q1 = corr.get("correction_q1") or (rec.get("output_row") or {}).get("correction_q1")
            q2 = corr.get("correction_q2") or (rec.get("output_row") or {}).get("correction_q2")
            corr_row = _load_corr_row(model_tag, exp_id, cue) or rec.get("output_row")
            n_corr_done += 1

        corr_label, corr_cls = _corr_badge(action)
        rendered = _rendered_pose_dict(cfg)

        card = "".join(
            [
                '<section class="card">',
                f'<div class="head"><h2>c{idx} <code>{_esc(cue)}</code></h2>'
                f'<div class="badges">{_badge_html(is_true)}'
                f'<span class="badge {corr_cls}" title="Correction">{_esc(corr_label)}</span></div></div>',
                '<div class="meta">',
                f"<div><b>GT</b>: {_esc(gt)}</div>",
                f"<div><b>Generation pose</b>: {_esc(gen_pose)}</div>",
                f"<div><b>Rendered start pose</b>: {_esc(rendered)}</div>",
                "</div>",
                '<div class="media-grid">',
                _fig(out_html, "Render", render_gif, cfg),
                _fig(
                    out_html,
                    "Correction 1",
                    corr_gif if action != "pending" else None,
                    corr_row,
                    q1=q1 if action != "pending" else None,
                    q2=q2 if action != "pending" else None,
                ),
                "</div>",
                "</section>",
            ]
        )
        (pass_cards if visual_pass else fail_cards).append(card)

    n_total = len(cfg_rows)
    gt_acc = (n_gt_true / n_total * 100.0) if n_total else 0.0
    visual_acc = (n_visual_pass / n_total * 100.0) if n_total else 0.0

    css = """
body { font-family: system-ui, sans-serif; margin: 20px; background: #f6f7fb; color: #1a1a1a; max-width: 1800px; }
header { background: #fff; border: 1px solid #dde3ee; border-radius: 10px; padding: 14px 16px; margin-bottom: 18px; }
h1 { margin: 0 0 6px; font-size: 1.25rem; }
.stats { color: #555; font-size: 0.9rem; line-height: 1.5; }
.legend { color: #666; font-size: 0.85rem; margin-top: 8px; }
.section { margin: 28px 0 10px; padding: 8px 12px; border-radius: 8px; font-size: 1.05rem; font-weight: 700; }
.section.true { background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }
.section.false { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr)); gap: 14px; }
.card { background: #fff; border: 1px solid #dde3ee; border-radius: 10px; padding: 14px; }
.head { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; flex-wrap: wrap; }
.head h2 { margin: 0; font-size: 1rem; }
.badges { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.meta { font-size: 0.82rem; color: #555; margin: 8px 0 10px; line-height: 1.45; }
.media-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
@media (max-width: 720px) { .media-grid { grid-template-columns: 1fr; } }
figure { margin: 0; }
figcaption { font-size: 0.8rem; color: #666; margin-bottom: 4px; font-weight: 600; }
img { width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; background: #111; }
.missing { color: #c00; font-size: 0.85rem; padding: 16px; text-align: center; border: 1px dashed #ddd; border-radius: 4px; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }
.badge.true { background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }
.badge.false { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
.badge-edited { background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }
.badge-verified { background: #ede9fe; color: #5b21b6; border: 1px solid #c4b5fd; }
.badge-pending { background: #e5e7eb; color: #374151; border: 1px solid #d1d5db; }
.cfg-toggle { margin-top: 8px; font-size: 0.8rem; }
.cfg-toggle summary { cursor: pointer; color: #444; font-weight: 600; user-select: none; }
.cfg-toggle summary:hover { color: #1e40af; }
.cfg-toggle pre {
  margin: 6px 0 0; padding: 8px; background: #f7f8fc; border: 1px solid #e8ecf4;
  border-radius: 6px; font-size: 0.72rem; line-height: 1.35; white-space: pre-wrap;
  word-break: break-word; max-height: 280px; overflow: auto;
}
.cfg-missing { color: #999; font-size: 0.78rem; margin-top: 6px; }
.vq { font-size: 0.82rem; margin-top: 10px; line-height: 1.4; }
"""

    doc = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Manipulator exp1 {model_tag} — render + correction</title>"
        f"<style>{css}</style></head><body>"
        "<header>"
        f"<h1>Manipulator exp1 — render + correction ({model_tag})</h1>"
        f'<div class="stats">Visual pass (orig render): <b>{n_visual_pass}/{n_total}</b> ({visual_acc:.1f}%) · '
        f"Generation vs GT: <b>{n_gt_true}/{n_total}</b> ({gt_acc:.1f}%) · "
        f"Render: {n_render}/{n_total} · Correction 1: {n_corr_done}</div>"
        '<div class="legend">Sections = manual visual pass on <b>Render</b> · '
        "Card badges: left = pose GT, right = correction status. "
        "Columns: <b>Render</b> = exp1 config + xyz-trim (hz=20) · "
        "<b>Correction 1</b> = corr1 when available.</div>"
        "</header>"
        f"<div class='section true'>PASS — {len(pass_cards)} cues</div>"
        f"<div class='grid'>{''.join(pass_cards)}</div>"
        f"<div class='section false'>FAIL — {len(fail_cards)} cues</div>"
        f"<div class='grid'>{''.join(fail_cards)}</div>"
        "</body></html>"
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(doc, encoding="utf-8")
    return out_html


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Manipulator exp1 render + correction HTML")
    ap.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    ap.add_argument("--exp", default=DEFAULT_EXP_ID)
    ap.add_argument("--out-html", type=Path, default=OUT_HTML)
    args = ap.parse_args()
    path = build(model_tag=args.model_tag, exp_id=args.exp, out_html=args.out_html)
    print(f"Wrote {path}")
