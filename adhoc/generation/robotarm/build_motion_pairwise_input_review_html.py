#!/usr/bin/env python3
"""HTML review: pilot-90 unified pairwise inputs (axis/joint/gt_side)."""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from motion_pairwise_media import MOTION_PAIRWISE_DIR, PAIRWISE_SPECS_NAME  # noqa: E402

OUT_HTML = MOTION_PAIRWISE_DIR / "pairwise_input_review.html"


def _esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def build_review_html(
    *,
    spec_path: Path,
    out_path: Path,
    title: str | None = None,
) -> Path:
    if not spec_path.is_file():
        raise SystemExit(f"Missing {spec_path} — run prepare_pilot90_motion_pairwise_mp4.py --force")

    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    rows = payload.get("mp4") or []
    n_left = sum(1 for r in rows if r.get("gt_side") == "left")
    n_same = sum(1 for r in rows if r.get("same_joint"))
    n_gen = sum(1 for r in rows if r.get("start_pose_source") == "generation")
    n_human = sum(1 for r in rows if r.get("start_pose_source") == "human_gt_tile_pick")

    sections: list[str] = []
    for e in rows:
        idx = int(e["idx"])
        cue = str(e["cue"])
        mp4_name = Path(str(e.get("pair_mp4", ""))).name or f"{idx:03d}_{cue}_pair_axis.mp4"
        local = MOTION_PAIRWISE_DIR / mp4_name
        gt_side = str(e.get("gt_side", "?"))
        ans_badge = f"ANSWER = {gt_side.upper()} (GT)"
        media = (
            f'<video class="media" src="{_esc(mp4_name)}" controls loop muted playsinline></video>'
            if local.is_file()
            else f'<div class="media missing">MP4 missing: {_esc(mp4_name)}</div>'
        )
        sections.append(
            f"""
<section id="c{idx}">
  <h2>{idx}. {_esc(cue)}</h2>
  <p class="answer">{ans_badge}</p>
  <table>
    <tr><th>GT axis</th><td>{_esc(e.get('true_axis'))}</td><th>neg axis</th><td>{_esc(e.get('neg_axis'))}</td></tr>
    <tr><th>GT joint</th><td>{_esc(e.get('gt_joint'))}</td><th>neg joint</th><td>{_esc(e.get('neg_joint'))}</td></tr>
    <tr><th>same joint?</th><td colspan="3">{_esc(e.get('same_joint'))}</td></tr>
    <tr><th>start pose</th><td colspan="3">{_esc(e.get('start_pose_source'))} — {_esc(e.get('start_pose_dir'))}, {_esc(e.get('start_pose_gripper'))} (pose_id={_esc(e.get('start_pose_id'))})</td></tr>
    <tr><th>neg variant</th><td colspan="3">{_esc((e.get('neg_axis_meta') or {}).get('variant', 'single_axis_swap'))}</td></tr>
    <tr><th>LEFT panel</th><td colspan="3">{_esc(e.get('left_tail_summary'))} <span class="dim">({_esc(e.get('left'))})</span></td></tr>
    <tr><th>RIGHT panel</th><td colspan="3">{_esc(e.get('right_tail_summary'))} <span class="dim">({_esc(e.get('right'))})</span></td></tr>
  </table>
  <div class="grid2">
    <div>{media}</div>
    <div><details open><summary>Prompt (Gemini + Qwen shared)</summary><pre>{_esc(e.get('prompt',''))}</pre></details></div>
  </div>
</section>"""
        )

    nav = " ".join(f'<a href="#c{int(e["idx"])}">{int(e["idx"])}</a>' for e in rows[:40])
    if len(rows) > 40:
        nav += " …"

    page_title = title or f"Pairwise inputs review (pilot-90, n={len(rows)})"
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>{_esc(page_title)}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:20px auto;padding:0 16px 48px;background:#fafafa}}
h1{{font-size:22px}} h2{{font-size:17px;margin-top:32px}}
.summary{{background:#fff;border:1px solid #ccc;border-radius:10px;padding:14px;margin:16px 0}}
.answer{{background:#e3f2fd;border:1px solid #64b5f6;border-radius:8px;padding:8px 12px;font-weight:600}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px}}
.media{{width:100%;border:1px solid #ccc;border-radius:8px;background:#000;min-height:160px}}
.missing{{background:#eee;color:#666;padding:40px;text-align:center;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left}}
.dim{{color:#888}}
pre{{background:#f0f0f2;padding:12px;border-radius:8px;font-size:11px;white-space:pre-wrap;max-height:360px;overflow:auto}}
nav a{{margin-right:8px;font-size:12px}}
</style></head><body>
<h1>{_esc(page_title)}</h1>
<div class="summary">
  <p>n={len(rows)} · GT left={n_left} · GT right={len(rows)-n_left} · same_joint={n_same}/{len(rows)} · start_pose: generation={n_gen}, human_gt={n_human}</p>
  <p>layout={_esc(payload.get('layout'))} · version={_esc(payload.get('version'))}</p>
  <p>Neg control: <b>axis swap only</b> (joint held fixed). <code>gt_side</code> randomized per cue (seeded).</p>
  <p>Specs: <code>{_esc(spec_path.name)}</code></p>
</div>
<nav><b>Jump:</b> {nav}</nav>
{''.join(sections)}
</body></html>"""

    out_path.write_text(doc, encoding="utf-8")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Build pairwise input review HTML")
    p.add_argument("--specs", type=Path, default=MOTION_PAIRWISE_DIR / PAIRWISE_SPECS_NAME)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--title", default=None)
    args = p.parse_args()
    out = args.out or (args.specs.parent / f"{args.specs.stem}_review.html")
    path = build_review_html(spec_path=args.specs, out_path=out, title=args.title)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
