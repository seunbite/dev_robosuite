#!/usr/bin/env python3
"""Build per-cue pilot-40 verify review HTML (generation + 4 verify columns)."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_CFG = _REPO / "data/seed/shots/google_robot/shot_configs_pilot40_mobile.json"
DEFAULT_MEDIA = _REPO / "data/results/render/google_robot/pilot40_media"
DEFAULT_VERIFY = _REPO / "data/results/verify/google_robot"
DEFAULT_OUT = _REPO / "data/results/html/google_robot/pilot40_verify_review.html"


def _safe_cue(cue: str) -> str:
    return cue.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _stem(row: dict) -> str:
    return f"mm19_g{int(row['idx']):02d}_{_safe_cue(str(row['cue']))}"


def _rel(path: Path) -> str:
    try:
        return path.relative_to(_REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _load_verify(path: Path) -> dict[tuple[int, str], dict]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("results", data.get("rows", data if isinstance(data, list) else []))
    out: dict[tuple[int, str], dict] = {}
    for row in items:
        key = (int(row.get("idx", -1)), str(row.get("cue", "")))
        out[key] = row.get("result", row)
    return out


def _media_href(out_html: Path, asset: Path) -> str:
    if not asset.is_file():
        return ""
    return Path(os.path.relpath(asset.resolve(), out_html.parent.resolve())).as_posix()


def _result_badge(result: dict | None, key: str) -> str:
    if not result:
        return '<span class="badge na">—</span>'
    if result.get("parse_error") or result.get("error"):
        return '<span class="badge err">ERR</span>'
    val = result.get(key)
    if val is True:
        return '<span class="badge ok">OK</span>'
    if val is False:
        return '<span class="badge bad">NO</span>'
    return '<span class="badge na">?</span>'


def _result_text(result: dict | None) -> str:
    if not result:
        return "<em>no result</em>"
    if result.get("error"):
        return html.escape(str(result["error"]))
    if result.get("parse_error"):
        return html.escape(f"parse: {result['parse_error']}")
    parts = []
    for k in ("visual_assessment", "why_change", "confidence"):
        if k in result and result[k] not in (None, ""):
            parts.append(f"<b>{k}</b>: {html.escape(str(result[k]))}")
    if not parts and result.get("raw_text"):
        parts.append(html.escape(str(result["raw_text"])[:800]))
    return "<br/>".join(parts) if parts else "<em>empty</em>"


def build(args: argparse.Namespace) -> None:
    rows = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    media = Path(args.media_dir)
    verify_dir = Path(args.verify_dir)

    pose_vlm = _load_verify(verify_dir / "pilot40_pose_verify_vlm.json")
    pose_text = _load_verify(verify_dir / "pilot40_pose_verify_text.json")
    mov_vlm = _load_verify(verify_dir / "pilot40_movement_verify_vlm.json")
    mov_text = _load_verify(verify_dir / "pilot40_movement_verify_text.json")

    out_path = Path(args.out_html)

    cards = []
    for row in rows:
        stem = _stem(row)
        key = (int(row["idx"]), str(row["cue"]))
        mp4 = media / "mp4" / f"{stem}.mp4"
        pose_png = media / "pose" / f"{stem}_pose.png"
        mp4_href = _media_href(out_path, mp4)
        pose_href = _media_href(out_path, pose_png)
        pv = pose_vlm.get(key)
        pt = pose_text.get(key)
        mv = mov_vlm.get(key)
        mt = mov_text.get(key)

        def _vid(href: str) -> str:
            return f'<video controls loop muted playsinline src="{html.escape(href)}"></video>' if href else '<p class="missing">missing mp4</p>'

        def _img(href: str) -> str:
            return f'<img src="{html.escape(href)}" alt="pose"/>' if href else '<p class="missing">missing png</p>'

        cards.append(
            f"""
<section class="card">
  <header>
    <h2>g{int(row['idx']):02d} · {html.escape(str(row['cue']))}</h2>
    <p>{html.escape(str(row.get('description', '')))}</p>
  </header>
  <div class="grid">
    <div class="col">
      <h3>generation (mp4)</h3>
      {_vid(mp4_href)}
    </div>
    <div class="col">
      <h3>pose verify VLM (png) {_result_badge(pv, 'pose_is_appropriate')}</h3>
      {_img(pose_href)}
      <div class="meta">{_result_text(pv)}</div>
    </div>
    <div class="col">
      <h3>pose verify text (png) {_result_badge(pt, 'pose_is_appropriate')}</h3>
      {_img(pose_href)}
      <div class="meta">{_result_text(pt)}</div>
    </div>
    <div class="col">
      <h3>movement verify VLM (mp4) {_result_badge(mv, 'movement_is_appropriate')}</h3>
      {_vid(mp4_href)}
      <div class="meta">{_result_text(mv)}</div>
    </div>
    <div class="col">
      <h3>movement verify text (mp4) {_result_badge(mt, 'movement_is_appropriate')}</h3>
      {_vid(mp4_href)}
      <div class="meta">{_result_text(mt)}</div>
    </div>
  </div>
</section>"""
        )
    css = """
body { font-family: system-ui, sans-serif; margin: 0; background: #0f1115; color: #e8eaed; }
h1 { padding: 16px 24px; margin: 0; font-size: 1.25rem; border-bottom: 1px solid #333; }
.card { margin: 24px; padding: 16px; border: 1px solid #333; border-radius: 12px; background: #171a21; }
.card header h2 { margin: 0 0 4px; font-size: 1.1rem; }
.card header p { margin: 0 0 12px; color: #9aa0a6; font-size: 0.9rem; }
.grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }
.col { background: #0f1115; border-radius: 8px; padding: 8px; min-width: 0; }
.col h3 { margin: 0 0 8px; font-size: 0.75rem; font-weight: 600; color: #bdc1c6; }
video, img { width: 100%; height: auto; border-radius: 4px; background: #000; display: block; }
.meta { margin-top: 8px; font-size: 0.72rem; line-height: 1.4; color: #c4c7c5; max-height: 120px; overflow: auto; }
.badge { font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; margin-left: 4px; }
.badge.ok { background: #1e4620; color: #81c995; }
.badge.bad { background: #5c1a1a; color: #f28b82; }
.badge.err { background: #4a3800; color: #fdd663; }
.badge.na { background: #333; color: #9aa0a6; }
.missing { color: #f28b82; font-size: 0.8rem; }
@media (max-width: 1400px) { .grid { grid-template-columns: repeat(2, 1fr); } }
"""

    out_path = Path(args.out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>pilot-40 Google Robot verify review</title>
<style>{css}</style>
</head><body>
<h1>pilot-40 Google Robot · generation + verify ({len(rows)} cues)</h1>
{''.join(cards)}
</body></html>"""
    out_path.write_text(doc, encoding="utf-8")
    print(f"Wrote {out_path} ({len(rows)} cues)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config-json", default=str(DEFAULT_CFG))
    p.add_argument("--media-dir", default=str(DEFAULT_MEDIA))
    p.add_argument("--verify-dir", default=str(DEFAULT_VERIFY))
    p.add_argument("--out-html", default=str(DEFAULT_OUT))
    build(p.parse_args())


if __name__ == "__main__":
    main()
