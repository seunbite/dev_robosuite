#!/usr/bin/env python3
"""
VLM verify for 40 GT-fixed-pose generated motions: MP4 + alpha_frame_trajectory (sim EE path).

Run this before compare experiments. Writes JSON + review HTML under data/results/verify/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _REPO / "adhoc" / "vlm_test", _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import testset_utils  # noqa: E402
from motion_media_paths import default_mp4_out, gif_to_mp4, pick_latest_gif  # noqa: E402
from render_and_verify_o_picked_motions import _extract_json  # noqa: E402
from score_pilot40_motion_gt_components import (  # noqa: E402
    _build_annotation_map,
    _tail_matches_component,
    _tail_steps,
)

BASE_CFG = (
    _REPO
    / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_gt_fixed_pose_pilot40.json"
)
BASE_GIF_DIR = _REPO / "data/results/visualize/gt_fixed_pose_pilot20_hz10/IIWA"
OUT_ROOT = _REPO / "data/results/render/manipulator/motion_vlm_verify_pilot40"
MP4_DIR = OUT_ROOT / "mp4"
ALPHA_DIR = OUT_ROOT / "alpha_frame_trajectory"
MANIFEST_PATH = OUT_ROOT / "manifest_pilot40.json"
OUT_JSON = _REPO / "data/results/verify/pilot40_motion_vlm_verify_gt_fixed_gemini.json"
OUT_HTML = _REPO / "data/results/html/manipulator/pilot40_motion_vlm_verify_gt_fixed.html"

ROBOT = "IIWA"
HZ = 10


def _pick_latest_gif(dir_path: Path, cue: str) -> Path | None:
    hit = pick_latest_gif(_REPO, cue)
    if hit is not None:
        return hit
    if not dir_path.is_dir():
        return None
    cands = sorted(
        [p for p in dir_path.glob("*.gif") if f"_{ROBOT}_{cue}" in p.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def _gif_to_mp4(gif: Path, mp4: Path) -> None:
    gif_to_mp4(gif, mp4)


def _vlm_call_video(model_id: str, user_text: str, mp4_path: Path) -> str:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini.")
    client = genai.Client(api_key=api_key)
    video = types.Part.from_bytes(data=mp4_path.read_bytes(), mime_type="video/mp4")
    resp = client.models.generate_content(model=model_id, contents=[video, user_text])
    return (resp.text or "").strip()


def _vlm_call_image(model_id: str, user_text: str, image_path: Path) -> str:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini.")
    client = genai.Client(api_key=api_key)
    part = types.Part.from_bytes(data=image_path.read_bytes(), mime_type="image/png")
    resp = client.models.generate_content(model=model_id, contents=[part, user_text])
    return (resp.text or "").strip()


def _vlm_prompt(cue: str, description: str) -> str:
    return f"""
You are evaluating a robot-arm motion (IIWA manipulator).

Target cue (ground truth label for this motion):
- cue: {cue}
- description: {description}

Does this motion appear to represent the target cue above?

Return ONLY strict JSON:
{{
  "represents_target_cue": true/false,
  "cue_match_reason": "string",
  "confidence": 0.0
}}
""".strip()


def _rows() -> list[dict[str, Any]]:
    return sorted(json.loads(BASE_CFG.read_text(encoding="utf-8")), key=lambda r: int(r["idx"]))


def _annotation_by_idx() -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for ann in _build_annotation_map():
        out[int(ann["cue_idx"])] = ann
    return out


def build_alpha_trajectory(
    row: dict[str, Any],
    *,
    force: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    idx = int(row["idx"])
    cue = row["cue"]
    pose_id = int((row.get("gt_fixed_first_pose") or {}).get("pose_id") or 0)
    labeled = ALPHA_DIR / f"{idx:03d}_{cue}_alpha_frame_trajectory.png"
    neutral = ALPHA_DIR / f"clip_{idx:03d}.png"

    sample = {
        "sample_id": testset_utils._safe_name(f"gt_fixed_{idx}_{cue}"),
        "testset": "iconic",
        "cue_idx": idx,
        "cue": cue,
        "gif_path": str(BASE_CFG),
        "config_path": str(BASE_CFG),
        "selected_pose_id": pose_id,
        "meta": {},
    }

    meta: dict[str, Any] = {}
    if not labeled.is_file() or force:
        print(f"[alpha_frame_trajectory] {cue} ...", flush=True)
        img, meta = testset_utils.build_tile_figure_sim_trajectory_panel(
            sample,
            ROBOT,
            HZ,
            canonical="alpha_frame_trajectory",
            force=force,
        )
        labeled.parent.mkdir(parents=True, exist_ok=True)
        try:
            img.save(labeled, format="PNG", optimize=False)
            img.save(neutral, format="PNG", optimize=False)
        except OSError as e:
            print(f"[warn] could not save alpha for {cue}: {e}", flush=True)
    else:
        print(f"[skip alpha] {cue}", flush=True)

    return neutral, labeled, meta


def prepare_manifest(rows: list[dict[str, Any]], *, force_alpha: bool = False) -> list[dict[str, Any]]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MP4_DIR.mkdir(parents=True, exist_ok=True)
    ALPHA_DIR.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for row in rows:
        idx = int(row["idx"])
        cue = row["cue"]
        gif = _pick_latest_gif(BASE_GIF_DIR, cue)
        mp4 = MP4_DIR / f"{idx:03d}_{cue}.mp4"
        if gif and not mp4.is_file():
            print(f"[mp4] {cue}", flush=True)
            _gif_to_mp4(gif, mp4)
        elif not gif:
            print(f"[warn] no gif for {cue}", flush=True)

        _neutral, labeled, meta = build_alpha_trajectory(row, force=force_alpha)

        manifest.append(
            {
                "cue_idx": idx,
                "cue": cue,
                "description": row.get("description", ""),
                "annotation_raw": (_annotation_by_idx().get(idx) or {}).get("annotation_raw", ""),
                "component_gt": (_annotation_by_idx().get(idx) or {}).get("component"),
                "config_path": str(BASE_CFG),
                "gif": str(gif) if gif else None,
                "mp4": str(mp4) if mp4.is_file() else None,
                "alpha_frame_trajectory": str(labeled),
                "alpha_frame_trajectory_neutral": str(_neutral),
                "alpha_meta": meta,
            }
        )
    MANIFEST_PATH.write_text(json.dumps({"rows": manifest}, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def run_vlm(manifest: list[dict[str, Any]], model: str, *, skip_vlm: bool = False) -> list[dict[str, Any]]:
    cfg_by_idx = {int(r["idx"]): r for r in _rows()}
    ann_by_idx = _annotation_by_idx()
    results: list[dict[str, Any]] = []

    for item in manifest:
        idx = int(item["cue_idx"])
        cue = item["cue"]
        cfg = cfg_by_idx[idx]
        tail = _tail_steps(cfg.get("movements") or [])
        comp = (ann_by_idx.get(idx) or {}).get("component")
        comp_match, via = (None, None)
        if comp is not None:
            comp_match, via = _tail_matches_component(tail, comp)

        merged = dict(item)
        merged["generation_tail"] = tail
        merged["component_structural_match"] = comp_match
        merged["component_matched_via"] = via

        if skip_vlm:
            results.append(merged)
            continue

        mp4 = item.get("mp4")
        alpha = item.get("alpha_frame_trajectory")
        desc = item.get("description", "")

        if mp4 and Path(mp4).is_file():
            raw = _vlm_call_video(model, _vlm_prompt(cue, desc), Path(mp4))
            try:
                merged["vlm_mp4"] = _extract_json(raw)
            except Exception as e:
                merged["vlm_mp4"] = {"parse_error": str(e), "raw_text": raw}
        else:
            merged["vlm_mp4"] = {"error": "missing_mp4"}

        if alpha and Path(alpha).is_file():
            raw = _vlm_call_image(model, _vlm_prompt(cue, desc), Path(alpha))
            try:
                merged["vlm_alpha_frame_trajectory"] = _extract_json(raw)
            except Exception as e:
                merged["vlm_alpha_frame_trajectory"] = {"parse_error": str(e), "raw_text": raw}
        else:
            merged["vlm_alpha_frame_trajectory"] = {"error": "missing_alpha"}

        print(
            f"[vlm] {cue} mp4={merged.get('vlm_mp4', {}).get('represents_target_cue')} "
            f"alpha={merged.get('vlm_alpha_frame_trajectory', {}).get('represents_target_cue')} "
            f"struct={comp_match}",
            flush=True,
        )
        results.append(merged)
    return results


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:.1f}%)" if d else "n/a"


def _write_html(payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    cards = []
    for r in rows:
        mp4 = r.get("mp4") or ""
        alpha = r.get("alpha_frame_trajectory") or ""
        rel_mp4 = os.path.relpath(mp4, OUT_HTML.parent) if mp4 else ""
        rel_alpha = os.path.relpath(alpha, OUT_HTML.parent) if alpha else ""
        v_mp4 = r.get("vlm_mp4") or {}
        v_alpha = r.get("vlm_alpha_frame_trajectory") or {}
        cards.append(
            f"""
<article class="card">
  <h2>{r['cue_idx']}. {r['cue']}</h2>
  <p class="meta">annotation: <code>{r.get('annotation_raw','')}</code> |
    structural GT match: <b>{r.get('component_structural_match')}</b></p>
  <p class="desc">{r.get('description','')}</p>
  <div class="grid2">
    <div><h3>MP4</h3>
      <p>represents_target_cue: <b>{v_mp4.get('represents_target_cue')}</b></p>
      <p class="reason">{v_mp4.get('cue_match_reason','')}</p>
      <video src="{rel_mp4}" controls loop muted playsinline></video>
    </div>
    <div><h3>alpha_frame_trajectory</h3>
      <p>represents_target_cue: <b>{v_alpha.get('represents_target_cue')}</b></p>
      <p class="reason">{v_alpha.get('cue_match_reason','')}</p>
      <img src="{rel_alpha}" alt="alpha trajectory"/>
    </div>
  </div>
</article>"""
        )
    summary = payload.get("summary", {})
    html = f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>Pilot40 motion VLM verify (GT-fixed pose)</title>
<style>
body{{font-family:system-ui,sans-serif;background:#f5f7fb;margin:0;padding:20px}}
.card{{background:#fff;border:1px solid #dce1ea;border-radius:10px;padding:14px;margin-bottom:14px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
video,img{{width:100%;border:1px solid #ddd;border-radius:8px;background:#fafafa}}
.reason{{font-size:13px;color:#444}}
.meta{{font-size:13px;color:#555}}
.summary{{background:#eef3ff;border:1px solid #c9d6f5;padding:12px;border-radius:8px;margin-bottom:16px}}
</style></head><body>
<h1>Pilot40 motion VLM verify</h1>
<div class="summary">
  <p>Model: {payload.get('model','')}</p>
  <p>MP4 represents_target_cue: {summary.get('vlm_mp4_ok')}</p>
  <p>alpha_frame_trajectory represents_target_cue: {summary.get('vlm_alpha_ok')}</p>
  <p>Either modality OK: {summary.get('vlm_either_ok')}</p>
  <p>Both modalities OK: {summary.get('vlm_both_ok')}</p>
  <p>Structural component match (generation tail vs your GT): {summary.get('structural_ok')}</p>
</div>
{''.join(cards)}
</body></html>"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    mp4_ok = alpha_ok = either_ok = both_ok = struct_ok = 0
    struct_n = vlm_n = 0
    for r in results:
        comp = r.get("component_gt")
        if comp is not None:
            struct_n += 1
            if r.get("component_structural_match"):
                struct_ok += 1
        if r.get("mp4") and r.get("alpha_frame_trajectory"):
            vlm_n += 1
            m = (r.get("vlm_mp4") or {}).get("represents_target_cue") is True
            a = (r.get("vlm_alpha_frame_trajectory") or {}).get("represents_target_cue") is True
            if m:
                mp4_ok += 1
            if a:
                alpha_ok += 1
            if m or a:
                either_ok += 1
            if m and a:
                both_ok += 1
    return {
        "vlm_mp4_ok": _pct(mp4_ok, vlm_n),
        "vlm_alpha_ok": _pct(alpha_ok, vlm_n),
        "vlm_either_ok": _pct(either_ok, vlm_n),
        "vlm_both_ok": _pct(both_ok, vlm_n),
        "structural_ok": _pct(struct_ok, struct_n),
        "n_vlm": vlm_n,
        "n_structural": struct_n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--limit", type=int, default=0, help="Only first N cues (0=all)")
    ap.add_argument("--force-alpha", action="store_true")
    ap.add_argument("--media-only", action="store_true", help="Build mp4 + alpha only, no VLM")
    ap.add_argument("--vlm-only", action="store_true", help="Reuse manifest; run VLM only")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing mp4/alpha; rebuild full manifest",
    )
    args = ap.parse_args()

    rows = _rows()
    if args.limit:
        rows = rows[: args.limit]

    if args.vlm_only and MANIFEST_PATH.is_file() and not args.resume:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["rows"]
        if args.limit:
            manifest = manifest[: args.limit]
    else:
        manifest = prepare_manifest(rows, force_alpha=args.force_alpha)

    results = run_vlm(manifest, args.model, skip_vlm=args.media_only)
    summary = _summarize(results)

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "robot": ROBOT,
        "hz": HZ,
        "config": str(BASE_CFG),
        "manifest": str(MANIFEST_PATH),
        "summary": summary,
        "rows": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_html(payload)

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
