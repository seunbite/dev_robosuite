#!/usr/bin/env python3
"""Build alpha_frame_trajectory PNGs for GT-o picked-pose motions; VLM Q1/Q2 on image."""
from __future__ import annotations

import json
import os
import re
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
from render_and_verify_o_picked_motions import (  # noqa: E402
    OUT_ROOT,
    VERIFY_JSON,
    _extract_json,
    write_review_html,
)


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

ALPHA_DIR = OUT_ROOT / "alpha_frame_trajectory"
OUT_JSON = _REPO / "data/results/verify/motion_alpha_traj_verify_o_picked_gemini.json"
ROBOT = "IIWA"
HZ = 4


def _vlm_prompt_q1(cue: str, description: str) -> str:
    return f"""
You are evaluating a single composite image of a robot-arm motion (IIWA manipulator).
The image shows an alpha-stack of motion frames with an end-effector trajectory overlay (yellow to purple).

Target cue (ground truth label for this motion):
- cue: {cue}
- description: {description}

Using ONLY this image, does this motion appear to represent the target cue above?

Return ONLY strict JSON:
{{
  "represents_target_cue": true/false,
  "cue_match_reason": "string",
  "confidence": 0.0
}}
""".strip()


def _vlm_prompt_q2_blind() -> str:
    return """
You are evaluating a single composite image of a robot-arm motion (IIWA manipulator).
The image shows an alpha-stack of motion frames with an end-effector trajectory overlay.

You are NOT given any target gesture name or description.

Using ONLY this image, describe what human gesture or social signal this motion appears to convey.

Return ONLY strict JSON:
{
  "freeform_gesture_description": "string",
  "confidence": 0.0
}
""".strip()


def _load_manifest() -> list[dict[str, Any]]:
    p = OUT_ROOT / "manifest_o_picked.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _attach_alpha_paths(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(manifest, 1):
        cue = item["cue"]
        cue_idx = int(item["cue_idx"])
        pose_id = int(item["pose_id"])
        png_path = ALPHA_DIR / f"clip_{i:02d}.png"
        labeled_path = ALPHA_DIR / f"{cue}_c{cue_idx}_p{pose_id}_alpha_traj.png"
        rows.append(
            {
                **item,
                "alpha_trajectory_png": str(png_path),
                "alpha_trajectory_png_labeled": str(labeled_path),
            }
        )
    return rows


def _load_configs_by_cue() -> dict[str, dict[str, Any]]:
    from render_and_verify_o_picked_motions import CONFIG_PATHS

    out: dict[str, dict[str, Any]] = {}
    for path in CONFIG_PATHS:
        for row in json.loads(Path(path).read_text(encoding="utf-8")):
            out[row["cue"]] = row
    return out


def build_alpha_images(manifest: list[dict[str, Any]], *, force: bool = False) -> list[dict[str, Any]]:
    ALPHA_DIR.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict[str, Any]] = []
    canonical = "alpha_frame_trajectory"

    for i, item in enumerate(manifest, 1):
        cue = item["cue"]
        cue_idx = int(item["cue_idx"])
        pose_id = int(item["pose_id"])
        config_path = item["config_json"]
        # Neutral filename for VLM (no cue string in path passed to model — bytes only).
        png_path = ALPHA_DIR / f"clip_{i:02d}.png"
        labeled_path = ALPHA_DIR / f"{cue}_c{cue_idx}_p{pose_id}_alpha_traj.png"

        sample = {
            "sample_id": testset_utils._safe_name(f"o_picked_{cue_idx}_{pose_id}"),
            "testset": "iconic",
            "cue_idx": cue_idx,
            "cue": cue,
            "gif_path": config_path,
            "config_path": config_path,
            "selected_pose_id": pose_id,
            "meta": {},
        }

        meta: dict[str, Any] = {}
        if not png_path.is_file() or force:
            print(f"[alpha] {cue} pose_id={pose_id} ...", flush=True)
            img, meta = testset_utils.build_tile_figure_sim_trajectory_panel(
                sample,
                ROBOT,
                HZ,
                canonical=canonical,
                force=force,
            )
            img.save(png_path, format="PNG", optimize=True)
            img.save(labeled_path, format="PNG", optimize=True)
        else:
            print(f"[skip alpha] {cue}", flush=True)

        out_rows.append(
            {
                **item,
                "alpha_trajectory_png": str(png_path),
                "alpha_trajectory_png_labeled": str(labeled_path),
                "alpha_meta": meta,
            }
        )
    return out_rows


def run_vlm(rows: list[dict[str, Any]], model: str = "gemini-2.5-pro") -> list[dict[str, Any]]:
    by_cue = _load_configs_by_cue()
    results: list[dict[str, Any]] = []

    for row in rows:
        cue = row["cue"]
        desc = by_cue[cue].get("description", "")
        png = Path(row["alpha_trajectory_png"])

        raw_q1 = _vlm_call_image(model, _vlm_prompt_q1(cue, desc), png)
        try:
            q1 = _extract_json(raw_q1)
        except Exception as e:
            q1 = {"parse_error": str(e), "raw_text": raw_q1}

        raw_q2 = _vlm_call_image(model, _vlm_prompt_q2_blind(), png)
        try:
            q2 = _extract_json(raw_q2)
        except Exception as e:
            q2 = {"parse_error": str(e), "raw_text": raw_q2}

        merged = dict(row)
        merged["vlm_alpha_model"] = model
        merged["vlm_alpha_q1"] = q1
        merged["vlm_alpha_q2_blind"] = q2
        results.append(merged)
        print(
            f"[vlm] {cue} q1={q1.get('represents_target_cue')} "
            f"q2={str(q2.get('freeform_gesture_description', ''))[:60]}...",
            flush=True,
        )
    return results


def write_alpha_html(rows: list[dict[str, Any]], payload: dict[str, Any]) -> Path:
    import html as html_mod
    import os

    html_dir = _REPO / "data/results/html/manipulator"
    html_dir.mkdir(parents=True, exist_ok=True)
    out = html_dir / "o_picked_alpha_traj_vlm_review.html"

    def rel(p: str) -> str:
        return os.path.relpath(p, html_dir).replace("\\", "/")

    cards = []
    for i, r in enumerate(rows, 1):
        q1 = r.get("vlm_alpha_q1") or {}
        q2 = r.get("vlm_alpha_q2_blind") or {}
        ok = q1.get("represents_target_cue")
        badge = "yes" if ok else "no"
        gen = r["generation"]
        cards.append(
            f"""
<article class="card {badge}">
  <header><h2>{i}. {html_mod.escape(r['cue'])}</h2><span class="badge">{badge.upper() if ok is not None else '—'}</span></header>
  <div class="meta"><b>GT</b> {html_mod.escape(r['groundtruth'])} · pose_id={r['pose_id']}</div>
  <img class="hero" src="{html_mod.escape(rel(r['alpha_trajectory_png_labeled']))}" alt="" />
  <p><b>Q1 (image, cue given)</b> <code>{html_mod.escape(str(ok))}</code></p>
  <p>{html_mod.escape(q1.get('cue_match_reason', ''))}</p>
  <p><b>Q2 blind (image only)</b></p>
  <p class="blind">{html_mod.escape(q2.get('freeform_gesture_description', ''))}</p>
</article>"""
        )

    n = payload["n"]
    n_yes = payload["q1_represents_target_cue_true"]
    page = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"/>
<title>alpha_frame_trajectory · o-picked VLM</title>
<style>
body{{font-family:system-ui,sans-serif;background:#f4f4f5;padding:24px;max-width:900px;margin:auto}}
.card{{background:#fff;border-radius:12px;padding:16px;margin-bottom:20px;border:1px solid #e4e4e7}}
.card.yes{{border-left:4px solid #16a34a}}.card.no{{border-left:4px solid #dc2626}}
.hero{{width:100%;border-radius:8px;border:1px solid #ddd}}
.blind{{background:#eff6ff;padding:10px;border-radius:8px}}
.badge{{font-weight:700;font-size:.75rem;padding:4px 10px;border-radius:999px}}
.card.yes .badge{{background:#dcfce7;color:#166534}}
.card.no .badge{{background:#fee2e2;color:#991b1b}}
</style></head><body>
<h1>alpha_frame_trajectory · GT-o picked poses (14)</h1>
<p>Q1 represents_target_cue: {n_yes}/{n} ({payload['q1_rate']:.1%})</p>
{''.join(cards)}
</body></html>"""
    out.write_text(page, encoding="utf-8")
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--vlm-only", action="store_true")
    ap.add_argument("--force-alpha", action="store_true")
    ap.add_argument("--model", default="gemini-2.5-pro")
    args = ap.parse_args()

    manifest = _load_manifest()
    if args.vlm_only:
        rows = _attach_alpha_paths(manifest)
    else:
        rows = build_alpha_images(manifest, force=args.force_alpha)

    if not args.build_only:
        rows = run_vlm(rows, model=args.model)
        n = len(rows)
        n_yes = sum(1 for r in rows if (r.get("vlm_alpha_q1") or {}).get("represents_target_cue") is True)
        payload = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": args.model,
            "media": "alpha_frame_trajectory_png",
            "n": n,
            "q1_represents_target_cue_true": n_yes,
            "q1_rate": n_yes / n if n else 0.0,
            "results": rows,
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {OUT_JSON}")
        print(f"Q1 yes: {n_yes}/{n} = {n_yes/n:.1%}" if n else "")
        html_path = write_alpha_html(rows, payload)
        print(f"wrote {html_path}")
        # refresh combined MP4 review page if prior verify exists
        if VERIFY_JSON.is_file():
            write_review_html()


if __name__ == "__main__":
    main()
