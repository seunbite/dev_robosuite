#!/usr/bin/env python3
"""Render motions for GT-o cues using human-picked pose tiles; VLM-verify MP4."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]  # .../dev_robosuite
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from generate_pose_group_tiles import _load_entries, _select_xyz_tertile_balanced  # noqa: E402
from legacy.motion_generation_core import generate  # noqa: E402

def _vlm_call(model_id: str, user_text: str, mp4_path: Path) -> str:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini.")
    client = genai.Client(api_key=api_key)
    video = types.Part.from_bytes(data=mp4_path.read_bytes(), mime_type="video/mp4")
    resp = client.models.generate_content(model=model_id, contents=[video, user_text])
    return (resp.text or "").strip()


def _extract_json(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        s = m.group(0)
    return json.loads(s)


CONFIG_PATHS = [
    _REPO / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot10.json",
    _REPO / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot20_more.json",
]
CONSOLIDATED = _REPO / "data/results/verify/pilot40_pose_eval_consolidated.json"
TILE_PICK = _REPO / "data/results/verify/pose_tile_pick_by_group.json"
JSONL = _REPO / "data/seed/_remainder/closest_poses_results.jsonl"
OUT_ROOT = _REPO / "data/results/render/manipulator/picked_pose_o"
MP4_DIR = OUT_ROOT / "mp4"
VERIFY_JSON = _REPO / "data/results/verify/motion_mp4_verify_o_picked_gemini.json"


def _load_tile_pick() -> dict[tuple[str, str], int]:
    data = json.loads(TILE_PICK.read_text(encoding="utf-8"))
    picks = data["picks"]
    out: dict[tuple[str, str], int] = {}
    for key, idx in picks.items():
        d, g = key.split("_", 1)
        out[(d, g)] = int(idx)
    return out


def _pose_id_for_pick(
    entries: list[dict],
    robot: str,
    d: str,
    g: str,
    tile_index_1based: int,
) -> dict[str, Any]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entries:
        if e.get("robot") == robot:
            buckets[(e.get("dir"), e.get("gripper_orientation"))].append(e)
    rows = buckets.get((d, g), [])
    reps = _select_xyz_tertile_balanced(rows, n=9)
    if not reps:
        raise ValueError(f"no poses for ({d}, {g})")
    i = tile_index_1based - 1
    if i < 0 or i >= len(reps):
        raise ValueError(f"tile index {tile_index_1based} out of range for ({d}, {g})")
    return reps[i]


def _configs_by_cue() -> dict[str, tuple[Path, dict[str, Any]]]:
    out: dict[str, tuple[Path, dict[str, Any]]] = {}
    for p in CONFIG_PATHS:
        for row in json.loads(p.read_text(encoding="utf-8")):
            out[row["cue"]] = (p, row)
    return out


def _gif_to_mp4(gif_path: Path, mp4_path: Path) -> None:
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    ff = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    cmd = [
        ff,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(gif_path),
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]
    r = subprocess.run(cmd, text=True, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "").strip() or "ffmpeg failed")


def _vlm_prompt(cue: str, description: str) -> str:
    return f"""
You are evaluating a short rendered robot-arm motion video (IIWA manipulator).

Target cue (ground truth label for this clip):
- cue: {cue}
- description: {description}

Answer both questions using ONLY the video.

1) Does this robot motion appear to represent the target cue above? Answer represents_target_cue as true or false.
2) In free form, what human gesture or social signal does this motion look like it is trying to convey?

Return ONLY strict JSON:
{{
  "represents_target_cue": true/false,
  "cue_match_reason": "string",
  "freeform_gesture_description": "string",
  "confidence": 0.0
}}
""".strip()


def _vlm_prompt_blind_freeform() -> str:
    return """
You are evaluating a short rendered robot-arm motion video (IIWA manipulator).

You are NOT given any target gesture name or description. Do not guess a label from metadata.

Using ONLY what you see in the video, describe what human gesture or social signal this motion appears to convey.

Return ONLY strict JSON:
{
  "freeform_gesture_description": "string",
  "confidence": 0.0
}
""".strip()


def _o_rows() -> list[dict[str, Any]]:
    cons = json.loads(CONSOLIDATED.read_text(encoding="utf-8"))
    return [r for r in cons["rows"] if str(r.get("groundtruth", "")).strip().startswith("o ")]


def run_render(sim_robot: str = "IIWA", hz: int = 4, skip_existing: bool = True) -> list[dict[str, Any]]:
    tile_pick = _load_tile_pick()
    entries = _load_entries(JSONL)
    by_cue = _configs_by_cue()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MP4_DIR.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for ev in _o_rows():
        cue = ev["cue"]
        gen = ev["generation"]
        d, g = gen["dir"], gen["gripper_orientation"]
        cfg_path, cfg_row = by_cue[cue]
        tile_idx = tile_pick[(d, g)]
        pose_row = _pose_id_for_pick(entries, sim_robot, d, g, tile_idx)
        pose_id = int(pose_row["pose_id"])

        mp4_path = MP4_DIR / f"{cue}_c{ev['cue_idx']}_p{pose_id}.mp4"
        if skip_existing and mp4_path.is_file():
            print(f"[skip render] {cue} -> {mp4_path.name}", flush=True)
            manifest.append(
                {
                    "cue": cue,
                    "cue_idx": ev["cue_idx"],
                    "groundtruth": ev["groundtruth"],
                    "generation": gen,
                    "tile_index": tile_idx,
                    "pose_id": pose_id,
                    "config_json": str(cfg_path),
                    "mp4": str(mp4_path),
                }
            )
            continue

        print(f"[render] {cue} pose_id={pose_id} tile={tile_idx} ({d},{g})", flush=True)
        gif_path = generate(
            robot=sim_robot,
            cue=cue,
            cue_idx=int(ev["cue_idx"]),
            pose_index=pose_id,
            jsonl_path=str(JSONL),
            config_path=str(cfg_path),
            output_dir=str(OUT_ROOT),
            hz=hz,
            top_k=1,
            gif_filename_suffix=f"pick_p{pose_id}",
        )
        if not gif_path:
            raise RuntimeError(f"no gif for {cue}")
        _gif_to_mp4(Path(gif_path), mp4_path)
        manifest.append(
            {
                "cue": cue,
                "cue_idx": ev["cue_idx"],
                "groundtruth": ev["groundtruth"],
                "generation": gen,
                "tile_index": tile_idx,
                "pose_id": pose_id,
                "config_json": str(cfg_path),
                "gif": gif_path,
                "mp4": str(mp4_path),
            }
        )
    manifest_path = OUT_ROOT / "manifest_o_picked.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def run_vlm(model: str = "gemini-2.5-pro", manifest: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if manifest is None:
        manifest_path = OUT_ROOT / "manifest_o_picked.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    by_cue = _configs_by_cue()
    results: list[dict[str, Any]] = []
    for item in manifest:
        cue = item["cue"]
        _, cfg_row = by_cue[cue]
        desc = cfg_row.get("description", "")
        mp4 = Path(item["mp4"])
        prompt = _vlm_prompt(cue, desc)
        raw = _vlm_call(model, prompt, mp4)
        try:
            parsed = _extract_json(raw)
        except Exception as e:
            parsed = {"parse_error": str(e), "raw_text": raw}

        results.append({**item, "vlm_model": model, "vlm_result": parsed})
        ok = parsed.get("represents_target_cue")
        print(f"[vlm] {cue} represents_target_cue={ok}", flush=True)

    n = len(results)
    n_yes = sum(1 for r in results if r.get("vlm_result", {}).get("represents_target_cue") is True)
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "n": n,
        "represents_target_cue_true": n_yes,
        "represents_target_cue_rate": n_yes / n if n else 0.0,
        "results": results,
    }
    VERIFY_JSON.parent.mkdir(parents=True, exist_ok=True)
    VERIFY_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nVLM yes rate: {n_yes}/{n} = {n_yes/n:.1%}" if n else "empty")
    print(f"wrote {VERIFY_JSON}")
    return payload


def run_vlm_blind_freeform(
    model: str = "gemini-2.5-pro",
    manifest: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if manifest is None:
        manifest_path = OUT_ROOT / "manifest_o_picked.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    prior: dict[str, Any] = {}
    if VERIFY_JSON.is_file():
        prior = json.loads(VERIFY_JSON.read_text(encoding="utf-8"))

    prior_by_cue = {r["cue"]: r for r in prior.get("results", [])}
    prompt = _vlm_prompt_blind_freeform()
    results: list[dict[str, Any]] = []

    for item in manifest:
        cue = item["cue"]
        mp4 = Path(item["mp4"])
        raw = _vlm_call(model, prompt, mp4)
        try:
            parsed = _extract_json(raw)
        except Exception as e:
            parsed = {"parse_error": str(e), "raw_text": raw}

        merged = {**prior_by_cue.get(cue, {}), **item}
        merged["vlm_blind_freeform_model"] = model
        merged["vlm_blind_freeform"] = parsed
        results.append(merged)
        print(
            f"[blind] {cue}: {str(parsed.get('freeform_gesture_description', ''))[:80]}...",
            flush=True,
        )

    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": prior.get("model", model),
        "n": len(results),
        "represents_target_cue_true": prior.get("represents_target_cue_true"),
        "represents_target_cue_rate": prior.get("represents_target_cue_rate"),
        "blind_freeform_note": "No cue name or description in prompt; video bytes only (no filename).",
        "results": results,
    }
    if prior.get("represents_target_cue_true") is not None:
        n = len(results)
        n_yes = sum(
            1
            for r in results
            if (r.get("vlm_result") or {}).get("represents_target_cue") is True
        )
        payload["represents_target_cue_true"] = n_yes
        payload["represents_target_cue_rate"] = n_yes / n if n else 0.0

    VERIFY_JSON.parent.mkdir(parents=True, exist_ok=True)
    VERIFY_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {VERIFY_JSON}")
    return payload


def write_review_html() -> Path:
    import html as html_mod
    import os

    verify = json.loads(VERIFY_JSON.read_text(encoding="utf-8"))
    html_dir = _REPO / "data/results/html/manipulator"
    html_dir.mkdir(parents=True, exist_ok=True)
    out = html_dir / "o_picked_motion_vlm_review.html"

    def rel(p: str) -> str:
        return os.path.relpath(p, html_dir).replace("\\", "/")

    rows = []
    for i, r in enumerate(verify["results"], 1):
        v = r.get("vlm_result") or {}
        b = r.get("vlm_blind_freeform") or {}
        ok = v.get("represents_target_cue")
        badge = "yes" if ok else "no"
        gen = r["generation"]
        mp4 = rel(r["mp4"])
        gif = rel(r.get("gif", ""))
        rows.append(
            f"""
<article class="card {badge}">
  <header>
    <h2>{i}. {html_mod.escape(r['cue'])} <span class="idx">c{r['cue_idx']}</span></h2>
    <span class="badge">{badge.upper() if ok is not None else '—'}</span>
  </header>
  <div class="meta">
    <div><b>GT</b> {html_mod.escape(r['groundtruth'])}</div>
    <div><b>pose</b> dir={gen['dir']}, grip={gen['gripper_orientation']} · tile #{r['tile_index']} · pose_id={r['pose_id']}</div>
  </div>
  <div class="media">
    <div><div class="label">MP4</div><video src="{html_mod.escape(mp4)}" controls loop muted playsinline></video></div>
    <div><div class="label">GIF</div><img src="{html_mod.escape(gif)}" alt="" loading="lazy" /></div>
  </div>
  <div class="vlm">
    <p><b>Q1 — represents target cue?</b> <code>{html_mod.escape(str(ok))}</code></p>
    <p class="reason">{html_mod.escape(v.get('cue_match_reason', ''))}</p>
    <p><b>Q2 — freeform (cue given in prompt)</b></p>
    <p class="freeform hinted">{html_mod.escape(v.get('freeform_gesture_description', ''))}</p>
    <p><b>Q2 — blind freeform (video only)</b></p>
    <p class="freeform blind">{html_mod.escape(b.get('freeform_gesture_description', ''))}</p>
  </div>
</article>"""
        )

    n = verify["n"]
    yes = verify.get("represents_target_cue_true", "?")
    rate = verify.get("represents_target_cue_rate", 0)
    page = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GT-o picked pose motions · VLM review</title>
  <style>
    :root {{ font-family: system-ui, -apple-system, sans-serif; background: #f4f4f5; color: #18181b; }}
    body {{ margin: 0; padding: 24px; max-width: 1200px; margin-inline: auto; }}
    h1 {{ font-size: 1.35rem; margin: 0 0 8px; }}
    .summary {{ color: #52525b; margin-bottom: 24px; line-height: 1.5; }}
    .card {{ background: #fff; border: 1px solid #e4e4e7; border-radius: 12px; padding: 16px; margin-bottom: 20px; }}
    .card.yes {{ border-left: 4px solid #16a34a; }}
    .card.no {{ border-left: 4px solid #dc2626; }}
    header {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }}
    h2 {{ margin: 0; font-size: 1.1rem; }}
    .idx {{ color: #71717a; font-weight: normal; font-size: 0.9rem; }}
    .badge {{ font-size: 0.75rem; font-weight: 700; padding: 4px 10px; border-radius: 999px; }}
    .card.yes .badge {{ background: #dcfce7; color: #166534; }}
    .card.no .badge {{ background: #fee2e2; color: #991b1b; }}
    .meta {{ font-size: 0.88rem; color: #3f3f46; margin: 10px 0 14px; line-height: 1.45; }}
    .media {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    @media (max-width: 800px) {{ .media {{ grid-template-columns: 1fr; }} }}
    .label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; color: #71717a; margin-bottom: 6px; }}
    video, img {{ width: 100%; max-height: 320px; object-fit: contain; background: #fafafa; border-radius: 8px; border: 1px solid #e4e4e7; }}
    .vlm {{ margin-top: 14px; font-size: 0.92rem; line-height: 1.5; }}
    .reason, .freeform {{ margin: 6px 0 12px; color: #27272a; }}
    .freeform.blind {{ background: #eff6ff; padding: 10px; border-radius: 8px; border-left: 3px solid #2563eb; }}
    .freeform.hinted {{ background: #fafafa; padding: 10px; border-radius: 8px; }}
    code {{ background: #f4f4f5; padding: 2px 6px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>GT-o cues · picked tile pose · rendered motion · VLM</h1>
  <p class="summary">
    Human GT <code>o</code> (14 cues) · picked tile start pose · config movements unchanged<br>
    <b>represents_target_cue:</b> {yes}/{n} ({rate * 100:.1f}%)<br>
    <b>Blind freeform</b> updated — no cue/description in that prompt.
  </p>
  {''.join(rows)}
</body>
</html>
"""
    out.write_text(page, encoding="utf-8")
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--vlm-only", action="store_true")
    ap.add_argument(
        "--vlm-blind-freeform-only",
        action="store_true",
        help="Re-run blind freeform VLM (video only, no cue) and refresh HTML.",
    )
    ap.add_argument("--html-only", action="store_true")
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--no-skip", action="store_true")
    args = ap.parse_args()

    if args.html_only:
        p = write_review_html()
        print(f"wrote {p}")
        return

    manifest: list[dict[str, Any]] | None = None
    if args.vlm_blind_freeform_only:
        run_vlm_blind_freeform(model=args.model)
        p = write_review_html()
        print(f"wrote {p}")
        return

    if not args.vlm_only:
        manifest = run_render(skip_existing=not args.no_skip)
    if not args.render_only:
        run_vlm(model=args.model, manifest=manifest)
        write_review_html()


if __name__ == "__main__":
    main()
