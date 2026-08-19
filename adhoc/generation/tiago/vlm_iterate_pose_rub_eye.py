#!/usr/bin/env python3
"""Iteratively refine mobile-manip *initial pose* for ``rub_eye_tired``, judged by VLM on renders.

Workflow (one cue hard-coded):
  1) Build a pose-only clip (single ``pose`` step), render GIF.
  2) Ask VLM: does this stance read as an appropriate setup for rubbing a tired eye?
  3) If No: Gemini revises only the pose dict → repeat.
  4) If Yes: attach the scripted ``movement`` step from seed config, render full motion, save artifacts.

Artifacts under ``data/results/vlm_pose_iterate/rub_eye_tired_<timestamp>/``::
  manifest.jsonl, iter_XXX/config.json, iter_XXX/pose_preview.gif,
  iter_XXX/vlm_judge.txt, iter_XXX/refine_prompt.txt (if any), …
  ``final_full_config.json``, ``final_full_motion.gif``

Requires ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``. Judge & refine use Gemini by default via
``adhoc/vlm_test/vlm_utils.py`` / ``google.genai``."""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adhoc.generation.tiago.legacy.render_mobile_config import CAM
from adhoc.generation.tiago.legacy import (
    _make_env as mobile_make_env,
    render_config as mobile_render_config,
)
from adhoc.vlm_test.vlm_utils import vlm_multimodal_call

CUE_ID = "rub_eye_tired"
DEFAULT_SEED = _REPO / "data/results/motion_configs/tiago/motion_configs_hf_xyz_demo.json"
OUT_PARENT = _REPO / "data/results/vlm_pose_iterate"

_JUDGE_RE = re.compile(r"^\s*(VERDICT|Judgment)\s*:\s*(YES|NO)\b", re.I | re.M)

_POSE_KEYS = frozenset(
    {
        "torso_height",
        "arm_position",
        "gripper_orientation",
        "head",
        "left_arm",
        "x",
        "y",
        "z",
    }
)


def _sanitize_pose(p: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in _POSE_KEYS:
        if k not in p:
            continue
        out[k] = p[k]
    return out


def _merge_pose(prev: dict[str, Any], llm_partial: dict[str, Any]) -> dict[str, Any]:
    out = dict(prev)
    out.update(_sanitize_pose(llm_partial))
    return out


def _load_rub_seed(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    for row in rows:
        if row.get("cue") == CUE_ID:
            return row
    raise SystemExit(f"No entry with cue={CUE_ID!r} in {path}")


def _clone_pose_only_config(seed_row: dict[str, Any], pose_params: dict[str, Any]) -> dict[str, Any]:
    m0 = (seed_row.get("movements") or [])[0]
    if not m0 or m0.get("type") != "pose":
        raise SystemExit("seed row must begin with pose")
    pose_block = dict(m0.get("parameters", {}).get("pose", {}) or {})
    pose_block.update(pose_params)
    return {
        "idx": seed_row.get("idx", -1),
        "cue": CUE_ID,
        "cue_text": seed_row.get("cue_text", ""),
        "group": seed_row.get("group", ""),
        "movements": [
            {"type": "pose", "duration": float(m0.get("duration", 0.5)), "parameters": {"pose": pose_block}}
        ],
    }


def _full_config(seed_row: dict[str, Any], final_pose: dict[str, Any]) -> dict[str, Any]:
    m = list(seed_row.get("movements") or [])
    if not m:
        raise SystemExit("seed has no movements")
    out_m = [{"type": "pose", **{k: v for k, v in m[0].items() if k != "parameters"}, "parameters": {"pose": dict(_sanitize_pose(final_pose))}}]
    out_m.extend(m[1:])
    base = {k: v for k, v in seed_row.items() if k != "movements"}
    base["movements"] = out_m
    base["cue"] = CUE_ID
    return base


def _save_gif(frames: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        frames[0].save(str(path), save_all=True, append_images=frames[1:], duration=50, loop=0)


def _extract_verdict(txt: str) -> str | None:
    """Return upper YES or NO if parseable."""
    m = _JUDGE_RE.search(txt or "")
    if m:
        return m.group(2).upper()
    t = (txt or "").strip().upper()
    if t.startswith("YES") or "\nYES" in t[:200]:
        return "YES"
    if t.startswith("NO") or "\nNO" in t[:200]:
        return "NO"
    return None


def _gemini_json_or_text(prompt: str, *, model_id: str) -> str:
    """Single-turn text-only Gemini call (reuse API key semantics from vlm_utils)."""
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY / GEMINI_API_KEY for refinement.")
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model=model_id, contents=prompt)
    return (resp.text or "").strip()


def _parse_pose_json(blob: str) -> dict[str, Any]:
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", blob, flags=re.I | re.S)
    blocks = fenced + [blob]
    decoder = json.JSONDecoder()
    for b in blocks:
        s = str(b).strip()
        i = s.find("{")
        if i < 0:
            continue
        try:
            obj, _ = decoder.raw_decode(s[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("could not decode pose JSON")


def _refine_pose_via_llm(
    *,
    model_id: str,
    seed_pose: dict[str, Any],
    judge_feedback: str,
    cue_text: str,
    description: str,
) -> dict[str, Any]:
    prompt = f"""You fix the FIRST KEYFRAME POSE JSON for TIAGo mobile manipulator cue \"{CUE_ID}\" only.

Cue text: {cue_text}
Intent: {description}

Current pose JSON (overwrite bad fields freely; keep schema valid):
{json.dumps(seed_pose, indent=2, ensure_ascii=False)}

VLM critic (must address this):
{judge_feedback}

Rules:
- Output ONE JSON object only (no markdown fences if possible) with keys among:
  torso_height (low|mid|high), arm_position, gripper_orientation (horizontal|vertical),
  head (center|left|right|up|down), left_arm optional, x y z optional (0-100 floats).
- arm_position one of: up, front, back, right, down+front, down+right, down+back, fold.
- For eye-rub prep: right hand should plausibly sit near face/eye level without absurd reach; prefer front + horizontal gripper when palm rubs eye.

JSON:"""
    raw = _gemini_json_or_text(prompt, model_id=model_id)
    return _parse_pose_json(raw)


def _append_manifest(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(
    seed_json: str | None = None,
    judge_model: str = "gemini-2.5-flash",
    refine_model: str | None = None,
    max_iter: int = 12,
    dry_run: bool = False,
) -> str:
    refine_model = refine_model or judge_model
    seed_path = Path(seed_json) if seed_json else DEFAULT_SEED
    seed_row = _load_rub_seed(seed_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUT_PARENT / f"{CUE_ID}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "seed_row.json").write_text(json.dumps(seed_row, indent=2, ensure_ascii=False), encoding="utf-8")

    m0 = seed_row["movements"][0]
    pose = dict(m0.get("parameters", {}).get("pose", {}) or {})

    judge_prompt = f"""You are grading a single video frame sequence of a TIAGo-class mobile manipulator (front view).
The intended micro-gesture prep is: \"{seed_row.get('cue_text', CUE_ID)}\" — the robot should look ready to rub its eye / near the eye with the right hand (tired-eye rub), not a random wave or point.

Reply with EXACTLY two lines:
Line 1: VERDICT: YES or NO
Line 2: One short reason (what works or what to fix: hand height, distance from face, wrong arm, pose unrelated, etc.)"""

    env = mobile_make_env()
    try:
        for it in range(int(max_iter)):
            sub = run_dir / f"iter_{it:03d}"
            sub.mkdir(parents=True, exist_ok=True)

            cfg_pose = _clone_pose_only_config(seed_row, pose)
            (sub / "config_pose_only.json").write_text(
                json.dumps(cfg_pose, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            frames = mobile_render_config(cfg_pose, env=env, camera=CAM, overlay_progress_bar=False)
            gif_path = sub / "pose_preview.gif"
            _save_gif(frames, gif_path)

            vlm_out = ""
            verdict: str | None = "YES" if dry_run else None
            if not dry_run:
                vlm_out = vlm_multimodal_call(
                    judge_model,
                    judge_prompt,
                    [(gif_path, "image/gif")],
                )
                (sub / "vlm_judge.txt").write_text(vlm_out, encoding="utf-8")
                verdict = _extract_verdict(vlm_out)

            if verdict is None:
                verdict = "NO"
                vlm_out = (vlm_out or "") + "\n(fallback: could not parse VERDICT; treating as NO)\n"
                (sub / "vlm_judge.txt").write_text(vlm_out, encoding="utf-8")

            _append_manifest(
                run_dir / "manifest.jsonl",
                {
                    "iter": it,
                    "verdict": verdict,
                    "pose": pose,
                    "gif": str(gif_path.relative_to(_REPO)),
                },
            )

            if verdict == "YES":
                final_cfg = _full_config(seed_row, pose)
                (run_dir / "final_full_config.json").write_text(
                    json.dumps(final_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                fr2 = mobile_render_config(final_cfg, env=env, camera=CAM, overlay_progress_bar=True)
                final_gif = run_dir / "final_full_motion.gif"
                _save_gif(fr2, final_gif)
                summary = {
                    "status": "ok",
                    "iterations": it + 1,
                    "run_dir": str(run_dir),
                    "seed_json": str(seed_path),
                }
                (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(json.dumps(summary, indent=2))
                return str(run_dir)

            if it == int(max_iter) - 1:
                break

            # Refine pose for next iter
            if dry_run:
                break
            try:
                crit = Path(sub / "vlm_judge.txt").read_text(encoding="utf-8")
                raw_pose = _refine_pose_via_llm(
                    model_id=refine_model,
                    seed_pose=pose,
                    judge_feedback=crit,
                    cue_text=str(seed_row.get("cue_text", "")),
                    description=str(seed_row.get("description", "")),
                )
                pose = _merge_pose(pose, raw_pose)
            except Exception as exc:
                (sub / "refine_error.txt").write_text(str(exc), encoding="utf-8")
                if "z" in pose:
                    pose["z"] = min(100.0, float(pose["z"]) + 3.0)
                else:
                    pose["z"] = 55.0

    finally:
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()

    summary = {"status": "fail", "max_iter": int(max_iter), "run_dir": str(run_dir), "seed_json": str(seed_path)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return str(run_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(run)
