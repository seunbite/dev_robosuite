"""Pilot-40 (39 cues) experiment map: prompts, Gemini baselines, Qwen output paths."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parents[2]

# --- prompts ---
PROMPT_POSE_GEN = _REPO / "data/seed/prompt/manipulator/prompt_v19_sophisticated.txt"
PROMPT_MOTION_GEN = _REPO / "data/seed/prompt/manipulator/prompt_gt_fixed_first_pose.txt"
PROMPT_MOTION_PAIRWISE = (
    _REPO / "data/seed/prompt/manipulator/prompt_motion_gt_neg_pairwise_alpha.txt"
)

# --- inputs ---
POSE_CFG = (
    _REPO
    / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_generation_pose_pilot40.json"
)
MOTION_CFG = (
    _REPO
    / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_gt_fixed_pose_pilot40.json"
)
CONSOLIDATED = _REPO / "data/results/verify/pilot40_pose_eval_consolidated.json"
MOTION_MANIFEST = (
    _REPO / "data/results/render/manipulator/motion_vlm_verify_pilot40/manifest_pilot40.json"
)
SHOTS = _REPO / "data/seed/shots/manipulator/shot_configs_v19_sophisticated.json"
TILE_DIR = _REPO / "data/results/visualize/pose_groups_12"
TILE_PICK = _REPO / "data/results/verify/pose_tile_pick_by_group.json"
MOTION_PAIRWISE_DIR = _REPO / "data/results/verify/samples/motion_gt_neg_pairwise"

DEFAULT_QWEN_OUT = _REPO / "data/results/verify/pilot40_qwen32b"

# Gemini reference outputs (steps 1–4, 7–10 already run; 5–6 are new for Qwen)
GEMINI_REF: dict[int, dict[str, str]] = {
    1: {
        "configs": str(POSE_CFG.relative_to(_REPO)),
        "scored_tsv": "data/results/verify/pilot40_pose_eval_consolidated_scored.tsv",
        "consolidated": str(CONSOLIDATED.relative_to(_REPO)),
    },
    2: {
        "vlm_verify": "data/results/verify/pose_tile_verify_pilot{10,20,20_more}_gemini.json (merged in consolidated)",
        "note": "Qwen re-run uses POSE_CFG (39 cues) single JSON",
    },
    3: {
        "text_verify": "data/results/verify/pose_textonly_verify_pilot{10,20,20_more}_gemini.json",
    },
    4: {
        "pairwise": "data/results/verify/pilot40_pose_pairwise_12_gemini.json",
    },
    5: {
        "multitile_grid6": "NEW — no pilot40 Gemini grid-6-only run",
        "closest": "data/results/verify/pilot20_pose_multitile_gt_gemini.json (pilot20 only)",
    },
    6: {
        "multitile_grid12": "NEW — no pilot40 Gemini grid-12-only run",
        "closest": "data/results/verify/pilot20_pose_multitile_gt_gemini.json (pilot20 only)",
    },
    7: {
        "configs": str(MOTION_CFG.relative_to(_REPO)),
        "component_gt": "data/results/verify/pilot40_motion_component_gt.json",
        "metrics": "data/results/verify/pilot40_motion_verify_metrics.json",
    },
    8: {
        "vlm_verify": "data/results/verify/pilot40_motion_component_verify_gemini.json",
    },
    9: {
        "text_verify": "data/results/verify/pilot40_motion_component_verify_text_gemini.json",
    },
    10: {
        "pairwise_mp4": "data/results/verify/samples/motion_gt_neg_pairwise/pairwise_eval_results*.json",
    },
}

EXPERIMENT_SPECS: list[dict[str, Any]] = [
    {
        "id": "1",
        "title": "Pose generation vs human GT",
        "kind": "pose_generation_score",
        "prompt": str(PROMPT_POSE_GEN.relative_to(_REPO)),
        "out_name": "exp01_pose_generation_score.json",
    },
    {
        "id": "2",
        "title": "Pose verify + regenerate — VLM (tile)",
        "kind": "pose_verify_vlm",
        "prompt": "verify_pose_tiles_gemini (few-shot in script)",
        "out_name": "exp02_pose_verify_vlm.json",
    },
    {
        "id": "3",
        "title": "Pose verify + regenerate — text",
        "kind": "pose_verify_text",
        "prompt": "verify_pose_textonly_gemini",
        "out_name": "exp03_pose_verify_text.json",
    },
    {
        "id": "4",
        "title": "Pose pairwise (VLM) — 2-way",
        "kind": "pose_pairwise",
        "out_name": "exp04_pose_pairwise_2way.json",
    },
    {
        "id": "5",
        "title": "Pose pairwise (VLM) — multitile grid 6",
        "kind": "multitile",
        "grid_sizes": "6",
        "out_name": "exp05_pose_multitile_grid6.json",
    },
    {
        "id": "6",
        "title": "Pose pairwise (VLM) — multitile grid 12",
        "kind": "multitile",
        "grid_sizes": "12",
        "out_name": "exp06_pose_multitile_grid12.json",
    },
    {
        "id": "7",
        "title": "Movement generation vs component GT",
        "kind": "motion_generation_score",
        "prompt": str(PROMPT_MOTION_GEN.relative_to(_REPO)),
        "out_name": "exp07_motion_generation_score.json",
    },
    {
        "id": "8",
        "title": "Movement verify + regenerate — VLM (MP4)",
        "kind": "motion_verify_vlm",
        "out_name": "exp08_motion_verify_vlm.json",
    },
    {
        "id": "9",
        "title": "Movement verify + regenerate — text",
        "kind": "motion_verify_text",
        "out_name": "exp09_motion_verify_text.json",
    },
    {
        "id": "10",
        "title": "Movement pairwise (VLM — MP4)",
        "kind": "motion_pairwise_mp4",
        "prompt": str(PROMPT_MOTION_PAIRWISE.relative_to(_REPO)),
        "out_name": "exp10_motion_pairwise_mp4.json",
    },
]


def pilot40_cue_names(consolidated_path: Path | None = None) -> list[str]:
    path = consolidated_path or CONSOLIDATED
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(data.get("rows") or [], key=lambda r: int(r.get("cue_idx", 0)))
    seen: set[str] = set()
    out: list[str] = []
    for r in rows:
        cue = r.get("cue")
        if cue and cue not in seen:
            seen.add(cue)
            out.append(str(cue))
    return out


def pilot40_cues_csv(consolidated_path: Path | None = None) -> str:
    return ",".join(pilot40_cue_names(consolidated_path))


def _parse_gt_poses(groundtruth: str) -> list[tuple[str, str]]:
    return [(a.strip(), b.strip()) for a, b in re.findall(r"\(([^,]+),\s*([^)]+)\)", groundtruth)]


def pose_generation_correct(generated: dict[str, Any] | None, groundtruth: str) -> bool | None:
    """Match pilot40_pose_eval_consolidated_scored.tsv generation_correct logic."""
    if not generated or not groundtruth:
        return None
    gt = groundtruth.strip()
    poses = _parse_gt_poses(gt)
    if not poses:
        return None
    d = str(generated.get("dir", "")).strip()
    g = str(generated.get("gripper_orientation", "")).strip()
    gen = (d, g)
    if gt.lower().startswith("o"):
        return gen == poses[0]
    if gt.lower().startswith("x"):
        return gen in poses
    return None


def load_consolidated_by_cue(path: Path | None = None) -> dict[str, dict[str, Any]]:
    p = path or CONSOLIDATED
    data = json.loads(p.read_text(encoding="utf-8"))
    return {str(r["cue"]): r for r in (data.get("rows") or []) if r.get("cue")}


def human_gt_pose_ok(groundtruth: str) -> bool:
    return str(groundtruth or "").strip().lower().startswith("o")


def metrics_from_json(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}

    data = json.loads(path.read_text(encoding="utf-8"))
    kind = spec["kind"]

    if kind == "pose_generation_score":
        ok = int(data.get("n_correct", 0))
        n = int(data.get("n", 0))
        acc = ok / n if n else None
        return {
            "status": "ok",
            "path": str(path),
            "ok": ok,
            "n": n,
            "accuracy": acc,
            "accuracy_pct": None if acc is None else round(100 * acc, 1),
            "headline": f"{ok}/{n} = {100 * acc:.1f}%" if acc is not None else "n/a",
        }

    if kind == "motion_generation_score":
        ok = int(data.get("n_correct", 0))
        n = int(data.get("n", 0))
        acc = ok / n if n else None
        return {
            "status": "ok",
            "path": str(path),
            "ok": ok,
            "n": n,
            "accuracy": acc,
            "accuracy_pct": None if acc is None else round(100 * acc, 1),
            "headline": f"{ok}/{n} = {100 * acc:.1f}%" if acc is not None else "n/a",
        }

    if kind == "multitile":
        summary = data.get("summary") or {}
        grid = spec.get("grid_sizes", "6")
        key = f"grid_{grid}"
        s = summary.get(key, {})
        acc = s.get("accuracy")
        ok, n = s.get("ok"), s.get("n")
        headline = f"{ok}/{n} = {100 * acc:.1f}%" if acc is not None else "n/a"
        return {
            "status": "ok",
            "path": str(path),
            "ok": ok,
            "n": n,
            "accuracy": acc,
            "accuracy_pct": None if acc is None else round(100 * acc, 1),
            "headline": headline,
        }

    if kind in {"pose_pairwise", "motion_pairwise_mp4"}:
        rows = data.get("comparisons") or data.get("mp4") or []
        scored = [c for c in rows if c.get("vlm_correct") is not None or c.get("correct") is not None]
        ok = sum(
            1
            for c in scored
            if (c.get("vlm_correct") if "vlm_correct" in c else c.get("correct"))
        )
        n = len(scored)
        acc = ok / n if n else None
        return {
            "status": "ok",
            "path": str(path),
            "ok": ok,
            "n": n,
            "accuracy": acc,
            "accuracy_pct": None if acc is None else round(100 * acc, 1),
            "headline": f"{ok}/{n} = {100 * acc:.1f}%" if acc is not None else "n/a",
        }

    if kind in {"pose_verify_vlm", "pose_verify_text"}:
        results = data.get("results") or []
        scored = [r for r in results if "error" not in r and isinstance(r.get("result"), dict)]
        human = load_consolidated_by_cue()
        agree_ok = agree_n = 0
        for r in scored:
            cue = r.get("cue")
            if not cue or cue not in human:
                continue
            model_ok = r.get("result", {}).get("pose_is_appropriate")
            if model_ok is None:
                continue
            agree_n += 1
            human_ok = human_gt_pose_ok(human[cue].get("groundtruth", ""))
            if bool(model_ok) == human_ok:
                agree_ok += 1
        acc = agree_ok / agree_n if agree_n else None
        return {
            "status": "ok",
            "path": str(path),
            "ok": agree_ok,
            "n": agree_n,
            "accuracy": acc,
            "accuracy_pct": None if acc is None else round(100 * acc, 1),
            "headline": f"human-agree {agree_ok}/{agree_n}"
            + (f" = {100 * acc:.1f}%" if acc is not None else ""),
        }

    if kind in {"motion_verify_vlm", "motion_verify_text"}:
        rows = data.get("rows") or []
        det_ok = det_n = 0
        from score_pilot40_motion_gt_components import (  # noqa: WPS433
            _build_annotation_map,
            _tail_matches_component,
            _tail_steps,
        )

        ann = {int(a["cue_idx"]): a for a in _build_annotation_map()}
        cfg_by = {int(r["idx"]): r for r in json.loads(MOTION_CFG.read_text(encoding="utf-8"))}
        for vr in rows:
            idx = int(vr["cue_idx"])
            comp = (ann.get(idx) or {}).get("component")
            row = cfg_by.get(idx)
            if not comp or not row:
                continue
            gen_tail = _tail_steps(row.get("movements") or [])
            gen_match, _ = _tail_matches_component(gen_tail, comp)
            appropriate = vr.get("movement_is_appropriate")
            if appropriate is None:
                continue
            det_n += 1
            if bool(appropriate) == bool(gen_match):
                det_ok += 1
        acc = det_ok / det_n if det_n else None
        return {
            "status": "ok",
            "path": str(path),
            "ok": det_ok,
            "n": det_n,
            "accuracy": acc,
            "accuracy_pct": None if acc is None else round(100 * acc, 1),
            "headline": f"detection {det_ok}/{det_n}"
            + (f" = {100 * acc:.1f}%" if acc is not None else ""),
        }

    return {"status": "unknown", "path": str(path)}


def print_summary_table(specs: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 92)
    print("PILOT-40 (39 cues) QWEN SUITE — ACCURACY SUMMARY")
    print("=" * 92)
    print(f"{'#':<4}  {'experiment':<46}  {'result'}")
    print("-" * 92)
    for spec, m in zip(specs, metrics):
        headline = m.get("headline", m.get("status", "?"))
        if m.get("status") == "missing":
            headline = "MISSING JSON"
        elif m.get("status") == "skipped":
            headline = m.get("note", "skipped")
        print(f"{spec['id']:<4}  {spec['title']:<46}  {headline}")
    print("=" * 92)
