"""Pilot-90 (non-essence manifest cues) experiment map for Qwen suite (10 steps)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pilot40_experiment_suite as p40
from pilot40_experiment_suite import (  # noqa: F401 — re-export shared helpers
    CONSOLIDATED,
    EXPERIMENT_SPECS as PILOT40_EXPERIMENT_SPECS,
    PROMPT_MOTION_GEN,
    PROMPT_POSE_GEN,
    SHOTS,
    TILE_DIR,
    TILE_PICK,
    _parse_gt_poses,
    human_gt_pose_ok,
    load_consolidated_by_cue,
    pose_generation_correct,
)

_REPO = Path(__file__).resolve().parent.parents[2]

POSE_CFG = (
    _REPO
    / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json"
)
MOTION_CFG = POSE_CFG
MOTION_COMPONENT_GT = _REPO / "data/results/verify/pilot40_motion_component_gt.json"
MOTION_MANIFEST = _REPO / "data/results/render/manipulator/motion_vlm_verify_pilot90/manifest_pilot90.json"
MOTION_PAIRWISE_DIR = _REPO / "data/results/verify/samples/motion_gt_neg_pairwise_pilot90"
MANIFEST_TSV = _REPO / "data/seed/yml/pilot100_manifest.tsv"
PAIRWISE_IMG_DIR = _REPO / "data/results/visualize/pose_pairwise_12_pilot90"
MULTITILE_IMG_DIR = _REPO / "data/results/visualize/pose_multitile_gt_pilot90"

N_CUES = 90


def qwen_out_dir(model_tag: str = "qwen32b") -> Path:
    return _REPO / "data/results/verify" / f"pilot90_{model_tag}"


DEFAULT_QWEN_OUT = qwen_out_dir("qwen32b")


def manifest90_cue_names() -> list[str]:
    out: list[str] = []
    for line in MANIFEST_TSV.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 4 or parts[1] == "pending_essence10":
            continue
        out.append(parts[3])
    return out


def manifest90_cues_csv() -> str:
    return ",".join(manifest90_cue_names())


def manifest90_rows_from_cfg(cfg_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest = set(manifest90_cue_names())
    return [r for r in cfg_rows if r.get("cue") in manifest]


def _all_poses_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    poses: list[dict[str, Any]] = []
    for step in row.get("movements") or []:
        if step.get("type") != "pose":
            continue
        pose = (step.get("parameters") or {}).get("pose") or {}
        if pose.get("dir") and pose.get("gripper_orientation"):
            poses.append(pose)
    return poses


def pose_generation_correct_any(row: dict[str, Any], groundtruth: str) -> bool | None:
    """True if any pose step in config matches human GT (any-pose rule)."""
    if not row or not groundtruth:
        return None
    targets = _parse_gt_poses(groundtruth.strip())
    if not targets:
        return None
    gen_set = {
        (str(p.get("dir", "")).strip(), str(p.get("gripper_orientation", "")).strip())
        for p in _all_poses_from_row(row)
    }
    if groundtruth.strip().lower().startswith("o"):
        return targets[0] in gen_set
    return any(t in gen_set for t in targets)


def experiment_specs_all() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for spec in PILOT40_EXPERIMENT_SPECS:
        s = dict(spec)
        if s["id"] in {"1", "2", "3", "4", "5", "6"}:
            s["input_config"] = str(POSE_CFG.relative_to(_REPO))
        elif s["id"] in {"7", "8", "9", "10"}:
            s["input_config"] = str(MOTION_CFG.relative_to(_REPO))
        if s["id"] == "8":
            s["media_dir"] = "data/results/render/manipulator/motion_vlm_verify_pilot90/mp4"
        specs.append(s)
    return specs


def metrics_from_json(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    old_motion = p40.MOTION_CFG
    try:
        p40.MOTION_CFG = MOTION_CFG
        return p40.metrics_from_json(path, spec)
    finally:
        p40.MOTION_CFG = old_motion


def print_summary_table(
    specs: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    *,
    n_cues: int = N_CUES,
) -> None:
    print("\n" + "=" * 92)
    print(f"PILOT-90 ({n_cues} cues, 10 steps) — QWEN ACCURACY SUMMARY")
    print("=" * 92)
    print(f"{'#':<4}  {'experiment':<46}  {'result'}")
    print("-" * 92)
    for spec, m in zip(specs, metrics):
        headline = m.get("headline", m.get("status", "?"))
        if m.get("status") == "missing":
            headline = "MISSING JSON"
        elif m.get("status") == "skipped":
            headline = m.get("note", "skipped")
        elif m.get("status") == "error":
            headline = f"ERROR: {m.get('error', '?')[:40]}"
        print(f"{spec['id']:<4}  {spec['title']:<46}  {headline}")
    print("=" * 92)


# backward compat
experiment_specs_pose_only = lambda: [s for s in experiment_specs_all() if s["id"] in {"1", "2", "3", "4", "5", "6"}]
