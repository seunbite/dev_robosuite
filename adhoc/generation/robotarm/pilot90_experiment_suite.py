"""Pilot-90 (non-essence manifest cues) pose experiment map for Qwen suite."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pilot40_experiment_suite import (  # noqa: F401 — re-export shared helpers
    CONSOLIDATED,
    EXPERIMENT_SPECS as PILOT40_EXPERIMENT_SPECS,
    PROMPT_POSE_GEN,
    SHOTS,
    TILE_DIR,
    TILE_PICK,
    _parse_gt_poses,
    human_gt_pose_ok,
    load_consolidated_by_cue,
    metrics_from_json,
    pose_generation_correct,
    print_summary_table as _print_summary_table,
)

_REPO = Path(__file__).resolve().parent.parents[2]

POSE_CFG = (
    _REPO
    / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json"
)
MANIFEST_TSV = _REPO / "data/seed/yml/pilot100_manifest.tsv"
PAIRWISE_IMG_DIR = _REPO / "data/results/visualize/pose_pairwise_12_pilot90"
MULTITILE_IMG_DIR = _REPO / "data/results/visualize/pose_multitile_gt_pilot90"

N_CUES = 90
POSE_STEP_IDS = frozenset({"1", "2", "3", "4", "5", "6"})


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


def experiment_specs_pose_only() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for spec in PILOT40_EXPERIMENT_SPECS:
        if spec["id"] not in POSE_STEP_IDS:
            continue
        s = dict(spec)
        s["input_config"] = str(POSE_CFG.relative_to(_REPO))
        specs.append(s)
    return specs


def print_summary_table(
    specs: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    *,
    n_cues: int = N_CUES,
) -> None:
    print("\n" + "=" * 92)
    print(f"PILOT-90 ({n_cues} cues, pose steps 1–6) — QWEN ACCURACY SUMMARY")
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
