#!/usr/bin/env python3
"""Compute paper factor tables from canonical traces and human annotations.

Table II reports pose and movement agreement, both by interface and for the
human YES/NO reference groups. Table III reports expression diversity by
interface, with human-authored GT as the reference rather than YES/NO groups.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
TRACE_DIR = REPO / "data/results/evaluation/motion_similarity_3d/canonical_traces"
PHASE_PATH = REPO / "data/results/evaluation/method_comparison/analysis_fail/pose_phase_index.json"
VELOCITY_PATH = REPO / "data/results/evaluation/motion_similarity_3d/pairwise_3d_scores.csv"
ANNOTATION_PATH = REPO / "data/results/evaluation/embodiment_cue_id/all_methods_human_yesno_c1_30.xlsx"
OUTPUT_PATH = REPO / "data/results/evaluation/method_comparison/analysis_fail/factor_metrics_by_annotation.json"

METHODS = {
    "Joint-Space": "joint",
    "Cartesian-Space": "xyzrpy",
    "Motion Program": "motion_program",
    "Trajectory Sketching": "trajectory",
}
PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "roomba")
HUMANLIKE = {"humanoid", "tiago"}
SINGLE_ARM = {"single_arm"}
NON_HUMANLIKE = {"stretch", "roomba"}
SCOPES = {
    "humanlike": HUMANLIKE,
    "single_arm": SINGLE_ARM,
    "non_humanoid": NON_HUMANLIKE,
}


def _load_trace(platform: str, cue_idx: int, method: str) -> dict:
    row = np.load(TRACE_DIR / f"{platform}_c{cue_idx:02d}_{method}.npz", allow_pickle=True)
    return {
        "names": list(row["names"]),
        "xyz": np.asarray(row["xyz"], float),
        "rot": np.asarray(row["rot"], float),
    }


def _normalize_xyz(xyz: np.ndarray, platform: str) -> np.ndarray:
    out = xyz.copy()
    root = out[:, 0].copy()
    if platform == "roomba":
        out -= root[0, None, :]
        return out / 0.25
    out[:, 1:] -= root[:, None]
    out[:, 0] = root - root[0]
    scale = max(float(np.median(np.linalg.norm(xyz[0, 1:] - xyz[0, 0], axis=1))), 0.15)
    return out / scale


def _pose_score(platform: str, cue_idx: int, method: str, phases: dict) -> dict:
    gt = _load_trace(platform, cue_idx, "gt")
    pred = _load_trace(platform, cue_idx, method)
    gt_idx = int(phases[f"{platform}|c{cue_idx:02d}|gt"]["pose_idx"])
    pred_idx = int(phases[f"{platform}|c{cue_idx:02d}|{method}"]["pose_idx"])
    gt_xyz = _normalize_xyz(gt["xyz"], platform)
    pred_xyz = _normalize_xyz(pred["xyz"], platform)
    hands = [name for name in gt["names"] if "hand" in name and name in pred["names"]]
    if not hands:
        raise ValueError(f"No shared hand landmarks: {platform} c{cue_idx:02d} {method}")

    distances, forward_angles, palm_angles = [], [], []
    for hand in hands:
        gi, pi = gt["names"].index(hand), pred["names"].index(hand)
        distances.append(np.linalg.norm(gt_xyz[gt_idx, gi] - pred_xyz[pred_idx, pi]))
        for axis, angles in ((0, forward_angles), (2, palm_angles)):
            cosine = float(np.clip(gt["rot"][gt_idx, gi, :, axis] @ pred["rot"][pred_idx, pi, :, axis], -1, 1))
            angles.append(math.acos(cosine))
    return {
        "location": math.exp(-float(np.mean(distances)) / 0.5),
        "ee_direction": (1 + math.cos(float(np.mean(forward_angles)))) / 2,
        "palm": (1 + math.cos(float(np.mean(palm_angles)))) / 2,
    }


def _movement_handles(trace: dict, platform: str) -> list[int]:
    names = trace["names"]
    if platform in HUMANLIKE:
        return [i for i, name in enumerate(names) if "hand" in name]
    if platform in {"single_arm", "stretch"}:
        return [names.index("hand")]
    return [names.index(name) for name in ("base", "heading") if name in names]


def _movement_segment(trace: dict, platform: str, cue_idx: int, method: str, phases: dict) -> np.ndarray:
    xyz = _normalize_xyz(trace["xyz"], platform)
    handles = _movement_handles(trace, platform)
    phase = phases.get(f"{platform}|c{cue_idx:02d}|{method}")
    if phase is None:
        return xyz[:, handles]
    return xyz[int(phase["move_lo"]): int(phase["move_hi"]) + 1, handles]


def _movement_eligible(platform: str, cue_idx: int, phases: dict) -> bool:
    gt = _load_trace(platform, cue_idx, "gt")
    segment = _movement_segment(gt, platform, cue_idx, "gt", phases)
    return float(np.linalg.norm(segment.max(axis=0) - segment.min(axis=0))) > 0.06


def _movement_score(platform: str, cue_idx: int, method: str, phases: dict, velocity: dict) -> dict:
    gt = _load_trace(platform, cue_idx, "gt")
    pred = _load_trace(platform, cue_idx, method)
    gt_segment = _movement_segment(gt, platform, cue_idx, "gt", phases)
    pred_segment = _movement_segment(pred, platform, cue_idx, method, phases)
    gt_span = gt_segment.max(axis=0) - gt_segment.min(axis=0)
    pred_span = pred_segment.max(axis=0) - pred_segment.min(axis=0)
    active_gt_axes = gt_span > 0.04
    axis = float(np.mean(pred_span[active_gt_axes] > 0.03)) if np.any(active_gt_axes) else 0.0

    # Symmetric span overlap over all body landmarks; 1 means equal travel extent.
    gt_full = _normalize_xyz(gt["xyz"], platform)
    pred_full = _normalize_xyz(pred["xyz"], platform)
    gt_body_span = gt_full.max(axis=0) - gt_full.min(axis=0)
    pred_body_span = pred_full.max(axis=0) - pred_full.min(axis=0)
    extent = float(
        np.minimum(gt_body_span, pred_body_span).sum()
        / max(float(np.maximum(gt_body_span, pred_body_span).sum()), 1e-9)
    )
    dynamics = math.exp(-velocity[(platform, cue_idx, method)])
    return {"axis": axis, "extent": extent, "dynamics": dynamics}


def _diversity_score(platform: str, cue_idx: int, method: str) -> dict:
    trace = _load_trace(platform, cue_idx, method)
    xyz = _normalize_xyz(trace["xyz"], platform)
    sample_ids = np.linspace(0, len(xyz) - 1, min(24, len(xyz))).round().astype(int)
    xyz = xyz[sample_ids]
    span = xyz.max(axis=0) - xyz.min(axis=0)
    max_displacement = np.linalg.norm(xyz - xyz[:1], axis=2).max(axis=0)
    return {
        "channel": float((max_displacement > 0.04).mean()),
        "direction": float((span > 0.04).mean()),
    }


def _mean_group(cells: list[dict], metrics: tuple[str, ...]) -> dict:
    result = {"n": len(cells)}
    for metric in metrics:
        result[metric] = float(np.mean([cell[metric] for cell in cells]))
    return result


def main() -> None:
    phases = json.loads(PHASE_PATH.read_text(encoding="utf-8"))
    annotations = pd.read_excel(ANNOTATION_PATH, sheet_name="Human yes-no")
    velocity_df = pd.read_csv(VELOCITY_PATH)
    velocity = {
        (str(row.robot), int(row.idx), str(row.a)): float(row.velocity_dtw)
        for row in velocity_df.itertuples()
        if row.b == "gt" and row.a in METHODS.values()
    }

    labels = {}
    for _, row in annotations.iterrows():
        embodiment = str(row["embodiment_index"]).lower()
        platform = next((name for name in PLATFORMS if name in embodiment), None)
        cue_match = re.match(r"c(\d+)", str(row["cue"]), flags=re.IGNORECASE)
        if platform is None or cue_match is None:
            continue
        cue_idx = int(cue_match.group(1))
        for column, method in METHODS.items():
            label = str(row[column]).strip().upper()
            if label in {"YES", "NO"}:
                labels[(platform, cue_idx, method)] = label

    pose_cells, movement_cells, diversity_cells = [], [], []
    skipped = []
    for (platform, cue_idx, method), label in labels.items():
        try:
            if platform in HUMANLIKE:
                pose_cells.append({"scope": "humanlike", "method": method, "annotation": label, **_pose_score(platform, cue_idx, method, phases)})
            if platform == "single_arm":
                pose_cells.append({"scope": "single_arm", "method": method, "annotation": label, **_pose_score(platform, cue_idx, method, phases)})
            scope = next(name for name, plats in SCOPES.items() if platform in plats)
            if _movement_eligible(platform, cue_idx, phases):
                movement_cells.append({"scope": scope, "method": method, "annotation": label, **_movement_score(platform, cue_idx, method, phases, velocity)})
            diversity_cells.append({"scope": scope, "method": method, **_diversity_score(platform, cue_idx, method)})
        except (FileNotFoundError, KeyError, ValueError) as exc:
            skipped.append({"platform": platform, "cue_idx": cue_idx, "method": method, "reason": str(exc)})

    # Human-authored diversity is computed directly from all 30 GT cues/body.
    for scope, platforms in SCOPES.items():
        for platform in platforms:
            for cue_idx in range(1, 31):
                try:
                    diversity_cells.append({"scope": scope, "method": "gt", **_diversity_score(platform, cue_idx, "gt")})
                except FileNotFoundError:
                    pass

    grouped = {}
    for label in ("YES", "NO"):
        grouped[label] = {
            "humanlike_pose": _mean_group(
                [x for x in pose_cells if x["scope"] == "humanlike" and x["annotation"] == label],
                ("location", "ee_direction", "palm"),
            ),
            "single_arm_pose": _mean_group(
                [x for x in pose_cells if x["scope"] == "single_arm" and x["annotation"] == label],
                ("location", "ee_direction", "palm"),
            ),
        }
        for scope in SCOPES:
            grouped[label][f"{scope}_movement"] = _mean_group(
                [x for x in movement_cells if x["scope"] == scope and x["annotation"] == label],
                ("axis", "extent", "dynamics"),
            )

    by_method = {}
    for display, method in METHODS.items():
        by_method[display] = {
            "humanlike_pose": _mean_group([x for x in pose_cells if x["scope"] == "humanlike" and x["method"] == method], ("location", "ee_direction", "palm")),
            "single_arm_pose": _mean_group([x for x in pose_cells if x["scope"] == "single_arm" and x["method"] == method], ("location", "ee_direction", "palm")),
        }
        for scope in SCOPES:
            by_method[display][f"{scope}_movement"] = _mean_group(
                [x for x in movement_cells if x["scope"] == scope and x["method"] == method],
                ("axis", "extent", "dynamics"),
            )

    diversity = {}
    for display, method in {**METHODS, "Human-authored": "gt"}.items():
        diversity[display] = {
            scope: _mean_group([x for x in diversity_cells if x["scope"] == scope and x["method"] == method], ("channel", "direction"))
            for scope in SCOPES
        }

    result = {
        "scope": {
            "humanlike_pose": "Humanoid + Tiago",
            "single_arm_pose_and_movement": "Single Arm",
            "non_humanoid_movement": "Stretch + Roomba",
        },
        "by_method": by_method,
        "grouped_by_human_annotation": grouped,
        "diversity_by_method_with_human_authored_reference": diversity,
        "formulas": {
            "pose_location": "exp(-mean_hand_distance_in_embodiment_normalized_coordinates/0.5)",
            "pose_axes": "(1+cos(mean_hand_axis_angle))/2",
            "movement_axis": "all GT-active signed travel axes covered",
            "movement_extent": "symmetric body-landmark span overlap",
            "movement_dynamics": "exp(-Velocity-DTW_to_GT)",
            "channel_diversity": "share of landmarks displaced by >0.04",
            "direction_diversity": "share of landmark coordinate axes spanning >0.04",
        },
        "source_annotation": str(ANNOTATION_PATH.relative_to(REPO)),
        "skipped": skipped,
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"by_method": by_method, "grouped": grouped, "diversity": diversity, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
