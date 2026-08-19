#!/usr/bin/env python3
"""Quantify diversity, pose fidelity, and expression overlap from cached 3-D traces."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
TRACE_DIR = REPO / "data/results/evaluation/motion_similarity_3d/canonical_traces"
DTW = REPO / "data/results/evaluation/motion_similarity_3d/pairwise_3d_scores.csv"
OUT = REPO / "data/results/evaluation/method_comparison"
PLATFORMS = ("humanoid", "tiago", "stretch", "single_arm", "roomba")
METHODS = ("joint", "xyzrpy", "motion_program", "trajectory", "gt")
GENERATED = METHODS[:-1]
LABEL = {"joint": "Joint-Space", "xyzrpy": "Cartesian-Space", "motion_program": "Motion Program",
         "trajectory": "Trajectory Sketching", "gt": "Human-authored"}
COLORS = {"joint": "#202020", "xyzrpy": "white", "motion_program": "#1F6FEB",
          "trajectory": "white", "gt": "#808080"}
HATCH = {"joint": None, "xyzrpy": None, "motion_program": None, "trajectory": "///", "gt": None}


def load_trace(platform: str, idx: int, method: str) -> dict | None:
    path = TRACE_DIR / f"{platform}_c{idx:02d}_{method}.npz"
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=False)
    return {"names": data["names"].tolist(), "xyz": np.asarray(data["xyz"], float), "hz": int(data["hz"][0])}


def normalize(trace: dict, platform: str) -> np.ndarray:
    x = trace["xyz"].copy()
    root = x[:, 0].copy()
    if platform == "roomba":
        x -= root[0, None, :]
        return x / 0.25
    x[:, 1:] -= root[:, None, :]
    x[:, 0] = root - root[0]
    scale = max(float(np.median(np.linalg.norm(trace["xyz"][0, 1:] - trace["xyz"][0, 0], axis=1))), 0.15)
    return x / scale


def resample(x: np.ndarray, n: int = 24) -> np.ndarray:
    ids = np.linspace(0, len(x) - 1, min(n, len(x))).round().astype(int)
    return x[ids]


def feature(trace: dict, platform: str) -> tuple[dict, np.ndarray, set[str], np.ndarray]:
    x = resample(normalize(trace, platform))
    duration_s = len(trace["xyz"]) / trace["hz"]
    sample_hz = max((len(x) - 1) / duration_s, 1e-8)
    names = trace["names"]
    span = x.max(axis=0) - x.min(axis=0)
    max_disp = np.linalg.norm(x - x[:1], axis=2).max(axis=0)
    active = max_disp > 0.04
    active_names = {name for name, flag in zip(names, active) if flag}
    path = np.linalg.norm(np.diff(x, axis=0), axis=2).sum(axis=0)
    vel = np.diff(x, axis=0) * sample_hz
    speed = np.linalg.norm(vel, axis=2)
    unit = vel / np.maximum(speed[..., None], 1e-8)
    dots = np.sum(unit[1:] * unit[:-1], axis=2)
    turns = np.degrees(np.arccos(np.clip(dots, -1, 1)))
    turn_mask = (speed[1:] > 0.04) & (speed[:-1] > 0.04)
    jerk = np.diff(vel, axis=0) * sample_hz
    signature = np.r_[span.ravel(), (x[-1] - x[0]).ravel(), path]
    metrics = {
        "duration_s": duration_s,
        "active_landmark_share": float(active.mean()),
        "active_axis_share": float((span > 0.04).mean()),
        "mean_path_length_norm": float(path[active].mean()) if active.any() else 0.0,
        "mean_turn_deg": float(turns[turn_mask].mean()) if turn_mask.any() else 0.0,
        "jerk_rms_norm_s3": float(np.sqrt(np.mean(jerk[:, active] ** 2))) if active.any() and len(jerk) else 0.0,
    }
    return metrics, signature, active_names, x


def participation_ratio(vectors: list[np.ndarray]) -> float:
    if len(vectors) < 3:
        return float("nan")
    x = np.asarray(vectors, float)
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-8)
    eig = np.linalg.eigvalsh(np.cov(x, rowvar=False))
    eig = np.clip(eig, 0, None)
    return float(eig.sum() ** 2 / np.maximum(np.square(eig).sum(), 1e-12))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def direction_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[-1] - a[0], b[-1] - b[0]
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = norm > 1e-8
    return float(np.mean(np.sum(a[valid] * b[valid], axis=1) / norm[valid])) if valid.any() else 1.0


def read_pose_dtw() -> dict[tuple[str, int, str], float]:
    values = {}
    for row in csv.DictReader(DTW.open()):
        if row["b"] == "gt" and row["a"] in GENERATED:
            values[(row["robot"], int(row["idx"]), row["a"])] = float(row["pose_dtw"])
    return values


def plot(summary: dict, pose: dict) -> None:
    methods = GENERATED
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4))
    diversity = ("effective_dim", "active_landmark_share", "mean_path_length_norm")
    titles = ("Effective motion dimension", "Active landmark share", "Mean path length (normalized)")
    for ax, metric, title in zip(axes, diversity, titles):
        values = [summary[m][metric] for m in methods]
        bars = ax.bar(range(4), values, color=[COLORS[m] for m in methods],
                      edgecolor="#202020", hatch=[HATCH[m] for m in methods], linewidth=1.2)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_title(title, fontweight="normal")
        ax.set_xticks(range(4), [LABEL[m] for m in methods], rotation=25, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "diversity_by_method.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, platform, title in zip(axes, ("humanoid", "tiago"), ("Humanoid", "Tiago")):
        values = [pose.get((platform, m), np.nan) for m in methods]
        bars = ax.bar(range(4), values, color=[COLORS[m] for m in methods],
                      edgecolor="#202020", hatch=[HATCH[m] for m in methods], linewidth=1.2)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_title(f"{title} · Pose-DTW to GT", fontweight="normal")
        ax.set_xticks(range(4), [LABEL[m] for m in methods], rotation=25, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Distance (lower is closer)")
    fig.tight_layout()
    fig.savefig(OUT / "pose_accuracy_humanoid_tiago.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pose_rows = read_pose_dtw()
    records, signatures, traces = [], defaultdict(list), {}
    for platform in PLATFORMS:
        for idx in range(1, 31):
            for method in METHODS:
                trace = load_trace(platform, idx, method)
                if trace is None:
                    continue
                metrics, signature, active, normalized = feature(trace, platform)
                records.append({"platform": platform, "idx": idx, "method": method, **metrics})
                signatures[(platform, method)].append(signature)
                traces[(platform, idx, method)] = (trace, active, normalized)

    overlap_rows = []
    for (platform, idx, method), (trace, active, norm) in traces.items():
        if method == "gt" or (platform, idx, "gt") not in traces:
            continue
        gt_trace, gt_active, gt_norm = traces[(platform, idx, "gt")]
        final_error = float(np.linalg.norm(norm[-1] - gt_norm[-1], axis=1).mean())
        ratio = (len(trace["xyz"]) / trace["hz"]) / (len(gt_trace["xyz"]) / gt_trace["hz"])
        overlap_rows.append({
            "platform": platform, "idx": idx, "method": method,
            "active_channel_jaccard": jaccard(active, gt_active),
            "dominant_direction_cosine": direction_cosine(norm, gt_norm),
            "terminal_pose_error_norm": final_error,
            "duration_ratio_to_gt": ratio,
            "pose_dtw_to_gt": pose_rows.get((platform, idx, method), np.nan),
        })

    summary = {}
    pose = {}
    for method in GENERATED:
        method_records = [r for r in records if r["method"] == method]
        summary[method] = {metric: float(np.mean([r[metric] for r in method_records]))
                           for metric in ("duration_s", "active_landmark_share", "active_axis_share",
                                          "mean_path_length_norm", "mean_turn_deg", "jerk_rms_norm_s3")}
        dims = [participation_ratio(signatures[(platform, method)]) for platform in PLATFORMS
                if signatures[(platform, method)]]
        summary[method]["effective_dim"] = float(np.nanmean(dims))
        matched = [r for r in overlap_rows if r["method"] == method]
        for metric in ("active_channel_jaccard", "dominant_direction_cosine", "terminal_pose_error_norm",
                       "duration_ratio_to_gt", "pose_dtw_to_gt"):
            summary[method][metric] = float(np.nanmean([r[metric] for r in matched]))
        for platform in ("humanoid", "tiago"):
            vals = [r["pose_dtw_to_gt"] for r in matched if r["platform"] == platform]
            pose[(platform, method)] = float(np.nanmean(vals))

    with (OUT / "per_instance_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    with (OUT / "gt_overlap_per_instance.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(overlap_rows[0]))
        writer.writeheader(); writer.writerows(overlap_rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(summary, pose)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
