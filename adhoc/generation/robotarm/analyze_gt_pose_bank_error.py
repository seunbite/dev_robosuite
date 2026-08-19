#!/usr/bin/env python3
"""Measure initial-pose bank error for Humanoid and Tiago GT."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from embodiment_renderers import _g1_select_feasible_pose  # noqa: E402
from adhoc.generation.tiago.legacy.render_mobile_config import (  # noqa: E402
    list_pose_bank_candidates,
)
from adhoc.generation.tiago.legacy.render_mobile_manip_schema import (  # noqa: E402
    _make_env,
)

GT = REPO / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
OUT_JSON = REPO / "data/results/evaluation/gt_initial_pose_bank_xyz_error.json"
OUT_MD = REPO / "data/results/evaluation/gt_initial_pose_bank_xyz_error.md"


def _pose(row: dict[str, Any]) -> dict[str, Any]:
    for step in row.get("movements") or []:
        if step.get("type") == "pose":
            return (step.get("parameters") or {}).get("pose") or {}
    return {}


def _targets(pose: dict[str, Any]) -> dict[str, float]:
    return {
        axis: float(pose[axis])
        for axis in ("x", "y", "z")
        if isinstance(pose.get(axis), (int, float))
    }


def _record(
    platform: str,
    entry: dict[str, Any],
    pose: dict[str, Any],
    selected: dict[str, Any],
    pool_count: int,
    source: str,
) -> dict[str, Any]:
    targets = _targets(pose)
    errors = {axis: abs(float(selected[axis]) - target) for axis, target in targets.items()}
    return {
        "platform": platform,
        "idx": int(entry["idx"]),
        "cue": str(entry["cue"]),
        "direction": pose.get("arm_position"),
        "gripper_orientation": pose.get("gripper_orientation"),
        "target": targets,
        "selected_xyz": {axis: float(selected[axis]) for axis in ("x", "y", "z")},
        "absolute_error": errors,
        "l1_error": sum(errors.values()),
        "euclidean_error": math.sqrt(sum(value * value for value in errors.values())),
        "max_axis_error": max(errors.values()),
        "specified_axis_count": len(errors),
        "candidate_pool_count": pool_count,
        "source": source,
        "pose_id": selected.get("pose_id"),
        "tile_idx": selected.get("tile_idx"),
    }


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_errors = [float(row["max_axis_error"]) for row in rows]
    l1_errors = [float(row["l1_error"]) for row in rows]
    bins = Counter()
    for value in max_errors:
        if value <= 5:
            bins["0–5"] += 1
        elif value <= 10:
            bins[">5–10"] += 1
        elif value <= 20:
            bins[">10–20"] += 1
        elif value <= 30:
            bins[">20–30"] += 1
        else:
            bins[">30"] += 1
    axes = {}
    for axis in ("x", "y", "z"):
        values = [
            float(row["absolute_error"][axis])
            for row in rows
            if axis in row["absolute_error"]
        ]
        axes[axis] = {
            "n": len(values),
            "mean_abs_error": mean(values),
            "median_abs_error": median(values),
            "p90_abs_error": _percentile(values, 90),
            "max_abs_error": max(values),
        }
    return {
        "n": len(rows),
        "mean_l1_error": mean(l1_errors),
        "median_l1_error": median(l1_errors),
        "p90_l1_error": _percentile(l1_errors, 90),
        "mean_max_axis_error": mean(max_errors),
        "median_max_axis_error": median(max_errors),
        "p90_max_axis_error": _percentile(max_errors, 90),
        "max_axis_error": max(max_errors),
        "within_5_pct": sum(value <= 5 for value in max_errors) / len(rows),
        "within_10_pct": sum(value <= 10 for value in max_errors) / len(rows),
        "within_20_pct": sum(value <= 20 for value in max_errors) / len(rows),
        "max_axis_error_bins": dict(bins),
        "axes": axes,
    }


def _fmt(value: float) -> str:
    return f"{value:.1f}"


def _write_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# GT initial-pose bank XYZ error",
        "",
        "Only explicitly specified axes are included. Selection filters semantic pose first, then",
        "uses minimum L1 distance in body-centered 0–100 percentile space.",
        "",
        "| Platform | N | Mean L1 | Median L1 | Mean max-axis | P90 max-axis | Max | ≤5 | ≤10 | ≤20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for platform in ("humanoid", "tiago"):
        s = payload["summary"][platform]
        lines.append(
            f"| {platform} | {s['n']} | {_fmt(s['mean_l1_error'])} | "
            f"{_fmt(s['median_l1_error'])} | {_fmt(s['mean_max_axis_error'])} | "
            f"{_fmt(s['p90_max_axis_error'])} | {_fmt(s['max_axis_error'])} | "
            f"{100*s['within_5_pct']:.1f}% | {100*s['within_10_pct']:.1f}% | "
            f"{100*s['within_20_pct']:.1f}% |"
        )
    lines.extend([
        "",
        "## Axis-wise absolute error",
        "",
        "| Platform | Axis | N | MAE | Median | P90 | Max |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for platform in ("humanoid", "tiago"):
        for axis in ("x", "y", "z"):
            a = payload["summary"][platform]["axes"][axis]
            lines.append(
                f"| {platform} | {axis} | {a['n']} | {_fmt(a['mean_abs_error'])} | "
                f"{_fmt(a['median_abs_error'])} | {_fmt(a['p90_abs_error'])} | "
                f"{_fmt(a['max_abs_error'])} |"
            )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    master = json.loads(GT.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for entry in master:
        pose = _pose(entry["platforms"]["humanoid"])
        targets = _targets(pose)
        if targets:
            selected = _g1_select_feasible_pose(pose)
            if selected is None:
                raise RuntimeError(f"No G1 candidate for {entry['cue']}: {pose}")
            semantic_pool = [
                row for row in json.loads(
                    (REPO / "data/seed/unitree_g1/g1_feasible_poses_metadata.json").read_text()
                )["poses"]
                if row["direction"] == pose["arm_position"]
            ]
            grip = str(pose.get("gripper_orientation", "vertical"))
            if grip in ("horizontal", "vertical"):
                semantic_pool = [
                    row for row in semantic_pool if row["gripper_orientation"] == grip
                ]
            rows.append(
                _record(
                    "humanoid", entry, pose, selected, len(semantic_pool), "approved_g1_47"
                )
            )

    env = _make_env()
    try:
        for entry in master:
            pose = _pose(entry["platforms"]["tiago"])
            if not _targets(pose):
                continue
            result = list_pose_bank_candidates(env, pose, top_k=1, ignore_xyz=False)
            if not result.get("entries"):
                raise RuntimeError(f"No Google candidate for {entry['cue']}: {pose}")
            selected = result["entries"][0]
            rows.append(
                _record(
                    "tiago",
                    entry,
                    pose,
                    selected,
                    int(result.get("n_pool", 0)),
                    str(result.get("source", selected.get("source", "unknown"))),
                )
            )
    finally:
        env.close()

    payload = {
        "definition": {
            "coordinate": "body-centered xyz percentile, 0–100",
            "distance_for_selection": "L1 on explicitly specified axes only",
            "humanoid_bank": "approved 47-pose Unitree G1 feasible set",
            "google_bank": "runtime plausible set first, full bank fallback",
        },
        "summary": {
            platform: _summary([row for row in rows if row["platform"] == platform])
            for platform in ("humanoid", "tiago")
        },
        "rows": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(payload)
    print(json.dumps(payload["summary"], indent=2))
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
