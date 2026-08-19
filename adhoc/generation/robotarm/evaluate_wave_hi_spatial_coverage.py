#!/usr/bin/env python3
r"""
Evaluate **Spatial variability** and **Coverage** for the six ``wave_hi`` scenarios in
``wave_hi_grounded_figure._WAVE_SCENARIOS`` via simulator EE trajectories
(``testset_utils.get_sim_bundle``).

**Spatial variability:** Mean / max pairwise mean per-frame trajectory RMSE (length-aligned).

**Coverage:** Pools all EE positions from motion-gate--ok runs and computes their **global**
axis-aligned bounding box (AABB): per-axis ranges \(R_x,R_y,R_z\), hypervolume
\(R_x R_y R_z\) (m\(^3\)), and diagonal span \(\|R\|_2\) (m).

::

    python adhoc/generation/robotarm/evaluate_wave_hi_spatial_coverage.py --robot IIWA
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
_ARM = Path(__file__).resolve().parent
for p in (_REPO, _ARM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import wave_hi_grounded_figure as wh  # noqa: E402
from adhoc.utils.repo_paths import dev_robosuite_root  # noqa: E402
from adhoc.vlm_test import testset_utils  # noqa: E402


def _traj_xyz(trajectory: list[dict[str, Any]]) -> np.ndarray:
    rows = []
    for t in trajectory:
        pos = np.asarray(t.get("pos"), dtype=np.float64).ravel()
        if pos.size < 3:
            continue
        rows.append(pos[:3])
    if not rows:
        raise ValueError("empty trajectory positions")
    return np.stack(rows, axis=0)


def _mean_pair_rmse(pa: np.ndarray, pb: np.ndarray) -> float:
    """Mean per-frame Euclidean distance (meters) after length alignment."""
    L = min(len(pa), len(pb))
    if L < 1:
        raise ValueError("trajectory too short")
    return float(np.linalg.norm(pa[:L] - pb[:L], axis=1).mean())


def _ee_dataset_aabb_coverage(trajs_xyz: list[np.ndarray]) -> dict[str, Any]:
    """
    Pool EE points from all trajectories (each row xyz), take global min/max per axis.
    Coverage proxies: axis ranges, box hypervolume (m^3), diagonal span (m).
    """
    parts = [x for x in trajs_xyz if x is not None and len(x) >= 1]
    if not parts:
        return {
            "n_points_pooled": 0,
            "min_xyz_m": None,
            "max_xyz_m": None,
            "axis_range_m": None,
            "aabb_volume_m3": float("nan"),
            "aabb_diag_span_m": float("nan"),
        }
    stacked = np.vstack(parts)
    amin = stacked.min(axis=0)
    amax = stacked.max(axis=0)
    span = amax - amin
    vol = float(np.prod(span))
    diag = float(np.linalg.norm(span))
    return {
        "n_points_pooled": int(stacked.shape[0]),
        "min_xyz_m": [float(v) for v in amin],
        "max_xyz_m": [float(v) for v in amax],
        "axis_range_m": [float(v) for v in span],
        "aabb_volume_m3": vol,
        "aabb_diag_span_m": diag,
    }


def _pairwise_rmse_stats(trajs: list[np.ndarray]) -> tuple[list[float], float, float]:
    ds: list[float] = []
    for i in range(len(trajs)):
        for j in range(i + 1, len(trajs)):
            if len(trajs[i]) < 2 or len(trajs[j]) < 2:
                continue
            ds.append(_mean_pair_rmse(trajs[i], trajs[j]))
    if not ds:
        return [], float("nan"), float("nan")
    arr = np.asarray(ds, dtype=np.float64)
    return ds, float(arr.mean()), float(arr.max())


def main() -> None:
    ap = argparse.ArgumentParser(description="Spatial + coverage metrics on 6 wave_hi sim trajectories.")
    ap.add_argument("--robot", default="IIWA")
    ap.add_argument("--cue", default="wave_hi")
    ap.add_argument("--hz", type=int, default=8)
    ap.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="Single motion-config JSON (else see_html_sources.yml manipulator.motion_config_json merge).",
    )
    ap.add_argument("--sources-yml", type=str, default=None)
    ap.add_argument("--fallback-latest-config", action="store_true")
    ap.add_argument(
        "--pairwise-delta-m",
        type=float,
        default=0.015,
        help="Diagnostic: count context pairs whose mean trajectory RMSE (m) ≥ this threshold.",
    )
    ap.add_argument(
        "--motion-range-min-m",
        type=float,
        default=0.02,
        help="Non-degenerate if axis peak-to-peak EE range exceeds this (meters).",
    )
    ap.add_argument(
        "--min-success-ratio",
        type=float,
        default=1.0,
        help="Interpretability_proxy: successes / n scenarios must ≥ this.",
    )
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument("--no-disk-cache", action="store_true")
    args = ap.parse_args()

    sources_yml = Path(args.sources_yml).resolve() if args.sources_yml else None
    if args.fallback_latest_config:
        cfg_paths = [wh._default_config_newest_glob()]
    elif args.config_json:
        cfg_paths = [Path(args.config_json).resolve()]
    else:
        cfg_paths, _ = wh._resolve_manipulator_configs(sources_yml=sources_yml, config_json=None)

    cue = str(args.cue).strip()
    base_row, _ = wh._pick_base_row(cue, cfg_paths, base_from_shots=False)

    testset_utils.clear_sim_bundle_cache()

    hz = max(1, int(args.hz))
    scenarios = wh._WAVE_SCENARIOS
    n_scene = len(scenarios)

    per_run: list[dict[str, Any]] = []
    trajs_xyz: list[np.ndarray] = []

    for i, spec in enumerate(scenarios):
        slug = str(spec["slug"])
        caption = str(spec.get("caption", slug))
        row = wh._apply_wave_scenario(base_row, spec)
        synth_idx = 9600 + i
        row["idx"] = synth_idx
        row["cue"] = cue

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix=f"eval_wave_{slug}_", delete=False) as tmp:
            json.dump([row], tmp, indent=2, ensure_ascii=False)
            cfg_path = Path(tmp.name)

        try:
            sample_id = testset_utils._safe_name(f"eval_wave_{slug}_{cue}")
            sample = {
                "sample_id": sample_id,
                "testset": "iconic",
                "cue_idx": synth_idx,
                "cue": cue,
                "config_path": str(cfg_path),
                "gif_path": str(cfg_path),
                "meta": {},
            }
            bundle = testset_utils.get_sim_bundle(
                sample,
                str(args.robot),
                hz,
                force=True,
                disk_cache=not bool(args.no_disk_cache),
            )
            traj = bundle["trajectory"]
            xyz = _traj_xyz(traj)
            trajs_xyz.append(xyz)

            rg = np.ptp(xyz, axis=0)
            degenerate = bool(rg.max() < float(args.motion_range_min_m) or len(xyz) < 3)

            per_run.append(
                {
                    "slug": slug,
                    "caption": caption,
                    "n_frames_sim": len(xyz),
                    "ee_xyz_range_m_per_axis": [float(v) for v in rg],
                    "ee_xyz_mean_m": [float(v) for v in xyz.mean(axis=0)],
                    "interpretability_motion_gate": not degenerate,
                }
            )
        except Exception as e:
            per_run.append({"slug": slug, "caption": caption, "error": str(e)[:280]})
            trajs_xyz.append(np.zeros((0, 3)))
        finally:
            cfg_path.unlink(missing_ok=True)

    successes = sum(1 for r in per_run if "error" not in r and r.get("interpretability_motion_gate"))
    gate_ok_ix = [
        i
        for i, r in enumerate(per_run)
        if "error" not in r and r.get("interpretability_motion_gate") and len(trajs_xyz[i]) >= 3
    ]
    traj_ok_subset = [trajs_xyz[k] for k in gate_ok_ix]

    pairwise_ds, spatial_mean_rmse_m, spatial_max_rmse_m = _pairwise_rmse_stats(
        traj_ok_subset if len(traj_ok_subset) >= 2 else []
    )

    thr = float(args.pairwise_delta_m)
    pairwise_above_thr = sum(1 for x in pairwise_ds if x >= thr) if pairwise_ds else 0
    n_pairs = len(pairwise_ds)

    coverage_aabb = _ee_dataset_aabb_coverage(traj_ok_subset)

    success_ratio = successes / float(n_scene)
    interpretability_aggregate = (
        success_ratio >= float(args.min_success_ratio) and len(gate_ok_ix) == n_scene
    )
    i_expr_proxy_mag = 1 if interpretability_aggregate else 0

    report: dict[str, Any] = {
        "robot": str(args.robot),
        "hz": hz,
        "cue": cue,
        "n_scenarios": n_scene,
        "spatial_variability_mean_pairwise_rmse_m": spatial_mean_rmse_m,
        "spatial_variability_max_pairwise_rmse_m": spatial_max_rmse_m,
        "spatial_pairwise_rmses_m_sample": pairwise_ds[:36],  # cap log size
        "coverage_ee_trajectory_aabb": coverage_aabb,
        "diagnostics_pairwise_rmse_above_delta_m": {
            "delta_m": thr,
            "count": pairwise_above_thr,
            "total_pairs_among_motion_gate_ok": n_pairs,
        },
        "coverage_interpretability": {
            "successes_motion_gate_ok": successes,
            "success_ratio": success_ratio,
            "min_success_ratio_requested": float(args.min_success_ratio),
            "motion_range_min_m": float(args.motion_range_min_m),
            "i_expr_cardinality_proxy": i_expr_proxy_mag,
            "note": "Assume single latent intent wave_hi across scenarios; cardinality 1 if all gates OK.",
        },
        "runs": per_run,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    outp = Path(args.out_json).resolve() if args.out_json else None
    if outp is not None:
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f" wrote {outp.relative_to(dev_robosuite_root())}", file=sys.stderr)


if __name__ == "__main__":
    main()
