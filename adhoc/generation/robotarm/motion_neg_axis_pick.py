#!/usr/bin/env python3
"""Pick neg-axis control variants maximally separated from the annotated true axis."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_JSONL = _REPO / "data/seed/_remainder/closest_poses_results.jsonl"
_AXIS_IDX = {"x": 0, "y": 1, "z": 2}
_ALL_AXES = ("x", "y", "z")
_PROBE_DEG = 22.0
_MIN_GAIN_M = 0.04


def primary_axis_from_component(comp: dict[str, Any] | None) -> str | None:
    if not comp:
        return None
    kind = comp.get("kind")
    if kind == "path_line":
        ax = comp.get("axis")
        return str(ax).lower() if ax else None
    if kind == "path_arc":
        return None
    axes = comp.get("axes") or {}
    if len(axes) == 1:
        return next(iter(axes.keys())).lower()
    return None


def _pose_def_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    gfp = cfg.get("gt_fixed_first_pose") or {}
    for st in cfg.get("movements") or []:
        if st.get("type") == "pose":
            pose = (st.get("parameters") or {}).get("pose") or {}
            return {**gfp, **pose}
    return dict(gfp)


@lru_cache(maxsize=1)
def _motion_generator():
    from motion_generation import MotionGenerator

    tmp = _REPO / "adhoc" / "test" / "_neg_axis_pick_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    return MotionGenerator(
        robot_name="IIWA",
        jsonl_path=str(_JSONL),
        output_dir=str(tmp),
        has_offscreen_renderer=True,
        hz=10,
    )


def _resolve_pose_record(cfg: dict[str, Any]) -> dict[str, Any] | None:
    gen = _motion_generator()
    pose_def = _pose_def_from_cfg(cfg)
    if not pose_def:
        return None
    matching = gen._find_matching_poses(pose_def)
    if not matching:
        return None
    pid = pose_def.get("pose_id")
    if pid is not None:
        for p in matching:
            if p.get("pose_id") == pid:
                return p
    return matching[0]


def _position_jacobian_at_cfg(cfg: dict[str, Any]) -> np.ndarray | None:
    """3 x n_active position Jacobian at the config's fixed pose."""
    gen = _motion_generator()
    pose = _resolve_pose_record(cfg)
    if not pose:
        return None
    gen.jacobian_calculator._set_pose_from_data(pose)
    model = gen.env.sim.model._model
    data = gen.env.sim.data._data
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, gen.jacobian_calculator.eef_site_name
    )
    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
    dof_ids = gen.jacobian_calculator.ik_solver.dof_ids
    jac_subset = np.vstack([jac_pos, jac_rot])[:, dof_ids]
    return jac_subset[0:3, :]


def _sorted_joint_scores(pos_jac: np.ndarray, axis: str) -> list[tuple[int, float]]:
    axis_idx = _AXIS_IDX[axis]
    target = pos_jac[axis_idx, :]
    total_sq = np.sum(pos_jac**2, axis=0) + 1e-6
    scores = (target**2) / total_sq
    order = np.argsort(-scores)
    return [(int(i), float(scores[i])) for i in order]


def _select_joint_column(
    pos_jac: np.ndarray,
    axis: str,
    joint_preference: str,
    *,
    score_threshold: float = 0.1,
) -> int | None:
    """Mirror MotionGenerator._select_joint (quiet) → column index in pos_jac."""
    sorted_joints = _sorted_joint_scores(pos_jac, axis)
    if not sorted_joints:
        return None

    filtered = [j for j in sorted_joints if j[1] >= score_threshold]
    if not filtered:
        filtered = sorted_joints[:1]

    n = len(filtered)
    if joint_preference in ("shoulder", "elbow", "wrist"):
        t1 = max(1, n // 3)
        t2 = max(t1 + 1, (2 * n + 2) // 3)
        groups = {
            "shoulder": filtered[:t1],
            "elbow": filtered[t1:t2],
            "wrist": filtered[t2:],
        }
        pool = groups.get(joint_preference) or filtered
    elif joint_preference == "proximal":
        pool = filtered[: max(1, n // 2)]
    elif joint_preference == "distal":
        pool = filtered[max(0, n // 2) :]
    else:
        pool = filtered

    best_col, _ = max(pool, key=lambda t: t[1])
    return best_col


def _probe_render_delta(
    cfg: dict[str, Any],
    command_axis: str,
    joint_preference: str,
    *,
    degrees: float = _PROBE_DEG,
) -> np.ndarray | None:
    """World EE delta using the same joint selection as ``render`` (``_select_joint``)."""
    gen = _motion_generator()
    pose = _resolve_pose_record(cfg)
    if pose is None:
        return None

    gen.jacobian_calculator._set_pose_from_data(pose)
    model = gen.env.sim.model._model
    data = gen.env.sim.data._data
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, gen.jacobian_calculator.eef_site_name
    )

    def eef() -> np.ndarray:
        return np.array(data.site_xpos[site_id], dtype=float).copy()

    try:
        _joint_idx, _joint_name, joint_dof_id, _score, _jac_sign = gen._select_joint(
            axis=command_axis,
            joint_preference=joint_preference,
        )
    except Exception:
        return None

    robot_idx = gen._find_joint_index_in_robot(joint_dof_id)
    if robot_idx is None:
        return None

    start = eef()
    joint_pos = gen._get_joint_positions().copy()
    joint_pos[robot_idx] += np.deg2rad(float(degrees))
    gen._set_joint_positions(joint_pos)
    return eef() - start


def _joint_preferences_to_try(base: str | None) -> tuple[str, ...]:
    base = (base or "shoulder").lower()
    order = ("shoulder", "elbow", "wrist")
    if base in order:
        return (base,) + tuple(j for j in order if j != base)
    return order


def pick_separated_axis_and_joint(
    cfg: dict[str, Any],
    true_axis: str,
    *,
    joint_preference: str | None = None,
) -> tuple[str, str, dict[str, float]]:
    """
    Pick (neg_axis, joint) whose rendered EE motion is most perpendicular to GT motion.

    Probes use ``_select_joint`` (same as GIF render), not Jacobian column alone.
    """
    true_axis = true_axis.lower()
    if joint_preference is None:
        st = None
        for step in cfg.get("movements") or []:
            if step.get("type") == "movement":
                st = step
                break
        joint_preference = str(((st or {}).get("parameters") or {}).get("joint") or "shoulder")

    delta_gt = _probe_render_delta(cfg, true_axis, joint_preference, degrees=_PROBE_DEG)
    gt_mag = float(np.linalg.norm(delta_gt)) if delta_gt is not None else 0.0

    candidates = [a for a in _ALL_AXES if a != true_axis]
    best: tuple[str, str, float, float, float] | None = None  # cand, jp, perp, parallel, mag

    for cand in candidates:
        for jp in _joint_preferences_to_try(joint_preference):
            delta = _probe_render_delta(cfg, cand, jp, degrees=_PROBE_DEG)
            if delta is None:
                continue
            mag = float(np.linalg.norm(delta))
            if mag < _MIN_GAIN_M:
                continue
            if gt_mag > 1e-5:
                parallel = abs(float(np.dot(delta, delta_gt) / gt_mag))
                perp = float(np.sqrt(max(0.0, mag * mag - parallel * parallel)))
            else:
                parallel = 0.0
                perp = mag
            # Prefer motion clearly different from GT (large perpendicular, small parallel).
            score = perp - 0.35 * parallel
            if best is None or score > best[4] - 1e-6:
                best = (cand, jp, perp, parallel, score)

    if best is None:
        fallback_ax = _cyclic_alt(true_axis)
        return fallback_ax, joint_preference, {"leak_true_m": 0.0, "gain_neg_m": 0.0}

    cand, jp, perp, parallel, _score = best
    meta = {
        "leak_true_m": round(parallel, 4),
        "gain_neg_m": round(perp, 4),
        "separation": round(perp / (parallel + 1e-4), 2),
        "gt_probe_mag_m": round(gt_mag, 4),
    }
    return cand, jp, meta


def pick_separated_axis(
    cfg: dict[str, Any],
    true_axis: str,
    *,
    joint_preference: str | None = None,
) -> tuple[str, float]:
    """Return (best_other_axis, separation_score)."""
    cand, _, meta = pick_separated_axis_and_joint(cfg, true_axis, joint_preference=joint_preference)
    return cand, float(meta.get("separation", 0.0))


def pick_neg_arc_plane(plane: str, avoid_axis: str | None) -> str:
    plane = plane.lower()
    if plane == "zx":
        plane = "xz"
    if avoid_axis:
        avoid_axis = avoid_axis.lower()
        options = [p for p in ("xy", "yz", "xz") if avoid_axis not in p]
        if options:
            return options[0]
    return {"xy": "yz", "yz": "xz", "xz": "xy"}.get(plane, "yz")


def _cyclic_alt(ax: str) -> str:
    return {"x": "y", "y": "z", "z": "x"}.get(ax, "y")
