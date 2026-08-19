#!/usr/bin/env python3
"""Core ablation: pose dir / gripper / movement axis / repetition on GT-matching steps."""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_MAG = 22.0

ALL_DIRS = ("front", "back", "left", "right", "up", "down")
ALL_GRIPS = ("horizontal", "vertical")


def _parse_gt_poses(groundtruth: str) -> list[tuple[str, str]]:
    return [(a.strip(), b.strip()) for a, b in re.findall(r"\(([^,]+),\s*([^)]+)\)", groundtruth or "")]


def _first_pose_step(row: dict[str, Any]) -> dict[str, Any] | None:
    for step in row.get("movements") or []:
        if step.get("type") == "pose":
            return step
    return None


def _pose_dict(step: dict[str, Any]) -> dict[str, Any]:
    return (step.get("parameters") or {}).get("pose") or {}


def find_gt_pose_step_indices(row: dict[str, Any], gt_poses: list[tuple[str, str]]) -> list[int]:
    gt_set = set(gt_poses)
    out: list[int] = []
    for i, step in enumerate(row.get("movements") or []):
        if step.get("type") != "pose":
            continue
        p = _pose_dict(step)
        pair = (str(p.get("dir", "")).strip(), str(p.get("gripper_orientation", "")).strip())
        if pair in gt_set:
            out.append(i)
    return out


def _rep_override_still_gt(rep_rule: str | None, override: str) -> bool:
    if rep_rule == "any":
        return True
    return False


def _variant_still_gt_valid(cfg: dict[str, Any], gt_row: dict[str, Any]) -> bool:
    ab = cfg.get("ablation") or {}
    kind = ab.get("kind")
    if kind in ("pose_dir", "pose_gripper"):
        pose_gt = str(gt_row.get("pose_gt") or gt_row.get("groundtruth") or "")
        gt_set = set(_parse_gt_poses(pose_gt))
        i = ab.get("step_idx")
        if not isinstance(i, int):
            return False
        movements = cfg.get("movements") or []
        if i < 0 or i >= len(movements):
            return False
        p = _pose_dict(movements[i])
        pair = (str(p.get("dir", "")).strip(), str(p.get("gripper_orientation", "")).strip())
        return pair in gt_set
    if kind == "movement_rep":
        comp = (gt_row.get("movement_gt") or {}).get("component") or {}
        override = str(ab.get("rep_override") or "")
        return _rep_override_still_gt(comp.get("repetition"), override)
    if kind == "movement_axis":
        comp = (gt_row.get("movement_gt") or {}).get("component") or {}
        from score_pilot40_motion_gt_components import _tail_matches_component, _tail_steps  # noqa: WPS433

        tail = _tail_steps(cfg.get("movements") or [])
        ok, _ = _tail_matches_component(tail, comp)
        return bool(ok)
    return False


def _alt_dir(cur: str, grip: str, gt_set: set[tuple[str, str]]) -> str | None:
    for d in ALL_DIRS:
        if d != cur and (d, grip) not in gt_set:
            return d
    return None


def ablate_pose_dir(row: dict[str, Any], gt_poses: list[tuple[str, str]]) -> dict[str, Any] | None:
    idxs = find_gt_pose_step_indices(row, gt_poses)
    if not idxs:
        return None
    v = copy.deepcopy(row)
    gt_set = set(gt_poses)
    i = idxs[0]
    p = _pose_dict(v["movements"][i])
    d, g = str(p.get("dir", "")), str(p.get("gripper_orientation", ""))
    nd = _alt_dir(d, g, gt_set)
    if not nd:
        return None
    p["dir"] = nd
    v["movements"][i]["parameters"]["pose"] = p
    in_gt = (nd, g) in gt_set
    v["ablation"] = {"kind": "pose_dir", "step_idx": i, "from": d, "to": nd, "still_gt_valid": in_gt}
    return v


def ablate_pose_gripper(row: dict[str, Any], gt_poses: list[tuple[str, str]]) -> dict[str, Any] | None:
    idxs = find_gt_pose_step_indices(row, gt_poses)
    if not idxs:
        return None
    v = copy.deepcopy(row)
    i = idxs[0]
    p = _pose_dict(v["movements"][i])
    d, g = str(p.get("dir", "")), str(p.get("gripper_orientation", ""))
    if g not in ALL_GRIPS:
        return None
    ng = "vertical" if g == "horizontal" else "horizontal"
    p["gripper_orientation"] = ng
    v["movements"][i]["parameters"]["pose"] = p
    in_gt = (d, ng) in set(gt_poses)
    v["ablation"] = {
        "kind": "pose_gripper",
        "step_idx": i,
        "from": g,
        "to": ng,
        "still_gt_valid": in_gt,
    }
    return v


def _import_scoring():
    _here = Path(__file__).resolve().parent
    _repo = _here.parents[2]
    for p in (_repo, _here):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from score_pilot40_motion_gt_components import (  # noqa: WPS433
        _match_movement_component,
        _match_path_component,
        _tail_steps,
    )

    return _match_movement_component, _match_path_component, _tail_steps


def find_gt_tail_step_index(row: dict[str, Any], comp: dict[str, Any]) -> int | None:
    _match_movement_component, _match_path_component, _tail_steps = _import_scoring()
    movements = row.get("movements") or []
    seen_first_pose = False
    for i, step in enumerate(movements):
        if step.get("type") == "pose":
            if not seen_first_pose:
                seen_first_pose = True
                continue
        if step.get("type") == "movement" and _match_movement_component(step, comp):
            return i
        if step.get("type") == "path" and _match_path_component(step, comp):
            return i
    # fallback: first movement/path after leading pose
    seen_first_pose = False
    for i, step in enumerate(movements):
        if step.get("type") == "pose":
            if not seen_first_pose:
                seen_first_pose = True
                continue
        if step.get("type") in ("movement", "path"):
            return i
    return None


def _apply_axis_on_step(cfg: dict[str, Any], step_idx: int, comp: dict[str, Any]) -> dict[str, Any] | None:
    from motion_gt_tail_builder import apply_single_element_variant  # noqa: WPS433

    stub = copy.deepcopy(cfg)
    stub["movements"] = [m for j, m in enumerate(stub.get("movements") or []) if j <= step_idx]
    # ensure stub tail step is last
    if not stub["movements"]:
        return None
    primary_ax = next(iter((comp.get("axes") or {}).keys()), None)
    variant = apply_single_element_variant(stub, "axis", primary_axis=primary_ax)
    if not variant:
        return None
    v = copy.deepcopy(cfg)
    v["movements"][step_idx] = variant["movements"][-1]
    meta = variant.get("neg_axis_meta") or {}
    v["ablation"] = {"kind": "movement_axis", "step_idx": step_idx, **meta}
    return v


def ablate_movement_axis(row: dict[str, Any], comp: dict[str, Any]) -> dict[str, Any] | None:
    idx = find_gt_tail_step_index(row, comp)
    if idx is None:
        return None
    return _apply_axis_on_step(row, idx, comp)


def ablate_movement_rep(row: dict[str, Any], comp: dict[str, Any]) -> dict[str, Any] | None:
    from motion_gt_tail_builder import build_tail_from_component_with_rep_override  # noqa: WPS433

    idx = find_gt_tail_step_index(row, comp)
    if idx is None:
        return None
    rep_rule = comp.get("repetition")
    if rep_rule == "non":
        override = "rep"
    elif rep_rule in ("rep", "any"):
        override = "non"
    else:
        override = "rep"
    new_tail = build_tail_from_component_with_rep_override(comp, rep_override=override)
    if not new_tail:
        return None
    v = copy.deepcopy(row)
    # replace tail from first GT tail step onward
    v["movements"] = (v.get("movements") or [])[:idx] + copy.deepcopy(new_tail)
    v["ablation"] = {"kind": "movement_rep", "step_idx": idx, "rep_override": override}
    return v


def both_correct_rows(
    exp1_path: Path,
    gt_by_cue: dict[str, dict[str, Any]],
    *,
    pose_ok_fn,
) -> list[dict[str, Any]]:
    from score_pilot40_motion_gt_components import _tail_matches_component, _tail_steps  # noqa: WPS433

    rows = json.loads(exp1_path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for r in rows:
        cue = str(r.get("cue", ""))
        ev = gt_by_cue.get(cue) or {}
        pose_gt = str(ev.get("pose_gt") or ev.get("groundtruth") or "")
        if not pose_ok_fn(r, pose_gt):
            continue
        mg = ev.get("movement_gt") or {}
        if mg.get("always_correct") or mg.get("annotation_raw") == "none":
            continue
        comp = mg.get("component")
        if not comp:
            continue
        tail = _tail_steps(r.get("movements") or [])
        ok, _ = _tail_matches_component(tail, comp)
        if ok:
            out.append(r)
    return out


def build_variants(row: dict[str, Any], gt_row: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    pose_gt = str(gt_row.get("pose_gt") or gt_row.get("groundtruth") or "")
    gt_poses = _parse_gt_poses(pose_gt)
    comp = (gt_row.get("movement_gt") or {}).get("component") or {}
    out = {
        "original": copy.deepcopy(row),
        "pose_dir": ablate_pose_dir(row, gt_poses),
        "pose_gripper": ablate_pose_gripper(row, gt_poses),
        "movement_axis": ablate_movement_axis(row, comp),
        "movement_rep": ablate_movement_rep(row, comp),
    }
    for key, cfg in out.items():
        if key == "original" or not isinstance(cfg, dict):
            continue
        ab = cfg.setdefault("ablation", {})
        ab["still_gt_valid"] = _variant_still_gt_valid(cfg, gt_row)
    return out
