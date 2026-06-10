#!/usr/bin/env python3
"""Build motion-config tail steps from pilot40 motion-component GT annotations."""
from __future__ import annotations

import copy
import re
from typing import Any

_MAG = 22.0
_OSC_MAG = 15.0  # back-and-forth rep cycles (e.g. beckon: -15°, +15° × 2)
_LINE_DIST = 0.15


def _deg_for_rule(rule: str, *, magnitude: float | None = None) -> float:
    mag = _MAG if magnitude is None else magnitude
    if rule == "-":
        return -mag
    return mag


def build_tail_from_component(comp: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not comp:
        return []

    kind = comp.get("kind")
    if kind == "path_arc":
        plane = (comp.get("plane") or "xz").lower()
        if plane == "zx":
            plane = "xz"
        sweep = comp.get("sweep", 360)
        direction = str(comp.get("direction", "ccw")).lower()
        if direction in ("counterclockwise", "counterclock"):
            direction = "ccw"
        elif direction == "clockwise":
            direction = "cw"
        if direction not in ("cw", "ccw"):
            direction = "ccw"
        return [
            {
                "type": "path",
                "parameters": {
                    "shape": "arc",
                    "plane": plane,
                    "radius": 0.04,
                    "sweep": float(sweep),
                    "direction": direction,
                    "speed": 2.0,
                    "hold_time": 0.2,
                },
            }
        ]

    if kind == "path_line":
        axis = (comp.get("axis") or "y").lower()
        return [
            {
                "type": "path",
                "parameters": {
                    "shape": "line",
                    "axis": axis,
                    "distance": _LINE_DIST,
                    "speed": 2.0,
                    "hold_time": 0.3,
                },
            }
        ]

    axes = comp.get("axes") or {}
    joint = comp.get("joint")
    rep_rule = comp.get("repetition")
    hold_rule = comp.get("hold")

    joint = joint or "shoulder"
    directions: list[dict[str, Any]] = []
    for ax, rule in axes.items():
        if rule == "+-":
            mag = _OSC_MAG if rep_rule == "rep" else _MAG
            directions.append(
                {
                    "degrees": {ax: mag},
                    "speed": 2.2,
                    "hold_time": 0.08,
                }
            )
            directions.append(
                {
                    "degrees": {ax: -mag},
                    "speed": 2.2,
                    "hold_time": 0.08,
                }
            )
        else:
            hold_t = 0.45 if hold_rule else 0.15
            osc = rep_rule == "rep"
            mag = _OSC_MAG if osc else None
            directions.append(
                {
                    "degrees": {ax: _deg_for_rule(rule, magnitude=mag)},
                    "speed": 2.0,
                    "hold_time": 0.08 if osc else hold_t,
                }
            )
            if osc:
                opposite = "+" if rule == "-" else "-"
                directions.append(
                    {
                        "degrees": {ax: _deg_for_rule(opposite, magnitude=_OSC_MAG)},
                        "speed": 2.0,
                        "hold_time": 0.08,
                    }
                )

    if not directions:
        directions.append({"degrees": {"z": _MAG}, "speed": 2.0, "hold_time": 0.2})

    repetition = 1
    if rep_rule == "rep":
        # pose → (+) → (−) → (+) → (−) as one flat sequence (no per-cycle anchor reset)
        directions = directions + directions
    elif rep_rule == "any":
        repetition = 1

    return [
        {
            "type": "movement",
            "parameters": {
                "joint": joint,
                "repetition": repetition,
                "directions": directions,
            },
        }
    ]


def _parse_gt_poses(groundtruth: str) -> list[tuple[str, str]]:
    return [(a.strip(), b.strip()) for a, b in re.findall(r"\(([^,]+),\s*([^)]+)\)", groundtruth)]


def _generation_pose_steps(row: dict[str, Any]) -> list[dict[str, Any]]:
    poses: list[dict[str, Any]] = []
    for step in row.get("movements") or []:
        if step.get("type") != "pose":
            continue
        pose = (step.get("parameters") or {}).get("pose") or {}
        if pose.get("dir") and pose.get("gripper_orientation"):
            poses.append(copy.deepcopy(step))
    return poses


def pose_generation_matches_human_gt(row: dict[str, Any], groundtruth: str) -> bool | None:
    """True when any generated pose step matches human GT (o=primary, x=any listed)."""
    gt = str(groundtruth or "").strip()
    if not gt:
        return None
    targets = _parse_gt_poses(gt)
    if not targets:
        return None
    gen_set = {
        (
            str((step.get("parameters") or {}).get("pose", {}).get("dir", "")).strip(),
            str((step.get("parameters") or {}).get("pose", {}).get("gripper_orientation", "")).strip(),
        )
        for step in _generation_pose_steps(row)
    }
    if gt.lower().startswith("o"):
        return targets[0] in gen_set
    if gt.lower().startswith("x"):
        return any(t in gen_set for t in targets)
    return None


def human_gt_fixed_pose_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Human tile-pick pose for the primary GT (dir, gripper_orientation) group."""
    gt = str(row.get("groundtruth") or "").strip()
    poses = _parse_gt_poses(gt)
    if not poses:
        return None
    from generate_motion_from_gt_pose import (  # noqa: WPS433 — shared tile-pick helper
        _build_fixed_pose,
        _first_pose_from_cfg,
        _load_tile_pick,
    )

    d, g = poses[0]
    return _build_fixed_pose(d, g, _first_pose_from_cfg(row), _load_tile_pick())


def resolve_first_pose_step(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Pick start pose: generation if human-GT-correct, else human GT tile-pick."""
    gen_steps = _generation_pose_steps(row)
    gen_step = gen_steps[0] if gen_steps else None
    groundtruth = str(row.get("groundtruth") or "").strip()
    correct = pose_generation_matches_human_gt(row, groundtruth) if groundtruth else None

    if correct is True and gen_step is not None:
        return gen_step, "generation"

    human = human_gt_fixed_pose_from_row(row)
    if human:
        return (
            {
                "type": "pose",
                "parameters": {"pose": human, "speed": 1.0, "hold_time": 0.0},
            },
            "human_gt_tile_pick",
        )

    if gen_step is not None:
        return gen_step, "fallback_generation"

    fixed = row.get("gt_fixed_first_pose") or {}
    if not fixed:
        return None, "missing"
    return (
        {
            "type": "pose",
            "parameters": {"pose": dict(fixed), "speed": 1.0, "hold_time": 0.0},
        },
        "gt_fixed_field",
    )


def first_pose_step(row: dict[str, Any]) -> dict[str, Any] | None:
    """First pose step from the generated motion config (legacy pairwise start)."""
    for st in row.get("movements") or []:
        if st.get("type") == "pose":
            return copy.deepcopy(st)
    fixed = row.get("gt_fixed_first_pose") or {}
    if not fixed:
        return None
    return {
        "type": "pose",
        "parameters": {
            "pose": dict(fixed),
            "speed": 1.0,
            "hold_time": 0.0,
        },
    }


_AXIS_CYCLIC_FWD = {"x": "y", "y": "z", "z": "x"}
_AXIS_CYCLIC_REV = {"x": "z", "z": "y", "y": "x"}


def extract_generation_choreography_tail(base_row: dict[str, Any]) -> list[dict[str, Any]]:
    """All choreography after the fixed start pose: intermediate poses + movements + paths."""
    tail: list[dict[str, Any]] = []
    saw_first_pose = False
    for st in base_row.get("movements") or []:
        if st.get("type") == "pose":
            if not saw_first_pose:
                saw_first_pose = True
                continue
        elif not saw_first_pose:
            continue
        tail.append(copy.deepcopy(st))
    return tail


def extract_generation_movement_tail(base_row: dict[str, Any]) -> list[dict[str, Any]]:
    """Movement-only tail (legacy); prefer extract_generation_choreography_tail."""
    return [st for st in extract_generation_choreography_tail(base_row) if st.get("type") == "movement"]


def tail_has_intermediate_poses(tail: list[dict[str, Any]]) -> bool:
    return any(st.get("type") == "pose" for st in tail)


def tail_has_multi_axis_degrees(tail: list[dict[str, Any]]) -> bool:
    for st in tail:
        if st.get("type") != "movement":
            continue
        for d in (st.get("parameters") or {}).get("directions") or []:
            if len((d.get("degrees") or {})) >= 2:
                return True
    return False


def _permute_degree_axes(deg: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ax, val in deg.items():
        key = str(ax).lower()
        out[mapping.get(key, key)] = val
    return out


def _degree_vec(deg: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(deg.get("x", 0.0) or 0.0),
        float(deg.get("y", 0.0) or 0.0),
        float(deg.get("z", 0.0) or 0.0),
    )


def pick_axis_permutation_mapping(deg: dict[str, Any]) -> dict[str, str]:
    """Pick cyclic permutation most orthogonal to GT degrees (x,y,z all present or not)."""
    import math

    gx, gy, gz = _degree_vec(deg)
    gmag = math.sqrt(gx * gx + gy * gy + gz * gz)
    if gmag < 1e-6:
        return dict(_AXIS_CYCLIC_FWD)

    best_map = _AXIS_CYCLIC_FWD
    best_score = -1.0
    for mapping in (_AXIS_CYCLIC_FWD, _AXIS_CYCLIC_REV):
        perm = _permute_degree_axes(deg, mapping)
        px, py, pz = _degree_vec(perm)
        pmag = math.sqrt(px * px + py * py + pz * pz)
        if pmag < 1e-6:
            continue
        cos = (gx * px + gy * py + gz * pz) / (gmag * pmag)
        score = 1.0 - abs(cos)
        if score > best_score:
            best_score = score
            best_map = mapping
    return dict(best_map)


def apply_multi_axis_permutation(
    cfg: dict[str, Any],
    *,
    mapping: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Neg control: relabel x/y/z on every direction entry (keeps joint + magnitudes)."""
    v = copy.deepcopy(cfg)
    ref_deg: dict[str, Any] | None = None
    for st in v.get("movements") or []:
        if st.get("type") != "movement":
            continue
        for d in (st.get("parameters") or {}).get("directions") or []:
            deg = d.get("degrees") or {}
            if len(deg) >= 2 and ref_deg is None:
                ref_deg = deg
    if ref_deg is None:
        return None

    perm = mapping or pick_axis_permutation_mapping(ref_deg)
    true_axes = sorted({str(k).lower() for st in v.get("movements") or [] if st.get("type") == "movement" for d in (st.get("parameters") or {}).get("directions") or [] for k in (d.get("degrees") or {})})
    neg_axes = sorted({perm.get(ax, ax) for ax in true_axes})

    for st in v.get("movements") or []:
        if st.get("type") != "movement":
            continue
        p = st.get("parameters") or {}
        new_dirs: list[dict[str, Any]] = []
        for d in p.get("directions") or []:
            nd = dict(d)
            nd["degrees"] = _permute_degree_axes(d.get("degrees") or {}, perm)
            new_dirs.append(nd)
        p["directions"] = new_dirs

    v["neg_axis_meta"] = {
        "variant": "multi_axis_permute",
        "true_axis": "+".join(true_axes),
        "neg_axis": "+".join(neg_axes),
        "axis_mapping": perm,
        "same_joint": True,
    }
    return v


def build_config_from_resolved_pose_and_tail(
    base_row: dict[str, Any],
    tail: list[dict[str, Any]],
    *,
    state_tag: str = "motion_component_gt",
) -> dict[str, Any] | None:
    pose, start_pose_source = resolve_first_pose_step(base_row)
    if pose is None or not tail:
        return None
    fixed_pose = copy.deepcopy((pose.get("parameters") or {}).get("pose") or {})
    return {
        "idx": int(base_row["idx"]),
        "cue": base_row["cue"],
        "description": base_row.get("description", ""),
        "groundtruth": base_row.get("groundtruth", ""),
        "gt_fixed_first_pose": fixed_pose,
        "start_pose_source": start_pose_source,
        "movements": [pose, *copy.deepcopy(tail)],
        "state": state_tag,
        "generation_mode": state_tag,
    }


def build_config_from_gt_pose_and_component(
    base_row: dict[str, Any],
    comp: dict[str, Any] | None,
    *,
    state_tag: str = "motion_component_gt",
) -> dict[str, Any] | None:
    pose = first_pose_step(base_row)
    if pose is None:
        return None
    tail = build_tail_from_component(comp)
    if comp and not tail:
        return None
    fixed_pose = copy.deepcopy((pose.get("parameters") or {}).get("pose") or {})
    out = {
        "idx": int(base_row["idx"]),
        "cue": base_row["cue"],
        "description": base_row.get("description", ""),
        "groundtruth": base_row.get("groundtruth", ""),
        "gt_fixed_first_pose": fixed_pose,
        "start_pose_source": "generation",
        "movements": [pose, *tail],
        "state": state_tag,
        "generation_mode": state_tag,
    }
    return out


def _first_tail_step(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """First non-start step after the leading pose (pose, movement, or path)."""
    seen_pose = False
    for st in cfg.get("movements") or []:
        if st.get("type") == "pose":
            seen_pose = True
            continue
        if seen_pose:
            return st
    return None


def _first_tail_movement_step(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """First movement step after the leading pose (skips intermediate poses)."""
    seen_pose = False
    for st in cfg.get("movements") or []:
        if st.get("type") == "pose":
            seen_pose = True
            continue
        if seen_pose and st.get("type") == "movement":
            return st
    return None


def _alt_joint(j: str) -> str:
    for cand in ("shoulder", "elbow", "wrist"):
        if cand != j:
            return cand
    return j


def apply_single_element_variant(
    cfg: dict[str, Any],
    kind: str,
    *,
    primary_axis: str | None = None,
    same_joint: bool = False,
) -> dict[str, Any] | None:
    """Flip one structural element (axis / joint / direction) on the first tail movement."""
    v = copy.deepcopy(cfg)
    st = _first_tail_step(v)
    if st is None:
        return None
    p = st.get("parameters") or {}
    t = st.get("type")

    if kind == "axis":
        from motion_neg_axis_pick import pick_neg_arc_plane, pick_separated_axis, pick_separated_axis_and_joint

        if t == "movement":
            dirs = p.get("directions") or []
            if not dirs:
                return None
            ax = (primary_axis or next(iter((dirs[0].get("degrees") or {}).keys()))).lower()
            base_joint = str(p.get("joint") or "shoulder")
            if same_joint:
                new_ax, sep = pick_separated_axis(v, ax, joint_preference=base_joint)
                neg_joint = base_joint
                probe = {"separation": sep, "same_joint": True}
            else:
                new_ax, neg_joint, probe = pick_separated_axis_and_joint(
                    v, ax, joint_preference=base_joint
                )

            # GT z+- (etc.): swap axis on every direction entry, not only the first —
            # otherwise neg looks stepwise (e.g. y+ then z-).
            new_dirs: list[dict[str, Any]] = []
            for d in dirs:
                deg = dict(d.get("degrees") or {})
                if ax not in deg:
                    continue
                val = deg.pop(ax)
                deg[new_ax] = val
                nd = dict(d)
                nd["degrees"] = deg
                new_dirs.append(nd)

            if not new_dirs:
                deg0 = dirs[0].get("degrees") or {}
                if not deg0:
                    return None
                val = deg0.get(ax, _MAG)
                new_dirs = [
                    {
                        "degrees": {new_ax: float(val) if isinstance(val, (int, float)) else _MAG},
                        "speed": dirs[0].get("speed", 2.0),
                        "hold_time": dirs[0].get("hold_time", 0.08),
                    }
                ]

            p["directions"] = new_dirs
            p["joint"] = neg_joint
            v["neg_axis_meta"] = {
                "true_axis": ax,
                "neg_axis": new_ax,
                "neg_joint": neg_joint,
                **probe,
            }
            return v
        if t == "path":
            if p.get("shape") == "line":
                true_ax = primary_axis or str(p.get("axis", "x")).lower()
                new_ax, _, probe = pick_separated_axis_and_joint(v, true_ax)
                p["axis"] = new_ax
                v["neg_axis_meta"] = {"true_axis": true_ax, "neg_axis": new_ax, **probe}
            else:
                plane = str(p.get("plane", "xy")).lower()
                p["plane"] = pick_neg_arc_plane(plane, primary_axis)
                v["neg_axis_meta"] = {
                    "true_plane": plane,
                    "neg_plane": p["plane"],
                    "avoid_axis": primary_axis,
                }
            st["parameters"] = p
            return v
        return None

    if kind == "joint":
        if t != "movement":
            return None
        p["joint"] = _alt_joint(str(p.get("joint", "shoulder")))
        st["parameters"] = p
        return v

    if kind == "direction":
        if t == "movement":
            dirs = p.get("directions") or []
            if not dirs:
                return None
            deg = dirs[0].get("degrees") or {}
            if not deg:
                return None
            ax = next(iter(deg.keys()))
            val = deg[ax]
            if isinstance(val, (int, float)):
                deg[ax] = -val
                dirs[0]["degrees"] = deg
                return v
            return None
        if t == "path":
            if p.get("shape") == "line" and isinstance(p.get("distance"), (int, float)):
                p["distance"] = -float(p["distance"])
            else:
                d = str(p.get("direction", "ccw")).lower()
                p["direction"] = "cw" if d == "ccw" else "ccw"
            st["parameters"] = p
            return v
        return None

    return None
