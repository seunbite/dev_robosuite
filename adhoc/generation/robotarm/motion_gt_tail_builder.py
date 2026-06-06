#!/usr/bin/env python3
"""Build motion-config tail steps from pilot40 motion-component GT annotations."""
from __future__ import annotations

import copy
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


def first_pose_step(row: dict[str, Any]) -> dict[str, Any] | None:
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
    out = {
        "idx": int(base_row["idx"]),
        "cue": base_row["cue"],
        "description": base_row.get("description", ""),
        "groundtruth": base_row.get("groundtruth", ""),
        "gt_fixed_first_pose": copy.deepcopy(base_row.get("gt_fixed_first_pose") or {}),
        "movements": [pose, *tail],
        "state": state_tag,
        "generation_mode": state_tag,
    }
    return out


def _first_tail_step(cfg: dict[str, Any]) -> dict[str, Any] | None:
    seen_pose = False
    for st in cfg.get("movements") or []:
        if st.get("type") == "pose":
            seen_pose = True
            continue
        if seen_pose:
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
) -> dict[str, Any] | None:
    """Flip one structural element (axis / joint / direction) on the first tail step."""
    v = copy.deepcopy(cfg)
    st = _first_tail_step(v)
    if st is None:
        return None
    p = st.get("parameters") or {}
    t = st.get("type")

    if kind == "axis":
        from motion_neg_axis_pick import pick_neg_arc_plane, pick_separated_axis_and_joint

        if t == "movement":
            dirs = p.get("directions") or []
            if not dirs:
                return None
            ax = (primary_axis or next(iter((dirs[0].get("degrees") or {}).keys()))).lower()
            base_joint = str(p.get("joint") or "shoulder")
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
