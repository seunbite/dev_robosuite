#!/usr/bin/env python3
"""Shared joint keywords for mobile / quadruped motion configs.

- ``joint: base`` bundles what used to be a top-level ``type: path`` (base translation / rotation).
- ``joint: right_arm`` / ``left_arm`` (spaces ok) map to existing arm DOF groups + ``side``.

Quadruped GIF recording still uses a legacy consumer that expects ``type: path`` for base
locomotion; use :func:`quadruped_movements_for_legacy_path` when writing per-cue JSON.
"""
from __future__ import annotations

import copy
from typing import Any


def canonical_joint_keyword(j: str | None) -> str | None:
    if j is None:
        return None
    n = str(j).strip().lower().replace(" ", "_").replace("-", "_")
    if n in ("base", "root", "omni", "chassis"):
        return "base"
    if n in ("right_arm", "rightarm", "r_arm", "arm_right"):
        return "right_arm"
    if n in ("left_arm", "leftarm", "l_arm", "arm_left"):
        return "left_arm"
    return n


_CART_AXES = frozenset({"x", "y", "z"})


def tiago_preprocess_movement_joints(joints: list[dict]) -> list[dict]:
    """Map high-level arm aliases onto ``joint`` + ``side`` understood by TIAGo renderer."""
    out: list[dict] = []
    for jspec in joints:
        jspec = dict(jspec)
        jspec["_motion_shape"] = str(jspec.get("shape", "line")).lower() or "line"
        cj = canonical_joint_keyword(jspec.get("joint"))
        if cj in ("right_arm", "left_arm"):
            axis = str(jspec.get("axis", "")).lower()
            link = jspec.pop("link", None) or jspec.pop("arm_link", None)
            if link:
                limb = str(link).strip().lower()
            elif axis in _CART_AXES:
                limb = "elbow"
            else:
                limb = "shoulder"
            jspec["joint"] = limb
            jspec["side"] = "right" if cj == "right_arm" else "left"
        out.append(jspec)
    return out


def movement_step_from_base_path(*, path: dict, duration: float | None = None) -> dict:
    step: dict[str, Any] = {
        "type": "movement",
        "parameters": {"movement": {"joints": [{"joint": "base", "path": copy.deepcopy(path)}]}},
    }
    if duration is not None:
        step["duration"] = float(duration)
    return step


def migrate_path_steps_to_base_movements(movements: list[dict]) -> list[dict]:
    """Replace each legacy ``type: path`` step with ``movement`` + ``joint: base``."""
    migrated: list[dict] = []
    for step in movements:
        if not isinstance(step, dict):
            continue
        if step.get("type") != "path":
            migrated.append(copy.deepcopy(step))
            continue
        params = step.get("parameters") or {}
        path = params.get("path")
        if not isinstance(path, dict):
            migrated.append(copy.deepcopy(step))
            continue
        new_step = movement_step_from_base_path(path=path, duration=step.get("duration"))
        migrated.append(new_step)
    return migrated


def migrate_config_row(row: dict) -> dict:
    """Deep-copy a cue row and migrate ``movements`` paths → base movements."""
    out = copy.deepcopy(row)
    mvs = out.get("movements")
    if isinstance(mvs, list):
        out["movements"] = migrate_path_steps_to_base_movements(mvs)
    return out


def quadruped_movements_for_legacy_path(movements: list[dict]) -> list[dict]:
    """Undo ``joint: base`` into standalone ``path`` steps for legacy MJLab scripts."""
    out: list[dict] = []
    for step in movements:
        if not isinstance(step, dict):
            continue
        st = step.get("type")
        if st != "movement":
            out.append(copy.deepcopy(step))
            continue
        mv = (step.get("parameters") or {}).get("movement") or {}
        joints = mv.get("joints") or []
        bases = []
        others = []
        for jspec in joints:
            if canonical_joint_keyword(jspec.get("joint")) == "base" and isinstance(jspec.get("path"), dict):
                bases.append(jspec["path"])
            else:
                others.append(jspec)

        # Only hoist when movement is purely base path(s); otherwise keep step as-is.
        if bases and not others:
            dur = step.get("duration")
            for i, bp in enumerate(bases):
                path_step: dict[str, Any] = {"type": "path", "parameters": {"path": copy.deepcopy(bp)}}
                if dur is not None and i == 0:
                    path_step["duration"] = dur
                out.append(path_step)
            continue

        if bases and others:
            # Preserve legacy-compatible base segments first (each base → path), keep remainder.
            for bp in bases:
                out.append({"type": "path", "parameters": {"path": copy.deepcopy(bp)}})
            rest = copy.deepcopy(step)
            r_mv = (rest.setdefault("parameters", {}).setdefault("movement", {}))
            r_mv["joints"] = others
            if not r_mv["joints"]:
                continue
            out.append(rest)
            continue

        out.append(copy.deepcopy(step))
    return out
