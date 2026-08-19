#!/usr/bin/env python3
"""Label each GT movement component as core | submotion | hold.

Rules (heuristic, editable JSON afterward):
- ``hold`` components → hold
- Solo non-hold component in a cue → core
- Base ``path`` with |degrees|≥270 (full spin / approach flourish) → submotion
  when another non-hold component exists
- Leading preparatory ``pose`` before a later pose/move/path that carries the
  cue (second pose, or move/path) → submotion
- Trailing base ``path`` after an arm EE path/move → submotion
- Otherwise → core

Writes:
  data/results/evaluation/gt_component_core_submotion.json
and annotates H4 glossary packs with ``motion_role`` on each question.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
GT = REPO / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
OUT = REPO / "data/results/evaluation/gt_component_core_submotion.json"
EVAL = REPO / "data/results/evaluation"
PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "roomba")


def _path_params(step: dict[str, Any]) -> dict[str, Any]:
    params = step.get("parameters") or {}
    nested = params.get("path")
    return nested if isinstance(nested, dict) else params


def _is_base_path(step: dict[str, Any]) -> bool:
    if str(step.get("type")).lower() != "path":
        return False
    return str(_path_params(step).get("mode") or "").lower() == "base"


def _abs_degrees(step: dict[str, Any]) -> float | None:
    p = _path_params(step)
    deg = p.get("degrees")
    if isinstance(deg, (int, float)):
        return abs(float(deg))
    if isinstance(deg, list) and deg:
        try:
            return max(abs(float(x)) for x in deg)
        except (TypeError, ValueError):
            return None
    return None


def _label_sequence(steps: list[dict[str, Any]]) -> list[str]:
    n = len(steps)
    roles = ["core"] * n
    types = [str(s.get("type") or "").lower() for s in steps]
    non_hold = [i for i, t in enumerate(types) if t != "hold"]
    if not non_hold:
        return ["hold" if t == "hold" else "core" for t in types]

    for i, t in enumerate(types):
        if t == "hold":
            roles[i] = "hold"

    if len(non_hold) == 1:
        return roles

    # Full spins on base: submotion if not the only content
    for i in non_hold:
        if _is_base_path(steps[i]):
            deg = _abs_degrees(steps[i])
            if deg is not None and deg >= 270:
                roles[i] = "submotion"

    # Leading pose before another pose / move / path → prep submotion
    pose_idxs = [i for i in non_hold if types[i] == "pose"]
    if len(pose_idxs) >= 2:
        for i in pose_idxs[:-1]:
            roles[i] = "submotion"

    # On stretch/tiago/humanoid: base path alongside arm content → submotion
    has_armish = any(
        (types[i] in ("pose", "move", "movement") and not _is_base_path(steps[i]))
        or (types[i] == "path" and not _is_base_path(steps[i]))
        for i in non_hold
    )
    if has_armish:
        for i in non_hold:
            if _is_base_path(steps[i]) and roles[i] != "hold":
                roles[i] = "submotion"

    # Roomba multi-path: first 360-ish already marked; if ≥3 paths, first is often approach flourish
    path_idxs = [i for i in non_hold if types[i] == "path"]
    if len(path_idxs) >= 3 and not has_armish:
        # keep last path(+hold) as core; earlier unmarked paths that are not last → submotion if line approach after spin already handled
        for i in path_idxs[:-1]:
            if roles[i] == "core":
                # only demote pure approach lines after a spin
                deg = _abs_degrees(steps[i])
                if deg is not None and deg >= 270:
                    roles[i] = "submotion"
                elif i == path_idxs[0] and any(roles[j] == "submotion" for j in path_idxs):
                    pass
                elif i < path_idxs[-1] and _path_params(steps[i]).get("shape") == "arc":
                    roles[i] = "submotion"

    # Ensure at least one core among non-hold
    if not any(roles[i] == "core" for i in non_hold):
        roles[non_hold[-1]] = "core"
    return roles


def build_labels() -> dict[str, Any]:
    master = json.loads(GT.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    counts = {"core": 0, "submotion": 0, "hold": 0}
    for entry in master:
        idx = int(entry["idx"])
        cue = str(entry["cue"])
        for platform in PLATFORMS:
            plat = (entry.get("platforms") or {}).get(platform)
            if not isinstance(plat, dict):
                continue
            steps = plat.get("movements") or []
            roles = _label_sequence(steps)
            for ci, (step, role) in enumerate(zip(steps, roles)):
                counts[role] = counts.get(role, 0) + 1
                rows.append(
                    {
                        "cue_idx": idx,
                        "cue": cue,
                        "platform": platform,
                        "component_index": ci,
                        "component_type": step.get("type"),
                        "motion_role": role,
                        "brief": _brief(step),
                    }
                )
    return {
        "definition": {
            "roles": ["core", "submotion", "hold"],
            "policy": (
                "Heuristic component roles for H4 attr accuracy splits. "
                "core = required for cue identity; submotion = flourish/prep/base "
                "that weights meaning but is not the cue itself; hold = timing hold."
            ),
            "source_gt": str(GT.relative_to(REPO)),
        },
        "summary": {"n_components": len(rows), "by_role": counts},
        "components": rows,
    }


def _brief(step: dict[str, Any]) -> str:
    t = str(step.get("type") or "")
    params = step.get("parameters") or {}
    if t == "pose":
        pose = params.get("pose") if isinstance(params.get("pose"), dict) else params
        keys = ("arm_position", "dir", "gripper_orientation", "z", "x", "y")
        bits = [f"{k}={pose.get(k)}" for k in keys if isinstance(pose, dict) and k in pose]
        return "pose(" + ", ".join(bits[:4]) + ")"
    if t == "path":
        p = _path_params(step)
        return (
            f"path(mode={p.get('mode')}, shape={p.get('shape')}, "
            f"deg={p.get('degrees')}, axis={p.get('axis')})"
        )
    if t in ("move", "movement"):
        return f"{t}(rep={params.get('repetition') or params.get('rep')})"
    if t == "hold":
        return f"hold(t={params.get('t') or params.get('hold_time')})"
    return t


def annotate_glossary_packs(label_index: dict[tuple[int, str, int], str]) -> None:
    stems = [
        "h4_glossary_value_english",
        "h4_glossary_k2",
        "h4_glossary_k3",
        "h4_glossary_k4",
        "h4_glossary_k5",
        "h4_glossary_component",
    ]
    for stem in stems:
        for suffix in ("", "_counterbalanced"):
            path = EVAL / f"{stem}{suffix}.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            n_miss = 0
            for q in payload.get("questions") or []:
                key = (int(q["cue_idx"]), str(q["platform"]), int(q["component_index"]))
                role = label_index.get(key)
                if role is None:
                    n_miss += 1
                    role = "core"
                q["motion_role"] = role
            payload.setdefault("definition", {})["motion_role"] = (
                "per-question component role from gt_component_core_submotion.json"
            )
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"annotated {path.name} miss={n_miss}")


def main() -> None:
    payload = build_labels()
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(OUT)
    index = {
        (int(r["cue_idx"]), str(r["platform"]), int(r["component_index"])): str(r["motion_role"])
        for r in payload["components"]
    }
    # roomba highfive sanity
    for r in payload["components"]:
        if r["cue"] == "highfive_invite" and r["platform"] == "roomba":
            print("roomba highfive", r["component_index"], r["motion_role"], r["brief"])
    annotate_glossary_packs(index)


if __name__ == "__main__":
    main()
