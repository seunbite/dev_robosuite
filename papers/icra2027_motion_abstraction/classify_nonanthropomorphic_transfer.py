#!/usr/bin/env python3
"""Classify non-anthropomorphic programs as retargeting or rematerialization.

The reference is the human-authored humanoid Motion Program for the same cue.
Retargeting requires an equivalent multiset of carrier-independent dynamic
signatures after carrier substitution. Tokens retain where a motion is enacted,
but compare only motion family, axis, and sign: a base path, EE path, and EE
movement therefore match when those three properties agree. Pose setup is
excluded from this comparison.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "data/results/motion_configs"
GT_PATH = CONFIGS / "embodiment_motion_code_gt/embodiment_motion_code_all.json"
LLM_PATH = CONFIGS / "embodiment_motion_code_gemini_gemini-2.5-pro/embodiment_motion_code_all.json"
OUT_DIR = Path(__file__).resolve().parent / "analysis"
PLATFORMS = ("single_arm", "stretch", "roomba")

# These equivalences express a carrier-independent movement direction, rather
# than an anatomical joint identity (e.g., head tilt and arm z are vertical).
AXIS_GROUPS = {
    "vertical": {"z", "tilt", "height", "pitch"},
    "lateral": {"y", "pan", "yaw", "roll"},
    "sagittal": {"x", "forward", "backward"},
}


def _axis_group(axis: object) -> str:
    value = str(axis).lower()
    for group, names in AXIS_GROUPS.items():
        if value in names:
            return group
    return value


def _numbers(value: object) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [n for child in value.values() for n in _numbers(child)]
    if isinstance(value, list):
        return [n for child in value for n in _numbers(child)]
    return []


def _family(value: object, repetition: object = 1) -> str:
    """Movement family: repeated/alternating motions versus single strokes."""
    nums = _numbers(value)
    alternating = bool(nums) and min(nums) < 0 < max(nums)
    repeated = isinstance(repetition, (int, float)) and repetition > 1
    return "Oscillation" if alternating or repeated else "Stroke"


def _sign(value: object) -> str:
    nums = _numbers(value)
    if not nums:
        return "unknown"
    if min(nums) < 0 < max(nums):
        return "+-"
    if max(nums) > 0:
        return "+"
    if min(nums) < 0:
        return "-"
    return "0"


def _token(family: str, axis: object, value: object, where: str) -> tuple[str, str, str, str]:
    """Create a motion token.

    A repeated action returns to its starting configuration between cycles.
    Thus a program written as ``[20, 0], rep=2`` is still a closed +/-
    oscillation, even though its explicit endpoint list is non-negative.
    """
    sign = "+-" if family == "Oscillation" else _sign(value)
    return (family, _axis_group(axis), sign, where)


def _dynamic_signature(program: dict) -> Counter[tuple[str, str, str, str]]:
    """Return ``(family, axis, sign, where)`` action tokens."""
    signature: Counter[tuple[str, str, str, str]] = Counter()
    for primitive in program.get("movements", []):
        kind = primitive.get("type")
        # Pose establishes a silhouette and hold only changes timing; neither
        # constitutes a dynamic movement type for retargeting comparison.
        if kind in {"pose", "hold"}:
            continue
        params = primitive.get("parameters", {})
        if kind == "movement":
            movement = params.get("movement", params)
            repetition = movement.get("repetition", params.get("repetition", 1))
            joints = movement.get("joints", [])
            if joints:
                for joint in joints:
                    degrees = joint.get("degrees", [])
                    family = _family(degrees, repetition)
                    signature[_token(family, joint.get("axis", "unknown"), degrees, "ee_movement")] += 1
            else:
                directions = movement.get("directions", params.get("directions", []))
                axes = set()
                for direction in directions:
                    axes.update(direction.get("degrees", {}).keys())
                if axes:
                    for axis in axes:
                        values = [direction.get("degrees", {}).get(axis) for direction in directions]
                        family = _family(values, repetition)
                        signature[_token(family, axis, values, "ee_movement")] += 1
                else:
                    signature[("unknown", "unknown", "unknown", "ee_movement")] += 1
        elif kind == "path":
            path = params.get("path", params)
            shape = str(path.get("shape", "line")).lower()
            repetition = path.get("repetition", 1)
            where = path.get("mode", "ee_path")
            axis = path.get("axis")

            if shape == "arc":
                # A planar arc is an azimuthal turn regardless of its radius.
                value = path.get("degrees", path.get("sweep", 0))
                if any(abs(number) > 1e-9 for number in _numbers(value)):
                    family = _family(value, repetition)
                    signature[_token(family, "pan", value, str(where))] += 1
            elif axis is not None:
                # EE line paths store displacement in ``distance`` rather than
                # under the named axis (e.g., axis=y, distance=+0.12).
                value = path.get(axis, path.get("distance", 0))
                if any(abs(number) > 1e-9 for number in _numbers(value)):
                    family = _family(value, repetition)
                    signature[_token(family, axis, value, str(where))] += 1
            else:
                # A diagonal line may explicitly contain multiple coordinates.
                # Preserve each non-zero component instead of collapsing it to
                # axis=unknown, sign=0.
                for coordinate in ("x", "y", "z"):
                    if coordinate not in path:
                        continue
                    value = path[coordinate]
                    if not any(abs(number) > 1e-9 for number in _numbers(value)):
                        continue
                    family = _family(value, repetition)
                    signature[_token(family, coordinate, value, str(where))] += 1
        else:
            signature[(str(kind), "unknown", "unknown", str(kind))] += 1
    return signature


def _axes_match(reference_axis: str, target_axis: str, platform: str) -> bool:
    """Return whether two carrier-independent axes preserve the same evidence.

    Roomba has no independently actuated vertical carrier. A humanoid vertical
    stroke is therefore realized as fore--aft base travel, just as a humanoid
    sagittal stroke is. This embodiment-specific equivalence does not apply to
    articulated target robots.
    """
    if platform == "roomba" and target_axis == "sagittal":
        return reference_axis in {"vertical", "sagittal"}
    return reference_axis == target_axis


def _signs_match(reference_sign: str, target_sign: str, reference_axis: str, target_axis: str) -> bool:
    """Treat mirrored single-direction lateral motions as equivalent.

    Left and right turns preserve the same lateral movement narrative. Closed
    oscillation (``+-``) remains distinct from a one-way stroke.
    """
    if reference_axis == "lateral" and target_axis == "lateral":
        if reference_sign in {"+", "-"} and target_sign in {"+", "-"}:
            return True
    return reference_sign == target_sign


def _classify(reference: dict, target: dict, platform: str) -> tuple[str, str, list[str], list[str]]:
    ref = _dynamic_signature(reference)
    tgt = _dynamic_signature(target)
    # ``where`` is descriptive only. A base path, EE path, and EE movement
    # preserve the same action when family, axis, and sign are identical.
    ref_items, tgt_items = list(ref.elements()), list(tgt.elements())
    remaining = list(tgt_items)
    for ref_item in ref_items:
        ref_family, ref_axis, ref_sign, _ = ref_item
        match_index = None
        for i, target_item in enumerate(remaining):
            tgt_family, tgt_axis, tgt_sign, _ = target_item
            if (
                ref_family == tgt_family
                and _axes_match(ref_axis, tgt_axis, platform)
                and _signs_match(ref_sign, tgt_sign, ref_axis, tgt_axis)
            ):
                match_index = i
                break
        if match_index is None:
            return "rematerialization", "dynamic signature differs", sorted(map(str, ref_items)), sorted(map(str, tgt_items))
        remaining.pop(match_index)
    if not remaining:
        return "retargeting", "equivalent dynamic signature", sorted(map(str, ref_items)), sorted(map(str, tgt_items))
    return "rematerialization", "dynamic signature differs", sorted(map(str, ref.elements())), sorted(map(str, tgt.elements()))


def _by_cue(records: list[dict]) -> dict[str, dict]:
    return {record["cue"]: record for record in records}


def main() -> None:
    gt = _by_cue(json.loads(GT_PATH.read_text()))
    llm = _by_cue(json.loads(LLM_PATH.read_text()))
    rows = []
    for author, source in (("Human", gt), ("LLM", llm)):
        for cue, reference_record in gt.items():
            reference = reference_record["platforms"]["humanoid"]
            target_record = source[cue]
            for platform in PLATFORMS:
                target = target_record["platforms"][platform]
                label, rationale, reference_sig, target_sig = _classify(reference, target, platform)
                rows.append({
                    "author": author,
                    "platform": platform,
                    "cue": cue,
                    "label": label,
                    "rationale": rationale,
                    "reference_signature": " | ".join(reference_sig),
                    "target_signature": " | ".join(target_sig),
                    "reference_program": reference.get("source_code", "").replace("\n", " "),
                    "target_program": target.get("source_code", "").replace("\n", " "),
                })

    summary = defaultdict(Counter)
    for row in rows:
        summary[(row["author"], row["platform"])][row["label"]] += 1
    OUT_DIR.mkdir(exist_ok=True)
    with (OUT_DIR / "nonanthropomorphic_transfer_rule_labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    output = {
        "scope": "30 cues shared by human- and LLM-authored datasets; the extension-20 set currently has no human-authored non-anthropomorphic programs.",
        "rule": "Each dynamic action is tokenized as (family, axis, sign, where). Retargeting iff the human-authored humanoid reference and target programs have equivalent multisets after ignoring where; thus base paths, EE paths, and EE movements match when family, axis, and sign agree. Otherwise: rematerialization.",
        "summary": [
            {
                "author": author,
                "robot": robot,
                "retargeting": counts["retargeting"],
                "rematerialization": counts["rematerialization"],
                "total": sum(counts.values()),
                "retargeting_percent": round(100 * counts["retargeting"] / sum(counts.values()), 1),
                "rematerialization_percent": round(100 * counts["rematerialization"] / sum(counts.values()), 1),
            }
            for (author, robot), counts in sorted(summary.items())
        ],
    }
    (OUT_DIR / "nonanthropomorphic_transfer_rule_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
