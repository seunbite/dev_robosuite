#!/usr/bin/env python3
"""Generate deterministic two-choice H4 questions from Human GT attributes."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
GT = REPO / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
OUT_JSON = REPO / "data/results/evaluation/h4_gt_binary_mcq.json"
OUT_COUNTERBALANCED_JSON = (
    REPO / "data/results/evaluation/h4_gt_binary_mcq_counterbalanced.json"
)
OUT_MD = REPO / "data/results/evaluation/h4_gt_binary_mcq_summary.md"
# Stretch replaces quadruped in the blank/binary paper eval set.
PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "roomba")
DIRECTIONS = ("front", "back", "in", "out", "up", "down")
AXES = ("x", "y", "z")


def _stable_index(key: str, n: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n


def _other(key: str, correct: str, choices: tuple[str, ...]) -> str:
    alternatives = tuple(value for value in choices if value != correct)
    return alternatives[_stable_index(key, len(alternatives))]


def _shift_percentile(key: str, value: float) -> float:
    candidates = [candidate for candidate in (value - 30, value + 30) if 0 <= candidate <= 100]
    return candidates[_stable_index(key, len(candidates))]


def _display(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _question(
    *,
    entry: dict[str, Any],
    platform: str,
    component_index: int,
    component_type: str,
    attribute: str,
    correct: Any,
    foil: Any,
    rule: str,
) -> dict[str, Any]:
    qid = (
        f"g{int(entry['idx']):02d}.{platform}.c{component_index + 1}."
        f"{attribute.replace('.', '_')}"
    )
    correct_first = _stable_index(qid + ".order", 2) == 0
    ordered = [correct, foil] if correct_first else [foil, correct]
    options = [
        {"id": option_id, "value": value, "text": _display(value)}
        for option_id, value in zip(("A", "B"), ordered)
    ]
    return {
        "question_id": qid,
        "cue_idx": int(entry["idx"]),
        "cue": str(entry["cue"]),
        "category": entry.get("category"),
        "platform": platform,
        "component_index": component_index,
        "component_number": component_index + 1,
        "component_type": component_type,
        "attribute": attribute,
        "prompt": (
            f"Gesture '{entry['cue']}'의 {platform} core motion에서 "
            f"{component_index + 1}번째 {component_type} component의 "
            f"'{attribute}' 값으로 맞는 것은?"
        ),
        "options": options,
        "correct_option": "A" if correct_first else "B",
        "correct_value": correct,
        "foil_value": foil,
        "mutation_rule": rule,
    }


def _component_payload(movement: dict[str, Any]) -> dict[str, Any]:
    parameters = movement.get("parameters") or {}
    kind = str(movement.get("type", "unknown")).lower()
    nested = parameters.get(kind)
    return nested if isinstance(nested, dict) else parameters


def _counterbalanced(question: dict[str, Any]) -> dict[str, Any]:
    """Return the paired item with A/B positions exactly reversed."""
    first, second = question["options"]
    swapped = [
        {**second, "id": "A"},
        {**first, "id": "B"},
    ]
    return {
        **question,
        "question_id": f"{question['question_id']}.reversed",
        "paired_question_id": question["question_id"],
        "options": swapped,
        "correct_option": "B" if question["correct_option"] == "A" else "A",
        "option_order": "reversed_from_original",
    }


def _extract(entry: dict[str, Any], platform: str) -> list[dict[str, Any]]:
    row = entry["platforms"][platform]
    questions: list[dict[str, Any]] = []
    for ci, movement in enumerate(row.get("movements") or []):
        kind = str(movement.get("type", "unknown")).lower()
        payload = _component_payload(movement)

        # Repetition is evaluated semantically: exactly one versus plural.
        rep = payload.get("repetition", payload.get("rep"))
        if isinstance(rep, (int, float)):
            correct = "1" if float(rep) <= 1 else "plural"
            foil = "plural" if correct == "1" else "1"
            questions.append(_question(
                entry=entry, platform=platform, component_index=ci,
                component_type=kind, attribute="repetition",
                correct=correct, foil=foil, rule="1_vs_plural",
            ))

        if kind == "pose":
            pose = payload.get("pose") if isinstance(payload.get("pose"), dict) else payload
            direction_key = "arm_position" if "arm_position" in pose else "dir"
            direction = str(pose.get(direction_key, "")).lower()
            if direction in DIRECTIONS:
                questions.append(_question(
                    entry=entry, platform=platform, component_index=ci,
                    component_type=kind, attribute="direction",
                    correct=direction,
                    foil=_other(f"{entry['idx']}.{platform}.{ci}.direction", direction, DIRECTIONS),
                    rule="correct_vs_one_of_other_5_directions",
                ))
            grip = str(pose.get("gripper_orientation", "")).lower()
            if grip in ("horizontal", "vertical"):
                questions.append(_question(
                    entry=entry, platform=platform, component_index=ci,
                    component_type=kind, attribute="gripper_orientation",
                    correct=grip,
                    foil="vertical" if grip == "horizontal" else "horizontal",
                    rule="horizontal_vs_vertical",
                ))
            left_arm = str(pose.get("left_arm", "")).lower()
            if left_arm in ("mirror", "still"):
                questions.append(_question(
                    entry=entry, platform=platform, component_index=ci,
                    component_type=kind, attribute="left_arm",
                    correct=left_arm,
                    foil="still" if left_arm == "mirror" else "mirror",
                    rule="mirror_vs_still",
                ))
            for axis in AXES:
                value = pose.get(axis)
                if isinstance(value, (int, float)):
                    questions.append(_question(
                        entry=entry, platform=platform, component_index=ci,
                        component_type=kind, attribute=f"pose.{axis}",
                        correct=value,
                        foil=_shift_percentile(
                            f"{entry['idx']}.{platform}.{ci}.pose.{axis}", float(value)
                        ),
                        rule="correct_vs_plus_or_minus_30_clipped_to_0_100",
                    ))

        if kind == "path":
            shape = str(payload.get("shape", "")).lower()
            axis = str(payload.get("axis", "")).lower()
            distance = payload.get("distance")

            def _is_zero_distance(value: Any) -> bool:
                if isinstance(value, (int, float)):
                    return float(value) == 0.0
                if isinstance(value, list) and value:
                    try:
                        return all(float(item) == 0.0 for item in value)
                    except (TypeError, ValueError):
                        return False
                return False

            # Zero-distance paths are hold pauses (often from legacy hold→path rewrite).
            # Do not ask path.axis / sign questions on them.
            if _is_zero_distance(distance):
                continue

            if axis in AXES:
                questions.append(_question(
                    entry=entry, platform=platform, component_index=ci,
                    component_type=kind, attribute="path.axis",
                    correct=axis,
                    foil=_other(f"{entry['idx']}.{platform}.{ci}.path.axis", axis, AXES),
                    rule="correct_vs_other_cartesian_axis",
                ))
            if shape == "line" and isinstance(distance, (int, float)) and float(distance) != 0:
                questions.append(_question(
                    entry=entry, platform=platform, component_index=ci,
                    component_type=kind, attribute="path.distance_sign",
                    correct=distance, foil=-float(distance),
                    rule="line_distance_sign_reversal",
                ))
            if shape == "line":
                for displacement_axis in AXES:
                    value = payload.get(displacement_axis)
                    if isinstance(value, (int, float)) and float(value) != 0:
                        questions.append(_question(
                            entry=entry, platform=platform, component_index=ci,
                            component_type=kind,
                            attribute=f"path.{displacement_axis}_sign",
                            correct=value, foil=-float(value),
                            rule="line_axis_displacement_sign_reversal",
                        ))
            for angle_key in ("deg", "degrees"):
                value = payload.get(angle_key)
                if isinstance(value, (int, float)) and float(value) != 0:
                    questions.append(_question(
                        entry=entry, platform=platform, component_index=ci,
                        component_type=kind, attribute=f"path.{angle_key}_sign",
                        correct=value, foil=-float(value),
                        rule="path_angle_sign_reversal",
                    ))
                elif isinstance(value, list) and value and all(
                    isinstance(item, (int, float)) for item in value
                ) and any(float(item) != 0 for item in value):
                    questions.append(_question(
                        entry=entry, platform=platform, component_index=ci,
                        component_type=kind, attribute=f"path.{angle_key}_sign",
                        correct=value, foil=[-float(item) for item in value],
                        rule="path_angle_list_sign_reversal",
                    ))
    return questions


def main() -> None:
    master = json.loads(GT.read_text(encoding="utf-8"))
    questions = [
        question
        for entry in master
        for platform in PLATFORMS
        for question in _extract(entry, platform)
    ]
    ids = [question["question_id"] for question in questions]
    if len(ids) != len(set(ids)):
        duplicates = [item for item, count in Counter(ids).items() if count > 1]
        raise RuntimeError(f"Duplicate question IDs: {duplicates}")
    by_attribute = Counter(question["attribute"] for question in questions)
    by_platform = Counter(question["platform"] for question in questions)
    payload = {
        "definition": {
            "source": str(GT.relative_to(REPO)),
            "format": "deterministic binary multiple choice",
            "option_order": "SHA-256 deterministic A/B shuffle",
            "rules": [
                "repetition: 1 versus plural",
                "direction: correct versus one alternative among six",
                "gripper_orientation: horizontal versus vertical",
                "left_arm: mirror versus still",
                "pose x/y/z: correct versus ±30 while remaining in [0,100]",
                "path axis: correct versus another Cartesian axis",
                "line/path signed displacement or angle: sign reversal",
            ],
        },
        "summary": {
            "questions": len(questions),
            "by_platform": dict(by_platform),
            "by_attribute": dict(by_attribute),
        },
        "questions": questions,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    counterbalanced_payload = {
        **payload,
        "definition": {
            **payload["definition"],
            "format": "paired counterbalanced binary multiple choice",
            "option_order": "exact A/B reversal of h4_gt_binary_mcq.json",
        },
        "questions": [_counterbalanced(question) for question in questions],
    }
    OUT_COUNTERBALANCED_JSON.write_text(
        json.dumps(counterbalanced_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# H4 GT binary multiple-choice set",
        "",
        f"- Total questions: **{len(questions)}**",
        f"- Source GT cells: **{len(master) * len(PLATFORMS)}**",
        "- Option ordering is deterministic and balanced by a stable hash.",
        "",
        "## Questions by platform",
        "",
        "| Platform | N |",
        "|---|---:|",
    ]
    lines.extend(f"| {platform} | {by_platform[platform]} |" for platform in PLATFORMS)
    lines.extend([
        "",
        "## Questions by attribute",
        "",
        "| Attribute | N |",
        "|---|---:|",
    ])
    lines.extend(f"| {attribute} | {count} |" for attribute, count in sorted(by_attribute.items()))
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(OUT_JSON)
    print(OUT_COUNTERBALANCED_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
