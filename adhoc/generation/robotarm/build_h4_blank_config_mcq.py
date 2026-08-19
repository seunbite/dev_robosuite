#!/usr/bin/env python3
"""Build H4 MCQs with one field blanked in the complete GT movements config."""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
EVAL = REPO / "data/results/evaluation"
GT = REPO / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
BASE = EVAL / "h4_gt_binary_mcq.json"
OUT = EVAL / "h4_gt_blank_config_mcq.json"
OUT_REVERSED = EVAL / "h4_gt_blank_config_mcq_counterbalanced.json"
BLANK = "<BLANK>"


def _display(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _payload(movement: dict[str, Any]) -> dict[str, Any]:
    parameters = movement.get("parameters") or {}
    nested = parameters.get(str(movement["type"]).lower())
    return nested if isinstance(nested, dict) else parameters


def _target_container(
    movement: dict[str, Any], attribute: str
) -> tuple[dict[str, Any], str]:
    payload = _payload(movement)
    if movement["type"] == "pose":
        pose = payload.get("pose") if isinstance(payload.get("pose"), dict) else payload
        key = {
            "direction": "arm_position" if "arm_position" in pose else "dir",
            "gripper_orientation": "gripper_orientation",
            "left_arm": "left_arm",
        }.get(attribute, attribute.removeprefix("pose."))
        return pose, key
    if attribute == "repetition":
        return payload, "repetition" if "repetition" in payload else "rep"
    if movement["type"] == "path" and attribute.startswith("path."):
        return payload, attribute.removeprefix("path.").removesuffix("_sign")
    raise KeyError(f"Cannot locate {attribute} in {movement['type']}")


def _option_rows(correct: Any, foil: Any, correct_option: str) -> list[dict[str, Any]]:
    values = [correct, foil] if correct_option == "A" else [foil, correct]
    return [
        {"id": option_id, "value": value, "text": _display(value)}
        for option_id, value in zip(("A", "B"), values)
    ]


def _blank_question(
    base: dict[str, Any], platform_gt: dict[str, Any]
) -> dict[str, Any]:
    movements = copy.deepcopy(platform_gt["movements"])
    movement = movements[int(base["component_index"])]
    container, key = _target_container(movement, str(base["attribute"]))
    if key not in container:
        raise KeyError(f"{base['question_id']}: missing target field {key}")
    literal_correct = container[key]
    container[key] = BLANK

    if base["attribute"] == "repetition":
        literal_foil = 1 if float(literal_correct) > 1 else 3
        rule = "literal_gt_repetition_vs_1_or_3"
    else:
        literal_foil = base["foil_value"]
        rule = base["mutation_rule"]

    prompt = (
        f"Gesture '{base['cue']}'의 {base['platform']} Human GT motion config 전체가 아래에 있다. "
        f"{base['component_number']}번째 {base['component_type']} component의 "
        f"'{base['attribute']}' field 하나가 blank token으로 대체되었다.\n"
        "전체 config의 문맥을 사용해 빈칸에 들어갈 정확한 attribute 값을 A/B 중 선택하라.\n\n"
        "[FULL GT MOVEMENTS CONFIG]\n"
        f"{json.dumps(movements, indent=2, ensure_ascii=False)}"
    )
    if prompt.count(BLANK) != 1:
        raise RuntimeError(f"{base['question_id']}: expected exactly one blank")

    qid = f"{base['question_id']}.blank"
    return {
        **base,
        "question_id": qid,
        "prompt": prompt,
        "options": _option_rows(literal_correct, literal_foil, base["correct_option"]),
        "correct_value": literal_correct,
        "foil_value": literal_foil,
        "mutation_rule": rule,
        "input_format": "complete_gt_movements_with_one_blank",
        "blank_target": {
            "component_index": int(base["component_index"]),
            "component_type": base["component_type"],
            "attribute": base["attribute"],
            "field_key": key,
        },
    }


def _reversed(question: dict[str, Any]) -> dict[str, Any]:
    first, second = question["options"]
    return {
        **question,
        "question_id": f"{question['question_id']}.reversed",
        "paired_question_id": question["question_id"],
        "options": [{**second, "id": "A"}, {**first, "id": "B"}],
        "correct_option": "B" if question["correct_option"] == "A" else "A",
        "option_order": "reversed_from_original",
    }


def main() -> None:
    master = json.loads(GT.read_text(encoding="utf-8"))
    gt_by_key = {
        (int(entry["idx"]), platform): row
        for entry in master
        for platform, row in entry["platforms"].items()
    }
    base_payload = json.loads(BASE.read_text(encoding="utf-8"))
    questions = [
        _blank_question(
            question,
            gt_by_key[(int(question["cue_idx"]), str(question["platform"]))],
        )
        for question in base_payload["questions"]
    ]
    summary = {
        "questions": len(questions),
        "by_platform": dict(Counter(row["platform"] for row in questions)),
        "by_attribute": dict(Counter(row["attribute"] for row in questions)),
        "correct_options": dict(Counter(row["correct_option"] for row in questions)),
    }
    definition = {
        **base_payload["definition"],
        "format": "complete GT movements config with exactly one blank",
        "source": str(GT.relative_to(REPO)),
        "source_code_included": False,
        "source_code_exclusion_reason": "duplicates target values and would leak answers",
    }
    payload = {"definition": definition, "summary": summary, "questions": questions}
    reversed_payload = {
        **payload,
        "definition": {
            **definition,
            "option_order": "exact A/B reversal of h4_gt_blank_config_mcq.json",
        },
        "summary": {
            **summary,
            "correct_options": dict(
                Counter(_reversed(row)["correct_option"] for row in questions)
            ),
        },
        "questions": [_reversed(row) for row in questions],
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_REVERSED.write_text(
        json.dumps(reversed_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(OUT)
    print(OUT_REVERSED)


if __name__ == "__main__":
    main()
