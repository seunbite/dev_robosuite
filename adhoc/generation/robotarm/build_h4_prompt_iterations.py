#!/usr/bin/env python3
"""Build prompt-ablation datasets for iterative H4 blank-config experiments."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
EVAL = REPO / "data/results/evaluation"
ORIGINAL = EVAL / "h4_gt_blank_config_mcq.json"
REVERSED = EVAL / "h4_gt_blank_config_mcq_counterbalanced.json"

DESCRIPTIONS = {
    "nod_yes": "Repeatedly nod to communicate yes or agreement.",
    "shake_head_no": "Repeatedly shake side to side to communicate no.",
    "draw_circle_repeat": "Trace a circular shape more than once.",
    "gaze_shift_avoid": "Turn or shift away to avoid gaze.",
    "fist_pump": "Raise a closed fist and pump it repeatedly in celebration.",
    "tilt_head_curious": "Tilt to one side to express curiosity.",
    "cover_yawn_sleepy": "Bring a hand toward the mouth as if covering a yawn.",
    "rub_eye_tired": "Bring a hand near an eye and rub with a short repeated stroke.",
    "point_you": "Extend toward another person to indicate you.",
    "point_self": "Turn inward toward the robot's own body to indicate self.",
    "beckon_come_here": "Gesture repeatedly inward to invite someone closer.",
    "stop_palm_out": "Present the palm outward as a stop signal.",
    "salute": "Raise the hand near the forehead and perform a salute.",
    "raise_index_one_moment": "Raise one finger to request one moment.",
    "highfive_invite": "Raise an open hand toward another person to invite a high five.",
    "shrug_idk": "Raise or open both sides to express uncertainty.",
    "flex_bicep": "Bend and raise the arm to display the bicep.",
    "air_quotes": "Raise the hand and repeat a small quote-like finger gesture.",
    "arms_open_welcome": "Open both arms outward in a welcoming gesture.",
    "hands_up_surrender": "Raise both hands to show surrender.",
    "stretch_arms_up": "Extend the arms upward in a stretch.",
    "raise_fists_victory": "Raise fists upward in victory.",
    "cross_arms_defensive": "Bring the arms inward across the body defensively.",
    "palms_together_thank_you": "Bring both palms inward together to express thanks.",
    "facepalm": "Bring a hand inward to the face.",
    "depict_tree": "Use the body or arms to depict a tall branching tree.",
    "depict_big_bowl": "Trace a broad bowl-shaped curve.",
    "depict_lion": "Use an expressive raised hand motion to depict a lion.",
    "depict_small_object": "Use a small, compact motion to indicate a small object.",
    "depict_flying_bird": "Move outward like wings to depict a flying bird.",
}

GLOSSARY = """Motion-code conventions:
- direction/arm_position describes the end-effector direction: front/back depth, in/out lateral, up/down vertical.
- pose x/y/z are body-centered task-space percentiles; x is depth, y is lateral, z is height.
- path.axis is the world Cartesian axis of the path. A signed distance or angle sets motion polarity.
- repetition is the literal number of cycles. Repeated semantic gestures normally use a value greater than 1.
- left_arm=mirror means bilateral mirrored motion; left_arm=still means only the primary arm moves.
- gripper_orientation describes the hand/palm orientation, not the arm direction."""

CHECKLIST = """Decision procedure:
1. Locate the single <BLANK> and identify its component and field.
2. Interpret the cue semantics and all neighboring components.
3. Compare both candidate values for physical and semantic consistency.
4. Do not prefer A, B, the first value, a default value, or a smaller number merely because of position.
5. Return the selected candidate value exactly."""

TARGET_GUIDANCE = {
    "direction": "Infer the end-effector direction required by the gesture meaning, not the arm's current location.",
    "gripper_orientation": "Infer hand/palm orientation from the communicative hand shape; do not infer it from arm direction.",
    "left_arm": "Use mirror for inherently bilateral gestures and still for clearly one-arm gestures.",
    "path.axis": "Map vertical motion to z, lateral side-to-side motion to y, and forward/back depth motion to x.",
    "path.degrees_sign": "Use the sign or signed sequence that preserves the intended rotation order.",
    "path.distance_sign": "Use the sign that moves along the intended positive or negative world axis.",
    "path.x_sign": "Choose the x polarity that matches forward/back motion in the gesture sequence.",
    "pose.x": "Infer approximate depth from near-body versus extended-forward placement.",
    "pose.y": "Infer lateral placement from inward, outward, left, or right placement.",
    "pose.z": "Infer height from low, torso, face, or overhead placement.",
    "repetition": "Choose 1 for a single/static action; choose the offered value greater than 1 for repeated, shaking, rubbing, pumping, or oscillating action.",
}

SYNTHETIC_EXAMPLES = {
    "direction": "Example: cue='reach_up', candidates up/down -> exact value up.",
    "gripper_orientation": "Example: a stop sign presents the palm rather than deriving orientation from arm direction.",
    "left_arm": "Example: cue='open_both_arms', candidates mirror/still -> exact value mirror.",
    "path.axis": "Example: a vertical lift with candidates x/z -> exact value z.",
    "path.degrees_sign": "Example: preserve the rotation order described by the cue and neighboring pose.",
    "path.distance_sign": "Example: movement toward negative x with candidates +0.1/-0.1 -> exact value -0.1.",
    "path.x_sign": "Example: forward and backward are opposite x polarities; use sequence context.",
    "pose.x": "Example: near-body placement has less forward depth than an extended reach.",
    "pose.y": "Example: outward placement differs from inward placement on the lateral axis.",
    "pose.z": "Example: an overhead hand has higher z than a waist-level hand.",
    "repetition": "Example: cue='wave_repeatedly', candidates 1/3 -> exact value 3.",
}

HUMAN_IDENTIFIABLE = {
    "direction",
    "left_arm",
    "path.axis",
    "path.degrees_sign",
    "path.distance_sign",
    "path.x_sign",
    "repetition",
}

ROUTED_SCHEME = {
    "direction": "fewshot_value",
    "left_arm": "targeted_ab",
    "path.axis": "checklist_value",
    "path.degrees_sign": "glossary_value",
    "path.distance_sign": "glossary_value",
    "path.x_sign": "targeted_ab",
    "repetition": "glossary_value",
}


def _variant(question: dict[str, Any], name: str) -> dict[str, Any]:
    row = copy.deepcopy(question)
    row["question_id"] = f"{row['question_id']}.{name}"
    row["experiment_id"] = name
    row["prompt"] = row["prompt"].replace("A/B 중 선택하라", "두 후보 값 중 선택하라")
    if name == "value_only":
        row["system_prompt"] = (
            "Recover the one missing value in a robot Human GT motion config. "
            "Choose semantically, independently of candidate order."
        )
    elif name == "glossary_value":
        row["system_prompt"] = (
            "You are an expert in embodied robot gesture motion codes. "
            "Recover the missing value from cue semantics and config context."
        )
        row["prompt"] = (
            f"[GESTURE MEANING]\n{DESCRIPTIONS[row['cue']]}\n\n"
            f"[CONVENTIONS]\n{GLOSSARY}\n\n{row['prompt']}"
        )
    elif name == "checklist_value":
        row["system_prompt"] = (
            "You are a careful robot-motion-code auditor. Apply the supplied decision "
            "procedure before selecting the missing value."
        )
        row["prompt"] = (
            f"[GESTURE MEANING]\n{DESCRIPTIONS[row['cue']]}\n\n"
            f"[CONVENTIONS]\n{GLOSSARY}\n\n[{CHECKLIST}]\n\n{row['prompt']}"
        )
    elif name in ("targeted_value", "targeted_ab", "fewshot_value"):
        row["system_prompt"] = (
            "You are a robot-motion-code auditor. Recover the one missing field from "
            "gesture meaning and the complete movement sequence."
        )
        guidance = TARGET_GUIDANCE[row["attribute"]]
        extra = (
            f"[GESTURE MEANING]\n{DESCRIPTIONS[row['cue']]}\n\n"
            f"[TARGET FIELD GUIDANCE]\n{guidance}\n"
        )
        if name == "fewshot_value":
            extra += f"\n[SYNTHETIC EXAMPLE]\n{SYNTHETIC_EXAMPLES[row['attribute']]}\n"
        row["prompt"] = f"{extra}\n{row['prompt']}"
    else:
        raise ValueError(name)
    if name == "targeted_ab":
        row["response_mode"] = "option"
        row["answer_instruction"] = (
            "Compare the candidate values, ignore their order, and answer only A or B."
        )
    else:
        row["response_mode"] = "value"
        row["answer_instruction"] = (
            "Output only the exact selected candidate VALUE, with no A/B label and no explanation."
        )
    return row


def _iteration_three_variant(question: dict[str, Any], name: str) -> dict[str, Any]:
    source_qid = question["question_id"]
    if name == "glossary_filtered":
        row = _variant(question, "glossary_value")
    else:
        row = _variant(question, ROUTED_SCHEME[question["attribute"]])
        if name == "routed_ab_filtered":
            row["response_mode"] = "option"
            row["answer_instruction"] = (
                "Compare the candidate values, ignore their order, and answer only A or B."
            )
    row["question_id"] = f"{source_qid}.{name}"
    row["experiment_id"] = name
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--variant", default=None)
    args = parser.parse_args()
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))["questions"]
    reversed_rows = json.loads(REVERSED.read_text(encoding="utf-8"))["questions"]
    if args.split == "dev":
        keep = lambda row: int(row["cue_idx"]) <= 20
    elif args.split == "holdout":
        keep = lambda row: int(row["cue_idx"]) > 20
    else:
        keep = lambda row: True
    variants = (
        ("value_only", "glossary_value", "checklist_value")
        if args.iteration == 1
        else ("targeted_value", "targeted_ab", "fewshot_value")
        if args.iteration == 2
        else ("routed_filtered", "glossary_filtered", "routed_ab_filtered")
    )
    if args.variant is not None:
        if args.variant not in variants:
            parser.error(f"--variant must be one of: {', '.join(variants)}")
        variants = (args.variant,)
    rows = []
    for variant in variants:
        for source in (original, reversed_rows):
            for row in source:
                if not keep(row):
                    continue
                if args.iteration == 3:
                    if row["attribute"] not in HUMAN_IDENTIFIABLE:
                        continue
                    rows.append(_iteration_three_variant(row, variant))
                else:
                    rows.append(_variant(row, variant))
    output = EVAL / f"h4_prompt_iteration_{args.iteration}_{args.split}.json"
    payload = {
        "definition": {
            "iteration": args.iteration,
            "split": args.split,
            "dev_cues": "1-20",
            "holdout_cues": "21-30",
            "variants": list(variants),
        },
        "summary": {
            "questions": len(rows),
            "by_experiment": dict(Counter(row["experiment_id"] for row in rows)),
        },
        "questions": rows,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(output)


if __name__ == "__main__":
    main()
