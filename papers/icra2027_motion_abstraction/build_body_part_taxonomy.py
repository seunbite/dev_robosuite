#!/usr/bin/env python3
"""Build the six-way required-body-part-combination taxonomy for Tables I/III."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent.parent
INVENTORY = PAPER / "gesture_critical_factors_90.json"
OUT_TAXONOMY = PAPER / "body_part_taxonomy.json"
OUT_AGGREGATE = PAPER / "cuespec_category_accuracy.json"
OUT_TABLE = PAPER / "table_humanoid_attr_by_category.json"


# The order is used consistently in Fig. 4, Table I, and Appendix Table III.
# Each mutually exclusive category lists every indispensable body part, not
# merely the body region at which the gesture is directed.
CATEGORIES: "OrderedDict[str, tuple[str, tuple[str, ...]]]" = OrderedDict(
    [
        (
            "Head / torso",
            (
                "The cue is produced by head, gaze, or whole-torso posture, without an arm.",
                (
                    "nod_yes", "shake_head_no", "gaze_shift_avoid", "tilt_head_curious",
                    "lean_forward_interest", "abandon_action_drop_turn",
                    "apology_forward_lowered_hold",
                ),
            ),
        ),
        (
            "Head / torso + one arm",
            (
                "One arm or hand must act at a head, face, neck, or torso target.",
                (
                    "cover_yawn_sleepy", "rub_eye_tired", "salute", "facepalm",
                    "self_scratch_head_confused", "fan_face_hot", "tap_temple_think",
                    "circle_temple_crazy", "cover_mouth_gasp", "throat_cut_stop",
                    "visor_search", "shh_gesture", "chin_flick_dismiss",
                    "point_self", "pat_stomach", "pat_self_back", "hand_over_heart_sincerity",
                ),
            ),
        ),
        (
            "One arm / hand",
            (
                "A single arm or hand is the sole required carrier, without an "
                "indispensable head or torso target.",
                (
                    "draw_circle_repeat", "fist_pump", "point_you",
                    "beckon_come_here", "stop_palm_out", "raise_index_one_moment",
                    "highfive_invite", "flex_bicep", "depict_tree", "depict_small_object",
                    "cheers_toast", "highfive_imitation", "fistbump_invite", "yield_turn",
                    "request_turn", "guiding_arm",
                    "protective_arm_out", "slow_down_request_palm_down",
                    "firm_accept_forward_reach", "hesitant_accept_half_reach",
                    "cancel_previous_offer_retract", "conditional_accept_pause_then_reach",
                    "final_decision_down_press", "resume_speaking_pull",
                    "emphasize_point_downbeat", "prepare_action_raise_hold", "exploration_reach",
                    "uncertainty_wave", "clarification_request_palm_tilt", "speed_up_circular_cue",
                    "polite_decline_small_push", "unsure_response_hand_hover_stop",
                    "continue_on_forward_roll", "start_action_forward_snap", "hesitation_pause_hold",
                    "interrupt_hard", "interrupt_soft", "escorting_gesture",
                    "wait_for_me_backward_palm", "prevent_fall_support_reach",
                    "frustration_down_throw", "thumbs_up_approval",
                ),
            ),
        ),
        (
            "Two arms / hands",
            (
                "The cue requires coordinated use of both arms or hands.",
                (
                    "air_quotes", "arms_open_welcome", "hands_up_surrender", "stretch_arms_up",
                    "raise_fists_victory", "cross_arms_defensive", "palms_together_thank_you",
                    "depict_big_bowl", "depict_lion", "depict_flying_bird", "scale_expand",
                    "self_hug", "rub_arms_cold", "dominance_high_open_hold", "contain_space",
                    "expand_space", "compress_space", "balance_scale", "push_pull_dual",
                    "disagreement_cross_then_release", "clap_applause", "cover_ears_too_loud",
                    "shrug_idk", "submission_lowered_open_palm",
                ),
            ),
        ),
    ]
)

EVAL_FILES = {
    "humanoid": {
        "Gemini": REPO / "data/results/evaluation/humanoid_gt_vs_gemini_humanlike90_loc_dir_move.json",
        "Qwen-32B": REPO / "data/results/evaluation/humanoid_gt_vs_qwen32b_humanlike90_loc_dir_move.json",
        "Qwen-7B": REPO / "data/results/evaluation/humanoid_gt_vs_qwen7b_humanlike90_loc_dir_move.json",
    },
    "tiago": {
        "Gemini": REPO / "data/results/evaluation/tiago_gt_vs_gemini_synth60_loc_dir_move.json",
        "Qwen-32B": REPO / "data/results/evaluation/tiago_gt_vs_qwen32b_synth60_loc_dir_move.json",
        "Qwen-7B": REPO / "data/results/evaluation/tiago_gt_vs_qwen7b_synth60_loc_dir_move.json",
    },
}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _axis_keys(keys: list[list[Any]]) -> set[tuple[str, str]]:
    """Project canonical movement keys onto the motion-axis attribute."""
    projected: set[tuple[str, str]] = set()
    for key in keys:
        if len(key) >= 2 and key[0] in {"ax", "arc"}:
            projected.add((str(key[0]), str(key[1])))
        elif len(key) >= 1 and key[0] == "height":
            projected.add(("height", "height"))
    return projected


def _sign_keys(keys: list[list[Any]]) -> set[str]:
    """Project signed linear/joint movement keys onto +, -, or +-."""
    return {
        str(key[2])
        for key in keys
        if len(key) >= 3 and key[0] == "ax" and key[2] in {"+", "-", "+-"}
    }


def _sign_covers(gt: set[str], pred: set[str]) -> bool:
    """A bidirectional prediction covers either one-way sign, as in joint scoring."""
    return all(sign in pred or (sign in {"+", "-"} and "+-" in pred) for sign in gt)


def _metrics(rows: list[dict[str, Any]], platform: str) -> dict[str, float | None]:
    """Same render-effective metrics used by score_humanoid_loc_dir_move.py."""
    loc: dict[str, list[float]] = {axis: [] for axis in "xyz"}
    reach_ok: list[bool] = []
    palm_ok: list[bool] = []
    move_ok: list[bool] = []
    move_axis_ok: list[bool] = []
    move_sign_ok: list[bool] = []
    for row in rows:
        location = row["location"]
        gt_effective = location.get("gt_effective") or location.get("gt") or {}
        delta = location.get("delta") or {}
        for axis in "xyz":
            if axis in gt_effective and isinstance(delta.get(axis), (int, float)):
                loc[axis].append(float(delta[axis]))
        direction = row["direction"]
        reach = direction.get("reach_angle_deg")
        if isinstance(reach, (int, float)):
            reach_ok.append(bool(direction.get("reach_ok", reach <= 1.0)))
        palm = direction.get("palm_angle_deg")
        if isinstance(palm, (int, float)):
            if "palm_ok" in direction:
                palm_ok.append(bool(direction["palm_ok"]))
            elif platform == "tiago":
                palm_ok.append(palm <= 1.0 or abs(palm - 180.0) <= 1.0)
            else:
                palm_ok.append(palm <= 1.0)
        movement = row["movement"]
        if movement.get("applicable"):
            move_ok.append(bool(movement.get("ok")))
            gt_keys = movement.get("gt_keys") or []
            llm_keys = movement.get("llm_keys") or []
            gt_axes = _axis_keys(gt_keys)
            llm_axes = _axis_keys(llm_keys)
            if gt_axes:
                move_axis_ok.append(gt_axes.issubset(llm_axes))
            gt_signs = _sign_keys(gt_keys)
            llm_signs = _sign_keys(llm_keys)
            if gt_signs:
                move_sign_ok.append(_sign_covers(gt_signs, llm_signs))

    def pct(values: list[bool]) -> float | None:
        return 100.0 * sum(values) / len(values) if values else None

    return {
        "locx": _mean(loc["x"]), "locy": _mean(loc["y"]), "locz": _mean(loc["z"]),
        "reach": pct(reach_ok), "palm": pct(palm_ok),
        "move_axis": pct(move_axis_ok), "move_sign": pct(move_sign_ok),
        "movement": pct(move_ok),
    }


def main() -> None:
    cues = [row["cue"] for row in json.loads(INVENTORY.read_text())["cues"]]
    cue_to_cat = {cue: cat for cat, (_, members) in CATEGORIES.items() for cue in members}
    missing = sorted(set(cues) - set(cue_to_cat))
    duplicate_count = sum(cues.count(cue) != 1 for cue in cue_to_cat)
    extra = sorted(set(cue_to_cat) - set(cues))
    if missing or duplicate_count or extra or len(cue_to_cat) != len(cues):
        raise ValueError(f"taxonomy must partition all 90 cues: missing={missing}, extra={extra}")

    definitions = {cat: description for cat, (description, _) in CATEGORIES.items()}
    tax = {
        "taxonomy_name": "required_body_part_combination6",
        "rule": "Each cue is assigned to one mutually exclusive set of indispensable body-part requirements.",
        "definitions": definitions,
        "categories": {cat: list(members) for cat, (_, members) in CATEGORIES.items()},
        "cue_to_cat": cue_to_cat,
        "cue_list": cues,
    }
    OUT_TAXONOMY.write_text(json.dumps(tax, indent=2) + "\n")

    by_cat: dict[str, dict[str, dict[str, float | None]]] = {}
    table: dict[str, dict[str, float | None]] = {}
    for platform, generators in EVAL_FILES.items():
        by_cat[platform] = {}
        for generator, path in generators.items():
            rows = json.loads(path.read_text())["rows"]
            all_metrics = _metrics(rows, platform)
            by_cat[platform][generator] = {"All": all_metrics}
            short_robot = "G1" if platform == "humanoid" else "Tiago"
            short_generator = {
                "Gemini": "Gem.",
                "Qwen-32B": "32B",
                "Qwen-7B": "7B",
            }[generator]
            table[f"{short_robot}|{short_generator}|All"] = all_metrics
            for category in CATEGORIES:
                category_rows = [row for row in rows if cue_to_cat[row["cue"]] == category]
                metrics = _metrics(category_rows, platform)
                by_cat[platform][generator][category] = metrics
                table[f"{short_robot}|{short_generator}|{category}"] = metrics

    aggregate = {
        "taxonomy_name": tax["taxonomy_name"],
        "rule": tax["rule"],
        "definitions": definitions,
        "cue_to_cat": cue_to_cat,
        "cue_list": cues,
        "by_cat": by_cat,
        "overall": {
            platform: {generator: by_cat[platform][generator]["All"] for generator in generators}
            for platform, generators in EVAL_FILES.items()
        },
    }
    OUT_AGGREGATE.write_text(json.dumps(aggregate, indent=2) + "\n")
    OUT_TABLE.write_text(json.dumps(table, indent=2) + "\n")
    print("taxonomy counts:", {cat: len(members) for cat, (_, members) in CATEGORIES.items()})
    print("wrote", OUT_TAXONOMY)
    print("wrote", OUT_AGGREGATE)
    print("wrote", OUT_TABLE)


if __name__ == "__main__":
    main()
