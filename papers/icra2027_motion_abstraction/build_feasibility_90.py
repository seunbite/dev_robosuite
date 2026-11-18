#!/usr/bin/env python3
"""Build the a-priori feasible/difficult partition for 90 cues x 8 robots."""
from __future__ import annotations

import csv
import json
from pathlib import Path

PAPER = Path(__file__).resolve().parent
SOURCE = PAPER / "gesture_critical_factors_90.json"
OUT_JSON = PAPER / "embodiment_physical_difficulty_90.json"
OUT_CSV = PAPER / "embodiment_physical_difficulty_90.csv"

ROBOTS = (
    "Humanoid",
    "Tiago",
    "Single Arm",
    "Stretch",
    "Roomba",
    "Paddleclaw",
    "Trileg Crown",
    "Spine Crab",
)

# These lists encode a morphology prior, not perceptual success. A cue is
# difficult when its indispensable relation requires a missing carrier,
# contact site, independently controllable bilateral pair, or fine digit/palm.
DIFFICULT = {
    "Humanoid": set(),
    "Tiago": set(),
    "Single Arm": {
        "cover_yawn_sleepy", "rub_eye_tired", "point_self", "salute",
        "raise_index_one_moment", "shrug_idk", "flex_bicep", "air_quotes",
        "arms_open_welcome", "hands_up_surrender", "stretch_arms_up",
        "raise_fists_victory", "cross_arms_defensive",
        "palms_together_thank_you", "facepalm",
        "self_scratch_head_confused", "pat_stomach", "fan_face_hot",
        "tap_temple_think", "circle_temple_crazy", "cover_mouth_gasp",
        "pat_self_back", "throat_cut_stop", "visor_search", "shh_gesture",
        "scale_expand", "self_hug", "rub_arms_cold",
        "dominance_high_open_hold", "contain_space", "expand_space",
        "compress_space", "balance_scale", "push_pull_dual",
        "wait_for_me_backward_palm", "frustration_down_throw",
        "submission_lowered_open_palm", "chin_flick_dismiss",
        "cover_ears_too_loud", "disagreement_cross_then_release",
        "thumbs_up_approval", "clap_applause", "hand_over_heart_sincerity",
    },
    "Stretch": {
        "cover_yawn_sleepy", "rub_eye_tired", "salute",
        "raise_index_one_moment", "shrug_idk", "flex_bicep", "air_quotes",
        "arms_open_welcome", "hands_up_surrender", "stretch_arms_up",
        "raise_fists_victory", "cross_arms_defensive",
        "palms_together_thank_you", "facepalm",
        "self_scratch_head_confused", "fan_face_hot", "tap_temple_think",
        "circle_temple_crazy", "cover_mouth_gasp", "pat_self_back",
        "shh_gesture", "scale_expand", "self_hug", "rub_arms_cold",
        "dominance_high_open_hold", "contain_space", "expand_space",
        "compress_space", "balance_scale", "push_pull_dual",
        "frustration_down_throw", "submission_lowered_open_palm",
        "chin_flick_dismiss", "cover_ears_too_loud",
        "disagreement_cross_then_release", "thumbs_up_approval",
        "clap_applause",
    },
    "Roomba": set(),  # Filled below as the complement of a movement whitelist.
    "Paddleclaw": {
        "raise_index_one_moment", "air_quotes", "shh_gesture",
        "chin_flick_dismiss", "cover_ears_too_loud", "thumbs_up_approval",
    },
    "Trileg Crown": {
        "cover_yawn_sleepy", "rub_eye_tired", "point_self", "stop_palm_out",
        "salute", "raise_index_one_moment", "highfive_invite", "flex_bicep",
        "air_quotes", "palms_together_thank_you", "facepalm",
        "self_scratch_head_confused", "pat_stomach", "fan_face_hot",
        "tap_temple_think", "circle_temple_crazy", "cover_mouth_gasp",
        "cheers_toast", "highfive_imitation", "fistbump_invite",
        "pat_self_back", "throat_cut_stop", "visor_search",
        "slow_down_request_palm_down", "shh_gesture",
        "clarification_request_palm_tilt", "polite_decline_small_push",
        "self_hug", "rub_arms_cold", "submission_lowered_open_palm",
        "chin_flick_dismiss", "cover_ears_too_loud", "thumbs_up_approval",
        "clap_applause", "hand_over_heart_sincerity",
    },
    "Spine Crab": {
        "cover_yawn_sleepy", "rub_eye_tired", "stop_palm_out",
        "raise_index_one_moment", "highfive_invite", "air_quotes",
        "palms_together_thank_you", "slow_down_request_palm_down",
        "shh_gesture", "clarification_request_palm_tilt",
        "polite_decline_small_push", "submission_lowered_open_palm",
        "chin_flick_dismiss", "cover_ears_too_loud", "thumbs_up_approval",
    },
}

ROOMBA_FEASIBLE = {
    "nod_yes", "shake_head_no", "draw_circle_repeat", "gaze_shift_avoid",
    "fist_pump", "tilt_head_curious", "beckon_come_here",
    "lean_forward_interest", "yield_turn", "guiding_arm",
    "firm_accept_forward_reach", "scale_expand",
    "hesitant_accept_half_reach", "cancel_previous_offer_retract",
    "conditional_accept_pause_then_reach", "resume_speaking_pull",
    "exploration_reach", "uncertainty_wave", "speed_up_circular_cue",
    "unsure_response_hand_hover_stop", "continue_on_forward_roll",
    "start_action_forward_snap", "hesitation_pause_hold", "interrupt_hard",
    "interrupt_soft", "expand_space", "abandon_action_drop_turn",
    "escorting_gesture",
}


def main() -> None:
    source = json.loads(SOURCE.read_text())
    cues = [row["cue"] for row in source["cues"]]
    cue_set = set(cues)
    depict = {cue for cue in cues if cue.startswith("depict_")}
    DIFFICULT["Roomba"] = cue_set - ROOMBA_FEASIBLE - depict

    unknown = {robot: sorted(items - cue_set) for robot, items in DIFFICULT.items() if items - cue_set}
    if unknown:
        raise ValueError(f"unknown cue names: {unknown}")

    matrix = []
    by_robot = {}
    for robot in ROBOTS:
        difficult = set(DIFFICULT[robot]) - depict
        feasible = cue_set - difficult
        by_robot[robot] = {
            "feasible_n": len(feasible),
            "difficult_n": len(difficult),
            "feasible": [cue for cue in cues if cue in feasible],
            "difficult": [cue for cue in cues if cue in difficult],
        }
    for row in source["cues"]:
        item = {"index": row["index"], "cue": row["cue"]}
        for robot in ROBOTS:
            item[robot] = "difficult" if row["cue"] in DIFFICULT[robot] and row["cue"] not in depict else "feasible"
        matrix.append(item)

    payload = {
        "source": SOURCE.name,
        "criterion": (
            "A-priori physical expressibility after morphology-appropriate carrier transfer. "
            "Feasible does not imply recognizable or correctly generated; difficult denotes missing "
            "contact, bilateral, fine digit/palm, or independently controllable carrier evidence."
        ),
        "forced_feasible_rule": "Every cue whose name begins with depict_ is feasible for all eight robots.",
        "robots": list(ROBOTS),
        "n_cues": len(cues),
        "by_robot": by_robot,
        "matrix": matrix,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "cue", *ROBOTS])
        writer.writeheader()
        writer.writerows(matrix)

    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_CSV}")
    for robot in ROBOTS:
        d = by_robot[robot]
        print(f"{robot}: feasible={d['feasible_n']}, difficult={d['difficult_n']}")


if __name__ == "__main__":
    main()
