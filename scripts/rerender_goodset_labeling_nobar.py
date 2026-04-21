from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
sys.path.insert(0, str(ROOT / "adhoc" / "robotarm"))
sys.path.insert(0, str(ROOT / "scripts"))

from motion_generation import MotionGenerator, _select_initial_poses  # type: ignore
import build_prompt19_goodset_labeling_pptx as label  # type: ignore


OUT_ROOT = ROOT / "data" / "motions" / "goodset_labeling_simplebar"

CONFIGS = {
    ("sophisticated", "iconic"): ROOT / "data" / "seed" / "motion_configs_prompt_v19_sophisticated.json",
    ("sophisticated", "contextual"): ROOT / "data" / "seed" / "motion_configs_prompt_v19_sophisticated_contextual.json",
    ("no_reasoning", "iconic"): ROOT / "data" / "seed" / "baseline_prompt19_full_no_reasoning" / "motion_configs_prompt_v19_sophisticated_no_reasoning_iconic.json",
    ("no_reasoning", "contextual"): ROOT / "data" / "seed" / "baseline_prompt19_full_no_reasoning" / "motion_configs_prompt_v19_sophisticated_no_reasoning_contextual.json",
}


def rerender(dataset: str, variant: str, cues: set[str]) -> dict[str, Path]:
    config_path = CONFIGS[(variant, dataset)]
    rows = label._load_json_rows(config_path)
    selected = [row for row in rows if row["cue"] in cues]
    out_dir = OUT_ROOT / variant / dataset / "IIWA"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = MotionGenerator(robot_name="IIWA", output_dir=str(out_dir))
    written: dict[str, Path] = {}
    try:
        for row in sorted(selected, key=lambda x: (int(x["idx"]), x["cue"])):
            pose_step = next((m for m in row.get("movements", []) if m.get("type") == "pose"), None)
            pose_index = None
            if pose_step:
                pose_def = pose_step.get("parameters", {}).get("pose")
                if isinstance(pose_def, dict):
                    matching = gen._find_matching_poses(pose_def)
                    selected_poses = _select_initial_poses(matching, pose_def, top_k=1)
                    if selected_poses:
                        pose_index = selected_poses[0]["pose_id"]
            gen.output_dir = str(out_dir)
            gen._set_joint_positions(gen.initial_joint_pos)
            gen.execute_cue(
                cue=row["cue"],
                pose_index=pose_index,
                config_path=str(config_path),
                hz=8,
                cue_idx=int(row["idx"]),
                save_gif=True,
                overlay_progress_bar=True,
                progress_bar_style="simple",
            )
            latest = label.latest_gif_by_cue(out_dir).get(row["cue"])
            if latest:
                written[row["cue"]] = latest
    finally:
        gen.close()
    return written


def main():
    iconic = label.load_ppt_good_bad(label.ICONIC_PPT, "iconic")
    contextual = label.load_ppt_good_bad(label.CONTEXTUAL_PPT, "contextual")

    iconic_good = {item.cue for item in iconic["good"]}
    contextual_good = {item.cue for item in contextual["good"]}

    pickbest_iconic = {
        "wave_hi", "raising_hand_greeting", "point_self", "rub_eye_tired", "circle_temple_crazy",
        "visor_search", "yield_turn", "interrupt_cue", "request_turn", "guiding_arm",
        "protective_arm_out", "slow_down_request_palm_down", "continue_on_forward_roll",
        "firm_accept_forward_reach", "hesitant_accept_half_reach", "cancel_previous_offer_retract",
        "conditional_accept_pause_then_reach", "final_decision_down_press", "beckon_come_here", "speed_up_circular_cue",
    }
    pickbest_contextual = {
        "nod_yes", "tilt_head_curious", "deep_bow_apology", "curl_fingers_give_me", "stretch_arms_up",
        "gentle_retreat_motion", "disagreement_cross_then_release", "allow_entry_open_side", "relief_release_drop",
        "frustration_down_throw", "disbelief_hold_then_drop", "submission_lowered_open_palm", "dominance_high_open_hold",
        "retry_action_reset_circle", "abandon_action_drop_turn", "prepare_action_raise_hold", "commit_action_fast_reach",
        "disengage_step_back_release", "small_rhythmic_bounce", "listening_lean_substitute", "arms_open_welcome", "stop_entry_cross_block",
    }

    rerender("iconic", "no_reasoning", iconic_good)
    rerender("contextual", "no_reasoning", contextual_good)
    rerender("iconic", "sophisticated", pickbest_iconic)
    rerender("contextual", "sophisticated", pickbest_contextual)

    print("WROTE", OUT_ROOT)


if __name__ == "__main__":
    main()
