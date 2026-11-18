# humanoid GT vs LLM — continuous location/direction, binary movement

Pred: `data/results/motion_configs/embodiment_motion_code_qwen_qwen32b_humanlike90/embodiment_motion_code_all.json`
Platform: `humanoid` · N = 90 cues.

## Table metrics (loc = mean |Δ| body-% ↓; rch/plm/mov = % ↑)

| Reach↑ | Palm↑ | Locx↓ | Locy↓ | Locz↓ | Movement↑ |
|---:|---:|---:|---:|---:|---:|
| 52.4 | 29.8 | 20.5 | 22.9 | 14.0 | 39.5 |

_Tiago Palm: 0° and 180° both count as match._

## Summary

| Metric | Value |
|---|---:|
| Location cues with GT xyz | 84 |
| Location mean L2 (body %) | 38.5 |
| Location median L2 | 36.6 |
| Location mean per-axis |Δ| | 19.0 |
| Direction mean reach ∠ (°) | 47.1 (n=84) |
| Direction mean palm ∠ (°) | 76.1 (n=84) |
| Direction mean RPY geodesic (°) | 103.6 (n=84) |
| Movement binary acc | 32/81 (40%) |

## Per cue

| # | Cue | Δx | Δy | Δz | L2 | reach° | palm° | geo° | Δr,Δp,Δy | move |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | nod_yes | — | — | — | — | — | — | — | — | ✓ |
| 2 | shake_head_no | — | — | — | — | — | — | — | — | ✓ |
| 3 | draw_circle_repeat | 57.2 | 28.7 | 10.0 | 64.8 | 90 | 180 | 120 | 90,90,0 | ✗ |
| 4 | gaze_shift_avoid | — | — | — | — | — | — | — | — | ✗ |
| 5 | fist_pump | 8.7 | 15.7 | 30.0 | 35.0 | 0 | 90 | 90 | 0,0,90 | ✓ |
| 6 | tilt_head_curious | — | — | — | — | — | — | — | — | ✗ |
| 7 | cover_yawn_sleepy | 3.7 | 25.7 | 25.0 | 36.1 | 0 | 180 | 180 | 0,0,180 | ✗ |
| 8 | rub_eye_tired | 5.0 | 0.0 | 10.0 | 11.2 | 90 | 0 | 90 | 90,90,90 | ✓ |
| 9 | point_you | 35.1 | 23.9 | 30.0 | 52.0 | 0 | 0 | 0 | 0,0,0 | n/a |
| 10 | point_self | 76.3 | 35.4 | 30.0 | 89.3 | 0 | 90 | 90 | 90,90,90 | n/a |
| 11 | beckon_come_here | 53.2 | 11.5 | 20.0 | 58.0 | 90 | 0 | 120 | 90,90,180 | ✗ |
| 12 | stop_palm_out | 6.8 | 39.1 | 10.0 | 41.0 | 90 | 90 | 120 | 90,0,90 | n/a |
| 13 | salute | 26.3 | 25.7 | 15.0 | 39.7 | 90 | 90 | 180 | 90,0,180 | ✗ |
| 14 | raise_index_one_moment | 11.3 | 40.7 | 45.0 | 61.7 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 15 | highfive_invite | 17.6 | 25.7 | 21.7 | 38.0 | 90 | 0 | 180 | 90,0,180 | ✗ |
| 16 | shrug_idk | 14.7 | 18.8 | 50.0 | 55.4 | 90 | 90 | 180 | 90,0,180 | ✓ |
| 17 | flex_bicep | 19.0 | 31.4 | 0.0 | 36.7 | 90 | 0 | 180 | 90,0,180 | ✗ |
| 18 | air_quotes | 13.7 | 15.7 | 20.0 | 28.9 | 90 | 90 | 90 | 90,0,0 | ✗ |
| 19 | arms_open_welcome | 61.7 | 50.0 | 5.0 | 79.6 | 90 | 90 | 90 | 0,0,90 | ✗ |
| 20 | hands_up_surrender | 3.7 | 30.7 | 20.0 | 36.8 | 0 | 90 | 90 | 0,0,90 | ✗ |
| 21 | stretch_arms_up | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 90 | 0 | 0,0,0 | n/a |
| 22 | raise_fists_victory | 1.3 | 35.7 | 5.0 | 36.1 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 23 | cross_arms_defensive | 8.5 | 16.5 | 10.0 | 21.1 | 0 | 90 | 180 | 180,0,180 | n/a |
| 24 | palms_together_thank_you | 8.5 | 16.5 | 21.7 | 28.6 | 90 | 0 | 180 | 90,0,180 | ✓ |
| 25 | facepalm | 5.0 | 25.0 | 20.0 | 32.4 | 90 | 90 | 90 | 90,0,0 | ✗ |
| 26 | depict_tree | 28.0 | 75.7 | 31.2 | 86.6 | 90 | 90 | 90 | 90,0,0 | n/a |
| 27 | depict_big_bowl | 31.7 | 47.6 | 5.0 | 57.4 | 0 | 90 | 0 | 0,0,0 | ✗ |
| 28 | depict_lion | 11.8 | — | 15.0 | 19.1 | 0 | 90 | 180 | 180,0,180 | ✗ |
| 29 | depict_small_object | 1.3 | — | 30.0 | 30.0 | 0 | 180 | 180 | 0,0,180 | ✗ |
| 30 | depict_flying_bird | 21.4 | 47.6 | 15.0 | 54.3 | 0 | 180 | 180 | 180,0,180 | ✓ |
| 31 | self_scratch_head_confused | 16.3 | 22.7 | 14.0 | 31.3 | 90 | 180 | 180 | 90,90,90 | ✗ |
| 32 | pat_stomach | 46.2 | 5.0 | 5.0 | 46.7 | 0 | 90 | 90 | 0,90,180 | ✗ |
| 33 | fan_face_hot | 3.0 | 29.7 | 18.0 | 34.9 | 90 | 90 | 120 | 0,90,90 | ✓ |
| 34 | tap_temple_think | 11.7 | 3.0 | 2.0 | 12.3 | 0 | 180 | 180 | 180,180,0 | ✗ |
| 35 | circle_temple_crazy | 12.4 | 3.0 | 1.0 | 12.8 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 36 | lean_forward_interest | — | — | — | — | — | — | — | — | ✓ |
| 37 | cover_mouth_gasp | 6.7 | 0.0 | 25.0 | 25.9 | 90 | 90 | 120 | 90,90,0 | ✗ |
| 38 | cheers_toast | 15.0 | 15.0 | 0.0 | 21.2 | 90 | 90 | 90 | 90,0,0 | ✗ |
| 39 | highfive_imitation | 25.0 | 28.8 | 5.0 | 38.5 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 40 | fistbump_invite | 2.0 | 9.3 | 2.0 | 9.7 | 0 | 90 | 180 | 180,0,180 | ✗ |
| 41 | pat_self_back | 2.5 | 10.0 | 8.0 | 13.0 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 42 | throat_cut_stop | 21.7 | 0.0 | 25.0 | 33.1 | 0 | 90 | 90 | 0,90,0 | ✓ |
| 43 | visor_search | 21.3 | 25.7 | 5.0 | 33.8 | 90 | 90 | 90 | 90,0,0 | ✗ |
| 44 | yield_turn | 50.2 | 8.7 | 18.0 | 54.0 | 180 | 0 | 180 | 0,0,180 | ✗ |
| 45 | request_turn | 9.7 | 39.4 | 10.0 | 41.8 | 90 | 180 | 180 | 180,90,0 | ✗ |
| 46 | guiding_arm | 10.3 | 5.0 | 15.0 | 18.9 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 47 | protective_arm_out | 1.7 | 12.0 | 0.0 | 12.1 | 90 | 90 | 120 | 90,0,90 | ✓ |
| 48 | slow_down_request_palm_down | 33.4 | 7.4 | 5.0 | 34.5 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 49 | firm_accept_forward_reach | 15.0 | 4.3 | 15.0 | 21.6 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 50 | shh_gesture | 28.0 | 2.0 | 26.0 | 38.3 | 0 | 0 | 0 | 0,0,0 | n/a |
| 51 | scale_expand | 16.7 | 20.0 | 5.0 | 26.5 | 0 | 90 | 90 | 180,90,0 | ✗ |
| 52 | hesitant_accept_half_reach | 10.0 | 24.3 | 5.0 | 26.7 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 53 | cancel_previous_offer_retract | 50.0 | 4.3 | 20.0 | 54.0 | 0 | 90 | 90 | 90,90,90 | ✓ |
| 54 | conditional_accept_pause_then_reach | 18.0 | 24.3 | 5.0 | 30.6 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 55 | final_decision_down_press | 41.6 | 24.4 | 2.0 | 48.3 | 90 | 90 | 120 | 180,90,90 | ✓ |
| 56 | resume_speaking_pull | 22.0 | 40.0 | 5.0 | 45.9 | 180 | 90 | 180 | 0,90,0 | ✗ |
| 57 | emphasize_point_downbeat | 5.7 | 25.7 | 52.0 | 58.3 | 90 | 0 | 90 | 0,90,0 | ✓ |
| 58 | prepare_action_raise_hold | 31.2 | 32.8 | 0.0 | 45.3 | 0 | 180 | 180 | 0,0,180 | ✗ |
| 59 | exploration_reach | 15.0 | 24.3 | 5.0 | 29.0 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 60 | uncertainty_wave | 5.3 | 10.0 | 0.0 | 11.3 | 90 | 0 | 90 | 0,0,90 | ✓ |
| 61 | clarification_request_palm_tilt | 40.0 | 23.9 | 5.0 | 46.9 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 62 | speed_up_circular_cue | 15.0 | 28.8 | 25.0 | 41.0 | 90 | 90 | 120 | 0,90,90 | ✗ |
| 63 | polite_decline_small_push | 21.7 | 20.0 | 5.0 | 29.9 | 90 | 90 | 120 | 90,0,90 | ✓ |
| 64 | unsure_response_hand_hover_stop | 20.0 | 25.3 | 15.0 | 35.5 | 0 | 90 | 90 | 90,90,90 | ✓ |
| 65 | continue_on_forward_roll | 38.4 | 7.4 | 5.0 | 39.4 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 66 | start_action_forward_snap | 32.7 | 44.4 | 2.0 | 55.2 | 90 | 90 | 120 | 0,90,90 | ✓ |
| 68 | hesitation_pause_hold | 5.7 | 39.4 | 5.0 | 40.1 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 69 | interrupt_hard | 33.4 | 7.4 | 5.0 | 34.5 | 0 | 90 | 180 | 180,0,180 | ✗ |
| 70 | interrupt_soft | 10.0 | 28.8 | 20.0 | 36.5 | 0 | 90 | 0 | 0,0,0 | ✓ |
| 71 | self_hug | 23.5 | 16.5 | 15.0 | 32.4 | 0 | 90 | 90 | 0,90,180 | ✗ |
| 72 | rub_arms_cold | 19.7 | 53.3 | 5.0 | 57.1 | 90 | 180 | 120 | 0,90,90 | ✗ |
| 73 | dominance_high_open_hold | 26.3 | 75.7 | 5.0 | 80.3 | 0 | 90 | 90 | 0,0,90 | n/a |
| 74 | contain_space | 1.5 | 51.5 | 15.0 | 53.7 | 90 | 0 | 180 | 90,0,180 | ✓ |
| 75 | expand_space | 21.7 | 15.0 | 5.0 | 26.9 | 90 | 0 | 180 | 90,0,180 | ✓ |
| 76 | compress_space | 40.2 | 30.1 | 15.0 | 52.4 | 90 | 90 | 90 | 90,0,0 | ✓ |
| 77 | balance_scale | 30.2 | 15.1 | 15.0 | 37.0 | 90 | 0 | 90 | 0,0,90 | ✓ |
| 78 | push_pull_dual | 38.4 | 7.4 | 5.0 | 39.4 | 0 | 90 | 90 | 90,90,90 | ✓ |
| 79 | abandon_action_drop_turn | — | — | — | — | — | — | — | — | ✗ |
| 80 | escorting_gesture | 10.3 | 5.0 | 15.0 | 18.9 | 0 | 90 | 180 | 180,0,180 | ✗ |
| 81 | wait_for_me_backward_palm | 4.8 | 16.7 | 18.0 | 25.0 | 90 | 180 | 90 | 90,0,0 | ✗ |
| 82 | prevent_fall_support_reach | 5.0 | 30.3 | 28.0 | 41.5 | 0 | 90 | 180 | 180,0,180 | ✗ |
| 83 | frustration_down_throw | 16.3 | 35.7 | 50.0 | 63.6 | 180 | 90 | 180 | 180,0,90 | ✓ |
| 84 | submission_lowered_open_palm | 29.6 | 14.4 | 10.0 | 34.4 | 90 | 0 | 90 | 90,0,0 | ✗ |
| 85 | apology_forward_lowered_hold | 22.0 | 8.1 | 10.0 | 25.5 | 0 | 90 | 0 | 0,0,0 | ✗ |
| 86 | chin_flick_dismiss | 8.1 | 10.0 | 28.0 | 30.8 | 180 | 180 | 180 | 180,0,0 | ✗ |
| 87 | cover_ears_too_loud | 1.7 | 37.4 | 16.0 | 40.7 | 0 | 180 | 180 | 0,0,180 | ✓ |
| 88 | disagreement_cross_then_release | 78.2 | 17.5 | 5.0 | 80.3 | 0 | 90 | 90 | 0,90,180 | ✓ |
| 89 | thumbs_up_approval | 24.5 | 28.4 | 18.0 | 41.6 | 90 | 90 | 120 | 0,90,90 | n/a |
| 90 | clap_applause | 22.3 | 34.8 | 5.0 | 41.7 | 90 | 90 | 120 | 90,90,0 | ✗ |
| 91 | hand_over_heart_sincerity | 0.3 | 2.0 | 12.0 | 12.2 | 0 | 90 | 90 | 0,90,180 | ✗ |

## Notes

- Location units are body-relative Motion Code percentages (0–100).
- Unspecified LLM xyz axes are filled from the G1 feasible-bank pose the renderer selects for dir|grip|elbow (render-effective EE %; no 50 penalty).
- Location match: |Δaxis| ≤ 5.0.
- Direction: `dir` → reach unit vector; `grip` palm token → palm normal; RPY from the reach–palm frame (geodesic = rotation angle between GT and LLM frames). Reach/palm angles are acos ∈ [0, 180°].
- Movement binary: GT (axis, sign) set ⊆ LLM set; line paths share the same (axis, sign) keys as joint moves; arcs stay as (arc, plane).
