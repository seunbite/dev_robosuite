# humanoid GT vs LLM — continuous location/direction, binary movement

Pred: `data/results/motion_configs/embodiment_motion_code_qwen_qwen7b_humanlike90/embodiment_motion_code_all.json`
Platform: `humanoid` · N = 90 cues.

## Table metrics (loc = mean |Δ| body-% ↓; rch/plm/mov = % ↑)

| Reach↑ | Palm↑ | Locx↓ | Locy↓ | Locz↓ | Movement↑ |
|---:|---:|---:|---:|---:|---:|
| 50.0 | 32.6 | 23.6 | 29.8 | 11.9 | 16.0 |

_Tiago Palm: 0° and 180° both count as match._

## Summary

| Metric | Value |
|---|---:|
| Location cues with GT xyz | 86 |
| Location mean L2 (body %) | 43.2 |
| Location median L2 | 38.0 |
| Location mean per-axis |Δ| | 21.7 |
| Direction mean reach ∠ (°) | 46.0 (n=86) |
| Direction mean palm ∠ (°) | 65.9 (n=86) |
| Direction mean RPY geodesic (°) | 91.7 (n=86) |
| Movement binary acc | 13/81 (16%) |

## Per cue

| # | Cue | Δx | Δy | Δz | L2 | reach° | palm° | geo° | Δr,Δp,Δy | move |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | nod_yes | — | — | — | — | — | — | — | — | ✓ |
| 2 | shake_head_no | — | — | — | — | — | — | — | — | ✓ |
| 3 | draw_circle_repeat | 30.0 | 7.9 | 10.0 | 32.5 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 4 | gaze_shift_avoid | — | — | — | — | — | — | — | — | ✗ |
| 5 | fist_pump | 51.2 | 42.8 | 20.0 | 69.7 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 6 | tilt_head_curious | — | — | — | — | — | — | — | — | ✗ |
| 7 | cover_yawn_sleepy | 8.5 | 36.5 | 5.0 | 37.8 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 8 | rub_eye_tired | 33.1 | 6.8 | 0.0 | 33.8 | 0 | 90 | 90 | 0,90,180 | ✗ |
| 9 | point_you | 53.0 | 48.1 | 0.0 | 71.6 | 0 | 0 | 0 | 0,0,0 | n/a |
| 10 | point_self | 28.5 | 16.5 | 10.0 | 34.4 | 90 | 90 | 90 | 0,0,90 | n/a |
| 11 | beckon_come_here | 18.1 | 18.2 | 0.0 | 25.7 | 90 | 0 | 120 | 90,90,180 | ✗ |
| 12 | stop_palm_out | 46.7 | 84.2 | 10.0 | 96.8 | 90 | 90 | 120 | 90,0,90 | n/a |
| 13 | salute | 13.3 | 32.4 | 10.0 | 36.4 | 90 | 90 | 180 | 90,0,180 | ✗ |
| 14 | raise_index_one_moment | 31.2 | 17.8 | 35.0 | 50.2 | 90 | 0 | 90 | 90,0,0 | ✗ |
| 15 | highfive_invite | 6.7 | 31.5 | 21.7 | 38.9 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 16 | shrug_idk | 21.4 | 29.4 | 40.0 | 54.1 | 90 | 90 | 180 | 90,0,180 | ✗ |
| 17 | flex_bicep | 27.0 | 36.4 | 20.0 | 49.5 | 90 | 0 | 120 | 90,90,180 | ✗ |
| 18 | air_quotes | 41.7 | 49.4 | 0.0 | 64.7 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 19 | arms_open_welcome | 80.2 | 45.1 | 15.0 | 93.3 | 90 | 90 | 90 | 0,0,90 | ✗ |
| 20 | hands_up_surrender | 16.7 | 27.4 | 15.0 | 35.4 | 0 | 90 | 90 | 0,0,90 | ✗ |
| 21 | stretch_arms_up | 0.0 | 0.0 | 10.0 | 10.0 | 0 | 90 | 90 | 0,0,90 | n/a |
| 22 | raise_fists_victory | 11.7 | 22.4 | 0.0 | 25.3 | 0 | 90 | 90 | 0,0,90 | ✗ |
| 23 | cross_arms_defensive | 14.4 | 30.0 | 10.0 | 34.7 | 0 | 90 | 180 | 180,0,180 | n/a |
| 24 | palms_together_thank_you | 8.5 | 16.5 | 21.7 | 28.6 | 90 | 0 | 180 | 90,0,180 | ✗ |
| 25 | facepalm | 58.2 | 22.5 | 20.0 | 65.5 | 0 | 90 | 90 | 0,90,180 | ✗ |
| 26 | depict_tree | 18.5 | 4.9 | 11.2 | 22.2 | 0 | 90 | 180 | 180,0,180 | n/a |
| 27 | depict_big_bowl | 50.2 | 42.6 | 15.0 | 67.6 | 0 | 90 | 0 | 0,0,0 | ✗ |
| 28 | depict_lion | 33.0 | — | 35.0 | 48.1 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 29 | depict_small_object | 18.0 | — | 10.0 | 20.6 | 90 | 90 | 120 | 0,90,90 | ✗ |
| 30 | depict_flying_bird | 54.2 | 61.3 | 15.0 | 83.2 | 0 | 90 | 90 | 0,90,180 | ✓ |
| 31 | self_scratch_head_confused | 11.5 | 33.5 | 11.0 | 37.1 | 0 | 90 | 90 | 0,90,180 | ✗ |
| 32 | pat_stomach | 38.0 | 29.3 | 5.0 | 48.2 | 90 | 90 | 90 | 90,0,0 | ✗ |
| 33 | fan_face_hot | 16.0 | 20.3 | 22.0 | 33.9 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 34 | tap_temple_think | 11.5 | 33.5 | 12.0 | 37.4 | 0 | 90 | 90 | 0,90,180 | ✓ |
| 35 | circle_temple_crazy | 11.5 | 33.5 | 11.0 | 37.1 | 0 | 90 | 90 | 0,90,180 | ✗ |
| 36 | lean_forward_interest | 8.0 | 24.3 | 5.0 | 26.0 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 37 | cover_mouth_gasp | 63.2 | 2.5 | 5.0 | 63.5 | 0 | 90 | 90 | 180,90,0 | ✗ |
| 38 | cheers_toast | 27.5 | 16.3 | 20.0 | 37.7 | 90 | 90 | 120 | 90,90,0 | ✗ |
| 39 | highfive_imitation | 23.0 | 24.3 | 25.0 | 41.7 | 90 | 90 | 120 | 0,90,90 | ✗ |
| 40 | fistbump_invite | 41.0 | 29.3 | 8.0 | 51.0 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 41 | pat_self_back | 27.7 | 49.4 | 12.0 | 57.9 | 0 | 180 | 180 | 180,180,0 | ✗ |
| 42 | throat_cut_stop | 18.1 | 6.8 | 25.0 | 31.6 | 0 | 90 | 180 | 180,0,180 | ✓ |
| 43 | visor_search | 21.3 | 25.7 | 5.0 | 33.8 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 44 | yield_turn | 31.7 | 29.4 | 8.0 | 44.0 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 45 | request_turn | 13.0 | 24.3 | 20.0 | 34.0 | 90 | 0 | 90 | 0,90,0 | ✗ |
| 46 | guiding_arm | 21.7 | 25.0 | 5.0 | 33.5 | 0 | 90 | 90 | 180,90,0 | ✓ |
| 47 | protective_arm_out | 28.2 | 13.1 | 0.0 | 31.1 | 90 | 0 | 90 | 90,90,90 | ✓ |
| 48 | slow_down_request_palm_down | 16.7 | 39.4 | 5.0 | 43.1 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 49 | firm_accept_forward_reach | 18.0 | 24.3 | 5.0 | 30.6 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 50 | shh_gesture | 28.3 | 6.1 | 14.0 | 32.2 | 0 | 0 | 0 | 0,0,0 | n/a |
| 51 | scale_expand | 35.2 | 15.1 | 15.0 | 41.2 | 0 | 90 | 90 | 180,90,0 | ✗ |
| 52 | hesitant_accept_half_reach | 3.0 | 24.3 | 5.0 | 25.0 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 53 | cancel_previous_offer_retract | 17.5 | 12.7 | 30.0 | 37.0 | 180 | 90 | 180 | 90,90,90 | ✗ |
| 54 | conditional_accept_pause_then_reach | 18.0 | 24.3 | 5.0 | 30.6 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 55 | final_decision_down_press | 33.7 | 39.4 | 18.0 | 54.9 | 90 | 90 | 120 | 180,90,90 | ✗ |
| 56 | resume_speaking_pull | 8.5 | 36.5 | 15.0 | 40.4 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 57 | emphasize_point_downbeat | 18.7 | 38.2 | 42.0 | 59.8 | 90 | 180 | 180 | 180,90,0 | ✗ |
| 58 | prepare_action_raise_hold | 16.7 | 39.4 | 10.0 | 44.0 | 0 | 180 | 180 | 0,0,180 | ✗ |
| 59 | exploration_reach | 18.0 | 24.3 | 5.0 | 30.6 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 60 | uncertainty_wave | 27.5 | 21.3 | 10.0 | 36.2 | 90 | 90 | 120 | 180,90,90 | ✓ |
| 61 | clarification_request_palm_tilt | 21.7 | 39.4 | 15.0 | 47.4 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 62 | speed_up_circular_cue | 1.5 | 36.5 | 15.0 | 39.5 | 90 | 180 | 120 | 90,90,180 | ✗ |
| 63 | polite_decline_small_push | 40.2 | 45.1 | 15.0 | 62.3 | 90 | 0 | 90 | 90,90,90 | ✗ |
| 64 | unsure_response_hand_hover_stop | 13.0 | 24.3 | 5.0 | 28.0 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 65 | continue_on_forward_roll | 13.0 | 24.3 | 5.0 | 28.0 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 66 | start_action_forward_snap | 35.0 | 29.3 | 8.0 | 46.3 | 90 | 90 | 120 | 0,90,90 | ✗ |
| 68 | hesitation_pause_hold | 8.0 | 24.3 | 5.0 | 26.0 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 69 | interrupt_hard | 11.3 | 25.7 | 35.0 | 44.9 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 70 | interrupt_soft | 36.6 | 43.0 | 0.0 | 56.5 | 0 | 90 | 0 | 0,0,0 | ✗ |
| 71 | self_hug | 23.5 | 16.5 | 15.0 | 32.4 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 72 | rub_arms_cold | 8.5 | 16.5 | 15.0 | 23.9 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 73 | dominance_high_open_hold | 26.3 | 75.7 | 0.0 | 80.2 | 0 | 90 | 90 | 0,0,90 | n/a |
| 74 | contain_space | 4.4 | 65.0 | 5.0 | 65.3 | 90 | 90 | 120 | 90,90,0 | ✗ |
| 75 | expand_space | 40.2 | 10.1 | 15.0 | 44.1 | 90 | 90 | 120 | 90,90,180 | ✓ |
| 76 | compress_space | 4.4 | 65.0 | 5.0 | 65.3 | 90 | 90 | 120 | 90,90,0 | ✗ |
| 77 | balance_scale | 31.6 | 73.0 | 5.0 | 79.7 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 78 | push_pull_dual | 13.0 | 24.3 | 5.0 | 28.0 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 79 | abandon_action_drop_turn | 26.9 | 22.7 | 15.0 | 38.3 | 90 | 90 | 120 | 0,90,90 | ✗ |
| 80 | escorting_gesture | 10.7 | 14.4 | 5.0 | 18.7 | 90 | 90 | 120 | 90,90,180 | ✗ |
| 81 | wait_for_me_backward_palm | 12.3 | 37.4 | 8.0 | 40.2 | 90 | 90 | 120 | 180,90,90 | ✗ |
| 82 | prevent_fall_support_reach | 48.0 | 29.3 | 2.0 | 56.2 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 83 | frustration_down_throw | 11.7 | 29.4 | 10.0 | 33.2 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 84 | submission_lowered_open_palm | 4.4 | 60.0 | 10.0 | 61.0 | 90 | 90 | 90 | 0,0,90 | ✗ |
| 85 | apology_forward_lowered_hold | 16.7 | 39.4 | 10.0 | 44.0 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 86 | chin_flick_dismiss | 8.5 | 46.5 | 2.0 | 47.3 | 90 | 180 | 180 | 90,0,180 | ✗ |
| 87 | cover_ears_too_loud | 13.1 | 1.8 | 6.0 | 14.5 | 90 | 0 | 180 | 90,0,180 | ✗ |
| 88 | disagreement_cross_then_release | 23.5 | 16.5 | 15.0 | 32.4 | 0 | 90 | 90 | 0,90,180 | ✗ |
| 89 | thumbs_up_approval | 46.2 | 32.8 | 18.0 | 59.5 | 90 | 0 | 90 | 0,90,0 | n/a |
| 90 | clap_applause | 20.5 | 7.5 | 5.0 | 22.4 | 90 | 0 | 180 | 90,0,180 | ✓ |
| 91 | hand_over_heart_sincerity | 8.5 | 34.5 | 12.0 | 37.5 | 0 | 90 | 90 | 0,90,180 | ✗ |

## Notes

- Location units are body-relative Motion Code percentages (0–100).
- Unspecified LLM xyz axes are filled from the G1 feasible-bank pose the renderer selects for dir|grip|elbow (render-effective EE %; no 50 penalty).
- Location match: |Δaxis| ≤ 5.0.
- Direction: `dir` → reach unit vector; `grip` palm token → palm normal; RPY from the reach–palm frame (geodesic = rotation angle between GT and LLM frames). Reach/palm angles are acos ∈ [0, 180°].
- Movement binary: GT (axis, sign) set ⊆ LLM set; line paths share the same (axis, sign) keys as joint moves; arcs stay as (arc, plane).
