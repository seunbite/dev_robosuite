# humanoid GT vs LLM — continuous location/direction, binary movement

Pred: `data/results/motion_configs/embodiment_motion_code_gemini_humanlike90/embodiment_motion_code_all.json`
Platform: `humanoid` · N = 90 cues.

## Table metrics (loc = mean |Δ| body-% ↓; rch/plm/mov = % ↑)

| Reach↑ | Palm↑ | Locx↓ | Locy↓ | Locz↓ | Movement↑ |
|---:|---:|---:|---:|---:|---:|
| 47.2 | 48.6 | 18.4 | 19.2 | 11.2 | 50.6 |

_Tiago Palm: 0° and 180° both count as match._

## Summary

| Metric | Value |
|---|---:|
| Location cues with GT xyz | 85 |
| Location mean L2 (body %) | 32.5 |
| Location median L2 | 29.6 |
| Location mean per-axis |Δ| | 16.1 |
| Direction mean reach ∠ (°) | 57.5 (n=72) |
| Direction mean palm ∠ (°) | 51.2 (n=72) |
| Direction mean RPY geodesic (°) | 98.8 (n=72) |
| Movement binary acc | 41/81 (51%) |

## Per cue

| # | Cue | Δx | Δy | Δz | L2 | reach° | palm° | geo° | Δr,Δp,Δy | move |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | nod_yes | — | — | — | — | — | — | — | — | ✓ |
| 2 | shake_head_no | — | — | — | — | — | — | — | — | ✓ |
| 3 | draw_circle_repeat | 21.4 | 26.4 | 20.0 | 39.5 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 4 | gaze_shift_avoid | — | — | — | — | — | — | — | — | ✗ |
| 5 | fist_pump | 28.0 | 34.3 | 0.0 | 44.2 | 90 | 0 | 90 | 0,90,0 | ✓ |
| 6 | tilt_head_curious | — | — | — | — | — | — | — | — | ✗ |
| 7 | cover_yawn_sleepy | 10.0 | 25.0 | 15.0 | 30.8 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 8 | rub_eye_tired | 15.0 | 10.0 | 0.0 | 18.0 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 9 | point_you | 30.0 | 56.2 | 20.0 | 66.8 | 0 | 0 | 0 | 0,0,0 | n/a |
| 10 | point_self | 10.0 | 5.0 | 10.0 | 15.0 | 180 | 90 | 180 | 90,90,90 | n/a |
| 11 | beckon_come_here | 7.7 | 14.2 | 15.0 | 22.0 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 12 | stop_palm_out | 6.5 | 43.3 | 20.0 | 48.1 | 90 | 0 | 180 | 90,0,180 | n/a |
| 13 | salute | 26.3 | 20.0 | 10.0 | 34.5 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 14 | raise_index_one_moment | 22.0 | 1.4 | 15.0 | 26.6 | 90 | 0 | 120 | 90,90,0 | ✓ |
| 15 | highfive_invite | 44.7 | 33.1 | 26.7 | 61.7 | 90 | 0 | 180 | 90,0,180 | ✗ |
| 16 | shrug_idk | 0.0 | 7.6 | 15.0 | 16.8 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 17 | flex_bicep | 8.1 | 0.0 | 0.0 | 8.1 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 18 | air_quotes | 16.7 | 5.0 | 15.0 | 23.0 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 19 | arms_open_welcome | 61.7 | 50.0 | 5.0 | 79.6 | 90 | 90 | 120 | 180,90,90 | ✗ |
| 20 | hands_up_surrender | 3.7 | 30.7 | 20.0 | 36.8 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 21 | stretch_arms_up | 0.0 | 0.0 | 10.0 | 10.0 | 0 | 90 | 90 | 0,0,90 | n/a |
| 22 | raise_fists_victory | 25.0 | 22.4 | 0.0 | 33.6 | 0 | 90 | 90 | 0,0,90 | ✗ |
| 23 | cross_arms_defensive | 50.2 | 65.1 | 10.0 | 82.8 | 180 | 90 | 180 | 0,90,0 | n/a |
| 24 | palms_together_thank_you | 12.8 | 5.0 | 21.7 | 25.7 | 90 | 0 | 180 | 90,0,180 | ✓ |
| 25 | facepalm | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 26 | depict_tree | 12.0 | 16.7 | 8.8 | 22.4 | 90 | 90 | 180 | 180,0,90 | n/a |
| 27 | depict_big_bowl | 10.3 | 27.4 | 5.0 | 29.8 | 180 | 0 | 180 | 0,0,180 | ✓ |
| 28 | depict_lion | 24.5 | — | 5.0 | 25.0 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 29 | depict_small_object | 12.0 | — | 0.0 | 12.0 | 90 | 90 | 120 | 0,90,90 | ✗ |
| 30 | depict_flying_bird | 0.0 | 0.0 | 5.0 | 5.0 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 31 | self_scratch_head_confused | 10.0 | 22.0 | 9.0 | 25.8 | 0 | 90 | 90 | 0,90,180 | ✓ |
| 32 | pat_stomach | 15.0 | 20.0 | 5.0 | 25.5 | 90 | 90 | 120 | 180,90,90 | ✓ |
| 33 | fan_face_hot | 5.7 | 14.0 | 8.0 | 17.1 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 34 | tap_temple_think | 8.1 | 17.0 | 8.0 | 20.4 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 35 | circle_temple_crazy | 10.0 | 12.0 | 9.0 | 18.0 | 180 | 180 | 180 | 180,0,0 | ✓ |
| 36 | lean_forward_interest | — | — | — | — | — | — | — | — | ✓ |
| 37 | cover_mouth_gasp | 10.0 | 25.0 | 15.0 | 30.8 | 0 | 90 | 90 | 180,90,0 | ✗ |
| 38 | cheers_toast | 27.0 | 20.0 | 15.0 | 36.8 | 90 | 0 | 90 | 90,0,0 | ✗ |
| 39 | highfive_imitation | 48.4 | 7.4 | 0.0 | 48.9 | 90 | 0 | 180 | 90,0,180 | ✓ |
| 40 | fistbump_invite | 18.0 | 3.1 | 8.0 | 19.9 | 0 | 90 | 180 | 180,0,180 | ✓ |
| 41 | pat_self_back | 5.0 | 25.0 | 2.0 | 25.6 | 90 | 90 | 120 | 90,90,0 | ✗ |
| 42 | throat_cut_stop | 2.5 | 15.0 | 15.0 | 21.4 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 43 | visor_search | 20.0 | 25.0 | 5.0 | 32.4 | 180 | 0 | 180 | 0,0,180 | ✓ |
| 44 | yield_turn | 5.0 | 18.1 | 7.0 | 20.1 | 0 | 180 | 180 | 180,0,180 | ✗ |
| 45 | request_turn | 13.0 | 24.3 | 10.0 | 29.3 | 90 | 0 | 90 | 0,90,0 | ✓ |
| 46 | guiding_arm | 0.3 | 50.0 | 5.0 | 50.3 | 180 | 0 | 180 | 0,0,180 | ✗ |
| 47 | protective_arm_out | 11.7 | 32.0 | 20.0 | 39.5 | 90 | 0 | 90 | 90,90,90 | ✓ |
| 48 | slow_down_request_palm_down | 33.4 | 7.4 | 5.0 | 34.5 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 49 | firm_accept_forward_reach | 5.0 | 25.7 | 5.0 | 26.6 | 0 | 90 | 90 | 90,90,90 | ✓ |
| 50 | shh_gesture | 28.0 | 23.0 | 16.0 | 39.6 | 90 | 90 | 120 | 90,90,0 | n/a |
| 51 | scale_expand | 23.6 | 55.0 | 5.0 | 60.1 | 180 | 90 | 180 | 180,90,180 | ✓ |
| 52 | hesitant_accept_half_reach | 28.7 | 5.4 | 5.0 | 29.6 | 0 | 90 | 90 | 90,90,90 | ✓ |
| 53 | cancel_previous_offer_retract | 50.0 | 23.9 | 0.0 | 55.4 | 0 | 180 | 180 | 180,0,180 | ✓ |
| 54 | conditional_accept_pause_then_reach | 5.0 | 12.7 | 5.0 | 14.5 | 180 | 180 | 180 | 180,0,0 | ✓ |
| 55 | final_decision_down_press | 50.4 | 7.4 | 38.0 | 63.5 | 0 | 90 | 90 | 90,90,90 | ✓ |
| 56 | resume_speaking_pull | 5.0 | 8.1 | 5.0 | 10.8 | 90 | 90 | 90 | 90,0,0 | ✓ |
| 57 | emphasize_point_downbeat | 5.0 | 16.4 | 22.0 | 27.9 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 58 | prepare_action_raise_hold | 33.4 | 7.4 | 20.0 | 39.6 | 90 | 90 | 120 | 90,0,90 | ✓ |
| 59 | exploration_reach | 5.0 | 25.3 | 5.0 | 26.2 | 0 | 90 | 90 | 90,90,90 | ✓ |
| 60 | uncertainty_wave | 5.3 | 7.6 | 10.0 | 13.6 | 90 | 0 | 90 | 0,0,90 | ✓ |
| 61 | clarification_request_palm_tilt | 21.7 | 50.0 | 5.0 | 54.7 | 90 | 90 | 90 | 0,0,90 | ✗ |
| 62 | speed_up_circular_cue | 17.0 | 16.4 | 5.0 | 24.1 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 63 | polite_decline_small_push | 0.0 | 25.0 | 5.0 | 25.5 | 90 | 0 | 180 | 90,0,180 | ✓ |
| 64 | unsure_response_hand_hover_stop | 38.7 | 5.4 | 5.0 | 39.4 | 0 | 90 | 90 | 90,90,90 | ✓ |
| 65 | continue_on_forward_roll | 38.4 | 7.4 | 5.0 | 39.4 | 0 | 90 | 90 | 90,90,90 | ✓ |
| 66 | start_action_forward_snap | 22.0 | 21.4 | 8.0 | 31.7 | 90 | 90 | 180 | 90,0,180 | ✓ |
| 68 | hesitation_pause_hold | 45.0 | 23.9 | 5.0 | 51.2 | 0 | 90 | 90 | 90,90,90 | ✓ |
| 69 | interrupt_hard | 8.3 | 25.0 | 15.0 | 30.3 | 90 | 0 | 90 | 0,0,90 | ✗ |
| 70 | interrupt_soft | 33.4 | 7.4 | 5.0 | 34.5 | 90 | 0 | 180 | 90,0,180 | ✓ |
| 71 | self_hug | 27.8 | 5.0 | 15.0 | 32.0 | 0 | 90 | 90 | 0,90,180 | ✗ |
| 72 | rub_arms_cold | 7.5 | 13.3 | 5.0 | 16.1 | 0 | 90 | 90 | 180,90,0 | ✗ |
| 73 | dominance_high_open_hold | 9.7 | 33.3 | 5.0 | 35.0 | 90 | 0 | 90 | 90,90,90 | n/a |
| 74 | contain_space | 21.7 | 15.0 | 5.0 | 26.9 | 90 | 0 | 90 | 90,0,0 | ✗ |
| 75 | expand_space | 46.2 | 55.0 | 5.0 | 72.0 | 90 | 90 | 90 | 90,0,0 | ✗ |
| 76 | compress_space | 21.7 | 10.0 | 5.0 | 24.4 | 90 | 0 | 90 | 90,0,0 | ✗ |
| 77 | balance_scale | 9.7 | 27.6 | 5.0 | 29.6 | 90 | 0 | 90 | 0,0,90 | ✗ |
| 78 | push_pull_dual | 13.0 | 24.3 | 5.0 | 28.0 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 79 | abandon_action_drop_turn | — | — | 5.0 | 5.0 | — | — | — | — | ✗ |
| 80 | escorting_gesture | 17.0 | 33.1 | 22.5 | 43.5 | — | — | — | — | ✗ |
| 81 | wait_for_me_backward_palm | 62.0 | 10.1 | 35.5 | 72.1 | — | — | — | — | ✗ |
| 82 | prevent_fall_support_reach | 18.0 | 3.1 | 19.5 | 26.8 | — | — | — | — | ✗ |
| 83 | frustration_down_throw | 27.0 | 18.1 | 7.5 | 33.4 | — | — | — | — | ✗ |
| 84 | submission_lowered_open_palm | 17.0 | 18.1 | 7.5 | 25.9 | — | — | — | — | ✗ |
| 85 | apology_forward_lowered_hold | 22.0 | 8.1 | 7.5 | 24.6 | — | — | — | — | ✗ |
| 86 | chin_flick_dismiss | 7.0 | 18.1 | 35.5 | 40.5 | — | — | — | — | ✗ |
| 87 | cover_ears_too_loud | 22.0 | 3.1 | 41.5 | 47.1 | — | — | — | — | ✗ |
| 88 | disagreement_cross_then_release | 8.0 | 11.9 | 22.5 | 26.7 | — | — | — | — | ✗ |
| 89 | thumbs_up_approval | 7.0 | 8.1 | 29.5 | 31.4 | — | — | — | — | n/a |
| 90 | clap_applause | 5.0 | 20.9 | 32.5 | 39.0 | — | — | — | — | ✗ |
| 91 | hand_over_heart_sincerity | 7.0 | 6.1 | 25.5 | 27.2 | — | — | — | — | ✗ |

## Notes

- Location units are body-relative Motion Code percentages (0–100).
- Unspecified LLM xyz axes are filled from the G1 feasible-bank pose the renderer selects for dir|grip|elbow (render-effective EE %; no 50 penalty).
- Location match: |Δaxis| ≤ 5.0.
- Direction: `dir` → reach unit vector; `grip` palm token → palm normal; RPY from the reach–palm frame (geodesic = rotation angle between GT and LLM frames). Reach/palm angles are acos ∈ [0, 180°].
- Movement binary: GT (axis, sign) set ⊆ LLM set; line paths share the same (axis, sign) keys as joint moves; arcs stay as (arc, plane).
