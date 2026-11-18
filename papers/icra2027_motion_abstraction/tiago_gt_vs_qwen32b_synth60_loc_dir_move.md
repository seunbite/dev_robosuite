# tiago GT vs LLM — continuous location/direction, binary movement

Pred: `data/results/motion_configs/embodiment_motion_code_qwen_qwen32b_humanlike90/embodiment_motion_code_all.json`
Platform: `tiago` · N = 60 cues.

## Table metrics (loc = mean |Δ| body-% ↓; rch/plm/mov = % ↑)

| Reach↑ | Palm↑ | Locx↓ | Locy↓ | Locz↓ | Movement↑ |
|---:|---:|---:|---:|---:|---:|
| 47.4 | 43.9 | 17.7 | 23.9 | 8.2 | 43.9 |

_Tiago Palm: 0° and 180° both count as match._

## Summary

| Metric | Value |
|---|---:|
| Location cues with GT xyz | 58 |
| Location mean L2 (body %) | 34.6 |
| Location median L2 | 31.6 |
| Location mean per-axis |Δ| | 16.4 |
| Direction mean reach ∠ (°) | 52.1 (n=57) |
| Direction mean palm ∠ (°) | 63.2 (n=57) |
| Direction mean RPY geodesic (°) | 85.3 (n=57) |
| Movement binary acc | 25/57 (44%) |

## Per cue

| # | Cue | Δx | Δy | Δz | L2 | reach° | palm° | geo° | Δr,Δp,Δy | move |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 31 | self_scratch_head_confused | 6.1 | 7.6 | 28.0 | 29.7 | 90 | 90 | 120 | 90,90,0 | ✓ |
| 32 | pat_stomach | 30.7 | 7.4 | 4.0 | 31.8 | 0 | 90 | 90 | 0,90,180 | ✗ |
| 33 | fan_face_hot | 6.1 | 40.0 | 25.0 | 47.6 | 90 | 90 | 120 | 90,90,180 | ✗ |
| 34 | tap_temple_think | 6.1 | 7.6 | 28.0 | 29.7 | 90 | 90 | 120 | 90,90,0 | ✗ |
| 35 | circle_temple_crazy | 19.4 | 4.9 | 38.0 | 42.9 | 90 | 90 | 90 | 90,0,0 | ✗ |
| 36 | lean_forward_interest | — | — | — | — | — | — | — | — | ✗ |
| 37 | cover_mouth_gasp | 21.1 | 39.6 | 8.0 | 45.6 | 90 | 90 | 180 | 90,0,180 | ✗ |
| 38 | cheers_toast | 16.1 | 25.2 | 0.0 | 29.9 | 90 | 90 | 120 | 90,90,0 | ✗ |
| 39 | highfive_imitation | 36.1 | 17.6 | 3.0 | 40.3 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 40 | fistbump_invite | 10.0 | 7.0 | 8.0 | 14.6 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 41 | pat_self_back | 10.4 | 8.1 | 13.0 | 18.5 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 42 | throat_cut_stop | 35.0 | 7.0 | 11.0 | 37.4 | 90 | 180 | 180 | 180,0,90 | ✓ |
| 43 | visor_search | 21.1 | 22.6 | 5.0 | 31.3 | 90 | 90 | 120 | 90,0,90 | ✓ |
| 44 | yield_turn | 31.0 | 9.9 | 25.0 | 41.1 | 180 | 90 | 180 | 90,90,90 | ✗ |
| 45 | request_turn | 0.0 | 14.9 | 11.0 | 18.5 | 90 | 0 | 90 | 0,90,0 | ✗ |
| 46 | guiding_arm | 11.1 | 20.0 | 5.0 | 23.4 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 47 | protective_arm_out | 1.1 | 15.0 | 3.0 | 15.3 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 48 | slow_down_request_palm_down | 11.1 | 50.0 | 5.0 | 51.5 | 90 | 0 | 90 | 0,0,90 | ✓ |
| 49 | firm_accept_forward_reach | 22.0 | 12.0 | 5.0 | 25.6 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 50 | shh_gesture | 33.1 | 42.6 | 0.0 | 54.0 | 0 | 0 | 0 | 0,0,0 | n/a |
| 51 | scale_expand | 6.1 | 18.0 | 0.0 | 19.0 | 0 | 90 | 90 | 0,90,180 | ✓ |
| 52 | hesitant_accept_half_reach | 2.0 | 4.9 | 5.0 | 7.3 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 53 | cancel_previous_offer_retract | 69.0 | 12.0 | 0.0 | 70.0 | 0 | 90 | 90 | 90,90,90 | ✓ |
| 54 | conditional_accept_pause_then_reach | 25.0 | 12.0 | 5.0 | 28.2 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 55 | final_decision_down_press | 21.1 | 37.5 | 2.0 | 43.1 | 90 | 180 | 180 | 180,90,0 | ✓ |
| 56 | resume_speaking_pull | 16.1 | 55.0 | 5.0 | 57.5 | 180 | 90 | 180 | 180,90,180 | ✓ |
| 57 | emphasize_point_downbeat | 21.1 | 31.5 | 5.0 | 38.2 | 90 | 180 | 180 | 180,90,0 | ✓ |
| 58 | prepare_action_raise_hold | 6.1 | 17.6 | 2.0 | 18.7 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 59 | exploration_reach | 20.0 | 12.0 | 5.0 | 23.9 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 60 | uncertainty_wave | 16.1 | 40.0 | 0.0 | 43.1 | 90 | 0 | 90 | 0,0,90 | ✓ |
| 61 | clarification_request_palm_tilt | 16.1 | 30.2 | 5.0 | 34.6 | 90 | 90 | 120 | 0,90,90 | ✓ |
| 62 | speed_up_circular_cue | 9.4 | 14.9 | 5.0 | 18.3 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 63 | polite_decline_small_push | 16.1 | 50.0 | 5.0 | 52.8 | 90 | 90 | 120 | 90,0,90 | ✓ |
| 64 | unsure_response_hand_hover_stop | 38.0 | 12.0 | 5.0 | 40.2 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 65 | continue_on_forward_roll | 30.0 | 12.0 | 5.0 | 32.7 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 66 | start_action_forward_snap | 22.0 | 12.0 | 5.0 | 25.6 | 90 | 90 | 120 | 0,90,90 | ✓ |
| 68 | hesitation_pause_hold | 42.0 | 12.0 | 5.0 | 44.0 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 69 | interrupt_hard | 11.1 | 50.0 | 0.0 | 51.2 | 90 | 0 | 90 | 0,0,90 | ✓ |
| 70 | interrupt_soft | — | — | 8.0 | 8.0 | — | — | — | — | ✗ |
| 71 | self_hug | 7.3 | 16.4 | 2.0 | 18.1 | 0 | 90 | 90 | 0,90,180 | ✗ |
| 72 | rub_arms_cold | 2.3 | 14.4 | 2.0 | 14.7 | 0 | 90 | 90 | 0,90,180 | ✗ |
| 73 | dominance_high_open_hold | 3.9 | 49.9 | 28.0 | 57.4 | 0 | 0 | 0 | 0,0,0 | n/a |
| 74 | contain_space | 12.3 | 59.4 | 5.0 | 60.9 | 90 | 90 | 120 | 90,90,0 | ✓ |
| 75 | expand_space | 6.1 | 12.0 | 0.0 | 13.5 | 90 | 90 | 180 | 90,0,180 | ✗ |
| 76 | compress_space | 12.3 | 59.4 | 5.0 | 60.9 | 90 | 90 | 120 | 90,90,0 | ✓ |
| 77 | balance_scale | 35.1 | 42.0 | 5.0 | 55.0 | 90 | 90 | 90 | 90,0,0 | ✓ |
| 78 | push_pull_dual | 30.0 | 12.0 | 5.0 | 32.7 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 79 | abandon_action_drop_turn | — | — | — | — | — | — | — | — | ✗ |
| 80 | escorting_gesture | 11.1 | 20.0 | 5.0 | 23.4 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 81 | wait_for_me_backward_palm | 42.6 | 17.0 | 18.0 | 49.3 | 90 | 90 | 90 | 90,0,0 | ✗ |
| 82 | prevent_fall_support_reach | 10.0 | 12.0 | 10.0 | 18.6 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 83 | frustration_down_throw | 1.1 | 27.5 | 30.0 | 40.7 | 180 | 180 | 180 | 180,0,0 | ✗ |
| 84 | submission_lowered_open_palm | 16.4 | 14.9 | 6.0 | 23.0 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 85 | apology_forward_lowered_hold | 14.4 | 4.9 | 6.0 | 16.4 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 86 | chin_flick_dismiss | 12.0 | 18.0 | 6.0 | 22.4 | 90 | 90 | 120 | 90,0,90 | ✓ |
| 87 | cover_ears_too_loud | 2.9 | 52.6 | 8.0 | 53.3 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 88 | disagreement_cross_then_release | 7.3 | 14.4 | 2.0 | 16.3 | 0 | 90 | 90 | 0,90,180 | ✓ |
| 89 | thumbs_up_approval | 19.1 | 19.6 | 15.0 | 31.2 | 90 | 0 | 90 | 0,90,0 | n/a |
| 90 | clap_applause | 29.1 | 78.0 | 17.0 | 85.0 | 90 | 90 | 180 | 90,0,180 | ✗ |
| 91 | hand_over_heart_sincerity | 17.0 | 22.0 | 2.0 | 27.9 | 90 | 90 | 90 | 90,0,0 | ✗ |

## Notes

- Location units are body-relative Motion Code percentages (0–100).
- Unspecified LLM xyz axes are filled from the G1 feasible-bank pose the renderer selects for dir|grip|elbow (render-effective EE %; no 50 penalty).
- Location match: |Δaxis| ≤ 5.0.
- Direction: `dir` → reach unit vector; `grip` palm token → palm normal; RPY from the reach–palm frame (geodesic = rotation angle between GT and LLM frames). Reach/palm angles are acos ∈ [0, 180°].
- Movement binary: GT (axis, sign) set ⊆ LLM set; line paths share the same (axis, sign) keys as joint moves; arcs stay as (arc, plane).
