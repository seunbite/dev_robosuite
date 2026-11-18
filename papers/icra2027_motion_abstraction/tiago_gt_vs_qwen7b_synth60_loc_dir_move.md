# tiago GT vs LLM — continuous location/direction, binary movement

Pred: `data/results/motion_configs/embodiment_motion_code_qwen_qwen7b_humanlike90/embodiment_motion_code_all.json`
Platform: `tiago` · N = 60 cues.

## Table metrics (loc = mean |Δ| body-% ↓; rch/plm/mov = % ↑)

| Reach↑ | Palm↑ | Locx↓ | Locy↓ | Locz↓ | Movement↑ |
|---:|---:|---:|---:|---:|---:|
| 48.3 | 51.7 | 12.2 | 26.0 | 8.6 | 12.3 |

_Tiago Palm: 0° and 180° both count as match._

## Summary

| Metric | Value |
|---|---:|
| Location cues with GT xyz | 60 |
| Location mean L2 (body %) | 33.0 |
| Location median L2 | 34.3 |
| Location mean per-axis |Δ| | 15.6 |
| Direction mean reach ∠ (°) | 48.0 (n=60) |
| Direction mean palm ∠ (°) | 49.5 (n=60) |
| Direction mean RPY geodesic (°) | 71.0 (n=60) |
| Movement binary acc | 7/57 (12%) |

## Per cue

| # | Cue | Δx | Δy | Δz | L2 | reach° | palm° | geo° | Δr,Δp,Δy | move |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 31 | self_scratch_head_confused | 19.4 | 4.9 | 38.0 | 42.9 | 90 | 90 | 90 | 90,0,0 | ✗ |
| 32 | pat_stomach | 0.4 | 34.9 | 4.0 | 35.1 | 90 | 90 | 90 | 90,0,0 | ✗ |
| 33 | fan_face_hot | 19.4 | 4.9 | 30.0 | 36.1 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 34 | tap_temple_think | 19.4 | 4.9 | 38.0 | 42.9 | 90 | 90 | 90 | 90,0,0 | ✓ |
| 35 | circle_temple_crazy | 19.4 | 4.9 | 38.0 | 42.9 | 90 | 90 | 90 | 90,0,0 | ✗ |
| 36 | lean_forward_interest | 19.4 | 14.9 | 2.0 | 24.5 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 37 | cover_mouth_gasp | 21.1 | 39.6 | 8.0 | 45.6 | 90 | 90 | 180 | 90,0,180 | ✗ |
| 38 | cheers_toast | 16.1 | 45.0 | 0.0 | 47.8 | 90 | 90 | 120 | 90,90,0 | ✗ |
| 39 | highfive_imitation | 10.6 | 14.9 | 17.0 | 25.0 | 90 | 90 | 120 | 0,90,90 | ✗ |
| 40 | fistbump_invite | 10.6 | 19.9 | 2.0 | 22.6 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 41 | pat_self_back | 10.4 | 8.1 | 18.0 | 22.3 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 42 | throat_cut_stop | 7.3 | 39.4 | 1.0 | 40.1 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 43 | visor_search | 21.1 | 42.5 | 0.0 | 47.4 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 44 | yield_turn | 21.1 | 32.5 | 5.0 | 39.1 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 45 | request_turn | 14.4 | 14.9 | 16.0 | 26.2 | 90 | 0 | 90 | 0,90,0 | ✗ |
| 46 | guiding_arm | 11.1 | 0.1 | 15.0 | 18.7 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 47 | protective_arm_out | 1.1 | 15.0 | 2.0 | 15.2 | 90 | 0 | 90 | 90,90,90 | ✗ |
| 48 | slow_down_request_palm_down | 11.1 | 37.5 | 5.0 | 39.4 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 49 | firm_accept_forward_reach | 3.6 | 14.9 | 5.0 | 16.1 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 50 | shh_gesture | 33.1 | 42.6 | 10.0 | 54.9 | 0 | 0 | 0 | 0,0,0 | n/a |
| 51 | scale_expand | 6.1 | 18.0 | 5.0 | 19.7 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 52 | hesitant_accept_half_reach | 16.4 | 14.9 | 5.0 | 22.7 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 53 | cancel_previous_offer_retract | 31.4 | 31.9 | 0.0 | 44.8 | 180 | 90 | 180 | 90,90,90 | ✗ |
| 54 | conditional_accept_pause_then_reach | 0.6 | 14.9 | 5.0 | 15.7 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 55 | final_decision_down_press | 21.1 | 37.5 | 8.0 | 43.7 | 90 | 180 | 180 | 180,90,0 | ✗ |
| 56 | resume_speaking_pull | 7.3 | 39.4 | 5.0 | 40.4 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 57 | emphasize_point_downbeat | 21.1 | 31.5 | 5.0 | 38.2 | 90 | 180 | 180 | 180,90,0 | ✗ |
| 58 | prepare_action_raise_hold | 6.1 | 37.5 | 8.0 | 38.8 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 59 | exploration_reach | 5.6 | 14.9 | 5.0 | 16.7 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 60 | uncertainty_wave | 16.1 | 40.0 | 5.0 | 43.4 | 90 | 0 | 90 | 0,0,90 | ✓ |
| 61 | clarification_request_palm_tilt | 16.1 | 37.5 | 5.0 | 41.1 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 62 | speed_up_circular_cue | 9.4 | 14.9 | 5.0 | 18.3 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 63 | polite_decline_small_push | 16.1 | 50.0 | 5.0 | 52.8 | 90 | 0 | 90 | 90,90,90 | ✗ |
| 64 | unsure_response_hand_hover_stop | 12.4 | 14.9 | 5.0 | 20.0 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 65 | continue_on_forward_roll | 4.4 | 14.9 | 5.0 | 16.3 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 66 | start_action_forward_snap | 3.6 | 14.9 | 5.0 | 16.1 | 90 | 90 | 120 | 0,90,90 | ✗ |
| 68 | hesitation_pause_hold | 16.4 | 14.9 | 5.0 | 22.7 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 69 | interrupt_hard | 14.4 | 14.9 | 5.0 | 21.3 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 70 | interrupt_soft | 9.1 | 37.5 | 2.0 | 38.6 | 0 | 90 | 90 | 0,0,90 | ✗ |
| 71 | self_hug | 7.3 | 16.4 | 2.0 | 18.1 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 72 | rub_arms_cold | 2.3 | 14.4 | 2.0 | 14.7 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 73 | dominance_high_open_hold | 11.7 | 57.0 | 28.0 | 64.6 | 0 | 90 | 90 | 0,0,90 | n/a |
| 74 | contain_space | 12.3 | 59.4 | 5.0 | 60.9 | 90 | 90 | 120 | 90,90,0 | ✗ |
| 75 | expand_space | 6.1 | 12.0 | 5.0 | 14.4 | 90 | 90 | 180 | 90,0,180 | ✓ |
| 76 | compress_space | 12.3 | 59.4 | 5.0 | 60.9 | 90 | 90 | 120 | 90,90,0 | ✗ |
| 77 | balance_scale | 4.0 | 44.9 | 5.0 | 45.3 | 90 | 90 | 90 | 90,0,0 | ✗ |
| 78 | push_pull_dual | 4.4 | 14.9 | 5.0 | 16.3 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 79 | abandon_action_drop_turn | 3.9 | 37.5 | 12.0 | 39.5 | 0 | 90 | 90 | 0,0,90 | ✗ |
| 80 | escorting_gesture | 14.4 | 15.1 | 5.0 | 21.5 | 90 | 90 | 120 | 90,90,180 | ✗ |
| 81 | wait_for_me_backward_palm | 17.0 | 9.9 | 8.0 | 21.3 | 90 | 90 | 120 | 180,90,90 | ✗ |
| 82 | prevent_fall_support_reach | 15.6 | 14.9 | 0.0 | 21.6 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 83 | frustration_down_throw | 1.1 | 27.5 | 10.0 | 29.2 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 84 | submission_lowered_open_palm | 16.4 | 4.9 | 16.0 | 23.4 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 85 | apology_forward_lowered_hold | 11.1 | 37.5 | 16.0 | 42.2 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 86 | chin_flick_dismiss | 2.4 | 32.9 | 6.0 | 33.5 | 90 | 0 | 90 | 0,90,0 | ✗ |
| 87 | cover_ears_too_loud | 28.4 | 49.9 | 8.0 | 58.0 | 90 | 0 | 90 | 0,90,0 | ✗ |
| 88 | disagreement_cross_then_release | 9.4 | 44.9 | 2.0 | 45.9 | 90 | 90 | 90 | 90,0,0 | ✗ |
| 89 | thumbs_up_approval | 19.1 | 39.5 | 5.0 | 44.1 | 90 | 0 | 90 | 0,90,0 | n/a |
| 90 | clap_applause | 5.7 | 16.4 | 22.0 | 28.0 | 90 | 90 | 120 | 90,90,0 | ✓ |
| 91 | hand_over_heart_sincerity | 0.3 | 22.4 | 2.0 | 22.5 | 0 | 0 | 0 | 0,0,0 | ✗ |

## Notes

- Location units are body-relative Motion Code percentages (0–100).
- Unspecified LLM xyz axes are filled from the G1 feasible-bank pose the renderer selects for dir|grip|elbow (render-effective EE %; no 50 penalty).
- Location match: |Δaxis| ≤ 5.0.
- Direction: `dir` → reach unit vector; `grip` palm token → palm normal; RPY from the reach–palm frame (geodesic = rotation angle between GT and LLM frames). Reach/palm angles are acos ∈ [0, 180°].
- Movement binary: GT (axis, sign) set ⊆ LLM set; line paths share the same (axis, sign) keys as joint moves; arcs stay as (arc, plane).
