# tiago GT vs LLM — continuous location/direction, binary movement

Pred: `data/results/motion_configs/embodiment_motion_code_gemini_humanlike90/embodiment_motion_code_all.json`
Platform: `tiago` · N = 60 cues.

## Table metrics (loc = mean |Δ| body-% ↓; rch/plm/mov = % ↑)

| Reach↑ | Palm↑ | Locx↓ | Locy↓ | Locz↓ | Movement↑ |
|---:|---:|---:|---:|---:|---:|
| 49.1 | 52.8 | 14.7 | 19.8 | 8.7 | 56.1 |

_Tiago Palm: 0° and 180° both count as match._

## Summary

| Metric | Value |
|---|---:|
| Location cues with GT xyz | 53 |
| Location mean L2 (body %) | 29.7 |
| Location median L2 | 26.0 |
| Location mean per-axis |Δ| | 14.4 |
| Direction mean reach ∠ (°) | 50.9 (n=53) |
| Direction mean palm ∠ (°) | 56.0 (n=53) |
| Direction mean RPY geodesic (°) | 74.2 (n=53) |
| Movement binary acc | 32/57 (56%) |

## Per cue

| # | Cue | Δx | Δy | Δz | L2 | reach° | palm° | geo° | Δr,Δp,Δy | move |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 31 | self_scratch_head_confused | 15.0 | 25.0 | 23.0 | 37.1 | 90 | 90 | 90 | 90,0,0 | ✓ |
| 32 | pat_stomach | 14.0 | 8.0 | 4.0 | 16.6 | 90 | 90 | 120 | 0,90,90 | ✓ |
| 33 | fan_face_hot | 10.0 | 20.0 | 25.0 | 33.5 | 0 | 90 | 90 | 90,90,90 | ✓ |
| 34 | tap_temple_think | 13.0 | 25.0 | 26.0 | 38.3 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 35 | circle_temple_crazy | 10.0 | 15.0 | 28.0 | 33.3 | 90 | 90 | 90 | 90,0,0 | ✗ |
| 36 | lean_forward_interest | — | — | — | — | — | — | — | — | ✗ |
| 37 | cover_mouth_gasp | 21.1 | 36.9 | 37.5 | 56.7 | 90 | 90 | 120 | 90,0,90 | ✓ |
| 38 | cheers_toast | 8.3 | 20.0 | 10.0 | 23.9 | 90 | 90 | 120 | 90,90,0 | ✗ |
| 39 | highfive_imitation | 36.1 | 17.6 | 8.0 | 41.0 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 40 | fistbump_invite | 25.0 | 19.9 | 2.0 | 32.0 | 0 | 90 | 90 | 90,90,90 | ✓ |
| 41 | pat_self_back | 12.0 | 10.0 | 13.0 | 20.3 | 90 | 90 | 120 | 90,90,180 | ✗ |
| 42 | throat_cut_stop | 16.1 | 0.0 | 11.0 | 19.5 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 43 | visor_search | 0.0 | 15.0 | 0.0 | 15.0 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 44 | yield_turn | 5.0 | 9.9 | 0.0 | 11.1 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 45 | request_turn | 0.0 | 14.9 | 11.0 | 18.5 | 90 | 0 | 90 | 0,90,0 | ✓ |
| 46 | guiding_arm | 12.3 | 74.4 | 0.0 | 75.4 | 180 | 180 | 180 | 180,0,0 | ✗ |
| 47 | protective_arm_out | 1.1 | 45.0 | 3.0 | 45.1 | 90 | 90 | 120 | 90,0,90 | ✓ |
| 48 | slow_down_request_palm_down | 14.4 | 14.9 | 5.0 | 21.3 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 49 | firm_accept_forward_reach | 18.0 | 14.9 | 5.0 | 23.9 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 50 | shh_gesture | 17.0 | 5.0 | 5.0 | 18.4 | 90 | 0 | 90 | 0,90,0 | n/a |
| 51 | scale_expand | 45.0 | 42.0 | 5.0 | 61.8 | 90 | 90 | 120 | 0,90,90 | ✓ |
| 52 | hesitant_accept_half_reach | 28.0 | 14.9 | 5.0 | 32.1 | 0 | 90 | 90 | 90,90,90 | ✓ |
| 53 | cancel_previous_offer_retract | — | — | — | — | — | — | — | — | ✗ |
| 54 | conditional_accept_pause_then_reach | 25.0 | 14.9 | 5.0 | 29.5 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 55 | final_decision_down_press | 10.0 | 17.6 | 13.0 | 24.1 | 0 | 90 | 90 | 90,90,90 | ✓ |
| 56 | resume_speaking_pull | 5.0 | 19.9 | 5.0 | 21.1 | 90 | 90 | 120 | 180,90,90 | ✓ |
| 57 | emphasize_point_downbeat | 30.0 | 18.0 | 5.0 | 35.4 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 58 | prepare_action_raise_hold | 19.4 | 4.9 | 28.0 | 34.4 | 90 | 0 | 90 | 0,90,0 | ✓ |
| 59 | exploration_reach | 25.0 | 4.9 | 5.0 | 26.0 | 0 | 90 | 90 | 90,90,90 | ✓ |
| 60 | uncertainty_wave | 9.4 | 4.9 | 5.0 | 11.7 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 61 | clarification_request_palm_tilt | 9.4 | 14.9 | 5.0 | 18.3 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 62 | speed_up_circular_cue | 19.4 | 10.0 | 5.0 | 22.4 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 63 | polite_decline_small_push | 5.0 | 14.9 | 5.0 | 16.5 | 90 | 90 | 90 | 90,0,0 | ✓ |
| 64 | unsure_response_hand_hover_stop | 12.4 | 14.9 | 0.0 | 19.4 | 0 | 90 | 90 | 90,90,90 | ✓ |
| 65 | continue_on_forward_roll | 10.0 | 14.9 | 5.0 | 18.6 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 66 | start_action_forward_snap | 28.0 | 14.9 | 0.0 | 31.7 | 90 | 90 | 120 | 0,90,90 | ✓ |
| 68 | hesitation_pause_hold | 16.4 | 14.9 | 5.0 | 22.7 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 69 | interrupt_hard | 11.1 | 30.1 | 20.0 | 37.8 | 90 | 0 | 90 | 0,0,90 | ✗ |
| 70 | interrupt_soft | 16.4 | 14.9 | 8.0 | 23.5 | 90 | 90 | 90 | 90,0,0 | ✓ |
| 71 | self_hug | 16.1 | 58.2 | 2.0 | 60.4 | 180 | 180 | 180 | 180,0,0 | ✗ |
| 72 | rub_arms_cold | 10.0 | 14.4 | 2.0 | 17.7 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 73 | dominance_high_open_hold | 3.9 | 49.9 | 13.0 | 51.8 | 0 | 0 | 0 | 0,0,0 | n/a |
| 74 | contain_space | 11.1 | 35.0 | 5.0 | 37.1 | 90 | 90 | 120 | 90,90,0 | ✗ |
| 75 | expand_space | 19.4 | 23.1 | 10.0 | 31.8 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 76 | compress_space | 11.1 | 35.0 | 0.0 | 36.7 | 90 | 90 | 180 | 90,0,180 | ✗ |
| 77 | balance_scale | 1.7 | 6.9 | 10.0 | 12.3 | 90 | 0 | 90 | 0,0,90 | ✗ |
| 78 | push_pull_dual | 10.0 | 17.6 | 5.0 | 20.9 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 79 | abandon_action_drop_turn | — | — | — | — | — | — | — | — | ✗ |
| 80 | escorting_gesture | 11.1 | 0.1 | 0.0 | 11.1 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 81 | wait_for_me_backward_palm | 33.0 | 17.0 | 18.0 | 41.3 | 90 | 90 | 90 | 90,0,0 | ✗ |
| 82 | prevent_fall_support_reach | 40.0 | 17.6 | 5.0 | 44.0 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 83 | frustration_down_throw | 1.1 | 7.6 | 25.0 | 26.2 | 180 | 180 | 180 | 180,0,0 | ✓ |
| 84 | submission_lowered_open_palm | 16.4 | 4.9 | 11.0 | 20.3 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 85 | apology_forward_lowered_hold | 11.1 | 37.5 | 4.0 | 39.3 | 90 | 90 | 120 | 90,0,90 | ✓ |
| 86 | chin_flick_dismiss | 7.0 | 2.0 | 6.0 | 9.4 | 90 | 180 | 180 | 180,90,0 | ✗ |
| 87 | cover_ears_too_loud | 2.9 | 45.0 | 2.0 | 45.1 | 90 | 90 | 180 | 90,0,180 | ✓ |
| 88 | disagreement_cross_then_release | — | — | — | — | — | — | — | — | ✗ |
| 89 | thumbs_up_approval | — | — | — | — | — | — | — | — | n/a |
| 90 | clap_applause | — | — | — | — | — | — | — | — | ✗ |
| 91 | hand_over_heart_sincerity | — | — | — | — | — | — | — | — | ✗ |

## Notes

- Location units are body-relative Motion Code percentages (0–100).
- Unspecified LLM xyz axes are filled from the G1 feasible-bank pose the renderer selects for dir|grip|elbow (render-effective EE %; no 50 penalty).
- Location match: |Δaxis| ≤ 5.0.
- Direction: `dir` → reach unit vector; `grip` palm token → palm normal; RPY from the reach–palm frame (geodesic = rotation angle between GT and LLM frames). Reach/palm angles are acos ∈ [0, 180°].
- Movement binary: GT (axis, sign) set ⊆ LLM set; line paths share the same (axis, sign) keys as joint moves; arcs stay as (arc, plane).
