# Human-like robots: carrier choice vs attribute errors

Location error = any axis |Δ| ≥ 20 body-%. Orientation error = reach or palm mismatch.

## G1

- Cues scored: 78; LLM uses the same carrier set as the human program: **58/78**
- Among those, at least one attribute error: **50/58** (orientation 44, location 38, movement 18)

| Cue | Same carrier | O err | L err | M err | GT carriers | LLM carriers |
|---|:---:|:---:|:---:|:---:|---|---|
| `nod_yes` | Y |  |  |  | head | head |
| `shake_head_no` | Y |  |  |  | head | head |
| `draw_circle_repeat` | Y |  | x | x | arm | arm |
| `gaze_shift_avoid` | N |  |  | x | head, torso | arm, head |
| `fist_pump` | Y | x | x |  | arm | arm |
| `tilt_head_curious` | N |  |  | x | head, torso | arm, head |
| `cover_yawn_sleepy` | Y | x | x | x | arm, head, torso | arm, head, torso |
| `rub_eye_tired` | N |  |  |  | arm | arm, head |
| `point_you` | Y |  | x |  | arm | arm |
| `point_self` | Y | x |  |  | arm | arm |
| `beckon_come_here` | Y | x |  | x | arm | arm |
| `stop_palm_out` | Y | x | x |  | arm | arm |
| `salute` | Y | x | x | x | arm | arm |
| `raise_index_one_moment` | Y | x | x |  | arm | arm |
| `highfive_invite` | Y | x | x | x | arm | arm |
| `shrug_idk` | Y |  |  | x | arm, second_arm | arm, second_arm |
| `flex_bicep` | Y |  |  |  | arm | arm |
| `air_quotes` | Y | x |  | x | arm, second_arm | arm, second_arm |
| `arms_open_welcome` | N | x | x | x | arm, second_arm | arm, second_arm, torso |
| `hands_up_surrender` | Y |  | x | x | arm, second_arm | arm, second_arm |
| `stretch_arms_up` | Y | x |  |  | arm, second_arm, torso | arm, second_arm, torso |
| `raise_fists_victory` | N | x | x | x | arm, second_arm | arm, second_arm, torso |
| `cross_arms_defensive` | Y | x | x |  | arm, second_arm | arm, second_arm |
| `palms_together_thank_you` | Y | x | x |  | arm, head, second_arm | arm, head, second_arm |
| `facepalm` | Y |  |  |  | arm, head, torso | arm, head, torso |
| `depict_tree` | Y | x |  |  | arm, second_arm, torso | arm, second_arm, torso |
| `depict_big_bowl` | Y | x | x |  | arm, second_arm | arm, second_arm |
| `depict_lion` | Y | x | x | x | arm, second_arm, torso | arm, second_arm, torso |
| `depict_small_object` | Y | x |  | x | arm | arm |
| `depict_flying_bird` | Y |  |  |  | arm, second_arm | arm, second_arm |
| `self_scratch_head_confused` | N | x | x |  | arm | arm, head |
| `pat_stomach` | Y | x | x |  | arm | arm |
| `fan_face_hot` | Y |  |  |  | arm | arm |
| `tap_temple_think` | N |  |  |  | arm | arm, head |
| `circle_temple_crazy` | Y | x |  |  | arm | arm |
| `lean_forward_interest` | N |  |  |  | arm, torso | torso |
| `cover_mouth_gasp` | N | x | x | x | arm | arm, torso |
| `cheers_toast` | Y | x | x | x | arm | arm |
| `highfive_imitation` | Y | x | x |  | arm | arm |
| `fistbump_invite` | Y | x |  |  | arm | arm |
| `pat_self_back` | Y | x | x | x | arm | arm |
| `throat_cut_stop` | Y |  |  |  | arm | arm |
| `visor_search` | Y | x | x |  | arm, head | arm, head |
| `yield_turn` | Y | x |  | x | arm | arm |
| `request_turn` | Y | x | x |  | arm | arm |
| `guiding_arm` | Y | x | x | x | arm | arm |
| `protective_arm_out` | Y | x | x |  | arm | arm |
| `slow_down_request_palm_down` | Y |  | x |  | arm | arm |
| `firm_accept_forward_reach` | Y | x | x |  | arm | arm |
| `shh_gesture` | Y | x | x |  | arm | arm |
| `scale_expand` | Y | x | x |  | arm, second_arm | arm, second_arm |
| `hesitant_accept_half_reach` | Y | x | x |  | arm | arm |
| `cancel_previous_offer_retract` | Y | x | x |  | arm | arm |
| `conditional_accept_pause_then_reach` | Y | x |  |  | arm | arm |
| `final_decision_down_press` | Y | x | x |  | arm | arm |
| `resume_speaking_pull` | Y | x |  |  | arm | arm |
| `emphasize_point_downbeat` | Y |  | x |  | arm | arm |
| `prepare_action_raise_hold` | Y | x | x |  | arm | arm |
| `exploration_reach` | N | x | x |  | arm | arm, head |
| `uncertainty_wave` | N | x |  |  | arm | arm, second_arm |
| `clarification_request_palm_tilt` | Y | x | x | x | arm | arm |
| `speed_up_circular_cue` | Y |  |  |  | arm | arm |
| `polite_decline_small_push` | N | x | x |  | arm | arm, head |
| `unsure_response_hand_hover_stop` | Y | x | x |  | arm | arm |
| `continue_on_forward_roll` | Y | x | x |  | arm | arm |
| `start_action_forward_snap` | Y | x | x |  | arm | arm |
| `hesitation_pause_hold` | N | x | x |  | arm | arm, head |
| `interrupt_hard` | Y | x | x | x | arm | arm |
| `interrupt_soft` | Y | x | x |  | arm | arm |
| `self_hug` | N | x | x | x | arm, second_arm | arm, head, second_arm, torso |
| `rub_arms_cold` | N | x |  | x | arm, second_arm | arm, second_arm, torso |
| `dominance_high_open_hold` | N | x | x |  | arm, second_arm | arm, head, second_arm, torso |
| `contain_space` | Y | x | x | x | arm, second_arm | arm, second_arm |
| `expand_space` | N | x | x | x | arm, second_arm | arm, second_arm, torso |
| `compress_space` | Y | x | x | x | arm, second_arm | arm, second_arm |
| `balance_scale` | N | x | x | x | arm, second_arm | arm, second_arm, torso |
| `push_pull_dual` | N |  | x | x | arm | arm, second_arm |
| `abandon_action_drop_turn` | N |  |  | x | arm, torso | head, torso |

## TIAGo

- Cues scored: 57; LLM uses the same carrier set as the human program: **38/57**
- Among those, at least one attribute error: **30/38** (orientation 25, location 19, movement 12)

| Cue | Same carrier | O err | L err | M err | GT carriers | LLM carriers |
|---|:---:|:---:|:---:|:---:|---|---|
| `self_scratch_head_confused` | N | x | x |  | arm | arm, head |
| `pat_stomach` | Y | x |  |  | arm | arm |
| `fan_face_hot` | Y | x | x |  | arm | arm |
| `tap_temple_think` | N |  | x | x | arm | arm, head |
| `circle_temple_crazy` | Y | x | x | x | arm | arm |
| `lean_forward_interest` | N |  |  | x | arm, base | torso |
| `cover_mouth_gasp` | N | x | x |  | arm | arm, torso |
| `cheers_toast` | Y | x | x | x | arm | arm |
| `highfive_imitation` | Y |  | x |  | arm | arm |
| `fistbump_invite` | Y | x | x |  | arm | arm |
| `pat_self_back` | Y | x |  | x | arm | arm |
| `throat_cut_stop` | Y |  |  |  | arm | arm |
| `visor_search` | Y |  |  |  | arm, head | arm, head |
| `yield_turn` | Y |  |  | x | arm | arm |
| `request_turn` | Y | x |  |  | arm | arm |
| `guiding_arm` | Y | x | x | x | arm | arm |
| `protective_arm_out` | Y | x | x |  | arm | arm |
| `slow_down_request_palm_down` | Y |  |  |  | arm | arm |
| `firm_accept_forward_reach` | Y |  |  |  | arm | arm |
| `shh_gesture` | Y | x |  |  | arm | arm |
| `scale_expand` | Y | x | x |  | arm, second_arm | arm, second_arm |
| `hesitant_accept_half_reach` | N | x | x |  | arm | arm, head |
| `cancel_previous_offer_retract` | N |  |  | x | arm |  |
| `conditional_accept_pause_then_reach` | Y |  | x |  | arm | arm |
| `final_decision_down_press` | Y | x |  |  | arm | arm |
| `resume_speaking_pull` | Y | x |  |  | arm | arm |
| `emphasize_point_downbeat` | Y |  | x |  | arm | arm |
| `prepare_action_raise_hold` | Y | x | x |  | arm | arm |
| `exploration_reach` | Y | x | x |  | arm | arm |
| `uncertainty_wave` | Y |  |  |  | arm | arm |
| `clarification_request_palm_tilt` | Y |  |  |  | arm | arm |
| `speed_up_circular_cue` | Y |  |  |  | arm | arm |
| `polite_decline_small_push` | N | x |  |  | arm | arm, head |
| `unsure_response_hand_hover_stop` | Y | x |  |  | arm | arm |
| `continue_on_forward_roll` | Y | x |  | x | arm | arm |
| `start_action_forward_snap` | Y | x | x |  | arm | arm |
| `hesitation_pause_hold` | Y |  |  |  | arm | arm |
| `interrupt_hard` | Y | x | x | x | arm | arm |
| `interrupt_soft` | Y | x |  |  | arm | arm |
| `self_hug` | Y | x | x | x | arm, second_arm | arm, second_arm |
| `rub_arms_cold` | N |  |  | x | arm, second_arm | arm, head, second_arm |
| `dominance_high_open_hold` | N |  | x |  | arm, second_arm | arm, second_arm, torso |
| `contain_space` | Y | x | x | x | arm, second_arm | arm, second_arm |
| `expand_space` | Y | x | x | x | arm, second_arm | arm, second_arm |
| `compress_space` | Y | x | x | x | arm, second_arm | arm, second_arm |
| `balance_scale` | N | x |  | x | arm, second_arm | arm, second_arm, torso |
| `push_pull_dual` | Y |  |  | x | arm, second_arm | arm, second_arm |
| `abandon_action_drop_turn` | N |  |  | x | arm, torso | head, torso |
| `escorting_gesture` | N |  |  | x | arm | arm, head |
| `wait_for_me_backward_palm` | N | x | x | x | arm | arm, base |
| `prevent_fall_support_reach` | N |  | x |  | arm | arm, torso |
| `frustration_down_throw` | Y | x | x |  | arm, second_arm, torso | arm, second_arm, torso |
| `submission_lowered_open_palm` | N |  |  | x | arm, second_arm | arm, second_arm, torso |
| `apology_forward_lowered_hold` | N | x | x |  | arm, second_arm, torso | arm, head, second_arm, torso |
| `chin_flick_dismiss` | N | x |  | x | arm | arm, head |
| `cover_ears_too_loud` | N | x | x |  | arm, second_arm | arm, head, second_arm, torso |
| `disagreement_cross_then_release` | N |  |  | x | arm, second_arm |  |

