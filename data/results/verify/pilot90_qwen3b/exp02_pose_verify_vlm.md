# Pose Verification — single selected tile

- backend: `transformers`
- model: `Qwen/Qwen2.5-VL-3B-Instruct`
- config: `/home/hoyeonk/sblee/dev_robosuite/data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json`
- group tiles: `/home/hoyeonk/sblee/dev_robosuite/data/results/visualize/pose_groups_12`
- tile picks: `/home/hoyeonk/sblee/dev_robosuite/data/results/verify/pose_tile_pick_by_group.json`

| idx | cue | tile | pose ok? | note |
|---:|---|---:|---|---|
| 7 | raise_fists_victory | 3 | False | The current pose does not align well with the target cue 'raise_fists_victory'. The current pose has the gripper oriented vertically, which  |
| 8 | fist_pump | 5 | False | The current pose does not align well with the intended cue of a fist pump. The current direction (front) and gripper orientation (horizontal |
| 9 | flex_bicep | 2 | False | The current pose does not align well with the desired cue. The robot should start with an extended arm to the side, which requires a differe |
| 10 | shrug_idk | 6 | False | The current pose does not align well with the desired cue 'shrug_idk'. The robot should start with an upward and outward lift from the shoul |
| 11 | self_scratch_head_confused | 3 | False | The current pose does not align well with the target cue. The robot should start by raising its arm to its head from the left side. |
| 12 | facepalm | 5 | False | The current pose does not align well with the target cue 'facepalm'. The current direction (front) and gripper orientation (horizontal) do n |
| 13 | point_self | 5 | False | The current pose does not align well with the target cue 'point_self'. The robot is currently facing forward with a horizontal gripper orien |
| 14 | point_you | 2 | False | The current pose does not align well with the target cue. The robot should start with a neutral position and then move forward to point dire |
| 15 | beckon_come_here | 5 | False | The current pose does not align well with the intended cue. The robot should start by pointing forward and then curl the wrist inward. |
| 16 | stop_palm_out | 2 | False | The current pose does not align well with the target cue 'stop_palm_out'. The current direction is down, which is incorrect for the cue. |
| 17 | pat_stomach | 5 | False | The current pose does not align well with the target cue 'pat_stomach'. The robot's current orientation (dir=front, gripper_orientation=hori |
| 18 | fan_face_hot | 3 | False | The current pose does not align well with the target cue 'fan_face_hot'. The current direction (left) and gripper orientation (vertical) do  |
| 19 | cover_yawn_sleepy | 6 | False | The current pose does not align well with the target cue 'cover_yawn_sleepy'. The current direction is down, which is not suitable for raisi |
| 20 | rub_eye_tired | 3 | False | The current pose does not align well with the target cue 'rub_eye_tired'. The current direction (left) and gripper orientation (vertical) do |
| 21 | tap_temple_think | 5 | False | The current pose does not align well with the target cue 'tap_temple_think'. The current direction (front) and gripper orientation (horizont |
| 22 | circle_temple_crazy | 5 | False | The current pose does not align well with the target cue 'circle_temple_crazy'. The current direction (front) and gripper orientation (horiz |
| 23 | lean_forward_interest | 5 | False | The current pose does not align well with the target cue 'lean_forward_interest'. The current direction (front) and gripper orientation (hor |
| 24 | cover_mouth_gasp | 5 | False | The current pose does not align well with the target cue 'cover_mouth_gasp'. The current direction (front) and gripper orientation (horizont |
| 25 | cheers_toast | 6 | False | The current pose does not align well with the desired cue 'cheers_toast'. The current pose is an upward-facing gesture with a horizontal gri |
| 26 | highfive_invite | 5 | False | The current pose does not align well with the desired cue. The robot should start with a raised arm in a high-five position. |
| 27 | highfive_imitation | 6 | False | The current pose does not align well with the desired cue. The robot should start with an arm raised in a ready position, which requires a d |
| 28 | fistbump_invite | 2 | False | The current pose does not align well with the intended cue. The robot should start with a more upright position to facilitate the fist bump. |
| 30 | pat_self_back | 5 | False | The current pose does not align well with the target cue 'pat_self_back'. The robot is currently facing forward with a horizontal gripper or |
| 31 | draw_circle_repeat | 3 | False | The current pose does not align well with the target cue 'draw_circle_repeat'. The current direction (front) and gripper orientation (vertic |
| 32 | throat_cut_stop | 5 | False | The current pose does not align well with the target cue 'throat_cut_stop'. The current direction (front) and gripper orientation (horizonta |
| 33 | chin_flick_dismiss | 5 | False | The current pose does not align well with the target cue 'chin_flick_dismiss'. The current direction (front) and gripper orientation (horizo |
| 34 | cover_ears_too_loud | 5 | False | The current pose does not align well with the target cue 'cover_ears_too_loud'. The robot should start by moving its hand towards the ear ar |
| 35 | salute | 5 | False | The current pose does not align well with the desired cue of a salute. The current direction (front) and gripper orientation (horizontal) do |
| 36 | visor_search | 3 | False | The current pose does not align well with the target cue. The current direction (left) and gripper orientation (vertical) do not match the r |
| 37 | yield_turn | 5 | False | The current pose (dir=back, gripper_orientation=vertical) is not suitable for the target cue 'yield_turn'. The robot needs to start from a n |
| 39 | request_turn | 6 | False | The current pose does not align well with the target cue. The robot should start with an upward direction and a horizontal gripper orientati |
| 40 | guiding_arm | 3 | False | The current pose does not align well with the target cue. The robot should start with the arm extended forward and the hand vertical. |
| 41 | protective_arm_out | 5 | False | The current pose does not align well with the required protective arm out motion. |
| 42 | escorting_gesture | 6 | False | The current pose does not align well with the intended cue of an escorting gesture. |
| 44 | clarification_request_palm_tilt | 5 | False | The current pose does not align well with the intended cue. The robot should start by raising its hand slightly and then turn its palm up an |
| 45 | slow_down_request_palm_down | 6 | False | The current pose does not align well with the target cue. The robot should start from a high forward position with the gripper horizontal (p |
| 46 | speed_up_circular_cue | 3 | False | The current pose does not align well with the target cue. The current direction (front) and gripper orientation (vertical) do not facilitate |
| 47 | wait_for_me_backward_palm | 5 | False | The current pose does not align well with the target cue 'wait_for_me_backward_palm'. The current direction is 'front', which is not suitabl |
| 48 | continue_on_forward_roll | 5 | False | The current pose does not align well with the intended cue. The current direction (front) and gripper orientation (horizontal) do not provid |
| 49 | polite_decline_small_push | 3 | False | The current pose does not align well with the target cue. The robot should start by pointing down to show hesitation before pushing upwards. |
| 50 | firm_accept_forward_reach | 5 | False | The current pose does not align well with the target cue 'firm_accept_forward_reach'. The current direction (front) is correct, but the grip |
| 51 | hesitant_accept_half_reach | 3 | False | The current pose does not align well with the intended cue. The current direction (front) and gripper orientation (vertical) do not match th |
| 52 | cancel_previous_offer_retract | 5 | False | The current pose does not align well with the target cue. The robot should start by retracting the arm sharply, followed by a lateral shake. |
| 53 | conditional_accept_pause_then_reach | 3 | False | The current pose does not align well with the target cue. The robot should start with a slight recoil or pause to show hesitation. |
| 54 | final_decision_down_press | 6 | False | The current pose does not align well with the target cue 'final_decision_down_press'. The current direction (down) is correct, but the gripp |
| 55 | resume_speaking_pull | 5 | False | The current pose does not align well with the required direction and gripper orientation for the 'resume_speaking_pull' cue. |
| 56 | emphasize_point_downbeat | 3 | False | The current pose does not align well with the intended cue. The robot should start with a pointing gesture, which requires a different direc |
| 57 | unsure_response_hand_hover_stop | 5 | False | The current pose does not align well with the intended cue. The robot should start with a more neutral or slightly upward-facing orientation |
| 58 | prevent_fall_support_reach | 5 | False | The current pose does not align well with the intended cue. The robot should start by moving the arm back and down to simulate losing balanc |
| 59 | nod_yes | 5 | False | The current pose does not align well with the desired cue. The robot should start with a high forward position and then perform two quick, s |
| 60 | shake_head_no | 3 | False | The current pose does not align well with the target cue. The current pose has the gripper oriented vertically, which is not suitable for th |
| 61 | air_quotes | 5 | False | The current pose does not align well with the target cue 'air_quotes'. The current pose has the gripper oriented horizontally, which is not  |
| 62 | self_hug | 3 | False | The current pose does not align well with the target cue 'self_hug'. The current direction (front) and gripper orientation (vertical) do not |
| 64 | tilt_head_curious | 5 | False | The current pose does not align well with the desired cue of tilting the head curiously. |
| 65 | raise_index_one_moment | 5 | False | The current pose does not align well with the target cue 'raise_index_one_moment'. The current direction (front) and gripper orientation (ho |
| 66 | shh_gesture | 2 | False | The current pose does not align well with the target cue. The robot should start with an arm raised (dir=up) and then bring the gripper forw |
| 67 | rub_arms_cold | 5 | False | The current pose does not align well with the intended cue of 'rub_arms_cold'. The current direction (front) and gripper orientation (horizo |
| 69 | cross_arms_defensive | 5 | False | The current pose does not align well with the target cue 'cross_arms_defensive'. The current direction (front) and gripper orientation (hori |
| 72 | palms_together_thank_you | 2 | False | The current pose does not align well with the desired cue 'pals_together_thank_you'. The current direction (down) and gripper orientation (v |
| 74 | hands_up_surrender | 5 | False | The current pose does not align well with the target cue 'hands_up_surrender'. The current direction (front) and gripper orientation (horizo |
| 75 | arms_open_welcome | 5 | False | The current pose (dir=back, gripper_orientation=vertical) does not align well with the cue 'arms_open_welcome'. The robot needs to face forw |
| 76 | stretch_arms_up | 5 | False | The current pose does not align well with the target cue 'stretch_arms_up'. The current direction (front) and gripper orientation (horizonta |
| 78 | disagreement_cross_then_release | 5 | False | The current pose does not align well with the target cue. The robot should start with a slight recoil or retraction before crossing an arm. |
| 82 | frustration_down_throw | 6 | False | The current pose does not align well with the intended cue. The robot should start with a more neutral or slightly forward-facing position t |
| 84 | submission_lowered_open_palm | 2 | False | The current pose does not align well with the target cue 'submission_lowered_open_palm'. The current pose is down and vertical, which does n |
| 85 | dominance_high_open_hold | 5 | False | The current pose does not align well with the target cue 'dominance_high_open_hold'. The current direction (front) and gripper orientation ( |
| 86 | apology_forward_lowered_hold | 2 | False | The current pose does not align well with the intended cue. The robot should start with a gesture that suggests a downward motion towards th |
| 90 | start_action_forward_snap | 2 | False | The current pose does not align well with the target cue 'start_action_forward_snap'. The current pose has the gripper oriented vertically,  |
| 92 | retry_action_reset_circle | 3 | False | The current pose does not align well with the required direction and gripper orientation for the 'retry_action_reset_circle' cue. |
| 94 | abandon_action_drop_turn | 5 | False | The current pose does not align well with the intended cue. The robot should start by pointing down to indicate a halt or stop, followed by  |
| 95 | prepare_action_raise_hold | 5 | False | The current pose does not align well with the required preparation for the 'prepare_action_raise_hold' cue. |
| 97 | hesitation_pause_hold | 3 | False | The current pose does not align well with the intended cue of hesitation_pause_hold. |
| 104 | scale_expand | 6 | False | The current pose does not align well with the target cue 'scale_expand'. The robot is currently pointing down, which is not suitable for ext |
| 106 | uncertainty_wave | 6 | False | The current pose does not align well with the intended cue of uncertainty_wave. |
| 107 | exploration_reach | 5 | False | The current pose does not align well with the intended cue of exploration_reach. The current direction (front) and gripper orientation (hori |
| 108 | interrupt_hard | 2 | False | The current pose does not align well with the intended cue. The robot should start by moving its arm upward to create the recoil effect. |
| 109 | interrupt_soft | 3 | False | The current pose does not align well with the intended cue. The robot should start with a more neutral or slightly upward-facing position to |
| 114 | contain_space | 3 | False | The current pose does not align well with the intended cue. The current direction (left) and gripper orientation (vertical) do not match the |
| 115 | expand_space | 3 | False | The current pose does not align well with the intended cue. The robot should start by moving the arm inward slightly to prepare for the swee |
| 116 | compress_space | 5 | False | The current pose does not align well with the target cue 'compress_space'. The current direction (front) and gripper orientation (horizontal |
| 117 | balance_scale | 6 | False | The current pose does not align well with the intended cue. The robot should start by moving its arm to the side to prepare for the balance  |
| 118 | push_pull_dual | 3 | False | The current pose does not align well with the intended cue. The robot should start by moving its shoulder backward slightly to prepare for t |
| 121 | weight_shift_forward | 5 | False | The current pose does not align well with the target cue 'weight_shift_forward'. The current direction (front) and gripper orientation (hori |
| 122 | weight_shift_backward | 5 | False | The current pose does not align well with the target cue 'weight_shift_backward'. The robot should start by moving its shoulder back and upw |
| 123 | torso_rotate_away | 5 | False | The current pose does not align well with the intended cue of 'torso_rotate_away'. The current direction (front) and gripper orientation (ho |
| 124 | torso_rotate_toward | 5 | False | The current pose does not align well with the target cue 'torso_rotate_toward'. The current direction (front) and gripper orientation (horiz |
| 125 | lean_side_doubt | 5 | False | The current pose does not align well with the intended cue. The robot should start with a more neutral or slightly forward-facing position t |
| 127 | expand_chest | 6 | False | The current pose does not align well with the target cue 'expand_chest'. The robot is currently pointing down, which is not suitable for the |
| 128 | eye_widen_surprise | 5 | False | The current pose does not align well with the target cue 'eye_widen_surprise'. The current direction (front) and gripper orientation (horizo |
| 130 | gaze_shift_avoid | 5 | False | The current pose does not align well with the target cue 'gaze_shift_avoid'. The current direction is front, which is not suitable for the i |
