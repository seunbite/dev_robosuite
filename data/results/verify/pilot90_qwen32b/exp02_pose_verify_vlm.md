# Pose Verification — single selected tile

- backend: `transformers`
- model: `Qwen/Qwen2.5-VL-32B-Instruct`
- config: `/home/hoyeonk/sblee/dev_robosuite/data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json`
- group tiles: `/home/hoyeonk/sblee/dev_robosuite/data/results/visualize/pose_groups_12`
- tile picks: `/home/hoyeonk/sblee/dev_robosuite/data/results/verify/pose_tile_pick_by_group.json`

| idx | cue | tile | pose ok? | note |
|---:|---|---:|---|---|
| 7 | raise_fists_victory | 3 | False | The current pose (dir=front, gripper_orientation=vertical) does not align well with the cue 'raise_fists_victory'. For this cue, the robot s |
| 8 | fist_pump | 5 | False | The current pose has the gripper oriented horizontally and pointing forward (dir=front). For a fist pump, the gripper should ideally be poin |
| 9 | flex_bicep | 2 | False | The current pose (dir=down, gripper_orientation=vertical) does not align with the requirements for the 'flex_bicep' cue. For a bicep flex, t |
| 10 | shrug_idk | 6 | True | The current pose (dir=up, gripper_orientation=horizontal) is appropriate for the 'shrug_idk' cue because the arm is already raised, which al |
| 11 | self_scratch_head_confused | 3 | True | The current pose (dir=left, gripper_orientation=vertical) is appropriate for the cue 'self_scratch_head_confused'. The arm is positioned to  |
| 12 | facepalm | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the requirements for a facepalm gesture. For a facepal |
| 13 | point_self | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align with the cue 'point_self'. For this cue, the robot should start  |
| 14 | point_you | 2 | False | The current pose has the gripper pointing downward (dir=down), which does not align with the requirement to point directly at the viewer (di |
| 15 | beckon_come_here | 5 | True | The current pose (dir=front, gripper_orientation=horizontal) is appropriate for the 'beckon_come_here' cue because the end effector is alrea |
| 16 | stop_palm_out | 2 | False | The current pose (dir=down, gripper_orientation=vertical) does not align with the requirements for the 'stop_palm_out' cue. For a 'stop' ges |
| 17 | pat_stomach | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the cue 'pat_stomach'. The arm is pointing forward, wh |
| 18 | fan_face_hot | 3 | True | The current pose (dir=left, gripper_orientation=vertical) is appropriate for the 'fan_face_hot' cue because the arm is already positioned to |
| 19 | cover_yawn_sleepy | 6 | False | The current pose (dir=down, gripper_orientation=horizontal) is not suitable for the 'cover_yawn_sleepy' cue because the gripper is pointing  |
| 20 | rub_eye_tired | 3 | True | The current pose (dir=left, gripper_orientation=vertical) is appropriate for the 'rub_eye_tired' cue because the arm is positioned to the si |
| 21 | tap_temple_think | 5 | False | The current pose has the gripper pointing forward (dir=front) and oriented horizontally. However, for the 'tap_temple_think' cue, the grippe |
| 22 | circle_temple_crazy | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the target cue 'circle_temple_crazy'. The arm is point |
| 23 | lean_forward_interest | 5 | True | The current pose (dir=front, gripper_orientation=horizontal) is appropriate for the 'lean_forward_interest' cue because the arm is already o |
| 24 | cover_mouth_gasp | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the target cue 'cover_mouth_gasp'. The arm is pointing |
| 25 | cheers_toast | 6 | False | The current pose (dir=up, gripper_orientation=horizontal) does not align with the requirements for the 'cheers_toast' cue. For a toast gestu |
| 26 | highfive_invite | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align with the requirements for a high-five invitation. For a high-fiv |
| 27 | highfive_imitation | 6 | False | The current pose (dir=up, gripper_orientation=horizontal) does not align with the requirements for the 'highfive_imitation' cue. For a high- |
| 28 | fistbump_invite | 2 | False | The current pose (dir=down, gripper_orientation=vertical) is not suitable for the 'fistbump_invite' cue because the end effector is pointing |
| 30 | pat_self_back | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) is not suitable for the cue 'pat_self_back' because the arm is pointing forward |
| 31 | draw_circle_repeat | 3 | True | The current pose (dir=front, gripper_orientation=vertical) is appropriate for the 'draw_circle_repeat' cue because the end effector is alrea |
| 32 | throat_cut_stop | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align with the target cue 'throat_cut_stop'. The cue requires the arm  |
| 33 | chin_flick_dismiss | 5 | False | The current pose has the gripper oriented horizontally and pointing forward (dir=front). However, for the 'chin_flick_dismiss' cue, the grip |
| 34 | cover_ears_too_loud | 5 | False | The current pose (dir=back, gripper_orientation=vertical) does not align well with the cue 'cover_ears_too_loud'. For this cue, the robot ne |
| 35 | salute | 5 | False | The current pose has the gripper pointing forward (dir=front) and oriented horizontally. For a formal salute, the gripper should initially p |
| 36 | visor_search | 3 | False | The current pose (dir=left, gripper_orientation=vertical) does not align well with the target cue 'visor_search'. The hand should start near |
| 37 | yield_turn | 5 | False | The current pose (dir=back, gripper_orientation=vertical) is not suitable for the 'yield_turn' cue because the arm is pointing backward and  |
| 39 | request_turn | 6 | False | The current pose (dir=up, gripper_orientation=horizontal) does not align well with the cue 'request_turn'. For this cue, the robot should st |
| 40 | guiding_arm | 3 | True | The current pose (dir=front, gripper_orientation=vertical) is appropriate for the 'guiding_arm' cue because the arm is extended forward with |
| 41 | protective_arm_out | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the cue 'protective_arm_out'. The arm is positioned to |
| 42 | escorting_gesture | 6 | False | The current pose (dir=up, gripper_orientation=horizontal) does not align with the requirements for an escorting gesture. For escorting, the  |
| 44 | clarification_request_palm_tilt | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the cue 'clarification_request_palm_tilt'. The palm sh |
| 45 | slow_down_request_palm_down | 6 | False | The current pose has dir=down and gripper_orientation=horizontal, but the cue requires starting from a high forward position with the grippe |
| 46 | speed_up_circular_cue | 3 | True | The current pose (dir=front, gripper_orientation=vertical) is appropriate for the 'speed_up_circular_cue'. The gripper is oriented verticall |
| 47 | wait_for_me_backward_palm | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the cue 'wait_for_me_backward_palm'. The gesture requi |
| 48 | continue_on_forward_roll | 5 | True | The current pose (dir=front, gripper_orientation=horizontal) is appropriate for the cue 'continue_on_forward_roll'. The arm is positioned to |
| 49 | polite_decline_small_push | 3 | False | The current pose (dir=front, gripper_orientation=vertical) does not align well with the cue 'polite_decline_small_push'. For this cue, the r |
| 50 | firm_accept_forward_reach | 5 | True | The current pose (dir=front, gripper_orientation=horizontal) is appropriate for the cue 'firm_accept_forward_reach'. The arm is already orie |
| 51 | hesitant_accept_half_reach | 3 | True | The current pose (dir=front, gripper_orientation=vertical) is appropriate for the 'hesitant_accept_half_reach' cue because the robot's end e |
| 52 | cancel_previous_offer_retract | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the cue 'cancel_previous_offer_retract'. The arm is po |
| 53 | conditional_accept_pause_then_reach | 3 | False | The current pose (dir=front, gripper_orientation=vertical) does not align well with the cue 'conditional_accept_pause_then_reach'. The front |
| 54 | final_decision_down_press | 6 | True | The current pose (dir=down, gripper_orientation=horizontal) is appropriate for the cue 'final_decision_down_press'. The EE is already pointi |
| 55 | resume_speaking_pull | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the cue 'resume_speaking_pull'. The arm is pointing fo |
| 56 | emphasize_point_downbeat | 3 | False | The current pose (dir=front, gripper_orientation=vertical) does not align well with the target cue 'emphasize_point_downbeat'. For this cue, |
| 57 | unsure_response_hand_hover_stop | 5 | True | The current pose (dir=front, gripper_orientation=horizontal) is appropriate for the cue 'unsure_response_hand_hover_stop'. The robot's end e |
| 58 | prevent_fall_support_reach | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the target cue 'prevent_fall_support_reach'. The arm i |
| 59 | nod_yes | 5 | True | The current pose (dir=front, gripper_orientation=horizontal) is appropriate for the 'nod_yes' cue because the gripper is positioned forward  |
| 60 | shake_head_no | 3 | True | The current pose (dir=front, gripper_orientation=vertical) is appropriate for the 'shake_head_no' cue because the wrist is already oriented  |
| 61 | air_quotes | 5 | True | The current pose (dir=front, gripper_orientation=horizontal) is appropriate for the 'air_quotes' cue because the gripper is oriented horizon |
| 62 | self_hug | 3 | False | The current pose (dir=front, gripper_orientation=vertical) is not suitable for the 'self_hug' cue because the arm is pointing forward, which |
| 64 | tilt_head_curious | 5 | True | The current pose (dir=front, gripper_orientation=horizontal) is appropriate for the 'tilt_head_curious' cue because the gripper is positione |
| 65 | raise_index_one_moment | 5 | True | The current pose (dir=front, gripper_orientation=horizontal) is appropriate for the 'raise_index_one_moment' cue because the gripper is posi |
| 66 | shh_gesture | 2 | False | The current pose (dir=up, gripper_orientation=vertical) does not align well with the 'shh_gesture' cue, which requires the robot to bring it |
| 67 | rub_arms_cold | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the cue 'rub_arms_cold'. The arm is extended forward,  |
| 69 | cross_arms_defensive | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the target cue 'cross_arms_defensive'. The arm is exte |
| 72 | palms_together_thank_you | 2 | False | The current pose (dir=down, gripper_orientation=vertical) is not suitable for the 'palms_together_thank_you' cue because the end effector is |
| 74 | hands_up_surrender | 5 | False | The current pose has the gripper pointing forward (dir=front) and oriented horizontally. For a 'hands_up_surrender' gesture, the gripper sho |
| 75 | arms_open_welcome | 5 | False | The current pose (dir=back, gripper_orientation=vertical) is not suitable for the 'arms_open_welcome' cue because the arm is pointing backwa |
| 76 | stretch_arms_up | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) is not suitable for the 'stretch_arms_up' cue because the end effector is point |
| 78 | disagreement_cross_then_release | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the cue 'disagreement_cross_then_release'. The arm is  |
| 82 | frustration_down_throw | 6 | False | The current pose (dir=up, gripper_orientation=horizontal) does not align well with the cue 'frustration_down_throw'. The cue requires a stro |
| 84 | submission_lowered_open_palm | 2 | False | The current pose (dir=down, gripper_orientation=vertical) does not align with the requirements for the 'submission_lowered_open_palm' cue. T |
| 85 | dominance_high_open_hold | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the target cue 'dominance_high_open_hold'. The arm is  |
| 86 | apology_forward_lowered_hold | 2 | False | The current pose (dir=down, gripper_orientation=vertical) does not align with the requirements for an apology gesture. For an apology, the r |
| 90 | start_action_forward_snap | 2 | False | The current pose (dir=up, gripper_orientation=vertical) does not align well with the cue 'start_action_forward_snap'. The cue requires a for |
| 92 | retry_action_reset_circle | 3 | False | The current pose (dir=front, gripper_orientation=vertical) does not align well with the cue 'retry_action_reset_circle'. The cue requires a  |
| 94 | abandon_action_drop_turn | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the cue 'abandon_action_drop_turn'. The robot's arm is |
| 95 | prepare_action_raise_hold | 5 | True | The current pose (dir=front, gripper_orientation=horizontal) is appropriate for the 'prepare_action_raise_hold' cue because the arm is posit |
| 97 | hesitation_pause_hold | 3 | True | The current pose (dir=front, gripper_orientation=vertical) is appropriate for the 'hesitation_pause_hold' cue because the robot's end effect |
| 104 | scale_expand | 6 | False | The current pose (dir=down, gripper_orientation=horizontal) does not align with the target cue 'scale_expand'. The cue requires the robot to |
| 106 | uncertainty_wave | 6 | True | The current pose (dir=up, gripper_orientation=horizontal) is appropriate for the 'uncertainty_wave' cue because the palm is already facing u |
| 107 | exploration_reach | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the exploration_reach cue. The arm is already extended |
| 108 | interrupt_hard | 2 | False | The current pose (dir=down, gripper_orientation=vertical) does not align well with the cue 'interrupt_hard'. For an explosive interruption g |
| 109 | interrupt_soft | 3 | True | The current pose (dir=front, gripper_orientation=vertical) is appropriate for the 'interrupt_soft' cue because the hand is positioned to nat |
| 114 | contain_space | 3 | False | The current pose (dir=left, gripper_orientation=vertical) does not align well with the cue 'contain_space'. For defining a space, the robot  |
| 115 | expand_space | 3 | True | The current pose (dir=front, gripper_orientation=vertical) is appropriate for the 'expand_space' cue because the arm is positioned to easily |
| 116 | compress_space | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align with the requirements for the 'compress_space' cue. The hand sho |
| 117 | balance_scale | 6 | False | The current pose (dir=up, gripper_orientation=horizontal) does not align with the requirements for the 'balance_scale' cue. For the balance  |
| 118 | push_pull_dual | 3 | True | The current pose (dir=front, gripper_orientation=vertical) is appropriate for the 'push_pull_dual' cue because the end effector is already o |
| 121 | weight_shift_forward | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the cue 'weight_shift_forward'. The robot's arm is ext |
| 122 | weight_shift_backward | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the cue 'weight_shift_backward'. The robot's arm is ex |
| 123 | torso_rotate_away | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the cue 'torso_rotate_away'. The robot's arm is extend |
| 124 | torso_rotate_toward | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the target cue 'torso_rotate_toward'. The arm is exten |
| 125 | lean_side_doubt | 5 | True | The current pose (dir=front, gripper_orientation=horizontal) is appropriate for the 'lean_side_doubt' cue because the robot's arm is positio |
| 127 | expand_chest | 6 | False | The current pose (dir=down, gripper_orientation=horizontal) is not suitable for the 'expand_chest' cue because the end effector is pointing  |
| 128 | eye_widen_surprise | 5 | False | The current pose (dir=front, gripper_orientation=horizontal) does not align well with the target cue 'eye_widen_surprise'. The pose suggests |
| 130 | gaze_shift_avoid | 5 | True | The current pose (dir=front, gripper_orientation=horizontal) is appropriate for the 'gaze_shift_avoid' cue because the robot's end effector  |
