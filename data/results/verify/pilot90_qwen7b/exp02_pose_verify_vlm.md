# Pose Verification — single selected tile

- backend: `transformers`
- model: `Qwen/Qwen2.5-VL-7B-Instruct`
- config: `/home/hoyeonk/sblee/dev_robosuite/data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json`
- group tiles: `/home/hoyeonk/sblee/dev_robosuite/data/results/visualize/pose_groups_12`
- tile picks: `/home/hoyeonk/sblee/dev_robosuite/data/results/verify/pose_tile_pick_by_group.json`

| idx | cue | tile | pose ok? | note |
|---:|---|---:|---|---|
| 7 | raise_fists_victory | 3 | False | The current pose does not align well with the cue's requirement of raising fists in a victory gesture. |
| 8 | fist_pump | 5 | False | The current pose does not align with the expected direction for a fist pump. |
| 9 | flex_bicep | 2 | False | The current pose does not align with the cue's requirement of starting with the arm extended to the side. |
| 10 | shrug_idk | 6 | False | The current pose does not align well with the shrug_idk cue. The arm should be raised and then moved outward and upward, which requires a di |
| 11 | self_scratch_head_confused | 3 | False | The current pose does not align well with the target cue. The robot should be facing front to scratch its head. |
| 12 | facepalm | 5 | False | The current pose does not align with the facepalm gesture. The arm should be raised initially, and the wrist should rotate to orient the gri |
| 13 | point_self | 5 | False | The current pose does not align with the cue 'point_self'. The robot should start from behind its back and move to the front to point at its |
| 14 | point_you | 2 | False | The current pose (dir=down, gripper_orientation=vertical) does not align with the cue requirement to point directly at the viewer. |
| 15 | beckon_come_here | 5 | False | The current pose does not align with the desired direction for the beckon_come_here cue. |
| 16 | stop_palm_out | 2 | False | The current pose does not align with the cue's requirement of starting with the arm raised and pulling back slightly before pushing forward. |
| 17 | pat_stomach | 5 | False | The current pose does not align with the target cue. The robot should start from the left side of its body to pat its stomach. |
| 18 | fan_face_hot | 3 | False | The current pose does not align with the cue's requirement of starting with the arm raised to the side. |
| 19 | cover_yawn_sleepy | 6 | False | The current pose does not align with the cue's requirement of covering a 'mouth' for a yawn. |
| 20 | rub_eye_tired | 3 | False | The current pose does not align well with the target cue. The robot should be positioned more towards the side of its head to start the rubb |
| 21 | tap_temple_think | 5 | False | The current pose does not align with the target cue. The arm should be extended left, not front. |
| 22 | circle_temple_crazy | 5 | False | The current pose does not align with the target cue. The robot should start from a side position to trace vertical circles. |
| 23 | lean_forward_interest | 5 | False | The current pose does not align well with the cue of leaning forward with curiosity and focused attention. |
| 24 | cover_mouth_gasp | 5 | False | The current pose does not align well with the cue 'cover_mouth_gasp'. The arm is extended forward, which is not suitable for a quick inward  |
| 25 | cheers_toast | 6 | False | The current pose does not align well with the cue's requirement of a raised 'toast' position. |
| 26 | highfive_invite | 5 | False | The current pose does not align well with the high-five invite cue. The arm should be positioned more vertically to match the high-five gest |
| 27 | highfive_imitation | 6 | False | The current pose does not align well with the high-five imitation cue. |
| 28 | fistbump_invite | 2 | False | The current pose does not align with the cue's requirement of extending the arm forward. |
| 30 | pat_self_back | 5 | False | The current pose does not align with the cue 'pat_self_back'. The robot should start with its arm behind its back. |
| 31 | draw_circle_repeat | 3 | False | The current pose does not align well with the target cue. The robot should start by pointing forward to establish a point, which is currentl |
| 32 | throat_cut_stop | 5 | False | The current pose does not align with the cue's requirement of starting with the arm extended to the left. |
| 33 | chin_flick_dismiss | 5 | False | The current pose does not align well with the cue's requirement to lift the 'head' in disdain. |
| 34 | cover_ears_too_loud | 5 | False | The current pose does not align well with the cue's requirement of recoiling sharply from a loud noise and then pressing the ear firmly. |
| 35 | salute | 5 | False | The current pose does not align with the desired gesture of a formal salute. |
| 36 | visor_search | 3 | False | The current pose does not align with the cue's starting position. |
| 37 | yield_turn | 5 | False | The current pose does not align with the cue's requirement of a neutral forward pose. |
| 39 | request_turn | 6 | False | The current pose does not align with the cue's requirement of starting with the arm raised and gripper held flat like a platter. |
| 40 | guiding_arm | 3 | False | The current pose does not align with the cue's requirement of extending the arm forward and making a smooth, deliberate sweep to the side. |
| 41 | protective_arm_out | 5 | False | The current pose does not align with the protective blocking motion required by the cue. |
| 42 | escorting_gesture | 6 | False | The current pose does not align with the escorting gesture, which requires a palm-up gesture. |
| 44 | clarification_request_palm_tilt | 5 | False | The current pose does not align with the cue's requirement for a palm-up and forward tilt. |
| 45 | slow_down_request_palm_down | 6 | True | The current pose is appropriate for the cue as the robot is already positioned with the gripper horizontal and pointing downward, which alig |
| 46 | speed_up_circular_cue | 3 | False | The current pose does not align well with the cue's requirement for a sharp poke followed by accelerating circular motions. |
| 47 | wait_for_me_backward_palm | 5 | False | The current pose does not align with the target cue. The robot should start by turning away, which requires a different direction. |
| 48 | continue_on_forward_roll | 5 | False | The current pose does not align well with the cue's requirement for a dynamic and guiding motion. |
| 49 | polite_decline_small_push | 3 | False | The current pose does not align well with the cue's requirement for a polite decline. The vertical gripper orientation suggests a pushing ac |
| 50 | firm_accept_forward_reach | 5 | True | The current pose is suitable for the cue as the arm is extended forward and the gripper is in a horizontal orientation, which aligns with th |
| 51 | hesitant_accept_half_reach | 3 | False | The current pose does not align well with the target cue. The robot should be positioned to signal hesitation and acceptance, which requires |
| 52 | cancel_previous_offer_retract | 5 | False | The current pose does not align well with the intended gesture of retracting an offer. The arm should be extended forward to present the off |
| 53 | conditional_accept_pause_then_reach | 3 | False | The current pose does not align well with the target cue. The robot should start with a more hesitant and back-oriented pose to convey the i |
| 54 | final_decision_down_press | 6 | False | The current pose does not align with the cue's requirement for a decisive downward press. |
| 55 | resume_speaking_pull | 5 | False | The current pose does not align well with the cue's requirement for a subtle upward lift followed by an inward pull. |
| 56 | emphasize_point_downbeat | 3 | False | The current pose does not align well with the cue's requirement for a clear pointing gesture. |
| 57 | unsure_response_hand_hover_stop | 5 | False | The current pose does not align well with the target cue. The robot should start with a hesitant forward reach, which requires a different d |
| 58 | prevent_fall_support_reach | 5 | False | The current pose does not align with the cue's requirement of a bracing motion. |
| 59 | nod_yes | 5 | True | The current pose is suitable for the cue as the robot is already in a high forward position with a horizontal gripper orientation. |
| 60 | shake_head_no | 3 | False | The current pose does not align well with the desired gesture of shaking the head side-to-side. |
| 61 | air_quotes | 5 | False | The current pose does not align well with the 'air_quotes' cue. The wrist flicks should be more dynamic and expressive. |
| 62 | self_hug | 3 | False | The current pose does not align well with the self-hug cue. The arm should be directed towards the chest and shoulder, not directly in front |
| 64 | tilt_head_curious | 5 | False | The current pose does not align with the cue's requirement of a curious head tilt. |
| 65 | raise_index_one_moment | 5 | False | The current pose does not align well with the target cue. The robot should be positioned to make a small anticipatory dip before crisply rai |
| 66 | shh_gesture | 2 | False | The current pose does not align with the cue's requirement of bringing the gripper forward smoothly. |
| 67 | rub_arms_cold | 5 | False | The current pose does not align well with the cue 'rub_arms_cold'. The robot should be positioned to mimic a shivering action. |
| 69 | cross_arms_defensive | 5 | False | The current pose does not align well with the defensive stance required by the cue. |
| 72 | palms_together_thank_you | 2 | False | The current pose does not align with the cue's requirement of bringing the hand to the chest in a prayer-like pose. |
| 74 | hands_up_surrender | 5 | False | The current pose does not align with the target cue. The arm should be raised higher and pitched forward to present a flat 'palm' in a class |
| 75 | arms_open_welcome | 5 | False | The current pose does not align with the cue's requirement of gathering attention towards the chest before sweeping outward. |
| 76 | stretch_arms_up | 5 | False | The current pose does not align with the cue's requirement for a stretch upwards. |
| 78 | disagreement_cross_then_release | 5 | False | The current pose does not align with the cue's requirement of recoiling slightly, crossing one arm over the chest, and then uncrossing slowl |
| 82 | frustration_down_throw | 6 | False | The current pose does not align with the cue's requirement of expressing frustration by recoiling sharply and throwing the hand down and for |
| 84 | submission_lowered_open_palm | 2 | False | The current pose does not align with the cue's requirement for a yielding gesture. |
| 85 | dominance_high_open_hold | 5 | False | The current pose does not align well with the cue's requirement for a high, wide, open-palmed position. |
| 86 | apology_forward_lowered_hold | 2 | False | The current pose does not align with the cue's requirement for a sincere apology. |
| 90 | start_action_forward_snap | 2 | False | The current pose does not align well with the target cue. The arm should be pulled back sharply before snapping forward. |
| 92 | retry_action_reset_circle | 3 | False | The current pose does not align well with the target cue. The robot should be facing sideways to perform a lateral swipe. |
| 94 | abandon_action_drop_turn | 5 | False | The current pose does not align well with the cue's requirement of abruptly halting and turning away. |
| 95 | prepare_action_raise_hold | 5 | False | The current pose does not align well with the target cue. The arm is extended forward, but the grip is horizontal, which may not convey the  |
| 97 | hesitation_pause_hold | 3 | False | The current pose does not align well with the cue's requirement for a tentative forward reach followed by a sharp recoil. |
| 104 | scale_expand | 6 | False | The current pose does not align with the cue's requirement of extending the arm to the left. |
| 106 | uncertainty_wave | 6 | False | The current pose does not align well with the cue's requirement of an upward-facing palm for uncertainty. |
| 107 | exploration_reach | 5 | False | The current pose does not align well with the exploration_reach cue. The robot should start with a hesitant side-to-side check before extend |
| 108 | interrupt_hard | 2 | False | The current pose does not align with the intended cue. The robot should be facing forward to execute a sharp recoil and explosive thrust. |
| 109 | interrupt_soft | 3 | False | The current pose does not align well with the desired cue. The robot should be facing forward (dir=front) but the gripper orientation should |
| 114 | contain_space | 3 | False | The current pose does not align well with the target cue. The robot should be facing towards the viewer (front) to effectively gather intent |
| 115 | expand_space | 3 | False | The current pose does not align well with the cue 'expand_space'. The robot should start by drawing its arm inward and then sweeping outward |
| 116 | compress_space | 5 | False | The current pose does not align well with the cue 'compress_space'. The robot should be positioned sideways to stage the hand to the side. |
| 117 | balance_scale | 6 | False | The current pose does not align with the cue's requirement of moving to the side into a palm-up pose. |
| 118 | push_pull_dual | 3 | False | The current pose does not align well with the cue's requirement for pushing and pulling motions. |
| 121 | weight_shift_forward | 5 | False | The current pose does not align with the target cue. The robot should start by recoiling slightly backward and down to gather momentum. |
| 122 | weight_shift_backward | 5 | False | The current pose does not align with the target cue. The robot should be positioned to convey a forward 'take' followed by a sharp recoil. |
| 123 | torso_rotate_away | 5 | False | The current pose does not align well with the target cue. The robot should be facing away from the viewer to indicate a backward recoil. |
| 124 | torso_rotate_toward | 5 | False | The current pose does not align with the desired cue. The robot should be facing the user (front) but the gripper orientation needs to be ve |
| 125 | lean_side_doubt | 5 | False | The current pose does not align well with the cue 'lean_side_doubt'. The robot should be positioned to lean to the side, which requires a di |
| 127 | expand_chest | 6 | False | The current pose does not align with the cue's requirement of placing the hand on the chest. |
| 128 | eye_widen_surprise | 5 | False | The current pose does not align well with the target cue. The robot should be positioned to express surprise more effectively. |
| 130 | gaze_shift_avoid | 5 | False | The current pose does not align with the desired gaze shift and avoidance cue. |
