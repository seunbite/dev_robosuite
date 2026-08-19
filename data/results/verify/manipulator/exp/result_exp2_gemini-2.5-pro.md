# Pose Verification — single selected tile

- backend: `gemini`
- model: `gemini-2.5-pro`
- config: `/Users/sb/Downloads/workspace/dev_robosuite/data/results/motion_configs/manipulator/exp/result_exp1_gemini-2.5-pro.json`
- group tiles: `/Users/sb/Downloads/workspace/dev_robosuite/data/results/visualize/pose_groups_12`
- tile picks: `/Users/sb/Downloads/workspace/dev_robosuite/data/results/verify/pose_tile_pick_by_group.json`

| idx | cue | tile | pose ok? | note |
|---:|---|---:|---|---|
| 7 | raise_fists_victory | 2 | False | The direction `dir=up` is appropriate, as the pose shown is the peak of the 'raise fist' motion, with the end-effector pointing towards the  |
| 8 | fist_pump | 3 | False | The labels 'dir=front' and 'gripper_orientation=vertical' correctly describe the image, as the gripper's pointing axis is directed forward t |
| 9 | flex_bicep | 3 | False | The labels dir=front and gripper_orientation=vertical accurately describe the pose in the image, where the gripper points toward the viewer  |
| 10 | shrug_idk | 6 | False | The provided pose (dir=up) shows the arm at the peak of a shrug, not the starting position. The cue description explicitly states the gestur |
| 11 | self_scratch_head_confused | 2 | False | The current labels dir=down and gripper_orientation=vertical are a correct description of the pose in the image. The end-effector's pointing |
| 12 | facepalm | 6 | False | The current labels (dir=up, gripper_orientation=horizontal) accurately describe the robot's pose in the image. However, this pose is not sui |
| 13 | point_self | 3 | True | The pose is correctly labeled as dir=front and gripper_orientation=vertical. The end-effector's pointing axis is directed forward toward the |
| 14 | point_you | 5 | False | The labels 'dir=front' and 'gripper_orientation=horizontal' are a correct description of the pose shown in the image. However, this pose its |
| 15 | beckon_come_here | 2 | False | The current pose with `dir=down` has the gripper pointing at the floor. A 'beckon' gesture is typically performed with the 'palm' facing up  |
| 16 | stop_palm_out | 5 | False | The 'dir' label 'front' is correct as the gripper is pointing towards the viewer. However, the 'gripper_orientation' label 'horizontal' is i |
| 17 | pat_stomach | 5 | False | The current labels (dir=front, gripper_orientation=horizontal) are incorrect for the cue 'pat_stomach'. The cue requires the robot to place  |
| 18 | fan_face_hot | 2 | False | The current pose `dir=down` shows the hand in a low, resting position. The 'fan face' gesture requires the hand to be raised near the robot' |
| 19 | cover_yawn_sleepy | 2 | False | The `dir=down` is correct as the gripper's pointing axis is oriented towards the floor. However, the `gripper_orientation=vertical` is incor |
| 20 | rub_eye_tired | 6 | False | The current pose (dir=up, gripper_orientation=horizontal) is incorrect. The end-effector is pointing straight up, which is a very unnatural  |
| 21 | tap_temple_think | 3 | False | The current pose (dir=front, gripper_orientation=vertical) points the robot's hand directly at the viewer. For a 'tap temple think' gesture, |
| 22 | circle_temple_crazy | 3 | True | The pose with `dir=front` and `gripper_orientation=vertical` is an excellent starting point. It perfectly matches the cue's initial action o |
| 23 | lean_forward_interest | 5 | False | The 'dir=front' label is incorrect. Based on the image, the end-effector's pointing axis (wrist to fingertips) is predominantly oriented upw |
| 24 | cover_mouth_gasp | 2 | False | The current pose `dir=down` is inappropriate. For a 'cover mouth' gesture, the hand must travel upward to the 'mouth' area of the robot's bo |
| 25 | cheers_toast | 2 | False | The pose shown in the image (dir=up, gripper_orientation=horizontal) is not appropriate for a 'cheers_toast' cue. While 'dir=up' is correct  |
| 26 | highfive_invite | 5 | False | The provided image shows the robot's end-effector pointing primarily upwards (+z axis), with the gripper jaws oriented horizontally. The cur |
| 27 | highfive_imitation | 5 | False | The current labels 'dir=front' and 'gripper_orientation=horizontal' correctly describe the final target pose of a classic high-five (palm fo |
| 28 | fistbump_invite | 3 | True | The pose is perfectly suited for a fist bump invitation. The end-effector points directly 'front' toward the user, and the 'vertical' grippe |
| 30 | pat_self_back | 5 | False | The image shows the gripper's pointing axis is predominantly in the +z direction (up), not +x (front) as labeled. The robot's 'fingertips' a |
| 31 | draw_circle_repeat | 3 | True | The labels dir=front and gripper_orientation=vertical accurately describe the pose. The robot's end-effector points directly towards the vie |
| 32 | throat_cut_stop | 5 | False | The provided labels (dir=front, gripper_orientation=horizontal) are not appropriate for a 'throat cut' gesture. The image actually shows a p |
| 33 | chin_flick_dismiss | 5 | False | The provided pose labels `dir=front` and `gripper_orientation=horizontal` are not appropriate for the 'chin_flick_dismiss' cue. The gesture  |
| 34 | cover_ears_too_loud | 2 | False | The current pose has the gripper pointing down, which is not suitable for a gesture that involves covering an 'ear' on the side of the robot |
| 35 | salute | 6 | True | The current pose is correctly labeled as dir=up and gripper_orientation=horizontal. The end-effector's pointing axis is clearly directed tow |
| 36 | visor_search | 5 | False | The current pose labels (dir=front, gripper_orientation=horizontal) are incorrect for the cue 'visor_search'. A visor gesture requires the h |
| 37 | yield_turn | 2 | False | The current pose is not appropriate for a 'yield_turn' gesture. While the 'dir' label is correct for the image (pointing left), the 'gripper |
| 39 | request_turn | 5 | False | The current labels are `dir=front`, `gripper_orientation=horizontal`. However, the robot in the image is clearly pointing its gripper upward |
| 40 | guiding_arm | 6 | False | The labels dir=up and gripper_orientation=horizontal are a correct description of the pose shown in the image. The gripper is clearly pointi |
| 41 | protective_arm_out | 5 | True | The current pose with dir=back and gripper_orientation=vertical is highly appropriate. The cue 'protective_arm_out' describes a motion that  |
| 42 | escorting_gesture | 4 | False | The labeled direction `dir=back` is incorrect. The image clearly shows the robot's end-effector pointing axis is directed upwards toward the |
| 44 | clarification_request_palm_tilt | 5 | False | The provided labels (dir=front, gripper_orientation=horizontal) are not appropriate. While 'gripper_orientation=horizontal' is correct, the  |
| 45 | slow_down_request_palm_down | 5 | True | The pose with dir=front and gripper_orientation=horizontal is appropriate. This configuration places the gripper's 'palm' facing downwards,  |
| 46 | speed_up_circular_cue | 3 | True | The labels `dir=front` and `gripper_orientation=vertical` are accurate. The robot's end-effector is pointing directly forward at the viewer, |
| 47 | wait_for_me_backward_palm | 5 | True | The labels `dir=front` and `gripper_orientation=horizontal` are appropriate. The end-effector points forward toward the viewer (+x), which c |
| 48 | continue_on_forward_roll | 3 | False | The current labels (dir=left, gripper_orientation=vertical) accurately describe the pose shown in the image. The robot's end-effector is poi |
| 49 | polite_decline_small_push | 3 | False | The 'front' direction is correct for a push-away gesture directed at the viewer. However, the 'vertical' gripper orientation is inappropriat |
| 50 | firm_accept_forward_reach | 5 | True | The current pose labels (dir=front, gripper_orientation=horizontal) are correct. The end effector's pointing axis is directed forward toward |
| 51 | hesitant_accept_half_reach | 3 | False | The 'front' direction is correct for a reaching gesture towards the user. However, the 'vertical' gripper orientation is not appropriate for |
| 52 | cancel_previous_offer_retract | 5 | False | The provided labels (dir=front, gripper_orientation=horizontal) describe a forward-extended, handshake-like pose. While this is a plausible  |
| 53 | conditional_accept_pause_then_reach | 5 | True | The pose with dir=back and gripper_orientation=vertical correctly places the end-effector in a 'hand-on-chin' position, which is the perfect |
| 54 | final_decision_down_press | 6 | False | The current pose labels are incorrect. The image clearly shows the robot's end-effector pointing straight up towards the ceiling (+z axis),  |
| 55 | resume_speaking_pull | 3 | False | The labels 'dir=front' and 'gripper_orientation=vertical' accurately describe the pose shown in the image. The end-effector is pointing forw |
| 56 | emphasize_point_downbeat | 5 | False | The 'dir=front' label is correct as the end-effector's pointing axis is directed towards the viewer. However, the 'gripper_orientation=horiz |
| 57 | unsure_response_hand_hover_stop | 5 | False | The provided labels (dir=front, gripper_orientation=horizontal) are incorrect for the image. The image shows the robot pointing to its right |
| 58 | prevent_fall_support_reach | 3 | True | The labels dir=front and gripper_orientation=vertical accurately describe the pose. This high, coiled starting position is an excellent repr |
| 59 | nod_yes | 3 | True | The labels dir=front and gripper_orientation=vertical are correct. The end-effector is pointing directly towards the viewer (+x axis), and t |
| 60 | shake_head_no | 2 | False | The current pose with dir=down shows the robot pointing at the floor. The cue 'shake_head_no' is described as starting from a 'stop' positio |
| 61 | air_quotes | 5 | False | The current `dir` is labeled 'front', but the image clearly shows the robot's end-effector pointing upwards towards the ceiling (+z). This u |
| 62 | self_hug | 2 | False | The 'down' direction is a plausible but not ideal starting point for a hug, as it requires a significant lift before the main cross-body mot |
| 64 | tilt_head_curious | 3 | True | The labels dir=front and gripper_orientation=vertical are correct for the image. The robot's end-effector is pointing directly at the viewer |
| 65 | raise_index_one_moment | 3 | False | The current pose with `dir=front` and `gripper_orientation=vertical` is a 'pointing at you' gesture. The cue `raise_index_one_moment` is uni |
| 66 | shh_gesture | 5 | False | The `dir=front` is appropriate, as the gripper's pointing axis is directed towards the viewer. However, the `gripper_orientation` is incorre |
| 67 | rub_arms_cold | 5 | False | The current pose is labeled as `dir=front`, `gripper_orientation=horizontal`. Visually, the end effector is pointing more towards the robot' |
| 69 | cross_arms_defensive | 4 | True | The pose is correctly labeled as `dir=right` and `gripper_orientation=vertical`. The end effector points to the robot's right, and the gripp |
| 72 | palms_together_thank_you | 5 | False | The provided labels (dir=front, gripper_orientation=horizontal) are incorrect for the image. The image shows the gripper's pointing axis (wr |
| 74 | hands_up_surrender | 6 | True | The current pose (dir=down, gripper_orientation=horizontal) represents a neutral, 'hands at rest' position. This is a perfectly suitable sta |
| 75 | arms_open_welcome | 2 | False | The current pose with the gripper pointing down is fundamentally wrong for an 'arms_open_welcome' cue, which requires an open, outward, and  |
| 76 | stretch_arms_up | 2 | True | The 'dir=down' label represents an excellent starting point for a 'stretch up' gesture, as it provides a low, relaxed initial state from whi |
| 78 | disagreement_cross_then_release | 5 | True | The labels dir=back and gripper_orientation=vertical are appropriate. The end-effector is pointing back towards the robot's torso, and the g |
| 82 | frustration_down_throw | 6 | True | The pose is correctly labeled as dir=down and gripper_orientation=horizontal. The pointing axis (wrist to fingertips) is oriented towards th |
| 84 | submission_lowered_open_palm | 5 | False | The provided labels (dir=front, gripper_orientation=horizontal) describe the intended final pose of a submissive gesture (offering an open p |
| 85 | dominance_high_open_hold | 2 | True | The current pose (dir=down, gripper_orientation=vertical) is an excellent starting point. It represents a low, neutral, or contracted state, |
| 86 | apology_forward_lowered_hold | 5 | False | The starting pose itself, being high and forward, could serve as the peak of a 'recoil' before the main apologetic slumping motion. However, |
| 90 | start_action_forward_snap | 3 | False | The current pose labels `dir=front` and `gripper_orientation=vertical` are technically correct for the image shown. The end effector's point |
| 92 | retry_action_reset_circle | 3 | True | The pose (dir=front, gripper_orientation=vertical) is perfectly suited. The 'front' direction allows the robot to draw the 'retry' circle di |
| 94 | abandon_action_drop_turn | 3 | True | The current pose labels (dir=front, gripper_orientation=vertical) are a correct representation of the robot's pose in the image. This pose i |
| 95 | prepare_action_raise_hold | 5 | True | The labels `dir=front` and `gripper_orientation=horizontal` are appropriate for the cue 'prepare_action_raise_hold'. A front-facing directio |
| 97 | hesitation_pause_hold | 3 | True | The labels dir=front and gripper_orientation=vertical are correct for the image and highly appropriate for the cue. This pose serves as a cl |
| 104 | scale_expand | 5 | False | The current pose labels (dir=front, gripper_orientation=horizontal) accurately describe the robot's configuration in the image. However, thi |
| 106 | uncertainty_wave | 3 | True | The labels `dir=front` and `gripper_orientation=vertical` are correct. The end-effector is pointing forward towards the viewer (+x), and the |
| 107 | exploration_reach | 3 | True | The labels 'dir=front' and 'gripper_orientation=vertical' accurately describe the pose shown. The end-effector is pointing directly towards  |
| 108 | interrupt_hard | 3 | False | The pose is correctly identified as dir=front and gripper_orientation=vertical. The end-effector is pointing directly at the viewer (+x), an |
| 109 | interrupt_soft | 6 | True | The pose labels are correct. The gripper's pointing axis is predominantly aligned with the world +z axis, making 'dir=up' accurate. The line |
| 114 | contain_space | 6 | False | The labels dir=up and gripper_orientation=horizontal correctly describe the pose shown in the image. The end-effector's pointing axis (wrist |
| 115 | expand_space | 5 | True | The pose is correctly labeled as `dir=back` and `gripper_orientation=vertical`. The gripper's pointing axis is directed back towards the rob |
| 116 | compress_space | 3 | False | The 'front' direction is correct as the gesture involves pushing forward toward the viewer. However, the 'vertical' gripper orientation is n |
| 117 | balance_scale | 6 | False | The provided image correctly matches the labels dir=up and gripper_orientation=horizontal. However, this pose is not suitable for a 'balance |
| 118 | push_pull_dual | 3 | True | The pose is correctly labeled as `dir=front` and `gripper_orientation=vertical`. The end effector's pointing axis is directed forward toward |
| 121 | weight_shift_forward | 2 | True | The labels dir=down and gripper_orientation=vertical accurately describe the robot's pose. The end effector's pointing axis is directed towa |
| 122 | weight_shift_backward | 3 | True | The labels 'dir=front' and 'gripper_orientation=vertical' accurately describe the pose. The end-effector is pointing directly at the viewer  |
| 123 | torso_rotate_away | 5 | False | The pose labels `dir=back` and `gripper_orientation=vertical` accurately describe the robot's configuration in the image. The end effector's |
| 124 | torso_rotate_toward | 2 | False | The 'dir=left' is correct as the gripper points along the robot's left (+y) axis. However, 'gripper_orientation=horizontal' is incorrect. Th |
| 125 | lean_side_doubt | 2 | False | The current `dir=down` label is correct as the end-effector's pointing axis is directed towards the floor. However, the `gripper_orientation |
| 127 | expand_chest | 5 | True | The pose is correctly labeled and highly appropriate for the cue. The gripper points back toward the robot's body (`dir=back`), which perfec |
| 128 | eye_widen_surprise | 3 | False | The current pose with `dir=front` is correct, as the robot is expressing surprise towards the viewer. However, the `gripper_orientation=vert |
| 130 | gaze_shift_avoid | 3 | True | The pose labels 'dir=front' and 'gripper_orientation=vertical' are correct. The end-effector is pointing directly towards the viewer (+x axi |
