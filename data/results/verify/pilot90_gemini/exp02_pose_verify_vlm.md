# Pose Verification — single selected tile

- backend: `gemini`
- model: `gemini-2.5-flash`
- config: `data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json`
- group tiles: `data/results/visualize/pose_groups_12`
- tile picks: `data/results/verify/pose_tile_pick_by_group.json`

| idx | cue | tile | pose ok? | note |
|---:|---|---:|---|---|
| 8 | fist_pump | None | False | The current labels 'dir=front, gripper_orientation=horizontal' are inappropriate for a 'fist pump'. This gesture is a celebratory upward thr |
| 9 | flex_bicep | None | False | The current labels (dir=down, gripper_orientation=vertical) are incorrect. A bicep flex gesture concludes with an upward curl, bringing the  |
| 11 | self_scratch_head_confused | None | True | The 'dir=left' and 'gripper_orientation=vertical' labels are perfect for this cue. 'left' correctly places the hand to the side of the 'head |
| 12 | facepalm | None | False | The current pose `dir=front` is fundamentally incorrect for a 'facepalm'. This gesture requires the robot's 'palm' to move towards its 'face |
| 14 | point_you | None | False | The current `dir=down` is incorrect. A 'point_you' gesture requires pointing forward (+x) at the viewer, not at the floor (-z). |
| 15 | beckon_come_here | None | False | The `dir=front` is appropriate for reaching towards the viewer. However, `gripper_orientation=horizontal` results in a palm-up or palm-down  |
| 16 | stop_palm_out | None | False | The current pose `dir=down, gripper_orientation=vertical` is incorrect. A 'stop_palm_out' gesture is directed at the viewer, requiring the e |
| 20 | rub_eye_tired | None | True | The 'left' direction correctly points the gripper towards the robot's own 'head' space, and the 'vertical' gripper orientation is suitable f |
| 25 | cheers_toast | None | True | The 'up' direction and 'horizontal' gripper orientation are perfectly suited for holding a glass upright, which is the core premise of a toa |
| 26 | highfive_invite | None | True | The 'dir=front' and 'gripper_orientation=horizontal' labels are correct for this cue. The hand is presented towards the user (front), and th |
| 30 | pat_self_back | None | False | The current `dir=front` is incorrect. To pat its own back, the robot's gripper must point towards its body (`dir=back`). The `gripper_orient |
| 32 | throat_cut_stop | None | False | The current pose (dir=front, gripper_orientation=horizontal) is incorrect. A 'throat cut' gesture is a lateral (side-to-side) motion across  |
| 35 | salute | None | False | The current label 'dir=front' is incorrect for a salute. The hand should be pointing upwards and outwards, not directly at the viewer. The ' |
| 36 | visor_search | None | True | The pose is appropriate. For a visor gesture, the hand is held flat across the forehead, so the pointing axis (wrist to fingertips) is corre |
| 39 | request_turn | None | True | The labels dir=up and gripper_orientation=horizontal correctly describe an 'open palm up' pose. This is the standard and most intuitive orie |
| 40 | guiding_arm | None | True | The 'front' direction is perfect for the initial 'extends its hand forward in an offering pose' part of the cue. The 'vertical' gripper orie |
| 45 | slow_down_request_palm_down | None | True | The 'dir=down' and 'gripper_orientation=horizontal' labels are perfect for a 'palm down' gesture, accurately representing a hand pressing do |
| 61 | air_quotes | None | False | The current `dir=front` is incorrect. For an 'air quotes' gesture, the hand (gripper) should be pointing upwards, not directly at the viewer |
| 64 | tilt_head_curious | None | False | The current labels `dir=front, gripper_orientation=horizontal` are inappropriate. The cue 'tilt_head_curious' is a body language gesture usi |
| 66 | shh_gesture | None | False | The cue 'shh_gesture' requires the gripper to be placed in front of the robot's 'mouth', meaning its pointing axis should be `dir=back`. The |
| 7 | raise_fists_victory | 3 | False | The current pose labels are incorrect. The robot's end effector is pointing almost directly upward toward the ceiling (+z direction), not 'f |
| 10 | shrug_idk | 9 | True | The end-effector's pointing axis (wrist to fingertips) is oriented towards the ceiling (+z), making 'dir=up' correct. The line connecting th |
| 13 | point_self | 5 | False | The current pose labels are incorrect. For the 'point_self' gesture, the robot's end-effector (EE) points toward its own body/chest. This co |
| 17 | pat_stomach | 5 | False | The `dir` label is incorrect. The gripper is positioned to pat the robot's own 'stomach', meaning its pointing axis is directed back towards |
| 18 | fan_face_hot | 3 | True | The robot's end-effector is pointing towards its left side (+y axis), making the direction 'left' correct. The line connecting the gripper's |
| 19 | cover_yawn_sleepy | 6 | False | The pose label 'dir=down' is inappropriate for the cue 'cover_yawn_sleepy'. Covering a mouth requires bringing the hand up to the face area, |
| 21 | tap_temple_think | 5 | False | The current pose labels are incorrect. The robot's end-effector is pointing towards its own left side to tap its temple, not 'front' towards |
| 22 | circle_temple_crazy | 5 | False | The pose is for a 'circle temple' gesture. The end-effector is correctly placed at the side of where the head would be. However, its pointin |
| 23 | lean_forward_interest | 5 | False | The gripper's pointing axis is predominantly directed upwards (+z), not towards the front (+x) as labeled. The gripper orientation is correc |
| 24 | cover_mouth_gasp | 5 | False | The provided labels (dir=front, gripper_orientation=horizontal) are incorrect. The image shows the robot's end-effector pointing back toward |
| 27 | highfive_imitation | 9 | False | The current pose (dir=up, gripper_orientation=horizontal) is incorrect for a high-five gesture. The robot is pointing its hand directly at t |
| 28 | fistbump_invite | 2 | False | The current pose is incorrect for a 'fistbump_invite'. The gripper is pointing down at the floor, which is not a natural or intuitive way to |
| 37 | yield_turn | 5 | False | The EE pointing axis is directed towards the robot's left (+y), not towards its back (-x). The line between the gripper jaws is oriented ver |
| 41 | protective_arm_out | 5 | False | The current labels are inappropriate for the cue 'protective_arm_out'. While 'dir=front' correctly captures the forward-facing nature of a b |
| 50 | firm_accept_forward_reach | 5 | False | The 'front' direction is correct as the end-effector points towards the viewer. However, the 'horizontal' gripper orientation is incorrect.  |
| 59 | nod_yes | 5 | False | The provided `dir=front` is incorrect. The gripper's pointing axis (wrist to fingertips) is predominantly directed `up` (+z) and slightly ba |
| 60 | shake_head_no | 3 | True | The robot's end-effector is pointing directly forward (+x) toward the viewer, which correctly corresponds to `dir=front`. The jaws of the gr |
| 74 | hands_up_surrender | 5 | False | The 'dir=front' is correct as the palm faces the viewer in a classic surrender pose. However, the 'gripper_orientation=horizontal' is incorr |
| 104 | scale_expand | 6 | False | The labels `dir=down` and `gripper_orientation=horizontal` technically match the rendered image, where the arm is raised high and the grippe |
| 31 | draw_circle_repeat | 3 | True | The pose is correctly labeled as dir=front and gripper_orientation=vertical. The end-effector points directly at the viewer, and the jaw-ope |
| 44 | clarification_request_palm_tilt | 5 | False | The current pose labels (dir=front, gripper_orientation=horizontal) are not the best fit for the cue or the image. The image itself shows th |
| 46 | speed_up_circular_cue | 3 | True | The pose is correctly labeled as dir=front and gripper_orientation=vertical. The end-effector points directly at the viewer, and the gripper |
| 49 | polite_decline_small_push | 3 | False | The current labels are a correct assessment of the robot's pose in the image. The end-effector's pointing axis is directed forward (+x) towa |
| 51 | hesitant_accept_half_reach | 3 | True | The robot's end-effector is pointing directly towards the viewer, which correctly corresponds to `dir=front`. The gripper's jaws are oriente |
| 52 | cancel_previous_offer_retract | 5 | False | The cue 'cancel_previous_offer_retract' describes a motion that begins with extending an offer, then retracting it. The current starting pos |
| 53 | conditional_accept_pause_then_reach | 3 | True | The pose (dir=front, gripper_orientation=vertical) is correct and highly appropriate. The robot is pointing towards the viewer, ready for in |
| 54 | final_decision_down_press | 6 | False | The pose labels (dir=down, gripper_orientation=horizontal) accurately describe the image, where the robot's end-effector points straight dow |
| 55 | resume_speaking_pull | 5 | True | The pose is correctly labeled. The gripper's pointing axis is directed forward towards the viewer (+x), making `dir=front` accurate. The lin |
| 56 | emphasize_point_downbeat | 3 | False | The current labels (dir=front, gripper_orientation=vertical) accurately describe the pose shown in the image. However, this pose is inapprop |
| 57 | unsure_response_hand_hover_stop | 5 | False | The `dir=front` label is correct as the gripper is pointing towards the viewer. However, the `gripper_orientation=horizontal` label is incor |
| 65 | raise_index_one_moment | 5 | False | The provided image shows the robot's end effector pointing almost directly upwards (+z), which is a perfect starting pose for a 'raise index |
| 90 | start_action_forward_snap | 2 | True | The robot's arm is pointing straight up (`dir=up`), which is an excellent preparatory pose for a downward and forward slashing motion. This  |
| 92 | retry_action_reset_circle | 3 | True | The labels dir=front and gripper_orientation=vertical are correct. The end-effector points directly towards the viewer, and the gripper jaws |
| 95 | prepare_action_raise_hold | 5 | False | The current labels 'dir=front' and 'gripper_orientation=horizontal' are incorrect for the pose shown in the image. The robot's end-effector  |
| 97 | hesitation_pause_hold | 3 | True | The labels `dir=front` and `gripper_orientation=vertical` are correct. The robot's end-effector is pointing directly forward (+x) towards th |
| 106 | uncertainty_wave | 6 | True | The pose with dir=up and gripper_orientation=horizontal is perfectly appropriate. It correctly positions the gripper as an 'upward-facing pa |
| 108 | interrupt_hard | 2 | False | The current labels are `dir=down` and `gripper_orientation=vertical`. The image shows the end-effector pointing towards the floor, making `d |
| 62 | self_hug | 3 | True | The labels dir=front and gripper_orientation=vertical are correct for the image. The gripper's pointing axis is directed forward towards the |
| 67 | rub_arms_cold | 5 | False | The current labels `dir=front` and `gripper_orientation=horizontal` are a reasonable interpretation of the pose shown in the image. The grip |
| 69 | cross_arms_defensive | 5 | False | The provided labels are incorrect for the image. The image shows the robot's end-effector pointing primarily to the robot's right, so the di |
| 72 | palms_together_thank_you | 2 | False | The labels dir=down and gripper_orientation=vertical correctly describe the image. The end-effector's pointing axis (wrist to fingertips) is |
| 75 | arms_open_welcome | 5 | True | The labels 'dir=back' and 'gripper_orientation=vertical' are appropriate. The end effector is pointing back towards the robot's central colu |
| 76 | stretch_arms_up | 5 | False | The current pose with `dir=front` is not suitable for a cue named 'stretch_arms_up'. The primary and most expressive part of this gesture is |
| 85 | dominance_high_open_hold | 5 | False | The current pose labels are incorrect. The robot's arm is held high and wide to its side, meaning the end-effector's pointing axis is predom |
| 94 | abandon_action_drop_turn | 5 | True | The pose is correctly labeled as 'dir=front' and 'gripper_orientation=horizontal'. The end-effector is pointing towards the viewer, and the  |
| 109 | interrupt_soft | 3 | True | The current labels are appropriate. The robot's end-effector is pointing directly forward (+x) towards the viewer, correctly labeled as `dir |
| 114 | contain_space | 3 | True | The labels `dir=left` and `gripper_orientation=vertical` are correct. The end-effector's pointing axis is directed towards the robot's left  |
| 115 | expand_space | 3 | True | The labels dir=front and gripper_orientation=vertical accurately describe the pose. The end-effector points directly towards the viewer (+x) |
| 116 | compress_space | 5 | False | The current pose labels `dir=front`, `gripper_orientation=horizontal` are not appropriate. The cue 'compress_space' describes creating a 'wa |
| 117 | balance_scale | 6 | True | The labels are correct. The end-effector's pointing axis is directed upwards along the +z axis, making 'dir=up' accurate. The line between t |
| 118 | push_pull_dual | 3 | True | The labels are correct. The end-effector points straight at the viewer, matching `dir=front`. The gripper jaws are oriented up-and-down, whi |
| 121 | weight_shift_forward | 5 | True | The pose labels (dir=front, gripper_orientation=horizontal) are accurate. The end-effector's pointing axis is directed towards the viewer (+ |
| 122 | weight_shift_backward | 5 | False | The current labels (dir=front, gripper_orientation=horizontal) do not match the robot's pose in the image. The end effector's pointing axis  |
| 123 | torso_rotate_away | 5 | True | The current pose (dir=front, gripper_orientation=horizontal) is appropriate. The arm is held in a neutral, forward-facing position, which pr |
| 124 | torso_rotate_toward | 5 | False | The labeled pose (dir=front, gripper_orientation=horizontal) is incorrect for the image provided. The image shows the robot's end-effector p |
| 125 | lean_side_doubt | 5 | False | The current labels (dir=front, gripper_orientation=horizontal) are a significant mismatch for the pose shown in the image. The image actuall |
| 33 | chin_flick_dismiss | 5 | False | The current pose labels (dir=front, gripper_orientation=horizontal) indicate the gripper's pointing axis is directed towards the human viewe |
| 34 | cover_ears_too_loud | 5 | True | The current pose with 'dir=back' means the end-effector is pointing towards the robot's body, which is ideal for interacting with its own 'e |
| 42 | escorting_gesture | 6 | True | The current pose with `dir=up` and `gripper_orientation=horizontal` accurately describes the robot's initial position in the image, where th |
| 47 | wait_for_me_backward_palm | 5 | False | The robot's end effector (gripper) in the image is pointing generally towards the viewer, consistent with 'dir=front'. The gripper jaws are  |
| 48 | continue_on_forward_roll | 5 | True | The robot's end-effector is clearly pointing forward (+x), aligning with the 'dir=front' label, which is suitable for initiating a 'continue |
| 58 | prevent_fall_support_reach | 5 | True | The robot's arm is extended forward, with the end effector (gripper) pointing predominantly towards the human viewer, which is consistent wi |
| 78 | disagreement_cross_then_release | 5 | True | The current pose with 'dir=front' and 'gripper_orientation=horizontal' is appropriate. 'dir=front' provides a suitable neutral starting poin |
| 82 | frustration_down_throw | 6 | True | The robot's end effector is clearly pointing upwards (+z), with the jaw opening oriented horizontally (left-right) when projected onto the p |
| 84 | submission_lowered_open_palm | 2 | False | The robot's end effector (gripper) is currently pointing towards the floor (dir=down), with its jaw opening oriented vertically. The cue des |
| 86 | apology_forward_lowered_hold | 2 | False | The current pose's 'dir=down' means the end-effector's pointing axis is directed towards the floor (-z). The 'gripper_orientation=vertical'  |
| 107 | exploration_reach | 5 | True | The robot's end effector is pointing forward (+x), which correctly corresponds to 'dir=front'. The gripper jaws are horizontally aligned, ma |
| 127 | expand_chest | 6 | False | The current pose shows the robot's gripper pointing downwards (dir=down) with the jaw opening horizontally. For the 'expand_chest' cue, the  |
| 128 | eye_widen_surprise | 5 | True | The robot's end-effector is pointing forward (+x) which aligns with 'dir=front'. The gripper jaws, when projected onto the yz-plane, appear  |
| 130 | gaze_shift_avoid | 5 | True | The robot's end effector (gripper) is clearly pointing towards the viewer (+x direction), making 'front' an appropriate label for its domina |
