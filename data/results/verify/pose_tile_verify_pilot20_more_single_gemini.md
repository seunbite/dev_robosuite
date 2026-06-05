# Pose Verification — single selected tile (Gemini)

- model: `gemini-2.5-pro`
- config: `data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot20_more.json`
- group tiles: `data/results/visualize/pose_groups_12`
- tile picks: `data/results/verify/pose_tile_pick_by_group.json`

| idx | cue | tile | pose ok? | note |
|---:|---|---:|---|---|
| 7 | raise_fists_victory | 3 | False | The current pose labels are incorrect. The robot's end effector is pointing almost directly upward toward the ceiling (+z direction), not 'f |
| 10 | shrug_idk | 9 | True | The end-effector's pointing axis (wrist to fingertips) is oriented towards the ceiling (+z), making 'dir=up' correct. The line connecting th |
| 11 | scratch_head_confused | 5 | False | The current labels 'dir=front, gripper_orientation=horizontal' are incorrect. The robot's end-effector is pointing towards its own right sid |
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
