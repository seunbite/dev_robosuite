# Pose Verification — single selected tile (Gemini)

- model: `gemini-2.5-pro`
- config: `/Users/sb/Downloads/workspace/dev_robosuite/data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot100_batch21.json`
- group tiles: `/Users/sb/Downloads/workspace/dev_robosuite/data/results/visualize/pose_groups_12`
- tile picks: `/Users/sb/Downloads/workspace/dev_robosuite/data/results/verify/pose_tile_pick_by_group.json`

| idx | cue | tile | pose ok? | note |
|---:|---|---:|---|---|
| 62 | self_hug | 3 | True | The labels dir=front and gripper_orientation=vertical are correct for the image. The gripper's pointing axis is directed forward towards the |
| 67 | rub_arms_cold | 5 | False | The current labels `dir=front` and `gripper_orientation=horizontal` are a reasonable interpretation of the pose shown in the image. The grip |
| 69 | cross_arms_defensive | 5 | False | The provided labels are incorrect for the image. The image shows the robot's end-effector pointing primarily to the robot's right, so the di |
| 72 | palms_together_thank_you | 2 | False | The labels dir=down and gripper_orientation=vertical correctly describe the image. The end-effector's pointing axis (wrist to fingertips) is |
| 75 | arms_open_welcome | 5 | True | The labels 'dir=back' and 'gripper_orientation=vertical' are appropriate. The end effector is pointing back towards the robot's central colu |
| 76 | stretch_arms_up | 5 | False | The current pose with `dir=front` is not suitable for a cue named 'stretch_arms_up'. The primary and most expressive part of this gesture is |
| 85 | dominance_high_open_hold | 5 | False | The current pose labels are incorrect. The robot's arm is held high and wide to its side, meaning the end-effector's pointing axis is predom |
| 94 | abandon_action_drop_turn | 5 | True | The pose is correctly labeled as 'dir=front' and 'gripper_orientation=horizontal'. The end-effector is pointing towards the viewer, and the  |
| 109 | interrupt_soft | 3 | True | The current labels are appropriate. The robot's end-effector is pointing directly forward (+x) towards the viewer, correctly labeled as `dir |
| 113 | hold_steady | 2 | False | The 'dir=down' label is correct as the end effector's pointing axis is directed towards the floor. However, the 'gripper_orientation=vertica |
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
