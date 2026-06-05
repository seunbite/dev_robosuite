# Pose Verification — single selected tile (Gemini)

- model: `gemini-2.5-pro`
- config: `data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot10.json`
- group tiles: `data/results/visualize/pose_groups_12`
- tile picks: `data/results/verify/pose_tile_pick_by_group.json`

| idx | cue | tile | pose ok? | note |
|---:|---|---:|---|---|
| 8 | fist_pump | 5 | False | The robot's end-effector is pointing primarily upwards toward the ceiling (+z axis), which is the characteristic peak of a fist pump gesture |
| 9 | flex_bicep | 2 | False | The analysis of the image confirms the labels `dir=down` and `gripper_orientation=vertical`. The end-effector's pointing axis is clearly dir |
| 11 | self_scratch_head_confused | 3 | True | The pose is correctly labeled. The end-effector's pointing axis is predominantly directed toward the robot's left (+y), making 'dir=left' ap |
| 12 | facepalm | 5 | False | The current labels (dir=front, gripper_orientation=horizontal) are incorrect for a 'facepalm' gesture. The pose shows the gripper pointing b |
| 14 | point_you | 2 | False | The current pose shows the robot pointing at the floor ('down'), which is fundamentally incorrect for a gesture cue named 'point_you'. The g |
| 15 | beckon_come_here | 5 | False | The provided image shows the robot's end-effector pointing predominantly upwards towards the ceiling (+z axis), not towards the viewer (+x a |
| 16 | stop_palm_out | 2 | False | The current pose with dir=down is incorrect. A 'stop_palm_out' gesture must be directed 'front' towards the person it is addressing. The pos |
| 20 | rub_eye_tired | 3 | True | The robot's end-effector is pointing towards the robot's left (+y), which correctly matches dir=left. The gripper jaws are aligned one above |
| 25 | cheers_toast | 9 | False | The labels dir=up and gripper_orientation=horizontal are technically correct for the robot's pose in the image. However, this specific pose  |
| 26 | highfive_invite | 5 | False | The current `dir=front` is incorrect. The image shows the robot's gripper held up in a classic high-five invitation posture. In this pose, t |
| 30 | pat_self_back | 5 | False | The provided pose shows the robot arm reaching over its own shoulder to pat its back. In this configuration, the gripper's pointing axis (wr |
| 32 | throat_cut_stop | 5 | False | The provided labels (dir=front, gripper_orientation=horizontal) are incorrect for the pose shown in the image. The robot's end-effector has  |
| 35 | salute | 5 | False | The provided labels 'dir=front' and 'gripper_orientation=horizontal' are incorrect for the given pose. The robot's end-effector is clearly p |
| 36 | visor_search | 3 | False | The provided labels (dir=left, gripper_orientation=vertical) accurately describe the pose in the image. However, this pose is not suitable f |
| 39 | request_turn | 9 | False | The current pose is correctly labeled as 'dir=up, gripper_orientation=horizontal'. The end-effector's pointing axis (wrist to fingertips) is |
| 40 | guiding_arm | 3 | False | The 'dir=front' is correct as the gripper is pointing forward towards the viewer. However, the 'gripper_orientation=vertical' is incorrect.  |
| 45 | slow_down_request_palm_down | 6 | True | The pose is correctly labeled. The end-effector's pointing axis is directed downwards (-z), matching 'dir=down'. The line connecting the gri |
| 61 | air_quotes | 5 | False | The gripper orientation is correctly identified as 'horizontal'. However, the direction is 'up' (pointing towards the ceiling), not 'front'  |
| 64 | tilt_head_curious | 5 | False | The provided labels are incorrect. The robot's end effector is pointing towards its own right side, not towards the viewer. The gripper jaws |
| 66 | shh_gesture | 2 | True | The pose correctly represents the 'shh' gesture. The gripper points upwards (dir=up) like an index finger, and the jaw orientation is vertic |
