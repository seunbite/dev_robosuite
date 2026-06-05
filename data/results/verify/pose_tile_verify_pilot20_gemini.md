# Pose Tile Verification (Gemini)

- model: `gemini-2.5-pro`
- config: `data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot10.json`
- tiles: `data/results/visualize/pose_groups_12`

| idx | cue | pose ok? | best tile | note |
|---:|---|---|---:|---|
| 8 | fist_pump | False | 6 | The current labels 'dir=front, gripper_orientation=horizontal' are inappropriate for a 'fist pump'. This gesture is a celebratory upward thr |
| 9 | flex_bicep | False | 8 | The current labels (dir=down, gripper_orientation=vertical) are incorrect. A bicep flex gesture concludes with an upward curl, bringing the  |
| 11 | self_scratch_head_confused | True | 4 | The 'dir=left' and 'gripper_orientation=vertical' labels are perfect for this cue. 'left' correctly places the hand to the side of the 'head |
| 12 | facepalm | False | 6 | The current pose `dir=front` is fundamentally incorrect for a 'facepalm'. This gesture requires the robot's 'palm' to move towards its 'face |
| 14 | point_you | False | 8 | The current `dir=down` is incorrect. A 'point_you' gesture requires pointing forward (+x) at the viewer, not at the floor (-z). |
| 15 | beckon_come_here | False | 5 | The `dir=front` is appropriate for reaching towards the viewer. However, `gripper_orientation=horizontal` results in a palm-up or palm-down  |
| 16 | stop_palm_out | False | 8 | The current pose `dir=down, gripper_orientation=vertical` is incorrect. A 'stop_palm_out' gesture is directed at the viewer, requiring the e |
| 20 | rub_eye_tired | True | 1 | The 'left' direction correctly points the gripper towards the robot's own 'head' space, and the 'vertical' gripper orientation is suitable f |
| 25 | cheers_toast | True | 2 | The 'up' direction and 'horizontal' gripper orientation are perfectly suited for holding a glass upright, which is the core premise of a toa |
| 26 | highfive_invite | True | 6 | The 'dir=front' and 'gripper_orientation=horizontal' labels are correct for this cue. The hand is presented towards the user (front), and th |
| 30 | pat_self_back | False | 6 | The current `dir=front` is incorrect. To pat its own back, the robot's gripper must point towards its body (`dir=back`). The `gripper_orient |
| 32 | throat_cut_stop | False | 4 | The current pose (dir=front, gripper_orientation=horizontal) is incorrect. A 'throat cut' gesture is a lateral (side-to-side) motion across  |
| 35 | salute | False | 6 | The current label 'dir=front' is incorrect for a salute. The hand should be pointing upwards and outwards, not directly at the viewer. The ' |
| 36 | visor_search | True | 2 | The pose is appropriate. For a visor gesture, the hand is held flat across the forehead, so the pointing axis (wrist to fingertips) is corre |
| 39 | request_turn | True | 4 | The labels dir=up and gripper_orientation=horizontal correctly describe an 'open palm up' pose. This is the standard and most intuitive orie |
| 40 | guiding_arm | True | 5 | The 'front' direction is perfect for the initial 'extends its hand forward in an offering pose' part of the cue. The 'vertical' gripper orie |
| 45 | slow_down_request_palm_down | True | 2 | The 'dir=down' and 'gripper_orientation=horizontal' labels are perfect for a 'palm down' gesture, accurately representing a hand pressing do |
| 61 | air_quotes | False | 6 | The current `dir=front` is incorrect. For an 'air quotes' gesture, the hand (gripper) should be pointing upwards, not directly at the viewer |
| 64 | tilt_head_curious | False | 6 | The current labels `dir=front, gripper_orientation=horizontal` are inappropriate. The cue 'tilt_head_curious' is a body language gesture usi |
| 66 | shh_gesture | False | 1 | The cue 'shh_gesture' requires the gripper to be placed in front of the robot's 'mouth', meaning its pointing axis should be `dir=back`. The |
