# Pose Tile Verification (Gemini)

- model: `gemini-2.5-pro`
- config: `data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot20_more.json`
- tiles: `data/results/visualize/pose_groups_12`

| idx | cue | pose ok? | best tile | note |
|---:|---|---|---:|---|
| 7 | raise_fists_victory | False | 3 | The current `dir=front` is incorrect. A gesture for 'raise_fists_victory' should culminate in the fist pointing `up` towards the ceiling. Th |
| 10 | shrug_idk | True | 3 | The labels `dir=up` and `gripper_orientation=horizontal` are correct. A classic shrug involves turning the palms up toward the ceiling, maki |
| 11 | scratch_head_confused | False | 3 | The current pose (dir=front, gripper_orientation=horizontal) is incorrect. A 'scratch head' gesture requires the robot to bring its hand to  |
| 13 | point_self | False | 5 | The assessment should reflect the final, most characteristic pose of the gesture. The current `dir=front` is incorrect because the defining  |
| 17 | pat_stomach | False | 5 | The current pose `dir=front` is incorrect. For a 'pat stomach' gesture, the gripper's palm must face the robot's body. This means the pointi |
| 18 | fan_face_hot | True | 2 | The labels dir=left and gripper_orientation=vertical are correct. Holding the hand sideways (dir=left) with the gripper jaws oriented vertic |
| 19 | cover_yawn_sleepy | False | 5 | The current pose `dir=down` is incorrect. Covering a yawn involves bringing the hand to the mouth, which requires pointing the gripper towar |
| 21 | tap_temple_think | False | 3 | The current `dir=front` is incorrect for a self-referential gesture like 'tap temple'. The end-effector should point inward towards the robo |
| 22 | circle_temple_crazy | True | 1 | The current labels `dir=front` and `gripper_orientation=horizontal` are appropriate for the *initial* phase of the gesture, which involves p |
| 23 | lean_forward_interest | True | 1 | The 'front' direction is essential for a gesture directed at a viewer, and the 'horizontal' gripper orientation is a neutral choice that doe |
| 24 | cover_mouth_gasp | False | 5 | The current pose (dir=front, gripper_orientation=horizontal) is incorrect. 'Front' points the hand at the viewer like a 'stop' gesture. 'Hor |
| 27 | highfive_imitation | False | 1 | The current direction 'up' is incorrect for a high-five, which requires presenting the palm forward to the user. The correct direction is 'f |
| 28 | fistbump_invite | False | 5 | The current 'dir=down' is incorrect. A 'fistbump_invite' gesture requires the robot to extend its fist *forward* toward the user to be bumpe |
| 37 | yield_turn | True | 5 | The 'back' direction and 'vertical' gripper orientation are perfectly suited for the start of a 'yield turn' gesture. It represents the hand |
| 41 | protective_arm_out | True | 1 | The 'front' direction and 'horizontal' gripper orientation are perfectly suited for this cue. It creates a classic 'palm-out' stop gesture,  |
| 50 | firm_accept_forward_reach | True | 5 | The 'front' direction and 'horizontal' gripper orientation are perfectly suited for a 'firm accept forward reach'. This configuration is ana |
| 59 | nod_yes | True | 5 | The 'front' direction and 'horizontal' gripper orientation are suitable. This configuration allows the end-effector to act as a proxy for a  |
| 60 | shake_head_no | True | 5 | The labels (dir=front, gripper_orientation=vertical) are perfect. 'Front' correctly orients the gesture towards the viewer, and 'vertical' a |
| 74 | hands_up_surrender | False | 2 | The current `dir=front` is incorrect for a 'hands up' gesture. This pose should have the hand pointing primarily 'up' to signify raising it  |
| 104 | scale_expand | False | 8 | The current `dir=down` is inappropriate. A 'scale expand' gesture is meant to be shown to a viewer and should occur in the interaction space |
