# Pose Verification — single selected tile (Gemini)

- model: `gemini-2.5-pro`
- config: `/Users/sb/Downloads/workspace/dev_robosuite/data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot100_batch20.json`
- group tiles: `/Users/sb/Downloads/workspace/dev_robosuite/data/results/visualize/pose_groups_12`
- tile picks: `/Users/sb/Downloads/workspace/dev_robosuite/data/results/verify/pose_tile_pick_by_group.json`

| idx | cue | tile | pose ok? | note |
|---:|---|---:|---|---|
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
| 73 | curl_fingers_give_me | 2 | False | The current pose is labeled with dir=down and gripper_orientation=vertical. The image confirms the gripper's pointing axis is directed downw |
| 90 | start_action_forward_snap | 2 | True | The robot's arm is pointing straight up (`dir=up`), which is an excellent preparatory pose for a downward and forward slashing motion. This  |
| 92 | retry_action_reset_circle | 3 | True | The labels dir=front and gripper_orientation=vertical are correct. The end-effector points directly towards the viewer, and the gripper jaws |
| 95 | prepare_action_raise_hold | 5 | False | The current labels 'dir=front' and 'gripper_orientation=horizontal' are incorrect for the pose shown in the image. The robot's end-effector  |
| 97 | hesitation_pause_hold | 3 | True | The labels `dir=front` and `gripper_orientation=vertical` are correct. The robot's end-effector is pointing directly forward (+x) towards th |
| 105 | scale_compress | 6 | True | The labels are correct. The robot's end-effector pointing axis (wrist to fingertips) is directed towards the floor, making 'dir=down' approp |
| 106 | uncertainty_wave | 6 | True | The pose with dir=up and gripper_orientation=horizontal is perfectly appropriate. It correctly positions the gripper as an 'upward-facing pa |
| 108 | interrupt_hard | 2 | False | The current labels are `dir=down` and `gripper_orientation=vertical`. The image shows the end-effector pointing towards the floor, making `d |
