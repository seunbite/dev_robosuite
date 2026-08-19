# Pose Verification — single selected tile (Gemini)

- model: `gemini-2.5-pro`
- config: `/Users/sb/Downloads/workspace/dev_robosuite/data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot100_batch20.json`
- note: exploration_reach / continue_on_forward_roll swapped in; verify pending for new cues

| idx | cue | tile | pose ok? | note |
|---:|---|---:|---|---|
| 31 | draw_circle_repeat |  | True | The pose is correctly labeled as dir=front and gripper_orientation=vertical. The end-effector points directly at the vie |
| 44 | clarification_request_palm_tilt |  | False | The current pose labels (dir=front, gripper_orientation=horizontal) are not the best fit for the cue or the image. The i |
| 46 | speed_up_circular_cue |  | True | The pose is correctly labeled as dir=front and gripper_orientation=vertical. The end-effector points directly at the vie |
| 49 | polite_decline_small_push |  | False | The current labels are a correct assessment of the robot's pose in the image. The end-effector's pointing axis is direct |
| 51 | hesitant_accept_half_reach |  | True | The robot's end-effector is pointing directly towards the viewer, which correctly corresponds to `dir=front`. The grippe |
| 52 | cancel_previous_offer_retract |  | False | The cue 'cancel_previous_offer_retract' describes a motion that begins with extending an offer, then retracting it. The  |
| 53 | conditional_accept_pause_then_reach |  | True | The pose (dir=front, gripper_orientation=vertical) is correct and highly appropriate. The robot is pointing towards the  |
| 54 | final_decision_down_press |  | False | The pose labels (dir=down, gripper_orientation=horizontal) accurately describe the image, where the robot's end-effector |
| 55 | resume_speaking_pull |  | True | The pose is correctly labeled. The gripper's pointing axis is directed forward towards the viewer (+x), making `dir=fron |
| 56 | emphasize_point_downbeat |  | False | The current labels (dir=front, gripper_orientation=vertical) accurately describe the pose shown in the image. However, t |
| 57 | unsure_response_hand_hover_stop |  | False | The `dir=front` label is correct as the gripper is pointing towards the viewer. However, the `gripper_orientation=horizo |
| 65 | raise_index_one_moment |  | False | The provided image shows the robot's end effector pointing almost directly upwards (+z), which is a perfect starting pos |
| 90 | start_action_forward_snap |  | True | The robot's arm is pointing straight up (`dir=up`), which is an excellent preparatory pose for a downward and forward sl |
| 92 | retry_action_reset_circle |  | True | The labels dir=front and gripper_orientation=vertical are correct. The end-effector points directly towards the viewer,  |
| 95 | prepare_action_raise_hold |  | False | The current labels 'dir=front' and 'gripper_orientation=horizontal' are incorrect for the pose shown in the image. The r |
| 97 | hesitation_pause_hold |  | True | The labels `dir=front` and `gripper_orientation=vertical` are correct. The robot's end-effector is pointing directly for |
| 106 | uncertainty_wave |  | True | The pose with dir=up and gripper_orientation=horizontal is perfectly appropriate. It correctly positions the gripper as  |
| 108 | interrupt_hard |  | False | The current labels are `dir=down` and `gripper_orientation=vertical`. The image shows the end-effector pointing towards  |
