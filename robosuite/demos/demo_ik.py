"""
This demo shows how to use robosuite's IK controller to control a robot arm.
The robot will move through several target positions sequentially.
"""

import numpy as np
import time
import robosuite as suite
from robosuite.utils.input_utils import *
import robosuite.utils.transform_utils as T
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config

# Maximum frames per second
MAX_FR = 25

if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = "Stack"  # or use choose_environment() for interactive selection
    options["robots"] = "Panda"    # or use choose_robots() for interactive selection

    # Load the default controller config
    controller_name = "IK_POSE"
    arm_controller_config = suite.load_part_controller_config(default_controller=controller_name)
    robot = options["robots"] if not isinstance(options["robots"], list) else options["robots"][0]
    options["controller_configs"] = refactor_composite_controller_config(
        arm_controller_config, robot, ["right"]  # Single arm robot, only use right arm config
    )

    # Initialize environment
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    
    # Reset environment
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # Get robot's initial state
    robot = env.robots[0]
    initial_pos = robot.ee_pos
    initial_ori = T.mat2euler(robot.ee_ori_mat)

    # Define relative movements
    delta_positions = [
        [0.2, 0.0, 0.0],     # Move right
        [0.0, 0.2, 0.0],     # Move forward
        [0.0, 0.0, 0.2],     # Move up
        [-0.2, 0.0, 0.0],    # Move left
        [0.0, -0.2, 0.0],    # Move back
        [0.0, 0.0, -0.2],    # Move down
    ]

    delta_rotations = [
        [0.3, 0.0, 0.0],     # Roll
        [0.0, 0.3, 0.0],     # Pitch
        [0.0, 0.0, 0.3],     # Yaw
        [-0.3, 0.0, 0.0],    # -Roll
        [0.0, -0.3, 0.0],    # -Pitch
        [0.0, 0.0, -0.3],    # -Yaw
    ]

    # Get gripper dimension
    gripper_dim = robot.gripper["right"].dof
    
    # Number of steps for each movement
    steps_per_action = 75
    steps_per_rest = 75

    print("Starting IK control demo...")
    print("The robot will move through several positions and orientations...")

    # First test position changes
    print("\nTesting position control...")
    for i, delta_pos in enumerate(delta_positions):
        print(f"\nExecuting position change {i+1}/{len(delta_positions)}")
        print(f"Delta position: {delta_pos}")

        # Create action with position change
        action = np.zeros(robot.action_dim)
        action[:3] = delta_pos  # Position change
        
        # Execute movement
        for _ in range(steps_per_action):
            start_time = time.time()
            env.step(action)
            env.render()

            elapsed = time.time() - start_time
            if elapsed < 1./MAX_FR:
                time.sleep(1./MAX_FR - elapsed)

        # Rest
        action[:3] = 0  # No movement
        for _ in range(steps_per_rest):
            start_time = time.time()
            env.step(action)
            env.render()

            elapsed = time.time() - start_time
            if elapsed < 1./MAX_FR:
                time.sleep(1./MAX_FR - elapsed)

    # Then test orientation changes
    print("\nTesting orientation control...")
    for i, delta_rot in enumerate(delta_rotations):
        print(f"\nExecuting rotation {i+1}/{len(delta_rotations)}")
        print(f"Delta rotation: {delta_rot}")

        # Create action with orientation change
        action = np.zeros(robot.action_dim)
        action[3:6] = delta_rot  # Orientation change
        
        # Execute movement
        for _ in range(steps_per_action):
            start_time = time.time()
            env.step(action)
            env.render()

            elapsed = time.time() - start_time
            if elapsed < 1./MAX_FR:
                time.sleep(1./MAX_FR - elapsed)

        # Rest
        action[3:6] = 0  # No movement
        for _ in range(steps_per_rest):
            start_time = time.time()
            env.step(action)
            env.render()

            elapsed = time.time() - start_time
            if elapsed < 1./MAX_FR:
                time.sleep(1./MAX_FR - elapsed)

    print("\nDemo finished!")
    env.close()