"""
This demo shows how to use the inverse kinematics controller to control a robot arm.
The robot will move to several target positions sequentially.
"""

import numpy as np
import time
import robosuite as suite
from robosuite.utils.input_utils import *
from inverse_kinematics import InverseKinematics

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

    # Initialize environment with IK controller
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

    # Initialize IK solver
    robot = env.robots[0]  # Get the first robot
    ik_solver = InverseKinematics(robot)

    # Define target positions to move to
    target_positions = [
        [0.3, 0.0, 0.5],    # Front
        [0.0, 0.3, 0.5],    # Right
        [-0.3, 0.0, 0.5],   # Back
        [0.0, -0.3, 0.5],   # Left
        [0.0, 0.0, 0.7],    # Up
        [0.0, 0.0, 0.3],    # Down
    ]

    # Define target orientations (in Euler angles)
    target_orientations = [
        [np.pi, 0, np.pi],       # Default orientation
        [np.pi, np.pi/4, np.pi],  # Rotated 45° around Y
        [np.pi, -np.pi/4, np.pi], # Rotated -45° around Y
        [np.pi, 0, np.pi + np.pi/4],  # Rotated 45° around Z
        [np.pi, 0, np.pi - np.pi/4],  # Rotated -45° around Z
        [np.pi, 0, np.pi],       # Back to default
    ]

    # Number of steps to stay at each target
    steps_per_target = 100

    print("Starting IK control demo...")
    print("The robot will move through several target positions...")

    # Move through each target position
    for i, (pos, ori) in enumerate(zip(target_positions, target_orientations)):
        print(f"\nMoving to target {i+1}/{len(target_positions)}")
        print(f"Position: {pos}")
        print(f"Orientation: {ori}")

        # Get joint angles for target pose
        joint_angles, success = ik_solver.get_joint_angles(
            target_pos=pos,
            target_rot=ori,
        )

        if not success:
            print(f"Failed to find IK solution for target {i+1}")
            continue

        # Move to target pose
        for _ in range(steps_per_target):
            start_time = time.time()

            # Create action - set joint positions
            action = np.zeros(robot.action_dim)
            action[:7] = joint_angles  # First 7 DOF are joint positions for Panda

            # Step the environment
            env.step(action)
            env.render()

            # Maintain timing
            elapsed = time.time() - start_time
            if elapsed < 1./MAX_FR:
                time.sleep(1./MAX_FR - elapsed)

    print("\nDemo finished!")
    env.close()
