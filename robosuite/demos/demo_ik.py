"""
This demo shows how to use robosuite's IK solver to control a robot arm.
The robot will move to several target positions sequentially.
"""

import numpy as np
import time
import robosuite as suite
from robosuite.utils.input_utils import *
from robosuite.utils.ik_utils import IKSolver
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
    options["controller_configs"] = refactor_composite_controller_config(
        arm_controller_config, options["robots"], ["right"]  # Single arm robot, only use right arm config
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

    # Get robot configuration
    robot = env.robots[0]  # Get the first robot
    
    robot_config = {
        "joint_names": robot.joint_indexes,
        "end_effector_sites": ["gripper0_right_grip_site"],  # Use actual site name from the model
        "nullspace_gains": [1.0] * len(robot.joint_indexes),  # Nullspace gains for each joint
    }

    # Initialize IK solver
    ik_solver = IKSolver(
        model=env.sim.model,  # MuJoCo model
        data=env.sim.data,    # MuJoCo data
        robot_config=robot_config,
        damping=0.05,
        integration_dt=1/20.0,
        max_dq=10.0,
    )

    # Define target positions and orientations
    target_positions = [
        [0.3, 0.0, 0.5],    # Front
        [0.0, 0.3, 0.5],    # Right
        [-0.3, 0.0, 0.5],   # Back
        [0.0, -0.3, 0.5],   # Left
        [0.0, 0.0, 0.7],    # Up
        [0.0, 0.0, 0.3],    # Down
    ]

    target_orientations = [
        np.array([0, 0, 0]),                # Default orientation (axis-angle)
        np.array([0, np.pi/4, 0]),          # Rotated 45° around Y
        np.array([0, -np.pi/4, 0]),         # Rotated -45° around Y
        np.array([0, 0, np.pi/4]),          # Rotated 45° around Z
        np.array([0, 0, -np.pi/4]),         # Rotated -45° around Z
        np.array([0, 0, 0]),                # Back to default
    ]

    # Get gripper dimension
    gripper_dim = robot.gripper["right"].dof
    
    # Number of steps to stay at each target
    steps_per_action = 75
    steps_per_rest = 75

    print("Starting IK control demo...")
    print("The robot will move through several target positions...")

    # Move through each target position
    for i, (pos, ori) in enumerate(zip(target_positions, target_orientations)):
        print(f"\nMoving to target {i+1}/{len(target_positions)}")
        print(f"Position: {pos}")
        print(f"Orientation: {ori}")

        # Prepare target action (position + axis angle)
        target_action = np.concatenate([np.array(pos), ori])

        # Move to target pose
        for _ in range(steps_per_action):
            start_time = time.time()

            # Get desired joint positions from IK solver
            q_des = ik_solver.solve(
                target_action.reshape(1, -1),  # Reshape for single end-effector
                Kpos=0.95,  # Position gain
                Kori=0.95,  # Orientation gain
            )

            # Create action - set joint positions and add gripper command
            action = np.zeros(robot.action_dim)
            action[:len(q_des)] = q_des  # Set joint positions
            
            # Step the environment
            env.step(action)
            env.render()

            # Maintain timing
            elapsed = time.time() - start_time
            if elapsed < 1./MAX_FR:
                time.sleep(1./MAX_FR - elapsed)

        # Rest at current position
        for _ in range(steps_per_rest):
            start_time = time.time()
            env.step(action)  # Keep the same action
            env.render()

            elapsed = time.time() - start_time
            if elapsed < 1./MAX_FR:
                time.sleep(1./MAX_FR - elapsed)

    print("\nDemo finished!")
    env.close()