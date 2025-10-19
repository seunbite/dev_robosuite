"""
This demo shows how to use the IKSolver class for robot control.
It demonstrates both absolute and relative IK control modes.
"""

import time
import numpy as np
import robosuite as suite
from robosuite.utils.input_utils import *
from robosuite.utils.ik_utils import IKSolver
import robosuite.utils.transform_utils as T

MAX_FR = 25  # max frame rate for running simulation

class IKController:
    def __init__(self, env, robot_config):
        """
        Initialize IK controller
        
        Args:
            env: robosuite environment
            robot_config: robot configuration dictionary containing:
                - joint_names: list of joint names
                - end_effector_sites: list of end effector site names
                - initial_keyframe: name of initial keyframe (optional)
                - nullspace_gains: list of nullspace gains for each joint
        """
        self.env = env
        self.model = env.sim.model
        self.data = env.sim.data
        
        # Initialize IK solver
        self.ik_solver = IKSolver(
            model=self.model,
            data=self.data,
            robot_config=robot_config,
            damping=0.05,  # damping coefficient for pseudo-inverse
            integration_dt=1.0 / env.control_freq,
            max_dq=0.5,  # maximum joint velocity
            input_type="keyboard",  # we'll use keyboard input mode
            debug=False
        )
        
        # Get control dimensions
        self.control_dim = len(robot_config["end_effector_sites"]) * 6  # 6 DOF per end effector
        self.gripper_dim = env.robots[0].gripper["right"].dof if hasattr(env.robots[0], "gripper") else 0
        
        # Initialize neutral pose
        self.neutral_pose = np.zeros(self.control_dim + self.gripper_dim)
        
    def reset(self):
        """Reset robot to initial pose"""
        self.ik_solver.reset_to_initial_state()
        
    def move_to_pose(self, target_pose, steps=100, relative=False):
        """
        Move end effector to target pose
        
        Args:
            target_pose: target pose as (pos, quat) or (dx, dy, dz, droll, dpitch, dyaw)
            steps: number of steps to take
            relative: if True, target_pose is relative to current pose
        """
        # Convert target pose to absolute coordinates if relative
        if relative:
            current_pos = np.array([self.data.site(site_id).xpos for site_id in self.ik_solver.site_ids])
            current_rot = np.array([self.data.site(site_id).xmat for site_id in self.ik_solver.site_ids])
            current_quat = np.array([T.mat2quat(rot.reshape(3, 3)) for rot in current_rot])
            
            # Separate position and orientation from target_pose
            delta_pos = target_pose[:3]
            delta_rot = target_pose[3:]
            
            # Convert delta rotation to quaternion
            delta_quat = T.axisangle2quat(delta_rot)
            
            # Compute new target position
            target_pos = current_pos + delta_pos
            
            # Compute new target orientation (quaternion multiplication)
            target_quat = np.array([T.quat_multiply(cur_q, delta_quat) for cur_q in current_quat])
            
            # Combine into final target pose
            target_pose = np.concatenate([target_pos.flatten(), target_quat.flatten()])
        
        # Interpolate from current to target pose
        for i in range(steps):
            alpha = (i + 1) / steps
            action = target_pose * alpha + self.neutral_pose * (1 - alpha)
            
            # Solve IK
            q_des = self.ik_solver.solve(action)
            
            # Set joint positions
            self.data.qpos[self.ik_solver.dof_ids] = q_des
            
            # Step simulation
            self.env.step(np.zeros(self.env.robots[0].dof))  # Zero action since we directly set qpos
            self.env.render()
            
            # Limit frame rate
            time.sleep(1.0 / MAX_FR)

def main():
    # Create dict to hold options that will be passed to env creation call
    options = {}

    # Choose environment and add it to options
    options["env_name"] = "Stack"  # or use choose_environment() for interactive selection
    options["robots"] = "Panda"    # or use choose_robots() for interactive selection

    # Initialize the task
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
    
    # Create robot configuration
    robot_config = {
        "joint_names": env.robots[0].joint_names,
        "end_effector_sites": ["right_ee"],  # Adjust based on your robot
        "nullspace_gains": [1.0] * len(env.robots[0].joint_names),
    }
    
    # Initialize controller
    controller = IKController(env, robot_config)
    controller.reset()
    
    # Example movements
    movements = [
        # Move in x direction
        (np.array([0.1, 0, 0, 0, 0, 0]), True),
        # Move in y direction
        (np.array([0, 0.1, 0, 0, 0, 0]), True),
        # Move in z direction
        (np.array([0, 0, 0.1, 0, 0, 0]), True),
        # Rotate around x axis
        (np.array([0, 0, 0, 0.1, 0, 0]), True),
        # Rotate around y axis
        (np.array([0, 0, 0, 0, 0.1, 0]), True),
        # Rotate around z axis
        (np.array([0, 0, 0, 0, 0, 0.1]), True),
    ]
    
    # Execute movements
    for target_pose, is_relative in movements:
        print(f"Executing {'relative' if is_relative else 'absolute'} movement")
        controller.move_to_pose(target_pose, steps=75, relative=is_relative)
        time.sleep(1.0)  # Pause between movements
        
    # Close environment
    env.close()

if __name__ == "__main__":
    main()
