"""
Inverse Kinematics implementation for robosuite robots.
Currently supports Panda robot arm.
"""

import numpy as np
from scipy.optimize import minimize

class InverseKinematics:
    def __init__(self, robot):
        self.robot = robot
        self.n_joints = len(self.robot.joint_indexes)
        
        # Joint limits for Panda robot
        self.joint_limits = {
            "lower": np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
            "upper": np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        }
        
        # Get robot link names
        self.robot_links = [name for name in self.robot.sim.model.body_names 
                          if name.startswith("robot0_")]
        self.eef_link_name = "robot0_link7"  # End effector link
        self.hand_link_name = "robot0_right_hand"  # Gripper base link
        
    def forward_kinematics(self, q):
        """
        Compute end-effector position for given joint angles.
        Args:
            q: Joint angles (7-DOF for Panda)
        Returns:
            pos: End-effector position (x,y,z)
            rot_mat: End-effector rotation matrix (3x3)
        """
        # Set robot joints
        original_joints = self.robot.sim.data.qpos[self.robot.joint_indexes].copy()
        self.robot.sim.data.qpos[self.robot.joint_indexes] = q
        self.robot.sim.forward()
        
        # Get end-effector position and rotation
        pos = self.robot.sim.data.get_body_xpos(self.eef_link_name)
        rot_mat = self.robot.sim.data.get_body_xmat(self.eef_link_name).reshape(3, 3)
        
        # Restore original joint positions
        self.robot.sim.data.qpos[self.robot.joint_indexes] = original_joints
        self.robot.sim.forward()
        
        return pos, rot_mat
        
    def inverse_kinematics(self, target_pos, target_rot=None, initial_guess=None):
        """
        Compute inverse kinematics for target end-effector pose.
        Args:
            target_pos: Target position (x,y,z)
            target_rot: Target rotation matrix (3x3) or None for position-only IK
            initial_guess: Initial joint angles guess (7-DOF) or None for current joints
        Returns:
            q: Joint angles solution (7-DOF)
        """
        if initial_guess is None:
            initial_guess = self.robot.sim.data.qpos[self.robot.joint_indexes].copy()
            
        def objective(q):
            # Get current end-effector pose
            current_pos, current_rot = self.forward_kinematics(q)
            
            # Position error
            pos_error = np.linalg.norm(current_pos - target_pos)
            
            # Rotation error (if target rotation provided)
            rot_error = 0
            if target_rot is not None:
                # Use rotation matrix difference
                rot_error = np.linalg.norm(current_rot - target_rot)
            
            # Joint limits penalty
            limit_penalty = 0
            for i in range(len(q)):
                if q[i] < self.joint_limits["lower"][i]:
                    limit_penalty += (self.joint_limits["lower"][i] - q[i])**2
                elif q[i] > self.joint_limits["upper"][i]:
                    limit_penalty += (q[i] - self.joint_limits["upper"][i])**2
            
            # Total cost
            cost = pos_error + (0.5 * rot_error if target_rot is not None else 0) + (0.1 * limit_penalty)
            return cost
            
        # Optimize joint angles
        result = minimize(
            objective,
            initial_guess,
            method='BFGS',
            options={'maxiter': 100}
        )
        
        if not result.success:
            print("Warning: IK optimization did not converge")
            
        return result.x
        
    def get_joint_angles(self, target_pos, target_rot=None, initial_guess=None):
        """
        High-level interface to get joint angles for target pose.
        Args:
            target_pos: Target position (x,y,z)
            target_rot: Target rotation matrix (3x3) or None
            initial_guess: Initial joint angles or None
        Returns:
            Joint angles (7-DOF) and success flag
        """
        try:
            # Print available robot links for debugging
            print("Available robot links:", self.robot_links)
            print("Using end-effector link:", self.eef_link_name)
            
            # Convert inputs to numpy arrays
            target_pos = np.array(target_pos)
            if target_rot is not None:
                target_rot = np.array(target_rot)
            
            # Get IK solution
            q = self.inverse_kinematics(target_pos, target_rot, initial_guess)
            
            # Verify solution
            final_pos, final_rot = self.forward_kinematics(q)
            pos_error = np.linalg.norm(final_pos - target_pos)
            
            print(f"IK solution found with position error: {pos_error:.3f}m")
            print(f"Target position: {target_pos}")
            print(f"Achieved position: {final_pos}")
            
            if pos_error > 0.01:  # 1cm threshold
                print(f"Warning: Large position error in IK solution: {pos_error:.3f}m")
                return None, False
                
            return q, True
            
        except Exception as e:
            print(f"IK failed: {str(e)}")
            return None, False