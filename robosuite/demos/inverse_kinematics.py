"""
Inverse Kinematics implementation using PyBullet.
Supports any robot loaded in PyBullet environment.
"""

import numpy as np
import pybullet

class InverseKinematics:
    def __init__(self, robot):
        """
        Initialize IK solver for a robot.
        Args:
            robot: PyBullet robot object (robot ID)
        """
        self.robot = robot
        
        # Get number of joints
        self.n_joints = pybullet.getNumJoints(self.robot)
        
        # Get joint info
        self.joint_info = []
        self.joint_indices = []
        self.joint_limits = {"lower": [], "upper": []}
        
        for i in range(self.n_joints):
            info = pybullet.getJointInfo(self.robot, i)
            if info[2] == pybullet.JOINT_REVOLUTE:  # Only consider revolute joints
                self.joint_info.append(info)
                self.joint_indices.append(i)
                self.joint_limits["lower"].append(info[8])  # Lower limit
                self.joint_limits["upper"].append(info[9])  # Upper limit
        
        self.joint_limits["lower"] = np.array(self.joint_limits["lower"])
        self.joint_limits["upper"] = np.array(self.joint_limits["upper"])
        
        # Get end effector link index (last revolute joint by default)
        self.eef_link_id = self.joint_indices[-1]
        
    def forward_kinematics(self, joint_angles):
        """
        Compute end-effector pose for given joint angles using PyBullet.
        Args:
            joint_angles: List/array of joint angles
        Returns:
            pos: End-effector position (x,y,z)
            rot_mat: End-effector rotation matrix (3x3)
        """
        # Store current joint states
        original_states = []
        for i in self.joint_indices:
            original_states.append(pybullet.getJointState(self.robot, i)[0])
        
        # Set joint angles
        for i, angle in zip(self.joint_indices, joint_angles):
            pybullet.resetJointState(self.robot, i, angle)
        
        # Get end effector state
        state = pybullet.getLinkState(self.robot, self.eef_link_id)
        pos = np.array(state[0])
        rot = np.array(state[1])  # Quaternion
        rot_mat = np.array(pybullet.getMatrixFromQuaternion(rot)).reshape(3, 3)
        
        # Restore original joint states
        for i, state in zip(self.joint_indices, original_states):
            pybullet.resetJointState(self.robot, i, state)
        
        return pos, rot_mat
        
    def inverse_kinematics(self, target_pos, target_rot=None, initial_guess=None):
        """
        Compute inverse kinematics using PyBullet's built-in IK solver.
        Args:
            target_pos: Target position (x,y,z)
            target_rot: Target rotation as quaternion [x,y,z,w] or None
            initial_guess: Initial joint angles or None
        Returns:
            Joint angles solution
        """
        if initial_guess is not None:
            # Set initial guess
            for i, angle in zip(self.joint_indices, initial_guess):
                pybullet.resetJointState(self.robot, i, angle)
        
        # Calculate IK
        if target_rot is not None:
            joint_angles = pybullet.calculateInverseKinematics(
                self.robot,
                self.eef_link_id,
                target_pos,
                target_rot,
                jointDamping=[0.01] * len(self.joint_indices),
                maxNumIterations=100,
                residualThreshold=1e-5
            )
        else:
            joint_angles = pybullet.calculateInverseKinematics(
                self.robot,
                self.eef_link_id,
                target_pos,
                maxNumIterations=100,
                residualThreshold=1e-5
            )
            
        # Extract only the relevant joint angles
        solution = np.array([joint_angles[i] for i in self.joint_indices])
        
        # Clip to joint limits
        solution = np.clip(solution, self.joint_limits["lower"], self.joint_limits["upper"])
        
        return solution
        
    def get_joint_angles(self, target_pos, target_rot=None, initial_guess=None):
        """
        High-level interface to get joint angles for target pose.
        Args:
            target_pos: Target position (x,y,z)
            target_rot: Target rotation as quaternion [x,y,z,w] or None
            initial_guess: Initial joint angles or None
        Returns:
            Joint angles and success flag
        """
        try:
            # Convert inputs to numpy arrays
            target_pos = np.array(target_pos)
            if target_rot is not None:
                target_rot = np.array(target_rot)
            
            # Get IK solution
            q = self.inverse_kinematics(target_pos, target_rot, initial_guess)
            
            # Verify solution
            final_pos, _ = self.forward_kinematics(q)
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