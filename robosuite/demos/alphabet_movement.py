"""
Generate non-verbal cues (movements) for robots using pose sets and movements.

This script:
1. Loads pose definitions from pose_set
2. Finds matching poses from closest_poses_results.jsonl
3. Generates movements using position_move and orientation_move
4. Executes movements in simulation
"""

import fire
import os
import json
import random
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from PIL import Image
from datetime import datetime

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.utils.ik_utils import IKSolver
import robosuite.utils.transform_utils as T

# Fixed joint indices (matching stack_preset.py)
FIXED_JOINT_INDICES = {
    'GR1': "0-2, 20-31"
}

# Pose definitions - maps pose names to target orientation
pose_set = {
    'Elbow_up': {'roll': 180, 'pitch': None, 'yaw': 0},
    'Stretched_out': {'roll': -90, 'pitch': None, 'yaw': -90},
    'Elbow_down': {'roll': 0, 'pitch': None, 'yaw': 0},
}

# Movement definitions - types and their parameters
movement_set = {
    'position_move': ['speed', 'distance', 'direction'],
    'orientation_move': ['speed', 'angle', 'joint', 'direction'],
}


class NonVerbalCueGenerator:
    """Generate and execute non-verbal cues using poses and movements."""
    
    def __init__(
        self,
        robot_name: str = "Panda",
        env_name: str = "EmptySpace",
        controller_name: str = "IK_POSE",
        jsonl_path: str = "data/poses/closest_poses_results.jsonl",
        has_renderer: bool = False,
        has_offscreen_renderer: bool = True,
        control_freq: int = 20,
        output_dir: str = "data/motions",
        capture_image_width: int = 512,
        capture_image_height: int = 512,
        camera_fov: float = 60.0,
        hz: int = 5,
    ):
        """
        Initialize the non-verbal cue generator.
        
        Args:
            robot_name: Name of the robot
            env_name: Name of the environment
            controller_name: Name of the controller (should support IK)
            jsonl_path: Path to JSONL file with pose data
            has_renderer: Whether to show on-screen rendering
            has_offscreen_renderer: Whether to enable offscreen rendering
            control_freq: Control frequency
        """
        self.robot_name = robot_name
        self.env_name = env_name
        self.controller_name = controller_name
        self.jsonl_path = jsonl_path
        self.control_freq = control_freq
        self.output_dir = os.path.join(output_dir, robot_name)
        self.camera_fov = camera_fov
        self.capture_image_width = capture_image_width
        self.capture_image_height = capture_image_height
        self.hz = hz  # Store hz for calculating frame counts
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
        print(f"Initializing robot: {robot_name}")
        
        # Load pose data from JSONL
        self.pose_database = self._load_pose_database(jsonl_path)
        print(f"Loaded {len(self.pose_database)} poses from {jsonl_path}")
        
        # Setup environment (matching connected_motion.py)
        options = {
            "env_name": env_name,
            "robots": robot_name,
            "has_renderer": has_renderer,
            "has_offscreen_renderer": has_offscreen_renderer,
            "ignore_done": True,
            "use_camera_obs": True,  # Need camera obs for rendering
            "camera_names": "frontview",
            "camera_heights": capture_image_height,
            "camera_widths": capture_image_width,
            "control_freq": control_freq,
        }
        
        # Load controller config
        arm_controller_config = suite.load_part_controller_config(default_controller=controller_name)
        options["controller_configs"] = refactor_composite_controller_config(
            arm_controller_config, robot_name, ["right", "left"]
        )
        
        # Create environment
        self.env = suite.make(**options, horizon=10000)
        self.env.reset()
        
        # Set camera FOV (matching connected_motion.py)
        camera_name = "frontview"
        try:
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            self.env.sim.model.cam_fovy[cam_id] = camera_fov
            print(f"Camera FOV set to {camera_fov} degrees")
        except Exception as e:
            print(f"Warning: Could not set camera FOV: {e}")
        
        # Disable gravity to prevent robot drift
        # This ensures the robot stays perfectly still when not being actively controlled
        self.env.sim.model.opt.gravity[:] = [0, 0, 0]
        print("Gravity disabled to prevent robot drift")
        
        # Initialize viewer if renderer is enabled
        if has_renderer:
            # Give viewer time to initialize
            if hasattr(self.env, 'viewer') and self.env.viewer is not None:
                try:
                    self.env.viewer.set_camera(camera_id=0)
                    # Call update() to initialize and render the viewer window
                    # (render() alone doesn't work because MjviewerRenderer.render() is empty)
                    self.env.viewer.update()
                    time.sleep(0.3)  # Give time for window to open
                except Exception as e:
                    print(f"Warning: Could not set camera or render: {e}")
        
        # Get robot
        self.robot = self.env.robots[0]
        
        # Get initial joint positions (matching stack_preset.py exactly)
        # stack_preset.py line 246: self.initial_joint_pos = self._get_joint_positions()
        # stack_preset.py line 316: _get_joint_positions returns self.robot._joint_positions.copy()
        self.initial_joint_pos = self.robot._joint_positions.copy()
        self.num_joints = len(self.initial_joint_pos)
        
        # Parse fixed joint indices (matching stack_preset.py)
        self.fixed_joint_indices = []
        if robot_name in FIXED_JOINT_INDICES:
            fixed_indices_str = FIXED_JOINT_INDICES[robot_name]
            try:
                for part in fixed_indices_str.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if "-" in part:
                        start, end = part.split("-", 1)
                        start = int(start.strip())
                        end = int(end.strip())
                        if start > end:
                            continue
                        self.fixed_joint_indices.extend(range(start, end + 1))
                    else:
                        self.fixed_joint_indices.append(int(part))
                self.fixed_joint_indices = sorted(list(set([idx for idx in self.fixed_joint_indices if 0 <= idx < self.num_joints])))
            except ValueError:
                self.fixed_joint_indices = []
        
        # Active joints (matching stack_preset.py line 293-294)
        # Exclude last joint (gripper) and fixed joints
        base_active_joint_indices = list(range(self.num_joints - 1))
        self.active_joint_indices = [idx for idx in base_active_joint_indices if idx not in self.fixed_joint_indices]
        
        # Get initial end effector position
        self.initial_eef_pos = self.robot._hand_pos["right"]
        self.initial_eef_orn = self.robot._hand_orn["right"]
        
        # Initialize IK solver for position movements
        # Get joint names from robot model
        try:
            joint_names = list(self.robot.robot_model.joints)
        except:
            joint_names = [self.env.sim.model.joint_id2name(idx) for idx in self.robot.joint_indexes]
        
        # Get end effector site name from gripper
        try:
            eef_site_name = self.robot.gripper["right"].important_sites["grip_site"]
        except:
            # Fallback: use standard naming convention
            eef_site_name = "gripper0_right_grip_site"
        
        robot_config = {
            "joint_names": joint_names,
            "end_effector_sites": [eef_site_name],
            "nullspace_gains": [1.0] * len(joint_names),
        }
        
        # Initialize IK solver (matching demo_ik_control.py)
        self.ik_solver = IKSolver(
            model=self.env.sim.model._model,
            data=self.env.sim.data._data,
            robot_config=robot_config,
            damping=0.05,  # damping coefficient for pseudo-inverse
            integration_dt=1.0 / self.control_freq,
            max_dq=0.5,  # maximum joint velocity
            input_type="keyboard",  # we'll use keyboard input mode
            debug=False,
            input_action_repr="absolute",  # use absolute positioning
            input_rotation_repr="axis_angle"  # use axis-angle for rotation
        )
        
        print(f"Total joints: {self.num_joints}")
        print(f"Active joints: {len(self.active_joint_indices)}")
        print(f"Fixed joints: {len(self.fixed_joint_indices)}")
        print(f"Initial EE position: {self.initial_eef_pos}")
        print(f"IK solver initialized")
        print(f"Robot initialized successfully!")
    
    def _load_pose_database(self, jsonl_path: str) -> List[Dict]:
        """Load pose database from JSONL file."""
        if not os.path.exists(jsonl_path):
            print(f"Warning: JSONL file not found: {jsonl_path}")
            return []
        
        poses = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    poses.append(json.loads(line))
        
        return poses
    
    def _find_matching_poses(
        self,
        roll_deg: Optional[float] = None,
        pitch_deg: Optional[float] = None,
        yaw_deg: Optional[float] = None,
        robot_name: Optional[str] = None,
    ) -> List[Dict]:
        """
        Find poses matching the given orientation criteria.
        
        Args:
            roll_deg: Target roll angle in degrees
            pitch_deg: Target pitch angle in degrees
            yaw_deg: Target yaw angle in degrees
            robot_name: Filter by robot name (if None, uses self.robot_name)
        
        Returns:
            List of matching pose dictionaries
        """
        if robot_name is None:
            robot_name = self.robot_name
        
        matching_poses = []
        
        for pose in self.pose_database:
            # Filter by robot name
            if pose.get("robot") != robot_name:
                continue
            
            # Filter by orientation (within tolerance)
            tolerance = 5.0  # degrees
            match = True
            
            if roll_deg is not None and pose.get("roll_deg") is not None:
                roll_diff = abs(pose.get("roll_deg") - roll_deg)
                # Handle angle wrapping (180 = -180)
                roll_diff = min(roll_diff, 360 - roll_diff)
                if roll_diff > tolerance:
                    match = False
            
            if pitch_deg is not None and pose.get("pitch_deg") is not None:
                pitch_diff = abs(pose.get("pitch_deg") - pitch_deg)
                pitch_diff = min(pitch_diff, 360 - pitch_diff)
                if pitch_diff > tolerance:
                    match = False
            
            if yaw_deg is not None and pose.get("yaw_deg") is not None:
                yaw_diff = abs(pose.get("yaw_deg") - yaw_deg)
                yaw_diff = min(yaw_diff, 360 - yaw_diff)
                if yaw_diff > tolerance:
                    match = False
            
            if match:
                matching_poses.append(pose)
        
        return matching_poses
    
    def _capture_image(self, width: int = None, height: int = None):
        """Capture current camera view as numpy array (matching connected_motion.py)."""
        if width is None:
            width = self.capture_image_width
        if height is None:
            height = self.capture_image_height
        obs = self.env.sim.render(
            camera_name="frontview",
            width=width,
            height=height,
            depth=False
        )
        return obs[::-1]
    
    def _get_joint_positions(self):
        """Get current joint positions."""
        joint_pos = self.robot._joint_positions.copy()
        return joint_pos
    
    def _set_joint_positions(self, joint_positions):
        """Set joint positions and update simulation (matching connected_motion.py)."""
        robot_joint_pos = joint_positions
        self.robot.set_robot_joint_positions(robot_joint_pos)
        self.env.sim.forward()
    
    def _set_pose_from_data(self, pose_data: Dict):
        """
        Set robot to a pose from pose data.
        
        Uses the EXACT same logic as reconstruct_pose.py's set_pose_from_filename:
        - reconstruct_pose.py line 218: joint_pos = self.initial_joint_pos.copy()
        - reconstruct_pose.py line 221-224: joint_pos[active_joint_idx] = joint_angles_rad[i]
        - reconstruct_pose.py line 229: self.robot.set_robot_joint_positions(joint_pos)
        - reconstruct_pose.py line 230: self.env.sim.forward()
        
        Args:
            pose_data: Dictionary containing pose information with joint_angles_deg and joint_names
        """
        joint_angles_deg = pose_data["joint_angles_deg"]
        joint_angles_rad = pose_data["joint_angles_rad"]
        active_joint_indices = pose_data.get("active_joint_indices", [])
        
        # Reconstruct full joint position array (matching reconstruct_pose.py line 218 exactly)
        joint_pos = self.initial_joint_pos.copy()
        
        # Set positions for active joints (matching reconstruct_pose.py line 221-224 exactly)
        # Use the active_joint_indices from the saved pose data
        for i, active_joint_idx in enumerate(active_joint_indices):
            if i < len(joint_angles_rad):
                if active_joint_idx < len(joint_pos):
                    joint_pos[active_joint_idx] = joint_angles_rad[i]
                else:
                    print(f"Warning: active_joint_idx {active_joint_idx} >= len(joint_pos) {len(joint_pos)}")
        
        # Set joint positions (matching reconstruct_pose.py line 229-230 exactly)
        self.robot.set_robot_joint_positions(joint_pos)
        self.env.sim.forward()
        
        # Update IK solver's q0 (nullspace control target) to current joint positions
        # This prevents nullspace control from trying to move joints back to zero
        current_joint_pos = self._get_joint_positions()
        self.ik_solver.q0 = current_joint_pos[self.ik_solver.dof_ids].copy()
        
        # Debug: print applied angles and verify (matching reconstruct_pose.py style)
        print(f"Applied joint angles (deg): {joint_angles_deg}")
        
        # Verify by reading back joint positions (matching reconstruct_pose.py line 233-240)
        current_joint_pos = self.robot._joint_positions.copy()
        if len(active_joint_indices) > 0:
            print(f"Verification - Current joint positions (deg) for active joints:")
            for i, active_idx in enumerate(active_joint_indices):
                if active_idx < len(current_joint_pos):
                    current_deg = np.rad2deg(current_joint_pos[active_idx])
                    expected_deg = joint_angles_deg[i] if i < len(joint_angles_deg) else 0
                    match = "✓" if abs(current_deg - expected_deg) < 1.0 else "✗"
                    print(f"  [{active_idx}] {current_deg:+.1f}° (expected {expected_deg:+.1f}°) {match}")
    
    def _move_position(
        self,
        direction: str,
        distance: float,
        speed: float = 1.0,
        capture_frames: bool = False,
    ):
        """
        Move end effector in a direction using inverse kinematics.
        No interpolation - moves directly to target position.
        
        Args:
            direction: Direction to move ('left', 'right', 'up', 'down', 'forward', 'backward')
            distance: Distance to move in meters
            speed: Speed in m/s (not used for frame calculation, only for duration)
            capture_frames: Whether to capture frame after movement (only 1 frame at end)
        """
        # Get current position and orientation from robot state
        current_pos = self.robot._hand_pos["right"].copy()
        
        # Update IK solver's q0 (nullspace control target) to current joint positions
        # This prevents nullspace control from trying to change the pose when distance is 0
        current_joint_pos = self._get_joint_positions()
        self.ik_solver.q0 = current_joint_pos[self.ik_solver.dof_ids].copy()
        
        # Determine direction vector
        direction_vectors = {
            'left': np.array([0, -1, 0]),  # -Y
            'right': np.array([0, 1, 0]),  # +Y
            'up': np.array([0, 0, 1]),     # +Z
            'down': np.array([0, 0, -1]),  # -Z
            'forward': np.array([1, 0, 0]), # +X
            'backward': np.array([-1, 0, 0]), # -X
        }
        
        if direction not in direction_vectors:
            print(f"Warning: Unknown direction '{direction}', using 'right'")
            direction = 'right'
        
        direction_vec = direction_vectors[direction]
        target_pos = current_pos + direction_vec * distance
        
        # Check if we're returning to initial pose (within 1mm tolerance)
        # If so, use stored joint positions directly instead of IK solver for perfect accuracy
        is_returning_to_initial = False
        if hasattr(self, 'initial_pose_eef_pos'):
            distance_to_initial = np.linalg.norm(target_pos - self.initial_pose_eef_pos)
            if distance_to_initial < 0.001:  # 1mm tolerance
                is_returning_to_initial = True
                print(f"  Returning to initial pose (distance: {distance_to_initial:.6f} m), using stored joint positions")
        
        # If distance is 0 or returning to initial pose, use stored joint positions
        if abs(distance) < 1e-6 or is_returning_to_initial:
            if is_returning_to_initial:
                # Restore exact initial joint positions
                self._set_joint_positions(self.initial_pose_joint_pos.copy())
                # Update IK solver's q0 to match
                self.ik_solver.q0 = self.initial_pose_joint_pos[self.ik_solver.dof_ids].copy()
            else:
                print(f"  Distance is 0, maintaining current pose (no IK solving)")
            
            # Zero velocities to prevent drift
            self.env.sim.data.qvel[:] = 0
            # Forward simulation after setting velocities (ensures state is updated)
            self.env.sim.forward()
            actual_pos = self.robot._hand_pos["right"].copy()
            actual_distance = np.linalg.norm(actual_pos - current_pos)
            print(f"  Position after restore: {actual_pos}")
            print(f"  Distance from initial: {np.linalg.norm(actual_pos - self.initial_pose_eef_pos):.6f} m")
        else:
            # Get current orientation (keep orientation constant during position movement)
            current_rot = self.env.sim.data.site(self.ik_solver.site_ids[0]).xmat
            current_quat = T.mat2quat(current_rot.reshape(3, 3))
            # Convert current quaternion to axis-angle to maintain current orientation
            # Since IK solver uses absolute mode with axis_angle representation, we need to pass
            # the current orientation as axis-angle, not zero rotation
            target_ori = T.quat2axisangle(current_quat)
            
            # Move to target position using iterative IK solver
            # IK solver needs multiple iterations to converge, especially for large distances
            target_action = np.concatenate([target_pos, target_ori])
            
            # Iteratively solve IK until convergence or max iterations
            max_ik_iterations = 50
            convergence_threshold = 0.001  # 1mm
            prev_pos = current_pos.copy()
            
            for ik_iter in range(max_ik_iterations):
                # Solve IK to get target joint positions
                q_des = self.ik_solver.solve(target_action)
                
                # Get current full joint positions
                current_joint_pos = self._get_joint_positions()
                
                # Update only the DOF indices that IK controls
                updated_joint_pos = current_joint_pos.copy()
                updated_joint_pos[self.ik_solver.dof_ids] = q_des
                
                # Set joint positions and forward simulation
                self._set_joint_positions(updated_joint_pos)
                
                # Check convergence
                actual_pos = self.robot._hand_pos["right"].copy()
                pos_error = np.linalg.norm(actual_pos - target_pos)
                pos_change = np.linalg.norm(actual_pos - prev_pos)
                
                if pos_error < convergence_threshold:
                    print(f"  IK converged after {ik_iter + 1} iterations (error: {pos_error:.6f} m)")
                    break
                
                if pos_change < convergence_threshold:
                    # No more progress, likely hit joint limits
                    print(f"  IK stopped making progress after {ik_iter + 1} iterations (change: {pos_change:.6f} m)")
                    break
                
                prev_pos = actual_pos.copy()
            
            # Verify actual movement by checking new position
            actual_pos = self.robot._hand_pos["right"].copy()
            actual_distance = np.linalg.norm(actual_pos - current_pos)
            print(f"  Actual position after IK: {actual_pos}")
            print(f"  Actual distance moved: {actual_distance:.4f} m ({actual_distance * 100:.1f} cm)")
            print(f"  Target distance: {distance:.4f} m ({distance * 100:.1f} cm)")
            print(f"  Position error from target: {np.linalg.norm(actual_pos - target_pos):.4f} m")
            if abs(actual_distance - distance) > 0.01:  # More than 1cm difference
                print(f"  Warning: Requested {distance:.4f} m but actually moved {actual_distance:.4f} m")
            
            # Zero velocities to prevent drift
            self.env.sim.data.qvel[:] = 0
            # Forward simulation after setting velocities (ensures state is updated)
            self.env.sim.forward()
        
        # Capture single frame if requested (only at end of movement)
        movement_frames = []
        if capture_frames:
            image = self._capture_image()
            movement_frames.append(Image.fromarray(image))
        
        if self.env.viewer is not None:
            self.env.viewer.update()  # Use update() instead of render()
        
        if capture_frames:
            return movement_frames
        return []
    
    def _move_orientation(
        self,
        joint: str,
        angle: float,
        direction: str,
        speed: float = 1.0,
        capture_frames: bool = False,
    ):
        """
        Rotate end effector around an axis using IK.
        No interpolation - rotates directly to target angle.
        
        Args:
            joint: Axis to rotate around ('roll', 'pitch', 'yaw')
            angle: Total angle to rotate in degrees
            direction: Direction ('left', 'right', 'up', 'down', 'forward', 'backward')
            speed: Angular speed in deg/s (not used for frame calculation, only for duration)
            capture_frames: Whether to capture frame after movement (only 1 frame at end)
        """
        # Get current position and orientation from robot state
        current_pos = self.robot._hand_pos["right"].copy()
        current_rot_mat = self.env.sim.data.site(self.ik_solver.site_ids[0]).xmat.reshape(3, 3)
        
        # Determine rotation axis based on joint
        axis_map = {
            'roll': np.array([1, 0, 0]),   # X axis
            'pitch': np.array([0, 1, 0]),  # Y axis
            'yaw': np.array([0, 0, 1]),    # Z axis
        }
        
        if joint not in axis_map:
            print(f"Warning: Unknown joint '{joint}', using 'yaw'")
            joint = 'yaw'
        
        rot_axis = axis_map[joint]
        
        # Determine rotation direction
        direction_mult = 1.0
        if direction in ['left', 'down', 'backward']:
            direction_mult = -1.0
        
        # Convert total angle to radians
        total_angle_deg = angle * direction_mult
        total_angle_rad = np.deg2rad(total_angle_deg)
        
        # Compute rotation matrix directly (no interpolation)
        rotation_mat = T.rotation_matrix(total_angle_rad, rot_axis)[:3, :3]
        target_rot_mat = current_rot_mat @ rotation_mat
        
        # Convert rotation matrix to quaternion, then to axis-angle
        target_quat = T.mat2quat(target_rot_mat)
        target_axis_angle = T.quat2axisangle(target_quat)
        
        # Create target action: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
        target_action = np.concatenate([current_pos, target_axis_angle])
        
        # Solve IK to get target joint positions
        q_des = self.ik_solver.solve(target_action)
        
        # Get current full joint positions
        current_joint_pos = self._get_joint_positions()
        
        # Update only the DOF indices that IK controls
        updated_joint_pos = current_joint_pos.copy()
        updated_joint_pos[self.ik_solver.dof_ids] = q_des
        
        # Set joint positions using _set_joint_positions (matching connected_motion.py)
        self._set_joint_positions(updated_joint_pos)
        
        # Zero velocities to prevent drift
        self.env.sim.data.qvel[:] = 0
        # Forward simulation after setting velocities (ensures state is updated)
        self.env.sim.forward()
        
        # Capture single frame if requested (only at end of movement)
        movement_frames = []
        if capture_frames:
            image = self._capture_image()
            movement_frames.append(Image.fromarray(image))
        
        if self.env.viewer is not None:
            self.env.viewer.update()  # Use update() instead of render()
        
        if capture_frames:
            return movement_frames
        return []
    
    def execute_cue(
        self,
        pose_name: str,
        movements: List[Dict],
        repeat: int = 1,
        hold_time: float = 2.0,
        save_gif: bool = True,
        output_filename: str = None,
        gif_duration: int = 100,
        pose_index: Optional[int] = None,
    ):
        """
        Execute a non-verbal cue and save as GIF.
        
        Args:
            pose_name: Name of the pose from pose_set
            movements: List of movement dictionaries, each with:
                - type: 'position_move' or 'orientation_move'
                - parameters: dict with movement parameters
            repeat: Number of times to repeat the movement sequence
            hold_time: Time to hold initial pose in seconds
            save_gif: Whether to save motion as GIF
            output_filename: Output filename for GIF (auto-generated if None)
            gif_duration: Duration of each frame in milliseconds for GIF
        """
        print(f"\n{'='*60}")
        print(f"Executing cue: {pose_name}")
        print(f"{'='*60}")
        
        # Get pose definition
        if pose_name not in pose_set:
            print(f"Error: Pose '{pose_name}' not found in pose_set")
            return
        
        pose_def = pose_set[pose_name]
        
        # Find matching poses
        matching_poses = self._find_matching_poses(
            roll_deg=pose_def.get('roll'),
            pitch_deg=pose_def.get('pitch'),
            yaw_deg=pose_def.get('yaw'),
        )
        
        if not matching_poses:
            print(f"Warning: No matching poses found for {pose_name}")
            print(f"  Looking for: roll={pose_def.get('roll')}, pitch={pose_def.get('pitch')}, yaw={pose_def.get('yaw')}")
            return
        
        # Select pose: use pose_index (pose_id) if specified, otherwise randomly select
        if pose_index is not None:
            # Find pose by pose_id in matching_poses
            selected_pose = None
            for pose in matching_poses:
                if pose.get('pose_id') == pose_index:
                    selected_pose = pose
                    break
            
            if selected_pose is None:
                print(f"Warning: pose_id {pose_index} not found in matching poses, using random selection")
                selected_pose = random.choice(matching_poses)
                pose_id = selected_pose['pose_id']
            else:
                pose_id = selected_pose['pose_id']
                print(f"Selected pose with pose_id {pose_id}: rank {selected_pose['rank']}")
        else:
            # Randomly select a pose
            selected_pose = random.choice(matching_poses)
            pose_id = selected_pose['pose_id']
            print(f"Randomly selected pose with pose_id {pose_id}: rank {selected_pose['rank']}")
        
        # Move to initial pose
        print("Moving to initial pose...")
        self._set_pose_from_data(selected_pose)
        
        # Stabilize (set joint velocities to zero to prevent any movement)
        for _ in range(10):
            self.env.sim.data.qvel[:] = 0
            self.env.sim.forward()
        
        # Store initial joint positions and end effector position for accurate return
        # This ensures we can return to exactly the same pose without IK solver errors
        self.initial_pose_joint_pos = self._get_joint_positions().copy()
        self.initial_pose_eef_pos = self.robot._hand_pos["right"].copy()
        self.initial_pose_eef_orn = self.robot._hand_orn["right"].copy()
        print(f"Stored initial pose: joint_pos shape={self.initial_pose_joint_pos.shape}, eef_pos={self.initial_pose_eef_pos}")
        
        # Collect frames for GIF (matching connected_motion.py)
        frames = []
        
        # Capture single frame at initial pose (1 frame total)
        print(f"Capturing initial pose...")
        self.env.sim.data.qvel[:] = 0
        self.env.sim.forward()
        image = self._capture_image()
        frames.append(Image.fromarray(image))
        
        # Store joint positions for each movement to enable exact repetition
        # First repeat: execute movements and store joint positions
        # Subsequent repeats: use stored joint positions for exact repetition
        stored_movement_joint_positions = []
        
        # Execute movements and capture one frame per movement (at end of each movement)
        for rep in range(repeat):
            print(f"\n--- Repeat {rep + 1}/{repeat} ---")
            
            for i, movement in enumerate(movements):
                move_type = movement.get('type')
                params = movement.get('parameters', {})
                
                print(f"  Movement {i+1}: {move_type} with params {params}")
                
                # If this is not the first repeat, use stored joint positions
                if rep > 0 and i < len(stored_movement_joint_positions):
                    print(f"    Using stored joint positions from first repeat")
                    stored_joint_pos = stored_movement_joint_positions[i]
                    self._set_joint_positions(stored_joint_pos.copy())
                    # Update IK solver's q0
                    self.ik_solver.q0 = stored_joint_pos[self.ik_solver.dof_ids].copy()
                    # Zero velocities
                    self.env.sim.data.qvel[:] = 0
                    self.env.sim.forward()
                    
                    # Capture frame
                    movement_frames = []
                    image = self._capture_image()
                    movement_frames.append(Image.fromarray(image))
                else:
                    # First repeat: execute movement normally
                    movement_frames = []
                    if move_type == 'position_move':
                        movement_frames = self._move_position(
                            direction=params.get('direction', 'right'),
                            distance=params.get('distance', 0.1),
                            speed=params.get('speed', 0.1),  # Speed in m/s
                            capture_frames=True,
                        )
                    elif move_type == 'orientation_move':
                        movement_frames = self._move_orientation(
                            joint=params.get('joint', 'yaw'),
                            angle=params.get('angle', 30),
                            direction=params.get('direction', 'right'),
                            speed=params.get('speed', 30.0),  # Angular speed in deg/s
                            capture_frames=True,
                        )
                    else:
                        print(f"    Warning: Unknown movement type '{move_type}'")
                    
                    # Store joint positions after movement (for first repeat only)
                    if rep == 0:
                        current_joint_pos = self._get_joint_positions()
                        stored_movement_joint_positions.append(current_joint_pos.copy())
                        print(f"    Stored joint positions for movement {i+1} (shape: {current_joint_pos.shape})")
                
                # Add frame captured at end of movement (should be exactly 1 frame)
                if len(movement_frames) > 0:
                    frames.extend(movement_frames)
                    print(f"    -> Captured {len(movement_frames)} frame(s) (total frames: {len(frames)})")
                else:
                    print(f"    Warning: No frames captured for movement {i+1}!")
                    # Even if no frames captured, add a frame to maintain frame count
                    if capture_frames:
                        image = self._capture_image()
                        frames.append(Image.fromarray(image))
                        print(f"    -> Added fallback frame (total frames: {len(frames)})")
        
        print(f"\n{'='*60}")
        print("Cue execution completed!")
        print(f"Captured {len(frames)} frames")
        expected_frames = 1 + len(movements) * repeat
        print(f"Expected: 1 (initial) + {len(movements)} (movements) * {repeat} (repeat) = {expected_frames}")
        
        # Save GIF if requested
        if save_gif and len(frames) > 0:
            if output_filename is None:
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{now}_{self.robot_name}_{pose_name}_p{pose_id}_r{repeat}_h{hold_time:.1f}.gif"
            
            filepath = os.path.join(self.output_dir, output_filename)
            
            # Ensure all frames are valid PIL Images
            valid_frames = []
            for i, frame in enumerate(frames):
                if frame is not None:
                    valid_frames.append(frame)
                else:
                    print(f"Warning: Frame {i} is None, skipping")
            
            print(f"Saving {len(valid_frames)} valid frames to GIF...")
            
            if len(valid_frames) > 0:
                valid_frames[0].save(
                    filepath,
                    save_all=True,
                    append_images=valid_frames[1:] if len(valid_frames) > 1 else [],
                    duration=gif_duration,
                    loop=0  # Infinite loop
                )
                print(f"Saved motion GIF: {filepath}")
                print(f"Total frames in GIF: {len(valid_frames)}")
                print(f"GIF duration: {len(valid_frames) * gif_duration / 1000:.2f} seconds")
            else:
                print("Error: No valid frames to save!")
        
        print(f"{'='*60}\n")
    
    def close(self):
        """Close the environment."""
        self.env.close()


def main(
    robot: str = "Panda",
    pose_index: Optional[int] = 416,
    env: str = "EmptySpace",
    controller: str = "IK_POSE",
    cue_name: str = "waving",
    repeat: int = 3,
    jsonl_path: str = "data/poses/closest_poses_results.jsonl",
    output_dir: str = "data/motions",
    save_gif: bool = True,
    output_filename: str = None,
    gif_duration: int = 100,
    hold_time: float = 2.0,
    capture_image_width: int = 512,
    capture_image_height: int = 512,
    camera_fov: float = 60.0,
    distance: float = 3,
    hz: int = 5,
):
    """
    Main function to execute non-verbal cues.
    
    Args:
        robot: Robot name
        env: Environment name
        controller: Controller name (should support IK)
        cue_name: Name of the cue to execute
        repeat: Number of times to repeat
        jsonl_path: Path to pose database JSONL file
    """
    
    # Define cues - maps cue names to pose + movements
    cues = {
        'waving': {
            'pose': 'Elbow_down',
            'movements': [
                {
                    'type': 'position_move',
                    'parameters': {
                        'direction': 'up',
                        'distance': distance,
                        'speed': 1.0,
                    }
                },
                {
                    'type': 'position_move',
                    'parameters': {
                        'direction': 'down',
                        'distance': distance,
                        'speed': 1.0,
                    }
                },
            ]
        },
        'nodding': {
            'pose': 'Elbow_up',
            'movements': [
                {
                    'type': 'orientation_move',
                    'parameters': {
                        'joint': 'pitch',
                        'angle': 15,
                        'direction': 'down',
                        'speed': 1.0,
                    }
                },
                {
                    'type': 'orientation_move',
                    'parameters': {
                        'joint': 'pitch',
                        'angle': 15,
                        'direction': 'up',
                        'speed': 1.0,
                    }
                },
            ]
        },
        'pointing': {
            'pose': 'Stretched_out',
            'movements': [
                {
                    'type': 'position_move',
                    'parameters': {
                        'direction': 'forward',
                        'distance': 0.2,
                        'speed': 0.8,
                    }
                },
            ]
        },
    }
    
    if cue_name not in cues:
        print(f"Error: Cue '{cue_name}' not defined. Available cues: {list(cues.keys())}")
        return
    
    cue_def = cues[cue_name]
    
    # Initialize generator
    generator = NonVerbalCueGenerator(
        robot_name=robot,
        env_name=env,
        controller_name=controller,
        jsonl_path=jsonl_path,
        has_renderer=False,
        has_offscreen_renderer=True,
        output_dir=output_dir,
        capture_image_width=capture_image_width,
        capture_image_height=capture_image_height,
        camera_fov=camera_fov,
        hz=hz,
    )
    
    try:
        # Execute cue
        generator.execute_cue(
            pose_name=cue_def['pose'],
            movements=cue_def['movements'],
            repeat=repeat,
            hold_time=hold_time,
            save_gif=save_gif,
            output_filename=output_filename,
            gif_duration=gif_duration,
            pose_index=pose_index,
        )
        
    finally:
        generator.close()


if __name__ == "__main__":
    fire.Fire(main)
