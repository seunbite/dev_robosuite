"""
Generate robot motions based on cue definitions.

This script:
1. Loads pose definitions and finds matching poses from JSONL
2. Uses JacobianCalculator to find best joints for specified axis
3. Selects joint based on proximal/distal preference
4. Executes movements and generates GIF
"""

import fire
import os
import json
import random
import numpy as np
from typing import Dict, List, Optional
from PIL import Image
from datetime import datetime

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.utils.ik_utils import IKSolver
import mujoco

# Import JacobianCalculator and pose configuration
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alphabet_jacobian import JacobianCalculator
from pose_config import direction_pose_set, pose_set, poses, pitch_poses, height_map

# Fixed joint indices
FIXED_JOINT_INDICES = {
    'GR1': "0-2, 20-31"
}


class MotionGenerator:
    """Generate robot motions based on cue definitions."""
    
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
        camera_distance: float = 1.8,
        hz: int = 4,
    ):
        """
        Initialize the motion generator.
        
        Args:
            robot_name: Name of the robot
            env_name: Name of the environment
            controller_name: Name of the controller
            jsonl_path: Path to JSONL file with pose data
            has_renderer: Whether to show on-screen rendering
            has_offscreen_renderer: Whether to enable offscreen rendering
            control_freq: Control frequency
            output_dir: Output directory for GIFs
            capture_image_width: Image width for capture
            capture_image_height: Image height for capture
            camera_distance: Multiplier for camera FOV to zoom out (1.8 = default, >1.0 = wider view)
            hz: Frame rate for GIF generation (frames per second)
        """
        self.robot_name = robot_name
        self.env_name = env_name
        self.controller_name = controller_name
        self.jsonl_path = jsonl_path
        self.control_freq = control_freq
        self.output_dir = os.path.join(output_dir, robot_name)
        self.capture_image_width = capture_image_width
        self.capture_image_height = capture_image_height
        self.camera_distance = camera_distance
        self.hz = hz
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
        # Initialize JacobianCalculator for joint analysis
        self.jacobian_calculator = JacobianCalculator(
            robot_name=robot_name,
            env_name=env_name,
            controller_name=controller_name,
            jsonl_path=jsonl_path,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            control_freq=control_freq,
        )
        
        # Get environment and robot from calculator
        self.env = self.jacobian_calculator.env
        self.robot = self.jacobian_calculator.robot
        self.initial_joint_pos = self.jacobian_calculator.initial_joint_pos
        
        # Adjust camera FOV to zoom out (instead of moving position)
        # Larger FOV = wider view = can see more
        camera_name = "frontview"
        try:
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            # Get current FOV (default is usually around 45-55 degrees)
            current_fov = self.env.sim.model.cam_fovy[cam_id]
            # Scale FOV: camera_distance > 1.0 means zoom out (larger FOV)
            new_fov = current_fov * camera_distance
            # Clamp to reasonable range (20-120 degrees) - increased max for upward pointing robots
            new_fov = max(20.0, min(120.0, new_fov))
            self.env.sim.model.cam_fovy[cam_id] = new_fov
            print(f"Camera FOV adjusted: {current_fov:.1f}° -> {new_fov:.1f}° (zoom factor: {camera_distance})")
        except Exception as e:
            print(f"Warning: Could not adjust camera FOV: {e}")
        
        print(f"Motion generator initialized for {robot_name}")
    
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
        pose_def,  # Can be string (pose name) or dict (pose definition)
        robot_name: Optional[str] = None,
    ) -> List[Dict]:
        """Find poses matching the given pose name or definition."""
        if robot_name is None:
            robot_name = self.robot_name
        
        # Handle both string (pose name) and dict (pose definition) formats
        if isinstance(pose_def, str):
            # Old format: pose name as string
            pose_name = pose_def
            if pose_name not in direction_pose_set:
                print(f"Error: Pose '{pose_name}' not found in direction_pose_set")
                return []
            pose_def = direction_pose_set[pose_name]
        elif isinstance(pose_def, dict):
            # New format: pose definition as dict with height/dir/pitch
            pass
        else:
            print(f"Error: Invalid pose format: {type(pose_def)}")
            return []
        
        # New format: uses direction and pitch (height is ignored for now as it's not stored in JSONL)
        # Get all roll/yaw combinations for the direction and all pitch values
        direction_name = pose_def.get('dir')
        pitch_name = pose_def.get('pitch')
        height = height_map.get(pose_def.get('height'))  # Note: height is not used for matching currently
        
        if not direction_name or not pitch_name:
            print(f"Error: Pose definition missing 'dir' or 'pitch' field: {pose_def}")
            return []
        
        # Get direction poses and pitch values
        direction_poses = poses.get(direction_name, [])
        pitch_values = pitch_poses.get(pitch_name, [])
        
        if not direction_poses:
            print(f"Warning: No roll/yaw combinations found for direction '{direction_name}'")
            print(f"Available directions: {list(poses.keys())}")
            return []
        
        if not pitch_values:
            print(f"Warning: No pitch values found for pitch '{pitch_name}'")
            print(f"Available pitches: {list(pitch_poses.keys())}")
            return []
        
        # Collect all matching poses from JSONL
        all_matching_poses = []
        print(f"  Searching for poses with direction='{direction_name}' ({len(direction_poses)} roll/yaw combos) and pitch='{pitch_name}' ({len(pitch_values)} values)")
        
        for dir_pose in direction_poses:
            for pitch_val in pitch_values:
                matching_poses = self.jacobian_calculator._find_matching_poses(
                    roll_deg=dir_pose['roll'],
                    pitch_deg=pitch_val,
                    yaw_deg=dir_pose['yaw'],
                    robot_name=robot_name,
                )
                if matching_poses:
                    print(f"    Found {len(matching_poses)} poses for roll={dir_pose['roll']}, pitch={pitch_val}, yaw={dir_pose['yaw']}")
                all_matching_poses.extend(matching_poses)
        
        if not all_matching_poses:
            print(f"  Warning: No poses found in JSONL for robot={robot_name}, direction={direction_name}, pitch={pitch_name}")
            print(f"  Tried {len(direction_poses) * len(pitch_values)} roll/pitch/yaw combinations")
        else:
            print(f"  Total: Found {len(all_matching_poses)} matching poses")
        
        return all_matching_poses
    
    def _select_joint(
        self,
        pose_def,  # Can be string (pose name) or dict (pose definition)
        axis: str,
        joint_preference: str,  # 'proximal' or 'distal'
        score_threshold: float = 0.1,  # Minimum score threshold
        max_pose_attempts: int = 10,  # Maximum number of pose attempts
    ) -> tuple:
        """
        Select a joint based on axis and joint preference.
        Tries multiple poses if needed to find a valid joint.
        
        Args:
            pose_def: Pose definition (string name or dict with height/dir/pitch)
            axis: 'x', 'y', or 'z'
            joint_preference: 'proximal' or 'distal'
            score_threshold: Minimum score threshold for joint selection
            max_pose_attempts: Maximum number of pose attempts
        
        Returns:
            Tuple of (joint_idx, joint_name, joint_dof_id, score)
        """
        matching_poses = self._find_matching_poses(pose_def)
        if not matching_poses:
            pose_display = pose_def if isinstance(pose_def, str) else f"{pose_def.get('height', '?')}_{pose_def.get('dir', '?')}_{pose_def.get('pitch', '?')}"
            raise ValueError(f"No matching poses found for {pose_display}")
        
        # Try multiple poses until we find a valid joint
        random.shuffle(matching_poses)  # Shuffle to try different poses
        attempted_poses = set()
        last_sorted_joints = None
        last_filtered_sorted_joints = None
        
        for attempt in range(min(max_pose_attempts, len(matching_poses))):
            # Select a pose that hasn't been tried yet
            remaining_poses = [p for p in matching_poses if id(p) not in attempted_poses]
            if not remaining_poses:
                break
            
            selected_pose = random.choice(remaining_poses)
            attempted_poses.add(id(selected_pose))
            
            self.jacobian_calculator._set_pose_from_data(selected_pose)
            
            # Compute Jacobian
            mujoco_model = self.env.sim.model._model
            mujoco_data = self.env.sim.data._data
            site_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_SITE, self.jacobian_calculator.eef_site_name)
            
            jac_pos = np.zeros((3, mujoco_model.nv))
            jac_rot = np.zeros((3, mujoco_model.nv))
            mujoco.mj_jacSite(mujoco_model, mujoco_data, jac_pos, jac_rot, site_id)
            jac_full = np.vstack([jac_pos, jac_rot])
            dof_ids = self.jacobian_calculator.ik_solver.dof_ids
            jac_subset = jac_full[:, dof_ids]
            
            # Get joint names
            joint_names_list = []
            for dof_id in dof_ids:
                if dof_id < len(self.jacobian_calculator.joint_names):
                    joint_names_list.append(self.jacobian_calculator.joint_names[dof_id])
                else:
                    joint_names_list.append(f"DOF_{dof_id}")
            
            # Get sorted joints
            sorted_joints = self.jacobian_calculator._find_and_sort_joints_for_axis(
                jac_subset, dof_ids, joint_names_list, axis=axis
            )
            last_sorted_joints = sorted_joints  # Keep for fallback
            
            # Filter joints by score threshold
            threshold_filtered_joints = [
                joint for joint in sorted_joints 
                if joint[3] >= score_threshold  # joint[3] is score
            ]
            
            if not threshold_filtered_joints:
                print(f"  Attempt {attempt + 1}: No joints above threshold {score_threshold}, trying different pose...")
                continue
            
            # Filter out distal joints that mainly contribute to roll rotation
            rot_jac = jac_subset[3:6, :]  # 3 x num_joints
            roll_jac = rot_jac[0, :]  # roll (x-axis rotation) contribution
            
            roll_threshold = 0.5
            high_roll_joints = set()
            
            sorted_by_dof = sorted(enumerate(dof_ids), key=lambda x: x[1], reverse=True)
            num_distal = max(1, len(dof_ids) // 3)
            
            for i, dof_id in sorted_by_dof[:num_distal]:
                if abs(roll_jac[i]) > roll_threshold:
                    high_roll_joints.add(dof_id)
            
            # Filter out high roll joints
            filtered_sorted_joints = [
                joint for joint in threshold_filtered_joints 
                if joint[2] not in high_roll_joints
            ]
            
            if not filtered_sorted_joints:
                filtered_sorted_joints = threshold_filtered_joints
            last_filtered_sorted_joints = filtered_sorted_joints  # Keep for fallback
            
            # Split joints into proximal and distal halves based on DOF ID
            sorted_by_dof_id = sorted(filtered_sorted_joints, key=lambda x: x[2])  # x[2] is joint_dof_id
            mid_point = len(sorted_by_dof_id) // 2
            proximal_half = sorted_by_dof_id[:mid_point]
            distal_half = sorted_by_dof_id[mid_point:]
            
            # Select based on joint preference
            if joint_preference == 'proximal':
                candidate_joints = proximal_half
                if not candidate_joints:
                    print(f"  Attempt {attempt + 1}: No joints in proximal half, trying different pose...")
                    continue
                selected = max(candidate_joints, key=lambda x: x[3])  # Highest score in proximal half
            elif joint_preference == 'distal':
                candidate_joints = distal_half
                if not candidate_joints:
                    print(f"  Attempt {attempt + 1}: No joints in distal half, trying different pose...")
                    continue
                selected = max(candidate_joints, key=lambda x: x[3])  # Highest score in distal half
            else:
                selected = filtered_sorted_joints[0]  # Best overall
            
            # Get z-axis Jacobian sign for the selected joint
            joint_idx_in_sorted = next(i for i, j in enumerate(sorted_joints) if j == selected)
            jac_subset_for_axis = jac_subset[0:3, :]  # Position Jacobian
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            axis_idx = axis_map.get(axis, 1)
            selected_joint_jac_value = jac_subset_for_axis[axis_idx, selected[0]]  # selected[0] is joint_idx in jac_subset
            
            print(f"\nSelected joint: {selected[1]} (DOF ID: {selected[2]}, Score: {selected[3]:.4f}, Rank: {sorted_joints.index(selected) + 1})")
            print(f"Joint preference: {joint_preference}")
            print(f"Pose attempt: {attempt + 1}")
            print(f"Jacobian {axis.upper()}-axis value for selected joint: {selected_joint_jac_value:.6f}")
            
            # Return: (joint_idx, joint_name, joint_dof_id, score, jac_sign)
            return selected + (np.sign(selected_joint_jac_value),)
        
        # If all attempts failed, use the best joint from the last pose
        print(f"  Warning: Could not find valid joint after {max_pose_attempts} attempts, using best joint from last pose")
        if last_filtered_sorted_joints:
            selected = last_filtered_sorted_joints[0]
        elif last_sorted_joints:
            selected = last_sorted_joints[0]
        else:
            pose_display = pose_def if isinstance(pose_def, str) else f"{pose_def.get('height', '?')}_{pose_def.get('dir', '?')}_{pose_def.get('pitch', '?')}"
            raise ValueError(f"Could not find any valid joint for {pose_display} with axis {axis} and preference {joint_preference}")
        
        # Get axis Jacobian sign for the selected joint (fallback case)
        # Use the last pose that was tried
        if matching_poses:
            self.jacobian_calculator._set_pose_from_data(matching_poses[0])
            mujoco_model = self.env.sim.model._model
            mujoco_data = self.env.sim.data._data
            site_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_SITE, self.jacobian_calculator.eef_site_name)
            jac_pos = np.zeros((3, mujoco_model.nv))
            jac_rot = np.zeros((3, mujoco_model.nv))
            mujoco.mj_jacSite(mujoco_model, mujoco_data, jac_pos, jac_rot, site_id)
            jac_full = np.vstack([jac_pos, jac_rot])
            dof_ids = self.jacobian_calculator.ik_solver.dof_ids
            jac_subset = jac_full[:, dof_ids]
            jac_subset_for_axis = jac_subset[0:3, :]
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            axis_idx = axis_map.get(axis, 1)
            selected_joint_jac_value = jac_subset_for_axis[axis_idx, selected[0]]
        else:
            selected_joint_jac_value = 1.0  # Default to positive if we can't compute
        
        print(f"Selected joint: {selected[1]} (DOF ID: {selected[2]}, Score: {selected[3]:.4f})")
        print(f"Jacobian {axis.upper()}-axis value for selected joint: {selected_joint_jac_value:.6f}")
        
        # Return: (joint_idx, joint_name, joint_dof_id, score, jac_sign)
        return selected + (np.sign(selected_joint_jac_value),)
    
    def _quantize_joint_angles(self, joint_angles_deg: List[float], standard_angles: List[float] = None) -> List[float]:
        """
        Quantize joint angles to nearest standard angles.
        
        Args:
            joint_angles_deg: List of joint angles in degrees
            standard_angles: List of standard angles to quantize to (default: [-90, -60, -45, -30, 0, 30, 45, 60, 90])
        
        Returns:
            List of quantized joint angles in degrees
        """
        if standard_angles is None:
            standard_angles = [-90, -60, -45, -30, 0, 30, 45, 60, 90]
        
        quantized = []
        for angle in joint_angles_deg:
            # Find nearest standard angle
            nearest = min(standard_angles, key=lambda x: abs(x - angle))
            quantized.append(nearest)
        
        return quantized
    
    def _find_closest_quantized_pose(
        self,
        current_pose: Dict,
        candidate_poses: List[Dict],
        standard_angles: List[float] = None,
    ) -> Dict:
        """
        Find the candidate pose that is closest to current pose after quantizing to standard angles.
        
        Args:
            current_pose: Current pose dictionary with joint_angles_deg
            candidate_poses: List of candidate pose dictionaries
            standard_angles: List of standard angles for quantization
        
        Returns:
            The closest candidate pose dictionary
        """
        if standard_angles is None:
            standard_angles = [-90, 0, 90]
        
        # Get current joint angles
        current_joint_angles_deg = current_pose.get("joint_angles_deg", [])
        current_active_indices = current_pose.get("active_joint_indices", [])
        
        # Quantize current pose angles
        current_quantized = self._quantize_joint_angles(current_joint_angles_deg, standard_angles)
        
        # Create a mapping from active joint index to quantized angle
        current_quantized_dict = {}
        for i, active_idx in enumerate(current_active_indices):
            if i < len(current_quantized):
                current_quantized_dict[active_idx] = current_quantized[i]
        
        # Score each candidate pose
        best_pose = None
        best_score = float('inf')
        
        for candidate_pose in candidate_poses:
            candidate_joint_angles_deg = candidate_pose.get("joint_angles_deg", [])
            candidate_active_indices = candidate_pose.get("active_joint_indices", [])
            
            # Quantize candidate pose angles
            candidate_quantized = self._quantize_joint_angles(candidate_joint_angles_deg, standard_angles)
            
            # Create mapping for candidate
            candidate_quantized_dict = {}
            for i, active_idx in enumerate(candidate_active_indices):
                if i < len(candidate_quantized):
                    candidate_quantized_dict[active_idx] = candidate_quantized[i]
            
            # Calculate difference (only for joints present in both)
            common_indices = set(current_quantized_dict.keys()) & set(candidate_quantized_dict.keys())
            if not common_indices:
                # No common joints, skip
                continue
            
            score = 0.0
            for idx in common_indices:
                diff = abs(current_quantized_dict[idx] - candidate_quantized_dict[idx])
                score += diff
            
            if score < best_score:
                best_score = score
                best_pose = candidate_pose
        
        return best_pose if best_pose is not None else candidate_poses[0]
    
    def _pose_data_to_joint_positions(self, pose_data: Dict) -> np.ndarray:
        """
        Convert pose data to joint positions array.
        
        Args:
            pose_data: Dictionary containing pose information with joint_angles_rad and active_joint_indices
        
        Returns:
            Joint positions array in radians
        """
        joint_angles_rad = pose_data.get("joint_angles_rad", [])
        active_joint_indices = pose_data.get("active_joint_indices", [])
        
        # Start with initial joint positions
        joint_pos = self.initial_joint_pos.copy()
        
        # Set positions for active joints
        for i, active_joint_idx in enumerate(active_joint_indices):
            if i < len(joint_angles_rad):
                if active_joint_idx < len(joint_pos):
                    joint_pos[active_joint_idx] = joint_angles_rad[i]
        
        return joint_pos
    
    def _get_joint_positions(self):
        """Get current joint positions."""
        return self.robot._joint_positions.copy()
    
    def _set_joint_positions(self, joint_positions):
        """Set joint positions."""
        self.robot.set_robot_joint_positions(joint_positions)
        self.env.sim.forward()
        
        # Stabilize
        for _ in range(10):
            self.env.sim.data.qvel[:] = 0
            self.env.sim.forward()
    
    def _capture_image(self):
        """Capture current camera view."""
        obs = self.env.sim.render(
            camera_name="frontview",
            width=self.capture_image_width,
            height=self.capture_image_height,
            depth=False
        )
        return obs[::-1]
    
    def _check_self_collision(self) -> bool:
        """
        Check if the robot has self-collision.
        
        Returns:
            bool: True if self-collision is detected, False otherwise
        """
        # Get robot contact geoms from robot_model
        robot_geoms = set(self.robot.robot_model.contact_geoms)
        
        # Check all contacts
        for i in range(self.env.sim.data.ncon):
            contact = self.env.sim.data.contact[i]
            geom1_name = self.env.sim.model.geom_id2name(contact.geom1)
            geom2_name = self.env.sim.model.geom_id2name(contact.geom2)
            
            # Check if both geoms are from the robot (self-collision)
            if geom1_name in robot_geoms and geom2_name in robot_geoms:
                return True
        
        return False
    
    def _find_joint_index_in_robot(self, joint_dof_id: int) -> Optional[int]:
        """Find the index in robot's joint position array for a given DOF ID."""
        if hasattr(self.robot, '_ref_joint_pos_indexes'):
            for i, qpos_addr in enumerate(self.robot._ref_joint_pos_indexes):
                if isinstance(qpos_addr, (int, np.integer)):
                    if qpos_addr == joint_dof_id:
                        return i
                elif isinstance(qpos_addr, (tuple, list)):
                    start_addr = qpos_addr[0] if isinstance(qpos_addr, tuple) else qpos_addr
                    if start_addr <= joint_dof_id < start_addr + len(qpos_addr):
                        return i
        return None
    
    def _load_cue_config(self, cue: str, config_path: str = "data/seed/motion_config.json") -> Dict:
        """
        Load cue configuration from JSON file.
        
        Args:
            cue: Name of the cue
            config_path: Path to JSON file with cue configurations
        
        Returns:
            Dictionary containing cue configuration
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Cue configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            configs = json.load(f)
        
        # configs is a list of cue configurations
        for config in configs:
            if config.get('cue') == cue:
                return config
        
        raise ValueError(f"Cue '{cue}' not found in configuration file: {config_path}")
    
    def execute_cue(
        self,
        cue: str,
        pose_index: Optional[int] = None,
        config_path: str = "data/seed/motion_config.json",
        proximal_degree_scale: float = 0.25,
        hz: int = 4,
        filename_suffix: Optional[str] = None,
        enable_self_collision_check: bool = False,
    ):
        """
        Execute a cue (e.g., 'waving').
        
        Args:
            cue: Name of the cue
            pose_index: Optional pose_id to use (if None, randomly selects)
            config_path: Path to JSONL file with cue configurations
        """
        print(f"\n{'='*60}")
        print(f"Executing cue: {cue}")
        print(f"{'='*60}\n")
        
        # Load cue configuration from JSON file
        cue_config_data = self._load_cue_config(cue, config_path)
        
        # Extract movements list
        movements = cue_config_data.get('movements', [])
        
        if not movements:
            raise ValueError(f"No movements found in cue '{cue}' configuration")
        
        # Process each movement item
        frames = []
        current_pose_name = None
        current_pose = None
        pose_id = None
        
        # Cache for joint selection: (axis, joint_preference) -> (joint_idx, joint_name, joint_dof_id, score)
        joint_cache = {}
        
        for movement_item in movements:
            movement_type = movement_item.get('type')
            parameters = movement_item.get('parameters', {})
            
            if movement_type == 'pose':
                # Set pose
                pose_param = parameters.get('pose')
                if pose_param is None:
                    raise ValueError("'pose' parameter is required for 'pose' type movement")
                
                # Handle both string (pose name) and dict (pose definition) formats
                if isinstance(pose_param, str):
                    pose_display_name = pose_param
                elif isinstance(pose_param, dict):
                    # Create display name from dict
                    pose_display_name = f"{pose_param.get('height', '?')}_{pose_param.get('dir', '?')}_{pose_param.get('pitch', '?')}"
                else:
                    raise ValueError(f"Invalid pose format: {type(pose_param)}")
                
                print(f"\n--- Setting Pose: {pose_display_name} ---")
                matching_poses = self._find_matching_poses(pose_param)
                
                if not matching_poses:
                    raise ValueError(f"No matching poses found for {pose_display_name}")
                
                # Select pose
                if pose_index is not None and current_pose is None:
                    # Only use pose_index for the first pose
                    selected_pose = None
                    for pose in matching_poses:
                        if pose.get('pose_id') == pose_index:
                            selected_pose = pose
                            break
                    
                    if selected_pose is None:
                        print(f"Warning: pose_id {pose_index} not found, using random selection")
                        selected_pose = random.choice(matching_poses)
                        pose_id = selected_pose['pose_id']
                    else:
                        pose_id = selected_pose['pose_id']
                        print(f"Selected pose with pose_id {pose_id}: rank {selected_pose.get('rank', 'N/A')}")
                else:
                    # If there's a current pose, find the closest quantized pose
                    if current_pose is not None:
                        selected_pose = self._find_closest_quantized_pose(current_pose, matching_poses)
                        print(f"Found closest quantized pose with pose_id {selected_pose['pose_id']}: rank {selected_pose.get('rank', 'N/A')}")
                    else:
                        selected_pose = random.choice(matching_poses)
                        print(f"Randomly selected pose with pose_id {selected_pose['pose_id']}: rank {selected_pose.get('rank', 'N/A')}")
                    if pose_id is None:
                        pose_id = selected_pose['pose_id']
                
                # Get speed and hold_time from parameters
                speed = parameters.get('speed', 1.0)
                hold_time = parameters.get('hold_time', 1.0)
                
                if current_pose is None:
                    # First pose: just set it and hold (no speed/interpolation needed)
                    print("Setting first pose...")
                    self.jacobian_calculator._set_pose_from_data(selected_pose)
                    current_pose_name = pose_param  # Store the pose parameter (string or dict)
                    current_pose = selected_pose
                    
                    # Hold at pose for hold_time
                    num_frames = int(hold_time * hz)
                    for _ in range(num_frames):
                        image = self._capture_image()
                        frames.append(Image.fromarray(image))
                    print(f"Captured {num_frames} frames (hold_time: {hold_time}s, hz: {hz})")
                else:
                    # Transition from current pose to selected pose with speed
                    # Create display names for logging
                    if isinstance(current_pose_name, str):
                        current_display_name = current_pose_name
                    else:
                        current_display_name = f"{current_pose_name.get('height', '?')}_{current_pose_name.get('dir', '?')}_{current_pose_name.get('pitch', '?')}"
                    
                    print(f"Transitioning from {current_display_name} to {pose_display_name} (speed: {speed})...")
                    
                    # Get joint positions for both poses
                    start_joint_pos = self._pose_data_to_joint_positions(current_pose)
                    end_joint_pos = self._pose_data_to_joint_positions(selected_pose)
                    
                    # Calculate movement duration and number of frames
                    duration = 1.0 / speed
                    num_transition_frames = int(duration * hz)
                    
                    # Interpolate from start to end pose
                    for frame_idx in range(num_transition_frames):
                        t = (frame_idx + 1) / num_transition_frames  # 0 to 1
                        interpolated_joint_pos = start_joint_pos * (1 - t) + end_joint_pos * t
                        
                        self._set_joint_positions(interpolated_joint_pos)
                        
                        # Capture frame
                        image = self._capture_image()
                        frames.append(Image.fromarray(image))
                    
                    print(f"  Captured {num_transition_frames} transition frames (speed: {speed}, duration: {duration:.2f}s)")
                    
                    # Set final pose (to ensure exact position)
                    self.jacobian_calculator._set_pose_from_data(selected_pose)
                    current_pose_name = pose_param  # Store the pose parameter (string or dict)
                    current_pose = selected_pose
                    
                    # Hold at final pose for hold_time
                    num_hold_frames = int(hold_time * hz)
                    for _ in range(num_hold_frames):
                        image = self._capture_image()
                        frames.append(Image.fromarray(image))
                    print(f"  Captured {num_hold_frames} hold frames (hold_time: {hold_time}s)")
                
            elif movement_type == 'movement':
                # Execute movement
                if current_pose_name is None:
                    raise ValueError("No pose set before movement. Please set a pose first.")
                
                repetition = parameters.get('repetition', 1)
                axis = parameters.get('axis')
                joint_preference = parameters.get('joint')
                
                # Support new format: degrees array (with +, - signs)
                degrees_array = parameters.get('degrees', [])
                # Support old format: directions array (for backward compatibility)
                directions = parameters.get('directions', [])
                
                if axis is None:
                    raise ValueError("'axis' parameter is required for 'movement' type")
                if joint_preference is None:
                    raise ValueError("'joint' parameter is required for 'movement' type")
                
                # Convert old format to new format if needed
                if directions and not degrees_array:
                    # Old format: convert directions to degrees array
                    degrees_array = []
                    for direction_config in directions:
                        direction = direction_config.get('direction')
                        degrees = direction_config.get('degrees', 0)
                        if direction == 'pos':
                            degrees_array.append(degrees)
                        elif direction == 'neg':
                            degrees_array.append(-degrees)
                        else:
                            # If no direction field, use degrees as-is (might be negative)
                            degrees_array.append(degrees)
                    speed = parameters.get('speed', directions[0].get('speed', 1.0) if directions else 1.0)
                    hold_time = parameters.get('hold_time', directions[0].get('hold_time', 0.5) if directions else 0.5)
                elif degrees_array:
                    # New format: degrees array with +, - signs
                    speed = parameters.get('speed', 1.0)
                    hold_time = parameters.get('hold_time', 0.5)
                else:
                    raise ValueError("Either 'degrees' or 'directions' parameter is required for 'movement' type")
                
                if not degrees_array:
                    raise ValueError("'degrees' array is empty")
                
                print(f"\n--- Movement ---")
                print(f"Repetition: {repetition}")
                print(f"Axis: {axis}")
                print(f"Joint preference: {joint_preference}")
                print(f"Degrees: {degrees_array}")
                print(f"Speed: {speed}")
                print(f"Hold time: {hold_time}")
                
                # Get current joint positions before joint selection
                current_joint_pos = self._get_joint_positions()
                
                # Check if we already selected a joint for this (axis, joint_preference) combination
                cache_key = (axis, joint_preference)
                if cache_key in joint_cache:
                    # Reuse the cached joint selection
                    joint_idx, joint_name, joint_dof_id, score, jac_sign = joint_cache[cache_key]
                    print(f"\nReusing cached joint selection: {joint_name} (DOF ID: {joint_dof_id}, Score: {score:.4f})")
                    print(f"Cache key: axis={axis}, joint_preference={joint_preference}")
                else:
                    # Select joint (this will set the pose again, so we need to restore after)
                    joint_idx, joint_name, joint_dof_id, score, jac_sign = self._select_joint(
                        pose_def=current_pose_name,
                        axis=axis,
                        joint_preference=joint_preference,
                    )
                    
                    # Cache the selected joint for this combination
                    joint_cache[cache_key] = (joint_idx, joint_name, joint_dof_id, score, jac_sign)
                    print(f"\nCached joint selection for axis={axis}, joint_preference={joint_preference}")
                
                # Restore pose after joint selection
                self._set_joint_positions(current_joint_pos)
                
                # Find joint index in robot's joint position array
                robot_joint_idx = self._find_joint_index_in_robot(joint_dof_id)
                if robot_joint_idx is None:
                    raise ValueError(f"Could not find joint index for DOF ID {joint_dof_id}")
                
                # Execute repetitions
                for rep in range(repetition):
                    print(f"\nRepetition {rep + 1}/{repetition}")
                    
                    # Execute each degree value in the array
                    for deg_idx, degrees in enumerate(degrees_array):
                        # Get expected sign if specified
                        expected_sign = None
                        if 'directions' in parameters and deg_idx < len(parameters['directions']):
                            sign_str = parameters['directions'][deg_idx].get('sign')
                            if sign_str == 'positive':
                                expected_sign = 1
                            elif sign_str == 'negative':
                                expected_sign = -1
                        
                        # Apply proximal degree scaling if joint is proximal
                        original_degrees = degrees
                        if joint_preference == 'proximal':
                            degrees = degrees * proximal_degree_scale
                            print(f"  Applied proximal scaling: {degrees}° (original: {original_degrees}°)")
                        else:
                            print(f"  Movement: {degrees}°")
                        
                        # Adjust degree sign based on Jacobian sign for z-axis upward movement
                        # For z-axis, if Jacobian is negative, we need to flip the sign to move upward
                        if axis == 'z' and jac_sign < 0:
                            degrees = -degrees
                            if original_degrees != degrees:
                                print(f"  Adjusted degree sign for upward movement: {degrees}° (original: {original_degrees}°)")
                        
                        # Verify expected sign if specified
                        if expected_sign is not None:
                            actual_sign = np.sign(degrees)
                            if actual_sign == 0:
                                actual_sign = 1  # Treat 0 as positive
                            
                            if actual_sign != expected_sign:
                                print(f"  Sign mismatch detected! Expected: {expected_sign}, Actual: {actual_sign}")
                                print(f"  Flipping degree sign: {degrees}° -> {-degrees}°")
                                degrees = -degrees
                        
                        # Calculate target angle offset (degrees already has sign)
                        original_target_angle_offset_rad = np.deg2rad(degrees)
                        
                        # Calculate movement duration and number of frames
                        # speed=1 means 1 second to reach target, speed=2 means 0.5 seconds
                        duration = 1.0 / speed
                        num_movement_frames = int(duration * hz)
                        
                        # Get starting angle
                        start_angle_rad = current_joint_pos[robot_joint_idx]
                        
                        # Try to find a safe angle by reducing the angle if collision is detected
                        if enable_self_collision_check:
                            min_angle_scale = 0.1  # Minimum scale (10% of original)
                            angle_scale = 1.0  # Start with full angle
                            safe_target_angle_offset_rad = original_target_angle_offset_rad
                            safe_degrees = degrees
                            max_attempts = 5
                            has_collision = True
                            
                            for attempt in range(max_attempts):
                                test_target_angle_rad = start_angle_rad + safe_target_angle_offset_rad
                                
                                # Check for self-collision
                                test_joint_pos = current_joint_pos.copy()
                                test_joint_pos[robot_joint_idx] = test_target_angle_rad
                                self._set_joint_positions(test_joint_pos)
                                has_collision = self._check_self_collision()
                                
                                if not has_collision:
                                    # Safe angle found
                                    if attempt > 0:
                                        print(f"  Found safe angle at {attempt + 1} attempt: {safe_degrees:.1f}° (reduced from {degrees:.1f}°)")
                                    break
                                
                                # Collision detected - reduce angle and try again
                                if attempt < max_attempts - 1:
                                    angle_scale *= 0.5  # Reduce by half
                                    if angle_scale < min_angle_scale:
                                        # Too small, skip this movement
                                        print(f"  Warning: Self-collision detected even at minimum angle ({min_angle_scale * 100:.0f}% of original), skipping this movement")
                                        self._set_joint_positions(current_joint_pos)
                                        has_collision = True  # Mark as still colliding
                                        break
                                    
                                    safe_target_angle_offset_rad = original_target_angle_offset_rad * angle_scale
                                    safe_degrees = degrees * angle_scale
                                    print(f"  Self-collision detected, reducing angle to {safe_degrees:.1f}° (attempt {attempt + 2}/{max_attempts})...")
                            
                            if has_collision:
                                # Could not find safe angle, skip this movement
                                self._set_joint_positions(current_joint_pos)
                                continue
                        else:
                            # No self-collision check - use original angle directly
                            safe_target_angle_offset_rad = original_target_angle_offset_rad
                            safe_degrees = degrees
                        
                        # Use the safe angle
                        target_angle_offset_rad = safe_target_angle_offset_rad
                        target_angle_rad = start_angle_rad + target_angle_offset_rad
                        
                        # Restore to start position for interpolation
                        self._set_joint_positions(current_joint_pos)
                        
                        # Interpolate from start to target
                        for frame_idx in range(num_movement_frames):
                            t = (frame_idx + 1) / num_movement_frames  # 0 to 1
                            interpolated_angle_rad = start_angle_rad + t * target_angle_offset_rad
                            
                            new_joint_pos = current_joint_pos.copy()
                            new_joint_pos[robot_joint_idx] = interpolated_angle_rad
                            self._set_joint_positions(new_joint_pos)
                            
                            # Capture frame
                            image = self._capture_image()
                            frames.append(Image.fromarray(image))
                        
                        if safe_degrees != degrees:
                            print(f"  Captured {num_movement_frames} movement frames: {safe_degrees:.1f}° (reduced from {degrees:.1f}°, speed: {speed}, duration: {duration:.2f}s)")
                        else:
                            print(f"  Captured {num_movement_frames} movement frames: {degrees}° (speed: {speed}, duration: {duration:.2f}s)")
                        
                        # Update current_joint_pos to target position
                        current_joint_pos[robot_joint_idx] = target_angle_rad
                        
                        # Hold at target position if hold_time > 0
                        if hold_time > 0:
                            num_hold_frames = int(hold_time * hz)
                            for _ in range(num_hold_frames):
                                image = self._capture_image()
                                frames.append(Image.fromarray(image))
                            print(f"  Captured {num_hold_frames} hold frames (hold_time: {hold_time}s)")
            else:
                raise ValueError(f"Unknown movement type: {movement_type}")
        
        # Save GIF
        if len(frames) > 0:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            if pose_id is not None:
                base_filename = f"{now}_{self.robot_name}_{cue}_p{pose_id}"
            else:
                base_filename = f"{now}_{self.robot_name}_{cue}"
            
            # Add suffix if provided (e.g., prompt variation)
            if filename_suffix:
                output_filename = f"{base_filename}_{filename_suffix}.gif"
            else:
                output_filename = f"{base_filename}.gif"
            
            filepath = os.path.join(self.output_dir, output_filename)
            
            # Calculate frame duration in ms (1000ms / hz)
            frame_duration_ms = int(1000 / hz)
            
            frames[0].save(
                filepath,
                save_all=True,
                append_images=frames[1:] if len(frames) > 1 else [],
                duration=frame_duration_ms,  # ms per frame (based on hz)
                loop=0  # Infinite loop
            )
            print(f"\n{'='*60}")
            print(f"Saved GIF: {filepath}")
            print(f"Total frames: {len(frames)}")
            print(f"{'='*60}\n")
        else:
            print("Error: No frames captured!")
            raise ValueError("No frames captured - all movements were skipped or failed")
    
    def close(self):
        """Close the environment."""
        self.jacobian_calculator.close()


def generate(
    robot: str = "IIWA",
    env: str = "EmptySpace",
    cue: str = "beckoning",
    pose_index: Optional[int] = None,
    controller: str = "IK_POSE",
    jsonl_path: str = "data/poses/closest_poses_results.jsonl",
    config_path: str = "data/seed/motion_config.json",
    proximal_degree_scale: float = 0.25,
    camera_distance: float = 1.8,
    hz: int = 4,
    enable_self_collision_check: bool = False,
):
    """
    Main function to generate robot motions.
    
    Args:
        robot: Robot name
        env: Environment name
        cue: Name of the cue to execute (e.g., 'waving')
        pose_index: Optional pose_id to use (if None, randomly selects)
        controller: Controller name
        jsonl_path: Path to pose database JSONL file
        config_path: Path to JSON file with cue configurations
        proximal_degree_scale: Scale factor for degrees when using proximal joints (default: 0.25 = 1/4)
        camera_distance: Multiplier for camera FOV to zoom out (default: 1.8 = 80% wider view)
        hz: Frame rate for GIF generation in frames per second (default: 4)
        enable_self_collision_check: Enable self-collision detection and angle reduction (default: False)
    """
    
    generator = MotionGenerator(
        robot_name=robot,
        env_name=env,
        controller_name=controller,
        jsonl_path=jsonl_path,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_distance=camera_distance,
    )
    
    try:
        generator.execute_cue(
            cue=cue,
            pose_index=pose_index,
            config_path=config_path,
            proximal_degree_scale=proximal_degree_scale,
            hz=hz,
            enable_self_collision_check=enable_self_collision_check,
        )
    finally:
        generator.close()
        
    return True


if __name__ == "__main__":
    fire.Fire(generate)
