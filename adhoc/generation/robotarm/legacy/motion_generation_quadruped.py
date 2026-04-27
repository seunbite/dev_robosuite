"""
Generate quadruped (dog) robot motions based on cue definitions.

This script (quadruped version):
1. Loads pose definitions from dog_pose_config and finds matching poses from JSONL
2. Directly controls 4 legs' joint angles (no IK needed)
3. Selects joints based on leg/axis/preference
4. Executes movements and generates GIF

Key differences from arm version:
- 4 independent legs instead of 1 end-effector
- Body height/tilt control affects multiple legs
- Direct joint control (no IK solver needed)

Supported robots:
- Go2, Unitree (generic quadrupeds)
- SpotWithArm, SpotWithArmFloating (Boston Dynamics Spot)

Usage:
    # Go2 robot
    python adhoc/generation/robotarm/motion_generation_quadruped.py \
        --robot Go2 --cue sit_down
    
    # Spot robot (floating base, easier to test)
    python adhoc/generation/robotarm/motion_generation_quadruped.py \
        --robot SpotWithArmFloating --cue paw_shake --camera-distance 2.2
    
    # Spot robot (full quadruped)
    python adhoc/generation/robotarm/motion_generation_quadruped.py \
        --robot SpotWithArm --cue play_bow --camera-distance 2.2
    
    # Custom pose database
    python adhoc/generation/robotarm/motion_generation_quadruped.py \
        --robot SpotWithArm --cue body_bounce \
        --jsonl-path data/poses/spot/closest_SpotWithArm_poses.jsonl

Note:
    For Spot robots, pose databases are auto-detected from:
    - data/poses/spot/closest_{robot}_poses.jsonl (preferred)
    - data/poses/spot/all_{robot}_poses.jsonl (fallback)
    
    If not found, run: python adhoc/spot/export_spot_poses.py --robot SpotWithArm
"""

import fire
import os
import json
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from PIL import Image
from datetime import datetime

import robosuite as suite

# Import pose configuration
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config.dog_pose_config import (
    dog_pose_set, pose_set,
    body_heights, body_tilts, leg_states, legs,
    body_height_joint_offset, body_tilt_joint_offset,
    movement_axes, joint_preference_map,
)


class QuadrupedMotionGenerator:
    """Generate quadruped robot motions based on cue definitions."""
    
    def __init__(
        self,
        robot_name: str = "Go2",
        env_name: str = "EmptySpace",
        controller_name: str = "JOINT_POSITION",
        jsonl_path: Optional[str] = None,
        has_renderer: bool = False,
        has_offscreen_renderer: bool = True,
        control_freq: int = 20,
        output_dir: str = "data/motions_quadruped",
        capture_image_width: int = 512,
        capture_image_height: int = 512,
        camera_distance: float = 2.5,
        hz: int = 4,
    ):
        """
        Initialize the quadruped motion generator.
        
        Args:
            robot_name: Name of the quadruped robot (e.g., "Go2", "Unitree", "SpotWithArm")
            env_name: Name of the environment
            controller_name: Name of the controller (usually JOINT_POSITION for quadrupeds)
            jsonl_path: Path to JSONL file with pose data (auto-detected if None)
            has_renderer: Whether to show on-screen rendering
            has_offscreen_renderer: Whether to enable offscreen rendering
            control_freq: Control frequency
            output_dir: Output directory for GIFs
            capture_image_width: Image width for capture
            capture_image_height: Image height for capture
            camera_distance: Multiplier for camera FOV to zoom out
            hz: Frame rate for GIF generation (frames per second)
        """
        self.robot_name = robot_name
        self.env_name = env_name
        self.controller_name = controller_name
        
        # Auto-detect jsonl_path if not provided
        if jsonl_path is None:
            jsonl_path = self._auto_detect_jsonl_path(robot_name)
        
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
        
        # Create environment
        self._create_environment(has_renderer, has_offscreen_renderer)
        
        # Get initial joint positions
        self.initial_joint_pos = self.robot._joint_positions.copy()
        
        # Build leg-to-joint mapping for this specific robot
        self.leg_joint_map = self._build_leg_joint_map()
        
        # Adjust camera FOV
        self._adjust_camera()
        
        print(f"Quadruped motion generator initialized for {robot_name}")
    
    def _auto_detect_jsonl_path(self, robot_name: str) -> str:
        """
        Auto-detect JSONL path based on robot name.
        
        Args:
            robot_name: Name of the robot
        
        Returns:
            Path to JSONL file
        """
        # Check for Spot-specific paths
        if "Spot" in robot_name:
            # Try closest poses first
            closest_poses_path = f"data/poses/spot/closest_{robot_name}_poses.jsonl"
            if os.path.exists(closest_poses_path):
                print(f"Using pre-queried poses: {closest_poses_path}")
                return closest_poses_path
            
            # Fallback to all poses
            all_poses_path = f"data/poses/spot/all_{robot_name}_poses.jsonl"
            if os.path.exists(all_poses_path):
                print(f"Using all poses: {all_poses_path}")
                return all_poses_path
            
            # If neither exists, show error
            print(f"Warning: Pose database not found for {robot_name}")
            print(f"Please run: python adhoc/spot/export_spot_poses.py --robot {robot_name}")
            return f"data/poses/spot/all_{robot_name}_poses.jsonl"
        
        # Default path for other quadrupeds
        default_path = f"data/poses/quadruped_{robot_name.lower()}_poses.jsonl"
        if os.path.exists(default_path):
            print(f"Using pose database: {default_path}")
            return default_path
        
        # Generic fallback
        generic_path = "data/poses/quadruped_poses_results.jsonl"
        if os.path.exists(generic_path):
            print(f"Using generic pose database: {generic_path}")
            return generic_path
        
        print(f"Warning: No pose database found, using default path: {generic_path}")
        return generic_path
    
    def _create_environment(self, has_renderer: bool, has_offscreen_renderer: bool):
        """Create the robosuite environment."""
        # Quadrupeds typically use JOINT_POSITION controller
        controller_config = {
            "type": "JOINT_POSITION",
            "interpolation": "linear",
            "ramp_ratio": 0.6,
        }
        
        self.env = suite.make(
            env_name=self.env_name,
            robots=self.robot_name,
            controller_configs=controller_config,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            control_freq=self.control_freq,
            render_camera="frontview",
        )
        
        self.env.reset()
        self.robot = self.env.robots[0]
        
        print(f"Environment created: {self.env_name}")
        print(f"Robot: {self.robot_name}")
        print(f"Total joints: {len(self.robot.joint_names)}")
    
    def _build_leg_joint_map(self) -> Dict[str, Dict[str, int]]:
        """
        Build mapping from leg identifiers to joint indices.
        
        Returns:
            Dict mapping leg names (FL, FR, HL, HR) to joint type (hip, shoulder, knee) to joint index
        """
        leg_joint_map = {}
        joint_names = self.robot.joint_names
        
        print("\nBuilding leg-to-joint mapping:")
        print(f"Available joints: {joint_names}")
        
        # Common quadruped joint naming patterns
        leg_prefixes = {
            'FL': ['fl', 'front_left', 'lf', 'fl0', 'fl1', 'fl2'],
            'FR': ['fr', 'front_right', 'rf', 'fr0', 'fr1', 'fr2'],
            'HL': ['hl', 'hind_left', 'rear_left', 'lh', 'hl0', 'hl1', 'hl2'],
            'HR': ['hr', 'hind_right', 'rear_right', 'rh', 'hr0', 'hr1', 'hr2'],
        }
        
        joint_type_keywords = {
            'hip': ['hip', 'abduction', 'abd', 'roll', 'hx', '0'],
            'shoulder': ['shoulder', 'thigh', 'flexion', 'gripper_orientation', 'hy', '1'],
            'knee': ['knee', 'calf', 'shank', 'kn', '2'],
        }
        
        for leg_id in legs:
            leg_joint_map[leg_id] = {}
            
            for joint_type, keywords in joint_type_keywords.items():
                found_idx = None
                best_match_score = 0
                
                # Try to find matching joint with scoring
                for idx, joint_name in enumerate(joint_names):
                    joint_name_lower = joint_name.lower()
                    
                    # Check if joint name matches leg prefix
                    leg_match_score = 0
                    for prefix in leg_prefixes[leg_id]:
                        if joint_name_lower.startswith(prefix):
                            leg_match_score = 2  # Strong match (starts with prefix)
                            break
                        elif prefix in joint_name_lower:
                            leg_match_score = 1  # Weak match (contains prefix)
                    
                    if leg_match_score == 0:
                        continue
                    
                    # Check if joint name matches joint type
                    type_match_score = 0
                    for keyword in keywords:
                        if keyword in joint_name_lower:
                            type_match_score = 1
                            break
                    
                    # Calculate total match score
                    total_score = leg_match_score + type_match_score
                    
                    if total_score > best_match_score:
                        best_match_score = total_score
                        found_idx = idx
                
                if found_idx is not None:
                    leg_joint_map[leg_id][joint_type] = found_idx
                    print(f"  {leg_id}_{joint_type} -> joint[{found_idx}]: {joint_names[found_idx]}")
                else:
                    print(f"  Warning: Could not find {joint_type} joint for {leg_id}")
        
        return leg_joint_map
    
    def _adjust_camera(self):
        """Adjust camera FOV to zoom out."""
        camera_name = "frontview"
        try:
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            current_fov = self.env.sim.model.cam_fovy[cam_id]
            new_fov = current_fov * self.camera_distance
            new_fov = max(20.0, min(120.0, new_fov))
            self.env.sim.model.cam_fovy[cam_id] = new_fov
            print(f"Camera FOV adjusted: {current_fov:.1f}° -> {new_fov:.1f}° (zoom factor: {self.camera_distance})")
        except Exception as e:
            print(f"Warning: Could not adjust camera FOV: {e}")
    
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
            # Pose name as string
            pose_name = pose_def
            if pose_name not in dog_pose_set:
                print(f"Error: Pose '{pose_name}' not found in dog_pose_set")
                return []
            pose_def = dog_pose_set[pose_name]
        elif isinstance(pose_def, dict):
            # Pose definition as dict
            pass
        else:
            print(f"Error: Invalid pose format: {type(pose_def)}")
            return []
        
        # Extract pose features
        body_height = pose_def.get('body_height')
        body_tilt = pose_def.get('body_tilt')
        leg_FL = pose_def.get('leg_FL')
        leg_FR = pose_def.get('leg_FR')
        leg_HL = pose_def.get('leg_HL')
        leg_HR = pose_def.get('leg_HR')
        
        # Load pose database
        pose_database = self._load_pose_database(self.jsonl_path)
        
        if not pose_database:
            print(f"Warning: Pose database is empty or not found: {self.jsonl_path}")
            return []
        
        # Filter poses matching all criteria
        matching_poses = []
        for pose_data in pose_database:
            if pose_data.get('robot') != robot_name:
                continue
            
            pose_features = pose_data.get('pose_features', {})
            
            # Check if all features match
            match = True
            if body_height and pose_features.get('body_height') != body_height:
                match = False
            if body_tilt and pose_features.get('body_tilt') != body_tilt:
                match = False
            if leg_FL and pose_features.get('leg_FL') != leg_FL:
                match = False
            if leg_FR and pose_features.get('leg_FR') != leg_FR:
                match = False
            if leg_HL and pose_features.get('leg_HL') != leg_HL:
                match = False
            if leg_HR and pose_features.get('leg_HR') != leg_HR:
                match = False
            
            if match:
                matching_poses.append(pose_data)
        
        print(f"  Found {len(matching_poses)} matching poses for {pose_def}")
        return matching_poses
    
    def _select_joint_for_movement(
        self,
        axis: str,  # e.g., 'body_z', 'leg_FL', 'body_pitch'
        joint_preference: str,  # 'proximal', 'middle', 'distal'
    ) -> Tuple[int, str, str]:
        """
        Select a joint for movement based on axis and preference.
        
        Args:
            axis: Movement axis (from movement_axes)
            joint_preference: Joint preference (proximal/middle/distal)
        
        Returns:
            Tuple of (joint_index, joint_name, leg_id)
        """
        if axis not in movement_axes:
            raise ValueError(f"Unknown axis: {axis}. Available axes: {list(movement_axes.keys())}")
        
        axis_info = movement_axes[axis]
        affects = axis_info['affects']
        primary_joint_type = axis_info['primary_joint']
        
        # Map joint preference to actual joint type
        if joint_preference in joint_preference_map:
            joint_type = joint_preference_map[joint_preference]
        else:
            joint_type = primary_joint_type
        
        # Determine which leg(s) to use
        if affects == 'all_legs':
            # Body height movement - pick one leg (all will move together)
            selected_leg = 'FL'  # Use front-left as representative
        elif affects == 'front_vs_back':
            # Body gripper_orientation - pick front or back leg
            selected_leg = 'FL' if joint_preference == 'proximal' else 'HL'
        elif affects == 'left_vs_right':
            # Body roll - pick left or right leg
            selected_leg = 'FL' if joint_preference == 'proximal' else 'FR'
        elif affects in legs:
            # Individual leg movement
            selected_leg = affects
        else:
            raise ValueError(f"Unknown affects value: {affects}")
        
        # Get joint index from leg_joint_map
        if selected_leg not in self.leg_joint_map:
            raise ValueError(f"Leg {selected_leg} not found in leg_joint_map")
        
        if joint_type not in self.leg_joint_map[selected_leg]:
            # Fallback to available joint
            available_joints = list(self.leg_joint_map[selected_leg].keys())
            if not available_joints:
                raise ValueError(f"No joints found for leg {selected_leg}")
            joint_type = available_joints[0]
            print(f"  Warning: {joint_type} not found for {selected_leg}, using {available_joints[0]}")
        
        joint_idx = self.leg_joint_map[selected_leg][joint_type]
        joint_name = self.robot.joint_names[joint_idx]
        
        print(f"\nSelected joint for {axis} movement:")
        print(f"  Leg: {selected_leg}, Joint type: {joint_type}")
        print(f"  Joint index: {joint_idx}, Joint name: {joint_name}")
        
        return joint_idx, joint_name, selected_leg
    
    def _pose_data_to_joint_positions(self, pose_data: Dict) -> np.ndarray:
        """
        Convert pose data to joint positions array.
        
        Args:
            pose_data: Dictionary containing pose information with joint_angles_rad
        
        Returns:
            Joint positions array in radians
        """
        joint_angles_rad = pose_data.get("joint_angles_rad", [])
        
        # Start with initial joint positions
        joint_pos = self.initial_joint_pos.copy()
        
        # Update with pose joint angles
        for i, angle in enumerate(joint_angles_rad):
            if i < len(joint_pos):
                joint_pos[i] = angle
        
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
        robot_geoms = set(self.robot.robot_model.contact_geoms)
        
        for i in range(self.env.sim.data.ncon):
            contact = self.env.sim.data.contact[i]
            geom1_name = self.env.sim.model.geom_id2name(contact.geom1)
            geom2_name = self.env.sim.model.geom_id2name(contact.geom2)
            
            if geom1_name in robot_geoms and geom2_name in robot_geoms:
                return True
        
        return False
    
    def _load_cue_config(self, cue: str, config_path: str = "data/results/motion_configs/manipulator/motion_config_quadruped.json") -> Dict:
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
        
        for config in configs:
            if config.get('cue') == cue:
                return config
        
        raise ValueError(f"Cue '{cue}' not found in configuration file: {config_path}")
    
    def execute_cue(
        self,
        cue: str,
        pose_index: Optional[int] = None,
        config_path: str = "data/results/motion_configs/manipulator/motion_config_quadruped.json",
        hz: int = 4,
        filename_suffix: Optional[str] = None,
        enable_self_collision_check: bool = False,
    ):
        """
        Execute a cue (e.g., 'sit_down', 'paw_shake').
        
        Args:
            cue: Name of the cue
            pose_index: Optional pose_id to use (if None, randomly selects)
            config_path: Path to JSON file with cue configurations
            hz: Frame rate for GIF
            filename_suffix: Optional suffix for output filename
            enable_self_collision_check: Enable self-collision detection
        """
        print(f"\n{'='*60}")
        print(f"Executing cue: {cue}")
        print(f"{'='*60}\n")
        
        # Load cue configuration
        cue_config_data = self._load_cue_config(cue, config_path)
        movements = cue_config_data.get('movements', [])
        
        if not movements:
            raise ValueError(f"No movements found in cue '{cue}' configuration")
        
        frames = []
        current_pose_name = None
        current_pose = None
        pose_id = None
        
        for movement_item in movements:
            movement_type = movement_item.get('type')
            parameters = movement_item.get('parameters', {})
            
            if movement_type == 'pose':
                # Set pose
                pose_param = parameters.get('pose')
                if pose_param is None:
                    raise ValueError("'pose' parameter is required for 'pose' type movement")
                
                # Handle both string and dict formats
                if isinstance(pose_param, str):
                    pose_display_name = pose_param
                elif isinstance(pose_param, dict):
                    pose_display_name = f"{pose_param.get('body_height', '?')}_{pose_param.get('body_tilt', '?')}"
                else:
                    raise ValueError(f"Invalid pose format: {type(pose_param)}")
                
                print(f"\n--- Setting Pose: {pose_display_name} ---")
                matching_poses = self._find_matching_poses(pose_param)
                
                if not matching_poses:
                    # Generate pose on-the-fly if not in database
                    print(f"  Warning: No matching poses in database, will use default position")
                    selected_pose = {'joint_angles_rad': self.initial_joint_pos.tolist()}
                else:
                    selected_pose = random.choice(matching_poses)
                    print(f"  Selected pose: {selected_pose.get('pose_id', 'N/A')}")
                
                speed = parameters.get('speed', 1.0)
                hold_time = parameters.get('hold_time', 1.0)
                
                if current_pose is None:
                    # First pose
                    print("Setting first pose...")
                    target_joint_pos = self._pose_data_to_joint_positions(selected_pose)
                    self._set_joint_positions(target_joint_pos)
                    current_pose_name = pose_param
                    current_pose = selected_pose
                    
                    # Hold
                    num_frames = int(hold_time * hz)
                    for _ in range(num_frames):
                        image = self._capture_image()
                        frames.append(Image.fromarray(image))
                    print(f"Captured {num_frames} frames (hold_time: {hold_time}s)")
                else:
                    # Transition
                    print(f"Transitioning to {pose_display_name} (speed: {speed})...")
                    
                    start_joint_pos = self._get_joint_positions()
                    end_joint_pos = self._pose_data_to_joint_positions(selected_pose)
                    
                    duration = 1.0 / speed
                    num_transition_frames = int(duration * hz)
                    
                    for frame_idx in range(num_transition_frames):
                        t = (frame_idx + 1) / num_transition_frames
                        interpolated_joint_pos = start_joint_pos * (1 - t) + end_joint_pos * t
                        self._set_joint_positions(interpolated_joint_pos)
                        
                        image = self._capture_image()
                        frames.append(Image.fromarray(image))
                    
                    print(f"  Captured {num_transition_frames} transition frames")
                    
                    # Set final pose
                    self._set_joint_positions(end_joint_pos)
                    current_pose_name = pose_param
                    current_pose = selected_pose
                    
                    # Hold
                    num_hold_frames = int(hold_time * hz)
                    for _ in range(num_hold_frames):
                        image = self._capture_image()
                        frames.append(Image.fromarray(image))
                    print(f"  Captured {num_hold_frames} hold frames")
            
            elif movement_type == 'movement':
                # Execute movement
                if current_pose_name is None:
                    raise ValueError("No pose set before movement")
                
                repetition = parameters.get('repetition', 1)
                axis = parameters.get('axis')
                joint_preference = parameters.get('joint', 'middle')
                directions = parameters.get('directions', [])
                
                if axis is None:
                    raise ValueError("'axis' parameter is required")
                if not directions:
                    raise ValueError("'directions' parameter is required")
                
                print(f"\n--- Movement ---")
                print(f"Axis: {axis}, Joint preference: {joint_preference}")
                print(f"Repetition: {repetition}")
                
                # Select joint
                joint_idx, joint_name, leg_id = self._select_joint_for_movement(axis, joint_preference)
                
                current_joint_pos = self._get_joint_positions()
                
                # Execute repetitions
                for rep in range(repetition):
                    print(f"\nRepetition {rep + 1}/{repetition}")
                    
                    for direction_config in directions:
                        degrees = direction_config.get('degrees', 0)
                        speed = direction_config.get('speed', 1.0)
                        hold_time = direction_config.get('hold_time', 0.5)
                        
                        print(f"  Movement: {degrees}°, speed: {speed}")
                        
                        # Calculate target
                        start_angle_rad = current_joint_pos[joint_idx]
                        target_angle_offset_rad = np.deg2rad(degrees)
                        target_angle_rad = start_angle_rad + target_angle_offset_rad
                        
                        # Movement duration
                        duration = 1.0 / speed
                        num_movement_frames = int(duration * hz)
                        
                        # Interpolate
                        for frame_idx in range(num_movement_frames):
                            t = (frame_idx + 1) / num_movement_frames
                            interpolated_angle_rad = start_angle_rad + t * target_angle_offset_rad
                            
                            new_joint_pos = current_joint_pos.copy()
                            new_joint_pos[joint_idx] = interpolated_angle_rad
                            self._set_joint_positions(new_joint_pos)
                            
                            image = self._capture_image()
                            frames.append(Image.fromarray(image))
                        
                        print(f"  Captured {num_movement_frames} movement frames")
                        
                        # Update current position
                        current_joint_pos[joint_idx] = target_angle_rad
                        
                        # Hold
                        if hold_time > 0:
                            num_hold_frames = int(hold_time * hz)
                            for _ in range(num_hold_frames):
                                image = self._capture_image()
                                frames.append(Image.fromarray(image))
                            print(f"  Captured {num_hold_frames} hold frames")
            else:
                raise ValueError(f"Unknown movement type: {movement_type}")
        
        # Save GIF
        if len(frames) > 0:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{now}_{self.robot_name}_{cue}"
            
            if filename_suffix:
                output_filename = f"{base_filename}_{filename_suffix}.gif"
            else:
                output_filename = f"{base_filename}.gif"
            
            filepath = os.path.join(self.output_dir, output_filename)
            frame_duration_ms = int(1000 / hz)
            
            frames[0].save(
                filepath,
                save_all=True,
                append_images=frames[1:] if len(frames) > 1 else [],
                duration=frame_duration_ms,
                loop=0
            )
            print(f"\n{'='*60}")
            print(f"Saved GIF: {filepath}")
            print(f"Total frames: {len(frames)}")
            print(f"{'='*60}\n")
        else:
            print("Error: No frames captured!")
            raise ValueError("No frames captured")
    
    def close(self):
        """Close the environment."""
        self.env.close()


def generate(
    robot: str = "Go2",
    env: str = "EmptySpace",
    cue: str = "sit_down",
    pose_index: Optional[int] = None,
    controller: str = "JOINT_POSITION",
    jsonl_path: Optional[str] = None,
    config_path: str = "data/results/motion_configs/manipulator/motion_config_quadruped.json",
    camera_distance: float = 2.5,
    hz: int = 4,
    enable_self_collision_check: bool = False,
):
    """
    Main function to generate quadruped robot motions.
    
    Args:
        robot: Robot name (e.g., "Go2", "Unitree", "SpotWithArm", "SpotWithArmFloating")
        env: Environment name
        cue: Name of the cue to execute
        pose_index: Optional pose_id to use
        controller: Controller name
        jsonl_path: Path to pose database JSONL file (auto-detected if None)
        config_path: Path to JSON file with cue configurations
        camera_distance: Multiplier for camera FOV to zoom out (2.2-2.5 recommended for quadrupeds)
        hz: Frame rate for GIF generation
        enable_self_collision_check: Enable self-collision detection
    
    Examples:
        # Go2 robot
        python adhoc/generation/robotarm/motion_generation_quadruped.py \
            --robot Go2 --cue sit_down
        
        # Spot robot (auto-detects pose database)
        python adhoc/generation/robotarm/motion_generation_quadruped.py \
            --robot SpotWithArm --cue paw_shake --camera-distance 2.2
        
        # Custom pose database
        python adhoc/generation/robotarm/motion_generation_quadruped.py \
            --robot SpotWithArm --cue play_bow \
            --jsonl-path data/poses/spot/closest_SpotWithArm_poses.jsonl
    """
    
    generator = QuadrupedMotionGenerator(
        robot_name=robot,
        env_name=env,
        controller_name=controller,
        jsonl_path=jsonl_path,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_distance=camera_distance,
        hz=hz,
    )
    
    try:
        generator.execute_cue(
            cue=cue,
            pose_index=pose_index,
            config_path=config_path,
            hz=hz,
            enable_self_collision_check=enable_self_collision_check,
        )
    finally:
        generator.close()
    
    return True


if __name__ == "__main__":
    fire.Fire(generate)
