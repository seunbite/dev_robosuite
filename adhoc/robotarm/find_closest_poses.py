"""
Find closest poses based on end effector orientation and root-to-EE distance.

Given target roll, pitch, yaw values, this script:
1. Generates all pose combinations (like export_ee_orientation.py)
2. Calculates orientation for each pose
3. Finds top 30 poses with closest orientation (absolute difference)
4. Sorts them by root-to-end-effector distance
"""

import fire
import os
import json
import time
import numpy as np
import math
from typing import Optional, List, Dict
from itertools import product
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.utils import transform_utils as T

# Import the same constants from stack_preset
FIXED_JOINT_INDICES = {
    'GR1': "0-2, 20-31"
}

SPECIFIED_JOINT_ANGLES = {
    'GR1': {
        "robot0_head_yaw": [-30, 0, 30],
        "robot0_head_roll": [-30, 0, 30],
        "robot0_head_pitch": [-30, 0, 30],
    }
}


class ClosestPoseFinder:
    """Find closest poses based on orientation and distance criteria."""
    
    def __init__(
        self,
        robot_name: str = "IIWA",
        env_name: str = "EmptySpace",
        controller_name: str = "OSC_POSE",
    ):
        """
        Initialize the finder.
        
        Args:
            robot_name: Name of the robot
            env_name: Name of the environment
            controller_name: Name of the controller
        """
        self.robot_name = robot_name
        self.env_name = env_name
        self.controller_name = controller_name
        
        print(f"Initializing robot: {robot_name}")
        
        # Setup environment
        options = {
            "env_name": env_name,
            "robots": robot_name,
            "has_renderer": False,
            "has_offscreen_renderer": False,  # Don't need rendering
            "ignore_done": True,
            "use_camera_obs": False,
            "control_freq": 20,
        }
        
        # Load controller config
        arm_controller_config = suite.load_part_controller_config(default_controller=controller_name)
        options["controller_configs"] = refactor_composite_controller_config(
            arm_controller_config, robot_name, ["right", "left"]
        )
        
        # Create environment
        self.env = suite.make(**options, horizon=1000)
        self.env.reset()
        
        # Get robot
        self.robot = self.env.robots[0]
        
        # Get initial joint positions
        self.initial_joint_pos = self.robot._joint_positions.copy()
        self.num_joints = len(self.initial_joint_pos)
        
        # Parse fixed joint indices
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
        
        # Active joints
        base_active_joint_indices = list(range(self.num_joints - 1))  # Exclude last joint (gripper)
        self.active_joint_indices = [idx for idx in base_active_joint_indices if idx not in self.fixed_joint_indices]
        
        print(f"Total joints: {self.num_joints}")
        print(f"Active joints: {len(self.active_joint_indices)}")
        if self.fixed_joint_indices:
            print(f"Fixed joints: {len(self.fixed_joint_indices)}")
    
    def _get_root_position(self):
        """
        Get root body position in world coordinates.
        
        Returns:
            np.ndarray: [x, y, z] position of root body
        """
        try:
            root_body = self.robot.robot_model.root_body
            root_pos = self.env.sim.data.get_body_xpos(root_body)
            return root_pos
        except Exception as e:
            print(f"Warning: Could not get root position: {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def _set_joint_positions(self, joint_positions_rad):
        """Set joint positions and update simulation."""
        self.robot.set_robot_joint_positions(joint_positions_rad)
        self.env.sim.forward()
    
    def _get_ee_position(self, arm="right"):
        """Get end effector position."""
        try:
            pos_dict = self.robot._hand_pos
            if arm in pos_dict:
                return pos_dict[arm].copy()
            else:
                available_arms = list(pos_dict.keys())
                if available_arms:
                    return pos_dict[available_arms[0]].copy()
                else:
                    raise ValueError("No arms available")
        except Exception as e:
            print(f"Warning: Could not get EE position: {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def _get_ee_orientation_rpy(self, arm="right"):
        """
        Get end effector orientation as roll, pitch, yaw in radians.
        
        Args:
            arm: Which arm to get orientation for ("right" or "left")
            
        Returns:
            np.ndarray: [roll, pitch, yaw] in radians
        """
        try:
            # Get rotation matrix
            orn_dict = self.robot._hand_orn
            if arm in orn_dict:
                rot_mat = orn_dict[arm]
                # Convert rotation matrix to euler angles (roll, pitch, yaw)
                rpy = T.mat2euler(rot_mat)
                return rpy
            else:
                # If arm not found, try with first available arm
                available_arms = list(orn_dict.keys())
                if available_arms:
                    arm = available_arms[0]
                    rot_mat = orn_dict[arm]
                    rpy = T.mat2euler(rot_mat)
                    return rpy
                else:
                    raise ValueError("No arms available")
        except Exception as e:
            print(f"Warning: Could not get EE orientation: {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def find_closest_poses(
        self,
        roll_deg: Optional[float] = None,
        pitch_deg: Optional[float] = 0,
        yaw_deg: Optional[float] = 0,
        angle_step_deg: float = 90.0,
        angle_min_deg: float = -90.0,
        angle_max_deg: float = 90.0,
        top_k: int = 30,
        output_file: Optional[str] = None,
        arm: str = "right",
        tile_size: int = 256,
        border_width: int = 2,
        stack_jsonl_path: Optional[str] = 'data/poses/closest_poses_results.jsonl',
        height: Optional[str] = None,
        height_low_bar: float = 33.0,
        height_high_bar: float = 33.0,
    ):
        """
        Generate poses and find closest ones based on orientation and distance.
        
        Args:
            roll_deg: Target roll angle in degrees (None to ignore)
            pitch_deg: Target pitch angle in degrees (None to ignore)
            yaw_deg: Target yaw angle in degrees (None to ignore)
            angle_step_deg: Step size in degrees for pose generation
            angle_min_deg: Minimum angle in degrees
            angle_max_deg: Maximum angle in degrees
            top_k: Number of top poses to return (default: 30)
            output_file: Path to save results JSON (default: print to stdout)
            arm: Which arm to use ("right" or "left")
            height: Filter by EE height relative to root ("high", "medium", "low", or None to disable)
            height_low_bar: Percentage of z_diff range for low threshold (default: 33.0)
            height_high_bar: Percentage of z_diff range for high threshold (default: 33.0)
        """
        print("\n" + "="*60)
        print("FINDING CLOSEST POSES")
        print("="*60)
        print(f"Target orientation:")
        print(f"  Roll:  {roll_deg}°" if roll_deg is not None else "  Roll:  None (ignored)")
        print(f"  Pitch: {pitch_deg}°" if pitch_deg is not None else "  Pitch: None (ignored)")
        print(f"  Yaw:   {yaw_deg}°" if yaw_deg is not None else "  Yaw:   None (ignored)")
        print(f"Top K: {top_k}")
        print(f"Arm: {arm}")
        print(f"Angle step: {angle_step_deg}°")
        print(f"Angle range: {angle_min_deg}° to {angle_max_deg}°")
        print("="*60 + "\n")
        
        # Convert target angles to radians
        target_roll = np.deg2rad(roll_deg) if roll_deg is not None else None
        target_pitch = np.deg2rad(pitch_deg) if pitch_deg is not None else None
        target_yaw = np.deg2rad(yaw_deg) if yaw_deg is not None else None
        
        # Check if at least one target is provided
        if target_roll is None and target_pitch is None and target_yaw is None:
            print("Warning: All target angles are None. No filtering will be applied.")
            print("Setting orientation_diff to 0.0 for all poses.")
        
        # Prepare angle arrays (same as export_ee_orientation.py)
        specified_angles = SPECIFIED_JOINT_ANGLES.get(self.robot_name, {})
        angle_min = np.deg2rad(angle_min_deg)
        angle_max = np.deg2rad(angle_max_deg)
        angle_step = np.deg2rad(angle_step_deg)
        default_angles = np.arange(angle_min, angle_max + angle_step/2, angle_step)
        
        # Create angle arrays for each active joint
        joint_angle_arrays = []
        joint_names = []
        for active_joint_idx in self.active_joint_indices:
            # Try to get joint name
            try:
                joint_name = f"joint_{active_joint_idx}"
                if hasattr(self.robot, 'robot_model') and hasattr(self.robot.robot_model, 'joints'):
                    robot_joints = list(self.robot.robot_model.joints)
                    if active_joint_idx < len(robot_joints):
                        joint_name = robot_joints[active_joint_idx]
            except:
                joint_name = f"joint_{active_joint_idx}"
            
            joint_names.append(joint_name)
            
            # Check if this joint has specified angles
            if joint_name in specified_angles:
                angles = np.deg2rad(np.array(specified_angles[joint_name]))
                joint_angle_arrays.append(angles)
            else:
                joint_angle_arrays.append(default_angles)
        
        # Calculate total combinations
        num_angles_per_joint = [len(angles) for angles in joint_angle_arrays]
        total_combinations = 1
        for num in num_angles_per_joint:
            total_combinations *= num
        
        print(f"Generating {total_combinations:,} pose combinations...")
        
        # Generate combinations
        selected_combinations = list(product(*[range(num) for num in num_angles_per_joint]))
        
        # Store all poses with their orientation differences
        scored_poses = []
        
        for combo_idx, angle_indices in tqdm(enumerate(selected_combinations), total=len(selected_combinations)):
            # Create joint position array
            joint_pos = self.initial_joint_pos.copy()
            
            # Set positions for active joints
            angle_values = []
            for i, active_joint_idx in enumerate(self.active_joint_indices):
                angle_idx = angle_indices[i]
                angle_value = joint_angle_arrays[i][angle_idx]
                joint_pos[active_joint_idx] = angle_value
                angle_values.append(angle_value)
            
            # Set joint positions
            self._set_joint_positions(joint_pos)
            
            # Get end effector orientation
            rpy = self._get_ee_orientation_rpy(arm=arm)
            roll_rad, pitch_rad, yaw_rad = rpy[0], rpy[1], rpy[2]
            
            # Calculate absolute difference for each angle (if target provided)
            orientation_diff = 0.0
            diff_components = {}
            num_targets = 0  # Count how many targets are provided
            
            if target_roll is not None:
                diff = abs(roll_rad - target_roll)
                # Handle angle wrapping (difference should be <= pi)
                diff = min(diff, 2 * np.pi - diff)
                orientation_diff += diff
                diff_components["roll_diff_rad"] = diff
                diff_components["roll_diff_deg"] = np.rad2deg(diff)
                num_targets += 1
            else:
                diff_components["roll_diff_rad"] = None
                diff_components["roll_diff_deg"] = None
            
            if target_pitch is not None:
                diff = abs(pitch_rad - target_pitch)
                diff = min(diff, 2 * np.pi - diff)
                orientation_diff += diff
                diff_components["pitch_diff_rad"] = diff
                diff_components["pitch_diff_deg"] = np.rad2deg(diff)
                num_targets += 1
            else:
                diff_components["pitch_diff_rad"] = None
                diff_components["pitch_diff_deg"] = None
            
            if target_yaw is not None:
                diff = abs(yaw_rad - target_yaw)
                diff = min(diff, 2 * np.pi - diff)
                orientation_diff += diff
                diff_components["yaw_diff_rad"] = diff
                diff_components["yaw_diff_deg"] = np.rad2deg(diff)
                num_targets += 1
            else:
                diff_components["yaw_diff_rad"] = None
                diff_components["yaw_diff_deg"] = None
            
            # If no targets provided, set orientation_diff to 0.0 (all poses are equally valid)
            if num_targets == 0:
                orientation_diff = 0.0
            
            # Generate angles string for reference
            angles_str = "_".join([f"j{self.active_joint_indices[j]}{int(np.rad2deg(angle_values[j])):+04d}" 
                                    for j in range(len(angle_indices))])
            
            # Store pose with score
            scored_pose = {
                "pose_id": combo_idx,
                "angles_str": angles_str,
                "joint_angles_deg": [float(np.rad2deg(angle_values[j])) for j in range(len(angle_indices))],
                "joint_angles_rad": [float(angle_values[j]) for j in range(len(angle_indices))],
                "active_joint_indices": self.active_joint_indices,
                "joint_names": joint_names,
                "end_effector": {
                    "orientation": {
                        "roll_rad": float(roll_rad),
                        "pitch_rad": float(pitch_rad),
                        "yaw_rad": float(yaw_rad),
                        "roll_deg": float(np.rad2deg(roll_rad)),
                        "pitch_deg": float(np.rad2deg(pitch_rad)),
                        "yaw_deg": float(np.rad2deg(yaw_rad)),
                    }
                },
                "orientation_diff_rad": orientation_diff,
                "orientation_diff_deg": np.rad2deg(orientation_diff),
                "orientation_diff_components": diff_components,
            }
            scored_poses.append(scored_pose)
            
            # Return to initial pose occasionally to prevent drift
            if (combo_idx + 1) % 500 == 0:
                self._set_joint_positions(self.initial_joint_pos)
        
        # Filter poses with orientation_diff > 60 degrees
        max_orientation_diff_deg = 60.0
        max_orientation_diff_rad = np.deg2rad(max_orientation_diff_deg)
        
        # Debug: print some statistics
        if scored_poses:
            orientation_diffs = [pose["orientation_diff_deg"] for pose in scored_poses]
            print(f"\nOrientation difference statistics:")
            print(f"  Min: {min(orientation_diffs):.2f}°")
            print(f"  Max: {max(orientation_diffs):.2f}°")
            print(f"  Mean: {np.mean(orientation_diffs):.2f}°")
            print(f"  Poses with diff <= {max_orientation_diff_deg}°: {sum(1 for d in orientation_diffs if d <= max_orientation_diff_deg)}")
        
        filtered_poses = [pose for pose in scored_poses if pose["orientation_diff_rad"] <= max_orientation_diff_rad]
        
        print(f"\nFiltered {len(scored_poses) - len(filtered_poses)} poses with orientation_diff > {max_orientation_diff_deg}°")
        print(f"Remaining poses: {len(filtered_poses)}")
        
        # Sort by orientation difference (smallest first)
        filtered_poses.sort(key=lambda x: x["orientation_diff_rad"])
        
        # Get top K poses (from filtered list)
        top_poses = filtered_poses[:top_k]
        print(f"Selected top {len(top_poses)} poses based on orientation similarity")
        
        # Now calculate root-to-EE distances and filter/sort by distance
        print("Calculating root-to-EE distances...")
        
        for pose in tqdm(top_poses, desc="Calculating distances"):
            # Set robot to this pose
            # Reconstruct full joint position array (including fixed joints)
            joint_pos = self.initial_joint_pos.copy()
            for i, active_joint_idx in enumerate(self.active_joint_indices):
                joint_pos[active_joint_idx] = pose["joint_angles_rad"][i]
            self._set_joint_positions(joint_pos)
            
            # Get root and EE positions
            root_pos = self._get_root_position()
            ee_pos = self._get_ee_position(arm=arm)
            
            # Calculate distance
            distance = np.linalg.norm(ee_pos - root_pos)
            
            pose["root_position"] = {
                "x": float(root_pos[0]),
                "y": float(root_pos[1]),
                "z": float(root_pos[2]),
            }
            pose["root_to_ee_distance"] = float(distance)
            pose["ee_position"] = {
                "x": float(ee_pos[0]),
                "y": float(ee_pos[1]),
                "z": float(ee_pos[2]),
            }
        
        # Filter poses where EE is in front of root (x > root_x)
        # This ensures the end-effector is positioned forward relative to the robot base
        front_poses = []
        behind_poses = []
        for pose in top_poses:
            ee_x = pose["ee_position"]["x"]
            root_x = pose["root_position"]["x"]
            if ee_x > root_x:
                front_poses.append(pose)
            else:
                behind_poses.append(pose)
        
        print(f"Filtered poses: {len(front_poses)} in front, {len(behind_poses)} behind root")
        
        # If we have front poses, use them; otherwise use all poses (fallback)
        if front_poses:
            top_poses = front_poses
            print(f"Using {len(top_poses)} poses with EE in front of root")
        else:
            print(f"Warning: No poses with EE in front of root. Using all {len(top_poses)} poses as fallback.")
        
        # Filter by height if specified
        if height is not None:
            # Calculate z_diff for all poses
            z_diffs = []
            for pose in top_poses:
                ee_z = pose["ee_position"]["z"]
                root_z = pose["root_position"]["z"]
                z_diff = ee_z - root_z
                z_diffs.append(z_diff)
                pose["z_diff"] = z_diff  # Store for later use
            
            # Calculate thresholds based on range (length), not percentile (count)
            min_z_diff = min(z_diffs)
            max_z_diff = max(z_diffs)
            z_diff_range = max_z_diff - min_z_diff
            
            # Calculate thresholds
            low_threshold = min_z_diff + z_diff_range * (height_low_bar / 100.0)
            high_threshold = max_z_diff - z_diff_range * (height_high_bar / 100.0)
            
            # Classify poses based on thresholds
            high_poses = []
            medium_poses = []
            low_poses = []
            
            for pose in top_poses:
                z_diff = pose["z_diff"]
                
                if z_diff > high_threshold:
                    high_poses.append(pose)
                elif z_diff < low_threshold:
                    low_poses.append(pose)
                else:
                    medium_poses.append(pose)
            
            print(f"\nHeight filtering (based on z_diff range):")
            print(f"  z_diff range: [{min_z_diff:.3f}, {max_z_diff:.3f}] m (total: {z_diff_range:.3f} m)")
            print(f"  height_low_bar: {height_low_bar}%, height_high_bar: {height_high_bar}%")
            print(f"  Low threshold: {low_threshold:.3f} m (z_diff < {low_threshold:.3f})")
            print(f"  High threshold: {high_threshold:.3f} m (z_diff > {high_threshold:.3f})")
            print(f"  Low poses: {len(low_poses)}")
            print(f"  Medium poses: {len(medium_poses)}")
            print(f"  High poses: {len(high_poses)}")
            
            # Select based on height parameter
            height_lower = height.lower()
            if height_lower == "high":
                top_poses = high_poses
                print(f"Selected {len(top_poses)} HIGH poses")
            elif height_lower == "medium":
                top_poses = medium_poses
                print(f"Selected {len(top_poses)} MEDIUM poses")
            elif height_lower == "low":
                top_poses = low_poses
                print(f"Selected {len(top_poses)} LOW poses")
            else:
                print(f"Warning: Unknown height value '{height}'. Expected 'high', 'medium', or 'low'. Using all poses.")
            
            if not top_poses:
                print(f"Warning: No poses found with height='{height}'. Results will be empty.")
        
        # Sort by root-to-EE distance (smallest first)
        top_poses.sort(key=lambda x: x["root_to_ee_distance"])
        
        # Create tiled image from top poses
        tiled_image_path = self._create_tiled_image(top_poses, output_file, tile_size, border_width, 
                                                    roll_deg, pitch_deg, yaw_deg, height)
        
        # Prepare output
        output_data = {
            "robot_name": self.robot_name,
            "env_name": self.env_name,
            "target_orientation": {
                "roll_deg": roll_deg,
                "pitch_deg": pitch_deg,
                "yaw_deg": yaw_deg,
            },
            "total_poses_searched": len(scored_poses),
            "top_k": len(top_poses),
            "arm": arm,
            "angle_step_deg": angle_step_deg,
            "angle_min_deg": angle_min_deg,
            "angle_max_deg": angle_max_deg,
            "tiled_image": tiled_image_path,
            "poses": top_poses,
        }
        
        # Save to JSONL file if specified
        if stack_jsonl_path:
            self._save_to_jsonl(output_data, stack_jsonl_path, roll_deg, pitch_deg, yaw_deg,
                               angle_step_deg, angle_min_deg, angle_max_deg, top_k, arm)
        
        # Output results
        if output_file:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n{'='*60}")
            print(f"Results saved to: {output_file}")
        else:
            # Print summary only (not individual poses)
            print(f"\n{'='*60}")
            print(f"Summary:")
            print(f"  robot_name:       {self.robot_name}")
            print(f"  env_name:         {self.env_name}")
            print(f"  target_roll:      {roll_deg}°")
            print(f"  target_pitch:     {pitch_deg}°")
            print(f"  target_yaw:       {yaw_deg}°")
            print(f"  top_k:            {len(top_poses)}")
            print(f"  tiled_image:      {tiled_image_path}")
            if top_poses:
                print(f"  avg_distance:     {np.mean([p['root_to_ee_distance'] for p in top_poses]):.4f} m")
                print(f"  min_distance:     {min([p['root_to_ee_distance'] for p in top_poses]):.4f} m")
                print(f"  max_distance:     {max([p['root_to_ee_distance'] for p in top_poses]):.4f} m")
            print(f"{'='*60}\n")
        
        return output_data
    
    def _create_tiled_image(self, top_poses: List[Dict], output_file: Optional[str] = None, 
                           tile_size: int = 256, border_width: int = 2, roll_deg: Optional[float] = None,
                           pitch_deg: Optional[float] = None, yaw_deg: Optional[float] = None,
                           height: Optional[str] = None):
        """
        Create a tiled image from the top poses.
        
        Args:
            top_poses: List of pose dictionaries
            output_file: Base output file path (if None, auto-generate)
            tile_size: Size to resize each tile
            border_width: Width of border between tiles
            roll_deg: Target roll (for filename generation)
            pitch_deg: Target pitch (for filename generation)
            yaw_deg: Target yaw (for filename generation)
            height: Height filter ("high", "medium", "low", or None) for filename generation
            
        Returns:
            str: Path to saved tiled image
        """
        print(f"\n{'='*60}")
        print("CREATING TILED IMAGE")
        print(f"{'='*60}")
        
        # Determine output filename (save to data/logs)
        if output_file:
            # Use output_file as base, change extension to .png
            base_name = os.path.splitext(output_file)[0]
            tiled_output = f"{base_name}_tiled.png"
        else:
            # Auto-generate filename in data/logs
            roll_str = f"r{int(roll_deg)}" if roll_deg is not None else "rNone"
            pitch_str = f"p{int(pitch_deg)}" if pitch_deg is not None else "pNone"
            yaw_str = f"y{int(yaw_deg)}" if yaw_deg is not None else "yNone"
            height_str = f"_h{height}" if height is not None else ""
            tiled_output = f"data/logs/{self.robot_name}_closest_{roll_str}_{pitch_str}_{yaw_str}{height_str}_tiled.png"
        
        print(f"Tiled image will be saved to: {tiled_output}")
        
        # Determine image directory (for finding source images)
        # Note: We still look for source images in data/poses/{robot_name}
        # but save the tiled image to data/logs
        image_dir = f"data/poses/{self.robot_name}"
        
        # Find image files for top poses
        image_files = []
        missing_files = []
        
        for i, pose in enumerate(top_poses, 1):
            # Try to find image file
            # Format: {robot_name}_pose_{pose_id:06d}_{angles_str}.png
            pose_id = pose["pose_id"]
            angles_str = pose["angles_str"]
            
            # Try different possible filenames
            possible_filenames = [
                f"{self.robot_name}_pose_{pose_id:06d}_{angles_str}.png",
                f"{self.robot_name}_pose_{pose_id:06d}.png",  # Fallback without angles
            ]
            
            found = False
            for filename in possible_filenames:
                filepath = os.path.join(image_dir, filename)
                if os.path.exists(filepath):
                    image_files.append((i, filepath, pose))
                    found = True
                    break
            
            if not found:
                missing_files.append((i, pose_id, angles_str))
        
        if not image_files:
            # No source images found, skip tiled image creation silently
            print(f"No source images found in '{image_dir}'. Skipping tiled image creation.")
            return None
        
        # Note: missing_files is tracked but no warning is printed
        
        num_images = len(image_files)
        print(f"Found {num_images} image files")
        
        # Calculate grid size (square grid)
        grid_size = int(math.ceil(math.sqrt(num_images)))
        print(f"Creating {grid_size}×{grid_size} grid ({grid_size**2} tiles)")
        
        # Calculate total canvas size including borders
        total_border_width = border_width * (grid_size - 1)
        total_border_height = border_width * (grid_size - 1)
        
        canvas_width = grid_size * tile_size + total_border_width
        canvas_height = grid_size * tile_size + total_border_height
        
        print(f"Final image size: {canvas_width}×{canvas_height} pixels")
        print(f"Tile size: {tile_size}×{tile_size} pixels")
        print(f"Border width: {border_width} pixels")
        
        # Create blank canvas
        canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
        
        # Try to load a font for tile labels
        font = None
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Helvetica.dfont",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 16)
                break
            except:
                continue
        
        if font is None:
            # Fallback to default font
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Tile images
        print("Tiling images...")
        draw = ImageDraw.Draw(canvas)
        
        for idx, (rank, img_path, pose) in enumerate(image_files):
            try:
                # Load image
                img = Image.open(img_path)
                
                # Resize if needed
                if img.size != (tile_size, tile_size):
                    img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate position in grid
                row = idx // grid_size
                col = idx % grid_size
                
                # Calculate pixel position (accounting for borders)
                x = col * (tile_size + border_width)
                y = row * (tile_size + border_width)
                
                # Paste image onto canvas
                canvas.paste(img, (x, y))
                
                # Get orientation from pose data
                orientation = pose["end_effector"]["orientation"]
                roll_deg_val = orientation["roll_deg"]
                pitch_deg_val = orientation["pitch_deg"]
                yaw_deg_val = orientation["yaw_deg"]
                
                # Format text: "r{roll}, p{pitch}, y{yaw}"
                roll_text = f"r{int(roll_deg_val)}"
                pitch_text = f"p{int(pitch_deg_val)}"
                yaw_text = f"y{int(yaw_deg_val)}"
                text = f"{roll_text}, {pitch_text}, {yaw_text}"
                
                # Draw text at top-left of tile (with small offset)
                text_x = x + 5
                text_y = y + 5
                
                # Draw text with white background for better visibility
                if font:
                    try:
                        # Get text size
                        try:
                            bbox = draw.textbbox((0, 0), text, font=font)
                            text_width = bbox[2] - bbox[0]
                            text_height_actual = bbox[3] - bbox[1]
                        except:
                            try:
                                text_width, text_height_actual = draw.textsize(text, font=font)
                            except:
                                text_width = len(text) * 10
                                text_height_actual = 16
                        
                        # Draw white background rectangle
                        padding = 2
                        draw.rectangle(
                            [text_x - padding, text_y - padding, 
                             text_x + text_width + padding, text_y + text_height_actual + padding],
                            fill=(255, 255, 255)
                        )
                        
                        # Draw text
                        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
                    except Exception as e:
                        # If font drawing fails, skip text for this tile
                        pass
                
                # Draw borders if specified
                if border_width > 0:
                    # Right border
                    if col < grid_size - 1:
                        for bw in range(border_width):
                            x_border = x + tile_size + bw
                            for by in range(tile_size):
                                if 0 <= x_border < canvas_width and 0 <= y + by < canvas_height:
                                    canvas.putpixel((x_border, y + by), (200, 200, 200))
                    
                    # Bottom border
                    if row < grid_size - 1:
                        for bw in range(border_width):
                            y_border = y + tile_size + bw
                            for bx in range(tile_size + border_width):
                                if 0 <= x + bx < canvas_width and 0 <= y_border < canvas_height:
                                    canvas.putpixel((x + bx, y_border), (200, 200, 200))
                
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                continue
        
        # Determine output filename (save to data/logs)
        if output_file:
            # Use output_file as base, change extension to .png
            base_name = os.path.splitext(output_file)[0]
            tiled_output = f"{base_name}_tiled.png"
        else:
            # Auto-generate filename in data/logs
            roll_str = f"r{int(roll_deg)}" if roll_deg is not None else "rNone"
            pitch_str = f"p{int(pitch_deg)}" if pitch_deg is not None else "pNone"
            yaw_str = f"y{int(yaw_deg)}" if yaw_deg is not None else "yNone"
            height_str = f"_h{height}" if height is not None else ""
            tiled_output = f"data/logs/{self.robot_name}_closest_{roll_str}_{pitch_str}_{yaw_str}{height_str}_tiled.png"
        
        # Create directory if it doesn't exist (no warning)
        os.makedirs(os.path.dirname(tiled_output) if os.path.dirname(tiled_output) else '.', exist_ok=True)
        
        # Save result
        print(f"Saving tiled image to: {tiled_output}")
        canvas.save(tiled_output, quality=95)
        
        file_size_mb = os.path.getsize(tiled_output) / (1024**2)
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"{'='*60}\n")
        
        return tiled_output
    
    def _save_to_jsonl(self, output_data: Dict, jsonl_path: str, roll_deg: Optional[float], 
                       pitch_deg: Optional[float], yaw_deg: Optional[float],
                       angle_step_deg: float, angle_min_deg: float, angle_max_deg: float,
                       top_k: int, arm: str):
        """
        Save results to JSONL file (append mode).
        Each pose becomes a separate line in the JSONL file.
        
        Args:
            output_data: Output data dictionary
            jsonl_path: Path to JSONL file
            roll_deg: Target roll angle
            pitch_deg: Target pitch angle
            yaw_deg: Target yaw angle
            angle_step_deg: Angle step size
            angle_min_deg: Minimum angle
            angle_max_deg: Maximum angle
            top_k: Top K value
            arm: Arm name
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(jsonl_path) if os.path.dirname(jsonl_path) else '.', exist_ok=True)
        
        # Prepare arguments that will be added to each pose entry
        arguments = {
            "robot": self.robot_name,
            "env": self.env_name,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            "top_k": top_k,
            "arm": arm,
            "angle_step_deg": angle_step_deg,
            "angle_min_deg": angle_min_deg,
            "angle_max_deg": angle_max_deg,
        }
        
        # Append each pose as a separate line
        entries_written = 0
        with open(jsonl_path, 'a') as f:
            for i, pose in enumerate(output_data["poses"], 1):
                # Create entry with pose info + arguments
                entry = {
                    "pose_id": pose["pose_id"],
                    "rank": i,  # Rank in results (1-based)
                    "angles_str": pose["angles_str"],
                    "joint_angles_deg": pose["joint_angles_deg"],
                    "joint_angles_rad": pose["joint_angles_rad"],
                    "active_joint_indices": pose["active_joint_indices"],
                    "joint_names": pose.get("joint_names", []),
                    "orientation": {
                        "roll_deg": pose["end_effector"]["orientation"]["roll_deg"],
                        "pitch_deg": pose["end_effector"]["orientation"]["pitch_deg"],
                        "yaw_deg": pose["end_effector"]["orientation"]["yaw_deg"],
                        "roll_rad": pose["end_effector"]["orientation"]["roll_rad"],
                        "pitch_rad": pose["end_effector"]["orientation"]["pitch_rad"],
                        "yaw_rad": pose["end_effector"]["orientation"]["yaw_rad"],
                    },
                    "orientation_diff_deg": pose["orientation_diff_deg"],
                    "orientation_diff_rad": pose["orientation_diff_rad"],
                    "orientation_diff_components": pose["orientation_diff_components"],
                    "root_to_ee_distance": pose.get("root_to_ee_distance"),
                    "root_position": pose.get("root_position"),
                    "ee_position": pose.get("ee_position"),
                    # Add arguments as separate columns
                    **arguments,
                }
                
                f.write(json.dumps(entry) + '\n')
                entries_written += 1
        
        print(f"Appended {entries_written} pose entries to JSONL: {jsonl_path}")
    
    def close(self):
        """Close the environment."""
        self.env.close()


def main(
    robot: str = "IIWA",
    roll: Optional[float] = 180,
    pitch: Optional[float] = None, # 이거 none
    yaw: Optional[float] = 0,
    top_k: int = 100,
    output_file: Optional[str] = None,
    arm: str = "right",
    angle_step: float = 90.0,
    angle_min: float = -90.0,
    angle_max: float = 90.0,
    tile_size: int = 256,
    border_width: int = 2,
    stack_jsonl_path: Optional[str] = 'data/poses/closest_poses_results.jsonl',
    height: Optional[str] = None,
    height_low_bar: float = 33.0,
    height_high_bar: float = 33.0,
    return_images: bool = False,
):
    """
    Find closest poses based on end effector orientation.
    
    Args:
        robot: Robot name (IIWA, Panda, Sawyer, etc.)
        roll: Target roll angle in degrees (None to ignore)
        pitch: Target pitch angle in degrees (None to ignore)
        yaw: Target yaw angle in degrees (None to ignore)
        top_k: Number of top poses to return (default: 30)
        output_file: Path to save results JSON (default: print to stdout)
        arm: Which arm to use ("right" or "left", default: "right")
        angle_step: Angle step size in degrees for pose generation
        angle_min: Minimum angle in degrees
        angle_max: Maximum angle in degrees
        height: Filter by EE height relative to root ("high", "medium", "low", or None to disable)
        height_low_bar: Percentage of z_diff range for low threshold (default: 33.0)
        height_high_bar: Percentage of z_diff range for high threshold (default: 33.0)
    
    Note:
        Environment is always set to "EmptySpace"
    
    Examples:
        # Find poses with specific orientation
        python find_closest_poses.py --robot Kinova3 --roll 10 --pitch 20 --yaw 30
        
        # Find poses matching only pitch
        python find_closest_poses.py --robot IIWA --pitch 45
        
        # Save results to file
        python find_closest_poses.py --robot Panda --roll 0 --pitch 90 --yaw 0 --output-file results.json
        
        # Use different top K and angle step
        python find_closest_poses.py --robot Kinova3 --roll 10 --pitch 20 --yaw 30 --top-k 50 --angle-step 45
        
        # Filter by height
        python find_closest_poses.py --robot IIWA --roll 0 --pitch 90 --yaw 0 --height high
        python find_closest_poses.py --robot Panda --roll 0 --pitch 0 --yaw 0 --height medium
        python find_closest_poses.py --robot Kinova3 --roll 180 --yaw 0 --height low
        
        # Customize height thresholds (40% low, 30% high -> 30% in middle)
        python find_closest_poses.py --robot IIWA --roll 0 --pitch 90 --yaw 0 --height high --height-low-bar 40 --height-high-bar 30
    """
    
    # Always use EmptySpace environment
    env = "EmptySpace"
    
    print("="*60)
    print("CLOSEST POSE FINDER")
    print("="*60)
    print(f"Robot: {robot}")
    print(f"Environment: {env}")
    print("="*60)
    
    # Create finder
    finder = ClosestPoseFinder(
        robot_name=robot,
    )
    
    try:
        results = finder.find_closest_poses(
            roll_deg=roll,
            pitch_deg=pitch,
            yaw_deg=yaw,
            angle_step_deg=angle_step,
            angle_min_deg=angle_min,
            angle_max_deg=angle_max,
            top_k=top_k,
            output_file=output_file,
            arm=arm,
            tile_size=tile_size,
            border_width=border_width,
            stack_jsonl_path=stack_jsonl_path,
            height=height,
            height_low_bar=height_low_bar,
            height_high_bar=height_high_bar,
        )
        
        if return_images:
            return results, images
        else:
            return results
    
    finally:
        finder.close()
        print("\nDone!")


if __name__ == "__main__":
    fire.Fire(main)
