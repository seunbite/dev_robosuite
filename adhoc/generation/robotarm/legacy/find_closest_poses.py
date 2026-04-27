"""
Find closest poses based on end effector orientation and root-to-EE distance.

Given target roll, gripper_orientation, yaw values, this script:
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
import sys
from typing import Optional, List, Dict
from itertools import product
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# Ensure current directory is in path for arm_pose_config import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

GRIPPER_OPENING_COLUMN = {
    'Panda': 1,
    'IIWA': 0,
    'XArm7': 0,
    'Sawyer': 0,
    'Jaco': 0,
    'Kinova3': 0,
    'UR5e': 0,
}


def classify_gripper_orientation(opening_vec, direction):
    """Classify gripper orientation from the opening vector and EE pointing direction.

    "vertical"   = opening plane perpendicular to ground (e.g. handshake)
    "horizontal" = opening plane parallel to ground (e.g. offering)

    The opening vector is the direction the jaws separate.
    If jaws open along z (large z-component) → palm is horizontal → "horizontal".
    If jaws open in the horizontal plane (small z) → palm is vertical → "vertical".

    For up/down: x-component distinguishes the two orientations instead of z.
    """
    if direction in ('up', 'down'):
        return 'horizontal' if abs(opening_vec[0]) > 0.5 else 'vertical'
    else:
        return 'horizontal' if abs(opening_vec[2]) > 0.5 else 'vertical'


class ClosestPoseFinder:
    """Find closest poses based on orientation and distance criteria."""
    
    def __init__(
        self,
        robot_name: str = "IIWA",
        env_name: str = "EmptySpace",
        controller_name: str = "OSC_POSE",
        render: bool = False,
    ):
        """
        Initialize the finder.
        
        Args:
            robot_name: Name of the robot
            env_name: Name of the environment
            controller_name: Name of the controller
            render: Whether to enable offscreen rendering
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
            "has_offscreen_renderer": render,
            "ignore_done": True,
            "use_camera_obs": render,
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
        Get end effector orientation as roll, gripper_orientation, yaw in radians.
        
        Args:
            arm: Which arm to get orientation for ("right" or "left")
            
        Returns:
            np.ndarray: [roll, gripper_orientation, yaw] in radians
        """
        try:
            # Get rotation matrix
            orn_dict = self.robot._hand_orn
            if arm in orn_dict:
                rot_mat = orn_dict[arm]
                # Convert rotation matrix to euler angles (roll, gripper_orientation, yaw)
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

    def _get_ee_rotation_matrix(self, arm="right"):
        """Get end effector rotation matrix."""
        try:
            orn_dict = self.robot._hand_orn
            if arm in orn_dict:
                return orn_dict[arm].copy()
            available_arms = list(orn_dict.keys())
            if available_arms:
                return orn_dict[available_arms[0]].copy()
            raise ValueError("No arms available")
        except Exception as e:
            print(f"Warning: Could not get EE rotation matrix: {e}")
            return np.eye(3)

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
        arm: any = "right", # Can be string or list of strings
        tile_size: int = 256,
        border_width: int = 2,
        stack_jsonl_path: Optional[str] = 'data/seed/_remainder/closest_poses_results.jsonl',
    ):
        """
        Generate poses and find closest ones based on orientation and distance.
        """
        arms = [arm] if isinstance(arm, str) else arm
            
        print("\n" + "="*60)
        print("FINDING CLOSEST POSES")
        print("="*60)
        print(f"Target orientation: R:{roll_deg}, P:{pitch_deg}, Y:{yaw_deg}")
        print(f"Arms: {arms}")
        
        # Convert target angles to radians
        target_roll = np.deg2rad(roll_deg) if roll_deg is not None else None
        target_pitch = np.deg2rad(pitch_deg) if pitch_deg is not None else None
        target_yaw = np.deg2rad(yaw_deg) if yaw_deg is not None else None
        
        # Prepare angle arrays
        specified_angles = SPECIFIED_JOINT_ANGLES.get(self.robot_name, {})
        angle_min = np.deg2rad(angle_min_deg)
        angle_max = np.deg2rad(angle_max_deg)
        angle_step = np.deg2rad(angle_step_deg)
        default_angles = np.arange(angle_min, angle_max + angle_step/2, angle_step)
        
        # If bimanual, we only vary one arm's joints to keep complexity low (3^7 instead of 3^14)
        num_arms = len(arms)
        search_indices = self.active_joint_indices
        if num_arms > 1 and len(self.active_joint_indices) % num_arms == 0:
            num_independent_joints = len(self.active_joint_indices) // num_arms
            search_indices = self.active_joint_indices[:num_independent_joints]
            print(f"Bimanual robot detected. Reducing search space to {num_independent_joints} joints (Symmetric search).")
        else:
            num_independent_joints = len(self.active_joint_indices)

        joint_angle_arrays = []
        joint_names = []
        for i in range(num_independent_joints):
            active_joint_idx = self.active_joint_indices[i]
            joint_name = f"joint_{active_joint_idx}"
            try:
                if hasattr(self.robot, 'robot_model') and hasattr(self.robot.robot_model, 'joints'):
                    robot_joints = list(self.robot.robot_model.joints)
                    if active_joint_idx < len(robot_joints):
                        joint_name = robot_joints[active_joint_idx]
            except:
                pass
            
            joint_names.append(joint_name)
            if joint_name in specified_angles:
                joint_angle_arrays.append(np.deg2rad(np.array(specified_angles[joint_name])))
            else:
                joint_angle_arrays.append(default_angles)
        
        num_angles_per_joint = [len(angles) for angles in joint_angle_arrays]
        total_combinations = int(np.prod(num_angles_per_joint))
        print(f"Generating {total_combinations:,} pose combinations for {len(arms)} arms...")
        
        selected_combinations = list(product(*[range(num) for num in num_angles_per_joint]))
        
        # Results storage per arm
        arm_scored_poses = {a: [] for a in arms}
        
        for combo_idx, angle_indices in tqdm(enumerate(selected_combinations), total=len(selected_combinations)):
            joint_pos = self.initial_joint_pos.copy()
            angle_values = []
            
            # Set positions for the independent joints
            for i in range(num_independent_joints):
                angle_value = joint_angle_arrays[i][angle_indices[i]]
                angle_values.append(angle_value)
                
                # Apply the same angle to corresponding joints in all arms (Symmetry)
                for arm_idx in range(num_arms):
                    target_joint_idx = self.active_joint_indices[i + arm_idx * num_independent_joints]
                    joint_pos[target_joint_idx] = angle_value
            
            self._set_joint_positions(joint_pos)
            
            # Calculate score for each arm
            for a in arms:
                rpy = self._get_ee_orientation_rpy(arm=a)
                roll_rad, pitch_rad, yaw_rad = rpy
                
                orientation_diff = 0.0
                diff_components = {}
                num_targets = 0
                
                if target_roll is not None:
                    diff = abs(roll_rad - target_roll)
                    diff = min(diff, 2 * np.pi - diff)
                    orientation_diff += diff
                    diff_components["roll_diff_deg"] = np.rad2deg(diff)
                    num_targets += 1
                
                if target_pitch is not None:
                    diff = abs(pitch_rad - target_pitch)
                    diff = min(diff, 2 * np.pi - diff)
                    orientation_diff += diff
                    diff_components["pitch_diff_deg"] = np.rad2deg(diff)
                    num_targets += 1
                
                if target_yaw is not None:
                    diff = abs(yaw_rad - target_yaw)
                    diff = min(diff, 2 * np.pi - diff)
                    orientation_diff += diff
                    diff_components["yaw_diff_deg"] = np.rad2deg(diff)
                    num_targets += 1
                
                if num_targets == 0: orientation_diff = 0.0
                
                # Construct angles_str for filename matching if needed
                angles_str = "_".join([f"j{i}+{int(np.rad2deg(v)):03d}" if v >= 0 else f"j{i}{int(np.rad2deg(v)):03d}" for i, v in enumerate(angle_values)])

                arm_scored_poses[a].append({
                    "pose_id": combo_idx,
                    "angles_str": angles_str,
                    "joint_angles_deg": [float(np.rad2deg(v)) for v in angle_values],
                    "joint_angles_rad": [float(v) for v in angle_values],
                    "active_joint_indices": [int(idx) for idx in self.active_joint_indices[:num_independent_joints]],
                    "end_effector": {
                        "orientation": {
                            "roll_deg": float(np.rad2deg(roll_rad)),
                            "pitch_deg": float(np.rad2deg(pitch_rad)),
                            "yaw_deg": float(np.rad2deg(yaw_rad)),
                            "roll_rad": float(roll_rad),
                            "pitch_rad": float(pitch_rad),
                            "yaw_rad": float(yaw_rad),
                        }
                    },
                    "orientation_diff_rad": float(orientation_diff),
                    "orientation_diff_deg": float(np.rad2deg(orientation_diff)),
                    "orientation_diff_components": diff_components,
                })
            
            if (combo_idx + 1) % 500 == 0:
                self._set_joint_positions(self.initial_joint_pos)

        # Final filtering and sorting for each arm
        final_arm_results = {}
        for a in arms:
            # First try strict tolerance (60 deg)
            filtered = [p for p in arm_scored_poses[a] if p["orientation_diff_deg"] <= 60.0]
            
            # If nothing found, relax tolerance to 90 deg
            if not filtered:
                filtered = [p for p in arm_scored_poses[a] if p["orientation_diff_deg"] <= 90.0]
                if filtered:
                    print(f"  Warning: No poses within 60° for arm {a}, relaxed tolerance to 90°")
            
            filtered.sort(key=lambda x: x["orientation_diff_rad"])
            top_poses = filtered[:top_k]
            
            # Distance and Region calculation
            for pose in top_poses:
                joint_pos = self.initial_joint_pos.copy()
                for i in range(num_independent_joints):
                    angle_value = pose["joint_angles_rad"][i]
                    for arm_idx in range(num_arms):
                        target_idx = self.active_joint_indices[i + arm_idx * num_independent_joints]
                        joint_pos[target_idx] = angle_value
                
                self._set_joint_positions(joint_pos)
                
                root_pos = self._get_root_position()
                ee_pos = self._get_ee_position(arm=a)
                pose["ee_position"] = {"x": float(ee_pos[0]), "y": float(ee_pos[1]), "z": float(ee_pos[2])}
                pose["root_position"] = {"x": float(root_pos[0]), "y": float(root_pos[1]), "z": float(root_pos[2])}
                pose["x_diff"] = pose["ee_position"]["x"] - pose["root_position"]["x"]
                pose["y_diff"] = pose["ee_position"]["y"] - pose["root_position"]["y"]
                pose["z_diff"] = pose["ee_position"]["z"] - pose["root_position"]["z"]
                pose["root_to_ee_distance"] = float(np.linalg.norm(ee_pos - root_pos))

            self._compute_percentiles(top_poses)
            top_poses.sort(key=lambda x: x["root_to_ee_distance"])
            final_arm_results[a] = top_poses

            # Save to JSONL if requested
            if stack_jsonl_path and top_poses:
                output_data = {
                    "robot_name": self.robot_name,
                    "env_name": self.env_name,
                    "poses": top_poses
                }
                self._save_to_jsonl(output_data, stack_jsonl_path, roll_deg, pitch_deg, yaw_deg,
                                   angle_step_deg, angle_min_deg, angle_max_deg, top_k, a)

        return final_arm_results

    def _compute_percentiles(self, poses):
        """Compute percentiles (0-100) for x, y, z diffs across all poses."""
        if not poses:
            return poses
        for axis in ['x', 'y', 'z']:
            values = np.array([p[f"{axis}_diff"] for p in poses])
            order = np.argsort(values)
            n = len(values)
            for rank, idx in enumerate(order):
                poses[idx][f"{axis}_pct"] = int(round(rank / max(n - 1, 1) * 100))
        return poses

    def _save_to_jsonl(self, output_data: Dict, jsonl_path: str, roll_deg: Optional[float], 
                       pitch_deg: Optional[float], yaw_deg: Optional[float],
                       angle_step_deg: float, angle_min_deg: float, angle_max_deg: float,
                       top_k: int, arm: str):
        """
        Save results to JSONL file (append mode).
        Each pose becomes a separate line in the JSONL file.
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
                    "angles_str": pose.get("angles_str", ""),
                    "joint_angles_deg": pose["joint_angles_deg"],
                    "joint_angles_rad": pose["joint_angles_rad"],
                    "active_joint_indices": pose["active_joint_indices"],
                    "orientation": pose["end_effector"]["orientation"],
                    "orientation_diff_deg": pose["orientation_diff_deg"],
                    "orientation_diff_rad": pose["orientation_diff_rad"],
                    "orientation_diff_components": pose["orientation_diff_components"],
                    "root_to_ee_distance": pose.get("root_to_ee_distance"),
                    "root_position": pose.get("root_position"),
                    "ee_position": pose.get("ee_position"),
                    "x_diff": pose.get("x_diff"),
                    "y_diff": pose.get("y_diff"),
                    "z_diff": pose.get("z_diff"),
                    "x_pct": pose.get("x_pct"),
                    "y_pct": pose.get("y_pct"),
                    "z_pct": pose.get("z_pct"),
                    **arguments,
                }
                f.write(json.dumps(entry) + '\n')
                entries_written += 1
        
        print(f"Appended {entries_written} pose entries to JSONL: {jsonl_path}")

    def brute_force_and_classify(
        self,
        angle_step_deg: float = 90.0,
        angle_min_deg: float = -90.0,
        angle_max_deg: float = 90.0,
        stack_jsonl_path: str = 'data/seed/_remainder/closest_poses_results.jsonl',
        orientation_tolerance_deg: float = 60.0,
        top_n_per_orientation: int = 20,
    ):
        """
        Brute force all joint combinations once and classify them into standard orientations and regions.
        """
        from arm_pose_config import poses, pitch_poses
        
        # 1. Prepare Target Orientations
        target_orientations = []
        for dir_name, dir_list in poses.items():
            for p_name, p_list in pitch_poses.items():
                for d_pose in dir_list:
                    for p_val in p_list:
                        target_orientations.append({
                            "dir": dir_name,
                            "pitch_type": p_name,
                            "roll": d_pose['roll'],
                            "gripper_orientation": p_val,
                            "yaw": d_pose['yaw']
                        })

        # 2. Generate combinations
        specified_angles = SPECIFIED_JOINT_ANGLES.get(self.robot_name, {})
        angle_min = np.deg2rad(angle_min_deg)
        angle_max = np.deg2rad(angle_max_deg)
        angle_step = np.deg2rad(angle_step_deg)
        default_angles = np.arange(angle_min, angle_max + angle_step/2, angle_step)
        
        is_bimanual = any(arm in self.robot_name.lower() for arm in ["gr1", "bimanual"])
        if is_bimanual and len(self.active_joint_indices) % 2 == 0:
            num_arms = 2
            num_independent_joints = len(self.active_joint_indices) // 2
        else:
            num_arms = 1
            num_independent_joints = len(self.active_joint_indices)

        joint_angle_arrays = []
        for i in range(num_independent_joints):
            active_joint_idx = self.active_joint_indices[i]
            joint_name = f"joint_{active_joint_idx}"
            try:
                if hasattr(self.robot, 'robot_model') and hasattr(self.robot.robot_model, 'joints'):
                    robot_joints = list(self.robot.robot_model.joints)
                    if active_joint_idx < len(robot_joints):
                        joint_name = robot_joints[active_joint_idx]
            except: pass
            
            if joint_name in specified_angles:
                joint_angle_arrays.append(np.deg2rad(np.array(specified_angles[joint_name])))
            else:
                joint_angle_arrays.append(default_angles)
        
        combinations = list(product(*[range(len(a)) for a in joint_angle_arrays]))
        print(f"Brute forcing {len(combinations):,} combinations for {self.robot_name}...")
        
        # Temporary storage for all poses and their orientations
        all_poses_data = []
        
        for combo_idx, angle_indices in tqdm(enumerate(combinations), total=len(combinations)):
            joint_pos = self.initial_joint_pos.copy()
            angle_values = []
            for i in range(num_independent_joints):
                v = joint_angle_arrays[i][angle_indices[i]]
                angle_values.append(v)
                for arm_idx in range(num_arms):
                    target_joint_idx = self.active_joint_indices[i + arm_idx * num_independent_joints]
                    joint_pos[target_joint_idx] = v
            
            self._set_joint_positions(joint_pos)
            
            arms_to_check = ["right", "left"] if num_arms > 1 else ["right"]
            for arm in arms_to_check:
                ee_pos = self._get_ee_position(arm=arm)
                rpy = self._get_ee_orientation_rpy(arm=arm)
                rot_mat = self._get_ee_rotation_matrix(arm=arm)
                root_pos = self._get_root_position()
                
                all_poses_data.append({
                    "pose_id": combo_idx,
                    "angle_values": angle_values,
                    "arm": arm,
                    "ee_pos": ee_pos,
                    "rpy": rpy,
                    "rot_mat": rot_mat,
                    "root_pos": root_pos,
                    "x_diff": ee_pos[0] - root_pos[0],
                    "y_diff": ee_pos[1] - root_pos[1],
                    "z_diff": ee_pos[2] - root_pos[2],
                })
            
            if (combo_idx + 1) % 500 == 0:
                self._set_joint_positions(self.initial_joint_pos)

        # 3. For each target orientation, find the top N closest poses
        results = []

        print(f"Matching {len(all_poses_data)} poses against {len(target_orientations)} targets...")
        for target in target_orientations:
            t_roll = np.deg2rad(target['roll'])
            t_pitch = np.deg2rad(target['gripper_orientation'])
            t_yaw = np.deg2rad(target['yaw'])
            
            scored_for_target = []
            for p in all_poses_data:
                diff = 0.0
                for v1, v2 in zip(p["rpy"], [t_roll, t_pitch, t_yaw]):
                    d = abs(v1 - v2)
                    d = min(d, 2 * np.pi - d)
                    diff += d
                
                diff_deg = np.rad2deg(diff)
                if diff_deg <= orientation_tolerance_deg:
                    scored_for_target.append((p, diff_deg))
            
            scored_for_target.sort(key=lambda x: x[1])
            top_for_target = scored_for_target[:top_n_per_orientation]
            
            for p, diff_deg in top_for_target:
                results.append({
                    "pose_id": p["pose_id"],
                    "joint_angles_rad": [float(v) for v in p["angle_values"]],
                    "joint_angles_deg": [float(np.rad2deg(v)) for v in p["angle_values"]],
                    "active_joint_indices": [int(idx) for idx in self.active_joint_indices[:num_independent_joints]],
                    "arm": p["arm"],
                    "roll_deg": float(np.rad2deg(p["rpy"][0])),
                    "pitch_deg": float(np.rad2deg(p["rpy"][1])),
                    "yaw_deg": float(np.rad2deg(p["rpy"][2])),
                    "target_roll": target['roll'],
                    "target_pitch": target['gripper_orientation'],
                    "target_yaw": target['yaw'],
                    "target_dir": target['dir'],
                    "target_pitch_type": target['pitch_type'],
                    "orientation_diff_deg": float(diff_deg),
                    "x_diff": float(p["x_diff"]),
                    "y_diff": float(p["y_diff"]),
                    "z_diff": float(p["z_diff"]),
                    "ee_position": {"x": float(p["ee_pos"][0]), "y": float(p["ee_pos"][1]), "z": float(p["ee_pos"][2])},
                    "root_position": {"x": float(p["root_pos"][0]), "y": float(p["root_pos"][1]), "z": float(p["root_pos"][2])},
                    "root_to_ee_distance": float(np.linalg.norm(p["ee_pos"] - p["root_pos"])),
                    "_rot_mat": p["rot_mat"],
                })

        if not results:
            print("No matching poses found.")
            return

        # 3b. Reclassify gripper_orientation from rotation matrix
        col_idx = GRIPPER_OPENING_COLUMN.get(self.robot_name, 0)
        for r in results:
            opening_vec = r["_rot_mat"][:, col_idx]
            r["target_pitch_type"] = classify_gripper_orientation(opening_vec, r["target_dir"])

        # 3c. Deduplicate by (pose_id, arm, dir, gripper_orientation), keep lowest diff
        seen = {}
        for r in results:
            key = (r["pose_id"], r["arm"], r["target_dir"], r["target_pitch_type"])
            if key not in seen or r["orientation_diff_deg"] < seen[key]["orientation_diff_deg"]:
                seen[key] = r
        results = list(seen.values())
        print(f"After reclassification + dedup: {len(results)} poses")

        # 4. Compute global percentiles (0-100) across all results
        self._compute_percentiles(results)

        # 5. Save to JSONL (drop internal _rot_mat before serialization)
        for p in results:
            p.pop("_rot_mat", None)
        os.makedirs(os.path.dirname(stack_jsonl_path) if os.path.dirname(stack_jsonl_path) else '.', exist_ok=True)
        with open(stack_jsonl_path, 'a') as f:
            for p in results:
                entry = {
                    "robot": self.robot_name,
                    "env": self.env_name,
                    "pose_id": p["pose_id"],
                    "angles_str": "_".join([f"j{i}+{int(v):03d}" if v >= 0 else f"j{i}{int(v):03d}" for i, v in enumerate(p["joint_angles_deg"])]),
                    "joint_angles_deg": p["joint_angles_deg"],
                    "joint_angles_rad": p["joint_angles_rad"],
                    "active_joint_indices": p["active_joint_indices"],
                    "orientation": {
                        "roll_deg": p["roll_deg"],
                        "pitch_deg": p["pitch_deg"],
                        "yaw_deg": p["yaw_deg"]
                    },
                    "roll_deg": p["target_roll"],
                    "pitch_deg": p["target_pitch"],
                    "yaw_deg": p["target_yaw"],
                    "dir": p["target_dir"],
                    "gripper_orientation": p["target_pitch_type"],
                    "orientation_diff_deg": p["orientation_diff_deg"],
                    "x_diff": p["x_diff"],
                    "y_diff": p["y_diff"],
                    "z_diff": p["z_diff"],
                    "x_pct": p.get("x_pct"),
                    "y_pct": p.get("y_pct"),
                    "z_pct": p.get("z_pct"),
                    "ee_position": p["ee_position"],
                    "root_position": p["root_position"],
                    "root_to_ee_distance": p["root_to_ee_distance"],
                    "arm": p["arm"],
                    "top_k": 0,
                }
                f.write(json.dumps(entry) + '\n')
        
        print(f"Saved {len(results)} poses for {self.robot_name} to {stack_jsonl_path}")

    def close(self):
        """Close the environment."""
        self.env.close()


def main(
    robot: str = "IIWA",
    roll: Optional[float] = 180,
    gripper_orientation: Optional[float] = None,
    yaw: Optional[float] = 0,
    top_k: int = 100,
    output_file: Optional[str] = None,
    arm: str = "right",
    angle_step: float = 90.0,
    angle_min: float = -90.0,
    angle_max: float = 90.0,
    stack_jsonl_path: Optional[str] = 'data/seed/_remainder/closest_poses_results.jsonl',
    brute_force: bool = False,
    orientation_tolerance: float = 30.0,
):
    """
    Find closest poses based on end effector orientation and 3D region.
    """
    env = "EmptySpace"
    print("="*60)
    print("CLOSEST POSE FINDER")
    print("="*60)
    print(f"Robot: {robot}")
    print(f"Environment: {env}")
    print("="*60)
    
    finder = ClosestPoseFinder(robot_name=robot)
    try:
        if brute_force:
            results = finder.brute_force_and_classify(
                angle_step_deg=angle_step,
                angle_min_deg=angle_min,
                angle_max_deg=angle_max,
                stack_jsonl_path=stack_jsonl_path,
                orientation_tolerance_deg=orientation_tolerance,
            )
        else:
            results = finder.find_closest_poses(
                roll_deg=roll,
                pitch_deg=gripper_orientation,
                yaw_deg=yaw,
                angle_step_deg=angle_step,
                angle_min_deg=angle_min,
                angle_max_deg=angle_max,
                top_k=top_k,
                output_file=output_file,
                arm=arm,
                stack_jsonl_path=stack_jsonl_path,
            )
        return results
    finally:
        finder.close()
        print("\nDone!")


if __name__ == "__main__":
    fire.Fire(main)
