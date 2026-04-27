"""
Export all possible poses for a robot ONCE, with their orientations.
This avoids recalculating poses for each target orientation query.

Usage:
    python adhoc/generation/robotarm/export_all_poses_once.py --robot IIWA
    python adhoc/generation/robotarm/export_all_poses_once.py --robot Panda --output data/poses/all_panda_poses.jsonl
"""

import fire
import os
import json
import numpy as np
from typing import Optional, List, Dict
from itertools import product
from tqdm import tqdm

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.utils import transform_utils as T

# Import the same constants
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


class AllPoseExporter:
    """Export all possible poses with their orientations."""
    
    def __init__(
        self,
        robot_name: str = "IIWA",
        env_name: str = "EmptySpace",
        controller_name: str = "OSC_POSE",
    ):
        """Initialize the exporter."""
        self.robot_name = robot_name
        self.env_name = env_name
        self.controller_name = controller_name
        
        print(f"Initializing robot: {robot_name}")
        
        # Setup environment
        options = {
            "env_name": env_name,
            "robots": robot_name,
            "has_renderer": False,
            "has_offscreen_renderer": False,
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
        base_active_joint_indices = list(range(self.num_joints - 1))
        self.active_joint_indices = [idx for idx in base_active_joint_indices if idx not in self.fixed_joint_indices]
        
        print(f"Total joints: {self.num_joints}")
        print(f"Active joints: {len(self.active_joint_indices)}")
        if self.fixed_joint_indices:
            print(f"Fixed joints: {len(self.fixed_joint_indices)}")
    
    def _get_root_position(self):
        """Get root body position in world coordinates."""
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
        """Get end effector orientation as roll, gripper_orientation, yaw in radians."""
        try:
            orn_dict = self.robot._hand_orn
            if arm in orn_dict:
                rot_mat = orn_dict[arm]
                rpy = T.mat2euler(rot_mat)
                return rpy
            else:
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
    
    def export_all_poses(
        self,
        angle_step_deg: float = 90.0,
        angle_min_deg: float = -90.0,
        angle_max_deg: float = 90.0,
        arm: str = "right",
        output_file: Optional[str] = None,
    ):
        """
        Generate ALL poses and save with their orientations.
        
        Args:
            angle_step_deg: Step size in degrees for pose generation
            angle_min_deg: Minimum angle in degrees
            angle_max_deg: Maximum angle in degrees
            arm: Which arm to use ("right" or "left")
            output_file: Path to save JSONL file (default: data/poses/all_{robot}_poses.jsonl)
        """
        print("\n" + "="*60)
        print("EXPORTING ALL POSES")
        print("="*60)
        print(f"Robot: {self.robot_name}")
        print(f"Arm: {arm}")
        print(f"Angle step: {angle_step_deg}°")
        print(f"Angle range: {angle_min_deg}° to {angle_max_deg}°")
        print("="*60 + "\n")
        
        # Prepare angle arrays
        specified_angles = SPECIFIED_JOINT_ANGLES.get(self.robot_name, {})
        angle_min = np.deg2rad(angle_min_deg)
        angle_max = np.deg2rad(angle_max_deg)
        angle_step = np.deg2rad(angle_step_deg)
        default_angles = np.arange(angle_min, angle_max + angle_step/2, angle_step)
        
        # Create angle arrays for each active joint
        joint_angle_arrays = []
        joint_names = []
        for active_joint_idx in self.active_joint_indices:
            try:
                joint_name = f"joint_{active_joint_idx}"
                if hasattr(self.robot, 'robot_model') and hasattr(self.robot.robot_model, 'joints'):
                    robot_joints = list(self.robot.robot_model.joints)
                    if active_joint_idx < len(robot_joints):
                        joint_name = robot_joints[active_joint_idx]
            except:
                joint_name = f"joint_{active_joint_idx}"
            
            joint_names.append(joint_name)
            
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
        
        # Determine output file
        if output_file is None:
            output_file = f"data/poses/all_{self.robot_name}_poses.jsonl"
        
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        # Export poses to JSONL (one pose per line)
        print(f"Saving to: {output_file}")
        
        with open(output_file, 'w') as f:
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
                
                # Get positions for distance calculation
                root_pos = self._get_root_position()
                ee_pos = self._get_ee_position(arm=arm)
                distance = np.linalg.norm(ee_pos - root_pos)
                
                # Calculate z_diff for height classification
                z_diff = ee_pos[2] - root_pos[2]
                
                # Check if EE is in front of root
                is_front = ee_pos[0] > root_pos[0]
                
                # Generate angles string
                angles_str = "_".join([f"j{self.active_joint_indices[j]}{int(np.rad2deg(angle_values[j])):+04d}" 
                                        for j in range(len(angle_indices))])
                
                # Create pose entry
                pose_entry = {
                    "robot": self.robot_name,
                    "pose_id": combo_idx,
                    "angles_str": angles_str,
                    "joint_angles_deg": [float(np.rad2deg(angle_values[j])) for j in range(len(angle_indices))],
                    "joint_angles_rad": [float(angle_values[j]) for j in range(len(angle_indices))],
                    "active_joint_indices": self.active_joint_indices,
                    "joint_names": joint_names,
                    "orientation": {
                        "roll_deg": float(np.rad2deg(roll_rad)),
                        "pitch_deg": float(np.rad2deg(pitch_rad)),
                        "yaw_deg": float(np.rad2deg(yaw_rad)),
                        "roll_rad": float(roll_rad),
                        "pitch_rad": float(pitch_rad),
                        "yaw_rad": float(yaw_rad),
                    },
                    "root_position": {
                        "x": float(root_pos[0]),
                        "y": float(root_pos[1]),
                        "z": float(root_pos[2]),
                    },
                    "ee_position": {
                        "x": float(ee_pos[0]),
                        "y": float(ee_pos[1]),
                        "z": float(ee_pos[2]),
                    },
                    "root_to_ee_distance": float(distance),
                    "z_diff": float(z_diff),
                    "is_front": bool(is_front),
                    "arm": arm,
                }
                
                # Write to JSONL
                f.write(json.dumps(pose_entry) + '\n')
                
                # Return to initial pose occasionally to prevent drift
                if (combo_idx + 1) % 500 == 0:
                    self._set_joint_positions(self.initial_joint_pos)
        
        print(f"\n{'='*60}")
        print(f"Exported {total_combinations:,} poses to: {output_file}")
        print(f"{'='*60}\n")
        
        return output_file
    
    def close(self):
        """Close the environment."""
        self.env.close()


def main(
    robot: str = "IIWA",
    output_file: Optional[str] = None,
    arm: str = "right",
    angle_step: float = 90.0,
    angle_min: float = -90.0,
    angle_max: float = 90.0,
):
    """
    Export all poses for a robot to JSONL file.
    
    Args:
        robot: Robot name (IIWA, Panda, Sawyer, etc.)
        output_file: Path to save JSONL (default: data/poses/all_{robot}_poses.jsonl)
        arm: Which arm to use ("right" or "left", default: "right")
        angle_step: Angle step size in degrees for pose generation
        angle_min: Minimum angle in degrees
        angle_max: Maximum angle in degrees
    
    Examples:
        # Export all IIWA poses
        python adhoc/generation/robotarm/export_all_poses_once.py --robot IIWA
        
        # Export with custom output
        python adhoc/generation/robotarm/export_all_poses_once.py --robot Panda --output-file my_poses.jsonl
        
        # Custom angle range
        python adhoc/generation/robotarm/export_all_poses_once.py --robot Kinova3 --angle-step 45
    """
    print("="*60)
    print("ALL POSE EXPORTER")
    print("="*60)
    print(f"Robot: {robot}")
    print("="*60)
    
    exporter = AllPoseExporter(robot_name=robot)
    
    try:
        output_path = exporter.export_all_poses(
            angle_step_deg=angle_step,
            angle_min_deg=angle_min,
            angle_max_deg=angle_max,
            arm=arm,
            output_file=output_file,
        )
        return output_path
    finally:
        exporter.close()
        print("\nDone!")


if __name__ == "__main__":
    fire.Fire(main)
