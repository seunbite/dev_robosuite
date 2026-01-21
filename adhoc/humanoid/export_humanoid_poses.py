"""
Export all possible poses for humanoid robots (one arm only).
This is specifically for humanoid robots where we fix all joints except one arm.

Supported robots:
- GR1ArmsOnly (14 joints: right arm [0-6], left arm [7-13])
- GR1FixedLowerBody (20 joints: head[0-2], torso[3-5], right arm[6-12], left arm[13-19])
- GR1FloatingBody (20 joints: head[0-2], torso[3-5], right arm[6-12], left arm[13-19])
- GR1 (32 joints: full humanoid)

Usage:
    # Arms-only version (simplest)
    python adhoc/humanoid/export_humanoid_poses.py --robot GR1ArmsOnly --active-arm right
    python adhoc/humanoid/export_humanoid_poses.py --robot GR1ArmsOnly --active-arm left
    
    # Fixed lower body version
    python adhoc/humanoid/export_humanoid_poses.py --robot GR1FixedLowerBody --active-arm right
"""

import fire
import os
import json
import numpy as np
from typing import Optional
from itertools import product
from tqdm import tqdm

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.utils import transform_utils as T

# Humanoid-specific fixed joint configurations
# Format: robot_name -> {arm -> fixed_joint_indices_str}
HUMANOID_FIXED_JOINTS = {
    'GR1ArmsOnly': {
        'right': "7-13",  # Fix left arm
        'left': "0-6",     # Fix right arm
    },
    'GR1FixedLowerBody': {
        'right': "0-5, 13-19",  # Fix head, torso, left arm
        'left': "0-5, 6-12",    # Fix head, torso, right arm
    },
    'GR1FloatingBody': {
        'right': "0-5, 13-19",  # Fix head, torso, left arm
        'left': "0-5, 6-12",    # Fix head, torso, right arm
    },
    'GR1': {
        'right': "0-5, 13-19, 20-31",  # Fix head, torso, left arm, legs
        'left': "0-5, 6-12, 20-31",    # Fix head, torso, right arm, legs
    },
}


class HumanoidPoseExporter:
    """Export all possible poses for humanoid robots with one arm active."""
    
    def __init__(
        self,
        robot_name: str = "GR1ArmsOnly",
        active_arm: str = "right",
        env_name: str = "EmptySpace",
        controller_name: str = "OSC_POSE",
    ):
        """Initialize the exporter."""
        self.robot_name = robot_name
        self.active_arm = active_arm
        self.env_name = env_name
        self.controller_name = controller_name
        
        print(f"Initializing humanoid robot: {robot_name} (active arm: {active_arm})")
        
        if robot_name not in HUMANOID_FIXED_JOINTS:
            raise ValueError(f"Robot {robot_name} not supported. Supported: {list(HUMANOID_FIXED_JOINTS.keys())}")
        
        if active_arm not in ['right', 'left']:
            raise ValueError(f"active_arm must be 'right' or 'left', got {active_arm}")
        
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
        fixed_indices_str = HUMANOID_FIXED_JOINTS[robot_name][active_arm]
        self.fixed_joint_indices = self._parse_fixed_indices(fixed_indices_str)
        
        # Active joints (excluding gripper joint, which is last)
        base_active_joint_indices = list(range(self.num_joints - 1))
        self.active_joint_indices = [idx for idx in base_active_joint_indices if idx not in self.fixed_joint_indices]
        
        print(f"Total joints: {self.num_joints}")
        print(f"Active joints ({active_arm} arm): {self.active_joint_indices}")
        print(f"Fixed joints: {self.fixed_joint_indices}")
    
    def _parse_fixed_indices(self, fixed_indices_str):
        """Parse fixed joint indices string like '0-5, 13-19'"""
        fixed_indices = []
        for part in fixed_indices_str.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                start = int(start.strip())
                end = int(end.strip())
                if start <= end:
                    fixed_indices.extend(range(start, end + 1))
            else:
                fixed_indices.append(int(part))
        return sorted(list(set([idx for idx in fixed_indices if 0 <= idx < self.num_joints])))
    
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
    
    def _get_ee_position(self):
        """Get end effector position for active arm."""
        try:
            pos_dict = self.robot._hand_pos
            if self.active_arm in pos_dict:
                return pos_dict[self.active_arm].copy()
            else:
                raise ValueError(f"Active arm '{self.active_arm}' not found in position dict")
        except Exception as e:
            print(f"Warning: Could not get EE position: {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def _get_ee_orientation_rpy(self):
        """Get end effector orientation for active arm as roll, pitch, yaw in radians."""
        try:
            orn_dict = self.robot._hand_orn
            if self.active_arm in orn_dict:
                rot_mat = orn_dict[self.active_arm]
                rpy = T.mat2euler(rot_mat)
                return rpy
            else:
                raise ValueError(f"Active arm '{self.active_arm}' not found in orientation dict")
        except Exception as e:
            print(f"Warning: Could not get EE orientation: {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def export_all_poses(
        self,
        angle_step_deg: float = 90.0,
        angle_min_deg: float = -90.0,
        angle_max_deg: float = 90.0,
        output_file: Optional[str] = None,
    ):
        """
        Generate ALL poses for the active arm and save with their orientations.
        
        Args:
            angle_step_deg: Step size in degrees for pose generation
            angle_min_deg: Minimum angle in degrees
            angle_max_deg: Maximum angle in degrees
            output_file: Path to save JSONL file (default: data/poses/humanoid/all_{robot}_{arm}_poses.jsonl)
        """
        print("\n" + "="*60)
        print("EXPORTING ALL HUMANOID POSES")
        print("="*60)
        print(f"Robot: {self.robot_name}")
        print(f"Active arm: {self.active_arm}")
        print(f"Angle step: {angle_step_deg}°")
        print(f"Angle range: {angle_min_deg}° to {angle_max_deg}°")
        print("="*60 + "\n")
        
        # Prepare angle arrays
        angle_min = np.deg2rad(angle_min_deg)
        angle_max = np.deg2rad(angle_max_deg)
        angle_step = np.deg2rad(angle_step_deg)
        default_angles = np.arange(angle_min, angle_max + angle_step/2, angle_step)
        
        # Create angle arrays for each active joint
        joint_angle_arrays = []
        joint_names = []
        for active_joint_idx in self.active_joint_indices:
            joint_name = f"joint_{active_joint_idx}"
            joint_names.append(joint_name)
            joint_angle_arrays.append(default_angles)
        
        # Calculate total combinations
        num_angles_per_joint = [len(angles) for angles in joint_angle_arrays]
        total_combinations = 1
        for num in num_angles_per_joint:
            total_combinations *= num
        
        print(f"Generating {total_combinations:,} pose combinations for {self.active_arm} arm...")
        
        # Generate combinations
        selected_combinations = list(product(*[range(num) for num in num_angles_per_joint]))
        
        # Determine output file
        if output_file is None:
            output_file = f"data/poses/humanoid/all_{self.robot_name}_{self.active_arm}_poses.jsonl"
        
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        # Export poses to JSONL
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
                rpy = self._get_ee_orientation_rpy()
                roll_rad, pitch_rad, yaw_rad = rpy[0], rpy[1], rpy[2]
                
                # Get positions
                root_pos = self._get_root_position()
                ee_pos = self._get_ee_position()
                distance = np.linalg.norm(ee_pos - root_pos)
                z_diff = ee_pos[2] - root_pos[2]
                is_front = ee_pos[0] > root_pos[0]
                
                # Generate angles string
                angles_str = "_".join([f"j{self.active_joint_indices[j]}{int(np.rad2deg(angle_values[j])):+04d}" 
                                        for j in range(len(angle_indices))])
                
                # Create pose entry
                pose_entry = {
                    "robot": self.robot_name,
                    "active_arm": self.active_arm,
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
                    "arm": self.active_arm,
                }
                
                # Write to JSONL
                f.write(json.dumps(pose_entry) + '\n')
                
                # Return to initial pose occasionally
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
    robot: str = "GR1ArmsOnly",
    active_arm: str = "right",
    output_file: Optional[str] = None,
    angle_step: float = 90.0,
    angle_min: float = -90.0,
    angle_max: float = 90.0,
):
    """
    Export all poses for a humanoid robot (one arm) to JSONL file.
    
    Args:
        robot: Robot name (GR1ArmsOnly, GR1FixedLowerBody, GR1FloatingBody, GR1)
        active_arm: Which arm to move ("right" or "left")
        output_file: Path to save JSONL (default: data/poses/humanoid/all_{robot}_{arm}_poses.jsonl)
        angle_step: Angle step size in degrees
        angle_min: Minimum angle in degrees
        angle_max: Maximum angle in degrees
    
    Examples:
        # Arms-only, right arm
        python adhoc/humanoid/export_humanoid_poses.py --robot GR1ArmsOnly --active-arm right
        
        # Arms-only, left arm
        python adhoc/humanoid/export_humanoid_poses.py --robot GR1ArmsOnly --active-arm left
        
        # Fixed lower body, right arm
        python adhoc/humanoid/export_humanoid_poses.py --robot GR1FixedLowerBody --active-arm right
        
        # Full humanoid (legs fixed), right arm
        python adhoc/humanoid/export_humanoid_poses.py --robot GR1 --active-arm right
    """
    print("="*60)
    print("HUMANOID POSE EXPORTER")
    print("="*60)
    print(f"Robot: {robot}")
    print(f"Active arm: {active_arm}")
    print("="*60)
    
    exporter = HumanoidPoseExporter(robot_name=robot, active_arm=active_arm)
    
    try:
        output_path = exporter.export_all_poses(
            angle_step_deg=angle_step,
            angle_min_deg=angle_min,
            angle_max_deg=angle_max,
            output_file=output_file,
        )
        return output_path
    finally:
        exporter.close()
        print("\nDone!")


if __name__ == "__main__":
    fire.Fire(main)
