"""
Export end effector orientations (pitch, yaw, roll) for generated poses.

This script loads generated poses and calculates end effector orientations
for each pose configuration.
"""

import fire
import os
import json
import numpy as np
from tqdm import tqdm

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


class EEOrientationExporter:
    """Export end effector orientations for robot poses."""
    
    def __init__(
        self,
        robot_name: str = "IIWA",
        env_name: str = "Lift",
        controller_name: str = "OSC_POSE",
    ):
        """
        Initialize the exporter.
        
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
            "has_offscreen_renderer": True,
            "ignore_done": True,
            "use_camera_obs": False,  # Don't need camera for this
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
    
    def _set_joint_positions(self, joint_positions):
        """Set joint positions and update simulation."""
        self.robot.set_robot_joint_positions(joint_positions)
        self.env.sim.forward()
    
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
                # mat2euler returns [roll, pitch, yaw]
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
    
    def _get_ee_position(self, arm="right"):
        """
        Get end effector position.
        
        Args:
            arm: Which arm to get position for ("right" or "left")
            
        Returns:
            np.ndarray: [x, y, z] position in meters
        """
        try:
            pos_dict = self.robot._hand_pos
            if arm in pos_dict:
                return pos_dict[arm].copy()
            else:
                # If arm not found, try with first available arm
                available_arms = list(pos_dict.keys())
                if available_arms:
                    arm = available_arms[0]
                    return pos_dict[arm].copy()
                else:
                    raise ValueError("No arms available")
        except Exception as e:
            print(f"Warning: Could not get EE position: {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def export_orientations(
        self,
        angle_step_deg: float = 90.0,
        angle_min_deg: float = -90.0,
        angle_max_deg: float = 90.0,
        output_file: str = None,
        arm: str = "right",
    ):
        """
        Generate poses and export end effector orientations.
        
        Args:
            angle_step_deg: Step size in degrees
            angle_min_deg: Minimum angle in degrees
            angle_max_deg: Maximum angle in degrees
            output_file: Path to output JSON file (default: data/ee_orientations/{robot_name}.json)
            arm: Which arm to get orientation for ("right" or "left")
        """
        print("\n" + "="*60)
        print("END EFFECTOR ORIENTATION EXPORT")
        print("="*60)
        
        # Default output file
        if output_file is None:
            output_file = f"data/ee_orientations/{self.robot_name}_ee_orientations.json"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
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
            # Try to get joint name (this might not work for all robots)
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
        
        print(f"Total poses to generate: {total_combinations:,}")
        print(f"Output file: {output_file}")
        print(f"Arm: {arm}")
        print("="*60 + "\n")
        
        # Generate combinations
        from itertools import product
        selected_combinations = list(product(*[range(num) for num in num_angles_per_joint]))
        
        results = []
        
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
            ee_pos = self._get_ee_position(arm=arm)
            
            # Generate angles string for filename reference
            angles_str = "_".join([f"j{self.active_joint_indices[j]}{int(np.rad2deg(angle_values[j])):+04d}" 
                                    for j in range(len(angle_indices))])
            
            # Store result
            result = {
                "pose_id": combo_idx,
                "angles_str": angles_str,
                "joint_angles_deg": [float(np.rad2deg(angle_values[j])) for j in range(len(angle_indices))],
                "joint_angles_rad": [float(angle_values[j]) for j in range(len(angle_indices))],
                "active_joint_indices": self.active_joint_indices,
                "joint_names": joint_names,
                "end_effector": {
                    "position": {
                        "x": float(ee_pos[0]),
                        "y": float(ee_pos[1]),
                        "z": float(ee_pos[2]),
                    },
                    "orientation": {
                        "roll_rad": float(rpy[0]),
                        "pitch_rad": float(rpy[1]),
                        "yaw_rad": float(rpy[2]),
                        "roll_deg": float(np.rad2deg(rpy[0])),
                        "pitch_deg": float(np.rad2deg(rpy[1])),
                        "yaw_deg": float(np.rad2deg(rpy[2])),
                    }
                }
            }
            
            results.append(result)
            
            # Return to initial pose occasionally to prevent drift
            if (combo_idx + 1) % 500 == 0:
                self._set_joint_positions(self.initial_joint_pos)
        
        # Save results
        output_data = {
            "robot_name": self.robot_name,
            "env_name": self.env_name,
            "total_poses": len(results),
            "active_joint_indices": self.active_joint_indices,
            "fixed_joint_indices": self.fixed_joint_indices,
            "angle_step_deg": angle_step_deg,
            "angle_min_deg": angle_min_deg,
            "angle_max_deg": angle_max_deg,
            "arm": arm,
            "poses": results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"COMPLETE: Exported {len(results):,} end effector orientations")
        print(f"Saved to: {output_file}")
        print(f"{'='*60}\n")
        
        # Print summary statistics
        rolls = [r["end_effector"]["orientation"]["roll_deg"] for r in results]
        pitches = [r["end_effector"]["orientation"]["pitch_deg"] for r in results]
        yaws = [r["end_effector"]["orientation"]["yaw_deg"] for r in results]
        
        print("Summary Statistics:")
        print(f"  Roll (deg):   min={min(rolls):.2f}, max={max(rolls):.2f}, mean={np.mean(rolls):.2f}")
        print(f"  Pitch (deg):  min={min(pitches):.2f}, max={max(pitches):.2f}, mean={np.mean(pitches):.2f}")
        print(f"  Yaw (deg):    min={min(yaws):.2f}, max={max(yaws):.2f}, mean={np.mean(yaws):.2f}")
        print()
    
    def close(self):
        """Close the environment."""
        self.env.close()


def main(
    robot: str = "IIWA",
    env: str = "Lift",
    angle_step: float = 90.0,
    angle_min: float = -90.0,
    angle_max: float = 90.0,
    output_file: str = None,
    arm: str = "right",
):
    """
    Export end effector orientations for robot poses.
    
    Args:
        robot: Robot name (IIWA, Panda, Sawyer, etc.)
        env: Environment name (Lift, EmptySpace, etc.)
        angle_step: Angle step size in degrees
        angle_min: Minimum angle in degrees
        angle_max: Maximum angle in degrees
        output_file: Path to output JSON file (default: data/ee_orientations/{robot}.json)
        arm: Which arm to get orientation for ("right" or "left", default: "right")
    
    Examples:
        # Export orientations for IIWA robot
        python export_ee_orientation.py --robot IIWA --angle-step 90
        
        # Export with custom output file
        python export_ee_orientation.py --robot Panda --output-file data/panda_orientations.json
        
        # Export for left arm (if robot has multiple arms)
        python export_ee_orientation.py --robot Baxter --arm left
    """
    
    print("="*60)
    print("END EFFECTOR ORIENTATION EXPORTER")
    print("="*60)
    print(f"Robot: {robot}")
    print(f"Environment: {env}")
    print(f"Angle step: {angle_step}°")
    print(f"Angle range: {angle_min}° to {angle_max}°")
    print(f"Arm: {arm}")
    print("="*60)
    
    # Create exporter
    exporter = EEOrientationExporter(
        robot_name=robot,
        env_name=env,
    )
    
    try:
        exporter.export_orientations(
            angle_step_deg=angle_step,
            angle_min_deg=angle_min,
            angle_max_deg=angle_max,
            output_file=output_file,
            arm=arm,
        )
    
    finally:
        exporter.close()
        print("\nDone!")


if __name__ == "__main__":
    fire.Fire(main)

