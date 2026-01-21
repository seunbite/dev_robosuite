"""
Export all possible poses for Spot robot arm.

This script exports poses for SpotWithArm or SpotWithArmFloating robots.
The arm has 6 DOF, similar to other single-arm manipulators.

Usage:
    # SpotWithArm (full quadruped with legs)
    python adhoc/spot/export_spot_poses.py --robot SpotWithArm
    
    # SpotWithArmFloating (floating base, no leg movement)
    python adhoc/spot/export_spot_poses.py --robot SpotWithArmFloating
"""

import fire
import os
import json
import numpy as np
from typing import Optional
from itertools import product
from tqdm import tqdm
from PIL import Image

import robosuite as suite
from robosuite.utils import transform_utils as T


class SpotPoseExporter:
    """Export all possible poses for Spot robot arm."""
    
    def __init__(
        self,
        robot_name: str = "SpotWithArmFloating",
        env_name: str = "EmptySpace",
        controller_name: str = "OSC_POSE",
        camera_distance: float = 2.5,
    ):
        """Initialize the exporter."""
        self.robot_name = robot_name
        self.env_name = env_name
        self.controller_name = controller_name
        
        print(f"Initializing Spot robot: {robot_name}")
        
        if robot_name not in ["SpotWithArm", "SpotWithArmFloating"]:
            raise ValueError(f"Robot {robot_name} not supported. Use 'SpotWithArm' or 'SpotWithArmFloating'")
        
        # Setup environment
        # SpotWithArm/SpotWithArmFloating are single-arm manipulators
        # We don't need to explicitly set controller_configs - it will use default
        options = {
            "env_name": env_name,
            "robots": robot_name,
            "has_renderer": False,
            "has_offscreen_renderer": True,  # Enable for rendering
            "ignore_done": True,
            "use_camera_obs": False,
            "control_freq": 20,
        }
        
        # Create environment
        self.env = suite.make(**options, horizon=1000)
        self.env.reset()
        
        # Get robot
        self.robot = self.env.robots[0]
        
        # Get initial joint positions from robot
        self.initial_joint_pos = self.robot._joint_positions.copy()
        self.num_arm_joints = len(self.initial_joint_pos)
        
        # Check if robot has leg joints
        # For SpotWithArm, legs are defined in robot_model.legs_joints
        self.has_leg_joints = hasattr(self.robot.robot_model, 'legs_joints') and len(self.robot.robot_model.legs_joints) > 0
        
        if self.has_leg_joints:
            # Get leg joint names and their qpos addresses
            self.leg_joint_names = self.robot.robot_model.legs_joints
            self.num_base_joints = len(self.leg_joint_names)
            
            # Get initial leg joint positions from sim
            self.leg_joint_qpos_addrs = [
                self.env.sim.model.get_joint_qpos_addr(joint_name) 
                for joint_name in self.leg_joint_names
            ]
            self.initial_base_joint_pos = np.array([
                self.env.sim.data.qpos[addr] 
                for addr in self.leg_joint_qpos_addrs
            ])
        else:
            self.leg_joint_names = []
            self.leg_joint_qpos_addrs = []
            self.initial_base_joint_pos = np.array([])
            self.num_base_joints = 0
        
        self.num_joints = self.num_arm_joints + self.num_base_joints
        
        # Get actual joint names to determine structure
        print(f"\n{'='*60}")
        print(f"Robot initialized: {robot_name}")
        print(f"Arm joints: {self.num_arm_joints}")
        print(f"Base (leg) joints: {self.num_base_joints}")
        if self.has_leg_joints:
            print(f"Leg joint names: {self.leg_joint_names[:3]}...")  # Show first 3
        print(f"Total joints: {self.num_joints}")
        print(f"Initial arm positions: {self.initial_joint_pos}")
        if self.has_leg_joints:
            print(f"Initial base positions: {self.initial_base_joint_pos}")
        
        # Determine joint ranges for arm and legs
        if "Floating" in robot_name:
            # Floating version: only arm joints (6)
            self.mobile_joint_indices = []
            self.arm_joint_indices = list(range(0, self.num_arm_joints))
            self.leg_joint_indices = []  # No legs
        else:
            # Full quadruped: base has legs, manipulator has arm
            self.arm_joint_indices = list(range(self.num_base_joints, self.num_base_joints + self.num_arm_joints))
            self.leg_joint_indices = list(range(0, self.num_base_joints)) if self.num_base_joints > 0 else []
            self.mobile_joint_indices = []
        
        print(f"\nJoint assignment:")
        print(f"  Arm joint indices: {self.arm_joint_indices}")
        print(f"  Leg joint indices: {self.leg_joint_indices}")
        print(f"  Mobile joint indices: {self.mobile_joint_indices}")
        print(f"{'='*60}\n")
        
        # Adjust camera distance for Spot (larger robot)
        # Move frontview camera further back to see full robot
        frontview_id = self.env.sim.model.camera_name2id("frontview")
        # Original: pos="1.6 0 1.45"
        # Adjust x (distance) and z (height) for better view
        self.env.sim.model.cam_pos[frontview_id] = np.array([camera_distance, 0, 1.5])
        print(f"Adjusted frontview camera position: {self.env.sim.model.cam_pos[frontview_id]}")
        
        # Adjust sideview camera as well
        sideview_id = self.env.sim.model.camera_name2id("sideview")
        # Move sideview camera further for better view
        self.env.sim.model.cam_pos[sideview_id] = np.array([0, camera_distance, 1.5])
        print(f"Adjusted sideview camera position: {self.env.sim.model.cam_pos[sideview_id]}")
        
        # Determine EEF site name
        if hasattr(self.robot, 'eef_site_id'):
            # Get site name from id
            site_id = self.robot.eef_site_id
            if isinstance(site_id, dict):
                site_id = site_id.get('right', site_id)
            self.eef_site_name = self.env.sim.model.site_id2name(site_id)
        else:
            # Fallback to common names
            self.eef_site_name = "gripper0_grip_site"
        
        print(f"End-effector site: {self.eef_site_name}")
    
    def _get_eef_orientation(self):
        """Get end-effector orientation as roll, pitch, yaw in degrees."""
        # Get end-effector name (handle both string and dict)
        eef_name = self.robot.robot_model.eef_name
        if isinstance(eef_name, dict):
            eef_name = eef_name.get('right', eef_name)
        
        # Get end-effector rotation matrix
        eef_rot_mat = T.quat2mat(self.robot.sim.data.get_body_xquat(eef_name))
        
        # Convert to Euler angles (roll, pitch, yaw)
        roll, pitch, yaw = T.mat2euler(eef_rot_mat)
        
        # Convert to degrees
        roll_deg = np.rad2deg(roll)
        pitch_deg = np.rad2deg(pitch)
        yaw_deg = np.rad2deg(yaw)
        
        return roll_deg, pitch_deg, yaw_deg
    
    def _get_eef_position(self):
        """Get end-effector position."""
        return self.robot.sim.data.get_site_xpos(self.eef_site_name)
    
    def _get_root_position(self):
        """Get robot root position."""
        return self.robot.sim.data.get_body_xpos(self.robot.robot_model.root_body)
    
    def _set_joint_angles(self, joint_angles_rad, active_joint_indices):
        """Set joint angles and simulate.
        
        Args:
            joint_angles_rad: Array of joint angles in radians
            active_joint_indices: List of joint indices to set (in global indexing)
        """
        # Create full joint position arrays
        arm_joint_positions = self.initial_joint_pos.copy()
        if self.has_leg_joints:
            base_joint_positions = self.initial_base_joint_pos.copy()
        
        # Set specified joint positions
        for i, joint_idx in enumerate(active_joint_indices):
            if i >= len(joint_angles_rad):
                break
            
            # Check if this is a base joint or arm joint
            if joint_idx < self.num_base_joints:
                # Base (leg) joint
                if self.has_leg_joints and joint_idx < len(base_joint_positions):
                    base_joint_positions[joint_idx] = joint_angles_rad[i]
            else:
                # Arm joint (adjust index to arm space)
                arm_idx = joint_idx - self.num_base_joints
                if arm_idx < len(arm_joint_positions):
                    arm_joint_positions[arm_idx] = joint_angles_rad[i]
        
        # Set base joint positions first (if exists) - directly to sim.data.qpos
        if self.has_leg_joints:
            for joint_idx, qpos_addr in enumerate(self.leg_joint_qpos_addrs):
                if joint_idx < len(base_joint_positions):
                    self.env.sim.data.qpos[qpos_addr] = base_joint_positions[joint_idx]
        
        # Set arm joint positions
        self.robot.set_robot_joint_positions(arm_joint_positions)
        
        # Forward simulation to update kinematics
        self.env.sim.forward()
        
        # Run physics simulation to let gravity and dynamics settle
        # Use step() instead of forward() to include gravity and dynamics
        for _ in range(50):
            # Apply zero control to let the robot settle under gravity
            zero_action = np.zeros(self.env.action_dim)
            self.env.step(zero_action)
        
        # Final stabilization - keep velocities low
        for _ in range(10):
            self.env.sim.data.qvel[:] *= 0.1  # Dampen velocities instead of zeroing
            self.env.sim.forward()
    
    def export_all_poses(
        self,
        output_path: str = None,
        num_angles: tuple = (-90, 90, 3),
        part: str = "arm",
        save_png: bool = False,
        image_width: int = 512,
        image_height: int = 512,
        exclude_ab_ad: bool = True,
    ):
        """
        Export all possible poses by trying all joint angle combinations.
        
        Args:
            output_path: Output JSONL file path
            num_angles: Tuple of (min_angle, max_angle, num_samples) in degrees
                       Example: (-90, 90, 3) means 3 angles from -90° to 90°
            part: Which part to move ("arm", "leg", or "all")
            save_png: Whether to save PNG images for each pose
            image_width: Image width for PNG (if save_png=True)
            image_height: Image height for PNG (if save_png=True)
            exclude_ab_ad: If True, exclude ab/ad (hx) joints from leg movement (default: True)
        """
        # Determine which joints to move
        if part == "arm":
            active_joint_indices = self.arm_joint_indices
            part_name = "arm"
        elif part == "leg":
            if not self.leg_joint_indices:
                raise ValueError(f"Robot {self.robot_name} has no legs (use 'arm' instead)")
            active_joint_indices = self.leg_joint_indices.copy()
            
            # Exclude ab/ad (hx) joints if requested
            # Spot leg joints order: [fr_hx, fr_hy, fr_kn, fl_hx, fl_hy, fl_kn, hr_hx, hr_hy, hr_kn, hl_hx, hl_hy, hl_kn]
            # ab/ad joints are at indices 0, 3, 6, 9 (every 3rd joint starting from 0)
            if exclude_ab_ad:
                # Get the first leg joint index (base offset)
                base_offset = self.leg_joint_indices[0] if self.leg_joint_indices else 0
                # ab/ad joints are at relative positions 0, 3, 6, 9 within leg joints
                ab_ad_indices = [base_offset + i for i in [0, 3, 6, 9]]
                active_joint_indices = [idx for idx in active_joint_indices if idx not in ab_ad_indices]
                print(f"Excluding ab/ad (hx) joints: {ab_ad_indices}")
                part_name = "leg_no_ab_ad"
            else:
                part_name = "leg"
        elif part == "all":
            active_joint_indices = list(range(self.num_joints))
            part_name = "all"
        else:
            raise ValueError(f"Invalid part: {part}. Must be 'arm', 'leg', or 'all'")
        
        num_active_joints = len(active_joint_indices)
        
        if output_path is None:
            output_path = f"data/poses/spot/all_{self.robot_name}_{part_name}_poses.jsonl"
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create PNG output directories if needed
        if save_png:
            png_dir_front = os.path.join(os.path.dirname(output_path), f"{self.robot_name}_{part_name}_images_front")
            png_dir_side = os.path.join(os.path.dirname(output_path), f"{self.robot_name}_{part_name}_images_side")
            os.makedirs(png_dir_front, exist_ok=True)
            os.makedirs(png_dir_side, exist_ok=True)
            print(f"Front view images will be saved to: {png_dir_front}")
            print(f"Side view images will be saved to: {png_dir_side}")
        
        # Parse num_angles tuple (min, max, num_samples)
        if isinstance(num_angles, (list, tuple)):
            angle_min, angle_max, num_angle_samples = num_angles
        else:
            # Fallback for backward compatibility
            angle_min, angle_max, num_angle_samples = -90, 90, num_angles
        
        # Generate angle samples
        angles_deg = np.linspace(angle_min, angle_max, num_angle_samples)
        angles_rad = np.deg2rad(angles_deg)
        
        print(f"\nGenerating poses for: {part_name}")
        print(f"Active joints: {active_joint_indices}")
        print(f"Number of active joints: {num_active_joints}")
        print(f"Angle range: {angle_min}° to {angle_max}° with {num_angle_samples} samples")
        print(f"Angle samples (deg): {angles_deg}")
        print(f"Total combinations: {num_angle_samples ** num_active_joints}")
        
        # Generate all combinations for active joints only
        all_angle_combinations = list(product(angles_rad, repeat=num_active_joints))
        
        print(f"\nExporting {len(all_angle_combinations)} poses...")
        
        # Export poses
        pose_id = 0
        with open(output_path, 'w') as f:
            for angle_combination in tqdm(all_angle_combinations, desc="Exporting poses"):
                # Set joint angles (only for active joints)
                self._set_joint_angles(angle_combination, active_joint_indices)
                
                # Get end-effector orientation and position
                roll_deg, pitch_deg, yaw_deg = self._get_eef_orientation()
                eef_pos = self._get_eef_position()
                root_pos = self._get_root_position()
                
                # Calculate distance from root to end-effector
                root_to_ee_distance = np.linalg.norm(eef_pos - root_pos)
                
                # Check if end-effector is in front of robot (positive y in robot frame)
                is_front = eef_pos[1] > root_pos[1]
                
                # Create pose data
                pose_data = {
                    "pose_id": pose_id,
                    "robot": self.robot_name,
                    "part": part_name,
                    "angles_str": "_".join([f"{int(np.rad2deg(a))}" for a in angle_combination]),
                    "joint_angles_deg": [float(np.rad2deg(a)) for a in angle_combination],
                    "joint_angles_rad": [float(a) for a in angle_combination],
                    "active_joint_indices": active_joint_indices,
                    "joint_names": [f"joint_{i}" for i in active_joint_indices],
                    "orientation": {
                        "roll_deg": float(roll_deg),
                        "pitch_deg": float(pitch_deg),
                        "yaw_deg": float(yaw_deg),
                    },
                    "root_to_ee_distance": float(root_to_ee_distance),
                    "root_position": root_pos.tolist(),
                    "ee_position": eef_pos.tolist(),
                    "z_diff": float(eef_pos[2] - root_pos[2]),
                    "is_front": bool(is_front),
                }
                
                # Write to JSONL
                f.write(json.dumps(pose_data) + '\n')
                
                # Save PNG if requested (both front and side views)
                if save_png:
                    # Front view
                    obs_front = self.env.sim.render(
                        camera_name="frontview",
                        width=image_width,
                        height=image_height,
                        depth=False
                    )
                    img_front = Image.fromarray(obs_front[::-1])
                    png_path_front = os.path.join(png_dir_front, f"pose_{pose_id:06d}.png")
                    img_front.save(png_path_front)
                    
                    # Side view
                    obs_side = self.env.sim.render(
                        camera_name="sideview",
                        width=image_width,
                        height=image_height,
                        depth=False
                    )
                    img_side = Image.fromarray(obs_side[::-1])
                    png_path_side = os.path.join(png_dir_side, f"pose_{pose_id:06d}.png")
                    img_side.save(png_path_side)
                
                pose_id += 1
        
        print(f"\n{'='*60}")
        print(f"Exported {pose_id} poses to: {output_path}")
        print(f"{'='*60}")
    
    def close(self):
        """Close the environment."""
        self.env.close()


def export_spot_poses(
    robot: str = "SpotWithArmFloating",
    env: str = "EmptySpace",
    controller: str = "OSC_POSE",
    output: str = None,
    num_angles: str = "(-30,30,3)",
    part: str = "arm",
    save_png: bool = True,
    image_width: int = 512,
    image_height: int = 512,
    camera_distance: float = 2.5,
    exclude_ab_ad: bool = True,
):
    """
    Export all poses for Spot robot.
    
    Args:
        robot: Robot name ("SpotWithArm" or "SpotWithArmFloating")
        env: Environment name
        controller: Controller name
        output: Output JSONL file path (default: auto-generated)
        num_angles: Angle range as "(min,max,num)" string (default: "(-30,30,3)")
                   Example: "(-90,90,5)" for 5 angles from -90° to 90°
        part: Which part to move ("arm", "leg", or "all")
        save_png: Whether to save PNG images for each pose
        image_width: Image width for PNG (default: 512)
        image_height: Image height for PNG (default: 512)
        camera_distance: Camera distance from origin (default: 2.5m for full robot view)
        exclude_ab_ad: If True, exclude ab/ad (hx) joints from leg movement (default: True)
    
    Examples:
        # Arm only (6 joints, 3^6 = 729 poses, -30° to 30°)
        python adhoc/spot/export_spot_poses.py --robot SpotWithArmFloating --part arm
        
        # Wider angle range (-90° to 90°, 5 samples)
        python adhoc/spot/export_spot_poses.py --robot SpotWithArmFloating --part arm --num-angles="(-90,90,5)"
        
        # Legs only, excluding ab/ad joints (8 joints instead of 12: 3^8 = 6,561 poses)
        python adhoc/spot/export_spot_poses.py --robot SpotWithArm --part leg --num-angles="(-45,45,3)"
        
        # Legs with ab/ad joints included (12 joints: 3^12 = 531,441 poses - not recommended!)
        python adhoc/spot/export_spot_poses.py --robot SpotWithArm --part leg --num-angles="(-30,30,2)" --exclude-ab-ad=False
        
        # All joints (fewer samples recommended!)
        python adhoc/spot/export_spot_poses.py --robot SpotWithArm --part all --num-angles="(-30,30,2)"
        
        # With PNG images
        python adhoc/spot/export_spot_poses.py --robot SpotWithArmFloating --part arm --save-png
    """
    # Parse num_angles string to tuple
    if isinstance(num_angles, str):
        # Remove parentheses and spaces, split by comma
        num_angles = num_angles.strip("()").replace(" ", "")
        num_angles = tuple(map(int, num_angles.split(",")))
    
    exporter = SpotPoseExporter(
        robot_name=robot,
        env_name=env,
        controller_name=controller,
        camera_distance=camera_distance,
    )
    
    try:
        exporter.export_all_poses(
            output_path=output,
            num_angles=num_angles,
            part=part,
            save_png=save_png,
            image_width=image_width,
            image_height=image_height,
            exclude_ab_ad=exclude_ab_ad,
        )
    finally:
        exporter.close()


if __name__ == "__main__":
    fire.Fire(export_spot_poses)
