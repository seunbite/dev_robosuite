import numpy as np
import os
import sys
import time
import json
import itertools
from itertools import product
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, List, Dict

# Local robosuite path setup
local_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if local_root not in sys.path:
    sys.path.insert(0, local_root)

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config

# Mobile Robot Configurations
# Mapping torso height and base yaw indices in qpos
MOBILE_ROBOT_CONFIGS = {
    'Tiago': {
        'torso_idx': 3,
        'torso_range': [0.05, 0.35], # Short, Tall
        'yaw_idx': 2,
        'arm_group': 'right',
        'head_indices': [0, 1] # Head Pan, Head Tilt in robot._joint_positions
    },
    'GoogleRobot': { # Proxy using Tiago
        'torso_idx': 3,
        'torso_range': [0.05, 0.35],
        'yaw_idx': 2,
        'arm_group': 'right',
        'robot_type': 'Tiago',
        'head_indices': [0, 1]
    },
    'PandaOmron': {
        'torso_idx': 3,
        'torso_range': [0.05, 0.30],
        'yaw_idx': 2,
        'arm_group': 'right',
        'head_indices': [] # No head joints in this proxy
    },
    'GR1': {
        'torso_idx': 3, 
        'torso_range': [0.0, 0.2],
        'yaw_idx': 2,
        'arm_group': 'right',
        'head_indices': [] # To be updated if GR1 has controllable head
    }
}

class MobilePresetPoseGenerator:
    """Generate and save preset poses for mobile robots with torso and base rotation."""
    
    def __init__(
        self,
        robot_name: str = "Tiago",
        env_name: str = "EmptySpace",
        controller_name: str = "OSC_POSE",
        output_dir: str = "data/poses_mobile",
        camera_fov: float = 60.0,
        capture_width: int = 1024,
        capture_height: int = 1024,
        save_local: bool = True,
    ):
        self.config_key = robot_name
        self.config = MOBILE_ROBOT_CONFIGS.get(self.config_key, MOBILE_ROBOT_CONFIGS['Tiago'])
        
        # Actual robot name to use in suite.make
        self.actual_robot_name = self.config.get('robot_type', robot_name)
        
        self.robot_name = robot_name # Original name for output
        self.env_name = env_name
        self.output_dir = os.path.join(output_dir, robot_name)
        self.save_local = save_local
        
        if self.save_local:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Local save enabled: {self.output_dir}")

        print(f"Initializing mobile robot: {robot_name} (Using model: {self.actual_robot_name})")
        
        # Setup environment
        options = {
            "env_name": env_name,
            "robots": self.actual_robot_name,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": True,
            "use_camera_obs": True,
            "camera_names": "frontview",
            "camera_heights": capture_height,
            "camera_widths": capture_width,
            "control_freq": 20,
        }
        
        # Load controller config
        arm_controller_config = suite.load_part_controller_config(default_controller=controller_name)
        options["controller_configs"] = refactor_composite_controller_config(
            arm_controller_config, self.actual_robot_name, ["right", "left"]
        )
        
        self.env = suite.make(**options, horizon=1000)
        self.env.reset()
        
        # Set camera FOV
        camera_name = "frontview"
        try:
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            self.env.sim.model.cam_fovy[cam_id] = camera_fov
        except Exception as e:
            print(f"Warning: Could not set camera FOV: {e}")
            
        self.robot = self.env.robots[0]
        
        # Identify joints
        self.active_joint_indices = []
        try:
            self.initial_joint_pos = self.robot._joint_positions.copy()
            self.num_joints = len(self.initial_joint_pos)
            
            all_joint_names = self.robot.robot_model.joints
            arm_group = self.config.get('arm_group', 'right')
            
            self.arm_joint_names = []
            for i, name in enumerate(all_joint_names):
                if i >= self.num_joints: break
                
                is_arm = False
                if self.actual_robot_name == "Tiago":
                    if "arm_" + arm_group in name:
                        is_arm = True
                elif self.actual_robot_name == "PandaOmron":
                    if "joint" in name:
                        is_arm = True
                
                if is_arm:
                    self.active_joint_indices.append(i)
                    self.arm_joint_names.append(name)

            if not self.active_joint_indices:
                self.active_joint_indices = list(range(self.num_joints))
        except Exception as e:
            print(f"Error identifying joints: {e}")
            self.initial_joint_pos = self.robot._joint_positions.copy()
            self.active_joint_indices = list(range(self.num_joints))

        # Explicitly set head indices from config
        self.head_joint_indices = self.config.get('head_indices', [])
        
        print(f"Identified {len(self.active_joint_indices)} arm joints.")
        print(f"Head joint indices: {self.head_joint_indices}")

    def _set_state(self, yaw_deg, height, arm_angles, head_angles=None):
        """Set robot state including base yaw, torso height, arm, and head joints."""
        # 1. Base Yaw
        self.env.sim.data.qpos[self.config['yaw_idx']] = np.deg2rad(yaw_deg)
        
        # 2. Torso Height
        self.env.sim.data.qpos[self.config['torso_idx']] = height
        
        # 3. Apply joint positions
        joint_pos = self.initial_joint_pos.copy()
        
        # Set arm joints
        for i, idx in enumerate(self.active_joint_indices):
            joint_pos[idx] = arm_angles[i]
            
        # Set head joints if provided
        if head_angles is not None and self.head_joint_indices:
            for i, idx in enumerate(self.head_joint_indices):
                if i < len(head_angles):
                    joint_pos[idx] = head_angles[i]
                    
        self.robot.set_robot_joint_positions(joint_pos)
        self.env.sim.forward()

    def generate_poses(
        self,
        angle_step_deg: float = 90.0,
        angle_min_deg: float = -90.0,
        angle_max_deg: float = 90.0,
        include_yaw: bool = False,
        head_only: bool = False,
    ):
        if head_only:
            print("\n--- Generating HEAD ONLY poses (9 combinations) ---")
            yaw_values = [0]
            height_values = [self.config['torso_range'][-1]] # Just use tall height
            arm_combinations = [ [self.initial_joint_pos[i] for i in self.active_joint_indices] ] # Fixed arms
            
            # Head combinations: Pan (-15, 0, 15), Tilt (-30, 0, 30)
            hp_range = np.deg2rad([-15, 0, 15])
            ht_range = np.deg2rad([-30, 0, 30])
            head_combinations = list(product(hp_range, ht_range))
        else:
            yaw_values = [0, 90, 180, 270] if include_yaw else [0]
            # Fixed height (just use tall) instead of brute-forcing tall/short
            height_values = [self.config['torso_range'][-1]]
            angle_range = np.deg2rad(np.arange(angle_min_deg, angle_max_deg + angle_step_deg/2, angle_step_deg))
            arm_combinations = list(product(angle_range, repeat=len(self.active_joint_indices)))
            head_combinations = [None] # No head variation

        total = len(yaw_values) * len(height_values) * len(arm_combinations) * len(head_combinations)
        print(f"Total combinations: {total:,}")
        
        count = 0
        pbar = tqdm(total=total)
        
        for yaw in yaw_values:
            for height in height_values:
                h_label = "tall" if height == max(height_values) else "short"
                for arm_angles in arm_combinations:
                    for head_angles in head_combinations:
                        self._set_state(yaw, height, arm_angles, head_angles)
                        
                        obs = self.env.sim.render(camera_name="frontview", width=1024, height=1024)
                        img = Image.fromarray(np.flipud(obs))
                        
                        # Filename generation
                        if head_only:
                            head_str = f"hp{int(np.rad2deg(head_angles[0])):+04d}_ht{int(np.rad2deg(head_angles[1])):+04d}"
                            filename = f"{self.robot_name}_HEAD_{head_str}.png"
                        else:
                            arm_str = "_".join([f"j{idx}{int(np.rad2deg(ang)):+04d}" for idx, ang in zip(self.active_joint_indices, arm_angles)])
                            filename = f"{self.robot_name}_y{yaw}_h{h_label}_{arm_str}.png"
                        
                        if self.save_local:
                            img.save(os.path.join(self.output_dir, filename))
                        
                        count += 1
                        pbar.update(1)
                        if count % 500 == 0: self.env.reset()
        
        pbar.close()
        print(f"Finished! Generated {count} images.")

    def close(self):
        self.env.close()

def main(robot="Tiago", angle_step=90.0, include_yaw=False, head_only=False):
    generator = MobilePresetPoseGenerator(robot_name=robot)
    try:
        generator.generate_poses(angle_step_deg=angle_step, include_yaw=include_yaw, head_only=head_only)
    finally:
        generator.close()

if __name__ == "__main__":
    import fire
    fire.Fire(main)
