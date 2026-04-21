"""
Efficiently export all possible poses for all robots into a single database.
This script scans the entire joint space of each robot ONCE and saves 
position, orientation, and region info for all combinations.
"""

import os
import json
import numpy as np
from itertools import product
from tqdm import tqdm
import fire

from find_closest_poses import ClosestPoseFinder
from arm_pose_config import poses as dir_configs, pitch_poses

def export_all_poses(
    output_path="data/seed/closest_poses_results.jsonl",
    angle_step=90.0, # 90 deg -> 3^N poses, 45 deg -> 5^N poses
    robots=["IIWA", "Panda", "Sawyer", "Kinova3", "Jaco", "UR5e", "XArm7"]
):
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed existing database: {output_path}")

    for robot_name in robots:
        print(f"\nScanning Pose Space for: {robot_name} (Step: {angle_step} deg)")
        finder = ClosestPoseFinder(robot_name=robot_name)
        
        # 1. Determine search space
        num_independent = len(finder.active_joint_indices)
        # For bimanual/symmetric robots, ClosestPoseFinder already handles reducing indices
        # If we want to force symmetry, we can check if it's Tiago etc.
        if robot_name in ["Tiago", "PandaOmron"]:
            num_independent = num_independent // 2
            
        angle_min, angle_max = np.deg2rad(-90), np.deg2rad(90)
        step_rad = np.deg2rad(angle_step)
        possible_angles = np.arange(angle_min, angle_max + step_rad/2, step_rad)
        
        combinations = list(product(possible_angles, repeat=num_independent))
        print(f"Total combinations to simulate: {len(combinations):,}")

        # 2. First pass: Find min/max bounds for global regions
        all_data = []
        x_vals, y_vals, z_vals = [], [], []
        
        print("Simulating and capturing EE data...")
        for combo_idx, combo in enumerate(tqdm(combinations)):
            joint_pos = finder.initial_joint_pos.copy()
            # Symmetric assignment
            for i, val in enumerate(combo):
                if len(finder.active_joint_indices) > num_independent: # Bimanual
                    joint_pos[finder.active_joint_indices[i]] = val
                    joint_pos[finder.active_joint_indices[i + num_independent]] = val
                else:
                    joint_pos[finder.active_joint_indices[i]] = val
            
            finder._set_joint_positions(joint_pos)
            
            # Get EE state
            ee_pos = finder._get_ee_position(arm="right")
            root_pos = finder._get_root_position()
            rpy = finder._get_ee_orientation_rpy(arm="right")
            
            dx, dy, dz = ee_pos - root_pos
            
            # Construct angles_str for reference
            angles_str = "_".join([f"j{i}{int(np.rad2deg(v)):+04d}" for i, v in enumerate(combo)])

            all_data.append({
                "pose_id": combo_idx,
                "angles_str": angles_str,
                "joint_angles_rad": [float(v) for v in combo],
                "joint_angles_deg": [float(np.rad2deg(v)) for v in combo],
                "active_joint_indices": [int(idx) for idx in finder.active_joint_indices[:num_independent]],
                "x_diff": float(dx), "y_diff": float(dy), "z_diff": float(dz),
                "ee_pos": [float(v) for v in ee_pos],
                "root_pos": [float(v) for v in root_pos],
                "roll_deg": float(np.rad2deg(rpy[0])),
                "pitch_deg": float(np.rad2deg(rpy[1])),
                "yaw_deg": float(np.rad2deg(rpy[2])),
            })
            x_vals.append(dx); y_vals.append(dy); z_vals.append(dz)

        # 3. Second pass: Compute percentiles and save
        for axis in ['x', 'y', 'z']:
            values = np.array([e[f"{axis}_diff"] for e in all_data])
            order = np.argsort(values)
            n = len(values)
            for rank, idx in enumerate(order):
                all_data[idx][f"{axis}_pct"] = int(round(rank / max(n - 1, 1) * 100))

        with open(output_path, 'a') as f:
            for entry in all_data:
                entry["robot"] = robot_name
                entry["orientation"] = {
                    "roll_deg": entry.pop("roll_deg"),
                    "pitch_deg": entry.pop("pitch_deg"),
                    "yaw_deg": entry.pop("yaw_deg")
                }
                
                f.write(json.dumps(entry) + '\n')
        
        finder.close()
        print(f"Successfully exported {len(all_data)} poses for {robot_name}")

    print(f"\nPose database complete: {output_path}")

if __name__ == "__main__":
    fire.Fire(export_all_poses)
