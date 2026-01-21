"""
Meta script to query ALL humanoid poses for all direction/height/pitch combinations.

This pre-generates a complete pose database that motion_generation_humanoid.py can use.

Usage:
    python adhoc/humanoid/meta_query_all_humanoid_poses.py --robot GR1ArmsOnly --active-arm right
    python adhoc/humanoid/meta_query_all_humanoid_poses.py --robot GR1ArmsOnly --active-arm left
"""

import sys
import os
import json
from itertools import product
from pathlib import Path

# Add parent directory to import pose_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robotarm'))
from pose_config import direction_pose_set, pitch_poses, poses, height_map

import fire


def meta_query_all_humanoid_poses(
    robot: str = "GR1ArmsOnly",
    active_arm: str = "right",
    output_jsonl: str = None,
    top_k: int = 30,
):
    """
    Query all humanoid poses for all direction/height/pitch combinations.
    
    This generates a complete pose database by querying:
    - All directions from pose_config (up, down, front, back, left, right)
    - All heights (high, medium, low)
    - All pitch orientations (vertical, horizontal)
    - All roll/yaw combinations per direction
    
    Args:
        robot: Humanoid robot name (GR1ArmsOnly, GR1FixedLowerBody, etc.)
        active_arm: Which arm ("right" or "left")
        output_jsonl: Output JSONL path (default: data/poses/humanoid/closest_{robot}_{arm}_poses.jsonl)
        top_k: Number of top poses per combination
    
    Examples:
        # Generate complete pose database for GR1ArmsOnly right arm
        python adhoc/humanoid/meta_query_all_humanoid_poses.py \
            --robot GR1ArmsOnly --active-arm right
        
        # For left arm
        python adhoc/humanoid/meta_query_all_humanoid_poses.py \
            --robot GR1ArmsOnly --active-arm left
    """
    print("="*80)
    print("META QUERY ALL HUMANOID POSES")
    print("="*80)
    print(f"Robot: {robot}")
    print(f"Active arm: {active_arm}")
    print("="*80)
    
    # Determine output file
    if output_jsonl is None:
        output_jsonl = f"data/poses/humanoid/closest_{robot}_{active_arm}_poses.jsonl"
    
    # Clear output JSONL
    if os.path.exists(output_jsonl):
        os.remove(output_jsonl)
        print(f"Cleared existing output file: {output_jsonl}")
    
    # Create directory
    os.makedirs(os.path.dirname(output_jsonl) if os.path.dirname(output_jsonl) else '.', exist_ok=True)
    
    query_count = 0
    success_count = 0
    
    # Iterate through all combinations
    for pose_name, pose_config in direction_pose_set.items():
        # Get configurations
        height_val = height_map[pose_config['height']]
        direction_name = pose_config['dir']
        ee_pitch_name = pose_config['pitch']
        
        # Get direction poses (roll/yaw combinations)
        direction_poses = poses[direction_name]
        
        # Get pitch values
        pitch_values = pitch_poses[ee_pitch_name]
        
        # Generate all combinations
        for dir_pose, pitch_val in product(direction_poses, pitch_values):
            roll = dir_pose['roll']
            yaw = dir_pose['yaw']
            
            print(f"\n{'='*60}")
            print(f"Querying: {pose_name}")
            print(f"  Robot: {robot} ({active_arm} arm)")
            print(f"  Height: {height_val}")
            print(f"  Roll: {roll}, Pitch: {pitch_val}, Yaw: {yaw}")
            print(f"{'='*60}")
            
            # Build query command
            cmd = f"python adhoc/humanoid/query_humanoid_poses.py"
            cmd += f" --robot {robot}"
            cmd += f" --active-arm {active_arm}"
            cmd += f" --roll {roll} --pitch {pitch_val} --yaw {yaw}"
            cmd += f" --top-k {top_k}"
            if height_val:
                cmd += f" --height {height_val}"
            
            # Temporary output file
            temp_output = f"data/poses/humanoid/temp_query_{robot}_{active_arm}_{query_count}.json"
            cmd += f" --output-file {temp_output}"
            
            # Execute query
            ret = os.system(cmd)
            query_count += 1
            
            if ret == 0 and os.path.exists(temp_output):
                # Read results and append to JSONL
                with open(temp_output, 'r') as f:
                    result = json.load(f)
                
                # Append each pose to JSONL
                with open(output_jsonl, 'a') as f_out:
                    for i, pose in enumerate(result.get("poses", []), 1):
                        # Add metadata for motion_generation compatibility
                        entry = {
                            "robot": robot,
                            "active_arm": active_arm,
                            "pose_name": pose_name,
                            "target_roll_deg": roll,
                            "target_pitch_deg": pitch_val,
                            "target_yaw_deg": yaw,
                            "target_height": height_val,
                            "rank": i,
                            "pose_id": pose["pose_id"],
                            "angles_str": pose["angles_str"],
                            "joint_angles_deg": pose["joint_angles_deg"],
                            "joint_angles_rad": pose["joint_angles_rad"],
                            "active_joint_indices": pose["active_joint_indices"],
                            "joint_names": pose.get("joint_names", []),
                            "orientation": pose["orientation"],
                            "orientation_diff_deg": pose["orientation_diff_deg"],
                            "orientation_diff_rad": pose["orientation_diff_rad"],
                            "root_to_ee_distance": pose["root_to_ee_distance"],
                            "root_position": pose["root_position"],
                            "ee_position": pose["ee_position"],
                            "z_diff": pose["z_diff"],
                            "is_front": pose["is_front"],
                            "arm": active_arm,
                        }
                        f_out.write(json.dumps(entry) + '\n')
                
                # Clean up temp file
                os.remove(temp_output)
                success_count += 1
                print(f"✓ Query successful ({len(result.get('poses', []))} poses)")
            else:
                print(f"✗ Query failed!")
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total queries: {query_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {query_count - success_count}")
    print(f"\nResults saved to: {output_jsonl}")
    
    # Count total poses in JSONL
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r') as f:
            total_poses = sum(1 for line in f if line.strip())
        print(f"Total poses in JSONL: {total_poses}")
    
    print("="*80)
    print("\nNow you can use motion_generation_humanoid.py without --pose-index!")
    print(f"It will automatically find poses from: {output_jsonl}")
    print("="*80)


if __name__ == "__main__":
    fire.Fire(meta_query_all_humanoid_poses)
