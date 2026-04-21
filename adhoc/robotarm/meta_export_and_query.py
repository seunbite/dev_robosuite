"""
Meta script to:
1. Export all poses for multiple robots (once)
2. Query all desired orientations from pre-computed data

This is MUCH faster than the old approach of recalculating poses for each query.
"""

from arm_pose_config import direction_pose_set, pitch_poses, poses, height_map
import sys
import os
import json
from itertools import product
from pathlib import Path
import argparse

# Configuration
robots = ["IIWA", "Panda", "Sawyer", "Kinova3", "Jaco", "UR5e", "XArm7"]
output_jsonl = "data/poses/closest_poses_results.jsonl"

# Parse arguments
parser = argparse.ArgumentParser(description="Meta script for exporting and querying poses")
parser.add_argument("--save-tile-images", action="store_true", help="Save tile images for each query")
args = parser.parse_args()

print("="*80)
print("META EXPORT AND QUERY")
print("="*80)

# Step 1: Export all poses for each robot (if not already done)
print("\nSTEP 1: Exporting all poses for each robot")
print("-"*80)

for robot in robots:
    export_file = f"data/poses/all_{robot}_poses.jsonl"
    
    if Path(export_file).exists():
        print(f"✓ {robot}: Already exported to {export_file}")
    else:
        print(f"⚙ {robot}: Exporting poses...")
        cmd = f"python adhoc/robotarm/export_all_poses_once.py --robot {robot}"
        ret = os.system(cmd)
        if ret == 0:
            print(f"✓ {robot}: Export complete")
        else:
            print(f"✗ {robot}: Export failed!")
            sys.exit(1)

print("\n" + "="*80)
print("All poses exported successfully!")
print("="*80)

# Step 2: Query all desired orientations
print("\nSTEP 2: Querying closest poses for all target orientations")
print("-"*80)

# Clear output JSONL
if os.path.exists(output_jsonl):
    os.remove(output_jsonl)
    print(f"Cleared existing output file: {output_jsonl}")

query_count = 0
success_count = 0

for robot in robots:
    for pose_name, pose_config in direction_pose_set.items():
        # Get configurations
        height_val = height_map[pose_config['height']]
        direction_name = pose_config['dir']
        ee_pitch_name = pose_config['gripper_orientation']
        
        # Get direction poses (roll/yaw combinations)
        direction_poses = poses[direction_name]
        
        # Get gripper_orientation values
        pitch_values = pitch_poses[ee_pitch_name]
        
        # Generate all combinations
        for dir_pose, pitch_val in product(direction_poses, pitch_values):
            roll = dir_pose['roll']
            yaw = dir_pose['yaw']
            
            print(f"\n{'='*60}")
            print(f"Querying: {pose_name}")
            print(f"  Robot: {robot}")
            print(f"  Height: {height_val}")
            print(f"  Roll: {roll}, Pitch: {pitch_val}, Yaw: {yaw}")
            print(f"{'='*60}")
            
            # Build query command
            cmd = f"python adhoc/robotarm/query_poses_from_export.py --robot {robot}"
            cmd += f" --roll {roll} --gripper_orientation {pitch_val} --yaw {yaw}"
            if height_val:
                cmd += f" --height {height_val}"
            
            # Add tile image flag if requested
            if args.save_tile_images:
                cmd += " --save-tile-image"
            
            # Temporary output file
            temp_output = f"data/poses/temp_query_{robot}_{query_count}.json"
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
                        entry = {
                            "robot": robot,
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
