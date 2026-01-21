"""
Filter dog poses from brute force results based on dog_pose_config requirements.

This script:
1. Loads dog_pose_set from config/dog_pose_config.py
2. Searches through generated poses (from stack_preset_dog.py)
3. Finds matching poses for each pose definition
4. Creates a filtered JSONL with closest matches

Usage:
    # Filter poses for all robots
    python adhoc/spot/meta_find_closest_poses_dog.py
    
    # Filter poses for specific robot
    python adhoc/spot/meta_find_closest_poses_dog.py --robot Go2
"""

import fire
import os
import sys
import json
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from config.dog_pose_config import dog_pose_set, height_map


def load_pose_database(jsonl_path: str) -> List[Dict]:
    """Load pose database from JSONL file."""
    if not os.path.exists(jsonl_path):
        print(f"Error: JSONL file not found: {jsonl_path}")
        return []
    
    poses = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                poses.append(json.loads(line))
    
    return poses


def find_matching_poses(
    pose_def: Dict,
    pose_database: List[Dict],
    robot_name: str,
) -> List[Dict]:
    """
    Find poses matching the given definition.
    
    Args:
        pose_def: Pose definition dict with body_height, body_tilt, leg_* fields
        pose_database: List of pose data dicts
        robot_name: Robot name to filter by
    
    Returns:
        List of matching pose dicts
    """
    # Extract required features
    body_height = pose_def.get('body_height')
    body_tilt = pose_def.get('body_tilt')
    leg_FL = pose_def.get('leg_FL')
    leg_FR = pose_def.get('leg_FR')
    leg_HL = pose_def.get('leg_HL')
    leg_HR = pose_def.get('leg_HR')
    
    matching_poses = []
    
    for pose_data in pose_database:
        if pose_data.get('robot_name') != robot_name:
            continue
        
        pose_features = pose_data.get('pose_features', {})
        
        # Check if all features match
        match = True
        if body_height and pose_features.get('body_height') != body_height:
            match = False
        if body_tilt and pose_features.get('body_tilt') != body_tilt:
            match = False
        if leg_FL and pose_features.get('leg_FL') != leg_FL:
            match = False
        if leg_FR and pose_features.get('leg_FR') != leg_FR:
            match = False
        if leg_HL and pose_features.get('leg_HL') != leg_HL:
            match = False
        if leg_HR and pose_features.get('leg_HR') != leg_HR:
            match = False
        
        if match:
            matching_poses.append(pose_data)
    
    return matching_poses


def main(
    robots: List[str] = None,
    input_dir: str = "data/poses/quadruped",
    output_file: str = "data/poses/quadruped/closest_dog_poses.jsonl",
):
    """
    Filter dog poses based on dog_pose_config requirements.
    
    Args:
        robots: List of robot names (None = all subdirectories in input_dir)
        input_dir: Directory containing pose JSONL files
        output_file: Output JSONL file for filtered poses
    
    Examples:
        # Process all robots in input_dir
        python adhoc/spot/meta_find_closest_poses_dog.py
        
        # Process specific robots
        python adhoc/spot/meta_find_closest_poses_dog.py --robots Go2 SpotWithArm
    """
    print("="*60)
    print("DOG POSE FILTERING")
    print("="*60)
    
    # Discover robots if not specified
    if robots is None:
        robots = []
        if os.path.exists(input_dir):
            for item in os.listdir(input_dir):
                item_path = os.path.join(input_dir, item)
                if os.path.isdir(item_path):
                    jsonl_path = os.path.join(item_path, f"{item}_dog_poses.jsonl")
                    if os.path.exists(jsonl_path):
                        robots.append(item)
        
        if not robots:
            print(f"Error: No robot pose databases found in {input_dir}")
            print(f"Please run: python adhoc/spot/stack_preset_dog.py --robot <ROBOT_NAME>")
            return
    
    print(f"Robots to process: {robots}")
    print(f"Pose definitions: {len(dog_pose_set)}")
    print(f"Output: {output_file}")
    print("="*60 + "\n")
    
    # Clear output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)
    
    total_matches = 0
    
    # Process each robot
    for robot in robots:
        print(f"\n{'='*60}")
        print(f"Processing robot: {robot}")
        print(f"{'='*60}")
        
        # Load pose database
        jsonl_path = os.path.join(input_dir, robot, f"{robot}_dog_poses.jsonl")
        pose_database = load_pose_database(jsonl_path)
        
        if not pose_database:
            print(f"Warning: No poses found for {robot}")
            continue
        
        print(f"Loaded {len(pose_database)} poses from {jsonl_path}")
        
        # Process each pose definition
        robot_matches = 0
        for pose_name, pose_def in dog_pose_set.items():
            matching_poses = find_matching_poses(pose_def, pose_database, robot)
            
            if matching_poses:
                # Take the first match (all should be identical for exact matches)
                best_pose = matching_poses[0]
                
                # Add pose name and save
                output_entry = best_pose.copy()
                output_entry['pose_name'] = pose_name
                output_entry['pose_definition'] = pose_def
                output_entry['num_matches'] = len(matching_poses)
                
                with open(output_file, 'a') as f:
                    f.write(json.dumps(output_entry) + '\n')
                
                robot_matches += 1
                total_matches += 1
                
                print(f"  ✓ {pose_name}: {len(matching_poses)} matches")
            else:
                print(f"  ✗ {pose_name}: No matches found")
        
        print(f"\n{robot}: Found {robot_matches}/{len(dog_pose_set)} poses")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Total filtered poses: {total_matches}")
    print(f"Output file: {output_file}")
    print(f"{'='*60}\n")
    
    # Print summary table
    print("\nSummary by pose:")
    print("-" * 60)
    
    # Load filtered results
    filtered_poses = []
    with open(output_file, 'r') as f:
        for line in f:
            if line.strip():
                filtered_poses.append(json.loads(line))
    
    # Count by pose name
    pose_counts = {}
    for entry in filtered_poses:
        pose_name = entry['pose_name']
        robot = entry['robot_name']
        if pose_name not in pose_counts:
            pose_counts[pose_name] = {}
        pose_counts[pose_name][robot] = 1
    
    # Print table
    print(f"{'Pose Name':<25} {' '.join([f'{r:>10}' for r in robots])} {'Total':>10}")
    print("-" * 60)
    
    for pose_name in sorted(dog_pose_set.keys()):
        row = f"{pose_name:<25}"
        total = 0
        for robot in robots:
            count = pose_counts.get(pose_name, {}).get(robot, 0)
            row += f"{count:>10}"
            total += count
        row += f"{total:>10}"
        print(row)
    
    print("-" * 60)
    print(f"{'TOTAL':<25} {' '.join([f'{sum(pose_counts.get(p, {}).get(r, 0) for p in dog_pose_set.keys()):>10}' for r in robots])} {total_matches:>10}")
    print("="*60)


if __name__ == "__main__":
    fire.Fire(main)
