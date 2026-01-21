from arm_pose_config import direction_pose_set, pitch_poses, poses, height_map

import sys
import os
import json
from itertools import product

jsonl_path = "data/poses/closest_poses_results.jsonl"

if os.path.exists(jsonl_path):
    os.remove(jsonl_path)
else:
    print("File does not exist, will create new one")

robots = ["IIWA", "Panda", "Sawyer", "Kinova3", "Jaco", "UR5e", "XArm7"]

# Generate all combinations from direction_pose_set
total_combinations = 0

for robot in robots:
    for pose_name, pose_config in direction_pose_set.items():
        # Get configurations
        height_val = height_map[pose_config['height']]  # Single height value
        direction_name = pose_config['dir']  # e.g., 'up', 'front', 'down'
        ee_pitch_name = pose_config['pitch']  # e.g., 'vertical', 'horizontal'
        
        # Get direction poses (roll/yaw combinations)
        direction_poses = poses[direction_name]  # List of {'roll': ..., 'yaw': ...}
        
        # Get pitch values
        pitch_values = pitch_poses[ee_pitch_name]  # List of pitch values: [0, 180] or [90, -90]
        
        # Generate all combinations: direction x pitch
        for dir_pose, pitch_val in product(direction_poses, pitch_values):
            roll = dir_pose['roll']
            yaw = dir_pose['yaw']
            
            print(f"\n{'='*60}")
            print(f"Generating: {pose_name}")
            print(f"  Robot: {robot}")
            print(f"  Height: {height_val}")
            print(f"  Roll: {roll}, Pitch: {pitch_val}, Yaw: {yaw}")
            print(f"{'='*60}")
            
            # Build command
            cmd = f"python adhoc/robotarm/find_closest_poses.py --robot {robot} --roll {roll} --pitch {pitch_val} --yaw {yaw}"
            if height_val:
                cmd += f" --height {height_val}"
            
            # Execute
            os.system(cmd)
            total_combinations += 1

print(f"\nTotal combinations generated: {total_combinations}")

# Read results and create summary table
print("\n" + "="*80)
print("POSE GENERATION SUMMARY")
print("="*80)

if not os.path.exists(jsonl_path):
    print(f"Error: JSONL file not found: {jsonl_path}")
    sys.exit(1)

# Load all poses
pose_results = []
with open(jsonl_path, 'r') as f:
    for line in f:
        if line.strip():
            pose_results.append(json.loads(line))

if not pose_results:
    print("No poses found in JSONL file")
    sys.exit(1)

# Get all unique robots
all_robots = sorted(set(p.get('robot') for p in pose_results))

# Get all unique (roll, pitch, yaw) combinations
combos = {}
for p in pose_results:
    key = (p.get('roll_deg'), p.get('pitch_deg'), p.get('yaw_deg'))
    if key not in combos:
        combos[key] = {robot: 0 for robot in all_robots}
    combos[key][p.get('robot')] += 1

# Sort combinations for consistent display
def sort_key(x):
    roll, pitch, yaw = x
    return (
        roll if roll is not None else 999,
        pitch if pitch is not None else 999,
        yaw if yaw is not None else 999
    )
combo_list = sorted(combos.keys(), key=sort_key)

# Format angle for display
def format_angle(angle):
    if angle is None:
        return "None"
    return str(int(angle))

# Print table header
header = f"{'Roll/Pitch/Yaw':<20}"
for robot in all_robots:
    header += f"{robot:>10}"
header += f"{'Total':>10}"
print(header)
print("-" * (20 + 10 * (len(all_robots) + 1)))

# Print each row
for combo in combo_list:
    roll, pitch, yaw = combo
    row_label = f"({format_angle(roll)}, {format_angle(pitch)}, {format_angle(yaw)})"
    row = f"{row_label:<20}"
    total = 0
    for robot in all_robots:
        count = combos[combo][robot]
        row += f"{count:>10}"
        total += count
    row += f"{total:>10}"
    print(row)

# Print totals row
print("-" * (20 + 10 * (len(all_robots) + 1)))
total_row = f"{'TOTAL':<20}"
grand_total = 0
for robot in all_robots:
    robot_total = sum(combos[combo][robot] for combo in combo_list)
    total_row += f"{robot_total:>10}"
    grand_total += robot_total
total_row += f"{grand_total:>10}"
print(total_row)

print("="*80)
print(f"Total poses generated: {len(pose_results)}")
print("="*80)
