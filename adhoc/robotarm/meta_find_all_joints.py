"""
Meta script to generate poses for all roll/pitch/yaw combinations.

Generates poses for:
- Roll: range(-180, 181, 90) = [-180, -90, 0, 90, 180] (5 values)
- Pitch: range(-180, 181, 90) = [-180, -90, 0, 90, 180] (5 values)
- Yaw: range(-180, 181, 90) = [-180, -90, 0, 90, 180] (5 values)
Total: 5 * 5 * 5 = 125 combinations

For each robot, runs find_closest_poses.py for all combinations,
then creates a summary table showing pose counts per combination.
"""

import sys
import os
import json
from itertools import product

# JSONL file path
jsonl_path = "data/poses/closest_poses_results.jsonl"

# Remove existing JSONL file if it exists
if os.path.exists(jsonl_path):
    os.remove(jsonl_path)
    print(f"Removed existing JSONL file: {jsonl_path}")
else:
    print(f"JSONL file does not exist, will create new one: {jsonl_path}")

# Define angle values: range(-180, 181, 90) = [-180, -90, 0, 90, 180]
angle_values = list(range(-180, 181, 90))

# Generate all combinations of (roll, pitch, yaw)
combinations = list(product(angle_values, angle_values, angle_values))
total_combinations = len(combinations)
print(f"\n{'='*80}")
print(f"POSE GENERATION: ALL ROLL/PITCH/YAW COMBINATIONS")
print(f"{'='*80}")
print(f"Total combinations: {total_combinations} (5 * 5 * 5)")
print(f"Angle values: {angle_values}")
print(f"{'='*80}\n")

# Robot list
# robots = ["IIWA", "Panda", "Sawyer", "Kinova3", "Jaco", "UR5e", "XArm7"]
robots = ["IIWA", "Panda"]

# Total number of executions
total_executions = len(robots) * total_combinations
print(f"Total executions: {total_executions} ({len(robots)} robots × {total_combinations} combinations)")
print(f"Starting pose generation...\n")

# Execute find_closest_poses.py for each robot and combination
execution_count = 0
for robot in robots:
    print(f"\n{'='*80}")
    print(f"Processing robot: {robot}")
    print(f"{'='*80}")
    
    for combo_idx, (roll, pitch, yaw) in enumerate(combinations, 1):
        execution_count += 1
        print(f"\n[{execution_count}/{total_executions}] Robot: {robot}, "
              f"Roll: {roll}°, Pitch: {pitch}°, Yaw: {yaw}°")
        
        # Build command
        cmd = (f"python adhoc/robotarm/find_closest_poses.py "
               f"--robot {robot} --roll {roll} --pitch {pitch} --yaw {yaw}")
        
        # Execute command
        exit_code = os.system(cmd)
        
        if exit_code != 0:
            print(f"Warning: Command failed with exit code {exit_code}")
            print(f"Command: {cmd}")

print(f"\n{'='*80}")
print("POSE GENERATION COMPLETE")
print(f"{'='*80}\n")

# Read results and create summary table
print(f"\n{'='*80}")
print("POSE GENERATION SUMMARY")
print(f"{'='*80}")

if not os.path.exists(jsonl_path):
    print(f"Error: JSONL file not found: {jsonl_path}")
    sys.exit(1)

# Load all poses
poses = []
with open(jsonl_path, 'r') as f:
    for line in f:
        if line.strip():
            try:
                poses.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
                continue

if not poses:
    print("No poses found in JSONL file")
    sys.exit(1)

print(f"Total poses loaded: {len(poses)}")

# Get all unique robots
all_robots = sorted(set(p.get('robot') for p in poses if p.get('robot')))
print(f"Robots found: {all_robots}")

# Get all unique (roll, pitch, yaw) combinations from the data
combos = {}
for p in poses:
    roll = p.get('roll_deg')
    pitch = p.get('pitch_deg')
    yaw = p.get('yaw_deg')
    key = (roll, pitch, yaw)
    
    if key not in combos:
        combos[key] = {robot: 0 for robot in all_robots}
    
    robot = p.get('robot')
    if robot in combos[key]:
        combos[key][robot] += 1

# Sort combinations for consistent display
# Sort by roll, then pitch, then yaw
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

# Print summary statistics
print(f"\nUnique (roll, pitch, yaw) combinations found: {len(combo_list)}")
print(f"Expected combinations: {total_combinations}")

# Print table header
# Use wider format for readability
col_width = 12
header = f"{'Roll/Pitch/Yaw':<25}"
for robot in all_robots:
    header += f"{robot:>{col_width}}"
header += f"{'Total':>{col_width}}"
print("\n" + header)
print("-" * (25 + col_width * (len(all_robots) + 1)))

# Print each row
for combo in combo_list:
    roll, pitch, yaw = combo
    row_label = f"({format_angle(roll)}, {format_angle(pitch)}, {format_angle(yaw)})"
    row = f"{row_label:<25}"
    total = 0
    for robot in all_robots:
        count = combos[combo][robot]
        row += f"{count:>{col_width}}"
        total += count
    row += f"{total:>{col_width}}"
    print(row)

# Print totals row
print("-" * (25 + col_width * (len(all_robots) + 1)))
total_row = f"{'TOTAL':<25}"
grand_total = 0
for robot in all_robots:
    robot_total = sum(combos[combo][robot] for combo in combo_list)
    total_row += f"{robot_total:>{col_width}}"
    grand_total += robot_total
total_row += f"{grand_total:>{col_width}}"
print(total_row)

# Print final summary
print(f"\n{'='*80}")
print(f"Total poses generated: {len(poses)}")
print(f"Total unique combinations: {len(combo_list)}")
print(f"Robots processed: {len(all_robots)}")
print(f"{'='*80}\n")

# Additional statistics
print("Statistics per robot:")
print("-" * 80)
for robot in all_robots:
    robot_poses = [p for p in poses if p.get('robot') == robot]
    robot_combos = len(set((p.get('roll_deg'), p.get('pitch_deg'), p.get('yaw_deg')) 
                       for p in robot_poses))
    print(f"  {robot:>10}: {len(robot_poses):>6} poses, {robot_combos:>4} unique combinations")
print("-" * 80)