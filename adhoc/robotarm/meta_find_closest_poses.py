import sys
import os
import json
from tqdm import tqdm
import fire

# Path to the shared pose database
jsonl_path = "data/seed/closest_poses_results.jsonl"

def main(
    reset: bool = False, 
    angle_step: float = 90.0, 
    robots: list = None
):
    # 1. Option to reset the database
    if reset and os.path.exists(jsonl_path):
        os.remove(jsonl_path)
        print(f"Removed {jsonl_path}")

    # 2. List of robots to process
    if robots is None:
        robots = ["IIWA", "Panda", "Sawyer", "Kinova3", "Jaco", "UR5e", "XArm7"]
    
    print(f"\nStarting Global Brute Force for {len(robots)} robots...")
    
    for robot in robots:
        print(f"\n{'='*60}")
        print(f"PROCESS ROBOT: {robot}")
        print(f"{'='*60}")
        
        # We call the script with --brute_force True
        # This will iterate through 3^N combinations and classify them by orientation and region.
        # It's much faster than calling the script 100 times per robot.
        cmd = f"python adhoc/robotarm/find_closest_poses.py --robot {robot} --brute_force True --angle_step {angle_step} --stack_jsonl_path {jsonl_path}"
        
        ret = os.system(cmd)
        if ret != 0:
            print(f"Warning: Failed to process {robot}")

    # 3. Print Summary
    print_summary(jsonl_path)


def print_summary(path=None):
    """Print a detailed summary table of the pose database."""
    if path is None:
        path = jsonl_path

    if not os.path.exists(path):
        print(f"Error: JSONL file not found: {path}")
        return

    import numpy as np

    poses = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                poses.append(json.loads(line))

    all_robots = sorted(set(p.get('robot', '?') for p in poses))
    all_dirs = ['up', 'down', 'front', 'back', 'left', 'right']
    all_grips = ['vertical', 'horizontal']

    # ── Table 1: Direction × Gripper Orientation ──
    print("\n" + "=" * 80)
    print("POSE DATABASE SUMMARY")
    print("=" * 80)

    header_cols = []
    for d in all_dirs:
        for g in all_grips:
            header_cols.append(f"{d[:2]}_{g[0]}")

    col_w = 7
    robot_w = 10
    header = f"{'Robot':<{robot_w}}" + "".join(f"{c:>{col_w}}" for c in header_cols) + f"{'TOTAL':>{col_w}}"
    print(f"\n[Direction × Gripper Orientation]")
    print(header)
    print("-" * len(header))

    for robot in all_robots:
        rp = [p for p in poses if p.get('robot') == robot]
        cells = []
        for d in all_dirs:
            for g in all_grips:
                cnt = sum(1 for p in rp if p.get('dir') == d and p.get('gripper_orientation') == g)
                cells.append(cnt)
        row = f"{robot:<{robot_w}}" + "".join(f"{c:>{col_w}}" for c in cells) + f"{sum(cells):>{col_w}}"
        print(row)

    # ── Table 2: Percentile distribution per axis per robot ──
    for axis in ['x', 'y', 'z']:
        pct_key = f"{axis}_pct"
        print(f"\n[{axis.upper()} Percentile Stats by Robot × Direction]")
        stat_header = f"{'Robot':<{robot_w}}{'Dir':<8}{'count':>7}{'min':>6}{'p25':>6}{'p50':>6}{'p75':>6}{'max':>6}"
        print(stat_header)
        print("-" * len(stat_header))
        for robot in all_robots:
            for d in all_dirs:
                rp = [p for p in poses if p.get('robot') == robot and p.get('dir') == d]
                vals = [p.get(pct_key, 0) for p in rp if pct_key in p]
                if not vals:
                    continue
                arr = np.array(vals)
                print(f"{robot:<{robot_w}}{d:<8}{len(vals):>7}{int(arr.min()):>6}{int(np.percentile(arr, 25)):>6}{int(np.percentile(arr, 50)):>6}{int(np.percentile(arr, 75)):>6}{int(arr.max()):>6}")

    print("\n" + "=" * 80)
    print("Done!")

if __name__ == "__main__":
    fire.Fire(main)
