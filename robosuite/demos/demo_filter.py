"""
Filter and rank robot poses based on beacon scoring criteria.

This script:
1. Loads pose data from JSONL file
2. Evaluates each pose using beacon_score function
3. Sorts and displays the top-ranked poses

Usage:
    python demo_filter.py --jsonl-path data/poses/Panda/Panda/3d_points/Panda_poses_3d.jsonl
    python demo_filter.py --jsonl-path data/poses/Panda/Panda/3d_points/Panda_poses_3d.jsonl --top-k 5
"""

import numpy as np
import json
import fire
from pathlib import Path


def unit(v):
    """Normalize vector to unit length."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def beacon_score(points, max_z=2.0, min_z=0.0):
    """
    Calculate beacon score for a robot pose based on joint positions.
    
    Args:
        points: Array of shape (N, 3) containing 3D positions of root + joints
        max_z: Maximum expected Z coordinate for normalization
        min_z: Minimum expected Z coordinate for normalization
    
    Returns:
        tuple: (S_total, S_vertical, S_orth, S_front, S_height, S_upward)
            - S_total: Total weighted score
            - S_vertical: Vertical alignment score
            - S_orth: Orthogonality score (forearm-wrist)
            - S_front: Front-facing score
            - S_height: Mid height score (middle height is best)
            - S_upward: Upward wrist score (j5-j6 more upward than j4-j5)
    """
    if len(points) < 7:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    z = np.array([0, 0, 1])
    front = np.array([1, 0, 0])  # 원하는 방향으로 변경

    # 1. vertical alignment
    vertical_scores = []
    for i in range(len(points) - 1):
        v = points[i+1] - points[i]
        v_norm = np.linalg.norm(v)
        if v_norm > 0:
            vertical_scores.append(abs(np.dot(unit(v), z)))
    S_vertical = np.mean(vertical_scores) if vertical_scores else 0.0

    # 2. orthogonality between forearm and wrist
    if len(points) >= 7:
        v_forearm = points[5] - points[4]
        v_wrist = points[6] - points[5]
        
        if np.linalg.norm(v_forearm) > 0 and np.linalg.norm(v_wrist) > 0:
            v_forearm = unit(v_forearm)
            v_wrist = unit(v_wrist)
            cos_angle = np.dot(v_forearm, v_wrist)
            S_orth = 1 - abs(cos_angle)
        else:
            S_orth = 0.0
    else:
        S_orth = 0.0

    # 3. wrist direction front-facing
    if len(points) >= 7:
        v_wrist = points[6] - points[5]
        if np.linalg.norm(v_wrist) > 0:
            S_front = (np.dot(unit(v_wrist), unit(front)) + 1) / 2
        else:
            S_front = 0.0
    else:
        S_front = 0.0
    
    # 4. End-effector height (middle height is best)
    # Last point is the end-effector
    ee_z = points[-1][2]  # Z coordinate of end-effector
    
    # Normalize to 0-1 range, then score based on distance from middle (0.5)
    # Middle height gets highest score (1.0), extremes get lower scores
    if max_z > min_z:
        normalized_z = (ee_z - min_z) / (max_z - min_z)
        normalized_z = np.clip(normalized_z, 0, 1)
        # Distance from middle: 0 at center (0.5), 0.5 at extremes (0 or 1)
        distance_from_middle = abs(normalized_z - 0.5)
        # Convert to score: 1.0 at middle, 0.0 at extremes
        S_height = 1.0 - 2.0 * distance_from_middle
    else:
        S_height = 1.0  # If all heights are the same, give max score

    # 5. Upward wrist orientation (j5-j6 more upward than j4-j5)
    # points[0] is root_body, so:
    # points[5] = joint4, points[6] = joint5, points[7] = joint6 (if 0-indexed including root)
    # Actually, if points = [root_body, joint1, joint2, ..., joint6]
    # then points[4] = joint3, points[5] = joint4, points[6] = joint5, points[7] = joint6
    # Wait, let me think about indexing...
    # If there are 6 joints: root_body + 6 joints = 7 elements
    # points[0] = root_body
    # points[1] = joint1
    # points[2] = joint2
    # points[3] = joint3
    # points[4] = joint4
    # points[5] = joint5
    # points[6] = joint6
    
    if len(points) >= 7:
        v_j4_j5 = points[6] - points[5]  # joint5 - joint4
        v_j5_j6 = points[7] - points[6] if len(points) >= 8 else points[-1] - points[6]  # joint6 - joint5
        
        # Get z components
        z_j4_j5 = v_j4_j5[2]
        z_j5_j6 = v_j5_j6[2]
        
        # Score: higher if z_j5_j6 > z_j4_j5
        # Normalize the difference
        z_diff = z_j5_j6 - z_j4_j5
        
        # Map to 0-1 range using sigmoid-like function
        # If z_diff > 0, score approaches 1; if z_diff < 0, score approaches 0
        # Using tanh for smooth transition: (tanh(z_diff * scale) + 1) / 2
        S_upward = (np.tanh(z_diff * 5) + 1) / 2  # scale=5 for reasonable sensitivity
    else:
        S_upward = 0.5  # neutral score if not enough joints

    # weighted total
    w1, w2, w3, w4, w5 = 0.20, 0.0, 0.0, 0.30, 0.50  # Added weight for upward wrist
    S_total = w1 * S_vertical + w2 * S_orth + w3 * S_front + w4 * S_height + w5 * S_upward

    return S_total, S_vertical, S_orth, S_front, S_height, S_upward


def load_poses_from_jsonl(jsonl_path):
    """
    Load pose data from JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
    
    Returns:
        list: List of pose dictionaries
    """
    poses = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                pose = json.loads(line)
                poses.append(pose)
    return poses


def evaluate_poses(poses):
    """
    Evaluate all poses and calculate scores.
    
    Args:
        poses: List of pose dictionaries
    
    Returns:
        list: List of tuples (pose_id, scores, pose_data)
    """
    results = []
    
    # First pass: find min/max Z for normalization
    all_ee_z = []
    for pose in poses:
        joints = [np.array(j) for j in pose['joint_positions_3d']['joints']]
        if joints:
            ee_z = joints[-1][2]  # Last joint Z coordinate
            all_ee_z.append(ee_z)
    
    if all_ee_z:
        min_z = min(all_ee_z)
        max_z = max(all_ee_z)
    else:
        min_z, max_z = 0.0, 2.0
    
    # Second pass: evaluate all poses
    for pose in poses:
        pose_id = pose['pose_id']
        
        # Get joint positions (combine root_body and joints)
        root_body = np.array(pose['joint_positions_3d']['root_body'])
        joints = [np.array(j) for j in pose['joint_positions_3d']['joints']]
        
        # Combine into single array: [root_body, joint1, joint2, ...]
        points = np.array([root_body] + joints)
        
        # Calculate scores
        scores = beacon_score(points, max_z=max_z, min_z=min_z)
        
        results.append({
            'pose_id': pose_id,
            'scores': scores,
            'S_total': scores[0],
            'S_vertical': scores[1],
            'S_orth': scores[2],
            'S_front': scores[3],
            'S_height': scores[4],
            'S_upward': scores[5],
            'ee_z': points[-1][2],  # End-effector Z coordinate
            'filename': pose['filename'],
            'joint_angles_deg': pose['joint_angles_deg'],
            'points': points
        })
    
    return results


def main(
    jsonl_path: str = "data/poses/Panda/Panda/3d_points/Panda_poses_3d.jsonl",
    top_k: int = 3,
    sort_by: str = "total",  # "total", "vertical", "orth", "front"
    show_details: bool = True
):
    """
    Filter and rank robot poses based on beacon scoring.
    
    Args:
        jsonl_path: Path to JSONL file containing pose data
        top_k: Number of top poses to display (default: 3)
        sort_by: Score type to sort by (default: "total")
        show_details: If True, show detailed scores for each pose
    
    Examples:
        # Basic usage - show top 3 poses
        python demo_filter.py --jsonl-path data/poses/Panda/Panda/3d_points/Panda_poses_3d.jsonl
        
        # Show top 10 poses
        python demo_filter.py --jsonl-path ... --top-k 10
        
        # Sort by vertical alignment only
        python demo_filter.py --jsonl-path ... --sort-by vertical
    """
    
    print("="*60)
    print("ROBOT POSE FILTERING AND RANKING")
    print("="*60)
    print(f"JSONL file: {jsonl_path}")
    print(f"Sorting by: {sort_by}")
    print(f"Top K: {top_k}")
    print("="*60 + "\n")
    
    # Check if file exists
    if not Path(jsonl_path).exists():
        print(f"Error: File not found: {jsonl_path}")
        return
    
    # Load poses
    print("Loading poses from JSONL...")
    poses = load_poses_from_jsonl(jsonl_path)
    print(f"Loaded {len(poses)} poses\n")
    
    # Evaluate all poses
    print("Evaluating poses...")
    results = evaluate_poses(poses)
    print(f"Evaluated {len(results)} poses\n")
    
    # Sort by selected criterion
    sort_key_map = {
        "total": "S_total",
        "vertical": "S_vertical",
        "orth": "S_orth",
        "front": "S_front",
        "height": "S_height",
        "upward": "S_upward"
    }
    sort_key = sort_key_map.get(sort_by, "S_total")
    results_sorted = sorted(results, key=lambda x: x[sort_key], reverse=True)
    
    # Display top K results
    print("="*60)
    print(f"TOP {top_k} POSES (sorted by {sort_by})")
    print("="*60 + "\n")
    
    for rank, result in enumerate(results_sorted[:top_k], 1):
        print(f"Rank {rank}:")
        print(f"  Pose ID: {result['pose_id']}")
        print(f"  Filename: {result['filename']}")
        print(f"  Joint Angles (deg): {result['joint_angles_deg']}")
        print(f"  Total Score: {result['S_total']:.4f}")
        
        if show_details:
            print(f"  Detailed Scores:")
            print(f"    - Vertical Alignment: {result['S_vertical']:.4f}")
            print(f"    - Orthogonality:      {result['S_orth']:.4f}")
            print(f"    - Front-Facing:       {result['S_front']:.4f}")
            print(f"    - Mid Height Score:   {result['S_height']:.4f} (EE Z: {result['ee_z']:.4f})")
            print(f"    - Upward Wrist:       {result['S_upward']:.4f}")
        
        print()
    
    # Print summary statistics
    print("="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    all_total_scores = [r['S_total'] for r in results]
    all_vertical_scores = [r['S_vertical'] for r in results]
    all_orth_scores = [r['S_orth'] for r in results]
    all_front_scores = [r['S_front'] for r in results]
    
    print(f"Total Score:      min={min(all_total_scores):.4f}, "
          f"max={max(all_total_scores):.4f}, "
          f"mean={np.mean(all_total_scores):.4f}")
    print(f"Vertical Score:   min={min(all_vertical_scores):.4f}, "
          f"max={max(all_vertical_scores):.4f}, "
          f"mean={np.mean(all_vertical_scores):.4f}")
    print(f"Orthogonality:    min={min(all_orth_scores):.4f}, "
          f"max={max(all_orth_scores):.4f}, "
          f"mean={np.mean(all_orth_scores):.4f}")
    print(f"Front-Facing:     min={min(all_front_scores):.4f}, "
          f"max={max(all_front_scores):.4f}, "
          f"mean={np.mean(all_front_scores):.4f}")
    print()
    
    # Print top 3 indices (for easy reference)
    print("="*60)
    print(f"TOP {min(top_k, 3)} POSE INDICES (for reference)")
    print("="*60)
    for i, result in enumerate(results_sorted[:min(top_k, 3)], 1):
        print(f"  #{i}: Pose ID {result['pose_id']}")
    print()


if __name__ == "__main__":
    fire.Fire(main)