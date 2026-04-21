import json
import os
import fire
from typing import Optional
from PIL import Image
import numpy as np

# Try to import MotionGenerator for rendering
try:
    from motion_generation import MotionGenerator
except ImportError:
    MotionGenerator = None

def find_pose(
    robot: str,
    dir: Optional[str] = None,
    gripper_orientation: Optional[str] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
    z: Optional[str] = None,
    jsonl_path: str = "data/seed/closest_poses_results.jsonl",
    show_png: bool = False,
    limit: int = 5
):
    """
    Find poses in the database matching specific criteria.
    
    Args:
        robot: Name of the robot (e.g., 'IIWA', 'Panda')
        dir: Direction label (e.g., 'front', 'left', 'right')
        gripper_orientation: Pitch label ('h' for horizontal, 'v' for vertical)
        x: X-region label (e.g., 'low', 'medium', 'high')
        y: Y-region label (e.g., 'low', 'medium', 'high')
        z: Z-region label (e.g., 'low', 'medium', 'high')
        jsonl_path: Path to the pose database
        show_png: Whether to render and show the matching poses
        limit: Maximum number of poses to show if show_png is True
    """
    if not os.path.exists(jsonl_path):
        print(f"Error: Database file not found at {jsonl_path}")
        return

    # Map gripper_orientation aliases
    pitch_map = {'h': 'horizontal', 'v': 'vertical'}
    target_pitch = pitch_map.get(gripper_orientation.lower(), gripper_orientation) if gripper_orientation else None

    # Load and filter database
    matches = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            pose = json.loads(line)
            
            # Filter by robot
            if pose.get("robot") != robot:
                continue
            
            # Filter by dir
            if dir is not None and pose.get("dir") != dir:
                continue

            # Filter by gripper_orientation
            if target_pitch is not None and pose.get("gripper_orientation") != target_pitch:
                continue
            
            # Filter by regions (x, y, z)
            if x is not None and pose.get("x_region") != x:
                continue
            if y is not None and pose.get("y_region") != y:
                continue
            if z is not None and pose.get("z_region") != z:
                continue
            
            matches.append(pose)

    print(f"\n{'='*60}")
    print(f"Search Criteria:")
    print(f"  Robot: {robot}")
    print(f"  Dir:   {dir if dir else 'Any'}")
    print(f"  Pitch: {target_pitch if target_pitch else 'Any'}")
    print(f"  X:     {x if x else 'Any'}")
    print(f"  Y:     {y if y else 'Any'}")
    print(f"  Z:     {z if z else 'Any'}")
    print(f"{'='*60}")
    print(f"Found {len(matches)} matching poses.")
    print(f"{'='*60}\n")

    if not matches:
        return

    # Display some info about matches
    for i, m in enumerate(matches[:limit]):
        print(f"Match {i+1}: Pose ID {m.get('pose_id')}")
        print(f"  Orientation: dir={m.get('dir')}, gripper_orientation={m.get('gripper_orientation')}")
        print(f"  Regions:     x={m.get('x_region')}, y={m.get('y_region')}, z={m.get('z_region')}")
        print(f"  Angles (deg): {m.get('joint_angles_deg')[:5]}...") # Show first 5 angles
        print("-" * 30)

    if show_png:
        if MotionGenerator is None:
            print("Error: Could not import MotionGenerator from motion_generation.py. Rendering unavailable.")
            return
        
        print(f"\nRendering top {min(len(matches), limit)} poses...")
        generator = MotionGenerator(robot_name=robot, jsonl_path=jsonl_path)
        
        try:
            for i, m in enumerate(matches[:limit]):
                print(f"Rendering Pose ID {m.get('pose_id')}...")
                generator.jacobian_calculator._set_pose_from_data(m)
                img_array = generator._capture_image()
                img = Image.fromarray(img_array)
                img.show(title=f"Robot: {robot} | Pose ID: {m.get('pose_id')}")
        finally:
            generator.close()

if __name__ == "__main__":
    fire.Fire(find_pose)
