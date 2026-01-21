"""
Generate Spot robot motions based on cue definitions.

This uses the same motion_config.json as other robots and generates
expressive motions for Spot's arm.

Usage:
    # SpotWithArmFloating (recommended)
    python adhoc/spot/motion_generation_spot.py \
        --robot SpotWithArmFloating \
        --cue waving \
        --controller OSC_POSE
    
    # SpotWithArm (full quadruped)
    python adhoc/spot/motion_generation_spot.py \
        --robot SpotWithArm \
        --cue beckoning \
        --controller OSC_POSE
"""

import fire
import os
import sys
from typing import Optional

# Add robotarm directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robotarm'))

from motion_generation import MotionGenerator


def generate_spot_motion(
    robot: str = "SpotWithArmFloating",
    env: str = "EmptySpace",
    cue: str = "waving",
    pose_index: Optional[int] = None,
    controller: str = "OSC_POSE",
    jsonl_path: str = None,
    config_path: str = "data/seed/motion_config.json",
    proximal_degree_scale: float = 0.25,
    camera_distance: float = 2.2,
    hz: int = 4,
    enable_self_collision_check: bool = False,
):
    """
    Generate Spot robot motion.
    
    Args:
        robot: Robot name ("SpotWithArm" or "SpotWithArmFloating")
        env: Environment name
        cue: Motion cue name (e.g., 'waving', 'beckoning')
        pose_index: Optional pose_id to use
        controller: Controller type (default: OSC_POSE)
        jsonl_path: Path to pose database (default: auto-detected)
        config_path: Path to motion config JSON
        proximal_degree_scale: Scale factor for proximal joints
        camera_distance: Camera FOV multiplier (2.2 = wider view for Spot)
        hz: Frame rate for GIF
        enable_self_collision_check: Enable collision checking
    
    Examples:
        # Basic waving
        python adhoc/spot/motion_generation_spot.py \
            --robot SpotWithArmFloating --cue waving
        
        # Beckoning with specific pose
        python adhoc/spot/motion_generation_spot.py \
            --robot SpotWithArmFloating --cue beckoning --pose-index 123
        
        # Full quadruped Spot
        python adhoc/spot/motion_generation_spot.py \
            --robot SpotWithArm --cue pointing
    """
    # Auto-detect jsonl_path if not provided
    if jsonl_path is None:
        # Try closest poses first
        closest_poses_path = f"data/poses/spot/closest_{robot}_poses.jsonl"
        if os.path.exists(closest_poses_path):
            jsonl_path = closest_poses_path
            print(f"Using pre-queried poses: {jsonl_path}")
        else:
            # Fallback to all poses
            jsonl_path = f"data/poses/spot/all_{robot}_poses.jsonl"
            if not os.path.exists(jsonl_path):
                print(f"Error: Pose database not found: {jsonl_path}")
                print(f"Please run: python adhoc/spot/export_spot_poses.py --robot {robot}")
                return False
            print(f"Using all poses: {jsonl_path}")
    
    # Create motion generator
    generator = MotionGenerator(
        robot_name=robot,
        env_name=env,
        controller_name=controller,
        jsonl_path=jsonl_path,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_distance=camera_distance,
        hz=hz,
    )
    
    try:
        generator.execute_cue(
            cue=cue,
            pose_index=pose_index,
            config_path=config_path,
            proximal_degree_scale=proximal_degree_scale,
            hz=hz,
            enable_self_collision_check=enable_self_collision_check,
        )
    finally:
        generator.close()
    
    return True


if __name__ == "__main__":
    fire.Fire(generate_spot_motion)
