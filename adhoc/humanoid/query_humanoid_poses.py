"""
Query closest poses for humanoid robots from pre-exported JSONL.
Uses the same interface as query_poses_from_export.py but for humanoid robots.

Usage:
    python adhoc/humanoid/query_humanoid_poses.py --robot GR1ArmsOnly --active-arm right --roll 0 --pitch 90 --yaw 0
    python adhoc/humanoid/query_humanoid_poses.py --robot GR1FixedLowerBody --active-arm left --roll 180 --yaw 0 --height high
"""

import sys
import os

# Add parent directory to path to import query_poses_from_export
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robotarm'))

from query_poses_from_export import query_closest_poses
import fire


def query_humanoid_poses(
    robot: str = "GR1ArmsOnly",
    active_arm: str = "right",
    roll: float = None,
    pitch: float = None,
    yaw: float = None,
    top_k: int = 100,
    height: str = None,
    height_low_bar: float = 33.0,
    height_high_bar: float = 33.0,
    output_file: str = None,
    max_orientation_diff_deg: float = 60.0,
    save_tile_image: bool = False,
    tile_size: int = 256,
    border_width: int = 2,
    tile_output: str = None,
):
    """
    Query closest poses for humanoid robots from pre-exported JSONL.
    
    Args:
        robot: Robot name (GR1ArmsOnly, GR1FixedLowerBody, etc.)
        active_arm: Which arm ("right" or "left")
        roll: Target roll angle in degrees
        pitch: Target pitch angle in degrees
        yaw: Target yaw angle in degrees
        top_k: Number of top poses to return
        height: Filter by height ("high", "medium", "low", or None)
        height_low_bar: Percentage for low threshold
        height_high_bar: Percentage for high threshold
        output_file: Output JSON file
        max_orientation_diff_deg: Maximum orientation difference to consider
        save_tile_image: Whether to save tiled image
        tile_size: Size of each tile in pixels
        border_width: Width of border between tiles
        tile_output: Custom tile image output path
    
    Examples:
        python adhoc/humanoid/query_humanoid_poses.py --robot GR1ArmsOnly --active-arm right --roll 0 --pitch 90 --yaw 0
        python adhoc/humanoid/query_humanoid_poses.py --robot GR1FixedLowerBody --active-arm left --roll 180 --yaw 0 --height high
        python adhoc/humanoid/query_humanoid_poses.py --robot GR1ArmsOnly --active-arm right --roll 0 --pitch 90 --yaw 0 --save-tile-image
    """
    # Construct input file path
    input_file = f"data/poses/humanoid/all_{robot}_{active_arm}_poses.jsonl"
    
    # Call the generic query function
    result = query_closest_poses(
        robot=f"{robot}_{active_arm}",  # Add arm suffix for identification
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        top_k=top_k,
        height=height,
        height_low_bar=height_low_bar,
        height_high_bar=height_high_bar,
        input_file=input_file,
        output_file=output_file,
        max_orientation_diff_deg=max_orientation_diff_deg,
        save_tile_image=save_tile_image,
        tile_size=tile_size,
        border_width=border_width,
        tile_output=tile_output,
    )
    
    return result


if __name__ == "__main__":
    fire.Fire(query_humanoid_poses)
