"""
Query closest poses from exported Spot pose database.

This is similar to query_poses_from_export.py but specifically for Spot robots.

Usage:
    python adhoc/spot/query_spot_poses.py --robot SpotWithArmFloating --roll 0 --pitch 90 --yaw 0
"""

import fire
import os
import sys

# Add robotarm directory to path to use query_poses_from_export
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robotarm'))

from query_poses_from_export import query_poses


def query_spot_poses(
    robot: str = "SpotWithArmFloating",
    roll: float = 0.0,
    pitch: float = 90.0,
    yaw: float = 0.0,
    height: str = None,
    top_k: int = 30,
    input_file: str = None,
    output_file: str = None,
    save_tile_image: bool = False,
    tile_size: int = 256,
    border_width: int = 2,
    tile_output: str = None,
):
    """
    Query closest Spot robot poses.
    
    This is a wrapper around query_poses_from_export with Spot-specific defaults.
    
    Args:
        robot: Robot name
        roll: Target roll angle in degrees
        pitch: Target pitch angle in degrees
        yaw: Target yaw angle in degrees
        height: Optional height filter
        top_k: Number of top poses to return
        input_file: Input JSONL file (default: auto-detected)
        output_file: Output file path (default: auto-generated)
        save_tile_image: Whether to save tile image
        tile_size: Size of each tile
        border_width: Border width between tiles
        tile_output: Tile image output path
    """
    # Auto-detect input file if not provided
    if input_file is None:
        input_file = f"data/poses/spot/all_{robot}_poses.jsonl"
    
    # Call the generic query function
    return query_poses(
        robot=robot,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        height=height,
        top_k=top_k,
        input_file=input_file,
        output_file=output_file,
        save_tile_image=save_tile_image,
        tile_size=tile_size,
        border_width=border_width,
        tile_output=tile_output,
    )


if __name__ == "__main__":
    fire.Fire(query_spot_poses)
