"""
Query closest poses from pre-exported JSONL file.
Much faster than recalculating poses every time.

Usage:
    python adhoc/robotarm/query_poses_from_export.py --robot IIWA --roll 0 --gripper_orientation 90 --yaw 0
    python adhoc/robotarm/query_poses_from_export.py --robot Panda --roll 180 --yaw 0 --height high
"""

import fire
import json
import numpy as np
import os
import math
from typing import Optional, List, Dict
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


def load_poses_from_jsonl(jsonl_path: str) -> List[Dict]:
    """Load all poses from JSONL file."""
    poses = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                poses.append(json.loads(line))
    return poses


def calculate_orientation_diff(
    pose_roll: float,
    pose_pitch: float,
    pose_yaw: float,
    target_roll: Optional[float],
    target_pitch: Optional[float],
    target_yaw: Optional[float],
) -> float:
    """Calculate orientation difference (in radians)."""
    diff = 0.0
    num_targets = 0
    
    if target_roll is not None:
        d = abs(pose_roll - target_roll)
        d = min(d, 2 * np.pi - d)  # Handle wrapping
        diff += d
        num_targets += 1
    
    if target_pitch is not None:
        d = abs(pose_pitch - target_pitch)
        d = min(d, 2 * np.pi - d)
        diff += d
        num_targets += 1
    
    if target_yaw is not None:
        d = abs(pose_yaw - target_yaw)
        d = min(d, 2 * np.pi - d)
        diff += d
        num_targets += 1
    
    # If no targets, all poses are equally valid
    if num_targets == 0:
        diff = 0.0
    
    return diff


def classify_height(poses: List[Dict], height_low_bar: float = 33.0, height_high_bar: float = 33.0) -> Dict[str, List[Dict]]:
    """Classify poses by height."""
    z_diffs = [p["z_diff"] for p in poses]
    
    min_z_diff = min(z_diffs)
    max_z_diff = max(z_diffs)
    z_diff_range = max_z_diff - min_z_diff
    
    low_threshold = min_z_diff + z_diff_range * (height_low_bar / 100.0)
    high_threshold = max_z_diff - z_diff_range * (height_high_bar / 100.0)
    
    high_poses = []
    medium_poses = []
    low_poses = []
    
    for pose in poses:
        z_diff = pose["z_diff"]
        if z_diff > high_threshold:
            high_poses.append(pose)
        elif z_diff < low_threshold:
            low_poses.append(pose)
        else:
            medium_poses.append(pose)
    
    return {
        "high": high_poses,
        "medium": medium_poses,
        "low": low_poses,
        "thresholds": {
            "low": low_threshold,
            "high": high_threshold,
            "range": [min_z_diff, max_z_diff],
        }
    }


def create_tiled_image(
    robot: str,
    top_poses: List[Dict],
    roll_deg: Optional[float],
    pitch_deg: Optional[float],
    yaw_deg: Optional[float],
    height: Optional[str],
    tile_size: int = 256,
    border_width: int = 2,
    output_file: Optional[str] = None,
) -> Optional[str]:
    """
    Create a tiled image from the top poses.
    
    Args:
        robot: Robot name
        top_poses: List of pose dictionaries
        roll_deg: Target roll (for filename)
        pitch_deg: Target gripper_orientation (for filename)
        yaw_deg: Target yaw (for filename)
        height: Height filter (for filename)
        tile_size: Size to resize each tile
        border_width: Width of border between tiles
        output_file: Custom output path (optional)
        
    Returns:
        str: Path to saved tiled image, or None if failed
    """
    print(f"\n{'='*60}")
    print("CREATING TILED IMAGE")
    print(f"{'='*60}")
    
    # Determine output filename
    if output_file:
        tiled_output = output_file
    else:
        roll_str = f"r{int(roll_deg)}" if roll_deg is not None else "rNone"
        pitch_str = f"p{int(pitch_deg)}" if pitch_deg is not None else "pNone"
        yaw_str = f"y{int(yaw_deg)}" if yaw_deg is not None else "yNone"
        height_str = f"_h{height}" if height is not None else ""
        tiled_output = f"data/logs/{robot}_closest_{roll_str}_{pitch_str}_{yaw_str}{height_str}_tiled.png"
    
    print(f"Tiled image will be saved to: {tiled_output}")
    
    # Find source images
    image_dir = f"data/poses/{robot}"
    
    if not os.path.exists(image_dir):
        print(f"Warning: Image directory not found: {image_dir}")
        print("Skipping tiled image creation.")
        return None
    
    image_files = []
    missing_files = []
    
    for i, pose in enumerate(top_poses, 1):
        pose_id = pose["pose_id"]
        angles_str = pose["angles_str"]
        
        # Try different possible filenames
        possible_filenames = [
            f"{robot}_pose_{pose_id:06d}_{angles_str}.png",
            f"{robot}_pose_{pose_id:06d}.png",
        ]
        
        found = False
        for filename in possible_filenames:
            filepath = os.path.join(image_dir, filename)
            if os.path.exists(filepath):
                image_files.append((i, filepath, pose))
                found = True
                break
        
        if not found:
            missing_files.append((i, pose_id, angles_str))
    
    if not image_files:
        print(f"No source images found in '{image_dir}'. Skipping tiled image creation.")
        return None
    
    num_images = len(image_files)
    print(f"Found {num_images} image files")
    
    if missing_files:
        print(f"Warning: {len(missing_files)} images not found (will be skipped)")
    
    # Calculate grid size
    grid_size = int(math.ceil(math.sqrt(num_images)))
    print(f"Creating {grid_size}×{grid_size} grid ({grid_size**2} tiles)")
    
    # Calculate canvas size
    total_border_width = border_width * (grid_size - 1)
    total_border_height = border_width * (grid_size - 1)
    
    canvas_width = grid_size * tile_size + total_border_width
    canvas_height = grid_size * tile_size + total_border_height
    
    print(f"Final image size: {canvas_width}×{canvas_height} pixels")
    print(f"Tile size: {tile_size}×{tile_size} pixels")
    print(f"Border width: {border_width} pixels")
    
    # Create blank canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    
    # Try to load a font
    font = None
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Helvetica.dfont",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, 16)
            break
        except:
            continue
    
    if font is None:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Tile images
    print("Tiling images...")
    draw = ImageDraw.Draw(canvas)
    
    for idx, (rank, img_path, pose) in enumerate(image_files):
        try:
            # Load image
            img = Image.open(img_path)
            
            # Resize if needed
            if img.size != (tile_size, tile_size):
                img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            
            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate position in grid
            row = idx // grid_size
            col = idx % grid_size
            
            # Calculate pixel position
            x = col * (tile_size + border_width)
            y = row * (tile_size + border_width)
            
            # Paste image
            canvas.paste(img, (x, y))
            
            # Get orientation
            orientation = pose["orientation"]
            roll_deg_val = orientation["roll_deg"]
            pitch_deg_val = orientation["pitch_deg"]
            yaw_deg_val = orientation["yaw_deg"]
            
            # Format text
            roll_text = f"r{int(roll_deg_val)}"
            pitch_text = f"p{int(pitch_deg_val)}"
            yaw_text = f"y{int(yaw_deg_val)}"
            text = f"{roll_text}, {pitch_text}, {yaw_text}"
            
            # Draw text
            text_x = x + 5
            text_y = y + 5
            
            if font:
                try:
                    # Get text size
                    try:
                        bbox = draw.textbbox((0, 0), text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height_actual = bbox[3] - bbox[1]
                    except:
                        try:
                            text_width, text_height_actual = draw.textsize(text, font=font)
                        except:
                            text_width = len(text) * 10
                            text_height_actual = 16
                    
                    # Draw white background
                    padding = 2
                    draw.rectangle(
                        [text_x - padding, text_y - padding, 
                         text_x + text_width + padding, text_y + text_height_actual + padding],
                        fill=(255, 255, 255)
                    )
                    
                    # Draw text
                    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
                except:
                    pass
            
            # Draw borders
            if border_width > 0:
                # Right border
                if col < grid_size - 1:
                    for bw in range(border_width):
                        x_border = x + tile_size + bw
                        for by in range(tile_size):
                            if 0 <= x_border < canvas_width and 0 <= y + by < canvas_height:
                                canvas.putpixel((x_border, y + by), (200, 200, 200))
                
                # Bottom border
                if row < grid_size - 1:
                    for bw in range(border_width):
                        y_border = y + tile_size + bw
                        for bx in range(tile_size + border_width):
                            if 0 <= x + bx < canvas_width and 0 <= y_border < canvas_height:
                                canvas.putpixel((x + bx, y_border), (200, 200, 200))
        
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")
            continue
    
    # Create directory
    os.makedirs(os.path.dirname(tiled_output) if os.path.dirname(tiled_output) else '.', exist_ok=True)
    
    # Save
    print(f"Saving tiled image to: {tiled_output}")
    canvas.save(tiled_output, quality=95)
    
    file_size_mb = os.path.getsize(tiled_output) / (1024**2)
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"{'='*60}\n")
    
    return tiled_output


def query_closest_poses(
    robot: str = "IIWA",
    roll: Optional[float] = None,
    gripper_orientation: Optional[float] = None,
    yaw: Optional[float] = None,
    top_k: int = 100,
    height: Optional[str] = None,
    height_low_bar: float = 33.0,
    height_high_bar: float = 33.0,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    max_orientation_diff_deg: float = 60.0,
    save_tile_image: bool = False,
    tile_size: int = 256,
    border_width: int = 2,
    tile_output: Optional[str] = None,
):
    """
    Query closest poses from pre-exported JSONL.
    
    Args:
        robot: Robot name
        roll: Target roll angle in degrees (None to ignore)
        gripper_orientation: Target gripper_orientation angle in degrees (None to ignore)
        yaw: Target yaw angle in degrees (None to ignore)
        top_k: Number of top poses to return
        height: Filter by height ("high", "medium", "low", or None)
        height_low_bar: Percentage for low threshold
        height_high_bar: Percentage for high threshold
        input_file: Input JSONL file (default: data/poses/all_{robot}_poses.jsonl)
        output_file: Output JSON file (default: None, print summary)
        max_orientation_diff_deg: Maximum orientation difference to consider (default: 60.0)
        save_tile_image: Whether to save tiled image (default: False)
        tile_size: Size of each tile in pixels (default: 256)
        border_width: Width of border between tiles (default: 2)
        tile_output: Custom tile image output path (default: auto-generate)
    
    Examples:
        # Basic query
        python adhoc/robotarm/query_poses_from_export.py --robot IIWA --roll 0 --gripper_orientation 90 --yaw 0
        
        # With height filter
        python adhoc/robotarm/query_poses_from_export.py --robot Panda --roll 180 --yaw 0 --height high --top-k 50
        
        # Save tile image
        python adhoc/robotarm/query_poses_from_export.py --robot IIWA --roll 0 --gripper_orientation 90 --yaw 0 --save-tile-image
        
        # Custom tile output
        python adhoc/robotarm/query_poses_from_export.py --robot Panda --roll 180 --yaw 0 --save-tile-image --tile-output my_poses.png
    """
    print("\n" + "="*60)
    print("QUERYING CLOSEST POSES")
    print("="*60)
    print(f"Robot: {robot}")
    print(f"Target orientation:")
    print(f"  Roll:  {roll}°" if roll is not None else "  Roll:  None (ignored)")
    print(f"  Pitch: {gripper_orientation}°" if gripper_orientation is not None else "  Pitch: None (ignored)")
    print(f"  Yaw:   {yaw}°" if yaw is not None else "  Yaw:   None (ignored)")
    print(f"Top K: {top_k}")
    if height:
        print(f"Height filter: {height}")
    print("="*60 + "\n")
    
    # Determine input file
    if input_file is None:
        input_file = f"data/poses/all_{robot}_poses.jsonl"
    
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        print(f"\nPlease run first:")
        print(f"  python adhoc/robotarm/export_all_poses_once.py --robot {robot}")
        return None
    
    # Load poses
    print(f"Loading poses from: {input_file}")
    all_poses = load_poses_from_jsonl(input_file)
    print(f"Loaded {len(all_poses):,} poses")
    
    # Convert target angles to radians
    target_roll = np.deg2rad(roll) if roll is not None else None
    target_pitch = np.deg2rad(gripper_orientation) if gripper_orientation is not None else None
    target_yaw = np.deg2rad(yaw) if yaw is not None else None
    
    # Calculate orientation differences
    print("Calculating orientation differences...")
    max_diff_rad = np.deg2rad(max_orientation_diff_deg)
    
    filtered_poses = []
    for pose in tqdm(all_poses):
        orn = pose["orientation"]
        diff = calculate_orientation_diff(
            orn["roll_rad"],
            orn["pitch_rad"],
            orn["yaw_rad"],
            target_roll,
            target_pitch,
            target_yaw,
        )
        
        # Filter by max orientation diff
        if diff <= max_diff_rad:
            pose["orientation_diff_rad"] = diff
            pose["orientation_diff_deg"] = np.rad2deg(diff)
            filtered_poses.append(pose)
    
    print(f"\nFiltered {len(all_poses) - len(filtered_poses)} poses with orientation_diff > {max_orientation_diff_deg}°")
    print(f"Remaining poses: {len(filtered_poses)}")
    
    if not filtered_poses:
        print("No poses found matching criteria!")
        return None
    
    # Sort by orientation difference
    filtered_poses.sort(key=lambda x: x["orientation_diff_rad"])
    
    # Take top K
    top_poses = filtered_poses[:top_k]
    print(f"Selected top {len(top_poses)} poses based on orientation similarity")
    
    # Filter by is_front (EE in front of root)
    front_poses = [p for p in top_poses if p.get("is_front", False)]
    
    if front_poses:
        top_poses = front_poses
        print(f"Filtered to {len(top_poses)} poses with EE in front of root")
    else:
        print(f"Warning: No poses with EE in front of root")
    
    # Filter by height if specified
    if height is not None:
        height_classified = classify_height(top_poses, height_low_bar, height_high_bar)
        
        print(f"\nHeight classification:")
        print(f"  Low: {len(height_classified['low'])} poses")
        print(f"  Medium: {len(height_classified['medium'])} poses")
        print(f"  High: {len(height_classified['high'])} poses")
        print(f"  Thresholds: {height_classified['thresholds']}")
        
        height_lower = height.lower()
        if height_lower in height_classified:
            top_poses = height_classified[height_lower]
            print(f"Selected {len(top_poses)} poses with height={height}")
        else:
            print(f"Warning: Unknown height value '{height}'. Using all poses.")
    
    if not top_poses:
        print("No poses found after filtering!")
        return None
    
    # Sort by root-to-EE distance (smallest first)
    top_poses.sort(key=lambda x: x["root_to_ee_distance"])
    
    # Create tiled image if requested
    tiled_image_path = None
    if save_tile_image:
        tiled_image_path = create_tiled_image(
            robot=robot,
            top_poses=top_poses,
            roll_deg=roll,
            pitch_deg=gripper_orientation,
            yaw_deg=yaw,
            height=height,
            tile_size=tile_size,
            border_width=border_width,
            output_file=tile_output,
        )
    
    # Prepare output
    result = {
        "robot": robot,
        "target_orientation": {
            "roll_deg": roll,
            "pitch_deg": gripper_orientation,
            "yaw_deg": yaw,
        },
        "height_filter": height,
        "total_poses_searched": len(all_poses),
        "poses_after_orientation_filter": len(filtered_poses),
        "top_k": len(top_poses),
        "tiled_image": tiled_image_path,
        "poses": top_poses,
    }
    
    # Save or print
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        if tiled_image_path:
            print(f"Tiled image saved to: {tiled_image_path}")
    else:
        print(f"\n{'='*60}")
        print("SUMMARY:")
        print(f"  Total poses: {len(top_poses)}")
        if top_poses:
            print(f"  Avg distance: {np.mean([p['root_to_ee_distance'] for p in top_poses]):.4f} m")
            print(f"  Min distance: {min([p['root_to_ee_distance'] for p in top_poses]):.4f} m")
            print(f"  Max distance: {max([p['root_to_ee_distance'] for p in top_poses]):.4f} m")
            print(f"  Avg orientation diff: {np.mean([p['orientation_diff_deg'] for p in top_poses]):.2f}°")
        if tiled_image_path:
            print(f"  Tiled image: {tiled_image_path}")
        print(f"{'='*60}\n")
    
    return result


if __name__ == "__main__":
    fire.Fire(query_closest_poses)
