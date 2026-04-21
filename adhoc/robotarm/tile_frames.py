"""
Create a tiled image from frames in a directory.

This script samples N frames evenly from a directory and combines them horizontally into a single PNG.
"""

import os
import fire
import numpy as np
from PIL import Image
from typing import Optional, Union, List


def tile_frames(
    frames_dir: str,
    n: int = 4,
    output_path: Optional[str] = None,
    pattern: str = "frame_*.png",
    frame_indexes: Optional[Union[str, List[int]]] = None,
    crop_margin: float = 0.0,
):
    """
    Create a horizontal tiled image from evenly sampled or specified frames.
    
    Args:
        frames_dir: Directory containing frame images
        n: Number of frames to sample (ignored if frame_indexes is provided)
        output_path: Output PNG path (default: same dir with _tiled.png suffix)
        pattern: File pattern to match (default: frame_*.png)
        frame_indexes: Comma-separated frame indices (e.g., "0,7,13,19")
        crop_margin: Fraction to crop from edges (e.g., 0.2 = crop 20% from each side)
    
    Returns:
        Path to the created tiled image
    """
    if not os.path.exists(frames_dir):
        raise ValueError(f"Directory not found: {frames_dir}")
    
    # Get all frame files sorted by name
    all_files = sorted([
        f for f in os.listdir(frames_dir) 
        if f.endswith('.png') and f.startswith('frame_')
    ])
    
    if not all_files:
        raise ValueError(f"No frame files found in {frames_dir}")
    
    total_frames = len(all_files)
    print(f"Found {total_frames} frames in {frames_dir}")
    
    # Determine which frames to use
    if frame_indexes is not None:
        # Use specified indices
        # Handle both string (comma-separated) and list/tuple
        if isinstance(frame_indexes, str):
            indices = [int(idx.strip()) for idx in frame_indexes.split(',')]
        elif isinstance(frame_indexes, (list, tuple)):
            indices = [int(idx) for idx in frame_indexes]
        else:
            raise ValueError(f"frame_indexes must be string, list, or tuple, got {type(frame_indexes)}")
        
        n = len(indices)
        print(f"Using specified frames: indices {indices}")
        
        # Validate indices
        for idx in indices:
            if idx < 0 or idx >= total_frames:
                raise ValueError(f"Frame index {idx} out of range (0-{total_frames-1})")
        
        selected_files = [all_files[i] for i in indices]
    else:
        # Sample n frames evenly
        if n > total_frames:
            print(f"Warning: Requested {n} frames but only {total_frames} available. Using all frames.")
            n = total_frames
        
        # Calculate indices for even sampling
        indices = np.linspace(0, total_frames - 1, n, dtype=int)
        selected_files = [all_files[i] for i in indices]
        
        print(f"Sampling {n} frames evenly: indices {indices.tolist()}")
    
    # Load images and crop if needed
    images = []
    for fname in selected_files:
        img_path = os.path.join(frames_dir, fname)
        img = Image.open(img_path)
        
        # Crop edges if crop_margin > 0
        if crop_margin > 0:
            original_w, original_h = img.size
            left = int(original_w * crop_margin)
            top = int(original_h * crop_margin)
            right = int(original_w * (1 - crop_margin))
            bottom = int(original_h * (1 - crop_margin))
            img = img.crop((left, top, right, bottom))
            print(f"  - {fname} ({original_w}x{original_h} → {img.size[0]}x{img.size[1]} after {crop_margin*100:.0f}% crop)")
        else:
            print(f"  - {fname} ({img.size[0]}x{img.size[1]})")
        
        images.append(img)
    
    # Get dimensions (assume all images have same size after cropping)
    width, height = images[0].size
    
    # Create tiled image (horizontal)
    tiled_width = width * n
    tiled_height = height
    tiled_image = Image.new('RGB', (tiled_width, tiled_height))
    
    # Paste images horizontally
    for i, img in enumerate(images):
        x_offset = i * width
        tiled_image.paste(img, (x_offset, 0))
    
    # Determine output path
    if output_path is None:
        # Use the directory name to create output filename
        dir_name = os.path.basename(frames_dir.rstrip('/'))
        parent_dir = os.path.dirname(frames_dir)
        output_path = os.path.join(parent_dir, f"{dir_name}_tiled_{n}.png")
    
    # Save
    tiled_image.save(output_path)
    print(f"\n✓ Saved tiled image ({tiled_width}x{tiled_height}) to:")
    print(f"  {output_path}")
    
    return output_path


def main(
    frames_dir: str,
    n: int = 4,
    output: Optional[str] = None,
    frame_indexes: Optional[Union[str, List[int]]] = None,
    crop_margin: float = 0.0,
):
    """
    Create a horizontal tiled image from evenly sampled or specified frames.
    
    Usage:
        # Sample 4 frames evenly
        python tile_frames.py /path/to/frames_dir --n=4
        
        # Specify exact frame indices
        python tile_frames.py /path/to/frames_dir --frame_indexes="0,7,13,19"
        
        # Crop 20% from edges
        python tile_frames.py /path/to/frames_dir --n=4 --crop_margin=0.2
        
        # Specify output path
        python tile_frames.py /path/to/frames_dir --n=6 --output=tiled.png
    
    Args:
        frames_dir: Directory containing frame_*.png files
        n: Number of frames to sample (ignored if frame_indexes is provided)
        output: Output PNG path (optional)
        frame_indexes: Comma-separated frame indices (e.g., "0,7,13,19")
        crop_margin: Fraction to crop from edges (0.0-0.5, e.g., 0.2 = 20%)
    """
    return tile_frames(frames_dir, n, output, frame_indexes=frame_indexes, crop_margin=crop_margin)


if __name__ == "__main__":
    fire.Fire(main)
