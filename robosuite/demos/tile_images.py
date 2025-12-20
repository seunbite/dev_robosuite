"""
Tile PNG images in a directory into a square grid.

This script:
1. Finds all PNG images in a directory (and optionally subdirectories)
2. Sorts them by filename
3. Arranges them in a square grid (e.g., 100 images → 10×10 grid)
4. Saves the result as a single large image

Usage:
    python tile_images.py --input-dir data/poses/Panda --output tiled.png
    python tile_images.py --input-dir data/poses/Panda/Panda/3d_points --output 3d_tiled.png
    python tile_images.py --input-dir data/poses/Panda --recursive True --output all_tiled.png
"""

import fire
import os
import numpy as np
from PIL import Image
from pathlib import Path
import math


def tile_images(
    input_dir: str,
    output: str = "tiled_images.png",
    recursive: bool = False,
    max_images: int = None,
    tile_size: int = None,
    background_color: tuple = (255, 255, 255),
    border_width: int = 0,
    border_color: tuple = (200, 200, 200)
):
    """
    Tile PNG images from a directory into a square grid.
    
    Args:
        input_dir: Directory containing PNG images
        output: Output filename for the tiled image
        recursive: If True, search subdirectories recursively
        max_images: Maximum number of images to tile (None = all)
        tile_size: Size to resize each tile (None = use original size)
        background_color: RGB color for background (default: white)
        border_width: Width of border between tiles in pixels (default: 0)
        border_color: RGB color for borders (default: light gray)
    
    Examples:
        # Basic usage
        python tile_images.py --input-dir data/poses/Panda
        
        # With options
        python tile_images.py --input-dir data/poses/Panda --tile-size 128 --border-width 2
        
        # Recursive search
        python tile_images.py --input-dir data/poses --recursive True --output all_poses.png
    """
    
    print("="*60)
    print("IMAGE TILING SCRIPT")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output}")
    print(f"Recursive search: {recursive}")
    print("="*60 + "\n")
    
    # Check if directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist")
        return
    
    # Find all PNG files
    print("Searching for PNG files...")
    
    if recursive:
        # Recursive search using pathlib
        png_files = sorted(Path(input_dir).rglob("*.png"))
    else:
        # Non-recursive search
        png_files = sorted(Path(input_dir).glob("*.png"))
    
    png_files = [str(f) for f in png_files]
    
    if not png_files:
        print(f"No PNG files found in '{input_dir}'")
        return
    
    print(f"Found {len(png_files)} PNG files")
    
    # Limit number of images if specified
    if max_images and len(png_files) > max_images:
        print(f"Limiting to first {max_images} images")
        png_files = png_files[:max_images]
    
    num_images = len(png_files)
    
    # Calculate grid size (square grid)
    grid_size = int(math.ceil(math.sqrt(num_images)))
    print(f"Creating {grid_size}×{grid_size} grid ({grid_size**2} tiles)")
    
    # Load first image to get dimensions
    first_img = Image.open(png_files[0])
    original_width, original_height = first_img.size
    
    # Determine tile size
    if tile_size is not None:
        img_width = img_height = tile_size
        print(f"Resizing each tile to {tile_size}×{tile_size} pixels")
    else:
        img_width, img_height = original_width, original_height
        print(f"Using original size: {img_width}×{img_height} pixels")
    
    # Calculate total canvas size including borders
    total_border_width = border_width * (grid_size - 1)
    total_border_height = border_width * (grid_size - 1)
    
    canvas_width = grid_size * img_width + total_border_width
    canvas_height = grid_size * img_height + total_border_height
    
    print(f"Final image size: {canvas_width}×{canvas_height} pixels")
    print(f"Estimated file size: ~{(canvas_width * canvas_height * 3) / (1024**2):.1f} MB (uncompressed)\n")
    
    # Create blank canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), background_color)
    
    # Tile images
    print("Tiling images...")
    
    for idx, png_file in enumerate(png_files):
        try:
            # Load image
            img = Image.open(png_file)
            
            # Resize if needed
            if tile_size is not None:
                img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed (in case of RGBA)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate position in grid
            row = idx // grid_size
            col = idx % grid_size
            
            # Calculate pixel position (accounting for borders)
            x = col * (img_width + border_width)
            y = row * (img_height + border_width)
            
            # Paste image onto canvas
            canvas.paste(img, (x, y))
            
            # Draw borders if specified
            if border_width > 0 and (col < grid_size - 1 or row < grid_size - 1):
                # Right border
                if col < grid_size - 1:
                    for bw in range(border_width):
                        x_border = x + img_width + bw
                        for by in range(img_height):
                            canvas.putpixel((x_border, y + by), border_color)
                
                # Bottom border
                if row < grid_size - 1:
                    for bw in range(border_width):
                        y_border = y + img_height + bw
                        for bx in range(img_width + border_width):
                            canvas.putpixel((x + bx, y_border), border_color)
            
            # Progress update
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{num_images} images...")
        
        except Exception as e:
            print(f"  Error processing {png_file}: {e}")
            continue
    
    # Save result
    print(f"\nSaving tiled image to: {output}")
    canvas.save(output, quality=95)
    
    # Print summary
    file_size_mb = os.path.getsize(output) / (1024**2)
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")
    print(f"Total images: {num_images}")
    print(f"Grid size: {grid_size}×{grid_size}")
    print(f"Image size: {canvas_width}×{canvas_height} pixels")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Saved to: {output}")
    print(f"{'='*60}\n")


def compare_directories(
    dir1: str,
    dir2: str,
    output: str = "comparison.png",
    tile_size: int = 256,
    max_images: int = 100
):
    """
    Create a side-by-side comparison of images from two directories.
    
    Args:
        dir1: First directory (e.g., camera images)
        dir2: Second directory (e.g., 3D plots)
        output: Output filename
        tile_size: Size to resize each tile
        max_images: Maximum number of image pairs to compare
    
    Example:
        python tile_images.py compare_directories \
            --dir1 data/poses/Panda \
            --dir2 data/poses/Panda/Panda/3d_points \
            --output comparison.png
    """
    print("="*60)
    print("DIRECTORY COMPARISON")
    print("="*60)
    print(f"Directory 1: {dir1}")
    print(f"Directory 2: {dir2}")
    print("="*60 + "\n")
    
    # Get sorted file lists
    files1 = sorted(Path(dir1).glob("*.png"))
    files2 = sorted(Path(dir2).glob("*.png"))
    
    # Match files by name
    names1 = {f.name: f for f in files1}
    names2 = {f.name: f for f in files2}
    
    common_names = sorted(set(names1.keys()) & set(names2.keys()))
    
    if not common_names:
        print("No common files found between directories")
        return
    
    print(f"Found {len(common_names)} common images")
    
    # Limit if needed
    if max_images and len(common_names) > max_images:
        common_names = common_names[:max_images]
        print(f"Limiting to first {max_images} images")
    
    num_images = len(common_names)
    grid_size = int(math.ceil(math.sqrt(num_images)))
    
    # Create canvas (2 columns per image: left from dir1, right from dir2)
    canvas_width = grid_size * tile_size * 2  # 2x width for side-by-side
    canvas_height = grid_size * tile_size
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    
    print(f"Creating {grid_size}×{grid_size} comparison grid")
    print(f"Canvas size: {canvas_width}×{canvas_height}\n")
    
    for idx, name in enumerate(common_names):
        try:
            # Load both images
            img1 = Image.open(names1[name]).resize((tile_size, tile_size), Image.Resampling.LANCZOS).convert('RGB')
            img2 = Image.open(names2[name]).resize((tile_size, tile_size), Image.Resampling.LANCZOS).convert('RGB')
            
            # Calculate position
            row = idx // grid_size
            col = idx % grid_size
            
            x_base = col * tile_size * 2
            y = row * tile_size
            
            # Paste side by side
            canvas.paste(img1, (x_base, y))
            canvas.paste(img2, (x_base + tile_size, y))
            
        except Exception as e:
            print(f"Error processing {name}: {e}")
    
    print(f"Saving to: {output}")
    canvas.save(output, quality=95)
    print(f"Complete! Saved {num_images} image pairs\n")


if __name__ == "__main__":
    fire.Fire({
        'tile': tile_images,
        'compare': compare_directories
    })



