from arm_pose_config import direction_pose_set, pitch_poses, poses, height_map
from find_closest_poses import ClosestPoseFinder
from PIL import Image
import math
import os

robots = ["IIWA", "Panda"]
for robot in robots:
    print(f"\n{'='*60}")
    print(f"Processing robot: {robot}")
    print(f"{'='*60}")
    
    # Create finder once per robot
    finder = ClosestPoseFinder(robot_name=robot)
    
    try:
        for pose_name, pose_config in direction_pose_set.items():
            height_str, direction, pitch_name = pose_config['height'], pose_config['dir'], pose_config['gripper_orientation']
            height = height_map[height_str]
            pitch_values = pitch_poses[pitch_name]
            roll_yaw_sets = poses[direction]
            
            print(f"\n  Pose: {pose_name}")
            
            # Collect all images in memory
            images_stack = []
            for roll_yaw in roll_yaw_sets:
                roll, yaw = roll_yaw['roll'], roll_yaw['yaw']
                for pitch_val in pitch_values:
                    print(f"    Generating r{int(roll)}, p{int(pitch_val)}, y{int(yaw)}, h{height}")
                    
                    # Generate poses and get tiled image directly
                    results = finder.find_closest_poses(
                        roll_deg=roll,
                        pitch_deg=pitch_val,
                        yaw_deg=yaw,
                        height=height,
                        top_k=100,
                        stack_jsonl_path=None,  # Don't save to JSONL
                    )
                    
                    # Get the tiled image from results
                    tiled_image_path = results.get('tiled_image')
                    if tiled_image_path and os.path.exists(tiled_image_path):
                        # Load image into memory
                        img = Image.open(tiled_image_path)
                        images_stack.append(img.copy())  # Copy to memory
                        img.close()
            
            if not images_stack:
                print(f"  No images generated, skipping")
                continue
            
            print(f"  Generated {len(images_stack)} images in memory")
            
            # Create tiled image from memory
            tile_size = 256
            border_width = 2
            num_images = len(images_stack)
            grid_size = int(math.ceil(math.sqrt(num_images)))
            
            canvas_width = grid_size * tile_size + border_width * (grid_size - 1)
            canvas_height = grid_size * tile_size + border_width * (grid_size - 1)
            
            canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
            
            for idx, img in enumerate(images_stack):
                try:
                    if img.size != (tile_size, tile_size):
                        img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    row = idx // grid_size
                    col = idx % grid_size
                    x = col * (tile_size + border_width)
                    y = row * (tile_size + border_width)
                    
                    canvas.paste(img, (x, y))
                except Exception as e:
                    print(f"  Error processing image: {e}")
                    continue
            
            # Save final tiled image
            os.makedirs("data/logs/poses", exist_ok=True)
            save_name = f"data/logs/poses/{robot}_{pose_name}_{height_str}_{direction}_{pitch_name}_tiled.png"
            canvas.save(save_name, quality=95)
            print(f"  Saved: {save_name}")
    
    finally:
        finder.close()

print("\n" + "="*60)
print("Done!")
print("="*60)
