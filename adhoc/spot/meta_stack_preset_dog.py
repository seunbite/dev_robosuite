import subprocess
import os
import itertools
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm

# Configuration ranges for brute force
KP_VALUES = [500]
KV_VALUES = [10]
KD_VALUES = [500]
FRICTION_VALUES = [2] # foot friction
FRONT_KP_VALUES = [1000] # front leg specific kp
FRONT_SHOULDER_VALUES = [10, 20]
FRONT_KNEE_VALUES = [0, -10, -20]

ROBOT_XML_PATH = "robosuite/models/assets/robots/spot/robot.xml"
OUTPUT_META_DIR = "data/poses/quadruped/meta_results"
os.makedirs(OUTPUT_META_DIR, exist_ok=True)

def update_xml_friction(friction_val):
    with open(ROBOT_XML_PATH, 'r') as f:
        content = f.read()
    
    # Replace friction value in spot_foot class
    # Matches any decimal value for the first friction component
    import re
    new_friction_str = f'friction="{friction_val} 0.02 0.01"'
    pattern = r'friction="[0-9.]+\s+0.02\s+0.01"'
    new_content = re.sub(pattern, new_friction_str, content)
    
    with open(ROBOT_XML_PATH, 'w') as f:
        f.write(new_content)

def run_single_test(kp, kv, kd, friction, front_sh, front_kn, front_kp):
    update_xml_friction(friction)
    
    cmd = [
        "python", "adhoc/spot/stack_preset_dog.py",
        "--robot", "SpotWithArm",
        "--kp", str(kp),
        "--kv", str(kv),
        "--kd", str(kd),
        "--front-kp", str(front_kp),
        "--front-shoulder", str(front_sh),
        "--front-knee", str(front_kn),
        "--target-pose", "sitting",
        "--physics-steps", "100"
    ]
    
    print(f"\nRunning: KP={kp}, KV={kv}, KD={kd}, Friction={friction}, F_SH={front_sh}, F_KN={front_kn}, F_KP={front_kp}")
    subprocess.run(cmd, check=True)
    
    # Copy result gifs to meta dir with descriptive names
    base_src = "data/poses/quadruped/SpotWithArm/step_gifs/SpotWithArm_sitting"
    results = {}
    
    import shutil
    for view in ["frontview", "sideview"]:
        src_gif = f"{base_src}_{view}.gif"
        dest_name = f"kp{kp}_kv{kv}_kd{kd}_f{friction}_sh{front_sh}_kn{front_kn}_fkp{front_kp}_{view}.gif"
        dest_path = os.path.join(OUTPUT_META_DIR, dest_name)
        
        if os.path.exists(src_gif):
            shutil.copy(src_gif, dest_path)
            results[view] = dest_path
            
    return results if results else None

def create_tiled_gif(gif_paths, combinations, output_name):
    if not gif_paths:
        return
    
    # Load all gifs
    all_gif_frames = []
    for p in gif_paths:
        img = Image.open(p)
        frames = []
        try:
            while True:
                frames.append(img.copy().convert("RGB"))
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        all_gif_frames.append(frames)
    
    # Synchronize lengths
    min_len = min(len(f) for f in all_gif_frames)
    
    tiled_frames = []
    
    # Determine grid size (approx square)
    n = len(gif_paths)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    frame_w, frame_h = all_gif_frames[0][0].size
    
    for i in range(min_len):
        new_frame = Image.new('RGB', (cols * frame_w, rows * frame_h))
        draw = ImageDraw.Draw(new_frame)
        
        for idx, (frames, combo) in enumerate(zip(all_gif_frames, combinations)):
            r, c = divmod(idx, cols)
            x, y = c * frame_w, r * frame_h
            new_frame.paste(frames[i], (x, y))
            
            # Label
            label = f"KP:{combo[0]} KV:{combo[1]} KD:{combo[2]} F:{combo[3]} SH:{combo[4]} KN:{combo[5]} FKP:{combo[6]}"
            draw.text((x + 10, y + 10), label, fill=(255, 0, 0))
            
        tiled_frames.append(new_frame)
    
    final_path = os.path.join(OUTPUT_META_DIR, f"{output_name}.gif")
    tiled_frames[0].save(
        final_path,
        save_all=True,
        append_images=tiled_frames[1:],
        duration=50,
        loop=0
    )
    print(f"\nFinal tiled GIF saved to: {final_path}")
    
    # Also save the last frame as a static tiled image for quick comparison
    final_image_path = os.path.join(OUTPUT_META_DIR, f"{output_name}.png")
    tiled_frames[-1].save(final_image_path)
    print(f"Final tiled static image saved to: {final_image_path}")

def main():
    combinations = list(itertools.product(
        KP_VALUES, KV_VALUES, KD_VALUES, FRICTION_VALUES, 
        FRONT_SHOULDER_VALUES, FRONT_KNEE_VALUES, FRONT_KP_VALUES
    ))
    gif_paths_front = []
    gif_paths_side = []
    successful_combos = []
    
    print(f"Starting meta-analysis for {len(combinations)} combinations...")
    
    try:
        for kp, kv, kd, friction, front_sh, front_kn, front_kp in combinations:
            paths = run_single_test(kp, kv, kd, friction, front_sh, front_kn, front_kp)
            if paths:
                if "frontview" in paths:
                    gif_paths_front.append(paths["frontview"])
                if "sideview" in paths:
                    gif_paths_side.append(paths["sideview"])
                successful_combos.append((kp, kv, kd, friction, front_sh, front_kn, front_kp))
    finally:
        # Create tiled gifs even if interrupted
        if gif_paths_front:
            print("\nCreating tiled GIF for Front View...")
            create_tiled_gif(gif_paths_front, successful_combos, "tiled_bruteforce_frontview")
        if gif_paths_side:
            print("\nCreating tiled GIF for Side View...")
            create_tiled_gif(gif_paths_side, successful_combos, "tiled_bruteforce_sideview")

if __name__ == "__main__":
    main()
