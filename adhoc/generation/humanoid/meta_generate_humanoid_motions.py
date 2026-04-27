import os
import json
import sys
import logging
import numpy as np
from tqdm import tqdm
import fire
from contextlib import contextmanager
from PIL import Image, ImageDraw, ImageFont
import math
import multiprocessing
from functools import partial
from datetime import datetime
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robotarm'))

from motion_generation import MotionGenerator
from motion_config_generate import generate_motion_config

# Silence robosuite's internal logger
for name in logging.root.manager.loggerDict:
    if "robosuite" in name:
        logging.getLogger(name).setLevel(logging.ERROR)
logging.getLogger("robosuite").setLevel(logging.ERROR)

@contextmanager
def suppress_stdout(enable=False):
    """Context manager to suppress stdout and stderr if enable is True."""
    if enable:
        with open(os.devnull, "w") as devnull:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = devnull, devnull
            try: yield
            finally: sys.stdout, sys.stderr = old_stdout, old_stderr
    else: yield

def combine_variations_horizontally(frames_list: list[list[Image.Image]], output_path: str, hz: int = 4, metadata_list: list = None, scores: list = None, max_per_row: int = 10):
    """Combine multiple variation frame lists into a grid GIF.
    
    Wraps into multiple rows when there are more than max_per_row variations.
    E.g., 20 variations with max_per_row=10 → 2 rows of 10.
    """
    if not frames_list or not frames_list[0]: return None, None
    
    max_frames = max(len(fs) for fs in frames_list)
    tile_w, tile_h = frames_list[0][0].size
    num_vars = len(frames_list)
    
    # Grid layout: wrap into rows
    cols = min(num_vars, max_per_row)
    rows = math.ceil(num_vars / max_per_row)
    total_w = cols * tile_w
    total_h = rows * tile_h
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        bold_font = ImageFont.truetype("/System/Library/Fonts/Helvetica-Bold.ttc", 16)
    except:
        font, bold_font = ImageFont.load_default(), ImageFont.load_default()

    combined_frames = []
    for i in range(max_frames):
        new_frame = Image.new("RGB", (total_w, total_h), (40, 40, 40))
        draw = ImageDraw.Draw(new_frame)
        
        for v_idx, fs in enumerate(frames_list):
            row = v_idx // max_per_row
            col = v_idx % max_per_row
            x_offset = col * tile_w
            y_offset = row * tile_h
            
            frame = fs[i % len(fs)]
            if frame.size != (tile_w, tile_h): frame = frame.resize((tile_w, tile_h))
            new_frame.paste(frame, (x_offset, y_offset))
            
            if metadata_list and v_idx < len(metadata_list):
                meta = metadata_list[v_idx]
                score = scores[v_idx] if scores else 0.0
                last_move = None
                if "movements" in meta:
                    for m in reversed(meta["movements"]):
                        if m.get("type") == "movement": last_move = m; break
                        elif m.get("type") == "parallel":
                            for sub_m in reversed(m.get("actions", [])):
                                if sub_m.get("type") == "movement": last_move = sub_m; break
                            if last_move: break
                
                v_label = f"Rank {v_idx+1}"
                if last_move:
                    defined_axis = last_move.get("parameters", {}).get("axis", "N/A")
                    implemented_axis = last_move.get("implemented_axis", "N/A")
                    label_str = f"DEF: {defined_axis.upper()} | IMP: {implemented_axis.upper()}"
                    
                    draw.rectangle([x_offset + 5, y_offset + 5, x_offset + 80, y_offset + 25], fill=(255, 255, 255, 200))
                    draw.text((x_offset + 10, y_offset + 7), v_label, fill="black", font=font)
                    
                    score_str = f"Match: {score*100:.1f}%"
                    sw = draw.textlength(score_str, font=font) if hasattr(draw, 'textlength') else 80
                    draw.rectangle([x_offset + tile_w - sw - 15, y_offset + 5, x_offset + tile_w - 5, y_offset + 25], fill=(0, 128, 0, 200) if score > 0.8 else (128, 0, 0, 200))
                    draw.text((x_offset + tile_w - sw - 10, y_offset + 7), score_str, fill="white", font=font)
                    
                    tw = draw.textlength(label_str, font=bold_font) if hasattr(draw, 'textlength') else 150
                    tx = x_offset + (tile_w - tw) // 2
                    draw.rectangle([tx - 5, y_offset + 28, tx + tw + 5, y_offset + 52], fill=(0, 0, 0, 180))
                    draw.text((tx, y_offset + 30), label_str, fill="cyan" if score > 0.8 else "yellow", font=bold_font)
            
            # Vertical separator between columns
            if col > 0:
                draw.line([(x_offset, y_offset), (x_offset, y_offset + tile_h)], fill="white", width=2)
            # Horizontal separator between rows
            if row > 0:
                draw.line([(x_offset, y_offset), (x_offset + tile_w, y_offset)], fill="white", width=2)
        combined_frames.append(new_frame)
    
    if combined_frames:
        # Sample for global palette to prevent flickering (increased sample count)
        sample_count = min(len(combined_frames), 100)
        sample_indices = np.unique(np.linspace(0, len(combined_frames) - 1, sample_count, dtype=int))
        palette_combined = Image.new("RGB", (total_w, total_h * len(sample_indices)))
        for idx_s, idx_f in enumerate(sample_indices):
            palette_combined.paste(combined_frames[idx_f], (0, idx_s * total_h))
        
        # Use adaptive palette with FLOYDSTEINBERG dithering for smoother transitions
        palette_img = palette_combined.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
        quantized_frames = [f.quantize(palette=palette_img, dither=Image.FLOYDSTEINBERG) for f in combined_frames]
        
        quantized_frames[0].save(
            output_path,
            save_all=True,
            append_images=quantized_frames[1:],
            duration=int(1000 / hz),
            loop=0,
            disposal=2,
            optimize=False
        )
        return output_path, combined_frames
    return None, None

def update_pose_info_yaml(robot: str, active_arm: str, cue: str, pose_data: dict, pose_info_file: str = "data/motions/initial_pose_info.yml"):
    """Update YAML file with pose info for a single cue (incremental update)."""
    import threading
    
    # Use a lock for thread-safe file access
    if not hasattr(update_pose_info_yaml, 'lock'):
        update_pose_info_yaml.lock = threading.Lock()
    
    with update_pose_info_yaml.lock:
        # Load existing data
        if os.path.exists(pose_info_file):
            with open(pose_info_file, 'r', encoding='utf-8') as f:
                existing_data = yaml.safe_load(f) or {}
        else:
            existing_data = {}
        
        # Use robot_arm as key for humanoid (e.g., "GR1_right")
        robot_key = f"{robot}_{active_arm}"
        
        # Update with new cue data
        if robot_key not in existing_data:
            existing_data[robot_key] = {}
        
        # Preserve existing best_manual_index if it exists
        if cue in existing_data[robot_key] and 'best_manual_index' in existing_data[robot_key][cue]:
            pose_data['best_manual_index'] = existing_data[robot_key][cue]['best_manual_index']
        
        existing_data[robot_key][cue] = pose_data
        
        # Save back to YAML
        os.makedirs(os.path.dirname(pose_info_file), exist_ok=True)
        with open(pose_info_file, 'w', encoding='utf-8') as f:
            yaml.dump(existing_data, f,
                     default_flow_style=False,
                     allow_unicode=True,
                     sort_keys=True,
                     indent=2)

def process_single_robot_arm(robot, active_arm, cues, jsonl_path, config_path, hz, top_k, proximal_degree_scale, verbose, save_separately, cue_yml_indices=None):
    """Process one robot-arm combination."""
    print(f"\n[Worker] Initializing generator for {robot} ({active_arm} arm)...")
    
    # Humanoid robots use predefined poses → top_k=1 (one pose per direction+pitch)
    is_humanoid = any(robot.startswith(pfx) for pfx in ['GR1'])
    if is_humanoid:
        effective_top_k = 1
        print(f"  [Humanoid] Using predefined poses → top_k=1")
    else:
        effective_top_k = top_k
    
    # Determine jsonl path for humanoid (still needed as fallback)
    if jsonl_path is None:
        jsonl_path = f"data/poses/{robot}/closest_{robot}_{active_arm}_poses.jsonl"
        if not os.path.exists(jsonl_path):
            jsonl_path = f"data/poses/{robot}/all_{robot}_{active_arm}_poses.jsonl"
            if is_humanoid and not os.path.exists(jsonl_path):
                # Predefined poses don't need JSONL; use a dummy path
                jsonl_path = f"data/poses/{robot}/all_{robot}_{active_arm}_poses.jsonl"
    
    try:
        generator = MotionGenerator(robot_name=robot, jsonl_path=jsonl_path, hz=hz, active_arm=active_arm)
    except Exception as e:
        cue_names = [c[1] if isinstance(c, tuple) else c for c in cues]
        return robot, active_arm, {}, [(robot, active_arm, cue, f"Init Failed: {str(e)}") for cue in cue_names], []

    robot_tiled_frames = {}
    robot_errors = []
    robot_metadata = []
    
    need_frames = save_separately or effective_top_k > 1

    for cue in tqdm(cues, desc=f"{robot}-{active_arm}"):
        try:
            with suppress_stdout(enable=not verbose):
                save_individual = save_separately and effective_top_k == 1
                
                # Use cue name matching directly (best_configs.json has one entry per cue)
                res_dict = generator.execute_cue(
                    cue=cue, 
                    config_path=config_path, 
                    hz=hz, 
                    top_k=effective_top_k, 
                    proximal_degree_scale=proximal_degree_scale,
                    save_gifs=save_individual,
                    return_frames=need_frames,
                )
            
            all_v_frames = res_dict.get("frames", [])
            all_metadata = res_dict.get("metadata", [])
            all_scores = res_dict.get("scores", [])
            all_paths = res_dict.get("paths", [])
            
            # Store metadata and pose info
            initial_pose_ids = []
            for idx, (meta, score) in enumerate(zip(all_metadata, all_scores)):
                pose_id = meta.get("pose_id")
                initial_pose_ids.append(pose_id)
                
                motion_meta = {
                    "robot": robot,
                    "active_arm": active_arm,
                    "cue": cue,
                    "pose_id": pose_id,
                    "candidate_idx": idx + 1,
                    "min_alignment": score,
                    "gif_path": all_paths[idx] if idx < len(all_paths) else None,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                robot_metadata.append(motion_meta)
            
            # Store initial pose indexes for this cue and update YAML
            if initial_pose_ids:
                pose_data = {
                    'initial_indexes': initial_pose_ids,
                    'best_manual_index': None
                }
                
                # Update YAML immediately after each cue
                try:
                    update_pose_info_yaml(robot, active_arm, cue, pose_data)
                    print(f"  ✓ Updated YAML for {robot}_{active_arm}/{cue}")
                except Exception as e:
                    print(f"  ⚠ Failed to update YAML for {robot}_{active_arm}/{cue}: {e}")
            
            if need_frames:
                if not all_v_frames: raise ValueError("No frames generated")
                
                if effective_top_k > 1 and len(all_v_frames) > 1:
                    if save_separately:
                        yml_idx = cue_yml_indices.get(cue, 0) if cue_yml_indices else 0
                        tiled_filename = f"{datetime.now().strftime('%Y%m%d')}_{yml_idx:03d}_{robot}_{active_arm}_{cue}_tiled.gif"
                        tiled_path = os.path.join("data/motions", robot, tiled_filename)
                        os.makedirs(os.path.dirname(tiled_path), exist_ok=True)
                        
                        tiled_path_out, tiled_fs = combine_variations_horizontally(
                            all_v_frames, tiled_path, hz=hz, 
                            metadata_list=all_metadata, scores=all_scores
                        )
                        if tiled_fs: robot_tiled_frames[(robot, active_arm, cue)] = tiled_fs
                        else: robot_tiled_frames[(robot, active_arm, cue)] = [f.convert("RGB") for f in all_v_frames[0]]
                    else:
                        dummy_path = f"temp_tiled_{robot}_{active_arm}_{cue}.gif"
                        _, tiled_fs = combine_variations_horizontally(
                            all_v_frames, dummy_path, hz=hz, 
                            metadata_list=all_metadata, scores=all_scores
                        )
                        if tiled_fs: robot_tiled_frames[(robot, active_arm, cue)] = tiled_fs
                        else: robot_tiled_frames[(robot, active_arm, cue)] = [f.convert("RGB") for f in all_v_frames[0]]
                        if os.path.exists(dummy_path):
                            os.remove(dummy_path)
                else:
                    robot_tiled_frames[(robot, active_arm, cue)] = [f.convert("RGB") for f in all_v_frames[0]]
                
                all_v_frames.clear()
            
            generator.reset_to_initial_state()
                
        except Exception as e:
            robot_errors.append((robot, active_arm, cue, str(e)))
            try:
                generator.reset_to_initial_state()
            except:
                pass

    generator.close()
    return robot, active_arm, robot_tiled_frames, robot_errors, robot_metadata

def main(
    robots=["GR1"],  # Full GR1 with legs (not fixed, not floating - just standard)
    arms=["right"],
    cues="iconic", 
    jsonl_path=None,
    verbose=False, 
    start_index=None, 
    save_separately=True, 
    hz=4, 
    top_k=20, 
    proximal_degree_scale=0.25, 
    num_workers=1,
    cues_yml="data/seed/yml/cues.yml",
    best_configs_path="data/seed/_remainder/best_configs.json",
    auto_generate_missing=True,
    meta_data_path="data/motions/meta_humanoid_motions.jsonl"
    ):
    """
    Generate humanoid motions for multiple robots, arms, and cues.
    
    Configs are loaded from best_configs.json, matched by cue name.
    
    Args:
        robots: List of humanoid robot names
        arms: List of arms ("right", "left", or both)
        cues: "iconic", "contextual", "all", or custom list
        jsonl_path: Path to poses JSONL (None = auto-detect)
        verbose: Print detailed logs
        start_index: Limit cue range (e.g., "0-10" or "5")
        save_separately: Save individual GIFs
        hz: Frame rate
        top_k: Number of variations to generate
        proximal_degree_scale: Scale for proximal joints
        num_workers: Number of parallel workers
        cues_yml: Path to cues YAML
        best_configs_path: Path to best_configs.json
        auto_generate_missing: Auto-generate missing configs
        meta_data_path: Path to metadata JSONL
    
    Examples:
        # Generate iconic cues for GR1 right arm
        python adhoc/humanoid/meta_generate_humanoid_motions.py \
            --robots='["GR1"]' --arms='["right"]' --cues="iconic"
        
        # Both arms
        python adhoc/humanoid/meta_generate_humanoid_motions.py \
            --robots='["GR1"]' --arms='["right","left"]' --cues="contextual"
    """
    def get_cues_from_yml(yml_path, category=None):
        if not os.path.exists(yml_path): return []
        with open(yml_path, 'r', encoding='utf-8') as f:
            cues_dict = yaml.safe_load(f)
        if category and category in cues_dict:
            return list(cues_dict[category].keys())
        all_cues = []
        for cat_cues in cues_dict.values():
            if isinstance(cat_cues, dict):
                all_cues.extend(cat_cues.keys())
        return all_cues
    
    def get_cues_from_best_configs(p):
        """Get cue names from best_configs.json."""
        if not os.path.exists(p): return []
        with open(p, 'r', encoding='utf-8') as f: configs = json.load(f)
        return [c.get('cue') for c in configs if 'cue' in c]
    
    def ensure_config_in_best_configs(cue_name, best_configs_path):
        """Check if cue exists in best_configs.json; auto-generate if missing."""
        if os.path.exists(best_configs_path):
            with open(best_configs_path, 'r', encoding='utf-8') as f:
                configs = json.load(f)
            if any(c.get('cue') == cue_name for c in configs):
                return True
        else:
            configs = []
        
        print(f"  Config for '{cue_name}' not found in best_configs.json. Generating...")
        try:
            # Generate to a temp motion_config, then copy to best_configs
            temp_config_path = "data/caches/_temp_motion_config.json"
            if not os.path.exists(temp_config_path):
                with open(temp_config_path, 'w') as f: json.dump([], f)
            generate_motion_config(
                cue_name=cue_name,
                target_json=temp_config_path,
                overwrite_existing=False,
                reindex=True
            )
            with open(temp_config_path, 'r', encoding='utf-8') as f:
                temp_configs = json.load(f)
            for mc in temp_configs:
                if mc.get('cue') == cue_name:
                    configs.append(mc)
                    with open(best_configs_path, 'w', encoding='utf-8') as f:
                        json.dump(configs, f, indent=2, ensure_ascii=False)
                    print(f"  ✓ Added '{cue_name}' to best_configs.json")
                    return True
            print(f"  ✗ Failed to generate config for '{cue_name}'")
            return False
        except Exception as e:
            print(f"  ✗ Error generating config for '{cue_name}': {e}")
            return False
    
    # Parse arms
    if isinstance(arms, str):
        arm_list = [a.strip() for a in arms.split(',')]
    else:
        arm_list = list(arms)
    
    # Parse robots
    if isinstance(robots, str):
        robot_list = [r.strip() for r in robots.split(',')]
    else:
        robot_list = list(robots)
    
    # Parse cues — always use cue names matched against best_configs.json
    if cues in ["iconic", "contextual"]:
        print(f"Loading cues from '{cues}' category in {cues_yml}...")
        cue_names = get_cues_from_yml(cues_yml, category=cues)
        print(f"Found {len(cue_names)} cues in '{cues}' category")
        
        if auto_generate_missing:
            print(f"Ensuring all configs exist in best_configs.json...")
            valid_cues = []
            for cue_name in cue_names:
                if ensure_config_in_best_configs(cue_name, best_configs_path):
                    valid_cues.append(cue_name)
            cues = valid_cues
        else:
            cues = cue_names
    elif cues == "all":
        cues = get_cues_from_best_configs(best_configs_path)
    
    if start_index is not None:
        if '-' in str(start_index):
            s, e = map(int, str(start_index).split('-'))
            cues = cues[s:e]
        else:
            s = int(start_index)
            cues = [cues[s]]

    print(f"\n{'='*80}")
    print(f"Starting execution for {len(robot_list)} robots, {len(arm_list)} arms, and {len(cues)} cues")
    print(f"  Robots: {', '.join(robot_list)}")
    print(f"  Arms: {', '.join(arm_list)}")
    print(f"  Total combinations: {len(robot_list) * len(arm_list) * len(cues)}")
    print(f"{'='*80}\n")
    
    # Build cue_name → yml index mapping (across all categories in cues.yml)
    cue_yml_indices = {}
    if os.path.exists(cues_yml):
        with open(cues_yml, 'r', encoding='utf-8') as f:
            all_yml = yaml.safe_load(f) or {}
        global_idx = 0
        for cat_cues in all_yml.values():
            if isinstance(cat_cues, dict):
                for cue_name in cat_cues.keys():
                    cue_yml_indices[cue_name] = global_idx
                    global_idx += 1

    all_tiled_frames, failed_errors, all_metadata = {}, [], []
    
    # Create list of (robot, arm) combinations
    robot_arm_combos = [(robot, arm) for robot in robot_list for arm in arm_list]
    
    worker_fn = partial(
        process_single_robot_arm, 
        cues=cues, jsonl_path=jsonl_path, config_path=best_configs_path, hz=hz, top_k=top_k, 
        proximal_degree_scale=proximal_degree_scale, verbose=verbose, save_separately=save_separately,
        cue_yml_indices=cue_yml_indices
    )

    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            worker_results = list(tqdm(pool.starmap(worker_fn, robot_arm_combos), total=len(robot_arm_combos), desc="Processing"))
            for r, arm, res_frames, err, meta in worker_results:
                all_tiled_frames.update(res_frames)
                failed_errors.extend(err)
                all_metadata.extend(meta)
    else:
        for robot, arm in tqdm(robot_arm_combos, desc="Processing"):
            _, _, res_frames, err, meta = worker_fn(robot, arm)
            all_tiled_frames.update(res_frames)
            failed_errors.extend(err)
            all_metadata.extend(meta)

    success_count = len(robot_list) * len(arm_list) * len(cues) - len(failed_errors)
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Success: {success_count}, Failed: {len(failed_errors)}")
    if failed_errors:
        for r, arm, c, err in failed_errors: 
            print(f"  - [{r} | {arm} | {c}]: {err}")
    print(f"{'='*80}\n")
    
    # Save metadata
    if all_metadata:
        os.makedirs(os.path.dirname(meta_data_path), exist_ok=True)
        print(f"Saving metadata to {meta_data_path}...")
        with open(meta_data_path, 'a', encoding='utf-8') as f:
            for meta in all_metadata:
                f.write(json.dumps(meta, ensure_ascii=False) + '\n')
        print(f"✓ Saved {len(all_metadata)} metadata entries to {meta_data_path}")
    
    all_tiled_frames.clear()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    fire.Fire(main)
