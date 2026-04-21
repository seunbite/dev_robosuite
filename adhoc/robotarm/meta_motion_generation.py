import os
import json
import sys
import logging
import numpy as np
from tqdm import tqdm
from motion_generation import generate, MotionGenerator
import fire
from contextlib import contextmanager
from PIL import Image, ImageDraw, ImageFont
import math
import multiprocessing
from functools import partial
from datetime import datetime
import yaml

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

def _frame_copy(f):
    """Return an independent RGB copy of a frame so GIF saves don't share state."""
    if f.mode != "RGB":
        f = f.convert("RGB")
    return f.copy()


def combine_variations_horizontally(frames_list: list[list[Image.Image]], output_path: str, hz: int = 10, metadata_list: list = None, scores: list = None):
    """Combine multiple variation frame lists side-by-side into a single high-quality GIF.
    Each save is independent: uses copies of all frames and a per-GIF palette.
    """
    if not frames_list or not frames_list[0]: return None, None
    
    max_frames = max(len(fs) for fs in frames_list)
    tile_w, tile_h = frames_list[0][0].size
    num_vars = len(frames_list)
    total_w, total_h = num_vars * tile_w, tile_h
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        bold_font = ImageFont.truetype("/System/Library/Fonts/Helvetica-Bold.ttc", 16)
    except:
        font, bold_font = ImageFont.load_default(), ImageFont.load_default()

    combined_frames = []
    for i in range(max_frames):
        new_frame = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(new_frame)
        
        for v_idx, fs in enumerate(frames_list):
            x_offset = v_idx * tile_w
            frame = _frame_copy(fs[i % len(fs)])
            if frame.size != (tile_w, tile_h): 
                frame = frame.resize((tile_w, tile_h))
            new_frame.paste(frame, (x_offset, 0))
            # Don't close frame here - it might be referenced by new_frame
            
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
                    
                    draw.rectangle([x_offset + 5, 5, x_offset + 80, 25], fill=(255, 255, 255, 200))
                    draw.text((x_offset + 10, 7), v_label, fill="black", font=font)
                    
                    score_str = f"Match: {score*100:.1f}%"
                    sw = draw.textlength(score_str, font=font) if hasattr(draw, 'textlength') else 80
                    draw.rectangle([x_offset + tile_w - sw - 15, 5, x_offset + tile_w - 5, 25], fill=(0, 128, 0, 200) if score > 0.8 else (128, 0, 0, 200))
                    draw.text((x_offset + tile_w - sw - 10, 7), score_str, fill="white", font=font)
                    
                    tw = draw.textlength(label_str, font=bold_font) if hasattr(draw, 'textlength') else 150
                    tx = x_offset + (tile_w - tw) // 2
                    draw.rectangle([tx - 5, 28, tx + tw + 5, 52], fill=(0, 0, 0, 180))
                    draw.text((tx, 30), label_str, fill="cyan" if score > 0.8 else "yellow", font=bold_font)
            
            if v_idx > 0: draw.line([(x_offset, 0), (x_offset, total_h)], fill="white", width=2)
        combined_frames.append(new_frame)
    
    if combined_frames:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Per-GIF palette from copies of this GIF's frames only (no shared state)
        # Speed-optimized: reduced samples, FASTOCTREE, no dithering
        sample_count = min(len(combined_frames), 50)
        sample_indices = np.unique(np.linspace(0, len(combined_frames) - 1, sample_count, dtype=int))
        palette_combined = Image.new("RGB", (total_w, total_h * len(sample_indices)))
        for idx_s, idx_f in enumerate(sample_indices):
            palette_combined.paste(combined_frames[idx_f].copy(), (0, idx_s * total_h))
        palette_img = palette_combined.quantize(colors=256, method=Image.FASTOCTREE, dither=Image.NONE)
        
        # Quantize all frames BEFORE deleting palette
        quantized_frames = [f.copy().quantize(palette=palette_img, dither=Image.NONE) for f in combined_frames]
        
        # Save GIF
        quantized_frames[0].save(
            output_path,
            save_all=True,
            append_images=quantized_frames[1:],
            duration=int(1000 / hz),
            loop=0,
            disposal=1,
            optimize=False
        )
        
        # Clean up AFTER saving, in correct order
        for f in quantized_frames:
            try:
                f.close()
            except:
                pass
        del quantized_frames
        del palette_img
        del palette_combined
        for f in combined_frames:
            try:
                f.close()
            except:
                pass
        combined_frames.clear()
        
        # Return path only, no frame copies (save memory)
        return output_path, None
    return None, None

def combine_gifs_grid(results_frames: dict, robots: list[str], cues: list, output_path: str, hz: int = 10):
    """Combine multiple frame sequences into a large grid GIF.
    
    Args:
        cues: List of cue names (str) or (idx, cue_name) tuples
    """
    if not results_frames: return
    
    # Extract cue names if they are tuples
    cue_names = [c[1] if isinstance(c, tuple) else c for c in cues]
    
    max_frames = 1
    tile_w, tile_h = 0, 0
    for frames in results_frames.values():
        if frames:
            max_frames = max(max_frames, len(frames))
            if tile_w == 0: tile_w, tile_h = frames[0].size
            
    if tile_w == 0: return

    header_h, header_w = 60, 150
    total_w = header_w + (len(cue_names) * tile_w)
    total_h = header_h + (len(robots) * tile_h)
    
    try: font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except: font = ImageFont.load_default()

    combined_frames = []
    for i in range(max_frames):
        new_frame = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(new_frame)
        for c_idx, cue_name in enumerate(cue_names):
            x = header_w + c_idx * tile_w
            tw = draw.textlength(cue_name, font=font) if hasattr(draw, 'textlength') else 100
            draw.text((x + (tile_w - tw)//2, (header_h - 20)//2), cue_name, fill="black", font=font)
            draw.line([(x, 0), (x, total_h)], fill="lightgrey", width=1)
        for r_idx, robot in enumerate(robots):
            y_cell = header_h + r_idx * tile_h
            draw.text((10, y_cell + (tile_h - 20)//2), robot, fill="black", font=font)
            draw.line([(0, y_cell), (total_w, y_cell)], fill="lightgrey", width=1)
            for c_idx, cue_name in enumerate(cue_names):
                x_cell = header_w + c_idx * tile_w
                fs = results_frames.get((robot, cue_name))
                if fs:
                    fr = _frame_copy(fs[i % len(fs)])
                    if fr.size != (tile_w, tile_h): fr = fr.resize((tile_w, tile_h))
                    new_frame.paste(fr, (x_cell, y_cell))
                else: draw.rectangle([x_cell, y_cell, x_cell + tile_w, y_cell + tile_h], fill=(200, 200, 200))
        combined_frames.append(new_frame)

    if combined_frames:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Speed-optimized: reduced samples, FASTOCTREE, no dithering
        sample_count = min(len(combined_frames), 50)
        sample_indices = np.unique(np.linspace(0, len(combined_frames) - 1, sample_count, dtype=int))
        palette_combined = Image.new("RGB", (total_w, total_h * len(sample_indices)))
        for idx_s, idx_f in enumerate(sample_indices):
            palette_combined.paste(combined_frames[idx_f].copy(), (0, idx_s * total_h))
        palette_img = palette_combined.quantize(colors=256, method=Image.FASTOCTREE, dither=Image.NONE)
        quantized_frames = [f.copy().quantize(palette=palette_img, dither=Image.NONE) for f in combined_frames]
        del palette_combined, palette_img

        quantized_frames[0].save(
            output_path,
            save_all=True,
            append_images=quantized_frames[1:],
            duration=int(1000 / hz),
            loop=0,
            disposal=1,
            optimize=False
        )

def update_pose_info_yaml(robot: str, cue: str, pose_data: dict, pose_info_file: str = "data/motions/initial_pose_info.yml"):
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
        
        # Update with new cue data
        if robot not in existing_data:
            existing_data[robot] = {}
        
        # Preserve existing best_manual_index if it exists
        if cue in existing_data[robot] and 'best_manual_index' in existing_data[robot][cue]:
            pose_data['best_manual_index'] = existing_data[robot][cue]['best_manual_index']
        
        existing_data[robot][cue] = pose_data
        
        # Save back to YAML
        os.makedirs(os.path.dirname(pose_info_file), exist_ok=True)
        with open(pose_info_file, 'w', encoding='utf-8') as f:
            yaml.dump(existing_data, f,
                     default_flow_style=False,
                     allow_unicode=True,
                     sort_keys=True,
                     indent=2)

def combine_gifs_single_robot(results_frames: dict, robot: str, cues: list, output_path: str, hz: int = 10):
    """Combine all cues for a single robot into a vertical summary GIF.
    
    Args:
        cues: List of cue names (str) or (idx, cue_name) tuples
    """
    # Extract cue names if they are tuples
    cue_names = [c[1] if isinstance(c, tuple) else c for c in cues]
    
    robot_cues_frames = {c: results_frames.get((robot, c)) for c in cue_names if (robot, c) in results_frames and results_frames[(robot, c)]}
    if not robot_cues_frames: return
    
    valid_cues = list(robot_cues_frames.keys())
    max_frames = max(len(fs) for fs in robot_cues_frames.values())
    tile_w, tile_h = robot_cues_frames[valid_cues[0]][0].size
    
    header_w = 200
    total_w, total_h = header_w + tile_w, len(valid_cues) * tile_h
    try: font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except: font = ImageFont.load_default()

    combined_frames = []
    for i in range(max_frames):
        new_frame = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(new_frame)
        for c_idx, cue in enumerate(valid_cues):
            y_offset = c_idx * tile_h
            draw.text((20, y_offset + (tile_h // 2) - 10), cue, fill="black", font=font)
            draw.line([(0, y_offset), (total_w, y_offset)], fill="lightgrey", width=1)
            fs = robot_cues_frames[cue]
            fr = _frame_copy(fs[i % len(fs)])
            if fr.size != (tile_w, tile_h): fr = fr.resize((tile_w, tile_h))
            new_frame.paste(fr, (header_w, y_offset))
        combined_frames.append(new_frame)

    if combined_frames:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Speed-optimized: reduced samples, FASTOCTREE, no dithering
        sample_count = min(len(combined_frames), 50)
        sample_indices = np.unique(np.linspace(0, len(combined_frames) - 1, sample_count, dtype=int))
        palette_combined = Image.new("RGB", (total_w, total_h * len(sample_indices)))
        for idx_s, idx_f in enumerate(sample_indices):
            palette_combined.paste(combined_frames[idx_f].copy(), (0, idx_s * total_h))
        palette_img = palette_combined.quantize(colors=256, method=Image.FASTOCTREE, dither=Image.NONE)
        quantized_frames = [f.copy().quantize(palette=palette_img, dither=Image.NONE) for f in combined_frames]
        del palette_combined, palette_img

        quantized_frames[0].save(
            output_path,
            save_all=True,
            append_images=quantized_frames[1:],
            duration=int(1000/hz),
            loop=0,
            disposal=1,
            optimize=False
        )

def _run_one_cue(generator, robot, cue_item, config_path, hz, top_k, proximal_degree_scale, need_frames, save_separately, verbose):
    """Run a single (robot, cue) and return (tiled_frames_entry, error, metadata_list, pose_info_entry)."""
    if isinstance(cue_item, tuple):
        if len(cue_item) == 3:
            _, cue, yml_index = cue_item
        else:
            _, cue = cue_item
            yml_index = None
    else:
        cue, yml_index = cue_item, None

    try:
        with suppress_stdout(enable=not verbose):
            save_individual = save_separately and top_k == 1
            res_dict = generator.execute_cue(
                cue=cue,
                config_path=config_path,
                hz=hz,
                top_k=top_k,
                proximal_degree_scale=proximal_degree_scale,
                save_gifs=save_individual,
                return_frames=need_frames,
            )

        all_v_frames = res_dict.get("frames", [])
        all_metadata = res_dict.get("metadata", [])
        all_scores = res_dict.get("scores", [])
        all_paths = res_dict.get("paths", [])

        initial_pose_ids = []
        robot_metadata = []
        for idx, (meta, score) in enumerate(zip(all_metadata, all_scores)):
            pose_id = meta.get("pose_id")
            initial_pose_ids.append(pose_id)
            motion_meta = {
                "robot": robot,
                "cue": cue,
                "config_idx": yml_index,
                "pose_id": pose_id,
                "candidate_idx": idx + 1,
                "min_alignment": score,
                "gif_path": all_paths[idx] if idx < len(all_paths) else None,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            robot_metadata.append(motion_meta)

        pose_info_entry = None
        if initial_pose_ids:
            pose_data = {'initial_indexes': initial_pose_ids, 'best_manual_index': None}
            pose_info_entry = (robot, cue, pose_data)
            try:
                update_pose_info_yaml(robot, cue, pose_data)
                print(f"  ✓ Updated YAML for {robot}/{cue}", flush=True)
            except Exception as e:
                print(f"  ⚠ Failed to update YAML for {robot}/{cue}: {e}", flush=True)

        tiled_entry = None
        if need_frames:
            if not all_v_frames:
                raise ValueError("No frames generated")
            if top_k > 1 and len(all_v_frames) > 1:
                if save_separately:
                    # Save to file but don't keep frames in memory
                    yml_prefix = f"p{yml_index}" if yml_index is not None else ""
                    safe_cue = cue.replace(' ', '_').replace('(', '').replace(')', '').replace("'", "").replace("/", "")
                    tiled_filename = f"{datetime.now().strftime('%Y%m%d')}_{yml_prefix}_{robot}_{safe_cue}_tiled.gif"
                    tiled_path = os.path.join("data/motions", robot, tiled_filename)
                    os.makedirs(os.path.dirname(tiled_path), exist_ok=True)
                    print(f"  💾 Saving tiled GIF to {tiled_path}", flush=True)
                    try:
                        _, tiled_fs = combine_variations_horizontally(
                            all_v_frames, tiled_path, hz=hz,
                            metadata_list=all_metadata, scores=all_scores
                        )
                        if os.path.exists(tiled_path):
                            print(f"  ✓ Saved tiled GIF for {robot}/{cue}", flush=True)
                        else:
                            print(f"  ✗ Failed to save tiled GIF for {robot}/{cue} - file not found", flush=True)
                    except Exception as save_err:
                        print(f"  ✗ Error saving tiled GIF for {robot}/{cue}: {save_err}", flush=True)
                    # Don't store frames when saving separately (save memory)
                    tiled_entry = None
                else:
                    # Keep frames for grid/byrobot combining
                    safe_cue = cue.replace(' ', '_').replace('(', '').replace(')', '').replace("'", "").replace("/", "")
                    dummy_path = f"temp_tiled_{robot}_{safe_cue}.gif"
                    _, tiled_fs = combine_variations_horizontally(
                        all_v_frames, dummy_path, hz=hz,
                        metadata_list=all_metadata, scores=all_scores
                    )
                    tiled_entry = tiled_fs
                    if os.path.exists(dummy_path):
                        os.remove(dummy_path)
            else:
                tiled_entry = None  # Don't store single variation frames
            
            # Clean up all frames immediately
            for variation_frames in all_v_frames:
                for frame in variation_frames:
                    try:
                        frame.close()
                    except:
                        pass
            all_v_frames.clear()
        
        # Clean up other large data structures
        all_metadata.clear()
        all_scores.clear()
        all_paths.clear()
        res_dict.clear()

        generator.reset_to_initial_state()
        return (robot, cue), None, robot_metadata, pose_info_entry, tiled_entry
    except Exception as e:
        try:
            generator.reset_to_initial_state()
        except Exception:
            pass
        # Clean up on error
        try:
            if 'all_v_frames' in locals():
                for variation_frames in all_v_frames:
                    for frame in variation_frames:
                        try:
                            frame.close()
                        except:
                            pass
                all_v_frames.clear()
            if 'res_dict' in locals():
                res_dict.clear()
        except:
            pass
        return (robot, cue), (robot, cue, str(e)), [], None, None


def process_single_robot_command(robot, configs, jsonl_path, config_path, hz, top_k, proximal_degree_scale, verbose, save_separately, script_path, python_bin=None):
    """Process a list of configs for a single robot using os.system calls."""
    print(f"\n[Worker] Processing {len(configs)} configs for {robot}...", flush=True)
    
    if python_bin is None:
        python_bin = sys.executable
    robot_errors = []
    
    for config in tqdm(configs, desc=f"{robot}"):
        cue_idx = config.get('idx')
        cue_name = config.get('cue')
        
        if cue_idx is None:
            continue
            
        cmd = (
            f"\"{python_bin}\" \"{script_path}\" "
            f"--robot=\"{robot}\" "
            f"--cue_idx={cue_idx} "
            f"--config_path=\"{config_path}\" "
            f"--jsonl_path=\"{jsonl_path}\" "
            f"--hz={hz} "
            f"--top_k={top_k} "
            f"--proximal_degree_scale={proximal_degree_scale} "
        )
        
        if not save_separately:
             # If not saving separately, we might want to pass a flag, but motion_generation 
             # usually saves by default. We'll assume default behavior is what we want.
             pass

        if verbose:
            print(f"Running: {cmd}")
            ret = os.system(cmd)
            stderr_msg = ""
        else:
            import subprocess as sp
            proc = sp.run(cmd, shell=True, stdout=sp.DEVNULL, stderr=sp.PIPE, text=True)
            ret = proc.returncode
            stderr_msg = proc.stderr.strip() if proc.stderr else ""
        
        if ret != 0:
            err_detail = f"exit code {ret}"
            if stderr_msg:
                last_lines = "\n".join(stderr_msg.splitlines()[-3:])
                err_detail += f" | {last_lines}"
            robot_errors.append((robot, cue_name, err_detail))
            
    print(f"[Worker] Completed {robot}", flush=True)
    return robot, {}, robot_errors, [], {}

def main(
    robots=["IIWA", "Panda", "XArm7"], 
    best_configs_path="data/seed/motion_configs.json",
    jsonl_path="data/seed/closest_poses_results.jsonl", 
    verbose=False, 
    start_index=18, 
    end_index=None,
    save_separately=True, 
    hz=10, 
    top_k=10, 
    proximal_degree_scale=0.25, 
    num_workers=1,
    ):
    
    print(f"Starting execution for {len(robots)} robots.")
    
    # Load configs
    with open(best_configs_path, 'r') as f:
        all_configs = json.load(f)
    
    # Filter configs by index
    filtered_configs = []
    for cfg in all_configs:
        idx = cfg.get('idx')
        if idx is not None:
            if start_index is not None and int(idx) < start_index:
                continue
            if end_index is not None and int(idx) > end_index:
                continue
            filtered_configs.append(cfg)
            
    print(f"Loaded {len(filtered_configs)} configs (Index range: {start_index} - {end_index})")
    
    failed_errors = []
    
    # Determine script path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "motion_generation.py")
    
    worker_fn = partial(
        process_single_robot_command,
        configs=filtered_configs, 
        jsonl_path=jsonl_path, 
        config_path=best_configs_path, 
        hz=hz, 
        top_k=top_k,
        proximal_degree_scale=proximal_degree_scale, 
        verbose=verbose,
        save_separately=save_separately,
        script_path=script_path,
        python_bin=sys.executable,
    )

    if num_workers > 1:
        # Parallel: robot-first (each worker = one robot, all cues)
        with multiprocessing.Pool(num_workers) as pool:
            worker_results = list(tqdm(pool.imap(worker_fn, robots), total=len(robots), desc="Processing Robots"))
            for r, _, err, _, _ in worker_results:
                failed_errors.extend(err)
    else:
        # Single-threaded: robot-first (each robot, then all cues)
        for robot in tqdm(robots, desc="Processing Robots"):
            r, _, err, _, _ = worker_fn(robot)
            failed_errors.extend(err)

    print(f"\nSummary: Failed: {len(failed_errors)}")
    if failed_errors:
        for r, c, err in failed_errors: print(f"  - [{r} | {c}]: {err}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    fire.Fire(main)
