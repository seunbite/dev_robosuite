import os
import json
import sys
from motion_generation import generate, MotionGenerator
import fire

def main(
    robots: list[str] = ["IIWA", "Panda", "Sawyer", "Kinova3", "Jaco", "UR5e", "XArm7"],
    cues: list[str] = "all", # all
    config_path: str = "data/seed/motion_config.json",
    all_poses: bool = False,
    jsonl_path: str = "data/poses/closest_poses_results.jsonl",
):
    # Load cues from config file
    def get_cues_from_config(config_path: str = "data/seed/motion_config.json"):
        """Load cue names from motion_config.json."""
        if not os.path.exists(config_path):
            print(f"Warning: Config file not found: {config_path}")
            return []
        
        with open(config_path, 'r') as f:
            configs = json.load(f)
        
        # Extract cue names
        cues = [config.get('cue') for config in configs if 'cue' in config]
        return cues
    
    if cues == "all":
        cues = get_cues_from_config(config_path=config_path)
        print(f"Found {len(cues)} cues")
    
    # Track success/failure counts
    total_success = 0
    total_failed = 0
    
    for cue in cues:
        for robot in robots:
            print(f"\n{'='*60}")
            print(f"Processing: {robot} - {cue}")
            print(f"{'='*60}\n")
            
            if all_poses:
                # Get all matching poses for the first pose in the cue
                # Load cue config to get first pose name
                with open(config_path, 'r') as f:
                    configs = json.load(f)
                
                cue_config = None
                for config in configs:
                    if config.get('cue') == cue:
                        cue_config = config
                        break
                
                if cue_config is None:
                    print(f"Warning: Cue '{cue}' not found in config file")
                    continue
                
                # Find first pose in movements
                movements = cue_config.get('movements', [])
                first_pose_def = None
                for movement in movements:
                    if movement.get('type') == 'pose':
                        first_pose_def = movement.get('parameters', {}).get('pose')
                        break
                
                if first_pose_def is None:
                    print(f"Warning: No pose found in cue '{cue}'")
                    continue
                
                # Create display name for logging
                if isinstance(first_pose_def, str):
                    first_pose_display = first_pose_def
                elif isinstance(first_pose_def, dict):
                    first_pose_display = f"{first_pose_def.get('height', '?')}_{first_pose_def.get('dir', '?')}_{first_pose_def.get('pitch', '?')}"
                else:
                    first_pose_display = str(first_pose_def)
                
                # Get matching poses using MotionGenerator
                generator = MotionGenerator(
                    robot_name=robot,
                    env_name="EmptySpace",
                    controller_name="IK_POSE",
                    jsonl_path=jsonl_path,
                    has_renderer=False,
                    has_offscreen_renderer=True,
                )
                
                try:
                    matching_poses = generator._find_matching_poses(first_pose_def)
                    generator.close()
                except Exception as e:
                    print(f"Error getting matching poses: {e}")
                    generator.close()
                    continue
                
                if not matching_poses:
                    print(f"Warning: No matching poses found for '{first_pose_display}'")
                    continue
                
                print(f"Found {len(matching_poses)} matching poses for '{first_pose_display}'")
                print(f"Trying all {len(matching_poses)} poses...\n")
                
                # Try each pose
                for pose_idx, pose in enumerate(matching_poses):
                    pose_id = pose.get('pose_id')
                    print(f"\n[{pose_idx + 1}/{len(matching_poses)}] Trying pose_id {pose_id} (rank {pose.get('rank', 'N/A')})")
                    
                    try:
                        generate(robot=robot, cue=cue, pose_index=pose_id, jsonl_path=jsonl_path, config_path=config_path)
                        total_success += 1
                        print(f"✓ Success for pose_id {pose_id}")
                    except Exception as e:
                        total_failed += 1
                        print(f"✗ Failed for pose_id {pose_id}: {e}")
                    except SystemExit:
                        total_failed += 1
                        print(f"✗ Failed for pose_id {pose_id} (system exit)")
            else:
                # Original behavior: single random execution
                try:
                    generate(robot=robot, cue=cue, jsonl_path=jsonl_path, config_path=config_path)
                    total_success += 1
                    print(f"✓ Success for {robot} - {cue}")
                except Exception as e:
                    total_failed += 1
                    print(f"✗ Failed for {robot} - {cue}: {e}")
                except SystemExit:
                    total_failed += 1
                    print(f"✗ Failed for {robot} - {cue} (system exit)")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Success: {total_success}")
    print(f"  Failed: {total_failed}")
    print(f"  Total: {total_success + total_failed}")
    print(f"{'='*60}\n")
    
    
if __name__ == "__main__":
    fire.Fire(main)
