"""
Meta script to generate motions with all personalities from personality_list.json using Gemini API.
"""

import os
import json
import sys
from motion_generation_with_persona import generate_with_persona
import fire


def main(
    robots: list[str] = ["IIWA", "Panda", "Sawyer", "Kinova3", "Jaco", "UR5e", "XArm7"],
    cues: list[str] = "all",
    personalities: list[str] = "all",
    config_path: str = "data/seed/motion_config.json",
    personality_list_path: str = "data/seed/personality_list.json",
    jsonl_path: str = "data/poses/closest_poses_results.jsonl",
    gemini_api_key: str = None,
    add_transitions: bool = True,
    transition_probability: float = 0.5,
):
    """
    Generate motions for all combinations of robots, cues, and personalities using Gemini API.
    
    Args:
        robots: List of robot names
        cues: List of cue names or "all"
        personalities: List of personality names or "all"
        config_path: Path to motion config file
        personality_list_path: Path to personality list JSON
        jsonl_path: Path to pose database
        gemini_api_key: Gemini API key (if None, uses hardcoded response for "Sad")
        add_transitions: Whether to add personality-based transitions
        transition_probability: Probability of adding transitions
    """
    # Load cues from config file
    def get_cues_from_config(config_path: str = "data/seed/motion_config.json"):
        """Load cue names from motion_config.json."""
        if not os.path.exists(config_path):
            print(f"Warning: Config file not found: {config_path}")
            return []
        
        with open(config_path, 'r') as f:
            configs = json.load(f)
        
        cues = [config.get('cue') for config in configs if 'cue' in config]
        return cues
    
    # Load personalities from personality list
    def get_personalities_from_file(personality_list_path: str):
        """Load personality names from personality_list.json."""
        if not os.path.exists(personality_list_path):
            print(f"Warning: Personality list not found: {personality_list_path}")
            return []
        
        with open(personality_list_path, 'r') as f:
            personality_dict = json.load(f)
        
        return list(personality_dict.keys())
    
    # Get cues
    if cues == "all":
        cues = get_cues_from_config(config_path=config_path)
        print(f"Found {len(cues)} cues")
    
    # Get personalities
    if personalities == "all":
        personalities = get_personalities_from_file(personality_list_path)
        print(f"Found {len(personalities)} personalities")
    
    # Track success/failure counts
    total_success = 0
    total_failed = 0
    
    # Track results per personality
    personality_stats = {p: {"success": 0, "failed": 0} for p in personalities}
    
    for personality in personalities:
        print(f"\n{'='*80}")
        print(f"PERSONALITY: {personality}")
        print(f"{'='*80}\n")
        
        for cue in cues:
            for robot in robots:
                print(f"\n{'='*60}")
                print(f"Processing: {robot} - {cue} - {personality}")
                print(f"{'='*60}\n")
                
                try:
                    generate_with_persona(
                        robot=robot,
                        cue=cue,
                        personality=personality,
                        personality_list_path=personality_list_path,
                        gemini_api_key=gemini_api_key,
                        jsonl_path=jsonl_path,
                        config_path=config_path,
                        add_transitions=add_transitions,
                        transition_probability=transition_probability,
                    )
                    total_success += 1
                    personality_stats[personality]["success"] += 1
                    print(f"✓ Success for {robot} - {cue} - {personality}")
                except Exception as e:
                    total_failed += 1
                    personality_stats[personality]["failed"] += 1
                    print(f"✗ Failed for {robot} - {cue} - {personality}: {e}")
                except SystemExit:
                    total_failed += 1
                    personality_stats[personality]["failed"] += 1
                    print(f"✗ Failed for {robot} - {cue} - {personality} (system exit)")
    
    # Print summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY:")
    print(f"  Total Success: {total_success}")
    print(f"  Total Failed: {total_failed}")
    print(f"  Total: {total_success + total_failed}")
    print(f"{'='*80}\n")
    
    print(f"\n{'='*80}")
    print("PER-PERSONALITY SUMMARY:")
    print(f"{'='*80}")
    for personality, stats in sorted(personality_stats.items()):
        total = stats["success"] + stats["failed"]
        if total > 0:
            success_rate = (stats["success"] / total) * 100
            print(f"{personality:20s}: {stats['success']:3d} / {total:3d} ({success_rate:5.1f}%)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    fire.Fire(main)
