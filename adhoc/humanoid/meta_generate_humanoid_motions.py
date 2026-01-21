"""
Meta script to generate multiple humanoid motions with different personalities.

Automatically generates motions for:
- Multiple cues from motion_config.json
- Multiple personalities from personality_list.json
- Both arms (right and left)

Usage:
    python adhoc/humanoid/meta_generate_humanoid_motions.py --robot GR1ArmsOnly
    python adhoc/humanoid/meta_generate_humanoid_motions.py --robot GR1ArmsOnly --arms right
    python adhoc/humanoid/meta_generate_humanoid_motions.py --robot GR1FixedLowerBody --cues waving,beckoning
"""

import fire
import os
import json
import sys
from typing import Optional, List


def meta_generate_humanoid_motions(
    robot: str = "GR1ArmsOnly",
    arms: str = "right",  # Comma-separated list of arms
    cues: Optional[str] = None,  # Comma-separated list of cues (None = all from config)
    personalities: Optional[str] = None,  # Comma-separated list (None = all from list)
    config_path: str = "data/seed/motion_config.json",
    personality_list_path: str = "data/seed/personality_list.json",
    controller: str = "OSC_POSE",  # Controller type (OSC_POSE works for all robots)
    proximal_degree_scale: float = 0.25,
    camera_distance: float = 1.8,
    hz: int = 4,
):
    """
    Generate multiple humanoid motions with different personalities.
    
    Args:
        robot: Humanoid robot name (GR1ArmsOnly, GR1FixedLowerBody, etc.)
        arms: Comma-separated list of arms ("right", "left", or "right,left")
        cues: Comma-separated list of cues (None = all cues from motion_config.json)
        personalities: Comma-separated list of personalities (None = all from personality_list.json)
        config_path: Path to motion_config.json
        personality_list_path: Path to personality_list.json
        controller: Controller type (default: OSC_POSE, works for all robots)
        proximal_degree_scale: Scale factor for proximal joints
        camera_distance: Camera FOV multiplier
        hz: Frame rate for GIF
    
    Examples:
        # Generate all combinations
        python adhoc/humanoid/meta_generate_humanoid_motions.py --robot GR1ArmsOnly
        
        # Only right arm
        python adhoc/humanoid/meta_generate_humanoid_motions.py --robot GR1ArmsOnly --arms right
        
        # Specific cues
        python adhoc/humanoid/meta_generate_humanoid_motions.py \
            --robot GR1ArmsOnly --cues "waving,beckoning,pointing"
        
        # Specific personalities
        python adhoc/humanoid/meta_generate_humanoid_motions.py \
            --robot GR1ArmsOnly --personalities "Excited,Sad,Calm"
    """
    print("="*80)
    print("META HUMANOID MOTION GENERATOR")
    print("="*80)
    print(f"Robot: {robot}")
    print(f"Arms: {arms}")
    print("="*80)
    
    # Parse arms list
    if isinstance(arms, str):
        arm_list = [arm.strip() for arm in arms.split(",")]
    elif isinstance(arms, (list, tuple)):
        arm_list = [str(arm).strip() for arm in arms]
    else:
        arm_list = [str(arms).strip()]
    
    # Load cues
    if cues is None:
        # Load all cues from config
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            return
        
        with open(config_path, 'r') as f:
            motion_configs = json.load(f)
        cue_list = [config["cue"] for config in motion_configs]
    else:
        # Handle both string and tuple (from fire parsing)
        if isinstance(cues, str):
            cue_list = [cue.strip() for cue in cues.split(",")]
        elif isinstance(cues, (list, tuple)):
            cue_list = [str(cue).strip() for cue in cues]
        else:
            cue_list = [str(cues).strip()]
    
    # Load personalities
    if personalities is "all":
        # Load all personalities from list
        if not os.path.exists(personality_list_path):
            print(f"Error: Personality list not found: {personality_list_path}")
            return
        
        with open(personality_list_path, 'r') as f:
            personality_data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(personality_data, list):
            personality_list = [p["name"] for p in personality_data if isinstance(p, dict)]
        elif isinstance(personality_data, dict):
            # If it's a dict, use the keys as personality names
            personality_list = list(personality_data.keys())
        else:
            print(f"Error: Unexpected personality list format")
            return
        
    elif personalities is not None:
        # Handle both string and tuple (from fire parsing)
        if isinstance(personalities, str):
            personality_list = [p.strip() for p in personalities.split(",")]
        elif isinstance(personalities, (list, tuple)):
            personality_list = [str(p).strip() for p in personalities]
        else:
            personality_list = [str(personalities).strip()]
            
    else:
        personality_list = ["None"]
    
    print(f"\nCues to generate: {len(cue_list)}")
    print(f"  {', '.join(cue_list)}")
    print(f"\nPersonalities: {len(personality_list)}")
    print(f"  {', '.join(personality_list)}")
    print(f"\nArms: {len(arm_list)}")
    print(f"  {', '.join(arm_list)}")
    
    total_combinations = len(arm_list) * len(cue_list) * len(personality_list)
    print(f"\nTotal combinations: {total_combinations}")
    print("="*80)
    
    # Generate all combinations
    success_count = 0
    fail_count = 0
    
    
    for arm in arm_list:
        for cue in cue_list:
            for personality in personality_list:
                print(f"\n{'='*60}")
                print(f"Generating: {robot} | {arm} arm | {cue} | {personality}")
                print(f"{'='*60}")
                
                # Build command
                cmd = f"python adhoc/humanoid/motion_generation_humanoid.py"
                cmd += f" --robot {robot}"
                cmd += f" --active-arm {arm}"
                cmd += f" --cue {cue}"
                
                # Only add personality if not None
                if personality != "None":
                    cmd += f" --personality {personality}"
                
                cmd += f" --controller {controller}"
                cmd += f" --proximal-degree-scale {proximal_degree_scale}"
                cmd += f" --camera-distance {camera_distance}"
                cmd += f" --hz {hz}"
                
                # Execute
                ret = os.system(cmd)
                
                if ret == 0:
                    success_count += 1
                    print(f"✓ Success")
                else:
                    fail_count += 1
                    print(f"✗ Failed")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total combinations: {total_combinations}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print("="*80)


if __name__ == "__main__":
    fire.Fire(meta_generate_humanoid_motions)
