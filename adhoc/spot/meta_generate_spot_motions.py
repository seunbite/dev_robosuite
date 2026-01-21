"""
Meta script to generate multiple Spot robot motions in batch.

Usage:
    # Generate all cues for SpotWithArmFloating
    python adhoc/spot/meta_generate_spot_motions.py --robot SpotWithArmFloating
    
    # Generate specific cues
    python adhoc/spot/meta_generate_spot_motions.py \
        --robot SpotWithArmFloating \
        --cues "waving,beckoning,pointing"
"""

import fire
import os
import json


def meta_generate_spot_motions(
    robot: str = "SpotWithArmFloating",
    env: str = "EmptySpace",
    controller: str = "OSC_POSE",
    cues: str = None,
    config_path: str = "data/seed/motion_config.json",
    proximal_degree_scale: float = 0.25,
    camera_distance: float = 2.2,
    hz: int = 4,
):
    """
    Generate multiple Spot robot motions in batch.
    
    Args:
        robot: Robot name
        env: Environment name
        controller: Controller type
        cues: Comma-separated list of cues (if None, uses all from config)
        config_path: Path to motion config
        proximal_degree_scale: Scale factor for proximal joints
        camera_distance: Camera FOV multiplier
        hz: Frame rate
    
    Examples:
        # All cues
        python adhoc/spot/meta_generate_spot_motions.py --robot SpotWithArmFloating
        
        # Specific cues
        python adhoc/spot/meta_generate_spot_motions.py \
            --robot SpotWithArmFloating \
            --cues "waving,beckoning,pointing,nodding_substitute"
    """
    print("="*80)
    print("META SPOT MOTION GENERATION")
    print("="*80)
    print(f"Robot: {robot}")
    print(f"Controller: {controller}")
    print("="*80)
    
    # Load motion config to get available cues
    if not os.path.exists(config_path):
        print(f"Error: Motion config not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        motion_configs = json.load(f)
    
    # Get cue list
    if cues:
        # Parse comma-separated cues
        if isinstance(cues, str):
            cue_list = [c.strip() for c in cues.split(",")]
        elif isinstance(cues, (list, tuple)):
            cue_list = [str(c).strip() for c in cues]
        else:
            cue_list = [str(cues).strip()]
    else:
        # Use all cues from config
        cue_list = [config['cue'] for config in motion_configs]
    
    print(f"\nCues to generate: {len(cue_list)}")
    print(f"  {', '.join(cue_list)}")
    print(f"\nTotal motions: {len(cue_list)}")
    print("="*80)
    
    # Generate motions
    success_count = 0
    fail_count = 0
    
    for cue in cue_list:
        print(f"\n{'='*60}")
        print(f"Generating: {robot} | {cue}")
        print(f"{'='*60}")
        
        # Build command
        cmd = f"python adhoc/spot/motion_generation_spot.py"
        cmd += f" --robot {robot}"
        cmd += f" --env {env}"
        cmd += f" --cue {cue}"
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
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total motions: {len(cue_list)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print("="*80)


if __name__ == "__main__":
    fire.Fire(meta_generate_spot_motions)
