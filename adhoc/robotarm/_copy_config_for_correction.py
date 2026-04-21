"""
Copy a config for manual correction.

This script copies an existing motion config and marks it as "corrected",
allowing you to manually edit it in the JSON file.
"""

import os
import json
import fire


def copy_for_correction(
    cue_index: int,
    target_json: str = "data/seed/motion_config.json",
):
    """
    Copy a config at the given index for manual correction.
    
    Args:
        cue_index: Index of the config to copy
        target_json: Path to motion config JSON file
    """
    if not os.path.exists(target_json):
        raise FileNotFoundError(f"Config file not found: {target_json}")
    
    # Load existing configs
    with open(target_json, 'r', encoding='utf-8') as f:
        configs = json.load(f)
    
    # Find config by index
    source_config = None
    for config in configs:
        if config.get('idx') == cue_index:
            source_config = config
            break
    
    if source_config is None:
        raise ValueError(f"Config with idx={cue_index} not found")
    
    # Find the position of source config in the list
    source_position = None
    for i, config in enumerate(configs):
        if config.get('idx') == cue_index:
            source_position = i
            break
    
    # Create a copy
    import copy
    new_config = copy.deepcopy(source_config)
    
    # Set state to "corrected"
    original_state = source_config.get('state', 'handmade')
    new_config['state'] = 'corrected'
    
    # Update description to indicate it's corrected
    original_desc = new_config.get('description', '')
    if original_desc:
        new_config['description'] = original_desc
    else:
        new_config['description'] = ""
    
    # Insert right after the source config
    configs.insert(source_position + 1, new_config)
    
    # Reorder all indices from 0
    for i, config in enumerate(configs):
        config['idx'] = i
    
    # Save back to file
    with open(target_json, 'w', encoding='utf-8') as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)
    
    new_idx = new_config['idx']
    
    print(f"\n{'='*60}")
    print("CONFIG COPIED FOR CORRECTION")
    print(f"{'='*60}")
    print(f"Source config:")
    print(f"  - idx: {cue_index} → {cue_index} (reordered)")
    print(f"  - cue: {source_config.get('cue')}")
    print(f"  - state: {original_state}")
    print(f"\nNew config created (inserted right after source):")
    print(f"  - idx: {new_idx}")
    print(f"  - cue: {new_config.get('cue')}")
    print(f"  - state: corrected")
    print(f"\n✅ All {len(configs)} configs reindexed (0 to {len(configs)-1})")
    print(f"\n📝 Please manually edit the config at idx={new_idx} in:")
    print(f"   {target_json}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    fire.Fire(copy_for_correction)
