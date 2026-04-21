"""
Get recent/best motion configs for each unique cue.

Priority order:
1. handmade (highest priority)
2. corrected
3. Most recent fewshot (by 'time' field)

Ensures each cue appears only once in the output.
"""

import os
import json
import fire
from typing import List, Dict, Optional
from datetime import datetime


def get_recent_motions(
    config_path: str = "data/seed/motion_config.json",
    output_format: str = "idx",  # "idx", "json", "cue", "full"
    output_file: Optional[str] = None,
    verbose: bool = False
) -> List:
    """
    Get the best/most recent motion config for each unique cue.
    
    Args:
        config_path: Path to motion_config.json
        output_format: Output format
            - "idx": List of indices
            - "cue": List of cue names
            - "json": Save selected configs to JSON file
            - "full": Return full config objects
        output_file: Output file path (for "json" format)
        verbose: Print detailed information
    
    Returns:
        List of indices, cue names, or full configs depending on output_format
    """
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found: {config_path}")
    
    # Load all configs
    with open(config_path, 'r', encoding='utf-8') as f:
        all_configs = json.load(f)
    
    # Group configs by cue
    cue_groups = {}
    for config in all_configs:
        cue = config.get('cue')
        if not cue:
            continue
        
        if cue not in cue_groups:
            cue_groups[cue] = []
        cue_groups[cue].append(config)
    
    # Select best config for each cue
    selected_configs = []
    
    for cue, configs in sorted(cue_groups.items()):
        # Separate by state
        handmade = [c for c in configs if c.get('state') == 'handmade']
        corrected = [c for c in configs if c.get('state') == 'corrected']
        fewshot = [c for c in configs if c.get('state') == 'fewshot']
        zeroshot = [c for c in configs if c.get('state') == 'zeroshot']
        
        selected = None
        selection_reason = ""
        
        # Priority: handmade > corrected > most recent fewshot > most recent zeroshot
        if handmade:
            selected = handmade[0]  # Assume only one handmade per cue
            selection_reason = "handmade"
        elif corrected:
            selected = corrected[0]  # Assume only one corrected per cue
            selection_reason = "corrected"
        elif fewshot:
            # Find most recent by 'time' field
            fewshot_with_time = [c for c in fewshot if 'time' in c]
            if fewshot_with_time:
                # Sort by time (most recent first)
                fewshot_with_time.sort(key=lambda x: x.get('time', ''), reverse=True)
                selected = fewshot_with_time[0]
                selection_reason = f"fewshot (most recent: {selected.get('time', 'N/A')})"
            else:
                # No time field, use highest idx
                selected = max(fewshot, key=lambda x: x.get('idx', 0))
                selection_reason = f"fewshot (highest idx: {selected.get('idx', 'N/A')})"
        elif zeroshot:
            selected = zeroshot[0]
            selection_reason = "zeroshot (fallback)"
        
        if selected:
            selected_configs.append(selected)
            if verbose:
                print(f"✓ {cue:40s} -> idx={selected.get('idx', 'N/A'):3d}  state={selected.get('state', 'N/A'):10s}  ({selection_reason})")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Selected {len(selected_configs)} unique cues")
    print(f"{'='*70}")
    
    state_counts = {}
    for config in selected_configs:
        state = config.get('state', 'unknown')
        state_counts[state] = state_counts.get(state, 0) + 1
    
    for state, count in sorted(state_counts.items()):
        print(f"  {state:10s}: {count}")
    
    # Output based on format
    if output_format == "idx":
        result = [c.get('idx') for c in selected_configs]
        print(f"\nIndices: {result[:10]}..." if len(result) > 10 else f"\nIndices: {result}")
        return result
    
    elif output_format == "cue":
        result = [c.get('cue') for c in selected_configs]
        print(f"\nCues: {result[:10]}..." if len(result) > 10 else f"\nCues: {result}")
        return result
    
    elif output_format == "json":
        if not output_file:
            output_file = "data/seed/motion_config_recent.json"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(selected_configs, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved {len(selected_configs)} configs to {output_file}")
        return selected_configs
    
    elif output_format == "full":
        return selected_configs
    
    else:
        raise ValueError(f"Invalid output_format: {output_format}. Use 'idx', 'cue', 'json', or 'full'")


def print_usage():
    """Print usage examples."""
    print("\nUsage Examples:")
    print("  # Get list of indices")
    print("  python get_recent_motions.py")
    print()
    print("  # Get list of cue names")
    print("  python get_recent_motions.py --output_format=cue")
    print()
    print("  # Save to JSON file")
    print("  python get_recent_motions.py --output_format=json --output_file=data/seed/best_configs.json")
    print()
    print("  # Verbose mode")
    print("  python get_recent_motions.py --verbose=True")


if __name__ == "__main__":
    fire.Fire(get_recent_motions)
