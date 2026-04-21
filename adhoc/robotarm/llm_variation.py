"""
Generate motion variations based on semantic prompts.

This script takes a semantic prompt (e.g., "make it look depressed", "make it look disappointed")
and generates motion variations by modifying parameters and waypoints accordingly.
Currently uses rule-based variation (LLM inference will be added later).
"""

import fire
import os
import json
import copy
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_generation import MotionGenerator


def parse_semantic_prompt(prompt: str) -> Dict:
    """
    Parse semantic prompt and return variation parameters.
    
    For now, uses rule-based mapping. Later will use LLM.
    
    Args:
        prompt: Semantic prompt (e.g., "make it look depressed", "make it look disappointed")
        
    Returns:
        Dictionary with variation parameters:
        - speed_multiplier: Speed adjustment (0.5-1.5)
        - degree_multiplier: Degree adjustment (0.5-1.5)
        - hold_time_multiplier: Hold time adjustment (0.5-2.0)
        - repetition_multiplier: Repetition adjustment (0.5-1.5)
        - initial_pose_offset: Dict with x, y, z offsets in meters (default: {"x": 0, "y": 0, "z": 0})
    """
    prompt_lower = prompt.lower()
    
    # Default variation (no change)
    variation = {
        "speed_multiplier": 1.0,
        "degree_multiplier": 1.0,
        "hold_time_multiplier": 1.0,
        "repetition_multiplier": 1.0,
        "initial_pose_offset": {"x": 0.0, "y": 0.0, "z": 0.0},
    }
    
    # Rule-based mapping (temporary, will be replaced with LLM)
    if any(keyword in prompt_lower for keyword in ["depressed", "disappointed", "sad", "down", "dejected"]):
        # Depressed/disappointed: slower, smaller movements, longer pauses, fewer repetitions, lower position
        variation["speed_multiplier"] = 0.7
        variation["degree_multiplier"] = 0.7
        variation["hold_time_multiplier"] = 1.5
        variation["repetition_multiplier"] = 0.7
        variation["initial_pose_offset"] = {"x": 0.0, "y": 0.0, "z": -0.05}  # Lower (down)
        
    elif any(keyword in prompt_lower for keyword in ["energetic", "excited", "enthusiastic", "cheerful"]):
        # Energetic/excited: faster, larger movements, shorter pauses, more repetitions, higher position
        variation["speed_multiplier"] = 1.3
        variation["degree_multiplier"] = 1.2
        variation["hold_time_multiplier"] = 0.7
        variation["repetition_multiplier"] = 1.3
        variation["initial_pose_offset"] = {"x": 0.0, "y": 0.0, "z": 0.05}  # Higher (up)
        
    elif any(keyword in prompt_lower for keyword in ["careful", "cautious", "slow", "gentle", "delicate"]):
        # Careful/cautious: slower, smaller movements, longer pauses
        variation["speed_multiplier"] = 0.6
        variation["degree_multiplier"] = 0.8
        variation["hold_time_multiplier"] = 1.3
        variation["repetition_multiplier"] = 1.0
        variation["initial_pose_offset"] = {"x": 0.0, "y": 0.0, "z": 0.0}  # Neutral
        
    elif any(keyword in prompt_lower for keyword in ["hasty", "quick", "fast", "rapid"]):
        # Hasty/quick: faster, shorter pauses
        variation["speed_multiplier"] = 1.5
        variation["degree_multiplier"] = 1.0
        variation["hold_time_multiplier"] = 0.5
        variation["repetition_multiplier"] = 1.0
        variation["initial_pose_offset"] = {"x": 0.0, "y": 0.0, "z": 0.0}  # Neutral
        
    elif any(keyword in prompt_lower for keyword in ["large", "exaggerated", "emphatic", "dramatic"]):
        # Large/exaggerated: larger movements, longer pauses for emphasis
        variation["speed_multiplier"] = 1.0
        variation["degree_multiplier"] = 1.5
        variation["hold_time_multiplier"] = 1.2
        variation["repetition_multiplier"] = 1.0
        variation["initial_pose_offset"] = {"x": 0.0, "y": 0.0, "z": 0.03}  # Slightly higher
        
    elif any(keyword in prompt_lower for keyword in ["small", "subtle", "gentle", "quiet", "minimal"]):
        # Small/subtle: smaller movements, shorter pauses
        variation["speed_multiplier"] = 0.9
        variation["degree_multiplier"] = 0.6
        variation["hold_time_multiplier"] = 0.8
        variation["repetition_multiplier"] = 1.0
        variation["initial_pose_offset"] = {"x": 0.0, "y": 0.0, "z": -0.02}  # Slightly lower
        
    elif any(keyword in prompt_lower for keyword in ["withdrawn", "shy", "retreating", "back"]):
        # Withdrawn/shy: smaller movements, pulled back position
        variation["speed_multiplier"] = 0.8
        variation["degree_multiplier"] = 0.7
        variation["hold_time_multiplier"] = 1.1
        variation["repetition_multiplier"] = 0.9
        variation["initial_pose_offset"] = {"x": -0.05, "y": 0.0, "z": 0.0}  # Backward
        
    elif any(keyword in prompt_lower for keyword in ["forward", "reaching", "extended"]):
        # Forward/reaching: extended position
        variation["speed_multiplier"] = 1.0
        variation["degree_multiplier"] = 1.0
        variation["hold_time_multiplier"] = 1.0
        variation["repetition_multiplier"] = 1.0
        variation["initial_pose_offset"] = {"x": 0.05, "y": 0.0, "z": 0.0}  # Forward
        
    elif any(keyword in prompt_lower for keyword in ["left", "to the left"]):
        # Left: move to the left
        variation["initial_pose_offset"] = {"x": 0.0, "y": 0.05, "z": 0.0}  # Left (positive y)
        
    elif any(keyword in prompt_lower for keyword in ["right", "to the right"]):
        # Right: move to the right
        variation["initial_pose_offset"] = {"x": 0.0, "y": -0.05, "z": 0.0}  # Right (negative y)
    
    return variation


def apply_parameter_variation(
    config: Dict,
    variation: Dict,
) -> Dict:
    """
    Apply parameter variation to motion configuration.
    
    Args:
        config: Original motion configuration
        variation: Variation parameters from parse_semantic_prompt
        
    Returns:
        Modified configuration
    """
    modified_config = copy.deepcopy(config)
    
    movements = modified_config.get("movements", [])
    
    for movement in movements:
        if movement.get("type") == "movement":
            parameters = movement.get("parameters", {})
            
            # Apply repetition multiplier
            if "repetition" in parameters:
                original_repetition = parameters["repetition"]
                new_repetition = max(1, int(original_repetition * variation["repetition_multiplier"]))
                parameters["repetition"] = new_repetition
            
            # Apply variations to directions
            directions = parameters.get("directions", [])
            for direction_config in directions:
                # Apply speed multiplier
                if "speed" in direction_config:
                    original_speed = direction_config["speed"]
                    new_speed = original_speed * variation["speed_multiplier"]
                    direction_config["speed"] = max(0.1, new_speed)  # Minimum speed
                
                # Apply degree multiplier
                if "degrees" in direction_config:
                    original_degrees = direction_config["degrees"]
                    new_degrees = original_degrees * variation["degree_multiplier"]
                    direction_config["degrees"] = max(5.0, new_degrees)  # Minimum degrees
                
                # Apply hold_time multiplier
                if "hold_time" in direction_config:
                    original_hold_time = direction_config["hold_time"]
                    new_hold_time = original_hold_time * variation["hold_time_multiplier"]
                    direction_config["hold_time"] = max(0.1, new_hold_time)  # Minimum hold time
        
        elif movement.get("type") == "pose":
            parameters = movement.get("parameters", {})
            
            # Apply speed multiplier (for pose transitions)
            if "speed" in parameters:
                original_speed = parameters["speed"]
                new_speed = original_speed * variation["speed_multiplier"]
                parameters["speed"] = max(0.1, new_speed)
            
            # Apply hold_time multiplier
            if "hold_time" in parameters:
                original_hold_time = parameters["hold_time"]
                new_hold_time = original_hold_time * variation["hold_time_multiplier"]
                parameters["hold_time"] = max(0.1, new_hold_time)
    
    return modified_config


def plan_waypoint_variation(
    config: Dict,
    variation: Dict,
) -> Dict:
    """
    Plan waypoint variation based on semantic prompt.
    
    Adds initial_pose_offset to the configuration to specify x, y, z offsets
    for the initial pose's end-effector position.
    
    Args:
        config: Motion configuration
        variation: Variation parameters with initial_pose_offset
        
    Returns:
        Configuration with waypoint variations (initial_pose_offset added)
    """
    modified_config = copy.deepcopy(config)
    
    # Add initial_pose_offset to config
    initial_pose_offset = variation.get("initial_pose_offset", {"x": 0.0, "y": 0.0, "z": 0.0})
    modified_config["initial_pose_offset"] = initial_pose_offset
    
    return modified_config


def apply_variation_to_config(
    config: Dict,
    prompt: str,
) -> Tuple[Dict, Dict]:
    """
    Apply semantic variation to motion configuration.
    
    Args:
        config: Original motion configuration
        prompt: Semantic prompt
        
    Returns:
        Tuple of (modified_config, variation_params)
    """
    # Parse semantic prompt
    variation = parse_semantic_prompt(prompt)
    
    print(f"\n{'='*60}")
    print("SEMANTIC VARIATION PLANNING")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"\nVariation parameters:")
    print(f"  Speed multiplier: {variation['speed_multiplier']:.2f}x")
    print(f"  Degree multiplier: {variation['degree_multiplier']:.2f}x")
    print(f"  Hold time multiplier: {variation['hold_time_multiplier']:.2f}x")
    print(f"  Repetition multiplier: {variation['repetition_multiplier']:.2f}x")
    offset = variation['initial_pose_offset']
    print(f"  Initial pose offset: x={offset['x']:.3f}m, y={offset['y']:.3f}m, z={offset['z']:.3f}m")
    print(f"{'='*60}\n")
    
    # Apply parameter variation
    modified_config = apply_parameter_variation(config, variation)
    
    # Plan waypoint variation
    modified_config = plan_waypoint_variation(modified_config, variation)
    
    return modified_config, variation


def main(
    robot: str = "Panda",
    env: str = "EmptySpace",
    cue: str = "waving",
    prompt: str = "make it look depressed",
    pose_index: Optional[int] = None,
    controller: str = "IK_POSE",
    jsonl_path: str = "data/seed/closest_poses_results.jsonl",
    config_path: str = "data/seed/motion_config.json",
    proximal_degree_scale: float = 0.25,
    camera_distance: float = 1.3,
    hz: int = 4,
    output_dir: str = "data/motions",
):
    """
    Generate motion with semantic variation based on prompt.
    
    Args:
        robot: Robot name
        env: Environment name
        cue: Name of the cue to execute (e.g., 'waving')
        prompt: Semantic prompt for variation (e.g., "make it look depressed")
        pose_index: Optional pose_id to use (if None, randomly selects)
        controller: Controller name
        jsonl_path: Path to pose database JSONL file
        config_path: Path to JSON file with cue configurations
        proximal_degree_scale: Scale factor for degrees when using proximal joints
        camera_distance: Multiplier for camera FOV to zoom out
        hz: Frame rate for GIF generation in frames per second
        output_dir: Output directory for GIFs
    """
    print(f"\n{'='*60}")
    print("LLM VARIATION MOTION GENERATION")
    print(f"{'='*60}")
    print(f"Robot: {robot}")
    print(f"Cue: {cue}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}\n")
    
    # Load original configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        configs = json.load(f)
    
    # Find the cue configuration
    cue_config = None
    for config in configs:
        if config.get("cue") == cue:
            cue_config = config
            break
    
    if cue_config is None:
        raise ValueError(f"Cue '{cue}' not found in config file")
    
    # Apply semantic variation
    modified_config, variation_params = apply_variation_to_config(cue_config, prompt)
    
    # Create a temporary config file with the modified configuration
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        tmp_config_path = tmp_file.name
        json.dump([modified_config], tmp_file, indent=2)
    
    try:
        # Initialize motion generator
        generator = MotionGenerator(
            robot_name=robot,
            env_name=env,
            controller_name=controller,
            jsonl_path=jsonl_path,
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_distance=camera_distance,
            output_dir=output_dir,
        )
        
        try:
            # Execute cue with modified configuration
            generator.execute_cue(
                cue=cue,
                pose_index=pose_index,
                config_path=tmp_config_path,
                proximal_degree_scale=proximal_degree_scale,
                hz=hz,
            )
        finally:
            generator.close()
    
    finally:
        # Clean up temporary config file
        if os.path.exists(tmp_config_path):
            os.unlink(tmp_config_path)
    
    print(f"\n{'='*60}")
    print("Motion generation completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    fire.Fire(main)
