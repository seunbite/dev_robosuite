"""
Generate humanoid robot motions with personality-based variations.

This is the humanoid version of motion_generation_with_persona.py, adapted for:
- Single arm motion (right or left)
- Other joints fixed (head, torso, other arm, legs)
- Same motion_config.json format
- Same personality system

Usage:
    python adhoc/humanoid/motion_generation_humanoid.py \
        --robot GR1ArmsOnly \
        --active-arm right \
        --cue waving \
        --personality Excited
"""

import fire
import os
import sys
import json
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robotarm'))

from motion_generation import MotionGenerator


# Import personality profile function from robotarm version
def get_personality_profile_from_gemini(
    cue_name: str,
    cue_description: str,
    personality_name: str,
    personality_description: str,
    api_key: Optional[str] = None,
) -> Dict:
    """
    Get personality profile from Gemini API or use hardcoded response.
    
    For now, returns hardcoded profiles. To use Gemini API:
    1. Uncomment google.generativeai import
    2. Pass gemini_api_key parameter
    
    Args:
        cue_name: Name of the motion cue
        cue_description: Description of the motion cue
        personality_name: Name of the personality
        personality_description: Description of the personality
        api_key: Gemini API key (if None, uses hardcoded response)
    
    Returns:
        Personality profile dictionary
    """
    # HARDCODED PROFILES FOR COMMON PERSONALITIES
    profiles = {
        "sad": {
            "speed_scale": [0.5, 0.8],
            "repetition_scale": [0.7, 0.9],
            "hold_time_scale": [1.3, 1.8],
            "degree_scale": [0.6, 0.8],
            "transition_bias": "down",
            "variation": 0.15,
        },
        "excited": {
            "speed_scale": [1.3, 1.8],
            "repetition_scale": [1.2, 1.5],
            "hold_time_scale": [0.3, 0.6],
            "degree_scale": [1.2, 1.5],
            "transition_bias": "up",
            "variation": 0.35,
        },
        "calm": {
            "speed_scale": [0.6, 1.0],
            "repetition_scale": [0.8, 1.0],
            "hold_time_scale": [1.2, 1.7],
            "degree_scale": [0.9, 1.1],
            "transition_bias": None,
            "variation": 0.1,
        },
        "anxious": {
            "speed_scale": [1.0, 1.5],
            "repetition_scale": [1.3, 1.7],
            "hold_time_scale": [0.3, 0.6],
            "degree_scale": [0.7, 1.0],
            "transition_bias": "random",
            "variation": 0.4,
        },
    }
    
    personality_lower = personality_name.lower()
    if personality_lower in profiles:
        print(f"Using hardcoded personality profile for '{personality_name}'")
        return profiles[personality_lower]
    
    # Default profile (neutral)
    print(f"Using default personality profile for '{personality_name}'")
    return {
        "speed_scale": [0.9, 1.1],
        "repetition_scale": [0.9, 1.1],
        "hold_time_scale": [0.9, 1.1],
        "degree_scale": [0.9, 1.1],
        "transition_bias": None,
        "variation": 0.1,
    }


class HumanoidMotionGenerator(MotionGenerator):
    """
    Motion generator for humanoid robots with personality variations.
    
    Extends MotionGenerator with:
    1. Personality-based parameter scaling
    2. Personality-based transitional movements
    3. Random variations within personality constraints
    """
    
    def __init__(
        self,
        robot_name: str,
        active_arm: str,
        personality: str = "Neutral",
        personality_description: Optional[str] = None,
        personality_list_path: str = "data/seed/personality_list.json",
        gemini_api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize humanoid motion generator with personality.
        
        Args:
            robot_name: Name of humanoid robot (e.g., GR1ArmsOnly)
            active_arm: Which arm to move ("right" or "left")
            personality: Personality name
            personality_description: Optional personality description
            personality_list_path: Path to personality list JSON
            gemini_api_key: Gemini API key for dynamic personality profiles
            **kwargs: Additional arguments passed to MotionGenerator
        """
        # Store humanoid-specific info
        self.humanoid_robot_name = robot_name
        self.active_arm = active_arm
        
        # Override jsonl_path for humanoid poses
        if 'jsonl_path' not in kwargs:
            # Try to use closest poses first (pre-queried)
            closest_poses_path = f"data/poses/humanoid/closest_{robot_name}_{active_arm}_poses.jsonl"
            if os.path.exists(closest_poses_path):
                kwargs['jsonl_path'] = closest_poses_path
                print(f"Using pre-queried poses: {closest_poses_path}")
            else:
                # Fallback to all poses
                kwargs['jsonl_path'] = f"data/poses/humanoid/all_{robot_name}_{active_arm}_poses.jsonl"
                print(f"Warning: Using all poses. Run meta_query_all_humanoid_poses.py first for better results.")
        
        # Initialize parent with combined robot name
        super().__init__(
            robot_name=f"{robot_name}",  # Use base robot name for env creation
            **kwargs
        )
        
        self.personality = personality
        self.gemini_api_key = gemini_api_key
        
        # Load personality description
        if personality_description is None:
            if os.path.exists(personality_list_path):
                try:
                    with open(personality_list_path, 'r') as f:
                        personality_data = json.load(f)
                        # Handle both list and dict formats
                        if isinstance(personality_data, list):
                            personality_list = personality_data
                        elif isinstance(personality_data, dict):
                            # If it's a dict, get the list (assuming key like "personalities")
                            personality_list = personality_data.get("personalities", list(personality_data.values()))
                        else:
                            personality_list = []
                        
                        # Find matching personality
                        for p in personality_list:
                            if isinstance(p, dict) and p.get("name", "").lower() == personality.lower():
                                personality_description = p.get("description", "")
                                break
                except Exception as e:
                    print(f"Warning: Could not load personality list: {e}")
                    personality_description = None
        
        self.personality_description = personality_description or f"A robot with {personality} personality"
        
        print(f"\n{'='*60}")
        print(f"Humanoid Motion Generator with Personality")
        print(f"{'='*60}")
        print(f"Robot: {robot_name}")
        print(f"Active arm: {active_arm}")
        print(f"Personality: {personality}")
        print(f"Description: {self.personality_description}")
        print(f"{'='*60}\n")
    
    def apply_personality_to_parameters(
        self,
        params: Dict,
        personality_profile: Dict,
    ) -> Dict:
        """
        Apply personality profile to movement parameters.
        
        Args:
            params: Original parameters
            personality_profile: Personality profile with scales and biases
        
        Returns:
            Modified parameters
        """
        modified = params.copy()
        variation = personality_profile.get("variation", 0.1)
        
        # Apply speed scale
        if "speed" in modified:
            speed_min, speed_max = personality_profile["speed_scale"]
            scale = random.uniform(speed_min, speed_max)
            scale += random.uniform(-variation, variation)
            modified["speed"] = max(0.1, modified["speed"] * scale)
        
        # Apply repetition scale
        if "repetition" in modified:
            rep_min, rep_max = personality_profile["repetition_scale"]
            scale = random.uniform(rep_min, rep_max)
            modified["repetition"] = max(1, int(modified["repetition"] * scale))
        
        # Apply hold_time scale
        if "hold_time" in modified:
            hold_min, hold_max = personality_profile["hold_time_scale"]
            scale = random.uniform(hold_min, hold_max)
            scale += random.uniform(-variation, variation)
            modified["hold_time"] = max(0.1, modified["hold_time"] * scale)
        
        # Apply degree scale to directions
        if "directions" in modified:
            deg_min, deg_max = personality_profile["degree_scale"]
            scale = random.uniform(deg_min, deg_max)
            scale += random.uniform(-variation, variation)
            
            new_directions = []
            for direction in modified["directions"]:
                new_dir = direction.copy()
                if "degrees" in new_dir:
                    new_dir["degrees"] = new_dir["degrees"] * scale
                new_directions.append(new_dir)
            modified["directions"] = new_directions
        
        return modified
    
    def execute_cue_with_personality(
        self,
        cue: str,
        pose_index: Optional[int] = None,
        config_path: str = "data/seed/motion_config.json",
        proximal_degree_scale: float = 0.25,
        hz: int = 4,
        enable_self_collision_check: bool = False,
        add_transitions: bool = True,
        transition_probability: float = 0.5,
    ):
        """
        Execute a motion cue with personality-based variations.
        
        Args:
            cue: Name of the cue from motion_config.json
            pose_index: Optional specific pose to use
            config_path: Path to motion_config.json
            proximal_degree_scale: Scale factor for proximal joints
            hz: Frame rate
            enable_self_collision_check: Enable collision checking
            add_transitions: Add personality-based transitions
            transition_probability: Probability of adding transitions
        """
        # Load motion config
        with open(config_path, 'r') as f:
            motion_configs = json.load(f)
        
        # Find cue config
        cue_config = None
        for config in motion_configs:
            if config["cue"] == cue:
                cue_config = config
                break
        
        if cue_config is None:
            raise ValueError(f"Cue '{cue}' not found in {config_path}")
        
        # Get personality profile
        cue_description = f"Motion cue: {cue}"
        personality_profile = get_personality_profile_from_gemini(
            cue_name=cue,
            cue_description=cue_description,
            personality_name=self.personality,
            personality_description=self.personality_description,
            api_key=self.gemini_api_key,
        )
        
        print(f"\nPersonality Profile:")
        print(json.dumps(personality_profile, indent=2))
        
        # Create modified config with personality
        modified_config = {
            "cue": cue,
            "movements": []
        }
        
        for movement in cue_config["movements"]:
            modified_movement = {
                "type": movement["type"],
                "parameters": {}
            }
            
            # Apply personality to parameters
            if movement["type"] == "movement":
                modified_params = self.apply_personality_to_parameters(
                    movement["parameters"],
                    personality_profile,
                )
                modified_movement["parameters"] = modified_params
            else:
                # Pose type - copy as is, but apply hold_time scale
                modified_params = movement["parameters"].copy()
                if "hold_time" in modified_params:
                    hold_min, hold_max = personality_profile["hold_time_scale"]
                    scale = random.uniform(hold_min, hold_max)
                    modified_params["hold_time"] = max(0.1, modified_params["hold_time"] * scale)
                modified_movement["parameters"] = modified_params
            
            modified_config["movements"].append(modified_movement)
        
        # Save modified config to temp file
        temp_config_path = f"temp_humanoid_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(temp_config_path, 'w') as f:
            json.dump([modified_config], f, indent=2)
        
        print(f"\nGenerated personality-modified config: {temp_config_path}")
        
        try:
            # Execute the modified motion
            self.execute_cue(
                cue=cue,
                pose_index=pose_index,
                config_path=temp_config_path,
                proximal_degree_scale=proximal_degree_scale,
                hz=hz,
                enable_self_collision_check=enable_self_collision_check,
            )
        finally:
            # Clean up temp file
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)


def generate_humanoid_motion(
    robot: str = "GR1ArmsOnly",
    active_arm: str = "right",
    env: str = "EmptySpace",
    cue: str = "waving",
    personality: str = None,
    personality_description: str = None,
    personality_list_path: str = "data/seed/personality_list.json",
    gemini_api_key: Optional[str] = None,
    pose_index: Optional[int] = None,
    controller: str = "IK_POSE",
    config_path: str = "data/seed/motion_config.json",
    proximal_degree_scale: float = 0.25,
    camera_distance: float = 1.8,
    hz: int = 4,
    enable_self_collision_check: bool = False,
    add_transitions: bool = True,
    transition_probability: float = 0.5,
):
    """
    Generate humanoid robot motions with or without personality-based variations.
    
    Args:
        robot: Humanoid robot name (GR1ArmsOnly, GR1FixedLowerBody, etc.)
        active_arm: Which arm to move ("right" or "left")
        env: Environment name
        cue: Name of the cue to execute
        personality: Personality name from personality_list.json (if None, generates without personality)
        personality_description: Optional personality description
        personality_list_path: Path to personality_list.json
        gemini_api_key: Gemini API key (if None, uses hardcoded profiles)
        pose_index: Optional pose_id to use
        controller: Controller name
        config_path: Path to motion config
        proximal_degree_scale: Scale factor for proximal joints
        camera_distance: Camera FOV multiplier
        hz: Frame rate for GIF
        enable_self_collision_check: Enable collision checking
        add_transitions: Whether to add personality-based transitions
        transition_probability: Probability of adding transitions
    
    Examples:
        # Without personality
        python adhoc/humanoid/motion_generation_humanoid.py \
            --robot GR1ArmsOnly --active-arm right --cue waving
        
        # With personality
        python adhoc/humanoid/motion_generation_humanoid.py \
            --robot GR1ArmsOnly --active-arm right --cue waving --personality Excited
        
        # With specific pose
        python adhoc/humanoid/motion_generation_humanoid.py \
            --robot GR1ArmsOnly --active-arm left --cue beckoning --personality Sad --pose-index 42
    """
    # If personality is None, use basic motion generation without personality
    if personality is None:
        print("\n" + "="*60)
        print("Generating motion WITHOUT personality")
        print("="*60 + "\n")
        
        # Determine jsonl path for humanoid
        jsonl_path = f"data/poses/humanoid/closest_{robot}_{active_arm}_poses.jsonl"
        if not os.path.exists(jsonl_path):
            jsonl_path = f"data/poses/humanoid/all_{robot}_{active_arm}_poses.jsonl"
        
        # Use the base MotionGenerator (already imported at the top)
        generator = MotionGenerator(
            robot_name=robot,
            env_name=env,
            controller_name=controller,
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_distance=camera_distance,
            hz=hz,
            jsonl_path=jsonl_path,
        )
        
        try:
            generator.execute_cue(
                cue=cue,
                pose_index=pose_index,
                config_path=config_path,
                proximal_degree_scale=proximal_degree_scale,
                hz=hz,
                enable_self_collision_check=enable_self_collision_check,
            )
        finally:
            generator.close()
    else:
        # Use personality-based generation
        generator = HumanoidMotionGenerator(
            robot_name=robot,
            active_arm=active_arm,
            personality=personality,
            personality_description=personality_description,
            personality_list_path=personality_list_path,
            gemini_api_key=gemini_api_key,
            env_name=env,
            controller_name=controller,
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_distance=camera_distance,
            hz=hz,
        )
        
        try:
            generator.execute_cue_with_personality(
                cue=cue,
                pose_index=pose_index,
                config_path=config_path,
                proximal_degree_scale=proximal_degree_scale,
                hz=hz,
                enable_self_collision_check=enable_self_collision_check,
                add_transitions=add_transitions,
                transition_probability=transition_probability,
            )
        finally:
            generator.close()
    
    return True


if __name__ == "__main__":
    fire.Fire(generate_humanoid_motion)
