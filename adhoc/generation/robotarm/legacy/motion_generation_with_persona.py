"""
Generate robot motions with personality-based variations using Gemini API.

This script extends motion_generation.py by adding:
1. Movement parameter variations based on personality (speed, repetition, hold_time, degrees)
2. Personality-based transitional movements between poses
3. Random variations within personality constraints

Uses Gemini API to dynamically generate personality profiles instead of hardcoded profiles.
"""

import fire
import os
import json
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
# import google.generativeai as genai  # Uncomment when ready to use API

from motion_generation import MotionGenerator


# GEMINI API PROMPT TEMPLATE
PERSONALITY_PROFILE_PROMPT = """You are an expert in robot motion design and personality expression through movement.

Given a robot motion cue and a personality trait, generate a personality profile that modifies the motion parameters to express that personality.

**Motion Cue:** {cue_name}
**Motion Description:** {cue_description}
**Personality:** {personality_name}
**Personality Description:** {personality_description}

Please generate a personality profile with the following parameters:

1. **speed_scale**: (min, max) - Range for speed modification (1.0 = normal speed, >1.0 = faster, <1.0 = slower)
2. **repetition_scale**: (min, max) - Range for repetition count modification (1.0 = normal, >1.0 = more repetitions, <1.0 = fewer)
3. **hold_time_scale**: (min, max) - Range for hold time modification (1.0 = normal, >1.0 = longer holds, <1.0 = shorter)
4. **degree_scale**: (min, max) - Range for movement degree modification (1.0 = normal range, >1.0 = larger movements, <1.0 = smaller)
5. **transition_bias**: Direction bias for transitional movements between poses. Options: "up", "down", "forward", "back", "random", or null
6. **variation**: Random variation amount (0.0 to 1.0, where higher values mean more randomness within the ranges)

**Guidelines:**
- For energetic/excited personalities: faster speed (1.2-1.8), more repetitions (1.2-1.5), shorter holds (0.3-0.6), larger movements (1.2-1.5), transition_bias "up" or "forward", higher variation (0.25-0.4)
- For sad/tired personalities: slower speed (0.4-0.8), fewer repetitions (0.6-0.9), longer holds (1.3-2.0), smaller movements (0.5-0.8), transition_bias "down", lower variation (0.1-0.15)
- For anxious/nervous personalities: variable speed (1.0-1.5), more repetitions (1.3-1.7), shorter holds (0.3-0.6), smaller movements (0.7-1.0), transition_bias "random", higher variation (0.3-0.45)
- For calm/relaxed personalities: slower speed (0.6-1.0), normal repetitions (0.8-1.0), longer holds (1.2-1.7), normal movements (0.9-1.1), transition_bias null, lower variation (0.05-0.15)

Respond ONLY with a valid JSON object in this exact format (no markdown, no extra text):
{{
  "speed_scale": [min, max],
  "repetition_scale": [min, max],
  "hold_time_scale": [min, max],
  "degree_scale": [min, max],
  "transition_bias": "up/down/forward/back/random/null",
  "variation": 0.0-1.0
}}"""


def get_personality_profile_from_gemini(
    cue_name: str,
    cue_description: str,
    personality_name: str,
    personality_description: str,
    api_key: Optional[str] = None,
) -> Dict:
    """
    Get personality profile from Gemini API.
    
    Args:
        cue_name: Name of the motion cue
        cue_description: Description of the motion cue
        personality_name: Name of the personality
        personality_description: Description of the personality
        api_key: Gemini API key (if None, uses hardcoded response for "Sad")
    
    Returns:
        Personality profile dictionary
    """
    # HARDCODED RESPONSE FOR "SAD" PERSONALITY
    # This simulates what Gemini would return
    if personality_name.lower() == "sad" or api_key is None:
        print(f"Using hardcoded personality profile for '{personality_name}'")
        return {
            "speed_scale": [0.5, 0.8],
            "repetition_scale": [0.7, 0.9],
            "hold_time_scale": [1.3, 1.8],
            "degree_scale": [0.6, 0.8],
            "transition_bias": "down",
            "variation": 0.15,
        }
    
    # UNCOMMENT THIS SECTION WHEN READY TO USE GEMINI API
    """
    # Configure Gemini API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    # Format prompt
    prompt = PERSONALITY_PROFILE_PROMPT.format(
        cue_name=cue_name,
        cue_description=cue_description,
        personality_name=personality_name,
        personality_description=personality_description,
    )
    
    try:
        # Generate response
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Parse JSON response
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        profile = json.loads(response_text)
        
        # Convert transition_bias "null" string to None
        if profile.get("transition_bias") == "null":
            profile["transition_bias"] = None
        
        # Convert lists to tuples for consistency
        profile["speed_scale"] = tuple(profile["speed_scale"])
        profile["repetition_scale"] = tuple(profile["repetition_scale"])
        profile["hold_time_scale"] = tuple(profile["hold_time_scale"])
        profile["degree_scale"] = tuple(profile["degree_scale"])
        
        print(f"Generated personality profile from Gemini API for '{personality_name}'")
        return profile
        
    except Exception as e:
        print(f"Error getting profile from Gemini API: {e}")
        print("Falling back to default profile")
        return DEFAULT_PROFILE
    """
    
    # For now, return default profile if not "sad"
    print(f"Using default personality profile for '{personality_name}'")
    return DEFAULT_PROFILE



# Default profile for personalities not in the list
DEFAULT_PROFILE = {
    "speed_scale": (0.9, 1.1),
    "repetition_scale": (0.9, 1.1),
    "hold_time_scale": (0.9, 1.1),
    "degree_scale": (0.9, 1.1),
    "transition_bias": None,
    "variation": 0.15,
}


class PersonalityMotionGenerator(MotionGenerator):
    """Motion generator with personality-based variations using Gemini API."""
    
    def __init__(
        self,
        personality: str = "Calm",
        personality_description: str = None,
        personality_list_path: str = "data/seed/_remainder/personality_list.json",
        gemini_api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize with a personality.
        
        Args:
            personality: Name of the personality from personality_list.json
            personality_description: Optional description of the personality (if None, loads from file)
            personality_list_path: Path to personality_list.json
            gemini_api_key: Gemini API key (if None, uses hardcoded response)
            **kwargs: Arguments passed to MotionGenerator
        """
        super().__init__(**kwargs)
        self.personality = personality
        self.gemini_api_key = gemini_api_key
        
        # Load personality description if not provided
        if personality_description is None:
            personality_description = self._load_personality_description(
                personality, personality_list_path
            )
        
        self.personality_description = personality_description
        self.profile = None  # Will be generated per cue
        
        print(f"Personality: {personality}")
        print(f"Description: {personality_description}")
        if gemini_api_key:
            print("Gemini API: Enabled")
        else:
            print("Gemini API: Disabled (using hardcoded responses)")
    
    def _load_personality_description(
        self,
        personality: str,
        personality_list_path: str
    ) -> str:
        """Load personality description from personality_list.json."""
        try:
            with open(personality_list_path, 'r') as f:
                personality_dict = json.load(f)
            
            description = personality_dict.get(personality, "")
            if not description:
                print(f"Warning: Personality '{personality}' not found in {personality_list_path}")
                return f"A personality trait: {personality}"
            
            return description
        except Exception as e:
            print(f"Error loading personality list: {e}")
            return f"A personality trait: {personality}"
    
    def _generate_cue_description(self, movements: List[Dict]) -> str:
        """Generate a human-readable description of the cue from movements."""
        description_parts = []
        
        for i, movement_item in enumerate(movements):
            movement_type = movement_item.get('type')
            
            if movement_type == 'pose':
                pose = movement_item.get('parameters', {}).get('pose')
                if isinstance(pose, dict):
                    height = pose.get('height', '?')
                    direction = pose.get('dir', '?')
                    gripper_orientation = pose.get('gripper_orientation', '?')
                    description_parts.append(f"pose {i+1}: {height} height, {direction} direction, {gripper_orientation} gripper_orientation")
                else:
                    description_parts.append(f"pose {i+1}: {pose}")
            
            elif movement_type == 'movement':
                params = movement_item.get('parameters', {})
                axis = params.get('axis', '?')
                joint = params.get('joint', '?')
                repetition = params.get('repetition', 1)
                description_parts.append(f"movement {i+1}: {repetition}x {axis}-axis motion using {joint} joint")
        
        if description_parts:
            return "This motion consists of: " + "; ".join(description_parts)
        else:
            return "A robot arm motion"
    
    def _apply_personality_variation(
        self,
        base_value: float,
        scale_range: Tuple[float, float],
        variation: float,
    ) -> float:
        """
        Apply personality-based variation to a parameter.
        
        Args:
            base_value: Original value
            scale_range: (min_scale, max_scale) from personality profile
            variation: Random variation amount (0.0 to 1.0)
        
        Returns:
            Modified value
        """
        # Sample from personality range
        scale = random.uniform(scale_range[0], scale_range[1])
        
        # Add random variation
        if variation > 0:
            variation_amount = random.uniform(-variation, variation)
            scale = scale * (1 + variation_amount)
        
        return base_value * scale
    
    def _add_transition_movement(
        self,
        movements: List[Dict],
        insert_index: int,
    ) -> List[Dict]:
        """
        Add a personality-based transition movement between poses.
        
        Args:
            movements: List of movement dictionaries
            insert_index: Where to insert the transition (after this index)
        
        Returns:
            Modified movements list
        """
        transition_bias = self.profile.get("transition_bias")
        
        if transition_bias is None or insert_index >= len(movements) - 1:
            return movements
        
        # Determine transition direction based on personality
        if transition_bias == "up":
            axis = "z"
            degrees = random.uniform(10, 25)
        elif transition_bias == "down":
            axis = "z"
            degrees = random.uniform(-25, -10)
        elif transition_bias == "forward":
            axis = "x"
            degrees = random.uniform(10, 20)
        elif transition_bias == "back":
            axis = "x"
            degrees = random.uniform(-20, -10)
        elif transition_bias == "random":
            axis = random.choice(["x", "y", "z"])
            degrees = random.uniform(-20, 20)
        else:
            return movements
        
        # Create transition movement
        transition = {
            "type": "movement",
            "parameters": {
                "repetition": 1,
                "axis": axis,
                "joint": random.choice(["proximal", "distal"]),
                "directions": [
                    {
                        "degrees": degrees,
                        "speed": self._apply_personality_variation(
                            1.0,
                            self.profile["speed_scale"],
                            self.profile["variation"],
                        ),
                        "hold_time": 0.1,
                    },
                    {
                        "degrees": -degrees,
                        "speed": self._apply_personality_variation(
                            1.0,
                            self.profile["speed_scale"],
                            self.profile["variation"],
                        ),
                        "hold_time": 0.1,
                    },
                ],
            },
        }
        
        # Insert after the specified index
        movements.insert(insert_index + 1, transition)
        return movements
    
    def execute_cue_with_personality(
        self,
        cue: str,
        pose_index: Optional[int] = None,
        config_path: str = "data/results/motion_configs/manipulator/motion_config.json",
        proximal_degree_scale: float = 0.25,
        hz: int = 4,
        filename_suffix: Optional[str] = None,
        enable_self_collision_check: bool = False,
        add_transitions: bool = True,
        transition_probability: float = 0.5,
    ):
        """
        Execute a cue with personality-based variations.
        
        Args:
            cue: Name of the cue
            pose_index: Optional pose_id to use
            config_path: Path to motion config file
            proximal_degree_scale: Scale factor for proximal joints
            hz: Frame rate for GIF
            filename_suffix: Optional filename suffix
            enable_self_collision_check: Enable collision checking
            add_transitions: Whether to add personality-based transitions
            transition_probability: Probability of adding transition (0.0 to 1.0)
        """
        print(f"\n{'='*60}")
        print(f"Executing cue: {cue} with personality: {self.personality}")
        print(f"{'='*60}\n")
        
        # Load cue configuration
        cue_config_data = self._load_cue_config(cue, config_path)
        movements = cue_config_data.get('movements', [])
        
        if not movements:
            raise ValueError(f"No movements found in cue '{cue}' configuration")
        
        # Generate cue description for Gemini API
        cue_description = self._generate_cue_description(movements)
        
        # Get personality profile from Gemini API
        print(f"\nGenerating personality profile for '{self.personality}' with cue '{cue}'...")
        self.profile = get_personality_profile_from_gemini(
            cue_name=cue,
            cue_description=cue_description,
            personality_name=self.personality,
            personality_description=self.personality_description,
            api_key=self.gemini_api_key,
        )
        print(f"Profile generated: {self.profile}\n")
        
        # Apply personality variations to movements
        modified_movements = []
        for i, movement_item in enumerate(movements):
            movement_type = movement_item.get('type')
            parameters = movement_item.get('parameters', {}).copy()
            
            if movement_type == 'movement':
                # Modify movement parameters based on personality
                if 'repetition' in parameters:
                    parameters['repetition'] = max(
                        1,
                        int(self._apply_personality_variation(
                            parameters['repetition'],
                            self.profile['repetition_scale'],
                            self.profile['variation'],
                        ))
                    )
                
                # Modify directions array
                if 'directions' in parameters:
                    modified_directions = []
                    for direction in parameters['directions']:
                        modified_dir = direction.copy()
                        
                        if 'speed' in modified_dir:
                            modified_dir['speed'] = self._apply_personality_variation(
                                modified_dir['speed'],
                                self.profile['speed_scale'],
                                self.profile['variation'],
                            )
                        
                        if 'hold_time' in modified_dir:
                            modified_dir['hold_time'] = self._apply_personality_variation(
                                modified_dir['hold_time'],
                                self.profile['hold_time_scale'],
                                self.profile['variation'],
                            )
                        
                        if 'degrees' in modified_dir:
                            modified_dir['degrees'] = self._apply_personality_variation(
                                modified_dir['degrees'],
                                self.profile['degree_scale'],
                                self.profile['variation'],
                            )
                        
                        modified_directions.append(modified_dir)
                    
                    parameters['directions'] = modified_directions
            
            elif movement_type == 'pose':
                # Modify pose parameters
                if 'speed' in parameters:
                    parameters['speed'] = self._apply_personality_variation(
                        parameters['speed'],
                        self.profile['speed_scale'],
                        self.profile['variation'],
                    )
                
                if 'hold_time' in parameters:
                    parameters['hold_time'] = self._apply_personality_variation(
                        parameters['hold_time'],
                        self.profile['hold_time_scale'],
                        self.profile['variation'],
                    )
            
            modified_movements.append({
                'type': movement_type,
                'parameters': parameters,
            })
            
            # Add transition movement with probability
            if add_transitions and i < len(movements) - 1 and random.random() < transition_probability:
                modified_movements = self._add_transition_movement(modified_movements, len(modified_movements) - 1)
        
        # Create modified cue config
        modified_config = {
            'cue': cue,
            'movements': modified_movements,
        }
        
        # Save to temporary file
        temp_config_path = f"data/temp_persona_{self.personality}_{cue}.json"
        os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)
        with open(temp_config_path, 'w') as f:
            json.dump([modified_config], f, indent=2)
        
        # Execute using parent class method
        try:
            # Add personality to filename suffix
            if filename_suffix:
                full_suffix = f"{filename_suffix}_{self.personality}"
            else:
                full_suffix = self.personality
            
            self.execute_cue(
                cue=cue,
                pose_index=pose_index,
                config_path=temp_config_path,
                proximal_degree_scale=proximal_degree_scale,
                hz=hz,
                filename_suffix=full_suffix,
                enable_self_collision_check=enable_self_collision_check,
            )
        finally:
            # Clean up temp file
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)


def generate_with_persona(
    robot: str = "Panda",
    env: str = "EmptySpace",
    cue: str = "waving",
    personality: str = "Calm",
    personality_description: str = None,
    personality_list_path: str = "data/seed/_remainder/personality_list.json",
    gemini_api_key: Optional[str] = None,
    pose_index: Optional[int] = None,
    controller: str = "IK_POSE",
    jsonl_path: str = "data/poses/closest_poses_results.jsonl",
    config_path: str = "data/results/motion_configs/manipulator/motion_config.json",
    proximal_degree_scale: float = 0.25,
    camera_distance: float = 1.8,
    hz: int = 4,
    enable_self_collision_check: bool = False,
    add_transitions: bool = True,
    transition_probability: float = 0.5,
):
    """
    Generate robot motions with personality-based variations using Gemini API.
    
    Args:
        robot: Robot name
        env: Environment name
        cue: Name of the cue to execute
        personality: Personality name from personality_list.json
        personality_description: Optional personality description (if None, loads from file)
        personality_list_path: Path to personality_list.json
        gemini_api_key: Gemini API key (if None, uses hardcoded response for "Sad")
        pose_index: Optional pose_id to use
        controller: Controller name
        jsonl_path: Path to pose database
        config_path: Path to motion config
        proximal_degree_scale: Scale factor for proximal joints
        camera_distance: Camera FOV multiplier
        hz: Frame rate for GIF
        enable_self_collision_check: Enable collision checking
        add_transitions: Whether to add personality-based transitions
        transition_probability: Probability of adding transitions (0.0 to 1.0)
    """
    generator = PersonalityMotionGenerator(
        personality=personality,
        personality_description=personality_description,
        personality_list_path=personality_list_path,
        gemini_api_key=gemini_api_key,
        robot_name=robot,
        env_name=env,
        controller_name=controller,
        jsonl_path=jsonl_path,
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
    fire.Fire(generate_with_persona)
