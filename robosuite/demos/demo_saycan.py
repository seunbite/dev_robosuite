"""
This demo shows how to use SayCan with robosuite.
SayCan combines LLM planning with robotic affordances for long-horizon tasks.
"""

import numpy as np
import robosuite as suite
import openai
import clip
import torch
import PIL.Image as Image
from collections import defaultdict
import datetime
import os

# Constants for the environment
PICK_TARGETS = {
    "blue block": None,
    "red block": None,
    "green block": None,
    "yellow block": None,
}

PLACE_TARGETS = {
    "blue bowl": None,
    "red bowl": None,
    "green bowl": None,
    "yellow bowl": None,
    "top left corner": (-0.3 + 0.05, -0.2 - 0.05, 0),
    "top right corner": (0.3 - 0.05, -0.2 - 0.05, 0),
    "middle": (0, -0.5, 0),
    "bottom left corner": (-0.3 + 0.05, -0.8 + 0.05, 0),
    "bottom right corner": (0.3 - 0.05, -0.8 + 0.05, 0),
}

class SayCanController:
    def __init__(self, env, openai_api_key):
        self.env = env
        openai.api_key = openai_api_key
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        self.clip_model.cuda().eval()
        self.llm_cache = {}
        
    def get_scene_description(self):
        """Get scene description using ViLD."""
        # Get top-down image
        img = self.env.get_camera_image_top()
        # TODO: Implement ViLD scene detection
        return "Scene contains: blue block, red block, green bowl"
        
    def get_llm_plan(self, task_description, scene_description):
        """Get high-level plan from LLM."""
        prompt = f"""Scene: {scene_description}
Task: {task_description}
Plan:
1."""
        
        response = self.call_llm(prompt)
        return self.parse_llm_response(response)
        
    def call_llm(self, prompt, engine="text-davinci-002"):
        """Call LLM with caching."""
        cache_key = (prompt, engine)
        if cache_key in self.llm_cache:
            return self.llm_cache[cache_key]
            
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=256,
            temperature=0
        )
        self.llm_cache[cache_key] = response
        return response
        
    def parse_llm_response(self, response):
        """Parse LLM response into action sequence."""
        # Extract steps from response
        steps = []
        for line in response.choices[0].text.split('\n'):
            if line.startswith('Pick') or line.startswith('Place'):
                steps.append(line.strip())
        return steps
        
    def get_action(self, step_description):
        """Convert step description to robot action."""
        # Parse step into pick/place targets
        if 'Pick' in step_description:
            target = step_description.split('Pick the ')[-1].split(' and')[0]
            action = {'pick': target}
        elif 'Place' in step_description:
            target = step_description.split('Place it on ')[-1]
            action = {'place': target}
        return action
        
    def execute_plan(self, task_description):
        """Execute full task plan."""
        # Get scene description
        scene_desc = self.get_scene_description()
        
        # Get plan from LLM
        plan = self.get_llm_plan(task_description, scene_desc)
        
        # Execute each step
        for step in plan:
            action = self.get_action(step)
            obs, reward, done, info = self.env.step(action)
            
        return obs

def main():
    # Create environment
    env = suite.make(
        "Stack",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=True,
        control_freq=20,
    )
    
    # Initialize controller
    controller = SayCanController(env, os.getenv("OPENAI_API_KEY"))
    
    # Reset environment
    obs = env.reset()
    
    # Example task
    task = "Pick up the blue block and stack it on the red block"
    
    # Execute task
    obs = controller.execute_plan(task)
    
if __name__ == "__main__":
    main()
