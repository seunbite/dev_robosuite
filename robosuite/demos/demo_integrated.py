"""
This demo integrates SayCan, ViLD and CLIPort for language-driven robotic manipulation.
"""

import os
import time
import numpy as np
import robosuite as suite
from robosuite.utils.input_utils import *
from demo_saycan import SayCanController
from demo_vild import ViLDDetector
from demo_cliport import CLIPortController
from PIL import Image
import datetime

MAX_FR = 25  # max frame rate for running simulation

class IntegratedSystem:
    def __init__(self, env):
        self.env = env
        
        # Initialize components
        self.saycan = SayCanController(env, os.getenv("OPENAI_API_KEY"))
        self.vild = ViLDDetector()
        self.cliport = CLIPortController(env)
        
        # Create directory for saving images
        self.save_dir = "execution_images"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Create timestamped subdirectory for this run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_dir = os.path.join(self.save_dir, timestamp)
        os.makedirs(self.current_run_dir, exist_ok=True)
        
        self.step_counter = 0
        
    def save_image(self, stage=""):
        """Save current simulator view as PNG."""
        image = self.get_camera_image()
        
        # Create filename with step counter and stage
        filename = f"step_{self.step_counter:03d}_{stage}.png"
        filepath = os.path.join(self.current_run_dir, filename)
        
        # Save image
        Image.fromarray(image).save(filepath)
        print(f"Saved image: {filepath}")
        
    def get_camera_image(self):
        """Get camera image from environment."""
        # Get camera matrix
        camera_id = 0
        width = height = 256  # Default size
        
        # Get image from simulator
        img = self.env.sim.render(
            width=width,
            height=height,
            camera_name="frontview",
            depth=False,
            device_id=0
        )
        
        return img
        
    def execute_task(self, task_description):
        """Execute high-level task using integrated components."""
        print(f"Executing task: {task_description}")
        
        # Save initial state
        self.save_image("initial_state")
        
        # Get scene description using ViLD
        image = self.get_camera_image()
        scene_desc = self.vild.get_scene_description(image)
        print(f"Scene description: {scene_desc}")
        
        # Get action plan using SayCan
        plan = self.saycan.get_llm_plan(task_description, scene_desc)
        print("Action plan:")
        for i, step in enumerate(plan):
            print(f"{i+1}. {step}")
            
        # Execute each step using CLIPort
        for i, step in enumerate(plan):
            print(f"\nExecuting step {i+1}: {step}")
            self.step_counter += 1
            
            # Save image before step
            self.save_image(f"step_{i+1}_before")
            
            # Execute step
            obs = self.cliport.execute_instruction(step)
            
            # Save image after step
            self.save_image(f"step_{i+1}_after")
            
            # Update visualization
            self.env.render()
            
        # Save final state
        self.save_image("final_state")
        return obs

def main():
    # Create dict to hold options that will be passed to env creation call
    options = {}

    # Choose environment and add it to options
    options["env_name"] = "Stack"  # or use choose_environment() for interactive selection
    options["robots"] = "Panda"    # or use choose_robots() for interactive selection

    # Initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=True,  # Need both renderers
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        camera_names=["frontview"],
        camera_heights=256,
        camera_widths=256,
    )
    
    # Reset environment
    env.reset()
    env.viewer.set_camera(camera_id=0)
    
    # Initialize system
    system = IntegratedSystem(env)
    
    # Example tasks
    tasks = [
        "Stack all blocks from tallest to shortest",
        "Sort the blocks into matching colored bowls",
        "Move all blocks to the corners",
    ]
    
    # Execute tasks
    for task in tasks:
        print(f"\nExecuting task: {task}")
        
        # Execute task with frame rate control
        start_time = time.time()
        obs = system.execute_task(task)
        env.render()
        
        # Limit frame rate
        elapsed = time.time() - start_time
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)
            
        input("\nPress Enter to continue to next task...")
        
if __name__ == "__main__":
    main()