"""
This demo implements Code as Policies for language-driven robotic manipulation.
It uses IK_POSE controller for precise end-effector control.

Code as Policies: Language Model Programs for Embodied Control
Paper: https://code-as-policies.github.io/

Copyright 2022 Google LLC. SPDX-License-Identifier: Apache-2.0
"""

import os
import time
import numpy as np
import robosuite as suite
from robosuite.utils.input_utils import *
from robosuite.controllers.composite.composite_controller_config import refactor_composite_controller_config
from PIL import Image
import datetime
import openai
import threading
import pybullet
import cv2
from shapely.geometry import *
from shapely.affinity import *

MAX_FR = 25  # max frame rate for running simulation

# Global constants for pick and place objects, colors, workspace bounds
COLORS = {
    'blue':   (78/255,  121/255, 167/255, 255/255),
    'red':    (255/255,  87/255,  89/255, 255/255),
    'green':  (89/255,  169/255,  79/255, 255/255),
    'orange': (242/255, 142/255,  43/255, 255/255),
    'yellow': (237/255, 201/255,  72/255, 255/255),
    'purple': (176/255, 122/255, 161/255, 255/255),
    'pink':   (255/255, 157/255, 167/255, 255/255),
    'cyan':   (118/255, 183/255, 178/255, 255/255),
    'brown':  (156/255, 117/255,  95/255, 255/255),
    'gray':   (186/255, 176/255, 172/255, 255/255),
}

CORNER_POS = {
    'top left corner':     (-0.3 + 0.05, -0.2 - 0.05, 0),
    'top side':            (0,           -0.2 - 0.05, 0),
    'top right corner':    (0.3 - 0.05,  -0.2 - 0.05, 0),
    'left side':           (-0.3 + 0.05, -0.5,        0),
    'middle':              (0,           -0.5,        0),
    'right side':          (0.3 - 0.05,  -0.5,        0),
    'bottom left corner':  (-0.3 + 0.05, -0.8 + 0.05, 0),
    'bottom side':         (0,           -0.8 + 0.05, 0),
    'bottom right corner': (0.3 - 0.05,  -0.8 + 0.05, 0),
}

BOUNDS = np.float32([[-0.3, 0.3], [-0.8, -0.2], [0, 0.15]])  # X Y Z
PIXEL_SIZE = 0.00267857

class CodeAsPoliciesController:
    """Controller that uses Code as Policies for language-driven manipulation."""
    
    def __init__(self, env, api_key):
        self.env = env
        openai.api_key = api_key
        self.model_name = 'code-davinci-002'  # or 'text-davinci-002'
        
        # Initialize LMP components
        self.setup_lmp_components()
        
        # Get IK controller settings
        self.action_dim = 6  # 6-DOF pose control
        self.gripper_dim = env.robots[0].gripper["right"].dof if hasattr(env.robots[0], "gripper") else 0
        self.neutral = np.zeros(self.action_dim + self.gripper_dim)
        
        # Create directory for saving images
        self.save_dir = "execution_images"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Create timestamped subdirectory for this run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_dir = os.path.join(self.save_dir, timestamp)
        os.makedirs(self.current_run_dir, exist_ok=True)
        
        self.step_counter = 0

    def setup_lmp_components(self):
        """Setup Language Model Programming (LMP) components."""
        # Initialize LMP configuration
        self.cfg_tabletop = {
            'lmps': {
                'tabletop_ui': {
                    'prompt_text': self.get_tabletop_ui_prompt(),
                    'engine': self.model_name,
                    'max_tokens': 512,
                    'temperature': 0,
                    'query_prefix': '# ',
                    'query_suffix': '.',
                    'stop': ['#', 'objects = ['],
                    'maintain_session': True,
                    'debug_mode': False,
                    'include_context': True,
                    'has_return': False,
                    'return_val_name': 'ret_val',
                }
            }
        }

        # Setup LMP environment wrapper
        self.lmp_env = self.setup_lmp_env()

    def get_tabletop_ui_prompt(self):
        """Get the prompt template for the tabletop UI LMP."""
        return """
        # Python 2D robot control script
        import numpy as np
        from env_utils import put_first_on_second, get_obj_pos, get_obj_names, say
        
        objects = ['blue block', 'red block', 'green block']
        # put the blue block on the red block.
        say('Ok - putting the blue block on the red block')
        put_first_on_second('blue block', 'red block')
        
        objects = ['blue block', 'red block', 'green block']
        # stack all blocks with green on top.
        say('Stacking blocks with green block on top')
        order_bottom_to_top = ['red block', 'blue block', 'green block']
        stack_objects_in_order(object_names=order_bottom_to_top)
        """

    def setup_lmp_env(self):
        """Setup the LMP environment wrapper."""
        # TODO: Implement LMP environment wrapper
        pass

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
        """Execute high-level task using Code as Policies."""
        print(f"Executing task: {task_description}")
        
        # Save initial state
        self.save_image("initial_state")
        
        # Generate and execute code using LMP
        try:
            # TODO: Implement LMP code generation and execution
            pass
            
        except Exception as e:
            print(f"Error executing task: {e}")
            
        # Save final state
        self.save_image("final_state")

def main(
    env_name = "Stack",
    robot = "Panda",
    api_key = None,
):
    if api_key is None:
        raise ValueError("Please provide an OpenAI API key")

    options = {}
    options["env_name"] = env_name
    options["robots"] = robot

    # Load the IK_POSE controller
    controller_name = "IK_POSE"
    arm_controller_config = suite.load_part_controller_config(default_controller=controller_name)
    robot = options["robots"][0] if isinstance(options["robots"], list) else options["robots"]
    options["controller_configs"] = refactor_composite_controller_config(
        arm_controller_config, robot, ["right", "left"]
    )

    # Initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=True,
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
    
    # Initialize Code as Policies controller
    controller = CodeAsPoliciesController(env, api_key)
    
    # Example tasks
    tasks = [
        "Stack red block on the green block",
        "Put the blue block in the yellow bowl",
        "Move all blocks to the right side",
    ]
    
    # Execute tasks
    for task in tasks:
        print(f"\nExecuting task: {task}")
        
        # Execute task with frame rate control
        start_time = time.time()
        controller.execute_task(task)
        env.render()
        
        # Limit frame rate
        elapsed = time.time() - start_time
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)
            
        input("\nPress Enter to continue to next task...")
        
if __name__ == "__main__":
    main(api_key=os.getenv("OPENAI_API_KEY"))
