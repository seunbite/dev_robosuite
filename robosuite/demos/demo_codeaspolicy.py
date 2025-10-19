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
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config

from PIL import Image
import datetime
import openai
import threading
import cv2
import shapely.geometry
import shapely.affinity
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
        
        # Initialize object tracking
        self.obj_name_to_id = {}
        self.setup_objects()
        
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
        
    def setup_objects(self):
        """Setup initial objects in the environment."""
        # Define available colors and objects
        self.COLORS = {
            'blue':   (78/255,  121/255, 167/255, 255/255),
            'red':    (255/255,  87/255,  89/255, 255/255),
            'green':  (89/255,  169/255,  79/255, 255/255),
            'yellow': (237/255, 201/255,  72/255, 255/255),
        }
        
        # Map existing objects in the environment
        self.map_existing_objects()
        
    def map_existing_objects(self):
        """Map existing objects in the Stack environment."""
        # Stack environment has cubeA and cubeB by default
        cubes = [self.env.cubeA, self.env.cubeB]
        colors = ['red', 'blue']  # Fixed colors for the two cubes
        
        for cube, color in zip(cubes, colors):
            block_name = f"{color} block"
            self.obj_name_to_id[block_name] = cube

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
                },
                'parse_obj_name': {
                    'prompt_text': self.get_parse_obj_name_prompt(),
                    'engine': self.model_name,
                    'max_tokens': 512,
                    'temperature': 0,
                    'query_prefix': '# ',
                    'query_suffix': '.',
                    'stop': ['#', 'objects = ['],
                    'maintain_session': False,
                    'debug_mode': False,
                    'include_context': True,
                    'has_return': True,
                    'return_val_name': 'ret_val',
                },
                'parse_position': {
                    'prompt_text': self.get_parse_position_prompt(),
                    'engine': self.model_name,
                    'max_tokens': 512,
                    'temperature': 0,
                    'query_prefix': '# ',
                    'query_suffix': '.',
                    'stop': ['#'],
                    'maintain_session': False,
                    'debug_mode': False,
                    'include_context': True,
                    'has_return': True,
                    'return_val_name': 'ret_val',
                },
                'parse_question': {
                    'prompt_text': self.get_parse_question_prompt(),
                    'engine': self.model_name,
                    'max_tokens': 512,
                    'temperature': 0,
                    'query_prefix': '# ',
                    'query_suffix': '.',
                    'stop': ['#', 'objects = ['],
                    'maintain_session': False,
                    'debug_mode': False,
                    'include_context': True,
                    'has_return': True,
                    'return_val_name': 'ret_val',
                },
                'transform_shape_pts': {
                    'prompt_text': self.get_transform_shape_pts_prompt(),
                    'engine': self.model_name,
                    'max_tokens': 512,
                    'temperature': 0,
                    'query_prefix': '# ',
                    'query_suffix': '.',
                    'stop': ['#'],
                    'maintain_session': False,
                    'debug_mode': False,
                    'include_context': True,
                    'has_return': True,
                    'return_val_name': 'new_shape_pts',
                },
                'fgen': {
                    'prompt_text': self.get_fgen_prompt(),
                    'engine': self.model_name,
                    'max_tokens': 512,
                    'temperature': 0,
                    'query_prefix': '# define function: ',
                    'query_suffix': '.',
                    'stop': ['# define', '# example'],
                    'maintain_session': False,
                    'debug_mode': False,
                    'include_context': True,
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
        from env_utils import put_first_on_second, get_obj_pos, get_obj_names, say, get_corner_name, get_side_name, is_obj_visible, stack_objects_in_order
        from plan_utils import parse_obj_name, parse_position, parse_question, transform_shape_pts

        objects = ['blue block', 'red block', 'green block']
        # put the blue block on the red block.
        say('Ok - putting the blue block on the red block')
        put_first_on_second('blue block', 'red block')
        
        objects = ['blue block', 'red block', 'green block']
        # stack all blocks with green on top.
        say('Stacking blocks with green block on top')
        order_bottom_to_top = ['red block', 'blue block', 'green block']
        stack_objects_in_order(object_names=order_bottom_to_top)
        
        objects = ['yellow block', 'green block', 'yellow bowl']
        # put the yellow block in its matching bowl.
        say('Putting the yellow block in the yellow bowl')
        put_first_on_second('yellow block', 'yellow bowl')
        
        objects = ['blue block', 'red block', 'green block']
        # arrange blocks in a triangle.
        say('Arranging blocks in a triangle formation')
        triangle_pts = parse_position('a triangle with size 10cm around the middle with 3 points')
        for block_name, pt in zip(['blue block', 'red block', 'green block'], triangle_pts):
            put_first_on_second(block_name, pt)
        """

    def get_parse_obj_name_prompt(self):
        """Get the prompt template for parsing object names."""
        return """
        import numpy as np
        from env_utils import get_obj_pos, parse_position
        from utils import get_obj_positions_np

        objects = ['blue block', 'cyan block', 'purple bowl']
        # the block closest to the purple bowl.
        block_names = ['blue block', 'cyan block']
        block_positions = get_obj_positions_np(block_names)
        closest_block_idx = get_closest_idx(points=block_positions, point=get_obj_pos('purple bowl'))
        closest_block_name = block_names[closest_block_idx]
        ret_val = closest_block_name
        """

    def get_parse_position_prompt(self):
        """Get the prompt template for parsing positions."""
        return """
        import numpy as np
        from shapely.geometry import *
        from shapely.affinity import *
        from env_utils import denormalize_xy, parse_obj_name, get_obj_names, get_obj_pos

        # a triangle with size 10cm with 3 points.
        polygon = make_triangle(size=0.1, center=denormalize_xy([0.5, 0.5]))
        points = get_points_from_polygon(polygon)
        ret_val = points
        """

    def get_parse_question_prompt(self):
        """Get the prompt template for parsing questions."""
        return """
        from utils import get_obj_pos, get_obj_names, parse_obj_name, bbox_contains_pt

        objects = ['yellow bowl', 'blue block', 'yellow block']
        # is the blue block to the right of the yellow bowl?
        ret_val = get_obj_pos('blue block')[0] > get_obj_pos('yellow bowl')[0]
        """

    def get_transform_shape_pts_prompt(self):
        """Get the prompt template for transforming shape points."""
        return """
        import numpy as np
        from utils import get_obj_pos, get_obj_names, parse_position, parse_obj_name

        # make it bigger by 1.5.
        new_shape_pts = scale_pts_around_centroid_np(shape_pts, scale_x=1.5, scale_y=1.5)
        """

    def get_fgen_prompt(self):
        """Get the prompt template for function generation."""
        return """
        import numpy as np
        from shapely.geometry import *
        from shapely.affinity import *
        from env_utils import get_obj_pos, get_obj_names
        from ctrl_utils import put_first_on_second

        # define function: total = get_total(xs=numbers).
        def get_total(xs):
            return np.sum(xs)
        """

    def setup_lmp_env(self):
        """Setup the LMP environment wrapper."""
        # Initialize environment configuration
        self.cfg_tabletop['env'] = {
            'init_objs': list(self.obj_name_to_id.keys()),
            'coords': {
                'bottom_left': (-0.3, -0.8),
                'top_right': (0.3, -0.2),
                'table_z': 0.0
            }
        }

        # Create LMP environment wrapper
        from codeaspolicies.env_wrapper import LMPWrapper
        lmp_env = LMPWrapper(self, self.cfg_tabletop, render=True)  # Pass self instead of env

        # Setup fixed and variable variables
        fixed_vars = {'np': np}
        fixed_vars.update({
            name: eval(name)
            for name in shapely.geometry.__all__ + shapely.affinity.__all__
        })

        variable_vars = {
            k: getattr(lmp_env, k)
            for k in [
                'get_bbox', 'get_obj_pos', 'get_color', 'is_obj_visible',
                'denormalize_xy', 'put_first_on_second', 'get_obj_names',
                'get_corner_name', 'get_side_name',
            ]
        }
        variable_vars['say'] = lambda msg: print(f'Robot says: {msg}')

        # Create function generator
        from codeaspolicies.lmp_utils import LMPFGen, LMP
        lmp_fgen = LMPFGen(self.cfg_tabletop['lmps']['fgen'], fixed_vars, variable_vars)

        # Create LMP components
        variable_vars.update({
            k: LMP(k, self.cfg_tabletop['lmps'][k], lmp_fgen, fixed_vars, variable_vars)
            for k in ['parse_obj_name', 'parse_position', 'parse_question', 'transform_shape_pts']
        })

        # Create main UI LMP
        self.lmp_tabletop_ui = LMP(
            'tabletop_ui',
            self.cfg_tabletop['lmps']['tabletop_ui'],
            lmp_fgen,
            fixed_vars,
            variable_vars
        )

        return lmp_env

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
            # Get current object list
            object_list = list(self.obj_name_to_id.keys())
            
            # Execute task using LMP
            self.lmp_tabletop_ui(task_description, f'objects = {object_list}')
            
            # Render environment
            self.env.render()
            
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

    # Create environment configuration
    options = {
        "env_name": env_name,
        "robots": robot,
        "controller_configs": None,  # Will be set after env creation
        
        # Basic parameters
        "control_freq": 20,
        "horizon": 1000,
        "ignore_done": True,
        
        # Visualization parameters
        "has_renderer": True,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        
        # Camera parameters
        "camera_names": ["agentview"],
        "camera_heights": 256,
        "camera_widths": 256,
    }

    # Load the IK_POSE controller
    controller_name = "IK_POSE"
    arm_controller_config = suite.load_part_controller_config(default_controller=controller_name)
    robot = options["robots"][0] if isinstance(options["robots"], list) else options["robots"]
    options["controller_configs"] = refactor_composite_controller_config(
        arm_controller_config, robot, ["right", "left"]
    )

    # Initialize the task
    env = suite.make(
        **options
    )
    
    # Reset environment and set camera
    env.reset()
    env.viewer.set_camera(camera_id=0)
    env.render()  # Initial render to setup viewer
    
    # Initialize Code as Policies controller
    controller = CodeAsPoliciesController(env, api_key)
    
    # Example tasks
    tasks = [
        "Stack red block on the blue block",
    ]
    
    # Execute tasks
    for task in tasks:
        print(f"\nExecuting task: {task}")
        
        # Execute task with frame rate control
        start_time = time.time()
        
        # Make sure environment is visible
        env.render()
        time.sleep(1)  # Give time for viewer to initialize
        
        # Execute task
        controller.execute_task(task)
        
        # Ensure final state is rendered
        env.render()
        
        # Limit frame rate
        elapsed = time.time() - start_time
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)
            
        input("\nPress Enter to continue to next task...")
        
if __name__ == "__main__":
    main(api_key=os.getenv("OPENAI_API_KEY"))
