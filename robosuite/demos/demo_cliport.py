"""
This demo implements a simplified version of CLIPort for robosuite.
Uses CLIP for language understanding and simple heuristic-based action selection.
"""

import numpy as np
import torch
import clip
from PIL import Image
import cv2

class CLIPortController:
    def __init__(self, env):
        self.env = env
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Define target positions
        self.target_positions = {
            "middle": np.array([0.0, -0.5, 0.15]),
            "left": np.array([-0.2, -0.5, 0.15]),
            "right": np.array([0.2, -0.5, 0.15]),
            "front": np.array([0.0, -0.3, 0.15]),
            "back": np.array([0.0, -0.7, 0.15]),
        }
        
        # Define object colors and their typical positions
        self.object_colors = ["red", "green", "blue", "yellow"]
        
    def get_camera_image(self):
        """Get camera image from environment."""
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
        
    def find_object_position(self, image, target_color):
        """Find object position using color detection."""
        # Convert image to HSV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define color ranges
        color_ranges = {
            "red": [(0, 50, 50), (10, 255, 255)],
            "green": [(35, 50, 50), (85, 255, 255)],
            "blue": [(100, 50, 50), (130, 255, 255)],
            "yellow": [(20, 50, 50), (35, 255, 255)]
        }
        
        if target_color not in color_ranges:
            return None
            
        # Create mask for target color
        lower, upper = color_ranges[target_color]
        mask = cv2.inRange(image, np.array(lower), np.array(upper))
        
        # Find centroid of the largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Convert to robot coordinates
        x = cx / 256.0 * 0.6 - 0.3  # Scale to workspace bounds
        y = -0.8 + cy / 256.0 * 0.6
        
        return np.array([x, y, 0.03])  # Fixed height for picking
        
    def parse_instruction(self, instruction):
        """Parse instruction to extract target object and location."""
        instruction = instruction.lower()
        
        # Find target object color
        target_color = None
        for color in self.object_colors:
            if color in instruction:
                target_color = color
                break
                
        # Find target location
        target_location = None
        for loc in self.target_positions.keys():
            if loc in instruction:
                target_location = loc
                break
                
        return target_color, target_location
        
    def get_action(self, instruction):
        """Get pick and place action from instruction."""
        # Parse instruction
        target_color, target_location = self.parse_instruction(instruction)
        if not target_color:
            print(f"Could not find target object in instruction: {instruction}")
            return None
            
        # Get current image
        image = self.get_camera_image()
        
        # Find object position
        pick_xyz = self.find_object_position(image, target_color)
        if pick_xyz is None:
            print(f"Could not find {target_color} object in image")
            return None
            
        # Get place position
        if target_location and target_location in self.target_positions:
            place_xyz = self.target_positions[target_location]
        else:
            # Default to middle if no location specified
            place_xyz = self.target_positions["middle"]
        
        return {
            'pick': pick_xyz,
            'place': place_xyz
        }
        
    def execute_instruction(self, instruction):
        """Execute language instruction."""
        # Get action
        action = self.get_action(instruction)
        if action is None:
            print("Failed to generate action")
            return None
        
        # Execute action
        obs, reward, done, info = self.env.step(action)
        
        return obs

def main():
    import robosuite as suite
    
    # Create environment
    env = suite.make(
        "Stack",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
    )
    
    # Initialize controller
    controller = CLIPortController(env)
    
    # Reset environment
    obs = env.reset()
    
    # Example instruction
    instruction = "Pick up the blue block and place it in the middle"
    
    # Execute instruction
    obs = controller.execute_instruction(instruction)
    
if __name__ == "__main__":
    main()