FIXED_JOINT_INDICES = {
    'GR1' : "0-2, 20-31"
}

SPECIFIED_JOINT_ANGLES = {
    'GR1' : {
        "robot0_head_yaw" : [-30, 0, 30],
        "robot0_head_roll" : [-30, 0, 30],
        "robot0_head_pitch" : [-30, 0, 30],
    }
}

import fire
import os
import numpy as np
from itertools import product
from PIL import Image
from tqdm import tqdm
import math

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config


class GivenJointPoseGenerator:
    """Generate poses for specified joints and save as tiled image."""
    
    def __init__(
        self,
        robot_name: str = "IIWA",
        env_name: str = "Lift",
        controller_name: str = "OSC_POSE",
        output_dir: str = "data/poses",
        capture_image_width: int = 512,
        capture_image_height: int = 512,
        camera_fov: float = 60.0,
    ):
        """
        Initialize the pose generator.
        
        Args:
            robot_name: Name of the robot
            env_name: Name of the environment
            controller_name: Name of the controller
            output_dir: Directory to save pose images
            capture_image_width: Width of captured images
            capture_image_height: Height of captured images
            camera_fov: Camera field of view in degrees
        """
        self.robot_name = robot_name
        self.env_name = env_name
        self.controller_name = controller_name
        self.output_dir = os.path.join(output_dir, robot_name)
        self.camera_fov = camera_fov
        self.capture_image_width = capture_image_width
        self.capture_image_height = capture_image_height
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
        print(f"Initializing robot: {robot_name}")
        
        # Setup environment with offscreen rendering
        options = {
            "env_name": env_name,
            "robots": robot_name,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": True,
            "use_camera_obs": True,
            "camera_names": "frontview",
            "camera_heights": capture_image_height,
            "camera_widths": capture_image_width,
            "control_freq": 20,
        }
        
        # Load controller config
        arm_controller_config = suite.load_part_controller_config(default_controller=controller_name)
        options["controller_configs"] = refactor_composite_controller_config(
            arm_controller_config, robot_name, ["right", "left"]
        )
        
        # Create environment
        self.env = suite.make(**options, horizon=1000)
        self.env.reset()
        
        # Set camera FOV
        camera_name = "frontview"
        try:
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            self.env.sim.model.cam_fovy[cam_id] = camera_fov
            print(f"Camera FOV set to {camera_fov} degrees")
        except Exception as e:
            print(f"Warning: Could not set camera FOV: {e}")
        
        # Get robot
        self.robot = self.env.robots[0]
        
        # Get initial joint positions
        self.initial_joint_pos = self._get_joint_positions()
        self.num_joints = len(self.initial_joint_pos)
        
        # Build joint names list
        try:
            robot_model_joints = list(self.robot.robot_model.joints)
            if len(robot_model_joints) >= len(self.initial_joint_pos):
                base_joint_names = robot_model_joints[:len(self.initial_joint_pos)]
            else:
                base_joint_names = robot_model_joints + [f"joint_{i}" for i in range(len(robot_model_joints), len(self.initial_joint_pos))]
            self.joint_names = base_joint_names
        except:
            self.joint_names = [f"joint_{i}" for i in range(len(self.initial_joint_pos))]
        
        # Parse fixed joint indices from FIXED_JOINT_INDICES
        self.fixed_joint_indices = []
        if robot_name in FIXED_JOINT_INDICES:
            fixed_indices_str = FIXED_JOINT_INDICES[robot_name]
            try:
                for part in fixed_indices_str.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if "-" in part:
                        start, end = part.split("-", 1)
                        start = int(start.strip())
                        end = int(end.strip())
                        if start > end:
                            continue
                        self.fixed_joint_indices.extend(range(start, end + 1))
                    else:
                        self.fixed_joint_indices.append(int(part))
                self.fixed_joint_indices = sorted(list(set([idx for idx in self.fixed_joint_indices if 0 <= idx < self.num_joints])))
            except ValueError as e:
                print(f"Warning: Could not parse fixed_joint_indices: {e}")
                self.fixed_joint_indices = []
    
    def _get_joint_positions(self):
        """Get current joint positions."""
        joint_pos = self.robot._joint_positions.copy()
        return joint_pos
    
    def _set_joint_positions(self, joint_positions):
        """Set joint positions and update simulation."""
        robot_joint_pos = joint_positions
        self.robot.set_robot_joint_positions(robot_joint_pos)
        self.env.sim.forward()
    
    def _capture_image(self, width: int = None, height: int = None):
        """Capture current camera view as numpy array."""
        if width is None:
            width = self.capture_image_width
        if height is None:
            height = self.capture_image_height
        obs = self.env.sim.render(
            camera_name="frontview",
            width=width,
            height=height,
            depth=False
        )
        # Flip vertically (OpenGL convention)
        return obs[::-1]
    
    def _parse_joint_indices(self, joint_indices_str: str) -> list:
        """
        Parse joint indices string like "3, 4, 5" into list of integers.
        
        Args:
            joint_indices_str: Comma-separated joint indices (e.g., "3, 4, 5")
        
        Returns:
            List of joint indices
        """
        joint_indices = []
        for part in joint_indices_str.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                idx = int(part)
                if idx < 0 or idx >= self.num_joints:
                    raise ValueError(f"Joint index {idx} is out of range [0, {self.num_joints-1}]")
                if idx in self.fixed_joint_indices:
                    print(f"Warning: Joint index {idx} is fixed and will be skipped.")
                    continue
                joint_indices.append(idx)
            except ValueError as e:
                raise ValueError(f"Invalid joint index: '{part}'. Error: {e}")
        
        # Remove duplicates and sort
        joint_indices = sorted(list(set(joint_indices)))
        return joint_indices
    
    def generate_tiled_poses(
        self,
        joint_indices_str: str,
        angle_step_deg: float = 30.0,
        angle_min_deg: float = -90.0,
        angle_max_deg: float = 90.0,
        output_filename: str = None,
        tiles_per_row: int = None,
    ):
        """
        Generate poses for specified joints and save as tiled image.
        
        Args:
            joint_indices_str: Comma-separated joint indices to move (e.g., "3, 4, 5")
            angle_step_deg: Step size in degrees
            angle_min_deg: Minimum angle in degrees
            angle_max_deg: Maximum angle in degrees
            output_filename: Output filename for tiled image. If None, auto-generates.
            tiles_per_row: Number of tiles per row. If None, auto-calculates to be roughly square.
        """
        # Parse joint indices
        joint_indices = self._parse_joint_indices(joint_indices_str)
        
        if not joint_indices:
            raise ValueError("No valid joint indices provided.")
        
        print("\n" + "="*60)
        print("GIVEN JOINT POSE GENERATION")
        print("="*60)
        print(f"Joint indices to move: {joint_indices}")
        print(f"Joint names: {[self.joint_names[idx] for idx in joint_indices]}")
        
        # Check if robot has specified joint angles
        specified_angles = SPECIFIED_JOINT_ANGLES.get(self.robot_name, {})
        
        # Default angle range
        angle_min = np.deg2rad(angle_min_deg)
        angle_max = np.deg2rad(angle_max_deg)
        angle_step = np.deg2rad(angle_step_deg)
        default_angles = np.arange(angle_min, angle_max + angle_step/2, angle_step)
        
        # Create angle arrays for each specified joint
        joint_angle_arrays = []
        joint_angle_info = []
        for joint_idx in joint_indices:
            joint_name = self.joint_names[joint_idx]
            
            # Check if this joint has specified angles
            if joint_name in specified_angles:
                angles = np.deg2rad(np.array(specified_angles[joint_name]))
                joint_angle_arrays.append(angles)
                joint_angle_info.append(f"[{joint_idx}] {joint_name}: {specified_angles[joint_name]} (specified)")
            else:
                angles = default_angles
                joint_angle_arrays.append(angles)
                joint_angle_info.append(f"[{joint_idx}] {joint_name}: default range")
        
        # Calculate number of angle positions per joint
        num_angles_per_joint = [len(angles) for angles in joint_angle_arrays]
        
        # Calculate total combinations
        total_combinations = 1
        for num in num_angles_per_joint:
            total_combinations *= num
        
        print(f"\nAngle range: {angle_min_deg}° to {angle_max_deg}° (step: {angle_step_deg}°)")
        print(f"Default angle values: {list(np.rad2deg(default_angles).astype(int))}")
        print(f"\nJoint angle configurations:")
        for i, (joint_idx, info) in enumerate(zip(joint_indices, joint_angle_info)):
            print(f"  {info}")
            print(f"      Angle values: {np.rad2deg(joint_angle_arrays[i]).astype(int).tolist()}")
            print(f"      Number of positions: {num_angles_per_joint[i]}")
        print(f"\nTotal combinations: {total_combinations}")
        print("="*60 + "\n")
        
        # Generate all combinations
        selected_combinations = list(product(*[range(num) for num in num_angles_per_joint]))
        
        # Generate images
        images = []
        print("Generating poses...")
        for combo_idx, angle_indices in tqdm(enumerate(selected_combinations), total=total_combinations):
            # Create joint position array (fixed joints keep initial values)
            joint_pos = self.initial_joint_pos.copy()
            
            # Set positions for specified joints
            for i, joint_idx in enumerate(joint_indices):
                angle_idx = angle_indices[i]
                angle_value = joint_angle_arrays[i][angle_idx]
                joint_pos[joint_idx] = angle_value
            
            # Set joint positions
            self._set_joint_positions(joint_pos)
            
            # Capture image
            image_array = self._capture_image()
            img = Image.fromarray(image_array)
            images.append(img)
            
            # Return to initial pose occasionally to prevent drift
            if (combo_idx + 1) % 100 == 0:
                self._set_joint_positions(self.initial_joint_pos)
        
        # Create tiled image
        print(f"\nCreating tiled image from {len(images)} poses...")
        
        # Calculate grid dimensions
        if tiles_per_row is None:
            # Auto-calculate to be roughly square
            tiles_per_row = int(math.ceil(math.sqrt(len(images))))
        
        num_rows = int(math.ceil(len(images) / tiles_per_row))
        
        print(f"Grid: {num_rows} rows × {tiles_per_row} columns")
        
        # Create tiled image
        tile_width = self.capture_image_width
        tile_height = self.capture_image_height
        tiled_width = tiles_per_row * tile_width
        tiled_height = num_rows * tile_height
        
        tiled_image = Image.new('RGB', (tiled_width, tiled_height), color='white')
        
        # Paste images into grid
        for idx, img in enumerate(images):
            row = idx // tiles_per_row
            col = idx % tiles_per_row
            x = col * tile_width
            y = row * tile_height
            tiled_image.paste(img, (x, y))
        
        # Generate output filename if not provided
        if output_filename is None:
            joint_str = "_".join([f"j{idx}" for idx in joint_indices])
            angles_str = f"step{int(angle_step_deg)}_min{int(angle_min_deg)}_max{int(angle_max_deg)}"
            output_filename = f"{self.robot_name}_tiled_{joint_str}_{angles_str}.png"
        
        # Save tiled image
        filepath = os.path.join(self.output_dir, output_filename)
        tiled_image.save(filepath)
        
        print(f"\n{'='*60}")
        print(f"Saved tiled image: {filepath}")
        print(f"Total poses: {len(images)}")
        print(f"Image size: {tiled_width} × {tiled_height} pixels")
        print(f"{'='*60}\n")
        
        return filepath
    
    def close(self):
        """Close the environment."""
        self.env.close()


def _normalize_joint_indices(joint_indices):
    """
    Normalize joint_indices input to string format.
    Fire may parse comma-separated strings as tuples, so we handle both cases.
    
    Args:
        joint_indices: String like "3, 4, 5" or tuple/list like (3, 4, 5)
    
    Returns:
        String like "3, 4, 5"
    """
    if isinstance(joint_indices, (tuple, list)):
        # Convert tuple/list to comma-separated string
        return ", ".join(str(x) for x in joint_indices)
    elif isinstance(joint_indices, str):
        return joint_indices
    else:
        return str(joint_indices)


def main(
    robot: str = "GR1",
    env: str = "Lift",
    joint_indices: str = "3, 4, 5",
    angle_step: float = 60.0,
    angle_min: float = -90.0,
    angle_max: float = 90.0,
    output_filename: str = None,
    tiles_per_row: int = None,
    output_dir: str = "data/poses",
    capture_image_width: int = 512,
    capture_image_height: int = 512,
    camera_fov: float = 60.0,
):
    """
    Generate poses for specified joints and save as tiled image.
    
    Args:
        robot: Robot name (IIWA, Panda, GR1, etc.)
        env: Environment name
        joint_indices: Comma-separated joint indices to move (e.g., "3, 4, 5")
        angle_step: Angle step size in degrees
        angle_min: Minimum angle in degrees
        angle_max: Maximum angle in degrees
        output_filename: Output filename for tiled image. If None, auto-generates.
        tiles_per_row: Number of tiles per row. If None, auto-calculates to be roughly square.
        output_dir: Directory to save pose images
        capture_image_width: Width of captured images
        capture_image_height: Height of captured images
        camera_fov: Camera field of view in degrees
    
    Examples:
        # Generate poses for joints 3, 4, 5
        python stack_given_joint.py --joint-indices "3, 4, 5"
        
        # Custom angle range and step
        python stack_given_joint.py --joint-indices "3, 4, 5" --angle-step 45 --angle-min -90 --angle-max 90
        
        # Custom output filename and tiles per row
        python stack_given_joint.py --joint-indices "3, 4, 5" --output-filename "my_tiled.png" --tiles-per-row 5
    """
    
    # Normalize joint_indices (Fire may parse comma-separated strings as tuples)
    joint_indices_str = _normalize_joint_indices(joint_indices)
    
    print("="*60)
    print("GIVEN JOINT POSE GENERATOR - TILED IMAGE")
    print("="*60)
    print(f"Robot: {robot}")
    print(f"Environment: {env}")
    print(f"Joint indices: {joint_indices_str}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Create generator
    generator = GivenJointPoseGenerator(
        robot_name=robot,
        env_name=env,
        output_dir=output_dir,
        capture_image_width=capture_image_width,
        capture_image_height=capture_image_height,
        camera_fov=camera_fov,
    )
    
    try:
        generator.generate_tiled_poses(
            joint_indices_str=joint_indices_str,
            angle_step_deg=angle_step,
            angle_min_deg=angle_min,
            angle_max_deg=angle_max,
            output_filename=output_filename,
            tiles_per_row=tiles_per_row,
        )
    
    finally:
        generator.close()
        print("\nDone!")


if __name__ == "__main__":
    fire.Fire(main)

