"""
Generate and save preset robot poses by rotating all joints in combinations.

This script:
1. Initializes a robot in robosuite
2. Systematically rotates ALL joints through combinations of angles (e.g., every 30°)
3. Captures either:
   - Camera images of each pose (default)
   - 3D plots of joint positions (with --use-3d-points True)
4. Saves them as PNG files

Example:
    Joint 0,1,2,3,4,5,6 = [  0°,   0°,   0°,   0°,   0°,   0°,   0°]
    Joint 0,1,2,3,4,5,6 = [  0°,   0°,   0°,   0°,   0°,   0°,  30°]
    Joint 0,1,2,3,4,5,6 = [  0°,   0°,   0°,   0°,   0°,  30°,   0°]
    ...

3D Points Mode:
    - Saves 3D matplotlib plots showing joint positions
    - Points: Purple → Yellow gradient (root body → last joint)
    - Lines: Black → White gradient connecting joints
    - Saved to: {output_dir}/{robot_name}/3d_points/

WARNING: This generates MANY poses! 
    - 7 joints, 30° step, ±90° range = 7^7 = 823,543 poses
    - 7 joints, 30° step, ±180° range = 13^7 = 62,748,517 poses

Usage:
    # Camera images
    python stack_preset.py --robot IIWA --angle-step 90
    
    # 3D point plots
    python stack_preset.py --robot Panda --angle-step 90 --use-3d-points True
"""

import fire
import os
import time
import numpy as np
import itertools
from itertools import product
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config


class PresetPoseGenerator:
    """Generate and save preset robot poses."""
    
    def __init__(
        self,
        robot_name: str = "IIWA",
        env_name: str = "Lift",
        controller_name: str = "OSC_POSE",
        output_dir: str = "data/poses",
        capture_image_width: int = 1024,
        capture_image_height: int = 1024,
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
            camera_fov: Camera field of view in degrees (larger = wider view)
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
        
        print(f"Initializing robot: {robot_name}")
        
        # Setup environment with offscreen rendering
        options = {
            "env_name": env_name,
            "robots": robot_name,
            "has_renderer": False,  # No on-screen rendering
            "has_offscreen_renderer": True,  # Enable image capture
            "ignore_done": True,
            "use_camera_obs": True,
            "camera_names": "frontview",  # or "agentview", "sideview"
            "camera_heights": 1024,
            "camera_widths": 1024,
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
        
        # Set camera FOV for wider view
        camera_name = "frontview"
        try:
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            # MuJoCo cam_fovy is stored in degrees
            self.env.sim.model.cam_fovy[cam_id] = camera_fov
            print(f"Camera FOV set to {camera_fov} degrees")
        except Exception as e:
            print(f"Warning: Could not set camera FOV: {e}")
        
        # Get robot
        self.robot = self.env.robots[0]
        
        # Get initial joint positions
        self.initial_joint_pos = self._get_joint_positions()
        self.num_joints = len(self.initial_joint_pos)
        
        print(f"Robot initialized with {self.num_joints} joints")
        print(f"Initial joint positions: {np.rad2deg(self.initial_joint_pos)}")
    
    def _get_joint_positions(self):
        """Get current joint positions."""
        return self.robot._joint_positions.copy()
    
    def _set_joint_positions(self, joint_positions):
        """Set joint positions and update simulation."""
        self.robot.set_robot_joint_positions(joint_positions)
        self.env.sim.forward()
        
        # Step a few times to stabilize
        for _ in range(10):
            self.env.step(np.zeros(self.robot.action_dim))
    
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
    
    def _save_image(self, image_array, filename):
        """Save image array as PNG."""
        img = Image.fromarray(image_array)
        filepath = os.path.join(self.output_dir, filename)
        img.save(filepath)
        return filepath
    
    def _save_pose_data_to_jsonl(self, data_entry, jsonl_path):
        """
        Save pose data to JSONL file (append mode).
        
        Args:
            data_entry: Dictionary containing pose information
            jsonl_path: Path to JSONL file
        """
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps(data_entry) + '\n')
    
    def _get_3d_joint_positions(self):
        """
        Get 3D positions of all joints and root body in world coordinates.
        
        Returns:
            np.ndarray: Array of shape (num_joints + 1, 3) with [x, y, z] positions
                First row is root body, remaining rows are joint positions
        """
        positions = []
        
        # Get root body position
        root_body = self.robot.robot_model.root_body
        root_pos = self.env.sim.data.get_body_xpos(root_body)
        positions.append(root_pos)
        
        # Get each joint position
        # Use the link/body associated with each joint
        for joint_name in self.robot.robot_model.joints:
            try:
                # Get the body associated with this joint
                joint_id = self.env.sim.model.joint_name2id(joint_name)
                body_id = self.env.sim.model.jnt_bodyid[joint_id]
                body_name = self.env.sim.model.body_id2name(body_id)
                body_pos = self.env.sim.data.get_body_xpos(body_name)
                positions.append(body_pos)
            except:
                # If we can't get body position, skip this joint
                continue
        
        return np.array(positions)
    
    def _get_object_positions(self):
        """
        Get positions of objects in the environment (table, cubes, etc.).
        
        Returns:
            dict: Dictionary mapping object names to their [x, y, z] positions
        """
        object_positions = {}
        
        # Get all body names in the simulation
        body_names = [self.env.sim.model.body_id2name(i) 
                     for i in range(self.env.sim.model.nbody)]
        
        # Keywords to identify environment objects
        object_keywords = ['table', 'cube', 'object', 'obj', 'goal', 'target', 
                          'obstacle', 'box', 'can', 'bin', 'peg', 'hole']
        
        for body_name in body_names:
            if body_name is None:
                continue
            
            # Check if this is an object (not part of robot)
            is_object = False
            for keyword in object_keywords:
                if keyword in body_name.lower():
                    is_object = True
                    break
            
            # Skip robot bodies
            if self.robot_name.lower() in body_name.lower():
                continue
            
            if is_object:
                try:
                    pos = self.env.sim.data.get_body_xpos(body_name)
                    object_positions[body_name] = pos.copy()
                except:
                    pass
        
        return object_positions
    
    def _save_3d_plot(self, positions, filename, object_positions=None):
        """
        Create and save a 3D plot of joint positions and environment objects.
        
        Args:
            positions: Array of shape (N, 3) with [x, y, z] positions (robot joints)
            filename: Name of file to save
            object_positions: Dict mapping object names to [x, y, z] positions
        
        Colors:
            - Robot Points: Purple to Yellow gradient
            - Robot Lines: Black to White gradient
            - Table: Light gray, semi-transparent
            - Objects: Cyan markers
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        num_points = len(positions)
        
        # Create color gradients
        # Points: Purple (138, 43, 226) to Yellow (255, 255, 0)
        purple = np.array([138, 43, 226]) / 255.0
        yellow = np.array([255, 255, 0]) / 255.0
        point_colors = [purple + (yellow - purple) * i / (num_points - 1) 
                       for i in range(num_points)]
        
        # Lines: Black (0, 0, 0) to White (255, 255, 255)
        black = np.array([0, 0, 0]) / 255.0
        white = np.array([255, 255, 255]) / 255.0
        line_colors = [black + (white - black) * i / (num_points - 2) 
                      for i in range(num_points - 1)]
        
        # Plot lines connecting joints
        for i in range(num_points - 1):
            ax.plot3D(
                positions[i:i+2, 0],
                positions[i:i+2, 1],
                positions[i:i+2, 2],
                color=line_colors[i],
                linewidth=2,
                alpha=0.8
            )
        
        # Plot points (joints)
        for i, pos in enumerate(positions):
            ax.scatter(
                pos[0], pos[1], pos[2],
                color=point_colors[i],
                s=200,  # Size
                edgecolors='black',
                linewidths=1,
                alpha=0.9,
                zorder=10  # Draw on top
            )
        
        # Set labels and title
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title(f'{self.robot_name} Joint Positions', fontsize=14)
        
        # Set equal aspect ratio
        max_range = np.array([
            positions[:, 0].max() - positions[:, 0].min(),
            positions[:, 1].max() - positions[:, 1].min(),
            positions[:, 2].max() - positions[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Plot environment objects if provided
        if object_positions:
            for obj_name, obj_pos in object_positions.items():
                # Different markers for different object types
                if 'table' in obj_name.lower():
                    # Table: large semi-transparent gray marker at bottom
                    ax.scatter(
                        obj_pos[0], obj_pos[1], obj_pos[2],
                        color='lightgray',
                        s=500,  # Large size
                        marker='s',  # Square
                        alpha=0.5,
                        edgecolors='gray',
                        linewidths=2,
                        label='Table',
                        zorder=1  # Draw below robot
                    )
                elif any(kw in obj_name.lower() for kw in ['cube', 'object', 'obj', 'box']):
                    # Objects: cyan markers
                    ax.scatter(
                        obj_pos[0], obj_pos[1], obj_pos[2],
                        color='cyan',
                        s=150,
                        marker='o',
                        alpha=0.8,
                        edgecolors='blue',
                        linewidths=1.5,
                        label=obj_name,
                        zorder=5
                    )
                elif any(kw in obj_name.lower() for kw in ['goal', 'target']):
                    # Goals: green markers
                    ax.scatter(
                        obj_pos[0], obj_pos[1], obj_pos[2],
                        color='limegreen',
                        s=120,
                        marker='*',  # Star
                        alpha=0.8,
                        edgecolors='darkgreen',
                        linewidths=1,
                        label=obj_name,
                        zorder=5
                    )
                else:
                    # Other objects: orange markers
                    ax.scatter(
                        obj_pos[0], obj_pos[1], obj_pos[2],
                        color='orange',
                        s=100,
                        marker='^',  # Triangle
                        alpha=0.7,
                        edgecolors='darkorange',
                        linewidths=1,
                        label=obj_name,
                        zorder=5
                    )
            
            # Add legend (avoid duplicates)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), 
                     loc='upper right', fontsize=9, framealpha=0.9)
        
        # Save
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
    
    def generate_combination_poses(
        self,
        angle_step_deg: float = 30.0,
        angle_min_deg: float = -90.0,
        angle_max_deg: float = 90.0,
        except_last_joint: bool = True,
        use_3d_points: bool = False
    ):
        """
        Generate poses using all combinations of joint angles.
        
        ALL joints move simultaneously in combinations:
        Example with 2 joints, angles [0°, 30°, 60°]:
            [0°,  0°]
            [0°, 30°]
            [0°, 60°]
            [30°, 0°]
            [30°,30°]
            [30°,60°]
            [60°, 0°]
            [60°,30°]
            [60°,60°]
        
        WARNING: This generates angle_positions^num_joints poses!
        For 7 joints with 7 angles each: 7^7 = 823,543 poses
        
        Args:
            angle_step_deg: Step size in degrees (larger = fewer combinations)
            angle_min_deg: Minimum angle in degrees
            angle_max_deg: Maximum angle in degrees
            except_last_joint: If True, skip last joint (gripper)
            use_3d_points: If True, save 3D joint position plots instead of images
        """
        print("\n" + "="*60)
        print("COMBINATION POSE GENERATION")
        print("="*60)
        print(f"All {self.num_joints} joints will move simultaneously in combinations")
        print(f"Mode: {'3D Points' if use_3d_points else 'Camera Images'}")
        
        # Create subdirectory for 3D points if needed
        jsonl_path = None
        if use_3d_points:
            original_output_dir = self.output_dir
            self.output_dir = os.path.join(original_output_dir, self.robot_name, "3d_points")
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Create JSONL file path
            jsonl_path = os.path.join(self.output_dir, f"{self.robot_name}_poses_3d.jsonl")
            
            # Clear existing JSONL file if it exists
            if os.path.exists(jsonl_path):
                os.remove(jsonl_path)
            
            print(f"Saving 3D point plots to: {self.output_dir}")
            print(f"Saving 3D coordinates to: {jsonl_path}")
        
        angle_min = np.deg2rad(angle_min_deg)
        angle_max = np.deg2rad(angle_max_deg)
        angle_step = np.deg2rad(angle_step_deg)
        
        angles = np.arange(angle_min, angle_max + angle_step/2, angle_step)
        num_angles = len(angles)
        
        total_combinations = num_angles ** (self.num_joints-1 if except_last_joint else self.num_joints)
        print(f"\nAngle range: {angle_min_deg}° to {angle_max_deg}° (step: {angle_step_deg}°)")
        print(f"Angles per joint: {list(np.rad2deg(angles).astype(int))}")
        print(f"Positions per joint: {num_angles}")
        print(f"Total combinations: {total_combinations:,}")
        
        selected_combinations = list(product(range(num_angles), repeat=self.num_joints-1 if except_last_joint else self.num_joints))
        pose_count = 0
        start_time = time.time()
        
        # Get object positions once (they don't change)
        object_positions = None
        if use_3d_points:
            object_positions = self._get_object_positions()
            if object_positions:
                print(f"Found {len(object_positions)} environment objects: {list(object_positions.keys())}")
        
        for combo_idx, angle_indices in tqdm(enumerate(selected_combinations)):
            # Create joint position array
            joint_pos = self.initial_joint_pos.copy()
            
            for joint_idx, angle_idx in enumerate(angle_indices):
                joint_pos[joint_idx] = angles[angle_idx]
            
            try:
                self._set_joint_positions(joint_pos)
                
                # Generate filename with joint angles
                angles_str = "_".join([f"j{j}{int(np.rad2deg(angles[idx])):+04d}" 
                                      for j, idx in enumerate(angle_indices)])
                
                if use_3d_points:
                    # Capture 3D joint positions
                    positions_3d = self._get_3d_joint_positions()
                    filename = f"{self.robot_name}_pose_{combo_idx:06d}_{angles_str}.png"
                    filepath = self._save_3d_plot(positions_3d, filename, object_positions)
                    
                    # Save to JSONL
                    if jsonl_path:
                        # Create data entry
                        joint_angles_deg = [float(np.rad2deg(angles[idx])) for idx in angle_indices]
                        
                        data_entry = {
                            "pose_id": combo_idx,
                            "filename": filename,
                            "robot_name": self.robot_name,
                            "joint_angles_deg": joint_angles_deg,
                            "joint_angles_rad": [float(angles[idx]) for idx in angle_indices],
                            "joint_positions_3d": {
                                "root_body": positions_3d[0].tolist(),
                                "joints": [pos.tolist() for pos in positions_3d[1:]]
                            },
                            "num_joints": len(positions_3d) - 1,  # Exclude root body
                        }
                        
                        # Add object positions if available
                        if object_positions:
                            data_entry["object_positions"] = {
                                name: pos.tolist() for name, pos in object_positions.items()
                            }
                        
                        self._save_pose_data_to_jsonl(data_entry, jsonl_path)
                else:
                    # Capture camera image
                    image = self._capture_image()
                    filename = f"{self.robot_name}_pose_{combo_idx:06d}_{angles_str}.png"
                    filepath = self._save_image(image, filename)
                
                pose_count += 1
            
            except Exception as e:
                print(f"Error at combination {combo_idx}: {e}")
                continue
            
            # Return to initial pose occasionally to prevent drift
            if pose_count % 500 == 0:
                self._set_joint_positions(self.initial_joint_pos)
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"COMPLETE: Generated {pose_count:,} combination poses")
        print(f"Time taken: {total_time/60:.1f} minutes ({total_time/pose_count:.2f} sec/pose)")
        print(f"Saved images to: {self.output_dir}")
        if jsonl_path and os.path.exists(jsonl_path):
            file_size_mb = os.path.getsize(jsonl_path) / (1024**2)
            print(f"Saved 3D coordinates to: {jsonl_path} ({file_size_mb:.2f} MB)")
        print(f"{'='*60}\n")
    
    def close(self):
        """Close the environment."""
        self.env.close()


def main(
    robot: str = "Baxter",
    env: str = "Lift",
    angle_step: float = 90.0,
    angle_min: float = -90.0,
    angle_max: float = 90.0,
    output_dir: str = "data/poses",
    use_3d_points: bool = False,
    capture_image_width: int = 1024,
    capture_image_height: int = 1024,
    camera_fov: float = 60.0,
):
    """
    Generate preset robot poses by combining all joint angles.
    
    Args:
        robot: Robot name (IIWA, Panda, etc.)
        env: Environment name
        angle_step: Angle step size in degrees (larger = fewer combinations)
        angle_min: Minimum angle in degrees (default: -90)
        angle_max: Maximum angle in degrees (default: +90)
        output_dir: Directory to save pose images
        use_3d_points: If True, save 3D joint position plots instead of images (default: False)
        capture_image_width: Width of captured images
        capture_image_height: Height of captured images
        camera_fov: Camera field of view in degrees (larger = wider view, default: 60.0)
    
    Examples:
        # Camera images mode (default)
        python stack_preset.py --robot Panda --angle-step 90
        # Saves to: data/poses/*.png
        
        # 3D points mode - save joint positions as 3D plots
        python stack_preset.py --robot Panda --angle-step 90 --use-3d-points True
        # Saves to: data/poses/Panda/3d_points/*.png
        
        # Quick test with 3D points
        python stack_preset.py --robot IIWA --angle-step 60 --use-3d-points True
        # Result: 4^7 = 16,384 3D plots
    """
    
    print("="*60)
    print("PRESET POSE GENERATOR - COMBINATION MODE")
    print("="*60)
    print(f"Robot: {robot}")
    print(f"Environment: {env}")
    print(f"Angle step: {angle_step}°")
    print(f"Angle range: {angle_min}° to {angle_max}°")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {'3D Points' if use_3d_points else 'Camera Images'}")
    print(f"Camera FOV: {camera_fov}°")
    print("="*60)
    
    # Create generator
    generator = PresetPoseGenerator(
        robot_name=robot,
        env_name=env,
        output_dir=output_dir,
        capture_image_width=capture_image_width,
        capture_image_height=capture_image_height,
        camera_fov=camera_fov,
    )
    
    try:
        generator.generate_combination_poses(
            angle_step_deg=angle_step,
            angle_min_deg=angle_min,
            angle_max_deg=angle_max,
            use_3d_points=use_3d_points,
        )
    
    finally:
        generator.close()
        print("\nDone!")


if __name__ == "__main__":
    fire.Fire(main)

