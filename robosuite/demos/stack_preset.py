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
        
<<<<<<< Updated upstream
        # Print all joints in the simulation model for debugging
        print("\n" + "="*60)
        print("ALL JOINTS IN SIMULATION MODEL")
        print("="*60)
        print(f"Total joints in model: {self.env.sim.model.njnt}")
        print(f"Total qpos dimension: {self.env.sim.model.nq}")
        print("\nAll joint names (with qpos addresses):")
        all_joint_info = []
        for i in range(self.env.sim.model.njnt):
            try:
                joint_name = self.env.sim.model.joint_id2name(i)
                if joint_name is None:
                    joint_name = f"joint_{i}"
            except:
                joint_name = f"joint_{i}"
            
            joint_type = self.env.sim.model.jnt_type[i]
            try:
                qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint_name)
                if isinstance(qpos_addr, (list, tuple, np.ndarray)):
                    qpos_val = self.env.sim.data.qpos[qpos_addr[0]:qpos_addr[1]]
                    qpos_dim = qpos_addr[1] - qpos_addr[0]
                else:
                    qpos_val = self.env.sim.data.qpos[qpos_addr]
                    qpos_dim = 1
                
                # Check if this joint name contains certain keywords
                keywords = []
                if joint_name and "head" in joint_name.lower():
                    keywords.append("HEAD")
                if joint_name and "torso" in joint_name.lower():
                    keywords.append("TORSO")
                if joint_name and ("arm" in joint_name.lower() or "right" in joint_name.lower() or "left" in joint_name.lower()):
                    keywords.append("ARM")
                if joint_name and "leg" in joint_name.lower():
                    keywords.append("LEG")
                if joint_name and ("gripper" in joint_name.lower() or "hand" in joint_name.lower()):
                    keywords.append("GRIPPER")
                
                keyword_str = f" [{', '.join(keywords)}]" if keywords else ""
                print(f"  [{i:2d}] {joint_name:40s} type={joint_type}, qpos_addr={qpos_addr}, qpos_dim={qpos_dim}, value={qpos_val}{keyword_str}")
                all_joint_info.append({
                    'name': joint_name,
                    'type': joint_type,
                    'qpos_addr': qpos_addr,
                    'qpos_dim': qpos_dim,
                    'keywords': keywords
                })
            except Exception as e:
                print(f"  [{i:2d}] {joint_name or f'joint_{i}':40s} (type: {joint_type}, error getting qpos: {e})")
        
        # Summary
        head_joints = [j for j in all_joint_info if "HEAD" in j['keywords']]
        torso_joints = [j for j in all_joint_info if "TORSO" in j['keywords']]
        arm_joints = [j for j in all_joint_info if "ARM" in j['keywords']]
        
        print(f"\nSummary:")
        print(f"  Head joints found: {len(head_joints)}")
        print(f"  Torso joints found: {len(torso_joints)}")
        print(f"  Arm-related joints found: {len(arm_joints)}")
        
        # Get all joints from robot model
        print("\n" + "="*60)
        print("ROBOT MODEL JOINTS")
        print("="*60)
        try:
            robot_model_joints = list(self.robot.robot_model.joints)
            print(f"Robot model joints ({len(robot_model_joints)}): {robot_model_joints}")
            
            # Get arm joints
            try:
                arm_joints = list(self.robot.robot_model.arm_joints)
                print(f"Arm joints ({len(arm_joints)}): {arm_joints}")
            except:
                print("Could not get arm joints")
            
            # Get head joints
            try:
                head_joints = list(self.robot.robot_model.head_joints)
                print(f"Head joints ({len(head_joints)}): {head_joints}")
            except:
                print("Could not get head joints")
            
            # Get torso joints
            try:
                torso_joints = list(self.robot.robot_model.torso_joints)
                print(f"Torso joints ({len(torso_joints)}): {torso_joints}")
            except:
                print("Could not get torso joints")
        except Exception as e:
            print(f"Error getting robot model joints: {e}")
        
        # Get robot's controlled joint positions
        print("\n" + "="*60)
        print("ROBOT CONTROLLED JOINTS (robot._joint_positions)")
        print("="*60)
        robot_joint_pos = self.robot._joint_positions.copy()
        print(f"Number of controlled joints: {len(robot_joint_pos)}")
        print(f"Initial positions (degrees): {np.rad2deg(robot_joint_pos)}")
        
        # Get initial joint positions (robot controlled + head joints if applicable)
=======
        # For GR1ArmsOnly, also include head joints
        self.include_head_joints = (robot_name == "GR1ArmsOnly")
        self.head_joint_names = []
        self.head_joint_qpos_addrs = []
        
        if self.include_head_joints:
            # Find head joint names and qpos addresses
            head_joint_candidates = ["head_yaw", "head_roll", "head_pitch"]
            
            for joint_name in head_joint_candidates:
                try:
                    # Try with robot prefix
                    prefixed_name = f"robot0_{joint_name}"
                    qpos_addr = self.env.sim.model.get_joint_qpos_addr(prefixed_name)
                    self.head_joint_names.append(prefixed_name)
                    self.head_joint_qpos_addrs.append(qpos_addr)
                except:
                    try:
                        # Try without prefix
                        qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint_name)
                        self.head_joint_names.append(joint_name)
                        self.head_joint_qpos_addrs.append(qpos_addr)
                    except:
                        pass
            
            if self.head_joint_names:
                print(f"Including {len(self.head_joint_names)} head joints: {self.head_joint_names}")
            else:
                print("Warning: Could not find head joints, continuing without them")
                self.include_head_joints = False
        
        # Get initial joint positions
>>>>>>> Stashed changes
        self.initial_joint_pos = self._get_joint_positions()
        self.num_joints = len(self.initial_joint_pos)
        
        # Build joint names list
        try:
            robot_model_joints = list(self.robot.robot_model.joints)
            if len(robot_model_joints) >= len(robot_joint_pos):
                base_joint_names = robot_model_joints[:len(robot_joint_pos)]
            else:
                base_joint_names = robot_model_joints + [f"joint_{i}" for i in range(len(robot_model_joints), len(robot_joint_pos))]
            
            # Add head joint names
            self.joint_names = base_joint_names
        except:
            base_joint_names = [f"joint_{i}" for i in range(len(robot_joint_pos))]
            self.joint_names = base_joint_names
        
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
                        # Range format: "0-2" -> [0, 1, 2]
                        start, end = part.split("-", 1)
                        start = int(start.strip())
                        end = int(end.strip())
                        if start > end:
                            print(f"Warning: Invalid range '{part}' (start > end), skipping")
                            continue
                        self.fixed_joint_indices.extend(range(start, end + 1))
                    else:
                        # Single number
                        self.fixed_joint_indices.append(int(part))
                # Remove duplicates and sort, and filter to valid range
                self.fixed_joint_indices = sorted(list(set([idx for idx in self.fixed_joint_indices if 0 <= idx < self.num_joints])))
                if self.fixed_joint_indices:
                    print(f"\nFixed joints from FIXED_JOINT_INDICES: {self.fixed_joint_indices}")
            except ValueError as e:
                print(f"Warning: Could not parse fixed_joint_indices '{fixed_indices_str}': {e}")
                self.fixed_joint_indices = []
        
        # Active joints: all joints except last (gripper) and fixed joints
        base_active_joint_indices = list(range(self.num_joints - 1))  # Exclude last joint (gripper)
        self.active_joint_indices = [idx for idx in base_active_joint_indices if idx not in self.fixed_joint_indices]
        
        print("\n" + "="*60)
        print("JOINT SPACE SUMMARY")
        print("="*60)
        print(f"Total joints in pose: {self.num_joints}")
        print("Joint names and initial positions:")
        for i, (name, pos) in enumerate(zip(self.joint_names, np.rad2deg(self.initial_joint_pos))):
            if i in self.fixed_joint_indices:
                marker = " [FIXED]"
            elif i in self.active_joint_indices:
                marker = " [ACTIVE]"
            else:
                marker = ""
            print(f"  [{i}] {name}: {pos:+.2f}°{marker}")
        print(f"\nActive joints: {len(self.active_joint_indices)} joints")
        if self.fixed_joint_indices:
            print(f"Fixed joints: {len(self.fixed_joint_indices)} joints (will keep initial values)")
        print("="*60 + "\n")
    
    def _get_joint_positions(self):
        """Get current joint positions, including head joints if applicable."""
        joint_pos = self.robot._joint_positions.copy()
<<<<<<< Updated upstream
=======
        
        # Add head joints if needed
        if self.include_head_joints and self.head_joint_qpos_addrs:
            head_positions = []
            for qpos_addr in self.head_joint_qpos_addrs:
                if isinstance(qpos_addr, (list, tuple, np.ndarray)):
                    # Joint has multiple DOF (e.g., free joint)
                    head_positions.append(self.env.sim.data.qpos[qpos_addr[0]])
                else:
                    # Single DOF joint
                    head_positions.append(self.env.sim.data.qpos[qpos_addr])
            joint_pos = np.concatenate([joint_pos, np.array(head_positions)])
        
>>>>>>> Stashed changes
        return joint_pos
    
    def _set_joint_positions(self, joint_positions):
        """Set joint positions and update simulation, including head joints if applicable."""
<<<<<<< Updated upstream
        robot_joint_pos = joint_positions
        # Set robot joint positions
        self.robot.set_robot_joint_positions(robot_joint_pos)
        
=======
        # Split joint positions into robot joints and head joints
        if self.include_head_joints and self.head_joint_qpos_addrs:
            num_head_joints = len(self.head_joint_qpos_addrs)
            robot_joint_pos = joint_positions[:-num_head_joints] if num_head_joints > 0 else joint_positions
            head_joint_pos = joint_positions[-num_head_joints:] if num_head_joints > 0 else []
        else:
            robot_joint_pos = joint_positions
            head_joint_pos = []
        
        # Set robot joint positions
        self.robot.set_robot_joint_positions(robot_joint_pos)
        
        # Set head joint positions directly in MuJoCo
        if head_joint_pos:
            for i, qpos_addr in enumerate(self.head_joint_qpos_addrs):
                if isinstance(qpos_addr, (list, tuple, np.ndarray)):
                    self.env.sim.data.qpos[qpos_addr[0]] = head_joint_pos[i]
                else:
                    self.env.sim.data.qpos[qpos_addr] = head_joint_pos[i]
        
>>>>>>> Stashed changes
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
        
        Active joints (arm + head joints) move simultaneously in combinations:
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
        
        WARNING: This generates angle_positions^num_active_joints poses!
        For 7 active joints with 7 angles each: 7^7 = 823,543 poses
        
        Args:
            angle_step_deg: Step size in degrees (larger = fewer combinations)
            angle_min_deg: Minimum angle in degrees
            angle_max_deg: Maximum angle in degrees
            except_last_joint: Deprecated - now automatically uses arm + head joints
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
        
        # Prepare angle arrays for each active joint
        # Check if robot has specified joint angles
        specified_angles = SPECIFIED_JOINT_ANGLES.get(self.robot_name, {})
        
        # Default angle range
        angle_min = np.deg2rad(angle_min_deg)
        angle_max = np.deg2rad(angle_max_deg)
        angle_step = np.deg2rad(angle_step_deg)
        default_angles = np.arange(angle_min, angle_max + angle_step/2, angle_step)
        
        # Create angle arrays for each active joint
        joint_angle_arrays = []
        joint_angle_info = []
        for active_joint_idx in self.active_joint_indices:
            joint_name = self.joint_names[active_joint_idx]
            
            # Check if this joint has specified angles
            if joint_name in specified_angles:
                # Use specified angles (convert to radians)
                angles = np.deg2rad(np.array(specified_angles[joint_name]))
                joint_angle_arrays.append(angles)
                joint_angle_info.append(f"{joint_name}: {specified_angles[joint_name]} (specified)")
            else:
                # Use default angle range
                angles = default_angles
                joint_angle_arrays.append(angles)
                joint_angle_info.append(f"{joint_name}: default range")
        
<<<<<<< Updated upstream
        # Calculate number of angle positions per joint
        num_angles_per_joint = [len(angles) for angles in joint_angle_arrays]
        
        # Calculate total combinations
        total_combinations = 1
        for num in num_angles_per_joint:
            total_combinations *= num
        
        print(f"\n{'='*60}")
        print("GENERATION PARAMETERS")
        print(f"{'='*60}")
        print(f"Default angle range: {angle_min_deg}° to {angle_max_deg}° (step: {angle_step_deg}°)")
        print(f"Default angle values: {list(np.rad2deg(default_angles).astype(int))}")
        print(f"\nActive joints ({len(self.active_joint_indices)}) and their angle ranges:")
        for i, (active_idx, info) in enumerate(zip(self.active_joint_indices, joint_angle_info)):
            print(f"  [{i}] [{active_idx}] {info}")
            print(f"      Angle values: {np.rad2deg(joint_angle_arrays[i]).astype(int).tolist()}")
            print(f"      Number of positions: {num_angles_per_joint[i]}")
        if self.fixed_joint_indices:
            print(f"\nFixed joints (excluded from combinations): {self.fixed_joint_indices}")
            for idx in self.fixed_joint_indices:
                print(f"  [{idx}] {self.joint_names[idx]}: {np.rad2deg(self.initial_joint_pos[idx]):+.2f}° (fixed)")
        print(f"\nTotal data points to generate: {total_combinations:,}")
        print(f"{'='*60}\n")
        
        # Generate combinations using product of different angle ranges
        selected_combinations = list(product(*[range(num) for num in num_angles_per_joint]))
=======
        # Calculate number of joints to include in combinations
        # If except_last_joint is True, exclude only the last arm joint (not head joints)
        num_head_joints = len(self.head_joint_qpos_addrs) if self.include_head_joints else 0
        num_arm_joints = self.num_joints - num_head_joints
        
        if except_last_joint:
            # Exclude last arm joint, but include all head joints
            num_joints_for_combinations = num_arm_joints - 1 + num_head_joints
        else:
            num_joints_for_combinations = self.num_joints
        
        total_combinations = num_angles ** num_joints_for_combinations
        print(f"\nAngle range: {angle_min_deg}° to {angle_max_deg}° (step: {angle_step_deg}°)")
        print(f"Angles per joint: {list(np.rad2deg(angles).astype(int))}")
        print(f"Positions per joint: {num_angles}")
        print(f"Joints in combinations: {num_joints_for_combinations} (arm: {num_arm_joints - (1 if except_last_joint else 0)}, head: {num_head_joints})")
        print(f"Total combinations: {total_combinations:,}")
        
        selected_combinations = list(product(range(num_angles), repeat=num_joints_for_combinations))
>>>>>>> Stashed changes
        pose_count = 0
        start_time = time.time()
        
        # Get object positions once (they don't change)
        object_positions = None
        if use_3d_points:
            object_positions = self._get_object_positions()
            if object_positions:
                print(f"Found {len(object_positions)} environment objects: {list(object_positions.keys())}")
        
        for combo_idx, angle_indices in tqdm(enumerate(selected_combinations)):
            # Create joint position array (fixed joints will keep initial values)
            joint_pos = self.initial_joint_pos.copy()
            
<<<<<<< Updated upstream
            # Map angle_indices to active joint positions only
            angle_idx_iter = iter(angle_indices)
=======
            # Map angle_indices to joint positions
            # If except_last_joint is True, skip the last arm joint
            angle_idx_iter = iter(angle_indices)
            
            # Set arm joint positions (excluding last if except_last_joint is True)
            num_arm_joints_to_set = num_arm_joints - (1 if except_last_joint else 0)
            for i in range(num_arm_joints_to_set):
                angle_idx = next(angle_idx_iter)
                joint_pos[i] = angles[angle_idx]
            
            # Set head joint positions (always included if head joints exist)
            if self.include_head_joints:
                for i in range(num_head_joints):
                    angle_idx = next(angle_idx_iter)
                    joint_pos[num_arm_joints + i] = angles[angle_idx]
>>>>>>> Stashed changes
            
            # Set positions only for active joints using their respective angle arrays
            angle_values = []
            for i, active_joint_idx in enumerate(self.active_joint_indices):
                angle_idx = angle_indices[i]
                angle_value = joint_angle_arrays[i][angle_idx]
                joint_pos[active_joint_idx] = angle_value
                angle_values.append(angle_value)
            
            # Fixed joints remain at initial positions (already set by copy())
            self._set_joint_positions(joint_pos)
            
            # Generate filename with joint angles (only for active joints)
            angles_str = "_".join([f"j{self.active_joint_indices[j]}{int(np.rad2deg(angle_values[j])):+04d}" 
                                    for j in range(len(angle_indices))])
            
            if use_3d_points:
                # Capture 3D joint positions
                positions_3d = self._get_3d_joint_positions()
                filename = f"{self.robot_name}_pose_{combo_idx:06d}_{angles_str}.png"
                filepath = self._save_3d_plot(positions_3d, filename, object_positions)
                
                # Save to JSONL
                if jsonl_path:
                    # Create data entry (only for active joints)
                    joint_angles_deg = [float(np.rad2deg(angle_values[j])) for j in range(len(angle_indices))]
                    
                    data_entry = {
                        "pose_id": combo_idx,
                        "filename": filename,
                        "robot_name": self.robot_name,
                        "active_joint_indices": self.active_joint_indices,
                        "fixed_joint_indices": self.fixed_joint_indices,
                        "joint_angles_deg": joint_angles_deg,
                            "joint_angles_rad": [float(angle_values[j]) for j in range(len(angle_indices))],
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
    robot: str = "GR1",
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

