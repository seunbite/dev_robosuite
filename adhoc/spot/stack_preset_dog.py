"""
Generate quadruped (dog) robot poses - 3 simple poses.

Simple Strategy (3 poses):
- Standing: Robot standing straight
- Bent: Robot slightly crouched
- Sitting: Robot sitting on the ground

All 4 legs move symmetrically (FL = FR = HL = HR).

Usage:
    # Generate 3 poses with both physics and kinematics versions
    python adhoc/spot/stack_preset_dog.py --robot SpotWithArm
"""

import fire
import os
import sys
import time
import numpy as np
import json
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import robosuite as suite


# Leg IDs
LEGS = ['FL', 'FR', 'HL', 'HR']

POSE_CONFIGS = {
    'standing': {
        'hip': 0,        # 몸체 평행 유지
        'shoulder': 20,  # 앞으로 살짝 뻗음 (Flexion)
        'knee': -40,     # 뒤로 강하게 굽힘 (역관절 구조 생성) -> 이래야 서 있을 수 있습니다.
    },
    'bent': {
        'hip': 0,
        'shoulder': 45,
        'knee': -90,     # 더 깊게 굽힘
    },
    'sitting': {
        'front' : {
            'hip': 0,
            'shoulder': 30,
            'knee': -40,    # 뒤보단 살짝 앞을 짚어서 안정적으로
        },
        'back' : {
            'hip': 0,
            'shoulder': 50,
            'knee': -60,    # 완전히 주저앉음 (다리를 접는 방향)
        },
    }
}


class DogPoseGenerator:
    """Generate 3 simple quadruped dog poses: standing, bent, sitting."""
    
    def __init__(
        self,
        robot_name: str = "SpotWithArm",
        env_name: str = "EmptySpace",
        output_dir: str = "data/poses/quadruped",
        capture_image_width: int = 512,
        capture_image_height: int = 512,
        camera_fov: float = 80.0,
        kp: int = 300,
        kv: int = 10,
        kd: int = 200,
        front_kp: int = None,
        front_shoulder: float = None,
        front_knee: float = None,
    ):
        print(f"kp: {kp}, kv: {kv}, kd: {kd}, front_kp: {front_kp}")
        """
        Initialize the dog pose generator.
        
        Args:
            robot_name: Name of the quadruped robot (SpotWithArm only)
            env_name: Environment name
            output_dir: Output directory for pose images and data
            capture_image_width: Image width
            capture_image_height: Image height
            camera_fov: Camera field of view (larger for quadrupeds)
            kp: Position gain for all legs (default/back legs)
            kv: Velocity gain for leg controller
            kd: Damping gain for leg controller
            front_kp: Position gain specifically for front legs
            front_shoulder: Override front shoulder angle for sitting pose (degrees)
            front_knee: Override front knee angle for sitting pose (degrees)
        """
        if robot_name not in ["SpotWithArm"]:
            raise ValueError(f"Robot {robot_name} not supported. Only SpotWithArm is supported (Spot without arm is not registered in robosuite).")
        
        self.robot_name = robot_name
        self.env_name = env_name
        self.output_dir = os.path.join(output_dir, robot_name)
        self.capture_image_width = capture_image_width
        self.capture_image_height = capture_image_height
        self.camera_fov = camera_fov
        self.kp = kp
        self.kv = kv
        self.kd = kd
        self.front_kp = front_kp if front_kp is not None else kp
        
        # Override sitting pose if parameters provided
        if front_shoulder is not None:
            POSE_CONFIGS['sitting']['front']['shoulder'] = front_shoulder
        if front_knee is not None:
            POSE_CONFIGS['sitting']['front']['knee'] = front_knee
            
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
        # Create environment (this also initializes joint info)
        self._create_environment()
        
        # Get initial joint positions
        self.initial_joint_pos = self.robot._joint_positions.copy()
        
        # Build leg-to-joint mapping
        self.leg_joint_map = self._build_leg_joint_map()
        
        print(f"Dog pose generator initialized for {robot_name}")
    
    def _create_environment(self):
        """Create robosuite environment."""
        # Load controller config with gravity compensation
        controller_config = suite.load_composite_controller_config(robot=self.robot_name)
        controller_config["gravity_compensation"] = True
        
        self.env = suite.make(
            env_name=self.env_name,
            robots=self.robot_name,
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,  # Important: don't terminate episode
            control_freq=100,  # Increased from 20 to 100Hz for better control stability
        )
        
        # Manually set the simulation timestep for even higher physics frequency
        # 0.001s = 1000Hz physics simulation
        self.env.sim.model.opt.timestep = 0.001
        print(f"Physics simulation frequency: {1.0/self.env.sim.model.opt.timestep}Hz")
        
        self.env.reset()
        self.robot = self.env.robots[0]
        
        # Detect leg joints and update num_base_joints
        self.has_leg_joints = hasattr(self.robot.robot_model, 'legs_joints') and len(self.robot.robot_model.legs_joints) > 0
        
        if self.has_leg_joints:
            # Get leg joint names and their qpos addresses
            self.leg_joint_names = self.robot.robot_model.legs_joints
            self.num_base_joints = len(self.leg_joint_names)
            
            # Get initial leg joint positions from sim
            self.leg_joint_qpos_addrs = [
                self.env.sim.model.get_joint_qpos_addr(joint_name) 
                for joint_name in self.leg_joint_names
            ]
            self.initial_leg_joint_pos = np.array([
                self.env.sim.data.qpos[addr] 
                for addr in self.leg_joint_qpos_addrs
            ])
            
            print(f"Leg joints detected: {self.num_base_joints}")
        else:
            self.leg_joint_names = []
            self.leg_joint_qpos_addrs = []
            self.initial_leg_joint_pos = np.array([])
            self.num_base_joints = 0

        # Adjust camera FOV
        try:
            cam_id = self.env.sim.model.camera_name2id("frontview")
            self.env.sim.model.cam_fovy[cam_id] = self.camera_fov
            print(f"Frontview camera FOV set to {self.camera_fov}°")
        except Exception as e:
            print(f"Warning: Could not set frontview camera FOV: {e}")
        
        # Adjust sideview camera FOV for wider view
        try:
            cam_id = self.env.sim.model.camera_name2id("sideview")
            self.env.sim.model.cam_fovy[cam_id] = 100.0  # Wider FOV for sideview
            print(f"Sideview camera FOV set to 100°")
        except Exception as e:
            print(f"Warning: Could not set sideview camera FOV: {e}")
        
        # Get joint names from robot model
        if hasattr(self.robot.robot_model, 'joints'):
            self.joint_names = list(self.robot.robot_model.joints)
        else:
            # Fallback: generate generic names
            num_joints = len(self.robot._joint_positions)
            self.joint_names = [f"joint_{i}" for i in range(num_joints)]
        
        # Pre-build mapping for kp_vector calculation (we need it here)
        self.leg_joint_map = self._build_leg_joint_map()

        # Increase PD gains for legs to make them stiffer and prevent collapsing
        if hasattr(self.robot, 'part_controllers') and 'legs' in self.robot.part_controllers:
            leg_controller = self.robot.part_controllers['legs']
            if hasattr(leg_controller, 'kp'):
                # Set up kp as a vector to allow different gains for front and back legs
                kp_vector = np.ones(self.num_base_joints) * self.kp
                
                # Identify front leg joint indices
                front_indices = []
                for leg_id in ['FL', 'FR']:
                    if leg_id in self.leg_joint_map:
                        for joint_idx in self.leg_joint_map[leg_id].values():
                            front_indices.append(joint_idx)
                
                # Apply front_kp to front leg indices
                for idx in front_indices:
                    if idx < len(kp_vector):
                        kp_vector[idx] = self.front_kp
                
                original_kp = leg_controller.kp
                leg_controller.kp = kp_vector
                print(f"Set leg kp vector (front: {self.front_kp}, back: {self.kp})")
            
            if hasattr(leg_controller, 'kv'):
                # Set velocity gain
                original_kv = leg_controller.kv
                leg_controller.kv = self.kv
                print(f"Set leg kv from {original_kv} to {leg_controller.kv}")
            if hasattr(leg_controller, 'kd'):
                # Increase damping proportionally
                original_kd = leg_controller.kd
                leg_controller.kd = self.kd  # Increased damping
                print(f"Increased leg kd from {original_kd} to {leg_controller.kd}")
        
        print(f"Environment created: {self.env_name}")
        print(f"Robot: {self.robot_name}")
        print(f"Total joints: {len(self.joint_names)}")
    
    def _build_leg_joint_map(self):
        """Build mapping from leg IDs to joint indices (in leg joint array)."""
        leg_joint_map = {}
        
        if not self.has_leg_joints:
            print("\nNo leg joints to map")
            return leg_joint_map
        
        joint_names = self.leg_joint_names
        
        print("\nBuilding leg-to-joint mapping:")
        print(f"Available leg joints ({len(joint_names)}): {joint_names[:6]}...")
        
        # Leg prefixes and joint keywords
        leg_prefixes = {
            'FL': ['fl', 'front_left', 'lf', 'fl0', 'fl1', 'fl2'],
            'FR': ['fr', 'front_right', 'rf', 'fr0', 'fr1', 'fr2'],
            'HL': ['hl', 'hind_left', 'rear_left', 'lh', 'hl0', 'hl1', 'hl2'],
            'HR': ['hr', 'hind_right', 'rear_right', 'rh', 'hr0', 'hr1', 'hr2'],
        }
        
        joint_type_keywords = {
            'hip': ['hip', 'abduction', 'abd', 'roll', 'hx', '0'],
            'shoulder': ['shoulder', 'thigh', 'flexion', 'pitch', 'hy', '1'],
            'knee': ['knee', 'calf', 'shank', 'kn', '2'],
        }
        
        for leg_id in LEGS:
            leg_joint_map[leg_id] = {}
            
            for joint_type, keywords in joint_type_keywords.items():
                best_idx = None
                best_score = 0
                
                for idx, joint_name in enumerate(joint_names):
                    joint_name_lower = joint_name.lower()
                    
                    # Score leg match
                    leg_score = 0
                    for prefix in leg_prefixes[leg_id]:
                        if joint_name_lower.startswith(prefix):
                            leg_score = 2
                            break
                        elif prefix in joint_name_lower:
                            leg_score = 1
                    
                    if leg_score == 0:
                        continue
                    
                    # Score joint type match
                    type_score = 0
                    for keyword in keywords:
                        if keyword in joint_name_lower:
                            type_score = 1
                            break
                    
                    total_score = leg_score + type_score
                    if total_score > best_score:
                        best_score = total_score
                        best_idx = idx
                
                if best_idx is not None:
                    leg_joint_map[leg_id][joint_type] = best_idx
                    print(f"  {leg_id}_{joint_type} -> joint[{best_idx}]: {joint_names[best_idx]}")
        
        return leg_joint_map
    
    def _set_joint_positions(self, leg_joint_positions, use_physics=True, num_steps=100, capture_gif=False, pose_name=""):
        """
        Set leg joint positions.
        
        For SpotWithArm:
        - Leg joints: Set directly to sim.data.qpos (not controlled by robot controller)
        - Arm joints: Keep at initial position (not used for body motions)
        
        Args:
            leg_joint_positions: Array of leg joint positions in radians
            use_physics: If True, use physics simulation; if False, use forward kinematics only
            num_steps: Number of physics simulation steps (only used if use_physics=True)
            capture_gif: If True, capture frames and save as GIF
            pose_name: Name of the pose for GIF filename
        
        Returns:
            None
        """
        # Set leg joint positions directly to sim.data.qpos
        if self.has_leg_joints:
            for joint_idx, qpos_addr in enumerate(self.leg_joint_qpos_addrs):
                if joint_idx < len(leg_joint_positions):
                    self.env.sim.data.qpos[qpos_addr] = leg_joint_positions[joint_idx]
        
        # Keep arm at initial position
        self.robot.set_robot_joint_positions(self.initial_joint_pos)
        
        # IMPORTANT: Lift robot slightly off the ground to prevent initial collision
        # Let it drop and settle naturally - much more stable than starting on ground
        if use_physics:
            # For mobile robots with floating base, the first 7 qpos values are: 
            # [x, y, z, qw, qx, qy, qz] (position + quaternion)
            # Lift Z position (index 2) by 0.4m
            original_z = self.env.sim.data.qpos[2]
            self.env.sim.data.qpos[2] = original_z - 0.1 # Down by 20cm
            self.env.sim.data.qvel[:] = 0 # Zero out velocity after teleport
            self.env.sim.forward()  # Update kinematics after position change
            print(f"  Lifted robot from z={original_z:.3f} to z={self.env.sim.data.qpos[2]:.3f}")
        else:
            self.env.sim.data.qvel[:] = 0
            self.env.sim.forward()
        
        if use_physics:
            # Lists to store frames for GIF
            frames_front = []
            frames_side = []
            
            # Option 1: Physics simulation - let robot settle with gravity
            # IMPORTANT: We need to send position-holding actions through the controller
            # to maintain the joint positions, not zero actions!
            
            # Zero out velocities once more before starting steps to prevent any teleport-induced velocity
            self.env.sim.data.qvel[:] = 0
            
            # The leg controller expects normalized joint position commands
            # Convert TARGET leg joint positions to normalized actions and keep them fixed
            leg_action = np.zeros(self.num_base_joints)
            for i, target_pos in enumerate(leg_joint_positions):
                if i >= self.num_base_joints:
                    break
                joint_name = self.leg_joint_names[i]
                joint_id = self.env.sim.model.joint_name2id(joint_name)
                joint_range = self.env.sim.model.jnt_range[joint_id]
                
                # Normalize target position to [-1, 1]
                if joint_range[1] > joint_range[0]:
                    normalized_pos = 2 * (target_pos - joint_range[0]) / (joint_range[1] - joint_range[0]) - 1
                    normalized_pos = np.clip(normalized_pos, -1, 1)
                else:
                    normalized_pos = 0
                leg_action[i] = normalized_pos
            
            for step_idx in range(num_steps):
                # Build full action based on robot type
                if self.robot_name == "Spot":
                    # Spot: only legs (12 dimensions)
                    full_action = leg_action
                else:
                    # SpotWithArm: arm (6) + gripper (1) + legs (12) = 19 dimensions
                    arm_action = np.zeros(6)  # Keep arm at current position
                    gripper_action = np.zeros(1)  # Keep gripper at current state
                    full_action = np.concatenate([arm_action, gripper_action, leg_action])
                self.env.step(full_action)
                
                # Capture frames for GIF (every 5 steps to reduce file size)
                if capture_gif and step_idx % 2 == 0:
                    frame_front = self._capture_image("frontview")
                    frame_side = self._capture_image("sideview")
                    frames_front.append(Image.fromarray(frame_front))
                    frames_side.append(Image.fromarray(frame_side))
            
            # After settling, zero out velocities to stop any residual motion
            self.env.sim.data.qvel[:] = 0
            self.env.sim.forward()
            
            # Save GIF if frames were captured
            if capture_gif and len(frames_front) > 0:
                gif_dir = os.path.join(self.output_dir, 'step_gifs')
                os.makedirs(gif_dir, exist_ok=True)
                
                gif_path_front = os.path.join(gif_dir, f"{pose_name}_frontview.gif")
                gif_path_side = os.path.join(gif_dir, f"{pose_name}_sideview.gif")
                
                # Save GIFs with 50ms per frame (20 fps)
                frames_front[0].save(
                    gif_path_front,
                    save_all=True,
                    append_images=frames_front[1:],
                    duration=50,
                    loop=0
                )
                frames_side[0].save(
                    gif_path_side,
                    save_all=True,
                    append_images=frames_side[1:],
                    duration=50,
                    loop=0
                )
                print(f"  Saved GIFs: {gif_path_front}, {gif_path_side}")
        else:
            # Option 2: Kinematics only (no physics, faster, no falling)
            # Just update kinematics without physics simulation
            for _ in range(20):
                self.env.sim.data.qvel[:] = 0  # Zero out all velocities
                self.env.sim.forward()  # Forward kinematics only
    
    def _capture_image(self, camera_name="frontview"):
        """Capture camera view.
        
        Args:
            camera_name: Name of the camera to render from (e.g., 'frontview', 'sideview')
        """
        obs = self.env.sim.render(
            camera_name=camera_name,
            width=self.capture_image_width,
            height=self.capture_image_height,
            depth=False
        )
        return obs[::-1]
    
    def _pose_params_to_joint_positions(self, pose_name: str) -> np.ndarray:
        """
        Convert pose name to leg joint positions.
        Handles both symmetric and separated (front/back) leg configurations.
        
        Args:
            pose_name: 'standing', 'bent', or 'sitting'
        
        Returns:
            Leg joint position array in radians (12 joints for SpotWithArm)
        """
        # Start with initial leg joint positions
        leg_joint_pos = self.initial_leg_joint_pos.copy()
        
        # Get angles for this pose
        pose_config = POSE_CONFIGS[pose_name]
        
        for leg_id in LEGS:
            if leg_id not in self.leg_joint_map:
                continue
            
            # Check if this pose has separated front/back configurations
            if 'front' in pose_config and ('back' in pose_config or 'hind' in pose_config):
                is_front = leg_id.startswith('F')
                back_key = 'back' if 'back' in pose_config else 'hind'
                target_angles = pose_config['front'] if is_front else pose_config[back_key]
            else:
                target_angles = pose_config
            
            for joint_type, joint_idx in self.leg_joint_map[leg_id].items():
                if joint_type in target_angles:
                    angle_deg = target_angles[joint_type]
                    
                    # Set leg joint position (joint_idx is index in leg_joint_pos array)
                    if joint_idx < len(leg_joint_pos):
                        leg_joint_pos[joint_idx] = np.deg2rad(angle_deg)
        
        return leg_joint_pos
    
    def generate_poses(self, physics_steps=100, target_pose=None):
        """Generate poses. If target_pose is specified, only generate that one."""
        print("\n" + "="*60)
        print(f"DOG POSE GENERATION - {'ALL' if target_pose is None else target_pose.upper()}")
        print("="*60)
        
        # Define poses
        if target_pose is not None:
            if target_pose not in POSE_CONFIGS:
                raise ValueError(f"Pose {target_pose} not found in POSE_CONFIGS")
            pose_names = [target_pose]
        else:
            pose_names = ['standing', 'bent', 'sitting']
        # pose_names = ['standing']
        
        print(f"Poses: {pose_names}")
        print(f"All 4 legs symmetric (FL = FR = HL = HR)")
        print(f"\nTotal poses: {len(pose_names)}")
        print("="*60 + "\n")
        
        # Create JSONL file
        jsonl_path = os.path.join(self.output_dir, f"{self.robot_name}_dog_poses.jsonl")
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)
        
        pose_count = 0
        start_time = time.time()
        
        # Create output directories for both step and forward versions
        os.makedirs(os.path.join(self.output_dir, 'step', 'frontview'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'step', 'sideview'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'forward', 'frontview'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'forward', 'sideview'), exist_ok=True)
        
        for pose_name in tqdm(pose_names):
            # Generate leg joint positions
            self._create_environment()
            leg_joint_pos = self._pose_params_to_joint_positions(pose_name)
            
            # ===== STEP VERSION (with physics) =====
            # Capture GIF to see how the robot settles/collapses
            self._set_joint_positions(leg_joint_pos, use_physics=True, num_steps=physics_steps, 
                                     capture_gif=True, pose_name=f"{self.robot_name}_{pose_name}")
            
            # Capture and save step frontview
            image_step_front = self._capture_image("frontview")
            filename_step_front = f"{self.robot_name}_pose_{pose_count:04d}_{pose_name}.png"
            filepath_step_front = os.path.join(self.output_dir, 'step', 'frontview', filename_step_front)
            Image.fromarray(image_step_front).save(filepath_step_front)
            
            # Capture and save step sideview
            image_step_side = self._capture_image("sideview")
            filename_step_side = f"{self.robot_name}_pose_{pose_count:04d}_{pose_name}_side.png"
            filepath_step_side = os.path.join(self.output_dir, 'step', 'sideview', filename_step_side)
            Image.fromarray(image_step_side).save(filepath_step_side)
            
            # ===== FORWARD VERSION (kinematics only) =====
            # Reset environment and set positions again with forward kinematics
            self._create_environment()
            leg_joint_pos = self._pose_params_to_joint_positions(pose_name)
            self._set_joint_positions(leg_joint_pos, use_physics=False, capture_gif=False)
            
            # Capture and save forward frontview
            image_forward_front = self._capture_image("frontview")
            filename_forward_front = f"{self.robot_name}_pose_{pose_count:04d}_{pose_name}.png"
            filepath_forward_front = os.path.join(self.output_dir, 'forward', 'frontview', filename_forward_front)
            Image.fromarray(image_forward_front).save(filepath_forward_front)
            
            # Capture and save forward sideview
            image_forward_side = self._capture_image("sideview")
            filename_forward_side = f"{self.robot_name}_pose_{pose_count:04d}_{pose_name}_side.png"
            filepath_forward_side = os.path.join(self.output_dir, 'forward', 'sideview', filename_forward_side)
            Image.fromarray(image_forward_side).save(filepath_forward_side)
            
            # Save to JSONL
            leg_joint_angles_deg = np.rad2deg(leg_joint_pos).tolist()
            data_entry = {
                "pose_id": pose_count,
                "pose_name": pose_name,
                "step_frontview": filename_step_front,
                "step_sideview": filename_step_side,
                "forward_frontview": filename_forward_front,
                "forward_sideview": filename_forward_side,
                "robot_name": self.robot_name,
                "leg_joint_angles_deg": leg_joint_angles_deg,
                "leg_joint_angles_rad": leg_joint_pos.tolist(),
                "num_leg_joints": len(leg_joint_pos),
            }
            
            with open(jsonl_path, 'a') as f:
                f.write(json.dumps(data_entry) + '\n')
            
            pose_count += 1
        
        # Create tiled summaries for easy viewing
        self._create_tiled_summary('step')
        self._create_tiled_summary('forward')
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"COMPLETE: Generated {pose_count} dog poses")
        print(f"Time taken: {total_time:.1f} seconds ({total_time/pose_count:.2f} sec/pose)")
        print(f"Saved images to: {self.output_dir}")
        print(f"Saved data to: {jsonl_path}")
        print(f"{'='*60}\n")
    
    def _create_tiled_summary(self, category):
        """Create a tiled image of all generated poses for a category."""
        from PIL import ImageDraw, ImageFont
        
        front_dir = os.path.join(self.output_dir, category, 'frontview')
        side_dir = os.path.join(self.output_dir, category, 'sideview')
        
        front_files = sorted([f for f in os.listdir(front_dir) if f.endswith('.png')])
        side_files = sorted([f for f in os.listdir(side_dir) if f.endswith('.png')])
        
        if not front_files:
            return
            
        n = len(front_files)
        img_w, img_h = Image.open(os.path.join(front_dir, front_files[0])).size
        
        # Create a grid: 2 rows (front, side) x N columns (poses)
        combined = Image.new('RGB', (n * img_w, 2 * img_h))
        draw = ImageDraw.Draw(combined)
        
        for i in range(n):
            # Paste front view
            f_img = Image.open(os.path.join(front_dir, front_files[i]))
            combined.paste(f_img, (i * img_w, 0))
            
            # Paste side view
            s_img = Image.open(os.path.join(side_dir, side_files[i]))
            combined.paste(s_img, (i * img_w, img_h))
            
            # Label pose name
            pose_name = front_files[i].split('_')[-1].replace('.png', '')
            draw.text((i * img_w + 10, 10), f"{category.upper()}: {pose_name}", fill=(255, 0, 0))
            
        save_path = os.path.join(self.output_dir, f"summary_tiled_{category}.png")
        combined.save(save_path)
        print(f"  Saved tiled summary: {save_path}")

    def close(self):
        """Close environment."""
        self.env.close()


def main(
    robot: str = "SpotWithArm",
    env: str = "EmptySpace",
    output_dir: str = "data/poses/quadruped",
    target_pose: str = "sitting",
    capture_image_width: int = 512,
    capture_image_height: int = 512,
    camera_fov: float = 80.0,
    physics_steps: int = 200,
    kp: int = 500,
    kv: int = 10,
    kd: int = 500,
    front_kp: int = 700,
    front_shoulder: float = 10,
    front_knee: float = -70,
):
    """
    Generate simple quadruped dog poses.
    
    Args:
        robot: Robot name (SpotWithArm only)
        env: Environment name
        output_dir: Output directory
        capture_image_width: Image width
        capture_image_height: Image height
        camera_fov: Camera FOV (degrees, larger = wider view)
        physics_steps: Number of physics simulation steps
        kp: Position gain for leg controller (default/back legs)
        kv: Velocity gain for leg controller
        kd: Damping gain for leg controller
        front_kp: Position gain specifically for front legs
        front_shoulder: Override front shoulder angle for sitting pose
        front_knee: Override front knee angle for sitting pose
        target_pose: If specified, only generate this pose (e.g., 'sitting')
    """
    print("="*60)
    print(f"DOG POSE GENERATOR - {target_pose if target_pose else 'ALL'}")
    print("="*60)
    print(f"Robot: {robot}")
    print(f"Environment: {env}")
    print(f"Output: {output_dir}/{robot}")
    print("="*60)
    
    generator = DogPoseGenerator(
        robot_name=robot,
        env_name=env,
        output_dir=output_dir,
        capture_image_width=capture_image_width,
        capture_image_height=capture_image_height,
        camera_fov=camera_fov,
        kp=kp,
        kv=kv,
        kd=kd,
        front_kp=front_kp,
        front_shoulder=front_shoulder,
        front_knee=front_knee,
    )
    
    try:
        generator.generate_poses(physics_steps=physics_steps, target_pose=target_pose)
    finally:
        generator.close()
        print("\nDone!")


if __name__ == "__main__":
    fire.Fire(main)
