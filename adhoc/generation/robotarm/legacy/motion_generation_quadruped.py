"""
Generate quadruped (dog) robot motions based on cue definitions.

This script (quadruped version):
1. Loads pose definitions from dog_pose_config and finds matching poses from JSONL
2. Directly controls 4 legs' joint angles (no IK needed)
3. Selects joints based on leg/axis/preference
4. Executes movements and generates GIF

Key differences from arm version:
- 4 independent legs instead of 1 end-effector
- Body height/tilt control affects multiple legs
- Direct joint control (no IK solver needed)

Supported robots:
- Go2, Unitree (generic quadrupeds)
- SpotWithArm, SpotWithArmFloating (Boston Dynamics Spot)

Usage:
    # Go2 robot
    python adhoc/generation/robotarm/motion_generation_quadruped.py \
        --robot Go2 --cue sit_down
    
    # Spot robot (floating base, easier to test)
    python adhoc/generation/robotarm/motion_generation_quadruped.py \
        --robot SpotWithArmFloating --cue paw_shake --camera-distance 2.2
    
    # Spot robot (full quadruped)
    python adhoc/generation/robotarm/motion_generation_quadruped.py \
        --robot SpotWithArm --cue play_bow --camera-distance 2.2
    
    # Custom pose database
    python adhoc/generation/robotarm/motion_generation_quadruped.py \
        --robot SpotWithArm --cue body_bounce \
        --jsonl-path data/poses/spot/closest_SpotWithArm_poses.jsonl

Note:
    For Spot robots, pose databases are auto-detected from:
    - data/poses/spot/closest_{robot}_poses.jsonl (preferred)
    - data/poses/spot/all_{robot}_poses.jsonl (fallback)
    
    If not found, run: python adhoc/spot/export_spot_poses.py --robot SpotWithArm
"""

import fire
import math
import os
import json
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from PIL import Image
from datetime import datetime

import robosuite as suite

# Import pose configuration
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config.dog_pose_config import (
    dog_pose_set, pose_set,
    body_heights, body_tilts, leg_states, legs,
    body_height_joint_offset, body_tilt_joint_offset,
    movement_axes, joint_preference_map,
)


class QuadrupedMotionGenerator:
    """Generate quadruped robot motions based on cue definitions."""
    
    def __init__(
        self,
        robot_name: str = "Go2",
        env_name: str = "EmptySpace",
        controller_name: str = "JOINT_POSITION",
        jsonl_path: Optional[str] = None,
        has_renderer: bool = False,
        has_offscreen_renderer: bool = True,
        control_freq: int = 20,
        output_dir: str = "data/motions_quadruped",
        capture_image_width: int = 512,
        capture_image_height: int = 512,
        camera_distance: float = 0.86,
        camera_spherical_m: float = 1.08,
        camera_elevation_deg: float = 21.0,
        camera_azimuth_deg: float = 44.0,
        camera_lookat_z_offset: float = 0.16,
        hz: int = 4,
    ):
        """
        Initialize the quadruped motion generator.
        
        Args:
            robot_name: Name of the quadruped robot (e.g., "Go2", "Unitree", "SpotWithArm")
            env_name: Name of the environment
            controller_name: Name of the controller (usually JOINT_POSITION for quadrupeds)
            jsonl_path: Path to JSONL file with pose data (auto-detected if None)
            has_renderer: Whether to show on-screen rendering
            has_offscreen_renderer: Whether to enable offscreen rendering
            control_freq: Control frequency
            output_dir: Output directory for GIFs
            capture_image_width: Image width for capture
            capture_image_height: Image height for capture
            camera_distance: Multiplier applied to default vertical FOV after framing (>1 wider, <1 tighter).
            camera_spherical_m: Distance from look-at point to camera (meters); closer = larger robot.
            camera_elevation_deg / camera_azimuth_deg: Direction from target → eye (world +X = 0°, +Y = 90°).
            camera_lookat_z_offset: Raise look-at above root body for nicer framing (meters).
            hz: Frame rate for GIF generation (frames per second)
        """
        self.robot_name = robot_name
        self.env_name = env_name
        self.controller_name = controller_name
        
        # Auto-detect jsonl_path if not provided
        if jsonl_path is None:
            jsonl_path = self._auto_detect_jsonl_path(robot_name)
        
        self.jsonl_path = jsonl_path
        self.control_freq = control_freq
        self.output_dir = os.path.join(output_dir, robot_name)
        self.capture_image_width = capture_image_width
        self.capture_image_height = capture_image_height
        self.camera_distance = camera_distance
        self.camera_spherical_m = camera_spherical_m
        self.camera_elevation_deg = camera_elevation_deg
        self.camera_azimuth_deg = camera_azimuth_deg
        self.camera_lookat_z_offset = camera_lookat_z_offset
        self.hz = hz
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
        # Create environment
        self._create_environment(has_renderer, has_offscreen_renderer)

        leg_joints = getattr(self.robot.robot_model, "legs_joints", None)
        self._leg_joint_names = list(leg_joints) if leg_joints is not None else []
        if not self._leg_joint_names:
            raise RuntimeError(
                f"Robot model for {robot_name!r} has no legs_joints; quadruped sim unsupported."
            )
        print(f"Leg joints ({len(self._leg_joint_names)}): {self._leg_joint_names}")
        
        # Get initial joint positions
        self.initial_joint_pos = self.robot._joint_positions.copy()
        
        # Build leg-to-joint mapping for this specific robot
        self.leg_joint_map = self._build_leg_joint_map()

        # Match mjlab ``record_go2_locomotion_config_mjlab_gif.py`` / ``--hold-base``:
        # keep the floating base upright at a fixed (x,y,z) so leg poses read as standing,
        # not collapsed on the floor when only joint targets are written.
        self._base_free_qpos_slice: Optional[slice] = None
        self._base_free_qvel_slice: Optional[slice] = None
        self._hold_base_qpos: Optional[np.ndarray] = None
        self._lifted_leg: Optional[str] = None
        self._init_mjlab_style_base_hold()
        
        # Place frontview for a closer 3/4 shot on the base, then scale FOV
        self._configure_camera()
        
        print(f"Quadruped motion generator initialized for {robot_name}")
    
    def _auto_detect_jsonl_path(self, robot_name: str) -> str:
        """
        Auto-detect JSONL path based on robot name.
        
        Args:
            robot_name: Name of the robot
        
        Returns:
            Path to JSONL file
        """
        # Check for Spot-specific paths
        if "Spot" in robot_name:
            # Try closest poses first
            closest_poses_path = f"data/poses/spot/closest_{robot_name}_poses.jsonl"
            if os.path.exists(closest_poses_path):
                print(f"Using pre-queried poses: {closest_poses_path}")
                return closest_poses_path
            
            # Fallback to all poses
            all_poses_path = f"data/poses/spot/all_{robot_name}_poses.jsonl"
            if os.path.exists(all_poses_path):
                print(f"Using all poses: {all_poses_path}")
                return all_poses_path
            
            # If neither exists, show error
            print(f"Warning: Pose database not found for {robot_name}")
            print(f"Please run: python adhoc/spot/export_spot_poses.py --robot {robot_name}")
            return f"data/poses/spot/all_{robot_name}_poses.jsonl"
        
        # Default path for other quadrupeds
        default_path = f"data/poses/quadruped_{robot_name.lower()}_poses.jsonl"
        if os.path.exists(default_path):
            print(f"Using pose database: {default_path}")
            return default_path
        
        # Generic fallback
        generic_path = "data/poses/quadruped_poses_results.jsonl"
        if os.path.exists(generic_path):
            print(f"Using generic pose database: {generic_path}")
            return generic_path
        
        print(f"Warning: No pose database found, using default path: {generic_path}")
        return generic_path
    
    def _create_environment(self, has_renderer: bool, has_offscreen_renderer: bool):
        """Create the robosuite environment."""
        try:
            controller_config = suite.load_composite_controller_config(robot=self.robot_name)
        except Exception:
            controller_config = suite.load_composite_controller_config(robot="Go2")

        self.env = suite.make(
            env_name=self.env_name,
            robots=self.robot_name,
            controller_configs=controller_config,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            control_freq=self.control_freq,
            render_camera="frontview",
        )
        
        self.env.reset()
        self.robot = self.env.robots[0]
        
        print(f"Environment created: {self.env_name}")
        print(f"Robot: {self.robot_name}")

    def _nominal_free_base_z_m(self) -> float:
        """
        Root free-joint height when robosuite ``reset()`` leaves the base at (0,0,0).
        mjlab Go2 uses ``INIT_STATE.pos = (0,0,0.32)`` (see ``unitree_go2/go2_constants.py``).
        """
        r = (self.robot_name or "").lower()
        if "spot" in r:
            return 0.35
        return 0.32

    def _init_mjlab_style_base_hold(self) -> None:
        """
        After env reset, canonicalize root pose like mjlab ``_set_canonical_root_state``:
        base x/y = 0, quaternion = identity, z from sim if valid else mjlab nominal (robosuite
        often leaves the free joint at z=0 so the body sits inside the floor).
        Store qpos (and zero qvel) so every leg update can re-apply this hold.
        """
        try:
            import mujoco as mj
        except ImportError:
            return

        model = self.env.sim.model
        data = self.env.sim.data
        root_name = str(self.robot.robot_model.root_body)
        try:
            bid = model.body_name2id(root_name)
        except Exception:
            return
        jadr = int(model.body_jntadr[bid])
        if jadr < 0:
            return
        if int(model.jnt_type[jadr]) != int(mj.mjtJoint.mjJNT_FREE):
            return
        q0 = int(model.jnt_qposadr[jadr])
        self._base_free_qpos_slice = slice(q0, q0 + 7)
        d0 = int(model.jnt_dofadr[jadr])
        self._base_free_qvel_slice = slice(d0, d0 + 6)

        self.env.sim.forward()
        h = np.asarray(data.qpos[self._base_free_qpos_slice], dtype=np.float64).copy()
        z_keep = float(h[2])
        nominal_z = self._nominal_free_base_z_m()
        # robosuite EmptySpace reset commonly leaves freejoint xyz = 0 → buried mesh / no motion read
        if z_keep < 0.12:
            z_keep = nominal_z
        h[0] = 0.0
        h[1] = 0.0
        h[2] = z_keep
        # MuJoCo free joint quat is w,x,y,z
        h[3] = 1.0
        h[4] = 0.0
        h[5] = 0.0
        h[6] = 0.0
        data.qpos[self._base_free_qpos_slice] = h
        data.qvel[self._base_free_qvel_slice] = 0.0
        self.env.sim.forward()
        self._hold_base_qpos = h.copy()

    def _apply_hold_base_qpos(self) -> None:
        if self._hold_base_qpos is None or self._base_free_qpos_slice is None:
            return
        self.env.sim.data.qpos[self._base_free_qpos_slice] = self._hold_base_qpos
        if self._base_free_qvel_slice is not None:
            self.env.sim.data.qvel[self._base_free_qvel_slice] = 0.0
    
    def _build_leg_joint_map(self) -> Dict[str, Dict[str, int]]:
        """
        Build mapping from leg identifiers to joint indices.
        
        Returns:
            Dict mapping leg names (FL, FR, HL, HR) to joint type (hip, shoulder, knee) to joint index
        """
        leg_joint_map = {}
        joint_names = self._leg_joint_names
        
        print("\nBuilding leg-to-joint mapping:")
        print(f"Available joints: {joint_names}")
        
        # Common quadruped joint naming patterns (Unitree Go2 uses RL_/RR_ for rear legs).
        leg_prefixes = {
            'FL': ['fl_', 'fl', 'front_left', 'lf', 'fl0', 'fl1', 'fl2'],
            'FR': ['fr_', 'fr', 'front_right', 'rf', 'fr0', 'fr1', 'fr2'],
            'HL': ['rl_', 'rl', 'hl', 'hind_left', 'rear_left', 'lh', 'hl0', 'hl1', 'hl2'],
            'HR': ['rr_', 'rr', 'hr', 'hind_right', 'rear_right', 'rh', 'hr0', 'hr1', 'hr2'],
        }
        
        joint_type_keywords = {
            'hip': ['hip', 'abduction', 'abd', 'roll', 'hx', '0'],
            'shoulder': ['shoulder', 'thigh', 'flexion', 'gripper_orientation', 'hy', '1'],
            'knee': ['knee', 'calf', 'shank', 'kn', '2'],
        }
        
        for leg_id in legs:
            leg_joint_map[leg_id] = {}
            
            for joint_type, keywords in joint_type_keywords.items():
                found_idx = None
                best_match_score = 0
                
                # Try to find matching joint with scoring
                for idx, joint_name in enumerate(joint_names):
                    joint_name_lower = joint_name.lower()
                    
                    # Check if joint name matches leg prefix
                    leg_match_score = 0
                    for prefix in leg_prefixes[leg_id]:
                        if joint_name_lower.startswith(prefix):
                            leg_match_score = 2  # Strong match (starts with prefix)
                            break
                        elif prefix in joint_name_lower:
                            leg_match_score = 1  # Weak match (contains prefix)
                    
                    if leg_match_score == 0:
                        continue
                    
                    # Check if joint name matches joint type
                    type_match_score = 0
                    for keyword in keywords:
                        if keyword in joint_name_lower:
                            type_match_score = 1
                            break
                    
                    # Calculate total match score
                    total_score = leg_match_score + type_match_score
                    
                    if total_score > best_match_score:
                        best_match_score = total_score
                        found_idx = idx
                
                if found_idx is not None:
                    leg_joint_map[leg_id][joint_type] = found_idx
                    print(f"  {leg_id}_{joint_type} -> joint[{found_idx}]: {joint_names[found_idx]}")
                else:
                    print(f"  Warning: Could not find {joint_type} joint for {leg_id}")
        
        return leg_joint_map
    
    @staticmethod
    def _lookat_quat_wxyz(
        eye: np.ndarray,
        target: np.ndarray,
        world_up: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """World orientation of fixed camera (MuJoCo ``cam_quat``: w,x,y,z). Camera looks along local -Z."""
        from robosuite.utils.transform_utils import mat2quat

        if world_up is None:
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        fwd = target - eye
        fn = float(np.linalg.norm(fwd))
        if fn < 1e-9:
            fwd = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            fwd = fwd / fn
        # glm-style: side = cross(forward, up); up = cross(side, forward)
        right = np.cross(fwd, world_up)
        rn = float(np.linalg.norm(right))
        if rn < 1e-8:
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            right = np.cross(fwd, world_up)
            rn = float(np.linalg.norm(right))
        right = right / max(rn, 1e-9)
        up = np.cross(right, fwd)
        rmat = np.stack([right, up, -fwd], axis=1)
        q_xyzw = mat2quat(rmat)
        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)

    def _configure_camera(self) -> None:
        """
        Move ``frontview`` to a closer 3/4 perspective on the robot root (EmptySpace default
        is very far: ~1.6 m on X), then scale vertical FOV with ``camera_distance``.
        """
        camera_name = "frontview"
        try:
            model = self.env.sim.model
            data = self.env.sim.data
            cam_id = model.camera_name2id(camera_name)
            self.env.sim.forward()
            root = str(self.robot.robot_model.root_body)
            bid = model.body_name2id(root)
            target = np.asarray(data.body_xpos[bid], dtype=np.float64).copy()
            target[2] += float(self.camera_lookat_z_offset)

            el = np.radians(float(self.camera_elevation_deg))
            az = np.radians(float(self.camera_azimuth_deg))
            m = float(self.camera_spherical_m)
            offset = m * np.array(
                [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
                dtype=np.float64,
            )
            eye = target + offset

            model.cam_pos[cam_id] = eye
            model.cam_quat[cam_id] = self._lookat_quat_wxyz(eye, target)
            self.env.sim.forward()

            base_fov = float(model.cam_fovy[cam_id])
            new_fov = base_fov * float(self.camera_distance)
            new_fov = max(22.0, min(95.0, new_fov))
            model.cam_fovy[cam_id] = new_fov
            print(
                f"Camera framed: eye={np.round(eye, 3)}, target_z={target[2]:.3f}, "
                f"FOV {base_fov:.1f}° → {new_fov:.1f}° (×{self.camera_distance})"
            )
        except Exception as e:
            print(f"Warning: Could not configure camera: {e}")
    
    def _load_pose_database(self, jsonl_path: str) -> List[Dict]:
        """Load pose database from JSONL file."""
        if not os.path.exists(jsonl_path):
            print(f"Warning: JSONL file not found: {jsonl_path}")
            return []
        
        poses = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    poses.append(json.loads(line))
        
        return poses
    
    def _find_matching_poses(
        self,
        pose_def,  # Can be string (pose name) or dict (pose definition)
        robot_name: Optional[str] = None,
    ) -> List[Dict]:
        """Find poses matching the given pose name or definition."""
        if robot_name is None:
            robot_name = self.robot_name
        
        # Handle both string (pose name) and dict (pose definition) formats
        if isinstance(pose_def, str):
            # Pose name as string
            pose_name = pose_def
            if pose_name not in dog_pose_set:
                print(f"Error: Pose '{pose_name}' not found in dog_pose_set")
                return []
            pose_def = dog_pose_set[pose_name]
        elif isinstance(pose_def, dict):
            # Pose definition as dict
            pass
        else:
            print(f"Error: Invalid pose format: {type(pose_def)}")
            return []
        
        # Extract pose features
        body_height = pose_def.get('body_height')
        body_tilt = pose_def.get('body_tilt')
        leg_FL = pose_def.get('leg_FL')
        leg_FR = pose_def.get('leg_FR')
        leg_HL = pose_def.get('leg_HL')
        leg_HR = pose_def.get('leg_HR')
        
        # Load pose database
        pose_database = self._load_pose_database(self.jsonl_path)
        
        if not pose_database:
            print(f"Warning: Pose database is empty or not found: {self.jsonl_path}")
            return []
        
        # Filter poses matching all criteria
        matching_poses = []
        for pose_data in pose_database:
            if pose_data.get('robot') != robot_name:
                continue
            
            pose_features = pose_data.get('pose_features', {})
            
            # Check if all features match
            match = True
            if body_height and pose_features.get('body_height') != body_height:
                match = False
            if body_tilt and pose_features.get('body_tilt') != body_tilt:
                match = False
            if leg_FL and pose_features.get('leg_FL') != leg_FL:
                match = False
            if leg_FR and pose_features.get('leg_FR') != leg_FR:
                match = False
            if leg_HL and pose_features.get('leg_HL') != leg_HL:
                match = False
            if leg_HR and pose_features.get('leg_HR') != leg_HR:
                match = False
            
            if match:
                matching_poses.append(pose_data)
        
        print(f"  Found {len(matching_poses)} matching poses for {pose_def}")
        return matching_poses
    
    def _select_joint_for_movement(
        self,
        axis: str,  # e.g., 'body_z', 'leg_FL', 'body_pitch'
        joint_preference: str,  # 'proximal', 'middle', 'distal'
    ) -> Tuple[int, str, str]:
        """
        Select a joint for movement based on axis and preference.
        
        Args:
            axis: Movement axis (from movement_axes)
            joint_preference: Joint preference (proximal/middle/distal)
        
        Returns:
            Tuple of (joint_index, joint_name, leg_id)
        """
        if axis not in movement_axes:
            raise ValueError(f"Unknown axis: {axis}. Available axes: {list(movement_axes.keys())}")
        
        axis_info = movement_axes[axis]
        affects = axis_info['affects']
        primary_joint_type = axis_info['primary_joint']
        
        # Map joint preference to actual joint type
        if joint_preference in joint_preference_map:
            joint_type = joint_preference_map[joint_preference]
        else:
            joint_type = primary_joint_type
        
        # Determine which leg(s) to use
        if affects == 'all_legs':
            # Body height movement - pick one leg (all will move together)
            selected_leg = 'FL'  # Use front-left as representative
        elif affects == 'front_vs_back':
            # Body gripper_orientation - pick front or back leg
            selected_leg = 'FL' if joint_preference == 'proximal' else 'HL'
        elif affects == 'left_vs_right':
            # Body roll - pick left or right leg
            selected_leg = 'FL' if joint_preference == 'proximal' else 'FR'
        elif affects in legs:
            # Individual leg movement
            selected_leg = affects
        else:
            raise ValueError(f"Unknown affects value: {affects}")
        
        # Get joint index from leg_joint_map
        if selected_leg not in self.leg_joint_map:
            raise ValueError(f"Leg {selected_leg} not found in leg_joint_map")
        
        if joint_type not in self.leg_joint_map[selected_leg]:
            # Fallback to available joint
            available_joints = list(self.leg_joint_map[selected_leg].keys())
            if not available_joints:
                raise ValueError(f"No joints found for leg {selected_leg}")
            joint_type = available_joints[0]
            print(f"  Warning: {joint_type} not found for {selected_leg}, using {available_joints[0]}")
        
        joint_idx = self.leg_joint_map[selected_leg][joint_type]
        joint_name = self._leg_joint_names[joint_idx]
        
        print(f"\nSelected joint for {axis} movement:")
        print(f"  Leg: {selected_leg}, Joint type: {joint_type}")
        print(f"  Joint index: {joint_idx}, Joint name: {joint_name}")
        
        return joint_idx, joint_name, selected_leg
    
    def _pose_data_to_joint_positions(self, pose_data: Dict) -> np.ndarray:
        """
        Convert pose data to joint positions array.
        
        Args:
            pose_data: Dictionary containing pose information with joint_angles_rad
        
        Returns:
            Joint positions array in radians
        """
        joint_angles_rad = pose_data.get("joint_angles_rad", [])
        
        # Start with initial joint positions
        joint_pos = self.initial_joint_pos.copy()
        
        # Update with pose joint angles
        for i, angle in enumerate(joint_angles_rad):
            if i < len(joint_pos):
                joint_pos[i] = angle
        
        return joint_pos
    
    def _get_joint_positions(self):
        """Get current joint positions."""
        return self.robot._joint_positions.copy()
    
    def _set_joint_positions(self, joint_positions):
        """Set joint positions."""
        self.robot.set_robot_joint_positions(joint_positions)
        self._apply_hold_base_qpos()
        self.env.sim.forward()
        
        # Stabilize
        for _ in range(10):
            self.env.sim.data.qvel[:] = 0
            self._apply_hold_base_qpos()
            self.env.sim.forward()
    
    def _capture_image(self):
        """Capture current camera view."""
        obs = self.env.sim.render(
            camera_name="frontview",
            width=self.capture_image_width,
            height=self.capture_image_height,
            depth=False
        )
        return obs[::-1]
    
    def _check_self_collision(self) -> bool:
        """
        Check if the robot has self-collision.
        
        Returns:
            bool: True if self-collision is detected, False otherwise
        """
        robot_geoms = set(self.robot.robot_model.contact_geoms)
        
        for i in range(self.env.sim.data.ncon):
            contact = self.env.sim.data.contact[i]
            geom1_name = self.env.sim.model.geom_id2name(contact.geom1)
            geom2_name = self.env.sim.model.geom_id2name(contact.geom2)
            
            if geom1_name in robot_geoms and geom2_name in robot_geoms:
                return True
        
        return False
    
    def _load_cue_config(
        self,
        cue: str,
        config_path: str = "data/results/motion_configs/manipulator/motion_config_quadruped.json",
        cue_idx: Optional[int] = None,
    ) -> Dict:
        """
        Load cue configuration from JSON file.
        
        Args:
            cue: Name of the cue
            config_path: Path to JSON file with cue configurations
            cue_idx: If set, pick the row with matching ``idx`` (same as arm ``MotionGenerator``).
        
        Returns:
            Dictionary containing cue configuration
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Cue configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            configs = json.load(f)
        
        if cue_idx is not None:
            for config in configs:
                if config.get("idx") == cue_idx:
                    return config
        
        for config in configs:
            if config.get('cue') == cue:
                return config
        
        raise ValueError(f"Cue '{cue}' not found in configuration file: {config_path}")

    @staticmethod
    def _is_lc19_pose_dict(pose_param) -> bool:
        if not isinstance(pose_param, dict):
            return False
        return any(
            k in pose_param
            for k in ("dir", "front_height", "back_height", "three_legs", "body_height", "body_tilt")
        )

    @staticmethod
    def _lifted_leg_from_lc19(pose_param: dict) -> Optional[str]:
        if not bool(pose_param.get("three_legs", False)):
            return None
        direction = str(pose_param.get("dir", "neutral")).lower()
        mapping = {
            "left": "FL",
            "right": "FR",
            "front": "FR",
            "back": "FR",
            "neutral": None,
        }
        if direction not in mapping:
            raise ValueError(f"Unsupported pose direction for lifted leg selection: {direction}")
        return mapping[direction]

    @staticmethod
    def _three_leg_joint_targets(hand_position: str) -> Tuple[float, float, float]:
        hp = str(hand_position or "back").lower()
        targets = {
            "up": (0.000, -1.571, -1.780),
            "front": (0.000, -1.571, -1.110),
            "right": (-1.047, 0.960, -0.838),
            "back": (0.000, 0.960, -2.723),
            "left": (1.047, 0.960, -0.838),
        }
        return targets.get(hp, targets["back"])

    def _lc19_pose_to_joint_positions(self, pose_param: dict) -> np.ndarray:
        q = np.asarray(self.initial_joint_pos, dtype=float).copy()
        front_level = str(pose_param.get("front_height", "mid")).lower()
        rear_level = str(pose_param.get("back_height", pose_param.get("rear_height", "mid"))).lower()
        height_thigh_delta = {"low": 0.14, "mid": 0.0, "high": -0.12}
        front_delta = height_thigh_delta.get(front_level, 0.0)
        rear_delta = height_thigh_delta.get(rear_level, 0.0)
        for leg_id in ("FL", "FR"):
            ji = self.leg_joint_map.get(leg_id, {}).get("shoulder")
            if ji is not None:
                q[ji] += front_delta
        for leg_id in ("HL", "HR"):
            ji = self.leg_joint_map.get(leg_id, {}).get("shoulder")
            if ji is not None:
                q[ji] += rear_delta

        lifted = self._lifted_leg_from_lc19(pose_param)
        if lifted is not None:
            tpl = pose_param.get("three_leg_pose") or {}
            hand_position = str(tpl.get("hand_position", "back"))
            hip, thigh, calf = self._three_leg_joint_targets(hand_position)
            leg_map = self.leg_joint_map.get(lifted, {})
            if "hip" in leg_map:
                q[leg_map["hip"]] = hip
            if "shoulder" in leg_map:
                q[leg_map["shoulder"]] = thigh
            if "knee" in leg_map:
                q[leg_map["knee"]] = calf
        return q

    def _pick_leg_joint_for_world_axis(
        self,
        leg_id: str,
        axis: str,
        q: np.ndarray,
    ) -> tuple[int | None, float]:
        """Pick hip/thigh/knee whose foot Jacobian best aligns with world x/y/z."""
        ax_idx = {"x": 0, "y": 1, "z": 2}.get(str(axis).lower())
        if ax_idx is None:
            return None, 1.0
        leg_map = self.leg_joint_map.get(leg_id) or {}
        candidates = [leg_map[k] for k in ("hip", "shoulder", "knee") if k in leg_map]
        if not candidates:
            return None, 1.0

        # Prefer a foot/toe site or distal body for this leg.
        sim = self.env.sim
        model = sim.model
        data = sim.data
        data.qpos[: len(q)] = q
        sim.forward()

        foot_body = None
        leg_l = leg_id.lower()
        for bid in range(model.nbody):
            name = model.body_id2name(bid) or ""
            nl = name.lower()
            if leg_l.replace("h", "r") in nl or leg_l in nl:
                if any(tok in nl for tok in ("foot", "toe", "calf", "ankle")):
                    foot_body = name
        if foot_body is None:
            # Fallback: use EE-like body from first candidate's child if available.
            return candidates[-1], 1.0

        try:
            jacp = data.get_body_jacp(foot_body).reshape(3, -1)
        except Exception:
            return candidates[-1], 1.0

        best_ji = None
        best_val = 0.0
        # Map qpos index → qvel dof. For revolute 1-DOF joints, dofadr ≈ qpos for limbs
        # after freejoint; use joint_qvel_addr if available.
        for ji in candidates:
            try:
                # robosuite/mujoco: joint qvel address
                jnt_id = None
                for j in range(model.njnt):
                    if int(model.jnt_qposadr[j]) == int(ji):
                        jnt_id = j
                        break
                if jnt_id is None:
                    continue
                dof = int(model.jnt_dofadr[jnt_id])
                val = float(jacp[ax_idx, dof])
            except Exception:
                continue
            if abs(val) > abs(best_val):
                best_val = val
                best_ji = ji
        if best_ji is None:
            return candidates[-1], 1.0
        return best_ji, best_val

    @staticmethod
    def _yaw_from_free_qpos(qpos: np.ndarray) -> float:
        qw, qx, qy, qz = float(qpos[3]), float(qpos[4]), float(qpos[5]), float(qpos[6])
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def _execute_lc19_movement(
        self,
        movement_item: dict,
        parameters: dict,
        frames: list,
        hz: int,
    ) -> None:
        """Prompt-19 / LC19 quadruped format: ``joints`` array + repetition + speed."""
        duration = float(movement_item.get("duration", 1.0))
        speed = float(parameters.get("speed", 1.0))
        n_frames = max(1, int(duration * hz / max(0.1, speed)))
        joints = parameters.get("joints") or []
        if not joints:
            # ``hold(t=…)`` is converted to an empty-joints movement. Freeze the
            # current configuration for the authored duration instead of no-oping.
            for _ in range(n_frames):
                self._apply_hold_base_qpos()
                image = self._capture_image()
                frames.append(Image.fromarray(image))
            print(f"  Captured {n_frames} hold frames (empty joints, duration={duration:.2f}s)")
            return
        reps = max(1, int(parameters.get("repetition", 1)))

        role = {"hip": "hip", "thigh": "shoulder", "knee": "knee", "calf": "knee", "foot": "knee", "auto": "auto", "shoulder": "shoulder", "elbow": "knee", "wrist": "knee"}
        base_q = self._get_joint_positions().copy()
        leg_order = ("FL", "FR", "HL", "HR")

        for fi in range(n_frames):
            t_global = fi / max(1, n_frames - 1)
            if reps > 1:
                cycle_t = (t_global * reps) % 1.0
                pp = 1.0 - abs(2.0 * cycle_t - 1.0)
            else:
                pp = t_global

            q_frame = base_q.copy()
            for jspec in joints:
                jname = str(jspec.get("joint", "")).lower()
                internal = role.get(jname, jname)
                deg = jspec.get("degrees", {})
                world_axis = str(jspec.get("axis") or "").lower()
                if isinstance(deg, dict):
                    for ak in ("x", "y", "z"):
                        if ak in deg:
                            world_axis = world_axis or ak
                            break
                leg_id = parameters.get("leg") or getattr(self, "_lifted_leg", None)
                if leg_id is None:
                    for lg in leg_order:
                        if any(r in self.leg_joint_map.get(lg, {}) for r in ("hip", "shoulder", "knee")):
                            leg_id = lg
                            break
                if leg_id is None:
                    continue

                # World-axis pick among hip/thigh/knee when joint=auto or axis is x/y/z.
                ji = None
                jac_val = 1.0
                if world_axis in ("x", "y", "z") and (
                    internal == "auto" or jname in ("auto", "shoulder", "elbow", "wrist")
                ):
                    ji, jac_val = self._pick_leg_joint_for_world_axis(leg_id, world_axis, base_q)
                elif internal in self.leg_joint_map.get(leg_id, {}):
                    ji = self.leg_joint_map[leg_id][internal]
                if ji is None:
                    continue

                delta_deg = 0.0
                if isinstance(deg, dict):
                    for axis_key in ("x", "y", "z"):
                        if axis_key not in deg:
                            continue
                        val = deg[axis_key]
                        if isinstance(val, (list, tuple)) and len(val) >= 2:
                            lo, hi = float(val[0]), float(val[1])
                            delta_deg = lo + (hi - lo) * pp
                        else:
                            delta_deg = float(val)
                        break
                elif isinstance(deg, (list, tuple)) and len(deg) >= 2:
                    lo, hi = float(deg[0]), float(deg[1])
                    delta_deg = lo + (hi - lo) * pp
                elif isinstance(deg, (int, float)):
                    delta_deg = float(deg)

                from adhoc.generation.world_axis_motion import align_offset_to_world_axis

                dq = np.deg2rad(delta_deg)
                if world_axis in ("x", "y", "z"):
                    dq = align_offset_to_world_axis(dq, jac_val)
                q_frame[ji] = base_q[ji] + dq

            self._set_joint_positions(q_frame)
            frames.append(Image.fromarray(self._capture_image()))

    def execute_cue(
        self,
        cue: str,
        pose_index: Optional[int] = None,
        config_path: str = "data/results/motion_configs/manipulator/motion_config_quadruped.json",
        hz: int = 4,
        filename_suffix: Optional[str] = None,
        enable_self_collision_check: bool = False,
        cue_idx: Optional[int] = None,
        save_gif: bool = True,
    ):
        """
        Execute a cue (e.g., 'sit_down', 'paw_shake').
        
        Args:
            cue: Name of the cue
            pose_index: Optional pose_id to use (if None, randomly selects)
            config_path: Path to JSON file with cue configurations
            hz: Frame rate for GIF
            filename_suffix: Optional suffix for output filename
            enable_self_collision_check: Enable self-collision detection
        """
        print(f"\n{'='*60}")
        print(f"Executing cue: {cue}")
        print(f"{'='*60}\n")
        
        # Load cue configuration
        cue_config_data = self._load_cue_config(cue, config_path, cue_idx=cue_idx)
        movements = cue_config_data.get('movements', [])
        
        if not movements:
            raise ValueError(f"No movements found in cue '{cue}' configuration")
        
        frames = []
        current_pose_name = None
        current_pose = None
        pose_id = None
        
        for movement_item in movements:
            movement_type = movement_item.get('type')
            parameters = movement_item.get('parameters', {})
            
            if movement_type == 'pose':
                # Set pose
                pose_param = parameters.get('pose')
                if pose_param is None:
                    raise ValueError("'pose' parameter is required for 'pose' type movement")
                
                # Handle both string and dict formats
                if isinstance(pose_param, str):
                    pose_display_name = pose_param
                elif isinstance(pose_param, dict):
                    pose_display_name = f"{pose_param.get('body_height', '?')}_{pose_param.get('body_tilt', '?')}"
                else:
                    raise ValueError(f"Invalid pose format: {type(pose_param)}")
                
                print(f"\n--- Setting Pose: {pose_display_name} ---")
                if self._is_lc19_pose_dict(pose_param):
                    target_joint_pos = self._lc19_pose_to_joint_positions(pose_param)
                    self._lifted_leg = self._lifted_leg_from_lc19(pose_param)
                    selected_pose = {"joint_angles_rad": target_joint_pos.tolist()}
                    print(f"  LC19 pose (lifted leg: {self._lifted_leg or 'none'})")
                else:
                    self._lifted_leg = None
                    matching_poses = self._find_matching_poses(pose_param)
                    if not matching_poses:
                        print("  Warning: No matching poses in database, will use default position")
                        selected_pose = {"joint_angles_rad": self.initial_joint_pos.tolist()}
                    else:
                        selected_pose = random.choice(matching_poses)
                        print(f"  Selected pose: {selected_pose.get('pose_id', 'N/A')}")
                
                speed = parameters.get('speed', 1.0)
                hold_time = float(movement_item.get('duration', parameters.get('hold_time', 1.0)))
                
                if current_pose is None:
                    # First pose
                    print("Setting first pose...")
                    target_joint_pos = self._pose_data_to_joint_positions(selected_pose)
                    self._set_joint_positions(target_joint_pos)
                    current_pose_name = pose_param
                    current_pose = selected_pose
                    
                    # Hold
                    num_frames = int(hold_time * hz)
                    for _ in range(num_frames):
                        image = self._capture_image()
                        frames.append(Image.fromarray(image))
                    print(f"Captured {num_frames} frames (hold_time: {hold_time}s)")
                else:
                    # Transition
                    print(f"Transitioning to {pose_display_name} (speed: {speed})...")
                    
                    start_joint_pos = self._get_joint_positions()
                    end_joint_pos = self._pose_data_to_joint_positions(selected_pose)
                    
                    duration = 1.0 / speed
                    num_transition_frames = int(duration * hz)
                    
                    for frame_idx in range(num_transition_frames):
                        t = (frame_idx + 1) / num_transition_frames
                        interpolated_joint_pos = start_joint_pos * (1 - t) + end_joint_pos * t
                        self._set_joint_positions(interpolated_joint_pos)
                        
                        image = self._capture_image()
                        frames.append(Image.fromarray(image))
                    
                    print(f"  Captured {num_transition_frames} transition frames")
                    
                    # Set final pose
                    self._set_joint_positions(end_joint_pos)
                    current_pose_name = pose_param
                    current_pose = selected_pose
                    
                    # Hold
                    num_hold_frames = int(hold_time * hz)
                    for _ in range(num_hold_frames):
                        image = self._capture_image()
                        frames.append(Image.fromarray(image))
                    print(f"  Captured {num_hold_frames} hold frames")
            
            elif movement_type == 'movement':
                # Execute movement
                if current_pose_name is None:
                    raise ValueError("No pose set before movement")

                raw_params = movement_item.get('parameters', {})
                if isinstance(raw_params.get('movement'), dict):
                    parameters = dict(raw_params['movement'])
                else:
                    parameters = raw_params

                if parameters.get('joints'):
                    self._execute_lc19_movement(movement_item, parameters, frames, hz)
                    continue

                repetition = parameters.get('repetition', 1)
                axis = parameters.get('axis')
                joint_preference = parameters.get('joint', 'middle')
                directions = parameters.get('directions', [])
                
                if axis is None:
                    raise ValueError("'axis' parameter is required")
                if not directions:
                    raise ValueError("'directions' parameter is required")
                
                print(f"\n--- Movement ---")
                print(f"Axis: {axis}, Joint preference: {joint_preference}")
                print(f"Repetition: {repetition}")
                
                # Select joint
                joint_idx, joint_name, leg_id = self._select_joint_for_movement(axis, joint_preference)
                
                current_joint_pos = self._get_joint_positions()
                
                # Execute repetitions
                for rep in range(repetition):
                    print(f"\nRepetition {rep + 1}/{repetition}")
                    
                    for direction_config in directions:
                        degrees = direction_config.get('degrees', 0)
                        speed = direction_config.get('speed', 1.0)
                        hold_time = direction_config.get('hold_time', 0.5)
                        
                        print(f"  Movement: {degrees}°, speed: {speed}")
                        
                        # Calculate target
                        start_angle_rad = current_joint_pos[joint_idx]
                        target_angle_offset_rad = np.deg2rad(degrees)
                        target_angle_rad = start_angle_rad + target_angle_offset_rad
                        
                        # Movement duration
                        duration = 1.0 / speed
                        num_movement_frames = int(duration * hz)
                        
                        # Interpolate
                        for frame_idx in range(num_movement_frames):
                            t = (frame_idx + 1) / num_movement_frames
                            interpolated_angle_rad = start_angle_rad + t * target_angle_offset_rad
                            
                            new_joint_pos = current_joint_pos.copy()
                            new_joint_pos[joint_idx] = interpolated_angle_rad
                            self._set_joint_positions(new_joint_pos)
                            
                            image = self._capture_image()
                            frames.append(Image.fromarray(image))
                        
                        print(f"  Captured {num_movement_frames} movement frames")
                        
                        # Update current position
                        current_joint_pos[joint_idx] = target_angle_rad
                        
                        # Hold
                        if hold_time > 0:
                            num_hold_frames = int(hold_time * hz)
                            for _ in range(num_hold_frames):
                                image = self._capture_image()
                                frames.append(Image.fromarray(image))
                            print(f"  Captured {num_hold_frames} hold frames")
            elif movement_type == "path":
                if current_pose_name is None:
                    raise ValueError("No pose set before path")
                path = (movement_item.get("parameters") or {}).get("path") or {}
                shape = str(path.get("shape", "line")).lower()
                speed = max(0.1, float(path.get("speed", 0.8)))
                if shape == "line":
                    x_dist = float(path.get("x", 0.0))
                    y_dist = float(path.get("y", 0.0))
                    if abs(x_dist) < 1e-9 and abs(y_dist) < 1e-9:
                        x_dist = float(path.get("distance", 0.3))
                    total_dist = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                    n_frames = max(6, int(total_dist / speed * hz))
                    start_xy = (0.0, 0.0)
                    if self._hold_base_qpos is not None:
                        start_xy = (float(self._hold_base_qpos[0]), float(self._hold_base_qpos[1]))
                    for fi in range(n_frames):
                        t = (fi + 1) / n_frames
                        if self._base_free_qpos_slice is not None:
                            self.env.sim.data.qpos[self._base_free_qpos_slice][0] = start_xy[0] + x_dist * t
                            self.env.sim.data.qpos[self._base_free_qpos_slice][1] = start_xy[1] + y_dist * t
                        self.env.sim.data.qvel[:] = 0
                        self.env.sim.forward()
                        frames.append(Image.fromarray(self._capture_image()))
                    if self._hold_base_qpos is not None:
                        self._hold_base_qpos[0] = start_xy[0] + x_dist
                        self._hold_base_qpos[1] = start_xy[1] + y_dist
                elif shape == "arc":
                    degrees = float(path.get("degrees", 360.0))
                    radius = float(path.get("radius", 0.0))
                    total_rad = math.radians(degrees)
                    if abs(radius) < 1e-6:
                        n_frames = max(24, int(abs(total_rad) / (speed * 0.12) * hz))
                        start_q = (
                            np.asarray(self._hold_base_qpos, dtype=float).copy()
                            if self._hold_base_qpos is not None
                            else np.zeros(7, dtype=float)
                        )
                        x0, y0, z0 = float(start_q[0]), float(start_q[1]), float(start_q[2])
                        start_yaw = self._yaw_from_free_qpos(start_q)
                        for fi in range(n_frames):
                            t = (fi + 1) / n_frames
                            yaw = start_yaw + total_rad * t
                            half = yaw * 0.5
                            q = start_q.copy()
                            q[0], q[1], q[2] = x0, y0, z0
                            q[3] = math.cos(half)
                            q[4] = 0.0
                            q[5] = 0.0
                            q[6] = math.sin(half)
                            if self._base_free_qpos_slice is not None:
                                self.env.sim.data.qpos[self._base_free_qpos_slice] = q
                            self.env.sim.data.qvel[:] = 0
                            self.env.sim.forward()
                            frames.append(Image.fromarray(self._capture_image()))
                    else:
                        arc_len = abs(total_rad) * abs(radius)
                        n_frames = max(24, int(arc_len / speed * hz))
                        start_q = (
                            np.asarray(self._hold_base_qpos, dtype=float).copy()
                            if self._hold_base_qpos is not None
                            else np.zeros(7, dtype=float)
                        )
                        cx = float(start_q[0]) - radius * math.cos(self._yaw_from_free_qpos(start_q))
                        cy = float(start_q[1]) - radius * math.sin(self._yaw_from_free_qpos(start_q))
                        start_angle = math.atan2(float(start_q[1]) - cy, float(start_q[0]) - cx)
                        z0 = float(start_q[2])
                        for fi in range(n_frames):
                            t = (fi + 1) / n_frames
                            ang = start_angle + total_rad * t
                            x = cx + radius * math.cos(ang)
                            y = cy + radius * math.sin(ang)
                            yaw = ang + (math.pi / 2.0 if total_rad >= 0 else -math.pi / 2.0)
                            half = yaw * 0.5
                            q = start_q.copy()
                            q[0], q[1], q[2] = x, y, z0
                            q[3] = math.cos(half)
                            q[4] = 0.0
                            q[5] = 0.0
                            q[6] = math.sin(half)
                            if self._base_free_qpos_slice is not None:
                                self.env.sim.data.qpos[self._base_free_qpos_slice] = q
                            self.env.sim.data.qvel[:] = 0
                            self.env.sim.forward()
                            frames.append(Image.fromarray(self._capture_image()))
                        if self._hold_base_qpos is not None:
                            self._hold_base_qpos[0] = x
                            self._hold_base_qpos[1] = y
                else:
                    raise ValueError(f"Unsupported quadruped path shape: {shape}")
            else:
                raise ValueError(f"Unknown movement type: {movement_type}")
        
        if len(frames) == 0:
            raise ValueError("No frames captured")
        if not save_gif:
            return frames, pose_id

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{now}_{self.robot_name}_{cue}"

        if filename_suffix:
            output_filename = f"{base_filename}_{filename_suffix}.gif"
        else:
            output_filename = f"{base_filename}.gif"

        filepath = os.path.join(self.output_dir, output_filename)
        frame_duration_ms = int(1000 / hz)

        frames[0].save(
            filepath,
            save_all=True,
            append_images=frames[1:] if len(frames) > 1 else [],
            duration=frame_duration_ms,
            loop=0
        )
        print(f"\n{'='*60}")
        print(f"Saved GIF: {filepath}")
        print(f"Total frames: {len(frames)}")
        print(f"{'='*60}\n")
    
    def close(self):
        """Close the environment."""
        self.env.close()


def generate(
    robot: str = "Go2",
    env: str = "EmptySpace",
    cue: str = "sit_down",
    pose_index: Optional[int] = None,
    controller: str = "JOINT_POSITION",
    jsonl_path: Optional[str] = None,
    config_path: str = "data/results/motion_configs/manipulator/motion_config_quadruped.json",
    camera_distance: float = 2.5,
    hz: int = 4,
    enable_self_collision_check: bool = False,
):
    """
    Main function to generate quadruped robot motions.
    
    Args:
        robot: Robot name (e.g., "Go2", "Unitree", "SpotWithArm", "SpotWithArmFloating")
        env: Environment name
        cue: Name of the cue to execute
        pose_index: Optional pose_id to use
        controller: Controller name
        jsonl_path: Path to pose database JSONL file (auto-detected if None)
        config_path: Path to JSON file with cue configurations
        camera_distance: Multiplier for camera FOV to zoom out (2.2-2.5 recommended for quadrupeds)
        hz: Frame rate for GIF generation
        enable_self_collision_check: Enable self-collision detection
    
    Examples:
        # Go2 robot
        python adhoc/generation/robotarm/motion_generation_quadruped.py \
            --robot Go2 --cue sit_down
        
        # Spot robot (auto-detects pose database)
        python adhoc/generation/robotarm/motion_generation_quadruped.py \
            --robot SpotWithArm --cue paw_shake --camera-distance 2.2
        
        # Custom pose database
        python adhoc/generation/robotarm/motion_generation_quadruped.py \
            --robot SpotWithArm --cue play_bow \
            --jsonl-path data/poses/spot/closest_SpotWithArm_poses.jsonl
    """
    
    generator = QuadrupedMotionGenerator(
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
        generator.execute_cue(
            cue=cue,
            pose_index=pose_index,
            config_path=config_path,
            hz=hz,
            enable_self_collision_check=enable_self_collision_check,
        )
    finally:
        generator.close()
    
    return True


if __name__ == "__main__":
    fire.Fire(generate)
