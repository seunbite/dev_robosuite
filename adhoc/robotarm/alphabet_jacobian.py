"""
Compute and print Jacobian matrices for robot joints at initial pose.

This script:
1. Loads pose definitions from pose_set
2. Finds matching poses from closest_poses_results.jsonl
3. Sets robot to initial pose
4. Computes and prints Jacobian matrices for each joint
"""

import fire
import os
import json
import random
import numpy as np
from typing import Dict, List, Optional
from PIL import Image
from datetime import datetime
import mujoco

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.utils.ik_utils import IKSolver

# Import pose configuration
from arm_pose_config import poses, pitch_poses, pose_set


def _debug_log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[jacobian_debug {ts}] {msg}", flush=True)

# Fixed joint indices - now arm-specific for humanoid robots
FIXED_JOINT_INDICES = {
    'GR1': {
        'right': "0-5, 13-19, 20-31",  # Fix head, torso, left arm, legs
        'left': "0-5, 6-12, 20-31",     # Fix head, torso, right arm, legs
    },
    'GR1FixedLowerBody': {
        'right': "0-5, 13-19",  # Fix head, torso, left arm
        'left': "0-5, 6-12",     # Fix head, torso, right arm
    },
    'GR1FloatingBody': {
        'right': "0-5, 13-19",
        'left': "0-5, 6-12",
    },
    'GR1ArmsOnly': {
        'right': "7-13",
        'left': "0-6",
    },
}


class JacobianCalculator:
    """Calculate and print Jacobian matrices for robot joints."""
    
    def __init__(
        self,
        robot_name: str = "Panda",
        env_name: str = "EmptySpace",
        controller_name: str = "IK_POSE",
        jsonl_path: str = "data/seed/closest_poses_results.jsonl",
        has_renderer: bool = False,
        has_offscreen_renderer: bool = True,
        control_freq: int = 20,
        save_jacobian_gif: bool = False,
        active_arm: Optional[str] = None,  # For humanoid robots: "right" or "left"
    ):
        """
        Initialize the Jacobian calculator.
        
        Args:
            robot_name: Name of the robot
            env_name: Name of the environment
            controller_name: Name of the controller (should support IK)
            jsonl_path: Path to JSONL file with pose data
            has_renderer: Whether to show on-screen rendering
            has_offscreen_renderer: Whether to enable offscreen rendering
            control_freq: Control frequency
            save_jacobian_gif: Whether to save GIF files for joint movements (default: False)
        """
        self.robot_name = robot_name
        self.env_name = env_name
        self.controller_name = controller_name
        self.jsonl_path = jsonl_path
        self.control_freq = control_freq
        self.save_jacobian_gif = save_jacobian_gif
        
        if self.save_jacobian_gif:
            self.output_dir = os.path.join("data/jacobian_test", robot_name)
            # Create output directory only if saving GIFs
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Output directory: {self.output_dir}")
        else:
            self.output_dir = None
        
        print(f"Initializing robot: {robot_name}")
        _debug_log("starting JacobianCalculator init")
        
        # Load pose data from JSONL
        _debug_log(f"loading pose database from {jsonl_path}")
        self.pose_database = self._load_pose_database(jsonl_path)
        print(f"Loaded {len(self.pose_database)} poses from {jsonl_path}")
        
        # Setup environment
        options = {
            "env_name": env_name,
            "robots": robot_name,
            "has_renderer": has_renderer,
            "has_offscreen_renderer": has_offscreen_renderer,
            "renderer": "mujoco",
            "ignore_done": True,
            "use_camera_obs": True,
            "camera_names": "frontview",
            "camera_heights": 512,
            "camera_widths": 512,
            "control_freq": control_freq,
        }
        
        # Load controller config
        _debug_log(f"loading controller config: {controller_name}")
        arm_controller_config = suite.load_part_controller_config(default_controller=controller_name)
        options["controller_configs"] = refactor_composite_controller_config(
            arm_controller_config, robot_name, ["right", "left"]
        )
        
        # Create environment
        _debug_log("calling suite.make")
        self.env = suite.make(**options, horizon=10000)
        _debug_log("suite.make returned")
        _debug_log("calling env.reset")
        self.env.reset()
        _debug_log("env.reset returned")
        
        # Disable gravity
        self.env.sim.model.opt.gravity[:] = [0, 0, 0]
        print("Gravity disabled")
        _debug_log("gravity disabled")
        
        # Get robot
        self.robot = self.env.robots[0]
        _debug_log("robot handle acquired")
        
        # Get initial joint positions
        self.initial_joint_pos = self.robot._joint_positions.copy()
        self.num_joints = len(self.initial_joint_pos)
        
        # Parse fixed joint indices (arm-specific for humanoid robots)
        self.fixed_joint_indices = []
        if robot_name in FIXED_JOINT_INDICES:
            fixed_data = FIXED_JOINT_INDICES[robot_name]
            # Check if arm-specific (dict) or global (string)
            if isinstance(fixed_data, dict) and active_arm:
                fixed_indices_str = fixed_data.get(active_arm, "")
            elif isinstance(fixed_data, str):
                fixed_indices_str = fixed_data
            else:
                fixed_indices_str = ""
            
            if fixed_indices_str:
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
                    if self.fixed_joint_indices:
                        print(f"Fixed joints for {robot_name} ({active_arm} arm): {len(self.fixed_joint_indices)} joints")
                except ValueError:
                    self.fixed_joint_indices = []
        
        # Active joints (exclude last joint (gripper) and fixed joints)
        base_active_joint_indices = list(range(self.num_joints - 1))
        self.active_joint_indices = [idx for idx in base_active_joint_indices if idx not in self.fixed_joint_indices]
        
        # Initialize IK solver to get joint names and end effector site
        try:
            all_joint_names = list(self.robot.robot_model.joints)
        except:
            all_joint_names = [self.env.sim.model.joint_id2name(idx) for idx in self.robot.joint_indexes]
        
        # Filter joint names to only include active joints
        # This ensures IK solver only considers active joints in dof_ids
        active_joint_names = [all_joint_names[i] for i in self.active_joint_indices if i < len(all_joint_names)]
        
        # Get end effector site name from gripper
        try:
            eef_site_name = self.robot.gripper["right"].important_sites["grip_site"]
        except:
            eef_site_name = "gripper0_right_grip_site"
        
        robot_config = {
            "joint_names": active_joint_names,  # Use only active joints
            "end_effector_sites": [eef_site_name],
            "nullspace_gains": [1.0] * len(active_joint_names),
        }
        
        # Initialize IK solver
        _debug_log("initializing IK solver")
        self.ik_solver = IKSolver(
            model=self.env.sim.model._model,
            data=self.env.sim.data._data,
            robot_config=robot_config,
            damping=0.05,
            integration_dt=1.0 / self.control_freq,
            max_dq=0.5,
            input_type="keyboard",
            debug=False,
            input_action_repr="absolute",
            input_rotation_repr="axis_angle"
        )
        _debug_log("IK solver initialized")
        
        # Convert active JOINT indices to MuJoCo DOF addresses for Jacobian indexing.
        # Robot joint indices (from robot.robot_model.joints) may NOT match MuJoCo
        # joint indices when the model has a free joint (e.g. GR1 floating body).
        # Use joint NAMES to bridge between the two index systems.
        mujoco_model = self.env.sim.model._model
        active_dof_addresses = []
        for j_idx in self.active_joint_indices:
            if j_idx < len(all_joint_names):
                jname = all_joint_names[j_idx]
                try:
                    mj_joint_id = mujoco_model.joint(jname).id
                    dof_addr = mujoco_model.jnt_dofadr[mj_joint_id]
                    active_dof_addresses.append(dof_addr)
                except Exception:
                    # Fallback: use robot joint index as DOF address (correct if no free joint)
                    active_dof_addresses.append(j_idx)
            else:
                active_dof_addresses.append(j_idx)
        self.active_dof_addresses = np.array(active_dof_addresses)
        self.ik_solver.dof_ids = self.active_dof_addresses
        
        # Check if DOF addresses differ from joint indices (indicates free joint)
        if not np.array_equal(self.active_dof_addresses, np.array(self.active_joint_indices)):
            print(f"Free joint detected: robot joint indices {self.active_joint_indices} → DOF addresses {list(self.active_dof_addresses)}")
        else:
            print(f"Active DOF ids (joint=DOF): {list(self.active_dof_addresses)}")
        
        self.eef_site_name = eef_site_name
        self.joint_names = active_joint_names  # Store active joint names
        self.all_joint_names = all_joint_names  # Store all joint names for reference
        
        print(f"Total joints: {self.num_joints}")
        print(f"Active joints: {len(self.active_joint_indices)}")
        print(f"Fixed joints: {len(self.fixed_joint_indices)}")
        print(f"End effector site: {eef_site_name}")
        print(f"Robot initialized successfully!")
        _debug_log("JacobianCalculator init complete")
    
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
        roll_deg: Optional[float] = None,
        pitch_deg: Optional[float] = None,
        yaw_deg: Optional[float] = None,
        robot_name: Optional[str] = None,
        tolerance: float = 30.0,
        dir_name: Optional[str] = None,
        pitch_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Find poses matching the given orientation criteria.
        """
        if robot_name is None:
            robot_name = self.robot_name
        
        matching_poses = []
        
        for pose in self.pose_database:
            if pose.get("robot") != robot_name:
                continue
            
            # 1. Check if we have high-level labels (dir, gripper_orientation)
            # This is much faster and matches our classification logic
            if dir_name is not None and pose.get("dir") != dir_name:
                continue
            if pitch_type is not None and pose.get("gripper_orientation") != pitch_type:
                continue

            # 2. Check by orientation (RPY)
            orientation = pose.get("orientation", {})
            pose_roll_deg = orientation.get("roll_deg")
            pose_pitch_deg = orientation.get("pitch_deg")
            pose_yaw_deg = orientation.get("yaw_deg")
            
            match = True
            if roll_deg is not None and pose_roll_deg is not None:
                roll_diff = abs(pose_roll_deg - roll_deg)
                roll_diff = min(roll_diff, 360 - roll_diff)
                if roll_diff > tolerance: match = False
            
            if pitch_deg is not None and pose_pitch_deg is not None:
                pitch_diff = abs(pose_pitch_deg - pitch_deg)
                pitch_diff = min(pitch_diff, 360 - pitch_diff)
                if pitch_diff > tolerance: match = False
            
            if yaw_deg is not None and pose_yaw_deg is not None:
                yaw_diff = abs(pose_yaw_deg - yaw_deg)
                yaw_diff = min(yaw_diff, 360 - yaw_diff)
                if yaw_diff > tolerance: match = False
            
            if match:
                matching_poses.append(pose)
        
        return matching_poses
    
    def _get_joint_positions(self):
        """Get current joint positions."""
        return self.robot._joint_positions.copy()
    
    def _capture_image(self, width: int = 512, height: int = 512):
        """Capture current camera view as numpy array."""
        obs = self.env.sim.render(
            camera_name="frontview",
            width=width,
            height=height,
            depth=False
        )
        return obs[::-1]
    
    def _set_pose_from_data(self, pose_data: Dict):
        """
        Set robot to a pose from pose data.
        
        Args:
            pose_data: Dictionary containing pose information with joint_angles_deg and joint_names
        """
        joint_angles_deg = pose_data["joint_angles_deg"]
        joint_angles_rad = pose_data["joint_angles_rad"]
        active_joint_indices = pose_data.get("active_joint_indices", [])
        
        # Reconstruct full joint position array
        joint_pos = self.initial_joint_pos.copy()
        
        # Set positions for active joints
        for i, active_joint_idx in enumerate(active_joint_indices):
            if i < len(joint_angles_rad):
                if active_joint_idx < len(joint_pos):
                    joint_pos[active_joint_idx] = joint_angles_rad[i]
                else:
                    print(f"Warning: active_joint_idx {active_joint_idx} >= len(joint_pos) {len(joint_pos)}")
        
        # Set joint positions
        self.robot.set_robot_joint_positions(joint_pos)
        self.env.sim.forward()
        
        # Stabilize
        for _ in range(10):
            self.env.sim.data.qvel[:] = 0
            self.env.sim.forward()
        
        print(f"Applied joint angles (deg): {joint_angles_deg}")
    
    def compute_and_print_jacobian(self, pose_name: str, pose_index: Optional[int] = None, axis: str = 'y'):
        """
        Set robot to initial pose and compute/print Jacobian matrices.
        
        Args:
            pose_name: Name of the pose from pose_set
            pose_index: Optional pose_id to use (if None, randomly selects)
            axis: 'x', 'y', or 'z' - the axis to optimize for
        """
        print(f"\n{'='*60}")
        print(f"Computing Jacobian for pose: {pose_name}")
        print(f"{'='*60}")
        
        # Get pose definition
        if pose_name not in pose_set:
            print(f"Error: Pose '{pose_name}' not found in pose_set")
            return
        
        pose_def = pose_set[pose_name]
        
        # Find matching poses
        matching_poses = self._find_matching_poses(
            roll_deg=pose_def.get('roll'),
            pitch_deg=pose_def.get('gripper_orientation'),
            yaw_deg=pose_def.get('yaw'),
        )
        
        if not matching_poses:
            print(f"Warning: No matching poses found for {pose_name}")
            return
        
        # Select pose
        if pose_index is not None:
            selected_pose = None
            for pose in matching_poses:
                if pose.get('pose_id') == pose_index:
                    selected_pose = pose
                    break
            
            if selected_pose is None:
                print(f"Warning: pose_id {pose_index} not found, using random selection")
                selected_pose = random.choice(matching_poses)
                pose_id = selected_pose['pose_id']
            else:
                pose_id = selected_pose['pose_id']
                print(f"Selected pose with pose_id {pose_id}: rank {selected_pose['rank']}")
        else:
            selected_pose = random.choice(matching_poses)
            pose_id = selected_pose['pose_id']
            print(f"Randomly selected pose with pose_id {pose_id}: rank {selected_pose['rank']}")
        
        # Move to initial pose
        print("\nMoving to initial pose...")
        self._set_pose_from_data(selected_pose)
        
        # Get MuJoCo model and data
        mujoco_model = self.env.sim.model._model
        mujoco_data = self.env.sim.data._data
        
        # Get end effector site ID
        site_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_SITE, self.eef_site_name)
        if site_id < 0:
            print(f"Error: Site '{self.eef_site_name}' not found")
            return
        
        print(f"\n{'='*60}")
        print("Computing Jacobian matrix for end effector")
        print(f"{'='*60}")
        
        # Compute Jacobian for end effector site (6xN: 3 for position, 3 for orientation)
        jac_pos = np.zeros((3, mujoco_model.nv))
        jac_rot = np.zeros((3, mujoco_model.nv))
        mujoco.mj_jacSite(mujoco_model, mujoco_data, jac_pos, jac_rot, site_id)
        
        # Combine position and rotation Jacobians (6xN)
        jac_full = np.vstack([jac_pos, jac_rot])
        
        # Get DOF indices that IK solver uses
        dof_ids = self.ik_solver.dof_ids
        jac_subset = jac_full[:, dof_ids]
        
        print(f"\nFull Jacobian shape: {jac_full.shape} (6 rows: 3 pos + 3 rot, {mujoco_model.nv} columns: all DOFs)")
        print(f"Subset Jacobian shape: {jac_subset.shape} (6 rows: 3 pos + 3 rot, {len(dof_ids)} columns: IK-controlled DOFs)")
        
        # Print Jacobian for each joint
        print(f"\n{'='*60}")
        print("Jacobian matrix for each joint (column of the full Jacobian)")
        print(f"{'='*60}\n")
        
        # Build joint names from active_joint_indices (not DOF addresses, which may differ for free joints)
        joint_names_list = [
            self.all_joint_names[j] if j < len(self.all_joint_names) else f"Joint_{j}"
            for j in self.active_joint_indices
        ]
        
        for i, (dof_id, jname) in enumerate(zip(dof_ids, joint_names_list)):
            jac_column = jac_subset[:, i]
            print(f"Joint {i+1}/{len(dof_ids)}: {jname} (DOF addr: {dof_id})")
            print(f"  Position Jacobian (3x1):")
            print(f"    {jac_column[0:3]}")
            print(f"  Rotation Jacobian (3x1):")
            print(f"    {jac_column[3:6]}")
            print()
        
        # Print full Jacobian matrix
        print(f"\n{'='*60}")
        print("Full Jacobian Matrix (6x{})".format(len(dof_ids)))
        print("Rows: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]")
        print("Columns: Joints (in order of DOF addresses)")
        print(f"{'='*60}")
        print("\nPosition Jacobian (3x{}):".format(len(dof_ids)))
        print(jac_subset[0:3, :])
        print("\nRotation Jacobian (3x{}):".format(len(dof_ids)))
        print(jac_subset[3:6, :])
        print(f"\n{'='*60}\n")
        
        print("Joint names (in order of columns):")
        print(joint_names_list)
        print()
        
        # Find joints sorted by score for the specified axis
        sorted_joints = self._find_and_sort_joints_for_axis(jac_subset, dof_ids, joint_names_list, axis=axis)
        
        # Create GIF for all joints (sorted by rank) only if save_jacobian_gif is True
        if self.save_jacobian_gif:
            print(f"\n{'='*60}")
            print(f"Creating GIFs for all {len(sorted_joints)} joints (sorted by rank)")
            print(f"{'='*60}\n")
            
            for rank, (joint_idx, joint_name, joint_dof_id, score) in enumerate(sorted_joints, start=1):
                print(f"\n--- Rank {rank}/{len(sorted_joints)}: {joint_name} (score: {score:.6f}) ---")
                self.create_gif_for_joint(
                    joint_idx_in_dof_list=joint_idx,
                    joint_dof_id=joint_dof_id,
                    joint_name=joint_name,
                    pose_name=pose_name,
                    pose_id=pose_id,
                    rank=rank,
                    angles_deg=[-30, 0, 30]
                )
    
    def _find_and_sort_joints_for_axis(self, jac_subset, dof_ids, joint_names_list, axis='y'):
        """
        Find and sort all joints by their score for movement along the specified axis.
        
        We want joints that:
        1. Have strong contribution to the specified axis movement
        2. Have minimal contribution to the other two axes (to maintain the plane)
        
        Args:
            jac_subset: Jacobian matrix (6 x num_joints)
            dof_ids: DOF indices
            joint_names_list: List of joint names
            axis: 'x', 'y', or 'z' - the axis to optimize for (Panda-style standard)
        
        Returns:
            List of tuples (joint_idx, joint_name, joint_dof_id, score) sorted by score (highest first)
        """
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_names = {'x': 'X-axis (forward/backward)', 'y': 'Y-axis (left/right)', 'z': 'Z-axis (up/down)'}
        
        if axis not in axis_map:
            print(f"Warning: Invalid axis '{axis}', using 'y'")
            axis = 'y'
        
        axis_idx = axis_map[axis]
        axis_name = axis_names[axis]
        
        print(f"\n{'='*60}")
        print(f"Finding and sorting joints for {axis_name} movement")
        print(f"{'='*60}\n")
        
        # Position Jacobian: rows 0-2 are [x, y, z]
        pos_jac = jac_subset[0:3, :]  # 3 x num_joints
        
        # Extract the target axis row
        target_axis_jac = pos_jac[axis_idx, :]  # 1 x num_joints
        
        # Calculate contributions for the other two axes (we want these to be small)
        other_axes_idx = [i for i in range(3) if i != axis_idx]
        other_axis1_jac = pos_jac[other_axes_idx[0], :]
        other_axis2_jac = pos_jac[other_axes_idx[1], :]
        
        # Score for each joint: target_axis_contribution^2 / (total_magnitude^2 + epsilon)
        # This represents the "dominance" or "alignment" of the joint with the target axis.
        # Squaring the ratio heavily penalizes joints that move in multiple axes.
        epsilon = 1e-6
        total_magnitude_sq = np.sum(pos_jac**2, axis=0) + epsilon
        scores = (target_axis_jac**2) / total_magnitude_sq
        
        # Create list of (joint_idx, joint_name, joint_dof_id, score) tuples
        joint_scores = []
        for i in range(len(dof_ids)):
            joint_scores.append((
                i,  # joint_idx
                joint_names_list[i],  # joint_name
                dof_ids[i],  # joint_dof_id
                scores[i]  # score
            ))
        
        # Sort by score (highest first)
        joint_scores.sort(key=lambda x: x[3], reverse=True)
        
        # Print analysis
        axis_labels = ['X-axis', 'Y-axis', 'Z-axis']
        other_labels = [axis_labels[other_axes_idx[0]], axis_labels[other_axes_idx[1]]]
        
        print("Analysis for each joint (sorted by rank):")
        print(f"{'Rank':<6} {'Joint Name':<30} {axis_labels[axis_idx]:>12} {other_labels[0]:>12} {other_labels[1]:>12} {'Score':>12}")
        print("-" * 90)
        
        for rank, (joint_idx, joint_name, joint_dof_id, score) in enumerate(joint_scores, start=1):
            print(f"{rank:<6} {joint_name:<30} {target_axis_jac[joint_idx]:>12.6f} {other_axis1_jac[joint_idx]:>12.6f} {other_axis2_jac[joint_idx]:>12.6f} {score:>12.6f}")
        
        print(f"\n{'='*60}")
        print(f"Sorted {len(joint_scores)} joints for {axis_name} movement")
        print(f"  Best joint: {joint_scores[0][1]} (score: {joint_scores[0][3]:.6f})")
        print(f"  Worst joint: {joint_scores[-1][1]} (score: {joint_scores[-1][3]:.6f})")
        print(f"{'='*60}\n")
        
        return joint_scores
    
    def create_gif_for_joint(self, joint_idx_in_dof_list: int, joint_dof_id: int, joint_name: str, pose_name: str, pose_id: int, rank: int, angles_deg: List[float] = [-30, 0, 30]):
        """
        Create GIF by rotating a joint at specified angles.
        
        Args:
            joint_idx_in_dof_list: Index of the joint in the DOF list (from IK solver)
            joint_dof_id: DOF ID of the joint in mujoco
            joint_name: Name of the joint
            pose_name: Name of the pose
            pose_id: ID of the pose
            rank: Rank of this joint (1 = best, 2 = second best, etc.)
            angles_deg: List of angles in degrees to rotate the joint
        """
        print(f"Creating GIF for joint: {joint_name} (DOF ID: {joint_dof_id}, Rank: {rank})")
        
        # Get initial joint positions (this is the pose after _set_pose_from_data)
        initial_joint_pos = self._get_joint_positions()
        
        frames = []
        
        # For each angle, set joint position and capture frame
        for angle_deg in angles_deg:
            # Calculate new joint position
            joint_pos = initial_joint_pos.copy()
            angle_rad = np.deg2rad(angle_deg)
            
            # joint_dof_id is the mujoco DOF ID (qpos address)
            # robot._joint_positions array corresponds to robot._ref_joint_pos_indexes
            # We need to find the index in _ref_joint_pos_indexes that matches joint_dof_id
            # Then use that index to modify joint_pos
            
            joint_idx = None
            # Find the index in robot's joint position array that corresponds to this DOF ID
            if hasattr(self.robot, '_ref_joint_pos_indexes'):
                for i, qpos_addr in enumerate(self.robot._ref_joint_pos_indexes):
                    # qpos_addr can be a single int or a tuple (for multi-DOF joints)
                    if isinstance(qpos_addr, (int, np.integer)):
                        if qpos_addr == joint_dof_id:
                            joint_idx = i
                            break
                    elif isinstance(qpos_addr, (tuple, list)):
                        # For multi-DOF joints, tuple is (start, end)
                        start_addr = qpos_addr[0] if isinstance(qpos_addr, tuple) else qpos_addr
                        if start_addr <= joint_dof_id < start_addr + len(qpos_addr):
                            joint_idx = i
                            break
            
            if joint_idx is None:
                print(f"  Warning: Could not find joint index for DOF ID {joint_dof_id}")
                if hasattr(self.robot, '_ref_joint_pos_indexes'):
                    print(f"    Available DOF IDs in _ref_joint_pos_indexes (first 10): {self.robot._ref_joint_pos_indexes[:10]}")
                    print(f"    Total joints: {len(self.robot._ref_joint_pos_indexes)}")
                continue
            
            # Set new joint position (add angle to initial position)
            joint_pos[joint_idx] = initial_joint_pos[joint_idx] + angle_rad
            
            # Apply joint positions
            self.robot.set_robot_joint_positions(joint_pos)
            
            # Stabilize
            for _ in range(10):
                self.env.sim.data.qvel[:] = 0
                self.env.sim.forward()
            
            # Capture frame
            image = self._capture_image()
            frames.append(Image.fromarray(image))
        
        # Save GIF
        if len(frames) > 0:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_joint_name = joint_name.replace('/', '_').replace(' ', '_')
            output_filename = f"{now}_{self.robot_name}_{pose_name}_p{pose_id}_rank{rank:02d}_joint{joint_dof_id}_{safe_joint_name}.gif"
            filepath = os.path.join(self.output_dir, output_filename)
            
            frames[0].save(
                filepath,
                save_all=True,
                append_images=frames[1:] if len(frames) > 1 else [],
                duration=500,  # 500ms per frame
                loop=0  # Infinite loop
            )
            print(f"  Saved GIF: {filepath}")
        else:
            print(f"  Error: No frames captured!")
    
    def close(self):
        """Close the environment."""
        self.env.close()


def main(
    robot: str = "Panda",
    pose_name: str = "Elbow_down",
    pose_index: Optional[int] = None,
    axis: str = "y",
    env: str = "EmptySpace",
    controller: str = "IK_POSE",
    jsonl_path: str = "data/seed/closest_poses_results.jsonl",
    save_jacobian_gif: bool = False,
):
    """
    Main function to compute and print Jacobian matrices.
    
    Args:
        robot: Robot name
        pose_name: Name of the pose from pose_set
        pose_index: Optional pose_id to use (if None, randomly selects)
        axis: 'x', 'y', or 'z' - the axis to optimize for (default: 'y' for left-right)
        env: Environment name
        controller: Controller name (should support IK)
        jsonl_path: Path to pose database JSONL file
        save_jacobian_gif: Whether to save GIF files for joint movements (default: False)
    """
    
    # Initialize calculator
    calculator = JacobianCalculator(
        robot_name=robot,
        env_name=env,
        controller_name=controller,
        jsonl_path=jsonl_path,
        has_renderer=False,
        has_offscreen_renderer=True,
        save_jacobian_gif=save_jacobian_gif,
    )
    
    try:
        # Compute and print Jacobian
        calculator.compute_and_print_jacobian(
            pose_name=pose_name,
            pose_index=pose_index,
            axis=axis,
        )
    finally:
        calculator.close()


if __name__ == "__main__":
    fire.Fire(main)
