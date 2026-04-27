"""
Generate robot motions based on cue definitions.

This script:
1. Loads pose definitions and finds matching poses from JSONL
2. Uses JacobianCalculator to find best joints for specified axis
3. Selects joint based on proximal/distal preference
4. Executes movements and generates GIF
"""

import fire
import os
import copy
import json
import random
import tempfile
import numpy as np
from typing import Dict, List, Optional
from PIL import Image, ImageDraw
from datetime import datetime

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.utils.ik_utils import IKSolver
import mujoco

# Import JacobianCalculator and pose configuration
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alphabet_jacobian import JacobianCalculator
from arm_pose_config import direction_pose_set, pose_set, poses, pitch_poses, height_map

# Fixed joint indices
FIXED_JOINT_INDICES = {
    'GR1': "0-2, 20-31"
}


STEP_PROGRESS_COLORS = {
    "pose": (88, 166, 255, 255),
    "movement": (63, 185, 80, 255),
    "path": (188, 140, 255, 255),
}

STEP_PROGRESS_BG = {
    "pose": (88, 166, 255, 64),
    "movement": (63, 185, 80, 64),
    "path": (188, 140, 255, 64),
}


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _orthogonal_axis(axis: str) -> str:
    return {"x": "y", "y": "z", "z": "x"}.get(axis, "y")


PATH_SPEED_DEG_PER_SEC = 30.0


def _path_duration_from_length(path_length_deg: float, speed: float) -> float:
    """Map path length and speed to a shared duration across line / arc paths."""
    effective_speed = max(0.1, float(speed))
    effective_length = max(1.0, float(path_length_deg))
    return effective_length / (effective_speed * PATH_SPEED_DEG_PER_SEC)


def _normalize_path_axis_value(axis: str, value: float, sign_multiplier: float) -> float:
    """
    Normalize path-axis values to match the desired visible sign convention.

    `sign_multiplier` is computed per-axis and may depend on the selected joint,
    robot, and current pose.
    """
    return float(value) * float(sign_multiplier)


def _swap_gripper_orientation_for_robot(robot_name: Optional[str], orientation: Optional[str]) -> Optional[str]:
    # Experimental compatibility shim: Panda currently looks visually inverted
    # relative to the shared vertical / horizontal pose labels, so swap only the
    # lookup label for Panda when querying the pose database.
    if orientation not in {"vertical", "horizontal"}:
        return orientation
    if str(robot_name) != "Panda":
        return orientation
    return "horizontal" if orientation == "vertical" else "vertical"


def _extract_render_modifiers(cue_config_data: Dict) -> Dict[str, float]:
    mods = cue_config_data.get("render_modifiers", {}) or {}
    return {
        "hesitation": _clamp(float(mods.get("hesitation", 0.0)), 0.0, 1.0),
        "elegant_curve": _clamp(float(mods.get("elegant_curve", 0.0)), 0.0, 1.0),
        "zittering": _clamp(float(mods.get("zittering", 0.0)), 0.0, 1.0),
    }


def _apply_style_offsets(
    joint_positions,
    *,
    t: float,
    primary_joint_idx: int | None,
    secondary_joint_idx: int | None,
    elegant_curve: float,
    zittering: float,
):
    if secondary_joint_idx is not None and elegant_curve > 0:
        curve_amp = np.deg2rad(4.0 + 8.0 * elegant_curve)
        joint_positions[secondary_joint_idx] += np.sin(np.pi * t) * curve_amp

    if primary_joint_idx is not None and zittering > 0:
        jitter_amp = np.deg2rad(1.5 + 4.5 * zittering)
        jitter_freq = 2.0 + 3.0 * zittering
        joint_positions[primary_joint_idx] += np.sin(2 * np.pi * jitter_freq * t) * jitter_amp

    if secondary_joint_idx is not None and zittering > 0:
        side_amp = np.deg2rad(1.0 + 2.5 * zittering)
        side_freq = 1.5 + 2.0 * zittering
        joint_positions[secondary_joint_idx] += np.cos(2 * np.pi * side_freq * t) * side_amp


def _overlay_progress_bar_on_frames(frames: List[Image.Image], step_spans: List[Dict]) -> List[Image.Image]:
    """Overlay a cumulative step progress bar directly onto every frame."""
    if not frames or not step_spans:
        return frames

    total_frames = len(frames)
    width, height = frames[0].size
    outer_pad = max(8, width // 48)
    bar_height = max(12, height // 36)
    overlay_height = bar_height + outer_pad * 2
    bar_left = outer_pad
    bar_top = outer_pad
    bar_width = width - outer_pad * 2
    bar_right = bar_left + bar_width
    bar_bottom = bar_top + bar_height
    marker_width = max(2, width // 180)

    overlaid = []
    for frame_idx, frame in enumerate(frames):
        base = frame.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        draw.rounded_rectangle(
            [bar_left - 4, bar_top - 4, bar_right + 4, bar_bottom + 4],
            radius=bar_height // 2 + 4,
            fill=(8, 12, 18, 170),
            outline=(230, 237, 243, 40),
            width=1,
        )

        for span in step_spans:
            start = span["start"]
            end = span["end"]
            if end <= start:
                continue
            step_type = span["type"]
            bg_color = STEP_PROGRESS_BG.get(step_type, (139, 148, 158, 64))
            fg_color = STEP_PROGRESS_COLORS.get(step_type, (139, 148, 158, 255))

            x0 = bar_left + int(round((start / total_frames) * bar_width))
            x1 = bar_left + int(round((end / total_frames) * bar_width))
            if x1 <= x0:
                x1 = min(bar_right, x0 + 1)

            draw.rounded_rectangle(
                [x0, bar_top, x1, bar_bottom],
                radius=bar_height // 2,
                fill=bg_color,
            )

            progress_end = min(frame_idx + 1, end)
            if progress_end > start:
                fill_x1 = bar_left + int(round((progress_end / total_frames) * bar_width))
                if fill_x1 <= x0:
                    fill_x1 = min(bar_right, x0 + 1)
                draw.rounded_rectangle(
                    [x0, bar_top, fill_x1, bar_bottom],
                    radius=bar_height // 2,
                    fill=fg_color,
                )

        progress_x = bar_left + int(round(((frame_idx + 1) / total_frames) * bar_width))
        progress_x = max(bar_left, min(bar_right, progress_x))
        draw.rectangle(
            [progress_x - marker_width, bar_top - 2, progress_x + marker_width, bar_bottom + 2],
            fill=(255, 255, 255, 220),
        )

        draw.rounded_rectangle(
            [bar_left, bar_top, bar_right, bar_bottom],
            radius=bar_height // 2,
            outline=(230, 237, 243, 120),
            width=1,
        )

        overlaid.append(Image.alpha_composite(base, overlay).convert("RGB"))

    return overlaid


def _overlay_simple_progress_bar_on_frames(frames: List[Image.Image]) -> List[Image.Image]:
    """Overlay a simple neutral playback progress bar without step-type hints."""
    if not frames:
        return frames

    total_frames = len(frames)
    width, height = frames[0].size
    outer_pad = max(8, width // 48)
    bar_height = max(10, height // 42)
    bar_left = outer_pad
    bar_top = outer_pad
    bar_width = width - outer_pad * 2
    bar_right = bar_left + bar_width
    bar_bottom = bar_top + bar_height

    overlaid = []
    for frame_idx, frame in enumerate(frames):
        base = frame.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        draw.rounded_rectangle(
            [bar_left - 3, bar_top - 3, bar_right + 3, bar_bottom + 3],
            radius=bar_height // 2 + 3,
            fill=(12, 16, 22, 110),
        )
        draw.rounded_rectangle(
            [bar_left, bar_top, bar_right, bar_bottom],
            radius=bar_height // 2,
            fill=(235, 239, 244, 180),
        )

        progress = (frame_idx + 1) / max(1, total_frames)
        fill_x1 = bar_left + int(round(progress * bar_width))
        fill_x1 = min(max(fill_x1, bar_left + 1), bar_right)
        draw.rounded_rectangle(
            [bar_left, bar_top, fill_x1, bar_bottom],
            radius=bar_height // 2,
            fill=(30, 30, 30, 230),
        )

        composed = Image.alpha_composite(base, overlay).convert("RGBA")
        overlaid.append(composed)

    return overlaid


def _debug_log(msg: str) -> None:
    print(f"[motion_debug] {msg}", flush=True)


class MotionGenerator:
    """Generate robot motions based on cue definitions."""
    
    def __init__(
        self,
        robot_name: str = "Panda",
        env_name: str = "EmptySpace",
        controller_name: str = "IK_POSE",
        jsonl_path: str = "data/seed/_remainder/closest_poses_results.jsonl",
        has_renderer: bool = False,
        has_offscreen_renderer: bool = True,
        control_freq: int = 20,
        output_dir: str = "data/results/render/manipulator",
        capture_image_width: int = 512,
        capture_image_height: int = 512,
        camera_distance: float = 1.8,
        hz: int = 4,
    ):
        """
        Initialize the motion generator.
        
        Args:
            robot_name: Name of the robot
            env_name: Name of the environment
            controller_name: Name of the controller
            jsonl_path: Path to JSONL file with pose data
            has_renderer: Whether to show on-screen rendering
            has_offscreen_renderer: Whether to enable offscreen rendering
            control_freq: Control frequency
            output_dir: Output directory for GIFs
            capture_image_width: Image width for capture
            capture_image_height: Image height for capture
            camera_distance: Multiplier for camera FOV to zoom out (1.8 = default, >1.0 = wider view)
            hz: Frame rate for GIF generation (frames per second)
        """
        self.robot_name = robot_name
        self.env_name = env_name
        self.controller_name = controller_name
        self.jsonl_path = jsonl_path
        self.control_freq = control_freq
        self.output_dir = os.path.join(output_dir, robot_name)
        self.capture_image_width = capture_image_width
        self.capture_image_height = capture_image_height
        self.camera_distance = camera_distance
        self.hz = hz
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        _debug_log("creating JacobianCalculator")
        
        # Initialize JacobianCalculator for joint analysis
        self.jacobian_calculator = JacobianCalculator(
            robot_name=robot_name,
            env_name=env_name,
            controller_name=controller_name,
            jsonl_path=jsonl_path,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            control_freq=control_freq,
        )
        
        # Get environment and robot from calculator
        self.env = self.jacobian_calculator.env
        self.robot = self.jacobian_calculator.robot
        self.initial_joint_pos = self.jacobian_calculator.initial_joint_pos
        _debug_log("JacobianCalculator ready; env and robot acquired")
        
        # Adjust camera FOV to zoom out (instead of moving position)
        # Larger FOV = wider view = can see more
        camera_name = "frontview"
        try:
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            # Get current FOV (default is usually around 45-55 degrees)
            current_fov = self.env.sim.model.cam_fovy[cam_id]
            # Scale FOV: camera_distance > 1.0 means zoom out (larger FOV)
            new_fov = current_fov * camera_distance
            # Clamp to reasonable range (20-120 degrees) - increased max for upward pointing robots
            new_fov = max(20.0, min(120.0, new_fov))
            self.env.sim.model.cam_fovy[cam_id] = new_fov
            print(f"Camera FOV adjusted: {current_fov:.1f}° -> {new_fov:.1f}° (zoom factor: {camera_distance})")
        except Exception as e:
            print(f"Warning: Could not adjust camera FOV: {e}")
        
        print(f"Motion generator initialized for {robot_name}")
        _debug_log("MotionGenerator init complete")
    
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
            # Old format: pose name as string
            pose_name = pose_def
            if pose_name not in direction_pose_set:
                print(f"Error: Pose '{pose_name}' not found in direction_pose_set")
                return []
            pose_def = direction_pose_set[pose_name]
        elif isinstance(pose_def, dict):
            # New format: pose definition as dict with height/dir/pitch
            pass
        else:
            print(f"Error: Invalid pose format: {type(pose_def)}")
            return []
        
        direction_name = pose_def.get('dir')
        grip_orient = pose_def.get('gripper_orientation') or pose_def.get('pitch')
        if not direction_name:
            print(f"Error: Pose definition missing 'dir' field: {pose_def}")
            return []
        
        if grip_orient:
            orientations_to_try = [_swap_gripper_orientation_for_robot(robot_name, grip_orient)]
        else:
            orientations_to_try = [
                _swap_gripper_orientation_for_robot(robot_name, 'vertical'),
                _swap_gripper_orientation_for_robot(robot_name, 'horizontal'),
            ]
            print(f"  gripper_orientation not specified, searching both vertical and horizontal")
        
        all_matching_poses = []
        for orient in orientations_to_try:
            poses = self.jacobian_calculator._find_matching_poses(
                robot_name=robot_name,
                dir_name=direction_name,
                pitch_type=orient,
            )
            all_matching_poses.extend(poses)
        
        orient_label = grip_orient or 'vertical+horizontal'
        query_label = orientations_to_try[0] if len(orientations_to_try) == 1 else "+".join(orientations_to_try)
        print(
            f"  Searching for poses: dir='{direction_name}', "
            f"gripper_orientation='{orient_label}'"
            + (f" (query='{query_label}')" if query_label != orient_label else "")
            + f" → {len(all_matching_poses)} found"
        )
        
        if not all_matching_poses:
            print(f"  Warning: No poses found in JSONL for robot={robot_name}, dir={direction_name}, gripper_orientation={orient_label}")
        
        return all_matching_poses
    
    def _select_joint(
        self,
        axis: str,
        joint_preference: str,  # 'proximal', 'distal', 'shoulder', 'elbow', 'wrist'
        score_threshold: float = 0.1,
        pose_def=None,  # Unused, kept for backward compatibility
        selection_mode: str = "default",
    ) -> tuple:
        """
        Select a joint based on axis and joint preference.
        Computes Jacobian at the current robot pose (sim state).
        
        Joint preference determines which portion of the kinematic chain to pick from:
          - 'proximal' / 'distal': 1/2 split by DOF ID
          - 'shoulder' / 'elbow' / 'wrist': 1/3 split by DOF ID
        
        Args:
            axis: 'x', 'y', or 'z'
            joint_preference: 'proximal', 'distal', 'shoulder', 'elbow', or 'wrist'
            score_threshold: Minimum score threshold for joint selection
        
        Returns:
            Tuple of (joint_idx, joint_name, joint_dof_id, score, jac_sign)
        """
        # Compute Jacobian at the current pose
        mujoco_model = self.env.sim.model._model
        mujoco_data = self.env.sim.data._data
        site_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_SITE, self.jacobian_calculator.eef_site_name)

        jac_pos = np.zeros((3, mujoco_model.nv))
        jac_rot = np.zeros((3, mujoco_model.nv))
        mujoco.mj_jacSite(mujoco_model, mujoco_data, jac_pos, jac_rot, site_id)
        jac_full = np.vstack([jac_pos, jac_rot])
        dof_ids = self.jacobian_calculator.ik_solver.dof_ids
        jac_subset = jac_full[:, dof_ids]

        # Get joint names
        joint_names_list = []
        for dof_id in dof_ids:
            if dof_id < len(self.jacobian_calculator.joint_names):
                joint_names_list.append(self.jacobian_calculator.joint_names[dof_id])
            else:
                joint_names_list.append(f"DOF_{dof_id}")

        # Get sorted joints by axis score
        sorted_joints = self.jacobian_calculator._find_and_sort_joints_for_axis(
            jac_subset, dof_ids, joint_names_list, axis=axis
        )

        # Filter joints by score threshold
        filtered_joints = [
            joint for joint in sorted_joints
            if joint[3] >= score_threshold
        ]

        if not filtered_joints:
            print(f"  Warning: No joints above threshold {score_threshold}, using best available")
            if not sorted_joints:
                raise ValueError(f"No joints available for axis {axis}")
            filtered_joints = sorted_joints[:1]

        # Filter out distal joints that mainly contribute to roll rotation
        rot_jac = jac_subset[3:6, :]
        roll_jac = rot_jac[0, :]

        roll_threshold = 0.5
        high_roll_joints = set()

        sorted_by_dof = sorted(enumerate(dof_ids), key=lambda x: x[1], reverse=True)
        num_distal_roll = max(1, len(dof_ids) // 3)

        for i, dof_id in sorted_by_dof[:num_distal_roll]:
            if abs(roll_jac[i]) > roll_threshold:
                high_roll_joints.add(dof_id)

        roll_filtered = [
            joint for joint in filtered_joints
            if joint[2] not in high_roll_joints
        ]

        if not roll_filtered:
            roll_filtered = filtered_joints

        # Select joint based on preference
        sorted_by_dof_id = sorted(roll_filtered, key=lambda x: x[2])
        n = len(sorted_by_dof_id)

        if joint_preference in ('shoulder', 'elbow', 'wrist'):
            t1 = max(1, n // 3)
            t2 = max(t1 + 1, (2 * n + 2) // 3)
            groups = {
                'shoulder': sorted_by_dof_id[:t1],
                'elbow': sorted_by_dof_id[t1:t2],
                'wrist': sorted_by_dof_id[t2:],
            }
            candidate_joints = groups[joint_preference]
            if not candidate_joints:
                print(f"  Warning: No joints in '{joint_preference}' group, falling back to best overall")
                candidate_joints = roll_filtered
            selected = max(candidate_joints, key=lambda x: x[3])
        elif joint_preference == 'proximal':
            mid = n // 2
            proximal_half = sorted_by_dof_id[:mid] or sorted_by_dof_id
            selected = max(proximal_half, key=lambda x: x[3])
        elif joint_preference == 'distal':
            mid = n // 2
            distal_half = sorted_by_dof_id[mid:] or sorted_by_dof_id
            selected = max(distal_half, key=lambda x: x[3])
        else:
            selected = roll_filtered[0]

        if selection_mode == "path" and axis == "x":
            axis_idx = 0
            refined_pool = candidate_joints if joint_preference in ('shoulder', 'elbow', 'wrist') else roll_filtered
            refined_pool = refined_pool or roll_filtered

            def _path_x_key(joint):
                joint_idx = joint[0]
                x_abs = abs(float(jac_subset[axis_idx, joint_idx]))
                y_abs = abs(float(jac_subset[1, joint_idx]))
                z_abs = abs(float(jac_subset[2, joint_idx]))
                off_axis = y_abs + z_abs
                purity = x_abs / (off_axis + 1e-6)
                strong_enough = 1.0 if x_abs >= 0.05 else 0.0
                return (strong_enough, purity, x_abs, -off_axis, joint[3])

            selected = max(refined_pool, key=_path_x_key)

        # Get axis Jacobian sign for the selected joint
        jac_subset_for_axis = jac_subset[0:3, :]
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_idx = axis_map.get(axis, 1)
        selected_joint_jac_value = jac_subset_for_axis[axis_idx, selected[0]]

        print(f"\nSelected joint: {selected[1]} (DOF ID: {selected[2]}, Score: {selected[3]:.4f}, Rank: {sorted_joints.index(selected) + 1})")
        print(f"Joint preference: {joint_preference}")
        print(f"Jacobian {axis.upper()}-axis value for selected joint: {selected_joint_jac_value:.6f}")

        return selected + (np.sign(selected_joint_jac_value),)
    
    def _find_closest_pose(
        self,
        current_joint_pos: np.ndarray,
        candidate_poses: List[Dict],
        pose_def: Optional[Dict] = None,
    ) -> tuple[Dict, Dict]:
        """
        Find the candidate pose that is closest to the current joint state.
        
        Args:
            current_joint_pos: Actual current joint positions from the simulator
            candidate_poses: List of candidate pose dictionaries
            pose_def: Optional target pose definition; x/y/z percentiles are used when provided
        
        Returns:
            The closest candidate pose dictionary and score details
        """
        XYZ_THRESHOLD_PCT = 18.0
        best_pose = None
        best_metrics = None
        candidate_metrics = []

        joint_count = len(current_joint_pos)
        proximal_count = max(1, joint_count // 3)
        # Heavier penalty on earlier / root-side joints.
        weighted_profile = np.linspace(2.5, 1.0, joint_count)
        
        for candidate_pose in candidate_poses:
            candidate_joint_pos = self._pose_data_to_joint_positions(candidate_pose)
            joint_delta_deg = np.abs(np.rad2deg(current_joint_pos - candidate_joint_pos))
            joint_score = float(np.sum(joint_delta_deg))
            weighted_joint_score = float(np.sum(joint_delta_deg * weighted_profile))
            proximal_score = float(np.sum(joint_delta_deg[:proximal_count]))

            xyz_axes = []
            xyz_distance = 0.0
            if isinstance(pose_def, dict):
                for axis in ("x", "y", "z"):
                    target_val = pose_def.get(axis)
                    if isinstance(target_val, (int, float)):
                        xyz_axes.append(axis)
                        pose_val = candidate_pose.get(f"{axis}_pct", 50)
                        xyz_distance += (float(pose_val) - float(target_val)) ** 2
            xyz_distance = xyz_distance ** 0.5 if xyz_axes else 0.0

            within_xyz_threshold = bool(xyz_axes) and xyz_distance <= XYZ_THRESHOLD_PCT

            if within_xyz_threshold:
                selection_key = (
                    proximal_score,
                    weighted_joint_score,
                    joint_score,
                    xyz_distance,
                    candidate_pose.get("pose_id", 0),
                )
                selection_mode = "xyz_threshold_then_joint"
            elif xyz_axes:
                selection_key = (
                    xyz_distance,
                    proximal_score,
                    weighted_joint_score,
                    joint_score,
                    candidate_pose.get("pose_id", 0),
                )
                selection_mode = "xyz_fallback_then_joint"
            else:
                selection_key = (
                    proximal_score,
                    weighted_joint_score,
                    joint_score,
                    candidate_pose.get("pose_id", 0),
                )
                selection_mode = "joint_only"

            metrics = {
                "combined_score": weighted_joint_score if within_xyz_threshold or not xyz_axes else xyz_distance,
                "joint_score": joint_score,
                "weighted_joint_score": weighted_joint_score,
                "proximal_score": proximal_score,
                "xyz_distance": xyz_distance,
                "xyz_axes": xyz_axes,
                "xyz_threshold_pct": XYZ_THRESHOLD_PCT,
                "within_xyz_threshold": within_xyz_threshold,
                "selection_key": selection_key,
                "selection_mode": selection_mode,
            }
            candidate_metrics.append((candidate_pose, metrics))

        threshold_candidates = [
            (pose, metrics)
            for pose, metrics in candidate_metrics
            if metrics["within_xyz_threshold"]
        ]

        search_pool = threshold_candidates if threshold_candidates else candidate_metrics
        for candidate_pose, metrics in search_pool:
            if best_metrics is None or metrics["selection_key"] < best_metrics["selection_key"]:
                best_pose = candidate_pose
                best_metrics = metrics

        if best_pose is None:
            fallback = candidate_poses[0]
            return fallback, {
                "combined_score": float("inf"),
                "joint_score": float("inf"),
                "weighted_joint_score": float("inf"),
                "proximal_score": float("inf"),
                "xyz_distance": float("inf"),
                "xyz_axes": [],
                "xyz_threshold_pct": XYZ_THRESHOLD_PCT,
                "within_xyz_threshold": False,
                "selection_key": (float("inf"), float("inf"), float("inf")),
                "selection_mode": "joint_only",
            }

        return best_pose, best_metrics
    
    def _pose_data_to_joint_positions(self, pose_data: Dict) -> np.ndarray:
        """
        Convert pose data to joint positions array.
        
        Args:
            pose_data: Dictionary containing pose information with joint_angles_rad and active_joint_indices
        
        Returns:
            Joint positions array in radians
        """
        joint_angles_rad = pose_data.get("joint_angles_rad", [])
        active_joint_indices = pose_data.get("active_joint_indices", [])
        
        # Start with initial joint positions
        joint_pos = self.initial_joint_pos.copy()
        
        # Set positions for active joints
        for i, active_joint_idx in enumerate(active_joint_indices):
            if i < len(joint_angles_rad):
                if active_joint_idx < len(joint_pos):
                    joint_pos[active_joint_idx] = joint_angles_rad[i]
        
        return joint_pos
    
    def _get_joint_positions(self):
        """Get current joint positions."""
        return self.robot._joint_positions.copy()
    
    def _set_joint_positions(self, joint_positions):
        """Set joint positions."""
        self.robot.set_robot_joint_positions(joint_positions)
        self.env.sim.forward()
        
        # Stabilize
        for _ in range(10):
            self.env.sim.data.qvel[:] = 0
            self.env.sim.forward()
    
    def _capture_image(self):
        """Capture current camera view."""
        _debug_log("capture_image:start")
        obs = self.env.sim.render(
            camera_name="frontview",
            width=self.capture_image_width,
            height=self.capture_image_height,
            depth=False
        )
        _debug_log("capture_image:done")
        return obs[::-1]

    def _get_eef_position(self):
        """Get current end-effector site position in world coordinates."""
        mujoco_model = self.env.sim.model._model
        mujoco_data = self.env.sim.data._data
        site_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_SITE, self.jacobian_calculator.eef_site_name)
        return np.array(mujoco_data.site_xpos[site_id], dtype=float).copy()

    def _path_axis_sign_multiplier(
        self,
        *,
        axis: str,
        robot_joint_idx: int,
        current_joint_pos,
        jac_sign: float,
        probe_deg: float = 2.0,
    ) -> float:
        """
        Decide the authored-sign multiplier for a path axis.

        For x-paths we probe the actual end-effector displacement from a small
        positive joint rotation at the current pose, because the visible
        front/back effect can differ by robot and selected joint.
        For z-paths we keep the existing jacobian-aware correction.
        """
        if axis == "z":
            return -1.0 if jac_sign < 0 else 1.0
        if axis != "x":
            return 1.0

        original_joint_pos = current_joint_pos.copy()
        start_pos = self._get_eef_position()
        probe_joint_pos = original_joint_pos.copy()
        probe_joint_pos[robot_joint_idx] += np.deg2rad(probe_deg)
        self._set_joint_positions(probe_joint_pos)
        probe_pos = self._get_eef_position()
        self._set_joint_positions(original_joint_pos)

        delta_x = float(probe_pos[0] - start_pos[0])
        if abs(delta_x) < 1e-6:
            multiplier = 1.0 if jac_sign >= 0 else -1.0
            print(f"  Path x probe nearly zero; falling back to jac_sign -> multiplier {multiplier:+.0f}")
            return multiplier

        multiplier = 1.0 if delta_x > 0 else -1.0
        print(f"  Path x probe delta_x={delta_x:+.6f} -> multiplier {multiplier:+.0f}")
        return multiplier
    
    def _check_self_collision(self) -> bool:
        """
        Check if the robot has self-collision.
        
        Returns:
            bool: True if self-collision is detected, False otherwise
        """
        # Get robot contact geoms from robot_model
        robot_geoms = set(self.robot.robot_model.contact_geoms)
        
        # Check all contacts
        for i in range(self.env.sim.data.ncon):
            contact = self.env.sim.data.contact[i]
            geom1_name = self.env.sim.model.geom_id2name(contact.geom1)
            geom2_name = self.env.sim.model.geom_id2name(contact.geom2)
            
            # Check if both geoms are from the robot (self-collision)
            if geom1_name in robot_geoms and geom2_name in robot_geoms:
                return True
        
        return False
    
    def _find_joint_index_in_robot(self, joint_dof_id: int) -> Optional[int]:
        """Find the index in robot's joint position array for a given DOF ID."""
        if hasattr(self.robot, '_ref_joint_pos_indexes'):
            for i, qpos_addr in enumerate(self.robot._ref_joint_pos_indexes):
                if isinstance(qpos_addr, (int, np.integer)):
                    if qpos_addr == joint_dof_id:
                        return i
                elif isinstance(qpos_addr, (tuple, list)):
                    start_addr = qpos_addr[0] if isinstance(qpos_addr, tuple) else qpos_addr
                    if start_addr <= joint_dof_id < start_addr + len(qpos_addr):
                        return i
        return None
    
    def _load_cue_config(self, cue: str, config_path: str = "data/results/motion_configs/manipulator/motion_config.json", cue_idx: int = None) -> Dict:
        """
        Load cue configuration from JSON file.
        
        Args:
            cue: Name of the cue
            config_path: Path to JSON file with cue configurations
            cue_idx: Optional idx to match exactly (avoids duplicate cue name issues)
        
        Returns:
            Dictionary containing cue configuration
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Cue configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            configs = json.load(f)
        
        if cue_idx is not None:
            for config in configs:
                if config.get('idx') == cue_idx:
                    return config
        
        for config in configs:
            if config.get('cue') == cue:
                return config
            if cue.isdigit() and config.get('idx') == int(cue):
                return config
        
        raise ValueError(f"Cue '{cue}' not found in configuration file: {config_path}")
    
    def execute_cue(
        self,
        cue: str,
        pose_index: Optional[int] = None,
        config_path: str = "data/results/motion_configs/manipulator/motion_config.json",
        proximal_degree_scale: float = 0.6,
        hz: int = 4,
        filename_suffix: Optional[str] = None,
        enable_self_collision_check: bool = False,
        cue_idx: Optional[int] = None,
        save_gif: bool = True,
        overlay_progress_bar: bool = True,
        progress_bar_style: str = "typed",
    ):
        """
        Execute a cue (e.g., 'waving').
        
        Args:
            cue: Name of the cue
            pose_index: Optional pose_id to use (if None, randomly selects)
            config_path: Path to JSONL file with cue configurations
            proximal_degree_scale: Deprecated compatibility argument. Degrees are no longer scaled by joint choice.
            cue_idx: Optional config idx for exact lookup (avoids duplicate cue name issues)
            save_gif: If True, save GIF to disk. If False, return (frames, pose_id) instead.
            overlay_progress_bar: If True, overlay a progress bar onto frames.
            progress_bar_style: "typed", "simple", or "none".
        """
        print(f"\n{'='*60}")
        print(f"Executing cue: {cue}")
        print(f"{'='*60}\n")
        _debug_log(f"execute_cue:start cue={cue} cue_idx={cue_idx}")
        
        # Load cue configuration from JSON file
        cue_config_data = self._load_cue_config(cue, config_path, cue_idx=cue_idx)
        _debug_log("execute_cue:config loaded")
        render_modifiers = _extract_render_modifiers(cue_config_data)
        hesitation_strength = render_modifiers["hesitation"]
        elegant_curve_strength = render_modifiers["elegant_curve"]
        zittering_strength = render_modifiers["zittering"]
        
        # Extract movements list
        movements = cue_config_data.get('movements', [])
        _debug_log(f"execute_cue:movements count={len(movements)}")
        
        if not movements:
            raise ValueError(f"No movements found in cue '{cue}' configuration")
        
        # Process each movement item
        frames = []
        step_spans = []
        current_pose_name = None
        current_pose = None
        pose_id = None
        
        # Cache for joint selection: (axis, joint_preference) -> (joint_idx, joint_name, joint_dof_id, score)
        joint_cache = {}
        
        for movement_item in movements:
            movement_type = movement_item.get('type')
            parameters = movement_item.get('parameters', {})
            step_start = len(frames)
            
            if movement_type == 'pose':
                # Set pose
                pose_param = parameters.get('pose')
                if pose_param is None:
                    raise ValueError("'pose' parameter is required for 'pose' type movement")
                
                # Handle both string (pose name) and dict (pose definition) formats
                if isinstance(pose_param, str):
                    pose_display_name = pose_param
                elif isinstance(pose_param, dict):
                    orient = pose_param.get('gripper_orientation') or pose_param.get('pitch') or 'any'
                    pose_display_name = f"{pose_param.get('height', '?')}_{pose_param.get('dir', '?')}_{orient}"
                else:
                    raise ValueError(f"Invalid pose format: {type(pose_param)}")
                
                print(f"\n--- Setting Pose: {pose_display_name} ---")
                matching_poses = self._find_matching_poses(pose_param)
                
                if not matching_poses:
                    raise ValueError(f"No matching poses found for {pose_display_name}")
                
                # Select pose
                if current_pose is None:
                    # First pose selection is always deterministic:
                    # 1. filter by dir / gripper_orientation
                    # 2. if x/y/z exist, sort by percentile distance
                    # 3. otherwise use a fixed-seed stable sample
                    selected_pose = None
                    if pose_index is not None:
                        for pose in matching_poses:
                            if pose.get('pose_id') == pose_index:
                                selected_pose = pose
                                break
                        if selected_pose is None:
                            print(f"Warning: pose_id {pose_index} not found, falling back to deterministic selection")

                    if selected_pose is None:
                        selected = _select_initial_poses(
                            matching_poses,
                            pose_param if isinstance(pose_param, dict) else None,
                            top_k=1,
                        )
                        if not selected:
                            raise ValueError(f"No deterministic initial pose found for {pose_display_name}")
                        selected_pose = selected[0]

                    pose_id = selected_pose['pose_id']
                    print(f"Selected pose with pose_id {pose_id}: rank {selected_pose.get('rank', 'N/A')}")
                else:
                    # For non-initial poses, keep dir / orientation filtering and
                    # choose the closest pose from the robot's actual current state.
                    if current_pose is not None:
                        current_joint_pos = self._get_joint_positions()
                        selected_pose, pose_metrics = self._find_closest_pose(
                            current_joint_pos,
                            matching_poses,
                            pose_param if isinstance(pose_param, dict) else None,
                        )
                        xyz_info = ""
                        if pose_metrics["xyz_axes"]:
                            xyz_axes = "/".join(pose_metrics["xyz_axes"])
                            xyz_info = (
                                f", {xyz_axes} pct delta {pose_metrics['xyz_distance']:.1f}"
                                f" (threshold {pose_metrics['xyz_threshold_pct']:.1f}, "
                                f"within={pose_metrics['within_xyz_threshold']})"
                            )
                        print(
                            f"Found closest pose with pose_id {selected_pose['pose_id']}: "
                            f"{pose_metrics['selection_mode']} | "
                            f"proximal delta {pose_metrics['proximal_score']:.1f} deg | "
                            f"weighted joint delta {pose_metrics['weighted_joint_score']:.1f} | "
                            f"joint delta {pose_metrics['joint_score']:.1f} deg{xyz_info}"
                        )
                    if pose_id is None:
                        pose_id = selected_pose['pose_id']
                
                # Get speed and hold_time from parameters
                speed = parameters.get('speed', 1.0)
                hold_time = parameters.get('hold_time', 1.0)
                
                if current_pose is None:
                    # First pose: just set it and hold (no speed/interpolation needed)
                    print("Setting first pose...")
                    self.jacobian_calculator._set_pose_from_data(selected_pose)
                    current_pose_name = pose_param  # Store the pose parameter (string or dict)
                    current_pose = selected_pose
                    
                    # Hold at pose for hold_time
                    num_frames = int(hold_time * hz)
                    for _ in range(num_frames):
                        image = self._capture_image()
                        frames.append(Image.fromarray(image))
                    print(f"Captured {num_frames} frames (hold_time: {hold_time}s, hz: {hz})")
                else:
                    # Transition from current pose to selected pose with speed
                    # Create display names for logging
                    if isinstance(current_pose_name, str):
                        current_display_name = current_pose_name
                    else:
                        current_display_name = f"{current_pose_name.get('height', '?')}_{current_pose_name.get('dir', '?')}_{current_pose_name.get('pitch', '?')}"
                    
                    print(f"Transitioning from {current_display_name} to {pose_display_name} (speed: {speed})...")
                    
                    # Transition from the robot's real current state so movement->pose
                    # respects whatever the preceding movement actually did.
                    start_joint_pos = self._get_joint_positions()
                    end_joint_pos = self._pose_data_to_joint_positions(selected_pose)
                    
                    # Calculate movement duration and number of frames
                    duration = 1.0 / speed
                    num_transition_frames = max(1, int(duration * hz))
                    
                    # Interpolate from start to end pose
                    for frame_idx in range(num_transition_frames):
                        t = (frame_idx + 1) / num_transition_frames  # 0 to 1
                        interpolated_joint_pos = start_joint_pos * (1 - t) + end_joint_pos * t
                        
                        self._set_joint_positions(interpolated_joint_pos)
                        
                        # Capture frame
                        image = self._capture_image()
                        frames.append(Image.fromarray(image))
                    
                    print(f"  Captured {num_transition_frames} transition frames (speed: {speed}, duration: {duration:.2f}s)")
                    
                    # Set final pose (to ensure exact position)
                    self.jacobian_calculator._set_pose_from_data(selected_pose)
                    current_pose_name = pose_param  # Store the pose parameter (string or dict)
                    current_pose = selected_pose
                    
                    # Hold at final pose for hold_time
                    num_hold_frames = int(hold_time * hz)
                    for _ in range(num_hold_frames):
                        image = self._capture_image()
                        frames.append(Image.fromarray(image))
                    print(f"  Captured {num_hold_frames} hold frames (hold_time: {hold_time}s)")
                
            elif movement_type == 'movement':
                # Execute movement
                if current_pose_name is None:
                    raise ValueError("No pose set before movement. Please set a pose first.")
                
                repetition = parameters.get('repetition', 1)
                axis = parameters.get('axis')
                joint_preference = parameters.get('joint')
                
                degrees_array = parameters.get('degrees', [])
                directions = parameters.get('directions', [])
                
                if joint_preference is None:
                    raise ValueError("'joint' parameter is required for 'movement' type")
                
                # Build direction_params: list of (degrees_dict, speed, hold_time)
                # Preserve all requested axes so multi-axis movements remain intact.
                direction_params = []
                
                if not directions:
                    raise ValueError("'directions' parameter is required for 'movement' type")
                
                for direction_config in directions:
                    deg_val = direction_config.get('degrees')
                    if not isinstance(deg_val, dict):
                        raise ValueError(
                            f"'degrees' must be a dict like {{\"z\": 45}}, got: {deg_val!r}"
                        )
                    
                    dir_speed = direction_config.get('speed')
                    if dir_speed is None:
                        raise ValueError(
                            f"'speed' is required per direction entry, got: {direction_config!r}"
                        )
                    dir_hold = direction_config.get('hold_time')
                    if dir_hold is None:
                        raise ValueError(
                            f"'hold_time' is required per direction entry, got: {direction_config!r}"
                        )
                    
                    direction_params.append((dict(deg_val), dir_speed, dir_hold))
                
                if not direction_params:
                    raise ValueError("No direction parameters found")
                
                degrees_array = [dp[0] for dp in direction_params]
                all_axes = sorted({ax for degs, _, _ in direction_params for ax in degs.keys()})
                axis_label = "+".join(all_axes) if all_axes else axis

                print(f"\n--- Movement ---")
                print(f"Repetition: {repetition}")
                print(f"Axis: {axis_label}")
                print(f"Joint preference: {joint_preference}")
                print(f"Degrees: {degrees_array}")
                print(f"Speeds: {[dp[1] for dp in direction_params]}")
                print(f"Hold times: {[dp[2] for dp in direction_params]}")
                
                # Get current joint positions before joint selection
                current_joint_pos = self._get_joint_positions()
                
                # Execute repetitions
                for rep in range(repetition):
                    print(f"\nRepetition {rep + 1}/{repetition}")
                    
                    # Execute each degree value in the array
                    for deg_idx, (degrees_dict, dir_speed, dir_hold_time) in enumerate(direction_params):
                        effective_dir_speed = max(0.5, dir_speed * (1.0 - 0.45 * hesitation_strength))
                        effective_dir_hold_time = dir_hold_time * (1.0 + 0.9 * hesitation_strength)
                        pre_pause_frames = int(hz * 0.12 * hesitation_strength)
                        if pre_pause_frames > 0:
                            for _ in range(pre_pause_frames):
                                image = self._capture_image()
                                frames.append(Image.fromarray(image))

                        # Get expected sign if specified
                        expected_sign = None
                        if 'directions' in parameters and deg_idx < len(parameters['directions']):
                            sign_str = parameters['directions'][deg_idx].get('sign')
                            if sign_str == 'positive':
                                expected_sign = 1
                            elif sign_str == 'negative':
                                expected_sign = -1

                        axis_moves = []
                        primary_axis = None
                        primary_axis_mag = -1.0
                        for move_axis, degrees in degrees_dict.items():
                            cache_key = (move_axis, joint_preference)
                            if cache_key in joint_cache:
                                joint_idx, joint_name, joint_dof_id, score, jac_sign = joint_cache[cache_key]
                                print(
                                    f"\nReusing cached joint selection: {joint_name} "
                                    f"(DOF ID: {joint_dof_id}, Score: {score:.4f})"
                                )
                                print(f"Cache key: axis={move_axis}, joint_preference={joint_preference}")
                            else:
                                joint_idx, joint_name, joint_dof_id, score, jac_sign = self._select_joint(
                                    axis=move_axis,
                                    joint_preference=joint_preference,
                                )
                                joint_cache[cache_key] = (joint_idx, joint_name, joint_dof_id, score, jac_sign)
                                print(f"\nCached joint selection for axis={move_axis}, joint_preference={joint_preference}")

                            robot_joint_idx = self._find_joint_index_in_robot(joint_dof_id)
                            if robot_joint_idx is None:
                                raise ValueError(f"Could not find joint index for DOF ID {joint_dof_id}")

                            scaled_degrees = degrees
                            print(f"  Movement {move_axis}: {scaled_degrees}°")

                            if move_axis == 'z' and jac_sign < 0:
                                scaled_degrees = -scaled_degrees
                                if degrees != scaled_degrees:
                                    print(
                                        f"  Adjusted {move_axis} sign for upward movement: "
                                        f"{scaled_degrees}° (original: {degrees}°)"
                                    )

                            if expected_sign is not None and len(degrees_dict) == 1:
                                actual_sign = np.sign(scaled_degrees) or 1
                                if actual_sign != expected_sign:
                                    print(
                                        f"  Sign mismatch detected! Expected: {expected_sign}, "
                                        f"Actual: {actual_sign}"
                                    )
                                    print(f"  Flipping degree sign: {scaled_degrees}° -> {-scaled_degrees}°")
                                    scaled_degrees = -scaled_degrees

                            axis_moves.append({
                                "axis": move_axis,
                                "joint_name": joint_name,
                                "joint_idx": robot_joint_idx,
                                "degrees": scaled_degrees,
                                "original_degrees": degrees,
                            })
                            if abs(float(degrees)) > primary_axis_mag:
                                primary_axis_mag = abs(float(degrees))
                                primary_axis = move_axis

                        secondary_joint_idx = None
                        if primary_axis is not None and (elegant_curve_strength > 0 or zittering_strength > 0):
                            secondary_axis = _orthogonal_axis(primary_axis)
                            cache_key = (secondary_axis, joint_preference)
                            try:
                                if cache_key in joint_cache:
                                    sec_joint = joint_cache[cache_key]
                                else:
                                    sec_joint = self._select_joint(axis=secondary_axis, joint_preference=joint_preference)
                                    joint_cache[cache_key] = sec_joint
                                secondary_joint_idx = self._find_joint_index_in_robot(sec_joint[2])
                            except Exception:
                                secondary_joint_idx = None

                        duration = 1.0 / effective_dir_speed
                        num_movement_frames = max(1, int(duration * hz))
                        joint_offsets = {}
                        for move in axis_moves:
                            joint_offsets.setdefault(move["joint_idx"], 0.0)
                            joint_offsets[move["joint_idx"]] += np.deg2rad(move["degrees"])

                        safe_joint_offsets = dict(joint_offsets)
                        safe_angle_scale = 1.0

                        if enable_self_collision_check:
                            min_angle_scale = 0.1
                            max_attempts = 5
                            has_collision = True
                            for attempt in range(max_attempts):
                                test_joint_pos = current_joint_pos.copy()
                                for joint_idx, offset in joint_offsets.items():
                                    test_joint_pos[joint_idx] += offset * safe_angle_scale
                                self._set_joint_positions(test_joint_pos)
                                has_collision = self._check_self_collision()
                                if not has_collision:
                                    if attempt > 0:
                                        print(
                                            f"  Found safe multi-axis scale at attempt {attempt + 1}: "
                                            f"{safe_angle_scale:.2f}x"
                                        )
                                    break
                                if attempt < max_attempts - 1:
                                    safe_angle_scale *= 0.5
                                    if safe_angle_scale < min_angle_scale:
                                        print(
                                            f"  Warning: Self-collision persists below "
                                            f"{min_angle_scale * 100:.0f}% scale, skipping movement"
                                        )
                                        self._set_joint_positions(current_joint_pos)
                                        break
                                    print(
                                        f"  Self-collision detected, reducing move scale to "
                                        f"{safe_angle_scale:.2f}x (attempt {attempt + 2}/{max_attempts})..."
                                    )
                            if has_collision and safe_angle_scale < min_angle_scale:
                                continue
                            if has_collision and safe_angle_scale >= min_angle_scale:
                                self._set_joint_positions(current_joint_pos)
                                continue

                        for joint_idx in list(safe_joint_offsets.keys()):
                            safe_joint_offsets[joint_idx] = joint_offsets[joint_idx] * safe_angle_scale

                        self._set_joint_positions(current_joint_pos)

                        for frame_idx in range(num_movement_frames):
                            t = (frame_idx + 1) / num_movement_frames
                            new_joint_pos = current_joint_pos.copy()
                            for joint_idx, offset in safe_joint_offsets.items():
                                new_joint_pos[joint_idx] += t * offset
                            _apply_style_offsets(
                                new_joint_pos,
                                t=t,
                                primary_joint_idx=axis_moves[0]["joint_idx"] if axis_moves else None,
                                secondary_joint_idx=secondary_joint_idx,
                                elegant_curve=elegant_curve_strength,
                                zittering=zittering_strength,
                            )
                            self._set_joint_positions(new_joint_pos)
                            image = self._capture_image()
                            frames.append(Image.fromarray(image))

                        move_summaries = []
                        for move in axis_moves:
                            applied = move["degrees"] * safe_angle_scale
                            move_summaries.append(f"{move['axis']}={applied:.1f}° via {move['joint_name']}")
                        print(
                            f"  Captured {num_movement_frames} movement frames: "
                            f"{', '.join(move_summaries)} (speed: {effective_dir_speed}, duration: {duration:.2f}s)"
                        )

                        for joint_idx, offset in safe_joint_offsets.items():
                            current_joint_pos[joint_idx] += offset
                        
                        # Hold at target position if dir_hold_time > 0
                        if effective_dir_hold_time > 0:
                            num_hold_frames = int(effective_dir_hold_time * hz)
                            for _ in range(num_hold_frames):
                                image = self._capture_image()
                                frames.append(Image.fromarray(image))
                            print(f"  Captured {num_hold_frames} hold frames (hold_time: {effective_dir_hold_time}s)")
            elif movement_type == 'path':
                if current_pose_name is None:
                    raise ValueError("No pose set before path. Please set a pose first.")
                
                shape = parameters.get('shape')
                joint_preference = parameters.get('joint')
                path_speed = parameters.get('speed', 1.0)
                target_pose = parameters.get('target_pose')

                # Some configs express a path as "move from current pose to target pose"
                # without specifying explicit axis / distance values. Treat those as
                # pose transitions so batch rendering can proceed without hand-fixing
                # otherwise valid configs.
                if target_pose is not None:
                    supports_explicit_line = (
                        shape == 'line'
                        and parameters.get('axis') is not None
                        and parameters.get('distance') is not None
                    )
                    supports_explicit_arc = (
                        shape == 'arc'
                        and parameters.get('plane') is not None
                        and parameters.get('radius') is not None
                        and parameters.get('sweep') is not None
                    )
                    if not supports_explicit_line and not supports_explicit_arc:
                        print(f"\n--- Path ({shape or 'target_pose'}) → pose fallback ---")
                        print(f"Target pose: {target_pose}, Speed: {path_speed}, Hold: {parameters.get('hold_time', 0.0)}")

                        matching_poses = self._find_matching_poses(target_pose)
                        if not matching_poses:
                            raise ValueError(f"No matching poses found for target_pose fallback: {target_pose}")

                        current_joint_pos = self._get_joint_positions()
                        selected_pose, pose_metrics = self._find_closest_pose(
                            current_joint_pos,
                            matching_poses,
                            target_pose if isinstance(target_pose, dict) else None,
                        )
                        xyz_info = ""
                        if pose_metrics["xyz_axes"]:
                            xyz_axes = "/".join(pose_metrics["xyz_axes"])
                            xyz_info = (
                                f", {xyz_axes} pct delta {pose_metrics['xyz_distance']:.1f}"
                                f" (threshold {pose_metrics['xyz_threshold_pct']:.1f}, "
                                f"within={pose_metrics['within_xyz_threshold']})"
                            )
                        print(
                            f"Found target pose with pose_id {selected_pose['pose_id']}: "
                            f"{pose_metrics['selection_mode']} | "
                            f"proximal delta {pose_metrics['proximal_score']:.1f} deg | "
                            f"weighted joint delta {pose_metrics['weighted_joint_score']:.1f} | "
                            f"joint delta {pose_metrics['joint_score']:.1f} deg{xyz_info}"
                        )

                        start_joint_pos = current_joint_pos
                        end_joint_pos = self._pose_data_to_joint_positions(selected_pose)
                        duration = 1.0 / path_speed
                        num_transition_frames = max(1, int(duration * hz))

                        for frame_idx in range(num_transition_frames):
                            t = (frame_idx + 1) / num_transition_frames
                            interpolated_joint_pos = start_joint_pos * (1 - t) + end_joint_pos * t
                            self._set_joint_positions(interpolated_joint_pos)
                            image = self._capture_image()
                            frames.append(Image.fromarray(image))

                        print(
                            f"  Captured {num_transition_frames} fallback transition frames "
                            f"(speed: {path_speed}, duration: {duration:.2f}s)"
                        )

                        self.jacobian_calculator._set_pose_from_data(selected_pose)
                        current_pose_name = target_pose
                        current_pose = selected_pose

                        hold_time = parameters.get('hold_time', 0.0)
                        num_hold_frames = int(hold_time * hz)
                        for _ in range(num_hold_frames):
                            image = self._capture_image()
                            frames.append(Image.fromarray(image))
                        if num_hold_frames > 0:
                            print(f"  Captured {num_hold_frames} hold frames (hold_time: {hold_time}s)")
                        continue
                
                if joint_preference is None:
                    raise ValueError("'joint' parameter is required for 'path' type")
                if shape is None:
                    raise ValueError("'shape' parameter is required for 'path' type")
                
                current_joint_pos = self._get_joint_positions()
                
                if shape == 'line':
                    axis = parameters.get('axis')
                    distance = parameters.get('distance')
                    
                    if axis is None or distance is None:
                        raise ValueError("'axis' and 'distance' are required for line path")

                    if isinstance(distance, dict):
                        axis_items = [(ax, float(val)) for ax, val in distance.items() if ax in "xyz"]
                        if not axis_items:
                            raise ValueError(f"line path distance dict must contain x/y/z values, got: {distance}")

                        print(f"\n--- Path (line multi-axis) ---")
                        print(f"Axes: {axis_items}, Speed: {path_speed}, Joint: {joint_preference}")

                        axis_moves = []
                        joint_offsets = {}
                        effective_path_speed = max(0.5, path_speed * (1.0 - 0.45 * hesitation_strength))
                        path_length_deg = float(np.sqrt(sum(single_distance ** 2 for _, single_distance in axis_items)))
                        duration = _path_duration_from_length(path_length_deg, effective_path_speed)
                        num_frames = max(1, int(duration * hz))

                        for single_axis, single_distance in axis_items:
                            cache_key = (single_axis, joint_preference)
                            if cache_key in joint_cache:
                                joint_idx, joint_name, joint_dof_id, score, jac_sign = joint_cache[cache_key]
                                print(f"  Reusing cached joint for {single_axis}-axis: {joint_name}")
                            else:
                                joint_idx, joint_name, joint_dof_id, score, jac_sign = self._select_joint(
                                    axis=single_axis, joint_preference=joint_preference, selection_mode="path",
                                )
                                joint_cache[cache_key] = (joint_idx, joint_name, joint_dof_id, score, jac_sign)

                            robot_joint_idx = self._find_joint_index_in_robot(joint_dof_id)
                            if robot_joint_idx is None:
                                raise ValueError(f"Could not find joint index for DOF ID {joint_dof_id}")

                            sign_multiplier = self._path_axis_sign_multiplier(
                                axis=single_axis,
                                robot_joint_idx=robot_joint_idx,
                                current_joint_pos=current_joint_pos,
                                jac_sign=jac_sign,
                            )
                            effective_distance = _normalize_path_axis_value(single_axis, single_distance, sign_multiplier)

                            offset_rad = np.deg2rad(effective_distance)
                            joint_offsets[robot_joint_idx] = joint_offsets.get(robot_joint_idx, 0.0) + offset_rad
                            axis_moves.append(
                                {
                                    "axis": single_axis,
                                    "degrees": effective_distance,
                                    "joint_name": joint_name,
                                    "joint_idx": robot_joint_idx,
                                }
                            )

                        secondary_joint_idx = None
                        if elegant_curve_strength > 0 or zittering_strength > 0:
                            try:
                                secondary_axis = axis_items[1][0] if len(axis_items) > 1 else _orthogonal_axis(axis_items[0][0])
                                cache_key2 = (secondary_axis, joint_preference)
                                if cache_key2 in joint_cache:
                                    sec_joint = joint_cache[cache_key2]
                                else:
                                    sec_joint = self._select_joint(axis=secondary_axis, joint_preference=joint_preference)
                                    joint_cache[cache_key2] = sec_joint
                                secondary_joint_idx = self._find_joint_index_in_robot(sec_joint[2])
                            except Exception:
                                secondary_joint_idx = None

                        for frame_idx in range(num_frames):
                            t = (frame_idx + 1) / num_frames
                            new_joint_pos = current_joint_pos.copy()
                            for joint_idx, offset in joint_offsets.items():
                                new_joint_pos[joint_idx] += t * offset
                            _apply_style_offsets(
                                new_joint_pos,
                                t=t,
                                primary_joint_idx=axis_moves[0]["joint_idx"] if axis_moves else None,
                                secondary_joint_idx=secondary_joint_idx,
                                elegant_curve=elegant_curve_strength,
                                zittering=zittering_strength,
                            )
                            self._set_joint_positions(new_joint_pos)
                            image = self._capture_image()
                            frames.append(Image.fromarray(image))

                        for joint_idx, offset in joint_offsets.items():
                            current_joint_pos[joint_idx] += offset
                        move_summaries = [f"{m['axis']}={m['degrees']:.1f}° via {m['joint_name']}" for m in axis_moves]
                        print(
                            f"  Captured {num_frames} frames ({', '.join(move_summaries)}, "
                            f"duration: {duration:.2f}s, speed: {effective_path_speed})"
                        )
                        continue
                    
                    print(f"\n--- Path (line) ---")
                    print(f"Axis: {axis}, Distance: {distance}°, Speed: {path_speed}, Joint: {joint_preference}")
                    
                    cache_key = (axis, joint_preference)
                    if cache_key in joint_cache:
                        joint_idx, joint_name, joint_dof_id, score, jac_sign = joint_cache[cache_key]
                        print(f"Reusing cached joint: {joint_name}")
                    else:
                        joint_idx, joint_name, joint_dof_id, score, jac_sign = self._select_joint(
                            axis=axis, joint_preference=joint_preference, selection_mode="path",
                        )
                        joint_cache[cache_key] = (joint_idx, joint_name, joint_dof_id, score, jac_sign)
                    
                    self._set_joint_positions(current_joint_pos)
                    
                    robot_joint_idx = self._find_joint_index_in_robot(joint_dof_id)
                    if robot_joint_idx is None:
                        raise ValueError(f"Could not find joint index for DOF ID {joint_dof_id}")
                    
                    sign_multiplier = self._path_axis_sign_multiplier(
                        axis=axis,
                        robot_joint_idx=robot_joint_idx,
                        current_joint_pos=current_joint_pos,
                        jac_sign=jac_sign,
                    )
                    effective_distance = _normalize_path_axis_value(axis, distance, sign_multiplier)
                    
                    offset_rad = np.deg2rad(effective_distance)
                    start_angle = current_joint_pos[robot_joint_idx]
                    
                    effective_path_speed = max(0.5, path_speed * (1.0 - 0.45 * hesitation_strength))
                    pre_pause_frames = int(hz * 0.12 * hesitation_strength)
                    if pre_pause_frames > 0:
                        for _ in range(pre_pause_frames):
                            image = self._capture_image()
                            frames.append(Image.fromarray(image))

                    secondary_joint_idx = None
                    if elegant_curve_strength > 0 or zittering_strength > 0:
                        secondary_axis = _orthogonal_axis(axis)
                        cache_key2 = (secondary_axis, joint_preference)
                        try:
                            if cache_key2 in joint_cache:
                                sec_joint = joint_cache[cache_key2]
                            else:
                                sec_joint = self._select_joint(axis=secondary_axis, joint_preference=joint_preference)
                                joint_cache[cache_key2] = sec_joint
                            secondary_joint_idx = self._find_joint_index_in_robot(sec_joint[2])
                        except Exception:
                            secondary_joint_idx = None

                    duration = _path_duration_from_length(abs(effective_distance), effective_path_speed)
                    num_frames = max(1, int(duration * hz))
                    
                    for frame_idx in range(num_frames):
                        t = (frame_idx + 1) / num_frames
                        new_joint_pos = current_joint_pos.copy()
                        new_joint_pos[robot_joint_idx] = start_angle + t * offset_rad
                        _apply_style_offsets(
                            new_joint_pos,
                            t=t,
                            primary_joint_idx=robot_joint_idx,
                            secondary_joint_idx=secondary_joint_idx,
                            elegant_curve=elegant_curve_strength,
                            zittering=zittering_strength,
                        )
                        self._set_joint_positions(new_joint_pos)
                        image = self._capture_image()
                        frames.append(Image.fromarray(image))
                    
                    current_joint_pos[robot_joint_idx] = start_angle + offset_rad
                    print(f"  Captured {num_frames} frames ({effective_distance:.1f}°, duration: {duration:.2f}s, speed: {effective_path_speed})")
                
                elif shape in ('arc', 'circle'):
                    plane = parameters.get('plane')
                    radius = parameters.get('radius')
                    sweep = parameters.get('sweep')
                    direction = parameters.get('direction', 'ccw')
                    destination = parameters.get('destination')
                    if shape == 'circle' and sweep is None:
                        sweep = 360
                    
                    if not all([plane, radius is not None, sweep is not None]):
                        if destination is not None:
                            base_pose = {}
                            if isinstance(current_pose_name, dict):
                                base_pose.update(current_pose_name)
                            if isinstance(current_pose, dict):
                                for key in ("dir", "gripper_orientation", "x_pct", "y_pct", "z_pct"):
                                    if current_pose.get(key) is not None:
                                        base_pose[key] = current_pose.get(key)
                            target_pose = {
                                "dir": base_pose.get("dir"),
                                "gripper_orientation": base_pose.get("gripper_orientation"),
                                "x": destination.get("x", base_pose.get("x_pct")),
                                "y": destination.get("y", base_pose.get("y_pct")),
                                "z": destination.get("z", base_pose.get("z_pct")),
                            }
                            target_pose = {k: v for k, v in target_pose.items() if v is not None}
                            print(f"\n--- Path ({shape}) destination → pose fallback ---")
                            print(f"Destination pose: {target_pose}, Speed: {path_speed}")

                            matching_poses = self._find_matching_poses(target_pose)
                            if not matching_poses:
                                raise ValueError(f"No matching poses found for destination fallback: {target_pose}")

                            current_joint_pos = self._get_joint_positions()
                            selected_pose, pose_metrics = self._find_closest_pose(
                                current_joint_pos,
                                matching_poses,
                                target_pose,
                            )
                            start_joint_pos = current_joint_pos
                            end_joint_pos = self._pose_data_to_joint_positions(selected_pose)
                            duration = 1.0 / path_speed
                            num_transition_frames = max(1, int(duration * hz))

                            for frame_idx in range(num_transition_frames):
                                t = (frame_idx + 1) / num_transition_frames
                                interpolated_joint_pos = start_joint_pos * (1 - t) + end_joint_pos * t
                                self._set_joint_positions(interpolated_joint_pos)
                                image = self._capture_image()
                                frames.append(Image.fromarray(image))

                            print(
                                f"  Captured {num_transition_frames} destination fallback frames "
                                f"(speed: {path_speed}, duration: {duration:.2f}s)"
                            )
                            self.jacobian_calculator._set_pose_from_data(selected_pose)
                            current_pose_name = target_pose
                            current_pose = selected_pose
                            hold_time = parameters.get('hold_time', 0.0)
                            num_hold_frames = int(hold_time * hz)
                            for _ in range(num_hold_frames):
                                image = self._capture_image()
                                frames.append(Image.fromarray(image))
                            continue
                        raise ValueError("'plane', 'radius', and 'sweep' are required for arc path")
                    if len(plane) != 2 or not all(c in 'xyz' for c in plane):
                        raise ValueError(f"'plane' must be two axes like 'xy', 'xz', 'yz', got: {plane}")
                    
                    axis1, axis2 = plane[0], plane[1]
                    
                    print(f"\n--- Path (arc) ---")
                    print(f"Plane: {plane}, Radius: {radius}°, Sweep: {sweep}°, Direction: {direction}, Speed: {path_speed}, Joint: {joint_preference}")
                    
                    joints_info = {}
                    for ax in [axis1, axis2]:
                        cache_key = (ax, joint_preference)
                        if cache_key in joint_cache:
                            joints_info[ax] = joint_cache[cache_key]
                            print(f"  Reusing cached joint for {ax}-axis: {joints_info[ax][1]}")
                        else:
                            result = self._select_joint(
                                axis=ax, joint_preference=joint_preference, selection_mode="path",
                            )
                            joint_cache[cache_key] = result
                            joints_info[ax] = result
                    
                    self._set_joint_positions(current_joint_pos)
                    
                    j1 = joints_info[axis1]
                    j2 = joints_info[axis2]
                    
                    robot_joint_idx1 = self._find_joint_index_in_robot(j1[2])
                    robot_joint_idx2 = self._find_joint_index_in_robot(j2[2])
                    
                    if robot_joint_idx1 is None or robot_joint_idx2 is None:
                        raise ValueError("Could not find joint indices for arc path")
                    
                    same_joint = (robot_joint_idx1 == robot_joint_idx2)
                    if same_joint:
                        print(f"  Warning: Same joint selected for both axes ({j1[1]}). Arc may be distorted.")
                    
                    jac_sign1 = j1[4]
                    jac_sign2 = j2[4]
                    sign_multiplier1 = self._path_axis_sign_multiplier(
                        axis=axis1,
                        robot_joint_idx=robot_joint_idx1,
                        current_joint_pos=current_joint_pos,
                        jac_sign=jac_sign1,
                    )
                    sign_multiplier2 = self._path_axis_sign_multiplier(
                        axis=axis2,
                        robot_joint_idx=robot_joint_idx2,
                        current_joint_pos=current_joint_pos,
                        jac_sign=jac_sign2,
                    )
                    
                    effective_radius = radius
                    
                    radius_rad = np.deg2rad(effective_radius)
                    sweep_rad = np.deg2rad(sweep)
                    
                    start1 = current_joint_pos[robot_joint_idx1]
                    start2 = current_joint_pos[robot_joint_idx2]
                    
                    effective_path_speed = max(0.5, path_speed * (1.0 - 0.45 * hesitation_strength))
                    pre_pause_frames = int(hz * 0.12 * hesitation_strength)
                    if pre_pause_frames > 0:
                        for _ in range(pre_pause_frames):
                            image = self._capture_image()
                            frames.append(Image.fromarray(image))

                    path_length_deg = abs(effective_radius * sweep_rad)
                    duration = _path_duration_from_length(path_length_deg, effective_path_speed)
                    num_frames = max(1, int(duration * hz))
                    
                    dir_sign = 1.0 if direction == 'ccw' else -1.0
                    
                    print(f"  Joint 1 ({axis1}): {j1[1]} (robot idx: {robot_joint_idx1}, jac_sign: {jac_sign1})")
                    print(f"  Joint 2 ({axis2}): {j2[1]} (robot idx: {robot_joint_idx2}, jac_sign: {jac_sign2})")
                    
                    for frame_idx in range(num_frames):
                        t = (frame_idx + 1) / num_frames
                        theta = t * sweep_rad
                        
                        offset1 = radius_rad * np.sin(theta) * dir_sign
                        offset2 = radius_rad * (1.0 - np.cos(theta))
                        
                        offset1 = _normalize_path_axis_value(axis1, offset1, sign_multiplier1)
                        offset2 = _normalize_path_axis_value(axis2, offset2, sign_multiplier2)
                        
                        new_joint_pos = current_joint_pos.copy()
                        if same_joint:
                            new_joint_pos[robot_joint_idx1] = start1 + offset1 + offset2
                        else:
                            new_joint_pos[robot_joint_idx1] = start1 + offset1
                            new_joint_pos[robot_joint_idx2] = start2 + offset2
                        _apply_style_offsets(
                            new_joint_pos,
                            t=t,
                            primary_joint_idx=robot_joint_idx1,
                            secondary_joint_idx=None if same_joint else robot_joint_idx2,
                            elegant_curve=elegant_curve_strength * 0.5,
                            zittering=zittering_strength,
                        )
                        
                        self._set_joint_positions(new_joint_pos)
                        image = self._capture_image()
                        frames.append(Image.fromarray(image))
                    
                    final_theta = sweep_rad
                    final_offset1 = radius_rad * np.sin(final_theta) * dir_sign
                    final_offset2 = radius_rad * (1.0 - np.cos(final_theta))
                    final_offset1 = _normalize_path_axis_value(axis1, final_offset1, sign_multiplier1)
                    final_offset2 = _normalize_path_axis_value(axis2, final_offset2, sign_multiplier2)
                    
                    if same_joint:
                        current_joint_pos[robot_joint_idx1] = start1 + final_offset1 + final_offset2
                    else:
                        current_joint_pos[robot_joint_idx1] = start1 + final_offset1
                        current_joint_pos[robot_joint_idx2] = start2 + final_offset2
                    
                    print(f"  Captured {num_frames} frames (radius: {effective_radius:.1f}°, sweep: {sweep}°, duration: {duration:.2f}s)")
                
                else:
                    raise ValueError(f"Unknown path shape: {shape}")
            
            elif movement_type == 'gripper':
                target_open = parameters.get('target_open')
                hold_time = parameters.get('hold_time', 0.0)
                print(f"\n--- Gripper (visual no-op) ---")
                print(f"Target open: {target_open}, Hold: {hold_time}")
                num_hold_frames = int(hold_time * hz)
                for _ in range(num_hold_frames):
                    image = self._capture_image()
                    frames.append(Image.fromarray(image))
                if num_hold_frames > 0:
                    print(f"  Captured {num_hold_frames} hold frames for gripper step")
            else:
                raise ValueError(f"Unknown movement type: {movement_type}")

            step_end = len(frames)
            if step_end > step_start:
                step_spans.append({
                    "type": movement_type,
                    "start": step_start,
                    "end": step_end,
                })
        
        if len(frames) == 0:
            raise ValueError("No frames captured - all movements were skipped or failed")

        if overlay_progress_bar and progress_bar_style == "typed":
            frames = _overlay_progress_bar_on_frames(frames, step_spans)
        elif overlay_progress_bar and progress_bar_style == "simple":
            frames = _overlay_simple_progress_bar_on_frames(frames)
        
        if not save_gif:
            return frames, pose_id
        
        # Save GIF
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_cue = cue.replace("/", "_").replace("\\", "_").replace(" ", "_")
        if pose_id is not None:
            base_filename = f"{now}_{self.robot_name}_{safe_cue}_p{pose_id}"
        else:
            base_filename = f"{now}_{self.robot_name}_{safe_cue}"
        
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
        self.jacobian_calculator.close()


def _select_initial_poses(matching_poses, pose_def, top_k=None, pct_threshold=15.0,
                          seed=42):
    """Select initial poses by Euclidean distance in percentile space.

    Args:
        matching_poses: Poses already filtered by dir/gripper_orientation.
        pose_def: Dict with target x/y/z as integers (0-100 percentile).
        top_k: Return closest k poses. If None, return all within threshold.
        pct_threshold: Max Euclidean distance when top_k is None (default 15).
        seed: Random seed for deterministic tie-breaking.
    """
    rng = random.Random(seed)

    seen = set()
    unique = []
    for p in matching_poses:
        pid = p.get('pose_id')
        if pid not in seen:
            seen.add(pid)
            unique.append(p)

    if not unique:
        return []

    target = {}
    if isinstance(pose_def, dict):
        for axis in ['x', 'y', 'z']:
            val = pose_def.get(axis)
            if val is not None and isinstance(val, (int, float)):
                target[axis] = float(val)

    if not target:
        rng.shuffle(unique)
        return unique[:top_k] if top_k is not None else unique

    for p in unique:
        dist_sq = 0.0
        for axis, target_val in target.items():
            pose_val = p.get(f'{axis}_pct', 50)
            dist_sq += (pose_val - target_val) ** 2
        p['_pct_distance'] = dist_sq ** 0.5

    unique.sort(key=lambda p: (p['_pct_distance'], p.get('pose_id', 0)))

    if top_k is not None:
        result = unique[:top_k]
    else:
        result = [p for p in unique if p['_pct_distance'] <= pct_threshold]
        if not result:
            result = unique[:1]

    for p in unique:
        p.pop('_pct_distance', None)

    return result


def _config_has_path(cue_config: Dict) -> bool:
    """Check if a cue config contains any path-type movement steps."""
    return any(m.get('type') == 'path' for m in cue_config.get('movements', []))


def _scaled_preview_config(
    cue_config: Dict,
    speed_scale: float = 1.0,
    hold_scale: float = 1.0,
    max_hold_time: Optional[float] = None,
) -> Dict:
    """Return a copy of a cue config with faster timings for preview rendering."""
    preview = copy.deepcopy(cue_config)
    for movement in preview.get("movements", []):
        params = movement.get("parameters", {})
        if "speed" in params and params["speed"] is not None:
            params["speed"] = max(0.1, float(params["speed"]) * speed_scale)
        if "hold_time" in params and params["hold_time"] is not None:
            hold = max(0.0, float(params["hold_time"]) * hold_scale)
            if max_hold_time is not None:
                hold = min(hold, max_hold_time)
            params["hold_time"] = hold
        for direction in params.get("directions", []) or []:
            if direction.get("speed") is not None:
                direction["speed"] = max(0.1, float(direction["speed"]) * speed_scale)
            if direction.get("hold_time") is not None:
                hold = max(0.0, float(direction["hold_time"]) * hold_scale)
                if max_hold_time is not None:
                    hold = min(hold, max_hold_time)
                direction["hold_time"] = hold
    return preview


def generate(
    robot: str = "IIWA",
    env: str = "EmptySpace",
    cue: str = "beckoning",
    cue_idx: Optional[int] = None,
    pose_index: Optional[int] = None,
    controller: str = "IK_POSE",
    jsonl_path: str = "data/seed/_remainder/closest_poses_results.jsonl",
    config_path: str = "data/results/motion_configs/manipulator/motion_configs.json",
    output_dir: str = "data/results/render/manipulator",
    proximal_degree_scale: float = 0.6,
    camera_distance: float = 1.8,
    hz: int = 4,
    path_hz: int = 12,
    top_k: Optional[int] = 5,
    enable_self_collision_check: bool = False,
    preview_speed_scale: float = 1.0,
    preview_hold_scale: float = 1.0,
    preview_max_hold_time: Optional[float] = None,
):
    """
    Main function to generate robot motions.

    Args:
        robot: Robot name
        env: Environment name
        cue: Name of the cue to execute (e.g., 'waving')
        pose_index: Optional pose_id to use (if None, selects from matching poses)
        controller: Controller name
        jsonl_path: Path to pose database JSONL file
        config_path: Path to JSON file with cue configurations
        proximal_degree_scale: Deprecated compatibility argument. Executor now uses configured degrees as-is.
        camera_distance: Multiplier for camera FOV to zoom out (default: 1.8 = 80% wider view)
        hz: Frame rate for GIF generation in frames per second (default: 4)
        path_hz: Frame rate used when the config contains path steps (default: 12)
        top_k: Number of unique initial poses to sample (default: 5)
        enable_self_collision_check: Enable self-collision detection and angle reduction (default: False)
        preview_speed_scale: Multiplier for all speeds when rendering a fast preview
        preview_hold_scale: Multiplier for all hold times when rendering a fast preview
        preview_max_hold_time: Optional clamp for hold times when rendering a fast preview
    """

    with open(config_path, 'r') as f:
        configs = json.load(f)

    cue_config = None
    if cue_idx is not None:
        for cfg in configs:
            if cfg.get('idx') == cue_idx:
                cue_config = cfg
                cue = cfg['cue']
                break
        if cue_config is None:
            raise ValueError(f"cue_idx={cue_idx} not found in {config_path}")
    else:
        for cfg in configs:
            if cfg.get('cue') == cue:
                cue_config = cfg
                break
        if cue_config is None:
            raise ValueError(f"cue='{cue}' not found in {config_path}")

    # Use one consistent render frame rate across movement and path steps so that
    # speed values stay comparable between configs that do and do not contain paths.
    effective_hz = hz
    if _config_has_path(cue_config) and path_hz != hz:
        print(f"Config contains path steps, but using unified hz={hz} for speed consistency (path_hz={path_hz} ignored)")

    effective_config_path = config_path
    temp_config_path = None
    if (
        preview_speed_scale != 1.0
        or preview_hold_scale != 1.0
        or preview_max_hold_time is not None
    ):
        scaled_configs = copy.deepcopy(configs)
        for i, cfg in enumerate(scaled_configs):
            if cfg.get("idx") == cue_config.get("idx"):
                scaled_configs[i] = _scaled_preview_config(
                    cfg,
                    speed_scale=preview_speed_scale,
                    hold_scale=preview_hold_scale,
                    max_hold_time=preview_max_hold_time,
                )
                break
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="motion_preview_",
            delete=False,
        ) as tmp:
            json.dump(scaled_configs, tmp, indent=2)
            temp_config_path = tmp.name
            effective_config_path = tmp.name
        print(
            "Using preview timing overrides: "
            f"speed_scale={preview_speed_scale}, "
            f"hold_scale={preview_hold_scale}, "
            f"max_hold={preview_max_hold_time}"
        )

    generator = MotionGenerator(
        robot_name=robot,
        env_name=env,
        controller_name=controller,
        jsonl_path=jsonl_path,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_distance=camera_distance,
        output_dir=output_dir,
    )
    _debug_log("generate: MotionGenerator constructed")

    try:
        if pose_index is not None:
            _debug_log(f"generate: direct pose_index path pose_index={pose_index}")
            generator.execute_cue(
                cue=cue,
                pose_index=pose_index,
                config_path=effective_config_path,
                proximal_degree_scale=proximal_degree_scale,
                hz=effective_hz,
                enable_self_collision_check=enable_self_collision_check,
                cue_idx=cue_idx,
                save_gif=True,
            )
        else:
            _debug_log("generate: selecting initial poses from config")
            first_pose_def = None
            for m in cue_config.get('movements', []):
                if m.get('type') == 'pose':
                    first_pose_def = m['parameters']['pose']
                    break

            if first_pose_def is None:
                raise ValueError("No pose movement found in cue config")

            _debug_log(f"generate: first_pose_def={first_pose_def}")
            matching_poses = generator._find_matching_poses(first_pose_def)
            _debug_log(f"generate: matching_poses count={len(matching_poses)}")
            selected_poses = _select_initial_poses(matching_poses, first_pose_def, top_k)
            _debug_log(f"generate: selected_poses count={len(selected_poses)}")

            if not selected_poses:
                raise ValueError("No matching poses found for initial pose")

            print(f"Selected {len(selected_poses)} unique initial poses (top_k={top_k})")

            if len(selected_poses) == 1:
                generator.execute_cue(
                    cue=cue,
                    pose_index=selected_poses[0]['pose_id'],
                    config_path=effective_config_path,
                    proximal_degree_scale=proximal_degree_scale,
                    hz=effective_hz,
                    enable_self_collision_check=enable_self_collision_check,
                    cue_idx=cue_idx,
                    save_gif=True,
                )
            else:
                variations = []
                for k, pose in enumerate(selected_poses):
                    if k > 0:
                        generator._set_joint_positions(generator.initial_joint_pos)
                    try:
                        frames, pid = generator.execute_cue(
                            cue=cue,
                            pose_index=pose['pose_id'],
                            config_path=effective_config_path,
                            proximal_degree_scale=proximal_degree_scale,
                            hz=effective_hz,
                            enable_self_collision_check=enable_self_collision_check,
                            cue_idx=cue_idx,
                            save_gif=False,
                        )
                        variations.append((frames, pid, pose))
                    except Exception as e:
                        print(f"  Variation {k+1}/{len(selected_poses)} failed: {e}")

                if not variations:
                    raise ValueError("All variations failed")

                _save_tiled_gif(variations, generator.output_dir, robot, cue, effective_hz, cue_idx)
    finally:
        generator.close()
        if temp_config_path and os.path.exists(temp_config_path):
            os.unlink(temp_config_path)

    return True


def _make_pose_label(pose_id, pose_data=None):
    """Build a compact label from pose metadata.

    Format: ``p{id} {dir} z:{pct} y:{pct} x:{pct} {orient}``
    """
    parts = [f"p{pose_id}" if pose_id is not None else "?"]
    if pose_data is None:
        return parts[0]
    d = pose_data.get('dir', '')
    if d:
        parts.append(d)
    for axis in ('z', 'y', 'x'):
        val = pose_data.get(f'{axis}_pct')
        if val is not None:
            parts.append(f"{axis}:{val}")
    orient = pose_data.get('gripper_orientation', '')
    if orient:
        parts.append('H' if orient == 'horizontal' else 'V')
    return ' '.join(parts)


def _save_tiled_gif(
    variations: List[tuple],
    output_dir: str,
    robot: str,
    cue: str,
    hz: int,
    cue_idx: Optional[int] = None,
    max_cols: int = 10,
):
    """Save multiple (frames, pose_id[, pose_data]) variations as a grid-tiled GIF."""
    from PIL import ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    label_h = 24
    tile_w, tile_h = variations[0][0][0].size
    num_vars = len(variations)
    max_frames = max(len(v[0]) for v in variations)

    num_cols = min(num_vars, max_cols)
    num_rows = (num_vars + num_cols - 1) // num_cols
    total_w = num_cols * tile_w
    total_h = num_rows * (tile_h + label_h)

    combined_frames = []
    for i in range(max_frames):
        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        for v_idx, variation in enumerate(variations):
            frames = variation[0]
            pose_id = variation[1]
            pose_data = variation[2] if len(variation) > 2 else None

            col = v_idx % num_cols
            row = v_idx // num_cols
            x_off = col * tile_w
            y_off = row * (tile_h + label_h)

            frame = frames[i % len(frames)]
            if frame.mode != "RGB":
                frame = frame.convert("RGB")
            if frame.size != (tile_w, tile_h):
                frame = frame.resize((tile_w, tile_h))
            canvas.paste(frame, (x_off, y_off))

            label = _make_pose_label(pose_id, pose_data)
            tw = draw.textlength(label, font=font) if hasattr(draw, "textlength") else 40
            draw.text((x_off + (tile_w - tw) / 2, y_off + tile_h + 4), label, fill="black", font=font)

            if col > 0:
                draw.line([(x_off, y_off), (x_off, y_off + tile_h + label_h)], fill=(200, 200, 200), width=1)
            if row > 0:
                draw.line([(x_off, y_off), (x_off + tile_w, y_off)], fill=(200, 200, 200), width=1)

        combined_frames.append(canvas)

    if not combined_frames:
        return

    os.makedirs(output_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_cue = cue.replace("/", "_").replace("\\", "_").replace(" ", "_")
    idx_tag = f"_c{cue_idx}" if cue_idx is not None else ""
    filename = f"{now}_{robot}_{safe_cue}{idx_tag}_tiled.gif"
    filepath = os.path.join(output_dir, filename)

    frame_duration_ms = int(1000 / hz)
    combined_frames[0].save(
        filepath,
        save_all=True,
        append_images=combined_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        disposal=1,
    )

    for f in combined_frames:
        try:
            f.close()
        except Exception:
            pass
    for frames, *_ in variations:
        for f in frames:
            try:
                f.close()
            except Exception:
                pass

    print(f"\n{'='*60}")
    print(f"Saved tiled GIF ({num_vars} variations): {filepath}")
    print(f"Total frames: {max_frames}, Tile size: {tile_w}x{tile_h}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    fire.Fire(generate)
