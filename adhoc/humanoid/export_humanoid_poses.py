"""
Export and query humanoid robot poses in a single step.

Pipeline:
1. Brute-force generate all pose combinations (with shoulder forward bias)
2. Classify each pose by direction, height, pitch
3. Select closest poses per direction/height/pitch combination
4. Save to JSONL files (all + closest)
5. Render tiled PNG previews for key directions (front, right, left, down)

Supported robots:
- GR1ArmsOnly (14 joints: right arm [0-6], left arm [7-13])
- GR1FixedLowerBody (20 joints: head[0-2], torso[3-5], right arm[6-12], left arm[13-19])
- GR1FloatingBody (20 joints: head[0-2], torso[3-5], right arm[6-12], left arm[13-19])
- GR1 (32 joints: full humanoid)

Usage:
    python adhoc/humanoid/export_humanoid_poses.py --robot GR1 --active-arm right
    python adhoc/humanoid/export_humanoid_poses.py --robot GR1 --active-arm right --shoulder-forward-deg 30
    python adhoc/humanoid/export_humanoid_poses.py --robot GR1 --active-arm right --skip-tiled-png
"""

import fire
import os
import sys
import json
import math
import numpy as np
from typing import Optional, List, Dict
from itertools import product
from tqdm import tqdm

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.utils import transform_utils as T
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to import arm_pose_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robotarm'))
from arm_pose_config import direction_pose_set, pitch_poses, poses, height_map

# Humanoid-specific fixed joint configurations
# Format: robot_name -> {arm -> fixed_joint_indices_str}
HUMANOID_FIXED_JOINTS = {
    'GR1ArmsOnly': {
        'right': "7-13",  # Fix left arm
        'left': "0-6",     # Fix right arm
    },
    'GR1FixedLowerBody': {
        'right': "0-5, 13-19",  # Fix head, torso, left arm
        'left': "0-5, 6-12",    # Fix head, torso, right arm
    },
    'GR1FloatingBody': {
        'right': "0-5, 13-19",
        'left': "0-5, 6-12",
    },
    'GR1': {
        'right': "0-5, 13-19, 20-31",  # Fix head, torso, left arm, legs
        'left': "0-5, 6-12, 20-31",    # Fix head, torso, right arm, legs
    },
}

# Shoulder pitch joint index per robot/arm
# This joint controls forward/backward arm swing
SHOULDER_PITCH_INDEX = {
    'GR1ArmsOnly': {'right': 0, 'left': 7},
    'GR1FixedLowerBody': {'right': 6, 'left': 13},
    'GR1FloatingBody': {'right': 6, 'left': 13},
    'GR1': {'right': 6, 'left': 13},
}


class HumanoidPoseExporter:
    """Export all possible poses for humanoid robots with one arm active."""

    # GR1 humanoid EE frame is inverted vs standalone robot arm convention.
    # All 6 directions are swapped: front↔back, up↔down, left↔right.
    _HUMANOID_DIR_INVERSION = {
        "front": "back", "back": "front",
        "up": "down", "down": "up",
        "left": "right", "right": "left",
    }

    def _classify_direction(self, roll_deg, pitch_deg, yaw_deg):
        """Classify EE direction by matching against arm_pose_config templates.

        Computes angular distance (roll + yaw) from the given RPY to every
        canonical template in arm_pose_config.poses and picks the closest
        matching direction.

        For GR1 humanoid robots the EE frame convention is opposite to
        standalone robot arms, so all directions are inverted after matching.

        Examples (GR1 right arm):
            (180, -1, -90)   → up    (arm_pose_config "down" inverted)
            (90,  -1, -90)   → front (arm_pose_config "back" inverted)
            (-130, 90, -130) → back  (arm_pose_config "front" inverted)
        """
        best_dir = "front"
        best_diff = float('inf')

        for dir_name, templates in poses.items():
            for tmpl in templates:
                roll_diff = abs(roll_deg - tmpl['roll']) % 360
                roll_diff = min(roll_diff, 360 - roll_diff)

                yaw_diff = abs(yaw_deg - tmpl['yaw']) % 360
                yaw_diff = min(yaw_diff, 360 - yaw_diff)

                total = roll_diff + yaw_diff
                if total < best_diff:
                    best_diff = total
                    best_dir = dir_name

        return self._HUMANOID_DIR_INVERSION.get(best_dir, best_dir)

    def _classify_pitch(self, pitch_deg, tolerance=45):
        """Classify EE pitch orientation.
        
        Unlike direction (which is inverted for GR1), pitch classification
        does NOT need inversion: vertical=0°/180°, horizontal=±90° is correct as-is.
        """
        pitch_deg = ((pitch_deg + 180) % 360) - 180
        if abs(pitch_deg) < tolerance or abs(abs(pitch_deg) - 180) < tolerance:
            return "vertical"
        else:
            return "horizontal"

    def __init__(
        self,
        robot_name: str = "GR1",
        active_arm: str = "right",
        env_name: str = "EmptySpace",
        controller_name: str = "OSC_POSE",
        has_offscreen_renderer: bool = True,
    ):
        """Initialize the exporter."""
        self.robot_name = robot_name
        self.active_arm = active_arm

        print(f"Initializing humanoid robot: {robot_name} (active arm: {active_arm})")

        if robot_name not in HUMANOID_FIXED_JOINTS:
            raise ValueError(f"Robot {robot_name} not supported. Supported: {list(HUMANOID_FIXED_JOINTS.keys())}")
        if active_arm not in ['right', 'left']:
            raise ValueError(f"active_arm must be 'right' or 'left', got {active_arm}")

        options = {
            "env_name": env_name,
            "robots": robot_name,
            "has_renderer": False,
            "has_offscreen_renderer": has_offscreen_renderer,
            "ignore_done": True,
            "use_camera_obs": False,
            "control_freq": 20,
        }

        arm_controller_config = suite.load_part_controller_config(default_controller=controller_name)
        options["controller_configs"] = refactor_composite_controller_config(
            arm_controller_config, robot_name, ["right", "left"]
        )

        self.env = suite.make(**options, horizon=1000)
        self.env.reset()
        self.robot = self.env.robots[0]
        self.initial_joint_pos = self.robot._joint_positions.copy()
        self.num_joints = len(self.initial_joint_pos)
        self.has_offscreen_renderer = has_offscreen_renderer

        # Parse fixed/active joints
        fixed_indices_str = HUMANOID_FIXED_JOINTS[robot_name][active_arm]
        self.fixed_joint_indices = self._parse_fixed_indices(fixed_indices_str)
        base_active = list(range(self.num_joints - 1))  # exclude gripper
        self.active_joint_indices = [idx for idx in base_active if idx not in self.fixed_joint_indices]

        # Shoulder pitch index
        self.shoulder_pitch_idx = SHOULDER_PITCH_INDEX.get(robot_name, {}).get(active_arm)

        # Fix GR1 height for proper standing
        if 'GR1' in robot_name:
            try:
                z_offset = 0.95
                if self.env.sim.model.nq >= 7:
                    self.env.sim.data.qpos[2] = z_offset
                    self.env.sim.forward()
                    print(f"Set {robot_name} base height to z={z_offset}m")
            except Exception as e:
                print(f"Warning: Could not adjust robot height: {e}")

        print(f"Total joints: {self.num_joints}")
        print(f"Active joints ({active_arm} arm): {self.active_joint_indices}")
        print(f"Fixed joints: {self.fixed_joint_indices}")
        print(f"Shoulder pitch joint: {self.shoulder_pitch_idx}")

    def _parse_fixed_indices(self, fixed_indices_str):
        """Parse fixed joint indices string like '0-5, 13-19'."""
        fixed_indices = []
        for part in fixed_indices_str.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                fixed_indices.extend(range(int(start.strip()), int(end.strip()) + 1))
            else:
                fixed_indices.append(int(part))
        return sorted(set(idx for idx in fixed_indices if 0 <= idx < self.num_joints))

    def _get_root_position(self):
        try:
            root_body = self.robot.robot_model.root_body
            return self.env.sim.data.get_body_xpos(root_body).copy()
        except Exception:
            return np.array([0.0, 0.0, 0.0])

    def _set_joint_positions(self, joint_positions_rad):
        self.robot.set_robot_joint_positions(joint_positions_rad)
        self.env.sim.forward()

    def _get_ee_position(self):
        try:
            pos_dict = self.robot._hand_pos
            return pos_dict[self.active_arm].copy()
        except Exception:
            return np.array([0.0, 0.0, 0.0])

    def _get_ee_orientation_rpy(self):
        try:
            orn_dict = self.robot._hand_orn
            rot_mat = orn_dict[self.active_arm]
            return T.mat2euler(rot_mat)
        except Exception:
            return np.array([0.0, 0.0, 0.0])

    def _capture_image(self, width=512, height=512, camera_name="frontview"):
        """Capture current camera view."""
        obs = self.env.sim.render(camera_name=camera_name, width=width, height=height, depth=False)
        return obs[::-1]

    # ─── Step 1: Brute-force export ───────────────────────────────────────────

    def export_all_poses(
        self,
        angle_step_deg: float = 90.0,
        angle_min_deg: float = -90.0,
        angle_max_deg: float = 90.0,
        shoulder_forward_deg: float = 30.0,
        output_file: Optional[str] = None,
    ) -> List[Dict]:
        """
        Generate ALL poses with shoulder forward bias.

        The shoulder_pitch joint gets an offset so the arm starts slightly
        forward, avoiding self-collision with the torso.

        Returns:
            List of all pose entries (also saved to JSONL).
        """
        print("\n" + "=" * 60)
        print("STEP 1: BRUTE-FORCE POSE GENERATION")
        print("=" * 60)
        print(f"Robot: {self.robot_name} | Arm: {self.active_arm}")
        print(f"Angle range: [{angle_min_deg}, {angle_max_deg}] step={angle_step_deg}")
        print(f"Shoulder forward offset: {shoulder_forward_deg} deg")
        print("=" * 60)

        angle_min = np.deg2rad(angle_min_deg)
        angle_max = np.deg2rad(angle_max_deg)
        angle_step = np.deg2rad(angle_step_deg)
        default_angles = np.arange(angle_min, angle_max + angle_step / 2, angle_step)

        shoulder_offset_rad = np.deg2rad(shoulder_forward_deg)

        # Build per-joint angle arrays
        joint_angle_arrays = []
        joint_names = []
        for active_idx in self.active_joint_indices:
            joint_names.append(f"joint_{active_idx}")
            if active_idx == self.shoulder_pitch_idx:
                # Shift shoulder pitch range forward
                shifted = default_angles + shoulder_offset_rad
                joint_angle_arrays.append(shifted)
                print(f"  joint_{active_idx} (shoulder_pitch): shifted by {shoulder_forward_deg} deg "
                      f"→ [{np.rad2deg(shifted[0]):.0f}, ..., {np.rad2deg(shifted[-1]):.0f}]")
            else:
                joint_angle_arrays.append(default_angles)

        num_per_joint = [len(a) for a in joint_angle_arrays]
        total = 1
        for n in num_per_joint:
            total *= n
        print(f"\nTotal combinations: {total:,}")

        all_combos = list(product(*[range(n) for n in num_per_joint]))

        if output_file is None:
            output_file = f"data/poses/{self.robot_name}/all_{self.robot_name}_{self.active_arm}_poses.jsonl"
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

        all_poses = []
        with open(output_file, 'w') as f:
            for combo_idx, angle_indices in tqdm(enumerate(all_combos), total=len(all_combos), desc="Generating"):
                joint_pos = self.initial_joint_pos.copy()
                angle_values = []
                for i, active_idx in enumerate(self.active_joint_indices):
                    val = joint_angle_arrays[i][angle_indices[i]]
                    joint_pos[active_idx] = val
                    angle_values.append(val)

                self._set_joint_positions(joint_pos)

                rpy = self._get_ee_orientation_rpy()
                root_pos = self._get_root_position()
                ee_pos = self._get_ee_position()
                distance = np.linalg.norm(ee_pos - root_pos)
                x_diff = float(ee_pos[0] - root_pos[0])
                y_diff = float(ee_pos[1] - root_pos[1])
                z_diff = float(ee_pos[2] - root_pos[2])

                roll_deg = float(np.rad2deg(rpy[0]))
                pitch_deg = float(np.rad2deg(rpy[1]))
                yaw_deg = float(np.rad2deg(rpy[2]))

                dir_label = self._classify_direction(roll_deg, pitch_deg, yaw_deg)
                pitch_label = self._classify_pitch(pitch_deg)

                x_region = "high" if x_diff > 0.2 else ("low" if x_diff < -0.2 else "medium")
                y_region = "high" if y_diff > 0.2 else ("low" if y_diff < -0.2 else "medium")
                z_region = "high" if z_diff > 0.2 else ("low" if z_diff < -0.2 else "medium")

                angles_str = "_".join(
                    f"j{self.active_joint_indices[j]}{int(np.rad2deg(angle_values[j])):+04d}"
                    for j in range(len(angle_indices))
                )

                entry = {
                    "robot": self.robot_name,
                    "active_arm": self.active_arm,
                    "pose_id": combo_idx,
                    "angles_str": angles_str,
                    "joint_angles_deg": [float(np.rad2deg(v)) for v in angle_values],
                    "joint_angles_rad": [float(v) for v in angle_values],
                    "active_joint_indices": self.active_joint_indices,
                    "joint_names": joint_names,
                    "orientation": {
                        "roll_deg": roll_deg, "pitch_deg": pitch_deg, "yaw_deg": yaw_deg,
                        "roll_rad": float(rpy[0]), "pitch_rad": float(rpy[1]), "yaw_rad": float(rpy[2]),
                    },
                    "dir": dir_label,
                    "pitch": pitch_label,
                    "root_position": {"x": float(root_pos[0]), "y": float(root_pos[1]), "z": float(root_pos[2])},
                    "ee_position": {"x": float(ee_pos[0]), "y": float(ee_pos[1]), "z": float(ee_pos[2])},
                    "x_diff": x_diff, "y_diff": y_diff, "z_diff": z_diff,
                    "x_region": x_region, "y_region": y_region, "z_region": z_region,
                    "root_to_ee_distance": float(distance),
                    "is_front": bool(ee_pos[0] > root_pos[0]),
                    "arm": self.active_arm,
                }
                f.write(json.dumps(entry) + '\n')
                all_poses.append(entry)

                if (combo_idx + 1) % 500 == 0:
                    self._set_joint_positions(self.initial_joint_pos)

        print(f"\nExported {len(all_poses):,} poses → {output_file}")

        # Print direction distribution
        dir_counts = {}
        for p in all_poses:
            d = p["dir"]
            dir_counts[d] = dir_counts.get(d, 0) + 1
        print("Direction distribution:")
        for d, c in sorted(dir_counts.items(), key=lambda x: -x[1]):
            print(f"  {d}: {c}")

        return all_poses

    # ─── Step 2: Query closest poses ──────────────────────────────────────────

    def _score_and_select_poses(
        self,
        all_poses: List[Dict],
        direction_poses_list: List[Dict],
        pitch_values: List[float],
        height_val: str,
        max_diff_rad: float,
        top_k: int,
        pose_name: str,
        seen_ids: set,
        selected_poses: List[Dict],
    ) -> int:
        """
        Score and select closest poses for a given direction/height/pitch combination.

        Returns:
            Number of new poses added (0 means no matching poses found).
        """
        added = 0
        for dir_pose, pitch_val in product(direction_poses_list, pitch_values):
            target_roll = np.deg2rad(dir_pose['roll'])
            target_pitch = np.deg2rad(pitch_val)
            target_yaw = np.deg2rad(dir_pose['yaw'])

            # Filter and score poses
            scored = []
            for p in all_poses:
                orn = p["orientation"]

                # Orientation distance
                diff = 0.0
                for actual, target in [
                    (orn["roll_rad"], target_roll),
                    (orn["pitch_rad"], target_pitch),
                    (orn["yaw_rad"], target_yaw),
                ]:
                    d = abs(actual - target)
                    d = min(d, 2 * np.pi - d)
                    diff += d

                if diff > max_diff_rad:
                    continue

                # Height filter
                if height_val == "high" and p["z_region"] != "high":
                    continue
                elif height_val == "low" and p["z_region"] != "low":
                    continue
                elif height_val == "medium" and p["z_region"] != "medium":
                    continue

                scored.append((diff, p))

            scored.sort(key=lambda x: x[0])

            for _, p in scored[:top_k]:
                if p["pose_id"] not in seen_ids:
                    entry = dict(p)
                    entry["pose_name"] = pose_name
                    entry["target_roll_deg"] = dir_pose['roll']
                    entry["target_pitch_deg"] = pitch_val
                    entry["target_yaw_deg"] = dir_pose['yaw']
                    entry["target_height"] = height_val
                    entry["orientation_diff_rad"] = scored[0][0] if scored else 0
                    entry["orientation_diff_deg"] = float(np.rad2deg(entry["orientation_diff_rad"]))
                    selected_poses.append(entry)
                    seen_ids.add(p["pose_id"])
                    added += 1
        return added

    def query_closest_poses(
        self,
        all_poses: List[Dict],
        top_k: int = 30,
        max_orientation_diff_deg: float = 60.0,
        output_file: Optional[str] = None,
    ) -> List[Dict]:
        """
        Select closest poses for each direction/height/pitch combination
        from arm_pose_config. All done in-memory (no subprocess calls).

        Pitch handling:
            1. First try with the specific pitch (horizontal or vertical).
            2. If no poses found, fallback to trying all pitch values.

        Returns:
            List of selected closest poses (also saved to JSONL).
        """
        print("\n" + "=" * 60)
        print("STEP 2: QUERY CLOSEST POSES")
        print("=" * 60)

        if output_file is None:
            output_file = f"data/poses/{self.robot_name}/closest_{self.robot_name}_{self.active_arm}_poses.jsonl"
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

        max_diff_rad = np.deg2rad(max_orientation_diff_deg)
        selected_poses = []
        seen_ids = set()
        query_count = 0
        pitch_fallback_count = 0

        for pose_name, pose_config in direction_pose_set.items():
            height_val = height_map[pose_config['height']]
            direction_name = pose_config['dir']
            ee_pitch_name = pose_config['pitch']

            direction_poses_list = poses[direction_name]

            # Step 1: Try with specific pitch (horizontal or vertical)
            specific_pitch_values = pitch_poses[ee_pitch_name]
            query_count += len(direction_poses_list) * len(specific_pitch_values)

            added = self._score_and_select_poses(
                all_poses, direction_poses_list, specific_pitch_values,
                height_val, max_diff_rad, top_k, pose_name, seen_ids, selected_poses,
            )

            # Step 2: Fallback — if no poses found, ignore pitch type and try all pitch values
            if added == 0:
                all_pitch_values = [p for v in pitch_poses.values() for p in v]
                query_count += len(direction_poses_list) * len(all_pitch_values)

                fallback_added = self._score_and_select_poses(
                    all_poses, direction_poses_list, all_pitch_values,
                    height_val, max_diff_rad, top_k, pose_name, seen_ids, selected_poses,
                )
                if fallback_added > 0:
                    pitch_fallback_count += 1
                    print(f"  {pose_name}: no poses with pitch={ee_pitch_name}, "
                          f"fallback to all pitches → found {fallback_added}")

        # Save
        with open(output_file, 'w') as f:
            for entry in selected_poses:
                f.write(json.dumps(entry) + '\n')

        print(f"Queries: {query_count}")
        print(f"Selected unique poses: {len(selected_poses)}")
        if pitch_fallback_count > 0:
            print(f"Pitch fallback used: {pitch_fallback_count} pose configs (no match with specific pitch)")
        print(f"Saved → {output_file}")

        dir_counts = {}
        for p in selected_poses:
            d = p["dir"]
            dir_counts[d] = dir_counts.get(d, 0) + 1
        print("Direction distribution (closest):")
        for d, c in sorted(dir_counts.items(), key=lambda x: -x[1]):
            print(f"  {d}: {c}")

        return selected_poses

    # ─── Step 3: Render tiled PNGs ────────────────────────────────────────────

    def render_tiled_pngs(
        self,
        all_poses: List[Dict],
        directions: List[str] = None,
        camera_views: List[str] = None,
        poses_per_direction: int = 16,
        tile_size: int = 256,
        output_dir: Optional[str] = None,
    ):
        """
        Render tiled PNG previews for specified directions.

        For each direction, selects representative poses, sets the robot,
        captures images from multiple camera views (front + side), and
        arranges them into a grid. Each tile shows all camera views
        side by side.

        Args:
            all_poses: List of all pose entries
            directions: List of directions to render (default: front, right, left, down)
            camera_views: Camera names to render per pose (default: frontview + sideview)
            poses_per_direction: Number of poses per direction
            tile_size: Pixel size per camera view
            output_dir: Output directory for PNGs
        """
        if not self.has_offscreen_renderer:
            print("Skipping tiled PNG: offscreen renderer not enabled")
            return

        if directions is None:
            directions = ["front", "right", "left", "down"]
        if camera_views is None:
            camera_views = ["frontview", "sideview_flip"]

        if output_dir is None:
            output_dir = f"data/poses/{self.robot_name}"
        os.makedirs(output_dir, exist_ok=True)

        num_views = len(camera_views)
        # Each tile = multiple camera views side by side
        cell_w = tile_size * num_views
        cell_h = tile_size

        print("\n" + "=" * 60)
        print("STEP 3: RENDER TILED PNGs")
        print("=" * 60)
        print(f"Directions: {directions}")
        print(f"Camera views: {camera_views}")
        print(f"Poses per direction: {poses_per_direction}")
        print(f"Tile size: {tile_size} x {num_views} = {cell_w} per pose")

        # Load font
        font = None
        for fpath in ["/System/Library/Fonts/Helvetica.ttc",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
            try:
                font = ImageFont.truetype(fpath, 14)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()

        font_small = None
        for fpath in ["/System/Library/Fonts/Helvetica.ttc",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            try:
                font_small = ImageFont.truetype(fpath, 11)
                break
            except Exception:
                continue
        if font_small is None:
            font_small = font

        for direction in directions:
            # Filter poses for this direction
            dir_poses = [p for p in all_poses if p["dir"] == direction]
            if not dir_poses:
                print(f"  {direction}: no poses found, skipping")
                continue

            # Sort: prefer front-facing, then by root-to-ee distance (larger = more extended)
            dir_poses.sort(key=lambda p: (-int(p.get("is_front", False)), -p["root_to_ee_distance"]))
            selected = dir_poses[:poses_per_direction]

            # Calculate grid
            n = len(selected)
            cols = int(math.ceil(math.sqrt(n)))
            rows = int(math.ceil(n / cols))

            canvas_w = cols * cell_w
            canvas_h = rows * cell_h
            canvas = Image.new("RGB", (canvas_w, canvas_h), (40, 40, 40))
            draw = ImageDraw.Draw(canvas)

            print(f"  {direction}: {n} poses → {cols}x{rows} grid ({canvas_w}x{canvas_h} px)")

            for idx, pose in enumerate(tqdm(selected, desc=f"  Rendering {direction}", leave=False)):
                # Set robot to this pose
                joint_pos = self.initial_joint_pos.copy()
                for i, active_idx in enumerate(pose["active_joint_indices"]):
                    if i < len(pose["joint_angles_rad"]) and active_idx < len(joint_pos):
                        joint_pos[active_idx] = pose["joint_angles_rad"][i]
                self._set_joint_positions(joint_pos)

                row, col = idx // cols, idx % cols
                base_x, base_y = col * cell_w, row * cell_h

                # Capture from each camera view
                for v_idx, cam_name in enumerate(camera_views):
                    img_array = self._capture_image(width=tile_size, height=tile_size, camera_name=cam_name)
                    img = Image.fromarray(img_array)
                    vx = base_x + v_idx * tile_size
                    canvas.paste(img, (vx, base_y))

                    # Camera label (small, top-right of each view)
                    cam_label = cam_name.replace("view", "")
                    try:
                        bbox = draw.textbbox((0, 0), cam_label, font=font_small)
                        clw = bbox[2] - bbox[0]
                        clh = bbox[3] - bbox[1]
                    except Exception:
                        clw, clh = len(cam_label) * 7, 12
                    draw.rectangle([vx + tile_size - clw - 6, base_y + 2, vx + tile_size - 2, base_y + clh + 6], fill=(0, 0, 0, 180))
                    draw.text((vx + tile_size - clw - 4, base_y + 4), cam_label, fill=(180, 180, 180), font=font_small)

                # Info label (on the first view, top-left)
                orn = pose["orientation"]
                label = f"id:{pose['pose_id']} r:{orn['roll_deg']:.0f} p:{orn['pitch_deg']:.0f} y:{orn['yaw_deg']:.0f}"
                try:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except Exception:
                    tw, th = len(label) * 8, 14
                draw.rectangle([base_x + 2, base_y + 2, base_x + tw + 6, base_y + th + 6], fill=(0, 0, 0))
                draw.text((base_x + 4, base_y + 4), label, fill=(255, 255, 0), font=font)

                # Height label (bottom-left of first view)
                h_label = f"z:{pose['z_region']}  x_diff:{pose['x_diff']:.2f}"
                draw.rectangle([base_x + 2, base_y + tile_size - th - 6, base_x + len(h_label) * 8 + 6, base_y + tile_size - 2], fill=(0, 0, 0))
                draw.text((base_x + 4, base_y + tile_size - th - 4), h_label, fill=(200, 200, 200), font=font)

            # Save
            png_path = os.path.join(output_dir, f"{self.robot_name}_{self.active_arm}_{direction}_poses.png")
            canvas.save(png_path, quality=95)
            file_mb = os.path.getsize(png_path) / (1024 ** 2)
            print(f"  Saved: {png_path} ({file_mb:.1f} MB)")

        # Reset robot
        self._set_joint_positions(self.initial_joint_pos)
        print("Done rendering tiled PNGs.\n")

    def render_pitch_samples(
        self,
        all_poses: List[Dict],
        camera_views: List[str] = None,
        tile_size: int = 512,
        output_dir: Optional[str] = None,
    ):
        """
        Render one horizontal and one vertical sample pose as PNG for verification.

        Picks one representative pose for each pitch type, renders front+side
        views side by side, and saves as separate PNGs with pitch info overlay.
        """
        if not self.has_offscreen_renderer:
            print("Skipping pitch sample PNGs: offscreen renderer not enabled")
            return

        if camera_views is None:
            camera_views = ["frontview", "sideview_flip"]
        if output_dir is None:
            output_dir = f"data/poses/{self.robot_name}"
        os.makedirs(output_dir, exist_ok=True)

        num_views = len(camera_views)

        # Load font
        font = None
        for fpath in ["/System/Library/Fonts/Helvetica.ttc",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
            try:
                font = ImageFont.truetype(fpath, 18)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()

        font_small = None
        for fpath in ["/System/Library/Fonts/Helvetica.ttc",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            try:
                font_small = ImageFont.truetype(fpath, 13)
                break
            except Exception:
                continue
        if font_small is None:
            font_small = font

        print("\n" + "=" * 60)
        print("PITCH SAMPLE PNGs (horizontal vs vertical)")
        print("=" * 60)

        for pitch_type in ["horizontal", "vertical"]:
            # Filter poses by pitch label and prefer front-facing, extended arm
            candidates = [p for p in all_poses if p.get("pitch") == pitch_type]
            if not candidates:
                print(f"  {pitch_type}: no poses found, skipping")
                continue

            # Pick the most representative: front-facing, most extended arm
            candidates.sort(key=lambda p: (
                -int(p.get("is_front", False)),
                -p.get("root_to_ee_distance", 0),
            ))
            pose = candidates[0]

            # Set robot to this pose
            joint_pos = self.initial_joint_pos.copy()
            for i, active_idx in enumerate(pose["active_joint_indices"]):
                if i < len(pose["joint_angles_rad"]) and active_idx < len(joint_pos):
                    joint_pos[active_idx] = pose["joint_angles_rad"][i]
            self._set_joint_positions(joint_pos)

            # Render side-by-side camera views
            canvas_w = tile_size * num_views
            canvas_h = tile_size
            canvas = Image.new("RGB", (canvas_w, canvas_h), (40, 40, 40))
            draw = ImageDraw.Draw(canvas)

            for v_idx, cam_name in enumerate(camera_views):
                img_array = self._capture_image(width=tile_size, height=tile_size, camera_name=cam_name)
                img = Image.fromarray(img_array)
                vx = v_idx * tile_size
                canvas.paste(img, (vx, 0))

                # Camera label (top-right)
                cam_label = cam_name.replace("view", "")
                try:
                    bbox = draw.textbbox((0, 0), cam_label, font=font_small)
                    clw = bbox[2] - bbox[0]
                    clh = bbox[3] - bbox[1]
                except Exception:
                    clw, clh = len(cam_label) * 8, 14
                draw.rectangle([vx + tile_size - clw - 8, 4, vx + tile_size - 2, clh + 10], fill=(0, 0, 0, 200))
                draw.text((vx + tile_size - clw - 5, 6), cam_label, fill=(180, 180, 180), font=font_small)

            # Title overlay (top-left, spanning first view)
            orn = pose["orientation"]
            title = f"pitch={pitch_type.upper()}"
            detail = (f"id:{pose['pose_id']}  dir:{pose.get('dir','?')}  "
                      f"r:{orn['roll_deg']:.0f} p:{orn['pitch_deg']:.0f} y:{orn['yaw_deg']:.0f}  "
                      f"z:{pose.get('z_region','?')}")
            color = (0, 255, 128) if pitch_type == "horizontal" else (128, 180, 255)

            try:
                bbox_t = draw.textbbox((0, 0), title, font=font)
                tw, th = bbox_t[2] - bbox_t[0], bbox_t[3] - bbox_t[1]
                bbox_d = draw.textbbox((0, 0), detail, font=font_small)
                dw, dh = bbox_d[2] - bbox_d[0], bbox_d[3] - bbox_d[1]
            except Exception:
                tw, th = len(title) * 10, 18
                dw, dh = len(detail) * 8, 14

            box_w = max(tw, dw) + 16
            box_h = th + dh + 20
            draw.rectangle([4, 4, 4 + box_w, 4 + box_h], fill=(0, 0, 0))
            draw.text((12, 8), title, fill=color, font=font)
            draw.text((12, 12 + th), detail, fill=(220, 220, 220), font=font_small)

            # Save
            png_path = os.path.join(
                output_dir,
                f"{self.robot_name}_{self.active_arm}_pitch_sample_{pitch_type}.png"
            )
            canvas.save(png_path, quality=95)
            file_kb = os.path.getsize(png_path) / 1024
            print(f"  {pitch_type}: pose_id={pose['pose_id']}  "
                  f"raw_pitch={orn['pitch_deg']:.1f}°  → {png_path} ({file_kb:.0f} KB)")

        # Reset robot
        self._set_joint_positions(self.initial_joint_pos)
        print("Done rendering pitch samples.\n")

    def close(self):
        self.env.close()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main(
    robot: str = "GR1",
    active_arm: str = "right",
    angle_step: float = 90.0,
    angle_min: float = -90.0,
    angle_max: float = 90.0,
    shoulder_forward_deg: float = 0.0,
    top_k: int = 30,
    max_orientation_diff_deg: float = 60.0,
    skip_tiled_png: bool = False,
    poses_per_direction: int = 16,
    tile_size: int = 256,
    directions: str = "front,right,left,down",
    camera_views: str = "frontview,sideview_flip",
):
    """
    Export and query humanoid poses in one step.

    Args:
        robot: Robot name (GR1, GR1ArmsOnly, GR1FixedLowerBody, GR1FloatingBody)
        active_arm: Which arm ("right" or "left")
        angle_step: Angle step size in degrees for brute force
        angle_min: Minimum angle in degrees
        angle_max: Maximum angle in degrees
        shoulder_forward_deg: Shoulder pitch offset (degrees) to bias arm forward.
                              Positive = forward for GR1. Set 0 to disable.
        top_k: Number of closest poses per direction/height/pitch query
        max_orientation_diff_deg: Max orientation difference for closest query
        skip_tiled_png: Skip rendering tiled PNG previews
        poses_per_direction: Number of example poses per direction in tiled PNG
        tile_size: Pixel size of each tile in tiled PNG
        directions: Comma-separated directions for tiled PNG (e.g. "front,right,left,down")
        camera_views: Comma-separated camera names per tile (e.g. "frontview,sideview")

    Examples:
        python adhoc/humanoid/export_humanoid_poses.py --robot GR1 --active-arm right
        python adhoc/humanoid/export_humanoid_poses.py --robot GR1 --active-arm right --shoulder-forward-deg 45
        python adhoc/humanoid/export_humanoid_poses.py --robot GR1 --active-arm right --skip-tiled-png
    """
    print("=" * 60)
    print("HUMANOID POSE EXPORT + QUERY")
    print("=" * 60)
    print(f"Robot: {robot} | Arm: {active_arm}")
    print(f"Shoulder forward offset: {shoulder_forward_deg} deg")
    print("=" * 60)

    need_renderer = not skip_tiled_png
    exporter = HumanoidPoseExporter(
        robot_name=robot,
        active_arm=active_arm,
        has_offscreen_renderer=need_renderer,
    )

    try:
        # Step 1: Brute-force export
        all_poses = exporter.export_all_poses(
            angle_step_deg=angle_step,
            angle_min_deg=angle_min,
            angle_max_deg=angle_max,
            shoulder_forward_deg=shoulder_forward_deg,
        )

        # Step 2: Query closest poses
        closest_poses = exporter.query_closest_poses(
            all_poses=all_poses,
            top_k=top_k,
            max_orientation_diff_deg=max_orientation_diff_deg,
        )

        # Step 3: Tiled PNGs
        if not skip_tiled_png:
            dir_list = [d.strip() for d in directions.split(",")]
            cam_list = [c.strip() for c in camera_views.split(",")]
            exporter.render_tiled_pngs(
                all_poses=all_poses,
                directions=dir_list,
                camera_views=cam_list,
                poses_per_direction=poses_per_direction,
                tile_size=tile_size,
            )

        # Step 4: Pitch sample PNGs (horizontal vs vertical verification)
        if not skip_tiled_png:
            cam_list = [c.strip() for c in camera_views.split(",")]
            exporter.render_pitch_samples(
                all_poses=all_poses,
                camera_views=cam_list,
                tile_size=tile_size,
            )

        print("\n" + "=" * 60)
        print("ALL DONE")
        print("=" * 60)
        print(f"All poses:     data/poses/{robot}/all_{robot}_{active_arm}_poses.jsonl ({len(all_poses)} entries)")
        print(f"Closest poses: data/poses/{robot}/closest_{robot}_{active_arm}_poses.jsonl ({len(closest_poses)} entries)")
        if not skip_tiled_png:
            print(f"Tiled PNGs:    data/poses/{robot}/{robot}_{active_arm}_{{direction}}_poses.png")
            print(f"Pitch samples: data/poses/{robot}/{robot}_{active_arm}_pitch_sample_horizontal.png")
            print(f"               data/poses/{robot}/{robot}_{active_arm}_pitch_sample_vertical.png")
        print("=" * 60)

    finally:
        exporter.close()
        print("\nDone!")


if __name__ == "__main__":
    fire.Fire(main)
