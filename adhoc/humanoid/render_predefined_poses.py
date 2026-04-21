#!/usr/bin/env python3
"""
Render all 12 predefined humanoid arm poses as a single tiled image.

6 directions (front, back, right, left, up, down) × 2 pitches (vertical, horizontal)
= 12 poses rendered with front + side views, labeled with direction, pitch, and angles.

Usage:
    python adhoc/humanoid/render_predefined_poses.py
    python adhoc/humanoid/render_predefined_poses.py --robot GR1 --arm right
"""

import os
import sys
import math
import numpy as np
import fire
from PIL import Image, ImageDraw, ImageFont

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robotarm'))
from motion_generation import (
    HUMANOID_PREDEFINED_POSES,
    HUMANOID_ARM_JOINT_INDICES,
    HUMANOID_ARM_POSE_OFFSETS,
)

# Direction display order (rows)
DIRECTIONS = ['front', 'back', 'right', 'left', 'up', 'down']
PITCHES = ['vertical', 'horizontal']

CAMERA_VIEWS = ['frontview', 'sideview_flip']


def load_fonts():
    font, font_small, font_bold = None, None, None
    for fpath in ["/System/Library/Fonts/Helvetica.ttc",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            font_bold = ImageFont.truetype(fpath, 18)
            font = ImageFont.truetype(fpath, 13)
            font_small = ImageFont.truetype(fpath, 11)
            break
        except Exception:
            continue
    if font is None:
        font = font_small = font_bold = ImageFont.load_default()
    return font, font_small, font_bold


def main(robot="GR1", arm="right", tile_size=400, output_dir="data/poses"):
    """Render predefined poses as a tiled verification image.
    
    Args:
        robot: Robot name (GR1, GR1FixedLowerBody, etc.)
        arm: Active arm (right or left)
        tile_size: Size of each camera view in pixels
        output_dir: Output directory for the PNG
    """
    # ── Find predefined poses ──
    base_name = None
    for prefix in HUMANOID_PREDEFINED_POSES:
        if robot.startswith(prefix):
            base_name = prefix
            break
    if not base_name:
        print(f"No predefined poses for {robot}")
        return

    arm_poses = HUMANOID_PREDEFINED_POSES[base_name].get(arm, {})
    if not arm_poses:
        print(f"No predefined poses for {robot} {arm} arm")
        return

    joint_indices = HUMANOID_ARM_JOINT_INDICES.get(robot, HUMANOID_ARM_JOINT_INDICES.get(base_name, {})).get(arm)
    if not joint_indices:
        print(f"No joint indices for {robot} {arm}")
        return

    # ── Get elbow offset ──
    pose_offsets = {}
    if robot in HUMANOID_ARM_POSE_OFFSETS and arm in HUMANOID_ARM_POSE_OFFSETS[robot]:
        pose_offsets = HUMANOID_ARM_POSE_OFFSETS[robot][arm]
    elif base_name in HUMANOID_ARM_POSE_OFFSETS and arm in HUMANOID_ARM_POSE_OFFSETS[base_name]:
        pose_offsets = HUMANOID_ARM_POSE_OFFSETS[base_name][arm]

    # ── Create environment ──
    print(f"Initializing {robot} ({arm} arm)...")
    options = {
        "env_name": "EmptySpace",
        "robots": robot,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_camera_obs": False,
        "control_freq": 20,
    }
    arm_ctrl = suite.load_part_controller_config(default_controller="OSC_POSE")
    options["controller_configs"] = refactor_composite_controller_config(arm_ctrl, robot, ["right", "left"])

    env = suite.make(**options, horizon=1000)
    env.reset()
    rob = env.robots[0]
    initial_pos = rob._joint_positions.copy()

    # Fix height
    if 'GR1' in robot and env.sim.model.nq >= 7:
        env.sim.data.qpos[2] = 0.95
        env.sim.forward()

    print(f"  Joints: {len(initial_pos)}, Active arm indices: {joint_indices}")
    print(f"  Elbow offsets: {pose_offsets}")

    # ── Layout: rows=directions, cols=pitches, each cell = front+side ──
    num_views = len(CAMERA_VIEWS)
    cell_w = tile_size * num_views  # front + side per cell
    cell_h = tile_size
    label_h = 55  # space for text labels at top of each cell

    rows = len(DIRECTIONS)
    cols = len(PITCHES)
    canvas_w = cols * cell_w
    canvas_h = rows * (cell_h + label_h)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    font, font_small, font_bold = load_fonts()

    print(f"\nRendering {rows}×{cols} = {rows*cols} poses ({canvas_w}×{canvas_h} px)...")

    for r_idx, direction in enumerate(DIRECTIONS):
        for c_idx, pitch in enumerate(PITCHES):
            angles_deg = arm_poses.get((direction, pitch))
            if angles_deg is None:
                print(f"  SKIP: ({direction}, {pitch}) — not defined")
                continue

            angles_rad = [np.deg2rad(a) for a in angles_deg]

            # Set joint positions (with elbow offset applied)
            pos = initial_pos.copy()
            for i, idx in enumerate(joint_indices):
                if i < len(angles_rad) and idx < len(pos):
                    pos[idx] = angles_rad[i]
            # Apply elbow offset
            for joint_idx, offset in pose_offsets.items():
                if joint_idx < len(pos):
                    pos[joint_idx] += offset

            rob.set_robot_joint_positions(pos)
            env.sim.forward()
            for _ in range(10):
                env.sim.data.qvel[:] = 0
                env.sim.forward()

            # Compute actual elbow angle for display
            elbow_stored = angles_deg[3]  # j9 stored
            elbow_actual = elbow_stored + 180  # after offset

            # Render camera views
            x_base = c_idx * cell_w
            y_base = r_idx * (cell_h + label_h)

            # Draw label background
            draw.rectangle([x_base, y_base, x_base + cell_w, y_base + label_h], fill=(50, 50, 50))

            # Direction + pitch label
            dir_label = f"{direction.upper()} / {pitch}"
            draw.text((x_base + 10, y_base + 4), dir_label, fill="white", font=font_bold)

            # Stored angles
            angles_str = f"stored: [{', '.join(f'{a:+.0f}' for a in angles_deg)}]"
            draw.text((x_base + 10, y_base + 24), angles_str, fill=(180, 220, 255), font=font_small)

            # Actual elbow
            actual_str = f"elbow actual: {elbow_actual:.0f}°  (shoulder_pitch={angles_deg[0]:.0f}°)"
            draw.text((x_base + 10, y_base + 38), actual_str, fill=(180, 255, 180), font=font_small)

            # Render views
            for v_idx, cam_name in enumerate(CAMERA_VIEWS):
                img_arr = env.sim.render(camera_name=cam_name, width=tile_size, height=tile_size, depth=False)
                img = Image.fromarray(img_arr[::-1])
                px = x_base + v_idx * tile_size
                py = y_base + label_h
                canvas.paste(img, (px, py))

                # Camera label
                cam_label = "FRONT" if "front" in cam_name else "SIDE"
                draw.text((px + 5, py + 5), cam_label, fill=(255, 255, 100), font=font_small)

            # Separator lines
            if c_idx > 0:
                draw.line([(x_base, y_base), (x_base, y_base + cell_h + label_h)], fill=(100, 100, 100), width=2)
            if r_idx > 0:
                draw.line([(x_base, y_base), (x_base + cell_w, y_base)], fill=(100, 100, 100), width=2)

            print(f"  ✓ {direction}/{pitch}: stored={angles_deg}, elbow_actual={elbow_actual:.0f}°")

    # ── Column headers ──
    # (Already labeled per-cell, but add overall title)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{robot}_{arm}_predefined_poses_tiled.png")
    canvas.save(out_path)
    print(f"\n✓ Saved tiled image: {out_path}")
    print(f"  Size: {canvas_w}×{canvas_h} px")
    print(f"  Grid: {rows} dirs × {cols} pitches = {rows*cols} poses")

    env.close()
    return out_path


if __name__ == "__main__":
    fire.Fire(main)
