"""Generate verification tile PNGs for gripper orientation classification.

Reads the JSONL pose database, renders a sample of poses per (robot, direction,
gripper_orientation) combo, and saves tiled images to data/debug_gripper_orientation/.
"""

import os
import sys
import json
import numpy as np
from itertools import groupby
from operator import itemgetter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from PIL import Image, ImageDraw, ImageFont

DIRECTIONS = ["up", "down", "front", "back", "left", "right"]
MAX_POSES_PER_GROUP = 10
TILE_W, TILE_H = 256, 256


def load_poses_from_jsonl(jsonl_path, robot_name):
    """Load all poses for a given robot from the JSONL file."""
    poses = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("robot") == robot_name:
                poses.append(entry)
    return poses


def setup_robot(robot_name):
    """Create a robosuite environment with offscreen rendering."""
    arm_controller_config = suite.load_part_controller_config(default_controller="OSC_POSE")
    controller_configs = refactor_composite_controller_config(
        arm_controller_config, robot_name, ["right", "left"]
    )
    env = suite.make(
        env_name="EmptySpace",
        robots=robot_name,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        controller_configs=controller_configs,
        horizon=1000,
    )
    env.reset()
    return env


def render_pose(env, joint_angles_rad, active_joint_indices):
    """Set joint positions and render a frame."""
    robot = env.robots[0]
    joint_pos = robot._joint_positions.copy()
    for i, idx in enumerate(active_joint_indices):
        if i < len(joint_angles_rad):
            joint_pos[idx] = joint_angles_rad[i]
            is_bimanual = any(arm in robot.name.lower() for arm in ["gr1", "bimanual"])
            if is_bimanual and len(robot._joint_positions) > idx + len(active_joint_indices):
                mirror_idx = idx + len(active_joint_indices)
                if mirror_idx < len(joint_pos):
                    joint_pos[mirror_idx] = joint_angles_rad[i]

    sim = env.sim
    for idx_j, val in enumerate(joint_pos):
        if idx_j < sim.data.qpos.shape[0]:
            sim.data.qpos[idx_j] = val
    sim.forward()

    frame = sim.render(width=TILE_W, height=TILE_H, camera_name="frontview")
    return Image.fromarray(np.flipud(frame))


def create_tile_image(images_v, images_h, robot_name, direction, output_path):
    """Create a tiled image with VERTICAL on top and HORIZONTAL on bottom."""
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()

    n_v = len(images_v)
    n_h = len(images_h)
    cols = max(n_v, n_h, 1)
    header_h = 30
    total_w = cols * TILE_W
    total_h = 2 * (header_h + TILE_H)

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([(0, 0), (total_w, header_h)], fill=(0, 100, 0))
    draw.text((10, 5), f"VERTICAL ({n_v} poses)", fill="white", font=font)
    for i, img in enumerate(images_v):
        canvas.paste(img.resize((TILE_W, TILE_H)), (i * TILE_W, header_h))

    y_offset = header_h + TILE_H
    draw.rectangle([(0, y_offset), (total_w, y_offset + header_h)], fill=(0, 0, 100))
    draw.text((10, y_offset + 5), f"HORIZONTAL ({n_h} poses)", fill="white", font=font)
    for i, img in enumerate(images_h):
        canvas.paste(img.resize((TILE_W, TILE_H)), (i * TILE_W, y_offset + header_h))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)
    print(f"  Saved: {output_path}")


def main():
    import fire

    def run(
        jsonl_path="data/seed/_remainder/closest_poses_results.jsonl",
        robots=None,
        directions=None,
        output_dir="data/debug_gripper_orientation",
    ):
        if robots is None:
            robots = ["IIWA", "Panda", "XArm7"]
        if directions is None:
            directions = DIRECTIONS

        all_poses = {}
        for robot in robots:
            all_poses[robot] = load_poses_from_jsonl(jsonl_path, robot)
            print(f"Loaded {len(all_poses[robot])} poses for {robot}")

        for robot in robots:
            print(f"\n{'='*60}")
            print(f"Rendering {robot}")
            print(f"{'='*60}")

            env = setup_robot(robot)

            for direction in directions:
                dir_poses = [p for p in all_poses[robot] if p.get("dir") == direction]
                if not dir_poses:
                    print(f"  {direction}: no poses, skipping")
                    continue

                v_poses = [p for p in dir_poses if p.get("gripper_orientation") == "vertical"]
                h_poses = [p for p in dir_poses if p.get("gripper_orientation") == "horizontal"]

                v_sample = v_poses[:MAX_POSES_PER_GROUP]
                h_sample = h_poses[:MAX_POSES_PER_GROUP]

                print(f"  {direction}: V={len(v_poses)} H={len(h_poses)}, rendering V={len(v_sample)} H={len(h_sample)}")

                images_v = []
                for p in v_sample:
                    img = render_pose(
                        env,
                        p["joint_angles_rad"],
                        p["active_joint_indices"],
                    )
                    images_v.append(img)

                images_h = []
                for p in h_sample:
                    img = render_pose(
                        env,
                        p["joint_angles_rad"],
                        p["active_joint_indices"],
                    )
                    images_h.append(img)

                out_path = os.path.join(output_dir, f"{robot}_{direction}_v3.png")
                create_tile_image(images_v, images_h, robot, direction, out_path)

            env.close()

    fire.Fire(run)


if __name__ == "__main__":
    main()
