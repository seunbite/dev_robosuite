"""Render 6 directions × 2 orientations = 12 example poses as a PNG grid."""

import os
import sys
import json
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from motion_generation import MotionGenerator, _select_initial_poses

for name in logging.root.manager.loggerDict:
    if "robosuite" in name:
        logging.getLogger(name).setLevel(logging.ERROR)
logging.getLogger("robosuite").setLevel(logging.ERROR)


def main(
    robot="IIWA",
    jsonl_path="data/seed/_remainder/closest_poses_results.jsonl",
    output_path="data/motions/pose_grid_6dir_2orient.png",
):
    directions = ["front", "back", "left", "right", "up", "down"]
    orientations = ["vertical", "horizontal"]

    gen = MotionGenerator(
        robot_name=robot,
        env_name="EmptySpace",
        controller_name="IK_POSE",
        jsonl_path=jsonl_path,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_distance=1.8,
    )

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        bold = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except Exception:
        font = bold = ImageFont.load_default()

    images = {}

    for d in directions:
        for o in orientations:
            pose_def = {"dir": d, "gripper_orientation": o}
            matching = gen._find_matching_poses(pose_def)
            if not matching:
                print(f"  {d}/{o}: NO POSES")
                continue

            selected = _select_initial_poses(matching, pose_def, top_k=1)
            if not selected:
                print(f"  {d}/{o}: selection failed")
                continue

            pose = selected[0]
            joint_pos = gen._pose_data_to_joint_positions(pose)
            gen._set_joint_positions(joint_pos)

            img_arr = gen._capture_image()
            img = Image.fromarray(img_arr)

            draw = ImageDraw.Draw(img)
            label = f"{d} / {o}"
            pid = pose.get("pose_id", "?")
            sub = f"p{pid}"

            tw = draw.textlength(label, font=bold) if hasattr(draw, 'textlength') else 150
            sw = draw.textlength(sub, font=font) if hasattr(draw, 'textlength') else 50

            draw.rectangle([0, 0, tw + 16, 52], fill=(0, 0, 0, 200))
            draw.text((8, 4), label, fill="white", font=bold)
            draw.text((8, 28), sub, fill=(180, 220, 255), font=font)

            images[(d, o)] = img
            print(f"  {d:6s}/{o:12s} → p{pid}")

            gen._set_joint_positions(gen.initial_joint_pos)

    if not images:
        print("No images generated")
        gen.close()
        return

    ncols = len(orientations)
    nrows = len(directions)
    sample = next(iter(images.values()))
    tw, th = sample.size

    col_header_h = 40
    row_header_w = 100
    total_w = row_header_w + ncols * tw
    total_h = col_header_h + nrows * th

    grid = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    for ci, o in enumerate(orientations):
        x = row_header_w + ci * tw + tw // 2
        ow = draw.textlength(o, font=bold) if hasattr(draw, 'textlength') else 80
        draw.text((x - ow // 2, 10), o, fill="black", font=bold)

    for ri, d in enumerate(directions):
        y = col_header_h + ri * th + th // 2 - 10
        draw.text((8, y), d, fill="black", font=bold)

    for ri, d in enumerate(directions):
        for ci, o in enumerate(orientations):
            img = images.get((d, o))
            if img:
                x = row_header_w + ci * tw
                y = col_header_h + ri * th
                grid.paste(img, (x, y))
            draw.line(
                [(row_header_w + ci * tw, col_header_h), (row_header_w + ci * tw, total_h)],
                fill="gray", width=1,
            )
        draw.line(
            [(0, col_header_h + ri * th), (total_w, col_header_h + ri * th)],
            fill="gray", width=1,
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid.save(output_path, quality=95)
    print(f"\nSaved: {output_path} ({os.path.getsize(output_path) / 1024:.0f}KB)")
    print(f"Grid: {nrows} rows × {ncols} cols, {total_w}×{total_h}px")

    gen.close()


if __name__ == "__main__":
    import fire
    fire.Fire(main)
