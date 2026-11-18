#!/usr/bin/env python3
"""Render FR3 sim-reset (robosuite init_qpos) with the same frontview as small10 GIFs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)

_REPO = Path(__file__).resolve().parents[3]
_OUT = _REPO / "papers/icra2027_motion_abstraction/droid_replay/small10"
_GIF = _OUT / "02_shake_head_no/llm/preview.gif"
_SIZE = 512
_CAM_ZOOM = 1.8


def _open_gripper(env) -> None:
    """DROID gripper_position 0 = open. Robotiq 2F-85 open is all finger joints at 0."""
    robot = env.robots[0]
    gripper = next(iter(robot.gripper.values()))
    for name in gripper.joints:
        jid = env.sim.model.joint_name2id(name)
        env.sim.data.qpos[env.sim.model.jnt_qposadr[jid]] = 0.0
    env.sim.forward()


def _env():
    robot = "FR3"
    arm_cfg = suite.load_part_controller_config(default_controller="JOINT_POSITION")
    env = suite.make(
        env_name="EmptySpace",
        robots=robot,
        gripper_types="DroidRobotiq85Gripper",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        ignore_done=True,
        control_freq=20,
        controller_configs=refactor_composite_controller_config(
            arm_cfg, robot, ["right", "left"]
        ),
        horizon=10000,
        camera_names="frontview",
        camera_heights=_SIZE,
        camera_widths=_SIZE,
    )
    env.reset()
    env.sim.model.opt.gravity[:] = [0.0, 0.0, 0.0]
    q0 = np.asarray(env.robots[0].robot_model.init_qpos, dtype=np.float64).reshape(-1)[:7]
    env.robots[0].set_robot_joint_positions(q0)
    _open_gripper(env)
    cam_id = env.sim.model.camera_name2id("frontview")
    fov = float(env.sim.model.cam_fovy[cam_id])
    env.sim.model.cam_fovy[cam_id] = max(20.0, min(120.0, fov * _CAM_ZOOM))
    env.sim.forward()
    return env


def _grab(env) -> Image.Image:
    rgb = env.sim.render(camera_name="frontview", width=_SIZE, height=_SIZE, depth=False)
    return Image.fromarray(np.asarray(rgb[::-1])).convert("RGB")


def _label(img: Image.Image, lines: list[str]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
        font_s = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        font_s = font
    y = 8
    for i, line in enumerate(lines):
        draw.rectangle([6, y - 2, 506, y + 18], fill=(0, 0, 0, 180))
        draw.text((10, y), line, fill=(255, 230, 80) if i == 0 else (255, 255, 255), font=font if i == 0 else font_s)
        y += 20
    return out


def main() -> None:
    env = _env()
    robot = env.robots[0]
    q = np.asarray(robot._joint_positions, dtype=np.float64)[:7]
    robot.set_robot_joint_positions(q)
    env.sim.forward()
    arm = "right" if "right" in robot.eef_site_id else next(iter(robot.eef_site_id))
    ee = np.asarray(env.sim.data.site_xpos[robot.eef_site_id[arm]], dtype=np.float64)
    cam = np.asarray(env.sim.data.cam_xpos[env.sim.model.camera_name2id("frontview")])
    print("q_deg", np.round(np.degrees(q), 1).tolist())
    print("EE", np.round(ee, 3).tolist(), "cam", np.round(cam, 3).tolist())
    print("EE.x - cam-side: EE is", "toward camera" if ee[0] > -0.6 else "behind base", "base offset empty=-0.6")

    still = _grab(env)
    still_path = _OUT / "neutral_same_camera.png"
    labeled = _label(
        still,
        [
            "FR3 desk-neutral (measured DROID joints)  ·  same camera as gallery GIFs",
            "camera = frontview at +X  (working side). Rear wall is -X behind the mount.",
            f"q° {np.round(np.degrees(q), 1).tolist()}",
            f"EE xyz {np.round(ee, 2).tolist()} m   cam {np.round(cam, 2).tolist()}",
        ],
    )
    still_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.save(still_path)
    print("wrote", still_path)

    gif0 = Image.open(_GIF).convert("RGB").resize((_SIZE, _SIZE), Image.Resampling.BILINEAR)
    gif0 = _label(gif0, ["gallery GIF frame 0  (02 shake_head_no LLM)", "same frontview — this dump starts at the same reset pose"])
    pair = Image.new("RGB", (_SIZE * 2 + 12, _SIZE + 36), (20, 20, 24))
    draw = ImageDraw.Draw(pair)
    draw.text((12, 8), "left: gallery GIF first frame     right: FR3 reset, identical camera", fill=(230, 230, 230))
    pair.paste(gif0, (0, 36))
    pair.paste(labeled, (_SIZE + 12, 36))
    pair_path = _OUT / "neutral_vs_gif0.png"
    pair.save(pair_path)
    print("wrote", pair_path)


if __name__ == "__main__":
    main()
