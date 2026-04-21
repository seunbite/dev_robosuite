import base64
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import fire
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOTARM_DIR = REPO_ROOT / "adhoc" / "robotarm"
if str(ROBOTARM_DIR) not in sys.path:
    sys.path.insert(0, str(ROBOTARM_DIR))

from vlm_direction_benchmark import _project_3d_to_2d  # noqa: E402


def _esc(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _image_to_data_uri(image_path: str) -> str:
    with Image.open(image_path).convert("RGB") as img:
        buf = BytesIO()
        img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _alpha_stack(frames: list[np.ndarray]) -> Image.Image:
    if not frames:
        raise ValueError("No frames to alpha-stack")
    pil_frames = [Image.fromarray(f).convert("RGB") for f in frames]
    final_rgb = pil_frames[-1]
    canvas = final_rgb.convert("RGBA")
    final_np = np.asarray(final_rgb).astype(np.int16)
    trail_frames = pil_frames[:-1]
    for order, frame_rgb in enumerate(trail_frames):
        frame_np = np.asarray(frame_rgb).astype(np.int16)
        diff = np.abs(frame_np - final_np).sum(axis=2)
        mask_np = np.where(diff > 28, 255, 0).astype(np.uint8)
        mask = Image.fromarray(mask_np, mode="L")
        alpha = int(100 + 120 * ((order + 1) / max(1, len(trail_frames))))
        frame_rgba = frame_rgb.convert("RGBA")
        frame_rgba.putalpha(mask.point(lambda px: min(255, int(px * alpha / 255.0))))
        canvas.alpha_composite(frame_rgba)
    return canvas.convert("RGB")


def _draw_arrow(img: Image.Image, start: tuple[int, int], end: tuple[int, int], color: str = "red", width: int = 5, head_size: int = 18) -> None:
    draw = ImageDraw.Draw(img)
    draw.line([start, end], fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max(1.0, float((dx**2 + dy**2) ** 0.5))
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    p1 = (int(end[0] - head_size * ux + head_size * 0.5 * px), int(end[1] - head_size * uy + head_size * 0.5 * py))
    p2 = (int(end[0] - head_size * ux - head_size * 0.5 * px), int(end[1] - head_size * uy - head_size * 0.5 * py))
    draw.polygon([end, p1, p2], fill=color)


def _tile_ab(a_path: str, b_path: str, output_path: str) -> str:
    left = Image.open(a_path).convert("RGB")
    right = Image.open(b_path).convert("RGB")
    width = left.width + right.width
    height = max(left.height, right.height) + 40
    canvas = Image.new("RGB", (width, height), (245, 247, 251))
    canvas.paste(left, (0, 40))
    canvas.paste(right, (left.width, 40))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    draw.text((16, 10), "A", fill=(20, 27, 34), font=font)
    draw.text((left.width + 16, 10), "B", fill=(20, 27, 34), font=font)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return output_path


def _classify_camera_direction(vec_cam: np.ndarray) -> str:
    depth_away = float(-vec_cam[2])
    right = float(vec_cam[0])
    up = float(vec_cam[1])
    mags = {
        "depth": abs(depth_away),
        "horizontal": abs(right),
        "vertical": abs(up),
    }
    dominant = max(mags, key=mags.get)
    if dominant == "depth":
        return "back" if depth_away >= 0 else "front"
    if dominant == "horizontal":
        return "right" if right >= 0 else "left"
    return "up" if up >= 0 else "down"


def _select_diverse(items: list[dict], key_name: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        buckets[str(item[key_name])].append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[dict] = []
    keys = list(buckets.keys())
    rng.shuffle(keys)
    while len(selected) < n and keys:
        progressed = False
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop())
                progressed = True
                if len(selected) >= n:
                    break
        if not progressed:
            break
    return selected[:n]


def _select_diverse_multi(items: list[dict], key_names: list[str], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for item in items:
        key = tuple(str(item[k]) for k in key_names)
        buckets[key].append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[dict] = []
    keys = list(buckets.keys())
    rng.shuffle(keys)
    while len(selected) < n and keys:
        progressed = False
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop())
                progressed = True
                if len(selected) >= n:
                    break
        if not progressed:
            break
    return selected[:n]


class PoseFactoredGenerator:
    def __init__(
        self,
        robot: str = "IIWA",
        img_size: int = 512,
        camera_fov_scale: float = 2.0,
        camera_name: str = "frontview",
        seed: int = 42,
    ):
        self.robot_name = robot
        self.img_size = int(img_size)
        self.camera_fov_scale = float(camera_fov_scale)
        self.camera_name = str(camera_name)
        self.rng = random.Random(seed)

        import robosuite as suite
        from robosuite.controllers.composite.composite_controller_factory import (
            refactor_composite_controller_config,
        )

        arm_ctrl = suite.load_part_controller_config(default_controller="OSC_POSE")
        ctrl_cfg = refactor_composite_controller_config(arm_ctrl, robot, ["right", "left"])
        self.env = suite.make(
            env_name="EmptySpace",
            robots=robot,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=True,
            control_freq=20,
            controller_configs=ctrl_cfg,
            horizon=1000,
        )
        self.env.reset()
        self.robot = self.env.robots[0]
        self.initial_joint_pos = self.robot._joint_positions.copy()
        self.num_joints = len(self.initial_joint_pos)
        self.active_joint_indices = list(range(self.num_joints - 1))
        self.arm_key = list(self.robot._hand_pos.keys())[0]
        self.cam_id = self.env.sim.model.camera_name2id(self.camera_name)
        original_fov = self.env.sim.model.cam_fovy[self.cam_id]
        self.env.sim.model.cam_fovy[self.cam_id] = min(120.0, original_fov * self.camera_fov_scale)

    def close(self) -> None:
        self.env.close()

    def _set_joint_positions(self, joint_pos: np.ndarray) -> None:
        self.robot.set_robot_joint_positions(joint_pos)
        self.env.sim.forward()

    def _camera_state(self) -> tuple[np.ndarray, np.ndarray, float]:
        cam_pos = self.env.sim.data.cam_xpos[self.cam_id].copy()
        cam_rot = self.env.sim.data.cam_xmat[self.cam_id].reshape(3, 3).copy()
        fovy = float(self.env.sim.model.cam_fovy[self.cam_id])
        return cam_pos, cam_rot, fovy

    def _tip_world_position(self) -> np.ndarray:
        model = self.env.sim.model
        tip_bodies = []
        for i in range(model.nbody):
            try:
                bname = model.body_id2name(i)
            except Exception:
                continue
            if bname and "finger" in bname and "tip" in bname:
                tip_bodies.append(self.env.sim.data.body_xpos[i].copy())
        if len(tip_bodies) >= 2:
            return np.mean(tip_bodies, axis=0)
        for i in range(model.nsite):
            try:
                sname = model.site_id2name(i)
            except Exception:
                continue
            if sname and "grip_site" in sname and "cylinder" not in sname:
                return self.env.sim.data.site_xpos[i].copy()
        return self.robot._hand_pos[self.arm_key].copy()

    def _camera_coords(self, world_point: np.ndarray) -> np.ndarray:
        cam_pos, cam_rot, _fovy = self._camera_state()
        p_cam = cam_rot.T @ (world_point - cam_pos)
        return np.array([-p_cam[2], p_cam[0], p_cam[1]], dtype=float)

    def _pointing_direction(self) -> tuple[str, np.ndarray]:
        ee_rot = self.robot._hand_orn[self.arm_key].copy()
        vec_world = ee_rot[:, 2]
        cam_pos, cam_rot, _fovy = self._camera_state()
        vec_cam = cam_rot.T @ vec_world
        return _classify_camera_direction(vec_cam), vec_cam

    def _capture_rgb(self) -> np.ndarray:
        obs = self.env.sim.render(
            camera_name=self.camera_name,
            width=self.img_size,
            height=self.img_size,
            depth=False,
        )
        return obs[::-1].copy()

    def _project_tip(self, world_point: np.ndarray) -> tuple[int, int] | None:
        cam_pos, cam_rot, fovy = self._camera_state()
        return _project_3d_to_2d(world_point, cam_pos, cam_rot, fovy, self.img_size)

    def _in_frame(self, world_point: np.ndarray) -> bool:
        pt = self._project_tip(world_point)
        if pt is None:
            return False
        margin = int(self.img_size * 0.1)
        return margin <= pt[0] <= self.img_size - margin and margin <= pt[1] <= self.img_size - margin

    def sample_static_pool(
        self,
        target_count: int = 80,
        max_attempts: int = 1200,
    ) -> list[dict]:
        pool: list[dict] = []
        seen = set()
        for _attempt in range(max_attempts):
            if len(pool) >= target_count:
                break
            joint_pos = self.initial_joint_pos.copy()
            move_count = self.rng.randint(2, min(4, len(self.active_joint_indices)))
            chosen = self.rng.sample(self.active_joint_indices, move_count)
            for idx in chosen:
                joint_pos[idx] = np.deg2rad(self.rng.choice([-90, 0, 90]))
            key = tuple(round(float(v), 4) for v in joint_pos.tolist())
            if key in seen:
                continue
            seen.add(key)
            self._set_joint_positions(joint_pos)
            tip_world = self._tip_world_position()
            if not self._in_frame(tip_world):
                continue
            direction, vec_cam = self._pointing_direction()
            camera_xyz = self._camera_coords(tip_world)
            if camera_xyz[0] < 0.05:
                continue
            pool.append(
                {
                    "joint_pos": joint_pos.copy(),
                    "direction": direction,
                    "camera_tip_xyz": camera_xyz.tolist(),
                    "tip_world": tip_world.tolist(),
                    "pointing_cam": vec_cam.tolist(),
                }
            )
        return pool

    def render_static_pose(self, joint_pos: np.ndarray, output_path: str) -> str:
        self._set_joint_positions(joint_pos)
        image = Image.fromarray(self._capture_rgb()).convert("RGB")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG")
        return output_path

    def render_static_pose_with_arrow(
        self,
        joint_pos: np.ndarray,
        output_path: str,
        arrow_color: str = "red",
        arrow_length: float = 0.15,
    ) -> str:
        self._set_joint_positions(joint_pos)
        image = Image.fromarray(self._capture_rgb()).convert("RGB")
        tip_pos = self._tip_world_position()
        ee_rot = self.robot._hand_orn[self.arm_key].copy()
        pointing_dir = ee_rot[:, 2]
        cam_pos, cam_rot, fovy = self._camera_state()
        s2d = _project_3d_to_2d(tip_pos, cam_pos, cam_rot, fovy, self.img_size)
        e2d = _project_3d_to_2d(tip_pos + pointing_dir * arrow_length, cam_pos, cam_rot, fovy, self.img_size)
        if s2d and e2d:
            dx, dy = e2d[0] - s2d[0], e2d[1] - s2d[1]
            screen_len = float((dx**2 + dy**2) ** 0.5)
            if 1 < screen_len < 40:
                scale = 40.0 / screen_len
                e2d = (int(s2d[0] + dx * scale), int(s2d[1] + dy * scale))
            _draw_arrow(image, s2d, e2d, color=arrow_color, width=5, head_size=18)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG")
        return output_path

    def build_move_candidates(
        self,
        pool: list[dict],
        max_candidates: int = 80,
        delta_deg: float = 15.0,
    ) -> list[dict]:
        candidates: list[dict] = []
        for base in pool:
            base_joint = np.array(base["joint_pos"], dtype=float)
            self._set_joint_positions(base_joint)
            start_tip = self._tip_world_position()
            start_px = self._project_tip(start_tip)
            if start_px is None:
                continue
            for joint_idx in self.active_joint_indices:
                for sign in (-1.0, 1.0):
                    end_joint = base_joint.copy()
                    end_joint[joint_idx] += np.deg2rad(delta_deg * sign)
                    self._set_joint_positions(end_joint)
                    end_tip = self._tip_world_position()
                    end_px = self._project_tip(end_tip)
                    if end_px is None or not self._in_frame(end_tip):
                        continue
                    world_delta = end_tip - start_tip
                    if float(np.linalg.norm(world_delta)) < 0.025:
                        continue
                    cam_pos, cam_rot, _fovy = self._camera_state()
                    cam_delta = cam_rot.T @ world_delta
                    label = _classify_camera_direction(cam_delta)
                    px_delta = float(np.linalg.norm(np.array(end_px) - np.array(start_px)))
                    if px_delta < 20:
                        continue
                    candidates.append(
                        {
                            "base_joint_pos": base_joint.copy(),
                            "end_joint_pos": end_joint.copy(),
                            "joint_idx": joint_idx,
                            "delta_deg": float(delta_deg * sign),
                            "move_direction": label,
                            "world_delta": world_delta.tolist(),
                            "camera_delta": [-float(cam_delta[2]), float(cam_delta[0]), float(cam_delta[1])],
                            "pixel_delta": px_delta,
                            "screen_dx": float(end_px[0] - start_px[0]),
                            "screen_dy": float(end_px[1] - start_px[1]),
                        }
                    )
                    if len(candidates) >= max_candidates:
                        return candidates
        return candidates

    def render_alpha_motion(self, start_joint_pos: np.ndarray, end_joint_pos: np.ndarray, output_path: str) -> str:
        frames: list[np.ndarray] = []
        n_frames = 5
        for i in range(n_frames):
            alpha = i / max(1, n_frames - 1)
            interp = start_joint_pos * (1 - alpha) + end_joint_pos * alpha
            self._set_joint_positions(interp)
            frames.append(self._capture_rgb())
        image = _alpha_stack(frames)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG")
        return output_path

    def build_xyz_pairs(
        self,
        pool: list[dict],
        n_pairs: int = 10,
        min_depth_diff: float = 0.12,
        min_side_diff: float = 0.12,
        min_height_diff: float = 0.12,
        dominance_ratio: float = 1.35,
    ) -> list[dict]:
        pairs: list[dict] = []
        used = set()
        for _ in range(1200):
            if len(pairs) >= n_pairs:
                break
            left, right = self.rng.sample(pool, 2)
            lid = tuple(round(float(v), 4) for v in np.array(left["joint_pos"], dtype=float).tolist())
            rid = tuple(round(float(v), 4) for v in np.array(right["joint_pos"], dtype=float).tolist())
            key = tuple(sorted([lid, rid]))
            if key in used:
                continue
            used.add(key)
            lx_depth, ly, lz = left["camera_tip_xyz"]
            rx_depth, ry, rz = right["camera_tip_xyz"]
            # For xyz_compare we define x so that larger x means closer to the camera.
            lx = -float(lx_depth)
            rx = -float(rx_depth)
            diffs = {
                "x": abs(lx - rx),
                "y": abs(ly - ry),
                "z": abs(lz - rz),
            }
            thresholds = {
                "x": min_depth_diff,
                "y": min_side_diff,
                "z": min_height_diff,
            }
            eligible = [axis for axis, diff in diffs.items() if diff >= thresholds[axis]]
            if not eligible:
                continue
            eligible.sort(key=lambda axis: diffs[axis], reverse=True)
            axis = eligible[0]
            next_best = diffs[eligible[1]] if len(eligible) > 1 else 0.0
            if next_best > 0 and diffs[axis] / next_best < dominance_ratio:
                continue
            pairs.append(
                {
                    "a_joint_pos": np.array(left["joint_pos"], dtype=float).copy(),
                    "b_joint_pos": np.array(right["joint_pos"], dtype=float).copy(),
                    "a_xyz": [float(lx), float(ly), float(lz)],
                    "b_xyz": [float(rx), float(ry), float(rz)],
                    "axis": axis,
                    "gt_choice": ("A" if {"x": lx, "y": ly, "z": lz}[axis] > {"x": rx, "y": ry, "z": rz}[axis] else "B"),
                    "axis_diff": float(diffs[axis]),
                }
            )
        return pairs[:n_pairs]


def _direction_prompt() -> str:
    return (
        "You will see a single rendered image of a static robot arm in a simulated room.\n\n"
        "Infer the direction the gripper tip is pointing.\n"
        "This is about the gripper's pointing direction, not its position.\n\n"
        "Camera-centric direction definitions:\n"
        "- front: points toward the camera.\n"
        "- back: points away from the camera, into the scene.\n"
        "- up: points toward the ceiling.\n"
        "- down: points toward the floor.\n"
        "- left: points toward the left side of the image.\n"
        "- right: points toward the right side of the image.\n\n"
        'Return exactly one JSON object: {"direction":"front|back|left|right|up|down"}'
    )


def _direction_arrow_prompt() -> str:
    return (
        "You will see a single rendered image of a static robot arm in a simulated room.\n\n"
        "Infer the direction the gripper tip is pointing.\n"
        "This is about the gripper's pointing direction, not its position.\n"
        "A RED arrow is drawn from the gripper tip to show the pointing direction.\n\n"
        "Camera-centric direction definitions:\n"
        "- front: points toward the camera.\n"
        "- back: points away from the camera, into the scene.\n"
        "- up: points toward the ceiling.\n"
        "- down: points toward the floor.\n"
        "- left: points toward the left side of the image.\n"
        "- right: points toward the right side of the image.\n\n"
        'Return exactly one JSON object: {"direction":"front|back|left|right|up|down"}'
    )


def _move_direction_prompt() -> str:
    return (
        "You will see a single overlaid alpha-stack image of a robot arm gesture.\n\n"
        "Infer which movement axis is most visually dominant for the gripper from earliest pose to latest pose.\n"
        "There are no white arrows in this image. Use only the overlaid arm positions.\n"
        "Choose the single axis that best explains the motion.\n\n"
        "Axis definitions:\n"
        "- x: front-back motion, meaning toward or away from the camera.\n"
        "- y: left-right motion across the image.\n"
        "- z: up-down motion toward the ceiling or floor.\n\n"
        "If multiple axes move, choose the most visually dominant axis.\n"
        "If it is genuinely ambiguous between two axes, you may return two axes.\n\n"
        'Return exactly one JSON object in one of these forms:\n'
        '- {"axes":["x"]}\n'
        '- {"axes":["x","y"]}'
    )


def _xyz_compare_prompt(axis: str) -> str:
    axis = str(axis)
    axis_hint = {
        "x": "Focus on x only. Choose the pose whose gripper tip has the larger x value. Larger x means closer to the camera.",
        "y": "Focus on y only. Choose the pose whose gripper tip has the larger y value. Larger y means farther to the right side of the image.",
        "z": "Focus on z only. Choose the pose whose gripper tip has the larger z value. Larger z means higher.",
    }[axis]
    return (
        "You will see a tiled image containing two static robot-arm poses: A and B.\n\n"
        "Compare the 3D position of the gripper tip between the two poses.\n"
        f"{axis_hint}\n\n"
        "Axis definitions:\n"
        "- x: front-back depth. Larger x means closer to the camera.\n"
        "- y: left-right. Larger y means farther to the right side of the image.\n"
        "- z: up-down. Larger z means higher.\n\n"
        f'Return exactly one JSON object: {{"axis":"{axis}","choice":"A|B"}}'
    )


def _write_preview_html(html_path: Path, samples: list[dict], prompts: dict[str, str], manifest_path: Path) -> None:
    sections: list[str] = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[sample["task"]].append(sample)

    for task in ("direction", "direction_arrow", "move_direction", "xyz_compare"):
        rows = grouped.get(task, [])
        sections.append(f"<section class='group'><h2>{_esc(task)} (n={len(rows)})</h2>")
        sections.append(f"<div class='prompt'><div class='label'>Prompt</div><pre>{_esc(prompts[task])}</pre></div>")
        for row in rows:
            gt_block = _json_dump(row["gt"])
            meta_block = _json_dump(row["meta"])
            sections.append(
                "<article class='card'>"
                f"<div class='media'><img src='{_image_to_data_uri(row['media_path'])}' alt='{_esc(row['sample_id'])}'></div>"
                "<div class='info'>"
                f"<div class='title'>{_esc(row['sample_id'])}</div>"
                f"<div class='box'><div class='label'>Ground Truth</div><pre>{_esc(gt_block)}</pre></div>"
                f"<div class='box'><div class='label'>Meta</div><pre>{_esc(meta_block)}</pre></div>"
                f"<div class='box'><div class='label'>Media Path</div><pre>{_esc(row['media_path'])}</pre></div>"
                "</div></article>"
            )
        sections.append("</section>")

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pose Factored Preview</title>
  <style>
    :root {{
      --bg: #f4f6f8; --surface: #ffffff; --surface2: #eef2f6; --border: #d8dde5;
      --text: #17212b; --muted: #617081;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .hero {{ margin-bottom: 20px; }}
    .chips {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 24px; }}
    .chip {{ background: var(--surface); border: 1px solid var(--border); border-radius: 999px; padding: 8px 12px; font-size: 13px; }}
    .group {{ margin-bottom: 30px; }}
    .prompt {{ background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 14px; margin-bottom: 16px; }}
    .label {{ font-size: 12px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px; }}
    .card {{ display: grid; grid-template-columns: minmax(360px, 560px) 1fr; gap: 16px; background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 16px; margin-bottom: 16px; }}
    .media {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; display: flex; align-items: center; justify-content: center; min-height: 280px; }}
    .media img {{ width: 100%; height: auto; display: block; }}
    .info {{ display: flex; flex-direction: column; gap: 10px; }}
    .title {{ font-weight: 800; font-size: 18px; }}
    .box {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 12px; padding: 12px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; margin: 0; font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    @media (max-width: 980px) {{ .card {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Pose Factored Preview</h1>
      <div class="chips">
        <div class="chip">manifest: {_esc(manifest_path)}</div>
        <div class="chip">direction: {len(grouped.get("direction", []))}</div>
        <div class="chip">move_direction: {len(grouped.get("move_direction", []))}</div>
        <div class="chip">xyz_compare: {len(grouped.get("xyz_compare", []))}</div>
      </div>
    </section>
    {''.join(sections)}
  </div>
</body>
</html>"""
    html_path.write_text(html, encoding="utf-8")


def _build_preview_samples(
    robot: str,
    n_per_task: int,
    output_dir: Path,
    seed: int,
    img_size: int,
    camera_name: str,
) -> tuple[list[dict], dict[str, str]]:
    prompts = {
        "direction": _direction_prompt(),
        "direction_arrow": _direction_arrow_prompt(),
        "move_direction": _move_direction_prompt(),
        "xyz_compare": "axis-specific prompt; see each sample prompt_text",
    }
    generator = PoseFactoredGenerator(robot=robot, img_size=img_size, camera_name=camera_name, seed=seed)
    try:
        pool = generator.sample_static_pool(target_count=max(80, n_per_task * 8))
        if len(pool) < max(30, n_per_task * 3):
            raise RuntimeError(f"Static pose pool too small: {len(pool)}")

        direction_rows = _select_diverse(pool, "direction", n_per_task, seed)
        direction_arrow_rows = direction_rows[:]
        move_candidates = generator.build_move_candidates(pool, max_candidates=max(80, n_per_task * 10))
        for cand in move_candidates:
            sx, sy = cand["screen_dx"], cand["screen_dy"]
            cand["move_direction"] = "right" if abs(sx) >= abs(sy) and sx >= 0 else (
                "left" if abs(sx) >= abs(sy) else ("down" if sy >= 0 else "up")
            )
        for cand in move_candidates:
            dx = abs(float(cand["camera_delta"][0]))
            dy = abs(float(cand["camera_delta"][1]))
            dz = abs(float(cand["camera_delta"][2]))
            axis_scores = {"x": dx, "y": dy, "z": dz}
            cand["move_axis"] = max(axis_scores, key=axis_scores.get)
            cand["move_axis_scores"] = axis_scores
        move_rows = _select_diverse(move_candidates, "move_axis", n_per_task, seed)
        xyz_rows = generator.build_xyz_pairs(pool, n_pairs=n_per_task)
        if len(direction_rows) < n_per_task or len(move_rows) < n_per_task or len(xyz_rows) < n_per_task:
            raise RuntimeError(
                f"Insufficient samples: direction={len(direction_rows)} move={len(move_rows)} xyz={len(xyz_rows)}"
            )

        samples: list[dict] = []
        image_dir = output_dir / "media"
        image_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in enumerate(direction_rows):
            media_path = image_dir / f"direction_{idx:02d}.png"
            generator.render_static_pose(np.array(row["joint_pos"], dtype=float), str(media_path))
            samples.append(
                {
                    "task": "direction",
                    "sample_id": f"direction_{idx:02d}",
                    "media_path": str(media_path),
                    "prompt_text": prompts["direction"],
                    "gt": {"direction": row["direction"]},
                    "meta": {
                        "camera_tip_xyz": row["camera_tip_xyz"],
                        "pointing_cam": row["pointing_cam"],
                    },
                }
            )

        for idx, row in enumerate(direction_arrow_rows):
            media_path = image_dir / f"direction_arrow_{idx:02d}.png"
            generator.render_static_pose_with_arrow(np.array(row["joint_pos"], dtype=float), str(media_path), arrow_color="red")
            samples.append(
                {
                    "task": "direction_arrow",
                    "sample_id": f"direction_arrow_{idx:02d}",
                    "media_path": str(media_path),
                    "prompt_text": prompts["direction_arrow"],
                    "gt": {"direction": row["direction"]},
                    "meta": {
                        "camera_tip_xyz": row["camera_tip_xyz"],
                        "pointing_cam": row["pointing_cam"],
                    },
                }
            )

        for idx, row in enumerate(move_rows):
            media_path = image_dir / f"move_direction_{idx:02d}.png"
            generator.render_alpha_motion(
                np.array(row["base_joint_pos"], dtype=float),
                np.array(row["end_joint_pos"], dtype=float),
                str(media_path),
            )
            samples.append(
                {
                    "task": "move_direction",
                    "sample_id": f"move_direction_{idx:02d}",
                    "media_path": str(media_path),
                    "prompt_text": prompts["move_direction"],
                    "gt": {"axis": row["move_axis"]},
                    "meta": {
                        "joint_idx": row["joint_idx"],
                        "delta_deg": row["delta_deg"],
                        "camera_delta_xyz": row["camera_delta"],
                        "move_axis_scores": row["move_axis_scores"],
                        "screen_dx": row["screen_dx"],
                        "screen_dy": row["screen_dy"],
                        "pixel_delta": row["pixel_delta"],
                    },
                }
            )

        for idx, row in enumerate(xyz_rows):
            left_path = image_dir / f"xyz_a_{idx:02d}.png"
            right_path = image_dir / f"xyz_b_{idx:02d}.png"
            tiled_path = image_dir / f"xyz_compare_{idx:02d}.png"
            generator.render_static_pose(np.array(row["a_joint_pos"], dtype=float), str(left_path))
            generator.render_static_pose(np.array(row["b_joint_pos"], dtype=float), str(right_path))
            _tile_ab(str(left_path), str(right_path), str(tiled_path))
            samples.append(
                {
                    "task": "xyz_compare",
                    "sample_id": f"xyz_compare_{idx:02d}",
                    "media_path": str(tiled_path),
                    "prompt_text": _xyz_compare_prompt(row["axis"]),
                    "gt": {"axis": row["axis"], "choice": row["gt_choice"]},
                    "meta": {
                        "a_xyz": row["a_xyz"],
                        "b_xyz": row["b_xyz"],
                        "axis_diff": row["axis_diff"],
                    },
                }
            )
    finally:
        generator.close()
    return samples, prompts


def _is_correct_eval_row(row: dict) -> bool:
    raw = str(row.get("raw_response", ""))
    task = row.get("task")
    gt = row.get("gt", {})

    if task in {"direction", "direction_arrow"}:
        m = re.search(r"front|back|left|right|up|down", raw.lower())
        pred = m.group(0) if m else None
        return pred == str(gt.get("direction", "")).lower()

    if task == "move_direction":
        axes = re.findall(r'"(x|y|z)"', raw.lower())
        uniq = []
        for axis in axes:
            if axis not in uniq:
                uniq.append(axis)
        gt_axis = str(gt.get("axis", "")).lower()
        return gt_axis in uniq[:2]

    if task == "xyz_compare":
        m_axis = re.search(r'"axis"\s*:\s*"(x|y|z)"', raw.lower())
        m_choice = re.search(r'"choice"\s*:\s*"(a|b)"', raw.lower())
        pred_axis = m_axis.group(1) if m_axis else None
        pred_choice = m_choice.group(1).upper() if m_choice else None
        return (
            pred_axis == str(gt.get("axis", "")).lower()
            and pred_choice == str(gt.get("choice", "")).upper()
        )

    return False


def _pred_summary(row: dict) -> str:
    raw = str(row.get("raw_response", ""))
    if row.get("task") in {"direction", "direction_arrow"}:
        m = re.search(r"front|back|left|right|up|down", raw.lower())
        return m.group(0) if m else "?"
    if row.get("task") == "move_direction":
        axes = re.findall(r'"(x|y|z)"', raw.lower())
        if axes:
            uniq = []
            for axis in axes:
                if axis not in uniq:
                    uniq.append(axis)
            return ",".join(uniq[:2])
        return "?"
    if row.get("task") == "xyz_compare":
        m_axis = re.search(r'"axis"\s*:\s*"(x|y|z)"', raw.lower())
        m_side = re.search(r'"choice"\s*:\s*"(a|b)"', raw.lower())
        axis = m_axis.group(1) if m_axis else "?"
        side = m_side.group(1).upper() if m_side else "?"
        return f"{axis}:{side}"
    return "?"


def _write_eval_html(rows: list[dict], html_path: Path) -> None:
    correct = sum(1 for row in rows if _is_correct_eval_row(row))
    total = len(rows)
    accuracy = (correct / total) if total else 0.0
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Pose Factored Eval</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#f4f6f8;color:#17212b;margin:0} .wrap{max-width:1400px;margin:0 auto;padding:24px} .summary{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px} .chip{background:#fff;border:1px solid #d8dde5;border-radius:999px;padding:8px 12px;font-weight:600} .card{display:grid;grid-template-columns:minmax(340px,520px) 1fr;gap:16px;background:#fff;border:1px solid #d8dde5;border-radius:16px;padding:16px;margin-bottom:16px;position:relative} .badge{position:absolute;top:14px;right:14px;padding:6px 10px;border-radius:999px;font-weight:700;font-size:12px;border:1px solid transparent} .badge.correct{background:#e8f7ee;color:#11643a;border-color:#b7e2c7} .badge.wrong{background:#fdecec;color:#9f1c1c;border-color:#f5c2c2} .media img{width:100%;display:block;border-radius:12px;border:1px solid #d8dde5} .box{background:#eef2f6;border:1px solid #d8dde5;border-radius:12px;padding:12px;margin-bottom:10px} pre{white-space:pre-wrap;word-break:break-word;margin:0;font-family:ui-monospace,Menlo,monospace;font-size:12px}</style>",
        "</head><body><div class='wrap'><h1>Pose Factored Eval</h1>",
        f"<div class='summary'><div class='chip'>N: {total}</div><div class='chip'>Correct: {correct}</div><div class='chip'>Accuracy: {accuracy:.2f}</div></div>",
    ]
    for row in rows:
        ok = _is_correct_eval_row(row)
        badge_text = "Correct" if ok else "Wrong"
        badge_class = "correct" if ok else "wrong"
        parts.append(
            f"<section class='card'><div class='badge {badge_class}'>{badge_text}</div><div class='media'><img src='{_image_to_data_uri(row['media_path'])}' alt='{_esc(row['sample_id'])}'></div><div>"
            f"<div class='box'><strong>{_esc(row['sample_id'])}</strong> | {_esc(row['task'])}</div>"
            f"<div class='box'><pre>{_esc(_json_dump(row['gt']))}</pre></div>"
            f"<div class='box'><pre>{_esc(_pred_summary(row))}</pre></div>"
            f"<div class='box'><pre>{_esc(row['prompt_text'])}</pre></div>"
            f"<div class='box'><pre>{_esc(row['raw_response'])}</pre></div>"
            "</div></section>"
        )
    parts.append("</div></body></html>")
    html_path.write_text("".join(parts), encoding="utf-8")


def preview(
    robot: str = "IIWA",
    n_per_task: int = 20,
    seed: int = 42,
    img_size: int = 512,
    camera_name: str = "frontview",
    output_dir: str = "adhoc/test/results",
    open_html: bool = True,
) -> str:
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stem = f"pose_factored_preview_{robot}_{camera_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / stem
    run_dir.mkdir(parents=True, exist_ok=True)

    samples, prompts = _build_preview_samples(
        robot=robot,
        n_per_task=int(n_per_task),
        output_dir=run_dir,
        seed=int(seed),
        img_size=int(img_size),
        camera_name=str(camera_name),
    )
    manifest = {
        "robot": robot,
        "n_per_task": int(n_per_task),
        "seed": int(seed),
        "img_size": int(img_size),
        "camera_name": str(camera_name),
        "prompts": prompts,
        "samples": samples,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path = run_dir / "preview.html"
    _write_preview_html(html_path, samples, prompts, manifest_path)

    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Preview HTML: {html_path.resolve()}")
    print(f"Preview URL: file://{html_path.resolve()}")
    if open_html:
        os.system(f"open '{html_path.resolve()}'")
    return str(html_path.resolve())


def run_eval(
    manifest_path: str,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.0,
    delay_sec: float = 1.0,
    task: str | None = None,
    open_html: bool = True,
) -> str:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY or GEMINI_API_KEY.")

    manifest_file = Path(manifest_path)
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    client = genai.Client(api_key=api_key)
    rows: list[dict] = []
    output_dir = manifest_file.parent
    suffix = str(task) if task is not None else "all"
    jsonl_path = output_dir / f"eval_results_{suffix}.jsonl"
    html_path = output_dir / f"eval_report_{suffix}.html"

    selected_samples = [
        sample for sample in data["samples"]
        if task is None or str(sample.get("task")) == str(task)
    ]
    if not selected_samples:
        raise ValueError(f"No samples found for task={task!r}")

    with open(jsonl_path, "w", encoding="utf-8", buffering=1) as f:
        for sample in selected_samples:
            image = Image.open(sample["media_path"]).convert("RGB")
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[sample["prompt_text"], image],
                    config=types.GenerateContentConfig(temperature=float(temperature)),
                )
                raw_text = response.text.strip()
            except Exception as exc:
                raw_text = f"ERROR: {exc}"
            row = dict(sample)
            row["raw_response"] = raw_text
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            time.sleep(float(delay_sec))

    _write_eval_html(rows, html_path)
    print(f"Eval JSONL: {jsonl_path.resolve()}")
    print(f"Eval HTML: {html_path.resolve()}")
    print(f"Eval URL: file://{html_path.resolve()}")
    if open_html:
        os.system(f"open '{html_path.resolve()}'")
    return str(html_path.resolve())


if __name__ == "__main__":
    fire.Fire(
        {
            "preview": preview,
            "run_eval": run_eval,
        }
    )
