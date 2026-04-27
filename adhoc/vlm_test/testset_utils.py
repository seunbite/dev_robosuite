import base64
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = REPO_ROOT / "data" / "seed"
MC_MANIP = REPO_ROOT / "data" / "results" / "motion_configs" / "manipulator"
MOTION_DIR = REPO_ROOT / "data" / "motions"


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text)).strip("_") or "item"


def load_cues() -> Dict[str, Dict[str, str]]:
    with open(SEED_DIR / "yml" / "cues.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _latest_match(patterns: List[str]) -> str | None:
    matches: List[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    matches = [m for m in matches if not m.endswith("_preview.gif")]
    if not matches:
        return None
    return sorted(set(matches), key=os.path.getmtime, reverse=True)[0]


def _find_iconic_gif(robot: str, cue_idx: int, motion_subdir: str = "v18") -> str | None:
    matches: List[str] = []
    patterns = [
        str(MOTION_DIR / motion_subdir / robot / f"*_c{cue_idx}_tiled.gif"),
        str(MOTION_DIR / f"{motion_subdir}_archive" / "*" / robot / f"*_c{cue_idx}_tiled.gif"),
    ]
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    matches = [m for m in matches if "contextual" not in str(m).lower()]
    if not matches:
        return None
    return sorted(set(matches), key=os.path.getmtime, reverse=True)[0]


def _find_contextual_gif(robot: str, cue_idx: int, motion_subdir: str = "v18_contextual") -> str | None:
    patterns = [
        str(MOTION_DIR / motion_subdir / robot / f"*_c{cue_idx}_tiled.gif"),
        str(MOTION_DIR / f"{motion_subdir}_archive" / "*" / robot / f"*_c{cue_idx}_tiled.gif"),
    ]
    return _latest_match(patterns)


def _find_persona_gif(persona_dir: Path, robot: str, cue: str) -> str | None:
    safe_cue = cue.replace("/", "_").replace("\\", "_").replace(" ", "_")
    patterns = [
        str(persona_dir / robot / f"*_{safe_cue}_tiled.gif"),
        str(persona_dir / robot / f"*_{safe_cue}_*.gif"),
    ]
    return _latest_match(patterns)


def build_samples(
    *,
    testset: str,
    robot: str = "IIWA",
    prompt_version: int = 18,
    config_json: str | None = None,
    iconic_motion_subdir: str = "v18",
    contextual_motion_subdir: str = "v18_contextual",
    persona_output_dir: str = "data/seed/persona_tag_dataset_v1",
    sample_offset: int = 0,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    testset = testset.lower()
    rows: List[Dict[str, Any]] = []

    if testset == "iconic":
        config_path = (
            Path(config_json)
            if config_json
            else (MC_MANIP / f"motion_configs_prompt_v{prompt_version}.json")
        )
        cfgs = _load_json(config_path)
        for cfg in cfgs:
            cue_idx = int(cfg["idx"])
            gif_path = _find_iconic_gif(robot, cue_idx, motion_subdir=iconic_motion_subdir)
            if not gif_path:
                continue
            rows.append(
                {
                    "sample_id": f"iconic_c{cue_idx}",
                    "testset": "iconic",
                    "cue_idx": cue_idx,
                    "cue": cfg["cue"],
                    "gif_path": gif_path,
                    "config_path": str(config_path),
                    "meta": {},
                }
            )
    elif testset == "contextual":
        config_path = (
            Path(config_json)
            if config_json
            else (MC_MANIP / f"motion_configs_prompt_v{prompt_version}_contextual.json")
        )
        cfgs = _load_json(config_path)
        for cfg in cfgs:
            cue_idx = int(cfg["idx"])
            gif_path = _find_contextual_gif(robot, cue_idx, motion_subdir=contextual_motion_subdir)
            if not gif_path:
                continue
            rows.append(
                {
                    "sample_id": f"contextual_c{cue_idx}",
                    "testset": "contextual",
                    "cue_idx": cue_idx,
                    "cue": cfg["cue"],
                    "gif_path": gif_path,
                    "config_path": str(config_path),
                    "meta": {},
                }
            )
    elif testset == "persona":
        persona_root = (REPO_ROOT / persona_output_dir).resolve()
        dataset_path = persona_root / "dataset.jsonl"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Persona dataset not found: {dataset_path}")
        for row in _load_jsonl(dataset_path):
            if row.get("status") != "success":
                continue
            tag_key = f"{row['tag_category']}__{row['tag_name']}"
            gif_path = _find_persona_gif(persona_root / "renders" / _safe_name(tag_key), robot, row["cue"])
            if not gif_path:
                continue
            rows.append(
                {
                    "sample_id": f"persona_{_safe_name(tag_key)}_c{row['cue_idx']}",
                    "testset": "persona",
                    "cue_idx": int(row["cue_idx"]),
                    "cue": row["cue"],
                    "gif_path": gif_path,
                    "edited_config": row["edited_config"],
                    "meta": {
                        "tag_category": row["tag_category"],
                        "tag_name": row["tag_name"],
                    },
                }
            )
    else:
        raise ValueError(f"Unknown testset: {testset}")

    rows.sort(key=lambda x: (x["cue_idx"], x["sample_id"]))
    if sample_offset is not None and int(sample_offset) > 0:
        rows = rows[int(sample_offset):]
    if limit is not None:
        rows = rows[: int(limit)]
    return rows


def extract_first_frame_bytes(gif_path: str, fmt: str = "PNG") -> bytes:
    img = Image.open(gif_path)
    frame = img.convert("RGB")
    buf = BytesIO()
    frame.save(buf, format=fmt)
    return buf.getvalue()


def first_frame_data_uri(gif_path: str) -> str:
    image_bytes = extract_first_frame_bytes(gif_path, fmt="PNG")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _extract_frame_image(gif_path: str, frame_idx: int) -> Image.Image:
    img = Image.open(gif_path)
    img.seek(frame_idx)
    return img.convert("RGB")


def _gif_frame_count(gif_path: str) -> int:
    img = Image.open(gif_path)
    return int(getattr(img, "n_frames", 1))


def extract_middle_frame_bytes(gif_path: str, fmt: str = "PNG") -> bytes:
    frame_count = _gif_frame_count(gif_path)
    frame = _extract_frame_image(gif_path, frame_count // 2)
    buf = BytesIO()
    frame.save(buf, format=fmt)
    return buf.getvalue()


def _save_png(image: Image.Image, output_path: str) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return output_path


def _build_alpha_stack_image(gif_path: str, stack_count: int = 10) -> Image.Image:
    img = Image.open(gif_path)
    total_frames = int(getattr(img, "n_frames", 1))
    if total_frames <= 1:
        return img.convert("RGBA").convert("RGB")

    indices = sorted(set(int(round(i * (total_frames - 1) / max(1, stack_count - 1))) for i in range(stack_count)))
    img.seek(total_frames - 1)
    final_rgb = img.convert("RGB")
    canvas = final_rgb.convert("RGBA")

    # Overlay only the regions that differ from the final frame, so the background
    # does not wash out the trails and the stacked motion stays visible.
    final_np = np.asarray(final_rgb).astype(np.int16)
    trail_indices = [idx for idx in indices if idx != total_frames - 1]
    for order, frame_idx in enumerate(trail_indices):
        img.seek(frame_idx)
        frame_rgb = img.convert("RGB")
        frame_np = np.asarray(frame_rgb).astype(np.int16)
        diff = np.abs(frame_np - final_np).sum(axis=2)
        mask_np = np.where(diff > 28, 255, 0).astype(np.uint8)
        mask = Image.fromarray(mask_np, mode="L").filter(ImageFilter.GaussianBlur(radius=1.5))

        frame_rgba = frame_rgb.convert("RGBA")
        trail_alpha = int(120 + 110 * ((order + 1) / max(1, len(trail_indices))))
        frame_rgba.putalpha(mask.point(lambda px: min(255, int(px * trail_alpha / 255.0))))
        canvas.alpha_composite(frame_rgba)

    final_outline = final_rgb.convert("RGBA")
    final_outline.putalpha(48)
    canvas.alpha_composite(final_outline)
    return canvas.convert("RGB")


def _estimate_step_frame_counts(cfg: Dict[str, Any], hz: int) -> List[int]:
    counts: List[int] = []
    for step in cfg.get("movements", []):
        step_type = step.get("type")
        params = step.get("parameters", {}) or {}
        total = 0
        if step_type == "pose":
            speed = float(params.get("speed", 1.0) or 1.0)
            hold_time = float(params.get("hold_time", 0.0) or 0.0)
            if counts:
                total += max(1, int((1.0 / max(0.5, speed)) * hz))
            total += max(0, int(hold_time * hz))
        elif step_type == "movement":
            repetition = int(params.get("repetition", 1) or 1)
            directions = params.get("directions", []) or []
            seq_total = 0
            for d in directions:
                dir_speed = float(d.get("speed", 1.0) or 1.0)
                dir_hold = float(d.get("hold_time", 0.0) or 0.0)
                seq_total += max(1, int((1.0 / max(0.5, dir_speed)) * hz))
                seq_total += max(0, int(dir_hold * hz))
            total += max(1, seq_total) * max(1, repetition)
        elif step_type == "path":
            path_speed = float(params.get("speed", 1.0) or 1.0)
            total += max(1, int((1.0 / max(0.5, path_speed)) * hz))
        counts.append(max(1, total))
    return counts


def _draw_thick_arrow(
    draw: ImageDraw.ImageDraw,
    pts: List[tuple[int, int]],
    color=(255, 255, 255),
    width: int = 5,
    head_len: int = 14,
    head_w: int = 10,
    bidirectional: bool = False,
) -> None:
    if len(pts) < 2:
        return
    draw.line(pts, fill=color, width=width)
    def _draw_head(a: tuple[int, int], b: tuple[int, int]) -> None:
        x1, y1 = a
        x2, y2 = b
        dx = x2 - x1
        dy = y2 - y1
        norm = max(1.0, (dx * dx + dy * dy) ** 0.5)
        ux, uy = dx / norm, dy / norm
        px, py = -uy, ux
        tip = (x2, y2)
        base = (x2 - ux * head_len, y2 - uy * head_len)
        left = (base[0] + px * head_w / 2, base[1] + py * head_w / 2)
        right = (base[0] - px * head_w / 2, base[1] - py * head_w / 2)
        draw.polygon([tip, left, right], fill=color)

    _draw_head(pts[-2], pts[-1])
    if bidirectional:
        _draw_head(pts[1], pts[0])


def _is_back_and_forth(seg: List[tuple[int, int]]) -> bool:
    if len(seg) < 3:
        return False
    net_dx = seg[-1][0] - seg[0][0]
    net_dy = seg[-1][1] - seg[0][1]
    net = (net_dx * net_dx + net_dy * net_dy) ** 0.5
    path = 0.0
    for i in range(1, len(seg)):
        dx = seg[i][0] - seg[i - 1][0]
        dy = seg[i][1] - seg[i - 1][1]
        path += (dx * dx + dy * dy) ** 0.5
    if net < 1.0 and path > 18:
        return True
    return path > max(24.0, net * 1.8)


def _draw_arrow_label(
    draw: ImageDraw.ImageDraw,
    pos: tuple[int, int],
    text: str,
    *,
    fill=(255, 255, 255),
    bg=(20, 20, 20),
) -> None:
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
    x, y = pos
    bbox = draw.textbbox((x, y), text, font=font)
    pad_x, pad_y = 5, 3
    rect = (
        bbox[0] - pad_x,
        bbox[1] - pad_y,
        bbox[2] + pad_x,
        bbox[3] + pad_y,
    )
    draw.rounded_rectangle(rect, radius=6, fill=bg)
    draw.text((x, y), text, fill=fill, font=font)


def _step_ranges_from_trajectory_len(step_counts: List[int], traj_len: int) -> List[tuple[int, int]]:
    if traj_len <= 1 or not step_counts:
        return []
    total = max(1, sum(step_counts))
    raw = [max(1, round(traj_len * (c / total))) for c in step_counts]
    diff = traj_len - sum(raw)
    if diff != 0:
        raw[-1] += diff
    ranges = []
    start = 0
    for c in raw:
        end = max(start + 1, start + c)
        end = min(end, traj_len)
        ranges.append((start, end))
        start = end
    if ranges:
        s, _ = ranges[-1]
        ranges[-1] = (s, traj_len)
    return ranges


def _build_alpha_stack_with_arrows(sample: Dict[str, Any], output_path: str, robot: str, hz: int) -> str:
    robotarm_dir = REPO_ROOT / "adhoc" / "robotarm"
    if str(robotarm_dir) not in sys.path:
        sys.path.insert(0, str(robotarm_dir))

    from motion_generation import MotionGenerator, _select_initial_poses
    from vlm_pose_benchmark import _project_3d, _simulate_trajectory

    cfg = _load_cue_config_from_sample(sample)
    base = _build_alpha_stack_image(sample["gif_path"]).convert("RGB")
    draw = ImageDraw.Draw(base)

    first_pose_def = None
    for movement in cfg.get("movements", []):
        if movement.get("type") == "pose":
            first_pose_def = movement.get("parameters", {}).get("pose")
            break

    jsonl_path = REPO_ROOT / "data" / "seed" / "closest_poses_results.jsonl"
    temp_output_root = (REPO_ROOT / "adhoc" / "test" / "_traj_tmp").resolve()
    temp_output_root.mkdir(parents=True, exist_ok=True)

    temp_config_path = None
    config_path = sample.get("config_path")
    if sample["testset"] == "persona":
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix=f"{sample['sample_id']}_", delete=False)
        json.dump([cfg], tmp, indent=2, ensure_ascii=False)
        tmp.close()
        temp_config_path = tmp.name
        config_path = temp_config_path

    gen = None
    try:
        gen = MotionGenerator(
            robot_name=robot,
            jsonl_path=str(jsonl_path),
            output_dir=str(temp_output_root),
            capture_image_width=512,
            capture_image_height=512,
            hz=hz,
        )
        pose_id = sample.get("selected_pose_id") or sample.get("pose_index")
        if pose_id is None and first_pose_def is not None:
            matching = gen._find_matching_poses(first_pose_def)
            selected = _select_initial_poses(matching, first_pose_def, 1)
            if selected:
                pose_id = selected[0]["pose_id"]

        _, trajectory, cam_pos, cam_rot, fovy = _simulate_trajectory(
            gen,
            sample["cue"],
            int(sample["cue_idx"]),
            str(config_path),
            hz=hz,
            pose_index=pose_id,
        )
        pts = []
        for t in trajectory:
            p = _project_3d(t["pos"], cam_pos, cam_rot, fovy, base.size[0])
            if p is not None:
                pts.append(p)
        pts = _offset_overlapping_projected_path(pts)
        if len(pts) < 2:
            return _save_png(base, output_path)

        step_counts = _estimate_step_frame_counts(cfg, hz)
        ranges = _step_ranges_from_trajectory_len(step_counts, len(pts))
        dyn_idx = 0
        for step, (s, e) in zip(cfg.get("movements", []), ranges):
            seg = pts[s:e]
            if len(seg) < 2:
                continue
            step_type = step.get("type")
            if step_type == "path":
                dyn_idx += 1
                sampled = [seg[0]]
                if len(seg) > 4:
                    sampled.append(seg[len(seg) // 2])
                sampled.append(seg[-1])
                bidirectional = _is_back_and_forth(seg)
                _draw_thick_arrow(
                    draw,
                    sampled,
                    color=(255, 255, 255),
                    width=6,
                    head_len=18,
                    head_w=12,
                    bidirectional=bidirectional,
                )
                tip = sampled[-1]
                _draw_arrow_label(draw, (tip[0] + 8, tip[1] - 20), f"m{dyn_idx}")
                if bidirectional:
                    _draw_arrow_label(draw, (sampled[0][0] + 8, sampled[0][1] - 20), f"m{dyn_idx}")
            elif step_type == "movement":
                dyn_idx += 1
                p0 = seg[0]
                p1 = seg[-1]
                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
                norm = max(1.0, (dx * dx + dy * dy) ** 0.5)
                ux, uy = dx / norm, dy / norm
                ox, oy = -uy * 20, ux * 20
                start = (int(p0[0] + ox), int(p0[1] + oy))
                end = (int(start[0] + ux * 60), int(start[1] + uy * 60))
                bidirectional = _is_back_and_forth(seg)
                _draw_thick_arrow(
                    draw,
                    [start, end],
                    color=(255, 255, 255),
                    width=5,
                    head_len=14,
                    head_w=10,
                    bidirectional=bidirectional,
                )
                _draw_arrow_label(draw, (end[0] + 8, end[1] - 20), f"m{dyn_idx}")
                if bidirectional:
                    _draw_arrow_label(draw, (start[0] + 8, start[1] - 20), f"m{dyn_idx}")

        return _save_png(base, output_path)
    finally:
        if gen is not None:
            try:
                gen.env.close()
            except Exception:
                pass
        if temp_config_path and os.path.exists(temp_config_path):
            os.unlink(temp_config_path)


def _load_cue_config_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    if sample["testset"] == "persona":
        return dict(sample["edited_config"])

    config_path = sample.get("config_path")
    if not config_path:
        raise ValueError(f"Missing config_path for sample {sample['sample_id']}")
    configs = _load_json(Path(config_path))
    for cfg in configs:
        if int(cfg.get("idx", -1)) == int(sample["cue_idx"]):
            return cfg
    raise ValueError(f"Config row not found for sample {sample['sample_id']}")


_SIM_BUNDLE_CACHE: dict[str, dict[str, Any]] = {}


def clear_sim_bundle_cache() -> None:
    _SIM_BUNDLE_CACHE.clear()


def get_sim_bundle(
    sample: Dict[str, Any],
    robot: str,
    hz: int,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """
    One simulation pass per (sample, robot, hz, config_path); cache for all trajectory-based media
    and mp4+trajectory in the same process.
    """
    cfg = _load_cue_config_from_sample(sample)
    first_pose_def = None
    for movement in cfg.get("movements", []):
        if movement.get("type") == "pose":
            first_pose_def = movement.get("parameters", {}).get("pose")
            break
    temp_config_path: str | None = None
    config_path = sample.get("config_path")
    if sample["testset"] == "persona":
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix=f"{sample['sample_id']}_", delete=False)
        json.dump([cfg], tmp, indent=2, ensure_ascii=False)
        tmp.close()
        temp_config_path = tmp.name
        config_path = temp_config_path
    ckey = f"{sample.get('sample_id')}|{robot}|{hz}|{os.path.normpath(str(config_path))}"
    if not force and ckey in _SIM_BUNDLE_CACHE:
        return _SIM_BUNDLE_CACHE[ckey]
    if force and ckey in _SIM_BUNDLE_CACHE:
        del _SIM_BUNDLE_CACHE[ckey]

    robotarm_dir = REPO_ROOT / "adhoc" / "robotarm"
    if str(robotarm_dir) not in sys.path:
        sys.path.insert(0, str(robotarm_dir))
    from motion_generation import MotionGenerator, _select_initial_poses
    from vlm_pose_benchmark import _simulate_trajectory

    jsonl_path = REPO_ROOT / "data" / "seed" / "closest_poses_results.jsonl"
    temp_output_root = (REPO_ROOT / "adhoc" / "test" / "_traj_tmp").resolve()
    temp_output_root.mkdir(parents=True, exist_ok=True)
    gen = None
    try:
        gen = MotionGenerator(
            robot_name=robot,
            jsonl_path=str(jsonl_path),
            output_dir=str(temp_output_root),
            capture_image_width=512,
            capture_image_height=512,
            hz=hz,
        )
        pose_id = sample.get("selected_pose_id") or sample.get("pose_index")
        if pose_id is None and first_pose_def is not None:
            matching = gen._find_matching_poses(first_pose_def)
            selected = _select_initial_poses(matching, first_pose_def, 1)
            if selected:
                pose_id = selected[0]["pose_id"]

        frames, trajectory, cam_pos, cam_rot, fovy = _simulate_trajectory(
            gen,
            sample["cue"],
            int(sample["cue_idx"]),
            str(config_path),
            hz=hz,
            pose_index=pose_id,
        )
        if not frames:
            raise RuntimeError(f"No frames captured for {sample['sample_id']}")
        frames_out = [np.asarray(frames[i]) for i in range(len(frames))]
        bundle: dict[str, Any] = {
            "frames": frames_out,
            "trajectory": trajectory,
            "cam_pos": cam_pos,
            "cam_rot": cam_rot,
            "fovy": float(fovy),
            "width": 512,
        }
        _SIM_BUNDLE_CACHE[ckey] = bundle
        return bundle
    finally:
        if gen is not None:
            try:
                gen.env.close()
            except Exception:
                pass
        if temp_config_path and os.path.exists(temp_config_path):
            os.unlink(temp_config_path)


def _project_ee_to_screen(
    trajectory: list[dict[str, Any]],
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    fovy: float,
    width: int,
) -> List[tuple[int, int]]:
    robotarm_dir = REPO_ROOT / "adhoc" / "robotarm"
    if str(robotarm_dir) not in sys.path:
        sys.path.insert(0, str(robotarm_dir))
    from vlm_pose_benchmark import _project_3d

    pts: list[tuple[int, int]] = []
    for t in trajectory:
        p = _project_3d(t["pos"], cam_pos, cam_rot, fovy, width)
        if p is not None:
            pts.append(p)
    return _offset_overlapping_projected_path(pts)


def _image_with_ee_path(base: Image.Image, pts: List[tuple[int, int]]) -> Image.Image:
    img = base.copy().convert("RGB")
    if len(pts) < 2:
        return img
    draw = ImageDraw.Draw(img)
    n = len(pts)
    for i in range(1, n):
        frac = i / max(1, n - 1)
        color = (
            int(255 * (1.0 - frac) + 128 * frac),
            int(220 * (1.0 - frac) + 60 * frac),
            int(40 * (1.0 - frac) + 220 * frac),
        )
        draw.line([pts[i - 1], pts[i]], fill=color, width=5)
    sx, sy = pts[0]
    ex, ey = pts[-1]
    draw.ellipse([sx - 6, sy - 6, sx + 6, sy + 6], fill=(255, 230, 80), outline="white", width=2)
    draw.ellipse([ex - 6, ey - 6, ex + 6, ey + 6], fill=(138, 80, 220), outline="white", width=2)
    return img


def _build_frame_with_ee_trajectory(
    sample: Dict[str, Any],
    output_path: str,
    robot: str,
    hz: int,
    *,
    frame_index: int | None = None,
    force: bool = False,
) -> str:
    """
    One camera frame of the sim with the yellow→purple end-effector path (same as mid-frame, different raster).
    frame_index: 0 = first frame; None = middle frame.
    """
    b = get_sim_bundle(sample, robot, hz, force=force)
    frames: list[np.ndarray] = b["frames"]
    if not frames:
        raise RuntimeError(f"No frames for {sample['sample_id']}")
    nf = len(frames)
    if frame_index is None:
        fidx = nf // 2
    else:
        fidx = max(0, min(int(frame_index), nf - 1))
    w = int(b["width"])
    pts = _project_ee_to_screen(b["trajectory"], b["cam_pos"], b["cam_rot"], b["fovy"], w)
    base = Image.fromarray(frames[fidx]).convert("RGB")
    return _save_png(_image_with_ee_path(base, pts), output_path)


def _build_first_frame_with_trajectory(
    sample: Dict[str, Any], output_path: str, robot: str, hz: int, *, force: bool = False
) -> str:
    return _build_frame_with_ee_trajectory(sample, output_path, robot, hz, frame_index=0, force=force)


def _build_middle_frame_with_trajectory(
    sample: Dict[str, Any], output_path: str, robot: str, hz: int, *, force: bool = False
) -> str:
    return _build_frame_with_ee_trajectory(sample, output_path, robot, hz, frame_index=None, force=force)


def _build_alpha_frame_with_ee(
    sample: Dict[str, Any],
    output_path: str,
    robot: str,
    hz: int,
    *,
    force: bool = False,
) -> str:
    """Stacked-gif underlay with full EE path (sim mid-frame) on top — not config arrows."""
    b = get_sim_bundle(sample, robot, hz, force=force)
    frames: list[np.ndarray] = b["frames"]
    mid = len(frames) // 2
    w = int(b["width"])
    pts = _project_ee_to_screen(b["trajectory"], b["cam_pos"], b["cam_rot"], b["fovy"], w)
    path_layer = _image_with_ee_path(Image.fromarray(frames[mid]).convert("RGB"), pts)
    under = _build_alpha_stack_image(sample["gif_path"]).convert("RGB")
    if under.size != path_layer.size:
        under = under.resize(path_layer.size, Image.LANCZOS)
    blended = Image.blend(under, path_layer, 0.5)
    return _save_png(blended, output_path)


def _build_mp4_plus_trajectory(
    sample: Dict[str, Any],
    output_path: str,
    robot: str,
    hz: int,
    *,
    force: bool = False,
) -> str:
    """MP4: each sim frame with cumulative yellow→purple EE path (same sim run as other trajectory media)."""
    out = Path(output_path)
    if out.exists() and not force:
        return str(out)
    b = get_sim_bundle(sample, robot, hz, force=force)
    frames: list[np.ndarray] = b["frames"]
    trajectory: list[dict[str, Any]] = b["trajectory"]
    if not frames:
        raise RuntimeError("no frames")
    w = int(b["width"])
    tdir = Path(tempfile.mkdtemp(prefix="mp4trj_", dir=str(out.parent)))
    try:
        n = len(frames)
        for k in range(n):
            sub = trajectory[: k + 1]
            pts = _project_ee_to_screen(sub, b["cam_pos"], b["cam_rot"], b["fovy"], w)
            base = Image.fromarray(frames[k]).convert("RGB")
            img = _image_with_ee_path(base, pts)
            img.save(tdir / f"frame_{k:04d}.png")
        ff = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        cmd = [
            ff,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(int(hz)),
            "-i",
            str(tdir / "frame_%04d.png"),
            "-movflags",
            "+faststart",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ]
        r = subprocess.run(cmd, text=True, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout or "").strip() or "ffmpeg mp4+trajectory failed")
    finally:
        try:
            shutil.rmtree(tdir, ignore_errors=True)
        except Exception:
            pass
    return str(out)


def _offset_overlapping_projected_path(
    pts: List[tuple[int, int]],
    *,
    quant: int = 10,
    step_px: int = 5,
    max_offset_px: int = 18,
) -> List[tuple[int, int]]:
    if len(pts) < 2:
        return pts

    visited_counts: dict[tuple[int, int], int] = {}
    adjusted: List[tuple[int, int]] = []

    for i, (x, y) in enumerate(pts):
        prev_pt = pts[max(0, i - 1)]
        next_pt = pts[min(len(pts) - 1, i + 1)]
        dx = next_pt[0] - prev_pt[0]
        dy = next_pt[1] - prev_pt[1]

        # Use screen-space normal so each revisit becomes a nearby parallel trace.
        norm = max(1.0, (dx * dx + dy * dy) ** 0.5)
        nx = -dy / norm
        ny = dx / norm

        key = (round(x / quant), round(y / quant))
        count = visited_counts.get(key, 0)
        visited_counts[key] = count + 1

        if count == 0:
            adjusted.append((x, y))
            continue

        mag = min(max_offset_px, count * step_px)
        # Keep the offset direction stable so back-and-forth passes separate cleanly.
        ox = int(round(nx * mag))
        oy = int(round(ny * mag))

        # If the local normal is numerically tiny, fall back to a simple upward shift.
        if abs(ox) + abs(oy) < 2:
            ox, oy = 0, -mag

        adjusted.append((x + ox, y + oy))

    return adjusted


def _find_latest_single_gif(render_dir: Path, cue: str) -> str | None:
    safe_cue = cue.replace("/", "_").replace("\\", "_").replace(" ", "_")
    patterns = [
        str(render_dir / f"*_{safe_cue}_*.gif"),
        str(render_dir / f"*_{safe_cue}_tiled.gif"),
    ]
    return _latest_match(patterns)


def gif_to_mp4(
    gif_path: str,
    *,
    output_path: str | None = None,
    force: bool = False,
) -> str:
    gif_file = Path(gif_path)
    if output_path is None:
        output_path = str(gif_file.with_suffix(".mp4"))
    mp4_path = Path(output_path)
    if mp4_path.exists() and not force:
        return str(mp4_path)

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    ff = shutil.which("ffmpeg")
    if not ff:
        ff = "/opt/homebrew/bin/ffmpeg"
    cmd = [ff, "-y", "-i", str(gif_file), "-movflags", "+faststart", "-pix_fmt", "yuv420p", str(mp4_path)]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffmpeg failed for {gif_path}")
    return str(mp4_path)


def _canonicalize_test_media_type(test_type: str) -> str:
    """
    Normalize CLI / alias names to internal tags. Trajectory types = EE path (yellow→purple) on a chosen raster;
    alpha_frame* = stacked GIF frames (2D), with or without motion arrows from sim.
    """
    s = (test_type or "").strip().lower().replace("-", "_").replace("+", "_")
    syn = {
        "mid_frame_trajectory": "middle_frame_trajectory",
        "mp4_trajectory": "mp4_plus_trajectory",
        "mp4_traj": "mp4_plus_trajectory",
        "alpha_stack": "alpha_frame",
        "alpha_stack_final": "alpha_frame",
        "alpha_stack_arrow": "alpha_frame_trajectory",
        "alpha_stack_final_arrow": "alpha_frame_trajectory",
        "trajectory": "middle_frame_trajectory",
    }
    s = syn.get(s, s)
    if s in ("first", "first_frame"):
        s = "first_frame_only"
    if s in ("middle", "middle_frame") and "trajectory" not in s:
        s = "middle_frame_only"
    allowed = {
        "first_frame_only",
        "first_frame_trajectory",
        "middle_frame_only",
        "middle_frame_trajectory",
        "alpha_frame",
        "alpha_frame_trajectory",
        "gif",
        "mp4",
        "mp4_plus_trajectory",
        # legacy
        "alpha_stack_final",
        "alpha_stack_final_arrow",
    }
    if s in ("alpha_stack_final",):
        s = "alpha_frame"
    if s in ("alpha_stack_final_arrow",):
        s = "alpha_frame_trajectory"
    if s not in allowed:
        raise ValueError(f"Unknown test_type: {test_type!r} (after alias: {s!r})")
    return s


def prepare_test_media(
    samples: List[Dict[str, Any]],
    *,
    test_type: str,
    robot: str = "IIWA",
    hz: int = 8,
    output_dir: str = "adhoc/vlm_test/eval_media_variants",
    force: bool = False,
) -> List[Dict[str, Any]]:
    canonical = _canonicalize_test_media_type(test_type)

    target_root = (REPO_ROOT / output_dir / canonical).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    prepared: List[Dict[str, Any]] = []

    for sample in samples:
        updated = dict(sample)
        stem = _safe_name(sample["sample_id"])
        gif_path = sample["gif_path"]

        if canonical == "gif":
            updated["test_media_type"] = "gif"
            updated["media_path"] = gif_path
            updated["media_mime"] = "image/gif"
            prepared.append(updated)
            continue

        if canonical == "mp4":
            mp4_path = gif_to_mp4(gif_path, output_path=str(target_root / f"{stem}.mp4"), force=force)
            updated["test_media_type"] = "mp4"
            updated["media_path"] = mp4_path
            updated["media_mime"] = "video/mp4"
            prepared.append(updated)
            continue

        if canonical == "mp4_plus_trajectory":
            out_mp4 = target_root / f"{stem}.mp4"
            _build_mp4_plus_trajectory(sample, str(out_mp4), robot, hz, force=force)
            updated["test_media_type"] = "mp4_plus_trajectory"
            updated["media_path"] = str(out_mp4)
            updated["media_mime"] = "video/mp4"
            prepared.append(updated)
            continue

        out_png = target_root / f"{stem}.png"
        if not out_png.exists() or force:
            if canonical == "first_frame_only":
                image = _extract_frame_image(gif_path, 0)
                _save_png(image, str(out_png))
            elif canonical == "middle_frame_only":
                image = _extract_frame_image(gif_path, _gif_frame_count(gif_path) // 2)
                _save_png(image, str(out_png))
            elif canonical == "alpha_frame":
                image = _build_alpha_stack_image(gif_path)
                _save_png(image, str(out_png))
            elif canonical == "alpha_frame_trajectory":
                _build_alpha_frame_with_ee(sample, str(out_png), robot=robot, hz=hz, force=force)
            elif canonical == "middle_frame_trajectory":
                _build_middle_frame_with_trajectory(sample, str(out_png), robot=robot, hz=hz, force=force)
            elif canonical == "first_frame_trajectory":
                _build_first_frame_with_trajectory(sample, str(out_png), robot=robot, hz=hz, force=force)

        updated["test_media_type"] = canonical
        updated["media_path"] = str(out_png)
        updated["media_mime"] = "image/png"
        prepared.append(updated)

    return prepared


def prepare_normalized_eval_media(
    samples: List[Dict[str, Any]],
    *,
    robot: str = "IIWA",
    render_hz: int = 8,
    preview_speed_scale: float = 1.0,
    preview_hold_scale: float = 1.0,
    top_k: int = 1,
    output_dir: str = "adhoc/vlm_test/eval_media",
    force: bool = False,
) -> List[Dict[str, Any]]:
    script_dir = REPO_ROOT / "adhoc" / "robotarm"
    motion_script = script_dir / "motion_generation.py"
    jsonl_path = REPO_ROOT / "data" / "seed" / "closest_poses_results.jsonl"
    target_root = (REPO_ROOT / output_dir).resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    prepared: List[Dict[str, Any]] = []
    for sample in samples:
        stable_gif = target_root / f"{sample['sample_id']}.gif"
        if stable_gif.exists() and not force:
            updated = dict(sample)
            updated["gif_path"] = str(stable_gif)
            prepared.append(updated)
            continue

        render_workspace = target_root / "_raw" / sample["sample_id"]
        render_workspace.mkdir(parents=True, exist_ok=True)

        config_path = sample.get("config_path")
        temp_config_path = None
        if sample["testset"] == "persona":
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix=f"{sample['sample_id']}_", delete=False)
            json.dump([sample["edited_config"]], tmp, indent=2, ensure_ascii=False)
            tmp.close()
            temp_config_path = tmp.name
            config_path = temp_config_path

        cmd = [
            sys.executable,
            str(motion_script),
            f"--robot={robot}",
            f"--cue_idx={int(sample['cue_idx'])}",
            f"--config_path={config_path}",
            f"--jsonl_path={jsonl_path}",
            f"--output_dir={render_workspace}",
            f"--hz={int(render_hz)}",
            f"--top_k={int(top_k)}",
            f"--preview_speed_scale={float(preview_speed_scale)}",
            f"--preview_hold_scale={float(preview_hold_scale)}",
        ]

        try:
            result = subprocess.run(cmd, text=True, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "render failed")
            latest = _find_latest_single_gif(render_workspace / robot, sample["cue"])
            if not latest:
                raise FileNotFoundError(f"No rendered gif found for {sample['sample_id']}")
            shutil.copy2(latest, stable_gif)
            updated = dict(sample)
            updated["gif_path"] = str(stable_gif)
            prepared.append(updated)
        finally:
            if temp_config_path and os.path.exists(temp_config_path):
                os.unlink(temp_config_path)

    return prepared


def build_binary_tasks(samples: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    cues = [sample["cue"] for sample in samples]
    tasks: List[Dict[str, Any]] = []
    for sample in samples:
        tasks.append(
            {
                "task_id": f"{sample['sample_id']}__pos",
                "task_family": "binary_match",
                "label": 1,
                "display_cue": sample["cue"],
                "sample": sample,
            }
        )
        distractors = _filter_nonoverlapping_cues(sample["cue"], [cue for cue in cues if cue != sample["cue"]])
        if not distractors:
            distractors = [cue for cue in cues if cue != sample["cue"]]
        if not distractors:
            continue
        tasks.append(
            {
                "task_id": f"{sample['sample_id']}__neg",
                "task_family": "binary_match",
                "label": 0,
                "display_cue": rng.choice(distractors),
                "sample": sample,
            }
        )
    return tasks


def build_mcq_tasks(samples: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    unique_cues = sorted({sample["cue"] for sample in samples})
    if len(unique_cues) < 4:
        raise ValueError("MCQ task requires at least 4 unique cues")
    tasks: List[Dict[str, Any]] = []
    for sample in samples:
        distractors = _filter_nonoverlapping_cues(sample["cue"], [cue for cue in unique_cues if cue != sample["cue"]])
        if len(distractors) < 3:
            distractors = [cue for cue in unique_cues if cue != sample["cue"]]
        choices = [sample["cue"]] + rng.sample(distractors, 3)
        rng.shuffle(choices)
        tasks.append(
            {
                "task_id": f"{sample['sample_id']}__mcq4",
                "task_family": "mcq4",
                "correct_choice": sample["cue"],
                "choices": choices,
                "sample": sample,
            }
        )
    return tasks


def _cue_tokens(cue: str) -> set[str]:
    normalized = []
    for ch in str(cue).lower():
        normalized.append(ch if ch.isalnum() else "_")
    return {tok for tok in "".join(normalized).split("_") if tok}


def _filter_nonoverlapping_cues(target_cue: str, candidate_cues: List[str]) -> List[str]:
    target_tokens = _cue_tokens(target_cue)
    filtered = []
    for cue in candidate_cues:
        if target_tokens.isdisjoint(_cue_tokens(cue)):
            filtered.append(cue)
    return filtered


# --- Human-eval VLM (run_exp / show_html): input_type names for prepare_test_media ---

INPUT_TYPE_CHOICES = (
    "mp4",
    "alpha_frame",
    "first_frame_trajectory",
    "alpha_frame_trajectory",
    "mp4_plus_trajectory",
    "mid_frame_trajectory",
    "all",
    "gif",
    "first_frame_only",
    "middle_frame_only",
)


def normalize_test_media_type(name: str) -> str:
    """Map CLI names to internal tags for ``prepare_test_media(..., test_type=...)``."""
    return _canonicalize_test_media_type(name)


def expand_input_types(input_type: str) -> List[str]:
    """``all`` = five: mp4, alpha_frame, first_frame_trajectory, alpha_frame_trajectory, mp4_plus_trajectory."""
    s = (input_type or "").strip().lower().replace("-", "_").replace("+", "_")
    if s == "all":
        return [
            "mp4",
            "alpha_frame",
            "first_frame_trajectory",
            "alpha_frame_trajectory",
            "mp4_plus_trajectory",
        ]
    return [normalize_test_media_type(s)]
