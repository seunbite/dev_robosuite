from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import fire
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SEED = ROOT / "data" / "seed"
MOTIONS = ROOT / "data" / "motions"
POSE_DB = SEED / "closest_poses_results.jsonl"
OUT_ROOT = SEED / "vlm_pairwise_prompt19_compare"
FALLBACK_GEMINI_KEY_FILE = Path("/Users/sb/Downloads/workspace/Motion2Mind/src/motion2mind/vlm/delete_gemini.py")

sys.path.insert(0, str(ROOT / "adhoc" / "robotarm"))
from motion_generation import MotionGenerator, _path_duration_from_length, _select_initial_poses  # noqa: E402
from vlm_pose_benchmark import _project_3d, _simulate_trajectory  # noqa: E402


DATASET_CONFIGS = {
    "iconic": {
        "sophisticated": SEED / "motion_configs_prompt_v19_sophisticated.json",
        "no_reasoning": SEED / "baseline_prompt19_full_no_reasoning" / "motion_configs_prompt_v19_sophisticated_no_reasoning_iconic.json",
        "sophisticated_gif_dir": MOTIONS / "v19_sophisticated" / "IIWA",
        "no_reasoning_gif_dir": MOTIONS / "baseline_prompt19_full_no_reasoning" / "no_reasoning_iconic" / "IIWA",
    },
    "contextual": {
        "sophisticated": SEED / "motion_configs_prompt_v19_sophisticated_contextual.json",
        "no_reasoning": SEED / "baseline_prompt19_full_no_reasoning" / "motion_configs_prompt_v19_sophisticated_no_reasoning_contextual.json",
        "sophisticated_gif_dir": MOTIONS / "v19_sophisticated_contextual" / "IIWA",
        "no_reasoning_gif_dir": MOTIONS / "baseline_prompt19_full_no_reasoning" / "no_reasoning_contextual" / "IIWA",
    },
}


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text)).strip("_") or "item"


def _load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_row(path: Path, cue_idx: int) -> dict:
    rows = _load_json(path)
    for row in rows:
        if int(row["idx"]) == int(cue_idx):
            return row
    raise ValueError(f"cue_idx={cue_idx} not found in {path}")


def _latest_single_gif(base: Path, cue: str) -> Path | None:
    safe = _safe_name(cue)
    matches = sorted(base.rglob(f"*_{safe}_p*.gif"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _ensure_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key
    if FALLBACK_GEMINI_KEY_FILE.exists():
        text = FALLBACK_GEMINI_KEY_FILE.read_text(encoding="utf-8")
        match = re.search(r'api_key="([^"]+)"', text)
        if match:
            api_key = match.group(1).strip()
            os.environ["GOOGLE_API_KEY"] = api_key
            return api_key
    raise ValueError("GOOGLE_API_KEY is not set and no local fallback key was found")


def _parse_json_object(raw_text: str) -> dict:
    raw = raw_text.strip()
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    candidates = [raw] + [blk.strip() for blk in fenced if blk.strip()]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for pos, ch in enumerate(candidate):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(candidate[pos:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError(f"Could not parse JSON object from model output: {raw[:500]}")


def _gif_frame_count(gif_path: Path) -> int:
    img = Image.open(gif_path)
    return int(getattr(img, "n_frames", 1))


def _extract_frame(gif_path: Path, frame_idx: int) -> Image.Image:
    img = Image.open(gif_path)
    img.seek(frame_idx)
    return img.convert("RGB")


def _sample_gif_frames(gif_path: Path, num_frames: int = 6) -> list[Image.Image]:
    total = _gif_frame_count(gif_path)
    if total <= 1:
        return [Image.open(gif_path).convert("RGB")]
    indices = sorted(set(int(round(i * (total - 1) / max(1, num_frames - 1))) for i in range(num_frames)))
    return [_extract_frame(gif_path, idx) for idx in indices]


def _all_gif_frames(gif_path: Path) -> list[Image.Image]:
    img = Image.open(gif_path)
    total = int(getattr(img, "n_frames", 1))
    frames = []
    for idx in range(total):
        img.seek(idx)
        frames.append(img.convert("RGB"))
    return frames


def _motion_signal_from_frames(frames: list[Image.Image]) -> np.ndarray:
    if not frames:
        return np.zeros((0,), dtype=np.float32)
    arrs = []
    for frame in frames:
        gray = frame.convert("L").resize((96, 96))
        arrs.append(np.asarray(gray, dtype=np.float32).reshape(-1))
    X = np.stack(arrs, axis=0)
    X = X - X.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        axis = vh[0]
        signal = X @ axis
    except np.linalg.LinAlgError:
        signal = np.arange(len(frames), dtype=np.float32)
    signal = signal.astype(np.float32)
    signal -= float(signal.mean())
    return signal


def _turning_point_indices_from_frames(frames: list[Image.Image], num_frames: int = 6) -> list[int]:
    if len(frames) <= num_frames:
        return list(range(len(frames)))
    if len(frames) < 3:
        return sorted(set([0, len(frames) - 1]))

    # Build a simple motion embedding from the rendered frames themselves.
    # We project each frame onto the first PCA axis, then pick local extrema
    # over time as turning points for oscillatory motions like fan / wave.
    signal = _motion_signal_from_frames(frames)

    extrema = []
    for i in range(1, len(signal) - 1):
        prev_v, cur_v, next_v = signal[i - 1], signal[i], signal[i + 1]
        if (cur_v >= prev_v and cur_v > next_v) or (cur_v > prev_v and cur_v >= next_v):
            extrema.append(i)
        elif (cur_v <= prev_v and cur_v < next_v) or (cur_v < prev_v and cur_v <= next_v):
            extrema.append(i)

    chosen = [0]
    for idx in extrema:
        if idx not in chosen:
            chosen.append(idx)
    if (len(frames) - 1) not in chosen:
        chosen.append(len(frames) - 1)

    # Rank candidates by how different they are from their neighbors so that
    # stronger turning points survive when we need to trim.
    def turning_strength(i: int) -> float:
        if i <= 0 or i >= len(signal) - 1:
            return float("inf")
        return abs(signal[i] - signal[i - 1]) + abs(signal[i] - signal[i + 1])

    endpoints = [0, len(frames) - 1]
    mids = [i for i in chosen if i not in endpoints]
    mids = sorted(mids, key=lambda i: (-turning_strength(i), i))
    chosen = endpoints[:1] + mids + endpoints[1:]

    # Trim or pad to the requested count.
    dedup = []
    for idx in chosen:
        if idx not in dedup:
            dedup.append(idx)
    chosen = dedup

    if len(chosen) > num_frames:
        keep = [0]
        middle = chosen[1:-1]
        slots = max(0, num_frames - 2)
        if slots > 0 and middle:
            middle = sorted(middle)
            picked = sorted(set(int(round(i * (len(middle) - 1) / max(1, slots - 1))) for i in range(slots)))
            keep.extend(middle[i] for i in picked)
        if num_frames > 1:
            keep.append(len(frames) - 1)
        chosen = sorted(set(keep))

    if len(chosen) < num_frames:
        uniform = sorted(set(int(round(i * (len(frames) - 1) / max(1, num_frames - 1))) for i in range(num_frames)))
        for idx in uniform:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= num_frames:
                break

    return sorted(chosen[:num_frames])


def _sample_gif_frames_turning_points(gif_path: Path, num_frames: int = 6) -> list[Image.Image]:
    frames = _all_gif_frames(gif_path)
    indices = _turning_point_indices_from_frames(frames, num_frames=num_frames)
    return [frames[idx] for idx in indices]


def _single_cycle_indices_from_frames(frames: list[Image.Image], num_frames: int = 6) -> list[int]:
    if len(frames) <= num_frames:
        return list(range(len(frames)))
    if len(frames) < 6:
        return _turning_point_indices_from_frames(frames, num_frames=num_frames)

    signal = _motion_signal_from_frames(frames)
    n = len(signal)
    max_lag = max(3, min(n // 2, 24))
    best_lag = None
    best_score = -float("inf")
    denom = float(np.dot(signal, signal)) + 1e-6
    for lag in range(3, max_lag + 1):
        left = signal[:-lag]
        right = signal[lag:]
        score = float(np.dot(left, right) / denom)
        # Prefer a meaningful oscillatory period rather than tiny lags.
        score -= 0.01 * lag
        if score > best_score:
            best_score = score
            best_lag = lag

    if best_lag is None or best_lag < 3:
        return _turning_point_indices_from_frames(frames, num_frames=num_frames)

    period = int(best_lag)
    if period >= n:
        return _turning_point_indices_from_frames(frames, num_frames=num_frames)

    diffs = np.abs(np.diff(signal))
    best_start = 0
    best_energy = -float("inf")
    for start in range(0, n - period):
        end = start + period
        energy = float(diffs[start:end].sum() + (signal[start:end + 1].max() - signal[start:end + 1].min()))
        if energy > best_energy:
            best_energy = energy
            best_start = start

    # Align the cycle to begin near a local extreme so the sampled phases are distinct.
    window_start = best_start
    window_end = min(n - 1, best_start + period)
    extrema = _turning_point_indices_from_frames(frames[window_start:window_end + 1], num_frames=max(4, num_frames))
    if extrema:
        window_start = min(n - 1, window_start + extrema[0])
        window_end = min(n - 1, window_start + period)

    span = max(1, window_end - window_start)
    phase_indices = []
    for i in range(num_frames):
        frac = i / max(1, num_frames - 1)
        idx = int(round(window_start + frac * span))
        phase_indices.append(min(n - 1, max(0, idx)))

    chosen = sorted(set(phase_indices))
    if len(chosen) < num_frames:
        fallback = _turning_point_indices_from_frames(frames[window_start:window_end + 1], num_frames=num_frames)
        for idx in fallback:
            global_idx = min(n - 1, window_start + idx)
            if global_idx not in chosen:
                chosen.append(global_idx)
            if len(chosen) >= num_frames:
                break
    return sorted(chosen[:num_frames])


def _sample_gif_frames_single_cycle(gif_path: Path, num_frames: int = 6) -> list[Image.Image]:
    frames = _all_gif_frames(gif_path)
    indices = _single_cycle_indices_from_frames(frames, num_frames=num_frames)
    return [frames[idx] for idx in indices]


def _estimate_step_segments(row: dict, hz: int) -> list[dict]:
    segments: list[dict] = []
    cursor = 0
    for step_idx, movement in enumerate(row.get("movements", [])):
        step_type = movement.get("type")
        params = movement.get("parameters", {})
        if step_type == "pose":
            speed = float(params.get("speed", 1.0) or 1.0)
            hold_time = float(params.get("hold_time", 0.0) or 0.0)
            pose_obj = params.get("pose")
            has_xyz = isinstance(pose_obj, dict) and any(k in pose_obj for k in ("x", "y", "z"))
            # First pose is set directly with hold only; later poses transition.
            transition_frames = 0 if step_idx == 0 else max(1, int((1.0 / max(speed, 1e-6)) * hz))
            hold_frames = int(hold_time * hz)
            total = transition_frames + hold_frames
            segments.append(
                {
                    "step_idx": step_idx,
                    "type": step_type,
                    "start": cursor,
                    "end": cursor + total,
                    "transition_frames": transition_frames,
                    "hold_frames": hold_frames,
                    "has_xyz": has_xyz,
                }
            )
            cursor += total
        elif step_type == "movement":
            repetition = int(params.get("repetition", 1) or 1)
            directions = list(params.get("directions", []))
            step_start = cursor
            subsegments = []
            for rep_idx in range(repetition):
                for dir_idx, direction in enumerate(directions):
                    speed = float(direction.get("speed", 1.0) or 1.0)
                    hold_time = float(direction.get("hold_time", 0.0) or 0.0)
                    move_frames = max(1, int((1.0 / max(speed, 1e-6)) * hz))
                    hold_frames = int(hold_time * hz)
                    sub_start = cursor
                    cursor += move_frames + hold_frames
                    subsegments.append(
                        {
                            "rep_idx": rep_idx,
                            "dir_idx": dir_idx,
                            "start": sub_start,
                            "move_end": sub_start + move_frames,
                            "end": cursor,
                            "degrees": direction.get("degrees", {}),
                        }
                    )
            segments.append(
                {
                    "step_idx": step_idx,
                    "type": step_type,
                    "start": step_start,
                    "end": cursor,
                    "repetition": repetition,
                    "directions": directions,
                    "subsegments": subsegments,
                }
            )
        elif step_type == "path":
            shape = params.get("shape")
            speed = float(params.get("speed", 1.0) or 1.0)
            hold_time = float(params.get("hold_time", 0.0) or 0.0)
            total_frames = 0
            if shape == "line":
                distance = params.get("distance")
                if isinstance(distance, dict):
                    path_length_deg = float(np.sqrt(sum(float(v) ** 2 for v in distance.values())))
                else:
                    path_length_deg = abs(float(distance or 0.0))
                total_frames = max(1, int(_path_duration_from_length(path_length_deg, speed) * hz))
            elif shape in {"arc", "circle"}:
                radius = float(params.get("radius", 0.0) or 0.0)
                sweep = float(params.get("sweep", 360 if shape == "circle" else 0.0) or 0.0)
                path_length_deg = abs(radius * math.radians(sweep))
                total_frames = max(1, int(_path_duration_from_length(path_length_deg, speed) * hz))
            else:
                total_frames = max(1, int((1.0 / max(speed, 1e-6)) * hz))
            hold_frames = int(hold_time * hz)
            segments.append(
                {
                    "step_idx": step_idx,
                    "type": step_type,
                    "start": cursor,
                    "move_end": cursor + total_frames,
                    "end": cursor + total_frames + hold_frames,
                    "shape": shape,
                }
            )
            cursor += total_frames + hold_frames
        elif step_type == "gripper":
            hold_time = float(params.get("hold_time", 0.0) or 0.0)
            total = int(hold_time * hz)
            segments.append(
                {
                    "step_idx": step_idx,
                    "type": step_type,
                    "start": cursor,
                    "end": cursor + total,
                }
            )
            cursor += total
    return segments


def _fit_estimated_index(idx: int, estimated_total: int, actual_total: int) -> int:
    if actual_total <= 1:
        return 0
    if estimated_total <= 1:
        return min(actual_total - 1, max(0, idx))
    frac = idx / max(1, estimated_total - 1)
    return min(actual_total - 1, max(0, int(round(frac * (actual_total - 1)))))


def _config_aware_indices(row: dict, total_frames: int, num_frames: int = 6, hz: int = 8) -> list[int]:
    segments = _estimate_step_segments(row, hz=hz)
    if total_frames <= num_frames:
        return list(range(total_frames))

    estimated_total = max(1, segments[-1]["end"] if segments else total_frames)
    chosen_estimated: list[int] = []

    # Start context.
    chosen_estimated.append(0)

    # Prefer repeated movement phase boundaries such as up/down/up/down.
    for seg in segments:
        if seg.get("type") != "movement":
            continue
        subsegments = seg.get("subsegments", [])
        if seg.get("repetition", 1) > 1 and len(seg.get("directions", [])) >= 2:
            for sub in subsegments:
                chosen_estimated.append(max(sub["start"], sub["move_end"] - 1))

    # Add ends of other dynamic steps if we still need more variety.
    if len(chosen_estimated) < num_frames:
        for seg in segments:
            if seg.get("type") == "movement":
                subsegments = seg.get("subsegments", [])
                if subsegments:
                    for sub in subsegments:
                        chosen_estimated.append(max(sub["start"], sub["move_end"] - 1))
                continue
            if seg.get("type") == "path":
                chosen_estimated.append(max(seg["start"], seg["move_end"] - 1))
            elif seg.get("type") == "pose" and seg.get("end", seg.get("start", 0)) > seg.get("start", 0):
                chosen_estimated.append(max(seg["start"], seg["end"] - 1))

    # End context.
    chosen_estimated.append(max(0, estimated_total - 1))

    dedup_estimated = []
    for idx in chosen_estimated:
        if idx not in dedup_estimated:
            dedup_estimated.append(idx)

    mapped = [_fit_estimated_index(idx, estimated_total, total_frames) for idx in dedup_estimated]
    chosen = []
    for idx in mapped:
        if idx not in chosen:
            chosen.append(idx)

    if len(chosen) > num_frames:
        # Preserve start/end and evenly subsample the middle.
        start = [chosen[0]]
        end = [chosen[-1]] if num_frames > 1 else []
        middle = chosen[1:-1]
        slots = max(0, num_frames - len(start) - len(end))
        picked = []
        if slots > 0 and middle:
            if len(middle) <= slots:
                picked = middle
            else:
                sel = sorted(set(int(round(i * (len(middle) - 1) / max(1, slots - 1))) for i in range(slots)))
                picked = [middle[i] for i in sel]
        chosen = start + picked + end

    if len(chosen) < num_frames:
        filler = _single_cycle_indices_from_frames([Image.new("RGB", (1, 1))] * total_frames, num_frames=num_frames) if False else []
        uniform = sorted(set(int(round(i * (total_frames - 1) / max(1, num_frames - 1))) for i in range(num_frames)))
        for idx in uniform:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= num_frames:
                break

    return sorted(chosen[:num_frames])


def _sample_gif_frames_config_aware(gif_path: Path, row: dict, num_frames: int = 6, hz: int = 8) -> list[Image.Image]:
    frames = _all_gif_frames(gif_path)
    indices = _config_aware_indices(row, len(frames), num_frames=num_frames, hz=hz)
    return [frames[idx] for idx in indices]


def _label_frame(img: Image.Image, text: str) -> Image.Image:
    label_h = 36
    canvas = Image.new("RGB", (img.width, img.height + label_h), "white")
    canvas.paste(img, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), text, fill="black")
    return canvas


def _make_frame_strip_compare(
    cue: str,
    left_gif: Path,
    right_gif: Path,
    out_path: Path,
    num_frames: int = 6,
    sampler: str = "uniform",
    left_row: dict | None = None,
    right_row: dict | None = None,
    hz: int = 8,
) -> Path:
    if sampler == "turning_points":
        left_src = _sample_gif_frames_turning_points(left_gif, num_frames=num_frames)
        right_src = _sample_gif_frames_turning_points(right_gif, num_frames=num_frames)
    elif sampler == "single_cycle":
        left_src = _sample_gif_frames_single_cycle(left_gif, num_frames=num_frames)
        right_src = _sample_gif_frames_single_cycle(right_gif, num_frames=num_frames)
    elif sampler == "config_aware":
        if left_row is None or right_row is None:
            raise ValueError("config_aware sampler requires left_row and right_row")
        left_src = _sample_gif_frames_config_aware(left_gif, left_row, num_frames=num_frames, hz=hz)
        right_src = _sample_gif_frames_config_aware(right_gif, right_row, num_frames=num_frames, hz=hz)
    else:
        left_src = _sample_gif_frames(left_gif, num_frames=num_frames)
        right_src = _sample_gif_frames(right_gif, num_frames=num_frames)
    left_frames = [_label_frame(img, "A Sophisticated") for img in left_src]
    right_frames = [_label_frame(img, "B No Reasoning") for img in right_src]
    cell_w = left_frames[0].width
    cell_h = left_frames[0].height
    pad = 10
    title_h = 50
    canvas = Image.new("RGB", (pad + (cell_w + pad) * num_frames, title_h + pad + (cell_h + pad) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 14), f"Cue: {cue} | Compare A vs B", fill="black")
    for i, frame in enumerate(left_frames):
        x = pad + i * (cell_w + pad)
        y = title_h
        canvas.paste(frame, (x, y))
    for i, frame in enumerate(right_frames):
        x = pad + i * (cell_w + pad)
        y = title_h + cell_h + pad
        canvas.paste(frame, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")
    return out_path


def _yellow_to_purple(frac: float) -> tuple[int, int, int]:
    frac = max(0.0, min(1.0, frac))
    yellow = np.array([255, 230, 0], dtype=float)
    purple = np.array([126, 87, 194], dtype=float)
    rgb = yellow + (purple - yellow) * frac
    return tuple(int(round(v)) for v in rgb)


def _render_trajectory_lastframe_image(
    base_frame: Image.Image,
    trajectory: list[dict],
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    fovy: float,
    img_size: int = 512,
) -> Image.Image:
    img = base_frame.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    pts = []
    for t in trajectory:
        p = _project_3d(t["pos"], cam_pos, cam_rot, fovy, img_size)
        if p:
            pts.append(p)
    if not pts:
        return img

    for i in range(1, len(pts)):
        frac = i / max(1, len(pts) - 1)
        draw.line([pts[i - 1], pts[i]], fill=_yellow_to_purple(frac), width=4)

    sx, sy = pts[0]
    ex, ey = pts[-1]
    draw.ellipse([sx - 7, sy - 7, sx + 7, sy + 7], fill=(255, 230, 0), outline="white", width=2)
    draw.ellipse([ex - 7, ey - 7, ex + 7, ey + 7], fill=(126, 87, 194), outline="white", width=2)
    return img


def _deterministic_pose_id(gen: MotionGenerator, row: dict) -> int | None:
    first_pose = None
    for movement in row.get("movements", []):
        if movement.get("type") == "pose":
            first_pose = movement["parameters"]["pose"]
            break
    if first_pose is None:
        return None
    matching = gen._find_matching_poses(first_pose)
    selected = _select_initial_poses(matching, first_pose, top_k=1)
    if not selected:
        return None
    return int(selected[0]["pose_id"])


def _make_trajectory_compare(
    cue: str,
    cue_idx: int,
    left_row: dict,
    right_row: dict,
    left_config: Path,
    right_config: Path,
    out_path: Path,
    hz: int = 8,
    img_size: int = 512,
) -> Path:
    gen = MotionGenerator(
        robot_name="IIWA",
        jsonl_path=str(POSE_DB),
        has_renderer=False,
        has_offscreen_renderer=True,
        output_dir=str(MOTIONS / "tmp_vlm_pairwise"),
        capture_image_width=img_size,
        capture_image_height=img_size,
        hz=hz,
    )
    try:
        left_pose_id = _deterministic_pose_id(gen, left_row)
        left_frames, left_traj, left_cam_pos, left_cam_rot, left_fovy = _simulate_trajectory(
            gen, cue, cue_idx, str(left_config), hz=hz, pose_index=left_pose_id
        )
        gen._set_joint_positions(gen.initial_joint_pos)
        right_pose_id = _deterministic_pose_id(gen, right_row)
        right_frames, right_traj, right_cam_pos, right_cam_rot, right_fovy = _simulate_trajectory(
            gen, cue, cue_idx, str(right_config), hz=hz, pose_index=right_pose_id
        )
    finally:
        gen.close()

    left_base = Image.fromarray(np.array(left_frames[-1]))
    right_base = Image.fromarray(np.array(right_frames[-1]))
    left_img = _label_frame(
        _render_trajectory_lastframe_image(left_base, left_traj, left_cam_pos, left_cam_rot, left_fovy, img_size=img_size),
        "A Sophisticated",
    )
    right_img = _label_frame(
        _render_trajectory_lastframe_image(right_base, right_traj, right_cam_pos, right_cam_rot, right_fovy, img_size=img_size),
        "B No Reasoning",
    )
    pad = 16
    title_h = 48
    canvas = Image.new("RGB", (left_img.width + right_img.width + pad * 3, max(left_img.height, right_img.height) + title_h + pad * 2), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 14), f"Cue: {cue} | Last frame + trajectory overlay", fill="black")
    canvas.paste(left_img, (pad, title_h))
    canvas.paste(right_img, (left_img.width + pad * 2, title_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")
    return out_path


def _gif_to_mp4(gif_path: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(gif_path),
        "-movflags",
        "faststart",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_path


def _media_parts_for_prompt(media_type: str, left_path: Path, right_path: Path):
    from google.genai import types

    if media_type == "mp4":
        return [
            types.Part.from_bytes(data=left_path.read_bytes(), mime_type="video/mp4"),
            types.Part.from_bytes(data=right_path.read_bytes(), mime_type="video/mp4"),
        ]
    return [
        types.Part.from_bytes(data=left_path.read_bytes(), mime_type="image/png"),
    ]


def _judge_prompt(cue: str, media_type: str) -> str:
    media_desc = {
        "mp4": "You will receive two separate videos. The first video is Motion A. The second video is Motion B.",
        "gif_strip": "You will receive one comparison image. The top row is Motion A and the bottom row is Motion B. Each row shows multiple sampled frames stitched left-to-right over time.",
        "trajectory": "You will receive one comparison image. The left panel is Motion A and the right panel is Motion B. Each panel shows the last frame with a yellow-to-purple trajectory overlay, where yellow is earlier and purple is later.",
    }[media_type]
    return (
        f"You are evaluating robot arm motions for the cue '{cue}'. "
        f"{media_desc} "
        "Choose which motion better communicates the target cue. "
        "Judge cue identifiability, not smoothness or visual appeal. "
        "Reply in JSON with keys winner, confidence, and reason. "
        "winner must be one of: 'A', 'B', 'tie'. "
        "confidence must be a number from 0 to 1."
    )


def compare(
    dataset: str,
    cue_idx: int,
    media_type: str = "gif_strip",
    model_name: str = "gemini-2.5-flash",
    robot: str = "IIWA",
    num_frames: int = 6,
    hz: int = 8,
    media_resolution: str = "high",
    strip_sampler: str = "single_cycle",
) -> str:
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"dataset must be one of {sorted(DATASET_CONFIGS)}")
    if robot != "IIWA":
        raise ValueError("This script currently supports only robot='IIWA'")
    if media_type not in {"mp4", "gif_strip", "trajectory"}:
        raise ValueError("media_type must be one of: mp4, gif_strip, trajectory")

    from google import genai
    from google.genai import types

    _ensure_google_api_key()
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    spec = DATASET_CONFIGS[dataset]
    sophisticated_row = _load_row(spec["sophisticated"], cue_idx)
    no_reasoning_row = _load_row(spec["no_reasoning"], cue_idx)
    cue = sophisticated_row["cue"]

    sophisticated_gif = _latest_single_gif(spec["sophisticated_gif_dir"], cue)
    no_reasoning_gif = _latest_single_gif(spec["no_reasoning_gif_dir"], cue)
    if sophisticated_gif is None or no_reasoning_gif is None:
        raise FileNotFoundError(
            f"Missing GIF for cue c{cue_idx} {cue}: "
            f"sophisticated={sophisticated_gif}, no_reasoning={no_reasoning_gif}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / f"{dataset}_c{cue_idx}_{_safe_name(cue)}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {
        "dataset": dataset,
        "cue_idx": int(cue_idx),
        "cue": cue,
        "media_type": media_type,
        "model_name": model_name,
        "robot": robot,
        "sophisticated_gif": str(sophisticated_gif),
        "no_reasoning_gif": str(no_reasoning_gif),
    }

    prompt = _judge_prompt(cue, media_type)
    contents: list[Any] = [prompt]

    if media_type == "mp4":
        left_mp4 = _gif_to_mp4(sophisticated_gif, out_dir / "A_sophisticated.mp4")
        right_mp4 = _gif_to_mp4(no_reasoning_gif, out_dir / "B_no_reasoning.mp4")
        artifacts["media_a"] = str(left_mp4)
        artifacts["media_b"] = str(right_mp4)
        contents.extend(_media_parts_for_prompt(media_type, left_mp4, right_mp4))
    elif media_type == "gif_strip":
        compare_png = _make_frame_strip_compare(
            cue,
            sophisticated_gif,
            no_reasoning_gif,
            out_dir / "compare_gif_strip.png",
            num_frames=num_frames,
            sampler=strip_sampler,
            left_row=sophisticated_row,
            right_row=no_reasoning_row,
            hz=hz,
        )
        artifacts["media_compare"] = str(compare_png)
        artifacts["strip_sampler"] = strip_sampler
        contents.extend(_media_parts_for_prompt(media_type, compare_png, compare_png))
    else:
        compare_png = _make_trajectory_compare(
            cue=cue,
            cue_idx=cue_idx,
            left_row=sophisticated_row,
            right_row=no_reasoning_row,
            left_config=spec["sophisticated"],
            right_config=spec["no_reasoning"],
            out_path=out_dir / "compare_trajectory.png",
            hz=hz,
        )
        artifacts["media_compare"] = str(compare_png)
        contents.extend(_media_parts_for_prompt(media_type, compare_png, compare_png))

    resolution_map = {
        "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
        "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
        "unspecified": types.MediaResolution.MEDIA_RESOLUTION_UNSPECIFIED,
    }
    config = types.GenerateContentConfig(media_resolution=resolution_map[media_resolution])
    response = client.models.generate_content(model=model_name, contents=contents, config=config)
    raw_text = response.text.strip()
    parsed = _parse_json_object(raw_text)

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts": artifacts,
        "prompt": prompt,
        "response_text": raw_text,
        "parsed": parsed,
    }
    out_json = out_dir / "result.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_json)
    return str(out_json)


if __name__ == "__main__":
    fire.Fire({"compare": compare})
