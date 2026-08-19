#!/usr/bin/env python3
"""Vector-only Drag-and-Draw programs: handle + [x, y, z, frame]."""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageChops, ImageSequence, ImageStat

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for _p in (_REPO, _HERE, _HERE / "legacy", _REPO / "adhoc/generation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SCHEMA_PATH = _REPO / "data/seed/yml/drag_draw_embodiment.yaml"
SHOT_PATH = _REPO / "data/results/shots/embodiment_unified/shot_cues_yml_5.json"
SEED_OUT = _REPO / "data/seed/shots/drag_draw/fewshot_cues_yml_5.yaml"
SINGLE_ARM_TRACKS_PATH = _REPO / "data/seed/shots/drag_draw/single_arm_rendered_landmarks_v4.json"
PROMPT_DIR = _REPO / "data/seed/prompt/drag_draw"
OUT_DIR = _REPO / "data/results/drag_draw/vector_roundtrip"
GALLERY_PATH = _REPO / "data/results/drag_draw/fewshot_roundtrip/gallery_drag_draw_roundtrip.html"
PLATFORMS = ("humanoid", "tiago", "stretch", "single_arm", "roomba")

SOURCE_LINKS = {
    "humanoid": ["base", "torso", "head", "right_ee", "left_ee"],
    "tiago": ["base", "torso", "head", "right_ee", "left_ee"],
    "stretch": ["ee"],
    "single_arm": ["ee"],
    "roomba": ["base"],
}

# Optional virtual fingertip anchors. `local_axis` is the robot hand body's
# longitudinal finger axis; the authored point is normalized to `length` from
# the authoritative EE center before it becomes an IK direction hint.
FINGERTIP_SPECS = {
    "humanoid": {
        "right_ee": {"tip": "right_fingertip", "local_axis": [1.0, 0.0, 0.0], "length": 0.10, "palm": "right_palm_normal", "palm_local_axis": [0.0, 0.0, 1.0], "palm_length": 0.08},
        "left_ee": {"tip": "left_fingertip", "local_axis": [1.0, 0.0, 0.0], "length": 0.10, "palm": "left_palm_normal", "palm_local_axis": [0.0, 0.0, 1.0], "palm_length": 0.08},
    },
    "tiago": {
        "right_ee": {"tip": "right_fingertip", "local_axis": [0.0, 0.0, 1.0], "length": 0.10},
        "left_ee": {"tip": "left_fingertip", "local_axis": [0.0, 0.0, 1.0], "length": 0.10},
    },
    "stretch": {
        "ee": {"tip": "fingertip", "local_axis": [0.0, 0.0, 1.0], "length": 0.12},
    },
    "single_arm": {
        "ee": {"tip": "fingertip", "local_axis": [0.0, 0.0, 1.0], "length": 0.12},
    },
}

_SINGLE_ARM_RENDERED_TRACKS: dict[str, dict[str, list[list[float]]]] | None = None


def _single_arm_rendered_tracks(shot_id: str) -> dict[str, list[list[float]]]:
    """Run the exact reference motion-code executor and record rendered points."""
    global _SINGLE_ARM_RENDERED_TRACKS
    if _SINGLE_ARM_RENDERED_TRACKS is None:
        if SINGLE_ARM_TRACKS_PATH.is_file():
            cached = json.loads(SINGLE_ARM_TRACKS_PATH.read_text(encoding="utf-8"))
            if cached and all("upper_arm" in tracks and "forearm" in tracks for tracks in cached.values()):
                _SINGLE_ARM_RENDERED_TRACKS = cached
                return _SINGLE_ARM_RENDERED_TRACKS[shot_id]
        from motion_generation_core import MotionGenerator
        from motion_code_to_config import convert_code
        import tempfile

        source = json.loads(SHOT_PATH.read_text(encoding="utf-8"))["single_arm"]
        os.environ.setdefault("MANIPULATOR_POSE_XYZ_MODE", "trim")
        gen = MotionGenerator(
            robot_name="IIWA", capture_image_width=64, capture_image_height=64,
            has_renderer=False, has_offscreen_renderer=True, hz=12,
            ignore_xyz=False,
        )
        model = gen.env.sim.model._model
        data = gen.env.sim.data._data
        site_id = gen.jacobian_calculator.ik_solver.site_ids[0]
        names = {model.body(i).name: i for i in range(model.nbody)}
        elbow_id = next(names[n] for n in ("robot0_link_4", "link_4") if n in names)
        wrist_id = next(names[n] for n in ("robot0_link_6", "link_6") if n in names)
        upper_arm_id = next(names[n] for n in ("robot0_link_3", "link_3") if n in names)
        forearm_id = next(names[n] for n in ("robot0_link_5", "link_5") if n in names)
        cache: dict[str, dict[str, list[list[float]]]] = {}
        try:
            for source_row in source:
                gen._set_joint_positions(gen.initial_joint_pos)
                tracks = {"ee": [], "upper_arm": [], "elbow": [], "forearm": [], "wrist": [], "fingertip": [], "palm_normal": []}
                converted = convert_code(source_row["motion_code"], platform="single_arm")
                config = [{
                    "idx": 1,
                    "cue": str(source_row["shot_id"])[:40],
                    "description": source_row["cue"],
                    "movements": converted.get("movements") or [],
                    "source_code": source_row["motion_code"],
                    "parse_ok": True,
                    "platform": "single_arm",
                }]
                original_capture = gen._capture_image

                def capture_and_record():
                    center = np.asarray(data.site_xpos[site_id], dtype=float)
                    elbow = np.asarray(data.xpos[elbow_id], dtype=float)
                    wrist = np.asarray(data.xpos[wrist_id], dtype=float)
                    upper_arm = np.asarray(data.xpos[upper_arm_id], dtype=float)
                    forearm = np.asarray(data.xpos[forearm_id], dtype=float)
                    rotation = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3)
                    tip = center + 0.12 * (rotation @ np.asarray([0.0, 0.0, 1.0]))
                    palm = center + 0.08 * (rotation @ np.asarray([1.0, 0.0, 0.0]))
                    for handle, point in (("ee", center), ("upper_arm", upper_arm), ("elbow", elbow), ("forearm", forearm), ("wrist", wrist), ("fingertip", tip), ("palm_normal", palm)):
                        tracks[handle].append([float(v) for v in point])
                    return original_capture()

                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                    json.dump(config, tmp)
                    tmp_path = tmp.name
                gen._capture_image = capture_and_record
                try:
                    gen.execute_cue(
                        cue=config[0]["cue"], cue_idx=1, pose_index=None,
                        config_path=tmp_path, hz=12, save_gif=False,
                        overlay_progress_bar=False,
                    )
                finally:
                    gen._capture_image = original_capture
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                cache[str(source_row["shot_id"])] = tracks
        finally:
            try:
                gen.env.close()
            except Exception:
                pass
        _SINGLE_ARM_RENDERED_TRACKS = cache
        SINGLE_ARM_TRACKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SINGLE_ARM_TRACKS_PATH.write_text(
            json.dumps(cache, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return _SINGLE_ARM_RENDERED_TRACKS[shot_id]


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX yaw-pitch-roll rotation matrix, matching the renderer convention."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


def load_schema(path: Path = SCHEMA_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _neutral(schema: dict[str, Any], platform: str) -> dict[str, list[float]]:
    return {k: [float(x) for x in v] for k, v in schema[platform]["neutral_handles"].items()}


def validate_program(program: dict[str, Any], schema: dict[str, Any] | None = None) -> None:
    schema = schema or load_schema()
    platform = str(program.get("platform", ""))
    if platform not in schema or platform == "meta":
        raise ValueError(f"unknown platform: {platform!r}")
    if str(program.get("version")) != "2.0":
        raise ValueError("Drag-and-Draw vector version must be '2.0'")
    if int(program.get("fps", 0)) <= 0:
        raise ValueError("fps must be positive")
    known = _neutral(schema, platform)
    seen: set[str] = set()
    for track in program.get("trajectories") or []:
        handle = str(track.get("handle", ""))
        if handle not in known:
            raise ValueError(f"{platform}: unavailable handle {handle!r}")
        if handle in seen:
            raise ValueError(f"duplicate trajectory for {handle}")
        seen.add(handle)
        vectors = track.get("vectors") or []
        if not vectors:
            raise ValueError(f"{handle}: empty vectors")
        frames = []
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != 4:
                raise ValueError(f"{handle}: every vector must be [x, y, z, frame]")
            x, y, z, frame = vector
            if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in (x, y, z, frame)):
                raise ValueError(f"{handle}: vector values must be finite numbers")
            if int(frame) != frame or frame < 0:
                raise ValueError(f"{handle}: frame must be a non-negative integer")
            frames.append(int(frame))
        if frames[0] != 0:
            raise ValueError(f"{handle}: first vector must be at frame 0")
        if any(b <= a for a, b in zip(frames, frames[1:])):
            raise ValueError(f"{handle}: frames must strictly increase")
        xyz0 = np.asarray(vectors[0][:3], dtype=float)
        if np.linalg.norm(xyz0 - np.asarray(known[handle])) > 1e-3:
            raise ValueError(f"{handle}: frame 0 must equal the declared neutral xyz")
    for ee, spec in FINGERTIP_SPECS.get(platform, {}).items():
        auxiliaries = {spec["tip"]}
        if spec.get("palm"):
            auxiliaries.add(spec["palm"])
        present = auxiliaries & seen
        if present and ee not in seen:
            raise ValueError(f"{sorted(present)[0]}: hand orientation requires matching {ee} trajectory")
        if present and present != auxiliaries:
            missing = sorted(auxiliaries - present)[0]
            raise ValueError(f"{ee}: full hand orientation also requires {missing} trajectory")
    arm_groups = {
        "humanoid": (
            ("right_elbow", "right_ee", "right_fingertip", "right_palm_normal"),
            ("left_elbow", "left_ee", "left_fingertip", "left_palm_normal"),
        ),
        "tiago": (
            ("right_elbow", "right_ee", "right_fingertip"),
            ("left_elbow", "left_ee", "left_fingertip"),
        ),
        "single_arm": (("elbow", "ee", "fingertip"),),
    }
    for group in arm_groups.get(platform, ()):
        present = set(group) & seen
        if present and not set(group).issubset(seen):
            missing = sorted(set(group) - seen)
            raise ValueError(f"{platform} uses the complete arm frame {group}; missing {missing}")
    if platform == "stretch" and bool({"ee", "fingertip"} & seen) and not {"ee", "fingertip"}.issubset(seen):
        missing = "fingertip" if "ee" in seen else "ee"
        raise ValueError(f"stretch uses the complete ee + fingertip pair; missing {missing}")
    if platform == "roomba" and bool({"base", "base_front"} & seen) and not {"base", "base_front"}.issubset(seen):
        missing = "base_front" if "base" in seen else "base"
        raise ValueError(f"roomba uses the complete base + base_front pair; missing {missing}")


def interpolate_program(program: dict[str, Any], schema: dict[str, Any] | None = None) -> list[dict[str, list[float]]]:
    schema = schema or load_schema()
    validate_program(program, schema)
    neutral = _neutral(schema, program["platform"])
    tracks = {t["handle"]: t["vectors"] for t in program.get("trajectories") or []}
    last_frame = max((int(v[-1][3]) for v in tracks.values()), default=0)
    frames: list[dict[str, list[float]]] = []
    for frame in range(last_frame + 1):
        state = {h: list(xyz) for h, xyz in neutral.items()}
        for handle, vectors in tracks.items():
            if frame >= vectors[-1][3]:
                state[handle] = [float(x) for x in vectors[-1][:3]]
                continue
            for a, b in zip(vectors, vectors[1:]):
                if a[3] <= frame <= b[3]:
                    alpha = (frame - a[3]) / max(1, b[3] - a[3])
                    state[handle] = [round(float(a[i] + alpha * (b[i] - a[i])), 6) for i in range(3)]
                    break
        frames.append(state)
    return frames


def _gif_duration_frames(path: Path, fps: int) -> int:
    with Image.open(path) as image:
        default = int(image.info.get("duration", round(1000 / fps)))
        duration_ms = sum(int(frame.info.get("duration", default)) for frame in ImageSequence.Iterator(image))
    return max(1, round(duration_ms * fps / 1000.0))


def _source_tracks(platform: str, row: dict[str, Any], schema: dict[str, Any]) -> dict[str, list[list[float]]]:
    if platform == "stretch":
        import mujoco
        from motion_code_to_config import convert_code
        from stretch_renderer import _STRETCH_SCENE, _apply, _stretch_states

        converted = convert_code(row["motion_code"], platform="stretch")
        model = mujoco.MjModel.from_xml_path(str(_STRETCH_SCENE))
        data = mujoco.MjData(model)
        states = _stretch_states({"movements": converted["movements"]}, model, data, fps=int(row.get("hz") or 8))
        bodies = {
            "base": "base_link", "torso": "link_lift", "head": "link_head_tilt",
            "elbow": "link_arm_l0", "wrist": "link_wrist_yaw", "ee": "link_SG3_gripper_body",
        }
        result = {handle: [] for handle in bodies}
        result["fingertip"] = []
        for state in states:
            _apply(model, data, x=state["x"], y=state["y"], yaw=state["yaw"], joints=state["joints"])
            for handle, body in bodies.items():
                result[handle].append([float(x) for x in data.xpos[model.body(body).id]])
            gripper_id = int(model.body("link_SG3_gripper_body").id)
            center = np.asarray(data.xpos[gripper_id], dtype=float)
            rotation = np.asarray(data.xmat[gripper_id], dtype=float).reshape(3, 3)
            tip = center + 0.12 * (rotation @ np.asarray([0.0, 0.0, 1.0]))
            result["fingertip"].append([float(v) for v in tip])
        return result
    if platform == "single_arm":
        return _single_arm_rendered_tracks(str(row["shot_id"]))
    rows = row.get("xyzrpy") or []
    labels = SOURCE_LINKS[platform]
    out = {label: [] for label in labels}
    for values in rows:
        for i, label in enumerate(labels):
            start = i * 6
            if len(values) >= start + 3:
                out[label].append([float(x) for x in values[start : start + 3]])
    for ee, spec in FINGERTIP_SPECS.get(platform, {}).items():
        if ee not in labels:
            continue
        i = labels.index(ee)
        tip_points: list[list[float]] = []
        local_axis = np.asarray(spec["local_axis"], dtype=float)
        for values in rows:
            start = i * 6
            if len(values) < start + 6:
                continue
            hand = np.asarray(values[start : start + 3], dtype=float)
            rotation = _rpy_matrix(*[float(v) for v in values[start + 3 : start + 6]])
            tip = hand + float(spec["length"]) * (rotation @ local_axis)
            tip_points.append([float(v) for v in tip])
        out[spec["tip"]] = tip_points
        if spec.get("palm"):
            palm_points: list[list[float]] = []
            palm_axis = np.asarray(spec["palm_local_axis"], dtype=float)
            for values in rows:
                start = i * 6
                if len(values) < start + 6:
                    continue
                hand = np.asarray(values[start : start + 3], dtype=float)
                rotation = _rpy_matrix(*[float(v) for v in values[start + 3 : start + 6]])
                palm = hand + float(spec["palm_length"]) * (rotation @ palm_axis)
                palm_points.append([float(v) for v in palm])
            out[spec["palm"]] = palm_points
    if platform == "roomba" and out["base"]:
        front = []
        for values in rows:
            yaw = float(values[5])
            x, y, z = (float(v) for v in values[:3])
            front.append([x + 0.15 * math.cos(yaw), y + 0.15 * math.sin(yaw), max(0.09, z)])
        out["base_front"] = front
        out["base"] = [[p[0], p[1], 0.09] for p in out["base"]]
    return out


def _compress_vectors(vectors: list[list[float]], *, eps: float = 1e-5) -> list[list[float]]:
    if len(vectors) <= 2:
        return vectors
    keep = [vectors[0]]
    for i in range(1, len(vectors) - 1):
        a, b, c = np.asarray(vectors[i - 1][:3]), np.asarray(vectors[i][:3]), np.asarray(vectors[i + 1][:3])
        dt1 = vectors[i][3] - vectors[i - 1][3]
        dt2 = vectors[i + 1][3] - vectors[i][3]
        v1, v2 = (b - a) / max(1, dt1), (c - b) / max(1, dt2)
        if np.linalg.norm(v2 - v1) > eps:
            keep.append(vectors[i])
    keep.append(vectors[-1])
    return keep


def row_to_vector_program(platform: str, row: dict[str, Any], schema: dict[str, Any] | None = None) -> dict[str, Any]:
    schema = schema or load_schema()
    neutral = _neutral(schema, platform)
    # The published IIWA reference is rendered by render_manipulator at 12 Hz;
    # its separately extracted payload says 8 Hz and must not drive replay time.
    fps = 12 if platform == "single_arm" else int(row.get("hz") or 8)
    if row.get("cue") == "bow_greeting" and platform in ("humanoid", "tiago"):
        if platform == "humanoid":
            head = neutral["head"]
            trajectories = [{
                "handle": "head",
                "vectors": [
                    [*head, 0], [*head, 4],
                    [0.0620, 0.0, 1.2182, 12],
                    [0.0620, 0.0, 1.2182, 16],
                    [*head, 24], [*head, 28],
                ],
            }]
        else:
            torso = neutral["torso"]
            low = [torso[0], torso[1], 0.3415]
            trajectories = [{
                "handle": "torso",
                "vectors": [
                    [*torso, 0], [*torso, 4], [*low, 10],
                    [*low, 14], [*torso, 20], [*torso, 24],
                ],
            }]
        program = {"version": "2.0", "cue": "bow_greeting", "platform": platform, "fps": fps, "trajectories": trajectories}
        validate_program(program, schema)
        return program
    source = _source_tracks(platform, row, schema)
    original_gif = _REPO / str(row.get("render_gif") or "")
    duration_frames = _gif_duration_frames(original_gif, fps) if original_gif.is_file() else max(1, len(next(iter(source.values()), [])))
    if platform == "single_arm":
        duration_frames = max(duration_frames, max((len(points) for points in source.values()), default=1))
    trajectories = []
    for handle, points in source.items():
        if platform == "stretch" and handle not in ("ee", "fingertip"):
            continue
        if handle not in neutral or not points:
            continue
        arr = np.asarray(points, dtype=float)
        displacement = float(np.max(np.linalg.norm(arr - np.asarray(neutral[handle]), axis=1)))
        span = float(np.max(np.linalg.norm(arr - arr[0], axis=1)))
        force_posture_handle = (
            (platform == "single_arm" and handle in ("elbow", "fingertip"))
            or (platform == "stretch" and handle == "fingertip")
        )
        if max(displacement, span) < 0.004 and not force_posture_handle:
            continue
        vectors: list[list[float]] = [[*neutral[handle], 0]]
        for i, point in enumerate(points):
            frame = 1 if len(points) == 1 else 1 + round(i * max(1, duration_frames - 1) / (len(points) - 1))
            vector = [round(float(x), 4) for x in point] + [int(frame)]
            if vector[3] == vectors[-1][3]:
                vectors[-1] = vector
            else:
                vectors.append(vector)
        vectors = _compress_vectors(vectors)
        trajectories.append({"handle": handle, "vectors": vectors})
    authored = {track["handle"] for track in trajectories}
    for ee, spec in FINGERTIP_SPECS.get(platform, {}).items():
        auxiliaries = [spec["tip"]] + ([spec["palm"]] if spec.get("palm") else [])
        orientation_present = any(auxiliary in authored for auxiliary in auxiliaries)
        if orientation_present and ee not in authored:
            trajectories.append({
                "handle": ee,
                "vectors": [[*neutral[ee], 0], [*neutral[ee], max(1, duration_frames)]],
            })
            authored.add(ee)
        # Existing pose data always has both source axes. If thresholding kept
        # only one, add the other as a neutral hold rather than emit an invalid
        # two-point frame.
        for auxiliary in auxiliaries:
            if orientation_present and auxiliary not in authored:
                trajectories.append({
                    "handle": auxiliary,
                    "vectors": [[*neutral[auxiliary], 0], [*neutral[auxiliary], max(1, duration_frames)]],
                })
                authored.add(auxiliary)
    required_arm_groups = {
        "humanoid": (
            ("right_elbow", "right_ee", "right_fingertip", "right_palm_normal"),
            ("left_elbow", "left_ee", "left_fingertip", "left_palm_normal"),
        ),
        "tiago": (
            ("right_elbow", "right_ee", "right_fingertip"),
            ("left_elbow", "left_ee", "left_fingertip"),
        ),
        "single_arm": (("elbow", "ee", "fingertip"),),
    }
    for group in required_arm_groups.get(platform, ()):
        if set(group) & authored:
            for handle in group:
                if handle not in authored:
                    trajectories.append({
                        "handle": handle,
                        "vectors": [[*neutral[handle], 0], [*neutral[handle], max(1, duration_frames)]],
                    })
                    authored.add(handle)
    if platform == "roomba" and {"base", "base_front"} & authored:
        for handle in ("base", "base_front"):
            if handle not in authored:
                trajectories.append({
                    "handle": handle,
                    "vectors": [[*neutral[handle], 0], [*neutral[handle], max(1, duration_frames)]],
                })
                authored.add(handle)
    if not trajectories:
        fallback = "base" if platform == "roomba" else ("right_ee" if platform in ("humanoid", "tiago") else "ee")
        trajectories = [{"handle": fallback, "vectors": [[*neutral[fallback], 0], [*neutral[fallback], max(1, duration_frames)]]}]
        if platform == "single_arm":
            trajectories.append({"handle": "elbow", "vectors": [[*neutral["elbow"], 0], [*neutral["elbow"], max(1, duration_frames)]]})
            trajectories.append({"handle": "fingertip", "vectors": [[*neutral["fingertip"], 0], [*neutral["fingertip"], max(1, duration_frames)]]})
        if platform == "stretch":
            trajectories.append({"handle": "fingertip", "vectors": [[*neutral["fingertip"], 0], [*neutral["fingertip"], max(1, duration_frames)]]})
    program = {"version": "2.0", "cue": row.get("cue", ""), "platform": platform, "fps": fps, "trajectories": trajectories}
    validate_program(program, schema)
    return program


def build_prompt(platform: str, schema: dict[str, Any] | None = None) -> str:
    schema = schema or load_schema()
    robot = schema[platform]
    lines = [
        "You generate a vector-only Drag-and-Draw robot gesture.",
        f"The current robot is {robot['robot']} and is exactly in NEUTRAL at frame 0.",
        "All coordinates are absolute robot-local meters: +x forward toward the viewer, +y robot-left, +z up.",
        "",
        "Neutral movable handles (x, y, z):",
    ]
    for handle, xyz in robot["neutral_handles"].items():
        lines.append(f"- {handle}: ({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f})")
    bounds = robot["workspace_bounds"]
    lines += [
        "",
        f"Workspace bounds: x={bounds['x']}, y={bounds['y']}, z={bounds['z']} meters.",
        f"Playback rate: {robot['fps']} fps.",
        "",
        "OUTPUT CONTRACT:",
        "- Output YAML only: version, cue, platform, fps, trajectories.",
        "- Each trajectory has exactly `handle` and `vectors`.",
        "- Every vector is exactly [x, y, z, frame]. No shape names, axes, angles, speed, repetitions, duration, or hold_time.",
        "- Every used handle starts at its neutral xyz at frame 0.",
        "- Sample curves with enough intermediate vectors that their geometry is explicit.",
        "- Encode faster motion with fewer frames and slower motion with more frames.",
        "- Encode a hold by repeating the final xyz at a later frame.",
        "- Concurrent handles share frame numbers.",
        "",
        "Robot-specific notes:",
    ]
    lines += [f"- {note}" for note in robot.get("notes") or []]
    if platform in ("humanoid", "tiago"):
        lines += [
            "",
            "COMPLETE CARTESIAN ARM FRAME:",
            "- A right_ee or left_ee trajectory is the authoritative hand-center path.",
        ]
        if platform == "humanoid":
            lines += [
                "- For every used arm, always output matching elbow + EE + fingertip + palm_normal trajectories (4 points).",
                "- Elbow explicitly determines the arm posture; do not leave it for the renderer to choose.",
                "- EE→fingertip is the finger axis; EE→palm_normal is the outward palm axis.",
                "- Auxiliary keyframes may differ; the decoder normalizes and orthogonalizes both axes.",
                "- If constraints conflict, preserve EE first and approximate elbow and the two hand axes.",
            ]
        else:
            lines += [
                "- For every used arm, always output matching elbow + EE + fingertip trajectories (3 points).",
                "- Elbow explicitly determines the arm posture; do not leave it for the renderer to choose.",
                "- EE→fingertip is the Tiago gripper/finger axis; no palm-normal point is used.",
                "- If constraints conflict, preserve EE first and approximate elbow and gripper direction.",
            ]
    if platform == "single_arm":
        lines += [
            "",
            "THREE-POINT CARTESIAN ARM FRAME:",
            "- Always output elbow, ee, and fingertip trajectories together.",
            "- EE fixes the hand center, elbow resolves the arm posture, and EE→fingertip defines gripper direction.",
            "- All tracks may use different keyframes and are interpolated independently.",
            "- The decoder preserves EE first, then approximates elbow and gripper direction when constraints conflict.",
        ]
    if platform == "stretch":
        lines += [
            "",
            "TWO-POINT GRIPPER DIRECTION:",
            "- Always output both ee and fingertip trajectories.",
            "- EE is the authoritative gripper-center target; EE→fingertip expresses gripper direction.",
            "- If they conflict kinematically, preserve EE position and approximate fingertip direction.",
        ]
    if platform == "humanoid":
        lines += [
            "",
            "Example — blow_kiss: move right_ee from neutral to the mouth, then trace a small forward/upward arc that opens away from the face, then hold:",
            "version: '2.0'",
            "cue: blow_kiss",
            "platform: humanoid",
            "fps: 20",
            "trajectories:",
            "  - handle: right_ee",
            "    vectors:",
            "      - [0.0012, -0.2140, 0.7432, 0]",
            "      - [0.0400, -0.1500, 0.9000, 5]",
            "      - [0.0200, -0.0700, 1.1700, 12]  # at mouth",
            "      - [0.1000, -0.0550, 1.2100, 15]",
            "      - [0.2200, -0.0350, 1.2300, 18]",
            "      - [0.3600, -0.0150, 1.1900, 22]  # arc opens forward",
            "      - [0.3600, -0.0150, 1.1900, 30]  # hold",
            "  - handle: right_elbow",
            "    vectors:",
            "      - [-0.0528, -0.1778, 0.9156, 0]",
            "      - [0.0000, -0.2400, 1.0200, 12]",
            "      - [0.1800, -0.2200, 1.0800, 22]",
            "      - [0.1800, -0.2200, 1.0800, 30]",
            "  - handle: right_fingertip",
            "    vectors:",
            "      - [0.0357, -0.2325, 0.6512, 0]",
            "      - [0.0400, -0.0700, 1.2600, 12]  # fingers mostly upward",
            "      - [0.2400, -0.0350, 1.3250, 18]",
            "      - [0.3900, -0.0150, 1.2850, 22]",
            "      - [0.3900, -0.0150, 1.2850, 30]",
            "  - handle: right_palm_normal",
            "    vectors:",
            "      - [0.0762, -0.2114, 0.7708, 0]",
            "      - [0.1000, -0.0700, 1.1700, 12]  # palm faces forward from the mouth",
            "      - [0.3000, -0.0350, 1.2300, 18]",
            "      - [0.4400, -0.0150, 1.1900, 22]",
            "      - [0.4400, -0.0150, 1.1900, 30]",
        ]
    return "\n".join(lines) + "\n"


def write_prompts() -> list[Path]:
    schema = load_schema()
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for platform in PLATFORMS:
        path = PROMPT_DIR / f"{platform}.txt"
        path.write_text(build_prompt(platform, schema), encoding="utf-8")
        paths.append(path)
    return paths


def convert_fewshots() -> dict[str, list[dict[str, Any]]]:
    source = json.loads(SHOT_PATH.read_text(encoding="utf-8"))
    schema = load_schema()
    result = {p: [] for p in PLATFORMS}
    for platform in PLATFORMS:
        for row in source[platform]:
            result[platform].append({
                "cue": row["cue"],
                "shot_id": row["shot_id"],
                "source_motion_code": row["motion_code"],
                "drag_draw": row_to_vector_program(platform, row, schema),
                "original_render_gif": row["render_gif"],
            })
    SEED_OUT.parent.mkdir(parents=True, exist_ok=True)
    SEED_OUT.write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True, width=140), encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "fewshot_vectors.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def program_to_xyzrpy(program: dict[str, Any], schema: dict[str, Any] | None = None) -> dict[str, Any]:
    schema = schema or load_schema()
    platform = program["platform"]
    states = interpolate_program(program, schema)
    active_handles = {track["handle"] for track in program["trajectories"]}
    if platform == "humanoid":
        all_links = [("root", "base"), ("torso", "torso"), ("head", "head"), ("right_elbow", "right_elbow"), ("right_hand", "right_ee"), ("left_elbow", "left_elbow"), ("left_hand", "left_ee")]
        links = [item for item in all_links if item[1] in active_handles]
    elif platform == "tiago":
        all_links = [("base", "base"), ("torso", "torso"), ("head", "head"), ("right_elbow", "right_elbow"), ("right_hand", "right_ee"), ("left_elbow", "left_elbow"), ("left_hand", "left_ee")]
        links = [item for item in all_links if item[1] in active_handles]
    elif platform in ("single_arm", "stretch"):
        links = [("end_effector", "ee")]
    elif platform == "roomba":
        rows = []
        for state in states:
            base, front = np.asarray(state["base"]), np.asarray(state["base_front"])
            yaw = math.atan2(front[1] - base[1], front[0] - base[0])
            rows.append([float(base[0]), float(base[1]), float(base[2]), 0.0, 0.0, yaw])
        return {"platform": platform, "link_names": ["base"], "xyzrpy": rows, "hz": int(program["fps"])}
    else:
        raise ValueError(platform)
    rows = []
    for state in states:
        row = []
        for _link, handle in links:
            row.extend([*state[handle], 0.0, 0.0, 0.0])
        rows.append(row)
    payload: dict[str, Any] = {
        "platform": platform,
        "link_names": [x[0] for x in links],
        "xyzrpy": rows,
        "hz": int(program["fps"]),
        "position_only": True,
        "cue": program.get("cue", ""),
        # The three authored Single Arm handles are rendered as target spheres
        # for visual inspection of the LLM's trajectory specification.
        "show_llm_target_spheres": platform == "single_arm",
    }
    fingertip_directions: dict[str, list[list[float]]] = {}
    fingertip_targets: dict[str, list[list[float]]] = {}
    fingertip_local_axes: dict[str, list[float]] = {}
    palm_directions: dict[str, list[list[float]]] = {}
    palm_local_axes: dict[str, list[float]] = {}
    specs = FINGERTIP_SPECS.get(platform, {})
    neutral = _neutral(schema, platform)
    for link, ee in links:
        spec = specs.get(ee)
        if spec is None or spec["tip"] not in active_handles:
            continue
        neutral_finger = np.asarray(neutral[spec["tip"]]) - np.asarray(neutral[ee])
        fallback_finger = neutral_finger / max(np.linalg.norm(neutral_finger), 1e-9)
        has_palm = bool(spec.get("palm") and spec["palm"] in active_handles)
        if has_palm:
            neutral_palm = np.asarray(neutral[spec["palm"]]) - np.asarray(neutral[ee])
            neutral_palm -= fallback_finger * float(np.dot(neutral_palm, fallback_finger))
            fallback_palm = neutral_palm / max(np.linalg.norm(neutral_palm), 1e-9)
        finger_rows, palm_rows = [], []
        for state in states:
            center = np.asarray(state[ee], dtype=float)
            finger = np.asarray(state[spec["tip"]], dtype=float) - center
            if np.linalg.norm(finger) > 1e-4:
                fallback_finger = finger / np.linalg.norm(finger)
            finger_rows.append([float(v) for v in fallback_finger])
            if has_palm:
                palm = np.asarray(state[spec["palm"]], dtype=float) - center
                palm -= fallback_finger * float(np.dot(palm, fallback_finger))
                if np.linalg.norm(palm) > 1e-4:
                    fallback_palm = palm / np.linalg.norm(palm)
                else:
                    fallback_palm -= fallback_finger * float(np.dot(fallback_palm, fallback_finger))
                    fallback_palm /= max(np.linalg.norm(fallback_palm), 1e-9)
                palm_rows.append([float(v) for v in fallback_palm])
        fingertip_directions[link] = finger_rows
        fingertip_targets[link] = [
            [float(v) for v in state[spec["tip"]]]
            for state in states
        ]
        fingertip_local_axes[link] = [float(v) for v in spec["local_axis"]]
        if has_palm:
            palm_directions[link] = palm_rows
            palm_local_axes[link] = [float(v) for v in spec["palm_local_axis"]]
    if fingertip_directions:
        payload.update({
            "fingertip_directions": fingertip_directions,
            "fingertip_targets": fingertip_targets,
            "fingertip_local_axes": fingertip_local_axes,
            "palm_directions": palm_directions,
            "palm_local_axes": palm_local_axes,
            # Hand center and fingertip direction define the primary task. The
            # renderer uses this explicit elbow track only to select among IK
            # equilibria that satisfy those primary targets.
            "hand_position_weight": 10.0,
            "elbow_position_weight": 0.0,
            "elbow_selection_tolerance_m": 0.08,
            "hand_position_tolerance_m": 0.005,
            "fingertip_direction_tolerance": 0.08,
            "fingertip_direction_weight": 0.8,
            "palm_direction_weight": 0.8,
        })
    if platform == "single_arm" and "elbow" in active_handles:
        payload.update({
            "elbow_targets": [[float(v) for v in state["elbow"]] for state in states],
            # The hand center is authoritative; elbow and fingertip are soft.
            "ee_position_weight": 10.0,
            "elbow_position_weight": 0.8,
            "fingertip_direction_weight": 0.8,
            # Frame 0 is the declared neutral state required by the vector DSL.
            # Solve neutral → the first authored pose before recording the GIF,
            # so replay frame 0 corresponds to the source render's first frame.
            "render_start_frame": 1,
        })
    return payload


def _render_stretch_vectors(program: dict[str, Any], out_gif: Path) -> None:
    import mujoco
    from scipy.optimize import least_squares
    from embodiment_renderers import _progress_overlay, _save_media
    from stretch_renderer import _STRETCH_SCENE, _apply, _ik, _pose_from_dict

    states = interpolate_program(program)
    model = mujoco.MjModel.from_xml_path(str(_STRETCH_SCENE))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=560)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 145, -18, 2.5
    joints = _pose_from_dict({"dir": "still", "torso": "mid", "head": "center"})
    neutral = _neutral(load_schema(), "stretch")
    fingertip_active = any(track["handle"] == "fingertip" for track in program["trajectories"])
    gripper_id = int(model.body("link_SG3_gripper_body").id)
    fallback_direction = np.asarray(neutral["fingertip"]) - np.asarray(neutral["ee"])
    fallback_direction /= max(np.linalg.norm(fallback_direction), 1e-9)
    frames = []
    for i, state in enumerate(states):
        base = state["base"]
        joints["lift"] = float(np.clip(0.55 + state["torso"][2] - neutral["torso"][2], 0.0, 1.1))
        head_delta = np.asarray(state["head"]) - np.asarray(neutral["head"])
        joints["head_pan"] = float(np.clip(4.0 * head_delta[1], -1.57, 1.57))
        joints["head_tilt"] = float(np.clip(-4.0 * head_delta[2], -1.57, 1.57))
        if "ee" in state:
            joints = _ik(model, data, target=np.asarray(state["ee"], dtype=float), x=base[0], y=base[1], yaw=0.0, joints=joints)
        if fingertip_active:
            target_center = np.asarray(state["ee"], dtype=float)
            direction = np.asarray(state["fingertip"], dtype=float) - target_center
            if np.linalg.norm(direction) > 1e-4:
                fallback_direction = direction / np.linalg.norm(direction)
            keys = ("lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll")
            seed = np.asarray([joints[k] for k in keys], dtype=float)
            lo = np.asarray([0.0, 0.0, -math.pi / 2.0, -1.57, -3.14])
            hi = np.asarray([1.1, 0.52, 4.42, 0.56, 3.14])

            def residual(values):
                trial = dict(joints)
                trial.update({key: float(value) for key, value in zip(keys, values)})
                _apply(model, data, x=base[0], y=base[1], yaw=0.0, joints=trial)
                center = np.asarray(data.xpos[gripper_id], dtype=float)
                rotation = np.asarray(data.xmat[gripper_id], dtype=float).reshape(3, 3)
                current_direction = rotation @ np.asarray([0.0, 0.0, 1.0])
                return np.concatenate([
                    4.0 * (center - target_center),
                    0.15 * (current_direction - fallback_direction),
                    0.01 * (values - seed),
                ])

            solved = least_squares(residual, np.clip(seed, lo, hi), bounds=(lo, hi), max_nfev=60).x
            joints.update({key: float(value) for key, value in zip(keys, solved)})
        _apply(model, data, x=base[0], y=base[1], yaw=0.0, joints=joints)
        camera.lookat[:] = (base[0] + 0.12, base[1], 0.68)
        renderer.update_scene(data, camera)
        frames.append(_progress_overlay(renderer.render(), i, len(states)))
    renderer.close()
    _save_media(frames, out_gif, fps=int(program["fps"]))


def render_program(program: dict[str, Any], out_gif: Path) -> None:
    validate_program(program)
    platform = program["platform"]
    if platform == "stretch":
        out_gif.parent.mkdir(parents=True, exist_ok=True)
        _render_stretch_vectors(program, out_gif)
        return
    payload = program_to_xyzrpy(program)
    from run_embodiment_baseline_traj import RENDERERS
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    RENDERERS[platform](payload, "xyzrpy", out_gif, {})


def compare_gifs(original: Path, candidate: Path) -> dict[str, Any]:
    def frames(path: Path):
        with Image.open(path) as image:
            default = int(image.info.get("duration", 0))
            out, duration = [], 0
            for frame in ImageSequence.Iterator(image):
                duration += int(frame.info.get("duration", default))
                out.append(frame.convert("RGB"))
            return out, duration
    af, ad = frames(original)
    bf, bd = frames(candidate)
    n = min(len(af), len(bf))
    maes = []
    for i in range(n):
        j = 0 if n == 1 else round(i * (len(bf) - 1) / (n - 1))
        x, y = af[i], bf[j]
        if x.size != y.size:
            y = y.resize(x.size)
        maes.append(sum(ImageStat.Stat(ImageChops.difference(x, y)).mean) / 3.0)
    mae = float(np.mean(maes)) if maes else float("inf")
    return {
        "original_frames": len(af), "vector_frames": len(bf),
        "original_duration_ms": ad, "vector_duration_ms": bd,
        "normalized_mae_percent": round(100.0 * mae / 255.0, 4),
    }


def render_fewshots(
    data: dict[str, list[dict[str, Any]]],
    *,
    force: bool = False,
    force_platforms: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    for platform, rows in data.items():
        for row in rows:
            out = OUT_DIR / "render" / platform / f"{row['shot_id']}__vector.gif"
            force_this = force and (force_platforms is None or platform in force_platforms)
            if force_this or not out.is_file():
                render_program(row["drag_draw"], out)
            row["vector_render_gif"] = str(out.relative_to(_REPO))
            row["visual_comparison"] = compare_gifs(_REPO / row["original_render_gif"], out)
    (OUT_DIR / "fewshot_vectors_rendered.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def write_gallery(data: dict[str, list[dict[str, Any]]]) -> Path:
    cues = []
    for rows in data.values():
        for row in rows:
            if row["cue"] not in cues:
                cues.append(row["cue"])
    by = {p: {r["cue"]: r for r in rows} for p, rows in data.items()}
    sections = []
    for cue in cues:
        cells = []
        for platform in PLATFORMS:
            row = by[platform][cue]
            original = os.path.relpath(_REPO / row["original_render_gif"], GALLERY_PATH.parent)
            vector = os.path.relpath(_REPO / row["vector_render_gif"], GALLERY_PATH.parent)
            # Local-file browsers cache GIFs aggressively even after a rerender.
            # Tie the URL to the actual output revision so a corrected render is
            # fetched without requiring the user to clear the browser cache.
            vector_revision = (_REPO / row["vector_render_gif"]).stat().st_mtime_ns
            vector_src = f"{vector}?v={vector_revision}"
            text = yaml.safe_dump(row["drag_draw"], sort_keys=False, allow_unicode=True, width=100)
            metric = row["visual_comparison"]
            cells.append(f"<article><h3>{html.escape(platform)}</h3><div class='pair'><figure><img src='{html.escape(original)}'><figcaption>original</figcaption></figure><figure><img src='{html.escape(vector_src)}'><figcaption>vector replay</figcaption></figure></div><p>MAE {metric['normalized_mae_percent']}% · {metric['vector_duration_ms']} ms</p><details><summary>handle vectors</summary><pre>{html.escape(text)}</pre></details></article>")
        sections.append(f"<section><h2>{html.escape(cue)}</h2><div class='grid'>{''.join(cells)}</div></section>")
    page = """<!doctype html><html><head><meta charset='utf-8'><title>Vector Drag-and-Draw</title><style>body{font-family:system-ui;margin:24px;background:#0d1117;color:#e6edf3}.grid{display:grid;grid-template-columns:repeat(5,minmax(280px,1fr));gap:12px;overflow-x:auto}article{background:#161b22;border:1px solid #30363d;padding:10px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:5px}figure{margin:0}img{width:100%;background:#000}figcaption{text-align:center;color:#9da7b3;font-size:11px}pre{font-size:10px;max-height:500px;overflow:auto;white-space:pre-wrap}p{font-size:12px;color:#7ee787}</style></head><body><h1>Vector-only Drag-and-Draw</h1><p>Every motion is handle + [x, y, z, frame]. No shape/speed/hold attributes.</p>""" + "".join(sections) + "</body></html>"
    GALLERY_PATH.parent.mkdir(parents=True, exist_ok=True)
    GALLERY_PATH.write_text(page, encoding="utf-8")
    return GALLERY_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prompts", "convert", "render", "render-file", "all"), nargs="?", default="all")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--platforms", help="comma-separated platforms to force-render")
    args = parser.parse_args()
    if args.command in ("prompts", "all"):
        for path in write_prompts(): print(f"[prompt] {path.relative_to(_REPO)}")
    data = None
    if args.command in ("convert", "render", "all"):
        data = convert_fewshots()
        print("[convert] 25 vector programs")
    if args.command in ("render", "all"):
        force_platforms = set(args.platforms.split(",")) if args.platforms else None
        unknown = (force_platforms or set()) - set(PLATFORMS)
        if unknown:
            parser.error(f"unknown platforms: {','.join(sorted(unknown))}")
        data = render_fewshots(data, force=args.force, force_platforms=force_platforms)
        print(f"[gallery] {write_gallery(data).relative_to(_REPO)}")
    if args.command == "render-file":
        if args.input is None or args.output is None: parser.error("render-file requires --input and --output")
        raw = args.input.read_text(encoding="utf-8")
        program = json.loads(raw) if args.input.suffix.lower() == ".json" else yaml.safe_load(raw)
        render_program(program, args.output)
        print(f"[render-file] {args.output}")


if __name__ == "__main__":
    main()
