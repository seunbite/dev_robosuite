"""Per-part joint / xyzrpy schemas for full-body few-shot baselines.

Keeps major body regions explicit so prompts, generation, and render all share
the same part inventory (torso / head / arms / …) instead of one opaque vector.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Controllable gesture DOFs only (legs / grippers stay at home in render).
PART_JOINTS: dict[str, dict[str, list[str]]] = {
    "humanoid": {
        "torso": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        "head": ["head_pan_joint", "head_tilt_joint"],
        "right_arm": [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        "left_arm": [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ],
    },
    "tiago": {
        "base": ["base_forward", "base_side", "base_yaw"],
        "torso": ["torso_lift"],
        "head": ["head_pan", "head_tilt"],
        "right_arm": [f"right_arm_{i}" for i in range(1, 7)],
        "left_arm": [f"left_arm_{i}" for i in range(1, 7)],
    },
}

PART_LINKS: dict[str, list[str]] = {
    "humanoid": ["root", "torso", "head", "right_hand", "left_hand"],
    "tiago": ["base", "torso", "head", "right_hand", "left_hand"],
    "single_arm": ["end_effector"],
}

# Canonical still / home rows used in few-shot prompts (radians).
# Tiago: identical L/R joint rows → mirrored hang in cartesian space.
REST_JOINTS: dict[str, dict[str, list[float]]] = {
    "tiago": {
        "base": [0.0, 0.0, 0.0],
        "torso": [0.18],
        "head": [0.0, 0.0],
        "right_arm": [0.0, 1.57, 0.0, 0.3, 0.0, 0.0],
        "left_arm": [0.0, 1.57, 0.0, 0.3, 0.0, 0.0],
    },
}


def part_order(plat: str) -> list[str]:
    return list(PART_JOINTS.get(plat, {}))


def link_order(plat: str) -> list[str]:
    return list(PART_LINKS.get(plat, ["end_effector"]))


def flat_joint_names(plat: str) -> list[str]:
    names: list[str] = []
    for part in part_order(plat):
        names.extend(PART_JOINTS[plat][part])
    return names


def _span(rows: list[list[float]]) -> float:
    if not rows:
        return 0.0
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return 0.0
    return float(np.max(arr.max(0) - arr.min(0)))


def motion_summary_joints(parts: dict[str, Any]) -> str:
    bits = []
    for part, block in parts.items():
        vals = block.get("joints_rad") or []
        span = _span(vals)
        bits.append(f"{part}: {'still' if span < 1e-4 else f'span={span:.3f}rad'}")
    return "; ".join(bits)


def motion_summary_links(links: dict[str, Any]) -> str:
    bits = []
    for link, block in links.items():
        vals = block.get("xyzrpy") or []
        if not vals:
            bits.append(f"{link}: still")
            continue
        arr = np.asarray(vals, dtype=float)
        xyz = float(np.max(arr[:, :3].max(0) - arr[:, :3].min(0))) if arr.size else 0.0
        rpy = float(np.max(arr[:, 3:].max(0) - arr[:, 3:].min(0))) if arr.shape[1] >= 6 else 0.0
        if xyz < 1e-4 and rpy < 1e-4:
            bits.append(f"{link}: still")
        else:
            bits.append(f"{link}: xyz_span={xyz:.3f}, rpy_span={rpy:.3f}")
    return "; ".join(bits)


def structure_joint_shot(plat: str, joint_names: list[str], joints_rad: list[list[float]], *, hz: int) -> dict[str, Any]:
    """Slice a flat joint trajectory into named parts (missing DOFs → zeros)."""
    name_to_i = {n: i for i, n in enumerate(joint_names)}
    n = len(joints_rad)
    parts: dict[str, Any] = {}
    for part, jnames in PART_JOINTS.get(plat, {}).items():
        cols = []
        for jn in jnames:
            if jn in name_to_i:
                cols.append([float(row[name_to_i[jn]]) for row in joints_rad])
            else:
                cols.append([0.0] * n)
        values = [list(frame) for frame in zip(*cols)] if cols and n else []
        parts[part] = {"joint_names": list(jnames), "joints_rad": values}
    return {"hz": hz, "parts": parts, "motion_summary": motion_summary_joints(parts)}


def structure_xyz_shot(plat: str, link_names: list[str], xyzrpy: list[list[float]], *, hz: int) -> dict[str, Any]:
    """Slice a flat concatenated xyzrpy trajectory into per-link blocks."""
    links: dict[str, Any] = {}
    order = link_order(plat)
    index = {n: i for i, n in enumerate(link_names)}
    for link in order:
        if link in index:
            i = index[link]
            values = [list(row[i * 6 : (i + 1) * 6]) for row in xyzrpy]
        else:
            values = [[0.0] * 6 for _ in xyzrpy]
        links[link] = {"xyzrpy": values}
    return {"hz": hz, "links": links, "link_names": order, "motion_summary": motion_summary_links(links)}


def _as_numeric_rows(rows: Any, *, where: str, width: int | None = None) -> list[list[float]]:
    """Accept real numeric rows only; reject schema placeholders like '[[...], ...]'."""
    if rows is None:
        return []
    if isinstance(rows, str):
        raise ValueError(f"{where}: joints/xyzrpy is a string placeholder, not a numeric array")
    if not isinstance(rows, list):
        raise ValueError(f"{where}: expected list of frames, got {type(rows).__name__}")
    out: list[list[float]] = []
    for i, row in enumerate(rows):
        if isinstance(row, str):
            raise ValueError(f"{where}[{i}]: frame is a string, not a numeric row")
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"{where}[{i}]: expected list, got {type(row).__name__}")
        if width is not None and len(row) != width:
            raise ValueError(f"{where}[{i}]: len={len(row)} expected {width}")
        try:
            out.append([float(x) for x in row])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}[{i}]: non-numeric value ({exc})") from exc
    return out


def _expand_constant_rows(rows: list[list[float]], block: dict[str, Any], n_hint: int) -> list[list[float]]:
    """Expand constant=true / n_frames shorthand to a full-length series."""
    if not rows:
        return rows
    n_frames = int(block.get("n_frames") or 0) or n_hint
    if block.get("constant") and len(rows) == 1 and n_frames > 1:
        return [list(rows[0]) for _ in range(n_frames)]
    if len(rows) == 1 and n_frames > 1 and n_hint > 1:
        # single row while other parts already imply a longer horizon
        return [list(rows[0]) for _ in range(n_hint)]
    return rows


def flatten_joint_parts(parts: dict[str, Any], plat: str) -> tuple[list[str], list[list[float]]]:
    """Merge part blocks back into a single joint trajectory (canonical order)."""
    names = flat_joint_names(plat)
    if not parts:
        return names, []
    parsed: dict[str, tuple[list[str], list[list[float]], dict[str, Any]]] = {}
    n_frames = 0
    for part, jnames in PART_JOINTS.get(plat, {}).items():
        block = parts.get(part) or {}
        local_names = list(block.get("joint_names") or jnames)
        rows = _as_numeric_rows(
            block.get("joints_rad"),
            where=f"parts.{part}.joints_rad",
            width=len(local_names),
        )
        n_frames = max(n_frames, len(rows), int(block.get("n_frames") or 0))
        parsed[part] = (local_names, rows, block)
    if n_frames == 0:
        return names, []
    series = {jn: [0.0] * n_frames for jn in names}
    for part, jnames in PART_JOINTS.get(plat, {}).items():
        local_names, rows, block = parsed[part]
        rows = _expand_constant_rows(rows, block, n_frames)
        for fi in range(n_frames):
            row = rows[fi] if fi < len(rows) else ([0.0] * len(local_names))
            for j, jn in enumerate(local_names):
                if jn in series and j < len(row):
                    series[jn][fi] = float(row[j])
    values = [[series[jn][fi] for jn in names] for fi in range(n_frames)]
    return names, values


def flatten_link_parts(links: dict[str, Any], plat: str) -> tuple[list[str], list[list[float]]]:
    order = link_order(plat)
    parsed: dict[str, tuple[list[list[float]], dict[str, Any]]] = {}
    n = 0
    for link in order:
        block = links.get(link) or {}
        frames = _as_numeric_rows(block.get("xyzrpy"), where=f"links.{link}.xyzrpy")
        for i, pose in enumerate(frames):
            if len(pose) not in (3, 6):
                raise ValueError(f"links.{link}.xyzrpy[{i}]: len={len(pose)} expected 3 or 6")
        n = max(n, len(frames), int(block.get("n_frames") or 0))
        parsed[link] = (frames, block)
    if n == 0:
        return order, []
    values: list[list[float]] = []
    for fi in range(n):
        row: list[float] = []
        for link in order:
            frames, block = parsed.get(link) or ([], {})
            frames = _expand_constant_rows(frames, block, n)
            pose = list(frames[fi]) if fi < len(frames) else [0.0] * 6
            if len(pose) == 3:
                pose = pose + [0.0, 0.0, 0.0]
            row.extend(float(x) for x in pose)
        values.append(row)
    return order, values


def compact_parts_for_prompt(parts: dict[str, Any], *, still_eps: float = 1e-4) -> dict[str, Any]:
    """Shrink still parts to 1 frame + n_frames so few-shot prompts stay small."""
    out: dict[str, Any] = {}
    for part, block in (parts or {}).items():
        names = list(block.get("joint_names") or [])
        rows = list(block.get("joints_rad") or [])
        n = len(rows)
        span = _span(rows) if rows and not isinstance(rows, str) else 0.0
        if n > 1 and span < still_eps:
            out[part] = {
                "joint_names": names,
                "n_frames": n,
                "constant": True,
                "joints_rad": [rows[0]],
            }
        else:
            out[part] = {"joint_names": names, "joints_rad": rows}
    return out


def compact_links_for_prompt(links: dict[str, Any], *, still_eps: float = 1e-4) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for link, block in (links or {}).items():
        rows = list(block.get("xyzrpy") or [])
        n = len(rows)
        if n > 1:
            arr = np.asarray(rows, dtype=float)
            span = float(np.max(arr.max(0) - arr.min(0))) if arr.size else 0.0
        else:
            span = 0.0
        if n > 1 and span < still_eps:
            out[link] = {"n_frames": n, "constant": True, "xyzrpy": [rows[0]]}
        else:
            out[link] = {"xyzrpy": rows}
    return out
