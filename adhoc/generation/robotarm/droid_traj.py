#!/usr/bin/env python3
"""DROID dump layout: q[0] is the first gesture pose, not sim-home."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

HOME = np.array(
    [0.0, np.pi / 16.0, 0.0, -np.pi / 2.0 - np.pi / 3.0, 0.0, np.pi - 0.2, np.pi / 4],
    dtype=np.float64,
)


def cue_start_index(q: np.ndarray, stored: Optional[int] = None) -> int:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or len(q) < 2:
        return 0
    if stored is not None:
        s = int(stored)
        if 0 < s < len(q):
            return s
    home_err = float(np.max(np.abs(q[0, :7] - HOME[: q.shape[1]])))
    if home_err > np.deg2rad(20.0):
        return 0
    dq = np.max(np.abs(np.diff(q[:, :7], axis=0)), axis=1)
    still = np.deg2rad(0.4)
    moving = dq > still
    i = 0
    while i < len(moving) and not moving[i]:
        i += 1
    if i >= len(moving):
        return 0
    while i < len(moving) and moving[i]:
        i += 1
    return min(i, len(q) - 1)


def strip_to_cue(
    q: np.ndarray,
    gripper: Optional[np.ndarray] = None,
    ee: Optional[np.ndarray] = None,
    pose_frame: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    q = np.asarray(q, dtype=np.float64)
    i = cue_start_index(q, pose_frame)
    q = q[i:]
    if gripper is None:
        g = np.zeros((len(q),), dtype=np.float64)
    else:
        g = np.asarray(gripper, dtype=np.float64).reshape(-1)
        g = g[i:] if len(g) > i else np.zeros((len(q),), dtype=np.float64)
        if len(g) < len(q):
            g = np.pad(g, (0, len(q) - len(g)))
        else:
            g = g[: len(q)]
    if ee is None:
        e = np.zeros((len(q), 3), dtype=np.float64)
    else:
        e = np.asarray(ee, dtype=np.float64)
        e = e[i:] if len(e) > i else np.zeros((len(q), 3), dtype=np.float64)
        if len(e) < len(q):
            pad = np.zeros((len(q) - len(e), e.shape[1] if e.ndim == 2 else 3), dtype=np.float64)
            e = np.concatenate([e, pad], axis=0) if e.size else np.zeros((len(q), 3))
        else:
            e = e[: len(q)]
    return q, g, e, 0


def rewrite_dump(path: Path) -> int:
    path = Path(path)
    z = np.load(path, allow_pickle=True)
    q = np.asarray(z["joints_rad"], dtype=np.float64)
    g = np.asarray(z["gripper"], dtype=np.float64) if "gripper" in z.files else None
    ee = np.asarray(z["ee_xyz"], dtype=np.float64) if "ee_xyz" in z.files else None
    stored = None
    if "pose_frame" in z.files:
        stored = int(np.asarray(z["pose_frame"]).reshape(-1)[0])
    i = cue_start_index(q, stored)
    q2, g2, e2, pose0 = strip_to_cue(q, g, ee, stored)
    extra = {k: z[k] for k in z.files if k not in {"joints_rad", "gripper", "ee_xyz", "pose_frame"}}
    np.savez(
        path,
        joints_rad=q2,
        gripper=g2,
        ee_xyz=e2,
        pose_frame=np.array(pose0, dtype=np.int32),
        **extra,
    )
    json_path = path.with_suffix(".json")
    if json_path.is_file():
        import json

        data = json.loads(json_path.read_text())
        data["joints_rad"] = q2.tolist()
        data["gripper"] = g2.tolist()
        data["ee_xyz"] = e2.tolist()
        data["pose_frame"] = pose0
        data["n_frames"] = int(q2.shape[0])
        hz = float(data.get("hz") or np.asarray(z["hz"]).reshape(-1)[0])
        data["duration_s"] = float(q2.shape[0] / max(hz, 1.0))
        json_path.write_text(json.dumps(data, indent=2))
    return i
