#!/usr/bin/env python3
"""Robotiq clocking vs sim, measured on the real FR3.

Franka cartesian pose is the flange, not the jaw opening. Measure by adding
``q7`` until the jaws look horizontal, then::

    python check_droid_env.py --try-q7 45
    python check_droid_env.py --write-mount 45 --check-only

``q7_to_horizontal_deg`` is that added wrist angle. The sim gripper body is
rotated by the opposite amount so the same joints match the real jaws.
Existing dumps then get ``+q7_to_horizontal_deg`` on joint 7 so grip=horiz/vert
still looks horiz/vert.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SMALL10_JSON = _HERE / "gripper_mount.json"
_ROBOSUITE_JSON = (
    Path(__file__).resolve().parents[4]
    / "robosuite"
    / "models"
    / "robots"
    / "manipulators"
    / "fr3_gripper_mount.json"
)
_Q7_LO = np.deg2rad(-166.0)
_Q7_HI = np.deg2rad(166.0)


def load_mount(path: Path | None = None) -> dict:
    path = Path(path) if path is not None else _SMALL10_JSON
    if not path.is_file():
        if _ROBOSUITE_JSON.is_file():
            path = _ROBOSUITE_JSON
        else:
            return {"q7_to_horizontal_deg": 0.0}
    data = json.loads(path.read_text())
    data["q7_to_horizontal_deg"] = float(data.get("q7_to_horizontal_deg") or 0.0)
    return data


def mount_z_deg(data: dict | None = None) -> float:
    """Extra sim gripper rotation about flange Z (opposite of the measured q7)."""
    if data is None:
        data = load_mount()
    return -float(data.get("q7_to_horizontal_deg") or 0.0)


def dump_q7_delta_rad(data: dict | None = None) -> float:
    if data is None:
        data = load_mount()
    return np.deg2rad(float(data.get("q7_to_horizontal_deg") or 0.0))


def write_mount(deg: float, *, path: Path | None = None, extra: dict | None = None) -> Path:
    path = Path(path) if path is not None else _SMALL10_JSON
    payload = {
        "q7_to_horizontal_deg": float(deg),
        "sim_gripper_z_deg": -float(deg),
        "dump_q7_delta_deg": float(deg),
        "note": (
            "q7_to_horizontal_deg = wrist angle added on the real FR3 until jaws "
            "looked horizontal. Sim Robotiq body is rotated by -that about flange Z. "
            "Dumps add +that to q7 so horiz/vert intent is preserved."
        ),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    # Laptop copy lives in ~/jellyho/droid/; only update robosuite when that JSON already exists.
    if _ROBOSUITE_JSON.is_file():
        _ROBOSUITE_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def offset_q7(q: np.ndarray, delta_rad: float) -> tuple[np.ndarray, dict]:
    """Add wrist offset. 2F-85 is 180°-symmetric, so wrap by π if q7 saturates."""
    out = np.asarray(q, dtype=np.float64).copy()
    if out.ndim == 1:
        out = out.reshape(1, -1)
    raw = out[:, 6] + float(delta_rad)
    wrapped = raw.copy()
    wrapped = np.where(wrapped < _Q7_LO, wrapped + np.pi, wrapped)
    wrapped = np.where(wrapped > _Q7_HI, wrapped - np.pi, wrapped)
    clipped = np.clip(wrapped, _Q7_LO, _Q7_HI)
    n_clip = int(np.sum(np.abs(clipped - wrapped) > 1e-6))
    n_wrap = int(np.sum(np.abs(wrapped - raw) > 1e-6))
    out[:, 6] = clipped
    return out if q.ndim == 2 else out[0], {
        "n_clip": n_clip,
        "n_wrap": n_wrap,
        "q7_deg_min": float(np.degrees(clipped.min())),
        "q7_deg_max": float(np.degrees(clipped.max())),
    }
