#!/usr/bin/env python3
"""Pack every small10 human/LLM trajectory into one portable NPZ.

The bundle uses concatenated numeric arrays plus offsets, so loading it does not
require pickled object arrays. Run this on the machine that has the per-motion
NPZ files, then copy ``small10_all.npz``, ``replay_droid.py``, and
``droid_traj.py`` to the robot.

    python build_bundle.py
    python replay_droid.py --list
    python replay_droid.py --idx 2 --author LLM --play --slow 0.1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from droid_traj import strip_to_cue

ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX = ROOT / "index.json"
DEFAULT_OUTPUT = ROOT / "small10_all.npz"


def _local_npz(idx: int, cue: str, author: str) -> Path:
    stem = f"{idx:02d}_{cue}"
    return ROOT / stem / author / f"{stem}_{author}.npz"


def _local_json(idx: int, cue: str, author: str) -> Path:
    stem = f"{idx:02d}_{cue}"
    return ROOT / stem / author / f"{stem}_{author}.json"


def _load_one(entry: dict, author: str) -> dict:
    idx = int(entry["idx"])
    cue = str(entry["cue"])
    meta = entry[author]
    advertised = Path(str(meta.get("npz") or ""))
    path = advertised if advertised.is_file() else _local_npz(idx, cue, author)

    if path.is_file():
        z = np.load(path, allow_pickle=False)
        q = np.asarray(z["joints_rad"], dtype=np.float64)
        g = np.asarray(z["gripper"], dtype=np.float64) if "gripper" in z.files else None
        ee = np.asarray(z["ee_xyz"], dtype=np.float64) if "ee_xyz" in z.files else None
        pose_frame = (
            int(np.asarray(z["pose_frame"]).reshape(-1)[0])
            if "pose_frame" in z.files
            else None
        )
        hz = float(np.asarray(z["hz"]).reshape(-1)[0])
        robot = (
            str(np.asarray(z["robot"]).reshape(-1)[0])
            if "robot" in z.files
            else "FR3"
        )
    else:
        json_path = _local_json(idx, cue, author)
        if not json_path.is_file():
            raise FileNotFoundError(
                f"missing both trajectory files for idx={idx} author={author}: "
                f"{path} and {json_path}"
            )
        data = json.loads(json_path.read_text())
        q = np.asarray(data["joints_rad"], dtype=np.float64)
        g = np.asarray(data.get("gripper", []), dtype=np.float64)
        ee = np.asarray(data.get("ee_xyz", []), dtype=np.float64)
        pose_frame = data.get("pose_frame")
        hz = float(data["hz"])
        robot = str(data.get("robot", "FR3"))
        path = json_path

    q, g, ee, pose_frame = strip_to_cue(q, g, ee, pose_frame)
    if q.ndim != 2 or q.shape[1] < 7 or len(q) == 0:
        raise ValueError(f"{path}: expected non-empty (N, >=7) joints, got {q.shape}")
    if ee.ndim != 2 or ee.shape[1] < 3:
        ee = np.full((len(q), 3), np.nan, dtype=np.float64)
    return {
        "idx": idx,
        "cue": cue,
        "author": author,
        "code": str(meta.get("code") or ""),
        "robot": robot,
        "hz": hz,
        "pose_frame": int(pose_frame),
        "joints_rad": q[:, :7],
        "gripper": g,
        "ee_xyz": ee[:, :3],
        "source": str(path),
    }


def build(index_path: Path, output_path: Path) -> list[dict]:
    index = json.loads(index_path.read_text())
    trajectories = [
        _load_one(entry, author)
        for entry in index
        for author in ("human", "llm")
    ]
    trajectories.sort(key=lambda t: (t["idx"], t["author"]))

    lengths = np.asarray([len(t["joints_rad"]) for t in trajectories], dtype=np.int64)
    offsets = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(lengths)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        format_name=np.array(["droid_small10_bundle"]),
        format_version=np.array([1], dtype=np.int32),
        idx=np.asarray([t["idx"] for t in trajectories], dtype=np.int32),
        cue=np.asarray([t["cue"] for t in trajectories]),
        author=np.asarray([t["author"] for t in trajectories]),
        code=np.asarray([t["code"] for t in trajectories]),
        robot=np.asarray([t["robot"] for t in trajectories]),
        hz=np.asarray([t["hz"] for t in trajectories], dtype=np.float64),
        pose_frame=np.asarray([t["pose_frame"] for t in trajectories], dtype=np.int32),
        offsets=offsets,
        joints_rad=np.concatenate([t["joints_rad"] for t in trajectories], axis=0),
        gripper=np.concatenate([t["gripper"] for t in trajectories], axis=0),
        ee_xyz=np.concatenate([t["ee_xyz"] for t in trajectories], axis=0),
        source=np.asarray([t["source"] for t in trajectories]),
    )
    return trajectories


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()
    trajectories = build(args.index, args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1_000_000:.2f} MB)")
    for t in trajectories:
        print(
            f"  idx={t['idx']:>2} author={t['author']:<5} "
            f"frames={len(t['joints_rad']):>3} cue={t['cue']}"
        )


if __name__ == "__main__":
    main()
