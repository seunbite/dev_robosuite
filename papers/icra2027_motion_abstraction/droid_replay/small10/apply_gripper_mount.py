#!/usr/bin/env python3
"""Apply measured Robotiq clocking to sim + dump q7.

    python apply_gripper_mount.py           # reads gripper_mount.json
    python apply_gripper_mount.py --dry-run

After check_droid_env.py --write-mount DEG on the laptop, copy
gripper_mount.json here, then run this to rotate the FR3 Robotiq in
robosuite and add that q7 to every small10 dump so horiz/vert still holds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gripper_mount import _ROBOSUITE_JSON, _SMALL10_JSON, dump_q7_delta_rad, load_mount, offset_q7, write_mount

_HERE = Path(__file__).resolve().parent


def _rewrite_npz(path: Path, delta: float) -> dict:
    z = np.load(path, allow_pickle=False)
    q = np.asarray(z["joints_rad"], dtype=np.float64)
        q2, meta = offset_q7(q, delta)
        extra = {k: z[k] for k in z.files if k != "joints_rad"}
        extra["q7_mount_delta_deg"] = np.array(float(np.degrees(delta)))
        np.savez(path, joints_rad=q2, **extra)
    meta["path"] = str(path.name)
    return meta


def _rewrite_json(path: Path, delta: float) -> None:
    if not path.is_file():
        return
    data = json.loads(path.read_text())
    q = np.asarray(data["joints_rad"], dtype=np.float64)
    q2, _ = offset_q7(q, delta)
    data["joints_rad"] = q2.tolist()
    data["q7_mount_delta_deg"] = float(np.degrees(delta))
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--deg", type=float, default=None, help="override q7_to_horizontal_deg (e.g. preview clips)")
    p.add_argument("--mount", type=Path, default=_SMALL10_JSON)
    args = p.parse_args()
    data = load_mount(args.mount)
    deg = float(args.deg) if args.deg is not None else float(data.get("q7_to_horizontal_deg") or 0.0)
    if abs(deg) < 1e-9:
        from robosuite.models.robots.manipulators.fr3_robot import BANK_Q7_OFFSET_RAD

        deg = float(np.degrees(BANK_Q7_OFFSET_RAD))
        print(f"mount JSON had 0; using FR3 BANK_Q7_OFFSET_RAD = {deg:g}°")
    data["q7_to_horizontal_deg"] = deg
    delta = dump_q7_delta_rad(data)
    print(f"q7_to_horizontal_deg={deg:g}  dump Δq7={np.degrees(delta):g}°  sim_gripper_z={-deg:g}°")
    dumps = sorted(_HERE.glob("*_*/*/*.npz"))
    dumps = [d for d in dumps if d.name.endswith("_human.npz") or d.name.endswith("_llm.npz")]
    if args.dry_run:
        n_clip = 0
        for path in dumps:
            z = np.load(path)
            q = np.asarray(z["joints_rad"], dtype=np.float64)
            _, meta = offset_q7(q, delta)
            print(f"  {path.parent.parent.name}/{path.parent.name} clip={meta['n_clip']} q7 {meta['q7_deg_min']:.1f}..{meta['q7_deg_max']:.1f}")
            n_clip += meta["n_clip"]
        print(f"dry-run {len(dumps)} dumps, {n_clip} clipped q7 samples")
        return
    write_mount(deg, path=_SMALL10_JSON, extra={k: data[k] for k in data if k not in {"note"}})
    print("wrote", _SMALL10_JSON)
    if _ROBOSUITE_JSON.parent.is_dir():
        print("wrote", _ROBOSUITE_JSON)
    n_clip = 0
    for path in dumps:
        meta = _rewrite_npz(path, delta)
        _rewrite_json(path.with_suffix(".json"), delta)
        print(f"  {path.name} clip={meta['n_clip']}")
        n_clip += meta["n_clip"]
    print(f"done {len(dumps)} dumps, {n_clip} clipped q7 samples")
    print("next: python build_bundle.py && python render_from_neutral.py --idx 1 --idx 2 ...")


if __name__ == "__main__":
    main()
