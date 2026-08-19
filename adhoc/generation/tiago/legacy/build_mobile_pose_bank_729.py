#!/usr/bin/env python3
"""Build ``mobile_pose_bank_729.json``: 3^6 right-arm grid + schema dir/orient + pad-mid x,y,z %.

Writes: ``data/seed/tiago/mobile_pose_bank_729.json`` (repo-relative).
Run once (or after changing brute grid): PYTHONPATH=. .venv/bin/python ..."""
from __future__ import annotations

import itertools
import json
import math
import os
import sys

import numpy as np

local_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if local_root not in sys.path:
    sys.path.insert(0, local_root)

from adhoc.generation.tiago.legacy import render_mobile_manip_schema as sch  # noqa: E402


def main() -> None:
    out_dir = os.path.join(local_root, "data", "seed", "tiago")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mobile_pose_bank_729.json")

    env = sch._make_env()
    try:
        deg_steps = list(sch._joint_angle_grid)
        base = sch._default_qpos(env)
        base[sch.QI_TORSO] = sch.TORSO_PRESETS["high"]
        base[sch.QI_HEAD_PAN] = 0.0
        base[sch.QI_HEAD_TILT] = 0.0

        records_raw: list[dict] = []
        combos = itertools.product(deg_steps, repeat=6)
        for tup in combos:
            q = base.copy()
            rad = [math.radians(float(d)) for d in tup]
            q[6:12] = np.asarray(rad, dtype=float)
            env.sim.data.qpos[:] = q
            env.sim.forward()
            d_lbl = sch._dir_six_way_from_anchor(env, q, cam_name=sch.CAM)
            h_lbl = sch._gripper_horizontal_vertical_by_reach_dir(env.sim, d_lbl)
            mid = sch._right_fingertip_pad_mid_world_xyz(env.sim)
            if mid is None or d_lbl == "?" or h_lbl == "?":
                continue
            r_arm_rad = np.asarray(env.sim.data.qpos[6:12], dtype=float).tolist()
            records_raw.append(
                {
                    "joint_deg": [int(x) for x in tup],
                    "r_arm_rad": r_arm_rad,
                    "dir": d_lbl,
                    "orient": h_lbl,
                    "_mid": np.asarray(mid, dtype=float),
                }
            )

        stack = np.stack([r["_mid"] for r in records_raw], axis=0)
        lo = stack.min(axis=0)
        hi = stack.max(axis=0)

        # Linear extrema: pad-mid min→0 … max→100 (G1-style uniform scale).
        # Body center stored for reference only.
        q_ref = base.copy()
        env.sim.data.qpos[:] = q_ref
        env.sim.forward()
        tid = env.sim.model.body_name2id("robot0_torso_lift_link")
        center = np.asarray(env.sim.data.body_xpos[tid], dtype=float).copy()
        center = sch._clamp_center_into_bounds(lo, hi, center)

        out_list: list[dict] = []
        for r in records_raw:
            mid = r.pop("_mid")
            px, py, pz = sch._world_xyz_to_pct_0100(mid, lo, hi, None)
            r["x"] = round(px, 3)
            r["y"] = round(py, 3)
            r["z"] = round(pz, 3)
            out_list.append(r)

        payload = {
            "meta": {
                "n": len(out_list),
                "world_pad_mid_bounds_m": {
                    "x": [float(lo[0]), float(hi[0])],
                    "y": [float(lo[1]), float(hi[1])],
                    "z": [float(lo[2]), float(hi[2])],
                },
                "world_body_center_m": {
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "z": float(center[2]),
                },
                "percentile_anchor": "linear_extrema",
                "note": (
                    "x,y,z are 0-100% linear over pad-mid extrema (min→0, max→100); "
                    "body center kept for reference only; pick nearest with same dir+orient"
                ),
            },
            "poses": out_list,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(out_list)} poses -> {out_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
