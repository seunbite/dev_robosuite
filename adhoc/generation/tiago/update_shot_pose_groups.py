#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from legacy.render_mobile_config import _apply_pose, _default_qpos, _make_env  # noqa: E402
from legacy import render_mobile_manip_schema as sch  # noqa: E402


DIR_TO_DISPLAY = {
    "front": "front",
    "back": "back",
    "left": "out",
    "right": "in",
    "up": "up",
    "down": "down",
}


def _first_pose(row: dict) -> dict:
    for step in row.get("movements", []):
        if step.get("type") == "pose":
            return (step.get("parameters") or {}).get("pose") or {}
    return {}


def run(args: argparse.Namespace) -> None:
    rows = json.loads(args.shots_json.read_text(encoding="utf-8"))
    env = _make_env()
    try:
        for row in rows:
            pose = _first_pose(row)
            if not pose:
                continue
            q = _default_qpos(env)
            _apply_pose(env, q, pose)
            env.sim.data.qpos[:] = q
            env.sim.forward()

            dir_raw = sch._dir_six_way_from_anchor(env, q, cam_name=sch.CAM)
            orient = sch._gripper_horizontal_vertical_by_reach_dir(env.sim, dir_raw)
            dir_disp = DIR_TO_DISPLAY.get(dir_raw, dir_raw)
            row["first_pose_group"] = {
                "direction_group": dir_disp,
                "gripper_orientation": orient,
                "group_key": f"{dir_disp}_{orient}",
            }
    finally:
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()

    args.shots_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"updated {args.shots_json} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shots-json",
        type=Path,
        default=Path("data/seed/shots/tiago/shot_configs_19_mobile.json"),
    )
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
