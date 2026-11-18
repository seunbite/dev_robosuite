#!/usr/bin/env python3
"""Dump a Panda Motion Code as a DROID-rate joint trajectory.

    python adhoc/generation/robotarm/export_droid_traj.py
    python adhoc/generation/robotarm/export_droid_traj.py --cue shake_head_no --idx 2 --hz 15
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_PAPER = _REPO / "papers/icra2027_motion_abstraction"
JSONL = _REPO / "data/seed/_remainder/closest_poses_results.jsonl"
GT_JSON = (
    _REPO
    / "data/results/motion_configs/embodiment_motion_code_gt"
    / "embodiment_motion_code_all.json"
)
DEFAULT_OUT = _PAPER / "droid_replay"


def _load_gt_single_arm(idx: int, cue: str) -> dict:
    import json

    data = json.loads(GT_JSON.read_text(encoding="utf-8"))
    for entry in data:
        plat = (entry.get("platforms") or {}).get("single_arm") or {}
        if int(entry.get("idx") or plat.get("idx") or 0) != idx:
            continue
        if str(entry.get("cue") or plat.get("cue") or "") != cue:
            continue
        if not plat.get("movements"):
            raise SystemExit(f"idx={idx} {cue}: empty single_arm movements")
        return {
            "idx": idx,
            "cue": cue,
            "category": str(entry.get("category") or ""),
            "description": plat.get("description") or "",
            "movements": plat["movements"],
            "parse_ok": bool(plat.get("parse_ok", True)),
            "source_code": plat.get("source_code") or "",
            "platform": "single_arm",
            "domain": "manipulator",
            "notes": plat.get("notes") or [],
        }
    raise SystemExit(f"idx={idx} cue={cue} not found in {GT_JSON}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cue", default="shake_head_no")
    p.add_argument("--idx", type=int, default=2)
    p.add_argument("--hz", type=int, default=15)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    row = _load_gt_single_arm(args.idx, args.cue)
    stem = f"{args.idx:02d}_{args.cue}"
    out_dir = args.out or (DEFAULT_OUT / stem)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "scratch"
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    tmp_cfg = scratch / "cfg.json"
    payload = {k: v for k, v in row.items() if not str(k).startswith("_")}
    tmp_cfg.write_text(__import__("json").dumps([payload], indent=2), encoding="utf-8")

    robotarm = _HERE
    google_legacy = _REPO / "adhoc/generation/tiago/legacy"
    sys.path = [x for x in sys.path if Path(x).resolve() != google_legacy.resolve()]
    if str(robotarm) in sys.path:
        sys.path.remove(str(robotarm))
    sys.path.insert(0, str(robotarm))
    for mod in list(sys.modules):
        if mod == "legacy" or mod.startswith("legacy."):
            del sys.modules[mod]
    from legacy.motion_generation_core import generate  # noqa: WPS433

    os.environ.setdefault("MANIPULATOR_POSE_XYZ_MODE", "trim")
    os.environ["MOTION_FRAME_SCALE"] = "1.0"
    traj_stem = str(out_dir / stem)
    generate(
        robot="FR3",
        cue=str(row["cue"]),
        cue_idx=int(row["idx"]),
        pose_index=None,
        jsonl_path=str(JSONL),
        config_path=str(tmp_cfg),
        output_dir=str(scratch),
        hz=int(args.hz),
        top_k=1,
        ignore_xyz=False,
        overlay_progress_bar=False,
        record_joints=True,
        traj_out_path=traj_stem,
    )
    replay_src = _HERE / "replay_droid.py"
    if replay_src.is_file():
        shutil.copy2(replay_src, out_dir / "replay_droid.py")
    print(f"traj: {traj_stem}.npz")
    print(f"json: {traj_stem}.json")


if __name__ == "__main__":
    main()
