#!/usr/bin/env python3
"""Build human-review JSON for GT-fixed motion generation (40 cues).

groundtruth_* fields initially copy generation (all treated as correct / 'o').
Edit groundtruth_tail or groundtruth after manual review.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _dev_root() -> Path:
    for anc in _HERE.parents:
        if (anc / "data" / "results" / "verify").is_dir():
            return anc
    raise SystemExit(f"Cannot find dev_robosuite root from {_HERE}")


_REPO = _dev_root()

MOTION_CONFIGS = (
    _REPO
    / "data/results/motion_configs/manipulator"
    / "motion_configs_prompt_v19_gt_fixed_pose_pilot40.json"
)
POSE_CONSOLIDATED = _REPO / "data/results/verify/pilot40_pose_eval_consolidated.json"
OUT_JSON = _REPO / "data/results/verify/pilot40_motion_gt_fixed_eval_consolidated.json"
OUT_TSV = _REPO / "data/results/verify/pilot40_motion_gt_fixed_eval_consolidated.tsv"


def _pose_label(d: str, g: str) -> str:
    return f"o ({d}, {g})"


def _first_pose(movements: list) -> dict:
    for step in movements:
        if step.get("type") == "pose":
            return copy.deepcopy((step.get("parameters") or {}).get("pose") or {})
    return {}


def _tail_movements(movements: list) -> list:
    out: list = []
    seen_pose = False
    for step in movements:
        if step.get("type") == "pose":
            seen_pose = True
            continue
        if seen_pose:
            out.append(copy.deepcopy(step))
    return out


def main() -> None:
    configs = json.loads(MOTION_CONFIGS.read_text(encoding="utf-8"))
    pose_rows = {r["cue"]: r for r in json.loads(POSE_CONSOLIDATED.read_text(encoding="utf-8"))["rows"]}

    rows: list[dict] = []
    for cfg in sorted(configs, key=lambda x: int(x.get("idx", 0))):
        cue = cfg["cue"]
        movements = cfg.get("movements") or []
        pose = _first_pose(movements)
        tail = _tail_movements(movements)
        d, g = pose.get("dir", "?"), pose.get("gripper_orientation", "?")
        gt_pose_str = cfg.get("groundtruth") or _pose_label(d, g)

        row = {
            "cue_idx": cfg.get("idx"),
            "cue": cue,
            "description": cfg.get("description", ""),
            "groundtruth": gt_pose_str,
            "groundtruth_pose": _pose_label(d, g),
            "generation_pose": {
                "dir": d,
                "gripper_orientation": g,
                "x": pose.get("x"),
                "y": pose.get("y"),
                "z": pose.get("z"),
                "pose_id": pose.get("pose_id"),
            },
            "groundtruth_tail": copy.deepcopy(tail),
            "generation_tail": copy.deepcopy(tail),
            "groundtruth_movements": copy.deepcopy(movements),
            "generation_movements": copy.deepcopy(movements),
            "n_tail_steps": len(tail),
            "human_pose_eval": pose_rows.get(cue, {}).get("groundtruth"),
            "config_meta": {
                "model": cfg.get("model"),
                "time": cfg.get("time"),
                "generation_mode": cfg.get("generation_mode"),
                "state": cfg.get("state"),
            },
        }
        rows.append(row)

    payload = {
        "sources": {
            "motion_configs": str(MOTION_CONFIGS.relative_to(_REPO)),
            "pose_human_eval": str(POSE_CONSOLIDATED.relative_to(_REPO)),
        },
        "groundtruth_note": (
            "Placeholder for manual review: groundtruth_tail and groundtruth_movements initially "
            "copy generation_tail / generation_movements (treated as all correct). "
            "Edit groundtruth_tail or set groundtruth to x (...) after review. "
            "groundtruth_pose uses GT-fixed first pose (o (...))."
        ),
        "n": len(rows),
        "rows": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    tsv_lines = [
        "cue_idx\tcue\tgroundtruth_pose\tn_tail\tmodel\ttime\treview_status",
    ]
    for r in rows:
        m = r.get("config_meta") or {}
        tsv_lines.append(
            f"{r.get('cue_idx')}\t{r['cue']}\t{r.get('groundtruth_pose','')}\t"
            f"{r.get('n_tail_steps',0)}\t{m.get('model','')}\t{m.get('time','')}\tplaceholder_o"
        )
    OUT_TSV.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUT_JSON} ({len(rows)} rows)")
    print(f"Wrote {OUT_TSV}")


if __name__ == "__main__":
    main()
