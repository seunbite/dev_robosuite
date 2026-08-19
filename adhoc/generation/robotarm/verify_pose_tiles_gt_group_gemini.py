#!/usr/bin/env python3
"""VLM verify using GT (dir, gripper_orientation) group 3x3 tiles (sanity / ceiling check)."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from google import genai

from verify_pose_tiles_gemini import (
    APPROPRIATE_MEANS_LINE,
    _extract_json,
    _fewshot_block,
    _first_pose,
    _load_json,
    _movement_summary,
)


def _parse_gt_poses(groundtruth: str) -> list[tuple[str, str]]:
    return [(a.strip(), b.strip()) for a, b in re.findall(r"\(([^,]+),\s*([^)]+)\)", groundtruth)]


def _row_with_pose_labels(cfg_row: dict[str, Any], d: str, g: str) -> dict[str, Any]:
    row = copy.deepcopy(cfg_row)
    for step in row.get("movements", []):
        if step.get("type") == "pose":
            step.setdefault("parameters", {})["pose"] = {
                **(step.get("parameters", {}).get("pose") or {}),
                "dir": d,
                "gripper_orientation": g,
            }
            break
    return row


def _prompt_grid(cue_row: dict[str, Any], fewshot_text: str) -> str:
    p = _first_pose(cue_row)
    return f"""
You are verifying robot gesture pose suitability from a 3x3 tiled image.
The tile image belongs to exactly one group: (dir={p.get('dir')}, gripper_orientation={p.get('gripper_orientation')}).
Tiles are numbered 1..9 left-to-right, top-to-bottom.

Task:
1) Judge whether current pose labels (dir + gripper_orientation) are appropriate for this cue.
{APPROPRIATE_MEANS_LINE}
2) If appropriate: propose next movement sequence guidance.
3) If not appropriate: propose corrected dir + gripper_orientation and explain.

Definitions:
Coordinate frame (world): +x = forward toward the human viewer, +y = robot left, +z = up (ceiling). The EE is the parallel-jaw gripper. Its **pointing axis** runs wrist → fingertips (palm normal / approach direction).
1. Direction (`dir`) — where the EE **points** in 3D (NOT gesture category):
Choose `dir` only from the dominant world direction of that pointing axis:
- **up**: pointing toward the ceiling (+z). Example: fingertips up, “air quotes” beside the head.
- **down**: pointing toward the floor (−z). Example: palm-down press, point at the ground.
- **front**: pointing toward the viewer (+x). Example: reach to shake hands, offer object forward.
- **back**: pointing toward the robot body (−x). Example: hand on own chest, retract toward torso.
- **left** / **right**: pointing toward the robot’s left (+y) / right (−y). Example: temple tap on one side.

2. Gripper orientation (`gripper_orientation`):
The **end line** is the jaw-opening line (between the two fingertip tips). Take the plane **perpendicular** to the pointing axis (`dir`). **Orthogonally project** the end line onto that plane. An observer stands **facing that plane** (looking straight at it). Read the projected line:
- **horizontal**: projected line is a **ㅡ** (left–right).
- **vertical**: projected line is a **|** (up–down).


Few-shot movement style examples:
{fewshot_text}

Target cue:
- cue: {cue_row.get("cue")}
- description: {cue_row.get("description", "")}
- current pose: dir={p.get("dir")}, gripper_orientation={p.get("gripper_orientation")}
- current movement summary: {_movement_summary(cue_row)}

Return ONLY strict JSON:
{{
  "pose_is_appropriate": true/false,
  "direction_orientation_assessment": "string",
  "if_appropriate": {{
    "recommended_movement_plan": [
      "step guidance 1",
      "step guidance 2",
      "step guidance 3"
    ]
  }},
  "if_not_appropriate": {{
    "recommended_dir": "front|back|left|right|up|down",
    "recommended_gripper_orientation": "horizontal|vertical",
    "why_change": "string",
    "recommended_movement_plan_after_change": [
      "step guidance 1",
      "step guidance 2",
      "step guidance 3"
    ]
  }},
  "confidence": 0.0
}}
""".strip()


def _load_configs_by_cue(*config_paths: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in config_paths:
        for row in _load_json(path):
            out[row["cue"]] = row
    return out


def run(args: argparse.Namespace) -> None:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set GOOGLE_API_KEY (or GEMINI_API_KEY).")

    consolidated = _load_json(args.consolidated_json)
    cfg_by_cue = _load_configs_by_cue(*args.config_json)
    shots = _load_json(args.shots_json)
    tile_dir = Path(args.tile_dir)
    out_json = Path(args.out_json)

    client = genai.Client(api_key=key)
    fewshot_text = _fewshot_block(shots, n=args.fewshot_n)

    results: list[dict[str, Any]] = []
    for ev in consolidated["rows"]:
        cue = ev["cue"]
        cfg = cfg_by_cue.get(cue)
        if not cfg:
            results.append({"cue": cue, "error": "missing config row"})
            continue

        gt_poses = _parse_gt_poses(ev["groundtruth"])
        if not gt_poses:
            results.append({"cue": cue, "error": "no gt poses"})
            continue

        poses_to_run = gt_poses if args.all_gt_poses else [gt_poses[0]]
        trial_results: list[dict[str, Any]] = []

        for d, g in poses_to_run:
            tile_path = tile_dir / f"group_{d}_{g}.png"
            if not tile_path.is_file():
                trial_results.append(
                    {
                        "gt_dir": d,
                        "gt_gripper_orientation": g,
                        "error": f"missing tile: {tile_path}",
                    }
                )
                continue

            labeled_row = _row_with_pose_labels(cfg, d, g)
            prompt = _prompt_grid(labeled_row, fewshot_text)
            img = Image.open(tile_path).convert("RGB")
            resp = client.models.generate_content(model=args.model, contents=[prompt, img])
            text = (resp.text or "").strip()
            try:
                parsed = _extract_json(text)
            except Exception as e:
                parsed = {"parse_error": str(e), "raw_text": text}

            trial_results.append(
                {
                    "gt_dir": d,
                    "gt_gripper_orientation": g,
                    "tile_image": str(tile_path),
                    "pose_is_appropriate": parsed.get("pose_is_appropriate"),
                    "result": parsed,
                }
            )
            print(
                f"[ok] {cue} gt=({d},{g}) appropriate={parsed.get('pose_is_appropriate')}",
                flush=True,
            )

        any_true = any(t.get("pose_is_appropriate") is True for t in trial_results)
        first_true = (
            trial_results[0].get("pose_is_appropriate") is True if trial_results else False
        )
        results.append(
            {
                "cue_idx": ev.get("cue_idx"),
                "cue": cue,
                "groundtruth": ev["groundtruth"],
                "generation": ev.get("generation"),
                "gt_poses": [{"dir": d, "gripper_orientation": g} for d, g in gt_poses],
                "trials": trial_results,
                "first_gt_appropriate": first_true,
                "any_gt_appropriate": any_true,
            }
        )
        _write_out(out_json, args, results)

    n_cue = len(results)
    n_first = sum(1 for r in results if r.get("first_gt_appropriate"))
    n_any = sum(1 for r in results if r.get("any_gt_appropriate"))
    n_trial = sum(len(r.get("trials", [])) for r in results)
    n_trial_true = sum(
        1
        for r in results
        for t in r.get("trials", [])
        if t.get("pose_is_appropriate") is True
    )

    summary = {
        "pose_is_appropriate_true_per_cue_first_gt": f"{n_first}/{n_cue}",
        "pose_is_appropriate_true_per_cue_any_gt": f"{n_any}/{n_cue}",
        "pose_is_appropriate_true_per_trial": f"{n_trial_true}/{n_trial}",
    }
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    payload = _load_json(out_json)
    payload["summary"] = summary
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_out(out_json: Path, args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "gt_group_3x3",
        "model": args.model,
        "consolidated_json": str(args.consolidated_json),
        "all_gt_poses": args.all_gt_poses,
        "total": len(results),
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--consolidated-json",
        type=Path,
        default=Path("data/results/verify/pilot40_pose_eval_consolidated.json"),
    )
    ap.add_argument(
        "--config-json",
        type=Path,
        nargs="+",
        default=[
            Path("data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot10.json"),
            Path("data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot20_more.json"),
        ],
    )
    ap.add_argument(
        "--shots-json",
        type=Path,
        default=Path("data/seed/shots/manipulator/shot_configs_v19_sophisticated.json"),
    )
    ap.add_argument(
        "--tile-dir",
        type=Path,
        default=Path("data/results/visualize/pose_groups_12"),
    )
    ap.add_argument(
        "--all-gt-poses",
        action="store_true",
        help="Run every GT pose listed per cue (not only the first).",
    )
    ap.add_argument("--model", type=str, default="gemini-2.5-pro")
    ap.add_argument("--fewshot-n", type=int, default=4)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/results/verify/pose_tile_verify_gt_group_first40_gemini.json"),
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
