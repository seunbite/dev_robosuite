#!/usr/bin/env python3
"""
Single-tile pose VLM verify with representative/iconic framing (same task as pairwise).

Uses human tile pick per (dir, gripper_orientation) group; one cropped PNG per cue.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from verify_pose_pairwise_12_gemini import REPRESENTATIVE_MEANS_LINE  # noqa: E402
from verify_pose_tiles_gemini import (  # noqa: E402
    _extract_json,
    _fewshot_block,
    _first_pose,
    _load_json,
    _load_tile_pick,
    _resolve_pose_image,
)

CONSOLIDATED = _REPO / "data/results/verify/pilot40_pose_eval_consolidated.json"
CONFIG_PATHS = [
    _REPO / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot10.json",
    _REPO / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot20_more.json",
]
TILE_DIR = _REPO / "data/results/visualize/pose_groups_12"
TILE_PICK = _REPO / "data/results/verify/pose_tile_pick_by_group.json"
SHOTS = _REPO / "data/seed/shots/manipulator/shot_configs_v19_sophisticated.json"
DEFAULT_OUT = _REPO / "data/results/verify/pilot40_pose_tile_single_representative_gemini.json"


def _gt_poses(groundtruth: str) -> set[tuple[str, str]]:
    return {(a.strip(), b.strip()) for a, b in re.findall(r"\(([^,]+),\s*([^)]+)\)", groundtruth)}


def _dedupe_consolidated(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        c = r.get("cue", "")
        if c in seen:
            continue
        seen.add(c)
        out.append(r)
    return out


def _configs_by_cue() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in CONFIG_PATHS:
        for row in _load_json(p):
            out[row["cue"]] = row
    return out


def _representative_prompt(cue_row: dict[str, Any], fewshot_text: str) -> str:
    p = _first_pose(cue_row)
    return f"""
You are judging one robot pose image for an iconic gesture cue.
The image shows a single cropped robot render from group (dir={p.get('dir')}, gripper_orientation={p.get('gripper_orientation')}).

Task: Is this pose MORE REPRESENTATIVE of the cue — would a human viewer readily recognize this static
arm configuration as an iconic snapshot of the gesture's meaning?
{REPRESENTATIVE_MEANS_LINE}
Do NOT judge based on whether this is a convenient starting pose for follow-on motion, or what movement should come next.

Definitions:
Coordinate frame: +x forward (viewer), +y robot left, +z up. EE pointing axis = wrist → fingertips.
- dir: dominant world direction of pointing axis (front|back|left|right|up|down).
- gripper_orientation: projected jaw-opening line on plane ⊥ to dir — horizontal (ㅡ) or vertical (|).

Few-shot movement style examples (context only; do not score movability):
{fewshot_text}

Target cue:
- cue: {cue_row.get("cue")}
- description: {cue_row.get("description", "")}
- labeled pose: dir={p.get("dir")}, gripper_orientation={p.get("gripper_orientation")}

Return ONLY strict JSON:
{{
  "pose_is_representative": true/false,
  "direction_orientation_assessment": "string: why this pose is or is not iconic/representative of the cue",
  "if_representative": {{
    "iconic_elements": ["what in the image reads as the gesture"]
  }},
  "if_not_representative": {{
    "recommended_dir": "front|back|left|right|up|down",
    "recommended_gripper_orientation": "horizontal|vertical",
    "why_change": "string",
    "what_would_be_more_iconic": "string"
  }},
  "confidence": 0.0
}}
""".strip()


def _pred_pose(row: dict[str, Any], parsed: dict[str, Any]) -> tuple[str, str]:
    p = _first_pose(row)
    if parsed.get("pose_is_representative"):
        return (p["dir"], p["gripper_orientation"])
    na = parsed.get("if_not_representative") or {}
    rd, rg = na.get("recommended_dir"), na.get("recommended_gripper_orientation")
    if rd and rg:
        return (rd, rg)
    return (p["dir"], p["gripper_orientation"])


def run(args: argparse.Namespace) -> None:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY.")

    consolidated = _load_json(args.consolidated_json)
    eval_rows = _dedupe_consolidated(consolidated["rows"])
    cfg_by_cue = _configs_by_cue()
    tile_dir = Path(args.tile_dir)
    tile_pick = _load_tile_pick(args.tile_pick_json)
    shots = _load_json(args.shots_json)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if args.cues:
        want = {c.strip() for c in args.cues.split(",") if c.strip()}
        eval_rows = [r for r in eval_rows if r.get("cue") in want]

    client = genai.Client(api_key=key)
    fewshot_text = _fewshot_block(shots, n=args.fewshot_n)
    results: list[dict[str, Any]] = []

    for ev in eval_rows:
        cue = ev["cue"]
        cfg = cfg_by_cue.get(cue)
        if not cfg:
            results.append({"cue": cue, "error": "missing motion config"})
            continue

        pose = _first_pose(cfg)
        d, g = pose.get("dir"), pose.get("gripper_orientation")
        if (d, g) not in tile_pick:
            results.append({"cue": cue, "error": f"no tile pick for ({d}, {g})"})
            continue

        group_path = tile_dir / f"group_{d}_{g}.png"
        if not group_path.is_file():
            results.append({"cue": cue, "error": f"missing {group_path}"})
            continue

        tile_index = tile_pick[(d, g)]
        try:
            img, saved = _resolve_pose_image(
                group_path, d, g, tile_index, None, False
            )
        except Exception as e:
            results.append({"cue": cue, "error": str(e)})
            continue

        prompt = _representative_prompt(cfg, fewshot_text)
        resp = client.models.generate_content(
            model=args.model,
            contents=[prompt, img],
        )
        text = (resp.text or "").strip()
        try:
            parsed = _extract_json(text)
        except Exception as e:
            parsed = {"parse_error": str(e), "raw_text": text}

        gt = _gt_poses(ev["groundtruth"])
        gen_pose = (ev["generation"]["dir"], ev["generation"]["gripper_orientation"])
        pred = _pred_pose(cfg, parsed)
        bool_rep = parsed.get("pose_is_representative")

        # strict: predicted pose in human GT set
        strict_correct = pred in gt
        # classifier: bool matches human o/x on generation labels
        human_o = ev["groundtruth"].strip().startswith("o ")
        clf_correct = (bool_rep is True and human_o) or (bool_rep is False and not human_o)

        results.append(
            {
                "cue_idx": ev.get("cue_idx"),
                "cue": cue,
                "groundtruth": ev["groundtruth"],
                "generation_pose": ev["generation"],
                "labeled_dir": d,
                "labeled_gripper_orientation": g,
                "tile_index": tile_index,
                "tile_image": str(group_path),
                "prompt_task": "representative_single_tile",
                "result": parsed,
                "pose_is_representative": bool_rep,
                "pred_pose": {"dir": pred[0], "gripper_orientation": pred[1]},
                "strict_correct": strict_correct,
                "classifier_correct": clf_correct,
                "generation_in_gt": gen_pose in gt,
            }
        )
        mark = "OK" if strict_correct else "MISS"
        print(
            f"[{mark}] {cue} rep={bool_rep} strict={strict_correct} "
            f"clf={clf_correct} tile={tile_index}",
            flush=True,
        )
        out_json.write_text(
            json.dumps(
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "mode": "single_tile_representative",
                    "model": args.model,
                    "prompt_task": "representative/iconic (not starting pose)",
                    "partial": True,
                    "results": results,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    strict_ok = sum(1 for r in results if r.get("strict_correct"))
    clf_ok = sum(1 for r in results if r.get("classifier_correct"))
    n = len([r for r in results if "strict_correct" in r])
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "single_tile_representative",
        "model": args.model,
        "prompt_task": "representative/iconic (not starting pose)",
        "consolidated_json": str(args.consolidated_json),
        "tile_pick_json": str(args.tile_pick_json),
        "n": n,
        "strict_accuracy": strict_ok / n if n else None,
        "classifier_accuracy": clf_ok / n if n else None,
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"\nWrote {out_json}\n"
        f"strict (pred pose in GT): {strict_ok}/{n}\n"
        f"classifier (bool vs o/x): {clf_ok}/{n}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Single-tile representative pose verify (Gemini)")
    p.add_argument("--consolidated-json", type=Path, default=CONSOLIDATED)
    p.add_argument("--tile-dir", type=Path, default=TILE_DIR)
    p.add_argument("--tile-pick-json", type=Path, default=TILE_PICK)
    p.add_argument("--shots-json", type=Path, default=SHOTS)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model", default="gemini-2.5-pro")
    p.add_argument("--fewshot-n", type=int, default=4)
    p.add_argument("--cues", type=str, default=None)
    run(p.parse_args())


if __name__ == "__main__":
    main()
