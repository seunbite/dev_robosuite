#!/usr/bin/env python3
"""
Round-2 pose verify: use round-1 VLM / Text predicted (dir, orient) as current pose input.
Score round-2 predictions against human GT pose sets.
"""
from __future__ import annotations

import argparse
import copy
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

from verify_pose_tiles_gemini import (  # noqa: E402
    APPROPRIATE_MEANS_LINE,
    _extract_json,
    _fewshot_block,
    _first_pose,
    _load_tile_pick,
    _movement_summary,
    _resolve_pose_image,
)
from verify_pose_textonly_gemini import _fewshot_block as _fewshot_text_block  # noqa: E402

CONSOLIDATED = _REPO / "data/results/verify/pilot40_pose_eval_consolidated.json"
CONFIG_PATHS = [
    _REPO / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot10.json",
    _REPO / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot20_more.json",
]
SHOTS = _REPO / "data/seed/shots/manipulator/shot_configs_v19_sophisticated.json"
TILE_DIR = _REPO / "data/results/visualize/pose_groups_12"
OUT_JSON = _REPO / "data/results/verify/pilot40_pose_round2_cascade.json"
OUT_TSV = _REPO / "data/results/verify/pilot40_pose_round2_cascade_scored.tsv"


def _gt_poses(s: str) -> set[tuple[str, str]]:
    return {(a.strip(), b.strip()) for a, b in re.findall(r"\(([^,]+),\s*([^)]+)\)", s)}


def _r1_pose(row: dict[str, Any], kind: str) -> tuple[str, str]:
    g = row["generation"]
    key = "VLM" if kind == "VLM" else "Text"
    if row.get(f"verify_{key}_bool"):
        return (g["dir"], g["gripper_orientation"])
    fix = row.get(f"verify_{key}_fix") or {}
    if fix.get("dir") and fix.get("gripper_orientation"):
        return (fix["dir"], fix["gripper_orientation"])
    return (g["dir"], g["gripper_orientation"])


def _r2_pose_from_result(
    r1_dir: str,
    r1_grip: str,
    parsed: dict[str, Any],
) -> tuple[str, str]:
    if parsed.get("pose_is_appropriate"):
        return (r1_dir, r1_grip)
    na = parsed.get("if_not_appropriate") or {}
    rd = na.get("recommended_dir")
    rg = na.get("recommended_gripper_orientation")
    if rd and rg:
        return (rd, rg)
    return (r1_dir, r1_grip)


def _row_with_pose(cfg_row: dict[str, Any], d: str, g: str) -> dict[str, Any]:
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


def _vlm_prompt(cue_row: dict[str, Any], fewshot_text: str) -> str:
    p = _first_pose(cue_row)
    return f"""
You are verifying robot gesture pose suitability from a single rendered robot image.
The image shows one representative pose from group (dir={p.get('dir')}, gripper_orientation={p.get('gripper_orientation')}).

Task:
1) Judge whether current pose labels (dir + gripper_orientation) are appropriate for this cue.
{APPROPRIATE_MEANS_LINE}
2) If appropriate: propose next movement sequence guidance.
3) If not appropriate: propose corrected dir + gripper_orientation and explain.

Definitions:
Coordinate frame (world): +x = forward toward the human viewer, +y = robot left, +z = up (ceiling). The EE is the parallel-jaw gripper. Its **pointing axis** runs wrist → fingertips (palm normal / approach direction).
1. Direction (`dir`) — where the EE **points** in 3D (NOT gesture category):
Choose `dir` only from the dominant world direction of that pointing axis:
- **up**: pointing toward the ceiling (+z).
- **down**: pointing toward the floor (−z).
- **front**: pointing toward the viewer (+x).
- **back**: pointing toward the robot body (−x).
- **left** / **right**: pointing toward the robot's left (+y) / right (−y).

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
    "recommended_movement_plan": ["step 1", "step 2", "step 3"]
  }},
  "if_not_appropriate": {{
    "recommended_dir": "front|back|left|right|up|down",
    "recommended_gripper_orientation": "horizontal|vertical",
    "why_change": "string",
    "recommended_movement_plan_after_change": ["step 1", "step 2", "step 3"]
  }},
  "confidence": 0.0
}}
""".strip()


def _text_prompt(cue_row: dict[str, Any], fewshot_text: str) -> str:
    p = _first_pose(cue_row)
    return f"""
You are verifying robot cue pose suitability using ONLY text (no image).

Definitions (must be strictly followed):
- World frame: +x = forward toward human viewer, +y = robot left, +z = up.
- End-effector (EE) pointing axis: wrist -> fingertips (approach direction).
- Direction `dir` is the dominant world direction of this pointing axis: up/down/front/back/left/right.
- `gripper_orientation`: project jaw-opening line onto plane perpendicular to `dir`; horizontal (ㅡ) or vertical (|).

Task:
1) Q1: judge if current pose labels (dir + gripper_orientation) are appropriate for the cue intent.
{APPROPRIATE_MEANS_LINE}
2) Q3: if appropriate, propose movement plan; if not appropriate, propose corrected pose and movement plan.

Few-shot style hints:
{fewshot_text}

Target:
- cue: {cue_row.get("cue")}
- description: {cue_row.get("description","")}
- current_pose: dir={p.get("dir")}, gripper_orientation={p.get("gripper_orientation")}

Return ONLY strict JSON:
{{
  "pose_is_appropriate": true/false,
  "direction_orientation_assessment": "string",
  "if_appropriate": {{
    "recommended_movement_plan": ["step 1", "step 2", "step 3"]
  }},
  "if_not_appropriate": {{
    "recommended_dir": "front|back|left|right|up|down",
    "recommended_gripper_orientation": "horizontal|vertical",
    "why_change": "string",
    "recommended_movement_plan_after_change": ["step 1", "step 2", "step 3"]
  }},
  "confidence": 0.0
}}
""".strip()


def _load_configs_by_cue() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in CONFIG_PATHS:
        for row in json.loads(path.read_text(encoding="utf-8")):
            out[row["cue"]] = row
    return out


def run(args: argparse.Namespace) -> None:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set GOOGLE_API_KEY (or GEMINI_API_KEY).")

    consolidated = json.loads(CONSOLIDATED.read_text(encoding="utf-8"))
    eval_rows = consolidated["rows"]
    cfg_by_cue = _load_configs_by_cue()
    shots = json.loads(SHOTS.read_text(encoding="utf-8"))
    fewshot_vlm = _fewshot_block(shots, n=args.fewshot_n)
    fewshot_text = _fewshot_text_block(shots, n=args.fewshot_n)
    tile_pick = _load_tile_pick(args.tile_pick_json)
    client = genai.Client(api_key=key)

    out_rows: list[dict[str, Any]] = []
    for ev in eval_rows:
        cue = ev["cue"]
        cfg = cfg_by_cue[cue]
        gt = ev["groundtruth"]
        gen = ev["generation"]

        r1_vlm = _r1_pose(ev, "VLM")
        r1_text = _r1_pose(ev, "Text")

        record: dict[str, Any] = {
            "cue_idx": ev.get("cue_idx"),
            "cue": cue,
            "groundtruth": gt,
            "generation": gen,
            "round1_VLM_pose": {"dir": r1_vlm[0], "gripper_orientation": r1_vlm[1]},
            "round1_Text_pose": {"dir": r1_text[0], "gripper_orientation": r1_text[1]},
            "round1_VLM_correct": r1_vlm in _gt_poses(gt),
            "round1_Text_correct": r1_text in _gt_poses(gt),
        }

        for kind, r1, fewshot, prompt_fn, use_image in [
            ("VLM", r1_vlm, fewshot_vlm, _vlm_prompt, True),
            ("Text", r1_text, fewshot_text, _text_prompt, False),
        ]:
            labeled = _row_with_pose(cfg, r1[0], r1[1])
            prompt = prompt_fn(labeled, fewshot)
            contents: list[Any] = [prompt]
            tile_meta: dict[str, Any] = {}
            if use_image:
                d, g = r1
                if (d, g) not in tile_pick:
                    record[f"round2_{kind}_error"] = f"no tile pick for ({d}, {g})"
                    continue
                group_path = TILE_DIR / f"group_{d}_{g}.png"
                if not group_path.is_file():
                    record[f"round2_{kind}_error"] = f"missing {group_path}"
                    continue
                img, _ = _resolve_pose_image(
                    group_path, d, g, tile_pick[(d, g)], None, False
                )
                contents.append(img)
                tile_meta = {"tile_index": tile_pick[(d, g)], "tile_group": f"{d}_{g}"}

            if args.dry_run:
                record[f"round2_{kind}_skipped"] = True
                continue

            resp = client.models.generate_content(model=args.model, contents=contents)
            text = (resp.text or "").strip()
            try:
                parsed = _extract_json(text)
            except Exception as e:
                parsed = {"parse_error": str(e), "raw_text": text}

            r2 = _r2_pose_from_result(r1[0], r1[1], parsed)
            record[f"round2_{kind}_bool"] = parsed.get("pose_is_appropriate")
            record[f"round2_{kind}_fix"] = (
                None
                if parsed.get("pose_is_appropriate")
                else {
                    "dir": (parsed.get("if_not_appropriate") or {}).get("recommended_dir"),
                    "gripper_orientation": (
                        (parsed.get("if_not_appropriate") or {}).get("recommended_gripper_orientation")
                    ),
                }
            )
            record[f"round2_{kind}_pose"] = {"dir": r2[0], "gripper_orientation": r2[1]}
            record[f"round2_{kind}_correct"] = r2 in _gt_poses(gt)
            record[f"round2_{kind}_raw"] = parsed
            if tile_meta:
                record[f"round2_{kind}_tile"] = tile_meta
            print(
                f"[r2 {kind}] {cue} r1={r1} -> r2={r2} gt_ok={record[f'round2_{kind}_correct']}",
                flush=True,
            )

        out_rows.append(record)
        if not args.dry_run:
            _write_checkpoint(out_rows, args.model)

    n = len(out_rows)
    r1v_ok = sum(1 for r in out_rows if r.get("round1_VLM_correct"))
    r1t_ok = sum(1 for r in out_rows if r.get("round1_Text_correct"))
    r2v_ok = sum(1 for r in out_rows if r.get("round2_VLM_correct"))
    r2t_ok = sum(1 for r in out_rows if r.get("round2_Text_correct"))

    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "mode": "round2_cascade_from_round1_pose",
        "n": len(out_rows),
        "accuracy": {
            "round1_VLM": f"{r1v_ok}/{n}",
            "round1_Text": f"{r1t_ok}/{n}",
            "round2_VLM": f"{r2v_ok}/{n}",
            "round2_Text": f"{r2t_ok}/{n}",
            "round1_VLM_rate": r1v_ok / n if n else 0,
            "round1_Text_rate": r1t_ok / n if n else 0,
            "round2_VLM_rate": r2v_ok / n if n else 0,
            "round2_Text_rate": r2t_ok / n if n else 0,
        },
        "rows": out_rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_tsv(out_rows)
    print("\n=== Accuracy (pred pose in GT set) ===")
    for k, v in payload["accuracy"].items():
        if k.endswith("_rate"):
            print(f"  {k}: {v:.1%}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_TSV}")


def _write_checkpoint(rows: list[dict[str, Any]], model: str) -> None:
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "partial": True,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_tsv(rows: list[dict[str, Any]]) -> None:
    lines = [
        "cue\tgroundtruth\t"
        "r1_vlm\tr1_vlm_ok\tr2_vlm\tr2_vlm_ok\t"
        "r1_text\tr1_text_ok\tr2_text\tr2_text_ok"
    ]
    for r in rows:
        def fmt(p: dict) -> str:
            return f"{p['dir']},{p['gripper_orientation']}"

        lines.append(
            "\t".join(
                [
                    r["cue"],
                    r["groundtruth"],
                    fmt(r["round1_VLM_pose"]),
                    str(r.get("round1_VLM_correct")),
                    fmt(r["round2_VLM_pose"]) if "round2_VLM_pose" in r else "",
                    str(r.get("round2_VLM_correct", "")),
                    fmt(r["round1_Text_pose"]),
                    str(r.get("round1_Text_correct")),
                    fmt(r["round2_Text_pose"]) if "round2_Text_pose" in r else "",
                    str(r.get("round2_Text_correct", "")),
                ]
            )
        )
    OUT_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--fewshot-n", type=int, default=4)
    ap.add_argument(
        "--tile-pick-json",
        type=Path,
        default=_REPO / "data/results/verify/pose_tile_pick_by_group.json",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
