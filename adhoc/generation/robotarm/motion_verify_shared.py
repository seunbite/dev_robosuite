#!/usr/bin/env python3
"""Shared motion verify prompt + component parsing."""
from __future__ import annotations

import json
from typing import Any

from verify_pose_tiles_gemini import APPROPRIATE_MEANS_LINE, _first_pose, _movement_summary


def motion_verify_prompt(row: dict[str, Any], fewshot_text: str, *, modality: str) -> str:
    p = _first_pose(row)
    fixed = row.get("gt_fixed_first_pose") or p
    tail_json = json.dumps(
        [s for s in row.get("movements", []) if s.get("type") != "pose"],
        indent=2,
        ensure_ascii=False,
    )
    intro = (
        "You see one composite image: alpha-stack of frames with end-effector trajectory (yellow → purple)."
        if modality == "vlm"
        else "You have NO image — use only the text below (fixed pose + tail JSON)."
    )
    return f"""
You are verifying a robot-arm motion (IIWA) for a social gesture cue.
{intro}

Context:
- The **first pose is fixed** (human GT); only the **tail movement** after that pose was generated.
- World frame: +x forward toward viewer, +y robot left, +z up.
- Movement uses joint rotations (shoulder/elbow/wrist) and/or Cartesian paths (line/arc).

Task:
1) Q1: Is the **current tail movement** appropriate for conveying this cue, given the fixed start pose?
{APPROPRIATE_MEANS_LINE.replace("this pose", "this fixed start pose").replace("subsequent movements", "the shown tail movement")}
2) Q2: If appropriate, note small optional refinements (short bullets).
3) Q3: If **not** appropriate, recommend how to **change the movement** using the component vocabulary below
   (same style as human motion annotations: e.g. "z +- rep wrist", "x + non hold", "arc xz", "line y").

Component vocabulary for recommendations:
- movement: axes x/y/z each +, -, or +- ; optional joint shoulder|elbow|wrist ; repetition non|rep|any ; optional hold
- path_arc: plane xy|yz|xz
- path_line: axis x|y|z

Few-shot examples (pose + movement style):
{fewshot_text}

Target:
- cue: {row.get("cue")}
- description: {row.get("description", "")}
- fixed_start_pose: dir={fixed.get("dir")}, gripper_orientation={fixed.get("gripper_orientation")}
- current_tail_summary: {_movement_summary(row)}
- current_tail_json:
{tail_json}

Return ONLY strict JSON:
{{
  "movement_is_appropriate": true/false,
  "movement_assessment": "string",
  "if_appropriate": {{
    "optional_refinements": ["string", "string"]
  }},
  "if_not_appropriate": {{
    "why_not": "string",
    "recommended_component": {{
      "kind": "movement|path_arc|path_line",
      "axes": {{"x": "+", "y": "-", "z": "+-"}},
      "joint": "shoulder|elbow|wrist|null",
      "repetition": "non|rep|any|null",
      "hold": true|null,
      "plane": "xy|yz|xz|null",
      "axis": "x|y|z|null"
    }},
    "recommended_tail_guidance": ["step 1", "step 2", "step 3"]
  }},
  "confidence": 0.0
}}
""".strip()


def normalize_component(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if not raw or not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    if kind not in ("movement", "path_arc", "path_line"):
        return None
    out: dict[str, Any] = {"kind": kind}
    if kind == "path_arc":
        plane = raw.get("plane") or "xz"
        if plane == "null":
            return None
        out["plane"] = str(plane).lower()
        return out
    if kind == "path_line":
        axis = raw.get("axis")
        if not axis or axis == "null":
            return None
        out["axis"] = str(axis).lower()
        return out
    axes = raw.get("axes") or {}
    if isinstance(axes, dict):
        clean = {}
        for ax in "xyz":
            if ax in axes and axes[ax] in ("+", "-", "+-"):
                clean[ax] = axes[ax]
        if clean:
            out["axes"] = clean
    j = raw.get("joint")
    if j and j != "null":
        out["joint"] = str(j).lower()
    rep = raw.get("repetition")
    if rep and rep != "null":
        out["repetition"] = str(rep).lower()
    if raw.get("hold") is True:
        out["hold"] = True
    return out if len(out) > 1 else None


def record_from_parsed(parsed: dict[str, Any]) -> dict[str, Any]:
    rec_comp = None
    if not parsed.get("movement_is_appropriate"):
        rec_comp = normalize_component(
            (parsed.get("if_not_appropriate") or {}).get("recommended_component")
        )
    return {
        "verify_result": parsed,
        "movement_is_appropriate": parsed.get("movement_is_appropriate"),
        "recommended_component": rec_comp,
    }
