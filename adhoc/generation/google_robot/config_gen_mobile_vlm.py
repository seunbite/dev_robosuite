"""Unified LLM config generation (Gemini API) for pilot-40 Google Robot exp 1 & 7."""
from __future__ import annotations

import json
import os
import re
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from pilot40_paths import (
    CUES_YAML,
    FEWSHOT_SHOTS,
    GT_CONSOLIDATED,
    SHOTS,
    load_config_list,
    pilot40_cue_names,
    prompt_exp_path,
    row_generation_done,
    upsert_config_row,
)

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
import sys

for p in (_REPO, _HERE, _HERE / "legacy"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DIR_TO_ARM = {
    "front": "front",
    "back": "back",
    "left": "out",
    "right": "in",
    "up": "up",
    "down": "down",
}


def _parse_gt_poses(groundtruth: str) -> list[tuple[str, str]]:
    return [(a.strip(), b.strip()) for a, b in re.findall(r"\(([^,]+),\s*([^)]+)\)", groundtruth)]


def _gt_by_cue() -> dict[str, dict[str, Any]]:
    if not GT_CONSOLIDATED.is_file():
        return {}
    data = json.loads(GT_CONSOLIDATED.read_text(encoding="utf-8"))
    return {str(r["cue"]): r for r in (data.get("rows") or []) if r.get("cue")}


def _cue_catalog() -> str:
    if not CUES_YAML.is_file():
        return ""
    data = yaml.safe_load(CUES_YAML.read_text(encoding="utf-8"))
    parts: list[str] = []
    for grp in ("iconic", "contextual"):
        block = data.get(grp)
        if not isinstance(block, dict):
            continue
        parts.append(f"[Available {grp} cues]")
        for cue, text in block.items():
            parts.append(f"- {cue}: {text}")
        parts.append("")
    return "\n".join(parts)


def _fewshot_block(max_examples: int = 6) -> str:
    from legacy.config_gen_single_mobile import _format_example_block, _load_json_list

    shots = _load_json_list(str(FEWSHOT_SHOTS)) if FEWSHOT_SHOTS.is_file() else []
    parts = [_format_example_block(sc) for sc in shots[:max_examples]]
    return "\n\n".join(parts)


def _llm_generate(prompt: str, *, model: str, vlm: Any | None = None) -> str:
    if vlm is not None:
        return vlm.generate(prompt)
    from google import genai

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(model=model, contents=prompt)
    return (resp.text or "").strip()


def _mobile_pose_from_gt_pair(d: str, g: str, ref: dict[str, Any] | None = None) -> dict[str, Any]:
    ref = ref or {}
    return {
        "torso_height": ref.get("torso_height", "mid"),
        "arm_position": DIR_TO_ARM.get(d.strip().lower(), d),
        "gripper_orientation": g,
        "head": ref.get("head", "center"),
        "left_arm": ref.get("left_arm", "still"),
        "x": ref.get("x", 50),
        "y": ref.get("y", 50),
        "z": ref.get("z", 50),
    }


def _fixed_pose_step(pose_fields: dict[str, Any], *, duration: float = 0.5) -> dict[str, Any]:
    return {
        "type": "pose",
        "duration": duration,
        "parameters": {"pose": pose_fields},
    }


def _strip_leading_poses(movements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = list(movements)
    while out and out[0].get("type") == "pose":
        out.pop(0)
    return out


def generate_exp1_row(
    *,
    cue: str,
    cue_idx: int,
    description: str,
    model: str,
    vlm: Any | None = None,
    out_path: Path | None = None,
    prompt_path: Path | None = None,
) -> dict[str, Any]:
    from legacy.config_gen_single_mobile import (
        _extract_reasoning_and_json,
        _sanitize_model_output,
        _validate_config,
        _validate_reasoning,
    )

    ppath = prompt_path or prompt_exp_path(1)
    template = ppath.read_text(encoding="utf-8")
    prompt = (
        template.replace("{{CUE_CATALOG}}", _cue_catalog())
        .replace("{{FEW_SHOT_EXAMPLES}}", _fewshot_block())
        .replace("{{CUE_NAME}}", cue)
    )
    validation_errors: list[str] = []
    reasoning_text = ""
    parsed: dict[str, Any] | None = None
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    max_attempts = 2

    for attempt in range(max_attempts):
        attempt_prompt = prompt
        if attempt > 0 and validation_errors:
            attempt_prompt += "\n\n# Fix these issues:\n" + "\n".join(
                f"# - {e}" for e in validation_errors
            )
        raw = _sanitize_model_output(_llm_generate(attempt_prompt, model=model, vlm=vlm))
        try:
            reasoning_text, parsed = _extract_reasoning_and_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            validation_errors = [f"JSON parse failed: {e}"]
            continue
        validation_errors = list(_validate_reasoning(reasoning_text))
        validation_errors.extend(_validate_config(parsed))
        if not validation_errors:
            break

    if parsed is None:
        raise ValueError(f"exp1 failed for {cue}: {validation_errors}")

    row = {
        "idx": cue_idx,
        "cue": cue,
        "description": parsed.get("description") or description,
        "movements": parsed.get("movements") or [],
        "state": "exp1_pose_generation",
        "model": model,
        "time": now,
        "experiment": "exp1",
        "generation_valid": True,
    }
    if reasoning_text:
        row["reasoning"] = reasoning_text
    if out_path is not None:
        upsert_config_row(out_path, row)
    return row


def generate_exp7_row(
    *,
    cue: str,
    cue_idx: int,
    description: str,
    fixed_pose: dict[str, Any],
    model: str,
    vlm: Any | None = None,
    out_path: Path | None = None,
    prompt_path: Path | None = None,
) -> dict[str, Any]:
    from legacy.config_gen_single_mobile import (
        _extract_reasoning_and_json,
        _sanitize_model_output,
        _validate_config,
        _validate_reasoning,
    )

    ppath = prompt_path or prompt_exp_path(7)
    template = ppath.read_text(encoding="utf-8")
    prompt = (
        template.replace("{{FEW_SHOT_EXAMPLES}}", _fewshot_block(max_examples=4))
        .replace("{{FIXED_FIRST_POSE_JSON}}", json.dumps(fixed_pose, ensure_ascii=False, indent=2))
        .replace("{{FIXED_START_POSE_JSON}}", json.dumps(fixed_pose, ensure_ascii=False, indent=2))
        .replace("{{CUE_NAME}}", cue)
        .replace("{{CUE_DESCRIPTION}}", description)
    )
    validation_errors: list[str] = []
    reasoning_text = ""
    tail: list[dict[str, Any]] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    max_attempts = 2

    for attempt in range(max_attempts):
        attempt_prompt = prompt
        if attempt > 0 and validation_errors:
            attempt_prompt += "\n\n# Fix these issues:\n" + "\n".join(
                f"# - {e}" for e in validation_errors
            )
        raw = _sanitize_model_output(_llm_generate(attempt_prompt, model=model, vlm=vlm))
        try:
            reasoning_text, parsed = _extract_reasoning_and_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            validation_errors = [f"JSON parse failed: {e}"]
            continue
        tail = parsed.get("movements") or parsed.get("movement_component") or []
        tail = _strip_leading_poses(tail)
        full = {
            "cue": cue,
            "movements": [_fixed_pose_step(fixed_pose)] + tail,
        }
        validation_errors = list(_validate_reasoning(reasoning_text))
        validation_errors.extend(_validate_config(full))
        if not tail:
            validation_errors.append("exp7 produced empty movement tail")
        if not validation_errors:
            break

    if validation_errors:
        raise ValueError(f"exp7 failed for {cue}: {validation_errors}")

    row = {
        "idx": cue_idx,
        "cue": cue,
        "description": description,
        "movements": [_fixed_pose_step(fixed_pose)] + tail,
        "state": "exp7_movement_generation",
        "model": model,
        "time": now,
        "experiment": "exp7",
        "generation_valid": True,
        "gt_fixed_first_pose": fixed_pose,
    }
    if reasoning_text:
        row["reasoning"] = reasoning_text
    if out_path is not None:
        upsert_config_row(out_path, row)
    return row


def run_exp_generation(
    exp_id: str,
    *,
    out_path: Path,
    model: str,
    backend: str = "gemini",
    vlm: Any | None = None,
    resume: bool = True,
    delay: float = 2.0,
) -> None:
    del backend
    shots = load_config_list(SHOTS)
    if not shots:
        raise FileNotFoundError(f"Missing shots list: {SHOTS}")
    gt = _gt_by_cue()
    existing = {r["cue"]: r for r in load_config_list(out_path) if r.get("cue")}
    catalog = yaml.safe_load(CUES_YAML.read_text(encoding="utf-8")) if CUES_YAML.is_file() else {}

    def _desc(cue: str) -> str:
        for grp in ("iconic", "contextual"):
            block = catalog.get(grp) if isinstance(catalog, dict) else None
            if isinstance(block, dict) and cue in block:
                return str(block[cue])
        row = next((r for r in shots if r.get("cue") == cue), {})
        return str(row.get("description") or row.get("cue_text") or "")

    for row in shots:
        cue = str(row.get("cue", ""))
        if not cue:
            continue
        if resume and row_generation_done(existing.get(cue)):
            continue
        cue_idx = int(row.get("idx", 0))
        description = _desc(cue)
        try:
            if exp_id == "1":
                generate_exp1_row(
                    cue=cue,
                    cue_idx=cue_idx,
                    description=description,
                    model=model,
                    vlm=vlm,
                    out_path=out_path,
                )
            elif exp_id == "7":
                ev = gt.get(cue) or {}
                gt_pairs = _parse_gt_poses(str(ev.get("groundtruth", "")))
                ref_pose = {}
                for step in row.get("movements") or []:
                    if step.get("type") == "pose":
                        ref_pose = (step.get("parameters") or {}).get("pose") or {}
                        break
                if gt_pairs:
                    d, g = gt_pairs[0]
                    fixed = _mobile_pose_from_gt_pair(d, g, ref_pose)
                else:
                    fixed = ref_pose or {"torso_height": "mid", "arm_position": "front", "gripper_orientation": "horizontal", "head": "center"}
                generate_exp7_row(
                    cue=cue,
                    cue_idx=cue_idx,
                    description=description,
                    fixed_pose=fixed,
                    model=model,
                    vlm=vlm,
                    out_path=out_path,
                )
            else:
                raise ValueError(f"Unsupported generation exp: {exp_id}")
            print(f"[gen] exp{exp_id} OK {cue}", flush=True)
        except Exception as e:
            print(f"[gen] exp{exp_id} FAIL {cue}: {e}", flush=True)
        time.sleep(delay)
