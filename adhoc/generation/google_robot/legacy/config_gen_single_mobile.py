#!/usr/bin/env python3
"""Generate one mobile-manipulator config for a single cue using Gemini.

Adapted from dev_locomotion/adhoc/locomotion/config_gen_single.py for
TIAGo mobile-manipulator configs (pose / movement / path schema).
"""
from __future__ import annotations

import json
import os
import re
import sys
import fire
import yaml
from datetime import datetime
from typing import Any

from google import genai
from google.genai import types

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from adhoc.utils.repo_paths import (  # noqa: E402
    motion_configs_results_dir,
    resolve_seed_prompt_txt,
    resolve_seed_shots_json,
    seed_yml_dir,
)

from adhoc.generation.joint_motion_schema import canonical_joint_keyword  # noqa: E402

_DEFAULT_PROMPT = str(resolve_seed_prompt_txt("google_robot"))
_DEFAULT_SHOTS = str(resolve_seed_shots_json("google_robot"))
_DEFAULT_CONFIG = str(motion_configs_results_dir("google_robot") / "motion_configs_19_mobile.json")
_DEFAULT_YAML = str(seed_yml_dir() / "cues_new.yml")


def _load_json_list(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"JSON root must be a list: {path}")
    return data


def _format_example_block(config: dict[str, Any]) -> str:
    parts: list[str] = []
    reasoning = config.get("reasoning")
    if reasoning:
        for line in str(reasoning).strip().splitlines():
            stripped = line.rstrip()
            if not stripped:
                continue
            parts.append(stripped if stripped.startswith("#") else f"# {stripped}")
    exclude = {"reasoning", "state", "model", "time", "group", "cue_text", "description"}
    json_config = {k: v for k, v in config.items() if k not in exclude}
    parts.append(json.dumps(json_config, indent=2, ensure_ascii=False))
    return "\n".join(parts)


def _sanitize_model_output(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n").lstrip("\ufeff")
    text = _CONTROL_CHAR_RE.sub("", text)
    return text.strip()


def _normalize_reasoning_text(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        lines.append(stripped if stripped.startswith("#") else f"# {stripped}")
    return "\n".join(lines)


def _parsed_object_score(obj: Any) -> int:
    if not isinstance(obj, dict):
        return -1
    score = 0
    if "movements" in obj:
        score += 8
    if "cue" in obj:
        score += 1
    return score


def _extract_reasoning_and_json(raw_text: str) -> tuple[str, dict[str, Any]]:
    text = _sanitize_model_output(raw_text)
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    decoder = json.JSONDecoder()
    candidates = [text]
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    parsed: list[tuple[int, int, str, dict[str, Any], str]] = []
    last_error = None

    for cand in candidates:
        for pos, char in enumerate(cand):
            if char != "{":
                continue
            try:
                obj, end = decoder.raw_decode(cand[pos:])
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if not isinstance(obj, dict):
                continue

            preamble = cand[:pos].strip()
            trailing = cand[pos + end:].strip()
            reasoning_parts = []
            if preamble:
                reasoning_parts.append(_normalize_reasoning_text(preamble))
            if trailing:
                reasoning_parts.append(_normalize_reasoning_text(trailing))
            reasoning_text = "\n".join(part for part in reasoning_parts if part)
            parsed.append((_parsed_object_score(obj), pos, cand, obj, reasoning_text))

    if parsed:
        parsed.sort(key=lambda item: (-item[0], item[1]))
        _, _, _, best_obj, best_reasoning = parsed[0]
        return best_reasoning, best_obj

    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON object found in model output", text, 0)


def _validate_reasoning(reasoning_text: str) -> list[str]:
    errors: list[str] = []
    if not reasoning_text.strip():
        return ["Missing planning comments before JSON"]
    lines = [line.strip() for line in reasoning_text.splitlines() if line.strip()]
    for expected in ("Q1", "Q2", "Q3", "Q4"):
        if not any(line.startswith(f"# {expected}:") for line in lines):
            errors.append(f"Missing '# {expected}:' planning line")
    return errors


def _validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    movements = config.get("movements", [])
    if not movements:
        errors.append("No movements defined")
        return errors

    step_types = [m.get("type") for m in movements]
    if step_types[0] != "pose":
        errors.append("First step must be type 'pose'")
    if all(t == "pose" for t in step_types):
        errors.append("Must include at least one 'movement', 'path', or 'pose_to_pose' step")
    if len(movements) < 2:
        errors.append(f"Only {len(movements)} step(s); need at least 2")
    if len(movements) > 10:
        errors.append(f"{len(movements)} steps exceeds max of 10")

    valid_types = {"pose", "movement", "path", "pose_to_pose"}
    valid_torso = {"low", "mid", "high"}
    valid_arm = {"front", "back", "in", "out", "up", "down", "still"}
    valid_grip = {"horizontal", "vertical"}
    valid_head = {"center", "left", "right", "up", "down"}
    valid_joints = {"shoulder", "elbow", "wrist", "torso", "head", "right_arm", "left_arm"}
    cartesian_axes = {"x", "y", "z"}
    valid_axes = {
        "shoulder": {"pitch", "roll"} | cartesian_axes,
        "elbow": {"pitch", "roll"} | cartesian_axes,
        "wrist": {"pitch", "roll"} | cartesian_axes,
        "torso": {"height"},
        "head": {"pan", "tilt"},
    }
    valid_path_shapes = {"line", "arc"}
    valid_left_arm = {"still", "mirror"} | valid_arm

    def _pose_field_errors(pose: dict, label: str) -> list[str]:
        pe: list[str] = []
        if pose.get("torso_height") and pose["torso_height"] not in valid_torso:
            pe.append(f"{label}: invalid torso_height {pose['torso_height']!r}")
        if pose.get("arm_position") and pose["arm_position"] not in valid_arm:
            pe.append(f"{label}: invalid arm_position {pose['arm_position']!r}")
        if pose.get("gripper_orientation") and pose["gripper_orientation"] not in valid_grip:
            pe.append(f"{label}: invalid gripper_orientation {pose['gripper_orientation']!r}")
        if pose.get("head") and pose["head"] not in valid_head:
            pe.append(f"{label}: invalid head {pose['head']!r}")
        la = pose.get("left_arm")
        if la is not None and la not in valid_left_arm:
            pe.append(f"{label}: invalid left_arm {la!r}")
        for coord in ("x", "y", "z"):
            if coord not in pose or pose[coord] is None:
                continue
            try:
                v = float(pose[coord])
            except (TypeError, ValueError):
                pe.append(f"{label}.{coord} must be a number in [0,100]")
                continue
            if not (0.0 <= v <= 100.0):
                pe.append(f"{label}.{coord} must be in [0,100], got {v}")
        return pe

    for m in movements:
        mtype = m.get("type")
        if mtype not in valid_types:
            errors.append(f"Invalid step type: {mtype}")
            continue
        params = m.get("parameters", {})

        if mtype == "pose":
            pose = params.get("pose", {})
            errors.extend(_pose_field_errors(pose, "pose"))

        elif mtype == "pose_to_pose":
            sp = params.get("start_pose")
            ep = params.get("end_pose")
            if not isinstance(sp, dict):
                errors.append("pose_to_pose requires parameters.start_pose (object)")
            else:
                errors.extend(_pose_field_errors(sp, "start_pose"))
            if not isinstance(ep, dict):
                errors.append("pose_to_pose requires parameters.end_pose (object)")
            else:
                errors.extend(_pose_field_errors(ep, "end_pose"))

        elif mtype == "movement":
            mv = params.get("movement", {})
            joints = mv.get("joints", [])
            if not joints:
                errors.append("Movement step has no joints")
            for j in joints:
                key = j.get("joint", "")
                cj = canonical_joint_keyword(key) or str(key).strip().lower()
                if cj == "base":
                    errors.append(
                        "joint 'base' is not supported in movement; use type: path with path.mode='base'"
                    )
                    continue
                if cj in ("right_arm", "left_arm"):
                    mshape = str(j.get("shape", "line")).lower()
                    if mshape == "arc":
                        if not j.get("plane"):
                            errors.append("right_arm/left_arm arc requires 'plane' (xy | xz | yz)")
                        if j.get("radius") is None:
                            errors.append("right_arm/left_arm arc requires 'radius' (m)")
                        if j.get("sweep", j.get("degrees")) is None:
                            errors.append("right_arm/left_arm arc requires 'sweep' or 'degrees'")
                        continue
                    axis = j.get("axis", "")
                    if not axis:
                        errors.append("right_arm / left_arm requires 'axis'")
                        continue
                    if "degrees" not in j:
                        errors.append(f"Missing 'degrees' for {key}.{axis}")
                    limb = (
                        str(j.get("link", "elbow" if axis in cartesian_axes else "shoulder")).strip().lower()
                    )
                    if limb not in valid_axes:
                        errors.append(f"Invalid link/group for arm alias: {limb}")
                    elif isinstance(axis, str) and axis not in valid_axes.get(limb, set()):
                        errors.append(f"Invalid axis '{axis}' for link '{limb}' on arm alias")
                    continue

                jname = str(key).strip().lower()
                if jname not in valid_joints:
                    errors.append(f"Invalid joint: {key}")
                    continue
                axis = j.get("axis", "")
                if axis not in valid_axes.get(jname, set()):
                    errors.append(f"Invalid axis '{axis}' for joint '{jname}'")
                if "degrees" not in j:
                    errors.append(f"Missing 'degrees' for {jname}.{axis}")

        elif mtype == "path":
            path = params.get("path", {})
            mode = path.get("mode")
            if mode is not None and mode not in {"ee", "base"}:
                errors.append(f"Invalid path mode: {mode!r} (use 'ee' or 'base')")
            shape = path.get("shape")
            if shape and shape not in valid_path_shapes:
                errors.append(f"Invalid path shape: {shape}")
            if mode == "ee":
                if shape == "line":
                    if path.get("axis") not in cartesian_axes:
                        errors.append("path.mode='ee' + line requires axis in {x,y,z}")
                    if path.get("distance") is None:
                        errors.append("path.mode='ee' + line requires distance (meters)")
                elif shape == "arc":
                    if path.get("plane") not in {"xy", "xz", "yz"}:
                        errors.append("path.mode='ee' + arc requires plane in {xy,xz,yz}")
                    if path.get("radius") is None:
                        errors.append("path.mode='ee' + arc requires radius (meters)")
            elif mode == "base":
                if shape == "line":
                    if path.get("x") is None and path.get("y") is None:
                        errors.append("path.mode='base' + line requires x and/or y (meters)")
                elif shape == "arc":
                    if path.get("degrees") is None:
                        errors.append("path.mode='base' + arc requires degrees")

    return errors


def _format_cue_catalog(yaml_path: str) -> str:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cues_dict = yaml.safe_load(f)
    parts: list[str] = []
    group_names = [g for g in ("iconic", "contextual") if g in cues_dict]
    if not group_names and isinstance(cues_dict, dict):
        group_names = [k for k, v in cues_dict.items() if isinstance(v, dict)]
    for group_name in group_names:
        grp = cues_dict.get(group_name)
        if not isinstance(grp, dict):
            continue
        parts.append(f"[Available {group_name} cues]")
        for cue, text in grp.items():
            parts.append(f"- {cue}: {text}")
        parts.append("")
    return "\n".join(parts)


def generate_motion_config(
    cue_name: str,
    cue_idx: int | None = None,
    model_name: str = "gemini-2.5-flash",
    prompt_file: str = _DEFAULT_PROMPT,
    shots_json: str = _DEFAULT_SHOTS,
    config_json: str = _DEFAULT_CONFIG,
    yaml_path: str = _DEFAULT_YAML,
    max_examples: int = 10,
    temperature: float | None = None,
    use_shots: bool = True,
    require_reasoning: bool = True,
):
    """Generate one mobile-manipulator config via Gemini and upsert into config_json."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY environment variable.")
    client = genai.Client(api_key=api_key)

    shot_configs = _load_json_list(shots_json) if use_shots else []
    generated_configs = _load_json_list(config_json)

    examples_str = ""
    if shot_configs:
        examples_parts = []
        for sc in shot_configs[:max_examples]:
            examples_parts.append(_format_example_block(sc))
        examples_str = "\n\n".join(examples_parts)
        print(f"  Using {len(shot_configs[:max_examples])} few-shot examples from {shots_json}")

    catalog_str = _format_cue_catalog(yaml_path)

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = prompt_template.replace("{{CUE_CATALOG}}", catalog_str)
    prompt = prompt.replace("{{FEW_SHOT_EXAMPLES}}", examples_str)
    prompt = prompt.replace("{{CUE_NAME}}", cue_name)

    print(f"Generating config for cue: '{cue_name}' (model={model_name})")

    max_attempts = 2
    new_config = None
    validation_errors: list[str] = []
    reasoning_text = ""

    for attempt in range(max_attempts):
        attempt_prompt = prompt
        if attempt > 0 and validation_errors:
            fix_msg = (
                "\n\n# IMPORTANT: Fix these problems:\n"
                + "\n".join(f"# - {e}" for e in validation_errors)
                + "\n# Regenerate the JSON now.\n"
            )
            attempt_prompt = prompt + fix_msg

        gen_config = None
        if temperature is not None:
            gen_config = types.GenerateContentConfig(temperature=float(temperature))
        response = client.models.generate_content(
            model=model_name,
            contents=attempt_prompt,
            config=gen_config,
        )
        raw_output = response.text.strip()

        try:
            reasoning_text, new_config = _extract_reasoning_and_json(raw_output)
        except (json.JSONDecodeError, ValueError):
            print(f"  Failed to parse JSON (attempt {attempt + 1})")
            if attempt == max_attempts - 1:
                raise ValueError(f"Failed to parse JSON for '{cue_name}'. Raw: {raw_output[:500]}")
            validation_errors = [
                "Output was not valid JSON",
                "Write '# Q1:', '# Q2:', '# Q3:' comment lines before the JSON",
                "After the planning comments, the next non-empty line must start with '{'",
            ]
            continue

        validation_errors = _validate_reasoning(reasoning_text) if require_reasoning else []
        validation_errors.extend(_validate_config(new_config))
        if not validation_errors:
            break
        print(f"  Validation issues (attempt {attempt + 1}): {validation_errors}")

    if new_config is None:
        raise ValueError(f"Failed to generate config for '{cue_name}' after retries.")
    if validation_errors:
        print(f"  Warning: {validation_errors}")

    existing = next(
        (c for c in generated_configs if c.get("cue") == cue_name and isinstance(c.get("idx"), int)),
        None,
    )
    if cue_idx is not None:
        target_idx = cue_idx
    elif existing is not None:
        target_idx = int(existing["idx"])
    else:
        target_idx = max((int(c.get("idx", -1)) for c in generated_configs), default=-1) + 1

    new_config["idx"] = target_idx
    new_config["cue"] = cue_name
    new_config["state"] = "generated"
    new_config["model"] = model_name
    new_config["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if reasoning_text:
        new_config["reasoning"] = reasoning_text
    if validation_errors:
        new_config["validation_warnings"] = validation_errors

    generated_configs = [
        cfg for cfg in generated_configs
        if cfg.get("idx") != target_idx and cfg.get("cue") != cue_name
    ]
    generated_configs.append(new_config)
    generated_configs.sort(key=lambda c: c.get("idx", 0))

    os.makedirs(os.path.dirname(config_json) or ".", exist_ok=True)
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(generated_configs, f, indent=2, ensure_ascii=False)

    print(f"  Saved '{cue_name}' to {config_json} (idx={target_idx})")


if __name__ == "__main__":
    fire.Fire(generate_motion_config)
