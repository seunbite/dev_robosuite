import copy
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List

import fire
from google import genai
from google.genai import types


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


PERSONA_EDIT_PROMPT = """You are an expert in expressive robot motion editing.

Your job is NOT to invent a new motion from scratch.
Your job is to take an existing motion config and apply a subtle persona-specific variation while preserving the original cue identity.

[Goal]
Keep the same cue, the same core primitive sequence, and the same recognizable meaning.
Only edit motion parameters to inject persona.

[Persona]
Name: {persona_name}
Description: {persona_description}
Edit Strength: {edit_strength}

[Allowed Edits]
You may make only small or moderate edits to:
- pose: x, y, z, dir, gripper_orientation, speed, hold_time
- movement: repetition, joint, direction degrees, speed, hold_time
- path: shape, joint, speed, and path-specific parameters

[Preferred Edit Style]
- Keep the same number of steps whenever possible.
- Keep the same step order and primitive types whenever possible.
- Keep the same main joint focus unless changing it is clearly necessary for persona readability.
- Prefer parameter edits over structural edits.
- Use subtle positional noise or timing variation when it helps convey persona.
- Preserve iconic readability of the original cue above all else.

[Hard Constraints]
1. The output must preserve the same cue name.
2. The first step must remain `pose`.
3. Do not output a pose-only sequence.
4. Keep total steps between 2 and 8.
5. Keep all speed values between 0.5 and 4.0.
6. Keep pose x, y, z in the range 0 to 100.
7. Prefer keeping the same primitive sequence and only changing parameters.
8. Do not add filler settle / return-to-rest steps.
9. Output reasoning first as `#` comment lines, then exactly one JSON object.
10. Do not use markdown code blocks.

[Planning Format]
Write exactly three comment lines before the JSON:
# Q1: What persona qualities should be expressed in this cue?
# Q2: Which parameters of the existing motion should change the most, and which should stay stable to preserve cue identity?
# Q3: What final edits best inject the persona while keeping the motion clearly recognizable?

[Output Requirements]
- The JSON object must preserve the cue and remain executable in the same motion schema.
- The JSON should contain the updated `cue`, `description`, and `movements`.
- Do not include extra prose after the JSON.

[Base Motion Config]
{base_config_json}
"""


def _sanitize_model_output(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n").lstrip("\ufeff")
    text = _CONTROL_CHAR_RE.sub("", text)
    return text.strip()


def _normalize_reasoning_text(text: str) -> str:
    lines: List[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        if stripped.startswith("#"):
            lines.append(stripped)
        else:
            lines.append(f"# {stripped}")
    return "\n".join(lines)


def _extract_reasoning_and_json(raw_text: str) -> tuple[str, Dict[str, Any]]:
    text = _sanitize_model_output(raw_text)
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    decoder = json.JSONDecoder()
    candidates = [text]
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    parsed_candidates: list[tuple[int, int, Dict[str, Any], str]] = []
    last_error = None
    for candidate_text in candidates:
        for pos, char in enumerate(candidate_text):
            if char != "{":
                continue
            try:
                obj, end = decoder.raw_decode(candidate_text[pos:])
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if not isinstance(obj, dict):
                continue
            preamble = candidate_text[:pos].strip()
            trailing = candidate_text[pos + end :].strip()
            reasoning_parts = []
            if preamble:
                reasoning_parts.append(_normalize_reasoning_text(preamble))
            if trailing:
                reasoning_parts.append(_normalize_reasoning_text(trailing))
            parsed_candidates.append(
                (
                    1 if "movements" in obj else 0,
                    pos,
                    obj,
                    "\n".join(part for part in reasoning_parts if part),
                )
            )

    if parsed_candidates:
        parsed_candidates.sort(key=lambda item: (-item[0], item[1]))
        _, _, best_obj, best_reasoning = parsed_candidates[0]
        return best_reasoning, best_obj

    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON object found in model output", text, 0)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _validate_reasoning(reasoning_text: str) -> List[str]:
    lines = [line.strip() for line in reasoning_text.splitlines() if line.strip()]
    errors: List[str] = []
    if len(lines) != 3:
        errors.append(f"Expected exactly 3 planning lines before JSON, got {len(lines)}")
    for expected in ("Q1", "Q2", "Q3"):
        if not any(line.startswith(f"# {expected}:") for line in lines):
            errors.append(f"Missing '# {expected}:' planning line")
    return errors


def _validate_edited_config(base_cfg: Dict[str, Any], edited_cfg: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    base_movements = base_cfg.get("movements", [])
    edited_movements = edited_cfg.get("movements", [])

    if edited_cfg.get("cue") != base_cfg.get("cue"):
        errors.append("Edited config must preserve the same cue name")

    if not edited_movements:
        errors.append("Edited config has no movements")
        return errors

    edited_types = [m.get("type") for m in edited_movements]
    if edited_types[0] != "pose":
        errors.append("First step must remain 'pose'")
    if all(t == "pose" for t in edited_types):
        errors.append("Edited config cannot be pose-only")
    if not (2 <= len(edited_movements) <= 8):
        errors.append(f"Edited config has {len(edited_movements)} steps; expected 2-8")

    if len(base_movements) != len(edited_movements):
        errors.append("Edited config changed the number of steps; preserve the base structure")

    for idx, (base_step, edited_step) in enumerate(zip(base_movements, edited_movements)):
        base_type = base_step.get("type")
        edited_type = edited_step.get("type")
        if base_type != edited_type:
            errors.append(f"Step {idx} changed primitive type from {base_type} to {edited_type}")
            continue

        params = edited_step.get("parameters", {})
        if base_type == "pose":
            pose = params.get("pose", {})
            for axis in ("x", "y", "z"):
                if axis in pose and not (0 <= float(pose[axis]) <= 100):
                    errors.append(f"Pose {axis}={pose[axis]} is outside [0, 100]")
            step_speed = params.get("speed")
            if isinstance(step_speed, (int, float)) and not (0.5 <= float(step_speed) <= 4.0):
                errors.append(f"Pose speed {step_speed} is outside [0.5, 4.0]")
            hold_time = params.get("hold_time")
            if isinstance(hold_time, (int, float)) and hold_time < 0:
                errors.append(f"Pose hold_time {hold_time} must be non-negative")

        elif base_type == "movement":
            repetition = params.get("repetition")
            if isinstance(repetition, (int, float)) and repetition < 1:
                errors.append(f"Movement repetition {repetition} must be >= 1")
            for direction in params.get("directions", []):
                direction_speed = direction.get("speed")
                if isinstance(direction_speed, (int, float)) and not (0.5 <= float(direction_speed) <= 4.0):
                    errors.append(f"Movement speed {direction_speed} is outside [0.5, 4.0]")
                hold_time = direction.get("hold_time")
                if isinstance(hold_time, (int, float)) and hold_time < 0:
                    errors.append(f"Movement hold_time {hold_time} must be non-negative")
                for axis, val in direction.get("degrees", {}).items():
                    if abs(float(val)) > 65:
                        errors.append(f"Movement degrees {axis}={val} is too extreme")

        elif base_type == "path":
            step_speed = params.get("speed")
            if isinstance(step_speed, (int, float)) and not (0.5 <= float(step_speed) <= 4.0):
                errors.append(f"Path speed {step_speed} is outside [0.5, 4.0]")

    return errors


def _postprocess_edited_config(base_cfg: Dict[str, Any], edited_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(edited_cfg)
    cfg["cue"] = base_cfg.get("cue")

    for step in cfg.get("movements", []):
        params = step.setdefault("parameters", {})
        mtype = step.get("type")

        if mtype == "pose":
            pose = params.setdefault("pose", {})
            for axis in ("x", "y", "z"):
                if axis in pose:
                    pose[axis] = int(round(_clamp(float(pose[axis]), 0, 100)))
            if "speed" in params:
                params["speed"] = round(_clamp(float(params["speed"]), 0.5, 4.0), 3)
            if "hold_time" in params:
                params["hold_time"] = round(max(0.0, float(params["hold_time"])), 3)

        elif mtype == "movement":
            if "repetition" in params:
                params["repetition"] = max(1, int(round(float(params["repetition"]))))
            for direction in params.get("directions", []):
                if "speed" in direction:
                    direction["speed"] = round(_clamp(float(direction["speed"]), 0.5, 4.0), 3)
                if "hold_time" in direction:
                    direction["hold_time"] = round(max(0.0, float(direction["hold_time"])), 3)
                if "degrees" in direction:
                    direction["degrees"] = {
                        axis: round(_clamp(float(val), -65, 65), 3)
                        for axis, val in direction["degrees"].items()
                    }

        elif mtype == "path":
            if "speed" in params:
                params["speed"] = round(_clamp(float(params["speed"]), 0.5, 4.0), 3)
            for key in ("distance", "radius", "sweep"):
                if key in params and isinstance(params[key], (int, float)):
                    params[key] = round(float(params[key]), 3)

    return cfg


def _load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"JSON root must be a list: {path}")
    return data


def _select_configs(data: List[Dict[str, Any]], cue_idxs: List[int] | None) -> List[Dict[str, Any]]:
    if cue_idxs is None:
        return [cfg for cfg in data if isinstance(cfg, dict) and isinstance(cfg.get("idx"), int)]
    allowed = set(int(idx) for idx in cue_idxs)
    return [cfg for cfg in data if isinstance(cfg, dict) and cfg.get("idx") in allowed]


def apply_persona_to_config(
    base_cfg: Dict[str, Any],
    *,
    persona_name: str,
    persona_description: str,
    client: genai.Client,
    model_name: str,
    edit_strength: str = "subtle",
    temperature: float = 0.3,
    max_attempts: int = 2,
) -> Dict[str, Any]:
    prompt = PERSONA_EDIT_PROMPT.format(
        persona_name=persona_name,
        persona_description=persona_description,
        edit_strength=edit_strength,
        base_config_json=json.dumps(base_cfg, indent=2, ensure_ascii=False),
    )

    validation_errors: List[str] = []
    reasoning_text = ""
    edited_cfg: Dict[str, Any] | None = None

    for attempt in range(max_attempts):
        attempt_prompt = prompt
        if attempt > 0 and validation_errors:
            fix_msg = (
                "\n\n# IMPORTANT: Your previous edit had these problems. Fix them:\n"
                + "\n".join(f"# - {e}" for e in validation_errors)
                + "\n# Keep the same cue and preserve the primitive sequence.\n"
            )
            attempt_prompt = prompt + fix_msg

        response = client.models.generate_content(
            model=model_name,
            contents=attempt_prompt,
            config=types.GenerateContentConfig(temperature=float(temperature)),
        )
        raw_output = response.text.strip()
        reasoning_text, candidate_cfg = _extract_reasoning_and_json(raw_output)
        candidate_cfg = _postprocess_edited_config(base_cfg, candidate_cfg)

        validation_errors = _validate_reasoning(reasoning_text)
        validation_errors.extend(_validate_edited_config(base_cfg, candidate_cfg))
        if not validation_errors:
            edited_cfg = candidate_cfg
            break

    if edited_cfg is None:
        raise ValueError(
            f"Failed persona edit for cue '{base_cfg.get('cue')}' after {max_attempts} attempts: "
            + "; ".join(validation_errors)
        )

    result = copy.deepcopy(base_cfg)
    result.update({
        "description": edited_cfg.get("description", base_cfg.get("description", "")),
        "movements": edited_cfg["movements"],
        "persona_name": persona_name,
        "persona_description": persona_description,
        "persona_edit_strength": edit_strength,
        "persona_model": model_name,
        "persona_temperature": temperature,
        "persona_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "persona_reasoning": reasoning_text,
        "base_config_snapshot": {
            "description": base_cfg.get("description", ""),
            "movements": base_cfg.get("movements", []),
        },
    })
    return result


def main(
    input_json: str,
    output_json: str | None = None,
    persona_name: str = "calm",
    persona_description: str = "A calm, restrained, smooth, deliberate personality with gentle timing changes.",
    cue_idxs: List[int] | None = None,
    model_name: str = "gemini-3.1-flash-lite-preview",
    edit_strength: str = "subtle",
    temperature: float = 0.3,
    max_attempts: int = 2,
):
    """Apply persona-specific parameter edits to existing motion configs using an LLM."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable.")

    data = _load_json_list(input_json)
    selected = _select_configs(data, cue_idxs)
    if not selected:
        raise ValueError("No configs selected for persona editing.")

    client = genai.Client(api_key=api_key)
    edited_by_idx: Dict[int, Dict[str, Any]] = {}

    print(f"input={input_json}")
    print(f"persona={persona_name} strength={edit_strength} model={model_name} temperature={temperature}")
    print(f"selected_cues={len(selected)}")

    for cfg in selected:
        cue = cfg.get("cue", "<unknown>")
        idx = cfg.get("idx")
        print(f"\nEditing c{idx}: {cue}")
        edited = apply_persona_to_config(
            cfg,
            persona_name=persona_name,
            persona_description=persona_description,
            client=client,
            model_name=model_name,
            edit_strength=edit_strength,
            temperature=temperature,
            max_attempts=max_attempts,
        )
        edited_by_idx[idx] = edited
        print(f"  done: c{idx}")

    merged: List[Dict[str, Any]] = []
    for cfg in data:
        idx = cfg.get("idx")
        if idx in edited_by_idx:
            merged.append(edited_by_idx[idx])
        else:
            merged.append(cfg)

    if output_json is None:
        stem, ext = os.path.splitext(input_json)
        suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", persona_name.strip()) or "persona"
        output_json = f"{stem}_{suffix}{ext}"

    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\nSaved persona-edited configs to: {output_json}")


if __name__ == "__main__":
    fire.Fire(main)
