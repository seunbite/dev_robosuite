import os
import json
import re
import fire
from google import genai
from google.genai import types
from typing import Any, Dict, List
from datetime import datetime

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _create_default_shot_configs(shot_json: str) -> None:
    """Create a minimal shot config file when missing."""
    os.makedirs(os.path.dirname(shot_json), exist_ok=True)
    default_example = [
        {
            "idx": 0,
            "cue": "wave",
            "description": "Raise arm and wave wrist side to side.",
            "movements": [
                {
                    "type": "pose",
                    "parameters": {
                        "pose": {"dir": "front", "z": "high"},
                        "speed": 1.0,
                        "hold_time": 0.3
                    }
                },
                {
                    "type": "movement",
                    "parameters": {
                        "repetition": 2,
                        "axis": "y",
                        "joint": "wrist",
                        "directions": [
                            {"degrees": 20, "speed": 1.2, "hold_time": 0.1, "sign": "positive"},
                            {"degrees": 20, "speed": 1.2, "hold_time": 0.1, "sign": "negative"}
                        ]
                    }
                }
            ],
            "state": "handmade",
            "time": "1970-01-01 00:00:00"
        }
    ]
    with open(shot_json, "w", encoding="utf-8") as f:
        json.dump(default_example, f, indent=2, ensure_ascii=False)
    print(f"Created default shot config file: {shot_json}")


def _load_json_list(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"JSON root must be a list: {path}")
    return data


def _resolve_shots_json(prompt_file: str, shots_json: str) -> str:
    """Use version-matched shot configs when available."""
    if shots_json != "data/seed/shot_configs.json":
        return shots_json
    prompt_name = os.path.basename(prompt_file)
    if prompt_name.startswith("prompt_v") and prompt_name.endswith(".txt"):
        version = prompt_name.replace("prompt_v", "").replace(".txt", "")
        candidate = os.path.join(os.path.dirname(os.path.dirname(prompt_file)), f"shot_configs_v{version}.json")
        if os.path.exists(candidate):
            return candidate
    return shots_json


def _format_example_block(config: Dict[str, Any], prefix: str | None = None) -> str:
    """Format a shot config with optional planning-shot commentary."""
    parts: List[str] = []
    reasoning = config.get("reasoning")
    if reasoning:
        parts.append("# EXAMPLE REASONING:")
        for line in str(reasoning).strip().splitlines():
            stripped = line.rstrip()
            if not stripped:
                continue
            parts.append(stripped if stripped.startswith("#") else f"# {stripped}")
    planning_shot = config.get("planning_shot")
    if planning_shot:
        parts.append("# PLANNING SHOT (how to reason before writing JSON):")
        for line in str(planning_shot).strip().splitlines():
            parts.append(f"# {line.rstrip()}")
    if prefix:
        parts.append(prefix)
    json_config = {k: v for k, v in config.items() if k not in {"planning_shot", "reasoning"}}
    parts.append(json.dumps(json_config, indent=2, ensure_ascii=False))
    return "\n".join(parts)


def _normalize_reasoning_text(text: str) -> str:
    """Normalize free-form preamble text into comment-style reasoning lines."""
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


def _sanitize_model_output(raw_text: str) -> str:
    """Remove control characters and normalize line endings before parsing."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n").lstrip("\ufeff")
    text = _CONTROL_CHAR_RE.sub("", text)
    return text.strip()


def _parsed_object_score(obj: Any) -> int:
    if not isinstance(obj, dict):
        return -1
    score = 0
    if "movements" in obj:
        score += 8
    if "description" in obj:
        score += 2
    if "cue" in obj:
        score += 1
    return score


def _extract_reasoning_and_json(raw_text: str) -> tuple[str, Dict[str, Any]]:
    """Extract reasoning text and the first valid top-level JSON object from model output."""
    text = _sanitize_model_output(raw_text)

    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    decoder = json.JSONDecoder()
    candidates = [text]
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    parsed_candidates: list[tuple[int, int, str, Dict[str, Any], str]] = []
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
            reasoning_text = "\n".join(part for part in reasoning_parts if part)
            parsed_candidates.append(
                (_parsed_object_score(obj), pos, candidate_text, obj, reasoning_text)
            )

    if parsed_candidates:
        parsed_candidates.sort(key=lambda item: (-item[0], item[1]))
        _, _, _, best_obj, best_reasoning = parsed_candidates[0]
        return best_reasoning, best_obj

    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON object found in model output", text, 0)


def _validate_reasoning(reasoning_text: str) -> List[str]:
    """Validate the required planning-comment format for the current prompt style."""
    errors: List[str] = []
    if not reasoning_text.strip():
        return ["Missing planning comments before JSON"]

    lines = [line.strip() for line in reasoning_text.splitlines() if line.strip()]
    if len(lines) not in (3, 4):
        errors.append(f"Expected exactly 3 or 4 planning lines before JSON, got {len(lines)}")

    q_map = {}
    for expected in ("Q1", "Q2", "Q3"):
        match_line = next((line for line in lines if line.startswith(f"# {expected}:")), None)
        if match_line is None:
            errors.append(f"Missing '# {expected}:' planning line")
            continue
        q_map[expected] = match_line

    q2 = q_map.get("Q2", "")
    q3 = q_map.get("Q3", "")

    if q2 and "winner=" not in q2.lower():
        errors.append("Q2 must compare multiple initial pose candidates and include a winner")
    if q2 and not re.search(r"candidates?\s*=", q2, flags=re.IGNORECASE):
        errors.append("Q2 must include ranked initial pose candidates")
    if q3 and "winner=" not in q3.lower():
        errors.append("Q3 must compare multiple step structures and include a winner")
    if q3 and not re.search(r"options?\s*=", q3, flags=re.IGNORECASE):
        errors.append("Q3 must include multiple candidate motion structures")

    q4 = next((line for line in lines if line.startswith("# Q4:")), "")
    if q4:
        if "winner=" not in q4.lower():
            errors.append("Q4 must compare multiple context-reinforcement options and include a winner")
        if not re.search(r"options?\s*=", q4, flags=re.IGNORECASE):
            errors.append("Q4 must include multiple contextual micro-action options")

    return errors


def _validate_config(config: Dict[str, Any], cue_name: str | None = None) -> List[str]:
    """Validate a generated motion config. Returns list of error strings (empty = valid)."""
    errors = []
    movements = config.get("movements", [])

    if not movements:
        errors.append("No movements defined")
        return errors

    types = [m.get("type") for m in movements]

    if types[0] != "pose":
        errors.append("First step must be type 'pose'")

    if all(t == "pose" for t in types):
        errors.append("Config is pose-only; must include at least one 'movement' or 'path' step")

    if len(movements) < 2:
        errors.append(f"Only {len(movements)} step(s); need at least 2")

    if len(movements) > 10:
        errors.append(f"{len(movements)} steps exceeds recommended max of 10")

    for m in movements:
        m_type = m.get("type")
        params = m.get("parameters", {})

        step_speed = params.get("speed")
        if isinstance(step_speed, (int, float)) and not (0.5 <= float(step_speed) <= 4.0):
            errors.append(f"{m_type} speed {step_speed} is outside allowed range [0.5, 4.0]")

        if m_type == "pose":
            hold_time = params.get("hold_time")
            if isinstance(hold_time, (int, float)) and hold_time < 0:
                errors.append(f"pose hold_time {hold_time} must be non-negative")

        if m.get("type") == "movement":
            for d in params.get("directions", []):
                direction_speed = d.get("speed")
                if isinstance(direction_speed, (int, float)) and not (0.5 <= float(direction_speed) <= 4.0):
                    errors.append(f"movement speed {direction_speed} is outside allowed range [0.5, 4.0]")

                hold_time = d.get("hold_time")
                if isinstance(hold_time, (int, float)) and hold_time < 0:
                    errors.append(f"movement hold_time {hold_time} must be non-negative")

                for axis, val in d.get("degrees", {}).items():
                    if abs(val) > 55:
                        errors.append(f"Extreme degree value {axis}={val} (max ±50 recommended)")

    target_cue = cue_name or config.get("cue", "")
    if "Beckon" in target_cue or "Come here" in target_cue:
        first_pose = next((m for m in movements if m.get("type") == "pose"), None)
        orientation = None
        if first_pose:
            orientation = first_pose.get("parameters", {}).get("pose", {}).get("gripper_orientation")
        if orientation not in ("horizontal", "vertical"):
            errors.append("Beckon cue should specify first-pose gripper_orientation as horizontal or vertical")

        movement_steps = [m for m in movements if m.get("type") == "movement"]
        repetitions = [m.get("parameters", {}).get("repetition", 1) for m in movement_steps]
        max_rep = max(repetitions, default=1)
        if not (3 <= max_rep <= 5):
            errors.append("Beckon cue should have a clear repeated core action with repetition 3-5")

        repeated_axes = set()
        for m in movement_steps:
            rep = m.get("parameters", {}).get("repetition", 1)
            if rep >= 3:
                for d in m.get("parameters", {}).get("directions", []):
                    degrees = d.get("degrees", {})
                    if isinstance(degrees, dict):
                        repeated_axes.update(degrees.keys())

        if orientation == "vertical" and "x" not in repeated_axes:
            errors.append("Beckon cue with vertical hand should use x-axis in the repeated beckon motion")
        if orientation == "horizontal" and "z" not in repeated_axes:
            errors.append("Beckon cue with horizontal hand should use z-axis in the repeated beckon motion")

    return errors


def generate_motion_config(
    cue_name: str,
    cue_idx: int | None = None,
    model_name: str = "gemini-3.1-flash-lite-preview",
    prompt_file: str = "data/seed/prompt/prompt_v1.txt",
    shots_json: str = "data/seed/shot_configs.json",
    config_json: str = "data/seed/motion_configs.json", # where to save
    max_handmade_examples: int = 10,
    max_correction_examples: int = 10,
    temperature: float | None = None,
    use_shots: bool = True,
    require_reasoning: bool = True,
):
    """
    Generate one few-shot motion config using Gemini.
    
    Args:
        cue_name: Name of the cue to generate
        shots_json: Path to shot examples JSON file (few-shot source)
        config_json: Path to generated motion config JSON file (few-shot output)
        prompt_file: Path to the prompt template
        model_name: Gemini model to use
        max_handmade_examples: Max handmade shot examples to include in prompt
        max_correction_examples: Max correction pairs to include in prompt
    """
    # 1. Setup Gemini API
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable.")
    client = genai.Client(api_key=api_key)

    shots_json = _resolve_shots_json(prompt_file, shots_json)

    # 2. Load shot configs (few-shot source) and generated configs (output target)
    if not os.path.exists(shots_json):
        _create_default_shot_configs(shots_json)
    shot_configs = _load_json_list(shots_json)
    generated_configs = _load_json_list(config_json)

    selected_handmade: List[Dict[str, Any]] = []
    choreography_configs: List[Dict[str, Any]] = []
    selected_pairs: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
    examples_str = ""

    if use_shots:
        # Build few-shot with handmade + before/after correction pairs from shot configs only
        handmade_configs = [c for c in shot_configs if c.get("state") == "handmade"]
        choreography_configs = [c for c in shot_configs if c.get("state") == "choreography"]
        corrected_configs = [c for c in shot_configs if c.get("state") == "corrected"]

        # Find correction pairs (fewshot/zeroshot -> corrected) in shot set
        correction_pairs = []
        for corrected in corrected_configs:
            cue = corrected.get("cue")
            if not cue:
                continue

            bad_example = next(
                (c for c in shot_configs if c.get("cue") == cue and c.get("state") == "fewshot"),
                None,
            )
            if not bad_example:
                bad_example = next(
                    (c for c in shot_configs if c.get("cue") == cue and c.get("state") == "zeroshot"),
                    None,
                )
            if bad_example:
                correction_pairs.append((bad_example, corrected))

        examples_parts: List[str] = []

        # Add handmade examples (good examples)
        num_handmade = min(max_handmade_examples, len(handmade_configs))
        import random

        selected_handmade = random.sample(handmade_configs, num_handmade) if handmade_configs else []
        for hm in selected_handmade:
            examples_parts.append(_format_example_block(hm))

        # Add choreography examples (complex multi-phase good examples)
        for ch in choreography_configs:
            examples_parts.append(_format_example_block(
                ch,
                "# CHOREOGRAPHY EXAMPLE (preferred multi-phase style with Setup → Core → Follow-through):"
            ))

        # Add correction pairs (bad -> good examples)
        num_pairs = min(max_correction_examples, len(correction_pairs))
        selected_pairs = correction_pairs[:num_pairs]
        for bad_ex, good_ex in selected_pairs:
            examples_parts.append(_format_example_block(
                bad_ex,
                "# BAD EXAMPLE (avoid this pattern):"
            ))
            examples_parts.append(_format_example_block(
                good_ex,
                "# GOOD EXAMPLE (corrected version):"
            ))

        examples_str = "\n\n".join(examples_parts)

    print(f"\nUsing few-shot examples from {shots_json}:")
    if use_shots:
        print(f"  - {len(selected_handmade)} handmade (good examples)")
        print(f"  - {len(choreography_configs)} choreography (complex examples)")
        print(f"  - {len(selected_pairs)} correction pairs (bad -> good)")
    else:
        print("  - disabled")

    # 3. Read prompt template
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # Fill placeholders
    prompt = prompt_template.replace("{{FEW_SHOT_EXAMPLES}}", examples_str)
    prompt = prompt.replace("{{CUE_NAME}}", cue_name)

    print(f"Generating config for cue: '{cue_name}'...")
    print(f"  model={model_name} temperature={temperature if temperature is not None else 'default'}")
    
    # 4. Inference with validation + retry
    max_attempts = 2
    new_config = None
    validation_errors: List[str] = []
    reasoning_text = ""

    for attempt in range(max_attempts):
        attempt_prompt = prompt
        if attempt > 0 and validation_errors:
            fix_msg = (
                "\n\n# IMPORTANT: Your previous output had these problems. Fix them:\n"
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
            print(f"Error: Failed to parse JSON (attempt {attempt+1})")
            if attempt == max_attempts - 1:
                raise ValueError(f"Failed to parse JSON for cue '{cue_name}'. Raw output: {raw_output[:500]}")
            validation_errors = [
                "Output was not valid JSON",
                "Write only '# Q1', '# Q2', '# Q3' comment lines before the JSON, and '# Q4' only if requested",
                "After the planning comments, the next non-empty line must start with '{'",
            ]
            continue

        validation_errors = _validate_reasoning(reasoning_text) if require_reasoning else []
        validation_errors.extend(_validate_config(new_config, cue_name=cue_name))
        if not validation_errors:
            break
        print(f"Validation issues (attempt {attempt+1}): {validation_errors}")

    if new_config is None:
        raise ValueError(f"Failed to generate config for cue '{cue_name}' after retries.")
    if validation_errors:
        raise ValueError(
            f"Failed to generate a valid config for cue '{cue_name}' after {max_attempts} attempts: "
            + "; ".join(validation_errors)
        )

    # 5. Upsert generated config in config_json
    existing_same_cue = next(
        (c for c in generated_configs if c.get("cue") == cue_name and isinstance(c.get("idx"), int)),
        None,
    )
    if cue_idx is not None:
        target_idx = cue_idx
    elif existing_same_cue is not None:
        target_idx = int(existing_same_cue["idx"])
    else:
        target_idx = max((int(c.get("idx", -1)) for c in generated_configs), default=-1) + 1

    new_config["idx"] = target_idx
    new_config["cue"] = cue_name
    new_config["state"] = "fewshot"
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

    os.makedirs(os.path.dirname(config_json), exist_ok=True)
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(generated_configs, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved '{cue_name}' to {config_json} (Index: {target_idx})")

if __name__ == "__main__":
    fire.Fire(generate_motion_config)
