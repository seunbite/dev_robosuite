import os
import json
import re
import sys
from pathlib import Path
import fire
from google import genai
from google.genai import types
from typing import Any, Dict, List
from datetime import datetime

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adhoc.utils.repo_paths import (  # noqa: E402
    infer_robot_from_prompt_path,
    resolve_seed_prompt_txt,
    resolve_seed_shots_json,
)

_DEFAULT_MANIP_PROMPT = str(resolve_seed_prompt_txt("manipulator"))

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
    """Single canonical ``shots.json`` (or newest ``shot*.json``) per robot under ``seed/shots/<robot>/``."""
    if shots_json and shots_json != "auto":
        return shots_json
    robot = infer_robot_from_prompt_path(prompt_file)
    if robot:
        return str(resolve_seed_shots_json(robot))
    return str(resolve_seed_shots_json("manipulator"))


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
    if "movements" in obj or "sequence" in obj:
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


def _coerce_step_parameters(step: Dict[str, Any]) -> Dict[str, Any]:
    """Hoist flat / alternate-schema fields into parameters (Gemini essence outputs)."""
    step = dict(step)
    step.pop("name", None)
    if step.get("parameters"):
        return step

    stype = step.get("type")
    params: Dict[str, Any] = {}

    if stype == "pose":
        pose: Dict[str, Any] = {}
        for k in ("dir", "gripper_orientation", "x", "y", "z", "pose_id"):
            if k in step:
                pose[k] = step.pop(k)
        loc = step.pop("location", None)
        if loc is not None:
            if isinstance(loc, dict):
                pose["x"] = int(float(loc.get("x", 0.55)) * 100) if float(loc.get("x", 0.55)) <= 1.5 else int(loc["x"])
                pose["y"] = int(float(loc.get("y", 0.5)) * 100) if abs(float(loc.get("y", 0.5))) <= 1.5 else int(loc["y"])
                pose["z"] = int(float(loc.get("z", 0.55)) * 100) if float(loc.get("z", 0.55)) <= 1.5 else int(loc["z"])
            elif isinstance(loc, (list, tuple)) and len(loc) >= 3:
                pose["x"] = min(90, max(35, int(abs(float(loc[0])) * 100)))
                pose["y"] = min(70, max(30, int((float(loc[1]) + 0.5) * 50)))
                pose["z"] = min(85, max(35, int(float(loc[2]) * 100)))
        pose.setdefault("dir", "front")
        pose.setdefault("gripper_orientation", "vertical")
        pose.setdefault("x", 55)
        pose.setdefault("y", 50)
        pose.setdefault("z", 55)
        params["pose"] = pose
        if "speed" in step:
            params["speed"] = step.pop("speed")
        if "hold_time" in step:
            params["hold_time"] = step.pop("hold_time")
        elif "duration" in step:
            params["hold_time"] = step.pop("duration")

    elif stype == "movement":
        if step.get("action") == "pause" or (step.get("axis") is None and step.get("joint") is None and "duration" in step):
            hold = float(step.pop("duration", step.pop("hold_time", 0.3)))
            step.pop("action", None)
            params = {
                "joint": "wrist",
                "directions": [{"degrees": {"z": 0}, "speed": 0.5, "hold_time": hold}],
            }
        elif "axis" in step and "distance" in step:
            step["type"] = "path"
            params = {
                "shape": "line",
                "axis": step.pop("axis"),
                "distance": abs(float(step.pop("distance"))),
                "speed": float(step.pop("speed", 1.5)),
            }
            step.pop("duration", None)
        else:
            for k in list(step.keys()):
                if k not in ("type", "parameters"):
                    params[k] = step.pop(k)

    elif stype == "path":
        for k in list(step.keys()):
            if k not in ("type", "parameters"):
                params[k] = step.pop(k)

    if params:
        step["parameters"] = params
    return step


def _normalize_motion_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize generated configs (e.g. strip legacy path.joint when EE fields exist)."""
    from legacy.path_ee_ik import normalize_path_parameters

    out = dict(config)
    raw = config.get("movements") or config.get("sequence") or []
    movements = []
    for step in raw:
        if not isinstance(step, dict):
            continue
        step = _coerce_step_parameters(step)
        if step.get("type") == "path":
            params = dict(step.get("parameters") or {})
            params = normalize_path_parameters(params)
            step["parameters"] = params
        movements.append(step)
    out["movements"] = movements
    if "sequence" in out:
        out.pop("sequence", None)
    return out


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

        if m_type == "path":
            from legacy.path_ee_ik import validate_path_parameters

            errors.extend(validate_path_parameters(params))

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
    prompt_file: str = _DEFAULT_MANIP_PROMPT,
    shots_json: str = "auto",
    config_json: str = "data/results/motion_configs/manipulator/motion_configs.json", # where to save
    max_handmade_examples: int = 10,
    max_correction_examples: int = 10,
    temperature: float | None = None,
    generation_seed: int | None = None,
    fewshot_seed: int | None = None,
    deterministic_fewshot: bool = False,
    use_shots: bool = True,
    require_reasoning: bool = True,
    max_attempts: int = 2,
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
        if deterministic_fewshot:
            ordered = sorted(handmade_configs, key=lambda c: str(c.get("cue", "")))
            selected_handmade = ordered[:num_handmade]
        elif fewshot_seed is not None:
            import random

            rng = random.Random(fewshot_seed)
            selected_handmade = (
                rng.sample(handmade_configs, num_handmade) if handmade_configs else []
            )
        else:
            import random

            selected_handmade = (
                random.sample(handmade_configs, num_handmade) if handmade_configs else []
            )
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
    print(
        f"  model={model_name} temperature={temperature if temperature is not None else 'default'}"
        f" generation_seed={generation_seed if generation_seed is not None else 'none'}"
        f" fewshot={'deterministic' if deterministic_fewshot else fewshot_seed if fewshot_seed is not None else 'random'}"
    )
    
    # 4. Inference with validation + retry
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

        gen_kwargs: dict[str, Any] = {}
        if temperature is not None:
            gen_kwargs["temperature"] = float(temperature)
        if generation_seed is not None:
            gen_kwargs["seed"] = int(generation_seed)
        gen_config = types.GenerateContentConfig(**gen_kwargs) if gen_kwargs else None
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

        new_config = _normalize_motion_config(new_config)
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
