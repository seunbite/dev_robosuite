import copy
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List

import fire
from google import genai
from google.genai import types


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_VALID_PATH_STYLES = {"none", "elegant", "wobble", "hesitant"}
_VALID_PATH_TARGETS = {"first_dynamic", "last_dynamic", "all_paths"}


VARIATION_SPEC_PROMPT = """You are an expert in expressive robot motion variation design.

You are given:
1. A base robot motion config
2. A persona

Your task is NOT to rewrite the whole motion config.
Instead, design a compact variation spec that can be applied algorithmically.

[Variation Categories]
You must design all three categories:
1. parameter_variation
2. pose_variation
3. path_variation

[Intent]
- Preserve the cue identity and core readability.
- Keep the primitive sequence mostly intact.
- Persona should come through as style, timing, hesitation, flourish, restraint, or energy.
- Favor subtle-to-moderate edits, not a different gesture.

[Definitions]
1. parameter_variation:
- Changes scalar motion parameters like speed, hold time, repetition, degrees, and path magnitude.

2. pose_variation:
- Adds slight x/y/z pose offsets to existing pose steps.
- This is not a new pose design; it is a slight spatial shift of the existing pose anchors.

3. path_variation:
- This is a generalized trajectory style policy.
- Allowed styles:
  - "none": no special path stylization
  - "elegant": soften straight motions into graceful curves or small arc flourishes
  - "wobble": inject slight wavering / unstable side-to-side deviation
  - "hesitant": inject slight delay, reverse-prep, or cautious micro-trajectory before the main action
- If the base motion already has path steps, the style should describe how to modify them.
- If the base motion has no path step, the style may request insertion of one small micro-path if it helps and remains readable.

[Output JSON Schema]
Return exactly one JSON object with this shape:
{{
  "persona_summary": "...",
  "parameter_variation": {{
    "speed_scale": [min, max],
    "hold_scale": [min, max],
    "degree_scale": [min, max],
    "repetition_delta": [min_int, max_int],
    "timing_jitter": 0.0-0.5
  }},
  "pose_variation": {{
    "enabled": true,
    "x_offset": [min_int, max_int],
    "y_offset": [min_int, max_int],
    "z_offset": [min_int, max_int],
    "anchor_weight": 0.0-1.0
  }},
  "path_variation": {{
    "style": "none|elegant|wobble|hesitant",
    "strength": 0.0-1.0,
    "insert_if_missing": true,
    "target": "first_dynamic|last_dynamic|all_paths"
  }}
}}

[Constraints]
- speed_scale should usually stay within [0.6, 1.6]
- hold_scale should usually stay within [0.5, 2.0]
- degree_scale should usually stay within [0.6, 1.5]
- repetition_delta should usually stay in [-1, 2]
- x/y/z offsets should usually stay within [-12, 12]
- anchor_weight controls how much the very first pose is protected from large offset; higher means more stable
- Choose one path_variation style only
- Be conservative when the cue is highly iconic and brittle

[Persona]
Name: {persona_name}
Description: {persona_description}
Edit Strength: {edit_strength}

[Base Motion Config]
{base_config_json}
"""


def _sanitize_model_output(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n").lstrip("\ufeff")
    text = _CONTROL_CHAR_RE.sub("", text)
    return text.strip()


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = _sanitize_model_output(raw_text)
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    decoder = json.JSONDecoder()
    candidates = [text]
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())
    for candidate_text in candidates:
        for pos, char in enumerate(candidate_text):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(candidate_text[pos:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise json.JSONDecodeError("No JSON object found in model output", text, 0)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clamp_range(values: List[float], lo: float, hi: float) -> List[float]:
    if len(values) != 2:
        raise ValueError(f"Expected range pair, got: {values}")
    a = _clamp(float(values[0]), lo, hi)
    b = _clamp(float(values[1]), lo, hi)
    return [min(a, b), max(a, b)]


def _stable_rng(*parts: Any) -> random.Random:
    seed_src = "||".join(str(part) for part in parts)
    seed = int(hashlib.md5(seed_src.encode("utf-8")).hexdigest()[:8], 16)
    return random.Random(seed)


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


def _esc(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _open_preview(path: str) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as exc:
        print(f"Preview open failed: {exc}")


def _safe_cue_name(cue: str) -> str:
    return cue.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _find_latest_render_gif(render_dir: str, cue_idx: int, cue_name: str | None = None) -> str | None:
    patterns = [
        os.path.join(render_dir, f"*_c{cue_idx}_tiled.gif"),
        os.path.join(render_dir, f"*_c{cue_idx}_*.gif"),
    ]
    if cue_name:
        safe_cue = _safe_cue_name(cue_name)
        patterns.extend([
            os.path.join(render_dir, f"*_{safe_cue}_tiled.gif"),
            os.path.join(render_dir, f"*_{safe_cue}_*.gif"),
        ])
    matches: List[str] = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            base = os.path.basename(path)
            if base.endswith("_preview.gif"):
                continue
            matches.append(path)
    if not matches:
        return None
    unique = sorted(set(matches), key=os.path.getmtime, reverse=True)
    return unique[0]


def _normalize_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(spec)
    out.setdefault("persona_summary", "")
    out.setdefault("_edit_strength", "subtle")

    pv = out.setdefault("parameter_variation", {})
    pv["speed_scale"] = _clamp_range(pv.get("speed_scale", [0.9, 1.1]), 0.6, 1.6)
    pv["hold_scale"] = _clamp_range(pv.get("hold_scale", [0.9, 1.1]), 0.5, 2.0)
    pv["degree_scale"] = _clamp_range(pv.get("degree_scale", [0.9, 1.1]), 0.6, 1.5)
    rep_delta = pv.get("repetition_delta", [0, 0])
    if len(rep_delta) != 2:
        rep_delta = [0, 0]
    pv["repetition_delta"] = [int(max(-1, min(2, rep_delta[0]))), int(max(-1, min(2, rep_delta[1])))]
    pv["repetition_delta"] = [min(pv["repetition_delta"]), max(pv["repetition_delta"])]
    pv["timing_jitter"] = _clamp(float(pv.get("timing_jitter", 0.1)), 0.0, 0.5)

    posev = out.setdefault("pose_variation", {})
    posev["enabled"] = bool(posev.get("enabled", True))
    posev["x_offset"] = [int(round(v)) for v in _clamp_range(posev.get("x_offset", [-4, 4]), -12, 12)]
    posev["y_offset"] = [int(round(v)) for v in _clamp_range(posev.get("y_offset", [-4, 4]), -12, 12)]
    posev["z_offset"] = [int(round(v)) for v in _clamp_range(posev.get("z_offset", [-4, 4]), -12, 12)]
    posev["anchor_weight"] = _clamp(float(posev.get("anchor_weight", 0.7)), 0.0, 1.0)

    pathv = out.setdefault("path_variation", {})
    style = str(pathv.get("style", "none")).strip().lower()
    pathv["style"] = style if style in _VALID_PATH_STYLES else "none"
    pathv["strength"] = _clamp(float(pathv.get("strength", 0.0)), 0.0, 1.0)
    pathv["insert_if_missing"] = bool(pathv.get("insert_if_missing", False))
    target = str(pathv.get("target", "first_dynamic")).strip().lower()
    pathv["target"] = target if target in _VALID_PATH_TARGETS else "first_dynamic"
    return out


def _first_dynamic_index(movements: List[Dict[str, Any]]) -> int | None:
    for idx, step in enumerate(movements):
        if step.get("type") != "pose":
            return idx
    return None


def _last_dynamic_index(movements: List[Dict[str, Any]]) -> int | None:
    for idx in range(len(movements) - 1, -1, -1):
        if movements[idx].get("type") != "pose":
            return idx
    return None


def _dominant_axis_from_movement(step: Dict[str, Any]) -> str:
    best_axis = "x"
    best_val = -1.0
    for direction in step.get("parameters", {}).get("directions", []):
        for axis, val in direction.get("degrees", {}).items():
            aval = abs(float(val))
            if aval > best_val:
                best_val = aval
                best_axis = axis
    return best_axis


def _dominant_axis_from_step(step: Dict[str, Any]) -> str:
    mtype = step.get("type")
    params = step.get("parameters", {})
    if mtype == "movement":
        return _dominant_axis_from_movement(step)
    if mtype == "path":
        if params.get("shape") == "line":
            return str(params.get("axis", "x"))
        plane = str(params.get("plane", "xz"))
        return plane[0] if plane else "x"
    return "x"


def _orthogonal_axis(axis: str) -> str:
    axis = axis.lower()
    return {"x": "y", "y": "z", "z": "x"}.get(axis, "y")


def _plane_for_axis(axis: str) -> str:
    return {"x": "xz", "y": "xy", "z": "yz"}.get(axis, "xz")


def _infer_joint(step: Dict[str, Any], fallback: str = "wrist") -> str:
    return str(step.get("parameters", {}).get("joint", fallback))


def _sample_range(rng: random.Random, values: List[float]) -> float:
    return rng.uniform(float(values[0]), float(values[1]))


def _sample_int_range(rng: random.Random, values: List[int]) -> int:
    return rng.randint(int(values[0]), int(values[1]))


def _edit_strength_gain(spec: Dict[str, Any]) -> float:
    label = str(spec.get("_edit_strength", "subtle")).strip().lower()
    return {
        "subtle": 1.35,
        "medium": 1.75,
        "strong": 2.2,
    }.get(label, 1.35)


def _amplify_around_one(value: float, gain: float) -> float:
    return 1.0 + (value - 1.0) * gain


def _apply_parameter_variation(cfg: Dict[str, Any], spec: Dict[str, Any], rng: random.Random) -> None:
    pv = spec["parameter_variation"]
    gain = _edit_strength_gain(spec)
    timing_jitter = _clamp(pv["timing_jitter"] * (1.0 + 0.55 * (gain - 1.0)), 0.0, 0.5)
    for step in cfg.get("movements", []):
        params = step.get("parameters", {})
        mtype = step.get("type")

        if mtype == "pose":
            if "speed" in params:
                scale = _amplify_around_one(_sample_range(rng, pv["speed_scale"]), gain)
                jitter = 1.0 + rng.uniform(-timing_jitter, timing_jitter)
                params["speed"] = round(_clamp(float(params["speed"]) * scale * jitter, 0.5, 4.0), 3)
            if "hold_time" in params:
                scale = _amplify_around_one(_sample_range(rng, pv["hold_scale"]), gain)
                jitter = 1.0 + rng.uniform(-timing_jitter, timing_jitter)
                params["hold_time"] = round(max(0.0, float(params["hold_time"]) * scale * jitter), 3)

        elif mtype == "movement":
            if "repetition" in params:
                delta = _sample_int_range(rng, pv["repetition_delta"])
                if delta == 0 and pv["repetition_delta"][1] > 0 and gain >= 1.3:
                    delta = 1
                params["repetition"] = max(1, int(params["repetition"]) + delta)
            for direction in params.get("directions", []):
                if "speed" in direction:
                    scale = _amplify_around_one(_sample_range(rng, pv["speed_scale"]), gain)
                    jitter = 1.0 + rng.uniform(-timing_jitter, timing_jitter)
                    direction["speed"] = round(_clamp(float(direction["speed"]) * scale * jitter, 0.5, 4.0), 3)
                if "hold_time" in direction:
                    scale = _amplify_around_one(_sample_range(rng, pv["hold_scale"]), gain)
                    jitter = 1.0 + rng.uniform(-timing_jitter, timing_jitter)
                    direction["hold_time"] = round(max(0.0, float(direction["hold_time"]) * scale * jitter), 3)
                degrees = direction.get("degrees", {})
                for axis, val in list(degrees.items()):
                    scale = _amplify_around_one(_sample_range(rng, pv["degree_scale"]), gain)
                    new_val = float(val) * scale
                    if abs(new_val - float(val)) < 3.0 and gain >= 1.3:
                        new_val = float(val) + (3.0 if float(val) >= 0 else -3.0)
                    degrees[axis] = round(_clamp(new_val, -65, 65), 3)

        elif mtype == "path":
            if "speed" in params:
                scale = _amplify_around_one(_sample_range(rng, pv["speed_scale"]), gain)
                jitter = 1.0 + rng.uniform(-timing_jitter, timing_jitter)
                params["speed"] = round(_clamp(float(params["speed"]) * scale * jitter, 0.5, 4.0), 3)
            if "distance" in params:
                scale = _amplify_around_one(_sample_range(rng, pv["degree_scale"]), gain)
                params["distance"] = round(float(params["distance"]) * scale, 3)
            if "radius" in params:
                scale = _amplify_around_one(_sample_range(rng, pv["degree_scale"]), gain)
                params["radius"] = round(max(1.0, float(params["radius"]) * scale), 3)
            if "sweep" in params:
                scale = _amplify_around_one(_sample_range(rng, pv["degree_scale"]), gain)
                params["sweep"] = round(float(params["sweep"]) * scale, 3)


def _apply_pose_variation(cfg: Dict[str, Any], spec: Dict[str, Any], rng: random.Random) -> None:
    posev = spec["pose_variation"]
    if not posev["enabled"]:
        return

    gain = _edit_strength_gain(spec)
    pose_steps = [step for step in cfg.get("movements", []) if step.get("type") == "pose"]
    for pose_i, step in enumerate(pose_steps):
        params = step.get("parameters", {})
        pose = params.get("pose", {})
        protection = posev["anchor_weight"] if pose_i == 0 else 0.0
        protection *= 0.65
        scale = max(0.45, (1.0 - protection) * (1.0 + 0.45 * (gain - 1.0)))
        for axis, key in (("x", "x_offset"), ("y", "y_offset"), ("z", "z_offset")):
            if axis not in pose:
                continue
            lo, hi = posev[key]
            offset = int(round(rng.uniform(lo, hi) * scale))
            if abs(offset) < 2 and hi - lo >= 4:
                offset = 2 if rng.random() < 0.5 else -2
            pose[axis] = int(round(_clamp(float(pose[axis]) + offset, 0, 100)))


def _make_micro_path(style: str, strength: float, joint: str, primary_axis: str, rng: random.Random) -> Dict[str, Any]:
    if style == "elegant":
        return {
            "type": "path",
            "parameters": {
                "shape": "arc",
                "joint": joint,
                "plane": _plane_for_axis(primary_axis),
                "radius": round(7 + 12 * strength + rng.uniform(0, 4), 3),
                "sweep": round(40 + 65 * strength + rng.uniform(0, 15), 3),
                "direction": rng.choice(["cw", "ccw"]),
                "speed": round(_clamp(1.0 - 0.15 * strength + rng.uniform(-0.1, 0.2), 0.5, 4.0), 3),
            },
        }
    if style == "wobble":
        return {
            "type": "path",
            "parameters": {
                "shape": "line",
                "joint": joint,
                "axis": _orthogonal_axis(primary_axis),
                "distance": round(rng.choice([-1, 1]) * (6 + 10 * strength + rng.uniform(0, 3)), 3),
                "speed": round(_clamp(1.1 + rng.uniform(-0.2, 0.3), 0.5, 4.0), 3),
            },
        }
    if style == "hesitant":
        return {
            "type": "path",
            "parameters": {
                "shape": "line",
                "joint": joint,
                "axis": primary_axis,
                "distance": round(rng.choice([-1, 1]) * (-4 - 9 * strength), 3),
                "speed": round(_clamp(0.7 + rng.uniform(-0.1, 0.1), 0.5, 4.0), 3),
            },
        }
    raise ValueError(f"Unsupported path style: {style}")


def _transform_existing_path(step: Dict[str, Any], style: str, strength: float, rng: random.Random) -> None:
    params = step.get("parameters", {})
    shape = params.get("shape")
    if style == "elegant":
        if shape == "line":
            axis = str(params.get("axis", "x"))
            distance = abs(float(params.get("distance", 8.0)))
            direction = "cw" if float(params.get("distance", 1.0)) >= 0 else "ccw"
            joint = params.get("joint", "wrist")
            params.clear()
            params.update({
                "shape": "arc",
                "joint": joint,
                "plane": _plane_for_axis(axis),
                "radius": round(max(4.0, distance * (0.45 + 0.25 * strength)), 3),
                "sweep": round(30 + distance * (1.2 + 0.6 * strength), 3),
                "direction": direction,
                "speed": round(_clamp(0.9 + rng.uniform(-0.1, 0.1), 0.5, 4.0), 3),
            })
        else:
            if "radius" in params:
                params["radius"] = round(max(1.0, float(params["radius"]) * (1.0 + 0.15 * strength)), 3)
            if "sweep" in params:
                params["sweep"] = round(float(params["sweep"]) * (1.0 + 0.08 * strength), 3)
            if "speed" in params:
                params["speed"] = round(_clamp(float(params["speed"]) * (0.92 - 0.08 * strength), 0.5, 4.0), 3)

    elif style == "wobble":
        if shape == "line" and "distance" in params:
            params["distance"] = round(float(params["distance"]) * (0.85 + 0.1 * strength), 3)
        if "speed" in params:
            params["speed"] = round(_clamp(float(params["speed"]) * (1.0 + 0.1 * strength), 0.5, 4.0), 3)

    elif style == "hesitant":
        if "speed" in params:
            params["speed"] = round(_clamp(float(params["speed"]) * (0.75 - 0.15 * strength), 0.5, 4.0), 3)
        if "distance" in params:
            params["distance"] = round(float(params["distance"]) * (0.8 + 0.1 * strength), 3)
        if "sweep" in params:
            params["sweep"] = round(float(params["sweep"]) * (0.85 + 0.05 * strength), 3)


def _insert_step(movements: List[Dict[str, Any]], index: int, step: Dict[str, Any]) -> None:
    if len(movements) < 8:
        movements.insert(index, step)


def _apply_path_variation(cfg: Dict[str, Any], spec: Dict[str, Any], rng: random.Random) -> None:
    pathv = spec["path_variation"]
    style = pathv["style"]
    strength = _clamp(pathv["strength"] * (1.0 + 0.5 * (_edit_strength_gain(spec) - 1.0)), 0.0, 1.0)
    if style == "none" or strength <= 0.0:
        return

    movements = cfg.get("movements", [])
    path_indices = [i for i, step in enumerate(movements) if step.get("type") == "path"]

    for idx in path_indices:
        _transform_existing_path(movements[idx], style, strength, rng)

    dynamic_targets: List[int] = []
    target_name = pathv["target"]
    if target_name == "first_dynamic":
        idx = _first_dynamic_index(movements)
        if idx is not None:
            dynamic_targets = [idx]
    elif target_name == "last_dynamic":
        idx = _last_dynamic_index(movements)
        if idx is not None:
            dynamic_targets = [idx]
    else:
        dynamic_targets = [i for i, step in enumerate(movements) if step.get("type") in {"movement", "path"}]

    if not path_indices and pathv["insert_if_missing"] and len(movements) < 8:
        for target_idx in dynamic_targets[: max(1, min(2, 8 - len(movements)))]:
            if len(movements) >= 8:
                break
            target_step = movements[target_idx]
            joint = _infer_joint(target_step)
            primary_axis = _dominant_axis_from_step(target_step)
            micro_path = _make_micro_path(style, strength, joint, primary_axis, rng)
            insert_at = target_idx if style == "hesitant" else target_idx + 1
            _insert_step(movements, insert_at, micro_path)

    if style == "wobble" and len(movements) < 8:
        target_idx = _last_dynamic_index(movements)
        if target_idx is not None:
            joint = _infer_joint(movements[target_idx])
            primary_axis = _dominant_axis_from_step(movements[target_idx])
            micro_path = _make_micro_path("wobble", min(1.0, strength * 0.9), joint, primary_axis, rng)
            _insert_step(movements, target_idx + 1, micro_path)


def _postprocess_config(cfg: Dict[str, Any]) -> None:
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
                if key in params:
                    params[key] = round(float(params[key]), 3)


def _validate_config(base_cfg: Dict[str, Any], edited_cfg: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    movements = edited_cfg.get("movements", [])
    if edited_cfg.get("cue") != base_cfg.get("cue"):
        errors.append("Cue name changed")
    if not movements:
        errors.append("No movements found")
        return errors
    if movements[0].get("type") != "pose":
        errors.append("First step must be pose")
    if all(step.get("type") == "pose" for step in movements):
        errors.append("Pose-only result is not allowed")
    if not (2 <= len(movements) <= 8):
        errors.append(f"Step count {len(movements)} is outside [2, 8]")
    return errors


def design_variation_spec(
    base_cfg: Dict[str, Any],
    *,
    persona_name: str,
    persona_description: str,
    edit_strength: str,
    client: genai.Client,
    model_name: str,
    temperature: float,
) -> Dict[str, Any]:
    prompt = VARIATION_SPEC_PROMPT.format(
        persona_name=persona_name,
        persona_description=persona_description,
        edit_strength=edit_strength,
        base_config_json=json.dumps(base_cfg, indent=2, ensure_ascii=False),
    )
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=float(temperature)),
    )
    spec = _normalize_spec(_extract_json_object(response.text.strip()))
    spec["_edit_strength"] = edit_strength
    return spec


def apply_variation_spec(base_cfg: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    rng = _stable_rng(base_cfg.get("cue"), base_cfg.get("idx"), json.dumps(spec, sort_keys=True))
    _apply_parameter_variation(cfg, spec, rng)
    _apply_pose_variation(cfg, spec, rng)
    _apply_path_variation(cfg, spec, rng)
    _postprocess_config(cfg)
    errors = _validate_config(base_cfg, cfg)
    if errors:
        raise ValueError(f"Invalid edited config for cue '{base_cfg.get('cue')}': {'; '.join(errors)}")
    return cfg


def _render_config(
    *,
    script_dir: str,
    config_path: str,
    cue_idx: int,
    cue_name: str,
    robot: str,
    output_dir: str,
    top_k: int,
    hz: int,
    preview_speed_scale: float = 1.0,
    preview_hold_scale: float = 1.0,
    preview_max_hold_time: float | None = None,
) -> str | None:
    motion_script = os.path.join(script_dir, "motion_generation.py")
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    jsonl_path = os.path.join(project_root, "data", "seed", "closest_poses_results.jsonl")
    cmd = [
        sys.executable,
        motion_script,
        f"--robot={robot}",
        f"--cue_idx={cue_idx}",
        f"--config_path={config_path}",
        f"--jsonl_path={jsonl_path}",
        f"--output_dir={output_dir}",
        f"--top_k={top_k}",
        f"--hz={hz}",
        f"--preview_speed_scale={preview_speed_scale}",
        f"--preview_hold_scale={preview_hold_scale}",
    ]
    if preview_max_hold_time is not None:
        cmd.append(f"--preview_max_hold_time={preview_max_hold_time}")
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        tail = result.stderr.strip().splitlines()[-5:] if result.stderr else []
        print(f"  render failed for c{cue_idx} ({os.path.basename(config_path)})")
        for line in tail:
            print(f"    {line}")
        return None
    return _find_latest_render_gif(os.path.join(output_dir, robot), cue_idx, cue_name=cue_name)


def _write_compare_html(
    *,
    output_path: str,
    title: str,
    persona_name: str,
    persona_description: str,
    rows: List[Dict[str, Any]],
) -> str:
    html_dir = os.path.dirname(output_path)
    parts = [f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_esc(title)}</title>
<style>
:root {{
  --bg: #f5f6f8; --surface: #ffffff; --surface2: #eef1f5;
  --border: #d5d9e0; --text: #1f2328; --text2: #5b6472; --accent: #0969da;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, 'SF Pro Text', 'Segoe UI', sans-serif; }}
.wrap {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
.hero {{ margin-bottom: 18px; }}
.hero h1 {{ margin: 0 0 8px; font-size: 28px; }}
.hero p {{ margin: 0 0 10px; color: var(--text2); }}
.chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.chip {{ padding: 4px 10px; border-radius: 999px; border: 1px solid var(--border); background: var(--surface2); color: var(--text2); font-size: 14px; }}
.row {{ margin-bottom: 20px; background: var(--surface); border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }}
.cue-header {{ padding: 14px 16px; border-bottom: 1px solid var(--border); background: var(--surface2); font-weight: 700; font-size: 18px; }}
.cue-header .idx {{ color: var(--accent); margin-right: 8px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; }}
.col {{ padding: 16px; border-right: 1px solid var(--border); }}
.col:last-child {{ border-right: 0; }}
.label {{ margin-bottom: 10px; font-size: 15px; font-weight: 700; color: var(--text2); text-transform: uppercase; letter-spacing: 0.04em; }}
.gif {{ min-height: 220px; display: flex; align-items: center; justify-content: center; margin-bottom: 12px; background: var(--surface2); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
.gif img {{ display: block; max-width: 100%; }}
.na {{ color: var(--text2); font-style: italic; }}
.meta {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }}
.block-title {{ margin: 0 0 6px; font-size: 12px; font-weight: 700; color: var(--text2); text-transform: uppercase; letter-spacing: 0.05em; }}
pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; background: var(--surface2); border-radius: 8px; padding: 10px; font-size: 12px; border: 1px solid var(--border); }}
@media (max-width: 980px) {{
  .grid {{ grid-template-columns: 1fr; }}
  .col {{ border-right: 0; border-bottom: 1px solid var(--border); }}
  .col:last-child {{ border-bottom: 0; }}
}}
</style>
</head>
<body>
<div class="wrap">
  <section class="hero">
    <h1>{_esc(title)}</h1>
    <p>{_esc(persona_description)}</p>
    <div class="chips">
      <span class="chip">persona: {_esc(persona_name)}</span>
      <span class="chip">cues: {len(rows)}</span>
    </div>
  </section>
"""]

    for row in rows:
        base_gif = row.get("base_gif")
        persona_gif = row.get("persona_gif")
        base_rel = _esc(os.path.relpath(base_gif, html_dir)) if base_gif else ""
        persona_rel = _esc(os.path.relpath(persona_gif, html_dir)) if persona_gif else ""
        base_cfg = row["base_cfg"]
        persona_cfg = row["persona_cfg"]
        spec = row["spec"]
        parts.append(f"""
  <section class="row">
    <div class="cue-header"><span class="idx">c{row['idx']}</span>{_esc(row['cue'])}</div>
    <div class="grid">
      <div class="col">
        <div class="label">Original</div>
        <div class="gif">{f'<img src="{base_rel}" alt="base c{row["idx"]}">' if base_gif else '<span class="na">No render found</span>'}</div>
        <div class="block-title">Description</div>
        <pre>{_esc(base_cfg.get('description', ''))}</pre>
      </div>
      <div class="col">
        <div class="label">Persona</div>
        <div class="gif">{f'<img src="{persona_rel}" alt="persona c{row["idx"]}">' if persona_gif else '<span class="na">No render found</span>'}</div>
        <div class="block-title">Description</div>
        <pre>{_esc(persona_cfg.get('description', ''))}</pre>
      </div>
    </div>
    <div class="grid">
      <div class="col">
        <div class="block-title">Original Movements</div>
        <pre>{_esc(json.dumps(base_cfg.get('movements', []), indent=2, ensure_ascii=False))}</pre>
      </div>
      <div class="col">
        <div class="block-title">Persona Spec</div>
        <pre>{_esc(json.dumps(spec, indent=2, ensure_ascii=False))}</pre>
      </div>
    </div>
  </section>
""")

    parts.append("""
</div>
</body>
</html>
""")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    return output_path


def main(
    input_json: str,
    output_json: str | None = None,
    persona_name: str = "calm",
    persona_description: str = "A calm, restrained, smooth, deliberate personality with gentle timing and softened motion.",
    cue_idxs: List[int] | None = None,
    model_name: str = "gemini-3.1-flash-lite-preview",
    edit_strength: str = "subtle",
    temperature: float = 0.3,
    render_html: bool = True,
    open_html: bool = True,
    render_robot: str = "IIWA",
    render_top_k: int = 1,
    render_hz: int = 4,
):
    """Design persona variation specs with an LLM, then apply parameter / pose / path variation in code."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable.")

    data = _load_json_list(input_json)
    selected = _select_configs(data, cue_idxs)
    if not selected:
        raise ValueError("No configs selected for persona variation.")

    client = genai.Client(api_key=api_key)
    edited_by_idx: Dict[int, Dict[str, Any]] = {}

    print(f"input={input_json}")
    print(f"persona={persona_name} strength={edit_strength} model={model_name} temperature={temperature}")
    print(f"selected_cues={len(selected)}")

    for cfg in selected:
        idx = cfg.get("idx")
        cue = cfg.get("cue", "<unknown>")
        print(f"\nDesigning variation for c{idx}: {cue}")
        spec = design_variation_spec(
            cfg,
            persona_name=persona_name,
            persona_description=persona_description,
            edit_strength=edit_strength,
            client=client,
            model_name=model_name,
            temperature=temperature,
        )
        edited = apply_variation_spec(cfg, spec)
        edited["persona_variation_spec"] = spec
        edited["persona_name"] = persona_name
        edited["persona_description"] = persona_description
        edited["persona_edit_strength"] = edit_strength
        edited["persona_model"] = model_name
        edited["persona_temperature"] = temperature
        edited["persona_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        edited["base_config_snapshot"] = {
            "description": cfg.get("description", ""),
            "movements": cfg.get("movements", []),
        }
        edited_by_idx[idx] = edited
        print(f"  path_style={spec['path_variation']['style']} target={spec['path_variation']['target']}")
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
        output_json = f"{stem}_{suffix}_var{ext}"

    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\nSaved persona-variation configs to: {output_json}")

    if render_html:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_stem = os.path.splitext(os.path.abspath(output_json))[0]
        render_root = f"{output_stem}_compare_renders"
        base_render_root = os.path.join(render_root, "base")
        persona_render_root = os.path.join(render_root, "persona")
        rows: List[Dict[str, Any]] = []

        print(f"\nRendering base vs persona for {len(selected)} cues...")
        for base_cfg in selected:
            idx = int(base_cfg["idx"])
            persona_cfg = edited_by_idx[idx]
            cue = str(base_cfg.get("cue", f"cue_{idx}"))
            print(f"  render c{idx}: {cue}")
            base_gif = _render_config(
                script_dir=script_dir,
                config_path=os.path.abspath(input_json),
                cue_idx=idx,
                cue_name=cue,
                robot=render_robot,
                output_dir=base_render_root,
                top_k=render_top_k,
                hz=render_hz,
            )
            persona_gif = _render_config(
                script_dir=script_dir,
                config_path=os.path.abspath(output_json),
                cue_idx=idx,
                cue_name=cue,
                robot=render_robot,
                output_dir=persona_render_root,
                top_k=render_top_k,
                hz=render_hz,
            )
            rows.append({
                "idx": idx,
                "cue": cue,
                "base_cfg": base_cfg,
                "persona_cfg": persona_cfg,
                "spec": persona_cfg.get("persona_variation_spec", {}),
                "base_gif": base_gif,
                "persona_gif": persona_gif,
            })

        html_path = f"{output_stem}_compare.html"
        _write_compare_html(
            output_path=html_path,
            title=f"Persona Motion Compare: {persona_name}",
            persona_name=persona_name,
            persona_description=persona_description,
            rows=rows,
        )
        html_abs = os.path.abspath(html_path)
        print(f"\nCompare HTML: {html_abs}")
        print(f"Compare URL: file://{html_abs}")
        if open_html:
            _open_preview(html_abs)


if __name__ == "__main__":
    fire.Fire(main)
