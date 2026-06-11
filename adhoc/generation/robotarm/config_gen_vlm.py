"""Unified LLM config generation (Gemini API or Qwen VLM) for pilot-90 exp 1 & 7."""
from __future__ import annotations

import json
import os
import re
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from pilot90_paths import (
    GT_PATH,
    SHOTS,
    TILE_PICK,
    load_config_list,
    load_gt_by_cue,
    manifest90_cue_names,
    prompt_exp_path,
    save_config_list,
    upsert_config_row,
)

_REPO = Path(__file__).resolve().parents[2]
_LEGACY = Path(__file__).resolve().parent / "legacy"

import sys

for p in (_REPO, Path(__file__).resolve().parent, _LEGACY):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

def _legacy():
    from legacy.config_gen_single import (  # noqa: WPS433
        _extract_reasoning_and_json,
        _format_example_block,
        _load_json_list,
        _normalize_motion_config,
        _resolve_shots_json,
        _sanitize_model_output,
        _validate_config,
        _validate_reasoning,
    )
    from legacy.path_ee_ik import validate_path_parameters  # noqa: WPS433
    from generate_pose_group_tiles import _load_entries, _select_xyz_tertile_balanced  # noqa: WPS433

    return (
        _extract_reasoning_and_json,
        _format_example_block,
        _load_json_list,
        _normalize_motion_config,
        _resolve_shots_json,
        _sanitize_model_output,
        _validate_config,
        _validate_reasoning,
        validate_path_parameters,
        _load_entries,
        _select_xyz_tertile_balanced,
    )


def _require_planning_comments(backend: str, *, env_key: str) -> bool:
    """Gemini enforces Q1–Q4 comments; Qwen/transformers skip unless env overrides."""
    env = os.getenv(env_key)
    if env is not None:
        return env.strip().lower() not in {"0", "false", "no", "off"}
    return backend.lower() == "gemini"


def _llm_generate(
    prompt: str,
    *,
    model: str,
    backend: str = "gemini",
    vlm: Any | None = None,
) -> str:
    if vlm is not None:
        return vlm.generate(prompt)
    b = backend.lower()
    if b == "gemini":
        from google import genai

        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(model=model, contents=prompt)
        return (resp.text or "").strip()
    from vlm_client import VLMClient

    client = vlm or VLMClient(backend=b, model=model)
    return client.generate(prompt)


def _fewshot_block(prompt_path: Path, shots_json: Path, *, max_examples: int = 6) -> str:
    _, _format_example_block, _load_json_list, _, _resolve_shots_json, *_ = _legacy()
    shots = _load_json_list(_resolve_shots_json(str(prompt_path), str(shots_json)))
    handmade = [c for c in shots if c.get("state") == "handmade"][:max_examples]
    parts = [_format_example_block(ex, "# EXAMPLE:") for ex in handmade]
    return "\n\n".join(parts)


def _parse_gt_poses(groundtruth: str) -> list[tuple[str, str]]:
    return [(a.strip(), b.strip()) for a, b in re.findall(r"\(([^,]+),\s*([^)]+)\)", groundtruth)]


def _load_tile_pick() -> dict[tuple[str, str], int]:
    data = json.loads(TILE_PICK.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], int] = {}
    for k, v in data["picks"].items():
        parts = k.split("_")
        if len(parts) >= 2:
            out[(parts[0], "_".join(parts[1:]))] = int(v)
    return out


def _pose_id_for_group(d: str, g: str, tile_pick: dict[tuple[str, str], int]) -> int | None:
    from collections import defaultdict

    *_, validate_path_parameters, _load_entries, _select_xyz_tertile_balanced = _legacy()
    jsonl = _REPO / "data/seed/_remainder/closest_poses_results.jsonl"
    entries = [e for e in _load_entries(jsonl) if e.get("robot") == "IIWA"]
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entries:
        ed, eg = e.get("dir"), e.get("gripper_orientation")
        if ed and eg:
            buckets[(ed, eg)].append(e)
    rows = buckets.get((d, g), [])
    if not rows:
        return None
    reps = _select_xyz_tertile_balanced(rows, n=9)
    idx = tile_pick.get((d, g), 1) - 1
    if 0 <= idx < len(reps):
        return int(reps[idx]["pose_id"])
    return int(reps[0]["pose_id"])


def _build_fixed_pose(
    d: str,
    g: str,
    ref_pose: dict[str, Any],
    tile_pick: dict[tuple[str, str], int],
) -> dict[str, Any]:
    pose = {
        "dir": d,
        "gripper_orientation": g,
        "x": ref_pose.get("x", 50),
        "y": ref_pose.get("y", 50),
        "z": ref_pose.get("z", 50),
    }
    pid = _pose_id_for_group(d, g, tile_pick)
    if pid is not None:
        pose["pose_id"] = pid
    return pose


def _strip_leading_poses(movements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = list(movements)
    while out and out[0].get("type") == "pose":
        out.pop(0)
    return out


def _clamp_speeds(movements: list[dict[str, Any]]) -> None:
    for step in movements:
        params = step.get("parameters", {})
        if isinstance(params.get("speed"), (int, float)):
            params["speed"] = max(0.5, min(4.0, float(params["speed"])))
        if step.get("type") == "movement":
            for d in params.get("directions", []):
                if isinstance(d.get("speed"), (int, float)):
                    d["speed"] = max(0.5, min(4.0, float(d["speed"])))


def _validate_tail(movements: list[dict[str, Any]], cue_name: str) -> list[str]:
    (
        _extract_reasoning_and_json,
        _format_example_block,
        _load_json_list,
        _normalize_motion_config,
        _resolve_shots_json,
        _sanitize_model_output,
        _validate_config,
        _validate_reasoning,
        validate_path_parameters,
        _load_entries,
        _select_xyz_tertile_balanced,
    ) = _legacy()
    if not movements:
        return ["Tail must include at least one movement/path step"]
    if len(movements) > 3:
        return [f"Tail has {len(movements)} steps; max 3 allowed"]
    dummy = {"type": "pose", "parameters": {"pose": {"dir": "front", "x": 50, "y": 50, "z": 50}}}
    errs = _validate_config({"movements": [dummy, *movements], "cue": cue_name}, cue_name=cue_name)
    skip = ("First step must be type 'pose'", "Config is pose-only")
    tail_errs = [e for e in errs if not any(s in e for s in skip)]
    for step in movements:
        if step.get("type") == "path":
            tail_errs.extend(validate_path_parameters(step.get("parameters", {})))
    return tail_errs


def generate_exp1_row(
    *,
    cue: str,
    cue_idx: int,
    description: str,
    model: str,
    backend: str = "gemini",
    vlm: Any | None = None,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    (
        _extract_reasoning_and_json,
        _format_example_block,
        _load_json_list,
        _normalize_motion_config,
        _resolve_shots_json,
        _sanitize_model_output,
        _validate_config,
        _validate_reasoning,
        *_rest,
    ) = _legacy()
    require_reasoning = _require_planning_comments(backend, env_key="EXP1_REQUIRE_REASONING")
    if max_attempts is None:
        max_attempts = 3 if backend.lower() != "gemini" else 2
    prompt_template = prompt_exp_path(1).read_text(encoding="utf-8")
    examples = _fewshot_block(prompt_exp_path(1), SHOTS)
    prompt = prompt_template.replace("{{FEW_SHOT_EXAMPLES}}", examples).replace("{{CUE_NAME}}", cue)

    validation_errors: list[str] = []
    reasoning_text = ""
    new_config: dict[str, Any] | None = None

    for attempt in range(max_attempts):
        attempt_prompt = prompt
        if attempt > 0 and validation_errors:
            attempt_prompt += "\n\n# Fix these issues:\n" + "\n".join(
                f"# - {e}" for e in validation_errors
            )
        raw = _sanitize_model_output(_llm_generate(attempt_prompt, model=model, backend=backend, vlm=vlm))
        try:
            reasoning_text, parsed = _extract_reasoning_and_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            validation_errors = [f"JSON parse failed: {e}"]
            continue
        parsed = _normalize_motion_config(parsed)
        validation_errors = list(_validate_reasoning(reasoning_text))
        validation_errors.extend(_validate_config(parsed, cue_name=cue))
        if not validation_errors:
            new_config = parsed
            break

    if new_config is None:
        raise ValueError(f"exp1 failed for {cue}: {validation_errors}")

    out = {
        "idx": cue_idx,
        "cue": cue,
        "description": new_config.get("description") or description,
        "movements": new_config.get("movements") or [],
        "state": "exp1_pose_generation",
        "model": model,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "exp1",
    }
    if reasoning_text:
        out["reasoning"] = reasoning_text
    return out


def generate_exp7_row(
    *,
    cue: str,
    cue_idx: int,
    pose_gt: str,
    description: str,
    movement_gt: dict[str, Any] | None = None,
    ref_pose: dict[str, Any] | None = None,
    model: str,
    backend: str = "gemini",
    vlm: Any | None = None,
    tile_pick: dict[tuple[str, str], int] | None = None,
    max_attempts: int = 3,
) -> dict[str, Any]:
    targets = _parse_gt_poses(pose_gt)
    if not targets:
        raise ValueError(f"exp7: no parseable pose GT for {cue}: {pose_gt!r}")

    tile_pick = tile_pick or _load_tile_pick()
    gt_d, gt_g = targets[0]
    ref = ref_pose or {"x": 50, "y": 50, "z": 50}
    fixed = _build_fixed_pose(gt_d, gt_g, ref, tile_pick)
    fixed_step = {
        "type": "pose",
        "parameters": {"pose": deepcopy(fixed), "speed": 1.0, "hold_time": 0.0},
    }

    (
        _extract_reasoning_and_json,
        _format_example_block,
        _load_json_list,
        _normalize_motion_config,
        _resolve_shots_json,
        _sanitize_model_output,
        _validate_config,
        _validate_reasoning,
        *_rest,
    ) = _legacy()
    prompt_template = prompt_exp_path(7).read_text(encoding="utf-8")
    shots = _load_json_list(_resolve_shots_json(str(prompt_exp_path(1)), str(SHOTS)))
    examples_parts: list[str] = []
    for ex in [c for c in shots if c.get("state") in ("handmade", "choreography")][:6]:
        mov = ex.get("movements", [])
        if mov and mov[0].get("type") == "pose":
            examples_parts.append(
                _format_example_block(
                    {k: v for k, v in ex.items() if k != "planning_shot"},
                    "# EXAMPLE (start pose + follow-on steps):",
                )
            )
    examples = "\n\n".join(examples_parts)
    prompt = (
        prompt_template.replace("{{FEW_SHOT_EXAMPLES}}", examples)
        .replace("{{CUE_NAME}}", cue)
        .replace("{{FIXED_FIRST_POSE_JSON}}", json.dumps(fixed, indent=2))
    )

    validation_errors: list[str] = []
    reasoning_text = ""
    tail_cfg: dict[str, Any] | None = None
    final_tail: list[dict[str, Any]] = []

    for attempt in range(max_attempts):
        attempt_prompt = prompt
        if attempt > 0 and validation_errors:
            attempt_prompt += "\n\n# Fix these issues:\n" + "\n".join(
                f"# - {e}" for e in validation_errors
            )
        raw = _sanitize_model_output(_llm_generate(attempt_prompt, model=model, backend=backend, vlm=vlm))
        try:
            reasoning_text, parsed = _extract_reasoning_and_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            validation_errors = [f"JSON parse failed: {e}"]
            continue
        tail_movements = _strip_leading_poses(parsed.get("movements", []))
        _clamp_speeds(tail_movements)
        full = {"cue": cue, "movements": [fixed_step, *tail_movements]}
        validation_errors = []
        if not reasoning_text.strip():
            validation_errors.append("Missing planning comments before JSON")
        validation_errors.extend(_validate_tail(tail_movements, cue_name=cue))
        validation_errors.extend(_validate_config(full, cue_name=cue))
        if not validation_errors:
            tail_cfg = parsed
            final_tail = tail_movements
            break

    if tail_cfg is None:
        raise ValueError(f"exp7 failed for {cue}: {validation_errors}")

    out: dict[str, Any] = {
        "idx": cue_idx,
        "cue": cue,
        "description": tail_cfg.get("description") or description,
        "groundtruth": pose_gt,
        "pose_gt": pose_gt,
        "gt_fixed_first_pose": fixed,
        "movements": [fixed_step, *final_tail],
        "state": "exp7_motion_generation",
        "model": model,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "exp7",
        "generation_mode": "gt_fixed_first_pose_tail_from_llm",
        "reasoning": reasoning_text,
    }
    if movement_gt is not None:
        out["movement_gt"] = movement_gt
    return out


def run_exp_generation(
    exp_id: str | int,
    *,
    out_path: Path,
    model: str,
    backend: str = "gemini",
    vlm: Any | None = None,
    cues: list[str] | None = None,
    resume: bool = True,
    delay: float = 2.0,
    on_progress: Callable[[str, bool], None] | None = None,
) -> tuple[int, int]:
    """Generate missing cues for exp 1 or 7. Returns (ok, failed)."""
    eid = str(exp_id)
    if eid not in {"1", "7"}:
        raise ValueError("Only exp 1 and 7 are generation steps")

    gt_rows = load_gt_by_cue()
    work_names = cues or manifest90_cue_names()
    existing = {r["cue"]: r for r in load_config_list(out_path)} if resume else {}
    tile_pick = _load_tile_pick()

    ok = failed = 0
    for cue in work_names:
        if resume and cue in existing and existing[cue].get("movements"):
            continue
        row_gt = gt_rows.get(cue)
        if not row_gt:
            failed += 1
            if on_progress:
                on_progress(cue, False)
            continue
        cue_idx = int(row_gt["cue_idx"])
        try:
            if eid == "1":
                gen = generate_exp1_row(
                    cue=cue,
                    cue_idx=cue_idx,
                    description=str(row_gt.get("description") or ""),
                    model=model,
                    backend=backend,
                    vlm=vlm,
                )
            else:
                pose_gt = str(row_gt.get("pose_gt") or "")
                ref_cfg = existing.get(cue) or {}
                ref_pose = {}
                for step in ref_cfg.get("movements") or []:
                    if step.get("type") == "pose":
                        ref_pose = step.get("parameters", {}).get("pose") or {}
                        break
                gen = generate_exp7_row(
                    cue=cue,
                    cue_idx=cue_idx,
                    pose_gt=pose_gt,
                    description=str(row_gt.get("description") or ""),
                    movement_gt=row_gt.get("movement_gt"),
                    ref_pose=ref_pose,
                    model=model,
                    backend=backend,
                    vlm=vlm,
                    tile_pick=tile_pick,
                )
            upsert_config_row(out_path, gen)
            ok += 1
            if on_progress:
                on_progress(cue, True)
        except Exception as e:
            failed += 1
            print(f"[FAIL] {cue}: {e}", flush=True)
            if on_progress:
                on_progress(cue, False)
        time.sleep(delay)

    return ok, failed
