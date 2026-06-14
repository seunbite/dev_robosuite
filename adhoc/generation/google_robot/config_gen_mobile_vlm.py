"""Unified LLM config generation (Gemini API or vLLM) for pilot-40 Google Robot exp 1 & 7."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from pilot40_paths import (
    CUES_YAML,
    FEWSHOT_SHOTS,
    GT_PATH,
    load_config_list,
    load_gt_by_cue,
    manifest90_cue_names,
    prompt_exp_path,
    row_generation_done,
    save_json,
    upsert_config_row,
)

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_ROBOTARM = _REPO / "adhoc/generation/robotarm"
for p in (_REPO, _ROBOTARM, _HERE):
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
    return load_gt_by_cue()


def _cue_idx(row: dict[str, Any], *, gt: dict[str, dict[str, Any]] | None = None) -> int:
    if row.get("idx") is not None:
        return int(row["idx"])
    cue = str(row.get("cue", ""))
    if gt and cue in gt:
        ev = gt[cue]
        if ev.get("cue_idx") is not None:
            return int(ev["cue_idx"])
    return 0


def _require_planning_comments(backend: str, *, env_key: str) -> bool:
    env = os.getenv(env_key)
    # Empty export (EXP7_REQUIRE_REASONING=) must not enable validation.
    if env is not None and env.strip():
        return env.strip().lower() not in {"0", "false", "no", "off"}
    return backend.lower() == "gemini"


def _is_local_backend(backend: str) -> bool:
    return backend.lower() in {"local", "vllm-local", "transformers", "hf", "vllm"}


def _cue_catalog(*, cue: str, backend: str) -> str:
    if _is_local_backend(backend):
        return f"(single target cue: {cue})"
    if not CUES_YAML.is_file():
        return ""
    data = yaml.safe_load(CUES_YAML.read_text(encoding="utf-8"))
    parts: list[str] = []
    for grp in ("iconic", "contextual"):
        block = data.get(grp)
        if not isinstance(block, dict):
            continue
        parts.append(f"[Available {grp} cues]")
        for name, text in block.items():
            parts.append(f"- {name}: {text}")
        parts.append("")
    return "\n".join(parts)


def _fewshot_block(*, max_examples: int = 6, backend: str = "gemini") -> str:
    from adhoc.generation.google_robot.legacy.config_gen_single_mobile import (  # noqa: WPS433
        _format_example_block,
        _load_json_list,
    )

    cap = 2 if _is_local_backend(backend) else max_examples
    shots = _load_json_list(str(FEWSHOT_SHOTS)) if FEWSHOT_SHOTS.is_file() else []
    parts = [_format_example_block(sc) for sc in shots[:cap]]
    return "\n\n".join(parts)


def _vllm_json_hint(backend: str) -> str:
    if not _is_local_backend(backend):
        return ""
    return (
        "\n\n# Model output requirement:\n"
        "# End with one JSON object containing \"movements\" (array). "
        "Optional # Q1–Q4 lines above JSON are allowed.\n"
    )


def _llm_generate(prompt: str, *, model: str, backend: str = "gemini", vlm: Any | None = None) -> str:
    if vlm is not None:
        return vlm.generate(prompt)
    if backend.lower() == "gemini":
        from google import genai

        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(model=model, contents=prompt)
        return (resp.text or "").strip()
    from vlm_client import VLMClient

    client = VLMClient(backend=backend, model=model)
    return client.generate(prompt)


def _llm_generate_many(
    prompts: list[str],
    *,
    model: str,
    backend: str = "gemini",
    vlm: Any | None = None,
) -> list[str]:
    if not prompts:
        return []
    from vlm_batch_util import vlm_generate_texts  # noqa: WPS433

    requests = [{"prompt": p} for p in prompts]
    return vlm_generate_texts(vlm, backend, requests)


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


def _build_exp1_prompt(*, cue: str, backend: str, prompt_path: Path | None = None) -> str:
    ppath = prompt_path or prompt_exp_path(1)
    template = ppath.read_text(encoding="utf-8")
    return (
        template.replace("{{CUE_CATALOG}}", _cue_catalog(cue=cue, backend=backend))
        .replace("{{FEW_SHOT_EXAMPLES}}", _fewshot_block(backend=backend))
        .replace("{{CUE_NAME}}", cue)
        + _vllm_json_hint(backend)
    )


def _build_exp7_prompt(
    *,
    cue: str,
    description: str,
    fixed_pose: dict[str, Any],
    backend: str,
    prompt_path: Path | None = None,
) -> str:
    ppath = prompt_path or prompt_exp_path(7)
    template = ppath.read_text(encoding="utf-8")
    return (
        template.replace("{{FEW_SHOT_EXAMPLES}}", _fewshot_block(max_examples=4, backend=backend))
        .replace("{{FIXED_FIRST_POSE_JSON}}", json.dumps(fixed_pose, ensure_ascii=False, indent=2))
        .replace("{{FIXED_START_POSE_JSON}}", json.dumps(fixed_pose, ensure_ascii=False, indent=2))
        .replace("{{CUE_NAME}}", cue)
        .replace("{{CUE_DESCRIPTION}}", description)
        + _vllm_json_hint(backend)
    )


def generate_exp1_row(
    *,
    cue: str,
    cue_idx: int,
    description: str,
    model: str,
    backend: str = "gemini",
    vlm: Any | None = None,
    out_path: Path | None = None,
    prompt_path: Path | None = None,
    raw_override: str | None = None,
) -> dict[str, Any]:
    from adhoc.generation.google_robot.legacy.config_gen_single_mobile import (  # noqa: WPS433
        _extract_reasoning_and_json,
        _sanitize_model_output,
        _validate_config,
        _validate_reasoning,
    )

    require_reasoning = _require_planning_comments(backend, env_key="EXP1_REQUIRE_REASONING")
    max_attempts = 3 if _is_local_backend(backend) else 2
    prompt = _build_exp1_prompt(cue=cue, backend=backend, prompt_path=prompt_path)
    validation_errors: list[str] = []
    reasoning_text = ""
    parsed: dict[str, Any] | None = None
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for attempt in range(max_attempts):
        if raw_override is not None and attempt == 0:
            raw = _sanitize_model_output(raw_override)
            raw_override = None
        else:
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
            if not raw.strip():
                validation_errors.append("model returned empty output")
            else:
                validation_errors.append(f"raw preview: {raw[:240]!r}")
            continue
        validation_errors = list(_validate_reasoning(reasoning_text)) if require_reasoning else []
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
    backend: str = "gemini",
    vlm: Any | None = None,
    out_path: Path | None = None,
    prompt_path: Path | None = None,
    raw_override: str | None = None,
) -> dict[str, Any]:
    from adhoc.generation.google_robot.legacy.config_gen_single_mobile import (  # noqa: WPS433
        _extract_reasoning_and_json,
        _sanitize_model_output,
        _validate_config,
        _validate_reasoning,
    )

    require_reasoning = _require_planning_comments(backend, env_key="EXP7_REQUIRE_REASONING")
    max_attempts = 3 if _is_local_backend(backend) else 2
    prompt = _build_exp7_prompt(
        cue=cue,
        description=description,
        fixed_pose=fixed_pose,
        backend=backend,
        prompt_path=prompt_path,
    )
    validation_errors: list[str] = []
    reasoning_text = ""
    tail: list[dict[str, Any]] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for attempt in range(max_attempts):
        if raw_override is not None and attempt == 0:
            raw = _sanitize_model_output(raw_override)
            raw_override = None
        else:
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
            if not raw.strip():
                validation_errors.append("model returned empty output")
            else:
                validation_errors.append(f"raw preview: {raw[:240]!r}")
            continue
        tail = parsed.get("movements") or parsed.get("movement_component") or []
        tail = _strip_leading_poses(tail)
        full = {
            "cue": cue,
            "movements": [_fixed_pose_step(fixed_pose)] + tail,
        }
        validation_errors = list(_validate_reasoning(reasoning_text)) if require_reasoning else []
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
    from vlm_client import vlm_batch_size  # noqa: WPS433

    if not GT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing ground truth: {GT_PATH}\n"
            "Run: python adhoc/generation/google_robot/build_gt_google_robot.py"
        )

    gt = _gt_by_cue()
    if not gt:
        raise FileNotFoundError(f"Empty ground truth: {GT_PATH}")

    work_names = manifest90_cue_names()
    if not work_names:
        raise FileNotFoundError("No cues in pilot100_manifest.tsv (manifest90)")

    cue_filter = os.getenv("CUES", "").strip()
    if cue_filter:
        want = {c.strip() for c in cue_filter.split(",") if c.strip()}
        work_names = [c for c in work_names if c in want]
        if not work_names:
            raise ValueError(f"No manifest cues matched CUES={cue_filter!r}")

    shots_by_cue = {
        str(r.get("cue", "")): r
        for r in load_config_list(_REPO / "data/seed/shots/google_robot/shot_configs_pilot40_mobile.json")
        if r.get("cue")
    }

    if not out_path.is_file():
        save_json(out_path, [])

    existing = {r["cue"]: r for r in load_config_list(out_path) if r.get("cue")}
    catalog = yaml.safe_load(CUES_YAML.read_text(encoding="utf-8")) if CUES_YAML.is_file() else {}
    batch_size = vlm_batch_size(backend) if vlm is not None else 1

    def _desc(cue: str, gt_row: dict[str, Any]) -> str:
        if gt_row.get("description"):
            return str(gt_row["description"])
        for grp in ("iconic", "contextual"):
            block = catalog.get(grp) if isinstance(catalog, dict) else None
            if isinstance(block, dict) and cue in block:
                return str(block[cue])
        shot = shots_by_cue.get(cue) or {}
        return str(shot.get("description") or shot.get("cue_text") or "")

    work: list[dict[str, Any]] = []
    for cue in work_names:
        if resume and row_generation_done(existing.get(cue)):
            continue
        gt_row = gt.get(cue)
        if not gt_row:
            print(f"[gen] exp{exp_id} SKIP {cue}: missing from {GT_PATH.name}", flush=True)
            continue
        cue_idx = int(gt_row.get("cue_idx", 0))
        description = _desc(cue, gt_row)
        item: dict[str, Any] = {
            "cue": cue,
            "cue_idx": cue_idx,
            "description": description,
            "shot_row": shots_by_cue.get(cue, {}),
        }
        if exp_id == "7":
            ev = gt_row
            gt_pairs = _parse_gt_poses(str(ev.get("pose_gt") or ev.get("groundtruth", "")))
            ref_pose = {}
            for step in (shots_by_cue.get(cue) or {}).get("movements") or []:
                if step.get("type") == "pose":
                    ref_pose = (step.get("parameters") or {}).get("pose") or {}
                    break
            if gt_pairs:
                d, g = gt_pairs[0]
                fixed = _mobile_pose_from_gt_pair(d, g, ref_pose)
            else:
                fixed = ref_pose or {
                    "torso_height": "mid",
                    "arm_position": "front",
                    "gripper_orientation": "horizontal",
                    "head": "center",
                }
            item["fixed_pose"] = fixed
        work.append(item)

    def _run_one(item: dict[str, Any], *, raw_override: str | None = None) -> None:
        cue = item["cue"]
        try:
            if exp_id == "1":
                generate_exp1_row(
                    cue=cue,
                    cue_idx=item["cue_idx"],
                    description=item["description"],
                    model=model,
                    backend=backend,
                    vlm=vlm,
                    out_path=out_path,
                    raw_override=raw_override,
                )
            elif exp_id == "7":
                generate_exp7_row(
                    cue=cue,
                    cue_idx=item["cue_idx"],
                    description=item["description"],
                    fixed_pose=item["fixed_pose"],
                    model=model,
                    backend=backend,
                    vlm=vlm,
                    out_path=out_path,
                    raw_override=raw_override,
                )
            else:
                raise ValueError(f"Unsupported generation exp: {exp_id}")
            print(f"[gen] exp{exp_id} OK {cue}", flush=True)
        except Exception as e:
            print(f"[gen] exp{exp_id} FAIL {cue}: {e}", flush=True)

    i = 0
    while i < len(work):
        chunk = work[i : i + batch_size]
        i += len(chunk)
        if batch_size > 1 and vlm is not None and len(chunk) > 1:
            if exp_id == "1":
                prompts = [_build_exp1_prompt(cue=item["cue"], backend=backend) for item in chunk]
            else:
                prompts = [
                    _build_exp7_prompt(
                        cue=item["cue"],
                        description=item["description"],
                        fixed_pose=item["fixed_pose"],
                        backend=backend,
                    )
                    for item in chunk
                ]
            raws = _llm_generate_many(prompts, model=model, backend=backend, vlm=vlm)
            for item, raw in zip(chunk, raws):
                _run_one(item, raw_override=raw)
                if delay > 0:
                    time.sleep(delay)
        else:
            for item in chunk:
                _run_one(item)
                if delay > 0:
                    time.sleep(delay)
