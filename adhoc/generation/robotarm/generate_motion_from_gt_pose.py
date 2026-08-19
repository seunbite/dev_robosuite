#!/usr/bin/env python3
"""Generate motion configs with human GT fixed as first pose; LLM generates tail movements."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import fire

_ROBOTARM = Path(__file__).resolve().parent
_REPO = _ROBOTARM.parents[2]
for p in (_REPO, _ROBOTARM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from adhoc.generation.embodiment_sources import resolve_embodiment_paths  # noqa: E402
from legacy.config_gen_single import (  # noqa: E402
    _extract_reasoning_and_json,
    _format_example_block,
    _load_json_list,
    _resolve_shots_json,
    _sanitize_model_output,
    _validate_config,
)
from legacy.path_ee_ik import validate_path_parameters  # noqa: E402
from generate_pose_group_tiles import _load_entries, _select_xyz_tertile_balanced  # noqa: E402

CONSOLIDATED = _REPO / "data/results/verify/pilot40_pose_eval_consolidated.json"
TILE_PICK = _REPO / "data/results/verify/pose_tile_pick_by_group.json"
JSONL = _REPO / "data/seed/_remainder/closest_poses_results.jsonl"
DEFAULT_OUT = (
    _REPO
    / "data/results/motion_configs/manipulator"
    / "motion_configs_prompt_v19_gt_fixed_pose_pilot40.json"
)
PROMPT_GT = _REPO / "data/seed/prompt/manipulator/prompt_gt_fixed_first_pose.txt"

CONFIG_PATHS = [
    _REPO / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot10.json",
    _REPO / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot20_more.json",
]


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

    entries = [e for e in _load_entries(JSONL) if e.get("robot") == "IIWA"]
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


def _first_pose_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    for step in cfg.get("movements", []):
        if step.get("type") == "pose":
            return deepcopy(step.get("parameters", {}).get("pose", {}) or {})
    return {"dir": "front", "x": 50, "y": 50, "z": 50, "gripper_orientation": "horizontal"}


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


def _pose_plus_tail_examples_from_shots(shots: list[dict], max_n: int = 6) -> str:
    """Few-shot: full config with leading pose + follow-on steps (model infers tail from pose)."""
    parts: list[str] = []
    picked = [c for c in shots if c.get("state") in ("handmade", "choreography")][:max_n]
    for ex in picked:
        slim = {k: v for k, v in ex.items() if k not in ("planning_shot",)}
        mov = slim.get("movements", [])
        if not mov or mov[0].get("type") != "pose":
            continue
        parts.append(
            _format_example_block(
                slim,
                "# EXAMPLE (start pose + follow-on steps; mirror this pose-then-motion pattern):",
            )
        )
    return "\n\n".join(parts)


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
    if not movements:
        return ["Tail must include at least one movement/path step"]
    if len(movements) > 3:
        return [f"Tail has {len(movements)} steps; max 3 allowed"]
    if not any(m.get("type") in ("movement", "path") for m in movements):
        return ["Tail must include at least one movement or path step"]
    # _validate_config requires a leading pose; use a dummy for tail-only checks.
    dummy_pose = {"type": "pose", "parameters": {"pose": {"dir": "front", "x": 50, "y": 50, "z": 50}}}
    errs = _validate_config(
        {"movements": [dummy_pose, *movements], "cue": cue_name},
        cue_name=cue_name,
    )
    skip = ("First step must be type 'pose'", "Config is pose-only")
    tail_errs = [e for e in errs if not any(s in e for s in skip)]
    for step in movements:
        if step.get("type") == "path":
            tail_errs.extend(validate_path_parameters(step.get("parameters", {})))
    return tail_errs


def _normalize_reasoning_gt(reasoning_text: str) -> str:
    lines = [ln.strip() for ln in reasoning_text.splitlines() if ln.strip().startswith("#")]
    q_lines = [ln for ln in lines if re.match(r"#\s*Q[1-4]:", ln)]
    if len(q_lines) >= 3:
        return "\n".join(q_lines[:3])
    return "\n".join(lines[:3]) if lines else ""


def generate_one(
    *,
    cue_name: str,
    cue_idx: int,
    groundtruth: str,
    ref_cfg: dict[str, Any],
    client: Any,
    model_name: str,
    prompt_template: str,
    examples_str: str,
    tile_pick: dict[tuple[str, str], int],
    require_reasoning: bool = True,
) -> dict[str, Any]:
    from google.genai import types

    gt_d, gt_g = _parse_gt_poses(groundtruth)[0]
    fixed = _build_fixed_pose(gt_d, gt_g, _first_pose_from_cfg(ref_cfg), tile_pick)
    fixed_step = {
        "type": "pose",
        "parameters": {"pose": fixed, "speed": 1.0, "hold_time": 0.0},
    }

    prompt = prompt_template.replace("{{FEW_SHOT_EXAMPLES}}", examples_str)
    prompt = prompt.replace("{{CUE_NAME}}", cue_name)
    prompt = prompt.replace("{{FIXED_FIRST_POSE_JSON}}", json.dumps(fixed, indent=2))

    validation_errors: list[str] = []
    reasoning_text = ""
    tail_cfg: dict[str, Any] | None = None
    final_tail: list[dict[str, Any]] = []

    for attempt in range(3):
        attempt_prompt = prompt
        if attempt > 0 and validation_errors:
            attempt_prompt += "\n\n# Fix these issues:\n" + "\n".join(
                f"# - {e}" for e in validation_errors
            )

        resp = client.models.generate_content(model=model_name, contents=attempt_prompt)
        raw = _sanitize_model_output((resp.text or "").strip())
        try:
            reasoning_text, parsed = _extract_reasoning_and_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            validation_errors = [f"JSON parse failed: {e}"]
            continue

        reasoning_text = _normalize_reasoning_gt(reasoning_text)
        tail_movements = _strip_leading_poses(parsed.get("movements", []))
        _clamp_speeds(tail_movements)
        full_movements = [fixed_step, *tail_movements]
        full = {
            "cue": cue_name,
            "description": parsed.get("description", ""),
            "movements": full_movements,
        }
        validation_errors = []
        if require_reasoning and not reasoning_text.strip():
            validation_errors.append("Missing planning comments before JSON")
        validation_errors.extend(_validate_tail(tail_movements, cue_name=cue_name))
        validation_errors.extend(_validate_config(full, cue_name=cue_name))
        if not validation_errors:
            tail_cfg = parsed
            final_tail = tail_movements
            break

    if tail_cfg is None:
        raise ValueError(f"Failed for {cue_name}: {validation_errors}")

    out = {
        "idx": cue_idx,
        "cue": cue_name,
        "description": tail_cfg.get("description", ""),
        "groundtruth": groundtruth,
        "gt_fixed_first_pose": fixed,
        "movements": [fixed_step, *final_tail],
        "state": "gt_fixed_pose_generation",
        "model": model_name,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generation_mode": "gt_fixed_first_pose_tail_from_llm",
    }
    if reasoning_text:
        out["reasoning"] = reasoning_text
    return out


def run(
    out_json: str | None = None,
    model: str = "gemini-2.5-pro",
    delay: float = 2.0,
    max_retries: int = 1,
    cues: str | None = None,
    run_render: bool = False,
) -> None:
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY.")

    from google import genai

    prompt_path, shots_path, _, _ = resolve_embodiment_paths("manipulator")
    out_path = Path(out_json) if out_json else DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.is_file():
        out_path.write_text("[]\n", encoding="utf-8")

    consolidated = json.loads(CONSOLIDATED.read_text(encoding="utf-8"))["rows"]
    cfg_by_cue: dict[str, dict[str, Any]] = {}
    for p in CONFIG_PATHS:
        for row in json.loads(p.read_text(encoding="utf-8")):
            cfg_by_cue[row["cue"]] = row

    work = consolidated
    if cues:
        cue_str = ",".join(cues) if isinstance(cues, (list, tuple)) else str(cues)
        names = {c.strip() for c in cue_str.split(",") if c.strip()}
        work = [r for r in work if r["cue"] in names]

    prompt_template = PROMPT_GT.read_text(encoding="utf-8")
    shots = _load_json_list(_resolve_shots_json(str(prompt_path), "auto"))
    examples_str = _pose_plus_tail_examples_from_shots(shots)
    tile_pick = _load_tile_pick()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))

    generated = _load_json_list(str(out_path))
    failed: list[str] = []

    print(f"Generating {len(work)} configs (GT fixed first pose + LLM tail)")
    print(f"Out: {out_path}")

    for ev in work:
        cue = ev["cue"]
        idx = int(ev["cue_idx"])
        ok = False
        last_err = ""
        for attempt in range(max_retries + 1):
            try:
                row = generate_one(
                    cue_name=cue,
                    cue_idx=idx,
                    groundtruth=ev["groundtruth"],
                    ref_cfg=cfg_by_cue[cue],
                    client=client,
                    model_name=model,
                    prompt_template=prompt_template,
                    examples_str=examples_str,
                    tile_pick=tile_pick,
                )
                generated = [c for c in generated if c.get("cue") != cue]
                generated.append(row)
                out_path.write_text(
                    json.dumps(generated, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                ok = True
                print(f"OK {cue} gt={_parse_gt_poses(ev['groundtruth'])[0]} steps={len(row['movements'])}")
                break
            except Exception as e:
                last_err = str(e)
                if attempt < max_retries:
                    time.sleep(delay * (2**attempt))
        if not ok:
            failed.append(f"{cue}: {last_err[:200]}")
            print(f"FAIL {cue}: {last_err[:300]}")

    print(f"Done failures={len(failed)}/{len(work)}")
    for f in failed:
        print(" ", f)

    if run_render and failed != list(work):
        from render import run as render_run  # noqa: E402

        render_run(config_json=str(out_path), sim_robot="IIWA", skip_existing=False)


if __name__ == "__main__":
    fire.Fire(run)
