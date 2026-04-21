import copy
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

import fire
import yaml
from google import genai
from google.genai import types

from config_persona_variation import _esc, _open_preview

DEFAULT_SELECTED_TAGS = ["joyful", "calm", "sad", "confident", "hesitant"]


SPEC_PROMPT = """You are an expert in expressive robot motion styling.

You are given:
1. a base motion config
2. a single persona tag and its category

Important:
- Do NOT redesign the config structure.
- Do NOT add or remove pose / movement / path elements.
- Do NOT change the order of existing motion elements.

You may only decide:
1. initial_pose_offset: x, y, z offsets for the FIRST pose only
2. parameter edits: speed, hold_time, repetition, angle size
3. render modifiers:
   - hesitation
   - elegant_curve
   - zittering

Those render modifiers will be applied by the renderer at execution time without changing the config structure.

Return exactly one JSON object in this schema:
{{
  "tag_summary": "...",
  "initial_pose_offset": {{"x": int, "y": int, "z": int}},
  "speed_scale": float,
  "hold_scale": float,
  "repetition_delta": int,
  "angle_scale": float,
  "render_modifiers": {{
    "hesitation": float,
    "elegant_curve": float,
    "zittering": float
  }}
}}

Constraints:
- x/y/z offsets should usually stay within [-12, 12]
- speed_scale in [0.6, 1.6]
- hold_scale in [0.5, 2.0]
- repetition_delta in [-1, 2]
- angle_scale in [0.7, 1.5]
- hesitation / elegant_curve / zittering in [0.0, 1.0]
- Preserve cue recognizability
- Keep the tag effect visible, not negligible

Category: {tag_category}
Tag: {tag_name}
Base config:
{base_config_json}
"""


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("No JSON object found in model output")


def _load_tags(tag_catalog_path: str) -> List[Dict[str, str]]:
    with open(tag_catalog_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    tags = []
    for category, items in data.items():
        for tag in items:
            tags.append({"category": category, "tag": str(tag)})
    return tags


def _load_cue_idxs(cue_subset_path: str) -> List[int]:
    with open(cue_subset_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [int(x) for x in data.get("cue_idxs", [])]


def _parse_selected_tags(selected_tags: Any) -> List[str]:
    if selected_tags is None:
        return list(DEFAULT_SELECTED_TAGS)
    if isinstance(selected_tags, str):
        stripped = selected_tags.strip()
        if not stripped:
            return list(DEFAULT_SELECTED_TAGS)
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [part.strip() for part in stripped.split(",") if part.strip()]
        if not isinstance(parsed, list):
            raise ValueError("selected_tags string must be a JSON list or comma-separated names")
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(selected_tags, list):
        return [str(item).strip() for item in selected_tags if str(item).strip()]
    raise ValueError("selected_tags must be None, a list, or a string")


def _filter_tags(all_tags: List[Dict[str, str]], selected_tags: Any) -> List[Dict[str, str]]:
    requested = _parse_selected_tags(selected_tags)
    by_name = {tag["tag"]: tag for tag in all_tags}
    missing = [name for name in requested if name not in by_name]
    if missing:
        available = ", ".join(sorted(by_name))
        raise ValueError(f"Unknown selected_tags: {missing}. Available tags: {available}")
    return [by_name[name] for name in requested]


def _extract_retry_delay_seconds(message: str) -> float | None:
    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", message, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"'retryDelay':\s*'([0-9]+)s'", message)
    if match:
        return float(match.group(1))
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _normalize_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    offset = spec.get("initial_pose_offset", {}) or {}
    render = spec.get("render_modifiers", {}) or {}
    return {
        "tag_summary": str(spec.get("tag_summary", "")),
        "initial_pose_offset": {
            "x": int(round(_clamp(float(offset.get("x", 0)), -12, 12))),
            "y": int(round(_clamp(float(offset.get("y", 0)), -12, 12))),
            "z": int(round(_clamp(float(offset.get("z", 0)), -12, 12))),
        },
        "speed_scale": _clamp(float(spec.get("speed_scale", 1.0)), 0.6, 1.6),
        "hold_scale": _clamp(float(spec.get("hold_scale", 1.0)), 0.5, 2.0),
        "repetition_delta": int(round(_clamp(float(spec.get("repetition_delta", 0)), -1, 2))),
        "angle_scale": _clamp(float(spec.get("angle_scale", 1.0)), 0.7, 1.5),
        "render_modifiers": {
            "hesitation": _clamp(float(render.get("hesitation", 0.0)), 0.0, 1.0),
            "elegant_curve": _clamp(float(render.get("elegant_curve", 0.0)), 0.0, 1.0),
            "zittering": _clamp(float(render.get("zittering", 0.0)), 0.0, 1.0),
        },
    }


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "item"


def _apply_spec(base_cfg: Dict[str, Any], spec: Dict[str, Any], tag_category: str, tag_name: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    movements = cfg.get("movements", [])
    first_pose = next((m for m in movements if m.get("type") == "pose"), None)
    if first_pose:
        pose = first_pose.setdefault("parameters", {}).setdefault("pose", {})
        for axis in ("x", "y", "z"):
            if axis in pose:
                pose[axis] = int(round(_clamp(float(pose[axis]) + spec["initial_pose_offset"][axis], 0, 100)))
        if "speed" in first_pose["parameters"]:
            first_pose["parameters"]["speed"] = round(_clamp(float(first_pose["parameters"]["speed"]) * spec["speed_scale"], 0.5, 4.0), 3)
        if "hold_time" in first_pose["parameters"]:
            first_pose["parameters"]["hold_time"] = round(max(0.0, float(first_pose["parameters"]["hold_time"]) * spec["hold_scale"]), 3)

    for step in movements:
        params = step.setdefault("parameters", {})
        if step.get("type") == "movement":
            if "repetition" in params:
                params["repetition"] = max(1, int(params["repetition"]) + spec["repetition_delta"])
            for direction in params.get("directions", []):
                if "speed" in direction:
                    direction["speed"] = round(_clamp(float(direction["speed"]) * spec["speed_scale"], 0.5, 4.0), 3)
                if "hold_time" in direction:
                    direction["hold_time"] = round(max(0.0, float(direction["hold_time"]) * spec["hold_scale"]), 3)
                if "degrees" in direction:
                    direction["degrees"] = {
                        axis: round(_clamp(float(val) * spec["angle_scale"], -65, 65), 3)
                        for axis, val in direction["degrees"].items()
                    }
        elif step.get("type") == "path":
            if "speed" in params:
                params["speed"] = round(_clamp(float(params["speed"]) * spec["speed_scale"], 0.5, 4.0), 3)
        elif step.get("type") == "pose":
            if "speed" in params:
                params["speed"] = round(_clamp(float(params["speed"]) * spec["speed_scale"], 0.5, 4.0), 3)
            if "hold_time" in params:
                params["hold_time"] = round(max(0.0, float(params["hold_time"]) * spec["hold_scale"]), 3)

    cfg["persona_tag"] = {"category": tag_category, "name": tag_name}
    cfg["render_modifiers"] = spec["render_modifiers"]
    cfg["tag_summary"] = spec["tag_summary"]
    return cfg


def _find_latest_render_gif(render_dir: str, cue_name: str) -> str | None:
    safe_cue = cue_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    patterns = [
        os.path.join(render_dir, f"*_{safe_cue}_tiled.gif"),
        os.path.join(render_dir, f"*_{safe_cue}_*.gif"),
    ]
    matches: List[str] = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            if os.path.basename(path).endswith("_preview.gif"):
                continue
            matches.append(path)
    if not matches:
        return None
    return sorted(set(matches), key=os.path.getmtime, reverse=True)[0]


def _render_config(
    *,
    script_dir: str,
    config_path: str,
    cue_idx: int,
    cue_name: str,
    robot: str,
    output_dir: str,
    hz: int,
    top_k: int,
    preview_speed_scale: float,
    preview_hold_scale: float,
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
        f"--hz={hz}",
        f"--top_k={top_k}",
        f"--preview_speed_scale={preview_speed_scale}",
        f"--preview_hold_scale={preview_hold_scale}",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        stderr_tail = result.stderr.strip().splitlines()[-5:] if result.stderr else []
        print(f"  render failed for c{cue_idx} ({os.path.basename(config_path)})")
        for line in stderr_tail:
            print(f"    {line}")
        return None
    return _find_latest_render_gif(os.path.join(output_dir, robot), cue_name)


def _write_render_input(output_dir: str, tag_key: str, cue_idx: int, edited_config: Dict[str, Any]) -> str:
    render_input_dir = os.path.join(output_dir, "_render_inputs", _safe_name(tag_key))
    os.makedirs(render_input_dir, exist_ok=True)
    config_path = os.path.join(render_input_dir, f"c{cue_idx}.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump([edited_config], f, indent=2, ensure_ascii=False)
    return config_path


def _render_pending_rows(
    *,
    pending_rows: List[Dict[str, Any]],
    preview_rows: List[Dict[str, Any]],
    preview_seen: set[tuple[str, int]],
    base_render_cache: Dict[int, str | None],
    script_dir: str,
    input_json: str,
    render_root: str,
    render_robot: str,
    render_hz: int,
    render_top_k: int,
    render_speed_scale: float,
    render_hold_scale: float,
    output_dir: str,
    deadline: float | None = None,
) -> None:
    while pending_rows and (deadline is None or time.time() < deadline):
        row = pending_rows[0]
        row_key = (f"{row['tag_category']}__{row['tag_name']}", int(row["cue_idx"]))
        if row_key in preview_seen:
            pending_rows.pop(0)
            continue

        cue_idx = int(row["cue_idx"])
        cue = str(row["cue"])
        tag_key = row_key[0]

        if cue_idx not in base_render_cache:
            base_render_cache[cue_idx] = _render_config(
                script_dir=script_dir,
                config_path=os.path.abspath(input_json),
                cue_idx=cue_idx,
                cue_name=cue,
                robot=render_robot,
                output_dir=os.path.join(render_root, "base"),
                hz=render_hz,
                top_k=render_top_k,
                preview_speed_scale=render_speed_scale,
                preview_hold_scale=render_hold_scale,
            )

        render_input_path = _write_render_input(output_dir, tag_key, cue_idx, row["edited_config"])
        tag_gif = _render_config(
            script_dir=script_dir,
            config_path=render_input_path,
            cue_idx=cue_idx,
            cue_name=cue,
            robot=render_robot,
            output_dir=os.path.join(render_root, _safe_name(tag_key)),
            hz=render_hz,
            top_k=render_top_k,
            preview_speed_scale=render_speed_scale,
            preview_hold_scale=render_hold_scale,
        )

        ready_row = dict(row)
        ready_row["base_gif"] = base_render_cache[cue_idx]
        ready_row["tag_gif"] = tag_gif
        preview_rows.append(ready_row)
        preview_seen.add(row_key)
        pending_rows.pop(0)


def _write_dataset_html(
    *,
    output_path: str,
    rows: List[Dict[str, Any]],
    render_root: str,
) -> str:
    html_dir = os.path.dirname(output_path)
    parts = ["""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Persona Tag Dataset Preview</title>
<style>
:root {
  --bg: #f5f6f8; --surface: #ffffff; --surface2: #eef1f5;
  --border: #d5d9e0; --text: #1f2328; --text2: #5b6472; --accent: #0969da;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, 'SF Pro Text', 'Segoe UI', sans-serif; }
.wrap { max-width: 1800px; margin: 0 auto; padding: 24px; }
.hero { margin-bottom: 18px; }
.hero h1 { margin: 0 0 8px; font-size: 28px; }
.hero p { margin: 0; color: var(--text2); }
.row { margin-bottom: 18px; background: var(--surface); border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }
.hdr { padding: 12px 14px; background: var(--surface2); border-bottom: 1px solid var(--border); font-weight: 700; }
.hdr .idx { color: var(--accent); margin-right: 8px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; }
.col { padding: 14px; border-right: 1px solid var(--border); }
.col:last-child { border-right: 0; }
.label { margin-bottom: 8px; font-size: 13px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; }
.gif { min-height: 220px; display: flex; align-items: center; justify-content: center; background: var(--surface2); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; margin-bottom: 10px; }
.gif img { display: block; max-width: 100%; }
.na { color: var(--text2); font-style: italic; }
pre { margin: 0; white-space: pre-wrap; word-break: break-word; background: var(--surface2); border-radius: 8px; padding: 10px; font-size: 12px; border: 1px solid var(--border); }
@media (max-width: 1180px) { .grid { grid-template-columns: 1fr; } .col { border-right: 0; border-bottom: 1px solid var(--border); } .col:last-child { border-bottom: 0; } }
</style>
</head>
<body>
<div class="wrap">
  <section class="hero">
    <h1>Persona Tag Dataset Preview</h1>
    <p>Base vs tag-conditioned render preview.</p>
  </section>
"""]
    for row in rows:
        base_gif = row.get("base_gif")
        tag_gif = row.get("tag_gif")
        base_rel = _esc(os.path.relpath(base_gif, html_dir)) if base_gif else ""
        tag_rel = _esc(os.path.relpath(tag_gif, html_dir)) if tag_gif else ""
        parts.append(f"""
  <section class="row">
    <div class="hdr"><span class="idx">c{row['cue_idx']}</span>{_esc(row['cue'])} · {_esc(row['tag_category'])}:{_esc(row['tag_name'])}</div>
    <div class="grid">
      <div class="col">
        <div class="label">Base</div>
        <div class="gif">{f'<img src="{base_rel}" alt="base">' if base_gif else '<span class="na">No render found</span>'}</div>
        <pre>{_esc(json.dumps(row['base_config']['movements'], indent=2, ensure_ascii=False))}</pre>
      </div>
      <div class="col">
        <div class="label">Tag Conditioned</div>
        <div class="gif">{f'<img src="{tag_rel}" alt="tag render">' if tag_gif else '<span class="na">No render found</span>'}</div>
        <pre>{_esc(json.dumps(row['variation_spec'], indent=2, ensure_ascii=False))}</pre>
      </div>
    </div>
  </section>
""")
    parts.append("</div></body></html>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    return output_path


def _persist_progress(
    *,
    output_dir: str,
    dataset_rows: List[Dict[str, Any]],
    all_configs: List[Dict[str, Any]],
    tag_to_configs: Dict[str, List[Dict[str, Any]]],
    input_json: str,
    tag_catalog_path: str,
    cue_subset_path: str,
    selected_tags: List[str],
    tags_count: int,
    cues_count: int,
    total_samples: int,
    succeeded: int,
    failed_items: List[Dict[str, Any]],
    model_name: str,
    temperature: float,
    request_sleep_sec: float,
    retry_backoff_sec: float,
    max_retries_per_item: int,
) -> tuple[str, str, str, Dict[str, str]]:
    jsonl_path = os.path.join(output_dir, "dataset.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in dataset_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    config_path = os.path.join(output_dir, "configs.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(all_configs, f, indent=2, ensure_ascii=False)

    tag_config_dir = os.path.join(output_dir, "tag_configs")
    os.makedirs(tag_config_dir, exist_ok=True)
    tag_config_paths: Dict[str, str] = {}
    for tag_key, configs in tag_to_configs.items():
        path = os.path.join(tag_config_dir, f"{_safe_name(tag_key)}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(configs, f, indent=2, ensure_ascii=False)
        tag_config_paths[tag_key] = path

    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_json": input_json,
                "tag_catalog_path": tag_catalog_path,
                "cue_subset_path": cue_subset_path,
                "selected_tags": selected_tags,
                "tags": tags_count,
                "cues": cues_count,
                "samples": len(dataset_rows),
                "planned_samples": total_samples,
                "succeeded_samples": succeeded,
                "failed_samples": len(failed_items),
                "failed_items": failed_items,
                "model_name": model_name,
                "temperature": temperature,
                "request_sleep_sec": request_sleep_sec,
                "retry_backoff_sec": retry_backoff_sec,
                "max_retries_per_item": max_retries_per_item,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return jsonl_path, config_path, meta_path, tag_config_paths


def main(
    input_json: str = "data/seed/motion_configs_prompt_v18.json",
    tag_catalog_path: str = "data/seed/persona_tag_catalog.yml",
    cue_subset_path: str = "data/seed/persona_cue_subset_v1.yml",
    output_dir: str = "data/seed/persona_tag_dataset_v1",
    model_name: str = "gemini-2.5-flash",
    selected_tags: Any = None,
    temperature: float = 0.3,
    request_sleep_sec: float = 13.0,
    retry_backoff_sec: float = 35.0,
    max_retries_per_item: int = 6,
    render_html: bool = True,
    open_html: bool = True,
    render_robot: str = "IIWA",
    render_hz: int = 10,
    render_top_k: int = 1,
    render_speed_scale: float = 0.8,
    render_hold_scale: float = 1.0,
):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable.")

    with open(input_json, "r", encoding="utf-8") as f:
        base_data = json.load(f)
    cue_idxs = set(_load_cue_idxs(cue_subset_path))
    base_subset = [cfg for cfg in base_data if cfg.get("idx") in cue_idxs]
    tags = _filter_tags(_load_tags(tag_catalog_path), selected_tags)

    os.makedirs(output_dir, exist_ok=True)
    client = genai.Client(api_key=api_key)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    total_samples = len(tags) * len(base_subset)
    render_root = os.path.join(output_dir, "renders")

    print("Persona tag dataset generation")
    print(f"selected_tags={','.join(tag['tag'] for tag in tags)}")
    print(f"cue_count={len(base_subset)} total_samples={total_samples}")
    print(f"model={model_name} temperature={temperature} request_sleep_sec={request_sleep_sec}")

    dataset_rows = []
    all_configs = []
    tag_to_configs: Dict[str, List[Dict[str, Any]]] = {}
    pending_render_rows: List[Dict[str, Any]] = []
    preview_rows: List[Dict[str, Any]] = []
    preview_seen: set[tuple[str, int]] = set()
    base_render_cache: Dict[int, str | None] = {}
    failed_items: List[Dict[str, Any]] = []
    completed = 0
    succeeded = 0
    for tag_info in tags:
        tag_category = tag_info["category"]
        tag_name = tag_info["tag"]
        tag_key = f"{tag_category}__{tag_name}"
        tag_to_configs[tag_key] = []
        print(f"\n=== {tag_category}:{tag_name} ===")
        for cfg in base_subset:
            completed += 1
            print(f"{completed}/{total_samples} {tag_category}:{tag_name} c{cfg['idx']}")
            prompt = SPEC_PROMPT.format(
                tag_category=tag_category,
                tag_name=tag_name,
                base_config_json=json.dumps(cfg, indent=2, ensure_ascii=False),
            )
            attempts = 0
            while True:
                attempts += 1
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=float(temperature)),
                    )
                    spec = _normalize_spec(_extract_json(response.text))
                    edited = _apply_spec(cfg, spec, tag_category, tag_name)
                    row = {
                        "cue_idx": cfg["idx"],
                        "cue": cfg["cue"],
                        "tag_category": tag_category,
                        "tag_name": tag_name,
                        "status": "success",
                        "base_config": {
                            "description": cfg.get("description", ""),
                            "movements": cfg.get("movements", []),
                        },
                        "variation_spec": spec,
                        "edited_config": edited,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    dataset_rows.append(row)
                    all_configs.append(edited)
                    tag_to_configs[tag_key].append(edited)
                    succeeded += 1
                    print(f"  success c{cfg['idx']}: {cfg['cue']}")
                    if render_html:
                        pending_render_rows.append(row)
                    if request_sleep_sec > 0:
                        print(f"  cooldown {request_sleep_sec:.1f}s")
                        deadline = time.time() + float(request_sleep_sec)
                        _persist_progress(
                            output_dir=output_dir,
                            dataset_rows=dataset_rows,
                            all_configs=all_configs,
                            tag_to_configs=tag_to_configs,
                            input_json=input_json,
                            tag_catalog_path=tag_catalog_path,
                            cue_subset_path=cue_subset_path,
                            selected_tags=[tag["tag"] for tag in tags],
                            tags_count=len(tags),
                            cues_count=len(base_subset),
                            total_samples=total_samples,
                            succeeded=succeeded,
                            failed_items=failed_items,
                            model_name=model_name,
                            temperature=temperature,
                            request_sleep_sec=request_sleep_sec,
                            retry_backoff_sec=retry_backoff_sec,
                            max_retries_per_item=max_retries_per_item,
                        )
                        if render_html:
                            _render_pending_rows(
                                pending_rows=pending_render_rows,
                                preview_rows=preview_rows,
                                preview_seen=preview_seen,
                                base_render_cache=base_render_cache,
                                script_dir=script_dir,
                                input_json=input_json,
                                render_root=render_root,
                                render_robot=render_robot,
                                render_hz=render_hz,
                                render_top_k=render_top_k,
                                render_speed_scale=render_speed_scale,
                                render_hold_scale=render_hold_scale,
                                output_dir=output_dir,
                                deadline=deadline,
                            )
                        remaining = max(0.0, deadline - time.time())
                        if remaining > 0:
                            time.sleep(remaining)
                    break
                except Exception as exc:
                    error_text = str(exc)
                    if _is_rate_limit_error(exc) and attempts <= int(max_retries_per_item):
                        retry_sec = _extract_retry_delay_seconds(error_text) or float(retry_backoff_sec)
                        print(f"  rate limited; retry {attempts}/{max_retries_per_item} in {retry_sec:.1f}s")
                        time.sleep(retry_sec)
                        continue
                    failed_row = {
                        "cue_idx": cfg["idx"],
                        "cue": cfg["cue"],
                        "tag_category": tag_category,
                        "tag_name": tag_name,
                        "status": "failed",
                        "error": error_text,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    dataset_rows.append(failed_row)
                    failed_items.append(failed_row)
                    print(f"  failed c{cfg['idx']}: {cfg['cue']}")
                    if error_text:
                        print(f"    {error_text.splitlines()[-1]}")
                    _persist_progress(
                        output_dir=output_dir,
                        dataset_rows=dataset_rows,
                        all_configs=all_configs,
                        tag_to_configs=tag_to_configs,
                        input_json=input_json,
                        tag_catalog_path=tag_catalog_path,
                        cue_subset_path=cue_subset_path,
                        selected_tags=[tag["tag"] for tag in tags],
                        tags_count=len(tags),
                        cues_count=len(base_subset),
                        total_samples=total_samples,
                        succeeded=succeeded,
                        failed_items=failed_items,
                        model_name=model_name,
                        temperature=temperature,
                        request_sleep_sec=request_sleep_sec,
                        retry_backoff_sec=retry_backoff_sec,
                        max_retries_per_item=max_retries_per_item,
                    )
                    break

    jsonl_path, config_path, meta_path, tag_config_paths = _persist_progress(
        output_dir=output_dir,
        dataset_rows=dataset_rows,
        all_configs=all_configs,
        tag_to_configs=tag_to_configs,
        input_json=input_json,
        tag_catalog_path=tag_catalog_path,
        cue_subset_path=cue_subset_path,
        selected_tags=[tag["tag"] for tag in tags],
        tags_count=len(tags),
        cues_count=len(base_subset),
        total_samples=total_samples,
        succeeded=succeeded,
        failed_items=failed_items,
        model_name=model_name,
        temperature=temperature,
        request_sleep_sec=request_sleep_sec,
        retry_backoff_sec=retry_backoff_sec,
        max_retries_per_item=max_retries_per_item,
    )

    print(f"\nSaved dataset rows: {len(dataset_rows)}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {len(failed_items)}")
    print(f"JSONL: {jsonl_path}")
    print(f"Configs: {config_path}")
    print(f"Meta: {meta_path}")

    if render_html:
        print("\nRendering HTML preview...")
        success_rows = [row for row in dataset_rows if row.get("status") == "success"]
        for row in success_rows:
            row_key = (f"{row['tag_category']}__{row['tag_name']}", int(row["cue_idx"]))
            if row_key not in preview_seen:
                pending_render_rows.append(row)
        _render_pending_rows(
            pending_rows=pending_render_rows,
            preview_rows=preview_rows,
            preview_seen=preview_seen,
            base_render_cache=base_render_cache,
            script_dir=script_dir,
            input_json=input_json,
            render_root=render_root,
            render_robot=render_robot,
            render_hz=render_hz,
            render_top_k=render_top_k,
            render_speed_scale=render_speed_scale,
            render_hold_scale=render_hold_scale,
            output_dir=output_dir,
            deadline=None,
        )

        html_path = os.path.join(output_dir, "preview.html")
        _write_dataset_html(
            output_path=html_path,
            rows=preview_rows,
            render_root=render_root,
        )
        html_abs = os.path.abspath(html_path)
        print(f"Success count: {succeeded}")
        print(f"Failure count: {len(failed_items)}")
        print(f"HTML: {html_abs}")
        print(f"HTML URL: file://{html_abs}")
        if open_html:
            _open_preview(html_abs)


if __name__ == "__main__":
    fire.Fire(main)
