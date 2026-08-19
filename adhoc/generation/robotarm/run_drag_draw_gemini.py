#!/usr/bin/env python3
"""Gemini few-shot inference and rendering for vector Trajectory Sketching.

Default pilot: c1-5 and c26-30 across the five paper embodiments (50 jobs).
The runner mirrors the other baseline pipelines: resumable JSONL predictions,
strict parsing/validation, retries, rendering, and a comparison gallery.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for _path in (_REPO, _HERE, _HERE / "legacy"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from drag_draw import PLATFORMS, load_schema, render_program, validate_program  # noqa: E402


CUES_PATH = _REPO / "data/seed/agent_fill/embodiment_short/cues.json"
FEWSHOT_PATH = _REPO / "data/seed/shots/drag_draw/fewshot_cues_yml_5.yaml"
PROMPT_DIR = _REPO / "data/seed/prompt/drag_draw"
GT_RENDER_ROOT = _REPO / "data/results/render/embodiment_motion_code_gt"
DEFAULT_OUT = _REPO / "data/results/drag_draw/gemini25pro_c1_5_c26_30"
DEFAULT_ONLY = "1-5,26-30"


def parse_indices(raw: str) -> set[int]:
    selected: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                raise ValueError(f"descending range: {token}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    return selected


def cue_index() -> dict[int, dict[str, Any]]:
    return {int(row["idx"]): row for row in json.loads(CUES_PATH.read_text(encoding="utf-8"))}


def make_client():
    from google import genai

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if key:
        return genai.Client(api_key=key), "gemini-api"

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    if not project and credentials_path and Path(credentials_path).is_file():
        credentials = json.loads(Path(credentials_path).read_text(encoding="utf-8"))
        project = credentials.get("project_id")
    if not project:
        raise RuntimeError(
            "Set GOOGLE_API_KEY/GEMINI_API_KEY, or configure "
            "GOOGLE_APPLICATION_CREDENTIALS with a project_id."
        )
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    return genai.Client(vertexai=True, project=project, location=location), f"vertex-ai/{location}"


def extract_text(response: Any) -> str:
    raw = (getattr(response, "text", None) or "").strip()
    if raw:
        return raw
    parts: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()


def extract_yaml(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.search(r"```(?:ya?ml)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    obj = yaml.safe_load(text)
    if not isinstance(obj, dict):
        raise ValueError("response must decode to one YAML mapping")
    return obj


def usage_dict(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage_metadata", None)
    return {
        "prompt_token_count": getattr(usage, "prompt_token_count", None),
        "candidates_token_count": getattr(usage, "candidates_token_count", None),
        "thoughts_token_count": getattr(usage, "thoughts_token_count", None),
        "total_token_count": getattr(usage, "total_token_count", None),
    }


def build_prompt(platform: str, cue: dict[str, Any], fewshots: dict[str, list[dict[str, Any]]]) -> str:
    rules = (PROMPT_DIR / f"{platform}.txt").read_text(encoding="utf-8").strip()
    examples: list[str] = []
    for index, row in enumerate(fewshots[platform], 1):
        program = yaml.safe_dump(
            row["drag_draw"], sort_keys=False, allow_unicode=True, width=120
        ).strip()
        examples.append(
            f"[EXAMPLE {index}]\n"
            f"Cue: {row['cue']}\n"
            f"Output:\n{program}"
        )
    return (
        f"{rules}\n\n"
        "The following five demonstrations are complete outputs for this same robot. "
        "Infer the target gesture at the same abstraction level; do not copy an example's motion blindly. "
        "Combine diverse motions to create a recognizable gesture that transcends the embodiment's limitations.\n\n"
        + "\n\n".join(examples)
        + "\n\n[TARGET]\n"
        + f"Cue id: c{int(cue['idx'])}\n"
        + f"Cue: {cue['cue']}\n"
        + f"Meaning: {cue['description']}\n\n"
        + "Return exactly one valid YAML program and nothing else."
    )


def validate_target(
    obj: dict[str, Any], *, platform: str, cue: dict[str, Any], schema: dict[str, Any]
) -> dict[str, Any]:
    obj = dict(obj)
    obj["version"] = "2.0"
    obj["cue"] = str(cue["cue"])
    obj["platform"] = platform
    obj["fps"] = int(schema[platform]["fps"])
    validate_program(obj, schema)

    bounds = schema[platform]["workspace_bounds"]
    max_frame = 0
    for trajectory in obj.get("trajectories") or []:
        for x, y, z, frame in trajectory.get("vectors") or []:
            for axis, value in zip(("x", "y", "z"), (x, y, z)):
                lo, hi = (float(v) for v in bounds[axis])
                if not lo - 1e-6 <= float(value) <= hi + 1e-6:
                    raise ValueError(f"{trajectory['handle']} {axis}={value} outside [{lo}, {hi}]")
            max_frame = max(max_frame, int(frame))
    if max_frame < 2:
        raise ValueError("motion must extend beyond frame 1")
    if max_frame > 100:
        raise ValueError("motion exceeds 100 frames (5 seconds at 20 fps)")
    return obj


def prediction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate(args: argparse.Namespace) -> list[dict[str, Any]]:
    from google.genai import types

    out = args.out_dir.resolve()
    pred_path = out / "predictions.jsonl"
    schema = load_schema()
    cues = cue_index()
    fewshots = yaml.safe_load(args.fewshot_path.read_text(encoding="utf-8"))
    selected = parse_indices(args.only_idx)
    platforms = [x.strip() for x in args.platforms.split(",") if x.strip()]
    existing = prediction_rows(pred_path)
    done = {row["job_id"] for row in existing if row.get("parse_ok")}
    jobs = [
        (idx, platform, cues[idx])
        for idx in sorted(selected)
        for platform in platforms
        if args.force or f"drag_draw.{platform}.c{idx:02d}" not in done
    ]
    client, backend = make_client()
    thinking_budget = max(128, int(args.thinking_budget)) if "2.5-pro" in args.model else int(args.thinking_budget)
    print(
        f"[generate] pending={len(jobs)} model={args.model} backend={backend} "
        f"thinking_budget={thinking_budget} workers={args.workers}",
        flush=True,
    )

    def work(job: tuple[int, str, dict[str, Any]]) -> dict[str, Any]:
        idx, platform, cue = job
        job_id = f"drag_draw.{platform}.c{idx:02d}"
        prompt = build_prompt(platform, cue, fewshots)
        last_error = "unknown"
        last_raw = ""
        for attempt in range(args.retries + 1):
            retry_note = "" if attempt == 0 else (
                f"\n\nYour previous response failed validation: {last_error}\n"
                "Return a corrected complete YAML program."
            )
            try:
                config_kwargs: dict[str, Any] = {
                    "temperature": args.temperature,
                    "max_output_tokens": args.max_output_tokens,
                }
                if thinking_budget >= 0:
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=thinking_budget
                    )
                response = client.models.generate_content(
                    model=args.model,
                    contents=prompt + retry_note,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                last_raw = extract_text(response)
                if not last_raw:
                    raise ValueError("empty Gemini response")
                program = validate_target(
                    extract_yaml(last_raw), platform=platform, cue=cue, schema=schema
                )
                return {
                    "job_id": job_id,
                    "idx": idx,
                    "cue": cue["cue"],
                    "description": cue["description"],
                    "platform": platform,
                    "model": args.model,
                    "backend": backend,
                    "thinking_budget": thinking_budget,
                    "attempt": attempt + 1,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "program": program,
                    "raw_output": last_raw,
                    "usage": usage_dict(response),
                    "parse_ok": True,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < args.retries:
                    time.sleep(min(2**attempt, 8))
        return {
            "job_id": job_id,
            "idx": idx,
            "cue": cue["cue"],
            "description": cue["description"],
            "platform": platform,
            "model": args.model,
            "backend": backend,
            "thinking_budget": thinking_budget,
            "error": last_error,
            "raw_output": last_raw,
            "parse_ok": False,
        }

    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(work, job): job for job in jobs}
            for future in as_completed(futures):
                row = future.result()
                append_jsonl(pred_path, row)
                status = "PASS" if row.get("parse_ok") else "FAIL"
                print(f"[infer] {row['job_id']} parse={status} attempt={row.get('attempt', '-')}", flush=True)

    rows = prediction_rows(pred_path)
    latest = {row["job_id"]: row for row in rows}
    selected_rows = [
        row for row in latest.values()
        if int(row["idx"]) in selected and row["platform"] in platforms
    ]
    selected_rows.sort(key=lambda row: (int(row["idx"]), PLATFORMS.index(row["platform"])))
    (out / "predictions_latest.json").write_text(
        json.dumps(selected_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    by_platform: dict[str, list[dict[str, Any]]] = {platform: [] for platform in platforms}
    for row in selected_rows:
        if row.get("parse_ok"):
            by_platform[row["platform"]].append({
                "idx": row["idx"], "cue": row["cue"], "drag_draw": row["program"]
            })
    (out / "predictions_latest.yaml").write_text(
        yaml.safe_dump(by_platform, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )
    return selected_rows


def render(args: argparse.Namespace) -> list[dict[str, Any]]:
    out = args.out_dir.resolve()
    source = out / "predictions_latest.json"
    if not source.is_file():
        raise FileNotFoundError(f"run generate first: {source}")
    rows = json.loads(source.read_text(encoding="utf-8"))
    rendered_path = out / "predictions_rendered.json"
    if rendered_path.is_file():
        previous = {
            row["job_id"]: row
            for row in json.loads(rendered_path.read_text(encoding="utf-8"))
        }
        for row in rows:
            old = previous.get(row["job_id"]) or {}
            if old.get("render_gif"):
                row["render_gif"] = old["render_gif"]
    selected = parse_indices(args.only_idx)
    platforms = {x.strip() for x in args.platforms.split(",") if x.strip()}
    good = [
        row for row in rows
        if row.get("parse_ok")
        and int(row["idx"]) in selected
        and row["platform"] in platforms
    ]
    for number, row in enumerate(good, 1):
        target = (
            out / "render" / row["platform"]
            / f"c{int(row['idx']):02d}_{row['cue']}_gemini-2.5-pro.gif"
        )
        if args.force_render or not target.is_file():
            print(f"[render {number}/{len(good)}] {row['platform']} c{int(row['idx']):02d}", flush=True)
            render_program(row["program"], target)
        row["render_gif"] = str(target.relative_to(_REPO))
    rendered_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return rows


def reference_gif(platform: str, idx: int, cue: str) -> Path | None:
    folder = GT_RENDER_ROOT / platform / "gif"
    exact = folder / f"emb_g{idx:02d}_{cue}.gif"
    if exact.is_file():
        return exact
    hits = sorted(folder.glob(f"*g{idx:02d}_{cue}*.gif")) if folder.is_dir() else []
    return hits[0] if hits else None


def gallery(args: argparse.Namespace) -> Path:
    out = args.out_dir.resolve()
    source = out / "predictions_rendered.json"
    if not source.is_file():
        raise FileNotFoundError(f"run render first: {source}")
    rows = json.loads(source.read_text(encoding="utf-8"))
    rows_by = {(int(row["idx"]), row["platform"]): row for row in rows}
    selected = sorted(parse_indices(args.only_idx))
    platforms = [x.strip() for x in args.platforms.split(",") if x.strip()]
    gallery_path = out / "gallery_trajectory_sketching_c1_5_c26_30.html"
    cue_label = ", ".join(f"c{idx}" for idx in selected)

    sections: list[str] = []
    for idx in selected:
        cards: list[str] = []
        cue_name = ""
        for platform in platforms:
            row = rows_by.get((idx, platform))
            if row is None:
                cards.append(f"<article><h3>{html.escape(platform)}</h3><p class='bad'>missing prediction</p></article>")
                continue
            cue_name = row["cue"]
            if not row.get("parse_ok") or not row.get("render_gif"):
                cards.append(
                    f"<article><h3>{html.escape(platform)}</h3><p class='bad'>"
                    f"{html.escape(str(row.get('error') or 'parse failed'))}</p></article>"
                )
                continue
            generated = os.path.relpath(_REPO / row["render_gif"], gallery_path.parent)
            revision = (_REPO / row["render_gif"]).stat().st_mtime_ns
            reference = reference_gif(platform, idx, row["cue"])
            reference_html = (
                f"<figure><img src='{html.escape(os.path.relpath(reference, gallery_path.parent))}'>"
                "<figcaption>human-authored reference</figcaption></figure>"
                if reference else "<figure><div class='missing'>reference unavailable</div></figure>"
            )
            program_text = yaml.safe_dump(
                row["program"], sort_keys=False, allow_unicode=True, width=100
            )
            cards.append(
                f"<article><h3>{html.escape(platform)}</h3><div class='pair'>{reference_html}"
                f"<figure><img src='{html.escape(generated)}?v={revision}'>"
                "<figcaption>Gemini trajectory sketch</figcaption></figure></div>"
                f"<details><summary>trajectory YAML</summary><pre>{html.escape(program_text)}</pre></details></article>"
            )
        sections.append(
            f"<section><h2>c{idx:02d} · {html.escape(cue_name)}</h2>"
            f"<div class='grid'>{''.join(cards)}</div></section>"
        )

    page = """<!doctype html><html><head><meta charset='utf-8'>
<title>Gemini-2.5-Pro · Trajectory Sketching · __CUE_LABEL__</title>
<style>
body{font-family:system-ui;margin:24px;background:#0d1117;color:#e6edf3}
h1{margin-bottom:4px}.meta{color:#9da7b3;margin-top:0}.grid{display:grid;grid-template-columns:repeat(5,minmax(300px,1fr));gap:12px;overflow-x:auto}
section{margin:28px 0}article{background:#161b22;border:1px solid #30363d;padding:10px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:5px}
figure{margin:0}img{display:block;width:100%;background:#000}figcaption{text-align:center;color:#9da7b3;font-size:11px;padding-top:4px}
pre{font-size:10px;max-height:520px;overflow:auto;white-space:pre-wrap}.bad{color:#ff7b72}.missing{padding:60px 10px;color:#9da7b3;text-align:center}
</style></head><body><h1>Gemini-2.5-Pro · Trajectory Sketching</h1>
<p class='meta'>__CUE_COUNT__ cues (__CUE_LABEL__) × __PLATFORM_COUNT__ embodiments · thinking budget 128 · five-shot prompting</p>
""" + "".join(sections) + "</body></html>"
    page = (
        page.replace("__CUE_LABEL__", html.escape(cue_label))
        .replace("__CUE_COUNT__", str(len(selected)))
        .replace("__PLATFORM_COUNT__", str(len(platforms)))
    )
    gallery_path.write_text(page, encoding="utf-8")
    print(f"[gallery] {gallery_path.relative_to(_REPO)}", flush=True)
    return gallery_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", nargs="?", default="all", choices=("generate", "render", "gallery", "all"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--fewshot-path",
        type=Path,
        default=FEWSHOT_PATH,
        help="Trajectory Sketching few-shot YAML source.",
    )
    parser.add_argument("--only-idx", default=DEFAULT_ONLY)
    parser.add_argument("--platforms", default=",".join(PLATFORMS))
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--thinking-budget", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="regenerate successful inference jobs")
    parser.add_argument("--force-render", action="store_true")
    args = parser.parse_args()
    unknown = sorted(set(x.strip() for x in args.platforms.split(",")) - set(PLATFORMS))
    if unknown:
        parser.error(f"unknown platforms: {', '.join(unknown)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.cmd in ("generate", "all"):
        generate(args)
    if args.cmd in ("render", "all"):
        render(args)
    if args.cmd in ("gallery", "all"):
        gallery(args)


if __name__ == "__main__":
    main()
