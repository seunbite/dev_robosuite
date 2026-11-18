#!/usr/bin/env python3
"""Generate Paddlebot (gcr_alpha1) Motion Codes for the humanlike-90 cue set.

Uses the authored 5-shot bank (motion_code/data/paddlebot_fewshot.json) plus the
gcr_alpha1 block from motion_code_synthetic_continuum.yaml.

Backends:
  --backend gemini   (local Google API; source APIKEY.sh first)
  --backend qwen     (local vLLM; needs GPU — usually run on Babel)

Examples::

    source APIKEY.sh
    python motion_code/generate_paddlebot90.py --backend gemini --resume

    python motion_code/generate_paddlebot90.py \\
      --backend qwen --model Qwen/Qwen2.5-VL-32B-Instruct --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

MC = Path(__file__).resolve().parent
PAPER = MC.parent
REPO = PAPER.parent.parent
ADHOC = REPO / "adhoc/generation/robotarm"
sys.path[:0] = [str(ADHOC), str(REPO), str(MC)]

from run_embodiment_exp0_gemini import (  # noqa: E402
    _extract_json_obj,
    extract_motion_code,
    gemini_generate,
)

PLATFORM = "gcr_alpha1"
SCHEMA_YAML = PAPER / "synthetic_morphology/yml/motion_code_synthetic_continuum.yaml"
FEWSHOT_JSON = MC / "data/paddlebot_fewshot.json"
DEFAULT_CUES = REPO / "data/seed/agent_fill/embodiment_short/cues_humanlike90.json"


def _fewshot_block() -> str:
    rows = json.loads(FEWSHOT_JSON.read_text(encoding="utf-8"))["cues"]
    lines = [
        "Match this dialect (do not copy cues verbatim; rematerialize the target cue):",
        "- Body facing stays `dir=front`. Arm direction = `claw=` / `paddle=` "
        "(still|front|back|in|out|up|down).",
        "- Optional claw EE percentiles `x,y,z` (0–100). Palm/grip on the active hand.",
        "- Prefer starting with `pose(...)`. Join steps with `+`.",
        "",
    ]
    for r in rows:
        code = " ".join(str(r["source_code"]).split())
        lines.append(f"- {r['cue']}: {code}")
    return "\n".join(lines)


def schema_block() -> str:
    root = yaml.safe_load(SCHEMA_YAML.read_text(encoding="utf-8"))
    if PLATFORM not in root:
        raise KeyError(f"{PLATFORM} missing in {SCHEMA_YAML}")
    return yaml.safe_dump(
        {PLATFORM: root[PLATFORM]},
        sort_keys=False,
        allow_unicode=True,
        width=100,
        default_flow_style=False,
    ).rstrip()


def build_prompt(*, cue: str, description: str) -> str:
    return f"""You are an expert in Oddkin α=1.0 Branching Gesture Tree (Paddlebot) gesture design using Motion Code.

Goal:
- Generate ONE complete short cue motion for this embodiment.
- Start with `pose(...)`. Total steps: 2~8. Join with `+`.
- Rematerialize the cue onto available channels (head, torso, claw, paddle, base)
  using the FEW-SHOT dialect and the embodiment schema.
- Do not invent attributes outside the schema / few-shot dialect.

[FEW-SHOT STYLE]
{_fewshot_block()}

[CUE]
name: {cue}
description: {description}

[EMBODIMENT SCHEMA — {PLATFORM}]
{schema_block()}

[Output]
Output ONE Motion Code line only (steps joined by +). No planning comments, no JSON, no explanation.
""".strip() + "\n"


def _model_tag(model: str, backend: str) -> str:
    if backend == "gemini":
        return model.replace("/", "_")
    name = model.strip().split("/")[-1].lower()
    name = name.replace("qwen2.5-vl-", "qwen").replace("-instruct", "")
    return name.replace(".", "").replace("_", "")


def _out_dir(backend: str, model: str) -> Path:
    tag = _model_tag(model, backend)
    if backend == "gemini":
        return REPO / "data/results/motion_configs" / f"embodiment_motion_code_gemini_{tag}_paddlebot90"
    return REPO / "data/results/motion_configs" / f"embodiment_motion_code_qwen_{tag}_paddlebot90"


def _pack_ok(*, cue_row: dict[str, Any], model: str, code: str, experiment: str) -> dict[str, Any]:
    return {
        "idx": int(cue_row["idx"]),
        "cue": cue_row["cue"],
        "description": str(cue_row.get("description") or ""),
        "category": cue_row.get("category", ""),
        "state": experiment,
        "model": model,
        "time": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment,
        "platform": PLATFORM,
        "parse_ok": True,
        "source_code": code,
        "notes": [],
    }


def _generate_qwen(prompt: str, *, model: str) -> str:
    from vllm_local import get_vllm_engine  # noqa: WPS433

    engine = get_vllm_engine(model=model)
    texts = engine.generate_batch([{"prompt": prompt}])
    return (texts[0] or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["gemini", "qwen"], required=True)
    ap.add_argument("--cues", type=Path, default=DEFAULT_CUES)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--model", default="")
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-cues", type=int, default=0)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    if args.backend == "gemini":
        model = args.model or "gemini-2.5-pro"
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY (source APIKEY.sh)")
        experiment = "embodiment_motion_code_gemini_paddlebot90"
    else:
        model = args.model or "Qwen/Qwen2.5-VL-32B-Instruct"
        experiment = "embodiment_motion_code_qwen_paddlebot90"

    cues = json.loads(args.cues.read_text(encoding="utf-8"))
    if args.max_cues > 0:
        cues = cues[: args.max_cues]

    out_json = args.out_json or _out_dir(args.backend, model)
    out_json.mkdir(parents=True, exist_ok=True)
    master_path = out_json / "embodiment_motion_code_all.json"
    raw_dir = out_json / "_raw" / PLATFORM
    raw_dir.mkdir(parents=True, exist_ok=True)

    by_cue: dict[str, dict[str, Any]] = {}
    if args.resume and master_path.is_file():
        for entry in json.loads(master_path.read_text(encoding="utf-8")):
            by_cue[entry["cue"]] = entry

    print(f"backend={args.backend} model={model}", flush=True)
    print(f"cues={len(cues)} out={out_json}", flush=True)
    print(f"fewshot={FEWSHOT_JSON}", flush=True)

    n_ok = n_fail = n_skip = 0
    for cue_row in cues:
        cue = cue_row["cue"]
        entry = by_cue.setdefault(
            cue,
            {
                "idx": int(cue_row["idx"]),
                "cue": cue,
                "category": cue_row.get("category", ""),
                "description": cue_row.get("description", ""),
                "platforms": {},
            },
        )
        prev = entry["platforms"].get(PLATFORM)
        if args.resume and prev and str(prev.get("source_code") or "").strip():
            n_skip += 1
            continue

        prompt = build_prompt(cue=cue, description=str(cue_row.get("description") or ""))
        raw = ""
        code = ""
        last_err: Exception | None = None
        for attempt in range(args.retries + 1):
            try:
                print(f"  gen [{n_ok + n_fail + 1}/{len(cues)}] {cue} (try {attempt + 1})", flush=True)
                if args.backend == "gemini":
                    raw = gemini_generate(prompt, model=model)
                else:
                    raw = _generate_qwen(prompt, model=model)
                obj = _extract_json_obj(raw)
                code = extract_motion_code(raw, obj)
                if not code:
                    raise RuntimeError("empty motion code")
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                msg = str(exc)
                print(f"  FAIL {cue}: {exc}", flush=True)
                if "RESOURCE_EXHAUSTED" in msg or "spending cap" in msg.lower():
                    _flush(master_path, by_cue, cues)
                    raise SystemExit(2) from exc
                time.sleep(1.0 + attempt)

        if last_err is not None or not code:
            n_fail += 1
            row = {
                "idx": int(cue_row["idx"]),
                "cue": cue,
                "description": str(cue_row.get("description") or ""),
                "category": cue_row.get("category", ""),
                "state": experiment,
                "model": model,
                "time": datetime.now().isoformat(timespec="seconds"),
                "experiment": experiment,
                "platform": PLATFORM,
                "parse_ok": False,
                "source_code": "",
                "notes": [f"error: {last_err}"],
            }
        else:
            n_ok += 1
            row = _pack_ok(cue_row=cue_row, model=model, code=code, experiment=experiment)

        (raw_dir / f"{cue}.txt").write_text(raw or "", encoding="utf-8")
        entry["platforms"][PLATFORM] = row
        _flush(master_path, by_cue, cues)
        if args.backend == "gemini":
            time.sleep(max(0.0, args.sleep))

    print(f"done ok={n_ok} fail={n_fail} skip={n_skip} → {master_path}", flush=True)


def _flush(master_path: Path, by_cue: dict[str, dict[str, Any]], cues: list[dict[str, Any]]) -> None:
    configs = [by_cue[c["cue"]] for c in cues if c["cue"] in by_cue]
    master_path.write_text(json.dumps(configs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
