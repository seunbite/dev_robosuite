#!/usr/bin/env python3
"""Generate Short Motion Codes with Qwen2.5-VL (vLLM) — same yaml_cue prompt as Gemini.

Writes the same master JSON layout as ``run_embodiment_exp0_gemini.py`` so
``score_humanoid_loc_dir_move.py`` can score Orientation / Locx / Locy / Locz /
Movement against human GT.

Example (Babel)::

    python adhoc/generation/robotarm/run_embodiment_exp0_qwen.py \\
      --platforms humanoid --resume \\
      --model Qwen/Qwen2.5-VL-32B-Instruct --batch-size 1
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

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from motion_code_to_config import convert_code, write_json_files  # noqa: E402
from run_embodiment_exp0_gemini import (  # noqa: E402
    DEFAULT_CUES,
    DEFAULT_FEWSHOT,
    PLATFORMS,
    SCHEMA_YAML,
    _extract_json_obj,
    build_prompt,
    extract_motion_code,
    load_cues,
)
from vllm_local import get_vllm_engine  # noqa: E402

DEFAULT_OUT = _REPO / "data/results/motion_configs/embodiment_motion_code_qwen"
DEFAULT_OUT_RENDER = _REPO / "data/results/render/embodiment_motion_code_qwen"


def _model_tag(model: str) -> str:
    name = model.strip().split("/")[-1].lower()
    name = name.replace("qwen2.5-vl-", "qwen").replace("-instruct", "")
    return name.replace(".", "").replace("_", "")


def _already_done(entry: dict[str, Any] | None, platform: str) -> bool:
    if not entry:
        return False
    plat = (entry.get("platforms") or {}).get(platform) or {}
    code = str(plat.get("source_code") or "").strip()
    movements = plat.get("movements") or []
    return bool(code) or bool(movements)


def _pack_row(
    *,
    cue_row: dict[str, Any],
    platform: str,
    model: str,
    raw: str,
) -> dict[str, Any]:
    cue = cue_row["cue"]
    description = str(cue_row.get("description") or "")
    obj = _extract_json_obj(raw)
    code = extract_motion_code(raw, obj)
    if not code and obj and isinstance(obj.get("movements"), list):
        converted = {
            "movements": obj["movements"],
            "parse_ok": True,
            "notes": ["json_movements_only"],
            "source_code": "",
        }
    else:
        converted = convert_code(code, platform=platform)

    now = datetime.now().isoformat(timespec="seconds")
    domain = {
        "single_arm": "manipulator",
        "tiago": "tiago",
        "stretch": "stretch",
        "humanoid": "unitree_g1",
        "quadruped": "unitree_go2",
        "roomba": "roomba",
    }[platform]
    return {
        "idx": int(cue_row["idx"]),
        "cue": cue,
        "description": description or f"Qwen embodiment ({platform}) for {cue}",
        "movements": converted["movements"],
        "state": "motion_code_qwen",
        "model": model,
        "time": now,
        "experiment": "embodiment_motion_code_qwen",
        "generation_valid": bool(converted["movements"]),
        "platform": platform,
        "domain": domain,
        "parse_ok": converted["parse_ok"],
        "notes": converted["notes"],
        "source_code": converted.get("source_code") or code,
        "raw_response": raw,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cues", type=Path, default=DEFAULT_CUES)
    ap.add_argument("--fewshot", type=Path, default=DEFAULT_FEWSHOT)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument(
        "--platforms",
        default="humanoid",
        help="Comma list; default humanoid (attribute table target).",
    )
    ap.add_argument("--max-cues", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-32B-Instruct")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--prompt-style",
        default="yaml_cue",
        choices=["yaml_cue", "exp0"],
    )
    args = ap.parse_args()

    tag = _model_tag(args.model)
    out_json = args.out_json or (DEFAULT_OUT.parent / f"embodiment_motion_code_qwen_{tag}")
    platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]
    for p in platforms:
        if p not in PLATFORMS:
            raise SystemExit(f"unknown platform: {p}")
    if args.prompt_style == "yaml_cue" and not SCHEMA_YAML.is_file():
        raise SystemExit(f"missing schema YAML: {SCHEMA_YAML}")

    cues = load_cues(args.cues)
    if args.max_cues > 0:
        cues = cues[: args.max_cues]

    out_json.mkdir(parents=True, exist_ok=True)
    master_path = out_json / "embodiment_motion_code_all.json"
    print(f"out-json: {out_json}", flush=True)
    print(f"model: {args.model}  platforms={platforms}", flush=True)

    by_cue: dict[str, dict[str, Any]] = {}
    if args.resume and master_path.is_file():
        for entry in json.loads(master_path.read_text(encoding="utf-8")):
            by_cue[entry["cue"]] = entry

    jobs: list[dict[str, Any]] = []
    for plat in platforms:
        for cue_row in cues:
            cue = cue_row["cue"]
            entry = by_cue.get(cue)
            if args.resume and _already_done(entry, plat):
                continue
            prompt = build_prompt(
                platform=plat,
                cue=cue,
                description=str(cue_row.get("description") or ""),
                cues=cues,
                fewshot_path=args.fewshot,
                category=str(cue_row.get("category") or ""),
                prompt_style=args.prompt_style,
            )
            jobs.append({"platform": plat, "cue_row": cue_row, "prompt": prompt})

    print(f"pending={len(jobs)}", flush=True)
    if not jobs:
        print("nothing to do", flush=True)
        return

    engine = get_vllm_engine(model=args.model)
    pending = jobs
    for attempt in range(args.retries + 1):
        if not pending:
            break
        retry: list[dict[str, Any]] = []
        bs = max(1, args.batch_size)
        for start in range(0, len(pending), bs):
            batch = pending[start : start + bs]
            texts = engine.generate_batch([{"prompt": j["prompt"]} for j in batch])
            for job, raw in zip(batch, texts):
                plat = job["platform"]
                cue_row = job["cue_row"]
                cue = cue_row["cue"]
                try:
                    row = _pack_row(
                        cue_row=cue_row, platform=plat, model=args.model, raw=raw
                    )
                    if not row["generation_valid"]:
                        raise RuntimeError(f"empty movements notes={row.get('notes')}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[fail] {plat} {cue}: {exc}", flush=True)
                    if attempt < args.retries:
                        retry.append(job)
                    continue

                entry = by_cue.setdefault(
                    cue,
                    {
                        "idx": int(cue_row["idx"]),
                        "cue": cue,
                        "category": cue_row.get("category", ""),
                        "platforms": {},
                    },
                )
                entry["platforms"][plat] = {
                    "idx": row["idx"],
                    "cue": row["cue"],
                    "description": row["description"],
                    "movements": row["movements"],
                    "state": row["state"],
                    "model": row["model"],
                    "time": row["time"],
                    "experiment": row["experiment"],
                    "generation_valid": row["generation_valid"],
                    "platform": row["platform"],
                    "domain": row["domain"],
                    "parse_ok": row["parse_ok"],
                    "notes": row["notes"],
                    "source_code": row["source_code"],
                }
                # keep raw separately
                raw_dir = out_json / "_raw" / plat
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / f"{cue}.txt").write_text(row.get("raw_response") or "", encoding="utf-8")

                ordered = sorted(by_cue.values(), key=lambda e: int(e["idx"]))
                write_json_files(ordered, out_json)
                print(
                    f"[ok] {plat} idx={cue_row['idx']} {cue} "
                    f"steps={len(row['movements'])}",
                    flush=True,
                )
        pending = retry
        if pending:
            print(f"retrying {len(pending)} after attempt {attempt + 1}", flush=True)
            time.sleep(1.0)

    print(f"done → {master_path}", flush=True)
    print(
        "Score (local, after rsync back):\n"
        f"  python papers/icra2027_motion_abstraction/score_humanoid_loc_dir_move.py "
        f"--pred {master_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
