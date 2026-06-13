#!/usr/bin/env python3
"""Text-only motion verify (pose text-verify style). Same JSON schema as VLM verify."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from motion_verify_shared import (  # noqa: E402
    load_verify_done_indices,
    motion_verify_prompt,
    record_from_parsed,
    resume_default,
)
from verify_pose_vlm import _fewshot_block  # noqa: E402
from vlm_batch_util import vlm_generate_texts  # noqa: E402
from vlm_client import setup_vlm_from_args, vlm_batch_size  # noqa: E402
from vlm_json import extract_json  # noqa: E402

BASE_CFG = (
    _REPO
    / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_gt_fixed_pose_pilot40.json"
)
SHOTS = _REPO / "data/seed/shots/manipulator/shot_configs_v19_sophisticated.json"
OUT_JSON = _REPO / "data/results/verify/pilot40_motion_component_verify_text_gemini.json"


def _gemini_client():
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY.")
    return genai.Client(api_key=api_key)


def _call_text_parse(text: str) -> dict[str, Any]:
    try:
        return extract_json(text)
    except Exception as e:
        return {"parse_error": str(e), "raw_text": text}


def _call_text(
    model_id: str,
    prompt: str,
    *,
    vlm_backend: str = "transformers",
    vlm: Any | None = None,
) -> dict[str, Any]:
    if vlm is not None:
        text = vlm.generate(prompt)
    elif vlm_backend == "gemini":
        client = _gemini_client()
        resp = client.models.generate_content(model=model_id, contents=[prompt])
        text = (resp.text or "").strip()
    else:
        raise RuntimeError(
            f"Non-gemini backend {vlm_backend!r} requires an in-process VLMClient (got vlm=None)."
        )
    try:
        return extract_json(text)
    except Exception as e:
        return {"parse_error": str(e), "raw_text": text}


def run(args: argparse.Namespace) -> None:
    out_path = Path(args.out_json) if getattr(args, "out_json", None) else OUT_JSON
    config_path = Path(getattr(args, "config_json", None) or BASE_CFG)
    rows = sorted(json.loads(config_path.read_text(encoding="utf-8")), key=lambda r: int(r["idx"]))
    if args.limit:
        rows = rows[: args.limit]
    shots = json.loads(SHOTS.read_text(encoding="utf-8"))
    fewshot = _fewshot_block(shots, n=args.fewshot_n)

    backend, vlm = setup_vlm_from_args(args)
    print(f"[motion-text] backend={backend} shared_vlm={'yes' if vlm else 'no'}", flush=True)

    out_rows: list[dict[str, Any]] = []
    done: set[int] = set()
    if args.resume and out_path.is_file():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        out_rows = prev.get("rows") or []
        done = load_verify_done_indices(out_path)
        if done:
            print(
                f"[resume] skipping {len(done)} cues already verified in {out_path.name}",
                flush=True,
            )

    batch_size = vlm_batch_size(backend) if vlm else 1
    pending: list[tuple[dict[str, Any], str]] = []

    def _append_text_row(row: dict[str, Any], text: str) -> None:
        nonlocal out_rows, done
        parsed = _call_text_parse(text)
        rec = record_from_parsed(parsed)
        idx = int(row["idx"])
        out_rows = [r for r in out_rows if int(r["cue_idx"]) != idx] + [
            {"cue_idx": idx, "cue": row["cue"], **rec}
        ]
        done.add(idx)
        print(
            f"[text] {row['cue']} appropriate={rec.get('movement_is_appropriate')} "
            f"rec={rec.get('recommended_component') is not None}",
            flush=True,
        )
        if not args.dry_run:
            out_path.write_text(
                json.dumps(
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "model": args.model,
                        "partial": True,
                        "rows": out_rows,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    def _flush_pending() -> None:
        nonlocal pending
        if not pending:
            return
        if vlm is not None:
            texts = vlm_generate_texts(
                vlm,
                backend,
                [{"prompt": prompt} for _, prompt in pending],
            )
            for (row, _), text in zip(pending, texts):
                _append_text_row(row, text)
        else:
            client = _gemini_client()
            for row, prompt in pending:
                resp = client.models.generate_content(model=args.model, contents=[prompt])
                _append_text_row(row, (resp.text or "").strip())
        pending = []

    for row in rows:
        idx = int(row["idx"])
        if idx in done and not args.force:
            continue
        pending.append((row, motion_verify_prompt(row, fewshot, modality="text")))
        if len(pending) >= batch_size:
            _flush_pending()
    _flush_pending()

    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "vlm_backend": backend,
        "model": args.model,
        "mode": "motion_component_verify_text_only",
        "config": str(config_path),
        "n": len(out_rows),
        "rows": sorted(out_rows, key=lambda r: int(r["cue_idx"])),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--vlm-backend",
        default=os.getenv("VLM_BACKEND", "transformers"),
        choices=["transformers", "hf", "local", "vllm-local", "vllm", "openai", "qwen", "gemini"],
    )
    ap.add_argument("--config-json", type=Path, default=BASE_CFG)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--fewshot-n", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=resume_default(),
        help="Skip cues already in out-json (default: on; RESUME=0 or --no-resume to disable)",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.model is None:
        args.model = (
            os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")
            if args.vlm_backend != "gemini"
            else "gemini-2.5-pro"
        )
    run(args)


if __name__ == "__main__":
    main()
