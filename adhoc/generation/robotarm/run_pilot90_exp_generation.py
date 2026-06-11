#!/usr/bin/env python3
"""Generate pilot-90 exp1 (pose) or exp7 (motion tail) configs — Gemini or Qwen.

  # Gemini exp1 (all 90 cues)
  source APIKEY.sh
  python adhoc/generation/robotarm/run_pilot90_exp_generation.py --exp 1 --backend gemini

  # Qwen exp1 on cluster
  python adhoc/generation/robotarm/run_pilot90_exp_generation.py --exp 1 --backend transformers --model Qwen/Qwen2.5-VL-32B-Instruct

  # exp7 only missing cues
  python adhoc/generation/robotarm/run_pilot90_exp_generation.py --exp 7 --backend gemini --resume
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config_gen_vlm import run_exp_generation  # noqa: E402
from gpu_check import require_cuda_gpu  # noqa: E402
from hf_cache_setup import setup_hf_cache  # noqa: E402
from pilot90_experiment_suite import score_exp1, score_exp7  # noqa: E402
from pilot90_paths import (  # noqa: E402
    model_to_tag,
    result_config_path,
    score_result_path,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Pilot-90 exp1/exp7 LLM generation")
    p.add_argument("--exp", type=int, choices=(1, 7), required=True)
    p.add_argument("--backend", default=os.getenv("VLM_BACKEND", "gemini"))
    p.add_argument("--model", default=os.getenv("VLM_MODEL", "gemini-2.5-pro"))
    p.add_argument("--delay", type=float, default=float(os.getenv("GEN_DELAY", "2.0")))
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--score", action="store_true", help="Write score_exp{N}_{tag}.json after generation")
    p.add_argument("--cues", type=str, default=None, help="Comma-separated cue subset")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    tag = model_to_tag(args.model)
    out_cfg = result_config_path(args.exp, tag)
    out_cfg.parent.mkdir(parents=True, exist_ok=True)

    backend = args.backend.lower()
    if backend == "vllm":
        backend = "local"

    vlm = None
    if backend != "gemini":
        require_cuda_gpu()
        setup_hf_cache(os.environ.get("HF_HOME"))
        from vlm_client import VLMClient, init_inprocess_engine  # noqa: WPS433

        init_inprocess_engine(backend, args.model)
        vlm = VLMClient(backend=backend, model=args.model)

    cues = [c.strip() for c in args.cues.split(",") if c.strip()] if args.cues else None
    print(f"exp{args.exp} → {out_cfg}  model={args.model}  tag={tag}  backend={backend}", flush=True)
    if args.dry_run:
        print("dry-run: would generate", cues or "all manifest cues")
        return

    ok, failed = run_exp_generation(
        args.exp,
        out_path=out_cfg,
        model=args.model,
        backend=backend,
        vlm=vlm,
        cues=cues,
        resume=args.resume,
        delay=args.delay,
        on_progress=lambda c, success: print(f"  [{'OK' if success else 'FAIL'}] {c}", flush=True),
    )
    print(f"Done exp{args.exp}: ok={ok} failed={failed} → {out_cfg}", flush=True)

    if args.score or args.exp in (1, 7):
        score_out = score_result_path(args.exp, tag)
        if args.exp == 1:
            payload = score_exp1(out_cfg, score_out)
        else:
            payload = score_exp7(out_cfg, score_out)
        acc = payload.get("accuracy")
        print(
            f"Score → {score_out}: {payload.get('n_correct')}/{payload.get('n')} "
            f"= {100 * acc:.1f}%" if acc is not None else f"Score → {score_out}",
            flush=True,
        )


if __name__ == "__main__":
    main()
