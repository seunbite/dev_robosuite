#!/usr/bin/env python3
"""Instance-level correction: wrapped exp1/exp7 prompt + Correction Q1/Q2 + render.

Examples:
  # Exp1 pose correction (manipulator)
  python adhoc/generation/robotarm/run_correction.py \\
    --cue flex_bicep --exp 1 --model-tag gemini-2.5-pro

  # Exp7 movement correction (google robot)
  python adhoc/generation/tiago/run_correction.py \\
    --cue beckon_come_here --exp 7

  # Verify only (parse Correction Q1/Q2, no config rewrite)
  python adhoc/generation/robotarm/run_correction.py \\
    --cue flex_bicep --exp 1 --verify-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE, _REPO / "adhoc/generation/tiago"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from correction_lib import default_config_path, run_instance_correction  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-cue correction: inner exp1/exp7 prompt + Correction Q1/Q2 + render"
    )
    ap.add_argument("--cue", required=True, help="Cue name, e.g. flex_bicep")
    ap.add_argument(
        "--exp",
        choices=("1", "7", "15"),
        default="1",
        help="Generation experiment to correct: 1=pose, 7=movement, 15=mime (default: 1)",
    )
    ap.add_argument(
        "--domain",
        choices=("manipulator", "tiago"),
        default="manipulator",
        help="Robot platform (default: manipulator)",
    )
    ap.add_argument("--model-tag", default="gemini-2.5-pro", help="Tag in result_exp{N}_{tag}.json")
    ap.add_argument("--model", default=None, help="VLM/LLM model id (default: model-tag)")
    ap.add_argument("--backend", default="gemini", help="gemini | transformers | vllm")
    ap.add_argument("--config-json", type=Path, default=None, help="Exp config JSON (default: result_exp{N}_{tag})")
    ap.add_argument("--render-dir", type=Path, default=None, help="Output dir for correction renders")
    ap.add_argument("--verify-only", action="store_true", help="Parse Correction Q1/Q2 only; do not rewrite config")
    ap.add_argument("--skip-render", action="store_true", help="Reuse existing renders in render-dir")
    ap.add_argument(
        "--no-save-config",
        action="store_true",
        help="Do not write versioned correction config snapshot (correction_config.json, …)",
    )
    ap.add_argument(
        "--update-main-config",
        action="store_true",
        help="Also upsert corrected row into result_exp{N}_{tag}.json (default: off)",
    )
    ap.add_argument(
        "--from-base",
        action="store_true",
        help="Always load input from base result_exp{N}_{tag}.json (ignore prior correction rounds)",
    )
    ap.add_argument(
        "--inner-exp",
        choices=("0", "1", "7", "15"),
        default=None,
        help="Inner generation prompt for correction wrapper (default: same as --exp; use 15 for mime)",
    )
    ap.add_argument("--no-rerender", action="store_true", help="Skip second render after correction")
    ap.add_argument(
        "--ignore-xyz",
        action="store_true",
        help="tiago: ignore pose x/y/z for pose-bank pick (multitile pool OK)",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=None,
        help="tiago+ignore_xyz: max pool tiles (default: all matching tiles)",
    )
    args = ap.parse_args()

    cfg = args.config_json or default_config_path(args.domain, args.model_tag, args.exp)  # type: ignore[arg-type]
    if not cfg.is_file():
        raise SystemExit(f"Config not found: {cfg}")

    vlm = None
    if args.backend.lower() not in {"gemini"}:
        from vlm_client import (  # noqa: E402
            VLMClient,
            init_inprocess_engine,
            is_inprocess_backend,
            is_vllm_http_backend,
            require_vllm_server,
        )

        if is_vllm_http_backend(args.backend):
            require_vllm_server()
        elif is_inprocess_backend(args.backend):
            init_inprocess_engine(args.backend, args.model or args.model_tag)
        vlm = VLMClient(backend=args.backend, model=args.model or args.model_tag)

    result = run_instance_correction(
        domain=args.domain,  # type: ignore[arg-type]
        exp_id=args.exp,  # type: ignore[arg-type]
        cue=args.cue,
        config_path=cfg,
        model_tag=args.model_tag,
        model=args.model,
        backend=args.backend,
        vlm=vlm,
        render_dir=args.render_dir,
        verify_only=args.verify_only,
        skip_render=args.skip_render,
        save_config=not args.no_save_config,
        update_main_config=args.update_main_config,
        chain_from_latest=not args.from_base,
        rerender=not args.no_rerender,
        inner_exp_id=args.inner_exp,  # type: ignore[arg-type]
        ignore_xyz=args.ignore_xyz,
        topk=args.topk,
    )

    summary = {
        "cue": result.cue,
        "domain": result.domain,
        "exp_id": result.exp_id,
        "round": result.round,
        "parent_round": result.parent_round,
        "correction_stem": f"{result.correction_json.stem}" if result.correction_json else None,
        "action": result.action,
        "is_correct": result.is_correct,
        "correction_q1": result.correction.get("correction_q1"),
        "correction_q2": result.correction.get("correction_q2"),
        "base_config_json": str(result.config_json),
        "config_snapshot_json": (
            str(result.config_snapshot_json) if result.config_snapshot_json else None
        ),
        "correction_json": str(result.correction_json) if result.correction_json else None,
        "media": {
            "gif": str(result.media.gif) if result.media.gif else None,
            "mp4": str(result.media.mp4) if result.media.mp4 else None,
            "pose_png": str(result.media.pose_png) if result.media.pose_png else None,
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if result.is_correct:
        print(f"[ok] {result.cue} (exp{result.exp_id}, round {result.round}): Correction Q1=yes")
    elif result.action == "verify_only":
        print(f"[verify] {result.cue} (round {result.round}): Q1={result.correction.get('correction_q1')}")
        print(f"         Q2={result.correction.get('correction_q2')}")
    else:
        snap = result.config_snapshot_json or result.correction_json
        print(f"[corrected] {result.cue} (exp{result.exp_id}, round {result.round}) → {snap}")

    if result.domain == "tiago" and result.exp_id == "1" and result.action == "corrected":
        try:
            sys.path.insert(0, str(_REPO / "adhoc/generation/tiago"))
            from build_correction_evolution_html import build_evolution_html  # noqa: WPS433

            html_path = build_evolution_html(model_tag=result.model_tag)
            print(f"[evolution html] {html_path}")
        except Exception as exc:
            print(f"[warn] evolution html not updated: {exc}")


if __name__ == "__main__":
    main()
