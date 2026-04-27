#!/usr/bin/env python3
"""
VLM pass over humeaneval-style instances (same data as PPTX: binary_classification, compare_baseline).
JSONL default: dev_robosuite/data/results/vlm/
Ends with a markdown summary (stdout): binary → robot×input_type metrics; compare → letter acc + keyword counts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_DEV_R = Path(__file__).resolve().parents[2]
_TEST = _DEV_R / "adhoc" / "vlm_test"
_VLM_DIR = _DEV_R / "data" / "results" / "vlm"
if str(_TEST) not in sys.path:
    sys.path.insert(0, str(_TEST))

from heval_data import (  # noqa: E402
    HevalVLMInstance,
    load_binary_instances,
    load_compare_instances,
)
from testset_utils import (  # noqa: E402
    expand_input_types,
    normalize_test_media_type,
    prepare_test_media,
)
from vlm_metrics import format_markdown_report  # noqa: E402
from vlm_utils import vlm_multimodal_call  # noqa: E402
from vlm_results_pptx import build_binary_results_pptx_from_records  # noqa: E402


def _one_call_for_binary(
    inst: HevalVLMInstance,
    input_type: str,
    *,
    hz: int,
    force_media: bool,
) -> tuple[str, list[tuple[Path, str | None]]]:
    sample = inst.meta.get("_sample")
    if not sample or not sample.get("gif_path"):
        raise ValueError("binary instance without meta._sample / gif_path")
    t = normalize_test_media_type(input_type)
    sim_robot = (sample.get("sim_robot") or "IIWA").strip()
    row = prepare_test_media(
        [dict(sample)], test_type=t, robot=sim_robot, hz=hz, force=force_media
    )[0]
    path = Path(row.get("media_path") or row["gif_path"])
    mime = (row.get("media_mime") or "image/png").split(";")[0]
    if mime == "image/gif" and t in ("gif",):
        mime = "image/gif"
    return inst.prompt, [(path, mime)]


def _media_tuples_from_instance(inst: HevalVLMInstance) -> list[tuple[Path, str | None]]:
    return [(p, m) for _, p, m in inst.media]


def _default_out_jsonl(args: argparse.Namespace) -> Path:
    tag = (args.robot or "all").replace(os.sep, "_").replace(" ", "").replace(",", "_")
    safe_model = "".join(c if c.isalnum() or c in "._-" else "_" for c in (args.model or "model"))[:80]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return _VLM_DIR / f"vlm_{args.task}_{tag}_{safe_model}_{stamp}.jsonl"


def run() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("binary_classification", "compare_baseline"), required=True)
    ap.add_argument(
        "--robot",
        type=str,
        default=None,
        help="default: manipulator,tiago (binary) or manipulator (compare). Comma list ok, e.g. manipulator,tiago",
    )
    ap.add_argument("--sample_n", type=int, default=None, help="default 20 (per robot, after shuffle cap)")
    ap.add_argument(
        "--first_n",
        type=int,
        default=None,
        help="After sample_n shuffle, keep only the first N items per robot (default: all).",
    )
    ap.add_argument(
        "--input_type",
        type=str,
        default="all",
        help=        "Binary: see testset_utils `all` = mp4, alpha_frame, first_frame_trajectory, "
        "alpha_frame_trajectory, mp4_plus_trajectory; also gif, mid_frame_trajectory, etc. Compare: ignored.",
    )
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--position_seed", type=int, default=20260424)
    ap.add_argument("--model", type=str, default="gemini-2.0-flash", help="Gemini or claude-… id")
    ap.add_argument("--hz", type=int, default=8)
    ap.add_argument("--force_media", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument(
        "--out_jsonl",
        type=str,
        default="",
        help=f"default: {_VLM_DIR}/vlm_<task>_<robot>_<model>_<utc_stamp>.jsonl",
    )
    ap.add_argument(
        "--out_pptx",
        type=str,
        default="",
        help="binary_classification: write compact per-instance PPTX (default: <jsonl>_summary.pptx if records exist).",
    )
    ap.add_argument(
        "--no_pptx",
        action="store_true",
        help="Skip PPTX summary even when --out_pptx would be set by default.",
    )
    args = ap.parse_args()

    if args.sample_n is None:
        args.sample_n = 20
    if args.robot is None:
        args.robot = "manipulator,tiago" if args.task == "binary_classification" else "manipulator"

    if args.task == "binary_classification":
        insts = load_binary_instances(
            robot=args.robot,
            sample_n=args.sample_n,
            seed=args.seed,
            first_n=args.first_n,
        )
    else:
        insts = load_compare_instances(
            robot=args.robot, sample_n=args.sample_n, position_seed=args.position_seed, seed=args.seed
        )
    if not insts:
        print("No instances (missing GIFs or empty sample).", file=sys.stderr)
        return 1

    itypes = expand_input_types(args.input_type) if args.task == "binary_classification" else ["gif"]
    out_path = Path(args.out_jsonl) if args.out_jsonl else _default_out_jsonl(args)
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    for inst in insts:
        if args.task == "compare_baseline":
            media = _media_tuples_from_instance(inst)
            prompt = inst.prompt
            if args.dry_run:
                print(
                    f"[dry] {inst.instance_id} media={len(media)} prompt_chars={len(prompt)}",
                    file=sys.stderr,
                )
                continue
            try:
                text = vlm_multimodal_call(args.model, prompt, media)
            except Exception as e:
                text = f"ERROR: {e}"
            rec = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "task": inst.task,
                "robot": inst.robot,
                "instance_id": inst.instance_id,
                "input_type": "multiclip",
                "model": args.model,
                "prompt": prompt,
                "response": text,
                "ground_truth": dict(inst.ground_truth),
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)
            print(f"OK {inst.instance_id} -> {out_path.name}", file=sys.stderr)
            continue

        for ity in itypes:
            key = f"{inst.instance_id}::{ity}"
            try:
                prompt, media = _one_call_for_binary(
                    inst, ity, hz=args.hz, force_media=args.force_media
                )
            except Exception as e:
                print(f"SKIP {key}: {e}", file=sys.stderr)
                continue
            if args.dry_run:
                print(f"[dry] {key} path={media[0][0]}", file=sys.stderr)
                continue
            try:
                text = vlm_multimodal_call(args.model, prompt, media)
            except Exception as e:
                text = f"ERROR: {e}"
            rec = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "task": inst.task,
                "robot": inst.robot,
                "instance_id": inst.instance_id,
                "input_type": ity,
                "model": args.model,
                "prompt": prompt,
                "response": text,
                "ground_truth": dict(inst.ground_truth),
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)
            print(f"OK {key} -> {out_path.name}", file=sys.stderr)

    if not args.dry_run and records:
        print(f"\nJSONL: {out_path}\n", file=sys.stderr)
    print(format_markdown_report(records, args.task))
    if (
        not args.dry_run
        and records
        and args.task == "binary_classification"
        and not args.no_pptx
    ):
        pptx_path = Path(args.out_pptx) if args.out_pptx else out_path.with_name(out_path.stem + "_summary.pptx")
        build_binary_results_pptx_from_records(records, pptx_path)
        print(f"PPTX: {pptx_path}\n", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
