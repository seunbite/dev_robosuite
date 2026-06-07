#!/usr/bin/env python3
"""
Pilot-90 (90 non-essence cues) × pose experiments 1–6 with one Qwen2.5-VL load.

  bash scripts/run_pilot90_qwen_suite.sh
  MODEL_SIZE=7b bash scripts/run_pilot90_qwen_suite.sh
  SUMMARY_ONLY=1 bash scripts/run_pilot90_qwen_suite.sh
"""
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

from hf_cache_setup import setup_hf_cache  # noqa: E402
from gpu_check import require_cuda_gpu  # noqa: E402
from pilot90_experiment_suite import (  # noqa: E402
    CONSOLIDATED,
    DEFAULT_QWEN_OUT,
    N_CUES,
    PAIRWISE_IMG_DIR,
    POSE_CFG,
    SHOTS,
    TILE_DIR,
    TILE_PICK,
    experiment_specs_pose_only,
    manifest90_cue_names,
    manifest90_cues_csv,
    metrics_from_json,
    pose_generation_correct_any,
    print_summary_table,
)


def _vlm_backend_name(backend: str) -> str:
    return "local" if backend == "vllm" else backend


def _init_model(args: argparse.Namespace) -> None:
    from vlm_client import init_inprocess_engine, is_vllm_local_backend  # noqa: WPS433

    backend = _vlm_backend_name(args.backend)
    os.environ["VLM_BACKEND"] = backend
    os.environ["VLM_MODEL"] = args.model
    require_cuda_gpu()
    print(f"\n{'=' * 72}\nLoading {args.model} (backend={backend}, once)\n{'=' * 72}\n", flush=True)
    if is_vllm_local_backend(backend):
        from vllm_local import get_vllm_engine

        get_vllm_engine(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    else:
        init_inprocess_engine(backend, args.model)


def _score_pose_generation(out_json: Path) -> None:
    consolidated = {
        r["cue"]: r for r in json.loads(CONSOLIDATED.read_text(encoding="utf-8")).get("rows", [])
    }
    cfg_rows = json.loads(POSE_CFG.read_text(encoding="utf-8"))
    manifest = set(manifest90_cue_names())
    rows_out: list[dict[str, Any]] = []
    ok = n = 0
    for row in sorted(cfg_rows, key=lambda r: int(r.get("idx", 0))):
        cue = row.get("cue")
        if cue not in manifest:
            continue
        ev = consolidated.get(cue)
        if not ev or not ev.get("groundtruth"):
            continue
        correct = pose_generation_correct_any(row, ev.get("groundtruth", ""))
        if correct is not None:
            n += 1
            if correct:
                ok += 1
        rows_out.append(
            {
                "cue_idx": row.get("idx"),
                "cue": cue,
                "groundtruth": ev.get("groundtruth"),
                "generation_correct": correct,
                "scoring": "any_pose_in_config",
            }
        )
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "pose_generation_vs_human_gt_any_pose",
        "config_json": str(POSE_CFG),
        "n_cues_manifest": len(manifest),
        "n": n,
        "n_correct": ok,
        "accuracy": ok / n if n else None,
        "rows": rows_out,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[1] pose generation score (any-pose): {ok}/{n}", flush=True)


def _run_pose_verify_vlm(args: argparse.Namespace, out_json: Path) -> None:
    from verify_pose_tiles_gemini import run

    ns = argparse.Namespace(
        config_json=POSE_CFG,
        shots_json=SHOTS,
        tile_dir=TILE_DIR,
        tile_pick_json=TILE_PICK,
        selected_tile_dir=TILE_DIR.parent / "pose_groups_12_selected",
        export_selected=False,
        vlm_backend=_vlm_backend_name(args.backend),
        model=args.model,
        fewshot_n=4,
        out_json=out_json,
        out_md=out_json.with_suffix(".md"),
        no_checkpoint=False,
    )
    run(ns)


def _run_pose_verify_text(args: argparse.Namespace, out_json: Path) -> None:
    from verify_pose_textonly_gemini import run

    ns = argparse.Namespace(
        config_json=POSE_CFG,
        shots_json=SHOTS,
        model=args.model,
        vlm_backend=_vlm_backend_name(args.backend),
        fewshot_n=4,
        out_json=out_json,
    )
    run(ns)


def _run_pose_pairwise(args: argparse.Namespace, out_json: Path) -> None:
    from verify_pose_pairwise_12_gemini import run

    ns = argparse.Namespace(
        consolidated_json=CONSOLIDATED,
        tile_dir=TILE_DIR,
        tile_pick_json=TILE_PICK,
        image_dir=PAIRWISE_IMG_DIR,
        out_json=out_json,
        model=args.model,
        vlm_backend=_vlm_backend_name(args.backend),
        dry_run=False,
        max_cues=None,
        max_pairs_per_cue=None,
        one_pair_per_cue=True,
        append_results=args.resume,
        replace_cues=False,
        exclude_cues=None,
        cue_indices=None,
        cues=manifest90_cues_csv(),
    )
    run(ns)


def _run_multitile(spec: dict[str, Any], args: argparse.Namespace, out_json: Path) -> None:
    from verify_pose_multitile_gt_gemini import run

    grid = spec.get("grid_sizes", "6")
    image_dir = TILE_DIR.parent / f"pose_multitile_gt_pilot90_grid{grid}"
    ns = argparse.Namespace(
        consolidated_json=CONSOLIDATED,
        tile_dir=TILE_DIR,
        tile_pick_json=TILE_PICK,
        image_dir=image_dir,
        out_json=out_json,
        model=args.model,
        vlm_backend=_vlm_backend_name(args.backend),
        grid_sizes=grid,
        max_cues=None,
        cue_indices=None,
        cues=manifest90_cues_csv(),
        temporal_prompt=False,
        dry_run=False,
        resume=args.resume,
    )
    run(ns)


def _run_one(spec: dict[str, Any], args: argparse.Namespace, out_dir: Path) -> Path:
    out_json = out_dir / spec["out_name"]

    print(f"\n{'=' * 72}", flush=True)
    print(f"EXP {spec['id']}: {spec['title']}", flush=True)
    print(f"→ {out_json}", flush=True)
    print("=" * 72, flush=True)

    kind = spec["kind"]
    if kind == "pose_generation_score":
        _score_pose_generation(out_json)
    elif kind == "pose_verify_vlm":
        _run_pose_verify_vlm(args, out_json)
    elif kind == "pose_verify_text":
        _run_pose_verify_text(args, out_json)
    elif kind == "pose_pairwise":
        _run_pose_pairwise(args, out_json)
    elif kind == "multitile":
        _run_multitile(spec, args, out_json)
    else:
        raise ValueError(kind)
    return out_json


def main() -> None:
    p = argparse.ArgumentParser(description="Run pilot-90 Qwen pose suite (steps 1–6, 90 cues)")
    p.add_argument("--backend", default=os.getenv("BACKEND", "transformers"))
    p.add_argument("--model", default=os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"))
    p.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")))
    p.add_argument("--max-model-len", type=int, default=int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")))
    p.add_argument("--gpu-memory-utilization", type=float, default=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")))
    p.add_argument("--out-dir", type=Path, default=DEFAULT_QWEN_OUT)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip all runs; read existing result JSONs and print the accuracy table",
    )
    p.add_argument("--only", type=str, default=None, help="Comma-separated step ids 1-6")
    p.add_argument("--skip-model-load", action="store_true", help="Score-only step 1 without GPU")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = setup_hf_cache(os.environ.get("HF_HOME"))
    print(f"[hf] cache root: {cache_root}", flush=True)

    specs = experiment_specs_pose_only()
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        specs = [s for s in specs if s["id"] in want]

    if args.summary_only:
        all_metrics = [
            {**metrics_from_json(args.out_dir / spec["out_name"], spec), "experiment_id": spec["id"], "title": spec["title"]}
            for spec in specs
        ]
        print_summary_table(specs, all_metrics)
        summary_path = args.out_dir / "pilot90_qwen_suite_summary.json"
        summary = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": args.model,
            "backend": args.backend,
            "n_cues": N_CUES,
            "scoring": "any_pose_in_config",
            "out_dir": str(args.out_dir),
            "summary_only": True,
            "table": [
                {
                    "id": spec["id"],
                    "title": spec["title"],
                    "json": str(args.out_dir / spec["out_name"]),
                    **{k: v for k, v in m.items() if k not in {"experiment_id", "title"}},
                }
                for spec, m in zip(specs, all_metrics)
            ],
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote suite summary → {summary_path}\n", flush=True)
        return

    needs_model = any(s["kind"] != "pose_generation_score" for s in specs)
    if needs_model and not args.skip_model_load:
        _init_model(args)
        from vlm_client import VLMClient  # noqa: WPS433

        args.vlm = VLMClient(backend=_vlm_backend_name(args.backend), model=args.model)
    else:
        args.vlm = None

    run_records: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []

    for spec in specs:
        t0 = datetime.now().isoformat(timespec="seconds")
        try:
            out_path = _run_one(spec, args, args.out_dir)
            m = metrics_from_json(out_path, spec)
        except Exception as e:
            out_path = args.out_dir / spec["out_name"]
            m = {"status": "error", "error": str(e), "path": str(out_path)}
            print(f"[ERROR] {spec['id']}: {e}", flush=True)
        m["experiment_id"] = spec["id"]
        m["title"] = spec["title"]
        all_metrics.append(m)
        run_records.append(
            {
                "id": spec["id"],
                "started": t0,
                "finished": datetime.now().isoformat(timespec="seconds"),
                "metrics": m,
            }
        )

    print_summary_table(specs, all_metrics)

    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "backend": args.backend,
        "n_cues": N_CUES,
        "scoring": "any_pose_in_config",
        "out_dir": str(args.out_dir),
        "experiments": run_records,
        "table": [
            {
                "id": spec["id"],
                "title": spec["title"],
                "json": str(args.out_dir / spec["out_name"]),
                **{k: v for k, v in m.items() if k not in {"experiment_id", "title"}},
            }
            for spec, m in zip(specs, all_metrics)
        ],
    }
    summary_path = args.out_dir / "pilot90_qwen_suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote suite summary → {summary_path}\n", flush=True)


if __name__ == "__main__":
    main()
