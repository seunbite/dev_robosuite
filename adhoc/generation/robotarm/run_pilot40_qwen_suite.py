#!/usr/bin/env python3
"""
Pilot-40 (39 cues) × 10 experiments with one Qwen2.5-VL-32B load.

  bash scripts/run_pilot40_qwen_suite.sh
  bash scripts/run_pilot40_qwen_suite.sh --only 5,6
  ONLY=5,6 bash scripts/run_pilot40_qwen_suite.sh
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
from pilot40_experiment_suite import (  # noqa: E402
    CONSOLIDATED,
    DEFAULT_QWEN_OUT,
    EXPERIMENT_SPECS,
    MOTION_CFG,
    MOTION_MANIFEST,
    POSE_CFG,
    SHOTS,
    TILE_DIR,
    TILE_PICK,
    metrics_from_json,
    pilot40_cues_csv,
    pose_generation_correct,
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


def _first_pose(row: dict[str, Any]) -> dict[str, Any]:
    for step in row.get("movements", []):
        if step.get("type") == "pose":
            return step.get("parameters", {}).get("pose", {}) or {}
    return {}


def _score_pose_generation(out_json: Path) -> None:
    consolidated = {
        r["cue"]: r for r in json.loads(CONSOLIDATED.read_text(encoding="utf-8")).get("rows", [])
    }
    cfg_rows = json.loads(POSE_CFG.read_text(encoding="utf-8"))
    rows_out: list[dict[str, Any]] = []
    ok = n = 0
    for row in sorted(cfg_rows, key=lambda r: int(r.get("idx", 0))):
        cue = row.get("cue")
        ev = consolidated.get(cue)
        if not ev:
            continue
        gen_pose = _first_pose(row)
        correct = pose_generation_correct(gen_pose, ev.get("groundtruth", ""))
        if correct is not None:
            n += 1
            if correct:
                ok += 1
        rows_out.append(
            {
                "cue_idx": ev.get("cue_idx"),
                "cue": cue,
                "groundtruth": ev.get("groundtruth"),
                "generation": gen_pose,
                "generation_correct": correct,
            }
        )
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "pose_generation_vs_human_gt",
        "config_json": str(POSE_CFG),
        "n": n,
        "n_correct": ok,
        "accuracy": ok / n if n else None,
        "rows": rows_out,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[1] pose generation score: {ok}/{n}", flush=True)


def _score_motion_generation(out_json: Path) -> None:
    from score_pilot40_motion_gt_components import (  # noqa: WPS433
        _tail_matches_component,
        _tail_steps,
    )

    gt_path = _REPO / "data/results/verify/pilot40_motion_component_gt.json"
    ann = {
        int(a["cue_idx"]): a
        for a in json.loads(gt_path.read_text(encoding="utf-8")).get("annotations", [])
    }
    cfg_rows = json.loads(MOTION_CFG.read_text(encoding="utf-8"))
    rows_out: list[dict[str, Any]] = []
    ok = n = 0
    for row in sorted(cfg_rows, key=lambda r: int(r["idx"])):
        idx = int(row["idx"])
        comp = (ann.get(idx) or {}).get("component")
        if not comp:
            continue
        tail = _tail_steps(row.get("movements") or [])
        match, _ = _tail_matches_component(tail, comp)
        n += 1
        if match:
            ok += 1
        rows_out.append(
            {
                "cue_idx": idx,
                "cue": row["cue"],
                "component_match": match,
            }
        )
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "motion_generation_vs_component_gt",
        "config_json": str(MOTION_CFG),
        "n": n,
        "n_correct": ok,
        "accuracy": ok / n if n else None,
        "rows": rows_out,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[7] motion generation score: {ok}/{n}", flush=True)


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
        image_dir=_REPO / "data/results/visualize/pose_pairwise_12_pilot40",
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
        cues=pilot40_cues_csv(),
    )
    run(ns)


def _run_multitile(spec: dict[str, Any], args: argparse.Namespace, out_json: Path) -> None:
    from verify_pose_multitile_gt_gemini import run

    ns = argparse.Namespace(
        consolidated_json=CONSOLIDATED,
        tile_dir=TILE_DIR,
        tile_pick_json=TILE_PICK,
        image_dir=_REPO / "data/results/visualize/pose_multitile_gt_pilot40",
        out_json=out_json,
        model=args.model,
        vlm_backend=_vlm_backend_name(args.backend),
        grid_sizes=spec.get("grid_sizes", "6"),
        max_cues=None,
        cue_indices=None,
        cues=pilot40_cues_csv(),
        temporal_prompt=False,
        dry_run=False,
        resume=args.resume,
    )
    run(ns)


def _prepare_motion_media_if_needed() -> tuple[int, list[str]]:
    if os.getenv("MOTION_PREPARE_MP4", "1") == "0":
        return 0, []
    from motion_media_paths import prepare_pilot40_motion_mp4s, write_pilot40_manifest  # noqa: WPS433

    rows = json.loads(MOTION_CFG.read_text(encoding="utf-8"))
    todo = [(int(r["idx"]), str(r["cue"])) for r in rows]
    ready, failures = prepare_pilot40_motion_mp4s(_REPO, _HERE, todo, config_json=MOTION_CFG)
    manifest = write_pilot40_manifest(_REPO, rows)
    print(f"[suite] motion media: {ready}/{len(todo)} mp4 ready → {manifest}", flush=True)
    if failures and ready < len(todo):
        print(f"[suite] {len(failures)} media issues (showing first 3):", flush=True)
        for line in failures[:3]:
            print(f"  {line}", flush=True)
    return ready, failures


def _run_motion_verify_vlm(args: argparse.Namespace, out_json: Path) -> None:
    from verify_motion_component_gemini import run

    ns = argparse.Namespace(
        model=args.model,
        vlm_backend=_vlm_backend_name(args.backend),
        vlm=getattr(args, "vlm", None),
        out_json=out_json,
        fewshot_n=4,
        limit=0,
        resume=args.resume,
        force=False,
        dry_run=False,
        prepare_media=os.getenv("MOTION_PREPARE_MP4", "1") != "0",
        manifest=MOTION_MANIFEST,
    )
    run(ns)


def _run_motion_verify_text(args: argparse.Namespace, out_json: Path) -> None:
    from verify_motion_component_text_gemini import run

    ns = argparse.Namespace(
        model=args.model,
        vlm_backend=_vlm_backend_name(args.backend),
        vlm=getattr(args, "vlm", None),
        out_json=out_json,
        fewshot_n=4,
        limit=0,
        resume=args.resume,
        force=False,
        dry_run=False,
    )
    run(ns)


def _run_motion_pairwise_mp4(args: argparse.Namespace, out_json: Path) -> None:
    from verify_motion_gt_neg_pairwise_vlm import run

    ns = argparse.Namespace(
        out_json=out_json,
        model=args.model,
        vlm_backend=_vlm_backend_name(args.backend),
        vlm=getattr(args, "vlm", None),
        pairwise_jsons=None,
        limit=0,
        resume=args.resume,
        force=False,
        dry_run=False,
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
    elif kind == "motion_generation_score":
        _score_motion_generation(out_json)
    elif kind == "motion_verify_vlm":
        _run_motion_verify_vlm(args, out_json)
    elif kind == "motion_verify_text":
        _run_motion_verify_text(args, out_json)
    elif kind == "motion_pairwise_mp4":
        _run_motion_pairwise_mp4(args, out_json)
    else:
        raise ValueError(kind)
    return out_json


def main() -> None:
    p = argparse.ArgumentParser(description="Run pilot-40 Qwen suite (10 steps, 39 cues)")
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
    p.add_argument("--only", type=str, default=None, help="Comma-separated step ids 1-10")
    p.add_argument("--skip-model-load", action="store_true", help="Score-only steps (1,7) without GPU")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = setup_hf_cache(os.environ.get("HF_HOME"))
    print(f"[hf] cache root: {cache_root}", flush=True)

    specs = EXPERIMENT_SPECS
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        specs = [s for s in specs if s["id"] in want]

    if args.summary_only:
        all_metrics = [
            {**metrics_from_json(args.out_dir / spec["out_name"], spec), "experiment_id": spec["id"], "title": spec["title"]}
            for spec in specs
        ]
        print_summary_table(specs, all_metrics)
        summary_path = args.out_dir / "pilot40_qwen_suite_summary.json"
        summary = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": args.model,
            "backend": args.backend,
            "n_cues": 39,
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

    needs_motion_media = any(s["kind"] == "motion_verify_vlm" for s in specs)
    if needs_motion_media:
        ready, _ = _prepare_motion_media_if_needed()
        if ready == 0:
            raise SystemExit(
                "Step 8 aborted: no motion MP4s found or built.\n"
                "  • Need data/seed/_remainder/closest_poses_results.jsonl (not in git)\n"
                "  • Or pre-copy MP4s to data/results/render/manipulator/motion_vlm_verify_pilot40/mp4/\n"
                "  • Run first: python adhoc/generation/robotarm/prepare_pilot40_motion_mp4.py\n"
                "  • Check [render fail] lines above for robosuite/mujoco errors"
            )

    needs_model = any(
        s["kind"]
        not in {
            "pose_generation_score",
            "motion_generation_score",
        }
        for s in specs
    )
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
        "n_cues": 39,
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
    summary_path = args.out_dir / "pilot40_qwen_suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote suite summary → {summary_path}\n", flush=True)


if __name__ == "__main__":
    main()
