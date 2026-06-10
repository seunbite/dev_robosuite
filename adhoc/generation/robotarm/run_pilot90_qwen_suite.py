#!/usr/bin/env python3
"""
Pilot-90 (90 non-essence cues) × 10 experiments with one Qwen2.5-VL load.

  bash scripts/run_pilot90_qwen_suite.sh
  MODEL_SIZE=7b RESUME=1 bash scripts/run_pilot90_qwen_suite.sh
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
    MOTION_CFG,
    MOTION_COMPONENT_GT,
    MOTION_MANIFEST,
    MOTION_PAIRWISE_DIR,
    N_CUES,
    PAIRWISE_IMG_DIR,
    POSE_CFG,
    SHOTS,
    TILE_DIR,
    TILE_PICK,
    experiment_specs_all,
    manifest90_cue_names,
    manifest90_cues_csv,
    manifest90_rows_from_cfg,
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
    cfg_rows = manifest90_rows_from_cfg(json.loads(POSE_CFG.read_text(encoding="utf-8")))
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


def _score_motion_generation(out_json: Path) -> None:
    from score_pilot40_motion_gt_components import (  # noqa: WPS433
        _tail_matches_component,
        _tail_steps,
    )

    ann = {
        int(a["cue_idx"]): a
        for a in json.loads(MOTION_COMPONENT_GT.read_text(encoding="utf-8")).get("annotations", [])
    }
    cfg_rows = manifest90_rows_from_cfg(json.loads(MOTION_CFG.read_text(encoding="utf-8")))
    rows_out: list[dict[str, Any]] = []
    ok = n = 0
    for row in sorted(cfg_rows, key=lambda r: int(r["idx"])):
        idx = int(row["idx"])
        entry = ann.get(idx) or {}
        comp = entry.get("component")
        raw = (entry.get("annotation_raw") or "").strip()
        if not comp and not entry.get("always_correct") and raw.lower() != "none":
            continue
        tail = _tail_steps(row.get("movements") or [])
        if entry.get("always_correct") or (comp and comp.get("kind") == "any"):
            match = True
        elif comp:
            match, _ = _tail_matches_component(tail, comp)
        else:
            continue
        n += 1
        if match:
            ok += 1
        rows_out.append(
            {
                "cue_idx": idx,
                "cue": row["cue"],
                "annotation_raw": raw,
                "component_match": match,
            }
        )
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "motion_generation_vs_component_gt",
        "config_json": str(MOTION_CFG),
        "groundtruth_json": str(MOTION_COMPONENT_GT),
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
        resume=args.resume,
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
        resume=args.resume,
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


def _prepare_motion_media_if_needed(
    out_dir: Path, *, resume: bool
) -> tuple[int, list[str]]:
    if os.getenv("MOTION_PREPARE_MP4", "1") == "0":
        return 0, []
    from motion_media_paths import prepare_pilot90_motion_mp4s, write_pilot90_manifest  # noqa: WPS433
    from motion_verify_shared import load_verify_done_indices  # noqa: WPS433

    rows = manifest90_rows_from_cfg(json.loads(MOTION_CFG.read_text(encoding="utf-8")))
    todo = [(int(r["idx"]), str(r["cue"])) for r in rows]
    if resume:
        skip = load_verify_done_indices(out_dir / "exp08_motion_verify_vlm.json")
        if skip:
            n_before = len(todo)
            todo = [(i, c) for i, c in todo if i not in skip]
            print(
                f"[suite] resume: MP4 prep for {len(todo)}/{n_before} cues "
                f"({len(skip)} already in exp08)",
                flush=True,
            )
    render_missing = os.getenv("MOTION_RENDER_MISSING", "0") == "1"
    ready, failures = prepare_pilot90_motion_mp4s(
        _REPO, _HERE, todo, config_json=MOTION_CFG, render_missing=render_missing
    )
    manifest = write_pilot90_manifest(_REPO, rows)
    print(f"[suite] pilot90 motion media: {ready}/{len(todo)} mp4 ready → {manifest}", flush=True)
    if failures and ready < len(todo):
        print(f"[suite] {len(failures)} media issues (showing first 3):", flush=True)
        for line in failures[:3]:
            print(f"  {line}", flush=True)
    return ready, failures


def _prepare_pairwise_mp4_if_needed() -> tuple[int, list[str]]:
    if os.getenv("MOTION_PREPARE_PAIRWISE", "1") == "0":
        return 0, []
    from build_pilot90_motion_pairwise_specs import main as refresh_pairwise_specs  # noqa: WPS433
    from motion_pairwise_media import prepare_pilot90_pairwise_mp4s  # noqa: WPS433

    ready, failures = prepare_pilot90_pairwise_mp4s()
    refresh_pairwise_specs()
    print(f"[suite] pilot90 pairwise mp4: {ready} ready → {MOTION_PAIRWISE_DIR}", flush=True)
    if failures:
        print(f"[suite] {len(failures)} pairwise issues (first 3):", flush=True)
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
        config_json=MOTION_CFG,
        fewshot_n=4,
        limit=0,
        resume=args.resume,
        force=False,
        dry_run=False,
        prepare_media=os.getenv("MOTION_PREPARE_MP4", "1") != "0",
        manifest=MOTION_MANIFEST,
        pilot90=True,
    )
    run(ns)


def _run_motion_verify_text(args: argparse.Namespace, out_json: Path) -> None:
    from verify_motion_component_text_gemini import run

    ns = argparse.Namespace(
        model=args.model,
        vlm_backend=_vlm_backend_name(args.backend),
        vlm=getattr(args, "vlm", None),
        out_json=out_json,
        config_json=MOTION_CFG,
        fewshot_n=4,
        limit=0,
        resume=args.resume,
        force=False,
        dry_run=False,
    )
    run(ns)


def _default_pairwise_specs() -> Path:
    from motion_pairwise_media import PAIRWISE_SPECS_MOTION_GT_CORRECT  # noqa: WPS433

    env = os.getenv("PAIRWISE_SPECS")
    if env:
        return Path(env)
    return MOTION_PAIRWISE_DIR / PAIRWISE_SPECS_MOTION_GT_CORRECT


def _validate_pairwise_specs(pairwise_json: Path) -> int:
    if not pairwise_json.is_file():
        raise SystemExit(
            f"Step 10 aborted: missing pairwise specs {pairwise_json}\n"
            "  • Run: bash scripts/build_pilot90_pairwise_media.sh\n"
            "  • Or rsync from local: bash scripts/rsync_to_babel.sh"
        )
    data = json.loads(pairwise_json.read_text(encoding="utf-8"))
    entries = data.get("mp4") or []
    missing = [
        str(e.get("cue", e.get("idx")))
        for e in entries
        if not (_REPO / str(e["pair_mp4"])).is_file()
    ]
    if missing:
        raise SystemExit(
            f"Step 10 aborted: {len(missing)}/{len(entries)} pairwise MP4s missing "
            f"(specs={pairwise_json.name}).\n"
            f"  • First missing: {missing[0]}\n"
            "  • bash scripts/rsync_to_babel.sh"
        )
    n = int(data.get("n_with_mp4") or data.get("n") or len(entries))
    print(
        f"[suite] exp10 specs: {pairwise_json.name} n={n} "
        f"version={data.get('version')} layout={data.get('layout')}",
        flush=True,
    )
    return n


def _run_motion_pairwise_mp4(args: argparse.Namespace, out_json: Path) -> None:
    from verify_motion_gt_neg_pairwise_vlm import run

    pairwise_json = _default_pairwise_specs()
    _validate_pairwise_specs(pairwise_json)
    ns = argparse.Namespace(
        out_json=out_json,
        model=args.model,
        vlm_backend=_vlm_backend_name(args.backend),
        vlm=getattr(args, "vlm", None),
        motion_cfg=MOTION_CFG,
        pairwise_dir=MOTION_PAIRWISE_DIR,
        pairwise_jsons=[pairwise_json] if pairwise_json.is_file() else None,
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
    p = argparse.ArgumentParser(description="Run pilot-90 Qwen suite (10 steps, 90 cues)")
    p.add_argument("--backend", default=os.getenv("BACKEND", "transformers"))
    p.add_argument("--model", default=os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"))
    p.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")))
    p.add_argument("--max-model-len", type=int, default=int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")))
    p.add_argument("--gpu-memory-utilization", type=float, default=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")))
    p.add_argument("--out-dir", type=Path, default=DEFAULT_QWEN_OUT)
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("RESUME", "1") != "0",
        help="Skip cues already in step output JSONs (default: on)",
    )
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

    specs = experiment_specs_all()
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
            "pose_scoring": "any_pose_in_config",
            "motion_groundtruth": str(MOTION_COMPONENT_GT),
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

    needs_pairwise_media = any(s["kind"] == "motion_pairwise_mp4" for s in specs)
    if needs_pairwise_media:
        if os.getenv("MOTION_PREPARE_PAIRWISE", "1") == "0":
            ready = len(list(MOTION_PAIRWISE_DIR.glob("*_pair_axis.mp4")))
            print(f"[suite] MOTION_PREPARE_PAIRWISE=0 — {ready} pairwise mp4 on disk", flush=True)
        else:
            ready, _ = _prepare_pairwise_mp4_if_needed()
        if ready == 0:
            raise SystemExit(
                "Step 10 aborted: no pilot90 pairwise MP4s built.\n"
                "  • Run: bash scripts/prepare_pilot90_motion_pairwise_mp4.sh\n"
                "  • Needs closest_poses_results.jsonl + ffmpeg + MuJoCo display/offscreen\n"
                "  • Set MOTION_PREPARE_PAIRWISE=0 to skip auto-build"
            )

    needs_motion_media = any(s["kind"] == "motion_verify_vlm" for s in specs)
    if needs_motion_media:
        ready, failures = _prepare_motion_media_if_needed(args.out_dir, resume=args.resume)
        exp08 = args.out_dir / "exp08_motion_verify_vlm.json"
        from motion_verify_shared import load_verify_done_indices  # noqa: WPS433

        already = len(load_verify_done_indices(exp08)) if args.resume and exp08.is_file() else 0
        if ready == 0 and already == 0:
            raise SystemExit(
                "Step 8 aborted: no pilot90 motion MP4s found or built.\n"
                "  • GIFs expected under run/IIWA/ (from render_manipulator_20260608)\n"
                "  • Or run: bash scripts/prepare_pilot90_motion_mp4.sh\n"
                "  • Set MOTION_PREPARE_MP4=0 to skip auto-build"
            )
        if failures and ready == 0 and already == 0:
            raise SystemExit(
                "Step 8 aborted: MP4 prep failed for all pending cues.\n"
                f"  • First issue: {failures[0]}"
            )

    needs_model = any(
        s["kind"] not in {"pose_generation_score", "motion_generation_score"} for s in specs
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
        "n_cues": N_CUES,
        "pose_scoring": "any_pose_in_config",
        "motion_groundtruth": str(MOTION_COMPONENT_GT),
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
