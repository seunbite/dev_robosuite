#!/usr/bin/env python3
"""
Pilot-90 experiment runner (exp 1–10).

  python adhoc/generation/robotarm/exp.py 1
  python adhoc/generation/robotarm/exp.py 7 --backend gemini
  python adhoc/generation/robotarm/exp.py 2,3,4
  python adhoc/generation/robotarm/exp.py all
  python adhoc/generation/robotarm/exp.py all --summary   # scores only, no runs

Legacy: bash exp.sh (or python adhoc/generation/robotarm/exp.py).
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
for p in (_REPO, _REPO / "adhoc/generation", _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hf_cache_setup import setup_hf_cache  # noqa: E402
from gpu_check import require_cuda_gpu  # noqa: E402
from pilot90_experiment_suite import (  # noqa: E402
    GT_PATH,
    MOTION_MANIFEST,
    MOTION_PAIRWISE_DIR,
    N_CUES,
    PAIRWISE_IMG_DIR,
    SHOTS,
    TILE_DIR,
    TILE_PICK,
    experiment_specs_all,
    manifest90_cues_csv,
    manifest90_rows_from_cfg,
    metrics_from_json,
    print_qwen_series_summary,
    print_summary_table,
    score_exp1,
    score_exp7,
    score_verify_motion_json,
    score_verify_pose_json,
)
from pilot90_paths import (  # noqa: E402
    VERIFY_EXP_DIR,
    config_for_experiment,
    model_to_tag,
    result_config_path,
    score_result_path,
    verify_result_path,
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


def _maybe_generate(spec: dict[str, Any], args: argparse.Namespace) -> None:
    eid = spec["id"]
    if eid not in {"1", "7"}:
        return
    if os.getenv("GENERATE", "1") == "0":
        return
    tag = args.model_tag
    out_cfg = result_config_path(eid, tag)
    need = os.getenv("FORCE_GENERATE", "0") == "1"
    if not need and out_cfg.is_file():
        from pilot90_paths import manifest90_cue_names, row_generation_done  # noqa: WPS433

        rows = json.loads(out_cfg.read_text(encoding="utf-8"))
        have = {r["cue"] for r in rows if row_generation_done(r)}
        if len(have) >= len(manifest90_cue_names()):
            return
    print(f"[suite] generating exp{eid} configs → {out_cfg}", flush=True)
    from config_gen_vlm import run_exp_generation  # noqa: WPS433

    run_exp_generation(
        eid,
        out_path=out_cfg,
        model=args.model,
        backend=_vlm_backend_name(args.backend),
        vlm=getattr(args, "vlm", None),
        resume=not need,
        delay=float(os.getenv("GEN_DELAY", "2.0")),
    )


def _run_pose_verify_vlm(args: argparse.Namespace, out_json: Path, pose_cfg: Path) -> None:
    from verify_pose_vlm import run

    ns = argparse.Namespace(
        config_json=pose_cfg,
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


def _run_pose_verify_text(args: argparse.Namespace, out_json: Path, pose_cfg: Path) -> None:
    from verify_pose_text import run

    ns = argparse.Namespace(
        config_json=pose_cfg,
        shots_json=SHOTS,
        model=args.model,
        vlm_backend=_vlm_backend_name(args.backend),
        fewshot_n=4,
        out_json=out_json,
        resume=args.resume,
    )
    run(ns)


def _run_pose_pairwise(args: argparse.Namespace, out_json: Path) -> None:
    from compare_pose_2 import run

    ns = argparse.Namespace(
        consolidated_json=GT_PATH,
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
    from compare_pose_multitile import run

    grid = spec.get("grid_sizes", "6")
    image_dir = TILE_DIR.parent / f"pose_multitile_gt_pilot90_grid{grid}"
    ns = argparse.Namespace(
        consolidated_json=GT_PATH,
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
    motion_cfg: Path, exp08_json: Path, *, resume: bool
) -> tuple[int, list[str]]:
    if os.getenv("MOTION_PREPARE_MP4", "1") == "0":
        return 0, []
    if not motion_cfg.is_file():
        print(
            f"[suite] skip MP4 prep — exp7 config not found yet: {motion_cfg}",
            flush=True,
        )
        return 0, []
    from motion_media_paths import prepare_pilot90_motion_mp4s, write_pilot90_manifest  # noqa: WPS433
    from motion_verify_shared import load_verify_done_indices  # noqa: WPS433

    rows = manifest90_rows_from_cfg(json.loads(motion_cfg.read_text(encoding="utf-8")))
    todo = [(int(r["idx"]), str(r["cue"])) for r in rows]
    if resume:
        skip = load_verify_done_indices(exp08_json)
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
        _REPO, _HERE, todo, config_json=motion_cfg, render_missing=render_missing
    )
    manifest = write_pilot90_manifest(_REPO, rows)
    print(f"[suite] pilot90 motion media: {ready}/{len(todo)} mp4 ready → {manifest}", flush=True)
    if failures and ready < len(todo):
        print(f"[suite] {len(failures)} media issues (showing first 3):", flush=True)
        for line in failures[:3]:
            print(f"  {line}", flush=True)
    return ready, failures


def _count_pairwise_mp4() -> int:
    return len(list(MOTION_PAIRWISE_DIR.glob("*_pair_axis.mp4")))


def _prepare_pairwise_mp4_if_needed() -> tuple[int, list[str]]:
    if os.getenv("MOTION_PREPARE_PAIRWISE", "1") == "0":
        return _count_pairwise_mp4(), []
    existing = _count_pairwise_mp4()
    if existing > 0:
        from build_pilot90_motion_pairwise_specs import main as refresh_pairwise_specs  # noqa: WPS433

        refresh_pairwise_specs()
        print(f"[suite] pairwise mp4: {existing} already on disk → {MOTION_PAIRWISE_DIR}", flush=True)
        return existing, []
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


def _run_motion_verify_vlm(args: argparse.Namespace, out_json: Path, motion_cfg: Path) -> None:
    from verify_movement_vlm import run

    ns = argparse.Namespace(
        model=args.model,
        vlm_backend=_vlm_backend_name(args.backend),
        vlm=getattr(args, "vlm", None),
        out_json=out_json,
        config_json=motion_cfg,
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


def _run_motion_verify_text(args: argparse.Namespace, out_json: Path, motion_cfg: Path) -> None:
    from verify_movement_text import run

    ns = argparse.Namespace(
        model=args.model,
        vlm_backend=_vlm_backend_name(args.backend),
        vlm=getattr(args, "vlm", None),
        out_json=out_json,
        config_json=motion_cfg,
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


def _run_motion_pairwise_mp4(args: argparse.Namespace, out_json: Path, motion_cfg: Path) -> None:
    from compare_movement_2 import run

    pairwise_json = _default_pairwise_specs()
    _validate_pairwise_specs(pairwise_json)
    ns = argparse.Namespace(
        out_json=out_json,
        model=args.model,
        vlm_backend=_vlm_backend_name(args.backend),
        vlm=getattr(args, "vlm", None),
        motion_cfg=motion_cfg,
        pairwise_dir=MOTION_PAIRWISE_DIR,
        pairwise_jsons=[pairwise_json] if pairwise_json.is_file() else None,
        limit=0,
        resume=args.resume,
        force=False,
        dry_run=False,
    )
    run(ns)


def _run_one(spec: dict[str, Any], args: argparse.Namespace) -> Path:
    tag = args.model_tag
    eid = spec["id"]
    out_json = (
        score_result_path(eid, tag)
        if spec["kind"] in {"pose_generation_score", "motion_generation_score"}
        else verify_result_path(eid, tag)
    )
    pose_cfg = config_for_experiment("1", tag)
    motion_cfg = config_for_experiment("7", tag)

    print(f"\n{'=' * 72}", flush=True)
    print(f"EXP {eid}: {spec['title']}", flush=True)
    print(f"→ {out_json}", flush=True)
    print("=" * 72, flush=True)

    _maybe_generate(spec, args)
    kind = spec["kind"]
    if kind == "pose_generation_score":
        score_exp1(pose_cfg, out_json)
        print(f"[1] scored {out_json}", flush=True)
    elif kind == "pose_verify_vlm":
        _run_pose_verify_vlm(args, out_json, pose_cfg)
        score_verify_pose_json(out_json, pose_cfg)
    elif kind == "pose_verify_text":
        _run_pose_verify_text(args, out_json, pose_cfg)
        score_verify_pose_json(out_json, pose_cfg)
    elif kind == "pose_pairwise":
        _run_pose_pairwise(args, out_json)
    elif kind == "multitile":
        _run_multitile(spec, args, out_json)
    elif kind == "motion_generation_score":
        score_exp7(motion_cfg, out_json)
        print(f"[7] scored {out_json}", flush=True)
    elif kind == "motion_verify_vlm":
        if not motion_cfg.is_file():
            raise FileNotFoundError(
                f"exp8 needs exp7 config: {motion_cfg}\n"
                "  Include task 7 before 8 (e.g. bash exp.sh or ONLY=7,8 bash exp.sh)"
            )
        exp08 = verify_result_path(8, tag)
        ready, failures = _prepare_motion_media_if_needed(motion_cfg, exp08, resume=args.resume)
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
        _run_motion_verify_vlm(args, out_json, motion_cfg)
        score_verify_motion_json(out_json, motion_cfg)
    elif kind == "motion_verify_text":
        _run_motion_verify_text(args, out_json, motion_cfg)
        score_verify_motion_json(out_json, motion_cfg)
    elif kind == "motion_pairwise_mp4":
        if not motion_cfg.is_file():
            raise FileNotFoundError(
                f"exp10 needs exp7 config: {motion_cfg}\n"
                "  Include task 7 before 10 (e.g. bash exp.sh or ONLY=7,10 bash exp.sh)"
            )
        ready = _count_pairwise_mp4()
        if ready == 0 and os.getenv("MOTION_PREPARE_PAIRWISE", "1") != "0":
            ready, failures = _prepare_pairwise_mp4_if_needed()
            if ready == 0:
                raise SystemExit(
                    "Step 10 aborted: no pilot90 pairwise MP4s built.\n"
                    "  • MuJoCo EGL needs a GPU compute node (not login node):\n"
                    "      salloc --gres=gpu:1 ...\n"
                    "      export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0\n"
                    "      bash scripts/prepare_pilot90_motion_pairwise_mp4.sh\n"
                    "  • Or skip exp10 for now: ONLY=1,2,3,4,5,6,7,8,9 bash exp.sh\n"
                    "  • Or rsync prebuilt MP4s: bash scripts/rsync_to_babel.sh\n"
                    + (f"  • First render error: {failures[0]}" if failures else "")
                )
        _run_motion_pairwise_mp4(args, out_json, motion_cfg)
    else:
        raise ValueError(kind)

    from pilot90_exp_html import write_exp_review_html  # noqa: WPS433

    write_exp_review_html(eid, tag, out_json, title=spec["title"], kind=kind)
    return out_json


def _parse_target(target: str | None) -> list[str] | None:
    if not target or target.lower() == "all":
        return None
    return [x.strip() for x in target.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run pilot-90 experiments (exp 1–10)")
    p.add_argument(
        "target",
        nargs="?",
        default="all",
        help="Experiment id (1–10), comma list (e.g. 2,3), or all",
    )
    p.add_argument("--backend", default=os.getenv("BACKEND", "transformers"))
    p.add_argument("--model", default=os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"))
    p.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")))
    p.add_argument("--max-model-len", type=int, default=int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")))
    p.add_argument("--gpu-memory-utilization", type=float, default=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")))
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("RESUME", "1") != "0",
        help="Skip cues already in step output JSONs (default: on)",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="Skip all runs; read existing result JSONs and print the accuracy table",
    )
    p.add_argument("--only", type=str, default=None, help="Deprecated alias for positional target")
    p.add_argument("--summary-only", action="store_true", help="Deprecated alias for --summary")
    p.add_argument("--skip-model-load", action="store_true", help="Deprecated — generation uses loaded model")
    args = p.parse_args(argv)

    args.model_tag = model_to_tag(args.model)
    cache_root = setup_hf_cache(os.environ.get("HF_HOME"))
    print(f"[hf] cache root: {cache_root}", flush=True)
    print(f"[suite] model_tag={args.model_tag}", flush=True)

    specs = experiment_specs_all(args.model_tag)
    want = _parse_target(args.only if args.only else args.target)
    if want is not None:
        by_id = {s["id"]: s for s in specs}
        specs = [by_id[eid] for eid in want if eid in by_id]

    if args.summary or args.summary_only:
        print_qwen_series_summary()
        return

    motion_cfg = config_for_experiment("7", args.model_tag)
    pose_cfg = config_for_experiment("1", args.model_tag)

    needs_model = any(
        s["kind"]
        not in {"pose_generation_score", "motion_generation_score", "pose_pairwise", "multitile"}
        for s in specs
    )
    if needs_model and not args.skip_model_load:
        backend = _vlm_backend_name(args.backend)
        from vlm_client import VLMClient  # noqa: WPS433

        if backend == "gemini":
            args.vlm = VLMClient(backend="gemini", model=args.model)
        else:
            _init_model(args)
            args.vlm = VLMClient(backend=backend, model=args.model)
    else:
        args.vlm = None

    run_records: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []

    for spec in specs:
        t0 = datetime.now().isoformat(timespec="seconds")
        try:
            out_path = _run_one(spec, args)
            m = metrics_from_json(
                out_path,
                spec,
                pose_cfg=pose_cfg,
                motion_cfg=motion_cfg,
                rescore_json=True,
            )
        except Exception as e:
            out_path = verify_result_path(spec["id"], args.model_tag)
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

    print_summary_table(specs, all_metrics, model_tag=args.model_tag)

    summary_path = VERIFY_EXP_DIR / f"pilot90_suite_summary_{args.model_tag}.json"
    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "model_tag": args.model_tag,
        "backend": args.backend,
        "n_cues": N_CUES,
        "groundtruth": str(GT_PATH),
        "experiments": run_records,
        "table": [
            {
                "id": spec["id"],
                "title": spec["title"],
                "json": str(out_path),
                **{k: v for k, v in m.items() if k not in {"experiment_id", "title"}},
            }
            for spec, m, out_path in zip(
                specs,
                all_metrics,
                [
                    score_result_path(s["id"], args.model_tag)
                    if s["kind"] in {"pose_generation_score", "motion_generation_score"}
                    else verify_result_path(s["id"], args.model_tag)
                    for s in specs
                ],
            )
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote suite summary → {summary_path}\n", flush=True)


if __name__ == "__main__":
    main()
