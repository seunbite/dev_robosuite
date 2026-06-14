#!/usr/bin/env python3
"""
Pilot-90 Google Robot experiment runner (exp 1–10, 90 cues).

  python adhoc/generation/google_robot/exp.py 1
  python adhoc/generation/google_robot/exp.py 7 --backend gemini --model gemini-2.5-pro
  python adhoc/generation/google_robot/exp.py all --summary

Via exp.sh:
  DOMAIN=google_robot MODEL_SIZE=gemini bash exp.sh all
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_ROBOTARM = _REPO / "adhoc/generation/robotarm"
# google_robot must precede robotarm on sys.path — both have a `legacy` package.
for p in (_REPO, _ROBOTARM, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from google_robot_experiment_suite import (  # noqa: E402
    GT_CONSOLIDATED,
    N_CUES,
    SHOTS,
    experiment_specs_all,
    metrics_from_json,
    print_summary_table,
    score_exp1,
    score_exp7,
)
from pilot40_paths import (  # noqa: E402
    LEGACY_CFG,
    MEDIA_DIR,
    MULTITILE_IMG_DIR,
    PAIRWISE_IMG_DIR,
    RENDER_DIR,
    TILE_DIR,
    VERIFY_EXP_DIR,
    config_for_experiment,
    model_to_tag,
    result_config_path,
    score_result_path,
    verify_result_path,
)


def _parse_target(target: str | None) -> list[str] | None:
    if not target or target.lower() == "all":
        return None
    return [x.strip() for x in target.split(",") if x.strip()]


def _vlm_backend_name(backend: str) -> str:
    return "local" if backend == "vllm" else backend


def _init_model(args: argparse.Namespace) -> None:
    from gpu_check import require_cuda_gpu  # noqa: WPS433
    from hf_cache_setup import setup_hf_cache  # noqa: WPS433
    from vlm_client import init_inprocess_engine, is_vllm_local_backend  # noqa: WPS433

    backend = _vlm_backend_name(args.backend)
    os.environ["VLM_BACKEND"] = backend
    os.environ["VLM_MODEL"] = args.model
    cache_root = setup_hf_cache(os.environ.get("HF_HOME"))
    print(f"[hf] cache root: {cache_root}", flush=True)
    require_cuda_gpu()
    print(
        f"\n{'=' * 72}\nLoading {args.model} (backend={backend}, once)\n{'=' * 72}\n",
        flush=True,
    )
    if is_vllm_local_backend(backend):
        from vllm_local import get_vllm_engine  # noqa: WPS433

        get_vllm_engine(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    else:
        init_inprocess_engine(backend, args.model)


def _gemini_client(model: str) -> Any:
    from google import genai

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
    return genai.Client(api_key=key), model


def _maybe_generate(spec: dict[str, Any], args: argparse.Namespace) -> None:
    eid = spec["id"]
    if eid not in {"1", "7"}:
        return
    if os.getenv("GENERATE", "1") == "0":
        return
    tag = args.model_tag
    out_cfg = result_config_path(eid, tag)
    if os.getenv("FORCE_GENERATE", "0") != "1" and out_cfg.is_file():
        from pilot40_paths import load_config_list, manifest90_cue_names, row_generation_done

        rows = load_config_list(out_cfg)
        by_cue = {r["cue"]: r for r in rows if r.get("cue")}
        if len(manifest90_cue_names()) > 0 and all(
            row_generation_done(by_cue.get(c)) for c in manifest90_cue_names()
        ):
            return
    print(f"[suite] generating exp{eid} → {out_cfg}", flush=True)
    from config_gen_mobile_vlm import run_exp_generation  # noqa: WPS433

    run_exp_generation(
        eid,
        out_path=out_cfg,
        model=args.model,
        backend=_vlm_backend_name(args.backend),
        vlm=getattr(args, "vlm", None),
        resume=not (os.getenv("FORCE_GENERATE", "0") == "1"),
        delay=float(os.getenv("GEN_DELAY", "2.0")),
    )


def _vlm_ns(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "vlm": getattr(args, "vlm", None),
        "vlm_backend": _vlm_backend_name(args.backend),
    }


def _prepare_media(config_path: Path) -> None:
    if os.getenv("MOTION_PREPARE_MP4", "1") == "0":
        return
    script = _HERE / "prepare_pilot40_media.py"
    if not script.is_file():
        return
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--config-json",
            str(config_path),
            "--gif-dir",
            str(RENDER_DIR),
            "--media-dir",
            str(MEDIA_DIR),
        ],
        cwd=str(_REPO),
        check=False,
    )


def _run_pose_verify(args: argparse.Namespace, out_json: Path, pose_cfg: Path, *, exp_id: str) -> None:
    from verify_components_gemini import run

    modality = "vlm" if exp_id == "2" else "text"
    ns = argparse.Namespace(
        config_json=pose_cfg,
        component="pose",
        modality=modality,
        model=args.model,
        exp_id=exp_id,
        prompt_file=None,
        render_dir=RENDER_DIR,
        media_dir=MEDIA_DIR,
        limit=int(os.getenv("LIMIT", "0") or 0),
        out_json=out_json,
        **_vlm_ns(args),
    )
    run(ns)


def _run_motion_verify(args: argparse.Namespace, out_json: Path, motion_cfg: Path, *, exp_id: str) -> None:
    from verify_components_gemini import run

    _prepare_media(motion_cfg)
    modality = "vlm" if exp_id == "8" else "text"
    ns = argparse.Namespace(
        config_json=motion_cfg,
        component="movement",
        modality=modality,
        model=args.model,
        exp_id=exp_id,
        prompt_file=None,
        render_dir=RENDER_DIR,
        media_dir=MEDIA_DIR,
        limit=int(os.getenv("LIMIT", "0") or 0),
        out_json=out_json,
        **_vlm_ns(args),
    )
    run(ns)


def _run_multitile(spec: dict[str, Any], args: argparse.Namespace, out_json: Path) -> None:
    from compare_pose_multitile_mobile import run

    grid = spec.get("grid_sizes", "6")
    img_dir = MULTITILE_IMG_DIR.parent / f"pose_multitile_gt_pilot40_grid{grid}"
    ns = argparse.Namespace(
        consolidated_json=GT_CONSOLIDATED,
        tile_dir=TILE_DIR,
        image_dir=img_dir,
        out_json=out_json,
        model=args.model,
        grid_sizes=grid,
        cues=os.getenv("CUES"),
        max_cues=int(os.getenv("MAX_CUES", "0") or 0),
        resume=args.resume,
        **_vlm_ns(args),
    )
    if not TILE_DIR.is_dir() or not any(TILE_DIR.glob("group_*.png")):
        gen_script = _HERE / "generate_pose_group_tiles.py"
        if gen_script.is_file():
            print(f"[suite] building tiles → {TILE_DIR}", flush=True)
            subprocess.run([sys.executable, str(gen_script), "--output-root", str(TILE_DIR)], cwd=str(_REPO), check=False)
    run(ns)


def _run_pose_pairwise(args: argparse.Namespace, out_json: Path) -> None:
    from compare_components_vlm_gemini import run

    tag = args.model_tag
    cfg_a = result_config_path("1", tag)
    cfg_b = LEGACY_CFG if LEGACY_CFG.is_file() else SHOTS
    ns = argparse.Namespace(
        component="pose",
        config_a=cfg_a,
        config_b=cfg_b,
        render_dir_a=RENDER_DIR,
        render_dir_b=RENDER_DIR,
        model=args.model,
        prompt_file=_REPO / "data/seed/prompt/google_robot/exp/prompt_exp4.txt",
        limit=int(os.getenv("LIMIT", "0") or 0),
        out_json=out_json,
        **_vlm_ns(args),
    )
    run(ns)


def _run_motion_pairwise(args: argparse.Namespace, out_json: Path, motion_cfg: Path) -> None:
    from compare_components_vlm_gemini import run

    _prepare_media(motion_cfg)
    ns = argparse.Namespace(
        component="movement",
        config_a=motion_cfg,
        config_b=LEGACY_CFG if LEGACY_CFG.is_file() else motion_cfg,
        render_dir_a=RENDER_DIR,
        render_dir_b=RENDER_DIR,
        model=args.model,
        prompt_file=_REPO / "data/seed/prompt/google_robot/exp/prompt_exp10.txt",
        limit=int(os.getenv("LIMIT", "0") or 0),
        out_json=out_json,
        **_vlm_ns(args),
    )
    run(ns)


def _run_one(spec: dict[str, Any], args: argparse.Namespace) -> Path:
    tag = args.model_tag
    eid = spec["id"]
    kind = spec["kind"]
    out_json = (
        score_result_path(eid, tag)
        if kind in {"pose_generation_score", "motion_generation_score"}
        else verify_result_path(eid, tag)
    )
    pose_cfg = config_for_experiment("1", tag)
    motion_cfg = config_for_experiment("7", tag)

    print(f"\n{'=' * 72}\nEXP {eid}: {spec['title']}\n→ {out_json}\n{'=' * 72}", flush=True)
    _maybe_generate(spec, args)

    if kind == "pose_generation_score":
        score_exp1(pose_cfg, out_json)
    elif kind == "pose_verify_vlm":
        _run_pose_verify(args, out_json, pose_cfg, exp_id="2")
    elif kind == "pose_verify_text":
        _run_pose_verify(args, out_json, pose_cfg, exp_id="3")
    elif kind == "pose_pairwise":
        _run_pose_pairwise(args, out_json)
    elif kind == "multitile":
        _run_multitile(spec, args, out_json)
    elif kind == "motion_generation_score":
        score_exp7(motion_cfg, out_json)
    elif kind == "motion_verify_vlm":
        _run_motion_verify(args, out_json, motion_cfg, exp_id="8")
    elif kind == "motion_verify_text":
        _run_motion_verify(args, out_json, motion_cfg, exp_id="9")
    elif kind == "motion_pairwise_mp4":
        _run_motion_pairwise(args, out_json, motion_cfg)
    else:
        raise ValueError(kind)

    from pilot40_exp_html import write_exp_html  # noqa: WPS433

    write_exp_html(eid, tag, out_json)
    return out_json


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run pilot-40 Google Robot experiments (exp 1–10)")
    p.add_argument("target", nargs="?", default="all", help="1–10, comma list, or all")
    p.add_argument("--backend", default=os.getenv("BACKEND", "gemini"))
    p.add_argument("--model", default=os.getenv("VLM_MODEL", "gemini-2.5-pro"))
    p.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")))
    p.add_argument("--max-model-len", type=int, default=int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")))
    p.add_argument("--gpu-memory-utilization", type=float, default=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")))
    p.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Do not load in-process VLM (summary / scoring only)",
    )
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=os.getenv("RESUME", "1") != "0")
    p.add_argument("--summary", action="store_true", help="Scores only from existing JSONs")
    p.add_argument("--force", action="store_true", help="Re-copy legacy artifacts into exp/")
    p.add_argument("--open", action="store_true", help="Open HTML index (macOS)")
    p.add_argument("--only", type=str, default=None, help="Deprecated alias for positional target")
    args = p.parse_args(argv)

    if args.force:
        from google_robot_experiment_suite import migrate_pilot40_layout  # noqa: WPS433

        for a in migrate_pilot40_layout(force=True):
            print(f"  [migrate] {a}")

    args.model_tag = model_to_tag(args.model)
    from hf_cache_setup import setup_hf_cache  # noqa: WPS433

    cache_root = setup_hf_cache(os.environ.get("HF_HOME"))
    print(f"[hf] cache root: {cache_root}", flush=True)
    print(f"[suite] robot=google_robot model_tag={args.model_tag} n_cues={N_CUES}", flush=True)

    specs = experiment_specs_all(args.model_tag)
    for spec in specs:
        if spec["id"] == "5":
            spec["grid_sizes"] = "6"
        if spec["id"] == "6":
            spec["grid_sizes"] = "12"

    want = _parse_target(os.getenv("ONLY") or args.only or args.target)
    if want is not None:
        by_id = {s["id"]: s for s in specs}
        specs = [by_id[eid] for eid in want if eid in by_id]

    if args.summary:
        print_summary_table(args.model_tag)
        return

    backend = _vlm_backend_name(args.backend)
    needs_model = any(
        s["kind"]
        not in {"pose_generation_score", "motion_generation_score"}
        for s in specs
    ) or os.getenv("GENERATE", "1") != "0"
    if needs_model and not args.skip_model_load:
        from vlm_client import VLMClient  # noqa: WPS433

        if backend == "gemini":
            args.vlm = VLMClient(backend="gemini", model=args.model)
        else:
            _init_model(args)
            args.vlm = VLMClient(backend=backend, model=args.model)
        print(f"[suite] shared_vlm=yes backend={backend}", flush=True)
    else:
        args.vlm = None

    all_metrics: list[dict[str, Any]] = []
    for spec in specs:
        try:
            out_path = _run_one(spec, args)
            m = metrics_from_json(out_path, spec["kind"])
        except Exception as e:
            out_path = verify_result_path(spec["id"], args.model_tag)
            m = {"status": "error", "error": str(e), "path": str(out_path)}
            print(f"[ERROR] exp{spec['id']}: {e}", flush=True)
        m["experiment_id"] = spec["id"]
        m["title"] = spec["title"]
        all_metrics.append(m)

    print_summary_table(args.model_tag)
    summary_path = VERIFY_EXP_DIR / f"pilot40_suite_summary_{args.model_tag}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "robot": "google_robot",
                "model": args.model,
                "model_tag": args.model_tag,
                "n_cues": N_CUES,
                "table": all_metrics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote suite summary → {summary_path}\n", flush=True)

    if want is None and os.getenv("SKIP_HTML", "0") != "1":
        from pilot40_exp_html import write_all_html  # noqa: WPS433

        try:
            write_all_html()
        except Exception as e:
            print(f"[html] write_all_html skipped: {e}", flush=True)

    if args.open and sys.platform == "darwin":
        idx = _REPO / "data/results/html/google_robot/index.html"
        if idx.is_file():
            subprocess.run(["open", str(idx)], check=False)


if __name__ == "__main__":
    main()
