#!/usr/bin/env python3
"""
Run all 7 pilot experiments with one Qwen2.5-VL-32B load (salloc-friendly).

  python adhoc/generation/robotarm/run_pilot32b_suite.py

Writes per-experiment JSON under data/results/verify/pilot32b_qwen32b/
and a suite summary JSON + accuracy table on stdout.
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

setup_hf_cache(os.environ.get("HF_HOME"))

from gpu_check import require_cuda_gpu  # noqa: E402
from pilot_experiment_suite import (  # noqa: E402
    EXPERIMENT_SPECS,
    human_gt_is_ok,
    load_consolidated_by_cue,
    metrics_from_json,
    pilot20_cue_names,
    print_summary_table,
    temporal_cue_names,
)
from vlm_client import VLMClient, init_inprocess_engine, is_vllm_local_backend  # noqa: E402

CONSOLIDATED = _REPO / "data/results/verify/pilot40_pose_eval_consolidated.json"
DEFAULT_OUT_DIR = _REPO / "data/results/verify/pilot32b_qwen32b"
CONFIG_PILOT10 = _REPO / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot10.json"
CONFIG_PILOT20 = _REPO / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot20_more.json"
GOOGLE_CFG_A = _REPO / "data/results/motion_configs/google_robot/motion_configs_google_robot_pilot40_manip_shots.json"
GOOGLE_CFG_B = _REPO / "data/results/motion_configs/google_robot/motion_configs_19_mobile.json"
GOOGLE_RENDER_A = _REPO / "data/results/render/google_robot/shots_manip"
GOOGLE_RENDER_B = _REPO / "data/results/render/google_robot/mobile19"


def _vlm_backend_name(backend: str) -> str:
    return "local" if backend == "vllm" else backend


def _init_model(args: argparse.Namespace) -> None:
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


def _run_multitile(spec: dict[str, Any], args: argparse.Namespace, out_json: Path) -> None:
    from verify_pose_multitile_gt_gemini import run

    cues: str | None = None
    if spec.get("cue_filter") == "temporal":
        names = temporal_cue_names()
        if not names:
            print("[warn] no temporal cues from manifest hashtags", flush=True)
        cues = ",".join(names)
    elif spec.get("max_cues") == 20 and not spec.get("cue_filter"):
        cues = ",".join(pilot20_cue_names(CONSOLIDATED))

    ns = argparse.Namespace(
        consolidated_json=CONSOLIDATED,
        tile_dir=_REPO / "data/results/visualize/pose_groups_12",
        tile_pick_json=_REPO / "data/results/verify/pose_tile_pick_by_group.json",
        image_dir=_REPO / "data/results/visualize/pose_multitile_gt",
        out_json=out_json,
        model=args.model,
        vlm_backend=_vlm_backend_name(args.backend),
        grid_sizes=spec.get("grid_sizes", "6,12"),
        max_cues=spec.get("max_cues"),
        cue_indices=None,
        cues=cues,
        temporal_prompt=bool(spec.get("temporal_prompt")),
        dry_run=False,
        resume=args.resume,
    )
    run(ns)


def _run_pairwise(spec: dict[str, Any], args: argparse.Namespace, out_json: Path) -> None:
    from verify_pose_pairwise_12_gemini import run

    cues = ",".join(pilot20_cue_names(CONSOLIDATED))
    ns = argparse.Namespace(
        consolidated_json=CONSOLIDATED,
        tile_dir=_REPO / "data/results/visualize/pose_groups_12",
        tile_pick_json=_REPO / "data/results/verify/pose_tile_pick_by_group.json",
        image_dir=_REPO / "data/results/visualize/pose_pairwise_12",
        out_json=out_json,
        model=args.model,
        vlm_backend=_vlm_backend_name(args.backend),
        dry_run=False,
        max_cues=spec.get("max_cues", 20),
        max_pairs_per_cue=None,
        one_pair_per_cue=True,
        append_results=args.resume,
        replace_cues=False,
        exclude_cues=None,
        cue_indices=None,
        cues=cues,
    )
    run(ns)


def _run_fewshot(spec: dict[str, Any], args: argparse.Namespace, out_json: Path) -> None:
    from verify_pose_tiles_gemini import _load_json, run

    rows10 = _load_json(CONFIG_PILOT10)
    rows20 = _load_json(CONFIG_PILOT20)
    merged = {str(r["cue"]): r for r in rows10 + rows20}
    pilot20 = pilot20_cue_names(CONSOLIDATED)
    merged_rows = [merged[c] for c in pilot20 if c in merged]

    tmp_cfg = out_json.parent / "_fewshot_pilot20_configs.json"
    tmp_cfg.write_text(json.dumps(merged_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    ns = argparse.Namespace(
        config_json=tmp_cfg,
        shots_json=_REPO / "data/seed/shots/manipulator/shot_configs_v19_sophisticated.json",
        tile_dir=_REPO / "data/results/visualize/pose_groups_12",
        tile_pick_json=_REPO / "data/results/verify/pose_tile_pick_by_group.json",
        selected_tile_dir=_REPO / "data/results/visualize/pose_groups_12_selected",
        export_selected=False,
        vlm_backend=_vlm_backend_name(args.backend),
        model=args.model,
        fewshot_n=4,
        out_json=out_json,
        out_md=out_json.with_suffix(".md"),
        no_checkpoint=False,
    )
    run(ns)

    human = load_consolidated_by_cue(CONSOLIDATED)
    data = json.loads(out_json.read_text(encoding="utf-8"))
    agree_ok = agree_n = 0
    for r in data.get("results") or []:
        cue = r.get("cue")
        if not cue or cue not in human or "error" in r:
            continue
        model_ok = r.get("result", {}).get("pose_is_appropriate")
        if model_ok is None:
            continue
        agree_n += 1
        if bool(model_ok) == human_gt_is_ok(human[cue].get("groundtruth", "")):
            agree_ok += 1
    data["agreement_with_human"] = {"ok": agree_ok, "n": agree_n}
    if agree_n:
        data["agreement_with_human"]["accuracy"] = agree_ok / agree_n
    out_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_google_robot(spec: dict[str, Any], args: argparse.Namespace, out_json: Path) -> None:
    from compare_google_robot_vlm import run as gr_run

    ns = argparse.Namespace(
        config_a=GOOGLE_CFG_A,
        config_b=GOOGLE_CFG_B,
        render_dir_a=GOOGLE_RENDER_A,
        render_dir_b=GOOGLE_RENDER_B,
        model=args.model,
        vlm_backend=_vlm_backend_name(args.backend),
        limit=spec.get("limit", 40),
        prompt_file=None,
        out_json=out_json,
    )
    vlm = VLMClient(backend=ns.vlm_backend, model=ns.model)
    gr_run(ns, vlm=vlm)


def _run_one(spec: dict[str, Any], args: argparse.Namespace, out_dir: Path) -> Path:
    out_json = out_dir / spec["out_name"]
    print(f"\n{'=' * 72}", flush=True)
    print(f"EXP {spec['id']}: {spec['title']}", flush=True)
    print(f"→ {out_json}", flush=True)
    print("=" * 72, flush=True)

    kind = spec["kind"]
    if kind == "multitile":
        _run_multitile(spec, args, out_json)
    elif kind == "pairwise":
        _run_pairwise(spec, args, out_json)
    elif kind == "fewshot":
        _run_fewshot(spec, args, out_json)
    elif kind == "google_robot":
        _run_google_robot(spec, args, out_json)
    else:
        raise ValueError(kind)
    return out_json


def main() -> None:
    p = argparse.ArgumentParser(description="Run 7 pilot Qwen-32B experiments in one session")
    p.add_argument("--backend", default=os.getenv("BACKEND", "transformers"))
    p.add_argument("--model", default=os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"))
    p.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")))
    p.add_argument("--max-model-len", type=int, default=int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")))
    p.add_argument("--gpu-memory-utilization", type=float, default=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")))
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--only", type=str, default=None, help="Comma-separated experiment ids (e.g. 1,3,7)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = setup_hf_cache(os.environ.get("HF_HOME"))
    print(f"[hf] cache root: {cache_root}", flush=True)

    specs = EXPERIMENT_SPECS
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        specs = [s for s in specs if s["id"].split("_")[0] in want or s["id"] in want]

    _init_model(args)

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
        run_records.append({"id": spec["id"], "started": t0, "finished": datetime.now().isoformat(timespec="seconds"), "metrics": m})

    print_summary_table(specs, all_metrics)

    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "backend": args.backend,
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
    summary_path = args.out_dir / "pilot32b_suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote suite summary → {summary_path}\n", flush=True)


if __name__ == "__main__":
    main()
