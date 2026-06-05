#!/usr/bin/env python3
"""
Single-process pose VLM evaluation (sbatch-friendly). No HTTP server.

Default backend: transformers (works on older NVIDIA drivers / Babel-style clusters).
Use --backend vllm only on nodes with a new driver + matching vLLM torch stack.

Examples:
  bash scripts/install_vlm_transformers.sh
  python adhoc/generation/robotarm/run_pose_vlm_eval.py --experiment multitile20
  sbatch --partition=YOUR_PART --gres=gpu:2 scripts/sbatch_pose_vlm.sh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hf_cache_setup import setup_hf_cache  # noqa: E402

# Default HF cache to /data before huggingface_hub import anywhere downstream
setup_hf_cache(os.environ.get("HF_HOME"))

from gpu_check import require_cuda_gpu  # noqa: E402
from vlm_client import init_inprocess_engine, is_transformers_backend, is_vllm_local_backend  # noqa: E402

_BACKEND_CHOICES = ("transformers", "vllm", "local")


def _vlm_backend_name(backend: str) -> str:
    if backend == "vllm":
        return "local"
    return backend


def _out_suffix(backend: str) -> str:
    return "hf_local" if is_transformers_backend(backend) else "vllm_local"


def _print_multitile_summary(path: Path) -> None:
    if not path.is_file():
        print(f"  (missing {path})")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary") or {}
    for key, s in sorted(summary.items()):
        n = s.get("n") or 0
        ok = s.get("ok") or 0
        acc = s.get("accuracy")
        acc_txt = f"{100 * acc:.1f}%" if acc is not None else "n/a"
        rand = s.get("random_baseline")
        rand_txt = f", random {100 * rand:.1f}%" if rand else ""
        print(f"  {key}: {ok}/{n} = {acc_txt}{rand_txt}")
    print(f"  → {path}")


def _print_pairwise_summary(path: Path) -> None:
    if not path.is_file():
        print(f"  (missing {path})")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    scored = data.get("n_scored") or sum(
        1 for c in (data.get("comparisons") or []) if "vlm_correct" in c
    )
    ok = sum(1 for c in (data.get("comparisons") or []) if c.get("vlm_correct"))
    acc = data.get("accuracy")
    if acc is None and scored:
        acc = ok / scored
    acc_txt = f"{100 * acc:.1f}%" if acc is not None else "n/a"
    print(f"  pairwise: {ok}/{scored} = {acc_txt}")
    print(f"  → {path}")


def _run_multitile(
    args: argparse.Namespace,
    *,
    max_cues: int,
    out_json: Path,
    vlm_backend: str,
) -> None:
    from verify_pose_multitile_gt_gemini import run

    ns = argparse.Namespace(
        consolidated_json=_REPO / "data/results/verify/pilot40_pose_eval_consolidated.json",
        tile_dir=_REPO / "data/results/visualize/pose_groups_12",
        tile_pick_json=_REPO / "data/results/verify/pose_tile_pick_by_group.json",
        image_dir=_REPO / "data/results/visualize/pose_multitile_gt",
        out_json=out_json,
        model=args.model,
        vlm_backend=vlm_backend,
        grid_sizes=args.grid_sizes,
        max_cues=max_cues,
        cue_indices=args.cue_indices,
        dry_run=False,
        resume=args.resume,
    )
    run(ns)


def _run_pairwise(
    args: argparse.Namespace,
    *,
    max_cues: int,
    out_json: Path,
    vlm_backend: str,
) -> None:
    from verify_pose_pairwise_12_gemini import run

    ns = argparse.Namespace(
        consolidated_json=_REPO / "data/results/verify/pilot40_pose_eval_consolidated.json",
        tile_dir=_REPO / "data/results/visualize/pose_groups_12",
        tile_pick_json=_REPO / "data/results/verify/pose_tile_pick_by_group.json",
        image_dir=_REPO / "data/results/visualize/pose_pairwise_12",
        out_json=out_json,
        model=args.model,
        vlm_backend=vlm_backend,
        dry_run=False,
        max_cues=max_cues,
        max_pairs_per_cue=None,
        one_pair_per_cue=True,
        append_results=args.resume,
        replace_cues=False,
        exclude_cues=None,
        cue_indices=args.cue_indices,
        cues=None,
    )
    run(ns)


def main() -> None:
    p = argparse.ArgumentParser(description="In-process pose VLM eval (sbatch-friendly)")
    p.add_argument(
        "--experiment",
        choices=["multitile20", "multitile100", "pairwise20", "all20"],
        default="multitile20",
    )
    p.add_argument(
        "--backend",
        choices=_BACKEND_CHOICES,
        default=os.getenv("BACKEND", os.getenv("VLM_BACKEND", "transformers")),
        help="transformers=HF (old drivers OK); vllm/local=vLLM in-process",
    )
    p.add_argument("--model", default=os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"))
    p.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")))
    p.add_argument("--max-model-len", type=int, default=int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")))
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")),
    )
    p.add_argument("--grid-sizes", default="6,12")
    p.add_argument("--cue-indices", type=str, default=None)
    p.add_argument("--out-dir", type=Path, default=_REPO / "data/results/verify")
    p.add_argument(
        "--hf-cache",
        type=Path,
        default=None,
        help="HF cache root (default: HF_HOME or /data/user_data/$USER/hf_cache)",
    )
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    if args.hf_cache:
        setup_hf_cache(args.hf_cache)
    cache_root = setup_hf_cache(os.environ.get("HF_HOME"))
    print(f"[hf] cache root: {cache_root}", flush=True)

    if args.backend == "vllm":
        args.backend = "local"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vlm_backend = _vlm_backend_name(args.backend)
    os.environ["VLM_BACKEND"] = vlm_backend
    os.environ["VLM_MODEL"] = args.model

    require_cuda_gpu()

    print(f"=== Loading model (backend={vlm_backend}, no HTTP) ===", flush=True)
    if is_vllm_local_backend(vlm_backend):
        from vllm_local import get_vllm_engine

        get_vllm_engine(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    else:
        init_inprocess_engine(vlm_backend, args.model)

    suffix = _out_suffix(vlm_backend)
    mt20_out = args.out_dir / f"pilot20_pose_multitile_{suffix}.json"
    mt100_out = args.out_dir / f"pilot100_pose_multitile_{suffix}.json"
    pw_out = args.out_dir / f"pilot20_pose_pairwise_{suffix}.json"

    if args.experiment in {"multitile20", "all20"}:
        print("\n=== Multitile GT (grid 6 + 12), 20 cues ===", flush=True)
        _run_multitile(args, max_cues=20, out_json=mt20_out, vlm_backend=vlm_backend)
        _print_multitile_summary(mt20_out)

    if args.experiment == "multitile100":
        print("\n=== Multitile GT (grid 6 + 12), 100 cues ===", flush=True)
        _run_multitile(args, max_cues=100, out_json=mt100_out, vlm_backend=vlm_backend)
        _print_multitile_summary(mt100_out)

    if args.experiment in {"pairwise20", "all20"}:
        print("\n=== Pairwise 2-way, 20 cues ===", flush=True)
        _run_pairwise(args, max_cues=20, out_json=pw_out, vlm_backend=vlm_backend)
        _print_pairwise_summary(pw_out)

    print("\n=== Done ===", flush=True)


if __name__ == "__main__":
    main()
