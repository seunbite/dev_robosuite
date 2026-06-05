"""CUDA / driver preflight before loading a VLM."""
from __future__ import annotations


def require_cuda_gpu() -> int:
    """Return visible GPU count or exit with a helpful message."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "ERROR: CUDA not available.\n"
            "  • Login node? Use salloc/sbatch on a GPU node first.\n"
            "  • sbatch: sbatch --partition=YOUR_PART --gres=gpu:1 scripts/sbatch_pose_vlm.sh\n"
            "  • List partitions: sinfo -s"
        )
    try:
        torch.zeros(1, device="cuda")
    except RuntimeError as e:
        msg = str(e)
        if "driver" in msg.lower() or "cuda" in msg.lower():
            raise SystemExit(
                "ERROR: PyTorch cannot use this GPU driver.\n"
                "  vLLM needs a very new driver; use transformers backend instead:\n"
                "    bash scripts/install_vlm_transformers.sh\n"
                "    BACKEND=transformers python adhoc/generation/robotarm/run_pose_vlm_eval.py ...\n"
                f"  Detail: {msg}"
            ) from e
        raise
    n = torch.cuda.device_count()
    print(f"[gpu] {n} device(s): {torch.cuda.get_device_name(0)}", flush=True)
    return n
