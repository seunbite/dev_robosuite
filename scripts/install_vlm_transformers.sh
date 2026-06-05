#!/usr/bin/env bash
# PyTorch + transformers stack for older cluster GPU drivers (no vLLM).
# Run inside salloc/sbatch GPU session after: micromamba activate robosuite-vlm
set -euo pipefail

CUDA_TAG="${CUDA_WHEEL:-cu118}"
echo "Installing torch (${CUDA_TAG}) + transformers + qwen-vl-utils ..."
echo "Override wheel tag if needed: CUDA_WHEEL=cu121 bash $0"

pip uninstall -y vllm 2>/dev/null || true
pip install --upgrade pip
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install "transformers>=4.45.0" "accelerate>=0.30.0" "qwen-vl-utils>=0.0.8" pillow pyyaml openai httpx

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA still not available — are you on a GPU node?"
torch.zeros(1, device="cuda")
print("CUDA smoke test OK on", torch.cuda.get_device_name(0))
PY

echo "Done. Run with:"
echo "  BACKEND=transformers python adhoc/generation/robotarm/run_pose_vlm_eval.py --experiment multitile20"
