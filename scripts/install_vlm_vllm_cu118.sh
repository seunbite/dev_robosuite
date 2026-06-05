#!/usr/bin/env bash
# Pin vLLM + PyTorch to cu118 for older NVIDIA drivers (Babel-style clusters).
#
# "driver too old" usually means: latest vLLM (0.22) pulled torch+cu124,
# not that vLLM itself is wrong. Fix = older vLLM + matching torch wheel.
#
# Run on a GPU node (salloc/sbatch), after: micromamba activate robosuite-vlm
set -euo pipefail

CUDA_TAG="${CUDA_WHEEL:-cu118}"
# Qwen2.5-VL needs vLLM >= 0.7.2; 0.8.x avoids the v1 engine in 0.22+.
VLLM_VER="${VLLM_PIN:-0.8.5}"
TORCH_VER="${TORCH_PIN:-2.4.0}"

echo "=== vLLM stack for old drivers ==="
echo "  CUDA wheel: ${CUDA_TAG}"
echo "  torch:      ${TORCH_VER}"
echo "  vllm:       ${VLLM_VER}"
echo "  Override:   VLLM_PIN=0.8.5 TORCH_PIN=2.4.0 CUDA_WHEEL=cu121 bash $0"
echo ""

pip uninstall -y vllm torch torchvision torchaudio 2>/dev/null || true
pip install --upgrade pip

echo "→ torch (${CUDA_TAG}) first (do NOT let vLLM upgrade this)"
pip install "torch==${TORCH_VER}" "torchvision==0.19.0" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "→ vLLM ${VLLM_VER} + vision deps"
pip install "vllm==${VLLM_VER}" \
  "transformers>=4.45.0" \
  "qwen-vl-utils>=0.0.8" \
  "accelerate>=0.30.0" \
  pillow pyyaml openai httpx

python - <<'PY'
import torch
print("torch", torch.__version__, "built_cuda", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available — run on GPU node (salloc/sbatch)")
torch.zeros(1, device="cuda")
print("CUDA OK:", torch.cuda.get_device_name(0))

import vllm
print("vllm", vllm.__version__)
PY

echo ""
echo "Done. Run experiments with vLLM in-process:"
echo "  BACKEND=vllm python adhoc/generation/robotarm/run_pose_vlm_eval.py --experiment multitile20"
echo "  BACKEND=vllm VLLM_TENSOR_PARALLEL_SIZE=2 VLM_MODEL=Qwen/Qwen2.5-VL-32B-Instruct ..."
