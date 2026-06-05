#!/usr/bin/env bash
# Start Qwen-VL on vLLM (OpenAI-compatible API). Run inside salloc / GPU node.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
TP="${VLLM_TENSOR_PARALLEL_SIZE:-1}"

if ! python -c "import vllm" 2>/dev/null; then
  echo "vllm not found — installing (needs CUDA on this node)..."
  pip install "vllm>=0.8.0"
fi

if command -v vllm >/dev/null 2>&1; then
  SERVE=(vllm serve)
else
  SERVE=(python -m vllm.entrypoints.openai.api_server)
fi

CMD=("${SERVE[@]}" "$MODEL" --host "$HOST" --port "$PORT" --max-model-len "$MAX_LEN")
if [[ "$TP" != "1" ]]; then
  CMD+=(--tensor-parallel-size "$TP")
fi

echo "Model:  $MODEL"
echo "Listen: http://${HOST}:${PORT}/v1"
echo "Set in .env: VLM_BASE_URL=http://127.0.0.1:${PORT}/v1"
echo "Command: ${CMD[*]}"
echo "---"
exec "${CMD[@]}"
