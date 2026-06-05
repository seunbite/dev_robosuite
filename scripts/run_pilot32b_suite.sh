#!/usr/bin/env bash
# One-shot pilot suite: 7 experiments × Qwen2.5-VL-32B, single model load, JSON + acc table.
#
# Usage (inside salloc GPU session):
#   bash scripts/run_pilot32b_suite.sh
#   bash scripts/run_pilot32b_suite.sh --only 1,3,7
#   RESUME=1 bash scripts/run_pilot32b_suite.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# HuggingFace cache on /data (avoid home quota)
# shellcheck disable=SC1091
source "$ROOT/scripts/cluster_env.sh" "${HF_HOME:-/data/user_data/${USER}/hf_cache}"

export BACKEND="${BACKEND:-transformers}"
export VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"

PY=python
ARGS=(adhoc/generation/robotarm/run_pilot32b_suite.py --backend "$BACKEND" --model "$VLM_MODEL")

if [[ "${RESUME:-0}" == "1" ]]; then
  ARGS+=(--resume)
fi

if [[ -n "${ONLY:-}" ]]; then
  ARGS+=(--only "$ONLY")
fi

echo "=== pilot32b suite ==="
echo "  model=$VLM_MODEL backend=$BACKEND"
echo "  HF_HOME=$HF_HOME"
echo "  out=data/results/verify/pilot32b_qwen32b/"
echo ""

"$PY" "${ARGS[@]}" "$@"
