#!/bin/bash
#SBATCH --job-name=pose-vlm
#SBATCH --output=logs/pose_vlm_%j.out
#SBATCH --error=logs/pose_vlm_%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
# Adjust partition/account for your cluster:
#SBATCH --partition=gpu

# In-process vLLM pose compare (no vllm serve / no HTTP port).
#
# Usage:
#   mkdir -p logs
#   EXPERIMENT=multitile20 sbatch scripts/sbatch_pose_vlm.sh
#   EXPERIMENT=all20 VLLM_TENSOR_PARALLEL_SIZE=2 sbatch scripts/sbatch_pose_vlm.sh
#
# Experiments: multitile20 | multitile100 | pairwise20 | all20

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export EXPERIMENT="${EXPERIMENT:-multitile20}"
export VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate robosuite-vlm
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate robosuite-vlm
fi

python adhoc/generation/robotarm/run_pose_vlm_eval.py \
  --experiment "$EXPERIMENT" \
  --model "$VLM_MODEL" \
  --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --resume \
  "$@"
