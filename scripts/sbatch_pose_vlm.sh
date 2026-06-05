#!/bin/bash
#SBATCH --job-name=pose-vlm
#SBATCH --output=logs/pose_vlm_%j.out
#SBATCH --error=logs/pose_vlm_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#
# NO hardcoded partition — pass at submit time, e.g.:
#   sinfo -s                                    # list partitions
#   sbatch --partition=YOUR_PART scripts/sbatch_pose_vlm.sh
#   sbatch --partition=YOUR_PART --gres=gpu:2 EXPERIMENT=multitile20 BACKEND=transformers \
#     VLM_MODEL=Qwen/Qwen2.5-VL-32B-Instruct scripts/sbatch_pose_vlm.sh

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export EXPERIMENT="${EXPERIMENT:-multitile20}"
export BACKEND="${BACKEND:-transformers}"
export VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
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

echo "=== GPU ==="
nvidia-smi || true
echo "partition=${SLURM_JOB_PARTITION:-?} gpus=${SLURM_GPUS_ON_NODE:-?} backend=${BACKEND} model=${VLM_MODEL}"

python adhoc/generation/robotarm/run_pose_vlm_eval.py \
  --experiment "$EXPERIMENT" \
  --backend "$BACKEND" \
  --model "$VLM_MODEL" \
  --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --resume \
  "$@"
