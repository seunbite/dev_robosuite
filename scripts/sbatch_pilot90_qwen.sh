#!/bin/bash
#SBATCH --job-name=pilot90-qwen
#SBATCH --output=logs/pilot90_qwen_%j.out
#SBATCH --error=logs/pilot90_qwen_%j.err
#SBATCH --time=24:00:00
#
# Pilot-90 Qwen suite via existing run_pilot90_qwen_suite.sh (ONLY= subset).
# Partition / GPU count — pass at submit time (your sbg2 / sbg4 aliases).
#
#   sbg2 --export=ALL,MODEL_SIZE=32b,ONLY=10,RESUME=0,MOTION_PREPARE_PAIRWISE=0 \
#     scripts/sbatch_pilot90_qwen.sh
#   sbg2 --export=ALL,MODEL_SIZE=32b,ONLY=5,6 \
#     scripts/sbatch_pilot90_qwen.sh
#   sbg4 --export=ALL,MODEL_SIZE=3b,ONLY=10 \
#     scripts/sbatch_pilot90_qwen.sh
#
# Env (optional):
#   MODEL_SIZE=32b|7b|3b   (default 32b)
#   ONLY=10                (default 10; e.g. 5,6 or 5,6,10)
#   RESUME=0 recommended for exp10 (49-cue v5 specs; ignores stale 88-cue results)
#   PAIRWISE_SPECS=.../pairwise_specs_motion_gt_correct.json  (default)
#   MOTION_PREPARE_MP4=0   MOTION_PREPARE_PAIRWISE=0  (skip re-render on cluster)

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}:$ROOT/adhoc/generation/robotarm"
# shellcheck disable=SC1091
source "$ROOT/scripts/cluster_env.sh" "${HF_HOME:-/data/user_data/${USER}/hf_cache}"

export MODEL_SIZE="${MODEL_SIZE:-32b}"
export ONLY="${ONLY:-10}"
export RESUME="${RESUME:-1}"
export BACKEND="${BACKEND:-transformers}"
export MOTION_PREPARE_MP4="${MOTION_PREPARE_MP4:-0}"
export MOTION_PREPARE_PAIRWISE="${MOTION_PREPARE_PAIRWISE:-0}"
export PAIRWISE_SPECS="${PAIRWISE_SPECS:-data/results/verify/samples/motion_gt_neg_pairwise_pilot90/pairwise_specs_motion_gt_correct.json}"

if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-m2m_caption32b}"
elif command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate robosuite-vlm
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate robosuite-vlm
fi

echo "=== sbatch pilot90 qwen ==="
echo "  host=$(hostname) job=${SLURM_JOB_ID:-local} partition=${SLURM_JOB_PARTITION:-?}"
echo "  model_size=$MODEL_SIZE only=$ONLY resume=$RESUME"
echo "  pairwise_specs=$PAIRWISE_SPECS"
nvidia-smi || true

git pull --ff-only || git pull

bash scripts/run_pilot90_qwen_suite.sh
