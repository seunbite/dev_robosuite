#!/usr/bin/env bash
#SBATCH --job-name=smp-qw32b-hum
#SBATCH --output=logs/smp_qwen32b_humanoid_%j.log
#SBATCH --error=logs/smp_qwen32b_humanoid_%j.log
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --export=ALL
#
# Humanoid Short Motion Code generation with Qwen2.5-VL-32B (yaml_cue).
# Attribute scoring (Orient / Locx / Locy / Locz / Movement) is run locally
# after rsyncing the master JSON back.
#
# Submit:
#   cd ~/sblee/dev_robosuite && mkdir -p logs
#   sbatch --partition=general scripts/sbatch_humanoid_exp0_qwen32b.sh
#   # or: sbg2 --export=ALL scripts/sbatch_humanoid_exp0_qwen32b.sh
#
# Smoke (2 cues):
#   MAX_CUES=2 sbatch --partition=general scripts/sbatch_humanoid_exp0_qwen32b.sh

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

CONDA_SH="${CONDA_SH:-/data/user_data/hoyeonk/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-m2m_caption32b}"
if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.cache}"
export VLLM_USE_CUSTOM_ALLREDUCE=0
export VLLM_USE_V1=0
export VLLM_ENFORCE_EAGER=1
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLM_TEMPERATURE="${VLM_TEMPERATURE:-0.3}"
export VLM_MAX_NEW_TOKENS="${VLM_MAX_NEW_TOKENS:-4096}"
export PYTHONPATH="$ROOT/adhoc/generation/robotarm:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
PLATFORMS="${PLATFORMS:-humanoid}"
MAX_CUES="${MAX_CUES:-0}"
OUT_JSON="${OUT_JSON:-$ROOT/data/results/motion_configs/embodiment_motion_code_qwen_qwen32b}"

EXTRA=()
if [[ "${MAX_CUES}" != "0" ]]; then
  EXTRA+=(--max-cues "${MAX_CUES}")
fi

python adhoc/generation/robotarm/run_embodiment_exp0_qwen.py \
  --platforms "$PLATFORMS" \
  --model "$MODEL" \
  --out-json "$OUT_JSON" \
  --batch-size 1 \
  --retries 2 \
  --resume \
  "${EXTRA[@]}"
