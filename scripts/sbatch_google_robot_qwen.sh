#!/bin/bash
#SBATCH --job-name=gr-qwen
#SBATCH --output=logs/google_robot_qwen_%j.out
#SBATCH --error=logs/google_robot_qwen_%j.err
#SBATCH --time=48:00:00
#
# Google Robot pilot-90 × Qwen (same prompts/code as Gemini via exp.py).
#
# Order: 1 → 7 → 7_1 → 2–6 → 8–10
#
#   sbg2 --export=ALL,MODEL_SIZE=32b \
#     scripts/sbatch_google_robot_qwen.sh
#   sbg  --export=ALL,MODEL_SIZE=7b,ONLY=1,7,7_1 \
#     scripts/sbatch_google_robot_qwen.sh
#   sbg4 --export=ALL,MODEL_SIZE=3b,ONLY=2,3,4,5,6,8,9,10 \
#     scripts/sbatch_google_robot_qwen.sh
#
# Env (optional):
#   MODEL_SIZE=32b|7b|3b          default 32b
#   ONLY=1,7,7_1,2,3,4,5,6,8,9,10  default full suite order
#   RESUME=1                      skip done cues/steps (default 1)
#   FORCE_GENERATE=1              re-run all 90 cues for exp1/7/7_1 (ignore existing configs)
#   RESUME=0                      re-run exp5/6 multitile from scratch (merge off)
#   BACKEND=transformers|vllm     default transformers
#   MOTION_PREPARE_MP4=0          skip PNG/MP4 rebuild if Gemini media exists
#   LIMIT=5                       smoke: cap verify steps 2–10 to N cues
#   SKIP_GIT_PULL=1               do not git pull on cluster

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}:$ROOT/adhoc/generation/google_robot"
# shellcheck disable=SC1091
source "$ROOT/scripts/cluster_env.sh" "${HF_HOME:-/data/user_data/${USER}/hf_cache}"

export DOMAIN=google_robot
export MODEL_SIZE="${MODEL_SIZE:-32b}"
export ONLY="${ONLY:-1,7,7_1,2,3,4,5,6,8,9,10}"
export RESUME="${RESUME:-1}"
export BACKEND="${BACKEND:-transformers}"
export FORCE_GENERATE="${FORCE_GENERATE:-0}"
export MOTION_PREPARE_MP4="${MOTION_PREPARE_MP4:-1}"
export LIMIT="${LIMIT:-0}"

# shellcheck disable=SC1091
source "$ROOT/scripts/activate_cluster_vlm.sh" "$ROOT"

echo "=== sbatch google_robot qwen ==="
echo "  host=$(hostname) job=${SLURM_JOB_ID:-local} partition=${SLURM_JOB_PARTITION:-?}"
echo "  model_size=$MODEL_SIZE only=$ONLY resume=$RESUME backend=$BACKEND"
echo "  motion_prepare_mp4=$MOTION_PREPARE_MP4 limit=$LIMIT force_generate=$FORCE_GENERATE"
nvidia-smi || true

if [[ "${SKIP_GIT_PULL:-0}" != "1" ]]; then
  if ! git pull --ff-only 2>/dev/null && ! git pull 2>/dev/null; then
    echo "[warn] git pull skipped — dirty working tree" >&2
  fi
fi

bash exp.sh "$ONLY" "" "$BACKEND" google_robot

echo "=== done ==="
echo "  summary: data/results/verify/google_robot/exp/pilot40_suite_summary_*"
echo "  html:    data/results/html/google_robot/index.html"
