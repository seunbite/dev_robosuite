#!/usr/bin/env bash
#SBATCH --job-name=gr-exp0-qw7b
#SBATCH --output=logs/gr_exp0_qwen7b_%j.log
#SBATCH --error=logs/gr_exp0_qwen7b_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#
# Google Robot exp0 generation (prompt_exp0.txt) — Qwen2.5-VL-7B
#
# Submit:
#   cd ~/sblee/dev_robosuite && mkdir -p logs
#   sbg --export=ALL scripts/sbatch_google_robot_exp0_qwen7b.sh
#
# Smoke test:
#   CUES=flex_bicep sbg --export=ALL scripts/sbatch_google_robot_exp0_qwen7b.sh

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

export DOMAIN=google_robot
export MODEL_SIZE=7b
export BACKEND=transformers
export VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
export ONLY=0
export GENERATE=1
export RESUME=1
export EXP0_REQUIRE_REASONING=0
export MOTION_PREPARE_MP4=0

exec bash exp.sh
