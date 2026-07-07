#!/usr/bin/env bash
#SBATCH --job-name=gr-exp0-qw32b
#SBATCH --output=logs/gr_exp0_qwen32b_%j.log
#SBATCH --error=logs/gr_exp0_qwen32b_%j.log
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#
# Google Robot exp0 generation (prompt_exp0.txt) — Qwen2.5-VL-32B
#
# Submit (set partition via your alias):
#   cd ~/sblee/dev_robosuite && mkdir -p logs
#   sbg2 --export=ALL scripts/sbatch_google_robot_exp0_qwen32b.sh
#   # or: sbatch --partition=YOUR_PART scripts/sbatch_google_robot_exp0_qwen32b.sh
#
# Smoke test (single cue):
#   CUES=flex_bicep sbg2 --export=ALL scripts/sbatch_google_robot_exp0_qwen32b.sh

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

export DOMAIN=google_robot
export MODEL_SIZE=32b
export BACKEND=transformers
export VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
export ONLY=0
export GENERATE=1
export RESUME=1
export EXP0_REQUIRE_REASONING=0
export MOTION_PREPARE_MP4=0

exec bash exp.sh
