#!/usr/bin/env bash
# Submit the append-only 300-job trajectory baseline generator on Babel.
set -euo pipefail

ROOT="${ROOT:-/data/user_data/hoyeonk/sb/motion2mind}"
CODE="$ROOT/adhoc/generation/robotarm"
OUT="$ROOT/data/results/baseline_traj/qwen25vl32b_fullbody_5x30x2_20260808"
CONDA_SH=/data/user_data/hoyeonk/miniconda3/etc/profile.d/conda.sh
JOB=baseline_qwen32b_300

sbatch --parsable \
  --job-name="$JOB" --partition=general --gres=gpu:L40S:4 \
  --cpus-per-task=8 --mem=120G --time=18:00:00 \
  --output="/home/hoyeonk/${JOB}_%j.out" \
  --error="/home/hoyeonk/${JOB}_%j.err" \
  --wrap="set -euo pipefail
source $CONDA_SH
conda activate m2m_caption32b
export HF_HOME=$ROOT/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=\$HF_HOME/hub
export XDG_CACHE_HOME=$ROOT/.cache
export VLLM_USE_CUSTOM_ALLREDUCE=0
export VLLM_USE_V1=0
export VLLM_ENFORCE_EAGER=1
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_MAX_MODEL_LEN=32768
export VLLM_MAX_NUM_SEQS=1
export VLLM_GPU_MEMORY_UTILIZATION=0.90
export VLM_TEMPERATURE=0.3
export VLM_MAX_NEW_TOKENS=8192
export PYTHONPATH=$CODE:$ROOT\${PYTHONPATH:+:\$PYTHONPATH}
mkdir -p $OUT
cd $CODE
python run_embodiment_baseline_qwen.py \
  --out-dir $OUT \
  --fewshot-shots $ROOT/data/seed/shots/embodiment_unified/shots_fullbody_parts.yaml \
  --model Qwen/Qwen2.5-VL-32B-Instruct --batch-size 1 --retries 2"
