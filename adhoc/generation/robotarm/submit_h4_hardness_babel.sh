#!/usr/bin/env bash
# Submit blank-hardness glossary packs bundled per model (3 jobs total).
# One job loads Qwen once, then runs k2/k3/k4/k5/component × orig+CB.
#
# Usage:
#   bash submit_h4_hardness_babel.sh [3b|7b|32b|all]
set -euo pipefail

ROOT=/data/user_data/hoyeonk/sb/motion2mind
CODE=$ROOT/adhoc/h4_mcq
EVAL=$ROOT/paper_eval/h4_mcq
ONLY_MODEL="${1:-all}"
CONDA_SH=/data/user_data/hoyeonk/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=m2m_caption32b
LOG_ROOT=/home/hoyeonk
MAX_TOKENS=32
MAX_MODEL_LEN=8192
GRES_32B="${GRES_32B:-gpu:2}"

submit_model() {
  local model="$1"
  local slug="$2"
  local gpus="$3"
  local time="$4"
  local batch="$5"
  local max_seqs="$6"
  local gpu_mem="$7"
  local gres="gpu:${gpus}"
  if [[ "$gpus" == "2" ]]; then
    gres="$GRES_32B"
  fi
  sbatch \
    --parsable \
    --job-name="h4h_all_${slug}" \
    --partition=general \
    --gres="$gres" \
    --cpus-per-task=8 \
    --mem=120G \
    --time="$time" \
    --output="${LOG_ROOT}/h4h_all_${slug}_%j.out" \
    --error="${LOG_ROOT}/h4h_all_${slug}_%j.err" \
    --wrap="set -euo pipefail
source $CONDA_SH
conda activate $CONDA_ENV
export XDG_CACHE_HOME=$ROOT/.cache
export HF_HOME=$ROOT/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$ROOT/.cache/huggingface/hub
export TRANSFORMERS_CACHE=$ROOT/.cache/huggingface/transformers
export VLLM_CACHE_ROOT=$ROOT/.cache/vllm
export VLLM_CONFIG_ROOT=$ROOT/.cache/vllm
export VLLM_NO_USAGE_STATS=1
export TRITON_CACHE_DIR=$ROOT/.cache/triton
export TRITON_HOME=$ROOT/.triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_CUSTOM_ALLREDUCE=0
export VLLM_USE_V1=0
cd $CODE
TAGS=(k2 k3 k4 k5 component)
DS=()
for tag in \"\${TAGS[@]}\"; do
  stem=h4_glossary_\${tag}
  DS+=(--dataset $EVAL/\${stem}.json $EVAL/results_hardness_\${tag}/$slug)
  DS+=(--dataset $EVAL/\${stem}_counterbalanced.json $EVAL/results_hardness_\${tag}_cb/$slug)
done
python run_h4_mcq_vllm.py \
  --model $model \
  --batch-size $batch \
  --max-num-seqs $max_seqs \
  --max-tokens $MAX_TOKENS \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-mem $gpu_mem \
  \"\${DS[@]}\""
}

echo "Bundled hardness: k2 k3 k4 k5 component (orig+CB) → 1 job / model"
if [[ "$ONLY_MODEL" == "all" || "$ONLY_MODEL" == "3b" ]]; then
  echo "3B $(submit_model Qwen/Qwen2.5-VL-3B-Instruct qwen_3b 1 24:00:00 4 4 0.90)"
fi
if [[ "$ONLY_MODEL" == "all" || "$ONLY_MODEL" == "7b" ]]; then
  echo "7B $(submit_model Qwen/Qwen2.5-VL-7B-Instruct qwen_7b 1 30:00:00 2 2 0.90)"
fi
if [[ "$ONLY_MODEL" == "all" || "$ONLY_MODEL" == "32b" ]]; then
  echo "32B $(submit_model Qwen/Qwen2.5-VL-32B-Instruct qwen_32b 2 48:00:00 1 1 0.92)"
fi
