#!/usr/bin/env bash
# Submit text-only H4 MCQ evaluation to Babel's general partition.
set -euo pipefail

ROOT=/data/user_data/hoyeonk/sb/motion2mind
CODE=$ROOT/adhoc/h4_mcq
EXPERIMENT="${1:-original}"
ONLY_MODEL="${2:-all}"
HAS_SECOND=0
MAX_TOKENS=8
MAX_MODEL_LEN=2048
# Prefer 80GB cards for long-context 32B; fall back to any 2 GPUs.
GRES_32B="${GRES_32B:-gpu:2}"
if [[ "$EXPERIMENT" == "glossary_en" ]]; then
  INPUT=$ROOT/paper_eval/h4_mcq/h4_glossary_value_english.json
  SECOND_INPUT=$ROOT/paper_eval/h4_mcq/h4_glossary_value_english_counterbalanced.json
  RESULTS=$ROOT/paper_eval/h4_mcq/results_glossary_en
  SECOND_RESULTS=$ROOT/paper_eval/h4_mcq/results_glossary_en_counterbalanced
  HAS_SECOND=1
  MAX_TOKENS=32
  # Schema YAML prompts are ~5k tokens; 8192 leaves KV headroom on 2x40–80GB.
  MAX_MODEL_LEN=8192
  JOB_PREFIX=h4glen
elif [[ "$EXPERIMENT" == "final" ]]; then
  INPUT=$ROOT/paper_eval/h4_mcq/h4_prompt_iteration_3_holdout.json
  RESULTS=$ROOT/paper_eval/h4_mcq/results_final_holdout
  SECOND_RESULTS=$RESULTS
  MAX_TOKENS=32
  JOB_PREFIX=h4final
elif [[ "$EXPERIMENT" == "iter3" ]]; then
  INPUT=$ROOT/paper_eval/h4_mcq/h4_prompt_iteration_3_dev.json
  RESULTS=$ROOT/paper_eval/h4_mcq/results_iteration_3_dev
  SECOND_RESULTS=$RESULTS
  MAX_TOKENS=32
  JOB_PREFIX=h4i3
elif [[ "$EXPERIMENT" == "iter2" ]]; then
  INPUT=$ROOT/paper_eval/h4_mcq/h4_prompt_iteration_2_dev.json
  RESULTS=$ROOT/paper_eval/h4_mcq/results_iteration_2_dev
  SECOND_RESULTS=$RESULTS
  MAX_TOKENS=32
  JOB_PREFIX=h4i2
elif [[ "$EXPERIMENT" == "iter1" ]]; then
  INPUT=$ROOT/paper_eval/h4_mcq/h4_prompt_iteration_1_dev.json
  RESULTS=$ROOT/paper_eval/h4_mcq/results_iteration_1_dev
  SECOND_RESULTS=$RESULTS
  MAX_TOKENS=32
  JOB_PREFIX=h4i1
elif [[ "$EXPERIMENT" == "blank" ]]; then
  INPUT=$ROOT/paper_eval/h4_mcq/h4_gt_blank_config_mcq.json
  SECOND_INPUT=$ROOT/paper_eval/h4_mcq/h4_gt_blank_config_mcq_counterbalanced.json
  RESULTS=$ROOT/paper_eval/h4_mcq/results_blank
  SECOND_RESULTS=$ROOT/paper_eval/h4_mcq/results_blank_counterbalanced
  HAS_SECOND=1
  JOB_PREFIX=h4blank
elif [[ "$EXPERIMENT" == "counterbalanced" ]]; then
  INPUT=$ROOT/paper_eval/h4_mcq/h4_gt_binary_mcq_counterbalanced.json
  RESULTS=$ROOT/paper_eval/h4_mcq/results_counterbalanced
  SECOND_RESULTS=$RESULTS
  JOB_PREFIX=h4cb
else
  INPUT=$ROOT/paper_eval/h4_mcq/h4_gt_binary_mcq.json
  RESULTS=$ROOT/paper_eval/h4_mcq/results
  SECOND_RESULTS=$RESULTS
  JOB_PREFIX=h4
fi
CONDA_SH=/data/user_data/hoyeonk/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=m2m_caption32b
LOG_ROOT=/home/hoyeonk

submit_model() {
  local model="$1"
  local slug="$2"
  local gpus="$3"
  local time="$4"
  local batch="$5"
  local max_seqs="$6"
  local gpu_mem="$7"
  local gres="gpu:${gpus}"
  local second_flags=""
  local exclude_args=()
  if [[ "$gpus" == "2" && "$EXPERIMENT" == "glossary_en" ]]; then
    gres="$GRES_32B"
  fi
  if [[ "$HAS_SECOND" == "1" ]]; then
    second_flags="--second-input $SECOND_INPUT --second-output-dir $SECOND_RESULTS/$slug"
  fi
  if [[ -n "${EXCLUDE:-}" ]]; then
    exclude_args=(--exclude="$EXCLUDE")
  fi

  sbatch \
    --parsable \
    "${exclude_args[@]}" \
    --job-name="${JOB_PREFIX}_${slug}" \
    --partition=general \
    --gres="$gres" \
    --cpus-per-task=8 \
    --mem=120G \
    --time="$time" \
    --output="${LOG_ROOT}/${JOB_PREFIX}_${slug}_%j.out" \
    --error="${LOG_ROOT}/${JOB_PREFIX}_${slug}_%j.err" \
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
mkdir -p $RESULTS/$slug $SECOND_RESULTS/$slug
cd $CODE
python run_h4_mcq_vllm.py \
  --input $INPUT \
  --output-dir $RESULTS/$slug \
  $second_flags \
  --model $model \
  --batch-size $batch \
  --max-num-seqs $max_seqs \
  --max-tokens $MAX_TOKENS \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-mem $gpu_mem"
}

if [[ "$ONLY_MODEL" == "all" || "$ONLY_MODEL" == "3b" ]]; then
  if [[ "$EXPERIMENT" == "glossary_en" ]]; then
    echo "3B $(submit_model Qwen/Qwen2.5-VL-3B-Instruct qwen_3b 1 08:00:00 4 4 0.90)"
  else
    echo "3B $(submit_model Qwen/Qwen2.5-VL-3B-Instruct qwen_3b 1 04:00:00 32 16 0.85)"
  fi
fi
if [[ "$ONLY_MODEL" == "all" || "$ONLY_MODEL" == "7b" ]]; then
  if [[ "$EXPERIMENT" == "glossary_en" ]]; then
    echo "7B $(submit_model Qwen/Qwen2.5-VL-7B-Instruct qwen_7b 1 10:00:00 2 2 0.90)"
  else
    echo "7B $(submit_model Qwen/Qwen2.5-VL-7B-Instruct qwen_7b 1 06:00:00 16 8 0.85)"
  fi
fi
if [[ "$ONLY_MODEL" == "all" || "$ONLY_MODEL" == "32b" ]]; then
  if [[ "$EXPERIMENT" == "glossary_en" ]]; then
    echo "32B $(submit_model Qwen/Qwen2.5-VL-32B-Instruct qwen_32b 2 18:00:00 1 1 0.92)"
  else
    echo "32B $(submit_model Qwen/Qwen2.5-VL-32B-Instruct qwen_32b 2 12:00:00 4 4 0.82)"
  fi
fi
