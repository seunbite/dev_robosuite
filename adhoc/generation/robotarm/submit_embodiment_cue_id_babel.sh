#!/usr/bin/env bash
# Submit embodiment cue-ID (yes/no + pairwise AB/BA) for GT / grok45 / opus48 / gemini on Babel.
# Usage:
#   bash adhoc/generation/robotarm/submit_embodiment_cue_id_babel.sh \
#     [all|gt|grok45|opus48|gemini] [all|3b|7b|32b] [all|yesno|pairwise]
set -euo pipefail

ROOT="${ROOT:-/data/user_data/hoyeonk/sb/motion2mind}"
CODE="$ROOT/adhoc/generation/robotarm"
EVAL="$ROOT/data/results/evaluation/embodiment_cue_id"
# RESULTS_SUBDIR lets easy-neg reruns keep prior random-foil numbers.
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results}"
RESULTS="$ROOT/data/results/evaluation/embodiment_cue_id/${RESULTS_SUBDIR}"
CONDA_SH=/data/user_data/hoyeonk/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=m2m_caption32b
LOG_ROOT=/home/hoyeonk
WHICH="${1:-all}"
ONLY_MODEL="${2:-all}"
ONLY_TASK="${3:-all}"
# 4×L40S TP=4 works for VL-32B; QOS normal caps gres/gpu=8 → 2 concurrent 32B jobs.
GRES_32B="${GRES_32B:-gpu:L40S:4}"

submit_one() {
  local source="$1"
  local task="$2"
  local model="$3"
  local slug="$4"
  local gpus="$5"
  local time="$6"
  local batch="$7"
  local max_seqs="$8"
  local gpu_mem="$9"
  local max_len="${10}"
  local input="$EVAL/${source}_${task}.json"
  local out="$RESULTS/${source}_${task}/$slug"
  local job="cueid_${source}_${task}_${slug}"
  local gres="gpu:${gpus}"
  if [[ "$slug" == "qwen_32b" ]]; then
    gres="$GRES_32B"
    # parse gpu count from GRES like gpu:L40S:4
    gpus="${GRES_32B##*:}"
  fi

  sbatch \
    --parsable \
    --job-name="$job" \
    --partition=general \
    --gres="$gres" \
    --cpus-per-task=8 \
    --mem=120G \
    --time="$time" \
    --output="${LOG_ROOT}/${job}_%j.out" \
    --error="${LOG_ROOT}/${job}_%j.err" \
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
export VLLM_ENFORCE_EAGER=1
export VLLM_TENSOR_PARALLEL_SIZE=${gpus}
export VLLM_MAX_MODEL_LEN=${max_len}
export VLLM_MAX_NUM_SEQS=${max_seqs}
export VLLM_GPU_MEMORY_UTILIZATION=${gpu_mem}
export VLM_MODEL=${model}
export VLM_TEMPERATURE=0
export VLM_MAX_NEW_TOKENS=8
export VLM_VIDEO_NFRAMES=4
export PYTHONPATH=$CODE:$ROOT\${PYTHONPATH:+:\$PYTHONPATH}
mkdir -p $out
cd $CODE
python run_embodiment_cue_id_vllm.py \
  --input $input \
  --output-dir $out \
  --model $model \
  --batch-size $batch \
  --tensor-parallel-size $gpus \
  --gpu-mem $gpu_mem \
  --max-model-len $max_len \
  --max-num-seqs $max_seqs"
}

sources=()
if [[ "$WHICH" == "all" || "$WHICH" == "gt" ]]; then sources+=(gt); fi
if [[ "$WHICH" == "all" || "$WHICH" == "grok45" || "$WHICH" == "agent_grok45" ]]; then
  sources+=(agent_grok45)
fi
if [[ "$WHICH" == "all" || "$WHICH" == "opus48" || "$WHICH" == "agent_opus48" || "$WHICH" == "opus" ]]; then
  sources+=(agent_opus48)
fi
if [[ "$WHICH" == "all" || "$WHICH" == "gemini" || "$WHICH" == "agent_gemini25pro" || "$WHICH" == "gemini25pro" ]]; then
  sources+=(agent_gemini25pro)
fi

# model|slug|gpus|time|batch|max_seqs|gpu_mem|max_model_len
models=()
if [[ "$ONLY_MODEL" == "all" || "$ONLY_MODEL" == "3b" ]]; then
  models+=("Qwen/Qwen2.5-VL-3B-Instruct|qwen_3b|1|08:00:00|1|1|0.90|8192")
fi
if [[ "$ONLY_MODEL" == "all" || "$ONLY_MODEL" == "7b" ]]; then
  models+=("Qwen/Qwen2.5-VL-7B-Instruct|qwen_7b|1|12:00:00|1|1|0.90|8192")
fi
if [[ "$ONLY_MODEL" == "all" || "$ONLY_MODEL" == "32b" ]]; then
  models+=("Qwen/Qwen2.5-VL-32B-Instruct|qwen_32b|4|18:00:00|1|1|0.90|4096")
fi

tasks=()
if [[ "$ONLY_TASK" == "all" || "$ONLY_TASK" == "yesno" ]]; then tasks+=(yesno); fi
if [[ "$ONLY_TASK" == "all" || "$ONLY_TASK" == "pairwise" ]]; then tasks+=(pairwise); fi

if [[ ${#sources[@]} -eq 0 || ${#models[@]} -eq 0 || ${#tasks[@]} -eq 0 ]]; then
  echo "No sources/models/tasks selected (which=$WHICH model=$ONLY_MODEL task=$ONLY_TASK)" >&2
  exit 1
fi

echo "RESULTS=$RESULTS GRES_32B=$GRES_32B"
for source in "${sources[@]}"; do
  for task in "${tasks[@]}"; do
    for spec in "${models[@]}"; do
      IFS='|' read -r model slug gpus time batch max_seqs gpu_mem max_len <<<"$spec"
      echo "$source $task $slug $(submit_one "$source" "$task" "$model" "$slug" "$gpus" "$time" "$batch" "$max_seqs" "$gpu_mem" "$max_len")"
    done
  done
done
