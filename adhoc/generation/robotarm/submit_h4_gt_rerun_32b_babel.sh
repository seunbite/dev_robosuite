#!/usr/bin/env bash
# Submit 32B H4 blank rerun using packs staged under $HOME (login has no /data).
# At job start, copy packs into $ROOT/paper_eval/h4_mcq (compute nodes mount /data).
set -euo pipefail

STAGE="${STAGE:-/home/hoyeonk/staging/h4_gt_rerun_20260805}"
ROOT=/data/user_data/hoyeonk/sb/motion2mind
CODE=$ROOT/adhoc/h4_mcq
EVAL=$ROOT/paper_eval/h4_mcq
CONDA_SH=/data/user_data/hoyeonk/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=m2m_caption32b
LOG_ROOT=/home/hoyeonk
GRES_32B="${GRES_32B:-gpu:2}"

submit_one() {
  local job_name="$1"
  local time="$2"
  local wrap="$3"
  sbatch \
    --parsable \
    --job-name="$job_name" \
    --partition=general \
    --gres="$GRES_32B" \
    --cpus-per-task=8 \
    --mem=120G \
    --time="$time" \
    --output="${LOG_ROOT}/${job_name}_%j.out" \
    --error="${LOG_ROOT}/${job_name}_%j.err" \
    --wrap="$wrap"
}

COPY_PACKS="mkdir -p $EVAL && cp -f $STAGE/packs/h4_glossary_*.json $STAGE/packs/gt_component_core_submotion.json $EVAL/ && \
source $CONDA_SH && conda activate $CONDA_ENV && \
export XDG_CACHE_HOME=$ROOT/.cache HF_HOME=$ROOT/.cache/huggingface \
  HUGGINGFACE_HUB_CACHE=$ROOT/.cache/huggingface/hub \
  TRANSFORMERS_CACHE=$ROOT/.cache/huggingface/transformers \
  VLLM_CACHE_ROOT=$ROOT/.cache/vllm VLLM_CONFIG_ROOT=$ROOT/.cache/vllm \
  VLLM_NO_USAGE_STATS=1 TRITON_CACHE_DIR=$ROOT/.cache/triton TRITON_HOME=$ROOT/.triton \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_CUSTOM_ALLREDUCE=0 VLLM_USE_V1=0 && \
cd $CODE"

# k1 glossary_en orig+CB
WRAP_K1="$COPY_PACKS && python run_h4_mcq_vllm.py \
  --model Qwen/Qwen2.5-VL-32B-Instruct \
  --batch-size 1 --max-num-seqs 1 --max-tokens 32 --max-model-len 8192 --gpu-mem 0.92 \
  --dataset $EVAL/h4_glossary_value_english.json $EVAL/results_glossary_en_gtrerun/qwen_32b \
  --dataset $EVAL/h4_glossary_value_english_counterbalanced.json $EVAL/results_glossary_en_gtrerun_cb/qwen_32b"

# hardness k2..component
WRAP_H="$COPY_PACKS && TAGS=(k2 k3 k4 k5 component); DS=(); \
for tag in \"\${TAGS[@]}\"; do \
  DS+=(--dataset $EVAL/h4_glossary_\${tag}.json $EVAL/results_hardness_\${tag}_gtrerun/qwen_32b); \
  DS+=(--dataset $EVAL/h4_glossary_\${tag}_counterbalanced.json $EVAL/results_hardness_\${tag}_gtrerun_cb/qwen_32b); \
done; \
python run_h4_mcq_vllm.py \
  --model Qwen/Qwen2.5-VL-32B-Instruct \
  --batch-size 1 --max-num-seqs 1 --max-tokens 32 --max-model-len 8192 --gpu-mem 0.92 \
  \"\${DS[@]}\""

echo "k1 $(submit_one h4glen_gtrerun_32b 18:00:00 "$WRAP_K1")"
echo "hardness $(submit_one h4h_gtrerun_32b 48:00:00 "$WRAP_H")"
squeue -u hoyeonk -o '%.18i %.28j %.8T %.10M' | head -15
