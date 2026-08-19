#!/usr/bin/env bash
# Submit Qwen2.5-VL-32B:
#   - pairwise: gt + agent_gemini25pro (full 300 each)  → compare to Gemini-judge
#   - yes/no no-quad: gt + agent_gemini25pro (240 each)
# Usage (on Babel login, or via ssh):
#   RESULTS_SUBDIR=results_qwen32b_20260802 bash adhoc/generation/robotarm/submit_cueid_32b_pairwise_and_yesno_noquad.sh
set -euo pipefail

ROOT="${ROOT:-/data/user_data/hoyeonk/sb/motion2mind}"
CODE="$ROOT/adhoc/generation/robotarm"
EVAL="$ROOT/data/results/evaluation/embodiment_cue_id"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results_qwen32b_20260802}"
RESULTS="$EVAL/${RESULTS_SUBDIR}"
CONDA_SH=/data/user_data/hoyeonk/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=m2m_caption32b
LOG_ROOT=/home/hoyeonk
GRES_32B="${GRES_32B:-gpu:L40S:4}"
MODEL="Qwen/Qwen2.5-VL-32B-Instruct"

submit_one() {
  local input_name="$1"   # e.g. gt_pairwise.json or gt_yesno_noquad.json
  local out_slug="$2"     # e.g. gt_pairwise
  local time="$3"
  local input="$EVAL/${input_name}"
  local out="$RESULTS/${out_slug}/qwen_32b"
  local job="cueid32_${out_slug}"
  local gpus="${GRES_32B##*:}"

  sbatch --parsable \
    --job-name="$job" \
    --partition=general \
    --gres="$GRES_32B" \
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
export VLLM_MAX_MODEL_LEN=4096
export VLLM_MAX_NUM_SEQS=1
export VLLM_GPU_MEMORY_UTILIZATION=0.90
export VLM_MODEL=${MODEL}
export VLM_TEMPERATURE=0
export VLM_MAX_NEW_TOKENS=8
export VLM_VIDEO_NFRAMES=4
export PYTHONPATH=$CODE:$ROOT\${PYTHONPATH:+:\$PYTHONPATH}
mkdir -p $out
cd $CODE
# wipe prior partials for a clean 32B pass on this RESULTS_SUBDIR
rm -f $out/predictions.jsonl $out/summary.json
python run_embodiment_cue_id_vllm.py \
  --input $input \
  --output-dir $out \
  --model $MODEL \
  --batch-size 1 \
  --tensor-parallel-size ${gpus} \
  --gpu-mem 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 1"
}

echo "RESULTS=$RESULTS"
mkdir -p "$RESULTS"
echo "pairwise gt: $(submit_one gt_pairwise.json gt_pairwise 18:00:00)"
echo "pairwise gemini: $(submit_one agent_gemini25pro_pairwise.json agent_gemini25pro_pairwise 18:00:00)"
echo "yesno_noquad gt: $(submit_one gt_yesno_noquad.json gt_yesno_noquad 12:00:00)"
echo "yesno_noquad gemini: $(submit_one agent_gemini25pro_yesno_noquad.json agent_gemini25pro_yesno_noquad 12:00:00)"
squeue -u "$USER" -o "%.10i %.40j %.2t %.10M %R" | head -20
