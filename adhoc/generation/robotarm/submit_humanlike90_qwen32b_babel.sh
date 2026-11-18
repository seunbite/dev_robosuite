#!/usr/bin/env bash
# Generate Short Motion Codes for the humanlike-90 cue set with Qwen2.5-VL-32B.
#
#   bash adhoc/generation/robotarm/submit_humanlike90_qwen32b_babel.sh
#   PLATFORMS=humanoid MAX_CUES=2 bash adhoc/generation/robotarm/submit_humanlike90_qwen32b_babel.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

REMOTE="${REMOTE:-hoyeonk@login2.babel.cs.cmu.edu}"
REMOTE_DIR="${REMOTE_DIR:-/home/hoyeonk/sblee/dev_robosuite}"
ACCOUNT="${ACCOUNT:-dmortens}"
QOS="${QOS:-preempt_qos}"
PARTITION="${PARTITION:-preempt}"
PLATFORMS="${PLATFORMS:-humanoid,tiago}"
OUT_TAG="${OUT_TAG:-qwen32b_humanlike90}"
CUES_REL="data/seed/agent_fill/embodiment_short/cues_humanlike90.json"

echo "=== rsync code + 90-cue list -> ${REMOTE}:${REMOTE_DIR} ==="
ssh "$REMOTE" "mkdir -p \
  ${REMOTE_DIR}/adhoc/generation/robotarm \
  ${REMOTE_DIR}/scripts \
  ${REMOTE_DIR}/data/seed/yml \
  ${REMOTE_DIR}/data/seed/agent_fill/embodiment_short \
  ${REMOTE_DIR}/logs \
  ${REMOTE_DIR}/data/results/motion_configs"

rsync -avz --progress \
  "$ROOT/adhoc/generation/robotarm/run_embodiment_exp0_qwen.py" \
  "$ROOT/adhoc/generation/robotarm/run_embodiment_exp0_gemini.py" \
  "$ROOT/adhoc/generation/robotarm/vllm_local.py" \
  "$ROOT/adhoc/generation/robotarm/motion_code_to_config.py" \
  "${REMOTE}:${REMOTE_DIR}/adhoc/generation/robotarm/"

rsync -avz --progress \
  "$ROOT/data/seed/yml/motion_code_embodiment.yaml" \
  "${REMOTE}:${REMOTE_DIR}/data/seed/yml/"

rsync -avz --progress \
  "$ROOT/data/seed/agent_fill/embodiment_short/cues_humanlike90.json" \
  "$ROOT/data/seed/agent_fill/embodiment_short/fewshot_synthetic.md" \
  "${REMOTE}:${REMOTE_DIR}/data/seed/agent_fill/embodiment_short/"

echo ""
echo "=== submit sbatch platforms=${PLATFORMS} out=${OUT_TAG} ==="
JOBID=$(ssh "$REMOTE" "bash -lc '
  set -euo pipefail
  cd ${REMOTE_DIR}
  mkdir -p logs data/results/motion_configs
  sbatch --parsable \
    --account=${ACCOUNT} \
    --qos=${QOS} \
    --partition=${PARTITION} \
    --chdir=${REMOTE_DIR} \
    --job-name=smp-qw32b-90 \
    --gres=gpu:L40S:4 \
    --cpus-per-task=8 \
    --mem=120G \
    --time=24:00:00 \
    --output=${REMOTE_DIR}/logs/smp_qwen32b_humanlike90_%j.log \
    --error=${REMOTE_DIR}/logs/smp_qwen32b_humanlike90_%j.log \
    --wrap=\"set -euo pipefail
source /data/user_data/hoyeonk/miniconda3/etc/profile.d/conda.sh
conda activate m2m_caption32b
cd ${REMOTE_DIR}
export HF_HOME=/data/user_data/hoyeonk/hf_cache
export HUGGINGFACE_HUB_CACHE=\\\$HF_HOME/hub
export TRANSFORMERS_CACHE=\\\$HF_HOME/transformers
export XDG_CACHE_HOME=/data/user_data/hoyeonk/.cache
export VLLM_USE_CUSTOM_ALLREDUCE=0
export VLLM_USE_V1=0
export VLLM_ENFORCE_EAGER=1
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_MAX_MODEL_LEN=32768
export VLLM_MAX_NUM_SEQS=1
export VLLM_GPU_MEMORY_UTILIZATION=0.90
export VLM_TEMPERATURE=0.3
export VLM_MAX_NEW_TOKENS=4096
export MAX_CUES=${MAX_CUES:-0}
export PYTHONPATH=\\\$PWD/adhoc/generation/robotarm:\\\$PWD\\\${PYTHONPATH:+:\\\$PYTHONPATH}
EXTRA=()
if [[ \\\${MAX_CUES} != 0 ]]; then EXTRA+=(--max-cues \\\${MAX_CUES}); fi
python adhoc/generation/robotarm/run_embodiment_exp0_qwen.py \
  --cues \\\$PWD/${CUES_REL} \
  --platforms ${PLATFORMS} \
  --model Qwen/Qwen2.5-VL-32B-Instruct \
  --out-json \\\$PWD/data/results/motion_configs/embodiment_motion_code_qwen_${OUT_TAG} \
  --batch-size 1 --retries 2 --resume \\\${EXTRA[@]}
\"
'")

echo "submitted job ${JOBID}"
echo ""
echo "Monitor:"
echo "  ssh ${REMOTE} squeue -j ${JOBID}"
echo "  ssh ${REMOTE} tail -f ${REMOTE_DIR}/logs/smp_qwen32b_humanlike90_${JOBID}.log"
echo ""
echo "After finish, pull + score:"
echo "  rsync -avz ${REMOTE}:${REMOTE_DIR}/data/results/motion_configs/embodiment_motion_code_qwen_${OUT_TAG}/ \\"
echo "    data/results/motion_configs/embodiment_motion_code_qwen_${OUT_TAG}/"
echo "  python papers/icra2027_motion_abstraction/score_humanlike_cuespec.py \\"
echo "    --pred data/results/motion_configs/embodiment_motion_code_qwen_${OUT_TAG}/embodiment_motion_code_all.json \\"
echo "    --tag qwen32b_humanlike90"
