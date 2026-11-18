#!/usr/bin/env bash
# Submit Paddlebot90 Motion Code generation (Qwen 32B or 7B) on Babel.
#
#   bash papers/icra2027_motion_abstraction/motion_code/submit_paddlebot90_qwen_babel.sh 32b
#   bash papers/icra2027_motion_abstraction/motion_code/submit_paddlebot90_qwen_babel.sh 7b
#   MAX_CUES=2 bash .../submit_paddlebot90_qwen_babel.sh 32b
set -euo pipefail

SIZE="${1:-32b}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

REMOTE="${REMOTE:-babel2}"
REMOTE_DIR="${REMOTE_DIR:-/home/hoyeonk/sblee/dev_robosuite}"
ACCOUNT="${ACCOUNT:-dmortens}"
QOS="${QOS:-preempt_qos}"
PARTITION="${PARTITION:-preempt}"

if [[ "$SIZE" == "32b" ]]; then
  MODEL="Qwen/Qwen2.5-VL-32B-Instruct"
  OUT_TAG="qwen32b_paddlebot90"
  GPUS=4
  MEM=120G
  JOBNAME="smp-pb-qw32b"
elif [[ "$SIZE" == "7b" ]]; then
  MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
  OUT_TAG="qwen7b_paddlebot90"
  GPUS=1
  MEM=48G
  JOBNAME="smp-pb-qw7b"
else
  echo "usage: $0 {32b|7b}" >&2
  exit 2
fi

CUES_REL="data/seed/agent_fill/embodiment_short/cues_humanlike90.json"
GEN_REL="papers/icra2027_motion_abstraction/motion_code/generate_paddlebot90.py"
FS_REL="papers/icra2027_motion_abstraction/motion_code/data/paddlebot_fewshot.json"
SCHEMA_REL="papers/icra2027_motion_abstraction/synthetic_morphology/yml/motion_code_synthetic_continuum.yaml"

echo "=== rsync paddlebot90 generator + fewshots → ${REMOTE}:${REMOTE_DIR} ==="
ssh "$REMOTE" "mkdir -p \
  ${REMOTE_DIR}/adhoc/generation/robotarm \
  ${REMOTE_DIR}/papers/icra2027_motion_abstraction/motion_code/data \
  ${REMOTE_DIR}/papers/icra2027_motion_abstraction/synthetic_morphology/yml \
  ${REMOTE_DIR}/data/seed/agent_fill/embodiment_short \
  ${REMOTE_DIR}/logs \
  ${REMOTE_DIR}/data/results/motion_configs"

rsync -avz --progress \
  "$ROOT/adhoc/generation/robotarm/run_embodiment_exp0_gemini.py" \
  "$ROOT/adhoc/generation/robotarm/run_embodiment_exp0_qwen.py" \
  "$ROOT/adhoc/generation/robotarm/vllm_local.py" \
  "${REMOTE}:${REMOTE_DIR}/adhoc/generation/robotarm/"

rsync -avz --progress \
  "$ROOT/${GEN_REL}" \
  "${REMOTE}:${REMOTE_DIR}/papers/icra2027_motion_abstraction/motion_code/"

rsync -avz --progress \
  "$ROOT/${FS_REL}" \
  "${REMOTE}:${REMOTE_DIR}/papers/icra2027_motion_abstraction/motion_code/data/"

rsync -avz --progress \
  "$ROOT/${SCHEMA_REL}" \
  "${REMOTE}:${REMOTE_DIR}/papers/icra2027_motion_abstraction/synthetic_morphology/yml/"

rsync -avz --progress \
  "$ROOT/${CUES_REL}" \
  "${REMOTE}:${REMOTE_DIR}/data/seed/agent_fill/embodiment_short/"

echo ""
echo "=== submit sbatch model=${MODEL} out=${OUT_TAG} ==="
JOBID=$(ssh "$REMOTE" "bash -lc '
  set -euo pipefail
  cd ${REMOTE_DIR}
  mkdir -p logs data/results/motion_configs
  sbatch --parsable \
    --account=${ACCOUNT} \
    --qos=${QOS} \
    --partition=${PARTITION} \
    --chdir=${REMOTE_DIR} \
    --job-name=${JOBNAME} \
    --gres=gpu:L40S:${GPUS} \
    --cpus-per-task=8 \
    --mem=${MEM} \
    --time=24:00:00 \
    --output=${REMOTE_DIR}/logs/smp_${OUT_TAG}_%j.log \
    --error=${REMOTE_DIR}/logs/smp_${OUT_TAG}_%j.log \
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
export VLLM_TENSOR_PARALLEL_SIZE=${GPUS}
export VLLM_MAX_MODEL_LEN=32768
export VLLM_MAX_NUM_SEQS=1
export VLLM_GPU_MEMORY_UTILIZATION=0.90
export VLM_TEMPERATURE=0.3
export VLM_MAX_NEW_TOKENS=4096
export MAX_CUES=${MAX_CUES:-0}
export PYTHONPATH=\\\$PWD/adhoc/generation/robotarm:\\\$PWD/papers/icra2027_motion_abstraction/motion_code:\\\$PWD\\\${PYTHONPATH:+:\\\$PYTHONPATH}
EXTRA=()
if [[ \\\${MAX_CUES} != 0 ]]; then EXTRA+=(--max-cues \\\${MAX_CUES}); fi
python ${GEN_REL} \
  --backend qwen \
  --model ${MODEL} \
  --cues \\\$PWD/${CUES_REL} \
  --out-json \\\$PWD/data/results/motion_configs/embodiment_motion_code_qwen_${OUT_TAG} \
  --retries 2 --resume \\\${EXTRA[@]}
\"
'")

echo "submitted job ${JOBID}"
echo "Monitor: ssh ${REMOTE} tail -f ${REMOTE_DIR}/logs/smp_${OUT_TAG}_${JOBID}.log"
echo "Pull later:"
echo "  rsync -avz ${REMOTE}:${REMOTE_DIR}/data/results/motion_configs/embodiment_motion_code_qwen_${OUT_TAG}/ \\"
echo "    data/results/motion_configs/embodiment_motion_code_qwen_${OUT_TAG}/"
