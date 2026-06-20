#!/usr/bin/env bash
# Pilot experiment entry point (manipulator pilot-90 | google_robot pilot-40).
#
# IMPORTANT: use bash, not sh:
#   bash exp.sh all Qwen/Qwen2.5-VL-3B-Instruct vllm google_robot
#
# Examples:
#   bash exp.sh all                                    # manipulator, Qwen 32B
#   DOMAIN=google_robot MODEL_SIZE=gemini bash exp.sh all
#   DOMAIN=google_robot ONLY=1,2,3,7 bash exp.sh all
#   SUMMARY=1 bash exp.sh all                    # Qwen 32B/7B/3B cross-model tables
#   SUMMARY=1 DOMAIN=google_robot bash exp.sh all
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# sbatch copies the job script to /var/spool/slurmd/job* — adhoc/ is not there.
if [[ ! -f "${ROOT}/adhoc/generation/robotarm/exp.py" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/exp.sh" ]]; then
    ROOT="${SLURM_SUBMIT_DIR}"
  elif [[ -n "${DEV_ROBOSUITE_ROOT:-}" && -f "${DEV_ROBOSUITE_ROOT}/exp.sh" ]]; then
    ROOT="${DEV_ROBOSUITE_ROOT}"
  elif [[ -f "${HOME}/sblee/dev_robosuite/exp.sh" ]]; then
    ROOT="${HOME}/sblee/dev_robosuite"
  fi
fi
cd "${ROOT}"

# Optional shell init (do not fail the run if missing)
set +e
source ~/.bashrc 2>/dev/null
set -e

# Cluster conda env (Babel)
if [[ -f /data/user_data/hoyeonk/miniconda3/etc/profile.d/conda.sh ]]; then
  set +e
  source /data/user_data/hoyeonk/miniconda3/etc/profile.d/conda.sh
  conda activate m2m_caption32b 2>/dev/null
  set -e
elif command -v micromamba >/dev/null 2>&1; then
  set +e
  eval "$(micromamba shell hook -s bash 2>/dev/null)"
  micromamba activate robosuite 2>/dev/null
  set -e
fi

# Babel: keep HuggingFace weights on /data (not ~/.cache — home quota is tiny).
if [[ -f "${ROOT}/scripts/cluster_env.sh" ]]; then
  HF_ROOT="${HF_HOME:-/data/user_data/${USER}/hf_cache}"
  # shellcheck source=/dev/null
  source "${ROOT}/scripts/cluster_env.sh" "${HF_ROOT}"
fi

task=${1:-"all"}
backend=${3:-"transformers"}
domain="${DOMAIN:-${4:-robotarm}}"

# Model selection
case "${MODEL_SIZE:-32b}" in
  gemini)
    model="${MODEL:-gemini-2.5-pro}"
    backend="gemini"
    [[ -f APIKEY.sh ]] && source APIKEY.sh
    ;;
  7b)  model="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}" ;;
  3b)  model="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}" ;;
  *)   model="${MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}" ;;
esac

# Positional override: exp.sh <task> <model> <backend> [domain]
if [[ -n "${2:-}" ]]; then
  model="$2"
fi
if [[ -n "${4:-}" ]]; then
  domain="$4"
fi

export BACKEND="${backend}"
export VLM_MODEL="${model}"
export GENERATE="${GENERATE:-1}"
export RESUME="${RESUME:-1}"

# vLLM v1 + flashinfer often breaks when torch ABI != torch_c_dlpack_ext wheel.
if [[ "${backend}" == "vllm" || "${backend}" == "local" ]]; then
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
  export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-10}"
  export VLM_BATCH_SIZE="${VLM_BATCH_SIZE:-10}"
fi

extra=()
if [[ "${SUMMARY:-0}" == "1" ]]; then
  extra+=(--summary)
fi

target="${ONLY:-$task}"
exp_py="adhoc/generation/${domain}/exp.py"

if [[ ! -f "${exp_py}" ]]; then
  echo "exp.py not found: ${ROOT}/${exp_py}" >&2
  echo "domain=${domain} — use robotarm or google_robot" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python not found in PATH (activate conda/micromamba first)" >&2
  exit 1
fi

cmd=(python "${exp_py}" "${target}" --backend "${backend}" --model "${model}")
if ((${#extra[@]})); then
  cmd+=("${extra[@]}")
fi

echo "[exp.sh] cwd=${ROOT}"
echo "[exp.sh] domain=${domain} backend=${backend} model=${model}"
if [[ -n "${HF_HOME:-}" ]]; then
  echo "[exp.sh] HF_HOME=${HF_HOME}"
  echo "[exp.sh] HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-}"
fi
if [[ "${backend}" == "vllm" || "${backend}" == "local" ]]; then
  echo "[exp.sh] VLM_BATCH_SIZE=${VLM_BATCH_SIZE:-10} (parallel vLLM requests, not cue limit)"
fi
echo "[exp.sh] ${cmd[*]}"
exec "${cmd[@]}"
