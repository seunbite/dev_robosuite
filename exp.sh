#!/usr/bin/env bash
# Pilot experiment entry point (manipulator pilot-90 | google_robot pilot-40).
#
# Examples:
#   bash exp.sh all                                    # manipulator, Qwen 32B
#   DOMAIN=google_robot MODEL_SIZE=gemini bash exp.sh all
#   DOMAIN=google_robot ONLY=1,2,3,7 bash exp.sh all
#   SUMMARY=1 DOMAIN=google_robot bash exp.sh all
set -euo pipefail

source ~/.bashrc 2>/dev/null || true

# Repo + env (cluster vs local)
if [[ -d "${HOME}/sblee/dev_robosuite" ]]; then
  cd "${HOME}/sblee/dev_robosuite"
  source /data/user_data/hoyeonk/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
  conda activate m2m_caption32b 2>/dev/null || true
elif [[ -d "${HOME}/Downloads/workspace/dev_robosuite" ]]; then
  cd "${HOME}/Downloads/workspace/dev_robosuite"
  if command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook -s zsh 2>/dev/null || micromamba shell hook -s bash)"
    micromamba activate robosuite 2>/dev/null || true
  fi
else
  echo "dev_robosuite repo not found" >&2
  exit 1
fi

task=${1:-"all"}
backend=${3:-"transformers"}
domain=${4:-"robotarm"}

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

# Allow positional override: exp.sh all <model> <backend> <domain>
if [[ -n "${2:-}" ]]; then
  model="$2"
fi
if [[ -n "${5:-}" ]]; then
  domain="$5"
fi

export BACKEND="${backend}"
export VLM_MODEL="${model}"
export GENERATE="${GENERATE:-1}"
export RESUME="${RESUME:-1}"

# vLLM v1 + flashinfer often breaks when torch ABI != torch_c_dlpack_ext wheel.
if [[ "${backend}" == "vllm" || "${backend}" == "local" ]]; then
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
fi

extra=()
if [[ "${SUMMARY:-0}" == "1" ]]; then
  extra+=(--summary)
fi

target="${ONLY:-$task}"

cmd="python adhoc/generation/${domain}/exp.py ${target} --backend ${backend} --model ${model} ${extra[*]:-}"
echo "${cmd}"
eval "${cmd}"
