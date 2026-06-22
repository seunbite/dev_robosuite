#!/usr/bin/env bash
#SBATCH --job-name=exp
#SBATCH --output=logs/exp_%j.log
#SBATCH --error=logs/exp_%j.log
#SBATCH --time=24:00:00
#
# Logs: logs/exp_<JOBID>.log  (stdout+stderr combined — tail this file)
# Single entry point: interactive, salloc, and sbatch (partition/GPU via your alias).
#
#   bash exp.sh all
#   ONLY=8,9 MODEL_SIZE=7b bash exp.sh
#   SUMMARY=1 bash exp.sh
#   ALL_MODELS=1 SUMMARY=1 bash exp.sh
#   DOMAIN=google_robot MODEL_SIZE=gemini bash exp.sh all
#
# Cluster sbatch (partition/GPU via your sbg/sbg2 alias):
#   sbg  --export=ALL,MODEL_SIZE=7b,ONLY=8,9,MOTION_PREPARE_MP4=0 exp.sh
#   sbg2 --export=ALL,MODEL_SIZE=32b,ONLY=4,5,6,10,MOTION_PREPARE_PAIRWISE=0 exp.sh
#   sbgd --export=ALL,MODEL_SIZE=3b,ONLY=4,5,6,10 exp.sh
#
# Positional override: exp.sh <task> <model> <backend> [domain]
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
mkdir -p logs

_on_err() {
  local rc=$?
  echo "[exp.sh] FATAL: exit $rc at line ${BASH_LINENO[0]}: ${BASH_COMMAND}" >&2
  exit "$rc"
}
trap _on_err ERR

_log_start() {
  echo "================================================================"
  echo "[exp.sh] start  time=$(date -Iseconds 2>/dev/null || date)"
  echo "[exp.sh] job=${SLURM_JOB_ID:-interactive} partition=${SLURM_JOB_PARTITION:-?}"
  echo "[exp.sh] submit=${SLURM_SUBMIT_DIR:-$ROOT} cwd=${ROOT}"
  echo "[exp.sh] ONLY=${ONLY:-all} MODEL_SIZE=${MODEL_SIZE:-32b} DOMAIN=${DOMAIN:-robotarm}"
  echo "================================================================"
}
_is_cluster() {
  [[ -n "${SLURM_JOB_ID:-}" ]] \
    || [[ -d /data/user_data ]] \
    || [[ -x "${ROOT}/y/envs/robosuite-vlm/bin/python" ]]
}

_setup_shell() {
  set +e
  source ~/.bashrc 2>/dev/null
  set -e
}

_setup_env() {
  if [[ "${SKIP_ENV:-0}" == "1" ]]; then
    return 0
  fi

  if _is_cluster; then
    export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}:$ROOT/adhoc/generation/${DOMAIN:-robotarm}"
    if [[ -f "${ROOT}/scripts/cluster_env.sh" ]]; then
      # shellcheck source=/dev/null
      source "${ROOT}/scripts/cluster_env.sh" "${HF_HOME:-/data/user_data/${USER}/hf_cache}"
    fi
    if [[ -f "${ROOT}/scripts/activate_cluster_vlm.sh" ]]; then
      echo "[exp.sh] activating VLM env..."
      # shellcheck source=/dev/null
      if ! source "${ROOT}/scripts/activate_cluster_vlm.sh" "${ROOT}"; then
        echo "[exp.sh] FATAL: VLM env setup failed" >&2
        exit 1
      fi
    fi
    return 0
  fi

  if [[ -f /data/user_data/hoyeonk/miniconda3/etc/profile.d/conda.sh ]]; then
    set +e
    # shellcheck source=/dev/null
    source /data/user_data/hoyeonk/miniconda3/etc/profile.d/conda.sh
    conda activate m2m_caption32b 2>/dev/null
    set -e
  elif command -v micromamba >/dev/null 2>&1; then
    set +e
    eval "$(micromamba shell hook -s bash 2>/dev/null)"
    micromamba activate robosuite 2>/dev/null
    set -e
  fi

  if [[ -f "${ROOT}/scripts/cluster_env.sh" ]]; then
    local hf_root="${HF_HOME:-${HOME}/.cache}"
    if [[ -d /data/user_data ]]; then
      hf_root="${HF_HOME:-/data/user_data/${USER}/hf_cache}"
    fi
    # shellcheck source=/dev/null
    source "${ROOT}/scripts/cluster_env.sh" "${hf_root}"
  fi
}

_maybe_git_pull() {
  if [[ "${SKIP_GIT_PULL:-0}" == "1" ]]; then
    return 0
  fi
  if ! git pull --ff-only 2>/dev/null && ! git pull 2>/dev/null; then
    echo "[exp.sh] warn: git pull skipped (dirty tree or offline)" >&2
  fi
}

_resolve_model() {
  local model_override="${1:-}"
  case "${MODEL_SIZE:-32b}" in
    gemini)
      model="${MODEL:-gemini-2.5-pro}"
      backend="gemini"
      [[ -f APIKEY.sh ]] && source APIKEY.sh
      ;;
    7b) model="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}" ;;
    3b) model="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}" ;;
    *)  model="${MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}" ;;
  esac
  backend="${BACKEND:-transformers}"
  if [[ "${MODEL_SIZE:-32b}" == "gemini" ]]; then
    backend="gemini"
  fi
  if [[ -n "$model_override" ]]; then
    model="$model_override"
  fi
}

_run_once() {
  local task="$1"
  local model="$2"
  local backend="$3"
  local domain="$4"

  export BACKEND="${backend}"
  export VLM_MODEL="${model}"
  export GENERATE="${GENERATE:-1}"
  export RESUME="${RESUME:-1}"
  export DOMAIN="${domain}"

  if [[ "${domain}" == "google_robot" ]]; then
    export MOTION_PREPARE_MP4="${MOTION_PREPARE_MP4:-1}"
  else
    export MOTION_PREPARE_MP4="${MOTION_PREPARE_MP4:-0}"
    export MOTION_PREPARE_PAIRWISE="${MOTION_PREPARE_PAIRWISE:-0}"
    export PAIRWISE_SPECS="${PAIRWISE_SPECS:-data/results/verify/samples/motion_gt_neg_pairwise_pilot90/pairwise_specs_motion_gt_correct.json}"
  fi

  if [[ "${backend}" == "vllm" || "${backend}" == "local" ]]; then
    export VLLM_USE_V1="${VLLM_USE_V1:-0}"
    export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-10}"
    export VLM_BATCH_SIZE="${VLM_BATCH_SIZE:-10}"
  fi

  local extra=()
  if [[ "${SUMMARY:-0}" == "1" ]]; then
    extra+=(--summary)
  fi

  local target="${ONLY:-$task}"
  local exp_py="adhoc/generation/${domain}/exp.py"

  if [[ ! -f "${exp_py}" ]]; then
    echo "[exp.sh] exp.py not found: ${ROOT}/${exp_py}" >&2
    echo "[exp.sh] domain=${domain} — use robotarm or google_robot" >&2
    exit 1
  fi

  if ! command -v python >/dev/null 2>&1; then
    echo "[exp.sh] python not found — set SKIP_ENV=1 if already activated" >&2
    exit 1
  fi

  export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}:$ROOT/adhoc/generation/${domain}"

  local cmd=(python "${exp_py}" "${target}" --backend "${backend}" --model "${model}")
  if ((${#extra[@]})); then
    cmd+=("${extra[@]}")
  fi

  echo "[exp.sh] host=$(hostname) job=${SLURM_JOB_ID:-local} partition=${SLURM_JOB_PARTITION:-?}"
  echo "[exp.sh] cwd=${ROOT}"
  echo "[exp.sh] domain=${domain} model_size=${MODEL_SIZE:-32b} backend=${backend} model=${model}"
  echo "[exp.sh] only=${target} resume=${RESUME} generate=${GENERATE}"
  if [[ -n "${HF_HOME:-}" ]]; then
    echo "[exp.sh] HF_HOME=${HF_HOME}"
  fi
  if _is_cluster; then
    nvidia-smi || true
  fi
  echo "[exp.sh] ${cmd[*]}"
  if "${cmd[@]}"; then
    echo "[exp.sh] === done OK ==="
  else
    local rc=$?
    echo "[exp.sh] === failed exit $rc ===" >&2
    exit "$rc"
  fi
}

main() {
  _log_start
  _setup_shell
  _setup_env
  if _is_cluster; then
    _maybe_git_pull
  fi

  local task="${1:-all}"
  local model_override="${2:-}"
  local domain="${DOMAIN:-${4:-robotarm}}"
  if [[ -n "${4:-}" ]]; then
    domain="$4"
  fi

  if [[ "${ALL_MODELS:-0}" == "1" ]]; then
    local want_summary="${SUMMARY:-0}"
    export SUMMARY=0
    for ms in 32b 7b 3b; do
      MODEL_SIZE="$ms"
      _resolve_model "$model_override"
      _run_once "$task" "$model" "$backend" "$domain"
    done
    if [[ "$want_summary" == "1" ]]; then
      SUMMARY=1
      _resolve_model "$model_override"
      _run_once "$task" "$model" "$backend" "$domain"
    fi
    return 0
  fi

  _resolve_model "$model_override"
  _run_once "$task" "$model" "$backend" "$domain"
}

main "$@"
