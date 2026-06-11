#!/usr/bin/env bash
# Pilot-90 experiments (tasks 1–10).
#
# Defaults: all 10 tasks, Qwen2.5-VL-32B, resume on, generate exp1/7 if missing.
#
#   bash exp.sh
#   SUMMARY=1 bash exp.sh
#   ONLY=1,2,3 bash exp.sh
#   MODEL_SIZE=gemini bash exp.sh
#   MODEL_SIZE=7b bash exp.sh
#   GENERATE=0 ONLY=2,3 bash exp.sh          # verify only (needs prior configs)
#   RESUME=0 ONLY=7 bash exp.sh
#   ALL_MODELS=1 SUMMARY=1 bash exp.sh       # summary for 32b, 7b, 3b
#
# Site (auto-detected): cluster → ~/sblee/dev_robosuite + conda m2m_caption32b
#                       local    → ~/Downloads/workspace/dev_robosuite + micromamba robosuite
#   EXP_SITE=local|cluster   override auto
#   SKIP_ENV=1               skip cd/activate (already in the right env)
set -euo pipefail

EXP_SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

_activate_conda() {
  local env_name="$1"
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$env_name" ]]; then
    return 0
  fi
  local cand
  for cand in \
    "${CONDA_EXE:+${CONDA_EXE%/*}/../etc/profile.d/conda.sh}" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"
  do
    [[ -n "$cand" && -f "$cand" ]] || continue
    # shellcheck disable=SC1090
    source "$cand"
    conda activate "$env_name"
    return 0
  done
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    conda activate "$env_name"
    return 0
  fi
  echo "exp.sh: could not activate conda env '$env_name' (set SKIP_ENV=1 if already active)" >&2
  exit 1
}

_activate_micromamba() {
  local env_name="$1"
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$env_name" || "${MAMBA_DEFAULT_ENV:-}" == "$env_name" ]]; then
    return 0
  fi
  if command -v micromamba >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    eval "$(micromamba shell hook -s bash)"
    micromamba activate "$env_name"
    return 0
  fi
  if [[ -f "$HOME/micromamba/etc/profile.d/micromamba.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/micromamba/etc/profile.d/micromamba.sh"
    micromamba activate "$env_name"
    return 0
  fi
  echo "exp.sh: could not activate micromamba env '$env_name' (set SKIP_ENV=1 if already active)" >&2
  exit 1
}

_setup_repo_and_env() {
  local need_qwen_env="${1:-1}"
  [[ "${SKIP_ENV:-0}" == "1" ]] && return 0

  local site="${EXP_SITE:-auto}"
  local server_root="${SERVER_ROOT:-$HOME/sblee/dev_robosuite}"
  local local_root="${LOCAL_ROOT:-$HOME/Downloads/workspace/dev_robosuite}"
  local server_env="${CONDA_ENV:-m2m_caption32b}"
  local local_env="${MAMBA_ENV:-robosuite}"

  if [[ "$site" == "auto" ]]; then
    if [[ -d "$server_root" ]] && [[ -d "/data/user_data/${USER:-}" || "${HOSTNAME:-}" == *babel* ]]; then
      site=cluster
    elif [[ -d "$local_root" ]]; then
      site=local
    elif [[ -d "$server_root" ]]; then
      site=cluster
    else
      site=local
      local_root="$(dirname "$EXP_SCRIPT")"
    fi
  fi

  case "$site" in
    cluster|server|babel)
      ROOT="$server_root"
      EXP_SITE_NAME="cluster"
      if [[ "$need_qwen_env" == "1" ]]; then
        _activate_conda "$server_env"
        EXP_PYTHON_ENV="$server_env (conda)"
      fi
      ;;
    local)
      ROOT="$local_root"
      EXP_SITE_NAME="local"
      if [[ "$need_qwen_env" == "1" ]]; then
        _activate_micromamba "$local_env"
        EXP_PYTHON_ENV="$local_env (micromamba)"
      fi
      ;;
    *)
      echo "Unknown EXP_SITE=$site (use auto, local, or cluster)" >&2
      exit 1
      ;;
  esac

  [[ -d "$ROOT" ]] || { echo "exp.sh: repo not found: $ROOT" >&2; exit 1; }
  cd "$ROOT"
  export EXP_SITE="$site"
}

# --- env aliases (read before site setup for ALL_MODELS) ---
SUMMARY="${SUMMARY:-${SUMMARY_ONLY:-0}}"
ONLY="${ONLY:-all}"
MODEL_SIZE="${MODEL_SIZE:-32b}"
RESUME="${RESUME:-1}"
GENERATE="${GENERATE:-1}"
export GENERATE RESUME MOTION_PREPARE_MP4 MOTION_PREPARE_PAIRWISE
# MuJoCo offscreen render (exp10 pairwise MP4) — needs GPU compute node + EGL
export MUJOCO_GL="${MUJOCO_GL:-egl}"

NEED_QWEN_ENV=1
[[ "$MODEL_SIZE" == "gemini" ]] && NEED_QWEN_ENV=0

_setup_repo_and_env "$NEED_QWEN_ENV"
mkdir -p logs

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}:$ROOT/adhoc/generation/robotarm"

# --- all Qwen sizes (batch mode) ---
if [[ "${ALL_MODELS:-0}" == "1" ]]; then
  if [[ "$SUMMARY" == "1" ]]; then
    for size in 32b 7b 3b; do
      echo ""
      echo "========== summary MODEL_SIZE=$size =========="
      MODEL_SIZE="$size" SUMMARY=1 ALL_MODELS=0 bash "$EXP_SCRIPT"
    done
    exit 0
  fi
  for size in 32b 7b 3b; do
    echo ""
    echo "========== run MODEL_SIZE=$size =========="
    MODEL_SIZE="$size" ALL_MODELS=0 bash "$EXP_SCRIPT"
  done
  echo ""
  echo "=== all model runs finished ==="
  ALL_MODELS=0 SUMMARY=1 bash "$EXP_SCRIPT"
  exit 0
fi

# --- model / backend ---
case "$MODEL_SIZE" in
  32b)
    export VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
    export BACKEND="${BACKEND:-transformers}"
    export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
    ;;
  7b)
    export VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
    export BACKEND="${BACKEND:-transformers}"
    export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
    ;;
  3b)
    export VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
    export BACKEND="${BACKEND:-transformers}"
    export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
    ;;
  gemini)
    export VLM_MODEL="${VLM_MODEL:-gemini-2.5-pro}"
    export BACKEND=gemini
    export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
    # shellcheck disable=SC1091
    [[ -f "$ROOT/APIKEY.sh" ]] && source "$ROOT/APIKEY.sh"
    ;;
  *)
    echo "Unknown MODEL_SIZE=$MODEL_SIZE (use 32b, 7b, 3b, or gemini)" >&2
    exit 1
    ;;
esac

export VLM_BACKEND="${VLM_BACKEND:-$BACKEND}"

if [[ "$BACKEND" != "gemini" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/cluster_env.sh" "${HF_HOME:-/data/user_data/${USER}/hf_cache}"
fi

PY="${PY:-$(command -v python3 || command -v python)}"
GT="data/seed/groundtruth/gt_manipulator.json"
PROMPT_EXP1="data/seed/prompt/manipulator/exp/prompt_exp1.txt"

ARGS=(adhoc/generation/robotarm/exp.py "$ONLY" --backend "$BACKEND" --model "$VLM_MODEL")
[[ "$RESUME" == "0" ]] && ARGS+=(--no-resume)
[[ "$SUMMARY" == "1" ]] && ARGS+=(--summary)
[[ $# -gt 0 ]] && ARGS+=("$@")

echo "=== pilot-90 exp ==="
echo "  site=${EXP_SITE_NAME:-${EXP_SITE:-auto}}  root=${ROOT}"
[[ -n "${EXP_PYTHON_ENV:-}" ]] && echo "  python_env=${EXP_PYTHON_ENV}"
echo "  tasks=${ONLY}  model_size=${MODEL_SIZE}  model=${VLM_MODEL}  backend=${BACKEND}"
echo "  summary=${SUMMARY}  generate=${GENERATE}  resume=${RESUME}"
[[ "$BACKEND" != "gemini" ]] && echo "  HF_HOME=${HF_HOME:-}"
echo "  gt=${GT}"
echo ""

for f in "$GT" "$PROMPT_EXP1"; do
  [[ -f "$f" ]] || { echo "Missing: $f — run: python adhoc/generation/robotarm/build_gt_manipulator.py" >&2; exit 1; }
done

if [[ "$SUMMARY" != "1" ]] && [[ "${ONLY}" == *"10"* || "${ONLY}" == "all" ]]; then
  "$PY" adhoc/generation/robotarm/build_pilot90_motion_pairwise_specs.py 2>/dev/null || true
fi

exec "$PY" "${ARGS[@]}"
