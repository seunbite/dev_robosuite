#!/usr/bin/env bash
# Activate the project VLM env on Babel (micromamba prefix: <repo>/y).
#
#   source scripts/activate_cluster_vlm.sh
#   source scripts/activate_cluster_vlm.sh /path/to/dev_robosuite
#
# Do NOT rely on m2m_caption32b unless you created it locally; this repo ships
# robosuite-vlm under y/envs/. If CUDA fails, on a GPU node run:
#   bash scripts/install_vlm_transformers.sh   # CUDA_WHEEL=cu118 (default)
set -euo pipefail

ROOT="${1:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
ENV_NAME="${VLM_ENV_NAME:-robosuite-vlm}"
ENV_BIN="$ROOT/y/envs/$ENV_NAME/bin"

if [[ ! -x "$ENV_BIN/python" ]]; then
  echo "Missing VLM env: $ENV_BIN/python" >&2
  echo "  Create with: micromamba create -p $ROOT/y/envs/$ENV_NAME python=3.10 -y" >&2
  echo "  Then on a GPU node: bash scripts/install_vlm_transformers.sh" >&2
  return 1 2>/dev/null || exit 1
fi

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$ROOT/y}"
export PATH="$ENV_BIN:$PATH"

if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash 2>/dev/null)" || true
  micromamba activate "$ENV_NAME" 2>/dev/null || true
elif [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-$ENV_NAME}" 2>/dev/null || true
fi

echo "[env] python=$(command -v python)"
echo "[env] env_prefix=$ENV_BIN"
"$ENV_BIN/python" - <<'PY' || true
import torch
print(f"[env] torch={torch.__version__} cuda_build={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[env] gpu={torch.cuda.get_device_name(0)}")
PY
