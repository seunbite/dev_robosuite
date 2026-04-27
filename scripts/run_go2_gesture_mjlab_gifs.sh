#!/usr/bin/env bash
# Record gesture GIFs in unitree_rl_mjlab (same Unitree-Go2-Flat-HF-Legacy env as velocity GIFs).
set -euo pipefail
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${WORKSPACE}/dev_robosuite:${WORKSPACE}/unitree_rl_mjlab${PYTHONPATH:+:$PYTHONPATH}"
export MICROMAMBA_ENV="${MICROMAMBA_ENV:-robosuite}"
exec micromamba run -n "$MICROMAMBA_ENV" python \
  "$WORKSPACE/unitree_rl_mjlab/scripts/record_go2_gesture_mjlab_gifs.py" "$@"
