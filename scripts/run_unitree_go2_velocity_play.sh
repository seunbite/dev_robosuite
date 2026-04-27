#!/usr/bin/env bash
# Wrapper: velocity-command Go2 policy (unitree_rl_mjlab + HF checkpoint).
# Requires: pip install -e unitree_rl_mjlab (see workspace unitree_rl_mjlab).
set -euo pipefail
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MICROMAMBA_ENV="${MICROMAMBA_ENV:-robosuite}"
exec bash "$WORKSPACE/unitree_rl_mjlab/scripts/play_go2_velocity_hf.sh" "$@"
