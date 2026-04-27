#!/usr/bin/env bash
# Record 3 GIFs (forward / strafe / turn) with fixed body-frame velocity commands.
set -euo pipefail
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MICROMAMBA_ENV="${MICROMAMBA_ENV:-robosuite}"
exec micromamba run -n "$MICROMAMBA_ENV" python \
  "$WORKSPACE/unitree_rl_mjlab/scripts/record_go2_velocity_headless_gifs.py" "$@"
