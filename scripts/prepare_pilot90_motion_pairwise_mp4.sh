#!/usr/bin/env bash
# Render GT vs neg-axis pairwise MP4s for pilot-90 step 10 (MuJoCo + ffmpeg).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}:$ROOT/adhoc/generation/robotarm:$ROOT/adhoc/vlm_test"
# shellcheck disable=SC1091
source "$ROOT/scripts/cluster_env.sh" "${HF_HOME:-/data/user_data/${USER}/hf_cache}" 2>/dev/null || true
python adhoc/generation/robotarm/prepare_pilot90_motion_pairwise_mp4.py "$@"
