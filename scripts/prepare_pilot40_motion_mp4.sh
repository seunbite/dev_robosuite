#!/usr/bin/env bash
# Prepare pilot-40 motion MP4s for step 8 (MuJoCo render + ffmpeg). No Qwen load.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC1091
source "$ROOT/scripts/cluster_env.sh" "${HF_HOME:-/data/user_data/${USER}/hf_cache}" 2>/dev/null || true
python adhoc/generation/robotarm/prepare_pilot40_motion_mp4.py "$@"
