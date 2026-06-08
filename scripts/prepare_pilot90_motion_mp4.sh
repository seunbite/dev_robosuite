#!/usr/bin/env bash
# Prepare pilot-90 motion MP4s for step 8 (run/IIWA GIF → MP4). No VLM load.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}:$ROOT/adhoc/generation/robotarm"
# shellcheck disable=SC1091
source "$ROOT/scripts/cluster_env.sh" "${HF_HOME:-/data/user_data/${USER}/hf_cache}" 2>/dev/null || true
python adhoc/generation/robotarm/prepare_pilot90_motion_mp4.py "$@"
