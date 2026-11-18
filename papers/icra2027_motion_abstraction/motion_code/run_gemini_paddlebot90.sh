#!/usr/bin/env bash
# Local Gemini paddlebot-90 generation (needs APIKEY approval in your shell).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PAPER="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/APIKEY.sh"
cd "${PAPER}"
mkdir -p "${ROOT}/logs"
LOG="${ROOT}/logs/gemini_paddlebot90.log"
echo "logging → ${LOG}"
exec /Users/sb/micromamba/envs/robosuite/bin/python motion_code/generate_paddlebot90.py \
  --backend gemini --model gemini-2.5-pro --resume --sleep 0.4 \
  2>&1 | tee "${LOG}"
