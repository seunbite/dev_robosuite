#!/usr/bin/env bash
# Run pilot-90 pose suite for 32B, 7B, and 3B (cluster / server batch).
#
#   bash scripts/run_pilot90_qwen_all_models.sh
#   RESUME=1 bash scripts/run_pilot90_qwen_all_models.sh
#   SUMMARY_ONLY=1 bash scripts/run_pilot90_qwen_all_models.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "${SUMMARY_ONLY:-0}" == "1" ]]; then
  for size in 32b 7b 3b; do
    echo ""
    echo "========== pilot90 summary MODEL_SIZE=$size =========="
    MODEL_SIZE="$size" SUMMARY_ONLY=1 bash scripts/run_pilot90_qwen_suite.sh
  done
  exit 0
fi

for size in 32b 7b 3b; do
  echo ""
  echo "========== pilot90 run MODEL_SIZE=$size =========="
  MODEL_SIZE="$size" RESUME="${RESUME:-1}" bash scripts/run_pilot90_qwen_suite.sh
done

echo ""
echo "=== all pilot90 model runs finished ==="
SUMMARY_ONLY=1 bash scripts/run_pilot90_qwen_all_models.sh
