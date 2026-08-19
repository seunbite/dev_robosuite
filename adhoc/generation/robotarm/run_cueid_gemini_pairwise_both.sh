#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
ROOT="$(pwd)"
PY="${PY:-/Library/Frameworks/Python.framework/Versions/3.10/bin/python3}"
OUT_ROOT="${OUT_ROOT:-$ROOT/data/results/evaluation/embodiment_cue_id/results_gemini25pro_judge_pairwise}"
LOG="${LOG:-$OUT_ROOT/run_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$OUT_ROOT"
# shellcheck disable=SC1091
set -a
source "$ROOT/APIKEY.sh"
set +a
exec >>"$LOG" 2>&1
echo "HOST=$(hostname) DATE=$(date) PY=$PY"
echo "=== GT pairwise ==="
"$PY" -u adhoc/generation/robotarm/run_embodiment_cue_id_gemini.py \
  --input data/results/evaluation/embodiment_cue_id/gt_pairwise.json \
  --output-dir "$OUT_ROOT/gt_pairwise/gemini_2.5_pro" \
  --model gemini-2.5-pro \
  --thinking-budget 0 \
  --workers 2
echo "=== Gemini25Pro pairwise ==="
"$PY" -u adhoc/generation/robotarm/run_embodiment_cue_id_gemini.py \
  --input data/results/evaluation/embodiment_cue_id/agent_gemini25pro_pairwise.json \
  --output-dir "$OUT_ROOT/agent_gemini25pro_pairwise/gemini_2.5_pro" \
  --model gemini-2.5-pro \
  --thinking-budget 0 \
  --workers 2
echo ALL_DONE
echo "LOG=$LOG"
