#!/usr/bin/env bash
# Build unified pairwise MP4s + specs + review HTML (Gemini + Qwen exp10).
#
#   bash scripts/build_pilot90_pairwise_media.sh
#   FORCE=1 bash scripts/build_pilot90_pairwise_media.sh
#   LIMIT=5 bash scripts/build_pilot90_pairwise_media.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}:$ROOT/adhoc/generation/robotarm"

PY="${PY:-python}"
if [[ -x /Users/sb/micromamba/envs/robosuite/bin/python ]]; then
  PY=/Users/sb/micromamba/envs/robosuite/bin/python
fi

PREP_ARGS=()
[[ "${FORCE:-0}" == "1" ]] && PREP_ARGS+=(--force)
[[ -n "${LIMIT:-}" ]] && PREP_ARGS+=(--limit "$LIMIT")
[[ -n "${SUBSET_JSON:-}" ]] && PREP_ARGS+=(--subset-json "$SUBSET_JSON")
[[ -n "${FROM_GEMINI_EXP10:-}" ]] && PREP_ARGS+=(--from-gemini-exp10 "$FROM_GEMINI_EXP10")
[[ "${FROM_MOTION_GT_SCORE:-0}" == "1" ]] && PREP_ARGS+=(--from-motion-gt-score)

if ((${#PREP_ARGS[@]})); then
  $PY adhoc/generation/robotarm/prepare_pilot90_motion_pairwise_mp4.py "${PREP_ARGS[@]}"
else
  $PY adhoc/generation/robotarm/prepare_pilot90_motion_pairwise_mp4.py
fi
$PY adhoc/generation/robotarm/build_motion_pairwise_input_review_html.py
echo "Open: data/results/verify/samples/motion_gt_neg_pairwise_pilot90/pairwise_input_review.html"
