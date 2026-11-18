#!/usr/bin/env bash
# Pull Babel Qwen humanlike90 outputs and score cue-spec Orient/Loc/Move.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
REMOTE="${REMOTE:-hoyeonk@login2.babel.cs.cmu.edu}"
REMOTE_DIR="${REMOTE_DIR:-/home/hoyeonk/sblee/dev_robosuite}"
TAG=qwen32b_humanlike90
LOCAL="data/results/motion_configs/embodiment_motion_code_qwen_${TAG}"
REMOTE_OUT="${REMOTE_DIR}/data/results/motion_configs/embodiment_motion_code_qwen_${TAG}"

mkdir -p "$LOCAL"
rsync -avz "${REMOTE}:${REMOTE_OUT}/" "$LOCAL/"

python papers/icra2027_motion_abstraction/score_humanlike_cuespec.py \
  --pred "$LOCAL/embodiment_motion_code_all.json" \
  --tag "$TAG" \
  --out-json "data/results/evaluation/humanlike_cuespec_scores_${TAG}.json" \
  --out-md "papers/icra2027_motion_abstraction/humanlike_cuespec_scores_${TAG}.md"

echo "Done. See papers/icra2027_motion_abstraction/humanlike_cuespec_scores_${TAG}.md"
