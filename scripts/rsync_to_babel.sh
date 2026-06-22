#!/usr/bin/env bash
# Sync pilot-90 code + pairwise artifacts to Babel.
#
#   bash scripts/rsync_to_babel.sh
#   REMOTE=user@login2.babel.cs.cmu.edu REMOTE_DIR=/path/to/dev_robosuite bash scripts/rsync_to_babel.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REMOTE="${REMOTE:-hoyeonk@login2.babel.cs.cmu.edu}"
REMOTE_DIR="${REMOTE_DIR:-/home/hoyeonk/sblee/dev_robosuite}"
PAIRWISE="data/results/verify/samples/motion_gt_neg_pairwise_pilot90"

echo "=== rsync -> ${REMOTE}:${REMOTE_DIR} ==="

rsync -avz --progress \
  "$ROOT/adhoc/generation/robotarm/" \
  "${REMOTE}:${REMOTE_DIR}/adhoc/generation/robotarm/"

rsync -avz --progress \
  "$ROOT/scripts/" \
  "${REMOTE}:${REMOTE_DIR}/scripts/"

rsync -avz --progress \
  "$ROOT/${PAIRWISE}/pairwise_specs_motion_gt_correct.json" \
  "$ROOT/${PAIRWISE}/motion_gt_correct_subset.json" \
  "$ROOT/${PAIRWISE}/pairwise_specs_motion_gt_correct_review.html" \
  "$ROOT/${PAIRWISE}/"*_pair_spec.json \
  "$ROOT/${PAIRWISE}/"*_pair_axis.mp4 \
  "${REMOTE}:${REMOTE_DIR}/${PAIRWISE}/"

echo ""
echo "On cluster:"
echo "  ssh ${REMOTE}"
echo "  cd ${REMOTE_DIR} && git pull"
echo "  sbg  --export=ALL,MODEL_SIZE=7b,ONLY=8,9,MOTION_PREPARE_MP4=0 exp.sh"
echo "  sbg2 --export=ALL,MODEL_SIZE=32b,ONLY=4,5,6,10,MOTION_PREPARE_PAIRWISE=0 exp.sh"
echo "  sbgd --export=ALL,MODEL_SIZE=3b,ONLY=4,5,6,10 exp.sh"
echo "  SUMMARY=1 bash exp.sh"
