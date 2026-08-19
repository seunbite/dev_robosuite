#!/usr/bin/env bash
# Sync embodiment cue-ID eval + media + runners to Babel, then sbatch 32B jobs.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
REMOTE="${REMOTE:-babel}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/user_data/hoyeonk/sb/motion2mind}"

echo "Syncing cue-ID code + eval JSON..."
ssh "$REMOTE" "mkdir -p $REMOTE_ROOT/data/results/evaluation/embodiment_cue_id \
  $REMOTE_ROOT/adhoc/generation/robotarm \
  $REMOTE_ROOT/data/results/render"

rsync -az --progress \
  "$REPO/adhoc/generation/robotarm/build_embodiment_cue_id_eval.py" \
  "$REPO/adhoc/generation/robotarm/run_embodiment_cue_id_vllm.py" \
  "$REPO/adhoc/generation/robotarm/submit_embodiment_cue_id_babel.sh" \
  "$REPO/adhoc/generation/robotarm/vllm_local.py" \
  "$REMOTE:$REMOTE_ROOT/adhoc/generation/robotarm/"

rsync -az --progress \
  "$REPO/data/results/evaluation/embodiment_cue_id/" \
  "$REMOTE:$REMOTE_ROOT/data/results/evaluation/embodiment_cue_id/"

echo "Syncing render mp4/gif trees (gt + grok45)..."
rsync -az --progress \
  "$REPO/data/results/render/embodiment_motion_code_gt/" \
  "$REMOTE:$REMOTE_ROOT/data/results/render/embodiment_motion_code_gt/"
rsync -az --progress \
  "$REPO/data/results/render/embodiment_motion_code_agent_grok45/" \
  "$REMOTE:$REMOTE_ROOT/data/results/render/embodiment_motion_code_agent_grok45/"

echo "Submitting cue-ID 32B jobs..."
ssh "$REMOTE" "bash $REMOTE_ROOT/adhoc/generation/robotarm/submit_embodiment_cue_id_babel.sh all 32b"
echo "Done. Watch: ssh babel squeue -u \$USER"
