#!/usr/bin/env bash
# Sync current cue-ID packs + updated GT/Gemini mp4s to Babel, then submit
# Qwen2.5-VL-32B: pairwise (gt + gemini) + yes/no no-quad (gt + gemini).
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
REMOTE="${REMOTE:-babel}"
# /data is only visible on compute nodes; stage via login home then srun-copy.
STAGE="${STAGE:-/home/hoyeonk/staging/cueid32b_sync}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/user_data/hoyeonk/sb/motion2mind}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results_qwen32b_20260802}"

echo "== stage to login home ($STAGE) =="
ssh "$REMOTE" "mkdir -p $STAGE/adhoc/generation/robotarm $STAGE/eval_json $STAGE/render_gt $STAGE/render_gem"
rsync -az \
  "$REPO/adhoc/generation/robotarm/run_embodiment_cue_id_vllm.py" \
  "$REPO/adhoc/generation/robotarm/vllm_local.py" \
  "$REPO/adhoc/generation/robotarm/hf_cache_setup.py" \
  "$REPO/adhoc/generation/robotarm/submit_cueid_32b_pairwise_and_yesno_noquad.sh" \
  "$REMOTE:$STAGE/adhoc/generation/robotarm/"
rsync -az "$REPO/data/results/evaluation/embodiment_cue_id/"*.json "$REMOTE:$STAGE/eval_json/"
rsync -az --include='*/' --include='*.mp4' --exclude='*' \
  "$REPO/data/results/render/embodiment_motion_code_gt/" "$REMOTE:$STAGE/render_gt/"
rsync -az --include='*/' --include='*.mp4' --exclude='*' \
  "$REPO/data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro/" "$REMOTE:$STAGE/render_gem/"

echo "== deploy /data + submit via srun =="
ssh "$REMOTE" "srun --partition=debug --time=00:25:00 --cpus-per-task=2 --mem=8G bash -lc '
set -euo pipefail
STAGE=$STAGE; ROOT=$REMOTE_ROOT; RESULTS_SUBDIR=$RESULTS_SUBDIR
mkdir -p \$ROOT/adhoc/generation/robotarm \$ROOT/data/results/evaluation/embodiment_cue_id \
  \$ROOT/data/results/render/embodiment_motion_code_gt \
  \$ROOT/data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro
cp -a \$STAGE/adhoc/generation/robotarm/. \$ROOT/adhoc/generation/robotarm/
cp -a \$STAGE/eval_json/. \$ROOT/data/results/evaluation/embodiment_cue_id/
rsync -a \$STAGE/render_gt/ \$ROOT/data/results/render/embodiment_motion_code_gt/
rsync -a \$STAGE/render_gem/ \$ROOT/data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro/
RESULTS_SUBDIR=\$RESULTS_SUBDIR bash \$ROOT/adhoc/generation/robotarm/submit_cueid_32b_pairwise_and_yesno_noquad.sh
'"
echo "Watch: ssh $REMOTE squeue -u \$USER"
