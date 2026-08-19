#!/usr/bin/env bash
# Sync rebuilt H4 glossary_value (English blank-config) packs to Babel and submit 3B/7B/32B.
# Protocol: blank-config + glossary_value (gesture meaning + motion-code glossary), fully English.
# Requires working SSH to Host `babel` in ~/.ssh/config.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
REMOTE="${REMOTE:-babel}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/user_data/hoyeonk/sb/motion2mind}"
LOCAL_EVAL="$REPO/data/results/evaluation"
REMOTE_EVAL="$REMOTE_ROOT/paper_eval/h4_mcq"

FILES=(
  h4_gt_binary_mcq.json
  h4_gt_binary_mcq_counterbalanced.json
  h4_gt_blank_config_mcq.json
  h4_gt_blank_config_mcq_counterbalanced.json
  h4_glossary_value_english.json
  h4_glossary_value_english_counterbalanced.json
)

echo "Syncing rebuilt H4 packs → $REMOTE:$REMOTE_EVAL"
ssh "$REMOTE" "mkdir -p '$REMOTE_EVAL'"
for f in "${FILES[@]}"; do
  scp "$LOCAL_EVAL/$f" "$REMOTE:$REMOTE_EVAL/$f"
done

scp "$REPO/adhoc/generation/robotarm/run_h4_mcq_vllm.py" \
  "$REMOTE:$REMOTE_ROOT/adhoc/h4_mcq/run_h4_mcq_vllm.py" || true
scp "$REPO/adhoc/generation/robotarm/submit_h4_mcq_babel.sh" \
  "$REMOTE:$REMOTE_ROOT/adhoc/generation/robotarm/submit_h4_mcq_babel.sh" || true

echo "Submitting glossary_en (blank-config + English glossary_value) for 3B/7B/32B..."
echo "Each model job runs original + counterbalanced in one load."
ssh "$REMOTE" "bash $REMOTE_ROOT/adhoc/generation/robotarm/submit_h4_mcq_babel.sh glossary_en all"
echo "Done submitting. Check squeue on babel."
