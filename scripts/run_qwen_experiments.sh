#!/usr/bin/env bash
# Run pilot pose-compare experiments with Qwen via vLLM (OpenAI-compatible API).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export VLM_BACKEND="${VLM_BACKEND:-openai}"
PY=python

run_multitile_pilot20() {
  echo "=== Exp 1: multitile GT (grid 6 + 12), pilot 20 cues ==="
  $PY adhoc/generation/robotarm/verify_pose_multitile_gt_gemini.py \
    --max-cues 20 \
    --grid-sizes 6,12 \
    --vlm-backend "$VLM_BACKEND" \
    --out-json data/results/verify/pilot20_pose_multitile_qwen.json \
    --resume
  $PY adhoc/generation/robotarm/build_pose_multitile_gt_review_html.py \
    --in-json data/results/verify/pilot20_pose_multitile_qwen.json \
    --out-html data/results/html/manipulator/pilot20_pose_multitile_qwen_review.html
}

run_multitile_pilot100() {
  echo "=== Exp 5: multitile GT (grid 6 + 12), all 100 cues ==="
  $PY adhoc/generation/robotarm/verify_pose_multitile_gt_gemini.py \
    --max-cues 100 \
    --grid-sizes 6,12 \
    --vlm-backend "$VLM_BACKEND" \
    --out-json data/results/verify/pilot100_pose_multitile_qwen.json \
    --resume
}

run_pairwise_pilot20() {
  echo "=== Pairwise 2-way baseline, pilot 20 ==="
  $PY adhoc/generation/robotarm/verify_pose_pairwise_12_gemini.py \
    --max-cues 20 \
    --one-pair-per-cue \
    --vlm-backend "$VLM_BACKEND" \
    --out-json data/results/verify/pilot20_pose_pairwise_qwen.json
}

run_fewshot_verify_pilot20() {
  echo "=== Exp 3: few-shot tile verify baseline ==="
  $PY adhoc/generation/robotarm/verify_pose_tiles_gemini.py \
    --vlm-backend "$VLM_BACKEND" 2>/dev/null || \
  echo "NOTE: wire verify_pose_tiles_gemini.py to vlm_client if not yet patched"
}

case "${1:-multitile20}" in
  multitile20) run_multitile_pilot20 ;;
  multitile100) run_multitile_pilot100 ;;
  pairwise20) run_pairwise_pilot20 ;;
  all20)
    run_pairwise_pilot20
    run_multitile_pilot20
    ;;
  *) echo "Usage: $0 {multitile20|multitile100|pairwise20|all20}"; exit 1 ;;
esac
