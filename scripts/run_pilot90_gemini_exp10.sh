#!/usr/bin/env bash
# Gemini exp10 on pilot-90 pairwise MP4s (API; no GPU).
#
#   bash scripts/run_pilot90_gemini_exp10.sh
#   RESUME=1 bash scripts/run_pilot90_gemini_exp10.sh
#   LIMIT=5 bash scripts/run_pilot90_gemini_exp10.sh   # smoke test
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs data/results/verify/pilot90_gemini

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}:$ROOT/adhoc/generation/robotarm"

PY="${PY:-python}"
if [[ -x /Users/sb/micromamba/envs/robosuite/bin/python ]]; then
  PY=/Users/sb/micromamba/envs/robosuite/bin/python
fi

POSE_CFG="data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json"
PAIRWISE_DIR="data/results/verify/samples/motion_gt_neg_pairwise_pilot90"
SPECS="${PAIRWISE_SPECS:-$PAIRWISE_DIR/pairwise_specs_motion_gt_correct.json}"
OUT="${OUT_JSON:-data/results/verify/pilot90_gemini/exp10_motion_pairwise_mp4.json}"
# Build unified MP4s + specs first:
#   python adhoc/generation/robotarm/prepare_pilot90_motion_pairwise_mp4.py --force
#   python adhoc/generation/robotarm/build_motion_pairwise_input_review_html.py
MODEL="${GEMINI_MODEL:-gemini-2.5-pro}"

[[ -f "$POSE_CFG" ]] || { echo "Missing $POSE_CFG" >&2; exit 1; }
[[ -f "$SPECS" ]] || "$PY" adhoc/generation/robotarm/build_pilot90_motion_pairwise_specs.py

READY=$(ls "$PAIRWISE_DIR"/*_pair_axis.mp4 2>/dev/null | wc -l | tr -d ' ')
echo "=== pilot90 gemini exp10 ==="
echo "  model=$MODEL"
echo "  specs=$SPECS"
echo "  pairwise_mp4_ready=$READY"
echo "  out=$OUT"
echo ""

ARGS=(
  adhoc/generation/robotarm/verify_motion_gt_neg_pairwise_vlm.py
  --vlm-backend gemini
  --model "$MODEL"
  --motion-cfg "$POSE_CFG"
  --pairwise-dir "$PAIRWISE_DIR"
  --pairwise-jsons "$SPECS"
  --out-json "$OUT"
)
[[ "${RESUME:-1}" == "0" ]] && ARGS+=(--no-resume)
[[ -n "${LIMIT:-}" ]] && ARGS+=(--limit "$LIMIT")

nohup "$PY" "${ARGS[@]}" > logs/pilot90_gemini_exp10.log 2>&1 &
echo "PID=$! log=logs/pilot90_gemini_exp10.log"
