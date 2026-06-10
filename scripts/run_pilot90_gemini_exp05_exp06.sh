#!/usr/bin/env bash
# Gemini exp05 (Pose grid-6) + exp06 (Pose grid-12) on pilot-90 manifest.
#
#   source APIKEY.sh
#   bash scripts/run_pilot90_gemini_exp05_exp06.sh
#   ONLY=5 bash scripts/run_pilot90_gemini_exp05_exp06.sh
#   ONLY=6 RESUME=0 bash scripts/run_pilot90_gemini_exp05_exp06.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs data/results/verify/pilot90_gemini

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}:$ROOT/adhoc/generation/robotarm"

PY="${PY:-python}"
if [[ -x /Users/sb/micromamba/envs/robosuite/bin/python ]]; then
  PY=/Users/sb/micromamba/envs/robosuite/bin/python
fi

CONSOLIDATED="data/results/verify/pilot40_pose_eval_consolidated.json"
POSE_CFG="data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json"
TILE_DIR="data/results/visualize/pose_groups_12"
TILE_PICK="data/results/verify/pose_tile_pick_by_group.json"
MODEL="${GEMINI_MODEL:-gemini-2.5-pro}"
CUES="$($PY -c "import sys; sys.path.insert(0,'adhoc/generation/robotarm'); from pilot90_experiment_suite import manifest90_cues_csv; print(manifest90_cues_csv())")"

for f in "$CONSOLIDATED" "$POSE_CFG" "$TILE_PICK"; do
  [[ -f "$f" ]] || { echo "Missing $f" >&2; exit 1; }
done
[[ -d "$TILE_DIR" ]] || { echo "Missing dir $TILE_DIR" >&2; exit 1; }

COMMON=(
  --vlm-backend gemini
  --model "$MODEL"
  --consolidated-json "$CONSOLIDATED"
  --tile-dir "$TILE_DIR"
  --tile-pick-json "$TILE_PICK"
  --cues "$CUES"
)
RESUME_FLAG=()
[[ "${RESUME:-1}" == "1" ]] && RESUME_FLAG=(--resume)
MAX_CUES="${LIMIT:-90}"

run_exp05() {
  echo "=== pilot90 gemini exp05 (Pose grid-6) ===" >&2
  nohup "$PY" adhoc/generation/robotarm/verify_pose_multitile_gt_gemini.py \
    "${COMMON[@]}" \
    --image-dir data/results/visualize/pose_multitile_gt_pilot90_grid6 \
    --grid-sizes 6 \
    --out-json data/results/verify/pilot90_gemini/exp05_pose_multitile_grid6.json \
    --max-cues "$MAX_CUES" \
    ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"} \
    > logs/pilot90_gemini_exp05.log 2>&1 &
  echo "exp05 PID=$! log=logs/pilot90_gemini_exp05.log"
}

run_exp06() {
  echo "=== pilot90 gemini exp06 (Pose grid-12) ===" >&2
  nohup "$PY" adhoc/generation/robotarm/verify_pose_multitile_gt_gemini.py \
    "${COMMON[@]}" \
    --image-dir data/results/visualize/pose_multitile_gt_pilot90_grid12 \
    --grid-sizes 12 \
    --out-json data/results/verify/pilot90_gemini/exp06_pose_multitile_grid12.json \
    --max-cues "$MAX_CUES" \
    ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"} \
    > logs/pilot90_gemini_exp06.log 2>&1 &
  echo "exp06 PID=$! log=logs/pilot90_gemini_exp06.log"
}

case "${ONLY:-all}" in
  5|05) run_exp05 ;;
  6|06) run_exp06 ;;
  all|5,6|05,06)
    run_exp05
    run_exp06
    ;;
  *)
    echo "ONLY must be 5, 6, or all (got ONLY=${ONLY:-all})" >&2
    exit 1
    ;;
esac
