#!/usr/bin/env bash
# Pilot-90 (90 non-essence cues) × pose steps 1–6 — Qwen2.5-VL (single model load).
#
#   bash scripts/run_pilot90_qwen_suite.sh
#   MODEL_SIZE=7b RESUME=1 bash scripts/run_pilot90_qwen_suite.sh
#   SUMMARY_ONLY=1 bash scripts/run_pilot90_qwen_suite.sh
#   bash scripts/run_pilot90_qwen_all_models.sh   # 32b → 7b → 3b on server
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}:$ROOT/adhoc/generation/robotarm"

# shellcheck disable=SC1091
source "$ROOT/scripts/cluster_env.sh" "${HF_HOME:-/data/user_data/${USER}/hf_cache}"

MODEL_SIZE="${MODEL_SIZE:-32b}"
case "$MODEL_SIZE" in
  32b)
    export VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
    OUT_DIR="${OUT_DIR:-data/results/verify/pilot90_qwen32b}"
    export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
    ;;
  7b)
    export VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
    OUT_DIR="${OUT_DIR:-data/results/verify/pilot90_qwen7b}"
    export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
    ;;
  3b)
    export VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
    OUT_DIR="${OUT_DIR:-data/results/verify/pilot90_qwen3b}"
    export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
    ;;
  *)
    echo "Unknown MODEL_SIZE=$MODEL_SIZE (use 32b, 7b, or 3b)" >&2
    exit 1
    ;;
esac

export BACKEND="${BACKEND:-transformers}"
export VLM_BACKEND="${VLM_BACKEND:-transformers}"

POSE_CFG="data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json"
CONSOLIDATED="data/results/verify/pilot40_pose_eval_consolidated.json"
SHOTS="data/seed/shots/manipulator/shot_configs_v19_sophisticated.json"
TILE_DIR="data/results/visualize/pose_groups_12"
TILE_PICK="data/results/verify/pose_tile_pick_by_group.json"

PY=python
BACKEND_FLAG="$BACKEND"
[[ "$BACKEND" == "vllm" ]] && BACKEND_FLAG="local"

SUITE_ARGS=(
  adhoc/generation/robotarm/run_pilot90_qwen_suite.py
  --backend "$BACKEND"
  --model "$VLM_MODEL"
  --out-dir "$OUT_DIR"
)

[[ "${RESUME:-0}" == "1" ]] && SUITE_ARGS+=(--resume)
[[ "${SUMMARY_ONLY:-0}" == "1" ]] && SUITE_ARGS+=(--summary-only)
[[ -n "${ONLY:-}" ]] && SUITE_ARGS+=(--only "$ONLY")
[[ $# -gt 0 ]] && SUITE_ARGS+=("$@")

echo "=== pilot90 qwen suite (90 cues, pose steps 1–6) ==="
echo "  model_size=$MODEL_SIZE model=$VLM_MODEL backend=$BACKEND"
echo "  HF_HOME=$HF_HOME"
echo "  pose_cfg=$POSE_CFG"
echo "  out=$OUT_DIR"
echo "  scoring=any_pose_in_config"
echo ""

if [[ ! -f "$POSE_CFG" ]]; then
  echo "Missing POSE_CFG: $POSE_CFG" >&2
  exit 1
fi
if [[ ! -f "$CONSOLIDATED" ]]; then
  echo "Missing CONSOLIDATED: $CONSOLIDATED" >&2
  exit 1
fi

if [[ "${SEPARATE_STEPS:-0}" == "1" ]]; then
  echo "=== SEPARATE_STEPS=1: one Python process per step ==="
  COMMON=(--vlm-backend "$BACKEND_FLAG" --model "$VLM_MODEL")
  CUES="$($PY -c "import sys; sys.path.insert(0,'adhoc/generation/robotarm'); from pilot90_experiment_suite import manifest90_cues_csv; print(manifest90_cues_csv())")"

  $PY adhoc/generation/robotarm/run_pilot90_qwen_suite.py --only 1 --skip-model-load --out-dir "$OUT_DIR"

  $PY adhoc/generation/robotarm/verify_pose_tiles_gemini.py \
    --config-json "$POSE_CFG" --shots-json "$SHOTS" \
    --tile-dir "$TILE_DIR" --tile-pick-json "$TILE_PICK" \
    "${COMMON[@]}" \
    --out-json "$OUT_DIR/exp02_pose_verify_vlm.json" \
    --out-md "$OUT_DIR/exp02_pose_verify_vlm.md"

  $PY adhoc/generation/robotarm/verify_pose_textonly_gemini.py \
    --config-json "$POSE_CFG" --shots-json "$SHOTS" \
    "${COMMON[@]}" \
    --out-json "$OUT_DIR/exp03_pose_verify_text.json"

  $PY adhoc/generation/robotarm/verify_pose_pairwise_12_gemini.py \
    --consolidated-json "$CONSOLIDATED" \
    --tile-dir "$TILE_DIR" --tile-pick-json "$TILE_PICK" \
    --image-dir data/results/visualize/pose_pairwise_12_pilot90 \
    --cues "$CUES" --one-pair-per-cue \
    "${COMMON[@]}" \
    --out-json "$OUT_DIR/exp04_pose_pairwise_2way.json"

  $PY adhoc/generation/robotarm/verify_pose_multitile_gt_gemini.py \
    --consolidated-json "$CONSOLIDATED" \
    --tile-dir "$TILE_DIR" --tile-pick-json "$TILE_PICK" \
    --image-dir data/results/visualize/pose_multitile_gt_pilot90_grid6 \
    --grid-sizes 6 --cues "$CUES" \
    "${COMMON[@]}" \
    --out-json "$OUT_DIR/exp05_pose_multitile_grid6.json" \
    --resume

  $PY adhoc/generation/robotarm/verify_pose_multitile_gt_gemini.py \
    --consolidated-json "$CONSOLIDATED" \
    --tile-dir "$TILE_DIR" --tile-pick-json "$TILE_PICK" \
    --image-dir data/results/visualize/pose_multitile_gt_pilot90_grid12 \
    --grid-sizes 12 --cues "$CUES" \
    "${COMMON[@]}" \
    --out-json "$OUT_DIR/exp06_pose_multitile_grid12.json" \
    --resume

  $PY adhoc/generation/robotarm/run_pilot90_qwen_suite.py --summary-only --out-dir "$OUT_DIR"
else
  "$PY" "${SUITE_ARGS[@]}"
fi
