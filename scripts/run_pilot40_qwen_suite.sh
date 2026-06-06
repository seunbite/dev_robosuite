#!/usr/bin/env bash
# Pilot-40 (39 cues) × 10 experiments — Qwen2.5-VL-32B (single model load).
#
# Usage (GPU node, e.g. salloc):
#   bash scripts/run_pilot40_qwen_suite.sh
#   bash scripts/run_pilot40_qwen_suite.sh --only 5,6        # new multitile 6/12 only
#   RESUME=1 bash scripts/run_pilot40_qwen_suite.sh
#   SEPARATE_STEPS=1 bash scripts/run_pilot40_qwen_suite.sh   # 10× model reload (slow)
#
# Gemini baselines (steps 1–4, 7–10 already done; 5–6 are new):
#   1  pose gen      → motion_configs_prompt_v19_generation_pose_pilot40.json
#                      scored: pilot40_pose_eval_consolidated_scored.tsv
#   2  pose VLM      → pose_tile_verify_pilot{10,20,20_more}_gemini.json
#   3  pose text     → pose_textonly_verify_pilot{10,20,20_more}_gemini.json
#   4  pose 2-way    → pilot40_pose_pairwise_12_gemini.json
#   5  multitile 6   → (no pilot40 Gemini; closest: pilot20_pose_multitile_gt_gemini.json)
#   6  multitile 12  → (same)
#   7  motion gen    → motion_configs_prompt_v19_gt_fixed_pose_pilot40.json
#                      metrics: pilot40_motion_verify_metrics.json
#   8  motion VLM    → pilot40_motion_component_verify_gemini.json
#   9  motion text   → pilot40_motion_component_verify_text_gemini.json
#  10  motion mp4    → samples/motion_gt_neg_pairwise/pairwise_eval_results*.json
#
# Qwen outputs: data/results/verify/pilot40_qwen32b/exp{01..10}_*.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# shellcheck disable=SC1091
source "$ROOT/scripts/cluster_env.sh" "${HF_HOME:-/data/user_data/${USER}/hf_cache}"

export BACKEND="${BACKEND:-transformers}"
export VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
export VLM_BACKEND="${VLM_BACKEND:-transformers}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"

OUT_DIR="${OUT_DIR:-data/results/verify/pilot40_qwen32b}"
POSE_CFG="data/results/motion_configs/manipulator/motion_configs_prompt_v19_generation_pose_pilot40.json"
MOTION_CFG="data/results/motion_configs/manipulator/motion_configs_prompt_v19_gt_fixed_pose_pilot40.json"
PROMPT_POSE="data/seed/prompt/manipulator/prompt_v19_sophisticated.txt"
PROMPT_MOTION="data/seed/prompt/manipulator/prompt_gt_fixed_first_pose.txt"
CONSOLIDATED="data/results/verify/pilot40_pose_eval_consolidated.json"
MANIFEST="data/results/render/manipulator/motion_vlm_verify_pilot40/manifest_pilot40.json"
SHOTS="data/seed/shots/manipulator/shot_configs_v19_sophisticated.json"
TILE_DIR="data/results/visualize/pose_groups_12"
TILE_PICK="data/results/verify/pose_tile_pick_by_group.json"

PY=python
BACKEND_FLAG="$BACKEND"
[[ "$BACKEND" == "vllm" ]] && BACKEND_FLAG="local"

SUITE_ARGS=(
  adhoc/generation/robotarm/run_pilot40_qwen_suite.py
  --backend "$BACKEND"
  --model "$VLM_MODEL"
  --out-dir "$OUT_DIR"
)

[[ "${RESUME:-0}" == "1" ]] && SUITE_ARGS+=(--resume)
[[ -n "${ONLY:-}" ]] && SUITE_ARGS+=(--only "$ONLY")
[[ $# -gt 0 ]] && SUITE_ARGS+=("$@")

echo "=== pilot40 qwen suite (39 cues, 10 steps) ==="
echo "  model=$VLM_MODEL backend=$BACKEND"
echo "  HF_HOME=$HF_HOME"
echo "  out=$OUT_DIR"
echo ""

if [[ "${SEPARATE_STEPS:-0}" == "1" ]]; then
  echo "=== SEPARATE_STEPS=1: one Python process per step (reloads model each time) ==="
  COMMON=(--vlm-backend "$BACKEND_FLAG" --model "$VLM_MODEL")

  # 1 — pose generation score (no GPU)
  $PY adhoc/generation/robotarm/run_pilot40_qwen_suite.py --only 1 --skip-model-load --out-dir "$OUT_DIR"

  # 2 — pose VLM verify
  $PY adhoc/generation/robotarm/verify_pose_tiles_gemini.py \
    --config-json "$POSE_CFG" --shots-json "$SHOTS" \
    --tile-dir "$TILE_DIR" --tile-pick-json "$TILE_PICK" \
    "${COMMON[@]}" \
    --out-json "$OUT_DIR/exp02_pose_verify_vlm.json" \
    --out-md "$OUT_DIR/exp02_pose_verify_vlm.md"

  # 3 — pose text verify
  $PY adhoc/generation/robotarm/verify_pose_textonly_gemini.py \
    --config-json "$POSE_CFG" --shots-json "$SHOTS" \
    "${COMMON[@]}" \
    --out-json "$OUT_DIR/exp03_pose_verify_text.json"

  # 4 — pose pairwise 2-way (39 cues)
  CUES="$($PY -c "import sys; sys.path.insert(0,'adhoc/generation/robotarm'); from pilot40_experiment_suite import pilot40_cues_csv; print(pilot40_cues_csv())")"
  $PY adhoc/generation/robotarm/verify_pose_pairwise_12_gemini.py \
    --consolidated-json "$CONSOLIDATED" \
    --tile-dir "$TILE_DIR" --tile-pick-json "$TILE_PICK" \
    --image-dir data/results/visualize/pose_pairwise_12_pilot40 \
    --cues "$CUES" --one-pair-per-cue \
    "${COMMON[@]}" \
    --out-json "$OUT_DIR/exp04_pose_pairwise_2way.json"

  # 5 — multitile grid 6
  $PY adhoc/generation/robotarm/verify_pose_multitile_gt_gemini.py \
    --consolidated-json "$CONSOLIDATED" \
    --tile-dir "$TILE_DIR" --tile-pick-json "$TILE_PICK" \
    --image-dir data/results/visualize/pose_multitile_gt_pilot40_grid6 \
    --grid-sizes 6 --cues "$CUES" \
    "${COMMON[@]}" \
    --out-json "$OUT_DIR/exp05_pose_multitile_grid6.json" \
    --resume

  # 6 — multitile grid 12
  $PY adhoc/generation/robotarm/verify_pose_multitile_gt_gemini.py \
    --consolidated-json "$CONSOLIDATED" \
    --tile-dir "$TILE_DIR" --tile-pick-json "$TILE_PICK" \
    --image-dir data/results/visualize/pose_multitile_gt_pilot40_grid12 \
    --grid-sizes 12 --cues "$CUES" \
    "${COMMON[@]}" \
    --out-json "$OUT_DIR/exp06_pose_multitile_grid12.json" \
    --resume

  # 7 — motion generation score (no GPU)
  $PY adhoc/generation/robotarm/run_pilot40_qwen_suite.py --only 7 --skip-model-load --out-dir "$OUT_DIR"

  # 8 — motion VLM verify
  $PY adhoc/generation/robotarm/verify_motion_component_gemini.py \
    --manifest "$MANIFEST" \
    "${COMMON[@]}" \
    --out-json "$OUT_DIR/exp08_motion_verify_vlm.json" \
    --resume

  # 9 — motion text verify
  $PY adhoc/generation/robotarm/verify_motion_component_text_gemini.py \
    "${COMMON[@]}" \
    --out-json "$OUT_DIR/exp09_motion_verify_text.json" \
    --resume

  # 10 — motion pairwise MP4
  $PY adhoc/generation/robotarm/verify_motion_gt_neg_pairwise_vlm.py \
    "${COMMON[@]}" \
    --out-json "$OUT_DIR/exp10_motion_pairwise_mp4.json" \
    --resume

  # Summary table
  $PY adhoc/generation/robotarm/run_pilot40_qwen_suite.py --only 1,7 --skip-model-load --out-dir "$OUT_DIR" 2>/dev/null || true
  echo "See per-step JSON under $OUT_DIR"
else
  # Default: single Python orchestrator, one model load
  "$PY" "${SUITE_ARGS[@]}"
fi

# Optional: re-generate with Gemini (steps 1 & 7) — requires GOOGLE_API_KEY
if [[ "${RUN_GEMINI_GENERATION:-0}" == "1" ]]; then
  [[ -f APIKEY.sh ]] && source APIKEY.sh
  echo "=== Gemini pose generation (prompt: $PROMPT_POSE) ==="
  $PY adhoc/generation/robotarm/generate_pilot_not_in_shots.py \
    --out_json="$POSE_CFG" --model="${GEMINI_MODEL:-gemini-2.5-pro}"
  echo "=== Gemini motion generation (prompt: $PROMPT_MOTION) ==="
  $PY adhoc/generation/robotarm/generate_motion_from_gt_pose.py \
    --out_json="$MOTION_CFG" --model="${GEMINI_MODEL:-gemini-2.5-pro}"
fi
