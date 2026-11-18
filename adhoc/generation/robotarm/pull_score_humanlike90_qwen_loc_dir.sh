#!/usr/bin/env bash
# Pull Babel Qwen humanlike90 Motion Codes and score G1 loc/dir/move vs GT.
# Tiago scoring is skipped until Tiago GT is ready.
#
#   TAG=qwen32b_humanlike90 bash adhoc/generation/robotarm/pull_score_humanlike90_qwen_loc_dir.sh
#   TAG=qwen7b_humanlike90  bash adhoc/generation/robotarm/pull_score_humanlike90_qwen_loc_dir.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
REMOTE="${REMOTE:-hoyeonk@login2.babel.cs.cmu.edu}"
REMOTE_DIR="${REMOTE_DIR:-/home/hoyeonk/sblee/dev_robosuite}"
TAG="${TAG:-qwen32b_humanlike90}"
LOCAL="data/results/motion_configs/embodiment_motion_code_qwen_${TAG}"
REMOTE_OUT="${REMOTE_DIR}/data/results/motion_configs/embodiment_motion_code_qwen_${TAG}"

# Prefer synth60 GT for cues 31–91; fall back to paper30 GT for shared cues.
GT_SYNTH="data/results/motion_configs/embodiment_motion_code_synth60_gt/embodiment_motion_code_all.json"
GT_PAPER="data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
GT_MERGED="data/results/motion_configs/embodiment_motion_code_gt_humanlike90/embodiment_motion_code_all.json"

mkdir -p "$LOCAL"
rsync -avz "${REMOTE}:${REMOTE_OUT}/" "$LOCAL/"

python3 - << PY
import json
from pathlib import Path
repo = Path(".")
synth = json.loads((repo / "${GT_SYNTH}").read_text())
paper = json.loads((repo / "${GT_PAPER}").read_text())
by = {e["cue"]: e for e in paper}
for e in synth:
    by[e["cue"]] = e  # synth overrides / adds
out = repo / "${GT_MERGED}"
out.parent.mkdir(parents=True, exist_ok=True)
rows = sorted(by.values(), key=lambda e: int(e.get("idx") or 0))
out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
print(f"merged GT n={len(rows)} -> {out}")
PY

PRED="$LOCAL/embodiment_motion_code_all.json"
if [[ ! -f "$PRED" ]]; then
  echo "Missing $PRED — generation not finished?"
  exit 1
fi

python papers/icra2027_motion_abstraction/score_humanoid_loc_dir_move.py \
  --gt "$GT_MERGED" \
  --pred "$PRED" \
  --out-json "data/results/evaluation/humanoid_gt_vs_${TAG}_loc_dir_move.json" \
  --out-md "papers/icra2027_motion_abstraction/humanoid_gt_vs_${TAG}_loc_dir_move.md"

echo "Done. See papers/icra2027_motion_abstraction/humanoid_gt_vs_${TAG}_loc_dir_move.md"
echo "Note: Tiago platform is generated but not scored until Tiago GT is ready."
