#!/bin/bash
# Run 5 representative cues for a given prompt version
# Usage: ./run_representative.sh <version> <python_bin>
set -e

VERSION=$1
PYTHON=$2
CUES=(0 14 21 32 47)
ROBOT="IIWA"
CONFIG="data/seed/motion_configs_v${VERSION}.json"

echo "=== Prompt v${VERSION} | ${#CUES[@]} cues ==="

for idx in "${CUES[@]}"; do
    echo "[v${VERSION}] Running cue_idx=${idx}..."
    $PYTHON adhoc/robotarm/motion_generation.py \
        --robot="$ROBOT" \
        --cue_idx=$idx \
        --config_path="$CONFIG" \
        --jsonl_path="data/seed/closest_poses_results.jsonl" \
        --hz=10 \
        --proximal_degree_scale=0.25 2>&1 | tail -10
    echo "[v${VERSION}] Done cue_idx=${idx}"
done

echo "=== Prompt v${VERSION} COMPLETE ==="
