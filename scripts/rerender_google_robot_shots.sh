#!/usr/bin/env bash
# Re-render selected google_robot shot GIFs and rebuild render HTML.
# Example:
#   bash scripts/rerender_google_robot_shots.sh circle_temple_crazy
#   bash scripts/rerender_google_robot_shots.sh circle_temple_crazy,bow_greeting
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/adhoc/generation/google_robot:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
CUES="${1:-circle_temple_crazy}"
CFG="${CONFIG_JSON:-data/seed/shots/google_robot/shot_configs_19_mobile.json}"
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python
fi
"$PYTHON" adhoc/generation/google_robot/render.py \
  --config_json="$CFG" \
  --do_html=True \
  --cues="$CUES"

# Tiles + HTML only (skip GIF re-render): add --html_only=True
