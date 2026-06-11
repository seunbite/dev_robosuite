#!/usr/bin/env bash
# Deprecated — use: MODEL_SIZE=gemini ONLY=1,7 bash exp.sh
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ONLY="${ONLY:-1,7}"
MODEL_SIZE=gemini exec "$ROOT/exp.sh"
