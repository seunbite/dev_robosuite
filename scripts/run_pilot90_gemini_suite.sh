#!/usr/bin/env bash
# Deprecated — use: MODEL_SIZE=gemini bash exp.sh
MODEL_SIZE=gemini exec "$(cd "$(dirname "$0")/.." && pwd)/exp.sh" "$@"
