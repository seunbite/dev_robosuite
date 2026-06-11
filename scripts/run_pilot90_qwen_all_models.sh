#!/usr/bin/env bash
# Deprecated — use: ALL_MODELS=1 bash exp.sh
ALL_MODELS=1 exec "$(cd "$(dirname "$0")/.." && pwd)/exp.sh" "$@"
