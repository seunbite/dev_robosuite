#!/usr/bin/env bash
# Deprecated — use: bash exp.sh
exec "$(cd "$(dirname "$0")/.." && pwd)/exp.sh" "$@"
