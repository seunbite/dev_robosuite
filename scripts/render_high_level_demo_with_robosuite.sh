#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
micromamba run -n robosuite python "$ROOT/adhoc/_legacy_scripts/locomotion/render_high_level_demo.py" "$@"
