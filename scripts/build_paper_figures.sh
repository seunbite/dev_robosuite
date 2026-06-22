#!/usr/bin/env bash
# Paper figures for pilot-90 benchmark (tasks 3–8)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
CLI="$ROOT/adhoc/generation/robotarm/paper_figures/cli.py"

case "${1:-help}" in
  acc)
    shift
    "$PY" "$CLI" acc "$@"
    ;;
  qual-pose)
    shift 2>/dev/null || true
    "$PY" "$CLI" qual --mode pose --idx "${IDX:-all}" "$@"
    ;;
  qual-movement)
    shift 2>/dev/null || true
    # Movement examples: only cues with correct pose generation (human GT, any-pose rule).
    "$PY" "$CLI" qual --mode movement --idx "${IDX:-all}" "$@"
    ;;
  pairwise)
    shift
    "$PY" "$CLI" pairwise --idx "${IDX:-0,1,0,1}" "$@"
    ;;
  qual-verify-pose)
    shift 2>/dev/null || true
    ARGS=(--mode pose --idx "${IDX:-sel}")
    [[ -n "${GOOD_IDX:-}" ]] && ARGS+=(--good-idx "$GOOD_IDX")
    [[ -n "${BAD_IDX:-}" ]] && ARGS+=(--bad-idx "$BAD_IDX")
    [[ -n "${SEED:-}" ]] && ARGS+=(--seed "$SEED")
    "$PY" "$CLI" qual-verify "${ARGS[@]}" "$@"
    ;;
  qual-verify-motion)
    shift 2>/dev/null || true
    ARGS=(--mode motion --idx "${IDX:-sel}")
    [[ -n "${GOOD_IDX:-}" ]] && ARGS+=(--good-idx "$GOOD_IDX")
    [[ -n "${BAD_IDX:-}" ]] && ARGS+=(--bad-idx "$BAD_IDX")
    [[ -n "${SEED:-}" ]] && ARGS+=(--seed "$SEED")
    "$PY" "$CLI" qual-verify "${ARGS[@]}" "$@"
    ;;
  components)
    shift
    "$PY" "$CLI" components "$@"
    ;;
  components-all)
    shift 2>/dev/null || true
    "$PY" "$CLI" components --all-embodiments "$@"
    ;;
  persona)
    shift
    "$PY" "$CLI" persona "$@"
    ;;
  essence10)
    shift
    "$PY" "$CLI" essence10 "$@"
    ;;
  all-static)
    "$PY" "$CLI" acc
    "$PY" "$CLI" qual --mode pose --idx all
    "$PY" "$CLI" qual --mode movement --idx all
    "$PY" "$CLI" pairwise --idx "${IDX:-0,1,0,1}"
    "$PY" "$CLI" components
    ;;
  help|*)
    cat <<'EOF'
Usage: bash scripts/build_paper_figures.sh <command> [args]

  acc              Line plots (pose tasks 1-3, movement 7-9, 4 models)
  qual-pose        Pose qualitative grid  (--idx all | 1,5,8,...)
  qual-movement    Movement alpha-traj grid (pose-GT-correct cues only)
  pairwise         8-panel comparison row
  qual-verify-pose    Pose pairwise qual (IDX=sel, SEED=0, GOOD_IDX/BAD_IDX optional)
  qual-verify-motion  Motion pairwise qual (task 10, 4 cols)
  components       Pose / movement / path definition figure (--embodiment IIWA|Panda|GoogleRobot|Quadruped)
  components-all   All four embodiment component figures
  persona          Persona variations (needs GOOGLE_API_KEY)
  essence10        Essence-10 generate + HTML (needs GOOGLE_API_KEY)
  all-static       acc + qual grids + pairwise + components (no API)

Env:
  IDX=1,5,8,2,15,28   for qual-* and pairwise selection
EOF
    ;;
esac
