#!/usr/bin/env bash
# Move existing ~/.cache/huggingface/hub weights to /data (one-time).
set -euo pipefail

DEST="${1:-/data/user_data/${USER}/hf_cache}"
SRC="${HOME}/.cache/huggingface/hub"

source "$(cd "$(dirname "$0")" && pwd)/cluster_env.sh" "$DEST"

if [[ ! -d "$SRC" ]]; then
  echo "No $SRC — nothing to migrate."
  exit 0
fi

echo "Copying $SRC -> $HUGGINGFACE_HUB_CACHE"
echo "(using rsync; safe to re-run)"
rsync -a --info=progress2 "$SRC/" "$HUGGINGFACE_HUB_CACHE/"

echo ""
echo "Done. Verify:"
du -sh "$HUGGINGFACE_HUB_CACHE" "$SRC" 2>/dev/null || true
echo ""
echo "If sizes look OK, free home quota:"
echo "  rm -rf ~/.cache/huggingface"
echo ""
echo "Then always:"
echo "  source scripts/cluster_env.sh"
echo "  # or: export HF_HOME=$HF_HOME"
