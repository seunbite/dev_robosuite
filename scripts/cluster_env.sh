#!/usr/bin/env bash
# Babel / shared cluster: point HuggingFace cache to /data (home quota is tiny).
# Usage (from repo root):
#   source scripts/cluster_env.sh
#   source scripts/cluster_env.sh /data/user_data/$USER/hf_cache
#
# One-time: move existing ~/.cache/huggingface weights:
#   bash scripts/migrate_hf_cache_to_data.sh
set -euo pipefail

HF_ROOT="${1:-${HF_HOME:-/data/user_data/${USER}/hf_cache}}"
mkdir -p "$HF_ROOT"/{hub,transformers}

export HF_HOME="$HF_ROOT"
export HUGGINGFACE_HUB_CACHE="$HF_ROOT/hub"
export TRANSFORMERS_CACHE="$HF_ROOT/transformers"
# Legacy alias some tools still read
export HF_DATASETS_CACHE="$HF_ROOT/datasets"

echo "HF_HOME=$HF_HOME"
echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"

if command -v quota >/dev/null 2>&1; then
  quota -s 2>/dev/null || true
fi

if [[ -d "$HUGGINGFACE_HUB_CACHE/models--Qwen--Qwen2.5-VL-32B-Instruct" ]]; then
  echo "Qwen2.5-VL-32B cache dir present under hub (may be partial — check size)"
fi

echo ""
echo "If download still fails with 'Disk quota exceeded':"
echo "  1) du -sh ~/.cache/huggingface ~/.cache/pip /data/user_data/$USER/* | sort -h"
echo "  2) rm -rf ~/.cache/huggingface   # after HF_HOME points to /data"
echo "  3) free space under /data/user_data/$USER (quota applies there too)"
echo "  4) use smaller model: VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct"
