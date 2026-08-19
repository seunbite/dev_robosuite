#!/usr/bin/env bash
#SBATCH --job-name=find-hf-cache
#SBATCH --output=logs/find_hf_cache_%j.log
#SBATCH --error=logs/find_hf_cache_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --export=ALL

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

echo "HOST=$(hostname)"
echo "=== /data ==="
ls -la /data 2>&1 | head -5 || true
echo "=== /data/user_data users ==="
ls /data/user_data 2>&1 | head -40 || true
echo "=== hf_cache sizes ==="
du -sh /data/user_data/*/hf_cache 2>/dev/null | sort -h || true
echo "=== Qwen 32B model dirs ==="
find /data/user_data -maxdepth 7 -type d -name 'models--Qwen--Qwen2.5-VL-32B-Instruct' 2>/dev/null || true
find /data/user_data -maxdepth 7 -type d -name 'models--Qwen--Qwen2.5-VL-7B-Instruct' 2>/dev/null || true
echo "=== hoykim ==="
ls -la /data/user_data/hoykim 2>&1 | head -15 || true
du -sh /data/user_data/hoykim/* 2>/dev/null | sort -h | tail -15 || true
echo "=== hoyeonk ==="
ls -la /data/user_data/hoyeonk 2>/dev/null | head -15 || true
echo "=== home hoyeonk hf ==="
du -sh "$HOME/.cache/huggingface" 2>/dev/null || true
ls "$HOME/.cache/huggingface/hub" 2>/dev/null | grep -i qwen || true
echo DONE
