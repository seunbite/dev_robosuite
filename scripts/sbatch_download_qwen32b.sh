#!/bin/bash
#SBATCH --job-name=hf-dl-qw32b
#SBATCH --account=dmortens
#SBATCH --qos=preempt_qos
#SBATCH --partition=preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/home/hoyeonk/sblee/dev_robosuite/logs/hf_download_qwen32b_%j.log
#SBATCH --error=/home/hoyeonk/sblee/dev_robosuite/logs/hf_download_qwen32b_%j.log

set -euo pipefail
source /data/user_data/hoyeonk/miniconda3/etc/profile.d/conda.sh
conda activate m2m_caption32b

export HF_HOME=/home/hoyeonk/.cache/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" /home/hoyeonk/sblee/dev_robosuite/logs

echo "host=$(hostname) starting download into $HF_HOME"
which huggingface-cli
huggingface-cli download Qwen/Qwen2.5-VL-32B-Instruct
echo DOWNLOAD_DONE

cd /home/hoyeonk/sblee/dev_robosuite
export PYTHONPATH="$PWD/adhoc/generation/robotarm:${PYTHONPATH:-}"
python - <<'PY'
import os, sys
sys.path.insert(0, "adhoc/generation/robotarm")
from hf_cache_setup import cache_shard_status, setup_hf_cache
setup_hf_cache("/home/hoyeonk/.cache/huggingface")
p, e, s = cache_shard_status("Qwen/Qwen2.5-VL-32B-Instruct")
print(f"shards {p}/{e} snap={s}")
if p != e or e == 0:
    raise SystemExit(f"incomplete cache {p}/{e}")
PY
