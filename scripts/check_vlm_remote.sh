#!/usr/bin/env bash
# Verify vLLM server + bundled experiment assets.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export VLM_BACKEND="${VLM_BACKEND:-vllm}"
: "${VLM_BASE_URL:?Set VLM_BASE_URL in .env (e.g. http://127.0.0.1:8000/v1)}"
: "${VLM_MODEL:=Qwen/Qwen2.5-VL-32B-Instruct}"

echo "Checking vLLM at $VLM_BASE_URL ..."
if ! curl -sf "${VLM_BASE_URL%/}/models" | head -c 200; then
  echo ""
  echo "ERROR: vLLM not running at $VLM_BASE_URL"
  echo "  1) salloc GPU node"
  echo "  2) tmux → bash scripts/start_vllm_server.sh"
  echo "  3) re-run this script"
  exit 1
fi
echo ""
echo ""

python - <<'PY'
import os
from openai import OpenAI
from adhoc.generation.robotarm.vlm_client import require_vllm_server

require_vllm_server()
c = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "EMPTY"), base_url=os.environ["VLM_BASE_URL"])
r = c.chat.completions.create(
    model=os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"),
    messages=[{"role": "user", "content": "Reply OK only"}],
    temperature=0,
)
print("VLM chat OK:", (r.choices[0].message.content or "").strip()[:40])
PY

echo "Assets:"
test -d data/results/visualize/pose_groups_12 && echo "  pose_groups_12 OK" || echo "  MISSING pose_groups_12"
test -f data/results/verify/pilot40_pose_eval_consolidated.json && echo "  consolidated OK" || echo "  MISSING consolidated json"
test -f data/seed/yml/pilot100_manifest.yml && echo "  pilot100 manifest OK" || echo "  MISSING manifest"
