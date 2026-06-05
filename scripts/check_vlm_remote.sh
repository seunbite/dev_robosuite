#!/usr/bin/env bash
# Quick env check on a remote GPU box running vLLM + Qwen-VL.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${VLM_BASE_URL:?Set VLM_BASE_URL in .env (e.g. http://127.0.0.1:8000/v1)}"
: "${VLM_MODEL:=Qwen/Qwen2.5-VL-32B-Instruct}"

python - <<'PY'
import os
from openai import OpenAI
c = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "EMPTY"), base_url=os.environ["VLM_BASE_URL"])
r = c.chat.completions.create(
    model=os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"),
    messages=[{"role": "user", "content": "Reply OK only"}],
    temperature=0,
)
print("VLM OK:", (r.choices[0].message.content or "").strip()[:40])
PY

echo "Assets:"
test -d data/results/visualize/pose_groups_12 && echo "  pose_groups_12 OK" || echo "  MISSING pose_groups_12"
test -f data/results/verify/pilot40_pose_eval_consolidated.json && echo "  consolidated OK" || echo "  MISSING consolidated json"
test -f data/seed/yml/pilot100_manifest.yml && echo "  pilot100 manifest OK" || echo "  MISSING manifest"
