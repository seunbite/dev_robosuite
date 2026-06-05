# Pilot gesture experiments (dev_robosuite)

Iconic gesture cue pipeline for **IIWA manipulator**: pose tiles, motion verify, GT-vs-neg compare.

**Remote Qwen via vLLM → [docs/REMOTE_QWEN_EXPERIMENTS.md](docs/REMOTE_QWEN_EXPERIMENTS.md)**

## Quick start (salloc + vLLM)

```bash
# Terminal 1 (GPU): inference server
bash scripts/start_vllm_server.sh

# Terminal 2: experiments
cp .env.example .env
bash scripts/check_vlm_remote.sh
bash scripts/run_qwen_experiments.sh multitile20
```

## Key scripts

| Script | Description |
|--------|-------------|
| `scripts/start_vllm_server.sh` | Launch Qwen-VL on vLLM |
| `verify_pose_multitile_gt_gemini.py` | Grid 6 / 12 GT tile pick |
| `verify_pose_pairwise_12_gemini.py` | 2-way pose compare |
| `verify_pose_tiles_gemini.py` | Single-tile verify (+ few-shot) |
| `build_pose_multitile_gt_capture_html.py` | Input images + prompts only (no API) |

## Manifest

- **39 done**: `data/seed/yml/pilot100_manifest.yml` → `completed_pilot40`
- **61 pending**: `pending_new51` + `pending_essence10`

Upstream robosuite docs: [robosuite.ai/docs](https://robosuite.ai/docs/overview.html)
