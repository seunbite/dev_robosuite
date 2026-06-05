# Pilot gesture experiments (dev_robosuite)

Iconic gesture cue pipeline for **IIWA manipulator**: pose tiles, motion verify, GT-vs-neg compare.

**Remote Qwen-32B setup → [docs/REMOTE_QWEN_EXPERIMENTS.md](docs/REMOTE_QWEN_EXPERIMENTS.md)**

## Quick start (VLM server)

```bash
cp .env.example .env   # set VLM_BASE_URL, VLM_MODEL
pip install -r requirements-vlm.txt
bash scripts/check_vlm_remote.sh
bash scripts/run_qwen_experiments.sh multitile20
```

## Key scripts

| Script | Description |
|--------|-------------|
| `verify_pose_multitile_gt_gemini.py` | Grid 6 / 12 GT tile pick |
| `verify_pose_pairwise_12_gemini.py` | 2-way pose compare |
| `verify_pose_tiles_gemini.py` | Single-tile verify (+ few-shot) |
| `build_pose_multitile_gt_capture_html.py` | Input images + prompts only (no API) |
| `build_pilot40_wrong_answer_notebook_html.py` | Wrong-case review HTML |

## Manifest

- **39 done**: `data/seed/yml/pilot100_manifest.yml` → `completed_pilot40`
- **61 pending**: `pending_new51` + `pending_essence10`

Upstream robosuite docs: [robosuite.ai/docs](https://robosuite.ai/docs/overview.html)
