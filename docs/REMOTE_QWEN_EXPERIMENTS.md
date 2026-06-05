# Remote Qwen-32B experiments

See setup, commands, and experiment backlog for the pilot gesture pipeline.

## Clone & env

```bash
git clone git@github.com:seunbite/dev_robosuite.git
cd dev_robosuite
micromamba env create -f environment-vlm.yml && micromamba activate robosuite-vlm
cp .env.example .env   # VLM_BASE_URL, VLM_MODEL
bash scripts/check_vlm_remote.sh
```

## vLLM example

```bash
vllm serve Qwen/Qwen2.5-VL-32B-Instruct --host 0.0.0.0 --port 8000 --max-model-len 8192
```

`.env`:

```
VLM_BACKEND=openai
VLM_BASE_URL=http://127.0.0.1:8000/v1
VLM_MODEL=Qwen/Qwen2.5-VL-32B-Instruct
OPENAI_API_KEY=EMPTY
```

## Run

```bash
bash scripts/run_qwen_experiments.sh multitile20
bash scripts/run_qwen_experiments.sh pairwise20
bash scripts/run_qwen_experiments.sh multitile100
```

Direct:

```bash
python adhoc/generation/robotarm/verify_pose_multitile_gt_gemini.py \
  --max-cues 20 --grid-sizes 6,12 --vlm-backend openai \
  --out-json data/results/verify/pilot20_pose_multitile_qwen.json --resume
```

Capture-only HTML (no API):

```bash
python adhoc/generation/robotarm/build_pose_multitile_gt_capture_html.py
```

## Bundled assets (git)

- `data/seed/yml/pilot100_manifest.yml`
- `data/seed/shots/`, `data/seed/prompt/`
- `data/results/motion_configs/manipulator/`
- `data/results/visualize/pose_groups_12/`, `pose_multitile_gt/`
- `data/results/verify/` (consolidated labels)

Not in git: `data/results/render/` (large GIFs) — rsync separately if needed.

---

## Experiment backlog

### 1. Multitile pose compare (6 & 12)

Pick human GT tile among shuffled grids. Scripts: `verify_pose_multitile_gt_gemini.py`. Random baseline: 16.7% (6), 8.3% (12).

### 2. Temporal category compare

Run compare on `dynamic_temporal` cues (21); optional tempo-aware prompt edits.

### 3. Few-shot baseline

`verify_pose_tiles_gemini.py` with shots from `data/seed/shots/manipulator/` — verify without pairwise.

### 4. Google Robot × ~40 cues

`adhoc/generation/google_robot/compare_components_vlm_gemini.py` (+ render assets).

### 5. Full pilot-100 multitile

200 calls (100 × grid6 × grid12). Analyze failures by category.

| Category | N | Done (39) | Undone (61) |
|----------|---:|---:|---:|
| pose_direction | 23 | 12 | 11 |
| pose_location | 22 | 15 | 7 |
| dynamic_direction | 24 | 10 | 14 |
| dynamic_temporal | 21 | 2 | 19 |
| abstract | 10 | 0 | 10 |

Backend: `adhoc/generation/robotarm/vlm_client.py` (`--vlm-backend openai` for Qwen/vLLM).
