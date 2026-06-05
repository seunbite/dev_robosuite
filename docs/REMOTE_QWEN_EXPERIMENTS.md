# Remote Qwen-32B experiments (vLLM)

Pilot gesture VLM experiments use **vLLM inference only** — not HuggingFace `transformers` in-process loading.

```
실험 스크립트  ──HTTP──►  vLLM serve (GPU)  ──►  Qwen2.5-VL-32B
              VLM_BASE_URL     port 8000
```

## 0. micromamba (first time)

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc   # or ~/.zshrc
```

## 1. Clone & env

```bash
git clone git@github.com:seunbite/dev_robosuite.git
cd dev_robosuite
micromamba env create -f environment-vlm.yml && micromamba activate robosuite-vlm
cp .env.example .env
```

## 2. salloc + vLLM (GPU node, terminal 1)

```bash
salloc ...                    # GPU 노드 진입
micromamba activate robosuite-vlm
cd dev_robosuite
tmux new -s vllm

bash scripts/start_vllm_server.sh
# → http://0.0.0.0:8000/v1  (모델 로딩 완료까지 대기)
```

Optional `.env` overrides for the server script:

```
VLLM_PORT=8000
VLLM_TENSOR_PARALLEL_SIZE=2   # 32B OOM 시
VLLM_MAX_MODEL_LEN=8192
```

`.env` client side (same node):

```
VLM_BACKEND=vllm
VLM_BASE_URL=http://127.0.0.1:8000/v1
VLM_MODEL=Qwen/Qwen2.5-VL-32B-Instruct
OPENAI_API_KEY=EMPTY
```

Port is arbitrary — if you use `--port 9000`, set `VLM_BASE_URL=...9000/v1` to match.

## 3. Preflight (terminal 2, same salloc session)

```bash
micromamba activate robosuite-vlm
cd dev_robosuite
bash scripts/check_vlm_remote.sh
```

## 4. Run experiments

```bash
bash scripts/run_qwen_experiments.sh multitile20
bash scripts/run_qwen_experiments.sh pairwise20
bash scripts/run_qwen_experiments.sh fewshot20
bash scripts/run_qwen_experiments.sh multitile100
```

Direct:

```bash
python adhoc/generation/robotarm/verify_pose_multitile_gt_gemini.py \
  --max-cues 20 --grid-sizes 6,12 \
  --out-json data/results/verify/pilot20_pose_multitile_qwen.json --resume
```

(`--vlm-backend` defaults to `vllm`; server check runs automatically.)

Capture-only HTML (no vLLM):

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

`verify_pose_tiles_gemini.py` with shots from `data/seed/shots/manipulator/`.

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

Backend: `adhoc/generation/robotarm/vlm_client.py` — default `VLM_BACKEND=vllm`.
