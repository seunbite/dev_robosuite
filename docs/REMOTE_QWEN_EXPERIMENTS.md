# Remote Qwen experiments (vLLM in-process)

**Recommended:** one Python process, no `vllm serve`, no port config. For sbatch / old driver setups.

```
python run_pose_vlm_eval.py  →  vLLM LLM.load  →  Qwen2.5-VL-32B on GPU
                            →  JSON + accuracy printed
```

## sbatch (babel 등)

```bash
cd dev_robosuite
micromamba activate robosuite-vlm
pip install -r requirements-vlm.txt   # vllm, qwen-vl-utils, transformers

mkdir -p logs
EXPERIMENT=multitile20 sbatch scripts/sbatch_pose_vlm.sh
# EXPERIMENT=all20 | multitile100 | pairwise20
# VLLM_TENSOR_PARALLEL_SIZE=2  (32B OOM 시)
```

로그: `logs/pose_vlm_<jobid>.out`

## 직접 실행 (salloc 1터미널)

```bash
micromamba activate robosuite-vlm
cd dev_robosuite
export PYTHONPATH=$PWD

python adhoc/generation/robotarm/run_pose_vlm_eval.py \
  --experiment multitile20 \
  --tensor-parallel-size 2 \
  --resume
```

출력 JSON:
- `data/results/verify/pilot20_pose_multitile_vllm_local.json`
- `data/results/verify/pilot20_pose_pairwise_vllm_local.json`
- `data/results/verify/pilot100_pose_multitile_vllm_local.json`

마지막에 grid별 / pairwise **accuracy** 출력.

## Experiments

| `--experiment` | 내용 |
|----------------|------|
| `multitile20` | grid 6+12, 20 cues |
| `multitile100` | grid 6+12, 100 cues |
| `pairwise20` | 2-way compare, 20 cues |
| `all20` | pairwise + multitile |

## micromamba (first time)

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc
git clone git@github.com:seunbite/dev_robosuite.git
cd dev_robosuite
micromamba env create -f environment-vlm.yml && micromamba activate robosuite-vlm
pip install -r requirements-vlm.txt
```

---

## Legacy: HTTP vLLM server

Only if you prefer two-terminal `vllm serve`:

```bash
bash scripts/start_vllm_server.sh   # terminal 1
bash scripts/run_qwen_experiments.sh multitile20   # terminal 2
```

---

## Experiment backlog

1. **Multitile 6/12** — `run_pose_vlm_eval.py --experiment multitile20`
2. **Temporal compare** — filter `dynamic_temporal` cues (TODO)
3. **Few-shot baseline** — `verify_pose_tiles_gemini.py --vlm-backend local`
4. **Google Robot ~40** — separate script + assets
5. **Pilot-100 multitile** — `--experiment multitile100`

| Category | N | Done | Undone |
|----------|---:|---:|---:|
| pose_direction | 23 | 12 | 11 |
| pose_location | 22 | 15 | 7 |
| dynamic_direction | 24 | 10 | 14 |
| dynamic_temporal | 21 | 2 | 19 |
| abstract | 10 | 0 | 10 |
