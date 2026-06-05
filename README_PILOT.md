# Pilot gesture experiments (dev_robosuite)

**Remote Qwen via in-process vLLM → [docs/REMOTE_QWEN_EXPERIMENTS.md](docs/REMOTE_QWEN_EXPERIMENTS.md)**

## sbatch (recommended)

```bash
pip install -r requirements-vlm.txt
mkdir -p logs
EXPERIMENT=multitile20 sbatch scripts/sbatch_pose_vlm.sh
```

## Direct run

```bash
python adhoc/generation/robotarm/run_pose_vlm_eval.py \
  --experiment multitile20 --tensor-parallel-size 2 --resume
```

Results → `data/results/verify/pilot*_vllm_local.json` + accuracy on stdout.
