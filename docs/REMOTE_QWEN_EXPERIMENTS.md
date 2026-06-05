# Remote Qwen pose experiments

## Two separate issues you may hit

| Error | Cause | Fix |
|-------|--------|-----|
| `invalid partition: gpu` | sbatch script had wrong partition name | `sinfo -s` then `sbatch --partition=YOUR_PART ...` |
| `NVIDIA driver too old` | vLLM 0.22 + latest torch need newer driver than Babel node | Use **transformers** backend + `install_vlm_transformers.sh` |
| `CUDA not available` | Running on login node | `salloc --gres=gpu:1` or sbatch first |

## Quick start (Babel / old driver)

```bash
git pull
micromamba activate robosuite-vlm

# GPU node (salloc) — NOT login node
salloc --partition=YOUR_PART --gres=gpu:1 --mem=64G --time=4:00:00

bash scripts/install_vlm_transformers.sh   # torch cu118 + transformers, no vllm

python adhoc/generation/robotarm/run_pose_vlm_eval.py \
  --backend transformers \
  --experiment multitile20 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --resume
```

32B (2 GPUs):

```bash
salloc --partition=YOUR_PART --gres=gpu:2 --mem=128G --time=8:00:00
VLM_MODEL=Qwen/Qwen2.5-VL-32B-Instruct \
  python adhoc/generation/robotarm/run_pose_vlm_eval.py \
  --backend transformers --experiment multitile20 --resume
```

## sbatch

```bash
sinfo -s   # find partition name

mkdir -p logs
sbatch --partition=YOUR_PART --gres=gpu:1 scripts/sbatch_pose_vlm.sh

# 32B
sbatch --partition=YOUR_PART --gres=gpu:2 \
  --export=ALL,VLM_MODEL=Qwen/Qwen2.5-VL-32B-Instruct \
  scripts/sbatch_pose_vlm.sh
```

Default: `BACKEND=transformers`, `VLM_MODEL=7B`. Logs: `logs/pose_vlm_<jobid>.out`

## Output

- `data/results/verify/pilot20_pose_multitile_hf_local.json`
- `data/results/verify/pilot20_pose_pairwise_hf_local.json`
- stdout: grid / pairwise **accuracy**

## vLLM (only if driver is new enough)

```bash
pip install vllm>=0.8.0
python adhoc/generation/robotarm/run_pose_vlm_eval.py --backend vllm --tensor-parallel-size 2 ...
```

If you see `driver too old` with vLLM, stay on `--backend transformers`.
