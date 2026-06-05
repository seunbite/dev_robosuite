# Remote Qwen pose experiments

## Two separate issues you may hit

| Error | Cause | Fix |
|-------|--------|-----|
| `invalid partition: gpu` | sbatch script had wrong partition name | `sinfo -s` then `sbatch --partition=YOUR_PART ...` |
| `NVIDIA driver too old` | `pip install vllm` pulled **torch+cu124** (vLLM 0.22); driver on node is older | Pin stack: `bash scripts/install_vlm_vllm_cu118.sh` **or** use transformers |
| `CUDA not available` | Running on login node | `salloc --gres=gpu:1` or sbatch first |
| `Disk quota exceeded` | HF model writes to `~/.cache` (home full) or `/data` quota full | `source scripts/cluster_env.sh` + free space (see below) |
| `Cannot activate robosuite_vlm` | Env name is **`robosuite-vlm`** (hyphen), or env not created on this path | See **Conda env** below |

## Disk quota / HuggingFace cache

`OSError: [Errno 122] Disk quota exceeded` = **디스크 할당량 초과**. 드라이버/vLLM 문제가 아님.

32B weights (~60GB+) need space under wherever `HF_HOME` points.

```bash
# 매 세션 (sbatch/salloc 안에서도)
source scripts/cluster_env.sh /data/user_data/$USER/hf_cache

du -sh ~/.cache/huggingface ~/.cache/pip /data/user_data/$USER/* 2>/dev/null | sort -h
quota -s   # if available

# home 캐시가 중복이면 (HF_HOME=/data 설정 후)
rm -rf ~/.cache/huggingface

# 32B 이미 partial download 되어 있으면 같은 HF_HOME에서 이어 받음
ls -lh $HUGGINGFACE_HUB_CACHE/models--Qwen--Qwen2.5-VL-32B-Instruct
```

공간 부족하면: (1) `/data/user_data/hoyeonk/` 아래 큰 폴더 정리, (2) **7B**로 먼저 테스트:
`VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct`

## Conda env

YAML 이름: **`robosuite-vlm`** (underscore `robosuite_vlm` 아님).

```bash
# env가 없을 때 — /data에 만들면 home quota 절약
micromamba env create -f environment-vlm.yml -p /data/user_data/$USER/envs/robosuite-vlm
micromamba activate /data/user_data/$USER/envs/robosuite-vlm

# 이미 m2m_caption32b에 torch/vllm 깔려 있으면 그 env 써도 됨:
conda activate m2m_caption32b
source scripts/cluster_env.sh
```
 (version pin — often what you want)

**vLLM 버전만** 내리면 안 되고, **PyTorch CUDA wheel**도 같이 맞춰야 합니다.

```
pip install vllm          → vLLM 0.22 + torch cu124 → driver too old
install_vlm_vllm_cu118.sh → vLLM 0.8.5 + torch cu118 → 보통 OK
```

```bash
salloc --partition=YOUR_PART --gres=gpu:2 --mem=128G --time=8:00:00
micromamba activate robosuite-vlm
bash scripts/install_vlm_vllm_cu118.sh

BACKEND=vllm VLLM_TENSOR_PARALLEL_SIZE=2 \
  VLM_MODEL=Qwen/Qwen2.5-VL-32B-Instruct \
  python adhoc/generation/robotarm/run_pose_vlm_eval.py \
  --backend vllm --experiment multitile20 --resume
```

Driver가 cu118도 안 받으면 (`CUDA smoke test` 실패) → `CUDA_WHEEL=cu121` 시도, 그래도 안 되면 transformers.

## Quick start (transformers fallback)

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

## vLLM (new driver clusters only)

```bash
pip install "vllm>=0.8.0"   # only if driver supports bundled torch CUDA
python adhoc/generation/robotarm/run_pose_vlm_eval.py --backend vllm ...
```

Old driver → `bash scripts/install_vlm_vllm_cu118.sh` instead of bare `pip install vllm`.
