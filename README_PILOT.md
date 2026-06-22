# Pilot gesture experiments (dev_robosuite)

**Main benchmark:** Pilot-90 — 90 non-essence cues × **10 experiments**, same protocol for **Gemini** and **Qwen2.5-VL** (only the model differs).

---

## Design principles (Pilot-90)

1. **Same experiment for every model** — generation, verify, and scoring use identical prompts, shots, and GT.
2. **Per-model outputs** — each model writes its own config / verify JSONs (no shared motion config for scoring).
3. **Task chain**
   - **Exp 1:** cue → LLM generates **full pose config** (`prompt_exp1`) → score vs `pose_gt`
   - **Exp 2–3:** verify **exp1 config** (VLM tile / text)
   - **Exp 7:** cue + **fixed human pose** → LLM generates **movement tail** (`prompt_exp7`) → score vs `movement_gt`
   - **Exp 8–9:** verify **exp7 config** (VLM MP4 / text)
   - **Exp 4–6, 10:** discriminative VLM benchmarks (pairwise / multitile / motion pairwise)

---

## Directory layout (Pilot-90)

```
data/seed/
  groundtruth/gt_manipulator.json          # 90 cues: pose_gt + movement_gt
  prompt/manipulator/exp/
    prompt_exp1.txt … prompt_exp10.txt     # all experiment prompts
    _shared_*.txt                          # snippets for verify prompts
  shots/manipulator/shot_configs_v19_sophisticated.json   # few-shot (exp1, exp7)

data/results/motion_configs/manipulator/exp/
  result_exp1_{model_tag}.json             # JSON array, 90 motion configs (exp1)
  result_exp7_{model_tag}.json             # JSON array, 90 motion configs (exp7)

data/results/verify/manipulator/exp/
  score_exp1_{model_tag}.json              # exp1 accuracy
  score_exp7_{model_tag}.json              # exp7 accuracy
  result_exp2_{model_tag}.json … result_exp10_{model_tag}.json
  pilot90_suite_summary_{model_tag}.json

data/results/html/
  exp1_{model_tag}.html … exp10_{model_tag}.html   # per-task review tables
```

`{model_tag}` examples: `gemini-2.5-pro`, `qwen32b`, `qwen7b`, `qwen3b`

Regenerate GT after pose/motion annotation changes:

```bash
python adhoc/generation/robotarm/build_gt_manipulator.py
```

---

## Experiment table (Pilot-90)

| # | Task | Prompt | Input config | JSON result | HTML review |
|---|------|--------|--------------|-------------|-------------|
| 1 | Pose generation vs GT | `prompt_exp1.txt` | — | `verify/.../score_exp1_{tag}.json` | `html/exp1_{tag}.html` |
| 2 | Pose verify VLM | `prompt_exp2.txt` | `result_exp1_{tag}.json` | `result_exp2_{tag}.json` | `html/exp2_{tag}.html` |
| 3 | Pose verify text | `prompt_exp3.txt` | `result_exp1_{tag}.json` | `result_exp3_{tag}.json` | `html/exp3_{tag}.html` |
| 4 | Pose pairwise 2-way | `prompt_exp4.txt` | tiles + GT | `result_exp4_{tag}.json` | `html/exp4_{tag}.html` |
| 5 | Multitile grid 6 | `prompt_exp5.txt` | multitile + GT | `result_exp5_{tag}.json` | `html/exp5_{tag}.html` |
| 6 | Multitile grid 12 | `prompt_exp6.txt` | multitile + GT | `result_exp6_{tag}.json` | `html/exp6_{tag}.html` |
| 7 | Movement generation vs GT | `prompt_exp7.txt` | GT pose fixed | `score_exp7_{tag}.json` | `html/exp7_{tag}.html` |
| 8 | Movement verify VLM | `prompt_exp8.txt` | `result_exp7_{tag}.json` + MP4 | `result_exp8_{tag}.json` | `html/exp8_{tag}.html` |
| 9 | Movement verify text | `prompt_exp9.txt` | `result_exp7_{tag}.json` | `result_exp9_{tag}.json` | `html/exp9_{tag}.html` |
| 10 | Motion pairwise MP4 | `prompt_exp10.txt` | `result_exp7_{tag}.json` + pairwise MP4 | `result_exp10_{tag}.json` | `html/exp10_{tag}.html` |

JSON paths are under `data/results/verify/manipulator/exp/` (scores + verify). HTML paths are under `data/results/html/`.  
`{tag}` examples: `gemini-2.5-pro`, `qwen32b`, `qwen7b`, `qwen3b`. Run via `bash exp.sh` or `exp.py`.

**Orchestrator:** `adhoc/generation/robotarm/exp.py` — `exp.py 1`, `exp.py all`, `exp.py all --summary`  
**Code registry:** `adhoc/generation/robotarm/pilot90_experiment_suite.py`  
**Path helpers:** `adhoc/generation/robotarm/pilot90_paths.py`  
**Unified LLM generation:** `adhoc/generation/robotarm/config_gen_vlm.py` (exp7 rows include `groundtruth` = human pose GT)

---

## Server commands (cluster)

```bash
git pull
salloc --partition=YOUR_PART --gres=gpu:2 --mem=128G --time=24:00:00
conda activate m2m_caption32b   # or robosuite-vlm
cd dev_robosuite
source scripts/cluster_env.sh /data/user_data/$USER/hf_cache
```

### Where to run

| Machine | Command |
|---------|---------|
| **Server (cluster)** | `cd ~/sblee/dev_robosuite && bash exp.sh` — manipulator, all 10 tasks |
| **Server (cluster)** | `DOMAIN=google_robot MODEL_SIZE=gemini bash exp.sh all` — mobile bimanipulator |
| **Local (Mac)** | `cd ~/Downloads/workspace/dev_robosuite && ONLY=1,2,3,7 bash exp.sh` |

### `exp.sh` — one entry point (interactive + sbatch)

`exp.sh` is the only orchestrator for pilot-90 / pilot-40 Google Robot. It works in:

- **interactive**: `bash exp.sh` or `ONLY=2,3 bash exp.sh`
- **sbatch**: `sbg --export=ALL,MODEL_SIZE=7b,ONLY=8,9 exp.sh` (partition/GPU/mem from your `sbg`/`sbg2` alias)

`#SBATCH` headers at the top of `exp.sh` set log paths (`logs/exp_%j.out`). Override partition/time/gpus when submitting.

| Site | Path | Env |
|------|------|-----|
| cluster (auto) | `~/sblee/dev_robosuite` | `y/envs/robosuite-vlm` via `activate_cluster_vlm.sh` |
| local (auto) | `~/Downloads/workspace/dev_robosuite` | `micromamba activate robosuite` |

Override: `SKIP_ENV=1` (already activated), `SKIP_GIT_PULL=1` (no `git pull` on cluster).

```bash
bash exp.sh                              # manipulator full suite (32b)
DOMAIN=google_robot MODEL_SIZE=gemini bash exp.sh all
SUMMARY=1 bash exp.sh                    # Qwen 32B/7B/3B cross-model tables
ONLY=1,2,3 bash exp.sh                   # subset
MODEL_SIZE=7b ONLY=8,9 bash exp.sh
GENERATE=0 ONLY=2,3 bash exp.sh          # verify only (needs prior exp1/7 configs)
ALL_MODELS=1 SUMMARY=1 bash exp.sh       # 32b → 7b → 3b, then summary

# sbatch (cluster)
sbg  --export=ALL,MODEL_SIZE=7b,ONLY=8,9,MOTION_PREPARE_MP4=0 exp.sh
sbg2 --export=ALL,MODEL_SIZE=32b,ONLY=4,5,6,10,MOTION_PREPARE_PAIRWISE=0 exp.sh
DOMAIN=google_robot sbg2 --export=ALL,MODEL_SIZE=32b exp.sh
```

Motion media (exp8, exp10) — run once before verify if needed:

```bash
bash scripts/prepare_pilot90_motion_mp4.sh
bash scripts/prepare_pilot90_motion_pairwise_mp4.sh
ONLY=8,9,10 bash exp.sh
```

---

## Pilot-40 Google Robot (39 cues, mobile bimanipulator)

Same **exp 1–10 protocol** as manipulator pilot-90, adapted to TIAGo pose/movement/path schema.

### Run (server)

```bash
cd ~/sblee/dev_robosuite
source APIKEY.sh                    # Gemini
DOMAIN=google_robot MODEL_SIZE=gemini bash exp.sh all
ONLY=1,2,3,7 bash exp.sh all        # subset (DOMAIN still google_robot)
SUMMARY=1 DOMAIN=google_robot bash exp.sh all
```

Local Mac:

```bash
cd ~/Downloads/workspace/dev_robosuite
DOMAIN=google_robot MODEL_SIZE=gemini ONLY=1 bash exp.sh all
```

**Orchestrator:** `adhoc/generation/google_robot/exp.py`  
**Generation:** `adhoc/generation/google_robot/config_gen_mobile_vlm.py` (exp 1 & 7)  
**Path helpers:** `adhoc/generation/google_robot/pilot40_paths.py`  
**Prompts:** `data/seed/prompt/google_robot/exp/prompt_exp{1..10}.txt` + `_shared_*.txt`

### Directory layout (symmetric to manipulator)

```
data/seed/
  shots/google_robot/shot_configs_pilot40_mobile.json   # 39 cue list + few-shot source
  shots/google_robot/diverse_shots_mobile.json          # few-shot examples (exp1)
  prompt/google_robot/exp/prompt_exp1.txt … prompt_exp10.txt
  prompt/google_robot/exp/_shared_appropriate_means.txt
  prompt/google_robot/exp/_shared_pose_definitions.txt
  prompt/google_robot/exp/_shared_representative_means.txt
  yml/_cues_new.yml                                     # cue catalog (exp1)

data/results/motion_configs/google_robot/exp/
  result_exp1_{model_tag}.json
  result_exp7_{model_tag}.json

data/results/verify/google_robot/exp/
  score_exp1_{model_tag}.json
  score_exp7_{model_tag}.json
  result_exp2_{model_tag}.json … result_exp10_{model_tag}.json
  pilot40_suite_summary_{model_tag}.json

data/results/render/google_robot/
  mm19_g*.gif                                           # rendered motions
  pilot40_media/{pose,mp4}/                             # exp2 PNG + exp8 MP4

data/results/visualize/
  google_pose_groups_12/                                # pose tiles (exp5/6)
  google_robot/pose_multitile_gt_pilot40_grid{6,12}/    # stitched grids

data/results/html/google_robot/
  exp{N}_{model_tag}.html
  index.html
```

`{model_tag}` examples: `gemini-2.5-pro`, `mobile-map` (legacy migrated configs)

### Experiment table (Pilot-40 Google Robot)

| # | Task | Prompt | Input | JSON result | HTML |
|---|------|--------|-------|-------------|------|
| 1 | Pose generation vs GT | `prompt_exp1.txt` | — | `score_exp1_{tag}.json` | `exp1_{tag}.html` |
| 2 | Pose verify VLM (PNG) | `prompt_exp2.txt` | `result_exp1_{tag}.json` | `result_exp2_{tag}.json` | `exp2_{tag}.html` |
| 3 | Pose verify text | `prompt_exp3.txt` | `result_exp1_{tag}.json` | `result_exp3_{tag}.json` | `exp3_{tag}.html` |
| 4 | Pose pairwise 2-way | `prompt_exp4.txt` | exp1 vs reference GIFs | `result_exp4_{tag}.json` | — |
| 5 | Multitile grid 6 | `prompt_exp5.txt` | pose tiles + GT | `result_exp5_{tag}.json` | `exp5_{tag}.html` |
| 6 | Multitile grid 12 | `prompt_exp6.txt` | pose tiles + GT | `result_exp6_{tag}.json` | — |
| 7 | Movement generation vs GT | `prompt_exp7.txt` | fixed human pose | `score_exp7_{tag}.json` | `exp7_{tag}.html` |
| 8 | Movement verify VLM (MP4) | `prompt_exp8.txt` | `result_exp7_{tag}.json` + MP4 | `result_exp8_{tag}.json` | `exp8_{tag}.html` |
| 9 | Movement verify text | `prompt_exp9.txt` | `result_exp7_{tag}.json` | `result_exp9_{tag}.json` | `exp9_{tag}.html` |
| 10 | Movement pairwise | `prompt_exp10.txt` | `result_exp7_{tag}.json` + GIF/MP4 | `result_exp10_{tag}.json` | — |

### Media prep (exp 2, 8, 10)

```bash
# After rendering GIFs under data/results/render/google_robot/
python adhoc/generation/google_robot/prepare_pilot40_media.py \
  --config-json data/results/motion_configs/google_robot/exp/result_exp7_gemini-2.5-pro.json

# Pose tiles for exp 5/6 (once)
python adhoc/generation/google_robot/generate_pose_group_tiles.py \
  --output-root data/results/visualize/google_pose_groups_12
```

### Legacy migrate (optional)

```bash
python adhoc/generation/google_robot/exp.py migrate --force
python adhoc/generation/google_robot/exp.py summary
```

---

## Pilot-40 (legacy manipulator, 39 cues)

Older 39-cue suite; paths and shared-config scoring remain in `pilot40_experiment_suite.py` / `run_pilot40_qwen_suite.sh`. Prefer Pilot-90 for new work.

---

## Cluster troubleshooting

| Error | Fix |
|-------|-----|
| `Disk quota exceeded` | `source scripts/cluster_env.sh /data/user_data/$USER/hf_cache` |
| `CUDA not available` | `salloc` with `--gres=gpu` |
| Step 8 `no mp4` | sync `run/IIWA/*.gif`, then `prepare_pilot90_motion_mp4.sh` |
| Step 10 `EGL_BAD_DISPLAY` | GPU **compute** node 필요 (login node X). `export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0` 후 `prepare_pilot90_motion_pairwise_mp4.sh`. 당장은 `ONLY=1,2,3,4,5,6,7,8,9 bash exp.sh` |
| Step 10 missing MP4 | `prepare_pilot90_motion_pairwise_mp4.sh` 또는 rsync |
| Gemini `RESOURCE_EXHAUSTED` | quota / billing cap; resume with `RESUME=1` |

---

## Paper figures

```bash
bash scripts/build_paper_figures.sh acc
```

Outputs: `data/results/paper_figures/`

---

## Files safe to delete (cleanup)

After migrating to per-model `result_exp*` paths, these are **candidates for removal** (archive first if unsure):

**Deprecated merged / shared configs**
- `data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json`
- `adhoc/generation/robotarm/run_pilot90_gemini_pose_generation.py` (superseded by `run_pilot90_exp_generation.py`)

**Old Qwen verify trees** (shared-config era; scores not per-model valid)
- `data/results/verify/pilot90_qwen32b/`
- `data/results/verify/pilot90_qwen7b/`
- `data/results/verify/pilot90_qwen3b/`

**Old Gemini partial runs** (mixed protocols)
- `data/results/verify/pilot90_gemini/exp01_pose_generation_score.json` (rescored shared config, not per-model generation)
- `data/results/verify/pilot90_gemini/pose_generation_checkpoint.json`

**Duplicate prompts** (copied to `prompt/manipulator/exp/`; keep one copy)
- `data/seed/prompt/pilot40/exp02_*.txt` … `exp10_*.txt` — optional once all code uses `prompt_loader` aliases

**Stale scoring / paper artifacts** (regenerate after new runs)
- `data/results/paper_figures/pilot90_acc_table.json` (manual table from wrong protocol)
- `data/results/verify/pilot40_pose_eval_consolidated_scored.tsv` (pilot-40 only)

**Temp / debug**
- `data/results/verify/_tmp_*.json`
- `data/results/verify/exp10_debug_sample/`

**Do not delete**
- `data/seed/groundtruth/gt_manipulator.json`
- `data/seed/prompt/manipulator/exp/prompt_exp*.txt`
- `data/seed/shots/manipulator/shot_configs_v19_sophisticated.json`
- `data/results/visualize/pose_*_pilot90/`
- `run/IIWA/` GIFs, pairwise MP4 prep outputs
