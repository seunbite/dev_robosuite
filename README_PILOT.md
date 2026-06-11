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
| **Server (cluster)** | `cd ~/sblee/dev_robosuite && bash exp.sh` — all 10 tasks, full rerun |
| **Local (Mac)** | `cd ~/Downloads/workspace/dev_robosuite && ONLY=1,2,3,7 bash exp.sh` |

### `exp.sh` — one entry point

Defaults: **all 10 tasks**, **Qwen 32B**, resume on.  
`exp.sh` picks repo + Python env automatically:

| Site | Path | Env |
|------|------|-----|
| cluster (auto) | `~/sblee/dev_robosuite` | `conda activate m2m_caption32b` |
| local (auto) | `~/Downloads/workspace/dev_robosuite` | `micromamba activate robosuite` |

Override: `EXP_SITE=local|cluster`, `SKIP_ENV=1` (already activated).

```bash
bash exp.sh                              # full suite (32b)
SUMMARY=1 bash exp.sh                    # scores only
ONLY=1,2,3 bash exp.sh                   # subset
MODEL_SIZE=gemini bash exp.sh            # Gemini API (source APIKEY.sh)
MODEL_SIZE=7b bash exp.sh
GENERATE=0 ONLY=2,3 bash exp.sh          # verify only (needs prior exp1/7 configs)
ALL_MODELS=1 bash exp.sh                 # 32b → 7b → 3b, then summary
```

Motion media (exp8, exp10) — run once before verify if needed:

```bash
bash scripts/prepare_pilot90_motion_mp4.sh
bash scripts/prepare_pilot90_motion_pairwise_mp4.sh
ONLY=8,9,10 bash exp.sh
```

### Direct Python CLI (optional)

```bash
python adhoc/generation/robotarm/exp.py all --summary
python adhoc/generation/robotarm/generate_all.py --backend gemini
python adhoc/generation/robotarm/generate_only_move.py --backend gemini --resume
```

### Motion media prep (exp8, exp10)

```bash
bash scripts/prepare_pilot90_motion_mp4.sh
bash scripts/prepare_pilot90_motion_pairwise_mp4.sh
ONLY=8,9,10 bash scripts/run_pilot90_qwen_suite.sh
```

---

## Pilot-40 (legacy, 39 cues)

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
