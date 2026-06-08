# Pilot gesture experiments (dev_robosuite)

**Main benchmark:** 39 iconic cues × **10 experiments** (pose + motion), evaluated with Gemini baselines and Qwen2.5-VL reruns.

| Item | Location |
|------|----------|
| Experiment registry (code) | `adhoc/generation/robotarm/pilot40_experiment_suite.py` |
| Suite runner | `scripts/run_pilot40_qwen_suite.sh` → `run_pilot40_qwen_suite.py` |
| Verify / eval prompts | `data/seed/prompt/pilot40/` |
| Generation prompts | `data/seed/prompt/manipulator/` |
| Gemini results | `data/results/verify/pilot40_*` and `pose_*_gemini.json` |
| Qwen results | `data/results/verify/pilot40_qwen32b/` (or `pilot40_qwen7b` via `OUT_DIR`) |
| Planned backlog | `data/seed/experiments/planned_backlog.yml` |

---

## Directory layout

```
data/seed/prompt/
  manipulator/          # LLM generation prompts (steps 1, 7)
  pilot40/              # VLM verify prompts (steps 2–6, 8–10) + shared snippets

data/results/
  motion_configs/manipulator/   # generated motion JSON (pose + motion configs)
  verify/
    pilot40_*                   # Gemini baselines, consolidated GT, metrics
    pilot40_qwen32b/            # Qwen suite outputs exp01..exp10
  render/manipulator/           # GIF/MP4 renders for VLM inputs
  visualize/                    # pose tiles, pairwise composites, multitile grids
  html/manipulator/             # human review HTML

adhoc/generation/robotarm/
  run_pilot40_qwen_suite.py     # orchestrator (single model load)
  pilot40_experiment_suite.py   # step specs + metrics
  verify_*.py                   # per-step runners (Gemini or Qwen)
  prompt_loader.py              # load data/seed/prompt/pilot40/*.txt
```

---

## Main suite — 10 experiments (39 cues)

Run on a GPU node (cluster):

```bash
git pull
salloc --partition=YOUR_PART --gres=gpu:2 --mem=128G --time=24:00:00
conda activate m2m_caption32b
cd dev_robosuite
source scripts/cluster_env.sh /data/user_data/$USER/hf_cache

bash scripts/run_pilot40_qwen_suite.sh              # all 10 steps
ONLY=5,6 bash scripts/run_pilot40_qwen_suite.sh     # subset
RESUME=1 bash scripts/run_pilot40_qwen_suite.sh      # skip finished JSONs
SUMMARY_ONLY=1 bash scripts/run_pilot40_qwen_suite.sh  # accuracy table only
```

Smaller model ablation:

```bash
VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct \
  OUT_DIR=data/results/verify/pilot40_qwen7b \
  bash scripts/run_pilot40_qwen_suite.sh
```

| # | Experiment | Prompt | Input / media | Code | Gemini result | Qwen result |
|---|------------|--------|---------------|------|---------------|-------------|
| 1 | Pose generation vs human GT | `manipulator/prompt_v19_sophisticated.txt` | `motion_configs_prompt_v19_generation_pose_pilot40.json` | score in `run_pilot40_qwen_suite.py` | `pilot40_pose_eval_consolidated_scored.tsv` | `pilot40_qwen32b/exp01_pose_generation_score.json` |
| 2 | Pose verify — VLM (tile) | `pilot40/exp02_pose_verify_vlm.txt` | pose config + tile PNG | `verify_pose_tiles_gemini.py` | `pose_tile_verify_pilot{10,20,20_more}_gemini.json` | `exp02_pose_verify_vlm.json` |
| 3 | Pose verify — text | `pilot40/exp03_pose_verify_text.txt` | pose config (no image) | `verify_pose_textonly_gemini.py` | `pose_textonly_verify_pilot{10,20,20_more}_gemini.json` | `exp03_pose_verify_text.json` |
| 4 | Pose pairwise 2-way | `pilot40/exp04_pose_pairwise_2way.txt` | `visualize/pose_pairwise_12_pilot40/` | `verify_pose_pairwise_12_gemini.py` | `pilot40_pose_pairwise_12_gemini.json` | `exp04_pose_pairwise_2way.json` |
| 5 | Multitile grid 6 | `pilot40/exp05_pose_multitile_grid.txt` | `visualize/pose_multitile_gt_pilot40_grid6/` | `verify_pose_multitile_gt_gemini.py` | *(no pilot-40 baseline; pilot20 only)* | `exp05_pose_multitile_grid6.json` |
| 6 | Multitile grid 12 | `pilot40/exp05_pose_multitile_grid.txt` | `visualize/pose_multitile_gt_pilot40_grid12/` | `verify_pose_multitile_gt_gemini.py` | *(no pilot-40 baseline; pilot20 only)* | `exp06_pose_multitile_grid12.json` |
| 7 | Motion generation vs component GT | `manipulator/prompt_gt_fixed_first_pose.txt` | `motion_configs_prompt_v19_gt_fixed_pose_pilot40.json` | score in `run_pilot40_qwen_suite.py` | `pilot40_motion_verify_metrics.json` | `exp07_motion_generation_score.json` |
| 8 | Motion verify — VLM (MP4) | `pilot40/exp08_motion_verify_vlm.txt` | `render/.../motion_vlm_verify_pilot40/mp4/` | `verify_motion_component_gemini.py` | `pilot40_motion_component_verify_gemini.json` | `exp08_motion_verify_vlm.json` |
| 9 | Motion verify — text | `pilot40/exp09_motion_verify_text.txt` | motion config JSON | `verify_motion_component_text_gemini.py` | `pilot40_motion_component_verify_text_gemini.json` | `exp09_motion_verify_text.json` |
| 10 | Motion pairwise (MP4) | `pilot40/exp10_motion_pairwise_mp4.txt` | neg-pairwise MP4 composites | `verify_motion_gt_neg_pairwise_vlm.py` | `samples/motion_gt_neg_pairwise/pairwise_eval_results*.json` | `exp10_motion_pairwise_mp4.json` |

Paths in the **Qwen result** column are relative to `data/results/verify/pilot40_qwen32b/` unless `OUT_DIR` is set.

**Human GT anchor:** `data/results/verify/pilot40_pose_eval_consolidated.json`

**Suite summary JSON:** `data/results/verify/pilot40_qwen32b/pilot40_qwen_suite_summary.json`

---

## Pilot-90 suite — 10 experiments (90 non-essence cues)

| Item | Location |
|------|----------|
| Experiment registry | `adhoc/generation/robotarm/pilot90_experiment_suite.py` |
| Suite runner | `scripts/run_pilot90_qwen_suite.sh` → `run_pilot90_qwen_suite.py` |
| Pose configs | `motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json` |
| Pose human GT | `pilot40_pose_eval_consolidated.json` (any-pose scoring) |
| Motion component GT | `pilot40_motion_component_gt.json` (90 cues) |
| Qwen outputs | `pilot90_qwen32b/`, `pilot90_qwen7b/`, `pilot90_qwen3b/` |

Run on cluster (**`RESUME=1` default** — skips cues already in output JSONs):

```bash
bash scripts/prepare_pilot90_motion_mp4.sh
bash scripts/run_pilot90_qwen_suite.sh
MODEL_SIZE=7b bash scripts/run_pilot90_qwen_suite.sh
RESUME=0 bash scripts/run_pilot90_qwen_suite.sh   # fresh run
SUMMARY_ONLY=1 bash scripts/run_pilot90_qwen_suite.sh
bash scripts/run_pilot90_qwen_all_models.sh       # 32b → 7b → 3b
```

| # | Experiment | Qwen result |
|---|------------|-------------|
| 1 | Pose generation vs human GT (any-pose) | `exp01_pose_generation_score.json` |
| 2 | Pose verify — VLM (tile) | `exp02_pose_verify_vlm.json` |
| 3 | Pose verify — text | `exp03_pose_verify_text.json` |
| 4 | Pose pairwise 2-way | `exp04_pose_pairwise_2way.json` |
| 5 | Multitile grid 6 | `exp05_pose_multitile_grid6.json` |
| 6 | Multitile grid 12 | `exp06_pose_multitile_grid12.json` |
| 7 | Motion generation vs component GT | `exp07_motion_generation_score.json` |
| 8 | Motion verify — VLM (MP4) | `exp08_motion_verify_vlm.json` |
| 9 | Motion verify — text | `exp09_motion_verify_text.json` |
| 10 | Motion pairwise (MP4) | `exp10_motion_pairwise_mp4.json` |

**Suite summary JSON:** `data/results/verify/pilot90_qwen32b/pilot90_qwen_suite_summary.json`

---

## Step 8 prep (motion MP4)

Before step 8 on cluster:

```bash
bash scripts/prepare_pilot40_motion_mp4.sh   # pilot-40 (auto-renders missing GIFs)
bash scripts/prepare_pilot90_motion_mp4.sh   # pilot-90: GIF→MP4 only (expects run/IIWA)
# or inside suite: MOTION_PREPARE_MP4=1 (default)
```

**Pilot-90 needs 88 motion GIFs** under `run/IIWA/` (not in git). Step 8 skips cues without GIF.

**Option A — sync from local** (fastest if you already rendered 90 cues locally):

```bash
# on laptop (repo root):
rsync -avz run/IIWA/ USER@login2.babel.cs.cmu.edu:PATH/TO/dev_robosuite/run/IIWA/

# on cluster:
bash scripts/prepare_pilot90_motion_mp4.sh   # expect 88/88 or 90/90 mp4 ready
```

**Option B — render on cluster** (needs `data/seed/_remainder/closest_poses_results.jsonl`):

```bash
bash scripts/prepare_pilot90_motion_mp4.sh --render-missing
# or: MOTION_RENDER_MISSING=1 bash scripts/prepare_pilot90_motion_mp4.sh
```

Then resume step 8–10:

```bash
ONLY=8,9,10 RESUME=1 bash scripts/run_pilot90_qwen_suite.sh
```

---

## Cluster troubleshooting

| Error | Fix |
|-------|-----|
| `Disk quota exceeded` | `source scripts/cluster_env.sh /data/user_data/$USER/hf_cache`; clear `~/.cache/huggingface` |
| `CUDA not available` | use `salloc` / sbatch with `--gres=gpu` |
| `'joint' parameter is required for 'path'` | pull latest `path_ee_ik.py` + `motion_generation_core.py` |
| Step 8 `no mp4` (pilot-40) | `bash scripts/prepare_pilot40_motion_mp4.sh` until 39/39 ready |
| Step 8 `no mp4` (pilot-90) | sync `run/IIWA/*.gif` from local, then `prepare_pilot90_motion_mp4.sh`; or `--render-missing` |

Env: `m2m_caption32b` or `robosuite-vlm` (hyphen). Set `HF_HOME=/data/user_data/$USER/hf_cache`.

---

## Planned experiments (not in main 10)

See `data/seed/experiments/planned_backlog.yml`:

| ID | Title | Status |
|----|-------|--------|
| `temporal_multitile` | Rhythmic cues — multitile with tempo-aware prompt | not run |
| `google_robot_vlm_compare` | TIAGo mobile ~40-cue pairwise VLM | not run |
| `google_robot_component_verify` | Google Robot component verify (Gemini) | not run |
| `pilot100_multitile` | Multitile 6+12 on full pilot-100 manifest | not run |
| `pilot100_batch_motion` | batch20/21 motion pipeline after pose batches | partial |
| `qwen7b_full_suite` | 7B model ablation on all 10 steps | not run |
| `humanoid_bimanual` | GR1 bimanual gestures | future |

---

## Review HTML

| Report | Path |
|--------|------|
| Pose generation eval | `data/results/html/manipulator/pose_generation_eval_review_pilot40.html` |
| Wrong-answer notebook | `data/results/html/manipulator/pilot40_wrong_answer_notebook.html` |
| Motion GT-fixed review | `data/results/html/manipulator/pilot40_motion_vlm_verify_gt_fixed.html` |
| Google Robot renders | `data/results/html/google_robot/render_google_robot_*.html` |

---

## Paper figures (pilot-90)

After Qwen suite finishes (`SUMMARY_ONLY=1 bash scripts/run_pilot90_qwen_suite.sh`):

```bash
bash scripts/build_paper_figures.sh acc
IDX=1,5,8,2,15,28 bash scripts/build_paper_figures.sh qual-pose
IDX=7,59,60 bash scripts/build_paper_figures.sh qual-movement
IDX=0,1,0,1 bash scripts/build_paper_figures.sh pairwise
bash scripts/build_paper_figures.sh components
bash scripts/build_paper_figures.sh persona      # GOOGLE_API_KEY
bash scripts/build_paper_figures.sh essence10    # GOOGLE_API_KEY
```

Outputs: `data/results/paper_figures/` (PDF line plots, PNG grids, captions).
CLI: `python adhoc/generation/robotarm/paper_figures/cli.py <acc|qual|pairwise|components|persona|essence10>`.
