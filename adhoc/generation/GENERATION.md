# Generation Pipeline Structure

Root level keeps only thin dispatchers:

- `preprocess.py`
- `motion_generation.py`
- `render.py`

They only receive `--robot=robotarm|manipulator|tiago|quadruped` and call each robot script via `os.system`.

## Per-robot implementation files

Each robot folder provides:

- `preprocess.py`
- `motion_generation.py`
- `render.py`

### Behavior

- `preprocess.py`:
  - robotarm: builds `closest_poses_results.jsonl` using `find_closest_poses.py`
  - tiago / quadruped: no cache step
  - then continues to `motion_generation.py` by default
- `motion_generation.py`:
  - iterates `data/seed/yml/cues_new.yml` in order
  - combines latest `prompt.txt` + `shots.json`
  - writes aggregate JSON to `data/results/motion_configs/<robot>/motion_configs_<cue_group>.json`
  - then continues to `render.py` by default
- `render.py`:
  - renders configs to `data/results/render/<robot>/`
  - `config_json=None` => latest `motion_configs*.json` by mtime
  - if config JSON is missing, auto-runs `motion_generation.py` first
  - `do_html=True` => generates HTML under `data/results/html/<robot>/` and auto-opens it

## robotarm

Core runtime modules:

- `find_closest_poses.py`
- `config_gen_single.py`
- `motion_generation_core.py` (former renderer core)
- `arm_pose_config.py`
- `alphabet_jacobian.py`

Legacy scripts are in `robotarm/legacy/`.

## tiago

Core runtime modules:

- `config_gen_single_mobile.py`
- `render_mobile_config.py`

Legacy scripts are in `tiago/legacy/`.

## quadruped

Core runtime modules:

- `config_gen_single.py`

Legacy scripts are in `quadruped/legacy/`.

## Commands

```bash
# Root dispatchers
python3 adhoc/generation/preprocess.py --robot=robotarm --reset=True
python3 adhoc/generation/motion_generation.py --robot=robotarm --cue_group=iconic
python3 adhoc/generation/render.py --robot=robotarm --sim_robot=IIWA
python3 adhoc/generation/render.py --robot=quadruped --do_html=True

# Direct robot scripts
python3 adhoc/generation/robotarm/preprocess.py
python3 adhoc/generation/tiago/motion_generation.py --cue_group=iconic
python3 adhoc/generation/quadruped/render.py
```
