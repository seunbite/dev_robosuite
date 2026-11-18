# Synth60 Motion Code

Authoring + render scripts for the 60-cue (non-paper-30) Motion Codes. Open [`index.html`](index.html) for galleries.

## Layout

| Path | What |
|---|---|
| `render_synth60_tiago_paddle.py` | **Current** Tiago + Paddlebot (`gcr_alpha1`) Motion Codes + GIFs |
| `render_synth60_motion60.py` | **Current** Unitree G1 Motion Codes + GIFs |
| `render_synth60_motion30.py` / `render_synth60_motion10.py` | Earlier G1 subsets; galleries go to `archive/` |
| `data/cue_xyz_60_humanoid_tiago.json` | Body-relative xyz used for Tiago/Paddlebot |
| `data/cue_xyz_90_humanoid_tiago.json` | Same table including the first 30 cues |
| `bank/g1_bank_gallery/` | Feasible G1 pose-bank tiles (orientation × grip) |
| `bank/_g1_xyz_iter/` | xyz IK iteration scratch |
| `archive/` | Intermediate galleries (motion10/30, xyz5, ori3, …) |

Rendered configs / GIFs are **not** stored here:

- `data/results/motion_configs/embodiment_motion_code_synth60_*`
- `data/results/render/embodiment_motion_code_synth60_*`

Cue-spec tokens (`REQUIRED`) still live in `../score_humanlike_cuespec.py`. The converter is `adhoc/generation/robotarm/motion_code_to_config.py`.

## Run

From this directory (needs the robosuite env + MuJoCo):

```bash
python render_synth60_tiago_paddle.py          # Tiago + Paddlebot
python render_synth60_tiago_paddle.py --skip-paddle
python render_synth60_motion60.py              # G1
```
