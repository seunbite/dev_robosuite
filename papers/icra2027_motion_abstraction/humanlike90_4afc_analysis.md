# Humanlike-90 4AFC + cue-spec analysis

Source answers: `~/Downloads/humanlike90_4afc_anon_1788247794504.json`

## Inventory
- **90 cues**, coarse taxonomy:
  - Reference 21 · Social 29 · Self/cognition 19 · Discourse 21

## Human 4AFC (Unitree G1 stimuli, 3 hard negatives)
| | Overall | Reference | Social | Self | Discourse |
|---|---:|---:|---:|---:|---:|
| Accuracy | **54.4%** (43/79) | 63.2% | 52.0% | 62.5% | 42.1% |

Skips 9, unanswered 2 (excluded from accuracy). Chance = 25%.

## Wrong trials → which cue-spec axis differs (GT vs chosen foil)
Among **36** incorrect choices:
- Orientation differs: **81%**
- Location differs: **75%**
- Movement/axis differs: **72%**
- All three differ: **17/36**

## Gemini cue-spec agreement (90-cue union, discrete GT)
| Body | Orient | Loc | Move | All applicable |
|---|---:|---:|---:|---:|
| Humanoid | 52.9% | 48.5% | 61.3% | 25.6% |
| Tiago | 53.8% | 44.6% | 65.2% | 30.6% |

## Qwen2.5-VL-32B
Generation job on Babel: `smp-qw32b-90` → `embodiment_motion_code_qwen_qwen32b_humanlike90`.
Score with:
```bash
python papers/icra2027_motion_abstraction/score_humanlike_cuespec.py \
  --pred data/results/motion_configs/embodiment_motion_code_qwen_qwen32b_humanlike90/embodiment_motion_code_all.json \
  --tag qwen32b_humanlike90
```
