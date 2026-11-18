# Reproducibility guide

## Environment

Use Python 3.10 and MuJoCo 3.3 or later. The minimal parser/validation path is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m motion_programs.cli validate data/paper
```

Install `requirements-vlm.txt` only when rerunning LLM authoring. API keys must
be supplied through environment variables and must never be committed.

## Paper-to-code map

| Paper artifact | Input | Analysis / renderer |
|---|---|---|
| Motion Program grammar | `data/paper/motion_program_schema.yaml` | `motion_programs.compiler` |
| Human-like attribute analysis | `data/paper/evaluation/humanoid_attribute_summary.json` | `papers/icra2027_motion_abstraction/score_humanoid_loc_dir_move.py` |
| Non-anthropomorphic transfer | `data/paper/evaluation/transfer_outcomes_50.json` | `papers/icra2027_motion_abstraction/analyze_transfer_50.py` |
| Program richness | `data/paper/evaluation/program_complexity.json` | `papers/icra2027_motion_abstraction/analyze_program_complexity_30.py` |
| Synthetic morphologies | `data/paper/cues50_synthetic.yaml` | `papers/icra2027_motion_abstraction/synthetic_morphology/` |
| Human recognition | `data/paper/evaluation/human_eval_4afc_responses.json` | Anonymized response and summary snapshots; raw survey media are archived |

The checked-in JSON files are immutable snapshots of model output. The compact
release supports schema validation, Motion Code compilation, and inspection of
the reported summary data. Re-running every analysis from raw intermediate
files requires the archived workspace on `codex/pre-paper-cleanup-20261118`;
the public branch intentionally omits those multi-gigabyte inputs.

Generated GIF/MP4 trees are excluded because they are several gigabytes. They
can be regenerated from the programs when the required simulator and robot
assets are available, but exact pixels may vary with renderer versions.

## Smoke checks

```bash
pytest -q tests/test_release.py
python -m motion_programs.cli compile --platform roomba \
  'pose(dir=front) + path(base, line, x=-0.15) + hold(t=1.0)'
```

Rendering requires embodiment-specific assets. G1 rendering additionally
expects the Unitree MJCF workspace; other release validation remains usable
without it.
