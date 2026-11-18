# Paper data

This directory is the compact, versioned source for the paper release.

- `motion_program_schema.yaml`: shared DSL and embodiment attribute schema.
- `cues50_synthetic.yaml`: 50-cue synthetic-morphology program inventory.
- `programs/`: human and model-authored program snapshots.
- `evaluation/`: anonymized responses and derived tables used by analyses.
- `pose_envelopes/`: closed-world G1 and TIAGo configuration allowlists and
  rule-based versus handcrafted selection provenance.

Filenames retain the experiment scope. The non-anthropomorphic comparison uses
the matched 30-cue subset; human-like model snapshots retain the larger source
pool from which the paper subset was selected. Do not interpret file length as
the sample size of every reported result.
