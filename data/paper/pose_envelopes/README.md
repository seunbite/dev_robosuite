# Anthropomorphic pose envelopes

`manifest.json` is the canonical validity contract for the Unitree G1 and
TIAGo pose banks used by the paper.

The contract is **closed-world**: a configuration is valid only when its ID is
listed in `valid_configuration_ids`. Absence means invalid for paper rendering;
it does not necessarily mean that the configuration received an individual
human-negative label.

| Embodiment | Enumerated | Valid | Retained | Configuration ID |
|---|---:|---:|---:|---|
| Unitree G1 | 1,458 | 206 | 14.1% | integer `pose_id` |
| TIAGo | 729 | 47 | 6.4% | `direction|orientation|tile_idx` |

The manifest records which stages were rule-based and which were handcrafted.
Full valid configurations are stored in:

- `data/seed/unitree_g1/g1_feasible_poses_metadata.json`
- `data/seed/tiago/plausible_poses_metadata.jsonl`

The G1 file also retains the manual removal IDs and reasons in
`filter_history`. For TIAGo, the manifest preserves the 32 explicitly marked
negative configurations in addition to the 47 positive selections.
