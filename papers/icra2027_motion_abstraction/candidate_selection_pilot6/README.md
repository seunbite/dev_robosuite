# Visual candidate selection: six-cue feasibility pilot

Run: 2026-09-12. Model: Gemini 2.5 Pro. Six completed API calls, no response-based retries. Reported token total: 2,838 (input, image, output and thinking combined). No pricing estimate is inferred.

## Protocol

Six preselected Unitree G1 cues from the existing humanlike90 GT/negative dataset. Each candidate set contains a human-authored GT plus one orientation-group, location-group, and movement-group foil. Existing clips were reused without re-rendering. Movement foils change axis/sign, or add a vertical stroke when the GT is static. Orientation can change both reach and palm; these are group-level, not single-scalar ablations.

Each PNG contains four neutrally labeled rows and eight uniformly sampled chronological frames per row. Camera and rendering are inherited from the source clips. Candidate position is shuffled deterministically; GT positions A, B, C, D, A, B. The VLM receives only the PNG, cue gloss, and selection instruction—not codes, GT labels, or candidate provenance. Temperature 0; output cap 2,048; thinking budget 128. All four sampled image sequences per cue have distinct pixel hashes.

| Cue | Selected candidate | GT recovered |
|---|---|---|
| salute | GT | Yes |
| highfive_invite | Location foil | No |
| stop_palm_out | Movement foil | No |
| cover_yawn_sleepy | GT | Yes |
| cover_ears_too_loud | GT | Yes |
| shh_gesture | Movement foil | No |

GT selection: 3/6 (50%); nominal uniform-choice baseline: 25%. This is reference agreement, not human-rated correctness. In particular, a location-modified high five might be an acceptable alternative. The model's movement-foil explanations emphasize pose and omit the added stroke.

## Limitations

- Convenience sample, one order per cue, no statistical superiority claim.
- Candidate position is approximately balanced, not exhaustively counterbalanced.
- Eight-frame storyboards and the provider's image processing may obscure fine palm orientation or brief motion. Each image was billed as 258 input image tokens.
- Code-group perturbations can couple in rendered space through IK/pose-bank selection. Distinct pixels do not establish perceptually isolated O/L/M manipulations.
- This is GT recovery among supplied alternatives, not autonomous generation improvement or a human recognition test.
- The existing 6-tile pose result uses 90 cues; the 12-tile cache is partial (80 cues). They concern reach/gripper pose selection and are not a matched 2/6/12 O/L/M comparison. Their design was reused, not their accuracies pooled.

## Files and reproducibility

`manifest.json` contains exact codes, source paths, candidate positions, and sampled-frame indices. `*.request.json` preserves prompts and generation settings without credentials or base64 payloads. `*.response.json` preserves raw answers and usage. `results.json` summarizes outcomes; `gallery.html` previews the exact PNGs supplied to the VLM.

From the repository root, run `papers/icra2027_motion_abstraction/.figvenv/bin/python papers/icra2027_motion_abstraction/run_candidate_selection_pilot.py` for preparation/summary only. Adding `--call` calls the API only for missing response files; existing results are reused. Do not remove response caches unless intentionally repeating the experiment.
