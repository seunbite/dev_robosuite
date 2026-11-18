# tiago GT vs LLM — continuous location/direction, binary movement

Pred: `data/results/motion_configs/embodiment_motion_code_qwen_qwen7b_humanlike90/embodiment_motion_code_all.json`
Platform: `tiago` · N = 30 cues.

## Table metrics (loc = mean |Δ| body-% ↓; rch/plm/mov = % ↑)

| Reach↑ | Palm↑ | Locx↓ | Locy↓ | Locz↓ | Movement↑ |
|---:|---:|---:|---:|---:|---:|
| 53.8 | 46.2 | n/a | n/a | 9.8 | 12.5 |

_Tiago Palm: 0° and 180° both count as match._

## Summary

| Metric | Value |
|---|---:|
| Location cues with GT xyz | 25 |
| Location mean L2 (body %) | 9.8 |
| Location median L2 | 10.0 |
| Location mean per-axis |Δ| | 9.8 |
| Direction mean reach ∠ (°) | 45.0 (n=26) |
| Direction mean palm ∠ (°) | 55.4 (n=26) |
| Direction mean RPY geodesic (°) | 72.7 (n=26) |
| Movement binary acc | 3/24 (12%) |

## Per cue

| # | Cue | Δx | Δy | Δz | L2 | reach° | palm° | geo° | Δr,Δp,Δy | move |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | nod_yes | — | — | — | — | — | — | — | — | ✓ |
| 2 | shake_head_no | — | — | — | — | — | — | — | — | ✓ |
| 3 | draw_circle_repeat | — | — | 10.0 | 10.0 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 4 | gaze_shift_avoid | — | — | — | — | — | — | — | — | ✓ |
| 5 | fist_pump | — | — | 10.0 | 10.0 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 6 | tilt_head_curious | — | — | — | — | — | — | — | — | ✗ |
| 7 | cover_yawn_sleepy | — | — | 10.0 | 10.0 | 90 | 90 | 180 | 90,0,180 | ✗ |
| 8 | rub_eye_tired | — | — | 12.0 | 12.0 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 9 | point_you | — | — | 0.0 | 0.0 | 0 | 0 | 0 | 0,0,0 | n/a |
| 10 | point_self | — | — | 10.0 | 10.0 | 90 | 90 | 90 | 90,0,0 | n/a |
| 11 | beckon_come_here | — | — | 10.0 | 10.0 | 90 | 90 | 90 | 90,0,0 | ✗ |
| 12 | stop_palm_out | — | — | 10.0 | 10.0 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 13 | salute | — | — | 15.0 | 15.0 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 14 | raise_index_one_moment | — | — | 10.0 | 10.0 | 90 | 0 | 90 | 0,90,0 | ✗ |
| 15 | highfive_invite | — | — | 5.0 | 5.0 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 16 | shrug_idk | — | — | 0.0 | 0.0 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 17 | flex_bicep | — | — | 20.0 | 20.0 | 90 | 90 | 120 | 90,90,180 | ✗ |
| 18 | air_quotes | — | — | 0.0 | 0.0 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 19 | arms_open_welcome | — | — | 5.0 | 5.0 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 20 | hands_up_surrender | — | — | 28.0 | 28.0 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 21 | stretch_arms_up | — | — | 5.0 | 5.0 | 0 | 90 | 90 | 0,0,90 | n/a |
| 22 | raise_fists_victory | — | — | 35.0 | 35.0 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 23 | cross_arms_defensive | — | — | 0.0 | 0.0 | 0 | 90 | 90 | 0,90,180 | n/a |
| 24 | palms_together_thank_you | — | — | 15.0 | 15.0 | 90 | 90 | 120 | 90,90,0 | n/a |
| 25 | facepalm | — | — | 0.0 | 0.0 | 90 | 180 | 180 | 90,90,90 | ✗ |
| 26 | depict_tree | — | — | 0.0 | 0.0 | 90 | 90 | 120 | 90,0,90 | n/a |
| 27 | depict_big_bowl | — | — | 5.0 | 5.0 | 180 | 90 | 180 | 0,90,0 | ✗ |
| 28 | depict_lion | — | — | 20.0 | 20.0 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 29 | depict_small_object | — | — | 10.0 | 10.0 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 30 | depict_flying_bird | — | — | — | — | 0 | 0 | 0 | 0,0,0 | ✗ |

## Notes

- Location units are body-relative Motion Code percentages (0–100).
- Unspecified LLM xyz axes are filled from the G1 feasible-bank pose the renderer selects for dir|grip|elbow (render-effective EE %; no 50 penalty).
- Location match: |Δaxis| ≤ 5.0.
- Direction: `dir` → reach unit vector; `grip` palm token → palm normal; RPY from the reach–palm frame (geodesic = rotation angle between GT and LLM frames). Reach/palm angles are acos ∈ [0, 180°].
- Movement binary: GT (axis, sign) set ⊆ LLM set; line paths share the same (axis, sign) keys as joint moves; arcs stay as (arc, plane).
