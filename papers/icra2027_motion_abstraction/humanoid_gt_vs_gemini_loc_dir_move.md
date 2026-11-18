# Humanoid GT vs Gemini — continuous location/direction, binary movement

N = 30 cues (Unitree G1).

## Summary

| Metric | Value |
|---|---:|
| Location cues with GT xyz | 25 |
| Location mean L2 (body %) | 43.6 |
| Location median L2 | 50.2 |
| Location mean per-axis |Δ| | 24.8 |
| Direction mean reach ∠ (°) | 55.4 (n=26) |
| Direction mean palm ∠ (°) | 41.5 (n=26) |
| Direction mean RPY geodesic (°) | 92.3 (n=26) |
| Movement binary acc | 11/24 (46%) |

## Per cue

| # | Cue | Δx | Δy | Δz | L2 | reach° | palm° | geo° | Δr,Δp,Δy | move |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | nod_yes | — | — | — | — | — | — | — | — | ✓ |
| 2 | shake_head_no | — | — | — | — | — | — | — | — | ✓ |
| 3 | draw_circle_repeat | — | — | 20.0 | 20.0 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 4 | gaze_shift_avoid | — | — | — | — | — | — | — | — | ✗ |
| 5 | fist_pump | — | — | 0.0 | 70.7 | 90 | 0 | 90 | 0,90,0 | ✗ |
| 6 | tilt_head_curious | — | — | — | — | — | — | — | — | ✗ |
| 7 | cover_yawn_sleepy | 10.0 | 25.0 | 15.0 | 30.8 | 90 | 90 | 120 | 90,0,90 | ✓ |
| 8 | rub_eye_tired | 15.0 | 10.0 | 0.0 | 18.0 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 9 | point_you | — | — | 20.0 | 20.0 | 0 | 0 | 0 | 0,0,0 | n/a |
| 10 | point_self | 10.0 | 5.0 | 10.0 | 15.0 | 180 | 90 | 180 | 90,90,90 | n/a |
| 11 | beckon_come_here | — | — | 15.0 | 52.2 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 12 | stop_palm_out | — | — | 20.0 | 20.0 | 90 | 0 | 180 | 90,0,180 | n/a |
| 13 | salute | — | 20.0 | 10.0 | 54.8 | 90 | 90 | 120 | 90,0,90 | ✗ |
| 14 | raise_index_one_moment | — | — | 15.0 | 72.3 | 90 | 0 | 120 | 90,90,0 | ✗ |
| 15 | highfive_invite | — | — | — | — | 90 | 0 | 180 | 90,0,180 | ✓ |
| 16 | shrug_idk | — | — | 15.0 | 52.2 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 17 | flex_bicep | — | 0.0 | 0.0 | 50.0 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 18 | air_quotes | — | 5.0 | 15.0 | 52.4 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 19 | arms_open_welcome | — | — | 5.0 | 50.2 | 90 | 90 | 120 | 180,90,90 | ✗ |
| 20 | hands_up_surrender | — | — | 20.0 | 73.5 | 0 | 0 | 0 | 0,0,0 | ✗ |
| 21 | stretch_arms_up | — | — | 10.0 | 10.0 | 0 | 90 | 90 | 0,0,90 | n/a |
| 22 | raise_fists_victory | 25.0 | — | 0.0 | 55.9 | 0 | 90 | 90 | 0,0,90 | ✗ |
| 23 | cross_arms_defensive | — | — | 10.0 | 71.4 | 180 | 90 | 180 | 0,90,0 | n/a |
| 24 | palms_together_thank_you | — | 5.0 | — | 50.2 | 90 | 0 | 180 | 90,0,180 | ✓ |
| 25 | facepalm | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0,0,0 | ✓ |
| 26 | depict_tree | — | — | — | 50.0 | 90 | 90 | 180 | 180,0,90 | n/a |
| 27 | depict_big_bowl | — | — | 5.0 | 50.2 | 180 | 0 | 180 | 0,0,180 | ✓ |
| 28 | depict_lion | — | — | 5.0 | 50.2 | 0 | 90 | 90 | 90,90,90 | ✗ |
| 29 | depict_small_object | — | — | 0.0 | 50.0 | 90 | 90 | 120 | 0,90,90 | ✗ |
| 30 | depict_flying_bird | — | — | 5.0 | 50.2 | 0 | 0 | 0 | 0,0,0 | ✓ |

## Notes

- Location units are body-relative Motion Code percentages (0–100).
- Direction: `dir` → reach unit vector; `grip` palm token → palm normal; RPY from the reach–palm frame (geodesic = rotation angle between GT and LLM frames).
- Movement binary: GT joint+axis (or path shape) set ⊆ LLM set.
