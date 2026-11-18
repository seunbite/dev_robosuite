# Transfer outcomes for the 30-cue screening set

**Status:** provisional program-derived coding. `H / L` denotes human-authored / LLM-authored. 
`P` = Preserved, `R` = Rematerialized, `A` = Augmented, and `L/D` = Lost or Distorted. 
Augmentation is an orthogonal flag, so combinations such as `P+A` and `L/D+A` are allowed.

The common semantic source is the human-authored G1 program. LLM outputs are additionally checked against the body-specific human program on the cue's required O/L/M channels. The JSON companion records every rule and mismatch. Independent coding and recognition data are still required before these labels are reported as empirical findings.

## Distribution

Counts are out of 30 cues. `A` overlaps the three primary outcomes.

| Robot | Author | P | R | L/D | A |
|---|---|---:|---:|---:|---:|
| G1 | Human | 30 | 0 | 0 | 0 |
| G1 | LLM | 9 | 0 | 21 | 8 |
| TIAGo | Human | 29 | 0 | 1 | 6 |
| TIAGo | LLM | 12 | 0 | 18 | 12 |
| IIWA | Human | 10 | 4 | 16 | 8 |
| IIWA | LLM | 5 | 2 | 23 | 7 |
| Stretch | Human | 17 | 1 | 12 | 12 |
| Stretch | LLM | 14 | 0 | 16 | 12 |
| Roomba | Human | 0 | 11 | 19 | 12 |
| Roomba | LLM | 0 | 10 | 20 | 14 |

## Per-cue coding

| # | Cue | Critical | G1 H / L | TIAGo H / L | IIWA H / L | Stretch H / L | Roomba H / L |
|---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `nod_yes` | M | P / P | P / P | R / R | P / P | R / R |
| 2 | `shake_head_no` | M | P / P | P / P | R / R | P / P | R / R |
| 3 | `draw_circle_repeat` | OM | P / P | P / P | P / P | P / P | R / R |
| 4 | `gaze_shift_avoid` | OM | P / P | P / P | R / L/D | P / P | R / R |
| 5 | `fist_pump` | OLM | P / L/D | P / L/D | P / P | P / P | L/D / L/D+A |
| 6 | `tilt_head_curious` | OM | P / L/D | P / P | R / L/D | P / P+A | R / L/D |
| 7 | `cover_yawn_sleepy` | OLM | P / L/D | P / L/D | L/D / L/D | P / P | R / R |
| 8 | `rub_eye_tired` | OLM | P / P | P / L/D | L/D+A / L/D | P / P | R / R |
| 9 | `point_you` | OL | P / P+A | P / P+A | P / P+A | P+A / P+A | L/D / L/D+A |
| 10 | `point_self` | OL | P / L/D+A | P / P+A | L/D / L/D | P+A / P+A | L/D+A / L/D+A |
| 11 | `beckon_come_here` | OLM | P / L/D | P+A / L/D | P / P | P / P+A | R / R |
| 12 | `stop_palm_out` | OL | P / L/D+A | P+A / L/D+A | P+A / L/D+A | P+A / P+A | L/D+A / L/D+A |
| 13 | `salute` | OL | P / L/D | P+A / L/D+A | L/D / L/D+A | L/D+A / L/D+A | L/D+A / L/D+A |
| 14 | `raise_index_one_moment` | OM | P / L/D | P / P | L/D / L/D | L/D / L/D | L/D / L/D |
| 15 | `highfive_invite` | OL | P / L/D | P+A / P+A | P+A / L/D+A | P+A / L/D+A | L/D+A / L/D+A |
| 16 | `shrug_idk` | OLM | P / L/D | P / L/D | L/D / L/D | L/D / L/D | L/D+A / R |
| 17 | `flex_bicep` | OL | P / P+A | P+A / L/D+A | L/D / L/D+A | L/D+A / L/D+A | L/D+A / L/D+A |
| 18 | `air_quotes` | OLM | P / L/D+A | P / L/D+A | L/D / L/D | L/D / L/D+A | L/D+A / L/D+A |
| 19 | `arms_open_welcome` | OLM | P / L/D | P / L/D+A | L/D / L/D | L/D / L/D | L/D / L/D |
| 20 | `hands_up_surrender` | OLM | P / L/D | P / P | L/D / L/D | L/D+A / L/D | L/D+A / L/D+A |
| 21 | `stretch_arms_up` | OLM | P / L/D | P / L/D | L/D+A / L/D+A | L/D+A / L/D+A | L/D / L/D |
| 22 | `raise_fists_victory` | OLM | P / L/D | P / L/D | L/D / L/D+A | L/D / L/D | L/D+A / L/D+A |
| 23 | `cross_arms_defensive` | OLM | P / L/D+A | P / L/D+A | L/D / L/D | L/D+A / L/D | L/D+A / L/D+A |
| 24 | `palms_together_thank_you` | OLM | P / L/D | L/D / L/D | L/D+A / L/D | R / L/D | L/D / L/D |
| 25 | `facepalm` | OLM | P / P | P / L/D | L/D / L/D | L/D / L/D | R / L/D |
| 26 | `depict_tree` | OLM | P / L/D+A | P / L/D+A | P+A / L/D | P+A / L/D | L/D / L/D+A |
| 27 | `depict_big_bowl` | OLM | P / L/D | P / L/D | P / L/D | P / L/D | R / R |
| 28 | `depict_lion` | OLM | P / L/D | P / L/D+A | L/D+A / L/D | L/D+A / L/D+A | L/D+A / L/D+A |
| 29 | `depict_small_object` | OL | P / L/D+A | P+A / P+A | P+A / P | P+A / P+A | L/D+A / L/D+A |
| 30 | `depict_flying_bird` | OLM | P / P | P / P | P / L/D | P / P | R / R |
