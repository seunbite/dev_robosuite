# Non-humanoid O/L/M agreement (30 cues, Gemini 2.5 Pro vs GT)

Location band = ±20 body-%. Roomba O/L always N/A.

| Robot | Ori % (n) | Loc % (n) | Mov % (n) | All applicable % (n) |
|---|---:|---:|---:|---:|
| IIWA | 46.7 (14/30) | 73.1 (19/26) | 66.7 (20/30) | 26.7 (8/30) |
| Stretch | 70.4 (19/27) | 60.0 (18/30) | 80.0 (24/30) | 36.7 (11/30) |
| Roomba | — | — | 66.7 (20/30) | 66.7 (20/30) |

## Per cue

| # | Cue | IIWA O L M | Stretch O L M | Roomba O L M |
|---:|---|---|---|---|
| 1 | `nod_yes` | N Y Y | Y Y Y | — — Y |
| 2 | `shake_head_no` | Y Y Y | — Y Y | — — Y |
| 3 | `draw_circle_repeat` | Y Y Y | Y N Y | — — Y |
| 4 | `gaze_shift_avoid` | N — Y | — Y Y | — — Y |
| 5 | `fist_pump` | Y — Y | Y N Y | — — N |
| 6 | `tilt_head_curious` | N Y Y | — Y Y | — — N |
| 7 | `cover_yawn_sleepy` | N N N | Y Y Y | — — Y |
| 8 | `rub_eye_tired` | N Y Y | Y Y Y | — — Y |
| 9 | `point_you` | Y Y N | Y Y Y | — — N |
| 10 | `point_self` | Y Y Y | Y Y Y | — — N |
| 11 | `beckon_come_here` | Y Y Y | Y Y Y | — — Y |
| 12 | `stop_palm_out` | N Y Y | Y Y N | — — N |
| 13 | `salute` | Y — N | Y Y Y | — — Y |
| 14 | `raise_index_one_moment` | Y — N | N Y N | — — Y |
| 15 | `highfive_invite` | N Y Y | N N N | — — N |
| 16 | `shrug_idk` | N Y Y | N Y Y | — — Y |
| 17 | `flex_bicep` | N N N | N N Y | — — N |
| 18 | `air_quotes` | N Y Y | Y Y N | — — N |
| 19 | `arms_open_welcome` | Y Y Y | N N Y | — — N |
| 20 | `hands_up_surrender` | N N N | N N N | — — Y |
| 21 | `stretch_arms_up` | Y Y Y | Y N Y | — — Y |
| 22 | `raise_fists_victory` | N N N | Y N Y | — — Y |
| 23 | `cross_arms_defensive` | Y Y Y | N Y Y | — — Y |
| 24 | `palms_together_thank_you` | N N Y | N Y Y | — — Y |
| 25 | `facepalm` | N Y Y | Y N Y | — — N |
| 26 | `depict_tree` | Y Y N | Y N Y | — — Y |
| 27 | `depict_big_bowl` | N N Y | Y N Y | — — Y |
| 28 | `depict_lion` | N Y N | Y N Y | — — Y |
| 29 | `depict_small_object` | Y Y N | Y Y N | — — Y |
| 30 | `depict_flying_bird` | Y N Y | Y Y Y | — — Y |

## Misses

- **IIWA** `nod_yes` — ori: grip horiz→None
- **Stretch** `draw_circle_repeat` — loc: x 40.0→∅
- **IIWA** `gaze_shift_avoid` — ori: grip horiz→None
- **Stretch** `fist_pump` — loc: x 15.0→∅
- **Roomba** `fist_pump` — mov: ('path', 'advance', 'plus') vs [('path', 'rock', 'pm')]
- **IIWA** `tilt_head_curious` — ori: grip horiz→vert
- **Roomba** `tilt_head_curious` — mov: ('path', 'yaw', 'plus') vs [('path', 'circle', 'plus')]
- **IIWA** `cover_yawn_sleepy` — ori: grip horiz→vert; loc: x 60.0→∅; z 75→40; mov: ('path', 'z', 'plus') vs [('path', 'x', 'pm')]
- **IIWA** `rub_eye_tired` — ori: dir in→back
- **IIWA** `point_you` — mov: static→('path', 'x', 'plus')
- **Roomba** `point_you` — mov: static→('path', 'advance', 'plus')
- **Roomba** `point_self` — mov: ('path', 'yaw', 'plus') vs [('path', 'advance', 'plus'), ('path', 'rock', 'pm')]
- **IIWA** `stop_palm_out` — ori: dir front→up
- **Stretch** `stop_palm_out` — mov: ('path', 'yaw', 'pm') vs [('path', 'x', 'plus')]
- **Roomba** `stop_palm_out` — mov: ('path', 'yaw', 'pm') vs [('path', 'advance', 'plus')]
- **IIWA** `salute` — mov: static→('path', 'z', 'plus')
- **IIWA** `raise_index_one_moment` — mov: ('path', 'z', 'pm') vs [('hold', 'static', None)]
- **Stretch** `raise_index_one_moment` — ori: dir up→front; mov: static→('path', 'z', 'plus')
- **IIWA** `highfive_invite` — ori: dir up→front; grip horiz→open
- **Stretch** `highfive_invite` — ori: dir up→front; loc: x 25.0→∅; torso high→mid; mov: ('path', 'x', 'plus') vs [('path', 'z', 'plus')]
- **Roomba** `highfive_invite` — mov: ('path', 'yaw', 'plus') vs [('path', 'advance', 'plus')]
- **IIWA** `shrug_idk` — ori: grip horiz→vert
- **Stretch** `shrug_idk` — ori: dir in→out
- **IIWA** `flex_bicep` — ori: dir in→up; loc: x 70.0→∅; z 65→40; mov: static→('move', 'z', 'pm')
- **Stretch** `flex_bicep` — ori: dir up→out; loc: x 25.0→∅
- **Roomba** `flex_bicep` — mov: ('path', 'circle', 'plus') vs [('path', 'advance', 'plus'), ('path', 'yaw', 'plus')]
- **IIWA** `air_quotes` — ori: grip horiz→vert
- **Stretch** `air_quotes` — mov: ('move', 'z', 'pm') vs [('move', 'x', 'pm')]
- **Roomba** `air_quotes` — mov: ('path', 'rock', 'pm') vs [('path', 'yaw', None)]
- **Stretch** `arms_open_welcome` — ori: dir up→front; loc: z 85→60; torso high→mid
- **Roomba** `arms_open_welcome` — mov: ('path', 'advance', 'plus') vs [('path', 'circle', 'plus'), ('path', 'circle', 'pm')]
- **IIWA** `hands_up_surrender` — ori: dir up→front; loc: z 90→45; mov: static→('path', 'line', 'plus')
- **Stretch** `hands_up_surrender` — ori: dir up→front; loc: z 85→60; torso high→mid; mov: ('path', 'rock', 'pm') vs [('path', 'xz', 'plus')]
- **Stretch** `stretch_arms_up` — loc: z 30→70
- **IIWA** `raise_fists_victory` — ori: dir up→∅; grip close→vert; loc: z 65→40; mov: ('path', 'z', 'plus') vs [('move', 'y', 'pm'), ('move', 'y', 'pm')]
- **Stretch** `raise_fists_victory` — loc: z 35→70; torso low→mid
- **Stretch** `cross_arms_defensive` — ori: dir in→out
- **IIWA** `palms_together_thank_you` — ori: dir in→up; loc: x 100→55
- **Stretch** `palms_together_thank_you` — ori: dir up→down
- **IIWA** `facepalm` — ori: dir in→back; grip vert→horiz
- **Stretch** `facepalm` — loc: z 50→95
- **Roomba** `facepalm` — mov: ('path', 'circle', 'plus') vs [('path', 'advance', 'plus'), ('path', 'rock', 'pm')]
- **IIWA** `depict_tree` — mov: ('move', 'y', 'plus') vs [('hold', 'static', None)]
- **Stretch** `depict_tree` — loc: z 85→60
- **IIWA** `depict_big_bowl` — ori: dir out→up; loc: x 20→70
- **Stretch** `depict_big_bowl` — loc: x 20.0→∅; z 55→80; torso mid→high
- **IIWA** `depict_lion` — ori: grip open→horiz; mov: ('path', 'x', 'pm') vs [('move', 'z', 'pm'), ('path', 'line', 'plus')]
- **Stretch** `depict_lion` — loc: x 25.0→∅
- **IIWA** `depict_small_object` — mov: ('path', 'yz', 'plus') vs [('hold', 'static', None)]
- **Stretch** `depict_small_object` — mov: ('path', 'advance', 'plus') vs [('path', 'z', 'plus')]
- **IIWA** `depict_flying_bird` — loc: y 100→20
