# Movement Mismatches by Embodiment

This audit uses exactly the strict Movement criterion plotted in Fig. 6:
the canonical Human-authored Humanoid motion form, axis, and sign must match
the Gemini-authored realization.  Lateral `y`, head `pan`, and head `tilt`
signs are mirrored (`±`).  Thus, a row is present if and only if it accounts
for one lost Movement-alignment point in the corresponding line-plot series.

`arc-xz` denotes a circular/arc path in the `xz` plane; `no-axis` means the
program was directional but did not expose a parsed stroke axis.

## Humanoid — 20 / 30 mismatches

| Cue | Original movement (human GT) | Changed movement (Gemini) |
|---|---|---|
| `draw_circle_repeat` | circular · arc-xz | circular · arc-yz |
| `fist_pump` | directional · z+ | oscillatory · z+, z- |
| `tilt_head_curious` | directional · height-, pan± | directional · pan± |
| `cover_yawn_sleepy` | directional · height-, tilt± | directional · tilt±, x- |
| `point_you` | static | directional · x+ |
| `point_self` | static | directional · x- |
| `beckon_come_here` | oscillatory · x+, x- | oscillatory · x- |
| `stop_palm_out` | static | directional · x+ |
| `salute` | directional · x-, z+ | static |
| `highfive_invite` | directional · x+ | static |
| `shrug_idk` | oscillatory · z+, z- | directional · z+ |
| `air_quotes` | oscillatory · z+, z- | oscillatory · y± |
| `arms_open_welcome` | directional · y± | static |
| `hands_up_surrender` | directional · z- | static |
| `stretch_arms_up` | static | directional · z+ |
| `raise_fists_victory` | directional · z+ | static |
| `cross_arms_defensive` | static | directional · y± |
| `depict_tree` | static | directional · y±, z+ |
| `depict_lion` | directional · height-, z- | directional · x+ |
| `depict_small_object` | directional · z- | directional · x-, z+ |

## Tiago — 19 / 30 mismatches

| Cue | Original movement (human GT) | Changed movement (Gemini) |
|---|---|---|
| `draw_circle_repeat` | circular · arc-xz | circular · arc-yz |
| `fist_pump` | directional · z+ | oscillatory · z+, z- |
| `tilt_head_curious` | directional · height-, pan± | directional · pan± |
| `cover_yawn_sleepy` | directional · height-, tilt± | directional · tilt±, x- |
| `point_you` | static | directional · x+ |
| `point_self` | static | directional · x- |
| `beckon_come_here` | oscillatory · x+, x- | directional · x- |
| `stop_palm_out` | static | directional · z+ |
| `salute` | directional · x-, z+ | directional · z+ |
| `highfive_invite` | directional · x+ | directional · tilt± |
| `air_quotes` | oscillatory · z+, z- | oscillatory · x+, x- |
| `arms_open_welcome` | directional · y± | directional · z+ |
| `hands_up_surrender` | directional · z- | directional · z+ |
| `stretch_arms_up` | static | directional · z+ |
| `cross_arms_defensive` | static | directional · tilt± |
| `facepalm` | directional · height-, tilt±, x- | directional · arc-yz, tilt± |
| `depict_tree` | static | directional · y±, z+ |
| `depict_lion` | directional · height-, z- | directional · tilt±, x+ |
| `depict_small_object` | directional · z- | directional · x+ |

## Single Arm — 20 / 30 mismatches

| Cue | Original movement (human GT) | Changed movement (Gemini) |
|---|---|---|
| `nod_yes` | oscillatory · tilt± | oscillatory · z+, z- |
| `shake_head_no` | oscillatory · pan± | oscillatory · y± |
| `draw_circle_repeat` | circular · arc-xz | circular · arc-yz |
| `gaze_shift_avoid` | directional · pan± | directional · y± |
| `tilt_head_curious` | directional · height-, pan± | directional · y± |
| `cover_yawn_sleepy` | directional · height-, tilt± | directional · x- |
| `point_you` | static | directional · x+ |
| `stop_palm_out` | static | directional · x+ |
| `salute` | directional · x-, z+ | directional · z+ |
| `raise_index_one_moment` | directional · z+ | static |
| `shrug_idk` | oscillatory · z+, z- | directional · y±, z+ |
| `flex_bicep` | directional · z+ | directional · z- |
| `arms_open_welcome` | directional · y± | static |
| `hands_up_surrender` | directional · z- | directional · no-axis |
| `stretch_arms_up` | static | directional · no-axis |
| `raise_fists_victory` | directional · z+ | directional · y± |
| `palms_together_thank_you` | directional · tilt± | directional · arc-xz |
| `facepalm` | directional · height-, tilt±, x- | directional · z- |
| `depict_lion` | directional · height-, z- | oscillatory · z+, z- |
| `depict_small_object` | directional · z- | static |

## Stretch — 23 / 30 mismatches

| Cue | Original movement (human GT) | Changed movement (Gemini) |
|---|---|---|
| `draw_circle_repeat` | circular · arc-xz | circular · arc-yz |
| `fist_pump` | directional · z+ | oscillatory · z+, z- |
| `tilt_head_curious` | directional · height-, pan± | directional · tilt± |
| `cover_yawn_sleepy` | directional · height-, tilt± | directional · arc-xz, tilt± |
| `point_you` | static | directional · x+ |
| `point_self` | static | directional · x- |
| `beckon_come_here` | oscillatory · x+, x- | oscillatory · y± |
| `stop_palm_out` | static | directional · x+ |
| `salute` | directional · x-, z+ | directional · z+ |
| `highfive_invite` | directional · x+ | directional · z+ |
| `shrug_idk` | oscillatory · z+, z- | directional · height+, height |
| `flex_bicep` | directional · z+ | directional · arc-yz |
| `air_quotes` | oscillatory · z+, z- | oscillatory · x+, x- |
| `arms_open_welcome` | directional · y± | directional · arc-xy |
| `hands_up_surrender` | directional · z- | directional · arc-xz |
| `stretch_arms_up` | static | directional · height+ |
| `raise_fists_victory` | directional · z+ | directional · height+ |
| `cross_arms_defensive` | static | directional · y± |
| `palms_together_thank_you` | directional · tilt± | directional · tilt±, z+ |
| `facepalm` | directional · height-, tilt±, x- | directional · height, tilt± |
| `depict_tree` | static | directional · arc-yz |
| `depict_lion` | directional · height-, z- | oscillatory · pan±, x+ |
| `depict_small_object` | directional · z- | directional · z+ |

## Roomba — 28 / 30 mismatches

| Cue | Original movement (human GT) | Changed movement (Gemini) |
|---|---|---|
| `nod_yes` | oscillatory · tilt± | oscillatory · x+, x- |
| `shake_head_no` | oscillatory · pan± | oscillatory · yaw+, yaw- |
| `draw_circle_repeat` | circular · arc-xz | circular · yaw+ |
| `gaze_shift_avoid` | directional · pan± | directional · yaw+ |
| `fist_pump` | directional · z+ | oscillatory · x+, x- |
| `tilt_head_curious` | directional · height-, pan± | directional · yaw+ |
| `cover_yawn_sleepy` | directional · height-, tilt± | directional · yaw- |
| `rub_eye_tired` | circular · arc-yz | oscillatory · x+, x-, yaw+, yaw- |
| `point_you` | static | directional · x+ |
| `point_self` | static | directional · x+, x- |
| `stop_palm_out` | static | directional · x+ |
| `salute` | directional · x-, z+ | directional · x+ |
| `raise_index_one_moment` | directional · z+ | directional · x+ |
| `shrug_idk` | oscillatory · z+, z- | directional · yaw+, yaw- |
| `flex_bicep` | directional · z+ | directional · x+, yaw+ |
| `air_quotes` | oscillatory · z+, z- | oscillatory · yaw+, yaw- |
| `arms_open_welcome` | directional · y± | directional · yaw+, yaw- |
| `hands_up_surrender` | directional · z- | directional · x- |
| `stretch_arms_up` | static | directional · x+ |
| `raise_fists_victory` | directional · z+ | directional · yaw+ |
| `cross_arms_defensive` | static | directional · x-, yaw- |
| `palms_together_thank_you` | directional · tilt± | directional · x+, yaw+, yaw- |
| `facepalm` | directional · height-, tilt±, x- | directional · x+, x- |
| `depict_tree` | static | oscillatory · x+, yaw+, yaw- |
| `depict_big_bowl` | directional · arc-yz | directional · yaw+ |
| `depict_lion` | directional · height-, z- | directional · x+, x- |
| `depict_small_object` | directional · z- | directional · x+, yaw+ |
| `depict_flying_bird` | oscillatory · z+, z- | directional · yaw+, yaw- |
