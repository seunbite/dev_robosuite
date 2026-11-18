# 50-cue non-anthropomorphic transfer audit

Automatic program-derived coding; these labels do not establish human recognition.

## Compact result

| Robot | Preserved | Carrier-transferred | Rematerialized | Augmented | Maintained | Lost |
|---|---:|---:|---:|---:|---:|---:|
| IIWA | 22 | 5 | 23 | 0 | 23 | 27 |
| Stretch | 21 | 2 | 27 | 7 | 21 | 22 |
| Roomba | 0 | 10 | 40 | 0 | 16 | 34 |

## Per-program coding

| ID | Cue | Robot | Required body parts | Feasible | Strategy | Consequence | Reason |
|---:|---|---|---|:---:|---|---|---|
| 1 | `nod_yes` | IIWA | Head / torso | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 1 | `nod_yes` | Stretch | Head / torso | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 1 | `nod_yes` | Roomba | Head / torso | Y | Carrier-transferred | Maintained | the required structure is feasible and no indispensable event is absent |
| 2 | `shake_head_no` | IIWA | Head / torso | Y | Carrier-transferred | Maintained | the required structure is feasible and no indispensable event is absent |
| 2 | `shake_head_no` | Stretch | Head / torso | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 2 | `shake_head_no` | Roomba | Head / torso | Y | Carrier-transferred | Maintained | the required structure is feasible and no indispensable event is absent |
| 3 | `draw_circle_repeat` | IIWA | One arm / hand | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 3 | `draw_circle_repeat` | Stretch | One arm / hand | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 3 | `draw_circle_repeat` | Roomba | One arm / hand | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 4 | `gaze_shift_avoid` | IIWA | Head / torso | Y | Carrier-transferred | Maintained | the required structure is feasible and no indispensable event is absent |
| 4 | `gaze_shift_avoid` | Stretch | Head / torso | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 4 | `gaze_shift_avoid` | Roomba | Head / torso | Y | Carrier-transferred | Maintained | the required structure is feasible and no indispensable event is absent |
| 5 | `fist_pump` | IIWA | One arm / hand | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 5 | `fist_pump` | Stretch | One arm / hand | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 5 | `fist_pump` | Roomba | One arm / hand | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 6 | `tilt_head_curious` | IIWA | Head / torso | Y | Carrier-transferred | Maintained | the required structure is feasible and no indispensable event is absent |
| 6 | `tilt_head_curious` | Stretch | Head / torso | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 6 | `tilt_head_curious` | Roomba | Head / torso | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 7 | `cover_yawn_sleepy` | IIWA | Head / torso + one arm | N | Carrier-transferred | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 7 | `cover_yawn_sleepy` | Stretch | Head / torso + one arm | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 7 | `cover_yawn_sleepy` | Roomba | Head / torso + one arm | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 8 | `rub_eye_tired` | IIWA | Head / torso + one arm | N | Preserved | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 8 | `rub_eye_tired` | Stretch | Head / torso + one arm | N | Preserved | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 8 | `rub_eye_tired` | Roomba | Head / torso + one arm | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 9 | `point_you` | IIWA | One arm / hand | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 9 | `point_you` | Stretch | One arm / hand | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 9 | `point_you` | Roomba | One arm / hand | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 10 | `point_self` | IIWA | Head / torso + one arm | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 10 | `point_self` | Stretch | Head / torso + one arm | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 10 | `point_self` | Roomba | Head / torso + one arm | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 11 | `beckon_come_here` | IIWA | One arm / hand | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 11 | `beckon_come_here` | Stretch | One arm / hand | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 11 | `beckon_come_here` | Roomba | One arm / hand | Y | Carrier-transferred | Maintained | the required structure is feasible and no indispensable event is absent |
| 12 | `stop_palm_out` | IIWA | One arm / hand | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 12 | `stop_palm_out` | Stretch | One arm / hand | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 12 | `stop_palm_out` | Roomba | One arm / hand | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 13 | `salute` | IIWA | Head / torso + one arm | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 13 | `salute` | Stretch | Head / torso + one arm | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 13 | `salute` | Roomba | Head / torso + one arm | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 14 | `raise_index_one_moment` | IIWA | One arm / hand | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 14 | `raise_index_one_moment` | Stretch | One arm / hand | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 14 | `raise_index_one_moment` | Roomba | One arm / hand | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 15 | `highfive_invite` | IIWA | One arm / hand | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 15 | `highfive_invite` | Stretch | One arm / hand | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 15 | `highfive_invite` | Roomba | One arm / hand | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 16 | `shrug_idk` | IIWA | Two arms / hands | N | Preserved | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 16 | `shrug_idk` | Stretch | Two arms / hands | N | Carrier-transferred | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 16 | `shrug_idk` | Roomba | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 17 | `flex_bicep` | IIWA | One arm / hand | N | Preserved | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 17 | `flex_bicep` | Stretch | One arm / hand | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 17 | `flex_bicep` | Roomba | One arm / hand | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 18 | `air_quotes` | IIWA | Two arms / hands | N | Preserved | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 18 | `air_quotes` | Stretch | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 18 | `air_quotes` | Roomba | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 19 | `arms_open_welcome` | IIWA | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 19 | `arms_open_welcome` | Stretch | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 19 | `arms_open_welcome` | Roomba | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 20 | `hands_up_surrender` | IIWA | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 20 | `hands_up_surrender` | Stretch | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 20 | `hands_up_surrender` | Roomba | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 21 | `stretch_arms_up` | IIWA | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 21 | `stretch_arms_up` | Stretch | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 21 | `stretch_arms_up` | Roomba | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 22 | `raise_fists_victory` | IIWA | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 22 | `raise_fists_victory` | Stretch | Two arms / hands | N | Carrier-transferred | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 22 | `raise_fists_victory` | Roomba | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 23 | `cross_arms_defensive` | IIWA | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 23 | `cross_arms_defensive` | Stretch | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 23 | `cross_arms_defensive` | Roomba | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 24 | `palms_together_thank_you` | IIWA | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 24 | `palms_together_thank_you` | Stretch | Two arms / hands | N | Preserved | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 24 | `palms_together_thank_you` | Roomba | Two arms / hands | N | Carrier-transferred | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 25 | `facepalm` | IIWA | Head / torso + one arm | N | Preserved | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 25 | `facepalm` | Stretch | Head / torso + one arm | N | Preserved | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 25 | `facepalm` | Roomba | Head / torso + one arm | N | Carrier-transferred | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 26 | `depict_tree` | IIWA | One arm / hand | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 26 | `depict_tree` | Stretch | One arm / hand | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 26 | `depict_tree` | Roomba | One arm / hand | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 27 | `depict_big_bowl` | IIWA | Two arms / hands | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 27 | `depict_big_bowl` | Stretch | Two arms / hands | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 27 | `depict_big_bowl` | Roomba | Two arms / hands | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 28 | `depict_lion` | IIWA | Two arms / hands | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 28 | `depict_lion` | Stretch | Two arms / hands | Y | Rematerialized | Augmented | the program recruits an additional independently controllable carrier |
| 28 | `depict_lion` | Roomba | Two arms / hands | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 29 | `depict_small_object` | IIWA | One arm / hand | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 29 | `depict_small_object` | Stretch | One arm / hand | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 29 | `depict_small_object` | Roomba | One arm / hand | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 30 | `depict_flying_bird` | IIWA | Two arms / hands | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 30 | `depict_flying_bird` | Stretch | Two arms / hands | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 30 | `depict_flying_bird` | Roomba | Two arms / hands | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 31 | `self_scratch_head_confused` | IIWA | Head / torso + one arm | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 31 | `self_scratch_head_confused` | Stretch | Head / torso + one arm | N | Preserved | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 31 | `self_scratch_head_confused` | Roomba | Head / torso + one arm | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 32 | `pat_stomach` | IIWA | Head / torso + one arm | N | Preserved | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 32 | `pat_stomach` | Stretch | Head / torso + one arm | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 32 | `pat_stomach` | Roomba | Head / torso + one arm | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 33 | `fan_face_hot` | IIWA | Head / torso + one arm | N | Preserved | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 33 | `fan_face_hot` | Stretch | Head / torso + one arm | N | Preserved | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 33 | `fan_face_hot` | Roomba | Head / torso + one arm | N | Carrier-transferred | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 37 | `cover_mouth_gasp` | IIWA | Head / torso + one arm | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 37 | `cover_mouth_gasp` | Stretch | Head / torso + one arm | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 37 | `cover_mouth_gasp` | Roomba | Head / torso + one arm | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 40 | `fistbump_invite` | IIWA | One arm / hand | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 40 | `fistbump_invite` | Stretch | One arm / hand | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 40 | `fistbump_invite` | Roomba | One arm / hand | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 42 | `throat_cut_stop` | IIWA | Head / torso + one arm | N | Preserved | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 42 | `throat_cut_stop` | Stretch | Head / torso + one arm | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 42 | `throat_cut_stop` | Roomba | Head / torso + one arm | N | Carrier-transferred | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 43 | `visor_search` | IIWA | Head / torso + one arm | N | Carrier-transferred | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 43 | `visor_search` | Stretch | Head / torso + one arm | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 43 | `visor_search` | Roomba | Head / torso + one arm | N | Carrier-transferred | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 44 | `yield_turn` | IIWA | One arm / hand | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 44 | `yield_turn` | Stretch | One arm / hand | Y | Rematerialized | Augmented | the program recruits an additional independently controllable carrier |
| 44 | `yield_turn` | Roomba | One arm / hand | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 45 | `request_turn` | IIWA | One arm / hand | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 45 | `request_turn` | Stretch | One arm / hand | Y | Rematerialized | Augmented | the program recruits an additional independently controllable carrier |
| 45 | `request_turn` | Roomba | One arm / hand | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 48 | `slow_down_request_palm_down` | IIWA | One arm / hand | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 48 | `slow_down_request_palm_down` | Stretch | One arm / hand | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 48 | `slow_down_request_palm_down` | Roomba | One arm / hand | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 51 | `scale_expand` | IIWA | Two arms / hands | N | Preserved | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 51 | `scale_expand` | Stretch | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 51 | `scale_expand` | Roomba | Two arms / hands | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 61 | `clarification_request_palm_tilt` | IIWA | One arm / hand | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 61 | `clarification_request_palm_tilt` | Stretch | One arm / hand | Y | Preserved | Augmented | the program recruits an additional independently controllable carrier |
| 61 | `clarification_request_palm_tilt` | Roomba | One arm / hand | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 65 | `continue_on_forward_roll` | IIWA | One arm / hand | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 65 | `continue_on_forward_roll` | Stretch | One arm / hand | Y | Rematerialized | Augmented | the program recruits an additional independently controllable carrier |
| 65 | `continue_on_forward_roll` | Roomba | One arm / hand | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 69 | `interrupt_hard` | IIWA | One arm / hand | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 69 | `interrupt_hard` | Stretch | One arm / hand | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 69 | `interrupt_hard` | Roomba | One arm / hand | Y | Rematerialized | Lost | the reference requires a dynamic event but the target program contains none |
| 70 | `interrupt_soft` | IIWA | One arm / hand | Y | Preserved | Maintained | the required structure is feasible and no indispensable event is absent |
| 70 | `interrupt_soft` | Stretch | One arm / hand | Y | Preserved | Augmented | the program recruits an additional independently controllable carrier |
| 70 | `interrupt_soft` | Roomba | One arm / hand | Y | Rematerialized | Maintained | the required structure is feasible and no indispensable event is absent |
| 71 | `self_hug` | IIWA | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 71 | `self_hug` | Stretch | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 71 | `self_hug` | Roomba | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 78 | `push_pull_dual` | IIWA | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 78 | `push_pull_dual` | Stretch | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 78 | `push_pull_dual` | Roomba | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 81 | `wait_for_me_backward_palm` | IIWA | One arm / hand | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 81 | `wait_for_me_backward_palm` | Stretch | One arm / hand | Y | Preserved | Augmented | the program recruits an additional independently controllable carrier |
| 81 | `wait_for_me_backward_palm` | Roomba | One arm / hand | N | Carrier-transferred | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 83 | `frustration_down_throw` | IIWA | One arm / hand | N | Preserved | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 83 | `frustration_down_throw` | Stretch | One arm / hand | N | Preserved | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 83 | `frustration_down_throw` | Roomba | One arm / hand | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 90 | `clap_applause` | IIWA | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 90 | `clap_applause` | Stretch | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
| 90 | `clap_applause` | Roomba | Two arms / hands | N | Rematerialized | Lost | an indispensable carrier or relation is unavailable under the a-priori feasibility rule |
