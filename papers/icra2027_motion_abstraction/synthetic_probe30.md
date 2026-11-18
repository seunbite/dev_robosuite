# Synthetic-morphology probe (30 cues x 3 bodies)

Program-level structural observations; not recognition measurements.

**Provenance:** Programs are hard-coded in synthetic_morphology/render_cues30.py (put(...)) together with joint-level render plans; they are not outputs of the Gemini authoring pipeline.

## Carrier usage (number of cues activating each carrier)

- **PaddleClaw**: claw 24, paddle 12, head 6, torso 6, base 2, ee 1
- **SpineCrab**: long_pincer 22, short_pincer 12, spine 11, base 5, antenna 1
- **TrilegCrown**: leg0 20, base 17, leg1 9, leg2 7, crown 1

## Motion structure

- **PaddleClaw**: dynamic 16, static 14
- **SpineCrab**: dynamic 16, static 14
- **TrilegCrown**: dynamic 19, static 11

Cues whose motion structure differs across the three bodies: **15/30**

## Exact collapse (identical non-neutral pose values and motions)

- **PaddleClaw** (0 cues in 0 groups)
- **SpineCrab** (2 cues in 1 groups)
  - `long=in | torso=low` <- cover_yawn_sleepy, facepalm
- **TrilegCrown** (5 cues in 2 groups)
  - `leg0=present` <- point_you, salute, flex_bicep
  - `path(base,arc(osc))` <- nod_yes, draw_circle_repeat

## Coarse collapse (same carriers and motion kinds)

- **PaddleClaw** (12 cues in 4 groups)
  - `claw :: static` <- point_you, point_self, salute, raise_index_one_moment, flex_bicep
  - `claw | paddle :: static` <- arms_open_welcome, raise_fists_victory, cross_arms_defensive
  - `claw :: osc:pitch` <- fist_pump, beckon_come_here
  - `claw | paddle | torso :: static` <- stretch_arms_up, depict_tree
- **SpineCrab** (13 cues in 5 groups)
  - `long_pincer | spine :: static` <- cover_yawn_sleepy, stop_palm_out, facepalm
  - `long_pincer :: static` <- point_you, point_self, salute
  - `long_pincer | short_pincer :: static` <- arms_open_welcome, raise_fists_victory, cross_arms_defensive
  - `long_pincer :: osc:tip` <- rub_eye_tired, beckon_come_here
  - `long_pincer | short_pincer | spine :: static` <- stretch_arms_up, depict_tree
- **TrilegCrown** (18 cues in 8 groups)
  - `leg0 :: static` <- point_you, salute, raise_index_one_moment, flex_bicep
  - `base :: path:arc(osc)` <- nod_yes, draw_circle_repeat
  - `base :: static` <- gaze_shift_avoid, tilt_head_curious
  - `leg0 :: osc:yaw` <- fist_pump, rub_eye_tired
  - `base :: path:arc` <- cover_yawn_sleepy, depict_big_bowl
  - `leg0 | leg1 | leg2 :: static` <- arms_open_welcome, cross_arms_defensive
  - `base | leg0 | leg1 | leg2 :: path:line` <- hands_up_surrender, depict_tree
  - `base | leg0 :: path:line` <- facepalm, depict_lion

## Schema violations

- beckon_come_here / PaddleClaw: pose `elbow_c=T` not in ['false', 'true']
- salute / PaddleClaw: pose `grip=edge-like` not in ['close', 'open', 'pinch', 'spread']
- raise_index_one_moment / PaddleClaw: pose `elbow_c=T` not in ['false', 'true']
- shrug_idk / SpineCrab: pose `short=out` not in ['back', 'down', 'front', 'still', 'up']
- shrug_idk / SpineCrab: move amount `high` is not a degree value
- flex_bicep / PaddleClaw: pose `elbow_c=T` not in ['false', 'true']
- arms_open_welcome / SpineCrab: pose `short=out` not in ['back', 'down', 'front', 'still', 'up']
- cross_arms_defensive / SpineCrab: pose `short=in` not in ['back', 'down', 'front', 'still', 'up']
- palms_together_thank_you / TrilegCrown: move with zero amplitude (no-op)
- depict_tree / SpineCrab: pose `short=out` not in ['back', 'down', 'front', 'still', 'up']
- depict_big_bowl / SpineCrab: pose `short=out` not in ['back', 'down', 'front', 'still', 'up']
- depict_flying_bird / SpineCrab: pose `short=out` not in ['back', 'down', 'front', 'still', 'up']

## Schema `rematerialize` examples reproduced verbatim by the 30-cue programs

none

## Per-cue programs

| Cue | PaddleClaw | SpineCrab | TrilegCrown | Same structure |
|---|---|---|---|:---:|
| `nod_yes` | `pose(head=center, torso=mid) + move(head, tilt, [-12,12], rep=3, spd=1.5)` | `pose(torso=mid) + move(torso, pitch, ±12, rep=3, spd=1.5)` | `path(base, arc, r=0, deg=±12, rep=3, spd=1.5)` | N |
| `shake_head_no` | `move(head, pan, ±20, rep=3, spd=2.0)` | `path(torso, arc, r=0, deg=±25, rep=3)` | `move(base, yaw, ±25, rep=3, spd=2.0)` | N |
| `draw_circle_repeat` | `pose(claw=front) + path(ee_claw, arc, plane=xy, r=0.08, sweep=360, rep=2)` | `pose(long=front) + path(ee_long, arc, plane=xy, r=0.08, sweep=360, rep=2)` | `path(base, arc, r=0.15, deg=360, rep=2, spd=1.4)` | Y |
| `gaze_shift_avoid` | `pose(head=left) + pose(head=right) + pose(head=center)` | `pose(dir=left) + pose(dir=right) + pose(dir=front)` | `pose(dir=left) + pose(dir=right) + pose(dir=front)` | Y |
| `fist_pump` | `pose(claw=up, grip=close) + move(claw, pitch, [0,30], rep=3, spd=2.2)` | `pose(long=up) + move(long_pincer, pitch, [0,30], rep=3, spd=2.2)` | `pose(leg0=raise) + move(hip0, yaw, [0,40], rep=3, spd=2.2)` | N |
| `tilt_head_curious` | `pose(head=left) + move(head, roll, +20, hold=1.0)` | `pose(dir=left, antenna=forward) + hold(t=1.0)` | `pose(dir=left) + hold(t=1.0)` | N |
| `cover_yawn_sleepy` | `pose(claw=in, paddle=still, head=down, torso=low) + hold(t=1.2)` | `pose(torso=low, long=in) + hold(t=1.2)` | `path(base, arc, r=0.25, deg=15, spd=0.8) + path(base, arc, r=0.25, deg=-30, spd=0.5, hold=1.2)` | N |
| `rub_eye_tired` | `pose(claw=in, head=center) + move(wrist_c, roll, ±25, rep=4)` | `pose(long=in) + move(long_pincer, tip, ±20, rep=4)` | `pose(leg0=present) + move(hip0, yaw, ±15, rep=4, spd=1.2)` | N |
| `point_you` | `pose(dir=front, claw=front, grip=pinch, paddle=still) + hold(t=1.4)` | `pose(dir=front, long=front, long_grip=edge) + hold(t=1.4)` | `pose(dir=front, leg0=present) + hold(t=1.4)` | Y |
| `point_self` | `pose(claw=back, grip=pinch) + hold(t=1.4)` | `pose(long=back) + hold(t=1.4)` | `pose(dir=back, leg0=present) + hold(t=1.4)` | Y |
| `beckon_come_here` | `pose(claw=front, grip=open, elbow_c=T) + move(claw, pitch, ±25, rep=3)` | `pose(long=front) + move(long_pincer, tip, ±30, rep=3)` | `pose(leg0=raise) + move(knee0, pitch, ±25, rep=3) + path(base, line, x=+0.08)` | N |
| `stop_palm_out` | `pose(paddle=front, palm=palmfront, claw=still, torso=mid) + hold(t=1.5)` | `pose(torso=high, long=front, long_grip=flat) + hold(t=1.5)` | `path(base, line, x=+0.12) + hold(t=1.5)` | N |
| `salute` | `pose(claw=up, grip=edge-like, head=center) + hold(t=1.4)` | `pose(long=up, short=still) + hold(t=1.4)` | `pose(leg0=present) + hold(t=1.4)` | Y |
| `raise_index_one_moment` | `pose(claw=up, grip=pinch, elbow_c=T) + hold(t=1.3)` | `pose(short=up) + hold(t=1.3)` | `pose(leg0=raise) + hold(t=1.3)` | Y |
| `highfive_invite` | `pose(claw=front, grip=open) + path(ee_claw, line, axis=x, dist=+0.15, hold=1.0)` | `pose(torso=high, long=front) + path(base, line, x=+0.15, hold=1.2)` | `path(base, arc, r=0, deg=360, spd=2.5) + path(base, line, x=+0.20, hold=1.0) + pose(leg0=present)` | N |
| `shrug_idk` | `pose(paddle=out, claw=out, palm=palmup) + move(torso, pitch, +15, hold=0.4) + move(paddle, yaw, ±15) + move(claw, yaw, ±15)` | `pose(long=out, short=out) + move(torso, height, high, hold=0.4) + move(long_pincer, yaw, ±20) + move(short_pincer, yaw, ±20)` | `pose(leg0=raise, leg1=raise, leg2=raise) + move(base, yaw, ±10, rep=2)` | N |
| `flex_bicep` | `pose(claw=out, elbow_c=T, grip=close) + hold(t=1.3)` | `pose(long=up) + move(long_pincer, pitch, -40, hold=1.2)` | `pose(leg0=present) + hold(t=1.3)` | N |
| `air_quotes` | `pose(claw=up, paddle=up, grip=pinch) + move(wrist_c, roll, ±30, rep=2) + move(wrist_p, roll, ±30, rep=2)` | `pose(long=up, short=up) + move(long_pincer, tip, ±25, rep=2) + move(short_pincer, pitch, ±25, rep=2)` | `pose(leg0=raise, leg1=raise) + move(hip0, yaw, ±20, rep=2) + move(hip1, yaw, ±20, rep=2)` | N |
| `arms_open_welcome` | `pose(paddle=out, claw=out, palm=palmup) + hold(t=1.4)` | `pose(long=out, short=out, torso=mid) + hold(t=1.4)` | `pose(leg0=raise, leg1=raise, leg2=raise) + hold(t=1.4)` | Y |
| `hands_up_surrender` | `pose(paddle=up, claw=up, grip=open) + path(base, line, x=-0.12) + hold(t=1.3)` | `pose(torso=high, long=up, short=up) + path(base, line, x=-0.15) + hold(t=1.3)` | `pose(leg0=raise, leg1=raise, leg2=raise) + path(base, line, x=-0.15) + hold(t=1.3)` | Y |
| `stretch_arms_up` | `pose(paddle=up, claw=up, torso=high) + hold(t=1.4)` | `pose(torso=high, long=up, short=up) + hold(t=1.4)` | `pose(leg0=raise, leg1=raise, leg2=raise, crown=half) + hold(t=1.4)` | Y |
| `raise_fists_victory` | `pose(paddle=up, claw=up, grip=close) + hold(t=1.4)` | `pose(long=up, short=up, long_grip=close) + hold(t=1.4)` | `pose(leg0=present, leg1=raise) + hold(t=1.4)` | Y |
| `cross_arms_defensive` | `pose(paddle=in, claw=in) + hold(t=1.4)` | `pose(long=in, short=in, torso=mid) + hold(t=1.4)` | `pose(leg0=plant, leg1=plant, leg2=plant) + hold(t=1.4)` | Y |
| `palms_together_thank_you` | `pose(paddle=front, claw=front, palm=palmin) + move(torso, pitch, +20, hold=1.0)` | `pose(long=front, short=front) + move(torso, pitch, +20, hold=1.0)` | `pose(dir=front) + path(base, line, x=+0.08) + move(base, yaw, 0, hold=1.2)` | N |
| `facepalm` | `pose(claw=in, head=down, paddle=still) + hold(t=1.3)` | `pose(long=in, torso=low) + hold(t=1.3)` | `pose(leg0=present) + path(base, line, x=-0.05) + hold(t=1.3)` | N |
| `depict_tree` | `pose(torso=high, paddle=up, claw=out) + hold(t=1.2)` | `pose(torso=high) + pose(long=up, short=out) + hold(t=1.2)` | `path(base, line, x=+0.12) + pose(leg0=present) + pose(leg1=raise) + pose(leg2=raise) + hold(t=1.0)` | N |
| `depict_big_bowl` | `pose(paddle=out, claw=out) + path(ee_claw, arc, plane=yz, r=0.14, sweep=-120)` | `pose(long=out, short=out) + path(ee_long, arc, plane=yz, r=0.12, sweep=-120)` | `path(base, arc, r=0.28, deg=-180, spd=0.9) + hold(t=0.8)` | Y |
| `depict_lion` | `pose(claw=front, grip=open, torso=low) + path(base, line, x=-0.10) + path(ee_claw, line, axis=x, dist=+0.20)` | `pose(long=front) + path(base, line, x=-0.08) + path(ee_long, line, axis=x, dist=+0.15)` | `path(base, line, x=-0.10) + path(base, line, x=+0.05) + path(base, line, x=+0.18) + pose(leg0=present)` | Y |
| `depict_small_object` | `pose(claw=front, grip=pinch, z=55) + move(claw, pitch, ±12, rep=2) + hold(t=0.8)` | `pose(short=front) + move(short_pincer, pitch, ±15, rep=2) + hold(t=0.8)` | `pose(leg0=raise) + move(knee0, pitch, ±12, rep=2) + hold(t=0.8)` | Y |
| `depict_flying_bird` | `pose(paddle=out, claw=out) + move(paddle, yaw, ±35, rep=4) + move(claw, yaw, ±35, rep=4)` | `pose(long=out, short=out) + move(long_pincer, yaw, ±35, rep=4) + move(short_pincer, yaw, ±35, rep=4)` | `move(hip1, yaw, ±30, rep=4) + move(hip2, yaw, ±30, rep=4)` | Y |
