# Misspecification vs underspecification (GT vs Gemini 2.5 Pro SMP)

N = 150 cells (30 cues × 5 bodies).

## Cell-level

| Split | Misspec cells | Underspec cells | Both | Misspec only | Underspec only | Neither | GT comps | LLM comps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 90.0% | 46.0% | 44.0% | 46.0% | 2.0% | 8.0% | 2.73 | 2.91 |
| humanoid | 96.7% | 66.7% | 63.3% | 33.3% | 3.3% | 0.0% | 2.47 | 2.73 |
| tiago | 100.0% | 50.0% | 50.0% | 50.0% | 0.0% | 0.0% | 2.30 | 2.87 |
| single_arm | 90.0% | 16.7% | 16.7% | 73.3% | 0.0% | 10.0% | 2.37 | 3.00 |
| stretch | 66.7% | 63.3% | 56.7% | 10.0% | 6.7% | 26.7% | 3.83 | 2.77 |
| roomba | 96.7% | 33.3% | 33.3% | 63.3% | 0.0% | 3.3% | 2.67 | 3.20 |

## Wrong-value attributes (misspec)

| Family | Count |
|---|---:|
| location | 149 |
| speed | 109 |
| extent | 41 |
| facing | 30 |
| path_shape | 28 |
| radius | 19 |
| axis | 16 |
| t | 10 |
| dir | 9 |
| hold | 8 |
| joint | 5 |
| sweep | 5 |
| repetition | 4 |
| elbow_bended | 2 |
| plane | 1 |
| laterality | 1 |

## GT slots missing on an aligned component

| Family | Count |
|---|---:|
| location | 61 |
| hold | 18 |
| repetition | 16 |
| radius | 13 |
| extent | 11 |
| sweep | 3 |
| use_feasible_bank | 2 |
| plane | 2 |
| axis | 1 |

## Missing GT components (underspec)

- by type: {'path': 55, 'hold': 21, 'movement': 17, 'pose': 5}
- by role: {'core': 49, 'hold': 27, 'unknown': 15, 'submotion': 7}
- extra LLM types: {'path': 62, 'pose': 36, 'movement': 22, 'hold': 6}

## Named examples

### `air_quotes` / humanoid  misspec=True underspec=False

GT:
```
pose(dir=front, grip=horiz|palmdown, z=70, x=80, y=60, left_arm=mirror) + move(wrist, z, [-20,20], rep=2, spd=2.0, hold=0.1)
```
LLM:
```
pose(dir=front, grip=vert|palmin, z=85, y=65, elbow=T, left_arm=mirror) + move(wrist, y, ±20, rep=2, spd=2.8, hold=0.1)
```
misspec: [{"type": "pose", "role": "core", "diffs": [{"family": "facing", "key": "pose.gripper_orientation", "gt": "horizontal|palmdown", "llm": "vertical|palmin", "kind": "wrong_value"}, {"family": "location", "key": "pose.x", "gt": 80, "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}, {"family": "location", "key": "pose.y", "gt": 60, "llm": 65, "kind": "wrong_value"}, {"family": "location", "key": "pose.z", "gt": 70, "llm": 85, "kind": "wrong_value"}]}, {"type": "movement", "role": "core", "diffs": [{"family": "axis", "key": "movement.joints[0].axis", "gt": "z", "llm": "y", "kind": "wrong_value"}, {"family": "speed", "key": "movement.speed", "gt": 2.0, "llm": 2.8, "kind": "wrong_value"}]}]

### `air_quotes` / tiago  misspec=True underspec=False

GT:
```
pose(dir=front, grip=horiz, z=60, x=75, y=60, left_arm=mirror, elbow=T) + move(wrist, z, [-60,60], rep=2, spd=2.0, hold=0.1)
```
LLM:
```
pose(dir=front, grip=vert, z=60, left_arm=mirror, elbow=T) + move(wrist, x, ±20, rep=2, spd=2.5, hold=0.1)
```
misspec: [{"type": "pose", "role": "core", "diffs": [{"family": "facing", "key": "pose.gripper_orientation", "gt": "horizontal", "llm": "vertical", "kind": "wrong_value"}, {"family": "location", "key": "pose.x", "gt": 75, "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}, {"family": "location", "key": "pose.y", "gt": 60, "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}]}, {"type": "movement", "role": "core", "diffs": [{"family": "axis", "key": "movement.joints[0].axis", "gt": "z", "llm": "x", "kind": "wrong_value"}, {"family": "extent", "key": "movement.joints[0].degrees", "gt": [-60.0, 60.0], "llm": [-20.0, 20.0], "kind": "wrong_value"}, {"family": "speed", "key": "movement.speed", "gt": 2.0, "llm": 2.5, "kind": "wrong_value"}]}]

### `air_quotes` / single_arm  misspec=True underspec=False

GT:
```
pose(dir=front, grip=horiz, y=40, z=75, elbow=T, spd=1.2) + move(wrist, z, ±15, rep=2, spd=3.0, hold=0.2)
```
LLM:
```
pose(dir=front, grip=vert, y=40, z=75, spd=1.2) + move(wrist, z, ±15, rep=2, spd=3.0, hold=0.2)
```
misspec: [{"type": "pose", "role": "core", "diffs": [{"family": "facing", "key": "pose.gripper_orientation", "gt": "horizontal", "llm": "vertical", "kind": "wrong_value"}]}]

### `air_quotes` / stretch  misspec=True underspec=True

GT:
```
pose(dir=front, grip=vert, torso=high, z=85) + path(base, line, x=+0.10, spd=2.0) + move(wrist, z, [-10, 10], rep=2, spd=3.2, hold=0.1)
```
LLM:
```
pose(dir=front, grip=vert, torso=high, z=85) + move(wrist, x, [-25, 25], rep=2, spd=3.2, hold=0.1)
```
underspec: [{"gt_i": 1, "type": "path", "role": "core", "n_attrs": 4}]
misspec: [{"type": "movement", "role": "unknown", "diffs": [{"family": "axis", "key": "movement.joints[0].axis", "gt": "z", "llm": "x", "kind": "wrong_value"}, {"family": "extent", "key": "movement.joints[0].degrees", "gt": [-10.0, 10.0], "llm": [-25.0, 25.0], "kind": "wrong_value"}]}]

### `air_quotes` / roomba  misspec=True underspec=False

GT:
```
path(base, line, x=+-0.05,rep=2, spd=2.5)
```
LLM:
```
pose(dir=front) + path(base, arc, r=0, deg=[22, -22], rep=2, spd=3.5, hold=0.2)
```
misspec: [{"type": "path", "role": "core", "diffs": [{"family": "path_shape", "key": "path.shape", "gt": "line", "llm": "arc", "kind": "wrong_value"}, {"family": "speed", "key": "path.speed", "gt": 2.5, "llm": 3.5, "kind": "wrong_value"}, {"family": "location", "key": "path.x", "gt": [-0.05, 0.05], "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}]}]

### `highfive_invite` / humanoid  misspec=True underspec=True

GT:
```
pose(dir=up, grip=horiz|palmfront) + path(ee, line, axis=x, dist=0.10, rep=1, spd=3, hold=0.2) + hold(t=1.5)
```
LLM:
```
pose(dir=front, grip=horiz|palmfront, z=65, elbow=T) + hold(t=1.5)
```
underspec: [{"gt_i": 2, "type": "hold", "role": "hold", "n_attrs": 2}]
misspec: [{"type": "pose", "role": "core", "diffs": [{"family": "location", "key": "pose.arm_position", "gt": "up", "llm": "front", "kind": "wrong_value"}]}, {"type": "path", "role": "core", "diffs": [{"family": "extent", "key": "path.distance", "gt": 0.1, "llm": 0.0, "kind": "wrong_value"}, {"family": "repetition", "key": "path.repetition", "gt": "single", "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}, {"family": "speed", "key": "path.speed", "gt": 3.0, "llm": 0.5, "kind": "wrong_value"}]}]

### `highfive_invite` / tiago  misspec=True underspec=True

GT:
```
pose(dir=up, grip=horiz, z=86, y=40, x=60) + path(ee, line, axis=x, dist=0.20, rep=1, spd=3, hold=0.2) + hold(t=1.5)
```
LLM:
```
pose(dir=front, grip=horiz, z=60, x=65, elbow=T) + move(head, tilt, [8], spd=0.9) + hold(t=1.3)
```
underspec: [{"gt_i": 2, "type": "hold", "role": "hold", "n_attrs": 2}]
misspec: [{"type": "pose", "role": "core", "diffs": [{"family": "location", "key": "pose.arm_position", "gt": "up", "llm": "front", "kind": "wrong_value"}, {"family": "location", "key": "pose.x", "gt": 60, "llm": 65, "kind": "wrong_value"}, {"family": "location", "key": "pose.y", "gt": 40, "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}, {"family": "location", "key": "pose.z", "gt": 86, "llm": 60, "kind": "wrong_value"}]}, {"type": "path", "role": "core", "diffs": [{"family": "extent", "key": "path.distance", "gt": 0.2, "llm": 0.0, "kind": "wrong_value"}, {"family": "repetition", "key": "path.repetition", "gt": "single", "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}, {"family": "speed", "key": "path.speed", "gt": 3.0, "llm": 0.5, "kind": "wrong_value"}]}]

### `highfive_invite` / single_arm  misspec=True underspec=True

GT:
```
pose(dir=up, grip=horiz, z=70) + move(shoulder, x, 10, rep=1, spd=2.0, hold=1.5)
```
LLM:
```
pose(dir=down, grip=vert, z=35, y=50, x=50) + pose(dir=front, grip=open, z=55, y=50, x=65, spd=0.9) + path(ee, line, axis=x, dist=0.03, spd=0.6) + hold(t=1.3)
```
underspec: [{"gt_i": 1, "type": "movement", "role": "core", "n_attrs": 8}]
misspec: [{"type": "pose", "role": "core", "diffs": [{"family": "dir", "key": "pose.dir", "gt": "up", "llm": "front", "kind": "wrong_value"}, {"family": "location", "key": "pose.z", "gt": 70, "llm": 55, "kind": "wrong_value"}]}]

### `highfive_invite` / stretch  misspec=True underspec=True

GT:
```
pose(dir=front, grip=horiz, z=60, torso=mid) + path(base, line, x=+0.20, spd=2.2) + pose(dir=up, grip=horiz, torso=high, z=80, x=25) + path(ee, line, axis=x, dist=0.12, spd=1.8) + hold(t=1.3)
```
LLM:
```
pose(dir=front, grip=horiz, z=60, torso=mid) + path(ee, line, axis=z, dist=0.15, spd=1.5) + hold(t=1.5)
```
underspec: [{"gt_i": 1, "type": "path", "role": "core", "n_attrs": 4}, {"gt_i": 2, "type": "pose", "role": "core", "n_attrs": 6}]
misspec: [{"type": "path", "role": "hold", "diffs": [{"family": "axis", "key": "path.axis", "gt": "x", "llm": "z", "kind": "wrong_value"}, {"family": "extent", "key": "path.distance", "gt": 0.12, "llm": 0.15, "kind": "wrong_value"}, {"family": "speed", "key": "path.speed", "gt": 1.8, "llm": 1.5, "kind": "wrong_value"}]}, {"type": "hold", "role": "unknown", "diffs": [{"family": "t", "key": "t", "gt": 1.3, "llm": 1.5, "kind": "wrong_value"}]}]

### `highfive_invite` / roomba  misspec=True underspec=True

GT:
```
path(base, arc, r=0, deg=360, spd=2.5) + path(base, line, x=+0.30, spd=2.0) + hold(t=1.5)
```
LLM:
```
pose(dir=front) + path(base, line, x=0.20, spd=1.5) + hold(t=1.5)
```
underspec: [{"gt_i": 0, "type": "path", "role": "submotion", "n_attrs": 5}]
misspec: [{"type": "path", "role": "core", "diffs": [{"family": "speed", "key": "path.speed", "gt": 2.0, "llm": 1.5, "kind": "wrong_value"}, {"family": "location", "key": "path.x", "gt": 0.3, "llm": 0.2, "kind": "wrong_value"}]}]

### `stop_palm_out` / humanoid  misspec=True underspec=True

GT:
```
pose(dir=up, grip=horiz|palmfront, z=60)
+ hold(t=1.5)
```
LLM:
```
pose(dir=front, grip=horiz|palmfront, z=80, x=60, elbow=T) + path(ee, line, axis=x, dist=0.05, spd=2.5) + hold(t=1.2)
```
underspec: [{"gt_i": 1, "type": "hold", "role": "hold", "n_attrs": 2}]
misspec: [{"type": "pose", "role": "core", "diffs": [{"family": "location", "key": "pose.arm_position", "gt": "up", "llm": "front", "kind": "wrong_value"}, {"family": "location", "key": "pose.z", "gt": 60, "llm": 80, "kind": "wrong_value"}]}]

### `stop_palm_out` / stretch  misspec=True underspec=True

GT:
```
pose(dir=front, grip=horiz, torso=high, z=80, x=25) + path(base, arc, r=0, deg=±12, rep=2, spd=2.0) + path(base, line, x=0.18, spd=2.4) + hold(t=1.5)
```
LLM:
```
pose(dir=front, grip=horiz, torso=high, z=80, x=25) + path(ee, line, axis=x, dist=+0.15, spd=1.8) + hold(t=1.5)
```
underspec: [{"gt_i": 1, "type": "path", "role": "core", "n_attrs": 6}]
misspec: [{"type": "path", "role": "hold", "diffs": [{"family": "path_shape", "key": "path.mode", "gt": "base", "llm": "ee", "kind": "wrong_value"}, {"family": "speed", "key": "path.speed", "gt": 2.4, "llm": 1.8, "kind": "wrong_value"}, {"family": "location", "key": "path.x", "gt": 0.18, "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}]}]

### `stop_palm_out` / roomba  misspec=True underspec=True

GT:
```
path(base, arc, r=0, deg=±15, rep=2, spd=2.0) + path(base, line, x=0.2, spd=2.5) + hold(t=1.5)
```
LLM:
```
pose(dir=front) + path(base, line, x=0.20, spd=3.5) + hold(t=1.5)
```
underspec: [{"gt_i": 0, "type": "path", "role": "core", "n_attrs": 6}]
misspec: [{"type": "path", "role": "core", "diffs": [{"family": "speed", "key": "path.speed", "gt": 2.5, "llm": 3.5, "kind": "wrong_value"}]}]

### `rub_eye_tired` / humanoid  misspec=True underspec=False

GT:
```
pose(dir=in, grip=vert, z=80, y=50, x=75)
+ path(ee, arc, plane=yz,
  r=0.03, sweep=360°, rep=3, spd=1.5)
```
LLM:
```
pose(dir=in, grip=vert|palmback, elbow=T, head=down, z=80, y=60, x=60) + path(ee, arc, plane=yz, r=0.02, sweep=360, rep=3, spd=1.2)
```
misspec: [{"type": "pose", "role": "core", "diffs": [{"family": "facing", "key": "pose.gripper_orientation", "gt": "vertical", "llm": "vertical|palmback", "kind": "wrong_value"}, {"family": "location", "key": "pose.head", "gt": "center", "llm": "down", "kind": "wrong_value"}, {"family": "location", "key": "pose.x", "gt": 75, "llm": 60, "kind": "wrong_value"}, {"family": "location", "key": "pose.y", "gt": 50, "llm": 60, "kind": "wrong_value"}]}, {"type": "path", "role": "core", "diffs": [{"family": "radius", "key": "path.radius", "gt": 0.03, "llm": 0.02, "kind": "wrong_value"}, {"family": "speed", "key": "path.speed", "gt": 1.5, "llm": 1.2, "kind": "wrong_value"}]}]

### `rub_eye_tired` / tiago  misspec=True underspec=False

GT:
```
pose(dir=in, grip=vert, z=62, y=70, x=70) + path(ee, arc, plane=yz, r=0.03, sweep=360°, rep=3, spd=0.8)
```
LLM:
```
pose(dir=front, grip=closed, elbow=T, head=down, x=70, y=65, z=60) + path(ee, arc, plane=yz, r=0.015, sweep=360, rep=3, spd=1.2)
```
misspec: [{"type": "pose", "role": "core", "diffs": [{"family": "location", "key": "pose.arm_position", "gt": "in", "llm": "front", "kind": "wrong_value"}, {"family": "location", "key": "pose.head", "gt": "center", "llm": "down", "kind": "wrong_value"}, {"family": "location", "key": "pose.y", "gt": 70, "llm": 65, "kind": "wrong_value"}, {"family": "location", "key": "pose.z", "gt": 62, "llm": 60, "kind": "wrong_value"}]}, {"type": "path", "role": "core", "diffs": [{"family": "radius", "key": "path.radius", "gt": 0.03, "llm": 0.015, "kind": "wrong_value"}, {"family": "speed", "key": "path.speed", "gt": 0.8, "llm": 1.2, "kind": "wrong_value"}]}]

### `rub_eye_tired` / single_arm  misspec=True underspec=False

GT:
```
pose(dir=in, grip=vert, z=80) + path(ee, line, axis=z, dist=±0.05, rep=3, spd=1, hold=0)
```
LLM:
```
pose(grip=vertical, dir=back, y=45, z=80) + path(ee, arc, plane=yz, r=0.02, sweep=360, rep=3, spd=1.5, hold=0.2)
```
misspec: [{"type": "pose", "role": "core", "diffs": [{"family": "dir", "key": "pose.dir", "gt": "left", "llm": "back", "kind": "wrong_value"}]}, {"type": "path", "role": "core", "diffs": [{"family": "axis", "key": "axis", "gt": "z", "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}, {"family": "extent", "key": "distance", "gt": [-0.05, 0.05], "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}, {"family": "hold", "key": "hold_time", "gt": "zero", "llm": "positive", "kind": "wrong_value"}, {"family": "path_shape", "key": "shape", "gt": "line", "llm": "arc", "kind": "wrong_value"}, {"family": "speed", "key": "speed", "gt": 1.0, "llm": 1.5, "kind": "wrong_value"}]}]

### `rub_eye_tired` / stretch  misspec=False underspec=False

GT:
```
pose(dir=in, grip=vert, torso=high, z=90) + path(ee, arc, plane=yz, r=0.03, sweep=360, rep=4, spd=0.9, hold=0.2)
```
LLM:
```
pose(dir=in, grip=vert, torso=high, z=90) + path(ee, arc, plane=yz, r=0.03, sweep=360, rep=4, spd=0.9, hold=0.2)
```

### `rub_eye_tired` / roomba  misspec=True underspec=False

GT:
```
path(base, line, y=±0.05, rep=3, spd=1.5) + path(base, arc, r=0.4, deg=-20, spd=0.2, hold=1.5)
```
LLM:
```
pose(dir=front) + path(base, line, x=0.1, y=-0.05, spd=0.7) + path(base, arc, r=0.08, deg=[-60, 60], rep=2, spd=1.2)
```
misspec: [{"type": "path", "role": "core", "diffs": [{"family": "repetition", "key": "path.repetition", "gt": "plural", "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}, {"family": "speed", "key": "path.speed", "gt": 1.5, "llm": 0.7, "kind": "wrong_value"}, {"family": "location", "key": "path.y", "gt": [-0.05, 0.05], "llm": -0.05, "kind": "wrong_value"}]}, {"type": "path", "role": "core", "diffs": [{"family": "extent", "key": "path.degrees", "gt": -20.0, "llm": [-60.0, 60.0], "kind": "wrong_value"}, {"family": "hold", "key": "path.hold_time", "gt": "positive", "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}, {"family": "radius", "key": "path.radius", "gt": 0.4, "llm": 0.08, "kind": "wrong_value"}, {"family": "speed", "key": "path.speed", "gt": 0.2, "llm": 1.2, "kind": "wrong_value"}]}]

### `shake_head_no` / humanoid  misspec=True underspec=False

GT:
```
pose(head=center, torso=mid)
+ move(head, pan, [-25,25],
  rep=3, spd=2.0, hold=0.1)
```
LLM:
```
pose(head=center) + move(head, pan, [-25, 25], rep=2, spd=2.2, hold=0.1)
```
misspec: [{"type": "movement", "role": "core", "diffs": [{"family": "speed", "key": "movement.speed", "gt": 2.0, "llm": 2.2, "kind": "wrong_value"}]}]

### `shake_head_no` / stretch  misspec=False underspec=False

GT:
```
pose(dir=still, torso=mid) + move(head, pan, [-30, 30], rep=2, spd=2.2, hold=0.1)
```
LLM:
```
pose(dir=still, torso=mid) + move(head, pan, [-30, 30], rep=2, spd=2.2, hold=0.1)
```

### `shake_head_no` / roomba  misspec=True underspec=False

GT:
```
path(base, arc, r=0,
  deg=±30, rep=3, spd=2.0)
```
LLM:
```
pose(dir=front) + path(base, arc, r=0, deg=[35, -70, 35], rep=2, spd=2.8)
```
misspec: [{"type": "path", "role": "core", "diffs": [{"family": "extent", "key": "path.degrees", "gt": [-30.0, 30.0], "llm": [35.0, 35.0], "kind": "wrong_value"}, {"family": "speed", "key": "path.speed", "gt": 2.0, "llm": 2.8, "kind": "wrong_value"}]}]

### `raise_fists_victory` / humanoid  misspec=True underspec=True

GT:
```
pose(dir=up, grip=horiz|palmback, z=65, x=65, y=40, left_arm=mirror, elbow=T, bank=F) + move(shoulder, z, [20], rep=1, spd=3.0, hold=0.2)
```
LLM:
```
pose(dir=up, grip=closed, z=90, x=40, elbow=F, left_arm=mirror, torso=tall) + hold(t=1.5)
```
underspec: [{"gt_i": 1, "type": "movement", "role": "core", "n_attrs": 6}]
misspec: [{"type": "pose", "role": "core", "diffs": [{"family": "elbow_bended", "key": "pose.elbow_bended", "gt": true, "llm": false, "kind": "wrong_value"}, {"family": "facing", "key": "pose.gripper_orientation", "gt": "horizontal|palmback", "llm": "vertical", "kind": "wrong_value"}, {"family": "location", "key": "pose.torso_height", "gt": "mid", "llm": "high", "kind": "wrong_value"}, {"family": "use_feasible_bank", "key": "pose.use_feasible_bank", "gt": false, "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}, {"family": "location", "key": "pose.x", "gt": 65, "llm": 40, "kind": "wrong_value"}, {"family": "location", "key": "pose.y", "gt": 40, "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}, {"family": "location", "key": "pose.z", "gt": 65, "llm": 90, "kind": "wrong_value"}]}]

### `raise_fists_victory` / stretch  misspec=True underspec=True

GT:
```
pose(dir=up, grip=vert, torso=mid, z=50) + path(base, arc, r=0, deg=720, spd=4.2) + path(base, arc, r=0.45, deg=40, spd=3.5) + path(base, line, x=0.08, spd=3.2) + hold(t=1.0)
```
LLM:
```
pose(dir=up, grip=vert, torso=mid, z=70) + move(torso, height, high, spd=2.8) + hold(t=1.5)
```
underspec: [{"gt_i": 1, "type": "path", "role": "core", "n_attrs": 5}, {"gt_i": 2, "type": "path", "role": "hold", "n_attrs": 5}, {"gt_i": 3, "type": "path", "role": "unknown", "n_attrs": 4}]
misspec: [{"type": "pose", "role": "core", "diffs": [{"family": "location", "key": "pose.z", "gt": 50, "llm": 70, "kind": "wrong_value"}]}, {"type": "hold", "role": "unknown", "diffs": [{"family": "t", "key": "t", "gt": 1.0, "llm": 1.5, "kind": "wrong_value"}]}]

### `raise_fists_victory` / roomba  misspec=True underspec=True

GT:
```
path(base, arc, r=0, deg=720, spd=4.5) + path(base, arc, r=0.55, deg=45, spd=3.8) + path(base, line, x=0.08, spd=3.5)
```
LLM:
```
pose(dir=front) + path(base, arc, r=0, deg=720, spd=4.0) + hold(t=1.5)
```
underspec: [{"gt_i": 1, "type": "path", "role": "core", "n_attrs": 5}]
misspec: [{"type": "path", "role": "submotion", "diffs": [{"family": "speed", "key": "path.speed", "gt": 4.5, "llm": 4.0, "kind": "wrong_value"}]}, {"type": "path", "role": "unknown", "diffs": [{"family": "speed", "key": "path.speed", "gt": 3.5, "llm": 0.5, "kind": "wrong_value"}, {"family": "location", "key": "path.x", "gt": 0.08, "llm": 0.0, "kind": "wrong_value"}]}]

### `gaze_shift_avoid` / humanoid  misspec=True underspec=True

GT:
```
pose(head=center, torso=mid) + move(head, pan, -45, spd=1.0, hold=1.5) & move(torso, pan, +20, spd=1.0)
```
LLM:
```
pose(head=center) + move(head, pan, -45, spd=0.8) + hold(t=1.5)
```
underspec: [{"gt_i": 2, "type": "movement", "role": "core", "n_attrs": 5}]
misspec: [{"type": "movement", "role": "core", "diffs": [{"family": "hold", "key": "movement.hold_time", "gt": "positive", "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}, {"family": "speed", "key": "movement.speed", "gt": 1.0, "llm": 0.8, "kind": "wrong_value"}]}]

### `gaze_shift_avoid` / stretch  misspec=False underspec=True

GT:
```
pose(dir=still, torso=mid) + move(head, pan, 45, spd=0.7) & path(base, arc, r=0, deg=-35, spd=0.7) + hold(t=1.3)
```
LLM:
```
pose(dir=still, torso=mid) + move(head, pan, 45, spd=0.7) + hold(t=1.3)
```
underspec: [{"gt_i": 2, "type": "path", "role": "hold", "n_attrs": 5}]

### `gaze_shift_avoid` / roomba  misspec=True underspec=False

GT:
```
path(base, arc, r=0, deg=120, spd=1.0, hold=1.5)
```
LLM:
```
pose(dir=front) + path(base, arc, r=0, deg=75, spd=0.9) + hold(t=1.5)
```
misspec: [{"type": "path", "role": "core", "diffs": [{"family": "extent", "key": "path.degrees", "gt": 120.0, "llm": 75.0, "kind": "wrong_value"}, {"family": "hold", "key": "path.hold_time", "gt": "positive", "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}, {"family": "speed", "key": "path.speed", "gt": 1.0, "llm": 0.9, "kind": "wrong_value"}]}]

### `hands_up_surrender` / humanoid  misspec=True underspec=False

GT:
```
pose(dir=up, grip=horiz|palmfront, z=85, y=20,
  left_arm=mirror)
+ hold(t=1.5)
```
LLM:
```
pose(dir=up, left_arm=mirror, grip=horiz|palmfront, z=95, elbow=T) + hold(t=1.5)
```
misspec: [{"type": "pose", "role": "core", "diffs": [{"family": "location", "key": "pose.y", "gt": 20, "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}, {"family": "location", "key": "pose.z", "gt": 85, "llm": 95, "kind": "wrong_value"}]}]

### `hands_up_surrender` / stretch  misspec=True underspec=True

GT:
```
pose(dir=up, grip=horiz, torso=mid, z=80) + path(base, arc, r=0, deg=90, spd=2.0) + hold(t=0.3) + path(base, arc, r=0, deg=-180, spd=2.0) + hold(t=0.3) + path(base, arc, r=0, deg=90, spd=2.0) + path(base, line, x=-0.20, spd=1.1) + hold(t=1.0)
```
LLM:
```
pose(dir=front, grip=horiz, torso=mid, z=60) + path(ee, arc, plane=xz, r=0.25, sweep=90, spd=1.8) + hold(t=1.5)
```
underspec: [{"gt_i": 1, "type": "path", "role": "core", "n_attrs": 5}, {"gt_i": 2, "type": "hold", "role": "core", "n_attrs": 2}, {"gt_i": 3, "type": "path", "role": "hold", "n_attrs": 5}, {"gt_i": 4, "type": "hold", "role": "unknown", "n_attrs": 2}, {"gt_i": 6, "type": "path", "role": "unknown", "n_attrs": 4}]
misspec: [{"type": "pose", "role": "submotion", "diffs": [{"family": "location", "key": "pose.arm_position", "gt": "up", "llm": "front", "kind": "wrong_value"}, {"family": "location", "key": "pose.z", "gt": 80, "llm": 60, "kind": "wrong_value"}]}, {"type": "path", "role": "unknown", "diffs": [{"family": "path_shape", "key": "path.mode", "gt": "base", "llm": "ee", "kind": "wrong_value"}, {"family": "radius", "key": "path.radius", "gt": 0.0, "llm": 0.25, "kind": "wrong_value"}, {"family": "speed", "key": "path.speed", "gt": 2.0, "llm": 1.8, "kind": "wrong_value"}]}, {"type": "hold", "role": "unknown", "diffs": [{"family": "t", "key": "t", "gt": 1.0, "llm": 1.5, "kind": "wrong_value"}]}]

### `hands_up_surrender` / roomba  misspec=True underspec=True

GT:
```
path(base, arc, r=0, deg=90, spd=2.0) + hold(t=0.35) + path(base, arc, r=0, deg=-180, spd=2.0) + hold(t=0.35) + path(base, arc, r=0, deg=90, spd=2.0) + path(base, line, x=-0.22, spd=1.1) + hold(t=1.0)
```
LLM:
```
pose(dir=front) + path(base, line, x=-0.15, spd=2.5) + hold(t=1.8)
```
underspec: [{"gt_i": 0, "type": "path", "role": "core", "n_attrs": 5}, {"gt_i": 1, "type": "path", "role": "submotion", "n_attrs": 5}, {"gt_i": 2, "type": "path", "role": "core", "n_attrs": 5}, {"gt_i": 3, "type": "path", "role": "unknown", "n_attrs": 5}, {"gt_i": 4, "type": "path", "role": "unknown", "n_attrs": 5}]
misspec: [{"type": "path", "role": "unknown", "diffs": [{"family": "speed", "key": "path.speed", "gt": 1.1, "llm": 2.5, "kind": "wrong_value"}, {"family": "location", "key": "path.x", "gt": -0.22, "llm": -0.15, "kind": "wrong_value"}]}]

### `cover_yawn_sleepy` / humanoid  misspec=True underspec=False

GT:
```
pose(head=up, dir=up, grip=horiz,
  z=65, x=70, y=50, torso=mid)
+ move(head, tilt, +15, spd=0.5, hold=1.0)
+ move(torso, height, low, spd=0.6)
```
LLM:
```
pose(dir=in, grip=vert|palmin, elbow=T, z=80, y=75, x=60, t=0.2) + move(torso, x, -0.05, spd=0.6) + move(head, tilt, -15, spd=0.5) + hold(t=1.0)
```
misspec: [{"type": "pose", "role": "core", "diffs": [{"family": "location", "key": "pose.arm_position", "gt": "up", "llm": "in", "kind": "wrong_value"}, {"family": "facing", "key": "pose.gripper_orientation", "gt": "horizontal", "llm": "vertical|palmin", "kind": "wrong_value"}, {"family": "location", "key": "pose.head", "gt": "up", "llm": "center", "kind": "wrong_value"}, {"family": "location", "key": "pose.x", "gt": 70, "llm": 60, "kind": "wrong_value"}, {"family": "location", "key": "pose.y", "gt": 50, "llm": 75, "kind": "wrong_value"}, {"family": "location", "key": "pose.z", "gt": 65, "llm": 80, "kind": "wrong_value"}]}, {"type": "movement", "role": "core", "diffs": [{"family": "hold", "key": "movement.hold_time", "gt": "positive", "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}, {"family": "axis", "key": "movement.joints[0].axis", "gt": "tilt", "llm": "height", "kind": "wrong_value"}, {"family": "extent", "key": "movement.joints[0].degrees", "gt": [0.0, 15.0], "llm": [0.18, 0.1795], "kind": "wrong_value"}, {"family": "joint", "key": "movement.joints[0].joint", "gt": "head", "llm": "torso", "kind": "wrong_value"}, {"family": "speed", "key": "movement.speed", "gt": 0.5, "llm": 0.6, "kind": "wrong_value"}]}, {"type": "movement", "role": "core", "diffs": [{"family": "axis", "key": "movement.joints[0].axis", "gt": "height", "llm": "tilt", "kind": "wrong_value"}, {"family": "location", "key": "movement.joints[0].height", "gt": "low", "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}, {"family": "joint", "key": "movement.joints[0].joint", "gt": "torso", "llm": "head", "kind": "wrong_value"}, {"family": "speed", "key": "movement.speed", "gt": 0.6, "llm": 0.5, "kind": "wrong_value"}]}]

### `cover_yawn_sleepy` / stretch  misspec=False underspec=False

GT:
```
pose(dir=front, grip=horiz, z=60, x=25) + path(ee, arc, plane=xz, r=0.15, sweep=90, spd=0.6) + move(head, tilt, -20, spd=0.5) + hold(t=1.5)
```
LLM:
```
pose(dir=front, grip=horiz, z=60, x=25) + path(ee, arc, plane=xz, r=0.15, sweep=90, spd=0.6) + move(head, tilt, -20, spd=0.5) + hold(t=1.5)
```

### `cover_yawn_sleepy` / roomba  misspec=True underspec=True

GT:
```
pose(dir=front) + path(base, arc, r=0.4, deg=10, spd=1) + path(base, arc, r=0.4, deg=-25, spd=0.5) + path(base, arc, r=0.4, deg=-10, spd=0.2, hold=1.5)
```
LLM:
```
pose(dir=front) + path(base, arc, r=0.4, deg=-25, spd=0.5) + hold(t=1.5)
```
underspec: [{"gt_i": 1, "type": "path", "role": "submotion", "n_attrs": 5}]
misspec: [{"type": "path", "role": "submotion", "diffs": [{"family": "extent", "key": "path.degrees", "gt": -10.0, "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}, {"family": "radius", "key": "path.radius", "gt": 0.4, "llm": "__ABSENT__", "kind": "gt_slot_missing_in_llm"}, {"family": "path_shape", "key": "path.shape", "gt": "arc", "llm": "line", "kind": "wrong_value"}, {"family": "speed", "key": "path.speed", "gt": 0.2, "llm": 0.5, "kind": "wrong_value"}]}]
