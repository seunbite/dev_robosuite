# Vector-only Drag-and-Draw

The robot starts in a declared neutral state. Each gesture is represented only as trajectories of movable handles:

```yaml
version: '2.0'
cue: nod_yes
platform: humanoid
fps: 20
trajectories:
  - handle: head
    vectors:
      - [-0.0040, 0.0000, 1.2240, 0]
      - [ 0.0100, 0.0000, 1.1700, 4]
      - [-0.0100, 0.0000, 1.2800, 8]
      - [-0.0040, 0.0000, 1.2240, 14]
```

Every vector is `[x, y, z, frame]` in absolute robot-local meters. There are no `shape`, `axis`, `speed`, `repetition`, `duration`, or `hold_time` attributes:

- Speed is determined by displacement per frame.
- A curve is explicitly sampled with intermediate xyz vectors.
- A hold repeats the endpoint at a later frame.
- Concurrent handles use the same frame numbers.

## Embodiment point budgets

The abstraction uses the smallest useful number of Cartesian point tracks:

- Humanoid hand: `ee + fingertip + palm_normal` (3)
- Tiago hand: `ee + fingertip` (2)
- Single-arm: `upper_arm + elbow + forearm + wrist + ee + fingertip + palm_normal` (7)
- Stretch: `ee + fingertip` (2)
- Roomba: `base + base_front` (2)

Humanoid hand orientation uses a complete three-point frame:

```yaml
trajectories:
  - handle: right_ee
    vectors:
      - [0.0012, -0.2140, 0.7432, 0]
      - [0.0200, -0.0700, 1.1700, 12]
  - handle: right_fingertip
    vectors:
      - [0.0357, -0.2325, 0.6512, 0]
      - [0.0700, -0.0700, 1.2500, 12]
  - handle: right_palm_normal
    vectors:
      - [0.0762, -0.2114, 0.7708, 0]
      - [0.1000, -0.0700, 1.1700, 12]
```

`right_ee` is always the authoritative position target. The renderer normalizes
`right_fingertip - right_ee`, projects `right_palm_normal - right_ee` orthogonal
to that finger axis, and uses both only as soft direction targets. The three
tracks may use different keyframes. Degenerate axes fall back to the last valid
frame, and kinematic conflicts preserve the hand center while approximating the
two directions. The same rules apply on the left. Other handles have no
orientation anchors.

Tiago normalizes only `fingertip - ee`; gripper roll is intentionally omitted.
For single-arm, upper-arm, elbow, forearm, wrist, and EE are strong Cartesian
chain constraints, while
`fingertip - ee` and `palm_normal - ee` define full gripper orientation. The
renderer uses continuation and deterministic multi-start IK to avoid selecting
a wrong arm branch.
Stretch likewise uses `fingertip - ee` to express the bent gripper direction.

## Files

- Neutral handle coordinates and workspace bounds: `data/seed/yml/drag_draw_embodiment.yaml`
- Conversion, interpolation and renderer bridge: `adhoc/generation/robotarm/drag_draw.py`
- Per-robot prompts: `data/seed/prompt/drag_draw/*.txt`
- Converted 25 few-shots: `data/seed/shots/drag_draw/fewshot_cues_yml_5.yaml`
- Comparison gallery: `data/results/drag_draw/fewshot_roundtrip/gallery_drag_draw_roundtrip.html`

Roomba keeps its original two-point planar frame: `base` is its center and
`base_front` is a point 0.15 m ahead. Their relative vector recovers yaw.

## Commands

```bash
/usr/local/bin/python3 adhoc/generation/robotarm/drag_draw.py prompts
/usr/local/bin/python3 adhoc/generation/robotarm/drag_draw.py convert
/usr/local/bin/python3 adhoc/generation/robotarm/drag_draw.py render --force
/usr/local/bin/python3 adhoc/generation/robotarm/drag_draw.py render-file --input gesture.yaml --output gesture.gif
/usr/local/bin/python3 -m pytest -q tests/test_drag_draw.py
```
