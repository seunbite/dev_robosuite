#!/usr/bin/env python3
"""Render pose/movement/path schema visualizations for TIAGo (Google Robot proxy).

Produces:
  - PNGs: pose parameter variations (torso_height, arm_position, gripper_orientation, head)
  - PNGs: movement joint ranges (shoulder, elbow, wrist, torso, head)
  - GIFs: path types (line with x/y, arc with radius/degrees)
  - HTML: combined dashboard
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

local_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if local_root not in sys.path:
    sys.path.insert(0, local_root)

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)

OUT_DIR = os.path.join(local_root, "data", "results", "render", "mobile_manip_schema")
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 512, 512
CAM = "frontview"
CAM_SIDE = "sideview"

# ── qpos index map ───────────────────────────────────────────────────────
QI_FWD = 0
QI_SIDE = 1
QI_YAW = 2
QI_TORSO = 3
QI_HEAD_PAN = 4   # +val = robot looks to its right (left in frontview)
QI_HEAD_TILT = 5  # +val = robot looks up
QI_R_ARM = slice(6, 12)   # arm_right joints 1-6
QI_L_ARM = slice(18, 24)

TORSO_PRESETS = {"low": 0.05, "mid": 0.18, "high": 0.34}

# ── Head presets (corrected directions) ──────────────────────────────────
# pan:  negative = robot looks LEFT,  positive = robot looks RIGHT
# tilt: positive = look UP,  negative = look DOWN
HEAD_PRESETS = {
    "center": [0.0,   0.0],
    "left":   [-0.8,  0.0],
    "right":  [0.8,   0.0],
    "up":     [0.0,   0.5],
    "down":   [0.0,  -0.6],
}

# ── Gripper orientation via arm_3 (elbow pitch) ─────────────────────────
GRIPPER_ORIENT_PRESETS = {
    "horizontal": 1.57,
    "vertical":   0.0,
}


def _make_env():
    arm_cfg = suite.load_part_controller_config(default_controller="OSC_POSE")
    ctrl = refactor_composite_controller_config(arm_cfg, "Tiago", ["right", "left"])
    env = suite.make(
        env_name="EmptySpace",
        robots="Tiago",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=CAM,
        camera_heights=H,
        camera_widths=W,
        control_freq=20,
        controller_configs=ctrl,
    )
    env.reset()
    try:
        for cname in [CAM, CAM_SIDE, "birdview"]:
            cid = env.sim.model.camera_name2id(cname)
            env.sim.model.cam_fovy[cid] = 55.0
    except Exception:
        pass
    return env


def _set_fov(env, fov: float):
    for cname in [CAM, CAM_SIDE, "birdview"]:
        try:
            cid = env.sim.model.camera_name2id(cname)
            env.sim.model.cam_fovy[cid] = fov
        except Exception:
            pass


def _default_qpos(env):
    q = env.sim.data.qpos.copy()
    q[:3] = 0
    q[QI_TORSO] = TORSO_PRESETS["mid"]
    q[QI_HEAD_PAN] = 0.0
    q[QI_HEAD_TILT] = 0.0
    q[QI_R_ARM] = [0.0, -0.3, 0.0, 0.3, 0.0, 0.0]
    q[QI_L_ARM] = [0.0, -0.3, 0.0, 0.3, 0.0, 0.0]
    return q


def _apply_and_capture(env, qpos, camera=CAM, steps=4):
    env.sim.data.qpos[:] = qpos
    env.sim.forward()
    action = np.zeros(env.action_dim)
    for _ in range(steps):
        env.step(action)
    frame = env.sim.render(camera_name=camera, width=W, height=H, depth=False)
    return Image.fromarray(np.flipud(frame))


def _label(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([(0, H - 36), (W, H)], fill=(0, 0, 0, 180))
    draw.text((10, H - 32), text, fill=(255, 255, 255), font=font)
    return img


def _make_tile(images: list[Image.Image], labels: list[str], cols: int, title: str = "") -> Image.Image:
    rows = math.ceil(len(images) / cols)
    tile_w = W * cols
    title_h = 50 if title else 0
    tile_h = H * rows + title_h
    canvas = Image.new("RGB", (tile_w, tile_h), (246, 248, 251))
    if title:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        except Exception:
            font = ImageFont.load_default()
        draw.text((20, 10), title, fill=(31, 35, 40), font=font)
    for i, (img, lbl) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        labeled = _label(img.copy(), lbl)
        canvas.paste(labeled, (c * W, r * H + title_h))
    return canvas


# ═══════════════════════════════════════════════════════════════════════════
# 1. POSE parameter tiles
# ═══════════════════════════════════════════════════════════════════════════

def render_pose_tiles(env):
    print("\n=== Rendering POSE parameter tiles ===")
    base = _default_qpos(env)

    # 1a. torso_height
    imgs, lbls = [], []
    for name, val in TORSO_PRESETS.items():
        q = base.copy()
        q[QI_TORSO] = val
        imgs.append(_apply_and_capture(env, q))
        lbls.append(f"torso_height: {name}")
    tile = _make_tile(imgs, lbls, 3, "pose.torso_height")
    tile.save(os.path.join(OUT_DIR, "pose_torso_height.png"))
    print(f"  torso_height: {len(imgs)} variations")

    # 1b. arm_position (brute-force 90° steps on shoulder joints 1 & 2)
    arm_imgs, arm_lbls = _render_arm_brute_force(env)
    if arm_imgs:
        tile = _make_tile(arm_imgs, arm_lbls, 4,
                          "pose.arm_position (right arm, 90° steps)")
        tile.save(os.path.join(OUT_DIR, "pose_arm_position.png"))
        print(f"  arm_position: {len(arm_imgs)} named positions")

    # 1c. head
    imgs, lbls = [], []
    for name, angles in HEAD_PRESETS.items():
        q = base.copy()
        q[QI_TORSO] = TORSO_PRESETS["high"]
        q[QI_HEAD_PAN] = angles[0]
        q[QI_HEAD_TILT] = angles[1]
        imgs.append(_apply_and_capture(env, q))
        lbls.append(f"head: {name}")
    tile = _make_tile(imgs, lbls, 5, "pose.head")
    tile.save(os.path.join(OUT_DIR, "pose_head.png"))
    print(f"  head: {len(imgs)} variations")

    # 1d. gripper_orientation (horizontal / vertical via arm_3)
    # Use wider FOV for this shot so grip is visible
    _set_fov(env, 75.0)
    imgs, lbls = [], []
    for name, arm3_val in GRIPPER_ORIENT_PRESETS.items():
        q = base.copy()
        q[QI_TORSO] = TORSO_PRESETS["high"]
        q[QI_R_ARM] = [1.0, 0.0, arm3_val, 0.8, 0.0, 0.0]
        imgs.append(_apply_and_capture(env, q, camera=CAM_SIDE))
        lbls.append(f"gripper: {name}")
    tile = _make_tile(imgs, lbls, 2, "pose.gripper_orientation (via elbow pitch)")
    tile.save(os.path.join(OUT_DIR, "pose_gripper_orientation.png"))
    _set_fov(env, 55.0)  # restore
    print(f"  gripper_orientation: {len(imgs)} variations")


def _render_arm_brute_force(env):
    """Brute-force arm_1 × arm_2 at 90° steps, label by EE direction from frontview.

    Verified EE positions (relative to shoulder at ~(-0.04, -0.19, 0.82)):
      a1=-90,a2=-90 → up (z+1.0)          a1=0,a2=-90 → up (dup)     a1=90,a2=-90 → up (dup)
      a1=-90,a2=0   → back (x-0.93,z+0.35) a1=0,a2=0  → right (y-0.93) a1=90,a2=0  → front (x+0.93)
      a1=-90,a2=90  → down+back             a1=0,a2=90 → down+right   a1=90,a2=90  → down+front
    """
    base = _default_qpos(env)
    base[QI_TORSO] = TORSO_PRESETS["high"]

    # Curated set: skip 2 duplicate "up" combos, keep unique 7 positions
    ARM_POSITIONS = [
        ("up",         -90, -90),
        ("back",       -90,   0),
        ("down+back",  -90,  90),
        ("right",        0,   0),
        ("down+right",   0,  90),
        ("front",       90,   0),
        ("down+front",  90,  90),
    ]

    imgs, lbls = [], []
    for direction, a1_deg, a2_deg in ARM_POSITIONS:
        q = base.copy()
        q[6] = np.deg2rad(a1_deg)
        q[7] = np.deg2rad(a2_deg)
        q[8] = 0.0
        q[9] = 0.3
        q[10] = 0.0
        q[11] = 0.0
        img = _apply_and_capture(env, q)
        lbls.append(f"{direction}  (a1={a1_deg:+d}° a2={a2_deg:+d}°)")
        imgs.append(img)

    return imgs, lbls


# ═══════════════════════════════════════════════════════════════════════════
# 2. MOVEMENT joint range tiles
# ═══════════════════════════════════════════════════════════════════════════

MOVEMENT_JOINTS = {
    "shoulder (arm_1: pitch)": (6,  [-1.0, 0.0, 1.5]),
    "shoulder (arm_2: roll)":  (7,  [-0.9, 0.0, 1.2]),
    "elbow (arm_3: pitch)":    (8,  [0.0,  0.8, 1.8]),
    "elbow (arm_4: roll)":     (9,  [0.0,  0.8, 2.0]),
    "wrist (arm_5: pitch)":    (10, [-1.5, 0.0, 1.5]),
    "wrist (arm_6: roll)":     (11, [-1.2, 0.0, 1.2]),
    "torso (height)":          (3,  [0.05, 0.18, 0.34]),
    "head (pan)":              (4,  [-0.8, 0.0, 0.8]),
    "head (tilt)":             (5,  [-0.6, 0.0, 0.5]),
}


def render_movement_tiles(env):
    print("\n=== Rendering MOVEMENT joint range tiles ===")
    base = _default_qpos(env)
    base[QI_TORSO] = TORSO_PRESETS["high"]
    base[QI_R_ARM] = [1.0, 0.0, 0.0, 0.8, 0.0, 0.0]

    all_imgs, all_lbls = [], []
    for jname, (qi, vals) in MOVEMENT_JOINTS.items():
        for vi, v in enumerate(vals):
            q = base.copy()
            q[qi] = v
            phase = ["min", "mid", "max"][vi]
            img = _apply_and_capture(env, q)
            all_imgs.append(img)
            all_lbls.append(f"{jname} [{phase}]={v:.2f}")

    tile = _make_tile(all_imgs, all_lbls, 3, "movement joints: min / mid / max")
    tile.save(os.path.join(OUT_DIR, "movement_joints.png"))
    print(f"  {len(MOVEMENT_JOINTS)} joints × 3 = {len(all_imgs)} frames")


# ═══════════════════════════════════════════════════════════════════════════
# 3. PATH GIFs — unified line(x,y,speed) + arc(radius,degrees,speed)
# ═══════════════════════════════════════════════════════════════════════════

def _render_path_gif(env, name: str, update_fn, n_frames: int = 60, camera=CAM):
    base = _default_qpos(env)
    base[QI_TORSO] = TORSO_PRESETS["high"]
    frames = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        q = base.copy()
        update_fn(q, t)
        env.sim.data.qpos[:] = q
        env.sim.forward()
        frame = env.sim.render(camera_name=camera, width=W, height=H, depth=False)
        img = Image.fromarray(np.flipud(frame))
        img = _label(img, f"path: {name}  t={t:.2f}")
        frames.append(img)
    path = os.path.join(OUT_DIR, f"path_{name}.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0)
    return path


def render_path_gifs(env):
    print("\n=== Rendering PATH GIFs ===")

    # --- line(x, y, speed) ---
    # Forward (x=1.5, y=0)
    def line_forward(q, t):
        q[QI_FWD] = t * 1.5

    # Backward (x=-1.0, y=0)
    def line_backward(q, t):
        q[QI_FWD] = -t * 1.0

    # Lateral right (x=0, y=-0.8)
    def line_lateral_right(q, t):
        q[QI_SIDE] = -t * 0.8

    # Diagonal (x=1.0, y=0.5)
    def line_diagonal(q, t):
        q[QI_FWD] = t * 1.0
        q[QI_SIDE] = t * 0.5

    # --- arc(radius, degrees, speed) ---
    # Wide arc left (radius=1.0, degrees=135)
    def arc_wide_left(q, t):
        angle = t * math.radians(135)
        r = 1.0
        q[QI_FWD] = r * math.sin(angle)
        q[QI_SIDE] = r * (1 - math.cos(angle))
        q[QI_YAW] = angle

    # Tight arc right (radius=0.5, degrees=-135)
    def arc_tight_right(q, t):
        angle = -t * math.radians(135)
        r = 0.5
        q[QI_FWD] = r * math.sin(-angle)
        q[QI_SIDE] = -r * (1 - math.cos(-angle))
        q[QI_YAW] = angle

    # In-place rotation = arc(radius=0, degrees=360)
    def arc_inplace_360(q, t):
        q[QI_YAW] = t * math.pi * 2

    # Small arc (radius=0.3, degrees=90)
    def arc_small_90(q, t):
        angle = t * math.radians(90)
        r = 0.3
        q[QI_FWD] = r * math.sin(angle)
        q[QI_SIDE] = r * (1 - math.cos(angle))
        q[QI_YAW] = angle

    BIRD = "birdview"
    paths_spec = [
        # line(x, y, speed) examples — all birdview
        ("line_x1.5_y0",       line_forward,       60, BIRD,
         "line(x=1.5, y=0) — forward"),
        ("line_x-1_y0",        line_backward,      60, BIRD,
         "line(x=-1.0, y=0) — backward"),
        ("line_x0_y-0.8",      line_lateral_right,  50, BIRD,
         "line(x=0, y=-0.8) — strafe right"),
        ("line_x1_y0.5",       line_diagonal,       60, BIRD,
         "line(x=1.0, y=0.5) — diagonal"),

        # arc(radius, degrees, speed) examples — all birdview
        ("arc_r1_d135",        arc_wide_left,       80, BIRD,
         "arc(radius=1.0, degrees=135) — wide left"),
        ("arc_r0.5_d-135",     arc_tight_right,     80, BIRD,
         "arc(radius=0.5, degrees=-135) — tight right"),
        ("arc_r0_d360",        arc_inplace_360,     80, BIRD,
         "arc(radius=0, degrees=360) — in-place rotation"),
        ("arc_r0.3_d90",       arc_small_90,        60, BIRD,
         "arc(radius=0.3, degrees=90) — small turn"),
    ]

    for fname, fn, nf, cam, desc in paths_spec:
        gif = _render_path_gif(env, fname, fn, n_frames=nf, camera=cam)
        print(f"  {desc} → {os.path.basename(gif)}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. HTML dashboard
# ═══════════════════════════════════════════════════════════════════════════

def build_html():
    print("\n=== Building HTML dashboard ===")
    html = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mobile Manipulator (TIAGo) Motion Schema</title>
<style>
:root{--bg:#f6f8fb;--surface:#fff;--border:#d0d7de;--text:#1f2328;--muted:#59636e;--accent:#0969da;--green:#1a7f37;--purple:#8250df}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,'SF Pro Text','Segoe UI',sans-serif;background:var(--bg);color:var(--text)}
.wrap{max-width:1600px;margin:0 auto;padding:24px}
h1{margin:0 0 8px;font-size:28px}
.sub{color:var(--muted);margin:0 0 24px}
.section{margin:0 0 32px}
.section h2{margin:0 0 12px;font-size:22px;border-bottom:2px solid var(--border);padding-bottom:6px}
.section h3{margin:16px 0 8px;font-size:17px;color:var(--accent)}
.desc{color:var(--muted);font-size:14px;margin:0 0 10px}
.tile{display:block;max-width:100%;border:1px solid var(--border);border-radius:12px;margin:0 0 16px}
.gif-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;margin:0 0 16px}
.gif-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden;padding:12px}
.gif-card img{display:block;width:100%;border-radius:8px}
.gif-card .label{font-weight:600;margin:8px 0 0;font-size:14px}
.gif-card .desc{font-size:13px;color:var(--muted);margin:4px 0 0}
.schema{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px;margin:0 0 16px;font-family:'SF Mono','Fira Code',monospace;font-size:13px;white-space:pre-wrap;overflow-x:auto}
.joint-table{width:100%;border-collapse:collapse;margin:0 0 16px}
.joint-table th,.joint-table td{border:1px solid var(--border);padding:8px 12px;text-align:left;font-size:14px}
.joint-table th{background:var(--surface);font-weight:600}
.joint-table tr:nth-child(even){background:#f0f3f6}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;margin:0 4px 4px 0}
.tag-pose{background:#0969da22;color:var(--accent)}
.tag-move{background:#1a7f3722;color:var(--green)}
.tag-path{background:#8250df22;color:var(--purple)}
</style>
</head>
<body>
<div class="wrap">
<h1>Mobile Manipulator Motion Schema</h1>
<p class="sub">TIAGo (Google Robot proxy) &mdash; pose / movement / path dimensions</p>

<!-- ==================== POSE ==================== -->
<div class="section">
<h2><span class="tag tag-pose">1</span> pose &mdash; Initial Stance</h2>
<div class="schema">{
  "type": "pose",
  "parameters": {
    "pose": {
      "torso_height": "low | mid | high",
      "arm_position": "&lt;direction label&gt;",
      "gripper_orientation": "horizontal | vertical",
      "head": "center | left | right | up | down"
    }
  }
}</div>

<h3>torso_height</h3>
<p class="desc">Torso lift joint controls overall robot height (0.05&ndash;0.34 m).</p>
<img class="tile" src="pose_torso_height.png">

<h3>arm_position (right arm, 90&deg; brute-force)</h3>
<p class="desc">Shoulder joints 1 &amp; 2 at -90&deg;/0&deg;/+90&deg;. Labels assigned by end-effector direction relative to robot body: up/down, front/back, left/right.</p>
<img class="tile" src="pose_arm_position.png">

<h3>head</h3>
<p class="desc">Pan (left/right, &plusmn;75&deg;) and tilt (up/down). Directions are from the robot&rsquo;s perspective.</p>
<img class="tile" src="pose_head.png">

<h3>gripper_orientation</h3>
<p class="desc">Controlled via elbow pitch (arm_3). <code>horizontal</code>: arm_3=0, <code>vertical</code>: arm_3=&pi;/2.</p>
<img class="tile" src="pose_gripper_orientation.png">
</div>

<!-- ==================== MOVEMENT ==================== -->
<div class="section">
<h2><span class="tag tag-move">2</span> movement &mdash; Joint Animation</h2>
<div class="schema">{
  "type": "movement",
  "parameters": {
    "movement": {
      "repetition": &lt;int&gt;,
      "speed": &lt;float&gt;,
      "joints": [
        { "joint": "shoulder | elbow | wrist | torso | head",
          "axis": "&lt;axis_name&gt;",
          "degrees": [&lt;lo&gt;, &lt;hi&gt;],
          "speed": &lt;optional float&gt; }
      ]
    }
  }
}</div>

<table class="joint-table">
<tr><th>joint</th><th>axis</th><th>MuJoCo joint</th><th>range</th><th>role</th></tr>
<tr><td>shoulder</td><td>pitch</td><td>arm_right_1_joint</td><td>-68&deg; ~ 90&deg;</td><td>Arm forward/backward</td></tr>
<tr><td>shoulder</td><td>roll</td><td>arm_right_2_joint</td><td>-68&deg; ~ 90&deg;</td><td>Arm lateral raise</td></tr>
<tr><td>elbow</td><td>pitch</td><td>arm_right_3_joint</td><td>-45&deg; ~ 225&deg;</td><td>Elbow bend / gripper orientation</td></tr>
<tr><td>elbow</td><td>roll</td><td>arm_right_4_joint</td><td>-23&deg; ~ 135&deg;</td><td>Forearm rotation</td></tr>
<tr><td>wrist</td><td>pitch</td><td>arm_right_5_joint</td><td>-120&deg; ~ 120&deg;</td><td>Wrist flex</td></tr>
<tr><td>wrist</td><td>roll</td><td>arm_right_6_joint</td><td>-81&deg; ~ 81&deg;</td><td>Wrist rotation</td></tr>
<tr><td>torso</td><td>height</td><td>torso_lift_joint</td><td>0.0 ~ 0.35 m</td><td>Body up/down</td></tr>
<tr><td>head</td><td>pan</td><td>head_1_joint</td><td>-75&deg; ~ 75&deg;</td><td>Look left/right</td></tr>
<tr><td>head</td><td>tilt</td><td>head_2_joint</td><td>-60&deg; ~ 45&deg;</td><td>Look up/down</td></tr>
</table>

<h3>Joint range visualization (min / mid / max)</h3>
<img class="tile" src="movement_joints.png">
</div>

<!-- ==================== PATH ==================== -->
<div class="section">
<h2><span class="tag tag-path">3</span> path &mdash; Base Navigation</h2>
<div class="schema">{
  "type": "path",
  "parameters": {
    "path": {
      "shape": "line | arc",

      // line: move in a straight direction (holonomic)
      "x": &lt;meters forward, negative=backward&gt;,
      "y": &lt;meters lateral, negative=right&gt;,
      "speed": &lt;float&gt;,

      // arc: curved or in-place rotation
      "radius": &lt;meters, 0 = in-place rotation&gt;,
      "degrees": &lt;total turn angle, negative=clockwise&gt;,
      "speed": &lt;float&gt;
    }
  }
}</div>

<table class="joint-table">
<tr><th>shape</th><th>params</th><th>description</th><th>special case</th></tr>
<tr><td>line</td><td>x (m), y (m), speed</td><td>Straight movement in any direction</td><td>x only = forward/back, y only = strafe, both = diagonal</td></tr>
<tr><td>arc</td><td>radius (m), degrees, speed</td><td>Curved path with turning</td><td>radius=0 = in-place rotation</td></tr>
</table>

<h3>line examples</h3>
<div class="gif-grid">
"""

    line_gifs = [
        ("path_line_x1.5_y0.gif", "line(x=1.5, y=0)", "Forward straight"),
        ("path_line_x-1_y0.gif", "line(x=-1.0, y=0)", "Backward straight"),
        ("path_line_x0_y-0.8.gif", "line(x=0, y=-0.8)", "Strafe right"),
        ("path_line_x1_y0.5.gif", "line(x=1.0, y=0.5)", "Diagonal forward-left"),
    ]
    for gif_name, label, desc in line_gifs:
        html += f"""<div class="gif-card">
  <img src="{gif_name}" loading="lazy">
  <div class="label">{label}</div>
  <div class="desc">{desc}</div>
</div>
"""

    html += """</div>
<h3>arc examples</h3>
<div class="gif-grid">
"""

    arc_gifs = [
        ("path_arc_r1_d135.gif", "arc(r=1.0, deg=135)", "Wide left turn"),
        ("path_arc_r0.5_d-135.gif", "arc(r=0.5, deg=-135)", "Tight right turn"),
        ("path_arc_r0_d360.gif", "arc(r=0, deg=360)", "In-place 360&deg; rotation"),
        ("path_arc_r0.3_d90.gif", "arc(r=0.3, deg=90)", "Small 90&deg; turn"),
    ]
    for gif_name, label, desc in arc_gifs:
        html += f"""<div class="gif-card">
  <img src="{gif_name}" loading="lazy">
  <div class="label">{label}</div>
  <div class="desc">{desc}</div>
</div>
"""

    html += """</div>
</div>

</div>
</body>
</html>"""

    path = os.path.join(OUT_DIR, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def main():
    env = _make_env()
    try:
        render_pose_tiles(env)
        render_movement_tiles(env)
        render_path_gifs(env)
    finally:
        env.close()

    html_path = build_html()
    print(f"\n✓ Dashboard: {html_path}")

    if sys.platform == "darwin":
        import subprocess
        subprocess.Popen(["open", html_path])


if __name__ == "__main__":
    main()
