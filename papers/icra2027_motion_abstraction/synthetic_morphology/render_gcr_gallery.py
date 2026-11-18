#!/usr/bin/env python3
"""Render Oddkin continuum attribute gallery → HTML.

Each card isolates ONE Motion Code attribute (pose / move / path) from
yml/motion_code_synthetic_continuum.yaml.
"""

from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
XML_DIR = ROOT / "xml"
OUT_DIR = ROOT / "renders"
FPS = 12
W, H = 520, 400


def _font(size: int = 17) -> ImageFont.ImageFont:
    for p in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _quat_yaw(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def _set_joint(data: mujoco.MjData, model: mujoco.MjModel, name: str, value: float) -> None:
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jnt_id < 0:
        return
    data.qpos[model.jnt_qposadr[jnt_id]] = value


def _set_base(data: mujoco.MjData, model: mujoco.MjModel, x: float, y: float, yaw: float, z: float) -> None:
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
    adr = model.jnt_qposadr[jnt_id]
    data.qpos[adr : adr + 3] = [x, y, z]
    data.qpos[adr + 3 : adr + 7] = _quat_yaw(yaw)


def _overlay(rgb: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, 0, img.width, 48], fill=(255, 255, 255, 230))
    draw.text((8, 5), title, fill=(15, 15, 20), font=_font(16))
    draw.text((8, 26), subtitle, fill=(50, 55, 65), font=_font(13))
    return np.asarray(img.convert("RGB"))


def _cam(lookat, distance, azimuth=180, elevation=-8):
    """azimuth=180 → camera on +x so robot front faces viewer."""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def _render(model, frames_fn, *, out_gif, title, subtitle, lookat, distance, base_z=0.12, azimuth=180):
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = _cam(lookat, distance, azimuth=azimuth)
    imgs = []
    for cfg in frames_fn():
        mujoco.mj_resetData(model, data)
        x, y, yaw = cfg.get("base", (0.0, 0.0, 0.0))
        _set_base(data, model, x, y, yaw, base_z)
        for jn, v in cfg.get("joints", {}).items():
            _set_joint(data, model, jn, v)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        imgs.append(_overlay(renderer.render(), title, subtitle))
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_gif, imgs, fps=FPS, loop=0)
    renderer.close()
    return out_gif


def _traj(n: int):
    for i in range(n):
        yield i / max(n - 1, 1)


def _lerp(a, b, t):
    return a + (b - a) * t


def _ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _ease_pose(home_fn, *, base=(0.0, 0.0, 0.0), extra=None, n=22, n_ease=10):
    """Lerp home → target pose, then hold (attribute-gallery clips)."""
    start_j = home_fn()
    end_j = home_fn(extra)
    start_b = (0.0, 0.0, 0.0)
    n_ease = min(n_ease, n)
    for i in range(n_ease):
        t = _ease(i / max(n_ease - 1, 1))
        j = {k: _lerp(start_j[k], end_j[k], t) for k in end_j}
        b = tuple(_lerp(start_b[i], base[i], t) for i in range(3))
        yield {"base": b, "joints": j}
    for _ in range(n - n_ease):
        yield {"base": base, "joints": end_j}


def _emit(specs, model, out_dir, title, lookat, distance, base_z):
    items = []
    for key, component, code, note, gen in specs:
        p = out_dir / f"{key}.gif"
        _render(
            model, gen, out_gif=p, title=title,
            subtitle=f"[{component}]  {code}  ·  {note}",
            lookat=lookat, distance=distance, base_z=base_z, azimuth=180,
        )
        items.append({"key": key, "component": component, "code": code, "note": note, "gif": p.name})
    return items


# -------------------- α=0 Trileg Crown --------------------

def clips_alpha0(model, out_dir):
    z = 0.12

    def home(extra=None):
        j = {
            "hip0": 0.2, "knee0": -0.6,
            "hip1": 0.15, "knee1": -0.55,
            "hip2": -0.1, "knee2": -0.65,
            "crown_spin": 0.0,
        }
        if extra:
            j.update(extra)
        return j

    specs = []

    # --- pose ---
    def pose_dir_left():
        yield from _ease_pose(home, base=(0, 0, math.pi / 2))
    specs.append(("pose_dir_left", "pose", "pose(dir=left)", "Facing only (base yaw)", pose_dir_left))

    def pose_leg0_present():
        yield from _ease_pose(home, extra={"hip0": 0.95, "knee0": -0.15})
    specs.append(("pose_leg0_present", "pose", "pose(leg0=present)", "Front leg present only", pose_leg0_present))

    def pose_crown_quarter():
        yield from _ease_pose(home, extra={"crown_spin": math.pi / 2})
    specs.append(("pose_crown_quarter", "pose", "pose(crown=quarter)", "Crown absolute angle only", pose_crown_quarter))

    # --- move ---
    def move_hip0():
        for t in _traj(36):
            s = math.sin(2 * math.pi * t * 2)
            yield {"base": (0, 0, 0), "joints": home({"hip0": 0.2 + 0.55 * s})}
    specs.append(("move_hip0", "move", "move(hip0, yaw, ±)", "Single hip joint beat", move_hip0))

    def move_base_yaw():
        for t in _traj(36):
            s = math.sin(2 * math.pi * t * 3)
            yield {"base": (0, 0, 0.55 * s), "joints": home()}
    specs.append(("move_base_yaw", "move", "move(base, yaw, ±, rep=3)", "Whole-body yaw; legs planted", move_base_yaw))

    def move_crown():
        for t in _traj(36):
            yield {"base": (0, 0, 0), "joints": home({"crown_spin": 2 * math.pi * t})}
    specs.append(("move_crown", "move", "move(crown, yaw, 360)", "Crown spin beat", move_crown))

    # --- path ---
    def path_base_line():
        for t in _traj(28):
            yield {"base": (0.28 * t, 0, 0), "joints": home()}
    specs.append(("path_base_line", "path", "path(base, line, x=+)", "Base translation only", path_base_line))

    def path_base_spin_r0():
        for t in _traj(36):
            yield {"base": (0, 0, 2 * math.pi * t), "joints": home()}
    specs.append(("path_base_spin_r0", "path", "path(base, arc, r=0, deg=360)", "Base in-place spin (r=0)", path_base_spin_r0))

    return _emit(specs, model, out_dir, "Oddkin α=0  Trileg Crown",
                 lookat=(0.0, 0.0, 0.14), distance=1.45, base_z=z)


# -------------------- α=0.5 Canted Spine Crab --------------------

def clips_alpha05(model, out_dir):
    z = 0.11

    def home(extra=None):
        j = {
            "flap_l": 0.2, "flap_r": -0.2,
            "v1_pitch": 0.1, "v1_yaw": 0.0,
            "v2_pitch": 0.08, "v2_twist": 0.15,
            "collar_spin": 0.0,
            "v3_pitch": 0.05,
            "pin_r0": 0.35, "pin_r1": -0.7, "pin_r2": 0.1,
            "pin_l0": -0.4, "pin_l1": -0.3,
            "antenna": 0.3,
        }
        if extra:
            j.update(extra)
        return j

    specs = []

    # --- pose ---
    def pose_torso_low():
        yield from _ease_pose(home, extra={
            "v1_pitch": 0.75, "v2_pitch": 0.65, "v3_pitch": 0.7,
        }, n=20)
    specs.append(("pose_torso_low", "pose", "pose(torso=low)", "Spine height only → folded", pose_torso_low))

    def pose_torso_high():
        yield from _ease_pose(home, extra={
            "v1_pitch": -0.45, "v2_pitch": -0.4, "v3_pitch": -0.5,
        }, n=20)
    specs.append(("pose_torso_high", "pose", "pose(torso=high)", "Spine height only → upright", pose_torso_high))

    def pose_dir_left():
        yield from _ease_pose(home, extra={
            "v1_yaw": 0.55, "v2_twist": 0.7,
        }, n=20)
    specs.append(("pose_dir_left", "pose", "pose(dir=left)", "Torso twist facing only", pose_dir_left))

    def pose_long_front():
        yield from _ease_pose(home, extra={
            "pin_r0": 0.1, "pin_r1": -0.25, "pin_r2": 0.0,
        }, n=20)
    specs.append(("pose_long_front", "pose", "pose(long=front)", "Long pincer EE only", pose_long_front))

    def pose_short_up():
        yield from _ease_pose(home, extra={
            "pin_l0": -0.2, "pin_l1": -1.2,
        }, n=20)
    specs.append(("pose_short_up", "pose", "pose(short=up)", "Short stub EE only", pose_short_up))

    # --- move ---
    def move_torso_pitch():
        for t in _traj(40):
            s = math.sin(2 * math.pi * t * 2)
            yield {"base": (0, 0, 0), "joints": home({
                "v1_pitch": 0.15 - 0.45 * s,
                "v2_pitch": 0.10 - 0.40 * s,
                "v3_pitch": 0.10 - 0.45 * s,
            })}
    specs.append(("move_torso_pitch", "move", "move(torso, pitch, ±, rep=2)", "Nod rematerialize on spine", move_torso_pitch))

    def move_long_only():
        for t in _traj(40):
            s = math.sin(2 * math.pi * t * 2)
            yield {"base": (0, 0, 0), "joints": home({
                "pin_r0": 0.4 + 0.7 * s,
                "pin_r1": -0.8 - 0.5 * abs(s),
                "pin_r2": 0.2 + 0.7 * s,
            })}
    specs.append(("move_long_pincer", "move", "move(long_pincer, yaw, ±)", "Long branch only", move_long_only))

    def move_short_only():
        for t in _traj(40):
            s = math.sin(2 * math.pi * t * 2)
            yield {"base": (0, 0, 0), "joints": home({
                "pin_l0": -0.5 - 0.4 * s,
                "pin_l1": -0.4 - 0.35 * abs(s),
            })}
    specs.append(("move_short_pincer", "move", "move(short_pincer, pitch, ±)", "Short branch only", move_short_only))

    # --- path ---
    def path_collar_r0():
        for t in _traj(36):
            yield {"base": (0, 0, 0), "joints": home({"collar_spin": 2 * math.pi * t})}
    specs.append(("path_collar_r0", "path", "path(collar, arc, r=0, deg=360)", "Collar in-place spin", path_collar_r0))

    def path_torso_r0():
        for t in _traj(36):
            s = math.sin(2 * math.pi * t * 2)
            yield {"base": (0, 0, 0), "joints": home({
                "v1_yaw": 0.55 * s,
                "v2_twist": 0.15 + 0.5 * s,
            })}
    specs.append(("path_torso_r0", "path", "path(torso, arc, r=0, deg=±)", "Torso twist in place (r=0)", path_torso_r0))

    def path_base_retreat():
        for t in _traj(28):
            yield {"base": (-0.22 * t, 0, 0), "joints": home()}
    specs.append(("path_base_retreat", "path", "path(base, line, x=-)", "Base retreat only", path_base_retreat))

    return _emit(specs, model, out_dir, "Oddkin α=0.5  Canted Spine Crab",
                 lookat=(0.0, 0.0, 0.30), distance=1.75, base_z=z)


# -------------------- α=1.0 Branching Gesture Tree --------------------

def clips_alpha1(model, out_dir):
    z = 0.13

    def home(extra=None):
        # still = arms lowered beside trunk (not raised overhead).
        j = {
            "trunk1_yaw": 0.0, "trunk1_pitch": 0.0,
            "trunk2_yaw": 0.0, "trunk2_pitch": 0.0,
            "sat_spin": 0.0,
            "trunk3_pitch": 0.0, "trunk3_roll": 0.0,
            "bl0": 0.25, "bl1": 0.25, "bl2": 0.0,
            "br0": -0.25, "br1": 0.30, "br2": 0.15, "br3": 0.0,
            "claw0": 0.25, "claw1": -0.25, "claw2": -0.05,
            "neck_yaw": 0.0, "neck_pitch": 0.0, "head_roll": 0.0,
        }
        if extra:
            j.update(extra)
        return j

    specs = []

    # --- pose ---
    def pose_dir_front():
        yield from _ease_pose(home, n=20, n_ease=6)
    specs.append(("pose_dir_front", "pose", "pose(dir=front, torso=mid)", "Facing + mid trunk; arms still/low", pose_dir_front))

    def pose_torso_high():
        yield from _ease_pose(home, extra={
            "trunk1_pitch": 0.0, "trunk2_pitch": 0.0, "trunk3_pitch": 0.0,
        }, n=20)
    specs.append(("pose_torso_high", "pose", "pose(torso=high)", "Trunk upright (max height)", pose_torso_high))

    def pose_torso_low():
        yield from _ease_pose(home, extra={
            "trunk1_pitch": 0.55, "trunk2_pitch": 0.45, "trunk3_pitch": 0.35,
        }, n=20)
    specs.append(("pose_torso_low", "pose", "pose(torso=low)", "Trunk folded low toward viewer", pose_torso_low))

    def pose_dir_left():
        yield from _ease_pose(home, extra={
            "trunk1_yaw": 0.6, "trunk2_yaw": 0.45,
        }, n=20)
    specs.append(("pose_dir_left", "pose", "pose(dir=left)", "Trunk twist facing only", pose_dir_left))

    def pose_paddle_palmfront():
        yield from _ease_pose(home, extra={
            "bl0": 0.1, "bl1": -0.2, "bl2": 1.2,
        }, n=20)
    specs.append(("pose_paddle_palm", "pose", "pose(paddle=front, palm=palmfront)", "Paddle EE + palm only", pose_paddle_palmfront))

    def pose_claw_close():
        for t in _traj(24):
            a = _lerp(0.15, 1.0, _ease(t))
            yield {"base": (0, 0, 0), "joints": home({
                "br0": -0.1, "br1": 0.1, "br2": 0.0,
                "claw0": a, "claw1": -a, "claw2": -0.6 * a,
            })}
        for _ in range(10):
            yield {"base": (0, 0, 0), "joints": home({
                "br0": -0.1, "br1": 0.1,
                "claw0": 1.0, "claw1": -1.0, "claw2": -0.6,
            })}
    specs.append(("pose_claw_grip", "pose", "pose(claw=front, grip=close)", "Claw grip attribute only", pose_claw_close))

    def pose_head_left():
        yield from _ease_pose(home, extra={"neck_yaw": 0.7}, n=20)
    specs.append(("pose_head_left", "pose", "pose(head=left)", "Head gaze only", pose_head_left))

    # --- move ---
    def move_head_tilt():
        for t in _traj(40):
            s = math.sin(2 * math.pi * t * 2)
            yield {"base": (0, 0, 0), "joints": home({
                "neck_pitch": 0.35 * s,
            })}
    specs.append(("move_head_tilt", "move", "move(head, tilt, ±, rep=2)", "Head nod only", move_head_tilt))

    def move_paddle():
        for t in _traj(36):
            s = math.sin(2 * math.pi * t * 2)
            yield {"base": (0, 0, 0), "joints": home({
                "bl0": 0.25 + 0.4 * s,
                "bl1": -1.3 - 0.2 * abs(s),
                "bl2": 0.8 + 0.4 * s,
            })}
    specs.append(("move_paddle", "move", "move(paddle, yaw, ±)", "Left paddle only", move_paddle))

    def move_claw():
        for t in _traj(36):
            s = math.sin(2 * math.pi * t * 2)
            yield {"base": (0, 0, 0), "joints": home({
                "br0": -0.25 + 0.4 * s,
                "br1": -1.2 - 0.2 * abs(s),
                "br2": -0.4,
            })}
    specs.append(("move_claw", "move", "move(claw, yaw, ±)", "Right claw arm only", move_claw))

    # --- path ---
    def path_sat_r0():
        for t in _traj(36):
            yield {"base": (0, 0, 0), "joints": home({"sat_spin": 2 * math.pi * t})}
    specs.append(("path_sat_r0", "path", "path(sat, arc, r=0, deg=360)", "Satellite ring in-place", path_sat_r0))

    def path_torso_r0():
        for t in _traj(36):
            s = math.sin(2 * math.pi * t * 2)
            yield {"base": (0, 0, 0), "joints": home({
                "trunk1_yaw": 0.5 * s,
                "trunk2_yaw": 0.35 * s,
            })}
    specs.append(("path_torso_r0", "path", "path(torso, arc, r=0, deg=±)", "Trunk twist in place (r=0)", path_torso_r0))

    def path_ee_claw():
        for t in _traj(28):
            yield {"base": (0.10 * t, 0, 0), "joints": home({
                "br0": _lerp(-0.25, 0.0, t),
                "br1": _lerp(0.30, -0.15, t),
                "br2": _lerp(0.15, 0.0, t),
                "claw0": 0.2, "claw1": -0.2,
            })}
    specs.append(("path_ee_claw", "path", "path(ee_claw, line, axis=x, dist=+)", "Claw EE stroke toward viewer", path_ee_claw))

    return _emit(specs, model, out_dir, "Oddkin α=1.0  Branching Gesture Tree",
                 lookat=(0.0, 0.0, 0.40), distance=2.05, base_z=z)


def write_html(gallery, out_html):
    labels = {
        "gcr_alpha0": "α=0 · Trileg Crown (radial arthropod)",
        "gcr_alpha05": "α=0.5 · Canted Spine Crab (asymmetric branches)",
        "gcr_alpha1": "α=1.0 · Branching Gesture Tree (Y-spine + claws)",
    }
    badge = {
        "pose": "#7dd3c0",
        "move": "#f0c674",
        "path": "#8ab4f8",
    }
    sections = []
    for morph, items in gallery.items():
        # group cards by component for readability
        by_comp = {"pose": [], "move": [], "path": []}
        for it in items:
            by_comp.setdefault(it.get("component", "pose"), []).append(it)

        blocks = []
        for comp in ("pose", "move", "path"):
            cards = by_comp.get(comp) or []
            if not cards:
                continue
            figs = []
            for it in cards:
                color = badge.get(comp, "#9aa6b8")
                figs.append(f"""
        <figure class="card">
          <img src="{morph}/{it['gif']}" alt="{it['key']}"/>
          <figcaption>
            <span class="badge" style="--b:{color}">{comp}</span>
            <code>{it['code']}</code>
            <span>{it['note']}</span>
          </figcaption>
        </figure>""")
            blocks.append(f"""
      <h3>{comp}</h3>
      <div class="grid">{''.join(figs)}</div>""")
        sections.append(f"""
    <section>
      <h2>{labels.get(morph, morph)}</h2>
      {''.join(blocks)}
    </section>""")

    out_html.write_text(f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>Oddkin Continuum — Attribute Gallery</title>
<style>
:root {{ --bg:#0c1017; --panel:#151a24; --text:#e9eef7; --muted:#9aa6b8; --accent:#7dd3c0; --line:#2a3344; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:32px 28px 72px; font-family:"IBM Plex Sans",sans-serif;
  background:radial-gradient(1000px 500px at 0% 0%, #1a2a28, var(--bg)); color:var(--text); }}
h1 {{ margin:0 0 8px; font-size:1.55rem; letter-spacing:-0.02em; }}
.lede {{ color:var(--muted); max-width:82ch; line-height:1.55; margin-bottom:28px; }}
h2 {{ margin:34px 0 6px; padding-bottom:8px; border-bottom:1px solid var(--line); color:var(--accent); font-size:1.1rem; }}
h3 {{ margin:18px 0 10px; font-size:.85rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:14px; }}
.card {{ margin:0; background:var(--panel); border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
.card img {{ display:block; width:100%; aspect-ratio:5/4; object-fit:cover; background:#000; }}
figcaption {{ padding:10px 12px 12px; display:flex; flex-direction:column; gap:5px; }}
.badge {{ align-self:flex-start; font-size:.68rem; font-weight:600; letter-spacing:.06em;
  text-transform:uppercase; color:#0c1017; background:var(--b); padding:2px 7px; border-radius:999px; }}
figcaption code {{ font-family:ui-monospace,monospace; font-size:.78rem; color:#c8f0e6; word-break:break-word; }}
figcaption span:not(.badge) {{ font-size:.82rem; color:var(--muted); }}
</style></head><body>
<h1>Oddkin continuum — attribute gallery</h1>
<p class="lede">
Each card isolates <b>one</b> Motion Code attribute from
<code>yml/motion_code_synthetic_continuum.yaml</code>
(pose silhouette · move joint beat · path stroke; collar/crown/sat/torso spinners use <code>r=0</code>).
Independently actuated channels (long vs short, paddle vs claw, hip0 vs hip1) are never forced to move together.
</p>
{''.join(sections)}
</body></html>""")
    print("wrote", out_html)


def main():
    if OUT_DIR.exists():
        for p in OUT_DIR.rglob("*.gif"):
            p.unlink()
        for p in OUT_DIR.rglob("*_f0*.png"):
            p.unlink()
    gallery = {}
    for name, xml, fn in [
        ("gcr_alpha0", XML_DIR / "gcr_alpha0.xml", clips_alpha0),
        ("gcr_alpha05", XML_DIR / "gcr_alpha05.xml", clips_alpha05),
        ("gcr_alpha1", XML_DIR / "gcr_alpha1.xml", clips_alpha1),
    ]:
        print("rendering", name, "...")
        model = mujoco.MjModel.from_xml_path(str(xml))
        print(f"  nq={model.nq} njnt={model.njnt} nu={model.nu}")
        gallery[name] = fn(model, OUT_DIR / name)
    write_html(gallery, OUT_DIR / "index.html")


if __name__ == "__main__":
    main()
