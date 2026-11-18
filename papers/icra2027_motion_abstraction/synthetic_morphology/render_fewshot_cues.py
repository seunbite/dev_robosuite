#!/usr/bin/env python3
"""Render the 5 unified few-shot cues rematerialized on Oddkin α=0 / 0.5 / 1.0."""

from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
XML_DIR = ROOT / "xml"
YML = ROOT / "yml" / "fewshot_cues_oddkin.yaml"
OUT_DIR = ROOT / "renders" / "fewshot_cues"
FPS = 12
W, H = 480, 360

CUES = [
    "raising_hand_greeting",
    "wave_hi",
    "bow_greeting",
    "blow_kiss",
    "circle_temple_crazy",
]
MORPHS = [
    ("gcr_alpha0", "α=0 Trileg Crown", (0.0, 0.0, 0.14), 1.45, 0.12),
    ("gcr_alpha05", "α=0.5 Spine Crab", (0.0, 0.0, 0.30), 1.75, 0.11),
    ("gcr_alpha1", "α=1.0 Gesture Tree", (0.0, 0.0, 0.40), 2.05, 0.13),
]


def _font(size: int = 15):
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
    h = 0.5 * yaw
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)], dtype=np.float64)


def _set_joint(data, model, name, value):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid >= 0:
        data.qpos[model.jnt_qposadr[jid]] = value


def _set_base(data, model, x, y, yaw, z):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
    adr = model.jnt_qposadr[jid]
    data.qpos[adr:adr + 3] = [x, y, z]
    data.qpos[adr + 3:adr + 7] = _quat_yaw(yaw)


def _overlay(rgb, title, subtitle):
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, 0, img.width, 46], fill=(255, 255, 255, 230))
    draw.text((8, 4), title, fill=(15, 15, 20), font=_font(14))
    draw.text((8, 24), subtitle[:78], fill=(50, 55, 65), font=_font(11))
    return np.asarray(img.convert("RGB"))


def _cam(lookat, distance):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = 180
    cam.elevation = -8
    return cam


def _traj(n):
    for i in range(n):
        yield i / max(n - 1, 1)


def _lerp(a, b, t):
    return a + (b - a) * t


def _hold(cfg, n=10):
    for _ in range(n):
        yield cfg


# -------------------- homes --------------------

def home0(extra=None):
    j = {
        "hip0": 0.2, "knee0": -0.6,
        "hip1": 0.15, "knee1": -0.55,
        "hip2": -0.1, "knee2": -0.65,
        "crown_spin": 0.0,
    }
    if extra:
        j.update(extra)
    return j


def home05(extra=None):
    j = {
        "flap_l": 0.2, "flap_r": -0.2,
        "v1_pitch": 0.1, "v1_yaw": 0.0,
        "v2_pitch": 0.08, "v2_twist": 0.15,
        "collar_spin": 0.0, "v3_pitch": 0.05,
        "pin_r0": 0.35, "pin_r1": -0.7, "pin_r2": 0.1,
        "pin_l0": -0.4, "pin_l1": -0.3,
        "antenna": 0.3,
    }
    if extra:
        j.update(extra)
    return j


def home1(extra=None):
    # still = arms lowered beside trunk (not raised). +br1/bl1 pitches tip down.
    j = {
        "trunk1_yaw": 0.0, "trunk1_pitch": 0.0,
        "trunk2_yaw": 0.0, "trunk2_pitch": 0.0,
        "sat_spin": 0.0, "trunk3_pitch": 0.0, "trunk3_roll": 0.0,
        "bl0": 0.25, "bl1": 0.25, "bl2": 0.0,
        "br0": -0.25, "br1": 0.30, "br2": 0.15, "br3": 0.0,
        "claw0": 0.25, "claw1": -0.25, "claw2": -0.05,
        "neck_yaw": 0.0, "neck_pitch": 0.0, "head_roll": 0.0,
    }
    if extra:
        j.update(extra)
    return j


# -------------------- α=0 programs --------------------

def frames_alpha0(cue: str):
    if cue == "raising_hand_greeting":
        for _ in _hold({"base": (0, 0, 0), "joints": home0()}, 8):
            yield _
        for t in _traj(14):
            yield {"base": (0, 0, 0), "joints": home0({
                "hip0": _lerp(0.2, 0.95, t), "knee0": _lerp(-0.6, -0.15, t),
            })}
        for t in _traj(16):
            yield {"base": (0.18 * t, 0, 0), "joints": home0({
                "hip0": 0.95, "knee0": -0.15,
            })}
        yield from _hold({"base": (0.18, 0, 0), "joints": home0({"hip0": 0.95, "knee0": -0.15})}, 8)
        return

    if cue == "wave_hi":
        for t in _traj(36):
            s = math.sin(2 * math.pi * t * 3)
            yield {"base": (0, 0, 0), "joints": home0({
                "hip0": 0.7 + 0.45 * s, "knee0": -0.25,
            })}
        return

    if cue == "bow_greeting":
        for t in _traj(14):
            yield {"base": (0.15 * t, 0, 0), "joints": home0()}
        yield from _hold({"base": (0.15, 0, 0), "joints": home0()}, 8)
        for t in _traj(14):
            yield {"base": (0.15 - 0.12 * t, 0, 0), "joints": home0()}
        yield from _hold({"base": (0.03, 0, 0), "joints": home0()}, 6)
        return

    if cue == "blow_kiss":
        for t in _traj(12):
            s = math.sin(2 * math.pi * t)
            yield {"base": (0, 0, 0.2 * s), "joints": home0()}
        yield from _hold({"base": (0, 0, 0), "joints": home0()}, 4)
        for t in _traj(16):
            yield {"base": (0.28 * t, 0, 0), "joints": home0({"crown_spin": 0.4 * t})}
        yield from _hold({"base": (0.28, 0, 0), "joints": home0({"crown_spin": 0.4})}, 6)
        return

    if cue == "circle_temple_crazy":
        for t in _traj(40):
            ang = 3 * math.pi * t  # 540°
            r = 0.12
            yield {"base": (r * math.cos(ang) - r, r * math.sin(ang), ang), "joints": home0({
                "crown_spin": ang,
            })}
        return

    raise KeyError(cue)


# -------------------- α=0.5 programs --------------------

def frames_alpha05(cue: str):
    high = {"v1_pitch": -0.45, "v2_pitch": -0.4, "v3_pitch": -0.5}
    low = {"v1_pitch": 0.75, "v2_pitch": 0.65, "v3_pitch": 0.7}
    mid = {"v1_pitch": 0.1, "v2_pitch": 0.08, "v3_pitch": 0.05}

    if cue == "raising_hand_greeting":
        yield from _hold({"base": (0, 0, 0), "joints": home05()}, 8)
        for t in _traj(16):
            yield {"base": (0, 0, 0), "joints": home05({
                "pin_r0": _lerp(0.35, 0.15, t),
                "pin_r1": _lerp(-0.7, -1.6, t),
                "pin_r2": _lerp(0.1, 0.0, t),
            })}
        yield from _hold({"base": (0, 0, 0), "joints": home05({
            "pin_r0": 0.15, "pin_r1": -1.6, "pin_r2": 0.0,
        })}, 10)
        return

    if cue == "wave_hi":
        for t in _traj(36):
            s = math.sin(2 * math.pi * t * 3)
            yield {"base": (0, 0, 0), "joints": home05({
                "pin_r0": 0.15 + 0.55 * s,
                "pin_r1": -1.5 - 0.2 * abs(s),
                "pin_r2": 0.4 * s,
            })}
        return

    if cue == "bow_greeting":
        for t in _traj(12):
            yield {"base": (0, 0, 0), "joints": home05({
                "v1_pitch": _lerp(mid["v1_pitch"], high["v1_pitch"], t),
                "v2_pitch": _lerp(mid["v2_pitch"], high["v2_pitch"], t),
                "v3_pitch": _lerp(mid["v3_pitch"], high["v3_pitch"], t),
            })}
        for t in _traj(14):
            yield {"base": (0, 0, 0), "joints": home05({
                "v1_pitch": _lerp(high["v1_pitch"], low["v1_pitch"], t),
                "v2_pitch": _lerp(high["v2_pitch"], low["v2_pitch"], t),
                "v3_pitch": _lerp(high["v3_pitch"], low["v3_pitch"], t),
            })}
        yield from _hold({"base": (0, 0, 0), "joints": home05(low)}, 8)
        for t in _traj(12):
            yield {"base": (0, 0, 0), "joints": home05({
                "v1_pitch": _lerp(low["v1_pitch"], mid["v1_pitch"], t),
                "v2_pitch": _lerp(low["v2_pitch"], mid["v2_pitch"], t),
                "v3_pitch": _lerp(low["v3_pitch"], mid["v3_pitch"], t),
            })}
        return

    if cue == "blow_kiss":
        yield from _hold({"base": (0, 0, 0), "joints": home05({
            "pin_r0": 0.2, "pin_r1": -1.4, "pin_r2": 0.8,
        })}, 8)
        for t in _traj(16):
            yield {"base": (0.10 * t, 0, 0), "joints": home05({
                "pin_r0": _lerp(0.2, 0.05, t),
                "pin_r1": _lerp(-1.4, -0.35, t),
                "pin_r2": _lerp(0.8, 0.0, t),
            })}
        yield from _hold({"base": (0.10, 0, 0), "joints": home05({
            "pin_r0": 0.05, "pin_r1": -0.35, "pin_r2": 0.0,
        })}, 8)
        return

    if cue == "circle_temple_crazy":
        for t in _traj(40):
            yield {"base": (0, 0, 0), "joints": home05({
                "pin_r0": -0.3, "pin_r1": -1.0,
                "collar_spin": 3 * math.pi * t,  # 540°
            })}
        return

    raise KeyError(cue)


# -------------------- α=1.0 programs --------------------

def frames_alpha1(cue: str):
    if cue == "raising_hand_greeting":
        yield from _hold({"base": (0, 0, 0), "joints": home1()}, 8)
        for t in _traj(16):
            yield {"base": (0, 0, 0), "joints": home1({
                "bl0": _lerp(0.25, 0.15, t),
                "bl1": _lerp(0.25, -1.5, t),
                "bl2": _lerp(0.0, 1.1, t),
            })}
        yield from _hold({"base": (0, 0, 0), "joints": home1({
            "bl0": 0.15, "bl1": -1.5, "bl2": 1.1,
        })}, 10)
        return

    if cue == "wave_hi":
        for t in _traj(36):
            s = math.sin(2 * math.pi * t * 3)
            yield {"base": (0, 0, 0), "joints": home1({
                "bl0": 0.15 + 0.35 * s,
                "bl1": -1.4 - 0.15 * abs(s),
                "bl2": 1.0 + 0.3 * s,
            })}
        return

    if cue == "bow_greeting":
        # +pitch leans crown toward camera (+x) = bow to viewer (NOT backward).
        for t in _traj(12):
            yield {"base": (0, 0, 0), "joints": home1({
                "trunk1_pitch": _lerp(0.0, 0.15, t),
                "trunk2_pitch": _lerp(0.0, 0.12, t),
            })}
        for t in _traj(14):
            yield {"base": (0, 0, 0), "joints": home1({
                "trunk1_pitch": _lerp(0.15, 0.55, t),
                "trunk2_pitch": _lerp(0.12, 0.50, t),
                "trunk3_pitch": _lerp(0.0, 0.40, t),
                "neck_pitch": _lerp(0.0, 0.45, t),
            })}
        yield from _hold({"base": (0, 0, 0), "joints": home1({
            "trunk1_pitch": 0.55, "trunk2_pitch": 0.50, "trunk3_pitch": 0.40,
            "neck_pitch": 0.45,
        })}, 8)
        for t in _traj(12):
            yield {"base": (0, 0, 0), "joints": home1({
                "trunk1_pitch": _lerp(0.55, 0.0, t),
                "trunk2_pitch": _lerp(0.50, 0.0, t),
                "trunk3_pitch": _lerp(0.40, 0.0, t),
                "neck_pitch": _lerp(0.45, 0.0, t),
            })}
        return

    if cue == "blow_kiss":
        yield from _hold({"base": (0, 0, 0), "joints": home1({
            "br0": 0.1, "br1": -1.5, "br2": -0.8, "br3": 0.3,
            "claw0": 0.9, "claw1": -0.9, "claw2": -0.5,
        })}, 8)
        for t in _traj(16):
            yield {"base": (0, 0, 0), "joints": home1({
                "br0": _lerp(0.1, -0.1, t),
                "br1": _lerp(-1.5, 0.15, t),
                "br2": _lerp(-0.8, 0.05, t),
                "br3": _lerp(0.3, 0.0, t),
                "claw0": _lerp(0.9, 0.2, t),
                "claw1": _lerp(-0.9, -0.2, t),
                "claw2": _lerp(-0.5, 0.0, t),
            })}
        yield from _hold({"base": (0, 0, 0), "joints": home1({
            "br0": -0.1, "br1": 0.15, "br2": 0.05, "claw0": 0.2, "claw1": -0.2,
        })}, 8)
        return

    if cue == "circle_temple_crazy":
        for t in _traj(40):
            ang = 3 * math.pi * t
            yield {"base": (0, 0, 0), "joints": home1({
                "br0": 0.55 + 0.25 * math.cos(ang),
                "br1": -1.2 + 0.2 * math.sin(ang),
                "br2": -0.55,
                "neck_yaw": 0.15 * math.sin(ang),
                "claw0": 0.8, "claw1": -0.8,
            })}
        return

    raise KeyError(cue)


FRAME_FN = {
    "gcr_alpha0": frames_alpha0,
    "gcr_alpha05": frames_alpha05,
    "gcr_alpha1": frames_alpha1,
}


def render_one(model, frames, *, out_gif, title, subtitle, lookat, distance, base_z):
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = _cam(lookat, distance)
    imgs = []
    for cfg in frames:
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


def write_html(programs, out_html: Path):
    cue_labels = {
        "raising_hand_greeting": "Raising hand / greeting",
        "wave_hi": "Wave hi",
        "bow_greeting": "Bow greeting",
        "blow_kiss": "Blow kiss",
        "circle_temple_crazy": "Circle temple (crazy)",
    }
    rows = []
    for cue in CUES:
        cells = []
        for morph, label, *_ in MORPHS:
            code = " ".join(programs[morph]["programs"][cue].split())
            gif = f"{morph}/{cue}.gif"
            cells.append(f"""
        <td>
          <figure class="cell">
            <img src="{gif}" alt="{morph}-{cue}"/>
            <figcaption>
              <div class="morph">{label}</div>
              <code>{code}</code>
            </figcaption>
          </figure>
        </td>""")
        rows.append(f"""
      <tr>
        <th scope="row"><div class="cue">{cue_labels[cue]}</div><div class="cue-id">{cue}</div></th>
        {''.join(cells)}
      </tr>""")

    out_html.write_text(f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>Oddkin · Few-shot cue rematerialization</title>
<style>
:root {{ --bg:#0c1017; --panel:#151a24; --text:#e9eef7; --muted:#9aa6b8; --accent:#7dd3c0; --line:#2a3344; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:28px 24px 64px; font-family:"IBM Plex Sans",sans-serif;
  background:radial-gradient(1000px 500px at 0% 0%, #1a2a28, var(--bg)); color:var(--text); }}
h1 {{ margin:0 0 8px; font-size:1.45rem; }}
.lede {{ color:var(--muted); max-width:90ch; line-height:1.5; margin-bottom:22px; }}
table {{ width:100%; border-collapse:separate; border-spacing:12px; }}
th {{ text-align:left; vertical-align:top; width:140px; padding-top:8px; }}
.cue {{ font-size:.95rem; color:var(--accent); }}
.cue-id {{ font-size:.72rem; color:var(--muted); font-family:ui-monospace,monospace; margin-top:2px; }}
td {{ vertical-align:top; width:28%; }}
.cell {{ margin:0; background:var(--panel); border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
.cell img {{ display:block; width:100%; aspect-ratio:4/3; object-fit:cover; background:#000; }}
figcaption {{ padding:8px 10px 10px; display:flex; flex-direction:column; gap:4px; }}
.morph {{ font-size:.72rem; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); }}
code {{ font-family:ui-monospace,monospace; font-size:.68rem; color:#c8f0e6; word-break:break-word; line-height:1.35; }}
</style></head><body>
<h1>Oddkin · few-shot cue rematerialization (5 × 3)</h1>
<p class="lede">
Same 5 Motion Code few-shot cues as <code>shots.yaml</code>
(<code>raising_hand_greeting</code>, <code>wave_hi</code>, <code>bow_greeting</code>,
<code>blow_kiss</code>, <code>circle_temple_crazy</code>), rematerialized onto Oddkin α=0 / 0.5 / 1.0.
Programs: <code>yml/fewshot_cues_oddkin.yaml</code>.
</p>
<table>
  <thead><tr><th></th>
    <th>α=0 Trileg Crown</th>
    <th>α=0.5 Spine Crab</th>
    <th>α=1.0 Gesture Tree</th>
  </tr></thead>
  <tbody>
  {''.join(rows)}
  </tbody>
</table>
</body></html>""")
    print("wrote", out_html)


def main():
    programs = yaml.safe_load(YML.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for morph, label, lookat, dist, base_z in MORPHS:
        model = mujoco.MjModel.from_xml_path(str(XML_DIR / f"{morph}.xml"))
        out_sub = OUT_DIR / morph
        out_sub.mkdir(parents=True, exist_ok=True)
        print("rendering", morph, "...")
        for cue in CUES:
            code = " ".join(programs[morph]["programs"][cue].split())
            frames = list(FRAME_FN[morph](cue))
            render_one(
                model, frames,
                out_gif=out_sub / f"{cue}.gif",
                title=f"{label} · {cue}",
                subtitle=code,
                lookat=lookat, distance=dist, base_z=base_z,
            )
            print(" ", cue)
    write_html(programs, OUT_DIR / "index.html")


if __name__ == "__main__":
    main()
