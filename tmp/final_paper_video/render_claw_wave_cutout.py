from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SRC = ROOT / "papers/icra2027_motion_abstraction/synthetic_morphology/render_fewshot_cues.py"
XML = ROOT / "papers/icra2027_motion_abstraction/synthetic_morphology/xml/gcr_alpha1.xml"
OUT = ROOT / "tmp/final_paper_video/paddleclaw_big_claw_wave_cutout.gif"

spec = importlib.util.spec_from_file_location("fewshot", SRC)
fewshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fewshot)

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)
cam = fewshot._cam((0.0, 0.0, 0.43), 1.85)
floor_id = 0

configs = []
for i in range(48):
    t = i / 47
    s = math.sin(2 * math.pi * t * 3)
    # Hold the large claw forward and drive the wave mainly from the wrist roll.
    configs.append({
        "base": (0.0, 0.0, 0.0),
        "joints": fewshot.home1({
            "br0": -0.12,
            "br1": -0.42,
            "br2": -0.28 + 0.06 * s,
            "br3": 0.78 * s,
            "claw0": 0.88 + 0.07 * s,
            "claw1": -0.88 + 0.07 * s,
            "claw2": -0.45 + 0.06 * s,
            "bl0": 0.25,
            "bl1": 0.25,
            "bl2": 0.0,
        }),
    })

rgb_frames = []
masks = []
for cfg in configs:
    mujoco.mj_resetData(model, data)
    x, y, yaw = cfg["base"]
    fewshot._set_base(data, model, x, y, yaw, 0.13)
    for name, value in cfg["joints"].items():
        fewshot._set_joint(data, model, name, value)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    seg = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    obj_id = seg[..., 0]
    obj_type = seg[..., 1]
    mask = (obj_id >= 0) & ~((obj_type == int(mujoco.mjtObj.mjOBJ_GEOM)) & (obj_id == floor_id))
    rgb_frames.append(rgb)
    masks.append(mask)

renderer.close()

union = np.logical_or.reduce(masks)
ys, xs = np.where(union)
if not len(xs):
    raise RuntimeError("Segmentation produced an empty robot mask")
pad = 18
x0, x1 = max(0, xs.min() - pad), min(union.shape[1], xs.max() + pad + 1)
y0, y1 = max(0, ys.min() - pad), min(union.shape[0], ys.max() + pad + 1)

frames = []
for rgb, mask in zip(rgb_frames, masks):
    crop_rgb = np.clip(rgb[y0:y1, x0:x1].astype(np.float32) * 0.72, 0, 255).astype(np.uint8)
    crop_mask = (mask[y0:y1, x0:x1] * 255).astype(np.uint8)
    rgba = np.dstack([crop_rgb, crop_mask])
    frames.append(Image.fromarray(rgba, "RGBA"))

OUT.parent.mkdir(parents=True, exist_ok=True)
frames[0].save(
    OUT,
    save_all=True,
    append_images=frames[1:],
    duration=round(1000 / 12),
    loop=0,
    disposal=2,
    transparency=0,
    optimize=False,
)
print(OUT)
print(f"size={frames[0].size}, frames={len(frames)}")
