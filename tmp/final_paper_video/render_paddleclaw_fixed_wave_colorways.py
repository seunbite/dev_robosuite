from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SRC = ROOT / "papers/icra2027_motion_abstraction/synthetic_morphology/render_fewshot_cues.py"
XML = ROOT / "papers/icra2027_motion_abstraction/synthetic_morphology/xml/gcr_alpha1.xml"
OUT = ROOT / "output/paddleclaw_fixed_wave_colorways"

BLACK = (0.10, 0.105, 0.115, 1.0)
WHITE = (0.86, 0.875, 0.90, 1.0)
FPS = 12
N_FRAMES = 48
MOTION_PROGRAM = (
    "pose(dir=front, base=still, torso=mid, paddle=still, claw=up, grip=open) "
    "+ move(claw, y, ±0.05, rep=3, spd=1.8, hold=0.12)"
)

# The long claw-side arm stays black and only its terminal wrist/claw is white
# in every colorway. Other body-part assignments vary within black and white.
COLORWAYS = [
    ("A_balanced", "A · Balanced", {"head", "face", "paddle", "base_rods", "claw_end"}),
    ("B_white_torso", "B · White torso", {"head", "face", "trunk", "paddle", "claw_end"}),
    ("C_white_head_base", "C · White head + base", {"head", "base", "base_rods", "paddle", "claw_end"}),
    ("D_white_short_arm", "D · White short arm", {"face", "short_arm", "paddle", "base_rods", "claw_end"}),
    ("E_white_core", "E · White core", {"face", "trunk", "base", "paddle", "claw_end"}),
    ("F_alternating", "F · Alternating", {"head", "face", "trunk_upper", "short_arm", "base_rods", "claw_end"}),
]

spec = importlib.util.spec_from_file_location("fewshot", SRC)
fewshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fewshot)


def bname(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
    ) or ""


def set_colorway(model: mujoco.MjModel, roles: set[str]) -> None:
    black_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "body")
    white_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "marker")
    model.mat_rgba[black_id] = BLACK
    model.mat_rgba[white_id] = WHITE
    model.geom_matid[:] = black_id

    for gid in range(1, model.ngeom):
        body = bname(model, gid)
        geom_type = int(model.geom_type[gid])
        white = False

        if "face" in roles and body == "head" and gid in {33, 34, 35}:
            white = True
        if "head" in roles and body in {"neck", "head"} and gid not in {33, 34, 35}:
            white = True
        if "trunk" in roles and body in {"trunk1", "trunk2", "trunk3"}:
            white = True
        if "trunk_upper" in roles and body in {"trunk2", "trunk3"}:
            white = True
        if "base" in roles and body == "base" and geom_type != int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            white = True
        if "base_rods" in roles and body == "base" and geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            white = True
        if "short_arm" in roles and body in {"branch_l0", "branch_l1"}:
            white = True
        if "paddle" in roles and body == "branch_l2":
            white = True
        # Fixed rule: long arm black; only terminal wrist hub and fingers white.
        if "claw_end" in roles and body in {"branch_r3", "claw0", "claw1", "claw2"}:
            white = True

        if white:
            model.geom_matid[gid] = white_id

    # Remove the claw-side spherical joint that appeared inside the claw silhouette.
    claw_sphere = 24  # branch_r1 sphere in gcr_alpha1.xml
    model.geom_matid[claw_sphere] = -1
    model.geom_rgba[claw_sphere] = (0.0, 0.0, 0.0, 0.0)
    model.geom_contype[claw_sphere] = 0
    model.geom_conaffinity[claw_sphere] = 0


def fixed_wave_joints(phase: float) -> dict[str, float]:
    # Only the claw-side lateral-yaw carrier changes. Base, torso, head,
    # paddle-side arm, wrist roll, and claw aperture remain fixed.
    lateral = math.sin(phase * 2.0 * math.pi * 3.0)
    return fewshot.home1({
        "trunk1_yaw": 0.0,
        "trunk1_pitch": 0.0,
        "trunk2_yaw": 0.0,
        "trunk2_pitch": 0.0,
        "trunk3_pitch": 0.0,
        "trunk3_roll": 0.0,
        "neck_yaw": 0.0,
        "neck_pitch": 0.0,
        "head_roll": 0.0,
        "bl0": 0.25,
        "bl1": 0.25,
        "bl2": 0.0,
        # Present the claw arm laterally and oscillate only its yaw carrier.
        # This keeps the linked black arm in the image plane and moves the
        # terminal white claw left/right without moving the base or torso.
        "br0": 0.65 + 0.125 * lateral,
        "br1": 0.0,
        "br2": 0.0,
        "br3": 0.0,
        "claw0": 0.92,
        "claw1": -0.92,
        "claw2": -0.44,
    })


def render_colorway(name: str, roles: set[str]) -> Path:
    model = mujoco.MjModel.from_xml_path(str(XML))
    set_colorway(model, roles)
    model.vis.headlight.ambient[:] = (0.82, 0.82, 0.82)
    model.vis.headlight.diffuse[:] = (0.76, 0.76, 0.76)
    model.vis.headlight.specular[:] = (0.18, 0.18, 0.18)
    if model.nlight:
        model.light_diffuse[:] = (0.66, 0.66, 0.66)
        model.light_specular[:] = (0.12, 0.12, 0.12)

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = fewshot._cam((0.0, 0.0, 0.43), 1.62)
    cam.azimuth = 180
    rgb_frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for frame_index in range(N_FRAMES):
        phase = frame_index / N_FRAMES
        mujoco.mj_resetData(model, data)
        fewshot._set_base(data, model, 0.0, 0.0, 0.0, 0.13)
        for joint, value in fixed_wave_joints(phase).items():
            fewshot._set_joint(data, model, joint, value)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        rgb = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        seg = renderer.render().copy()
        renderer.disable_segmentation_rendering()

        obj_id = seg[..., 0]
        obj_type = seg[..., 1]
        mask = (obj_id >= 0) & ~(
            (obj_type == int(mujoco.mjtObj.mjOBJ_GEOM)) & (obj_id == 0)
        )
        rgb_frames.append(rgb)
        masks.append(mask)

    renderer.close()

    # Use one union crop for the entire animation. Per-frame cropping makes a
    # stationary base appear to drift whenever the moving hand changes the
    # frame bounds.
    union = np.logical_or.reduce(masks)
    ys, xs = np.where(union)
    pad = 22
    x0, x1 = max(0, int(xs.min()) - pad), min(union.shape[1], int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(union.shape[0], int(ys.max()) + pad + 1)

    frames: list[Image.Image] = []
    for rgb, mask in zip(rgb_frames, masks):
        crop = Image.fromarray(rgb[y0:y1, x0:x1], "RGB")
        crop = ImageEnhance.Brightness(crop).enhance(1.06)
        alpha = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8), "L")
        rgba = crop.convert("RGBA")
        rgba.putalpha(alpha)
        rgba.thumbnail((590, 590), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (640, 640), (255, 255, 255, 0))
        canvas.alpha_composite(rgba, ((640 - rgba.width) // 2, (640 - rgba.height) // 2))
        # Preserve the black/white palette while keeping white parts legible on
        # the deck's white background.
        alpha_full = canvas.getchannel("A")
        border = ImageChops.subtract(alpha_full.filter(ImageFilter.MaxFilter(5)), alpha_full)
        outlined = Image.new("RGBA", canvas.size, (174, 184, 194, 0))
        outlined.putalpha(border.point(lambda value: min(155, value)))
        outlined.alpha_composite(canvas)
        canvas = outlined
        frames.append(canvas)

    dst = OUT / f"{name}.gif"
    frames[0].save(
        dst, save_all=True, append_images=frames[1:],
        duration=round(1000 / FPS), loop=0, disposal=2,
        transparency=0, optimize=False,
    )
    return dst


def make_sheet(paths: list[Path]) -> Path:
    clips = [Image.open(path) for path in paths]
    label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 27, index=1)
    detail_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17, index=0)
    cell_w, cell_h = 480, 500
    sheets: list[Image.Image] = []
    for frame_index in range(N_FRAMES):
        sheet = Image.new("RGB", (cell_w * 3, cell_h * 2), "#F4F6F8")
        draw = ImageDraw.Draw(sheet)
        for i, (clip, (_, label, _)) in enumerate(zip(clips, COLORWAYS)):
            clip.seek(frame_index % clip.n_frames)
            robot = clip.convert("RGBA")
            robot.thumbnail((405, 405), Image.Resampling.LANCZOS)
            x, y = (i % 3) * cell_w, (i // 3) * cell_h
            sheet.paste(robot, (x + (cell_w - robot.width) // 2, y + 8), robot)
            draw.text((x + cell_w // 2, y + 430), label, font=label_font, fill="#132431", anchor="mm")
            draw.text((x + cell_w // 2, y + 467), "identical lateral hand motion", font=detail_font, fill="#667680", anchor="mm")
        sheets.append(sheet)
    dst = OUT / "paddleclaw_fixed_wave_colorways.gif"
    sheets[0].save(
        dst, save_all=True, append_images=sheets[1:],
        duration=round(1000 / FPS), loop=0, optimize=False,
    )
    return dst


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "motion_program.txt").write_text(MOTION_PROGRAM + "\n", encoding="utf-8")
    rendered = [render_colorway(name, roles) for name, _, roles in COLORWAYS]
    sheet = make_sheet(rendered)
    print(MOTION_PROGRAM)
    for path in rendered:
        print(path)
    print(sheet)
