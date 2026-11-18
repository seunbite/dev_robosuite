from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SRC = ROOT / "papers/icra2027_motion_abstraction/synthetic_morphology/render_fewshot_cues.py"
XML = ROOT / "papers/icra2027_motion_abstraction/synthetic_morphology/xml/gcr_alpha1.xml"
OUT = ROOT / "output/paddleclaw_hi_variants"

BLACK = (0.10, 0.105, 0.115, 1.0)
WHITE = (0.86, 0.875, 0.90, 1.0)
FPS = 12
N_FRAMES = 48

VARIANTS = [
    ("A_wrist_wave", "A · Clear wrist wave"),
    ("B_gentle_wave", "B · Gentle wrist wave"),
    ("C_bouncy_wave", "C · Wrist + small bounce"),
    ("D_claw_wave", "D · Wrist + claw motion"),
    ("E_side_wave", "E · Side-to-side wave"),
    ("F_friendly_wave", "F · Wrist + friendly lean"),
]

spec = importlib.util.spec_from_file_location("fewshot", SRC)
fewshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fewshot)


def body_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        int(model.geom_bodyid[geom_id]),
    ) or ""


def apply_black_white_scheme(model: mujoco.MjModel) -> None:
    black_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "body")
    white_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "marker")
    marker_id = white_id
    model.mat_rgba[black_id] = BLACK
    model.mat_rgba[white_id] = WHITE

    # Start with every visible robot geom black.
    model.geom_matid[:] = black_id
    for geom_id in range(model.ngeom):
        bname = body_name(model, geom_id)
        original_mat = int(model.geom_matid[geom_id])
        geom_type = int(model.geom_type[geom_id])

        is_face_dot = bname == "head" and original_mat == marker_id
        is_wrist_joint = bname in {"branch_l1", "branch_r1"} and geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE)
        # branch_r2 is the short distal wrist link seen nearly end-on as the
        # apparent circle behind the claw.
        is_end_effector = bname in {"branch_l2", "branch_r1", "branch_r2", "branch_r3", "claw0", "claw1", "claw2"}
        is_base_rod = bname == "base" and geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
        if is_face_dot or is_wrist_joint or is_end_effector or is_base_rod:
            model.geom_matid[geom_id] = white_id

    # The material ids were overwritten above, so explicitly restore the face dots.
    # In gcr_alpha1.xml the final three geoms in the head body are eyes and nose.
    head_geoms = [gid for gid in range(model.ngeom) if body_name(model, gid) == "head"]
    for geom_id in head_geoms[-3:]:
        model.geom_matid[geom_id] = white_id


def motion(name: str, phase: float) -> dict[str, float]:
    s = math.sin(phase * 2 * math.pi * 3)
    s2 = math.sin(phase * 2 * math.pi * 3 + math.pi / 2)
    joints = {
        "br0": -0.12,
        "br1": -0.42,
        "br2": -0.28,
        "br3": 0.72 * s,
        "claw0": 0.90,
        "claw1": -0.90,
        "claw2": -0.44,
        "bl0": 0.25,
        "bl1": 0.25,
        "bl2": 0.0,
    }
    if name == "B_gentle_wave":
        joints["br3"] = 0.46 * math.sin(phase * 2 * math.pi * 2)
    elif name == "C_bouncy_wave":
        joints["br3"] = 0.62 * s
        joints["br2"] = -0.28 + 0.11 * s2
    elif name == "D_claw_wave":
        joints["br3"] = 0.58 * s
        pinch = 0.11 * (0.5 + 0.5 * s2)
        joints["claw0"] = 0.84 + pinch
        joints["claw1"] = -0.84 - pinch
        joints["claw2"] = -0.38 - 0.55 * pinch
    elif name == "E_side_wave":
        joints["br3"] = 0.44 * s
        joints["br0"] = -0.12 + 0.18 * s
    elif name == "F_friendly_wave":
        joints["br3"] = 0.58 * s
        joints["trunk3_roll"] = 0.09 * s2
        joints["head_roll"] = -0.13 * s2
    return joints


def render_variant(name: str) -> Path:
    model = mujoco.MjModel.from_xml_path(str(XML))
    original_matids = model.geom_matid.copy()
    apply_black_white_scheme(model)

    # Bright neutral studio lighting, retaining highlights on black surfaces.
    model.vis.headlight.ambient[:] = (0.82, 0.82, 0.82)
    model.vis.headlight.diffuse[:] = (0.76, 0.76, 0.76)
    model.vis.headlight.specular[:] = (0.20, 0.20, 0.20)
    if model.nlight:
        model.light_diffuse[:] = (0.66, 0.66, 0.66)
        model.light_specular[:] = (0.14, 0.14, 0.14)

    # Restore knowledge of which head geoms were marker-based before recoloring.
    marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "marker")
    head_geoms = [gid for gid in range(model.ngeom) if body_name(model, gid) == "head"]
    for gid in head_geoms:
        model.geom_matid[gid] = marker_id if original_matids[gid] == marker_id else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "body")

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = fewshot._cam((0.0, 0.0, 0.43), 1.62)
    frames: list[Image.Image] = []

    for frame_index in range(N_FRAMES):
        phase = frame_index / N_FRAMES
        mujoco.mj_resetData(model, data)
        fewshot._set_base(data, model, 0.0, 0.0, 0.0, 0.13)
        for joint, value in fewshot.home1(motion(name, phase)).items():
            fewshot._set_joint(data, model, joint, value)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        # Shadows cast onto the removed floor otherwise survive as dark circular
        # artifacts in the transparent cutout.
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
        ys, xs = np.where(mask)
        pad = 22
        x0, x1 = max(0, int(xs.min()) - pad), min(rgb.shape[1], int(xs.max()) + pad + 1)
        y0, y1 = max(0, int(ys.min()) - pad), min(rgb.shape[0], int(ys.max()) + pad + 1)
        crop = Image.fromarray(rgb[y0:y1, x0:x1], "RGB")
        crop = ImageEnhance.Brightness(crop).enhance(1.07)
        alpha = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8), "L")
        rgba = crop.convert("RGBA")
        rgba.putalpha(alpha)
        rgba.thumbnail((590, 590), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (640, 640), (255, 255, 255, 0))
        canvas.alpha_composite(rgba, ((640 - rgba.width) // 2, (640 - rgba.height) // 2))
        frames.append(canvas)

    renderer.close()
    dst = OUT / f"{name}.gif"
    frames[0].save(
        dst,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / FPS),
        loop=0,
        disposal=2,
        transparency=0,
        optimize=False,
    )
    return dst


def make_contact_sheet(paths: list[Path]) -> Path:
    clips = [Image.open(path) for path in paths]
    label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 27, index=1)
    detail_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17, index=0)
    cell_w, cell_h = 480, 500
    frames: list[Image.Image] = []
    for frame_index in range(N_FRAMES):
        sheet = Image.new("RGB", (cell_w * 3, cell_h * 2), "#F4F6F8")
        draw = ImageDraw.Draw(sheet)
        for i, (clip, (_, label)) in enumerate(zip(clips, VARIANTS)):
            clip.seek(frame_index % clip.n_frames)
            robot = clip.convert("RGBA")
            robot.thumbnail((405, 405), Image.Resampling.LANCZOS)
            x, y = (i % 3) * cell_w, (i // 3) * cell_h
            sheet.paste(robot, (x + (cell_w - robot.width) // 2, y + 8), robot)
            draw.text((x + cell_w // 2, y + 430), label, font=label_font, fill="#132431", anchor="mm")
            draw.text((x + cell_w // 2, y + 467), "same black/white body scheme", font=detail_font, fill="#667680", anchor="mm")
        frames.append(sheet)
    dst = OUT / "paddleclaw_hi_variants_contact_sheet.gif"
    frames[0].save(
        dst,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / FPS),
        loop=0,
        optimize=False,
    )
    return dst


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rendered = [render_variant(name) for name, _ in VARIANTS]
    sheet = make_contact_sheet(rendered)
    for path in rendered:
        print(path)
    print(sheet)
