from __future__ import annotations

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SRC = ROOT / "papers/icra2027_motion_abstraction/synthetic_morphology/render_fewshot_cues.py"
XML = ROOT / "papers/icra2027_motion_abstraction/synthetic_morphology/xml/gcr_alpha1.xml"
OUT = ROOT / "output/paddleclaw_bw_variants"

BLACK = (0.07, 0.075, 0.085, 1.0)
WHITE = (0.92, 0.93, 0.95, 1.0)

# Each option uses only the same near-black and near-white materials.
VARIANTS = {
    "A_balanced": {
        "body": BLACK, "leg": BLACK, "arm": WHITE, "spine": WHITE,
        "head": WHITE, "joint": BLACK, "marker": BLACK,
    },
    "B_white_shell": {
        "body": WHITE, "leg": BLACK, "arm": WHITE, "spine": BLACK,
        "head": WHITE, "joint": BLACK, "marker": BLACK,
    },
    "C_black_head": {
        "body": BLACK, "leg": WHITE, "arm": WHITE, "spine": WHITE,
        "head": BLACK, "joint": BLACK, "marker": WHITE,
    },
    "D_black_branches": {
        "body": WHITE, "leg": WHITE, "arm": BLACK, "spine": BLACK,
        "head": WHITE, "joint": BLACK, "marker": BLACK,
    },
    "E_white_claw_head": {
        "body": BLACK, "leg": BLACK, "arm": WHITE, "spine": BLACK,
        "head": WHITE, "joint": WHITE, "marker": BLACK,
    },
    "F_black_claw_head": {
        "body": WHITE, "leg": BLACK, "arm": BLACK, "spine": WHITE,
        "head": BLACK, "joint": WHITE, "marker": WHITE,
    },
}

LABELS = {
    "A_balanced": "A · Balanced",
    "B_white_shell": "B · White shell",
    "C_black_head": "C · Black head",
    "D_black_branches": "D · Black branches",
    "E_white_claw_head": "E · White claw + head",
    "F_black_claw_head": "F · Black claw + head",
}

spec = importlib.util.spec_from_file_location("fewshot", SRC)
fewshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fewshot)


def set_material(model: mujoco.MjModel, name: str, rgba: tuple[float, ...]) -> None:
    material_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
    if material_id < 0:
        raise KeyError(name)
    model.mat_rgba[material_id] = rgba


def render_variant(name: str, palette: dict[str, tuple[float, ...]]) -> Path:
    model = mujoco.MjModel.from_xml_path(str(XML))
    for role, rgba in palette.items():
        set_material(model, role, rgba)

    # Brighter, soft, neutral studio lighting.
    model.vis.headlight.ambient[:] = (0.78, 0.78, 0.78)
    model.vis.headlight.diffuse[:] = (0.78, 0.78, 0.78)
    model.vis.headlight.specular[:] = (0.16, 0.16, 0.16)
    if model.nlight:
        model.light_diffuse[:] = (0.62, 0.62, 0.62)
        model.light_specular[:] = (0.12, 0.12, 0.12)

    data = mujoco.MjData(model)
    fewshot._set_base(data, model, 0.0, 0.0, 0.0, 0.13)
    joints = fewshot.home1({
        "br0": -0.12,
        "br1": -0.42,
        "br2": -0.28,
        "br3": 0.62,
        "claw0": 0.90,
        "claw1": -0.90,
        "claw2": -0.44,
        "bl0": 0.25,
        "bl1": 0.25,
        "bl2": 0.0,
    })
    for joint, value in joints.items():
        fewshot._set_joint(data, model, joint, value)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = fewshot._cam((0.0, 0.0, 0.43), 1.62)
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    seg = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    renderer.close()

    obj_id = seg[..., 0]
    obj_type = seg[..., 1]
    # The plane is the first geom in this model. Everything else is the robot.
    mask = (obj_id >= 0) & ~((obj_type == int(mujoco.mjtObj.mjOBJ_GEOM)) & (obj_id == 0))
    ys, xs = np.where(mask)
    if not len(xs):
        raise RuntimeError("Segmentation produced an empty robot mask")
    pad = 28
    x0, x1 = max(0, int(xs.min()) - pad), min(rgb.shape[1], int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(rgb.shape[0], int(ys.max()) + pad + 1)

    crop = Image.fromarray(rgb[y0:y1, x0:x1], "RGB")
    crop = ImageEnhance.Brightness(crop).enhance(1.08)
    alpha = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8), "L")
    rgba = crop.convert("RGBA")
    rgba.putalpha(alpha)

    canvas = Image.new("RGBA", (640, 640), (255, 255, 255, 0))
    rgba.thumbnail((590, 590), Image.Resampling.LANCZOS)
    canvas.alpha_composite(rgba, ((640 - rgba.width) // 2, (640 - rgba.height) // 2))
    dst = OUT / f"{name}.png"
    canvas.save(dst)
    return dst


def make_contact_sheet(paths: list[Path]) -> Path:
    cell_w, cell_h = 480, 540
    sheet = Image.new("RGB", (cell_w * 3, cell_h * 2), "white")
    draw = ImageDraw.Draw(sheet)
    font_path = "/System/Library/Fonts/Helvetica.ttc"
    label_font = ImageFont.truetype(font_path, 28, index=1)
    detail_font = ImageFont.truetype(font_path, 18, index=0)
    for i, path in enumerate(paths):
        x, y = (i % 3) * cell_w, (i // 3) * cell_h
        robot = Image.open(path).convert("RGBA")
        robot.thumbnail((420, 420), Image.Resampling.LANCZOS)
        shadow = Image.new("RGBA", robot.size, (0, 0, 0, 0))
        shadow.putalpha(robot.getchannel("A").point(lambda v: int(v * 0.13)))
        sheet.paste(shadow, (x + (cell_w - robot.width) // 2 + 7, y + 30 + 9), shadow)
        sheet.paste(robot, (x + (cell_w - robot.width) // 2, y + 30), robot)
        label = LABELS[path.stem]
        draw.text((x + cell_w // 2, y + 465), label, font=label_font, fill="#132431", anchor="mm")
        draw.text((x + cell_w // 2, y + 505), "bright studio light · transparent PNG", font=detail_font, fill="#667680", anchor="mm")
    dst = OUT / "paddleclaw_bw_contact_sheet.png"
    sheet.save(dst)
    return dst


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rendered = [render_variant(name, palette) for name, palette in VARIANTS.items()]
    contact_sheet = make_contact_sheet(rendered)
    for path in rendered:
        print(path)
    print(contact_sheet)
