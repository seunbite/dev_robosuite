"""Draw a small palm-facing arrow on pose review renders (G1 / Google)."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw


ARROW_RGBA = (1.0, 0.35, 0.08, 1.0)  # vivid orange
ARROW_RGB = (255, 90, 20)
ARROW_TIP_RGB = (255, 220, 60)


def add_mj_arrow(
    scene,
    start: np.ndarray,
    end: np.ndarray,
    *,
    width: float = 0.014,
    rgba: Sequence[float] = ARROW_RGBA,
) -> None:
    """Append a MuJoCo scene ARROW geom (for mujoco.Renderer after update_scene)."""
    import mujoco

    i = int(scene.ngeom)
    if i >= int(scene.maxgeom):
        return
    mujoco.mjv_initGeom(
        scene.geoms[i],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        scene.geoms[i],
        mujoco.mjtGeom.mjGEOM_ARROW,
        float(width),
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
    )
    scene.ngeom = i + 1


def g1_palm_origin_normal(model, data) -> tuple[np.ndarray, np.ndarray]:
    """Palm origin + outward normal (right_palm site +Z / project convention)."""
    sid = int(model.site("right_palm").id)
    origin = np.asarray(data.site_xpos[sid], dtype=float)
    rot = np.asarray(data.site_xmat[sid], dtype=float).reshape(3, 3)
    normal = rot[:, 2]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    return origin, normal


def overlay_g1_palm_arrow(
    renderer,
    model,
    data,
    *,
    length: float = 0.13,
    width: float = 0.015,
) -> None:
    """Call after renderer.update_scene(...); mutates renderer.scene."""
    origin, normal = g1_palm_origin_normal(model, data)
    add_mj_arrow(renderer.scene, origin, origin + length * normal, width=width)


def project_sim_cam(
    sim,
    camera_name: str,
    world_xyz: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[float, float] | None:
    """Project world point to image pixels (MuJoCo camera, look along -Z)."""
    cam_id = sim.model.camera_name2id(camera_name)
    pos = np.asarray(sim.data.cam_xpos[cam_id], dtype=float)
    mat = np.asarray(sim.data.cam_xmat[cam_id], dtype=float).reshape(3, 3)
    pc = mat.T @ (np.asarray(world_xyz, dtype=float) - pos)
    depth = -float(pc[2])
    if depth <= 1e-6:
        return None
    fovy = float(sim.model.cam_fovy[cam_id])
    f = 0.5 * height / math.tan(math.radians(fovy) / 2.0)
    u = 0.5 * width + f * float(pc[0]) / depth
    v = 0.5 * height - f * float(pc[1]) / depth
    return float(u), float(v)


def google_palm_origin_normal(sim) -> tuple[np.ndarray, np.ndarray] | None:
    """Palm at grip site; normal = grip-site +Y (= pad-plane normal)."""
    try:
        sid = sim.model.site_name2id("gripper0_right_grip_site")
    except Exception:
        return None
    origin = np.asarray(sim.data.site_xpos[sid], dtype=float)
    rot = np.asarray(sim.data.site_xmat[sid], dtype=float).reshape(3, 3)
    normal = rot[:, 1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    return origin, normal


def draw_arrow_pil(
    image: Image.Image,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    *,
    color: tuple[int, int, int] = ARROW_RGB,
    tip_color: tuple[int, int, int] = ARROW_TIP_RGB,
    width: int = 5,
    head: float = 14.0,
) -> None:
    dr = ImageDraw.Draw(image)
    a = (float(start_xy[0]), float(start_xy[1]))
    b = (float(end_xy[0]), float(end_xy[1]))
    dr.line([a, b], fill=color, width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for da in (2.55, -2.55):
        p = (b[0] - head * math.cos(ang + da), b[1] - head * math.sin(ang + da))
        dr.line([b, p], fill=color, width=width)
    r = max(3, width)
    dr.ellipse((a[0] - r, a[1] - r, a[0] + r, a[1] + r), fill=tip_color)


def overlay_google_palm_arrow(
    image: Image.Image,
    sim,
    camera_name: str,
    *,
    length: float = 0.13,
    width: int = 5,
) -> Image.Image:
    """Draw palm arrow on an already-captured Tiago RGB frame."""
    pn = google_palm_origin_normal(sim)
    if pn is None:
        return image
    origin, normal = pn
    w, h = image.size
    a = project_sim_cam(sim, camera_name, origin, width=w, height=h)
    b = project_sim_cam(sim, camera_name, origin + length * normal, width=w, height=h)
    if a is None or b is None:
        return image
    # Skip if completely off-frame
    if max(a[0], b[0]) < -20 or min(a[0], b[0]) > w + 20:
        return image
    if max(a[1], b[1]) < -20 or min(a[1], b[1]) > h + 20:
        return image
    draw_arrow_pil(image, a, b, width=width)
    return image
