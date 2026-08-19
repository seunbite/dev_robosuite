#!/usr/bin/env python3
"""Render pose/movement/path schema visualizations for TIAGo (Tiago proxy).

Produces:
  - PNGs: pose parameter variations (torso_height, arm_position, gripper_orientation, head)
  - PNG: brute grid (``--arm6-bruteforce``); 12 stripes; cells: ``[joints]``, ``dir, orient``, ``(x, y, z)`` pct.
  - PNGs: 729 clustered ``dir × (horizontal|vertical)`` (``--arm6-cluster-12``). Cell captions: ``[joints]``, ``dir, orient``, ``(x, y, z)`` pct.
  - PNGs: movement joint ranges (shoulder, elbow, wrist, torso, head)
  - GIFs: path types (line with x/y, arc with radius/degrees)
  - HTML: combined dashboard

Run::
  PYTHONPATH=. .venv/bin/python adhoc/generation/tiago/legacy/render_mobile_manip_schema.py --arm6-bruteforce
  PYTHONPATH=. .venv/bin/python adhoc/generation/tiago/legacy/render_mobile_manip_schema.py --arm6-cluster-12

Optional env — render width/height in pixels (square recommended; clamped 256–2048):
``MOBILE_MANIP_SCHEMA_RENDER_W``, ``MOBILE_MANIP_SCHEMA_RENDER_H``.
"""
from __future__ import annotations

import itertools
import math
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

local_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if local_root not in sys.path:
    sys.path.insert(0, local_root)

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)

OUT_DIR = os.path.join(local_root, "data", "results", "render", "mobile_manip_schema")
os.makedirs(OUT_DIR, exist_ok=True)

# Offscreen render size (pixels). Larger = sharper schema tiles / bruteforce / cluster PNGs (slower, bigger files).
_DEFAULT_RENDER_WH = (896, 896)
_dw, _dh = _DEFAULT_RENDER_WH
_w_env = os.environ.get("MOBILE_MANIP_SCHEMA_RENDER_W")
_h_env = os.environ.get("MOBILE_MANIP_SCHEMA_RENDER_H")
W = int(_w_env) if _w_env else _dw
H = int(_h_env) if _h_env else _dh
# MuJoCo / GPU friendly square bounds.
W = max(256, min(2048, W))
H = max(256, min(2048, H))

CAM = "frontview"
CAM_SIDE = "sideview"

# Anchored to legacy 512px layout so captions scale when W/H changes.
RENDER_BASE_PX = 512


def _render_ui_scale() -> float:
    return float(min(H, W)) / float(RENDER_BASE_PX)


def _scaled_thumb_layout(*, n_caption_lines: int = 3) -> tuple[int, int, int, int]:
    """(cell_w, img_crop_h, cap_h, caption_font_px) for ``_cell_thumb_with_caption``."""
    s = _render_ui_scale()
    nlines = max(2, min(6, int(n_caption_lines)))
    cw = max(206, int(round(206 * s)))
    ih = max(112, int(round(112 * s)))
    fz = max(11, int(round(10 * s)))
    gap = max(8, int(round(6 * s)))
    line_px = fz + gap
    pad_bottom = max(14, int(round(14 * s)))
    cap = line_px * nlines + pad_bottom
    return cw, ih, max(56, cap), fz


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

# Right-arm wrist frame (MuJoCo link before gripper); extension dir = midpoint(fingertips) − wrist.
_WRIST_BODY_RIGHT = "robot0_arm_right_6_link"
_joint_angle_grid = (-90, 0, 90)

# Lexical 12-strip layout for brute montage (6-way dir alphabet × pads H/V alphabet).
_BF_STRIP_DIRS = tuple(sorted(("back", "down", "front", "left", "right", "up")))
_BF_STRIP_ORIENTS = tuple(sorted(("horizontal", "vertical")))

# World ±Z as ceiling/floor (“up/down”); horizontal front/left/right from frontview gaze.
_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# Robotiq85 on Tiago right arm — fingertip collision geoms from composed MJCF.
_GRIP_RIGHT_FINGERTIP_GEOMS = (
    "gripper0_right_left_fingertip_collision",
    "gripper0_right_right_fingertip_collision",
)
# Min projected length (meters) for pad–pad segment to count as a “line” vs “point” after dir-dependent projection.
_GRIP_PROJECT_MIN_LINE_M = 0.012


def _right_fingertips_world_xyz(sim) -> tuple[np.ndarray, np.ndarray] | None:
    """Robotiq85 right gripper fingertip collision frame centers (world)."""
    try:
        g0 = sim.model.geom_name2id(_GRIP_RIGHT_FINGERTIP_GEOMS[0])
        g1 = sim.model.geom_name2id(_GRIP_RIGHT_FINGERTIP_GEOMS[1])
    except Exception:
        return None
    return (
        np.asarray(sim.data.geom_xpos[g0], dtype=np.float64),
        np.asarray(sim.data.geom_xpos[g1], dtype=np.float64),
    )


def _right_fingertip_pad_mid_world_xyz(sim) -> np.ndarray | None:
    """World-frame midpoint of the two right fingertip collision sites (gripper pad center)."""
    pair = _right_fingertips_world_xyz(sim)
    if pair is None:
        return None
    p0, p1 = pair
    return np.asarray(0.5 * (p0 + p1), dtype=np.float64)


def _world_xyz_pct_bounds(stack_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis min / max for mapping world xyz → 0…100 % (see ``_world_xyz_to_pct_0100``)."""
    arr = np.asarray(stack_xyz, dtype=np.float64)
    return arr.min(axis=0), arr.max(axis=0)


def _clamp_center_into_bounds(lo: np.ndarray, hi: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Keep body-center inside [lo, hi] with a tiny interior margin so piecewise spans stay valid."""
    a = np.asarray(lo, dtype=np.float64).reshape(3,)
    b = np.asarray(hi, dtype=np.float64).reshape(3,)
    c = np.asarray(center, dtype=np.float64).reshape(3,).copy()
    for i in range(3):
        span = float(b[i] - a[i])
        if span < 1e-12:
            c[i] = float(a[i])
            continue
        margin = max(1e-6, 1e-4 * span)
        c[i] = float(np.clip(c[i], a[i] + margin, b[i] - margin))
    return c


def _world_axis_to_pct_0100(coord: float, lo: float, hi: float, center: float | None = None) -> float:
    """Linear map ``lo→0`` … ``hi→100`` (``center`` ignored; kept for call-site compat)."""
    del center
    span = float(hi - lo)
    if span < 1e-12:
        return 50.0
    return float(np.clip((float(coord) - float(lo)) / span * 100.0, 0.0, 100.0))


def _pct_0100_to_world_axis(pct: float, lo: float, hi: float, center: float | None = None) -> float:
    """Inverse of linear ``_world_axis_to_pct_0100`` (``center`` ignored)."""
    del center
    p = float(np.clip(float(pct), 0.0, 100.0))
    span = float(hi - lo)
    if span < 1e-12:
        return float(lo)
    return float(lo) + p / 100.0 * span


def _world_xyz_to_pct_0100(
    xyz: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    center: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Map world xyz → [0, 100] linearly over pad-mid extrema (``center`` ignored).

    Matches G1-style uniform percentile scale: min→0, max→100.
    """
    del center
    v = np.asarray(xyz, dtype=np.float64).reshape(3,)
    a = np.asarray(lo, dtype=np.float64).reshape(3,)
    b = np.asarray(hi, dtype=np.float64).reshape(3,)
    return (
        _world_axis_to_pct_0100(float(v[0]), float(a[0]), float(b[0])),
        _world_axis_to_pct_0100(float(v[1]), float(a[1]), float(b[1])),
        _world_axis_to_pct_0100(float(v[2]), float(a[2]), float(b[2])),
    )


def _project_orthogonal_to_floor(v: np.ndarray, up: np.ndarray = _WORLD_UP) -> np.ndarray:
    """Project vector onto horizontal plane (⊥ gravity / floor span)."""
    u = np.asarray(up, dtype=np.float64).reshape(3,)
    nu = np.linalg.norm(u)
    if nu < 1e-15:
        return np.asarray(v, dtype=np.float64).copy()
    u = u / nu
    return np.asarray(v, dtype=np.float64) - float(np.dot(v, u)) * u


def _project_onto_world_yz_plane(v: np.ndarray) -> np.ndarray:
    """Orthogonal projection onto world-fixed YZ plane (remove world X component)."""
    vv = np.asarray(v, dtype=np.float64).reshape(3,)
    return np.array([0.0, vv[1], vv[2]], dtype=np.float64)


def _gripper_horizontal_vertical_by_reach_dir(sim, reach_dir: str) -> str:
    """H/V from pad segment after projection plane chosen by wrist→pads six-way dir.

    * ``front/back/left/right``: project onto horizontal (floor / ⊥ gravity): line ⇒ horizontal, dot ⇒ vertical.
    * ``up/down``: project onto world YZ: same line vs dot rule.
    """
    if reach_dir in ("?", ""):
        return "?"
    pair = _right_fingertips_world_xyz(sim)
    if pair is None:
        return "?"
    e = pair[1] - pair[0]
    if reach_dir in ("up", "down"):
        span = float(np.linalg.norm(_project_onto_world_yz_plane(e)))
    elif reach_dir in ("front", "back", "left", "right"):
        span = float(np.linalg.norm(_project_orthogonal_to_floor(e)))
    else:
        return "?"
    return "horizontal" if span >= _GRIP_PROJECT_MIN_LINE_M else "vertical"


def _cam_world_R_from_corr_cam(sim, camera_name: str) -> np.ndarray:
    """Same rotation as robosuite ``get_camera_extrinsic_matrix`` (avoids importing camera_utils→h5py)."""
    from robosuite.utils import transform_utils as Tu

    cam_id = sim.model.camera_name2id(camera_name)
    pose = Tu.make_pose(sim.data.cam_xpos[cam_id], sim.data.cam_xmat[cam_id].reshape(3, 3))
    corr = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    return (pose @ corr)[:3, :3]


def _horizontal_gaze_flat(sim, cam_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors (world): ``fh`` horizontal camera gaze (+ = into scene), ``rr`` viewer's right (= cross(fh, world_up))."""
    Rw = _cam_world_R_from_corr_cam(sim, cam_name)
    cam_fwd_world = Rw[:, 2]  # +Z_corr = optical axis after robosuite fix
    u = _WORLD_UP
    fh = cam_fwd_world - float(np.dot(cam_fwd_world, u)) * u
    nf = np.linalg.norm(fh)
    if nf < 1e-9:
        fh = np.array([1.0, 0.0, 0.0])
        nf = 1.0
    fh /= nf
    r_view_right = np.cross(fh, u)
    nr = np.linalg.norm(r_view_right)
    if nr < 1e-9:
        r_view_right = np.array([0.0, 1.0, 0.0])
        nr = 1.0
    r_view_right /= nr
    return fh, r_view_right


def _direction_label_viewer_frontview(sim, vec_world: np.ndarray, cam_name: str) -> str:
    """Label ``vec_world`` (e.g. grip extension) into up/down/front/back/left/right (viewer frontview + world Z)."""
    fh, rr = _horizontal_gaze_flat(sim, cam_name)
    dz = float(np.dot(vec_world, _WORLD_UP))
    df = float(np.dot(vec_world, fh))
    dr = float(np.dot(vec_world, rr))
    ax = [(abs(dz), "ud", dz), (abs(df), "fb", df), (abs(dr), "lr", dr)]
    _, tag, dv = max(ax, key=lambda t: t[0])
    if tag == "ud":
        return "up" if dv > 0 else "down"
    if tag == "fb":
        # ``fh`` points into the scene along the camera; empirical check: flip so “front” matches user view.
        return "front" if dv < 0 else "back"
    return "right" if dv > 0 else "left"


def _direction_grip_wrist_to_pads(sim, wrist_body: str, cam_name: str) -> str:
    """Grip “reach” dir: midpoint(fingertip pads) − wrist (same 6-way as before, viewer + world Z)."""
    fts = _right_fingertips_world_xyz(sim)
    if fts is None:
        return "?"
    p0, p1 = fts
    try:
        wid = sim.model.body_name2id(wrist_body)
    except ValueError:
        return "?"
    pw = np.asarray(sim.data.body_xpos[wid], dtype=np.float64)
    pad_mid = 0.5 * (p0 + p1)
    return _direction_label_viewer_frontview(sim, pad_mid - pw, cam_name)


def _dir_six_way_from_anchor(env, qpos: np.ndarray, *, cam_name: str = CAM) -> str:
    env.sim.data.qpos[:] = qpos
    env.sim.forward()
    return _direction_grip_wrist_to_pads(env.sim, _WRIST_BODY_RIGHT, cam_name)


def _caption_font_try(font_px: int, font_try: tuple[str, ...]):
    for fp in font_try:
        try:
            return ImageFont.truetype(fp, font_px)
        except Exception:
            continue
    return ImageFont.load_default()


def _measure_caption_area_height(
    texts: list[str],
    font,
    *,
    pad_between: int,
    pad_top: int,
    pad_bottom: int,
) -> int:
    """Pixel height needed below the thumbnail for ``texts`` (single-column lines)."""
    draw_probe = ImageDraw.Draw(Image.new("RGB", (16, 16)))
    h = pad_top
    for i, line in enumerate(texts):
        try:
            bbox = font.getbbox(line)
            line_h = bbox[3] - bbox[1]
        except Exception:
            try:
                bbox = draw_probe.textbbox((0, 0), line, font=font)
                line_h = bbox[3] - bbox[1]
            except Exception:
                line_h = max(12, pad_between + 8)
        h += max(1, line_h)
        if i < len(texts) - 1:
            h += pad_between
    return h + pad_bottom


def _max_caption_line_width_px(texts: list[str], font) -> int:
    draw_probe = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    mw = 0
    for t in texts:
        try:
            bbox = font.getbbox(t)
            w = bbox[2] - bbox[0]
        except Exception:
            try:
                bb = draw_probe.textbbox((0, 0), t, font=font)
                w = bb[2] - bb[0]
            except Exception:
                w = len(t) * 6
        mw = max(mw, int(w))
    return mw


def _cell_thumb_with_caption(
    thumb: Image.Image,
    joint_deg: tuple[int, ...],
    dir_lbl: str,
    grip_lbl: str,
    *,
    cell_w: int | None = None,
    img_h: int | None = None,
    cap_h: int | None = None,
    caption_font_px: int | None = None,
    ee_pad_mid_pct_xyz: tuple[float, float, float] | None = None,
    font_try: tuple[str, ...] = (
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ),
) -> Image.Image:
    n_cap_lines = 3 if ee_pad_mid_pct_xyz is not None else 2
    d_cw, d_ih, d_cap, d_fz = _scaled_thumb_layout(n_caption_lines=n_cap_lines)
    cell_w_u = d_cw if cell_w is None else cell_w
    img_h_u = d_ih if img_h is None else img_h
    caption_font_px_u = d_fz if caption_font_px is None else caption_font_px
    jline = "[" + ",".join(f"{int(x):d}" for x in joint_deg) + "]"
    texts = [
        jline,
        f"{dir_lbl}, {grip_lbl}",
    ]
    if ee_pad_mid_pct_xyz is not None:
        px, py, pz = ee_pad_mid_pct_xyz
        texts.append(f"({px:.0f}, {py:.0f}, {pz:.0f})")

    su = _render_ui_scale()
    pad_between = max(2, int(round(2 * su)))
    pad_top = max(3, int(round(3 * su)))
    pad_bottom = max(10, int(round(12 * su)))
    sx = max(4, int(round(4 * su)))

    # Shrink monospace caption font until three lines fit the tile width (avoids clipping).
    margin = cell_w_u - 2 * sx - max(4, int(round(4 * su)))
    fz = int(caption_font_px_u)
    font = _caption_font_try(fz, font_try)
    while fz > 6 and _max_caption_line_width_px(texts, font) > margin:
        fz -= 1
        font = _caption_font_try(fz, font_try)

    cap_meas = _measure_caption_area_height(
        texts,
        font,
        pad_between=pad_between,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
    )
    cap_floor = d_cap if cap_h is None else cap_h
    cap_final = max(cap_floor, cap_meas)

    out = Image.new("RGB", (cell_w_u, img_h_u + cap_final), (255, 255, 255))
    th = thumb.resize((cell_w_u, img_h_u), Image.LANCZOS)
    out.paste(th, (0, 0))
    draw = ImageDraw.Draw(out)
    y = img_h_u + pad_top
    for i, line in enumerate(texts):
        draw.text((sx, y), line, fill=(20, 30, 40), font=font)
        try:
            bbox = font.getbbox(line)
            dy = max(1, bbox[3] - bbox[1])
        except Exception:
            dy = max(12, int(round(12 * su)))
        y += dy
        if i < len(texts) - 1:
            y += pad_between
    return out


def _vstack_rgb_montages(
    chunks: list[Image.Image],
    *,
    gutter: int | None = None,
    bg: tuple[int, int, int] = (246, 248, 251),
    align_pad_to_max_width: bool = True,
) -> Image.Image:
    """Paste caption-montages top-to-bottom; optional center-short strips to max width."""
    s = _render_ui_scale()
    g = gutter if gutter is not None else max(8, int(round(10 * s)))
    rgbs = [c.convert("RGB") for c in chunks if getattr(c, "size", None) and c.size[1] >= 2]
    if not rgbs:
        return Image.new("RGB", (32, 32), bg)

    wmax = max(im.size[0] for im in rgbs)
    htot = sum(im.size[1] for im in rgbs) + g * max(0, len(rgbs) - 1)
    canvas = Image.new("RGB", (wmax if align_pad_to_max_width else rgbs[0].size[0], htot), bg)
    y = 0
    for im in rgbs:
        x = 0
        if align_pad_to_max_width and im.size[0] < wmax:
            x = (wmax - im.size[0]) // 2
        canvas.paste(im, (x, y))
        y += im.size[1] + g
    return canvas


def render_arm6_bruteforce_grid(env) -> str:
    """3^6 right-arm grids: twelve lexical (dir × H/V) bands; within each band sort by pad-mid world (x,y,z)."""
    deg_steps = list(_joint_angle_grid)
    n = len(deg_steps) ** 6
    cols = 27
    base = _default_qpos(env)
    base[QI_TORSO] = TORSO_PRESETS["high"]
    base[QI_HEAD_PAN] = 0.0
    base[QI_HEAD_TILT] = 0.0

    cell_template = Image.new("RGB", (32, 32))
    _, _ = _cell_thumb_with_caption(
        cell_template,
        (0,) * 6,
        "?",
        "?",
        ee_pad_mid_pct_xyz=(12.3, 45.6, 78.9),
    ).size

    s = _render_ui_scale()
    title_h = max(48, int(round(52 * s)))
    combos = itertools.product(deg_steps, repeat=6)

    records: list[tuple[tuple[int, ...], np.ndarray, str, str]] = []
    for tup in tqdm(combos, total=n, desc="arm 3^6 classify"):
        q = base.copy()
        rad = [math.radians(float(d)) for d in tup]
        q[6:12] = np.asarray(rad, dtype=float)
        env.sim.data.qpos[:] = q
        env.sim.forward()
        lbl_dir = _dir_six_way_from_anchor(env, q, cam_name=CAM)
        lbl_grip = _gripper_horizontal_vertical_by_reach_dir(env.sim, lbl_dir)
        mid_w = _right_fingertip_pad_mid_world_xyz(env.sim)
        if mid_w is None:
            mid_w = np.zeros(3, dtype=np.float64)
        records.append((tup, mid_w, lbl_dir, lbl_grip))

    mids_stack = np.stack([r[1] for r in records], axis=0)
    lo_xyz, hi_xyz = _world_xyz_pct_bounds(mids_stack)

    buckets: defaultdict = defaultdict(list)
    unclassified: list[tuple[tuple[int, ...], np.ndarray, str, str]] = []
    for rec in records:
        tup, mid_w, lbl_dir, lbl_grip = rec
        if lbl_dir in _BF_STRIP_DIRS and lbl_grip in _BF_STRIP_ORIENTS:
            buckets[(lbl_dir, lbl_grip)].append(rec)
        else:
            unclassified.append(rec)

    sort_xyz = lambda recs: sorted(
        recs, key=lambda z: (float(z[1][0]), float(z[1][1]), float(z[1][2]))
    )

    def _capture_cell(tup, mid_w, lbl_dir, lbl_grip):
        qc = base.copy()
        rad = [math.radians(float(d)) for d in tup]
        qc[6:12] = np.asarray(rad, dtype=float)
        frame = _apply_and_capture_kinematic(env, qc, camera=CAM)
        pct = _world_xyz_to_pct_0100(mid_w, lo_xyz, hi_xyz)
        return _cell_thumb_with_caption(frame, tup, lbl_dir, lbl_grip, ee_pad_mid_pct_xyz=pct)

    stripe_imgs: list[Image.Image] = []
    pbar_render = tqdm(total=n, desc="arm 3^6 render (12 lexical stripes + xyz)")
    try:
        for d in _BF_STRIP_DIRS:
            for ori in _BF_STRIP_ORIENTS:
                bucket = sort_xyz(buckets[(d, ori)])
                cells_run: list[Image.Image] = []
                for tup, mid_w, lbl_dir, lbl_grip in bucket:
                    cells_run.append(_capture_cell(tup, mid_w, lbl_dir, lbl_grip))
                    pbar_render.update(1)

                ttl = f"{d} · {ori} · {len(bucket)}"
                stripe_imgs.append(_montage_caption_cells(cells_run, cols=cols, title=ttl))

        stray_items = sort_xyz(unclassified)
        if stray_items:
            cells_s: list[Image.Image] = []
            for tup, mid_w, lbl_dir, lbl_grip in stray_items:
                cells_s.append(_capture_cell(tup, mid_w, lbl_dir, lbl_grip))
                pbar_render.update(1)
            stripe_imgs.append(
                _montage_caption_cells(
                    cells_s,
                    cols=cols,
                    title=f"? · {len(stray_items)}",
                )
            )

    finally:
        pbar_render.close()

    body = _vstack_rgb_montages(stripe_imgs, gutter=max(10, int(round(12 * s))))
    out_w = body.size[0]
    out_full = Image.new("RGB", (out_w, body.size[1] + title_h), (246, 248, 251))
    try:
        tfont = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(14, int(round(18 * s))))
    except Exception:
        tfont = ImageFont.load_default()
    pad_txt = max(12, int(round(12 * s)))
    banner = "729 · 12 bands · xyz↑"
    ImageDraw.Draw(out_full).text((pad_txt, max(10, int(round(12 * s)))), banner, fill=(31, 35, 40), font=tfont)
    out_full.paste(body, (0, title_h))
    path_out = os.path.join(OUT_DIR, "pose_arm_bruteforce_3pow6_deg.png")
    out_full.save(path_out, optimize=True)
    print(f"  arm 3^6 grid saved: {path_out} ({n} cells)")
    print(
        "  pct basis pad-mid world m (729): "
        f"x[{lo_xyz[0]:.5f},{hi_xyz[0]:.5f}] "
        f"y[{lo_xyz[1]:.5f},{hi_xyz[1]:.5f}] "
        f"z[{lo_xyz[2]:.5f},{hi_xyz[2]:.5f}]"
    )
    return path_out


# Order for 12-way clustering: 6 reach dirs × pad opening (floor/YZ proj).
CLUSTER_DIRS_ORDER = ("front", "back", "left", "right", "up", "down")
CLUSTER_ORIENT_ORDER = ("horizontal", "vertical")


def _cluster_cols_for_cell_count(n: int) -> int:
    if n <= 0:
        return 1
    return min(27, max(5, int(math.ceil(math.sqrt(max(1, n))))))


def _montage_caption_cells(
    cells: list[Image.Image],
    *,
    cols: int,
    title: str,
    title_h: int | None = None,
    bg_rgb: tuple[int, int, int] = (246, 248, 251),
    title_font_px: int | None = None,
) -> Image.Image:
    s = _render_ui_scale()
    th = title_h if title_h is not None else max(52, int(round(52 * s)))
    tfp = title_font_px if title_font_px is not None else max(14, int(round(20 * s)))
    stub_w = max(420, int(round(420 * s)))
    stub_h = max(200, int(round(200 * s)))

    if cols < 1:
        cols = 1
    n_cells = len(cells)
    if n_cells == 0:
        out = Image.new("RGB", (stub_w, th + stub_h), bg_rgb)
        dr = ImageDraw.Draw(out)
        try:
            tfont = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", tfp)
        except Exception:
            tfont = ImageFont.load_default()
        dr.text((16, th // 4), title, fill=(31, 35, 40), font=tfont)
        dr.text(
            (16, th + max(36, int(round(40 * s)))), "(no poses in this bucket)", fill=(96, 108, 120), font=tfont
        )
        return out

    cw, ch = cells[0].size
    rows = int(math.ceil(n_cells / cols))
    canvas_w = cw * cols
    canvas_h = ch * rows
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_rgb)
    for i, cell in enumerate(cells):
        r, ci = divmod(i, cols)
        canvas.paste(cell, (ci * cw, r * ch))

    out = Image.new("RGB", (canvas_w, canvas_h + th), bg_rgb)
    out.paste(canvas, (0, th))
    dr = ImageDraw.Draw(out)
    try:
        tfont = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", tfp)
    except Exception:
        tfont = ImageFont.load_default()
    pad = max(12, int(round(12 * s)))
    dr.text((pad, max(10, th // 4)), title, fill=(31, 35, 40), font=tfont)
    return out


def _clusters_6x2_overview(
    thumb_by_key: dict[tuple[str, str], Image.Image],
    *,
    top_title: str = "729 pool · dir × pads h/v clusters (thumbnail)",
    overview_thumb_w: int | None = None,
    gutter_x: int | None = None,
    gutter_y: int | None = None,
    col_header_h: int | None = None,
    top_title_h: int | None = None,
    left_col_w: int | None = None,
) -> Image.Image:
    """Stack 12 cluster montages in 6×2 grid (columns = horizontal | vertical)."""
    s = _render_ui_scale()
    otw = overview_thumb_w if overview_thumb_w is not None else max(580, min(1200, int(round(670 * s))))
    gx = gutter_x if gutter_x is not None else max(11, int(round(14 * s)))
    gy = gutter_y if gutter_y is not None else max(28, int(round(38 * s)))
    chh = col_header_h if col_header_h is not None else max(24, int(round(28 * s)))
    tph = top_title_h if top_title_h is not None else max(40, int(round(48 * s)))
    lcol = left_col_w if left_col_w is not None else max(124, int(round(120 * s)))

    norm: dict[tuple[str, str], Image.Image] = {}
    for d in CLUSTER_DIRS_ORDER:
        for o in CLUSTER_ORIENT_ORDER:
            im = thumb_by_key[(d, o)].convert("RGB")
            w0, h0 = im.size
            if w0 < 2:
                w0 = 2
            scale_thumb = otw / float(w0)
            nh = max(8, int(round(h0 * scale_thumb)))
            norm[(d, o)] = im.resize((otw, nh), Image.LANCZOS)

    hfp = max(15, int(round(18 * s)))
    cfp = max(14, int(round(16 * s)))
    dfp = max(13, int(round(15 * s)))
    try:
        hfont = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", hfp)
        cfont = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", cfp)
        dfont = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", dfp)
    except Exception:
        hfont = cfont = dfont = ImageFont.load_default()

    row_heights: list[int] = []
    for d in CLUSTER_DIRS_ORDER:
        hmax = max(norm[(d, o)].size[1] for o in CLUSTER_ORIENT_ORDER)
        row_heights.append(int(hmax + gy))

    body_w = lcol + gx + (otw + gx) * len(CLUSTER_ORIENT_ORDER)
    body_h_top = tph + chh
    canvas_h = body_h_top + sum(row_heights)
    canvas = Image.new("RGB", (body_w, canvas_h), (246, 248, 251))
    dr = ImageDraw.Draw(canvas)
    dr.text((max(14, int(round(16 * s))), max(12, tph // 4)), top_title, fill=(31, 35, 40), font=hfont)

    x0_cols = lcol + gx
    for jo, ori in enumerate(CLUSTER_ORIENT_ORDER):
        xh = x0_cols + jo * (otw + gx)
        dr.text((xh, tph + 4), ori, fill=(31, 35, 40), font=cfont)

    y = body_h_top
    for ri, d in enumerate(CLUSTER_DIRS_ORDER):
        dr.text((max(11, int(round(14 * s))), y + row_heights[ri] // 3), d, fill=(41, 45, 50), font=dfont)
        y_row_top = y
        for jo, ori in enumerate(CLUSTER_ORIENT_ORDER):
            cell = norm[(d, ori)]
            x = x0_cols + jo * (otw + gx)
            canvas.paste(cell, (x, y_row_top))
        y += row_heights[ri]

    return canvas


def render_arm6_dir_orient_clusters(env) -> tuple[list[str], str]:
    """Group 729 right-arm combos into fixed 12 bins (six-way dir × horizontal|vertical pad proj)."""

    cluster_root = os.path.join(OUT_DIR, "dir_orient_clusters")
    os.makedirs(cluster_root, exist_ok=True)

    deg_steps = list(_joint_angle_grid)
    n_tot = len(deg_steps) ** 6
    base = _default_qpos(env)
    base[QI_TORSO] = TORSO_PRESETS["high"]
    base[QI_HEAD_PAN] = 0.0
    base[QI_HEAD_TILT] = 0.0

    combos = itertools.product(deg_steps, repeat=6)
    buckets: dict[tuple[str, str], list[tuple[tuple[int, ...], np.ndarray]]] = defaultdict(list)
    skips = 0

    for tup in tqdm(combos, total=n_tot, desc="arm 3^6 cluster"):
        q = base.copy()
        rad = [math.radians(float(d)) for d in tup]
        q[6:12] = np.asarray(rad, dtype=float)
        _apply_and_capture_kinematic(env, q, camera=CAM)
        lbl_dir = _dir_six_way_from_anchor(env, q, cam_name=CAM)
        lbl_grip = _gripper_horizontal_vertical_by_reach_dir(env.sim, lbl_dir)
        if lbl_dir == "?" or lbl_grip == "?":
            skips += 1
            continue
        if lbl_dir not in CLUSTER_DIRS_ORDER or lbl_grip not in CLUSTER_ORIENT_ORDER:
            skips += 1
            continue
        mid_w = _right_fingertip_pad_mid_world_xyz(env.sim)
        if mid_w is None:
            skips += 1
            continue
        buckets[(lbl_dir, lbl_grip)].append((tup, mid_w))

    mids_stack = np.stack([m for lst in buckets.values() for (_, m) in lst], axis=0)
    lo_xyz, hi_xyz = _world_xyz_pct_bounds(mids_stack)

    cell_template = Image.new("RGB", (32, 32))
    _cell_thumb_with_caption(
        cell_template, (0,) * 6, "?", "?", ee_pad_mid_pct_xyz=(50.0, 50.0, 50.0)
    )

    thumb_by_key: dict[tuple[str, str], Image.Image] = {}
    saved_paths: list[str] = []
    counts: dict[tuple[str, str], int] = {}

    for direction in CLUSTER_DIRS_ORDER:
        for ori in CLUSTER_ORIENT_ORDER:
            key = (direction, ori)
            entries = sorted(buckets[key], key=lambda tm: tm[0])
            counts[key] = len(entries)
            cols = _cluster_cols_for_cell_count(len(entries))
            cells: list[Image.Image] = []
            for tup, mid_w in entries:
                q = base.copy()
                rad = [math.radians(float(d)) for d in tup]
                q[6:12] = np.asarray(rad, dtype=float)
                frame = _apply_and_capture_kinematic(env, q, camera=CAM)
                lbl_d = _dir_six_way_from_anchor(env, q, cam_name=CAM)
                lbl_g = _gripper_horizontal_vertical_by_reach_dir(env.sim, lbl_d)
                pct = _world_xyz_to_pct_0100(mid_w, lo_xyz, hi_xyz)
                cells.append(
                    _cell_thumb_with_caption(
                        frame, tup, lbl_d, lbl_g, ee_pad_mid_pct_xyz=pct
                    )
                )

            title = f"{direction}, {ori} ({len(entries)})"
            montage = _montage_caption_cells(cells, cols=cols, title=title)

            fname = f"cluster__{direction}__{ori}.png"
            path_png = os.path.join(cluster_root, fname)
            montage.save(path_png, optimize=True)
            saved_paths.append(path_png)

            w0, h0 = montage.size
            s_ui = _render_ui_scale()
            tw = max(620, min(1300, int(round(980 * min(s_ui, 2.2)))))
            scale = tw / float(max(1, w0))
            nh = max(int(round(220 * s_ui)), int(round(h0 * scale)))
            thumb_by_key[key] = montage.resize((tw, nh), Image.LANCZOS)

    overview = _clusters_6x2_overview(
        thumb_by_key,
        top_title="729 clusters · [joints] / dir, orient / (x,y,z)",
    )
    overview_path = os.path.join(cluster_root, "clusters_6x2_overview.png")
    overview.save(overview_path, optimize=True)

    print("\n=== dir × orientation cluster counts ===")
    for d in CLUSTER_DIRS_ORDER:
        n_h = counts.get((d, "horizontal"), 0)
        n_v = counts.get((d, "vertical"), 0)
        print(f"  {d:>5}:  horizontal={n_h:>3}  vertical={n_v:>3}  (sum={n_h + n_v:>3})")
    summed = sum(counts.values())
    print(f"  assigned={summed}/{n_tot}  skipped_unknown={skips}")
    print(
        "  pct basis pad-mid world m (729): "
        f"x[{lo_xyz[0]:.5f},{hi_xyz[0]:.5f}] "
        f"y[{lo_xyz[1]:.5f},{hi_xyz[1]:.5f}] "
        f"z[{lo_xyz[2]:.5f},{hi_xyz[2]:.5f}]"
    )

    print(f"\n  cluster PNGs saved under:\n    {cluster_root}/")
    print(f"  overview: {overview_path}")
    return saved_paths, overview_path


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


def _apply_and_capture_kinematic(env, qpos, camera=CAM):
    """Pure FK + render (no env.step). Needed for large brute grids: stepping can terminate the episode."""
    env.sim.data.qpos[:] = qpos
    env.sim.forward()
    frame = env.sim.render(camera_name=camera, width=W, height=H, depth=False)
    return Image.fromarray(np.flipud(frame))


def _label(img: Image.Image, text: str) -> Image.Image:
    s = _render_ui_scale()
    bar_h = max(34, int(round(36 * s)))
    fz = max(18, int(round(22 * s)))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", fz)
    except Exception:
        font = ImageFont.load_default()
    pad_b = max(8, int(round(10 * s)))
    draw.rectangle([(0, H - bar_h), (W, H)], fill=(0, 0, 0, 180))
    draw.text((pad_b, H - bar_h + max(4, int(round(4 * s)))), text, fill=(255, 255, 255), font=font)
    return img


def _make_tile(images: list[Image.Image], labels: list[str], cols: int, title: str = "") -> Image.Image:
    s = _render_ui_scale()
    rows = math.ceil(len(images) / cols)
    tile_w = W * cols
    title_h = (max(48, int(round(50 * s))) if title else 0)
    tile_h = H * rows + title_h
    canvas = Image.new("RGB", (tile_w, tile_h), (246, 248, 251))
    if title:
        draw = ImageDraw.Draw(canvas)
        tfz = max(22, int(round(32 * s)))
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", tfz)
        except Exception:
            font = ImageFont.load_default()
        draw.text((max(16, int(round(20 * s))), max(10, int(round(10 * s)))), title, fill=(31, 35, 40), font=font)
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
<p class="sub">TIAGo (Tiago proxy) &mdash; pose / movement / path dimensions</p>

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
    if "--arm6-bruteforce" in sys.argv:
        env = _make_env()
        try:
            render_arm6_bruteforce_grid(env)
        finally:
            env.close()
        print(os.path.join(OUT_DIR, "pose_arm_bruteforce_3pow6_deg.png"))
        if sys.platform == "darwin":
            import subprocess

            subprocess.Popen(
                ["open", os.path.join(OUT_DIR, "pose_arm_bruteforce_3pow6_deg.png")]
            )
    elif "--arm6-cluster-12" in sys.argv:
        env = _make_env()
        try:
            _, ov = render_arm6_dir_orient_clusters(env)
        finally:
            env.close()
        print(ov)
        if sys.platform == "darwin":
            import subprocess

            subprocess.Popen(["open", ov])
    else:
        main()
