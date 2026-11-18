#!/usr/bin/env python3
"""Table-1 attribute build-up grid: G1 beckon | gap | Tiago flying bird.

One row of 8 cells (4 + gap + 4):
  cue-name | Orientation | Location | Movement
Negative → orientation → location → movement (loc frame @ α=0.5 + traj).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("G1_POSE_XYZ_MODE", "translate")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path("/Users/sb/Downloads/workspace/dev_robosuite")
PAPER = REPO / "papers/icra2027_motion_abstraction"
MC = PAPER / "motion_code"
OUT_DIR = PAPER / "figures" / "_attr_buildup"
OUT_PNG = PAPER / "figures" / "figure_attr_buildup_beckon_bird.png"
OUT_PDF = PAPER / "figures" / "figure_attr_buildup_beckon_bird.pdf"

sys.path[:0] = [
    str(MC),
    str(PAPER),
    str(REPO),
    str(REPO / "adhoc/generation/robotarm"),
    str(REPO / "adhoc/generation/tiago"),
    str(REPO / "adhoc/vlm_test"),
]

from build_fig1_gaze_rub_grid import fill_cell, track_from_first, zoom_crop  # noqa: E402
from motion_code_to_config import convert_code  # noqa: E402

CELL = 260
HEADER_H = 38
GAP_W = 28
ROW_GAP = 0
CUE_BLUE = (31, 111, 235)
LABELS_ATTR = ("Orientation", "Location", "Movement")
MOV_ALPHA = 0.50
PROGRAM_BOX_BG = (255, 255, 255)
PROGRAM_BOX_FG = (20, 20, 20)

# Progressive stages: neg / ori / loc / ghost / mov.
# Location and Movement share the loc pose. Ghost is the other extreme
# (e.g. wings down) composited as a true α=0.5 robot overlay — background stays opaque.
G1_STAGES = [
    (
        "neg",
        "pose(dir=out, grip=vert|palmdown, x=36, y=48, z=36, left_arm=still)\n+ hold(t=1.0)",
    ),
    (
        "ori",
        "pose(dir=front, grip=vert|palmin, x=36, y=48, z=36, left_arm=still)\n+ hold(t=1.0)",
    ),
    (
        "loc",
        "pose(dir=front, grip=vert|palmin, x=60, y=50, z=80, left_arm=still)\n+ hold(t=1.0)",
    ),
    (
        "ghost",
        # Front camera: x is depth, so afterimage must move y/z.
        "pose(dir=front, grip=vert|palmin, x=60, y=50, z=42, left_arm=still)\n+ hold(t=1.0)",
    ),
    (
        "mov",
        "pose(dir=front, grip=vert|palmin, z=80, x=60)\n"
        "+ move(wrist, x, [-35,+35], rep=3, spd=3.5, hold=0.1)",
    ),
]

TIAGO_STAGES = [
    (
        "neg",
        "pose(left_arm=still, torso=mid)\n+ hold(t=1.0)",
    ),
    (
        "ori",
        "pose(dir=out, grip=vert, x=70, y=35, z=88, left_arm=mirror, torso=mid)\n+ hold(t=1.0)",
    ),
    (
        "loc",
        "pose(dir=out, grip=horiz, y=0, z=88, left_arm=mirror, torso=mid)\n+ hold(t=1.0)",
    ),
    (
        "ghost",
        "pose(dir=out, grip=horiz, y=0, z=18, left_arm=mirror, torso=mid)\n+ hold(t=1.0)",
    ),
    (
        "mov",
        "pose(dir=out, grip=horiz, y=0, left_arm=mirror)\n"
        "+ move(shoulder, z, [-25,25], rep=4, spd=2.5)",
    ),
]

IIWA_STAGES = [
    (
        "neg",
        "pose(dir=front, grip=horiz, z=40)\n+ hold(t=1.0)",
    ),
    (
        "ori",
        "pose(dir=up, grip=horiz, z=22, x=35)\n+ hold(t=1.0)",
    ),
    (
        "loc",
        "pose(dir=up, grip=horiz, z=90, x=78)\n+ hold(t=1.0)",
    ),
    (
        "ghost",
        "pose(dir=up, grip=horiz, z=32)\n+ hold(t=1.0)",
    ),
    (
        "mov",
        "pose(dir=up, grip=horiz, z=70) + move(shoulder, x, 10, rep=1, spd=2.0, hold=1.5)",
    ),
]

STRETCH_STAGES = [
    (
        "neg",
        "pose(dir=front, grip=horiz, torso=mid, z=50, x=25)\n+ hold(t=1.0)",
    ),
    (
        "ori",
        "pose(dir=up, grip=horiz, torso=mid, z=50, x=25)\n+ hold(t=1.0)",
    ),
    (
        "loc",
        "pose(dir=up, grip=horiz, torso=high, z=85)\n+ hold(t=1.0)",
    ),
    (
        "ghost",
        "pose(dir=up, grip=horiz, torso=high, z=40)\n+ hold(t=1.0)",
    ),
    (
        "mov",
        "pose(dir=up, grip=horiz, torso=high, z=85) & path(base, arc, r=0, deg=+90, spd=3.5)\n"
        "+ path(ee, line, axis=y, dist=-0.15, spd=1.8) + hold(t=1.2)",
    ),
]

# G1 front-camera framing (upper body, head + both arms in frame)
G1_CAM = dict(
    camera_lookat=(0.0, 0.0, 0.88),
    camera_distance=2.75,  # room for head + outstretched / hanging arms
    camera_azimuth=180.0,  # true front
    camera_elevation=0.0,  # eye-level front (less low-angle)
)


def _font(size: int = 18) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _load_frames(gif: Path, platform: str) -> list[np.ndarray]:
    from build_analysis_qual_grids import crop_progress, load_gif_frames

    raw = load_gif_frames(gif)
    if platform == "humanoid":
        return [fr[:-14] for fr in raw]
    if platform == "tiago":
        return [fr[28:] for fr in raw]
    if platform in ("single_arm", "stretch"):
        return crop_progress(raw)
    return raw


def _g1_cell(img: Image.Image) -> Image.Image:
    """Square cover crop without sky-padding (pad wasted space and clipped arms)."""
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    x0 = (w - side) // 2
    # slight upward bias so head isn't cropped when square-cutting a tall frame
    y0 = max(0, (h - side) // 2 - int(round(0.04 * side)))
    y0 = min(y0, h - side)
    return img.crop((x0, y0, x0 + side, y0 + side)).resize(
        (CELL, CELL), Image.Resampling.LANCZOS
    )


def _tiago_cell(img: Image.Image) -> Image.Image:
    """Fit by width so outstretched hands stay in frame (square center-crop clips them)."""
    img = img.convert("RGB")
    w, h = img.size
    scale = CELL / float(w)
    nh = max(1, int(round(h * scale)))
    resized = img.resize((CELL, nh), Image.Resampling.LANCZOS)
    if nh >= CELL:
        # slight downward bias: drop ceiling, keep arms/base
        slack = nh - CELL
        y0 = int(round(slack * 0.55))
        y0 = max(0, min(slack, y0))
        return resized.crop((0, y0, CELL, y0 + CELL))
    canvas = Image.new("RGB", (CELL, CELL), resized.getpixel((0, 0)))
    canvas.paste(resized, (0, (CELL - nh) // 2))
    return canvas


def _tiago_hand_tip(img: Image.Image) -> tuple[int, int]:
    """Outer gripper tip on the viewer's-left wing (right EE)."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    lum = arr.mean(axis=2)
    h, w = lum.shape
    dark = (lum < 95.0) & (lum > 18.0)
    dark[int(h * 0.78) :, :] = False  # ignore floor
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    band = dark & (xs < int(w * 0.42)) & (ys > int(h * 0.12)) & (ys < int(h * 0.62))
    yy, xx = np.where(band)
    if len(xx) < 8:
        return int(w * 0.12), int(h * 0.38)
    # leftmost cluster (avoid single edge-noise pixels via percentile)
    thr = float(np.percentile(xx, 18))
    keep = xx <= thr
    # sit on the gripper body (not past the outer silhouette edge)
    tx = int(np.percentile(xx[keep], 60))
    ty = int(np.percentile(yy[keep], 28))  # slightly toward the raised tip
    return tx, ty


def _wing_flap_traj(
    cx: int,
    cy: int,
    *,
    amp: int,
    h: int = CELL,
    strokes: int = 3,
    n_per: int = 11,
    bulge: float = 5.5,
) -> list[tuple[int, int]]:
    """Wing-flap orbit: mostly vertical, tiny overlapping arcs, never above the hand.

    Hand tip ``cy`` is the top of the stroke; path only goes downward and back up
    along a slightly offset arc so repeated flaps read as a soft bundle, not a ruler.
    """
    y_top = int(cy)
    y_bot = min(h - 8, y_top + max(24, int(amp)))
    pts: list[tuple[int, int]] = []
    for s in range(strokes):
        # alternate a hair left/right so strokes overlap without coinciding
        side = 1.0 if (s % 2 == 0) else -1.0
        x_base = cx + side * (1.2 + 0.7 * s)
        b = bulge * (1.0 - 0.12 * s)  # later flaps a bit flatter
        # downstroke
        for t in np.linspace(0.0, 1.0, n_per):
            y = y_top + t * (y_bot - y_top)
            # shallow parabola — peak bulge mid-way (very slight arc)
            x = x_base + side * b * (4.0 * t * (1.0 - t))
            pts.append((int(round(x)), int(round(y))))
        # upstroke on a slightly different arc (the "overlap")
        for t in np.linspace(0.0, 1.0, n_per):
            y = y_bot - t * (y_bot - y_top)
            x = x_base - side * (b * 0.65) * (4.0 * t * (1.0 - t))
            pts.append((int(round(x)), int(round(y))))
    # end mid-downstroke so the arrow reads as a downward flap
    for t in np.linspace(0.0, 0.85, max(6, n_per // 2)):
        y = y_top + t * (y_bot - y_top)
        x = cx + bulge * 0.4 * (4.0 * t * (1.0 - t))
        pts.append((int(round(x)), int(round(y))))
    return pts


def _pick_still_array(frames: list[np.ndarray], platform: str) -> np.ndarray:
    """Prefer a non-snow frame (Tiago EGL glitches); hold poses use an early frame."""
    from build_fig1_gaze_rub_grid import _frame_is_snow

    if not frames:
        raise ValueError("no frames")
    if platform != "tiago":
        return frames[min(3, len(frames) - 1)]
    clean = [fr for fr in frames if not _frame_is_snow(fr)]
    if not clean:
        return frames[0]
    return clean[0]


def _square_cover(img: Image.Image, *, y_bias: float = 0.0) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    x0 = (w - side) // 2
    y0 = max(0, min(h - side, (h - side) // 2 + int(round(y_bias * side))))
    return img.crop((x0, y0, x0 + side, y0 + side)).resize(
        (CELL, CELL), Image.Resampling.LANCZOS
    )


def _to_cell(img: Image.Image, platform: str) -> Image.Image:
    if platform == "humanoid":
        return _g1_cell(img)
    if platform == "tiago":
        return _tiago_cell(img)
    if platform == "stretch":
        return _square_cover(img, y_bias=0.06)
    return _square_cover(img, y_bias=0.04)


def _cell_still(gif: Path, platform: str) -> Image.Image:
    frames = _load_frames(gif, platform)
    mid = Image.fromarray(_pick_still_array(frames, platform)).convert("RGB")
    return _to_cell(mid, platform)


def _robot_mask(arr: np.ndarray, *, platform: str) -> np.ndarray:
    """Robot pixels only (True). Backdrop is never blended."""
    h, w, _ = arr.shape
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = arr.mean(axis=2)
    if platform == "tiago":
        band = np.concatenate(
            [arr[: max(2, h // 12), : w // 4], arr[: max(2, h // 12), -w // 4 :]],
            axis=1,
        )
        bg = band.reshape(-1, 3).mean(axis=0)
        dist = np.linalg.norm(arr - bg[None, None, :], axis=2)
        is_wood = (r > g + 4) & (r > b + 8) & (r > 70)
        ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        white_body = (dist > 18.0) & (lum > 115.0) & (~is_wood)
        dark_arm = (lum < 110.0) & (lum > 12.0) & (~is_wood) & (ys < 0.78)
        mask = white_body | dark_arm
    elif platform == "stretch":
        bg = arr[: max(2, h // 10)].reshape(-1, 3).mean(axis=0)
        dist = np.linalg.norm(arr - bg[None, None, :], axis=2)
        mask = (dist > 22.0) & (lum < 150.0)
    elif platform == "single_arm":
        bg = arr[: max(2, h // 12)].reshape(-1, 3).mean(axis=0)
        dist = np.linalg.norm(arr - bg[None, None, :], axis=2)
        is_wood = (r > g + 4) & (r > b + 8) & (r > 70)
        mask = (dist > 16.0) & (lum > 70.0) & (~is_wood)
    else:
        # White G1 vs pale sky: chroma, not distance-from-sky mean.
        # Floor is the dark lower band only — black head/joints are not floor.
        ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        is_sky = (b > r + 6) & (b > g - 2) & (lum > 135)
        is_floor = (lum < 72) & (ys > 0.58)
        mask = ~(is_sky | is_floor)
    from PIL import ImageFilter

    m_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").filter(
        ImageFilter.MaxFilter(3)
    )
    return np.asarray(m_img) > 127


def _overlay_afterimage(
    base: Image.Image,
    overlay: Image.Image,
    alpha: float,
    *,
    platform: str,
) -> Image.Image:
    """True α layer of overlay *robot pixels only*, composited on opaque base.

    Non-robot pixels in the overlay stay fully transparent, so the sky/floor
    are never washed out. Unmoved body over itself is a no-op.
    """
    if overlay.size != base.size:
        overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
    o = np.asarray(overlay.convert("RGB"))
    robot = _robot_mask(o, platform=platform)
    rgba = np.zeros((o.shape[0], o.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = o
    rgba[:, :, 3] = np.where(robot, int(round(float(alpha) * 255.0)), 0).astype(np.uint8)
    layer = Image.fromarray(rgba, mode="RGBA")
    return Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")


def _draw_traj_on(img: Image.Image, pts: list[tuple[int, int]]) -> Image.Image:
    from testset_utils import _draw_thick_arrow, _image_with_ee_path

    if len(pts) < 2:
        return img
    out = _image_with_ee_path(img.convert("RGB"), pts)
    draw = ImageDraw.Draw(out)
    tail = pts[-min(8, len(pts)) :]
    _draw_thick_arrow(draw, tail, color=(138, 80, 220), width=5, head_len=14, head_w=11)
    return out


def _map_pts(
    pts: list[tuple[int, int]],
    src_wh: tuple[int, int],
    dst_wh: tuple[int, int],
    *,
    platform: str,
) -> list[tuple[int, int]]:
    """Map trajectory points from raw frame coords into final CELL coords."""
    sw, sh = src_wh
    dw, dh = dst_wh
    if platform == "humanoid":
        # _g1_cell: square cover, slight upward bias
        side = min(sw, sh)
        x0 = (sw - side) // 2
        y0 = max(0, (sh - side) // 2 - int(round(0.04 * side)))
        y0 = min(y0, sh - side)
        out = []
        for x, y in pts:
            xx = (x - x0) * dw / max(1, side)
            yy = (y - y0) * dh / max(1, side)
            out.append((int(xx), int(yy)))
        return out
    # tiago: zoom_crop 0.82 y_floor=0.78 then fit_cell square
    keep = 0.82
    nw, nh = int(round(sw * keep)), int(round(sh * keep))
    x_slack, y_slack = sw - nw, sh - nh
    x0 = int(round(x_slack / 2))
    y0 = int(round(y_slack * 0.78))
    side = min(nw, nh)
    cx0 = (nw - side) // 2
    cy0 = (nh - side) // 2
    out = []
    for x, y in pts:
        xx, yy = x - x0 - cx0, y - y0 - cy0
        out.append((int(xx * dw / max(1, side)), int(yy * dh / max(1, side))))
    return out


def _movement_cell(
    loc_img: Image.Image,
    ghost_img: Image.Image,
    mov_gif: Path,
    platform: str,
    idx: int,
) -> Image.Image:
    """Opaque location pose + translucent other-extreme pose + trajectory."""
    stacked = _overlay_afterimage(loc_img, ghost_img, MOV_ALPHA, platform=platform)
    if platform == "tiago":
        hx, hy = _tiago_hand_tip(loc_img)
        gx, gy = _tiago_hand_tip(ghost_img)
        # One downstroke on the viewer's-left wing — do not paint a flap bundle
        # (that reads as a beige smear over the afterimage).
        n = 8
        pts2 = [
            (
                int(round(hx + (gx - hx) * t / (n - 1))),
                int(round(hy + (gy - hy) * t / (n - 1))),
            )
            for t in range(n)
        ]
        return _draw_traj_on(stacked, pts2)
    frames = _load_frames(mov_gif, platform)
    if not frames:
        return stacked
    cell_frames = [np.asarray(_to_cell(Image.fromarray(fr), platform)) for fr in frames]
    pts2 = track_from_first(cell_frames, platform, idx=idx)
    return _draw_traj_on(stacked, pts2)


def _entry(idx: int, cue: str, platform: str, source: str, tag: str) -> dict:
    frag = convert_code(source, platform=platform)
    row = {
        "idx": idx,
        "cue": f"{cue}__{tag}",
        "description": tag,
        "movements": frag.get("movements") or [],
        "state": "attr_buildup",
        "model": "attr_buildup",
        "experiment": "attr_buildup",
        "generation_valid": True,
        "platform": platform,
        "source_code": source,
        "parse_ok": True,
        "notes": [],
    }
    return {"idx": idx, "cue": row["cue"], "platforms": {platform: row}}


def _render_humanoid(stages: list[tuple[str, str]], cue: str) -> list[Image.Image]:
    from embodiment_renderers import render_unitree_g1

    out_dir = OUT_DIR / "render_g1" / "humanoid"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_idx = 9100
    rows: list[dict] = []
    gifs: list[Path] = []
    for i, (tag, source) in enumerate(stages):
        idx = base_idx + i
        entry = _entry(idx, cue, "humanoid", source, tag)
        row = entry["platforms"]["humanoid"]
        rows.append(row)
        # drop stale gifs (render_unitree_g1 names emb_g{idx:02d}_{cue}.gif)
        for p in out_dir.glob(f"emb_g{idx:02d}_*"):
            p.unlink(missing_ok=True)

    render_unitree_g1(rows, out_dir, force=True, **G1_CAM)
    for row in rows:
        gif = out_dir / f"emb_g{int(row['idx']):02d}_{row['cue']}.gif"
        gifs.append(gif)
        print(f"[g1] {row['description']} -> {gif.name}", flush=True)

    neg = _cell_still(gifs[0], "humanoid")
    ori = _cell_still(gifs[1], "humanoid")
    loc = _cell_still(gifs[2], "humanoid")
    ghost = _cell_still(gifs[3], "humanoid")
    mov = _movement_cell(loc, ghost, gifs[4], "humanoid", idx=11)
    return [neg, ori, loc, mov]


def _render_tiago(stages: list[tuple[str, str]], cue: str) -> list[Image.Image]:
    from correction_lib import _import_tiago_render

    _make_env, render_config, _ = _import_tiago_render()
    env = _make_env()
    gifs: list[Path] = []
    from build_fig1_gaze_rub_grid import _frame_is_snow

    try:
        for tag, source in stages:
            frag = convert_code(source, platform="tiago")
            frames = None
            for attempt in range(4):
                cand = render_config(
                    {"movements": frag.get("movements") or []},
                    env=env,
                    overlay_progress_bar=False,
                    ignore_xyz=False,
                )
                # At least one non-snow frame after banner crop
                cropped = [np.array(fr.convert("RGB"))[28:] for fr in cand]
                if any(not _frame_is_snow(fr) for fr in cropped):
                    frames = cand
                    break
                print(f"[tiago] {tag} snow retry {attempt + 1}", flush=True)
            if frames is None:
                frames = cand
            gif_path = OUT_DIR / f"tiago_{tag}.gif"
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=50,
                loop=0,
            )
            gifs.append(gif_path)
            print(f"[tiago] {tag} -> {gif_path.name}", flush=True)
    finally:
        env.close()

    neg = _cell_still(gifs[0], "tiago")
    ori = _cell_still(gifs[1], "tiago")
    loc = _cell_still(gifs[2], "tiago")
    ghost = _cell_still(gifs[3], "tiago")
    mov = _movement_cell(loc, ghost, gifs[4], "tiago", idx=30)
    return [neg, ori, loc, mov]


def _cells_from_gifs(gifs: list[Path], platform: str, idx: int) -> list[Image.Image]:
    neg = _cell_still(gifs[0], platform)
    ori = _cell_still(gifs[1], platform)
    loc = _cell_still(gifs[2], platform)
    ghost = _cell_still(gifs[3], platform)
    mov = _movement_cell(loc, ghost, gifs[4], platform, idx=idx)
    return [neg, ori, loc, mov]


def _render_iiwa(stages: list[tuple[str, str]], cue: str) -> list[Image.Image]:
    from motion_code_to_config import render_manipulator

    out_root = OUT_DIR / "render_iiwa"
    rows: list[dict] = []
    base_idx = 9200
    for i, (tag, source) in enumerate(stages):
        idx = base_idx + i
        entry = _entry(idx, cue, "single_arm", source, tag)
        rows.append(entry["platforms"]["single_arm"])
    render_manipulator(rows, out_root=out_root, force=True)
    gif_dir = out_root / "manipulator" / "gif" / "IIWA"
    gifs = [gif_dir / f"emb_g{base_idx + i:02d}_{cue}__{tag}.gif" for i, (tag, _) in enumerate(stages)]
    for g in gifs:
        print(f"[iiwa] {g.name} exists={g.is_file()}", flush=True)
    return _cells_from_gifs(gifs, "single_arm", idx=15)


def _render_stretch(stages: list[tuple[str, str]], cue: str) -> list[Image.Image]:
    from embodiment_renderers import render_stretch

    out_dir = OUT_DIR / "render_stretch"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    base_idx = 9300
    for i, (tag, source) in enumerate(stages):
        idx = base_idx + i
        entry = _entry(idx, cue, "stretch", source, tag)
        rows.append(entry["platforms"]["stretch"])
        for p in out_dir.glob(f"emb_g{idx:02d}_*"):
            p.unlink(missing_ok=True)
    render_stretch(rows, out_dir, force=True, fps=8)
    gifs = [
        out_dir / f"emb_g{base_idx + i:02d}_{cue}__{tag}.gif"
        for i, (tag, _) in enumerate(stages)
    ]
    for g in gifs:
        print(f"[stretch] {g.name} exists={g.is_file()}", flush=True)
    return _cells_from_gifs(gifs, "stretch", idx=19)


def _mono(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = []
    if bold:
        names += [
            "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
            "/Library/Fonts/Courier New Bold.ttf",
            "Courier New Bold.ttf",
        ]
    names += [
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/Library/Fonts/Courier New.ttf",
        "Courier New.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# Final SMP shown as a short banner spanning each 4-cell group.
# (pose, movement_or_path) — keep to two short lines so the robot stays visible.
PROGRAMS = {
    "beckon": (
        "POSE",
        "gripper(front, palmin, xyz=(60,50,80))",
        "MOVEMENT",
        "move(wrist, x, +/-35, rep=3)",
    ),
    "flying bird": (
        "POSE",
        "gripper(out, horiz, y=0, z=88, mirror)",
        "MOVEMENT",
        "move(shoulder, z, +/-25, rep=4)",
    ),
    "highfive": (
        "POSE",
        "gripper(up, horiz, xyz=(78,-,90))",
        "MOVEMENT",
        "move(shoulder, x, +10, rep=1)",
    ),
    "open welcome": (
        "POSE",
        "gripper(up, horiz, torso=high, z=85)",
        "PATH",
        "base arc(+90°)  +  ee line(y)",
    ),
}


PROG_TITLE = 18
PROG_BODY = 17


def _box_size(draw: ImageDraw.ImageDraw, title: str, body: str, max_w: int) -> tuple[int, int]:
    title_f = _mono(PROG_TITLE, bold=True)
    body_f = _mono(PROG_BODY)
    pad_x, pad_y = 10, 4
    tw = max(draw.textlength(title, font=title_f), draw.textlength(body, font=body_f))
    box_w = int(min(max_w, tw + 2 * pad_x + 2))
    box_h = pad_y + PROG_TITLE + 3 + PROG_BODY + pad_y
    return box_w, box_h


def _draw_code_box(
    draw: ImageDraw.ImageDraw,
    *,
    xy: tuple[int, int],
    title: str,
    body: str,
    box_w: int,
    box_h: int,
) -> None:
    title_f = _mono(PROG_TITLE, bold=True)
    body_f = _mono(PROG_BODY)
    pad_x, pad_y = 10, 4
    x0, y0 = xy
    draw.rounded_rectangle(
        [x0, y0, x0 + box_w, y0 + box_h],
        radius=4,
        fill=PROGRAM_BOX_BG,
        outline=PROGRAM_BOX_FG,
        width=1,
    )
    draw.text((x0 + pad_x, y0 + pad_y - 1), title, fill=PROGRAM_BOX_FG, font=title_f)
    draw.text((x0 + pad_x, y0 + pad_y + PROG_TITLE + 1), body, fill=PROGRAM_BOX_FG, font=body_f)


def _draw_program_banner(
    canvas: Image.Image,
    *,
    x0: int,
    y_img: int,
    block_w: int,
    cell_h: int,
    cue: str,
    at_bottom: bool,
) -> None:
    title_l, body_l, title_r, body_r = PROGRAMS[cue]
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    max_each = int(block_w * 0.55)
    wl, hl = _box_size(draw, title_l, body_l, max_each)
    wr, hr = _box_size(draw, title_r, body_r, max_each)
    box_h = max(hl, hr)
    gap = 10
    pair_w = wl + gap + wr
    x_left = x0 + (block_w - pair_w) // 2
    x_right = x_left + wl + gap
    y = y_img + cell_h - box_h - 5 if at_bottom else y_img + 5
    _draw_code_box(draw, xy=(x_left, y), title=title_l, body=body_l, box_w=wl, box_h=box_h)
    _draw_code_box(draw, xy=(x_right, y), title=title_r, body=body_r, box_w=wr, box_h=box_h)
    y_line = y + box_h // 2
    draw.line([(x_left + wl, y_line), (x_right, y_line)], fill=(20, 20, 20), width=1)
    canvas.paste(Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB"))


def _compose_panel(
    canvas: Image.Image,
    draw,
    cells: list[Image.Image],
    *,
    y0: int,
    cue: str,
    at_bottom: bool,
) -> None:
    n = 4
    font = _font(21)
    font_cue = _font(20)

    def _center_text(text: str, x0: int, block_w: int, *, fill, font_) -> None:
        bbox = draw.textbbox((0, 0), text, font=font_)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = x0 + (block_w - tw) // 2
        y = y0 + max(4, (HEADER_H - th) // 2 - 1)
        draw.text((x, y), text, fill=fill, font=font_)

    _center_text(cue, 0, CELL, fill=CUE_BLUE, font_=font_cue)
    for i, lab in enumerate(LABELS_ATTR):
        _center_text(lab, (i + 1) * CELL, CELL, fill=(20, 20, 20), font_=font)
    for i in range(n):
        canvas.paste(cells[i], (i * CELL, y0 + HEADER_H))

    block_w = n * CELL
    _draw_program_banner(
        canvas,
        x0=0,
        y_img=y0 + HEADER_H,
        block_w=block_w,
        cell_h=CELL,
        cue=cue,
        at_bottom=at_bottom,
    )


def _compose(
    rows: tuple[
        tuple[list[Image.Image], str, bool],
        tuple[list[Image.Image], str, bool],
        tuple[list[Image.Image], str, bool],
    ],
) -> Image.Image:
    n = 4
    width = n * CELL
    row_h = HEADER_H + CELL
    height = len(rows) * row_h + (len(rows) - 1) * ROW_GAP
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for row_idx, (cells, cue, at_bottom) in enumerate(rows):
        _compose_panel(
            canvas,
            draw,
            cells,
            y0=row_idx * (row_h + ROW_GAP),
            cue=cue,
            at_bottom=at_bottom,
        )
    return canvas


def recompose_from_gifs() -> None:
    """Rebuild Fig.~2 from cached stage GIFs (no MuJoCo)."""
    g1_dir = OUT_DIR / "render_g1" / "humanoid"
    g1_gifs = [
        g1_dir / f"emb_g{9100 + i:02d}_beckon_come_here__{tag}.gif"
        for i, (tag, _) in enumerate(G1_STAGES)
    ]
    tiago_gifs = [OUT_DIR / f"tiago_{tag}.gif" for tag, _ in TIAGO_STAGES]
    iiwa_dir = OUT_DIR / "render_iiwa" / "manipulator" / "gif" / "IIWA"
    iiwa_gifs = [
        iiwa_dir / f"emb_g{9200 + i:02d}_highfive_invite__{tag}.gif"
        for i, (tag, _) in enumerate(IIWA_STAGES)
    ]
    missing = [p for p in g1_gifs + tiago_gifs + iiwa_gifs if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing stage gifs: " + ", ".join(p.name for p in missing))
    grid = _compose((
        (_cells_from_gifs(g1_gifs, "humanoid", idx=11), "beckon", False),
        (_cells_from_gifs(tiago_gifs, "tiago", idx=30), "flying bird", False),
        (_cells_from_gifs(iiwa_gifs, "single_arm", idx=15), "highfive", True),
    ))
    grid.save(OUT_PNG)
    grid.convert("RGB").save(OUT_PDF, "PDF", resolution=220.0)
    print("recomposed", OUT_PNG, grid.size)


def overlay_only() -> None:
    recompose_from_gifs()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Rendering G1 beckon stages…", flush=True)
    g1 = _render_humanoid(G1_STAGES, "beckon_come_here")
    print("Rendering Tiago flying-bird stages…", flush=True)
    tiago = _render_tiago(TIAGO_STAGES, "depict_flying_bird")
    print("Rendering IIWA highfive stages…", flush=True)
    iiwa = _render_iiwa(IIWA_STAGES, "highfive_invite")
    grid = _compose((
        (g1, "beckon", False),
        (tiago, "flying bird", False),
        (iiwa, "highfive", True),
    ))
    grid.save(OUT_PNG)
    grid.convert("RGB").save(OUT_PDF, "PDF", resolution=220.0)
    meta = {
        "g1": [{"tag": t, "code": c} for t, c in G1_STAGES],
        "tiago": [{"tag": t, "code": c} for t, c in TIAGO_STAGES],
        "iiwa": [{"tag": t, "code": c} for t, c in IIWA_STAGES],
        "mov_alpha": MOV_ALPHA,
        "out": str(OUT_PNG),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", OUT_PNG)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    if "--overlay-only" in sys.argv:
        overlay_only()
    else:
        main()
