#!/usr/bin/env python3
"""Figure 1: two cues × six morphologies, peak-pose frame with trajectory overlay.

Columns: Humanoid, Tiago, Single Arm, Stretch, Roomba, Synthetic.
Synthetic uses Paddleclaw (α=1) for gaze_shift_avoid and Spine Crab (α=0.5) for rub_eye_tired.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "adhoc/generation/robotarm"))
sys.path.insert(0, str(REPO / "adhoc/vlm_test"))

from build_analysis_qual_grids import (  # noqa: E402
    CELL,
    HEADER_H,
    _smooth_path,
    _subsample_path,
    crop_progress,
    edge_trim,
    fit_cell,
    font_pair,
    gif_path,
    load_gif_frames,
    pick_pose_frame_index,
)
from testset_utils import _draw_thick_arrow, _image_with_ee_path  # noqa: E402

PAPER = REPO / "papers/icra2027_motion_abstraction/figures"
SRC_GIF = PAPER / "_src_gifs"
ODDKIN = REPO / "papers/icra2027_motion_abstraction/synthetic_morphology/renders/cues30"
PLATFORMS = ["humanoid", "tiago", "single_arm", "stretch", "roomba", "synthetic"]
TRAJ_PLATFORMS = {"single_arm", "stretch", "roomba"}
CUES = {4: "gaze_shift_avoid", 8: "rub_eye_tired"}
ROWS = [4, 8]
# Synthetic column: Paddleclaw (α=1) on gaze row; Spine Crab (α=0.5) on rub row.
SYNTHETIC_GIF = {
    4: ODDKIN / "gcr_alpha1" / "gaze_shift_avoid.gif",
    8: ODDKIN / "gcr_alpha05" / "rub_eye_tired.gif",
}
COL_LABEL = {
    "humanoid": "Humanoid",
    "tiago": "Tiago",
    "single_arm": "Single Arm",
    "stretch": "Stretch",
    "roomba": "Roomba",
    "synthetic": "Synthetic",
}
CUE_LABEL_H = 32
ROW_PAD = 6


def _roomba_stick(arr: np.ndarray) -> tuple[int, int] | None:
    lum = arr.mean(axis=2)
    dark = lum < 52
    ys_d, xs_d = np.where(dark)
    if len(xs_d) < 80:
        return None
    bx, by = int(np.median(xs_d)), int(np.median(ys_d))
    blue = (
        (arr[:, :, 2] > 150)
        & (arr[:, :, 2] > arr[:, :, 0] + 45)
        & (arr[:, :, 2] > arr[:, :, 1] + 35)
        & (arr[:, :, 0] < 120)
    )
    ys_b, xs_b = np.where(blue)
    if len(xs_b) < 4:
        return bx, by
    d2 = (xs_b.astype(float) - bx) ** 2 + (ys_b.astype(float) - by) ** 2
    outer = d2 >= max(36.0, float(np.percentile(d2, 60)))
    if outer.any():
        xs_o, ys_o, d_o = xs_b[outer], ys_b[outer], d2[outer]
        k = int(np.argmax(d_o))
        return int(xs_o[k]), int(ys_o[k])
    k = int(np.argmax(d2))
    return int(xs_b[k]), int(ys_b[k])


def _accum_motion(frames: list[np.ndarray]) -> np.ndarray:
    acc = np.zeros(frames[0].shape[:2], dtype=np.float32)
    prev = frames[0].astype(np.float32)
    for fr in frames[1:]:
        arr = fr.astype(np.float32)
        acc += np.abs(arr - prev).sum(axis=2)
        prev = arr
    H = acc.shape[0]
    acc[: max(2, H // 50), :] = 0
    acc[-max(8, H // 12) :, :] = 0
    return acc


def _seed_from_motion(
    frames: list[np.ndarray], platform: str, *, head_only: bool = False
) -> tuple[int, int]:
    """Frame-0 actor: head/EE of the region that actually moves, not sky/floor."""
    H, W = frames[0].shape[:2]
    acc = _accum_motion(frames)
    if float(acc.max()) < 1e-3:
        return W // 2, H // 3
    thr = float(np.percentile(acc[acc > 0], 88))
    mask = acc >= max(thr, float(acc.max()) * 0.18)
    ys, xs = np.where(mask)
    if len(xs) < 8:
        yx = np.unravel_index(int(np.argmax(acc)), acc.shape)
        return int(yx[1]), int(yx[0])
    if platform in ("humanoid", "tiago", "stretch"):
        pct = 8 if head_only else 22
        keep = ys <= np.percentile(ys, pct)
        return int(np.median(xs[keep])), int(np.median(ys[keep]))
    bx, by = W * 0.5, H * 0.92
    d2 = (xs.astype(float) - bx) ** 2 + (ys.astype(float) - by) ** 2
    k = int(np.argmax(d2))
    return int(xs[k]), int(ys[k])


def track_from_first(
    frames: list[np.ndarray], platform: str, *, idx: int | None = None
) -> list[tuple[int, int]]:
    """One point per frame, starting at frame 0's actor (no skipped lead-in)."""
    if not frames:
        return []
    H, W = frames[0].shape[:2]
    head_only = idx == 4
    pts: list[tuple[int, int]] = []
    if platform == "roomba":
        prev: tuple[int, int] | None = None
        for fr in frames:
            hit = _roomba_stick(fr.astype(np.float32))
            prev = hit if hit is not None else (prev or (W // 2, H // 2))
            pts.append(prev)
        pts = _smooth_path(pts, k=5)
        return _subsample_path(pts, min_step=1.2)

    prev = _seed_from_motion(frames, platform, head_only=head_only)
    pts.append(prev)
    prev_fr = frames[0].astype(np.float32)
    rad = (0.07 if head_only else 0.16) * max(H, W)
    yy, xx = np.ogrid[:H, :W]
    for fr in frames[1:]:
        arr = fr.astype(np.float32)
        mag = np.abs(arr - prev_fr).sum(axis=2)
        mag[-max(8, H // 12) :, :] = 0
        mag[: max(2, H // 50), :] = 0
        local = mag * (((xx - prev[0]) ** 2 + (yy - prev[1]) ** 2) < rad**2)
        if float(local.max()) >= 6.0:
            thr = float(np.percentile(local, 90))
            mask = local >= max(thr, 6.0)
            ys, xs = np.where(mask)
            if len(xs) >= 4:
                cx, cy = int(np.median(xs)), int(np.median(ys))
            else:
                cx, cy = prev
        else:
            cx, cy = prev
        jump = ((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2) ** 0.5
        if jump > 0.18 * max(H, W):
            cx = int(0.85 * prev[0] + 0.15 * cx)
            cy = int(0.85 * prev[1] + 0.15 * cy)
        prev = (int(cx), int(cy))
        pts.append(prev)
        prev_fr = arr
    pts = _smooth_path(pts, k=5)
    return _subsample_path(pts, min_step=1.2)


def zoom_crop(
    img: Image.Image, keep: float, *, x_shift: float = 0.0, y_floor: float = 0.22
) -> Image.Image:
    w, h = img.size
    nw, nh = max(1, int(round(w * keep))), max(1, int(round(h * keep)))
    x_slack, y_slack = w - nw, h - nh
    x0 = int(round((x_slack / 2) + x_shift * x_slack))
    x0 = max(0, min(x_slack, x0))
    y0 = int(round(y_slack * y_floor))
    y0 = max(0, min(y_slack, y0))
    return img.crop((x0, y0, x0 + nw, y0 + nh))


def fill_cell(img: Image.Image, size: int, *, platform: str = "") -> Image.Image:
    """Center-crop to a square, then scale to CELL×CELL (cover, no letterbox)."""
    img = img.convert("RGB")
    if platform == "humanoid":
        w, h = img.size
        pad = int(round(h * 0.20))
        sky = img.crop((0, 0, w, min(8, h))).resize((1, 1)).getpixel((0, 0))
        canvas = Image.new("RGB", (w, h + pad), sky)
        canvas.paste(img, (0, pad))
        cropped = canvas.crop((0, 0, w, w))
        return cropped.resize((size, size), Image.Resampling.LANCZOS)
    return fit_cell(img, size, size)


def _frame_is_snow(arr: np.ndarray) -> bool:
    """EGL/offscreen snow: high pixel std or collapsed luminance."""
    a = arr.astype(np.float32)
    return float(a.std()) > 55.0 or float(a.mean()) < 100.0


def _gif_snow_frac(gif: Path) -> float:
    raw = load_gif_frames(gif)
    if not raw:
        return 1.0
    return float(sum(_frame_is_snow(fr) for fr in raw)) / len(raw)


def peak_pose_with_traj(gif: Path, platform: str, idx: int) -> Image.Image:
    raw = load_gif_frames(gif)
    if platform == "humanoid":
        # Bottom scrubber only; crop_progress over-crops and clips the feet.
        frames = [fr[:-14] for fr in raw]
    elif platform == "tiago":
        # Top typed bar only; keep the full mobile base. Drop intermittent snow.
        frames = [fr[28:] for fr in raw if not _frame_is_snow(fr)]
        if not frames:
            frames = [fr[28:] for fr in raw]
    else:
        frames = crop_progress(raw)
    pi = pick_pose_frame_index(frames, platform=platform, method="gt", idx=idx)
    base = Image.fromarray(frames[pi]).convert("RGB")
    if platform in TRAJ_PLATFORMS:
        pts = track_from_first(frames, platform, idx=idx)
        if len(pts) >= 2:
            img = _image_with_ee_path(base, pts)
            draw = ImageDraw.Draw(img)
            tail = pts[-min(8, len(pts)) :]
            _draw_thick_arrow(
                draw, tail, color=(138, 80, 220), width=5, head_len=14, head_w=11
            )
        else:
            img = base
    else:
        img = base
    if platform == "tiago":
        # Original default frontview: drop ceiling so the body fills the cell.
        img = zoom_crop(img, 0.82, y_floor=0.78)
    if platform not in ("humanoid", "tiago"):
        img = edge_trim(img)
    return img


def _cue_title(cue: str) -> str:
    return cue.replace("_", " ").title()


def _gt_rows() -> dict[str, list[dict]]:
    import json

    path = REPO / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
    entries = json.loads(path.read_text(encoding="utf-8"))
    want = set(CUES.values())
    by_plat: dict[str, list[dict]] = {"humanoid": [], "tiago": []}
    for e in entries:
        if e.get("cue") not in want:
            continue
        for plat in by_plat:
            by_plat[plat].append(e["platforms"][plat])
    return by_plat


def ensure_front_gifs(*, force: bool = False, force_humanoid: bool = False, force_tiago: bool = False) -> None:
    """Re-render Humanoid/Tiago from a true front camera, full body in frame."""
    rows = _gt_rows()
    hum_dir = SRC_GIF / "humanoid"
    have_h = all(
        (hum_dir / f"emb_g{int(r['idx']):02d}_{r['cue']}.gif").is_file() for r in rows["humanoid"]
    )
    have_t = all(
        (SRC_GIF / "tiago" / "gif" / f"emb_g{int(r['idx']):02d}_{r['cue']}.gif").is_file()
        for r in rows["tiago"]
    )
    if force or force_humanoid or not have_h:
        from embodiment_renderers import render_unitree_g1

        # azimuth 180 = on-axis front. Distance ~1/2 of 3.35 so the body fills the cell.
        # Lookat slightly below mid-torso so closer FOV still includes feet.
        render_unitree_g1(
            rows["humanoid"],
            hum_dir,
            force=True,
            camera_lookat=(0.0, 0.0, 0.64),
            camera_distance=1.68,
            camera_azimuth=180,
            camera_elevation=-8,
        )
    if force or force_tiago or not have_t:
        from motion_code_to_config import render_tiago

        SRC_GIF.mkdir(parents=True, exist_ok=True)
        # Higher overhead-front: z≈1.75, still on +x, look down at mid-body so the
        # base stays in frame (default XML quat is only ~15° down).
        cam_pos = (2.00, 0.0, 1.72)
        cam_quat = (0.6166, 0.34614, 0.34614, 0.6166)
        for attempt in range(1, 7):
            render_tiago(
                rows["tiago"],
                out_root=SRC_GIF,
                force=True,
                cam_pos=cam_pos,
                cam_quat=cam_quat,
            )
            snow = {
                int(r["idx"]): _gif_snow_frac(
                    SRC_GIF / "tiago" / "gif" / f"emb_g{int(r['idx']):02d}_{r['cue']}.gif"
                )
                for r in rows["tiago"]
            }
            print(f"[fig1] tiago snow frac attempt {attempt}: {snow}", flush=True)
            if all(v <= 0.08 for v in snow.values()):
                break


def cell_gif(plat: str, idx: int) -> Path | None:
    if plat == "synthetic":
        path = SYNTHETIC_GIF.get(idx)
        return path if path is not None and path.is_file() else None
    if plat == "tiago":
        return gif_path("gt", plat, idx, CUES, {})
    name = f"emb_g{idx:02d}_{CUES[idx]}.gif"
    for cand in (SRC_GIF / plat / name, SRC_GIF / plat / "gif" / name):
        if cand.is_file():
            return cand
    return gif_path("gt", plat, idx, CUES, {})


def main() -> None:
    ensure_front_gifs()
    font_cue, font_s = font_pair(20, 18)
    n_c, n_r = len(PLATFORMS), len(ROWS)
    row_stride = CELL + ROW_PAD + CUE_LABEL_H
    W = CELL * n_c
    H = HEADER_H + n_r * row_stride
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    for j, plat in enumerate(PLATFORMS):
        lab = COL_LABEL[plat]
        bbox = draw.textbbox((0, 0), lab, font=font_s)
        tw = bbox[2] - bbox[0]
        x = j * CELL
        draw.text((x + (CELL - tw) / 2, 10), lab, fill="#111111", font=font_s)

    for i, idx in enumerate(ROWS):
        y0 = HEADER_H + i * row_stride
        for j, plat in enumerate(PLATFORMS):
            x0 = j * CELL
            g = cell_gif(plat, idx)
            print(f"{plat} c{idx:02d} -> {g}")
            if g is None or not g.is_file():
                fitted = Image.new("RGB", (CELL, CELL), "#f2f2f2")
                ImageDraw.Draw(fitted).text(
                    (CELL // 2 - 28, CELL // 2 - 8), "missing", fill="#888888", font=font_s
                )
            else:
                # Oddkin cells: no traj overlay; mild edge crop like non-humanoid frames.
                if plat == "synthetic":
                    raw = load_gif_frames(g)
                    frames = crop_progress(raw)
                    pi = pick_pose_frame_index(frames, platform="stretch", method="gt", idx=idx)
                    img = edge_trim(Image.fromarray(frames[pi]).convert("RGB"))
                    fitted = fill_cell(img, CELL, platform="")
                else:
                    img = peak_pose_with_traj(g, plat, idx)
                    fitted = fill_cell(img, CELL, platform=plat)
            canvas.paste(fitted, (x0, y0))
        label = _cue_title(CUES[idx])
        bbox = draw.textbbox((0, 0), label, font=font_cue)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        lx = (W - tw) / 2
        ly = y0 + CELL + max(0, (CUE_LABEL_H - th) // 2)
        draw.text((lx, ly), label, fill="#111111", font=font_cue)

    PAPER.mkdir(parents=True, exist_ok=True)
    out = PAPER / "figure1_gaze_rub_grid.png"
    canvas.save(out, optimize=True)
    print("wrote", out, canvas.size)


if __name__ == "__main__":
    main()
