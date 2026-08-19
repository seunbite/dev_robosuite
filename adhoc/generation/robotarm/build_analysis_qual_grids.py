#!/usr/bin/env python3
"""Rebuild paper Analysis qualitative grids (Humanoid / Single Arm / Roomba).

Design goals:
  - Thin green/red YES/NO outlines (no text badges, no white gutter)
  - Crop progress-bar overlays; optional outer trim
  - Same-cue method cells abut; small gap between cues
  - Row labels = cue name only (no cNN)
  - Roomba path tracks blue stick outer tip
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from PIL import Image, ImageDraw, ImageFont, ImageSequence

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "adhoc/vlm_test"))
from testset_utils import _draw_thick_arrow, _image_with_ee_path  # noqa: E402

OUT = REPO / "data/results/evaluation/method_comparison/analysis_fail"
CACHE = OUT / "path_overlay_cache"

METHODS = ["gt", "joint", "xyzrpy", "motion_program", "trajectory"]
MLAB = {
    "gt": "GT",
    "joint": "Joint",
    "xyzrpy": "Cartesian",
    "motion_program": "Motion Program",
    "trajectory": "Trajectory",
}

# Fig3: three humanoid communicative cues + one Tiago where MP+Traj succeed.
HU_ROWS: list[tuple[str, int]] = [
    ("humanoid", 7),
    ("humanoid", 18),
    ("humanoid", 16),
    ("tiago", 10),  # point_self — MP/Traj YES
]
# Fig7: salute replaced by point_self.
SA_ROWS: list[tuple[str, int]] = [
    ("single_arm", 5),
    ("single_arm", 9),
    ("single_arm", 14),
    ("single_arm", 10),
]
RB_ROWS: list[tuple[str, int]] = [
    ("roomba", 28),
    ("roomba", 18),
    ("roomba", 8),
    ("roomba", 19),
]
# Main-paper cross-embodiment grid: one diagnostic row per embodiment.
MAIN_ROWS: list[tuple[str, int]] = [
    ("humanoid", 18),  # air_quotes
    ("tiago", 10),     # point_self
    ("single_arm", 14), # raise_index_one_moment
    ("stretch", 21),    # stretch_arms_up
    ("roomba", 28),     # depict_lion
]

PLATFORM_LABEL = {
    "humanoid": "Humanoid",
    "tiago": "Tiago",
    "single_arm": "IIWA",
    "stretch": "Stretch",
    "roomba": "Roomba",
}

# Three representative target/foil requests for the compact main-paper panel.
CONTRASTIVE_ROWS: list[tuple[int, int]] = [
    (15, 20),  # highfive_invite != hands_up_surrender
    (20, 15),  # hands_up_surrender != highfive_invite
    (22, 5),   # raise_fists_victory != fist_pump
]

CELL = 240
LABEL_W = 200
HEADER_H = 40
TITLE_H = 40
CUE_GAP = 4
OUTLINE = 2
EDGE_TRIM = 10  # px trimmed from each side of source before fit
YES_COLOR = (32, 168, 78)
NO_COLOR = (210, 48, 48)
YES_BG = (21, 61, 37)
NO_BG = (72, 31, 39)
YES_FG = (140, 233, 154)  # gallery .yes
NO_FG = (255, 180, 180)   # gallery .no


def cues_map() -> dict[int, str]:
    rows = json.loads(
        (REPO / "data/seed/agent_fill/embodiment_short/cues.json").read_text(encoding="utf-8")
    )
    return {int(r["idx"]): str(r["cue"]) for r in rows}


def load_labels() -> dict[tuple[str, int, str], bool]:
    labels: dict[tuple[str, int, str], bool] = {}
    xlsx = REPO / "data/results/evaluation/embodiment_cue_id/all_methods_human_yesno_c1_30.xlsx"
    wb = load_workbook(xlsx, data_only=True)
    sh = wb.active
    for row in sh.iter_rows(min_row=2, values_only=True):
        cue = str(row[0])
        idx = int(re.match(r"c(\d+)", cue.lower()).group(1))
        plat = str(row[1]).split("·")[-1].strip()
        for m, i in [
            ("gt", 2),
            ("joint", 3),
            ("xyzrpy", 4),
            ("motion_program", 5),
            ("trajectory", 6),
        ]:
            labels[(plat, idx, m)] = str(row[i] or "").upper() == "YES"
    return labels


def traj_renders() -> dict[tuple[str, int], Path]:
    out: dict[tuple[str, int], Path] = {}
    for src in [
        REPO / "data/results/drag_draw/gemini25pro_c1_25/predictions_rendered.json",
        REPO / "data/results/drag_draw/gemini25pro_c1_5_c26_30/predictions_rendered.json",
        REPO / "data/results/drag_draw/roomba_sophisticated_full30/predictions_rendered.json",
    ]:
        if not src.is_file():
            continue
        for row in json.loads(src.read_text(encoding="utf-8")):
            if row.get("parse_ok") and row.get("render_gif"):
                out[(row["platform"], int(row["idx"]))] = REPO / row["render_gif"]
    return out


def gif_path(
    method: str,
    platform: str,
    idx: int,
    cues: dict[int, str],
    traj: dict[tuple[str, int], Path],
) -> Path | None:
    cue = cues[idx]
    if method == "trajectory":
        return traj.get((platform, idx))
    stem = f"emb_g{idx:02d}_{cue}"
    if method in ("joint", "xyzrpy"):
        if platform == "roomba":
            base = (
                REPO
                / f"data/results/baseline_traj/roomba_sophisticated_full30/render_{method}/roomba"
            )
            for name in (f"{stem}_gemini-2.5-pro.gif", f"{stem}.gif"):
                p = base / name
                if p.is_file():
                    return p
        base = REPO / f"data/results/baseline_traj/ours5/render_{method}/{platform}"
        for name in (f"{stem}_gemini-2.5-pro.gif", f"{stem}.gif"):
            p = base / name
            if p.is_file():
                return p
        return None
    if method == "motion_program" and platform == "roomba":
        p = REPO / f"data/results/roomba_sophisticated_full30_motion_code/render/{stem}.gif"
        return p if p.is_file() else None
    root = (
        "embodiment_motion_code_gt"
        if method == "gt"
        else "embodiment_motion_code_gemini_gemini-2.5-pro"
    )
    rplat = "manipulator" if platform == "single_arm" else platform
    if method == "motion_program" and platform in ("humanoid", "tiago"):
        ee = REPO / f"data/results/render/{root}_ee_translate/{rplat}/gif" / f"{stem}.gif"
        if ee.is_file():
            return ee
    base = REPO / f"data/results/render/{root}/{rplat}/gif"
    if platform == "single_arm":
        base = base / "IIWA"
    p = base / f"{stem}.gif"
    return p if p.is_file() else None


def load_gif_frames(path: Path) -> list[np.ndarray]:
    im = Image.open(path)
    return [np.asarray(fr.convert("RGB")) for fr in ImageSequence.Iterator(im)]


def detect_progress_bar_height(frames: list[np.ndarray]) -> tuple[int, int]:
    """Return (top_crop, bottom_crop) for typed/scrubber progress overlays."""
    if not frames:
        return 0, 0
    fr = frames[min(len(frames) - 1, max(1, len(frames) // 3))]
    arr = fr.astype(np.float32)
    H = arr.shape[0]
    lum = arr.mean(axis=2)

    # Top typed bar: thin dark horizontal rule near the top.
    top = 0
    for y in range(0, min(48, H // 6)):
        dark = float((lum[y] < 90).mean())
        if dark > 0.12:
            top = y + 1
    if top:
        top = min(H // 6, max(top + 2, 10))

    # Bottom scrubber bar.
    bottom = 0
    for y in range(H - 2, max(H - 56, H // 8), -1):
        row = lum[y]
        dark = float((row < 60).mean())
        band = arr[max(0, y - 3) : min(H, y + 4)]
        blue = float(
            (
                (band[:, :, 2] > 140)
                & (band[:, :, 2] > band[:, :, 0] + 40)
                & (band[:, :, 2] > band[:, :, 1] + 20)
            ).mean()
        )
        if dark > 0.10 and blue > 0.0002:
            bottom = int(min(H // 5, max(22, H - y + 8)))
            break
    if bottom == 0:
        for y in range(H - 2, max(H - 48, H // 8), -1):
            if float((lum[y] < 50).mean()) > 0.22:
                bottom = int(min(H // 5, max(20, H - y + 6)))
                break
    if bottom == 0 and float((lum[-28:] < 70).mean()) > 0.08:
        bottom = 28
    if bottom == 0:
        bottom = 24
    return int(top), int(bottom)


def crop_progress(frames: list[np.ndarray]) -> list[np.ndarray]:
    top, bottom = detect_progress_bar_height(frames)
    out = []
    for fr in frames:
        h = fr.shape[0]
        y0 = min(top, h // 5)
        y1 = max(y0 + 1, h - bottom)
        out.append(fr[y0:y1])
    return out


def edge_trim(img: Image.Image, px: int = EDGE_TRIM) -> Image.Image:
    if px <= 0:
        return img
    w, h = img.size
    if w <= 2 * px or h <= 2 * px:
        return img
    return img.crop((px, px, w - px, h - px))


def _smooth_path(pts: list[tuple[int, int]], k: int = 5) -> list[tuple[int, int]]:
    if len(pts) < 3:
        return pts
    arr = np.asarray(pts, dtype=float)
    k = min(k, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
    k = max(3, k)
    if k % 2 == 0:
        k -= 1
    pad = k // 2
    ker = np.ones(k) / k
    xs = np.convolve(np.pad(arr[:, 0], (pad, pad), mode="edge"), ker, mode="valid")
    ys = np.convolve(np.pad(arr[:, 1], (pad, pad), mode="edge"), ker, mode="valid")
    return [(int(round(x)), int(round(y))) for x, y in zip(xs, ys)]


def _subsample_path(pts: list[tuple[int, int]], min_step: float = 1.5) -> list[tuple[int, int]]:
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    thr = min_step * min_step
    for p in pts[1:]:
        dx = p[0] - out[-1][0]
        dy = p[1] - out[-1][1]
        if dx * dx + dy * dy >= thr:
            out.append(p)
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


def track_centroid(frames: list[np.ndarray], platform: str) -> list[tuple[int, int]]:
    if not frames:
        return []
    f0 = frames[0].astype(np.float32)
    H, W = f0.shape[:2]
    pts: list[tuple[int, int]] = []
    prev_pt: tuple[int, int] | None = None
    prev_fr = f0

    for fr in frames:
        arr = fr.astype(np.float32)
        lum = arr.mean(axis=2)
        if platform == "roomba":
            dark = lum < 52
            ys_d, xs_d = np.where(dark)
            if len(xs_d) < 80:
                if prev_pt is not None:
                    pts.append(prev_pt)
                prev_fr = arr
                continue
            bx, by = int(np.median(xs_d)), int(np.median(ys_d))
            # Blue stick tip = farthest saturated-blue pixel from body center.
            blue = (
                (arr[:, :, 2] > 150)
                & (arr[:, :, 2] > arr[:, :, 0] + 45)
                & (arr[:, :, 2] > arr[:, :, 1] + 35)
                & (arr[:, :, 0] < 120)
            )
            ys_b, xs_b = np.where(blue)
            if len(xs_b) >= 4:
                d2 = (xs_b.astype(float) - bx) ** 2 + (ys_b.astype(float) - by) ** 2
                # Keep candidates clearly outside the disk core.
                outer = d2 >= max(36.0, float(np.percentile(d2, 60)))
                if outer.any():
                    xs_o, ys_o, d_o = xs_b[outer], ys_b[outer], d2[outer]
                    k = int(np.argmax(d_o))
                    cx, cy = int(xs_o[k]), int(ys_o[k])
                else:
                    k = int(np.argmax(d2))
                    cx, cy = int(xs_b[k]), int(ys_b[k])
            else:
                cx, cy = bx, by
        else:
            mag = np.abs(arr - prev_fr).sum(axis=2)
            mag[-max(8, H // 12) :, :] = 0
            mag[: max(2, H // 50), :] = 0
            thr = float(np.percentile(mag, 96))
            mask = mag >= max(thr, 18.0)
            ys, xs = np.where(mask)
            if len(xs) < 25:
                if prev_pt is not None:
                    pts.append(prev_pt)
                prev_fr = arr
                continue
            cx, cy = int(np.median(xs)), int(np.median(ys))
            if prev_pt is not None:
                near = (xs - prev_pt[0]) ** 2 + (ys - prev_pt[1]) ** 2 < (
                    0.28 * max(H, W)
                ) ** 2
                if int(near.sum()) >= 15:
                    cx = int(np.median(xs[near]))
                    cy = int(np.median(ys[near]))

        if prev_pt is not None:
            jump = ((cx - prev_pt[0]) ** 2 + (cy - prev_pt[1]) ** 2) ** 0.5
            if jump > 0.28 * max(H, W):
                cx = int(0.75 * prev_pt[0] + 0.25 * cx)
                cy = int(0.75 * prev_pt[1] + 0.25 * cy)
        prev_pt = (cx, cy)
        pts.append(prev_pt)
        prev_fr = arr

    pts = _smooth_path(pts, k=5)
    pts = _subsample_path(pts, min_step=1.5)
    if len(pts) >= 2:
        path_len = sum(
            ((pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2) ** 0.5
            for i in range(1, len(pts))
        )
        if path_len < 6:
            return []
    return pts


def load_phase_index() -> dict[tuple[str, int, str], dict]:
    """Optional precomputed (neutral→pose→movement→neutral) indices from traces."""
    path = OUT / "pose_phase_index.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, int, str], dict] = {}
    for key, val in raw.items():
        # key format: platform|cNN|method
        try:
            plat, cue, method = key.split("|")
            idx = int(cue[1:])
            out[(plat, idx, method)] = val
        except Exception:
            continue
    return out


def pick_pose_frame_index(
    frames: list[np.ndarray],
    *,
    platform: str,
    method: str,
    idx: int,
    phase_index: dict[tuple[str, int, str], dict] | None = None,
) -> int:
    """Pick communicative pose frame: not first/last neutral.

    Prefer trace-based pose_idx mapped by timeline fraction; else max visual
    change from the opening neutral frame (hold near peak change).
    """
    n = len(frames)
    if n <= 1:
        return 0
    if phase_index:
        ph = phase_index.get((platform, idx, method))
        if ph and not ph.get("static", False):
            # Map from trace length if known via move_hi, else assume pose_idx is absolute on unknown T → use fraction with move_hi as T proxy
            pose_i = int(ph.get("pose_idx", n // 2))
            t_trace = max(pose_i + 1, int(ph.get("move_hi", 0)) or (pose_i + 1))
            mapped = int(round(pose_i / max(t_trace - 1, 1) * (n - 1)))
            return int(np.clip(mapped, 0, n - 1))
        if ph and ph.get("static", False):
            # Static hold already at pose — mid frame is fine.
            return n // 2

    f0 = frames[0].astype(np.float32)
    # Mean abs diff from opening neutral; ignore tiny lead/trail.
    lo = max(1, int(0.08 * n))
    hi = min(n - 1, int(0.92 * n))
    if hi <= lo:
        lo, hi = 1, n - 1
    scores = []
    for i in range(lo, hi + 1):
        scores.append((i, float(np.abs(frames[i].astype(np.float32) - f0).mean())))
    if not scores:
        return n // 2
    # Peak change, then prefer a nearby lower-diff "hold" (local plateau).
    peak_i, peak_s = max(scores, key=lambda t: t[1])
    if peak_s < 1.0:
        return n // 2
    thr = 0.85 * peak_s
    near = [i for i, s in scores if s >= thr]
    # Among near-peak, pick the one with smallest frame-to-frame change (hold).
    best = peak_i
    best_step = 1e18
    for i in near:
        step = float(np.abs(frames[i].astype(np.float32) - frames[max(0, i - 1)].astype(np.float32)).mean())
        if step < best_step:
            best_step = step
            best = i
    return best


def pick_mouth_frame(frames: list[np.ndarray]) -> Image.Image:
    """Pick a mid-motion frame where the moving hand is nearest the face/mouth region."""
    if len(frames) < 3:
        return Image.fromarray(frames[-1]).convert("RGB")
    f0 = frames[0].astype(np.float32)
    H, W = f0.shape[:2]
    mouth = np.array([0.50 * W, 0.32 * H], dtype=float)  # upper-center face
    best_i = len(frames) // 2
    best = 1e18
    lo = max(1, int(0.20 * len(frames)))
    hi = min(len(frames) - 1, int(0.85 * len(frames)))
    for i in range(lo, hi + 1):
        diff = np.abs(frames[i].astype(np.float32) - f0).sum(axis=2)
        # Focus on upper-middle where the hand-to-mouth contact happens.
        roi = np.zeros_like(diff, dtype=bool)
        roi[int(0.12 * H) : int(0.55 * H), int(0.25 * W) : int(0.75 * W)] = True
        ys, xs = np.where((diff > 40) & roi)
        if len(xs) < 40:
            continue
        # Prefer the highest moving mass (hand raised to mouth).
        cy = float(np.percentile(ys, 25))
        cx = float(np.median(xs[ys <= np.percentile(ys, 40)]))
        score = (cx - mouth[0]) ** 2 + (cy - mouth[1]) ** 2
        if score < best:
            best = score
            best_i = i
    return Image.fromarray(frames[best_i]).convert("RGB")


def font_pair(size: int, size_s: int) -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    font = font_s = None
    for path in candidates:
        try:
            font = ImageFont.truetype(path, size)
            font_s = ImageFont.truetype(path, size_s)
            break
        except Exception:
            continue
    if font is None:
        font = font_s = ImageFont.load_default()
    return font, font_s


def overlay_path_on_first_frame(
    gif: Path, platform: str
) -> tuple[Image.Image, list[tuple[int, int]]]:
    frames = crop_progress(load_gif_frames(gif))
    base = Image.fromarray(frames[0]).convert("RGB")
    pts = track_centroid(frames, platform)
    if len(pts) < 2:
        return edge_trim(base), pts
    img = _image_with_ee_path(base, pts)
    draw = ImageDraw.Draw(img)
    tail = pts[-min(8, len(pts)) :]
    _draw_thick_arrow(
        draw, tail, color=(138, 80, 220), width=5, head_len=14, head_w=11
    )
    return edge_trim(img), pts


def prepare_cell_image(
    gif: Path | None,
    *,
    platform: str,
    method: str,
    idx: int,
    dynamics: bool,
    force_cache: bool,
    write_cache: bool = True,
    phase_index: dict[tuple[str, int, str], dict] | None = None,
) -> Image.Image | None:
    if gif is None or not gif.is_file():
        return None
    if dynamics:
        cache = CACHE / f"{platform}_c{idx:02d}_{method}_v7_bluetip.png"
        if force_cache or not cache.is_file():
            img, pts = overlay_path_on_first_frame(gif, platform)
            if write_cache:
                img.save(cache)
            print(f"  {platform} c{idx:02d} {method}: pts={len(pts)}")
            return img
        return Image.open(cache).convert("RGB")
    frames = crop_progress(load_gif_frames(gif))
    # Humanoid yawn Cartesian: mouth-contact heuristic still helps when the
    # arm barely clears the neutral silhouette.
    if platform == "humanoid" and idx == 7 and method == "xyzrpy":
        return edge_trim(pick_mouth_frame(frames))
    # Pose grids: communicative pose phase, not first/last neutral.
    pi = pick_pose_frame_index(
        frames, platform=platform, method=method, idx=idx, phase_index=phase_index
    )
    return edge_trim(Image.fromarray(frames[pi]).convert("RGB"))


def fit_cell(img: Image.Image, size: int, height: int | None = None) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = img.crop((left, top, left + side, top + side))
    if height is None:
        height = size
    if height < size:
        # Remove unused upper background from each rendered panel.
        cropped = cropped.crop((0, size - height, size, size))
    return cropped.resize((size, height), Image.Resampling.LANCZOS)


def stroke_outline(img: Image.Image, color: tuple[int, int, int], width: int) -> Image.Image:
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for t in range(width):
        draw.rectangle([t, t, w - 1 - t, h - 1 - t], outline=color)
    return out


def draw_ox_badge(img: Image.Image, yes: bool, font: ImageFont.ImageFont, *, lower_right: bool = False) -> Image.Image:
    """Compact O/X sticker matching the human-eval gallery badge style."""
    out = img.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    mark = "O" if yes else "X"
    fg = YES_FG if yes else NO_FG
    bg = YES_BG if yes else NO_BG
    border = YES_COLOR if yes else NO_COLOR
    # Measure text for a compact pill.
    bbox = draw.textbbox((0, 0), mark, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 8, 4
    bw, bh = tw + 2 * pad_x, th + 2 * pad_y
    margin = 8
    x1 = out.size[0] - margin
    y0 = out.size[1] - margin - bh if lower_right else margin
    x0 = x1 - bw
    y1 = y0 + bh
    draw.rounded_rectangle([x0, y0, x1, y1], radius=5, fill=(*bg, 230), outline=(*border, 255), width=2)
    # Center the glyph in the pill.
    tx = x0 + (bw - tw) / 2 - bbox[0]
    ty = y0 + (bh - th) / 2 - bbox[1]
    draw.text((tx, ty), mark, fill=(*fg, 255), font=font)
    # Soft drop shadow under the badge (like gallery box-shadow).
    shadow = Image.new("RGBA", out.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle([x0 + 1, y0 + 2, x1 + 1, y1 + 2], radius=5, fill=(0, 0, 0, 70))
    composed = Image.alpha_composite(out, shadow)
    composed = Image.alpha_composite(composed, overlay)
    return composed.convert("RGB")


def make_grid(
    rows: list[tuple[str, int]],
    out_path: Path,
    title: str,
    *,
    cues: dict[int, str],
    labels: dict[tuple[str, int, str], bool],
    traj: dict[tuple[str, int], Path],
    dynamics: bool,
    force_cache: bool,
    phase_index: dict[tuple[str, int, str], dict] | None = None,
    show_ox: bool = True,
    show_title: bool = True,
    methods: list[str] | None = None,
    cue_in_gt: bool = False,
    lower_right_badge: bool = False,
    crop_top_fifth: bool = False,
    write_cache: bool = True,
    platform_in_label: bool = False,
) -> None:
    font, font_s = font_pair(24, 18)
    use_methods = methods if methods is not None else METHODS
    n_m = len(use_methods)
    n_r = len(rows)
    title_h = TITLE_H if show_title and title else 0
    label_w = 0 if cue_in_gt else LABEL_W
    cell_h = int(CELL * 0.8) if crop_top_fifth else CELL
    W = label_w + CELL * n_m
    H = title_h + HEADER_H + n_r * cell_h + max(0, n_r - 1) * CUE_GAP
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    if title_h:
        draw.text((10, 8), title, fill="#111111", font=font)
    for j, m in enumerate(use_methods):
        x = label_w + j * CELL
        lab = MLAB[m]
        bbox = draw.textbbox((0, 0), lab, font=font_s)
        tw = bbox[2] - bbox[0]
        draw.text((x + (CELL - tw) / 2, title_h + 10), lab, fill="#111111", font=font_s)

    for i, (platform, idx) in enumerate(rows):
        y0 = title_h + HEADER_H + i * (cell_h + CUE_GAP)
        label = cues[idx].replace("_", " ")
        if platform_in_label:
            label = f"{PLATFORM_LABEL.get(platform, platform)} · {label}"
        if len(label) > 22 and not cue_in_gt:
            label = label[:21] + "…"
        bbox = draw.textbbox((0, 0), label, font=font_s)
        th = bbox[3] - bbox[1]
        if not cue_in_gt:
            draw.text((10, y0 + (cell_h - th) / 2), label, fill="#222222", font=font_s)

        for j, m in enumerate(use_methods):
            x0 = label_w + j * CELL
            g = gif_path(m, platform, idx, cues, traj)
            img = prepare_cell_image(
                g,
                platform=platform,
                method=m,
                idx=idx,
                dynamics=dynamics,
                force_cache=force_cache,
                write_cache=write_cache,
                phase_index=phase_index,
            )
            yes = labels.get((platform, idx, m), False)
            if img is not None:
                fitted = fit_cell(img, CELL, cell_h)
            else:
                fitted = Image.new("RGB", (CELL, cell_h), "#f2f2f2")
                md = ImageDraw.Draw(fitted)
                md.text((CELL // 2 - 28, CELL // 2 - 8), "missing", fill="#888888", font=font_s)
            if cue_in_gt and j == 0:
                fd = ImageDraw.Draw(fitted)
                lines = label.split(" · ", 1) if platform_in_label else [label]
                boxes = [fd.textbbox((0, 0), line, font=font_s) for line in lines]
                widths = [b[2] - b[0] for b in boxes]
                heights = [b[3] - b[1] for b in boxes]
                box_w = min(CELL - 12, max(widths) + 18)
                box_h = sum(heights) + 10 + 4 * max(0, len(lines) - 1)
                fd.rounded_rectangle([6, 6, box_w, 6 + box_h], radius=3, fill="white", outline="#dddddd")
                yy = 10
                for line, hh in zip(lines, heights):
                    fd.text((12, yy), line, fill="#111111", font=font_s)
                    yy += hh + 4
            if show_ox:
                fitted = draw_ox_badge(fitted, yes, font_s, lower_right=lower_right_badge)
            canvas.paste(fitted, (x0, y0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print("wrote", out_path, canvas.size)


def contrastive_gif_path(method: str, idx: int, cue: str, foil: str) -> Path | None:
    root = REPO / "data/results/roomba_contrastive_distinguish"
    if method == "gt":
        return gif_path("gt", "roomba", idx, {idx: cue}, {})
    if method == "joint":
        p = root / "baseline_joint/render_joint/roomba" / f"emb_g{idx:02d}_{cue}__vs_{foil}_gemini-2.5-pro.gif"
    elif method == "xyzrpy":
        p = root / "baseline_xyzrpy/render_xyzrpy/roomba" / f"emb_g{idx:02d}_{cue}__vs_{foil}_gemini-2.5-pro.gif"
    elif method == "motion_program":
        p = root / "motion_program/render" / f"emb_g{idx:02d}_{cue}__vs_{foil}.gif"
    elif method == "trajectory":
        p = root / "trajectory/render/roomba" / f"c{idx:02d}_{cue}__vs_{foil}_gemini-2.5-pro.gif"
    else:
        return None
    return p if p.is_file() else None


def make_contrastive_grid(
    out_path: Path,
    *,
    cues: dict[int, str],
) -> None:
    """Build a three-row contrastive grid matching the five-column Fig. 5."""
    _, font_s = font_pair(24, 18)
    cell_h = CELL
    W = CELL * len(METHODS)
    H = HEADER_H + len(CONTRASTIVE_ROWS) * cell_h + (len(CONTRASTIVE_ROWS) - 1) * CUE_GAP
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    for j, method in enumerate(METHODS):
        lab = MLAB[method]
        bbox = draw.textbbox((0, 0), lab, font=font_s)
        tw = bbox[2] - bbox[0]
        draw.text((j * CELL + (CELL - tw) / 2, 10), lab, fill="#111111", font=font_s)

    for i, (idx, foil_idx) in enumerate(CONTRASTIVE_ROWS):
        cue, foil = cues[idx], cues[foil_idx]
        y0 = HEADER_H + i * (cell_h + CUE_GAP)
        for j, method in enumerate(METHODS):
            path = contrastive_gif_path(method, idx, cue, foil)
            if path is not None:
                img, _ = overlay_path_on_first_frame(path, "roomba")
                fitted = fit_cell(img, CELL, cell_h)
            else:
                fitted = Image.new("RGB", (CELL, cell_h), "#f2f2f2")
                ImageDraw.Draw(fitted).text((CELL // 2 - 28, CELL // 2 - 8), "missing", fill="#888888", font=font_s)
            canvas.paste(fitted, (j * CELL, y0))

        # Keep the full contrastive request on one line.  It intentionally may
        # extend beyond the first cell, but remains readable over a white strip.
        label = f"{cue.replace('_', ' ')}  !=  {foil.replace('_', ' ')}"
        bbox = draw.textbbox((0, 0), label, font=font_s)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        lx, ly = 8, y0 + cell_h - th - 16
        draw.rounded_rectangle([lx - 4, ly - 4, lx + tw + 8, ly + th + 8], radius=3, fill="white", outline="#dddddd")
        draw.text((lx, ly), label, fill="#111111", font=font_s)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print("wrote", out_path, canvas.size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platforms", default="humanoid,single_arm,roomba")
    ap.add_argument("--force-cache", action="store_true")
    args = ap.parse_args()

    cues = cues_map()
    labels = load_labels()
    traj = traj_renders()
    phase_index = load_phase_index()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.force_cache and CACHE.exists():
        shutil.rmtree(CACHE)
    CACHE.mkdir(parents=True, exist_ok=True)

    plats = {p.strip() for p in args.platforms.split(",") if p.strip()}
    if "humanoid" in plats:
        make_grid(
            HU_ROWS,
            OUT / "qual_grid_humanoid_pose.png",
            "Humanoid / Tiago · pose-phase frame",
            cues=cues,
            labels=labels,
            traj=traj,
            dynamics=False,
            force_cache=args.force_cache,
            phase_index=phase_index,
        )
    if "single_arm" in plats:
        make_grid(
            SA_ROWS,
            OUT / "qual_grid_single_arm_pose.png",
            "Single Arm · pose-phase frame",
            cues=cues,
            labels=labels,
            traj=traj,
            dynamics=False,
            force_cache=args.force_cache,
            phase_index=phase_index,
        )
    if "roomba" in plats:
        make_grid(
            RB_ROWS,
            OUT / "qual_grid_roomba_dynamics.png",
            "Roomba · first frame + trajectory path",
            cues=cues,
            labels=labels,
            traj=traj,
            dynamics=True,
            force_cache=args.force_cache,
            phase_index=phase_index,
        )
    if "main" in plats:
        make_grid(
            MAIN_ROWS, OUT / "qual_grid_main_cross_embodiment.png", "",
            cues=cues, labels=labels, traj=traj, dynamics=True,
            force_cache=args.force_cache, phase_index=phase_index,
            show_title=False, cue_in_gt=True, lower_right_badge=True,
            crop_top_fifth=False, write_cache=False,
            platform_in_label=True,
        )
    if "contrastive" in plats:
        make_contrastive_grid(
            OUT / "roomba_contrastive_distinguish_grid_compact.png",
            cues=cues,
        )


if __name__ == "__main__":
    main()
