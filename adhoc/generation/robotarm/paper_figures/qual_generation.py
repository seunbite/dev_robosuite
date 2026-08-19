"""Qualitative grids: task-1 pose PNG / task-7 alpha trajectory (generation only)."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
_ROBOTARM = _HERE.parent
for p in (_ROBOTARM.parents[2], _ROBOTARM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from paper_figures._media import (  # noqa: E402
    CACHE,
    alpha_trajectory_for_row,
    cfg_by_idx,
    manifest90_rows,
    motion_split_idxs,
    parse_idx_arg,
    pose_correct_idxs,
    pose_png_for_row,
    pose_split_idxs,
    repo,
)

FONT_IDX = 20
FONT_CUE = 18
FONT_MISS = 24
CELL_SIZE = 220
VERIFY_CELL_SIZE = 420
FOOTER_H = 32
CROP_TOP_FRAC = 0.08
CROP_BOTTOM_FRAC = 0.06
PAD = 6


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _parse_csv_ints(s: str | None) -> list[int]:
    if not s or not str(s).strip():
        return []
    return [int(p.strip()) for p in str(s).replace("[", "").replace("]", "").split(",") if p.strip()]


def _text_outline(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    anchor: str | None = None,
) -> None:
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0), anchor=anchor)
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _crop_vertical(
    img: Image.Image,
    *,
    top_frac: float = CROP_TOP_FRAC,
    bottom_frac: float = CROP_BOTTOM_FRAC,
) -> Image.Image:
    w, h = img.size
    top = max(1, int(h * top_frac))
    bottom = min(h - 1, int(h * (1 - bottom_frac)))
    if bottom <= top:
        return img
    return img.crop((0, top, w, bottom))


def _fit_into_box(img: Image.Image, width: int, height: int) -> Image.Image:
    """Resize with aspect ratio preserved; center on white canvas."""
    iw, ih = img.size
    scale = min(width / iw, height / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(resized, ((width - nw) // 2, (height - nh) // 2))
    return canvas


def _fit_to_width(img: Image.Image, width: int) -> Image.Image:
    """Resize to exact width; height follows aspect ratio (no letterbox)."""
    iw, ih = img.size
    nh = max(1, int(round(width * ih / iw)))
    return img.resize((width, nh), Image.Resampling.LANCZOS)


def _draw_pair_markers(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    gt_side: str,
    mark_wrong: bool,
) -> None:
    """Green circle / red X at top-right of each panel (0.75× base size)."""
    mid = width // 2
    margin = 10
    r = 10
    lw = 2
    s = 8
    y = margin + r
    gt_side = gt_side.lower()
    left_cx = mid - margin - r
    right_cx = width - margin - r
    if gt_side == "left":
        gx, wx = left_cx, right_cx
    else:
        gx, wx = right_cx, left_cx
    draw.ellipse((gx - r, y - r, gx + r, y + r), outline=(34, 180, 34), width=lw)
    if mark_wrong:
        draw.line((wx - s, y - s, wx + s, y + s), fill=(210, 35, 35), width=lw)
        draw.line((wx - s, y + s, wx + s, y - s), fill=(210, 35, 35), width=lw)


def _cell_overlay(img: Image.Image, *, idx: int, cue: str, size: int = CELL_SIZE) -> Image.Image:
    """Full-grid tile: square image + idx/cue overlaid on PNG."""
    canvas = img.resize((size, size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas)
    _text_outline(draw, (PAD, PAD), f"#{idx}", font=_font(FONT_IDX), fill=(255, 255, 255))
    cue_short = cue.replace("_", " ")
    _text_outline(
        draw,
        (size - PAD, PAD),
        cue_short,
        font=_font(FONT_CUE),
        fill=(255, 255, 255),
        anchor="ra",
    )
    return canvas


def _pose_state_label(row: dict) -> str | None:
    from verify_pose_tiles_gemini import _first_pose

    p = _first_pose(row)
    d, g = p.get("dir"), p.get("gripper_orientation")
    if d and g:
        return f"({d}, {g})"
    return None


def _path_shape_label(p: dict) -> str:
    shape = str(p.get("shape") or "")
    if shape in ("arc", "circle"):
        return "arc"
    return "line"


def _path_axis_label(p: dict) -> str:
    axis = p.get("axis") or p.get("plane")
    return str(axis).lower() if axis else _path_shape_label(p)


def _movement_axis_label(step: dict) -> str:
    p = step.get("parameters") or {}
    for d in p.get("directions") or []:
        deg = d.get("degrees") or {}
        for ax in ("x", "y", "z"):
            if ax in deg:
                return ax
    return "line"


def _repetitive_movement_label(step: dict) -> str:
    from score_pilot40_motion_gt_components import _step_repetition

    p = step.get("parameters") or {}
    rep = _step_repetition(step)
    n_dirs = len(p.get("directions") or [])
    if rep >= 2 or n_dirs >= 2:
        return "rep"
    return "single"


def _repetitive_path_label(step: dict, *, n_path_steps: int) -> str:
    p = step.get("parameters") or {}
    shape = str(p.get("shape") or "")
    if shape in ("arc", "circle"):
        sweep = float(p.get("sweep") or 0)
        return "rep" if sweep >= 360 else "single"
    if n_path_steps >= 2:
        return "rep"
    return "single"


def _movement_state_label(row: dict) -> str | None:
    """Generation tail summary as (axis, joint, repetitive)."""
    from score_pilot40_motion_gt_components import _step_joint, _tail_steps

    tail = _tail_steps(row.get("movements") or [])
    if not tail:
        return None

    arcs = [
        s
        for s in tail
        if s.get("type") == "path" and str((s.get("parameters") or {}).get("shape") or "") in ("arc", "circle")
    ]
    if arcs:
        step = max(arcs, key=lambda s: float((s.get("parameters") or {}).get("sweep") or 0))
        p = step.get("parameters") or {}
        kind = _path_shape_label(p)
        return f"({_path_axis_label(p)}, {kind}, {_repetitive_path_label(step, n_path_steps=len(arcs))})"

    paths = [s for s in tail if s.get("type") == "path"]
    movements = [s for s in tail if s.get("type") == "movement"]
    if paths and not movements:
        step = paths[0]
        p = step.get("parameters") or {}
        kind = _path_shape_label(p)
        return f"({_path_axis_label(p)}, {kind}, {_repetitive_path_label(step, n_path_steps=len(paths))})"

    step = movements[0] if movements else None
    if step is None:
        return None
    joint = _step_joint(step) or "line"
    return f"({_movement_axis_label(step)}, {joint}, {_repetitive_movement_label(step)})"


def _cell_selective(
    img: Image.Image,
    *,
    cue: str,
    width: int = CELL_SIZE,
    footer_h: int = FOOTER_H,
    crop_top_frac: float = CROP_TOP_FRAC,
    crop_bottom_frac: float = CROP_BOTTOM_FRAC,
) -> Image.Image:
    """Generation paper pick: crop, fixed cell, aspect-preserving fit, cue on footer."""
    img_h = width - footer_h
    body = _crop_vertical(img, top_frac=crop_top_frac, bottom_frac=crop_bottom_frac)
    body = _fit_into_box(body, width, img_h)
    cue_short = cue.replace("_", " ")
    canvas = Image.new("RGB", (width, img_h + footer_h), (255, 255, 255))
    canvas.paste(body, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (width / 2, img_h + footer_h / 2),
        cue_short,
        fill=(0, 0, 0),
        font=_font(FONT_CUE),
        anchor="mm",
    )
    return canvas


def _cell_selective_verify(
    img: Image.Image,
    *,
    cue: str,
    gt_side: str,
    mark_wrong: bool,
    width: int = VERIFY_CELL_SIZE,
    footer_h: int | None = None,
    crop_top_frac: float = 0.10,
    crop_bottom_frac: float = 0.08,
) -> Image.Image:
    """Verification pairwise: tight width-fit, GT circle / distractor X, cue on footer."""
    footer_h = footer_h or max(FOOTER_H, int(round(FOOTER_H * width / CELL_SIZE)))
    font_sz = max(FONT_CUE, int(round(FONT_CUE * width / CELL_SIZE)))
    body = _crop_vertical(img, top_frac=crop_top_frac, bottom_frac=crop_bottom_frac)
    body = _fit_to_width(body, width)
    img_h = body.height
    cue_short = cue.replace("_", " ")
    canvas = Image.new("RGB", (width, img_h + footer_h), (255, 255, 255))
    canvas.paste(body, (0, 0))
    draw = ImageDraw.Draw(canvas)
    _draw_pair_markers(draw, width=width, gt_side=gt_side, mark_wrong=mark_wrong)
    draw.text(
        (width / 2, img_h + footer_h / 2),
        cue_short,
        fill=(0, 0, 0),
        font=_font(font_sz),
        anchor="mm",
    )
    return canvas


def _load_media(idx: int, *, mode: str, by_idx: dict[int, dict], size: int) -> tuple[Image.Image, str] | None:
    row = by_idx.get(idx)
    if not row:
        print(f"[skip] missing config idx={idx}", flush=True)
        return None
    cue = str(row["cue"])
    cache_dir = CACHE / mode / f"c{idx}"
    if mode == "pose":
        media = pose_png_for_row(row, cache_dir / "pose.png")
    elif mode == "movement":
        media = alpha_trajectory_for_row(row, cache_dir / "alpha_traj.png")
    else:
        raise ValueError(mode)
    if media is None:
        placeholder = Image.new("RGB", (size, size), (200, 200, 200))
        d = ImageDraw.Draw(placeholder)
        _text_outline(d, (size // 2, size // 2), "missing", font=_font(FONT_MISS), fill=(80, 80, 80), anchor="mm")
        img = placeholder
    else:
        img = Image.open(media).convert("RGB")
    return img, cue


def _stitch(cells: list[Image.Image], ncols: int) -> Image.Image | None:
    if not cells:
        return None
    cw = cells[0].width
    rows = [cells[i : i + ncols] for i in range(0, len(cells), ncols)]
    row_heights = [max(c.height for c in row) for row in rows]
    grid = Image.new("RGB", (ncols * cw, sum(row_heights)), (255, 255, 255))
    y = 0
    for row, rh in zip(rows, row_heights):
        for c_idx, cell in enumerate(row):
            grid.paste(cell, (c_idx * cw, y))
        y += rh
    return grid


def _stack(top: Image.Image | None, bottom: Image.Image | None) -> Image.Image:
    parts = [p for p in (top, bottom) if p is not None]
    if not parts:
        raise SystemExit("No panels rendered")
    if len(parts) == 1:
        return parts[0]
    w = max(p.width for p in parts)
    h = sum(p.height for p in parts)
    out = Image.new("RGB", (w, h))
    y = 0
    for p in parts:
        out.paste(p, (0, y))
        y += p.height
    return out


def build_grid(
    *,
    mode: str,
    idx_arg: str,
    ncols: int = 3,
    out_path: Path | None = None,
    require_pose_correct: bool | None = None,
    cell_size: int = CELL_SIZE,
    good_idxs: list[int] | None = None,
    bad_idxs: list[int] | None = None,
    selective: bool = False,
) -> Path:
    by_idx = cfg_by_idx()
    selective = selective or bool(good_idxs or bad_idxs)
    good_idxs = good_idxs or []
    bad_idxs = bad_idxs or []

    if selective:
        correct_idxs, wrong_idxs = good_idxs, bad_idxs
        label_ok, label_bad = "good", "bad"
        ncols = ncols or 3
    elif mode == "movement":
        rows_all = manifest90_rows()
        filter_pose = True if require_pose_correct is None else require_pose_correct
        pose_ok = set(pose_correct_idxs()) if filter_pose else None
        pool = [r for r in rows_all if not filter_pose or int(r["idx"]) in pose_ok]
        if idx_arg.strip().lower() == "all":
            idxs = [int(r["idx"]) for r in sorted(pool, key=lambda r: int(r["idx"]))]
        else:
            idxs = parse_idx_arg(idx_arg, pool)
        if filter_pose and pose_ok is not None:
            skipped = [i for i in parse_idx_arg(idx_arg, rows_all) if i not in pose_ok]
            for i in skipped:
                print(f"[skip] idx={i}: pose generation not GT-correct (movement pool only)", flush=True)
            print(f"[movement] pose-GT-correct pool: {len(pose_ok)} cues", flush=True)
        correct_idxs, wrong_idxs = motion_split_idxs(idxs)
        label_ok, label_bad = "motion-GT-correct", "motion-GT-wrong"
        ncols = ncols if idx_arg.lower() != "all" else 3
    else:
        rows_all = manifest90_rows()
        idxs = parse_idx_arg(idx_arg, rows_all)
        correct_idxs, wrong_idxs = pose_split_idxs(idxs)
        label_ok, label_bad = "pose-GT-correct", "pose-GT-wrong"
        ncols = ncols if idx_arg.lower() != "all" else 3

    def _cells_for(id_list: list[int]) -> list[Image.Image]:
        out: list[Image.Image] = []
        for idx in id_list:
            loaded = _load_media(idx, mode=mode, by_idx=by_idx, size=cell_size)
            if loaded is None:
                continue
            img, cue = loaded
            row = by_idx.get(idx) or {}
            if selective:
                out.append(_cell_selective(img, cue=cue, width=cell_size))
            else:
                out.append(_cell_overlay(img, idx=idx, cue=cue, size=cell_size))
        return out

    top = _stitch(_cells_for(correct_idxs), ncols)
    bottom = _stitch(_cells_for(wrong_idxs), ncols)
    grid = _stack(top, bottom)

    od = repo() / "data/results/paper_figures"
    od.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if out_path is None:
        if selective:
            out_path = od / f"qual_{mode}_sel.png"
        else:
            tag = "all" if idx_arg.lower() == "all" else "sel"
            suffix = "_poseok" if mode == "movement" else ""
            out_path = od / f"qual_{mode}{suffix}_{tag}.png"
    grid.save(out_path)
    written.append(out_path)
    if selective:
        if top is not None:
            good_path = od / f"good_{mode}_example.png"
            top.save(good_path)
            written.append(good_path)
        if bottom is not None:
            bad_path = od / f"bad_{mode}_example.png"
            bottom.save(bad_path)
            written.append(bad_path)
    print(
        f"Wrote {', '.join(str(p.name) for p in written)} "
        f"({label_ok}={len(correct_idxs)}, {label_bad}={len(wrong_idxs)}, "
        f"ncols={ncols}, selective={selective})",
        flush=True,
    )
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Generation qualitative grid (pose or movement)")
    p.add_argument("--mode", choices=("pose", "movement"), required=True)
    p.add_argument("--idx", default="all", help='Cue idx list or "all" (ignored if --good-idx/--bad-idx set)')
    p.add_argument("--good-idx", default=None, help="Comma-separated good example idx (top row block)")
    p.add_argument("--bad-idx", default=None, help="Comma-separated bad example idx (bottom row block)")
    p.add_argument(
        "--require-pose-correct",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Movement only: restrict to pose-GT-correct cues (default: on for movement)",
    )
    p.add_argument("--ncols", type=int, default=3)
    p.add_argument("--cell-size", type=int, default=CELL_SIZE)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    build_grid(
        mode=args.mode,
        idx_arg=args.idx,
        ncols=args.ncols,
        out_path=Path(args.out) if args.out else None,
        require_pose_correct=args.require_pose_correct,
        cell_size=args.cell_size,
        good_idxs=_parse_csv_ints(args.good_idx),
        bad_idxs=_parse_csv_ints(args.bad_idx),
    )


if __name__ == "__main__":
    main()
