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
    parse_idx_arg,
    pose_png_for_row,
    repo,
)


def _font(size: int = 11) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _cell(img: Image.Image, *, idx: int, cue: str, cell_w: int, img_h: int, footer_h: int) -> Image.Image:
    canvas = Image.new("RGB", (cell_w, img_h + footer_h), (255, 255, 255))
    iw, ih = img.size
    scale = min((cell_w - 8) / iw, (img_h - 20) / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    ox = (cell_w - nw) // 2
    oy = 18 + (img_h - 20 - nh) // 2
    canvas.paste(resized, (ox, oy))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), f"#{idx}", fill=(80, 80, 80), font=_font(10))
    cue_short = cue.replace("_", " ")
    draw.text((4, img_h + 2), cue_short, fill=(0, 0, 0), font=_font(9))
    return canvas


def build_grid(
    *,
    mode: str,
    idx_arg: str,
    ncols: int = 3,
    out_path: Path | None = None,
) -> Path:
    rows_all = manifest90_rows()
    by_idx = cfg_by_idx()
    idxs = parse_idx_arg(idx_arg, rows_all)
    if not idxs:
        raise SystemExit("No indices selected")

    cells: list[Image.Image] = []
    for idx in idxs:
        row = by_idx.get(idx)
        if not row:
            print(f"[skip] missing config idx={idx}", flush=True)
            continue
        cue = str(row["cue"])
        cache_dir = CACHE / mode / f"c{idx}"
        if mode == "pose":
            media = pose_png_for_row(row, cache_dir / "pose.png")
        elif mode == "movement":
            media = alpha_trajectory_for_row(row, cache_dir / "alpha_traj.png")
        else:
            raise ValueError(mode)
        if media is None:
            placeholder = Image.new("RGB", (200, 160), (240, 240, 240))
            d = ImageDraw.Draw(placeholder)
            d.text((20, 70), "missing", fill=(120, 120, 120), font=_font(12))
            img = placeholder
        else:
            img = Image.open(media).convert("RGB")
        cells.append(_cell(img, idx=idx, cue=cue, cell_w=220, img_h=180, footer_h=28))

    if not cells:
        raise SystemExit("No cells rendered")

    ncols = ncols if idx_arg.lower() != "all" else 3
    nrows = math.ceil(len(cells) / ncols)
    cell_w, cell_h = cells[0].size
    grid = Image.new("RGB", (ncols * cell_w, nrows * cell_h), (255, 255, 255))
    for i, cell in enumerate(cells):
        r, c = divmod(i, ncols)
        grid.paste(cell, (c * cell_w, r * cell_h))

    od = repo() / "data/results/paper_figures"
    od.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        tag = "all" if idx_arg.lower() == "all" else "sel"
        out_path = od / f"qual_{mode}_{tag}.png"
    grid.save(out_path)
    print(f"Wrote {out_path} ({len(cells)} panels)")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Generation qualitative grid (pose or movement)")
    p.add_argument("--mode", choices=("pose", "movement"), required=True)
    p.add_argument("--idx", default="all", help='Cue idx list or "all"')
    p.add_argument("--ncols", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    build_grid(mode=args.mode, idx_arg=args.idx, ncols=args.ncols, out_path=Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
