"""Pose / movement / path definition figure (3 subplots, different robots)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
_ROBOTARM = _HERE.parent
_REPO = _ROBOTARM.parents[2]
for p in (_REPO, _ROBOTARM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from paper_figures._media import (  # noqa: E402
    alpha_trajectory_for_row,
    cfg_by_idx,
    mid_frame_from_gif,
    pick_gif,
    pose_png_for_row,
    repo,
)

# Curated examples: pose-only, movement-heavy, path-heavy (google robot config)
DEFAULT_POSE_IDX = 16  # stop_palm_out — pose-primary
DEFAULT_MOVEMENT_IDX = 59  # nod_yes — movement-primary
GOOGLE_PATH_CFG = _REPO / "data/results/motion_configs/google_robot/motion_configs_google_robot_pilot40_manip_shots.json"


def _font(size: int = 11) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _panel(img: Image.Image, label: str, w: int = 240, h: int = 200, footer: int = 22) -> Image.Image:
    canvas = Image.new("RGB", (w, h + footer), (255, 255, 255))
    iw, ih = img.size
    scale = min((w - 10) / iw, (h - 10) / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas.paste(resized, ((w - nw) // 2, (h - nh) // 2))
    draw = ImageDraw.Draw(canvas)
    tw = draw.textlength(label, font=_font(11))
    draw.text(((w - tw) / 2, h + 4), label, fill=(0, 0, 0), font=_font(11))
    return canvas


def _path_panel_google(*, cache: Path) -> Image.Image | None:
    if not GOOGLE_PATH_CFG.is_file():
        return None
    rows = json.loads(GOOGLE_PATH_CFG.read_text(encoding="utf-8"))
    # pick circle_temple_crazy equivalent — use flex_bicep with path if exists
    row = next((r for r in rows if any(s.get("type") == "path" for s in r.get("movements", []))), rows[0])
    gif_dir = _REPO / "data/results/render/google_robot"
    gifs = sorted(gif_dir.glob(f"*_{row['cue']}*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not gifs:
        gifs = sorted(gif_dir.glob("*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)
    if gifs:
        mid = mid_frame_from_gif(gifs[0], cache / "google_mid.png")
        if mid:
            # overlay trajectory if possible via alpha on same config mapped to IIWA-style sample
            return Image.open(mid).convert("RGB")
    return None


def build(
    *,
    pose_idx: int = DEFAULT_POSE_IDX,
    movement_idx: int = DEFAULT_MOVEMENT_IDX,
    out: str | None = None,
) -> Path:
    by_idx = cfg_by_idx()
    cache = repo() / "data/results/paper_figures/cache/component_def"
    cache.mkdir(parents=True, exist_ok=True)

    pose_row = by_idx[pose_idx]
    mov_row = by_idx[movement_idx]

    pose_png = pose_png_for_row(pose_row, cache / "pose.png")
    mov_traj = alpha_trajectory_for_row(mov_row, cache / "movement_traj.png")
    mov_gif = pick_gif(str(mov_row["cue"]))
    mov_mid = mid_frame_from_gif(mov_gif, cache / "movement_mid.png") if mov_gif else None

    panels: list[Image.Image] = []

    if pose_png:
        panels.append(_panel(Image.open(pose_png).convert("RGB"), "Pose"))
    else:
        panels.append(_panel(Image.new("RGB", (200, 160), (230, 230, 230)), "Pose"))

    if mov_traj and mov_mid:
        traj = Image.open(mov_traj).convert("RGBA")
        mid = Image.open(mov_mid).convert("RGBA")
        mid = mid.resize(traj.size, Image.Resampling.LANCZOS)
        comp = Image.alpha_composite(mid, traj).convert("RGB")
        panels.append(_panel(comp, "Movement"))
    elif mov_traj:
        panels.append(_panel(Image.open(mov_traj).convert("RGB"), "Movement"))
    else:
        panels.append(_panel(Image.new("RGB", (200, 160), (230, 230, 230)), "Movement"))

    path_img = _path_panel_google(cache=cache)
    if path_img is None:
        # fallback: circle_temple_crazy on IIWA with trajectory
        path_row = by_idx.get(22) or mov_row
        path_traj = alpha_trajectory_for_row(path_row, cache / "path_traj.png")
        path_gif = pick_gif(str(path_row["cue"]))
        path_mid = mid_frame_from_gif(path_gif, cache / "path_mid.png") if path_gif else None
        if path_traj and path_mid:
            traj = Image.open(path_traj).convert("RGBA")
            mid = Image.open(path_mid).convert("RGBA").resize(traj.size, Image.Resampling.LANCZOS)
            path_img = Image.alpha_composite(mid, traj).convert("RGB")
        elif path_traj:
            path_img = Image.open(path_traj).convert("RGB")
        else:
            path_img = Image.new("RGB", (200, 160), (230, 230, 230))
    panels.append(_panel(path_img, "Path"))

    w, h = panels[0].size
    fig = Image.new("RGB", (w * 3, h), (255, 255, 255))
    for i, p in enumerate(panels):
        fig.paste(p, (i * w, 0))

    od = repo() / "data/results/paper_figures"
    od.mkdir(parents=True, exist_ok=True)
    out_path = Path(out) if out else od / "component_def_pose_movement_path.png"
    fig.save(out_path)
    (od / "component_def_caption.txt").write_text(
        "Motion primitives on distinct embodiments: static end-effector pose (IIWA), "
        "joint-space movement with trajectory overlay (IIWA), and end-effector path (Google Robot).\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Pose / movement / path definition figure")
    p.add_argument("--pose-idx", type=int, default=DEFAULT_POSE_IDX)
    p.add_argument("--movement-idx", type=int, default=DEFAULT_MOVEMENT_IDX)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    build(pose_idx=args.pose_idx, movement_idx=args.movement_idx, out=args.out)


if __name__ == "__main__":
    main()
