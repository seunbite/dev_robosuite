"""Render helpers: pose tiles, alpha trajectories, GIF lookup."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
_ROBOTARM = _HERE.parent
_REPO = _ROBOTARM.parents[2]
for p in (_REPO, _ROBOTARM, _REPO / "adhoc" / "vlm_test"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from verify_pose_tiles_gemini import (  # noqa: E402
    _crop_tile_from_group,
    _first_pose,
    _load_tile_pick,
)

ROBOT = "IIWA"
HZ = 10
POSE_CFG = (
    _REPO
    / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json"
)
TILE_DIR = _REPO / "data/results/visualize/pose_groups_12"
TILE_PICK = _REPO / "data/results/verify/pose_tile_pick_by_group.json"
CACHE = _REPO / "data/results/paper_figures/cache"


def repo() -> Path:
    return _REPO


def load_pose_cfg() -> list[dict[str, Any]]:
    return json.loads(POSE_CFG.read_text(encoding="utf-8"))


def cfg_by_idx() -> dict[int, dict[str, Any]]:
    return {int(r["idx"]): r for r in load_pose_cfg()}


def manifest90_rows() -> list[dict[str, Any]]:
    from pilot90_experiment_suite import manifest90_rows_from_cfg

    return manifest90_rows_from_cfg(load_pose_cfg())


def parse_idx_arg(idx_arg: str, rows: list[dict[str, Any]]) -> list[int]:
    if idx_arg.strip().lower() == "all":
        return [int(r["idx"]) for r in sorted(rows, key=lambda r: int(r["idx"]))]
    out: list[int] = []
    for part in idx_arg.replace("[", "").replace("]", "").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def pick_gif(cue: str) -> Path | None:
    from motion_media_paths import pick_latest_gif

    return pick_latest_gif(_REPO, cue, pilot90=True)


def pose_png_for_row(row: dict[str, Any], out_png: Path) -> Path | None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_png.is_file():
        return out_png
    pose = _first_pose(row)
    d, g = pose.get("dir"), pose.get("gripper_orientation")
    if not d or not g:
        return None
    tile_pick = _load_tile_pick(TILE_PICK)
    idx = tile_pick.get((d, g))
    if idx is None:
        return None
    group_path = TILE_DIR / f"group_{d}_{g}.png"
    if not group_path.is_file():
        return None
    cell = _crop_tile_from_group(Image.open(group_path).convert("RGB"), idx)
    cell.save(out_png)
    return out_png


def alpha_trajectory_for_row(row: dict[str, Any], out_png: Path, *, force: bool = False) -> Path | None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_png.is_file() and not force:
        return out_png
    gif = pick_gif(str(row["cue"]))
    if not gif:
        return None
    try:
        import testset_utils

        pose = _first_pose(row)
        pose_id = int(pose.get("pose_id") or 0)
        sample = {
            "sample_id": testset_utils._safe_name(f"paper_{row['idx']}_{row['cue']}"),
            "testset": "iconic",
            "cue_idx": int(row["idx"]),
            "cue": row["cue"],
            "config_path": str(POSE_CFG),
            "gif_path": str(gif),
            "selected_pose_id": pose_id,
            "meta": {},
        }
        img, _ = testset_utils.build_tile_figure_sim_trajectory_panel(
            sample, ROBOT, HZ, canonical="alpha_frame_trajectory", force=force
        )
        img.save(out_png, format="PNG", optimize=False)
        return out_png
    except Exception as e:
        print(f"[alpha fail] {row.get('cue')}: {e}", flush=True)
        return None


def mid_frame_from_gif(gif: Path, out_png: Path) -> Path | None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_png.is_file():
        return out_png
    try:
        im = Image.open(gif)
        n = getattr(im, "n_frames", 1)
        im.seek(n // 2)
        im.convert("RGB").save(out_png)
        return out_png
    except Exception:
        return None


def _font(size: int = 10) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def label_box(
    img: Image.Image,
    text: str,
    *,
    color: tuple[int, int, int, int],
    corner: str = "top_right",
) -> Image.Image:
    """Overlay semi-transparent True/False label."""
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font(11)
    pad = 4
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w, h = base.size
    if corner == "top_right":
        x0, y0 = w - tw - 2 * pad - 4, 4
    else:
        x0, y0 = 4, 4
    draw.rectangle([x0 - pad, y0 - pad, x0 + tw + pad, y0 + th + pad], fill=color)
    draw.text((x0, y0), text, fill=(255, 255, 255, 255), font=font)
    return Image.alpha_composite(base, overlay).convert("RGB")


def gif_to_mp4(gif: Path, mp4: Path) -> bool:
    ff = shutil.which("ffmpeg")
    if not ff:
        return False
    mp4.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(gif), "-pix_fmt", "yuv420p", str(mp4)],
        capture_output=True,
    )
    return r.returncode == 0 and mp4.is_file()
