"""Pairwise qualitative figure: task-4 pose + task-10 motion (8 panels in one row)."""
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

from paper_figures._media import label_box, repo  # noqa: E402
from pilot90_experiment_suite import MOTION_PAIRWISE_DIR, PAIRWISE_IMG_DIR  # noqa: E402


def _font(size: int = 9) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _split_pair(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    w, h = img.size
    mid = w // 2
    return img.crop((0, 0, mid, h)), img.crop((mid, 0, w, h))


def _load_pose_pairwise_entries(json_path: Path) -> list[dict]:
    if not json_path.is_file():
        return []
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return list(data.get("comparisons") or [])


def _pose_panels(entry: dict) -> tuple[Image.Image, Image.Image, str, bool, bool]:
    img_path = Path(entry.get("pair_image") or "")
    if not img_path.is_file():
        img_path = PAIRWISE_IMG_DIR / img_path.name
    if not img_path.is_file():
        raise FileNotFoundError(img_path)
    pair = Image.open(img_path).convert("RGB")
    left, right = _split_pair(pair)
    gt_side = str(entry.get("gt_side", "left")).lower()
    left_ok = gt_side == "left"
    right_ok = not left_ok
    left = label_box(left, "True" if left_ok else "False", color=(34, 139, 34, 160) if left_ok else (200, 40, 40, 160))
    right = label_box(right, "True" if right_ok else "False", color=(34, 139, 34, 160) if right_ok else (200, 40, 40, 160))
    return left, right, str(entry.get("cue", "?")), left_ok, right_ok


def _motion_panels_from_spec(spec: dict, cfg_by: dict) -> tuple[Image.Image, Image.Image, str, bool, bool]:
    from paper_figures._media import alpha_trajectory_for_row

    cue = str(spec["cue"])
    row = cfg_by.get(int(spec["idx"]))
    if row is None:
        raise KeyError(cue)
    cache = repo() / "data/results/paper_figures/cache/pairwise_motion"
    gt_side = str(spec.get("gt_side", "left")).lower()
    left_ok = gt_side == "left"
    right_ok = not left_ok

    # Both sides use generation config; labels indicate GT vs distractor side
    left_img = alpha_trajectory_for_row(row, cache / f"c{spec['idx']}_left.png")
    right_img = alpha_trajectory_for_row(row, cache / f"c{spec['idx']}_right.png")
    if left_img is None or right_img is None:
        ph = Image.new("RGB", (180, 140), (230, 230, 230))
        left = right = ph
    else:
        left = Image.open(left_img).convert("RGB")
        right = Image.open(right_img).convert("RGB")
    left = label_box(left, "True" if left_ok else "False", color=(34, 139, 34, 160) if left_ok else (200, 40, 40, 160))
    right = label_box(right, "True" if right_ok else "False", color=(34, 139, 34, 160) if right_ok else (200, 40, 40, 160))
    return left, right, cue, left_ok, right_ok


def _footer(img: Image.Image, cue: str, w: int = 160, h: int = 24) -> Image.Image:
    canvas = Image.new("RGB", (w, img.height + h), (255, 255, 255))
    canvas.paste(img, ((w - img.width) // 2, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((2, img.height + 4), cue.replace("_", " "), fill=(0, 0, 0), font=_font(8))
    return canvas


def parse_selection(idx_arg: str, n_pose: int, n_motion: int) -> tuple[list[int], list[int]]:
    if idx_arg.strip().lower() == "all":
        return [0, 1][:n_pose], [0, 1][:n_motion]
    parts = [p.strip() for p in idx_arg.replace("[", "").replace("]", "").split(",") if p.strip()]
    if len(parts) == 4:
        return [int(parts[0]), int(parts[1])], [int(parts[2]), int(parts[3])]
    if len(parts) == 8:
        # legacy: 4 pose slot indices + 4 motion slot indices (pairs duplicated)
        return [int(parts[0]), int(parts[2])], [int(parts[4]), int(parts[6])]
    raise SystemExit('Use --idx "0,1,0,1" (2 pose-pair + 2 motion-pair indices) or "all"')


def build(
    *,
    idx_arg: str = "0,1,0,1",
    pose_json: str | None = None,
    out: str | None = None,
) -> Path:
    pose_json_path = Path(pose_json) if pose_json else _REPO / "data/results/verify/pilot90_gemini/exp04_pose_pairwise_2way.json"
    if not pose_json_path.is_file():
        pose_json_path = _REPO / "data/results/verify/pilot40_pose_pairwise_12_gemini.json"

    pose_entries = _load_pose_pairwise_entries(pose_json_path)
    motion_specs = json.loads((MOTION_PAIRWISE_DIR / "pairwise_specs_pilot90.json").read_text(encoding="utf-8")).get("mp4", [])

    pose_sel, motion_sel = parse_selection(idx_arg, len(pose_entries), len(motion_specs))

    from paper_figures._media import cfg_by_idx

    cfg_by = cfg_by_idx()
    panels: list[Image.Image] = []

    for pi in pose_sel:
        if pi >= len(pose_entries):
            continue
        left, right, cue, _, _ = _pose_panels(pose_entries[pi])
        panels.append(_footer(left, cue))
        panels.append(_footer(right, cue))

    for mi in motion_sel:
        if mi >= len(motion_specs):
            continue
        left, right, cue, _, _ = _motion_panels_from_spec(motion_specs[mi], cfg_by)
        panels.append(_footer(left, cue))
        panels.append(_footer(right, cue))

    if not panels:
        raise SystemExit("No panels rendered")

    ph = max(p.height for p in panels)
    pw = max(p.width for p in panels)
    norm = [Image.new("RGB", (pw, ph), (255, 255, 255)) for _ in panels]
    for i, p in enumerate(panels):
        norm[i].paste(p, ((pw - p.width) // 2, 0))

    row = Image.new("RGB", (pw * len(norm), ph), (255, 255, 255))
    for i, p in enumerate(norm):
        row.paste(p, (i * pw, 0))

    od = repo() / "data/results/paper_figures"
    od.mkdir(parents=True, exist_ok=True)
    out_path = Path(out) if out else od / "qual_pairwise_8panel.png"
    row.save(out_path)
    cap = od / "qual_pairwise_8panel_caption.txt"
    cap.write_text(
        "Pairwise comparison examples: pose (tasks 1–4) and motion (task 10). "
        "Green/red semi-transparent labels mark True/False sides.\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_path} ({len(norm)} panels)")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="8-panel pairwise qualitative figure")
    p.add_argument("--idx", default="0,1,0,1", help='4 indices: pose_i1,pose_i2,motion_i1,motion_i2 or "all"')
    p.add_argument("--pose-json", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    build(idx_arg=args.idx, pose_json=args.pose_json, out=args.out)


if __name__ == "__main__":
    main()
