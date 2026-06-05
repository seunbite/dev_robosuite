#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from motion_generation import MotionGenerator  # noqa: E402


ORDER_DIR = ["front", "back", "left", "right", "up", "down"]
ORDER_GRIP = ["horizontal", "vertical"]
AXES = ("x_pct", "y_pct", "z_pct")
BIN_NAMES = ("low", "mid", "high")


def _tertile_bin(v: float) -> str:
    if v < 33.3333:
        return "low"
    if v < 66.6667:
        return "mid"
    return "high"


def _select_xyz_tertile_balanced(rows: list[dict], n: int = 9) -> list[dict]:
    """
    Pick n poses so each axis (x/y/z) is as evenly distributed across low/mid/high as possible.
    For n=9, ideal target is 3 per bin on each axis.
    """
    if len(rows) <= n:
        return rows

    target = n // 3
    counts = {a: {b: 0 for b in BIN_NAMES} for a in AXES}
    selected: list[dict] = []
    used_pose_ids: set[int] = set()

    # Pre-sort for stable tie-breaking: prefer lower orientation diff.
    candidates = sorted(rows, key=lambda r: float(r.get("orientation_diff_deg", 999)))

    def score(r: dict) -> tuple[float, float]:
        # Main score: how much this row helps fill axis-bin deficits.
        benefit = 0.0
        for axis in AXES:
            b = _tertile_bin(float(r.get(axis, 50)))
            deficit = max(0, target - counts[axis][b])
            benefit += deficit
        # Tie-break: smaller orientation diff is better (negative for max score sort).
        tie = -float(r.get("orientation_diff_deg", 999))
        return benefit, tie

    while len(selected) < n:
        best = None
        best_score = (-1.0, -9999.0)
        for r in candidates:
            pid = int(r.get("pose_id", -1))
            if pid in used_pose_ids:
                continue
            s = score(r)
            if s > best_score:
                best_score = s
                best = r
        if best is None:
            break
        selected.append(best)
        used_pose_ids.add(int(best.get("pose_id", -1)))
        for axis in AXES:
            b = _tertile_bin(float(best.get(axis, 50)))
            counts[axis][b] += 1

    if len(selected) < n:
        for r in candidates:
            pid = int(r.get("pose_id", -1))
            if pid in used_pose_ids:
                continue
            selected.append(r)
            used_pose_ids.add(pid)
            if len(selected) >= n:
                break
    return selected[:n]


def _load_entries(jsonl_path: Path) -> list[dict]:
    out: list[dict] = []
    for ln in jsonl_path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            out.append(json.loads(ln))
    return out


def _make_tile_for_group(
    mg: MotionGenerator,
    group_rows: list[dict],
    title: str,
    out_path: Path,
) -> None:
    reps = _select_xyz_tertile_balanced(group_rows, n=9)
    imgs: list[Image.Image] = []
    for r in reps:
        joint_pos = mg._pose_data_to_joint_positions(r)
        mg._set_joint_positions(joint_pos)
        arr = mg._capture_image()
        im = Image.fromarray(arr).convert("RGB")
        draw = ImageDraw.Draw(im)
        label = (
            f"p{r.get('pose_id','?')} "
            f"x{int(round(float(r.get('x_pct', 50))))} "
            f"y{int(round(float(r.get('y_pct', 50))))} "
            f"z{int(round(float(r.get('z_pct', 50))))}"
        )
        draw.rectangle([0, 0, 280, 30], fill=(0, 0, 0))
        draw.text((8, 6), label, fill="white")
        imgs.append(im)
        mg._set_joint_positions(mg.initial_joint_pos)

    if not imgs:
        return

    cw, ch = imgs[0].size
    cols = 3
    rows = math.ceil(len(imgs) / cols)
    pad = 6
    header = 58
    canvas = Image.new("RGB", (pad + cols * (cw + pad), header + pad + rows * (ch + pad)), (250, 250, 250))
    dr = ImageDraw.Draw(canvas)
    try:
        ft = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except Exception:
        ft = ImageFont.load_default()
    dr.text((10, 10), title, fill=(20, 20, 20), font=ft)

    for i, im in enumerate(imgs):
        rr = i // cols
        cc = i % cols
        x = pad + cc * (cw + pad)
        y = header + pad + rr * (ch + pad)
        canvas.paste(im, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jsonl-path",
        type=Path,
        default=Path("data/seed/_remainder/closest_poses_results.jsonl"),
    )
    ap.add_argument(
        "--robots",
        type=str,
        default="IIWA",
        help="Comma-separated robot list, or 'all'.",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/results/visualize/pose_groups_12"),
    )
    args = ap.parse_args()

    entries = _load_entries(args.jsonl_path)
    all_robots = sorted({e.get("robot") for e in entries if e.get("robot")})
    if args.robots.strip().lower() == "all":
        robots = all_robots
    else:
        robots = [x.strip() for x in args.robots.split(",") if x.strip()]

    for robot in robots:
        robot_entries = [e for e in entries if e.get("robot") == robot]
        if not robot_entries:
            print(f"[skip] {robot}: no entries")
            continue
        buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for e in robot_entries:
            key = (e.get("dir"), e.get("gripper_orientation"))
            buckets[key].append(e)

        print(f"[robot] {robot}")
        mg = MotionGenerator(
            robot_name=robot,
            env_name="EmptySpace",
            controller_name="IK_POSE",
            jsonl_path=str(args.jsonl_path),
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_distance=1.8,
        )
        for d in ORDER_DIR:
            for g in ORDER_GRIP:
                key = (d, g)
                if key not in buckets or not buckets[key]:
                    continue
                out_dir = args.output_root if len(robots) == 1 else (args.output_root / robot)
                out_path = out_dir / f"group_{d}_{g}.png"
                title = f"{robot} | {d}+{g} | xyz-tertile-balanced 9"
                _make_tile_for_group(mg, buckets[key], title, out_path)
                print(f"  wrote {out_path}")
        mg.close()


if __name__ == "__main__":
    main()
