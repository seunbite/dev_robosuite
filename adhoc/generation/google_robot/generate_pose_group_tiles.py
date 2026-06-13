#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from legacy.render_mobile_config import (  # noqa: E402
    HEAD_REF_BODY,
    QI_R_ARM,
    R_EE_BODY,
    TORSO_REF_BODY,
    _capture,
    _default_qpos,
    _get_pose_bank_entries,
    _make_env,
)

ORDER_DIR = ["front", "back", "in", "out", "up", "down"]
ORDER_GRIP = ["horizontal", "vertical"]
AXES = ("x", "y", "z")
DEFAULT_TILES_PER_GROUP = 25
DEFAULT_GRID_COLS = 5
# hand_x - head_x may be up to 10cm behind head (was +20cm only)
DEFAULT_HEAD_HAND_GAP_MIN_M = -0.1
SOURCE_DIR_FOR_DISPLAY = {
    "front": "front",
    "back": "back",
    "in": "right",
    "out": "left",
    "up": "up",
    "down": "down",
}


def _bend_score(r: dict) -> float:
    jd = r.get("joint_deg")
    if not isinstance(jd, list) or len(jd) != 6:
        return 1e9
    return float(sum(abs(float(v)) for v in jd))


def _joint_key(r: dict) -> tuple[float, ...]:
    return tuple(float(x) for x in r.get("r_arm_rad", []))


def _select_xy_grid_balanced(rows: list[dict], *, grid_side: int = 5) -> list[dict]:
    """Pick grid_side² poses spread across (x,y) EE percent space; z breaks ties."""
    n = grid_side * grid_side
    if len(rows) <= n:
        return rows

    candidates = sorted(
        rows,
        key=lambda r: (
            int(round(float(r.get("x", 50)))),
            int(round(float(r.get("y", 50)))),
            int(round(float(r.get("z", 50)))),
            _bend_score(r),
        ),
    )
    used: set[tuple[float, ...]] = set()
    selected: list[dict] = []

    mins = {a: min(float(r.get(a, 50)) for r in candidates) for a in AXES}
    maxs = {a: max(float(r.get(a, 50)) for r in candidates) for a in AXES}
    z_mid = (mins["z"] + maxs["z"]) / 2.0

    def _ee_key(r: dict) -> tuple[int, int, int]:
        return (
            int(round(float(r.get("x", 50)))),
            int(round(float(r.get("y", 50)))),
            int(round(float(r.get("z", 50)))),
        )

    span = max(grid_side - 1, 1)
    targets: list[tuple[float, float]] = []
    for i in range(grid_side):
        for j in range(grid_side):
            tx = mins["x"] + (maxs["x"] - mins["x"]) * i / span
            ty = mins["y"] + (maxs["y"] - mins["y"]) * j / span
            targets.append((tx, ty))

    for tx, ty in targets:
        best = None
        best_d = float("inf")
        for r in candidates:
            key = _joint_key(r)
            if key in used:
                continue
            same_ee = [c for c in candidates if _ee_key(c) == _ee_key(r)]
            if same_ee and min(same_ee, key=_bend_score) is not r:
                continue
            x = float(r.get("x", 50))
            y = float(r.get("y", 50))
            z = float(r.get("z", 50))
            d = (x - tx) ** 2 + (y - ty) ** 2 + 0.15 * (z - z_mid) ** 2
            if d < best_d:
                best_d = d
                best = r
        if best is None:
            continue
        selected.append(best)
        used.add(_joint_key(best))

    if len(selected) < n:
        for r in candidates:
            key = _joint_key(r)
            if key in used:
                continue
            selected.append(r)
            used.add(key)
            if len(selected) >= n:
                break
    return selected[:n]


def _select_xyz_tertile_balanced(rows: list[dict], n: int = 9) -> list[dict]:
    """Legacy 9-pick: x/y/z each min, mid, max."""
    if n == 9:
        grid_side = 3
        if len(rows) > 9:
            return _select_xy_grid_balanced(rows, grid_side=grid_side)
    return _select_xy_grid_balanced(rows, grid_side=max(1, int(round(n**0.5))))


def _is_forward_of_torso_and_head(env, r: dict) -> bool:
    """Keep poses with hand_x - head_x >= MOBILE_POSE_HEAD_HAND_X_GAP (default -0.1 m)."""
    q = _default_qpos(env)
    q[QI_R_ARM] = [float(v) for v in r["r_arm_rad"]]
    env.sim.data.qpos[:] = q
    env.sim.forward()
    from legacy.render_mobile_config import _right_hand_forward_vs_torso_head_ok  # noqa: WPS433

    return _right_hand_forward_vs_torso_head_ok(env.sim)


def _make_group_tile(
    env,
    group_rows: list[dict],
    title: str,
    out_path: Path,
    *,
    n_tiles: int = DEFAULT_TILES_PER_GROUP,
    grid_cols: int = DEFAULT_GRID_COLS,
) -> None:
    grid_side = grid_cols
    reps = _select_xy_grid_balanced(group_rows, grid_side=grid_side)
    imgs: list[Image.Image] = []
    for r in reps:
        q = _default_qpos(env)
        q[QI_R_ARM] = [float(v) for v in r["r_arm_rad"]]
        im = _capture(env, q).convert("RGB")
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, 470, 52], fill=(0, 0, 0))
        label_xyz = f"x{int(round(float(r.get('x', 50))))} y{int(round(float(r.get('y', 50))))} z{int(round(float(r.get('z', 50))))}"
        jd = r.get("joint_deg")
        if isinstance(jd, list) and len(jd) == 6:
            label_joint = "j=[" + ",".join(str(int(round(float(v)))) for v in jd) + "]"
        else:
            label_joint = "j=[?]"
        dr.text((8, 6), label_xyz, fill="white")
        dr.text((8, 28), label_joint, fill="white")
        imgs.append(im)

    if not imgs:
        return

    cw, ch = imgs[0].size
    cols = grid_cols
    grid_rows = math.ceil(len(imgs) / cols)
    pad = 6
    header = 60
    canvas = Image.new("RGB", (pad + cols * (cw + pad), header + pad + grid_rows * (ch + pad)), (248, 248, 248))
    dr = ImageDraw.Draw(canvas)
    try:
        ft = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except Exception:
        ft = ImageFont.load_default()
    dr.text((10, 10), title, fill=(20, 20, 20), font=ft)

    for i, im in enumerate(imgs):
        rr = i // cols
        cc = i % cols
        x0 = pad + cc * (cw + pad)
        y0 = header + pad + rr * (ch + pad)
        canvas.paste(im, (x0, y0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _write_gallery_html(out_root: Path, *, n_tiles: int, grid_cols: int) -> None:
    cards = []
    for d in ORDER_DIR:
        for g in ORDER_GRIP:
            fname = f"group_{d}_{g}.png"
            if not (out_root / fname).is_file():
                continue
            cards.append(
                f'<article><h3>{fname.replace(".png", "")}</h3>'
                f'<img src="{fname}" loading="lazy"></article>'
            )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>google pose groups 12</title>
<style>body{{font-family:sans-serif;margin:0;background:#0f172a;color:#e2e8f0}}
.wrap{{padding:16px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:12px}}
article{{background:#111827;border:1px solid #334155;border-radius:10px;padding:10px}}
h3{{margin:0 0 8px;font-size:14px}}img{{width:100%;height:auto;border-radius:6px}}</style></head>
<body><div class="wrap"><h1>Google Robot Pose Groups (front/back/in/out/up/down × 2 orientation)</h1>
<p>Selection: {grid_cols}×{grid_cols} = {n_tiles} tiles per group, spread across (x,y) EE percent; z tie-break; hand-head x gap ≥ −10cm.</p>
<p>Generated groups: {len(cards)}</p>
<section class="grid">{''.join(cards)}</section></div></body></html>"""
    (out_root / "gallery.html").write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/results/visualize/google_pose_groups_12"),
    )
    ap.add_argument("--n-tiles", type=int, default=DEFAULT_TILES_PER_GROUP)
    ap.add_argument("--grid-cols", type=int, default=DEFAULT_GRID_COLS)
    ap.add_argument(
        "--head-hand-gap-min",
        type=float,
        default=DEFAULT_HEAD_HAND_GAP_MIN_M,
        help="Min hand_x - head_x in meters (default -0.1 = allow 10cm behind head)",
    )
    args = ap.parse_args()
    os.environ["MOBILE_POSE_HEAD_HAND_X_GAP"] = str(args.head_hand_gap_min)
    grid_cols = args.grid_cols
    if grid_cols * grid_cols != args.n_tiles:
        grid_cols = max(1, int(round(args.n_tiles**0.5)))

    env = _make_env()
    try:
        bank = _get_pose_bank_entries()
        buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for e in bank:
            src_dir = str(e.get("dir"))
            orient = str(e.get("orient"))
            disp_dir = None
            for k, v in SOURCE_DIR_FOR_DISPLAY.items():
                if v == src_dir:
                    disp_dir = k
                    break
            if disp_dir in ORDER_DIR and orient in ORDER_GRIP:
                if not _is_forward_of_torso_and_head(env, e):
                    continue
                buckets[(disp_dir, orient)].append(e)

        for d in ORDER_DIR:
            for g in ORDER_GRIP:
                rows = buckets.get((d, g), [])
                if not rows:
                    continue
                out = args.output_root / f"group_{d}_{g}.png"
                title = f"google_robot | {d}+{g} | xy-grid {grid_cols}x{grid_cols} ({args.n_tiles})"
                _make_group_tile(env, rows, title, out, n_tiles=args.n_tiles, grid_cols=grid_cols)
                print(f"wrote {out} ({min(len(rows), args.n_tiles)} tiles)")
        _write_gallery_html(args.output_root, n_tiles=args.n_tiles, grid_cols=grid_cols)
        print(f"wrote {args.output_root / 'gallery.html'}")
    finally:
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()


if __name__ == "__main__":
    main()
