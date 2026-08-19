#!/usr/bin/env python3
"""Remap Google xyz to linear min→0…max→100, then build zigzag reject + remain HTML.

Mirrors data/results/html/g1_zigzag_rejects.html:
  - per-category REJECT / REMAIN sheets
  - yellow tile numbers + x/y/z labels
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE, _HERE / "legacy"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from generate_pose_group_tiles import SOURCE_DIR_FOR_DISPLAY  # noqa: E402
from legacy.render_mobile_config import (  # noqa: E402
    QI_R_ARM,
    R_ELBOW_BODY,
    R_EE_BODY,
    _capture,
    _default_qpos,
    _make_env,
)
from legacy.render_mobile_manip_schema import (  # noqa: E402
    _right_fingertip_pad_mid_world_xyz,
)

try:
    from pose_palm_arrow import overlay_google_palm_arrow  # noqa: E402
except Exception:  # pragma: no cover
    overlay_google_palm_arrow = None  # type: ignore

BANK_PATH = _REPO / "data/seed/tiago/mobile_pose_bank_729.json"
PLAUSIBLE_PATH = _REPO / "data/seed/tiago/plausible_poses_metadata.jsonl"
OUT_REJ = _REPO / "data/results/visualize/google_zigzag_rejects"
OUT_REM = _REPO / "data/results/visualize/google_remain_after_zigzag"
OUT_HTML = _REPO / "data/results/html/google_zigzag_rejects.html"

TURN_ELBOW_MIN_DEG = 45.0
TURN_WRIST_MIN_DEG = 60.0
TILE, COLS = 240, 5
ORDER_DIR = ["front", "back", "out", "in", "up", "down"]
ORDER_GRIP = ["horizontal", "vertical"]
R_SHOULDER_BODY = "robot0_arm_right_1_link"
R_WRIST_BODY = "robot0_arm_right_6_link"

# Manual keep: zigzag metric may fire, but keep these in the remain / feasible pool.
# (was REJECT · in|horizontal tile #2)
FORCE_REMAIN_JOINT_DEG: set[tuple[int, ...]] = {
    (90, 0, 0, 90, 0, -90),
}

# bank source dir → display dir (inverse of SOURCE_DIR_FOR_DISPLAY)
_BANK_DIR_TO_DISPLAY = {v: k for k, v in SOURCE_DIR_FOR_DISPLAY.items()}


def _old_piecewise_pct_to_world(pct: float, lo: float, hi: float, center: float) -> float:
    """Inverse of the pre-linear body-centered encoding (for one-shot bank remap)."""
    p = float(np.clip(float(pct), 0.0, 100.0))
    if p >= 50.0:
        span = float(hi - center)
        if span < 1e-12:
            return float(center)
        return float(center) + (p - 50.0) / 50.0 * span
    span = float(center - lo)
    if span < 1e-12:
        return float(center)
    return float(center) - (50.0 - p) / 50.0 * span


def _linear_world_to_pct(coord: float, lo: float, hi: float) -> float:
    span = float(hi - lo)
    if span < 1e-12:
        return 50.0
    return float(np.clip((float(coord) - float(lo)) / span * 100.0, 0.0, 100.0))


def _display_cat(row: dict[str, Any]) -> str:
    d = _BANK_DIR_TO_DISPLAY.get(str(row.get("dir", "")), str(row.get("dir", "")))
    o = str(row.get("orient", row.get("gripper_orientation", "")))
    return f"{d}|{o}"


def _xyz_int_key(r: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(round(float(r.get("x", 50)))),
        int(round(float(r.get("y", 50)))),
        int(round(float(r.get("z", 50)))),
    )


def _ang_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.degrees(
            np.arccos(
                np.clip(
                    float(np.dot(a, b))
                    / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)) + 1e-12),
                    -1.0,
                    1.0,
                )
            )
        )
    )


def remap_bank_xyz_linear(payload: dict[str, Any]) -> dict[str, Any]:
    meta = payload.get("meta") or {}
    wb = meta["world_pad_mid_bounds_m"]
    wc = meta.get("world_body_center_m") or {}
    lo = np.array([float(wb["x"][0]), float(wb["y"][0]), float(wb["z"][0])], dtype=float)
    hi = np.array([float(wb["x"][1]), float(wb["y"][1]), float(wb["z"][1])], dtype=float)
    center = np.array(
        [
            float(wc.get("x", 0.5 * (lo[0] + hi[0]))),
            float(wc.get("y", 0.5 * (lo[1] + hi[1]))),
            float(wc.get("z", 0.5 * (lo[2] + hi[2]))),
        ],
        dtype=float,
    )
    if meta.get("percentile_anchor") == "linear_extrema":
        print("bank already linear_extrema — skip remap", flush=True)
        return payload

    for r in payload["poses"]:
        old = np.array([float(r["x"]), float(r["y"]), float(r["z"])], dtype=float)
        world = np.array(
            [
                _old_piecewise_pct_to_world(old[0], lo[0], hi[0], center[0]),
                _old_piecewise_pct_to_world(old[1], lo[1], hi[1], center[1]),
                _old_piecewise_pct_to_world(old[2], lo[2], hi[2], center[2]),
            ],
            dtype=float,
        )
        r["x"] = round(_linear_world_to_pct(world[0], lo[0], hi[0]), 3)
        r["y"] = round(_linear_world_to_pct(world[1], lo[1], hi[1]), 3)
        r["z"] = round(_linear_world_to_pct(world[2], lo[2], hi[2]), 3)
        r["xyz_world_pad_mid_m"] = [float(world[0]), float(world[1]), float(world[2])]

    meta["percentile_anchor"] = "linear_extrema"
    meta["note"] = (
        "x,y,z are 0–100% linear over pad-mid extrema (min→0, max→100); "
        "same uniform scale as G1 feasible bank. Body center kept in meta for reference only."
    )
    notes = meta.setdefault("notes", [])
    if isinstance(notes, list):
        notes.append(
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} "
            "remapped body_center piecewise → linear_extrema"
        )
    payload["meta"] = meta
    return payload


def sync_plausible_xyz(bank_poses: list[dict[str, Any]]) -> int:
    by_jd = {tuple(int(x) for x in p["joint_deg"]): p for p in bank_poses}
    rows = [
        json.loads(l)
        for l in PLAUSIBLE_PATH.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    n = 0
    for e in rows:
        key = tuple(int(x) for x in e["joint_deg"])
        b = by_jd.get(key)
        if not b:
            continue
        e["x"], e["y"], e["z"] = b["x"], b["y"], b["z"]
        n += 1
    PLAUSIBLE_PATH.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in rows),
        encoding="utf-8",
    )
    return n


def zigzag_metrics(env, r: dict[str, Any]) -> dict[str, Any]:
    q = _default_qpos(env)
    q[QI_R_ARM] = [float(v) for v in r["r_arm_rad"]]
    env.sim.data.qpos[:] = q
    env.sim.forward()
    sim = env.sim
    sh = np.asarray(sim.data.body_xpos[sim.model.body_name2id(R_SHOULDER_BODY)], float)
    el = np.asarray(sim.data.body_xpos[sim.model.body_name2id(R_ELBOW_BODY)], float)
    wr = np.asarray(sim.data.body_xpos[sim.model.body_name2id(R_WRIST_BODY)], float)
    palm = _right_fingertip_pad_mid_world_xyz(sim)
    if palm is None:
        palm = np.asarray(sim.data.body_xpos[sim.model.body_name2id(R_EE_BODY)], float)
    u, v, w = el - sh, wr - el, palm - wr
    turn_el = _ang_deg(u, v)
    turn_wr = _ang_deg(v, w)
    n1 = np.cross(u, v)
    n2 = np.cross(v, w)
    opp = bool(float(np.dot(n1, n2)) < 0.0)
    is_zig = turn_el >= TURN_ELBOW_MIN_DEG and turn_wr >= TURN_WRIST_MIN_DEG and opp
    return {
        "turn_elbow_deg": round(turn_el, 2),
        "turn_wrist_deg": round(turn_wr, 2),
        "opposite_sense": opp,
        "zigzag": is_zig,
    }


def _fonts():
    try:
        ft = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        ft_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        ft_h = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        ft_num = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except Exception:
        ft = ft_sm = ft_h = ft_num = ImageFont.load_default()
    return ft, ft_sm, ft_h, ft_num


def draw_num(im: Image.Image, n: int, ft_num) -> None:
    dr = ImageDraw.Draw(im)
    lab = str(n)
    try:
        bb = dr.textbbox((0, 0), lab, font=ft_num)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        tw, th = 12, 16
    x1 = im.width - 5
    x0 = x1 - tw - 10
    dr.rectangle((x0, 4, x1, th + 12), fill=(0, 0, 0))
    dr.text((x0 + 5, 6), lab, fill=(255, 255, 80), font=ft_num)


def render_sheet(
    env,
    items: list[dict[str, Any]],
    title: str,
    path: Path,
    *,
    reject: bool,
    fonts,
) -> None:
    if not items:
        Image.new("RGB", (400, 80), (240, 240, 240)).save(path)
        return
    ft, ft_sm, ft_h, ft_num = fonts
    imgs: list[Image.Image] = []
    for idx, r in enumerate(items, 1):
        q = _default_qpos(env)
        q[QI_R_ARM] = [float(v) for v in r["r_arm_rad"]]
        env.sim.data.qpos[:] = q
        env.sim.forward()
        if overlay_google_palm_arrow is not None:
            try:
                overlay_google_palm_arrow(env)
            except Exception:
                pass
        im = _capture(env, q).convert("RGB").resize((TILE, TILE), Image.Resampling.BILINEAR)
        dr = ImageDraw.Draw(im)
        dr.rectangle((0, 0, im.width, 40), fill=(0, 0, 0))
        xi, yi, zi = _xyz_int_key(r)
        jd = r.get("joint_deg")
        jd_s = ",".join(str(int(x)) for x in jd) if isinstance(jd, list) else "?"
        dr.text((4, 2), f"[{jd_s}]  x{xi} y{yi} z{zi}", fill=(255, 255, 255), font=ft)
        if reject:
            m = r.get("zigzag_metrics") or {}
            dr.text(
                (4, 22),
                f"el={m.get('turn_elbow_deg', 0):.0f}° wr={m.get('turn_wrist_deg', 0):.0f}° opp",
                fill=(255, 170, 170),
                font=ft_sm,
            )
        else:
            dr.text((4, 22), f"dir={r.get('dir')} · {r.get('orient')}", fill=(200, 200, 200), font=ft_sm)
        draw_num(im, idx, ft_num)
        imgs.append(im)

    pad, header = 5, 48
    cw, ch = imgs[0].size
    rn = math.ceil(len(imgs) / COLS)
    canvas = Image.new(
        "RGB",
        (pad + COLS * (cw + pad), header + pad + rn * (ch + pad)),
        (240, 240, 240),
    )
    dr = ImageDraw.Draw(canvas)
    dr.text((10, 12), f"{title} · n={len(imgs)} · 5-col", fill=(20, 20, 20), font=ft_h)
    for i, im in enumerate(imgs):
        canvas.paste(
            im,
            (pad + (i % COLS) * (cw + pad), header + pad + (i // COLS) * (ch + pad)),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92, subsampling=0)
    print("wrote", path, "n=", len(imgs), flush=True)


def write_html(
    *,
    n_pool: int,
    n_rej: int,
    n_rem: int,
    rej_by: dict[str, list],
    rem_by: dict[str, list],
) -> None:
    def slug(cat: str) -> str:
        return cat.replace("|", "-")

    rej_links = []
    rem_links = []
    for d in ORDER_DIR:
        for g in ORDER_GRIP:
            cat = f"{d}|{g}"
            short = f"{d[0]}{g[0]}"
            rej_links.append(f"<a href='#zigzag-{slug(cat)}'>z:{short}</a>")
            rem_links.append(f"<a href='#remain-{slug(cat)}'>r:{short}</a>")

    rej_articles = [
        f"<article><h3>ALL zigzag rejects (n={n_rej})</h3>"
        f"<img src='../visualize/google_zigzag_rejects/zigzag_all.jpg' loading='lazy'></article>"
    ]
    for d in ORDER_DIR:
        for g in ORDER_GRIP:
            cat = f"{d}|{g}"
            n = len(rej_by.get(cat, []))
            rej_articles.append(
                f"<article id='zigzag-{slug(cat)}'><h3>REJECT · {cat} · n={n}</h3>"
                f"<img src='../visualize/google_zigzag_rejects/zigzag_{d}_{g}.jpg' loading='lazy'></article>"
            )
    rem_articles = []
    for d in ORDER_DIR:
        for g in ORDER_GRIP:
            cat = f"{d}|{g}"
            n = len(rem_by.get(cat, []))
            rem_articles.append(
                f"<article id='remain-{slug(cat)}'><h3>REMAIN · {cat} · n={n}</h3>"
                f"<img src='../visualize/google_remain_after_zigzag/remain_{d}_{g}.jpg' loading='lazy'></article>"
            )

    page = f"""<!DOCTYPE html><html><head><meta charset=utf-8>
<title>Tiago zigzag rejects + remain pool</title>
<style>
body{{margin:0;font-family:system-ui;background:#0c0e14;color:#e6eaf2}}
header{{padding:14px 18px;border-bottom:1px solid #2a3140;position:sticky;top:0;background:#0c0e14;z-index:5}}
nav a{{color:#7ab4f8;margin-right:8px;font-size:.75rem}}
main{{padding:16px 18px 60px}}
.grid-1col{{display:grid;grid-template-columns:1fr;gap:16px;max-width:1100px}}
article{{background:#12161f;border:1px solid #242a36;border-radius:10px;padding:10px}}
img{{width:100%;border-radius:6px;background:#000}}
.note{{color:#9aa3b5;font-size:.88rem;max-width:1050px}}
h2{{border-top:1px solid #2a3140;padding-top:18px;margin-top:28px}}
code{{color:#9bc6ff}}
</style></head><body>
<header>
<h1>Tiago · zigzag rejects + remaining pool</h1>
<p class="note">xyz: <b>linear</b> pad-mid min→0 … max→100 (G1-style). Soft reject: opposite-sense Z-fold
(elbow≥{TURN_ELBOW_MIN_DEG:.0f}° &amp; wrist≥{TURN_WRIST_MIN_DEG:.0f}° &amp; opp).<br>
Pool <b>{n_pool}</b> · remain <b>{n_rem}</b> (zigzag rejects <b>{n_rej}</b>). Yellow # + x/y/z on every tile.</p>
<nav><b>rejects</b> {' '.join(rej_links)}<br><b>remain</b> {' '.join(rem_links)}</nav>
</header>
<main>
<section id="rejects">
<h2>1. Zigzag / Z-fold rejects ({n_rej})</h2>
<div class="grid-1col">{''.join(rej_articles)}</div>
</section>
<section id="remain">
<h2>2. Remaining pool for feasible pick ({n_rem})</h2>
<p class="note">Category-full · 5-col. Display dirs: bank left→out, right→in. Tell me e.g. <code>front|horizontal: 3,7,12</code>.</p>
<div class="grid-1col">{''.join(rem_articles)}</div>
</section>
</main></body></html>
"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(page, encoding="utf-8")
    print("wrote", OUT_HTML, flush=True)


def main() -> None:
    payload = json.loads(BANK_PATH.read_text(encoding="utf-8"))
    payload = remap_bank_xyz_linear(payload)
    BANK_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    n_sync = sync_plausible_xyz(payload["poses"])
    print(f"synced plausible xyz for {n_sync} rows", flush=True)

    # hist check
    xs = [float(r["x"]) for r in payload["poses"]]
    ys = [float(r["y"]) for r in payload["poses"]]
    zs = [float(r["z"]) for r in payload["poses"]]
    for name, a in ("x", xs), ("y", ys), ("z", zs):
        arr = np.asarray(a)
        hist, _ = np.histogram(arr, bins=np.linspace(0, 100, 11))
        print(f"linear {name}: [{arr.min():.1f},{arr.max():.1f}] med={np.median(arr):.1f} hist10={hist.tolist()}", flush=True)

    # clear render caches
    import legacy.render_mobile_config as rmc

    rmc._POSE_BANK_ENTRIES = None
    rmc._POSE_BANK_BOUNDS = None
    rmc._PLAUSIBLE_POSES = None

    env = _make_env()
    fonts = _fonts()
    try:
        zig_rows: list[dict[str, Any]] = []
        rem_rows: list[dict[str, Any]] = []
        for i, r in enumerate(payload["poses"]):
            m = zigzag_metrics(env, r)
            r["zigzag_metrics"] = m
            jd_key = tuple(int(x) for x in r["joint_deg"])
            is_zig = bool(m["zigzag"]) and jd_key not in FORCE_REMAIN_JOINT_DEG
            if bool(m["zigzag"]) and jd_key in FORCE_REMAIN_JOINT_DEG:
                m["force_remain"] = True
            r["zigzag_filter_fail"] = is_zig
            if is_zig:
                zig_rows.append(r)
            else:
                rem_rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"zigzag scored {i+1}/{len(payload['poses'])}", flush=True)
        print(
            f"pool {len(payload['poses'])}  zigzag rejects {len(zig_rows)}  remain {len(rem_rows)}",
            flush=True,
        )
        print("reject by cat", Counter(_display_cat(r) for r in zig_rows), flush=True)

        BANK_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        rej_by: dict[str, list] = defaultdict(list)
        rem_by: dict[str, list] = defaultdict(list)
        for r in zig_rows:
            rej_by[_display_cat(r)].append(r)
        for r in rem_rows:
            rem_by[_display_cat(r)].append(r)

        OUT_REJ.mkdir(parents=True, exist_ok=True)
        OUT_REM.mkdir(parents=True, exist_ok=True)

        # all rejects
        all_sorted = sorted(zig_rows, key=lambda r: (_display_cat(r), _xyz_int_key(r), tuple(r["joint_deg"])))
        render_sheet(env, all_sorted, "Google REJECT ALL zigzag", OUT_REJ / "zigzag_all.jpg", reject=True, fonts=fonts)

        picks = {"n_pool": len(payload["poses"]), "n_reject": len(zig_rows), "n_remain": len(rem_rows), "categories": {}}
        for d in ORDER_DIR:
            for g in ORDER_GRIP:
                cat = f"{d}|{g}"
                zr = sorted(rej_by.get(cat, []), key=lambda r: (_xyz_int_key(r), tuple(r["joint_deg"])))
                rr = sorted(rem_by.get(cat, []), key=lambda r: (_xyz_int_key(r), tuple(r["joint_deg"])))
                render_sheet(
                    env, zr, f"Google REJECT {cat}", OUT_REJ / f"zigzag_{d}_{g}.jpg", reject=True, fonts=fonts
                )
                render_sheet(
                    env, rr, f"Google REMAIN {cat}", OUT_REM / f"remain_{d}_{g}.jpg", reject=False, fonts=fonts
                )
                picks["categories"][cat] = {
                    "reject": [
                        {"tile": i + 1, "joint_deg": r["joint_deg"], "xyz_int": list(_xyz_int_key(r)), "dir": r["dir"], "orient": r["orient"]}
                        for i, r in enumerate(zr)
                    ],
                    "remain": [
                        {"tile": i + 1, "joint_deg": r["joint_deg"], "xyz_int": list(_xyz_int_key(r)), "dir": r["dir"], "orient": r["orient"]}
                        for i, r in enumerate(rr)
                    ],
                }

        (OUT_REM / "picks.json").write_text(json.dumps(picks, indent=2, ensure_ascii=False) + "\n")
        (OUT_REJ / "zigzag_rejects.json").write_text(
            json.dumps(
                {
                    "rule": {
                        "rule": "opposite_sense_Z_fold",
                        "turn_elbow_min_deg": TURN_ELBOW_MIN_DEG,
                        "turn_wrist_min_deg": TURN_WRIST_MIN_DEG,
                    },
                    "n_pool": len(payload["poses"]),
                    "n_reject": len(zig_rows),
                    "n_remain": len(rem_rows),
                    "joint_deg": [r["joint_deg"] for r in zig_rows],
                },
                indent=2,
            )
            + "\n"
        )
        write_html(
            n_pool=len(payload["poses"]),
            n_rej=len(zig_rows),
            n_rem=len(rem_rows),
            rej_by=rej_by,
            rem_by=rem_by,
        )
    finally:
        env.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
