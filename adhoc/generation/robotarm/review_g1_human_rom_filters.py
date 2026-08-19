#!/usr/bin/env python3
"""Review human-AROM lattice cuts (A/B/C) on the G1 {-90,0,+90} pose bank.

Retained / rejected are split ONLY by joint-range limits — hard filters
(collision, collapsed_arm, ee_x) are ignored in this view.
"""
from __future__ import annotations

import html as H
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from embodiment_renderers import _make_g1_scene_xml  # noqa: E402

CAND = REPO / "data/results/visualize/g1_pose_bank_90deg_review/g1_pose_bank_90deg_candidates.json"
OUT_ROOT = REPO / "data/results/visualize/g1_human_rom_filter_review"
OUT_HTML = REPO / "data/results/html/g1_human_rom_filter_review.html"

TILE = 160
COLS = 6
ROWS_PER_PAGE = 5
PAGE_W = COLS * TILE
PAGE_H = ROWS_PER_PAGE * TILE

# Short names for reject labels
_SHORT = {
    "right_shoulder_pitch_joint": "pitch",
    "right_shoulder_roll_joint": "roll",
    "right_shoulder_yaw_joint": "yaw",
    "right_elbow_joint": "elbow",
    "right_wrist_roll_joint": "w_roll",
    "right_wrist_pitch_joint": "w_pitch",
    "right_wrist_yaw_joint": "w_yaw",
}

VARIANTS: list[dict[str, Any]] = [
    {
        "key": "A_clear_human",
        "title": "A · clear human AROM",
        "description": (
            "Drop clear impossibles only: shoulder_pitch +90 (extension ~90°, human ~60°) "
            "and wrist_yaw ±90 (radial/ulnar only ~20–30°)."
        ),
        "lattice": {
            "right_shoulder_pitch_joint": (-90.0, 0.0),
            "right_shoulder_roll_joint": (-90.0, 0.0, 90.0),
            "right_shoulder_yaw_joint": (-90.0, 0.0, 90.0),
            "right_elbow_joint": (0.0, 90.0),
            "right_wrist_roll_joint": (-90.0, 0.0, 90.0),
            "right_wrist_pitch_joint": (-90.0, 0.0, 90.0),
            "right_wrist_yaw_joint": (0.0,),
        },
    },
    {
        "key": "B_clear_plus_roll",
        "title": "B · A + drop roll +90",
        "description": (
            "A plus shoulder_roll +90 (strong cross-body / non-abduction extreme)."
        ),
        "lattice": {
            "right_shoulder_pitch_joint": (-90.0, 0.0),
            "right_shoulder_roll_joint": (-90.0, 0.0),
            "right_shoulder_yaw_joint": (-90.0, 0.0, 90.0),
            "right_elbow_joint": (0.0, 90.0),
            "right_wrist_roll_joint": (-90.0, 0.0, 90.0),
            "right_wrist_pitch_joint": (-90.0, 0.0, 90.0),
            "right_wrist_yaw_joint": (0.0,),
        },
    },
    {
        "key": "C_strict_wrist",
        "title": "C · B + wrist_pitch 0 only",
        "description": (
            "B plus wrist_pitch only 0 (drop ±90 flex/ext lattice extremes)."
        ),
        "lattice": {
            "right_shoulder_pitch_joint": (-90.0, 0.0),
            "right_shoulder_roll_joint": (-90.0, 0.0),
            "right_shoulder_yaw_joint": (-90.0, 0.0, 90.0),
            "right_elbow_joint": (0.0, 90.0),
            "right_wrist_roll_joint": (-90.0, 0.0, 90.0),
            "right_wrist_pitch_joint": (0.0,),
            "right_wrist_yaw_joint": (0.0,),
        },
    },
]


def _fonts():
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        return ImageFont.load_default()


def _fwd(model: mujoco.MjModel, data: mujoco.MjData, row: dict[str, Any]) -> None:
    data.qpos[:] = np.asarray(row["qpos"], dtype=float)
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def _rom_check(row: dict[str, Any], lattice: dict[str, tuple[float, ...]]) -> tuple[bool, str]:
    """Return (rejected, reason). Rejected = outside human lattice."""
    jd = row["joint_deg"]
    bad: list[str] = []
    for joint, allowed in lattice.items():
        val = float(jd[joint])
        if val not in allowed:
            bad.append(f"{_SHORT[joint]}={val:+.0f}")
    if bad:
        return True, ",".join(bad)
    return False, "ok"


def _rel(path: Path) -> str:
    return str(Path("..") / path.relative_to(OUT_HTML.parent.parent)).replace("\\", "/")


def _render_pages(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    camera: mujoco.MjvCamera,
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    font,
    reason_by_id: dict[int, str],
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # clear old pages
    for old in out_dir.glob("page_*.jpg"):
        old.unlink()
    if not rows:
        empty = out_dir / "page_01.jpg"
        img = Image.new("RGB", (PAGE_W, 80), (30, 30, 30))
        ImageDraw.Draw(img).text((12, 30), "(empty)", fill=(200, 200, 200), font=font)
        img.save(empty, quality=88)
        return [empty]

    paths: list[Path] = []
    page_size = COLS * ROWS_PER_PAGE
    for page_i in range(0, len(rows), page_size):
        page_rows = rows[page_i : page_i + page_size]
        canvas = Image.new("RGB", (PAGE_W, PAGE_H), (235, 235, 235))
        for index, row in enumerate(page_rows):
            _fwd(model, data, row)
            renderer.update_scene(data, camera)
            tile = Image.fromarray(renderer.render()).convert("RGB").resize(
                (TILE, TILE), Image.BILINEAR
            )
            draw = ImageDraw.Draw(tile)
            pid = int(row["pose_id"])
            reason = reason_by_id.get(pid, "")
            label = f"#{pid} {reason}".strip()
            draw.rectangle((0, 0, TILE, 22), fill=(0, 0, 0))
            draw.text((3, 3), label[:42], fill=(255, 255, 255), font=font)
            canvas.paste(tile, ((index % COLS) * TILE, (index // COLS) * TILE))
        path = out_dir / f"page_{page_i // page_size + 1:02d}.jpg"
        canvas.save(path, quality=85)
        paths.append(path)
    return paths


def main() -> None:
    payload = json.loads(CAND.read_text(encoding="utf-8"))
    pool = list(payload.get("poses") or [])
    pool.sort(
        key=lambda r: (
            str(r.get("direction")),
            str(r.get("gripper_orientation")),
            int(r["pose_id"]),
        )
    )
    print(f"pool (full robot lattice, no hard-filter split) = {len(pool)}")

    font = _fonts()
    xml_path = Path(_make_g1_scene_xml())
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=TILE, width=TILE)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.85)
    camera.distance = 2.7
    camera.azimuth = 145
    camera.elevation = -8

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []

    for variant in VARIANTS:
        lattice = variant["lattice"]
        product_n = 1
        for vals in lattice.values():
            product_n *= len(vals)

        rejected: list[dict[str, Any]] = []
        retained: list[dict[str, Any]] = []
        reasons: dict[int, str] = {}
        for row in pool:
            bad, reason = _rom_check(row, lattice)
            reasons[int(row["pose_id"])] = reason
            (rejected if bad else retained).append(row)

        key = variant["key"]
        var_dir = OUT_ROOT / key
        print(
            f"[{key}] lattice={product_n} retained={len(retained)} "
            f"rejected={len(rejected)} (joint-ROM only)"
        )
        rej_pages = _render_pages(
            model, data, renderer, camera, rejected, var_dir / "rejected",
            font=font, reason_by_id=reasons,
        )
        ret_pages = _render_pages(
            model, data, renderer, camera, retained, var_dir / "retained",
            font=font, reason_by_id=reasons,
        )

        lattice_txt = {
            _SHORT[j]: list(v) for j, v in lattice.items()
        }
        summary.append(
            {
                "key": key,
                "title": variant["title"],
                "description": variant["description"],
                "lattice": lattice_txt,
                "lattice_product": product_n,
                "pool": len(pool),
                "rejected": len(rejected),
                "retained": len(retained),
                "reject_rate": len(rejected) / max(len(pool), 1),
                "rejected_page_rels": [_rel(p) for p in rej_pages],
                "retained_page_rels": [_rel(p) for p in ret_pages],
            }
        )

    renderer.close()

    (OUT_ROOT / "summary.json").write_text(
        json.dumps(
            {
                "pool": len(pool),
                "note": (
                    "Split is joint-ROM only on the full 90° robot lattice. "
                    "Hard filters (collision / collapsed_arm / ee_x) are NOT applied here."
                ),
                "variants": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    parts = [
        """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>G1 human-ROM filter review (A/B/C)</title>
<style>
body{margin:0;font-family:ui-sans-serif,system-ui,sans-serif;background:#0c0e14;color:#e6eaf2}
header{padding:18px 20px;border-bottom:1px solid #2a3140;position:sticky;top:0;background:#0c0e14f2;z-index:5}
.note{color:#9aa3b5;max-width:1100px;line-height:1.45;font-size:.92rem}
nav{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
nav a{color:#7ab4f8;text-decoration:none;border:1px solid #2a3140;padding:4px 9px;border-radius:8px;font-size:.8rem;background:#12161f}
main{padding:10px 20px 80px}
section{border-top:1px solid #2a3140;margin-top:28px;padding-top:16px}
h2{margin:0 0 6px;font-size:1.2rem}
.desc{color:#9aa3b5;font-size:.88rem;max-width:1000px;margin:0 0 8px;line-height:1.4}
.lattice{font-size:.78rem;color:#9bc6ff;margin:0 0 12px;white-space:pre-wrap}
.stats{font-size:.9rem;color:#c5cde0;margin:0 0 12px}
.stats b{color:#fff}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start}
@media(max-width:1000px){.pair{grid-template-columns:1fr}}
figure{margin:0;background:#12161f;border:1px solid #242a36;border-radius:10px;padding:8px}
figcaption{font-size:.78rem;color:#9aa3b5;margin-bottom:6px;letter-spacing:.04em;text-transform:uppercase}
.sheets{display:flex;flex-direction:column;gap:8px;max-height:80vh;overflow:auto}
img{width:100%;border-radius:6px;background:#000;display:block}
.rej{color:#e8a09a}.ret{color:#9dcc8a}
code{color:#9bc6ff}
</style></head><body>
"""
    ]
    parts.append(
        f"""<header>
<h1>G1 human-ROM lattice cuts · A / B / C</h1>
<p class="note">
Pool = full robot 90° lattice (<b>{len(pool)}</b> poses). Elbow already <code>[0,90]</code> from MJCF.<br>
<b>Retained / rejected = joint-range only.</b> Hard filters (collision · collapsed_arm · ee_x&lt;10)
are <em>not</em> used for this split — labels on rejected tiles show which joints are out of AROM.
</p>
<nav>
"""
    )
    for s in summary:
        parts.append(
            f'<a href="#{H.escape(s["key"])}">{H.escape(s["title"])}</a>'
        )
    parts.append("</nav></header><main>\n")

    for s in summary:
        lat_lines = "\n".join(f"{j}: {vals}" for j, vals in s["lattice"].items())
        parts.append(f'<section id="{H.escape(s["key"])}">')
        parts.append(f'<h2>{H.escape(s["title"])}</h2>')
        parts.append(f'<p class="desc">{H.escape(s["description"])}</p>')
        parts.append(
            f'<pre class="lattice">lattice product = {s["lattice_product"]}\n{H.escape(lat_lines)}</pre>'
        )
        parts.append(
            f'<p class="stats">pool <b>{s["pool"]}</b> · '
            f'<span class="rej">rejected <b>{s["rejected"]}</b> '
            f'({100 * s["reject_rate"]:.1f}%)</span> · '
            f'<span class="ret">retained <b>{s["retained"]}</b></span></p>'
        )
        parts.append('<div class="pair">')
        rej_imgs = "".join(
            f'<img loading="lazy" src="{H.escape(p)}">' for p in s["rejected_page_rels"]
        )
        ret_imgs = "".join(
            f'<img loading="lazy" src="{H.escape(p)}">' for p in s["retained_page_rels"]
        )
        parts.append(
            f'<figure><figcaption class="rej">Rejected · joint ROM ({s["rejected"]})</figcaption>'
            f'<div class="sheets">{rej_imgs}</div></figure>'
        )
        parts.append(
            f'<figure><figcaption class="ret">Retained · joint ROM ({s["retained"]})</figcaption>'
            f'<div class="sheets">{ret_imgs}</div></figure>'
        )
        parts.append("</div></section>\n")

    parts.append("</main></body></html>\n")
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text("".join(parts), encoding="utf-8")
    print("HTML", OUT_HTML)


if __name__ == "__main__":
    main()
