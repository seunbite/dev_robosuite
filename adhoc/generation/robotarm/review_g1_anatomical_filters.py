#!/usr/bin/env python3
"""Run anatomical hard-filter candidates on the G1 soft-remain pose pool.

For each algorithm, emit rejected / retained tiled sheets (no selected/red borders)
and an index HTML so a human can pick which filters to keep.
"""
from __future__ import annotations

import html as H
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from embodiment_renderers import _make_g1_scene_xml  # noqa: E402

CAND = REPO / "data/results/visualize/g1_pose_bank_90deg_review/g1_pose_bank_90deg_candidates.json"
OUT_ROOT = REPO / "data/results/visualize/g1_anatomical_filter_review"
OUT_HTML = REPO / "data/results/html/g1_anatomical_filter_review.html"

TILE = 280
COLS = 3
ROWS_PER_PAGE = 4  # 12 tiles / page
PAGE_W = COLS * TILE
PAGE_H = ROWS_PER_PAGE * TILE


@dataclass(frozen=True)
class Algo:
    key: str
    title: str
    description: str
    reject_fn: Callable[[dict[str, Any], mujoco.MjModel, mujoco.MjData], tuple[bool, str]]


def _fonts():
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception:
        return ImageFont.load_default()


def _fwd(model: mujoco.MjModel, data: mujoco.MjData, row: dict[str, Any]) -> None:
    data.qpos[:] = np.asarray(row["qpos"], dtype=float)
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def _links(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
    return {
        "torso": np.asarray(data.xpos[model.body("torso_link").id], float),
        "shoulder": np.asarray(data.xpos[model.body("right_shoulder_pitch_link").id], float),
        "elbow": np.asarray(data.xpos[model.body("right_elbow_link").id], float),
        "wrist": np.asarray(data.xpos[model.body("right_wrist_yaw_link").id], float),
        "palm": np.asarray(data.site_xpos[model.site("right_palm").id], float),
    }


def _ang_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)) + 1e-12
    nb = float(np.linalg.norm(b)) + 1e-12
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _point_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / (float(np.dot(ab, ab)) + 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _segment_points(a: np.ndarray, b: np.ndarray, n: int = 8) -> list[np.ndarray]:
    return [a + (b - a) * (i / (n - 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

def reject_elbow_twist(
    row: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[bool, str]:
    """Reject extreme forearm twist about the upper-arm axis (non-hinge elbow).

    Threshold: |twist| > 70° from the zero-twist reference (upper × world-up).
    """
    _fwd(model, data, row)
    L = _links(model, data)
    upper = L["elbow"] - L["shoulder"]
    fore = L["wrist"] - L["elbow"]
    axis = upper / (float(np.linalg.norm(upper)) + 1e-12)
    # Zero-twist reference in plane ⊥ upper: world-up projected
    world_up = np.array([0.0, 0.0, 1.0])
    ref = world_up - np.dot(world_up, axis) * axis
    if float(np.linalg.norm(ref)) < 1e-6:
        ref = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), axis) * axis
    ref = ref / (float(np.linalg.norm(ref)) + 1e-12)
    fore_p = fore - np.dot(fore, axis) * axis
    if float(np.linalg.norm(fore_p)) < 1e-4:
        return False, "colinear"
    fore_p = fore_p / (float(np.linalg.norm(fore_p)) + 1e-12)
    # Signed twist about upper axis
    twist = math.degrees(
        math.atan2(float(np.dot(axis, np.cross(ref, fore_p))), float(np.dot(ref, fore_p)))
    )
    if abs(twist) > 70.0:
        return True, f"twist={twist:.0f}"
    return False, f"twist={twist:.0f}"


def reject_segment_length(
    row: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[bool, str]:
    """Reject unnatural upper/forearm lengths or broken triangle inequality."""
    _fwd(model, data, row)
    L = _links(model, data)
    u = float(np.linalg.norm(L["elbow"] - L["shoulder"]))
    f = float(np.linalg.norm(L["wrist"] - L["elbow"]))
    r = float(np.linalg.norm(L["wrist"] - L["shoulder"]))
    # G1 nominal segments ~0.2–0.28 m; allow lattice slack
    if u < 0.14 or u > 0.36:
        return True, f"upper={u:.2f}"
    if f < 0.14 or f > 0.36:
        return True, f"fore={f:.2f}"
    if r > u + f - 0.005:
        return True, f"tri={r:.2f}>{u+f:.2f}"
    if r < abs(u - f) + 0.005:
        return True, f"tri_collapse"
    ratio = u / (f + 1e-12)
    if ratio < 0.55 or ratio > 1.80:
        return True, f"ratio={ratio:.2f}"
    return False, f"u={u:.2f}/f={f:.2f}"


def reject_wrist_below_elbow(
    row: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[bool, str]:
    """Reject raised-arm poses where the wrist sits below the elbow (inverted)."""
    _fwd(model, data, row)
    L = _links(model, data)
    direction = str(row.get("direction", "")).lower()
    z_pct = float(row.get("z", 50.0))
    raised = direction == "up" or z_pct >= 55.0
    dz = float(L["wrist"][2] - L["elbow"][2])
    if raised and dz < -0.03:
        return True, f"dz={dz:.2f}"
    # Also: for dir=up, elbow should not be far above the wrist
    if direction == "up" and float(L["elbow"][2] - L["wrist"][2]) > 0.05:
        return True, f"elbow_over_wrist"
    return False, f"dz={dz:.2f}"


def reject_behind_torso(
    row: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[bool, str]:
    """Reject hands clearly behind the torso for non-back directions.

    Contact hard-filter already removes geom collisions; this catches
    anatomically awkward 'arm through / behind body' silhouettes that still
    clear meshes (esp. lattice folds).
    """
    _fwd(model, data, row)
    L = _links(model, data)
    direction = str(row.get("direction", "")).lower()
    if direction == "back":
        return False, "back_ok"
    dx = float(L["wrist"][0] - L["torso"][0])
    # torso +X is forward; wrist behind by >8 cm
    if dx < -0.08:
        return True, f"dx={dx:.2f}"
    # right wrist far across midline (+y) looks like body-crossing pretzel
    if float(L["wrist"][1]) > 0.08 and direction in ("out", "front", "up"):
        return True, f"cross_y={L['wrist'][1]:.2f}"
    return False, f"dx={dx:.2f}"


def reject_upper_arm_sense(
    row: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[bool, str]:
    """Reject upper-arm directions that contradict the classified reach dir.

    Human scapulohumeral prior (proxy): for ``up`` the elbow shouldn't sit below
    the shoulder; for ``front``/``up`` the upper arm shouldn't point strongly back.
    """
    _fwd(model, data, row)
    L = _links(model, data)
    upper = L["elbow"] - L["shoulder"]
    nu = float(np.linalg.norm(upper)) + 1e-12
    upper_u = upper / nu
    direction = str(row.get("direction", "")).lower()
    if direction == "up" and float(upper_u[2]) < -0.15:
        return True, f"up_z={upper_u[2]:.2f}"
    if direction in ("front", "up") and float(upper_u[0]) < -0.35:
        return True, f"back={upper_u[0]:.2f}"
    if direction == "out" and float(upper_u[1]) > 0.25:
        # right arm "out" should go −y (robot right), not inward
        return True, f"out_y={upper_u[1]:.2f}"
    if direction == "in" and float(upper_u[1]) < -0.35:
        return True, f"in_y={upper_u[1]:.2f}"
    if direction == "down" and float(upper_u[2]) > 0.35:
        return True, f"down_z={upper_u[2]:.2f}"
    return False, f"u=({upper_u[0]:.1f},{upper_u[1]:.1f},{upper_u[2]:.1f})"

def reject_elbow_line_collapse(
    row: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[bool, str]:
    """Reject bent elbows whose elbow joint lies too close to the shoulder–wrist line.

    Catches 'Z-ish' folds that look broken even when segments have length.
    """
    _fwd(model, data, row)
    L = _links(model, data)
    if not bool(row.get("elbow_bended")):
        return False, "straight"
    d = _point_segment_dist(L["elbow"], L["shoulder"], L["wrist"])
    # With ~90° bend, offset should be sizable; < 4 cm looks collapsed/twisted
    if d < 0.04:
        return True, f"off={d:.3f}"
    return False, f"off={d:.3f}"


def reject_zigzag_existing(
    row: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[bool, str]:
    """Existing soft filter: opposite-sense Z-fold (already in candidate JSON)."""
    del model, data
    m = row.get("zigzag_metrics") or {}
    if row.get("zigzag_filter_fail") or m.get("zigzag"):
        return True, f"el={m.get('turn_elbow_deg','?')}/wr={m.get('turn_wrist_deg','?')}"
    return False, "ok"


ALGOS: list[Algo] = [
    Algo(
        "01_elbow_twist",
        "1. Elbow twist (hinge prior)",
        "Reject |forearm twist about upper-arm| > 70° from world-up zero-twist reference. "
        "Targets non-hinge / anatomically twisted elbows.",
        reject_elbow_twist,
    ),
    Algo(
        "02_segment_length",
        "2. Segment length / triangle",
        "Reject upper/forearm outside [0.14, 0.36] m, bad length ratio, or broken "
        "triangle inequality on S–E–W.",
        reject_segment_length,
    ),
    Algo(
        "03_wrist_above_elbow",
        "3. Wrist-above-elbow (raised poses)",
        "For dir=up or z≥55: reject if wrist is >3 cm below elbow (inverted raised arm).",
        reject_wrist_below_elbow,
    ),
    Algo(
        "04_behind_torso",
        "4. Behind-torso / midline cross",
        "For non-back dirs: reject wrist behind torso (Δx < −8 cm) or right wrist "
        "crossing far past midline (+y) on out/front/up.",
        reject_behind_torso,
    ),
    Algo(
        "05_upper_arm_sense",
        "5. Upper-arm vs direction sense",
        "Reject when upper-arm vector contradicts dir (e.g. dir=up but elbow below "
        "shoulder; dir=front/up but upper arm points strongly backward).",
        reject_upper_arm_sense,
    ),
    Algo(
        "06_elbow_line_collapse",
        "6. Elbow–SW line collapse",
        "For elbow_bended: reject if elbow is <4 cm from the shoulder–wrist segment "
        "(bent flag but visually collapsed / Z-fold).",
        reject_elbow_line_collapse,
    ),
    Algo(
        "07_zigzag_zfold",
        "7. Existing soft: zigzag Z-fold",
        "Baseline already used: opposite-sense elbow/wrist turns (from candidate JSON).",
        reject_zigzag_existing,
    ),
]


def _load_pool() -> list[dict[str, Any]]:
    """Soft-remain pool: hard-pass ∧ ¬zigzag ∧ xyz-palm-dedupe (pre-manual-cull)."""
    payload = json.loads(CAND.read_text(encoding="utf-8"))
    rows = []
    for row in payload.get("poses") or []:
        if not row.get("hard_filter_pass"):
            continue
        if row.get("zigzag_filter_fail"):
            # Keep zigzag fails in pool only for algo 07 comparison — actually for
            # evaluating NEW filters, start from remain (no zigzag fail).
            continue
        if row.get("xyz_palm_dedupe_pass") is False:
            continue
        rows.append(row)
    return rows


def _load_hard_pass_for_zigzag_demo() -> list[dict[str, Any]]:
    """Hard-pass only (incl. zigzag fails) — used solely for algo 07 sheets."""
    payload = json.loads(CAND.read_text(encoding="utf-8"))
    return [
        row
        for row in payload.get("poses") or []
        if row.get("hard_filter_pass") and row.get("xyz_palm_dedupe_pass") is not False
    ]


def _render_sheet(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    camera: mujoco.MjvCamera,
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    font,
    reason_by_id: dict[int, str] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        img = Image.new("RGB", (PAGE_W, 80), (30, 30, 30))
        ImageDraw.Draw(img).text((12, 30), "(empty)", fill=(200, 200, 200), font=font)
        img.save(out_path, quality=88)
        return
    pages = []
    for page_i in range(0, len(rows), COLS * ROWS_PER_PAGE):
        page_rows = rows[page_i : page_i + COLS * ROWS_PER_PAGE]
        canvas = Image.new("RGB", (PAGE_W, PAGE_H), (235, 235, 235))
        for index, row in enumerate(page_rows):
            _fwd(model, data, row)
            renderer.update_scene(data, camera)
            tile = Image.fromarray(renderer.render()).convert("RGB").resize(
                (TILE, TILE), Image.BILINEAR
            )
            draw = ImageDraw.Draw(tile)
            pid = int(row["pose_id"])
            reason = (reason_by_id or {}).get(pid, "")
            label = f"#{pid} {reason}".strip()
            draw.rectangle((0, 0, TILE, 26), fill=(0, 0, 0))
            draw.text((4, 4), label[:48], fill=(255, 255, 255), font=font)
            # NO selected / red borders
            canvas.paste(tile, ((index % COLS) * TILE, (index // COLS) * TILE))
        pages.append(canvas)
    if len(pages) == 1:
        pages[0].save(out_path, quality=88)
    else:
        # Stack pages vertically into one image for simpler HTML
        total_h = sum(p.height for p in pages) + 4 * (len(pages) - 1)
        big = Image.new("RGB", (PAGE_W, total_h), (20, 20, 20))
        y = 0
        for p in pages:
            big.paste(p, (0, y))
            y += p.height + 4
        big.save(out_path, quality=85)


def main() -> None:
    font = _fonts()
    remain = _load_pool()
    hard_deduped = _load_hard_pass_for_zigzag_demo()
    print(f"pool remain (hard∧¬zigzag∧dedupe) = {len(remain)}")
    print(f"pool hard∧dedupe (for zigzag demo) = {len(hard_deduped)}")

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

    summary: list[dict[str, Any]] = []
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for algo in ALGOS:
        pool = hard_deduped if algo.key.startswith("07_") else remain
        rejected: list[dict[str, Any]] = []
        retained: list[dict[str, Any]] = []
        reasons: dict[int, str] = {}
        for row in pool:
            bad, reason = algo.reject_fn(row, model, data)
            reasons[int(row["pose_id"])] = reason
            (rejected if bad else retained).append(row)
        # Stable order for browsing
        rejected.sort(key=lambda r: (str(r.get("direction")), str(r.get("gripper_orientation")), int(r["pose_id"])))
        retained.sort(key=lambda r: (str(r.get("direction")), str(r.get("gripper_orientation")), int(r["pose_id"])))

        algo_dir = OUT_ROOT / algo.key
        rej_path = algo_dir / "rejected.jpg"
        ret_path = algo_dir / "retained.jpg"
        print(
            f"[{algo.key}] pool={len(pool)} rejected={len(rejected)} retained={len(retained)}"
        )
        _render_sheet(
            model, data, renderer, camera, rejected, rej_path, font=font, reason_by_id=reasons
        )
        _render_sheet(
            model, data, renderer, camera, retained, ret_path, font=font, reason_by_id=reasons
        )
        # Per-category mini counts
        by_cat = defaultdict(lambda: [0, 0])
        for r in rejected:
            by_cat[f"{r['direction']}|{r['gripper_orientation']}"][0] += 1
        for r in retained:
            by_cat[f"{r['direction']}|{r['gripper_orientation']}"][1] += 1
        summary.append(
            {
                "key": algo.key,
                "title": algo.title,
                "description": algo.description,
                "pool": len(pool),
                "rejected": len(rejected),
                "retained": len(retained),
                "reject_rate": len(rejected) / max(len(pool), 1),
                "by_category": {k: {"rejected": v[0], "retained": v[1]} for k, v in sorted(by_cat.items())},
                "rejected_ids": [int(r["pose_id"]) for r in rejected],
                "retained_rel": str(Path("..") / ret_path.relative_to(OUT_HTML.parent.parent)),
                "rejected_rel": str(Path("..") / rej_path.relative_to(OUT_HTML.parent.parent)),
            }
        )

    renderer.close()

    (OUT_ROOT / "summary.json").write_text(
        json.dumps(
            {
                "pool_remain": len(remain),
                "pool_hard_deduped": len(hard_deduped),
                "note": (
                    "Algos 1–6 scored on remain pool (hard ∧ ¬zigzag ∧ dedupe). "
                    "Algo 7 scored on hard∧dedupe to show zigzag cut."
                ),
                "algorithms": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    parts = [
        """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>G1 anatomical filter review</title>
<style>
body{margin:0;font-family:ui-sans-serif,system-ui,sans-serif;background:#0c0e14;color:#e6eaf2}
header{padding:18px 20px;border-bottom:1px solid #2a3140;position:sticky;top:0;background:#0c0e14f2;z-index:5}
.note{color:#9aa3b5;max-width:1100px;line-height:1.45;font-size:.92rem}
nav{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
nav a{color:#7ab4f8;text-decoration:none;border:1px solid #2a3140;padding:4px 9px;border-radius:8px;font-size:.75rem;background:#12161f}
main{padding:10px 20px 80px}
section{border-top:1px solid #2a3140;margin-top:28px;padding-top:16px}
h2{margin:0 0 6px;font-size:1.15rem}
.desc{color:#9aa3b5;font-size:.88rem;max-width:1000px;margin:0 0 12px;line-height:1.4}
.stats{font-size:.85rem;color:#c5cde0;margin:0 0 12px}
.stats b{color:#fff}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start}
@media(max-width:900px){.pair{grid-template-columns:1fr}}
figure{margin:0;background:#12161f;border:1px solid #242a36;border-radius:10px;padding:8px}
figcaption{font-size:.78rem;color:#9aa3b5;margin-bottom:6px;letter-spacing:.04em;text-transform:uppercase}
img{width:100%;border-radius:6px;background:#000;display:block}
code{color:#9bc6ff}
.rej{color:#e8a09a}.ret{color:#9dcc8a}
</style></head><body>
"""
    ]
    parts.append(
        f"""<header>
<h1>G1 anatomical filter candidates</h1>
<p class="note">
Pool for algos <b>1–6</b>: hard ∧ ¬zigzag ∧ xyz-palm-dedupe = <b>{len(remain)}</b> poses
(pre-manual-cull remain).<br>
Algo <b>7</b> (existing zigzag) scored on hard∧dedupe = <b>{len(hard_deduped)}</b> to show its cut.<br>
No selected / red borders — pick filters you like; we will apply only those.<br>
Existing hard (already applied to pool): collision · collapsed_arm · ee_x&lt;10.
</p>
<nav>
"""
    )
    for s in summary:
        parts.append(f'<a href="#{H.escape(s["key"])}">{H.escape(s["title"].split(". ",1)[-1] if ". " in s["title"] else s["title"])}</a>')
    parts.append("</nav></header><main>\n")

    for s in summary:
        parts.append(f'<section id="{H.escape(s["key"])}">')
        parts.append(f'<h2>{H.escape(s["title"])}</h2>')
        parts.append(f'<p class="desc">{H.escape(s["description"])}</p>')
        parts.append(
            f'<p class="stats">pool <b>{s["pool"]}</b> · '
            f'<span class="rej">rejected <b>{s["rejected"]}</b> ({100*s["reject_rate"]:.1f}%)</span> · '
            f'<span class="ret">retained <b>{s["retained"]}</b></span></p>'
        )
        parts.append('<div class="pair">')
        parts.append(
            f'<figure><figcaption class="rej">Rejected ({s["rejected"]})</figcaption>'
            f'<img loading="lazy" src="{H.escape(s["rejected_rel"])}"></figure>'
        )
        parts.append(
            f'<figure><figcaption class="ret">Retained ({s["retained"]})</figcaption>'
            f'<img loading="lazy" src="{H.escape(s["retained_rel"])}"></figure>'
        )
        parts.append("</div></section>\n")

    parts.append("</main></body></html>\n")
    OUT_HTML.write_text("".join(parts), encoding="utf-8")
    print("HTML", OUT_HTML)


if __name__ == "__main__":
    main()
