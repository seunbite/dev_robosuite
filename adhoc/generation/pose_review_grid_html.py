#!/usr/bin/env python3
"""
Build an HTML gallery: each *pose* step in merged motion configs → one indexed row.

Each indexed row lists a **GIF** (full-motion render beside the PNGs, when present), an **original**
pose still, plus three deltas: ``direction``, ``gripper_orientation``, and ``xyz`` percent offsets.
Grip variation enforces authored H/V visually (Tiago: elbow.roll after pose-bank; manipulator:
JSONL matches filtered by ``gripper_orientation``).

Row caps (--max-*): use **0** (default) for no limit (covers every pose step in the loaded motion JSON).

Examples:
  .venv/bin/python -m adhoc.generation.pose_review_grid_html \\
      --out-dir data/results/html/pose_review_grid_demo

  Only rows for **hand-authored shots** (see_html_sources.yml ``shots_json``), not the full merged set:
      --shots-only \\
      --out-dir data/results/html/pose_review_grid_shots

Paths default from data/seed/yml/see_html_sources.yml (manipulator + tiago blocks).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import html
import io
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image

from adhoc.generation.embodiment_sources import (
    repo_rel_to_path,
    resolve_embodiment_paths,
)
from adhoc.utils.repo_paths import dev_robosuite_root


_NO_CAP_BUCKETS = 10**12  # "unlimited": larger than any practical pose-row count


# ── deterministic variation deltas (percentile grid for xyz) ─────────────
_XYZ_DELTA = (14, -12, 10)

# ── TIAGo arm_preset key rotation (matches render_mobile_config.ARM_PRESETS order) ─
_TIAGO_ARM_ORDER = [
    "up",
    "back",
    "down+back",
    "right",
    "down+right",
    "front",
    "down+front",
    "fold",
]

# ── Manipulator (IIWA pose DB) directional cycle ───────────────────────────
_MANIP_DIRS = ["front", "up", "down", "left", "right", "back"]


def _slug(s: str, max_len: int = 64) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = s.strip("_") or "x"
    return s[:max_len]


def _clamp_pct(v: float) -> int:
    return int(round(max(0.0, min(100.0, v))))


@dataclass(frozen=True)
class FrameSpec:
    key: str
    label: str
    subtitle: str


def _variation_google(pose: dict) -> tuple[dict, dict, dict]:
    arm = pose.get("arm_position", "down+right")
    if arm not in _TIAGO_ARM_ORDER:
        i0 = 0
    else:
        i0 = _TIAGO_ARM_ORDER.index(arm)
    p_dir = copy.deepcopy(pose)
    p_dir["arm_position"] = _TIAGO_ARM_ORDER[(i0 + 1) % len(_TIAGO_ARM_ORDER)]

    g0 = pose.get("gripper_orientation", "vertical") or "vertical"
    p_grip = copy.deepcopy(pose)
    p_grip["gripper_orientation"] = "horizontal" if g0 == "vertical" else "vertical"

    dx, dy, dz = _XYZ_DELTA
    p_xyz = copy.deepcopy(pose)
    ox = float(pose.get("x", 50))
    oy = float(pose.get("y", 50))
    oz = float(pose.get("z", 50))
    p_xyz["x"] = _clamp_pct(ox + dx)
    p_xyz["y"] = _clamp_pct(oy + dy)
    p_xyz["z"] = _clamp_pct(oz + dz)

    return p_dir, p_grip, p_xyz


def _count_pose_steps(configs: list[dict]) -> int:
    return sum(
        1
        for row in configs
        for m in (row.get("movements") or [])
        if m.get("type") == "pose"
    )


def _budget(n: int) -> int:
    """CLI caps: ``n <= 0`` means no limit (very large bucket)."""
    return n if int(n) > 0 else _NO_CAP_BUCKETS


_MANIP_RENDER_SUBDIR = "IIWA"


def _cue_safe_for_motion_gif(cue: str) -> str:
    return str(cue).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _resolve_google_motion_gif(render_dir: Path, cue: str, motion_idx: Any) -> Path | None:
    try:
        mid = int(motion_idx)
    except (TypeError, ValueError):
        return None
    cand = Path(render_dir) / f"mm19_g{mid:02d}_{_cue_safe_for_motion_gif(cue)}.gif"
    return cand if cand.is_file() else None


def _resolve_manipulator_motion_gif(render_dir: Path, cue: str, motion_idx: Any) -> Path | None:
    try:
        mid = int(motion_idx)
    except (TypeError, ValueError):
        return None

    rr = Path(render_dir)
    roots: list[Path] = []
    sub = rr / _MANIP_RENDER_SUBDIR
    if sub.is_dir():
        roots.append(sub)
    if rr.is_dir():
        roots.append(rr)

    best: Path | None = None
    best_mtime = -1.0
    for rt in roots:
        if not rt.is_dir():
            continue
        for p in rt.glob(f"*_c{mid}_tiled.gif"):
            try:
                m = float(p.stat().st_mtime)
            except OSError:
                continue
            if m > best_mtime:
                best_mtime = m
                best = p
    if best is not None:
        return best

    safe_full = _cue_safe_for_motion_gif(cue)
    if not safe_full.strip():
        return None
    for rt in roots:
        if not rt.is_dir():
            continue
        for p in rt.glob("*.gif"):
            try:
                m = float(p.stat().st_mtime)
            except OSError:
                continue
            if safe_full in p.name and m > best_mtime:
                best_mtime = m
                best = p
    return best


def _bundle_motion_gif(
    *,
    embodiment_slug: str,
    cue: str,
    motion_idx: Any,
    render_dir_abs: Path | None,
    out_gifs_dir: Path,
    copy_by_canon: dict[str, str],
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "ok": False,
            "relpath": "",
            "subtitle": "",
            "note": "GIF column disabled (--skip-motion-gifs)",
        }
    if render_dir_abs is None or not render_dir_abs.is_dir():
        return {
            "ok": False,
            "relpath": "",
            "subtitle": "",
            "note": "render_dir missing; run embodiment render pipeline first.",
        }

    src: Path | None
    if embodiment_slug == "tiago":
        src = _resolve_google_motion_gif(render_dir_abs, cue, motion_idx)
    elif embodiment_slug == "manipulator":
        src = _resolve_manipulator_motion_gif(render_dir_abs, cue, motion_idx)
    else:
        src = None

    if src is None:
        return {
            "ok": False,
            "relpath": "",
            "subtitle": "",
            "note": f"No GIF in {render_dir_abs.name}",
        }

    try:
        canon = os.path.realpath(str(src))
    except OSError:
        canon = str(src.resolve())

    if canon not in copy_by_canon:
        h = hashlib.md5(canon.encode("utf-8", errors="ignore")).hexdigest()[:10]
        try:
            mid_i = int(motion_idx)
        except (TypeError, ValueError):
            mid_i = -1
        ext = src.suffix or ".gif"
        fname = f"{_slug(embodiment_slug, 32)}_c{mid_i}_{h}{ext}"
        dest = out_gifs_dir / fname
        shutil.copy2(src, dest)
        copy_by_canon[canon] = fname

    out_name = copy_by_canon[canon]
    return {
        "ok": True,
        "relpath": str((Path("gifs") / out_name)).replace("\\", "/"),
        "subtitle": src.name,
        "source_abspath_real": canon,
        "note": "",
    }


def _finalize_tiago_authored_grip_qpos(q: np.ndarray, pose: dict) -> None:
    """After pose-bank resolution, enforce authored ``gripper_orientation`` where presets use dof 2."""
    from adhoc.generation.tiago.legacy.render_mobile_config import (
        ARM_PRESETS,
        GRIPPER_MAP,
        QI_L_ARM,
        QI_R_ARM,
        _arm_rad,
        _mirror_offset,
    )

    arm = str(pose.get("arm_position", "down+right"))
    preset = ARM_PRESETS.get(arm, ARM_PRESETS["down+right"])
    grip = str(pose.get("gripper_orientation") or "vertical").strip().lower()
    if grip not in GRIPPER_MAP:
        grip = "vertical"

    if len(preset) >= 6 and int(preset[2]) == 0:
        r = np.asarray(q[QI_R_ARM], dtype=float).copy()
        r[2] = float(GRIPPER_MAP[grip])
        q[QI_R_ARM] = r

    left = pose.get("left_arm", "still")
    if left == "mirror":
        l_rad = list(np.asarray(q[QI_R_ARM], dtype=float))
        _mirror_offset(l_rad, arm)
        q[QI_L_ARM] = l_rad
    elif left != "still":
        q[QI_L_ARM] = _arm_rad(str(left), grip)


def _maybe_int(val: Any) -> int | None:
    """Parse JSON ``idx`` or similar integers."""
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return int(val)
    if isinstance(val, float) and float(val).is_integer():
        return int(val)
    if isinstance(val, str) and val.strip():
        try:
            return int(val.strip(), 10)
        except ValueError:
            return None
    return None


def filter_motion_configs_by_shots_only(
    configs: list[dict],
    *,
    slug: str,
    shots_rows: list[dict],
) -> tuple[list[dict], str]:
    """Keep cue configs referenced by the embodiment's shots catalog.

    - **manipulator**: match merged ``motion_config`` rows whose ``idx`` appears in shots.
      (Cue strings differ between shots vs merged catalogs; indices align.)
    - **tiago**: match merged rows whose ``cue`` string equals any shot row's ``cue``.

    ``shots_rows`` is the JSON root list loaded from ``shots_json``.
    """
    if slug == "manipulator":
        allowed_idx = {_maybe_int(s.get("idx")) for s in shots_rows}
        allowed_idx.discard(None)
        if allowed_idx:
            kept = []
            dropped = 0
            for c in configs:
                cid = _maybe_int(c.get("idx"))
                if cid is not None and cid in allowed_idx:
                    kept.append(c)
                else:
                    dropped += 1
            return kept, (
                f"manipulator: idx filter from shots ({len(allowed_idx)} ids) "
                f"→ {len(kept)} cue configs kept, {dropped} dropped."
            )

        cues = _shot_cues_set(shots_rows)
        kept = []
        dropped = 0
        for c in configs:
            q = str(c.get("cue", "")).strip()
            if q and q in cues:
                kept.append(c)
            else:
                dropped += 1
        return kept, (
            f"manipulator (no idx in shots): cue-string filter → "
            f"{len(kept)} kept, {dropped} dropped."
        )

    if slug == "tiago":
        cues = _shot_cues_set(shots_rows)
        kept = []
        dropped = 0
        for c in configs:
            q = str(c.get("cue", "")).strip()
            if q and q in cues:
                kept.append(c)
            else:
                dropped += 1
        return kept, (
            f"tiago: cue-string filter from shots ({len(cues)} cues in shots) "
            f"→ {len(kept)} cue configs kept, {dropped} dropped."
        )

    raise SystemExit(f"filter_motion_configs_by_shots_only: unknown slug {slug!r}")


def _shot_cues_set(shots_rows: list[dict]) -> set[str]:
    return {
        str(s["cue"]).strip()
        for s in shots_rows
        if isinstance(s, dict) and isinstance(s.get("cue"), str) and str(s["cue"]).strip()
    }


def _variation_manip(pose: dict) -> tuple[dict, dict, dict]:
    d0 = pose.get("dir") or "front"
    if d0 not in _MANIP_DIRS:
        i = 0
    else:
        i = _MANIP_DIRS.index(d0)
    p_dir = copy.deepcopy(pose)
    p_dir["dir"] = _MANIP_DIRS[(i + 1) % len(_MANIP_DIRS)]

    g0_raw = pose.get("gripper_orientation") or pose.get("pitch") or "horizontal"
    g0 = str(g0_raw).lower()
    p_grip = copy.deepcopy(pose)
    flipped = "horizontal" if g0 == "vertical" else "vertical"
    if "pitch" in p_grip and "gripper_orientation" not in pose:
        p_grip["pitch"] = flipped
    else:
        p_grip["gripper_orientation"] = flipped
        p_grip.pop("pitch", None)

    dx, dy, dz = _XYZ_DELTA
    p_xyz = copy.deepcopy(pose)
    ox = float(pose.get("x", 50))
    oy = float(pose.get("y", 50))
    oz = float(pose.get("z", 50))
    p_xyz["x"] = _clamp_pct(ox + dx)
    p_xyz["y"] = _clamp_pct(oy + dy)
    p_xyz["z"] = _clamp_pct(oz + dz)

    return p_dir, p_grip, p_xyz


def _summarize_google(p: dict) -> str:
    parts = [
        f"arm={p.get('arm_position', '?')}",
        f"grip={p.get('gripper_orientation', '?')}",
    ]
    if any(k in p for k in ("x", "y", "z")):
        parts.append(
            "xyz=("
            + ",".join(str(int(p.get(a, 50))) for a in ("x", "y", "z"))
            + ")"
        )
    return "; ".join(parts)


def _summarize_manip(p: dict) -> str:
    g = p.get("gripper_orientation") or p.get("pitch") or "?"
    parts = [
        f"dir={p.get('dir', '?')}",
        f"grip={g}",
    ]
    if any(k in p for k in ("x", "y", "z")):
        parts.append(
            "xyz=("
            + ",".join(str(int(p.get(a, 50))) for a in ("x", "y", "z"))
            + ")"
        )
    return "; ".join(parts)


def _iter_pose_steps(configs: list[dict]) -> Iterable[tuple[int, dict, int, dict]]:
    """Yield (cfg_list_index, cue_config_dict, pose_step_index, pose_dict)."""
    for ci, cue_cfg in enumerate(configs):
        movements = cue_cfg.get("movements") or []
        pi = 0
        for step in movements:
            if step.get("type") != "pose":
                continue
            pose = (step.get("parameters") or {}).get("pose") or {}
            yield ci, cue_cfg, pi, pose
            pi += 1


def _make_google_renderer():
    from adhoc.generation.tiago.legacy.render_mobile_config import (
        CAM,
        QI_FWD,
        QI_SIDE,
        QI_YAW,
        _apply_pose,
        _capture,
        _default_qpos,
        _make_env,
    )

    env = _make_env()
    env.reset()

    def capture(pose: dict) -> Image.Image:
        q = _default_qpos(env)
        _apply_pose(env, q, pose)
        _finalize_tiago_authored_grip_qpos(q, pose)
        q[QI_FWD] = 0.0
        q[QI_SIDE] = 0.0
        q[QI_YAW] = 0.0
        return _capture(env, q, CAM)

    return capture


def _make_manip_renderer():
    repo = dev_robosuite_root()
    jsonl_abs = repo / "data" / "seed" / "_remainder" / "closest_poses_results.jsonl"
    if not jsonl_abs.is_file():
        raise SystemExit(f"Manipulator pose DB missing: {jsonl_abs}")

    # Import heavy stack only after user opts in via --embodiments
    from adhoc.generation.robotarm.legacy.motion_generation_core import (
        MotionGenerator,
        _select_initial_poses,
    )

    mg = MotionGenerator(
        robot_name="IIWA",
        env_name="EmptySpace",
        controller_name="IK_POSE",
        jsonl_path=str(jsonl_abs),
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_distance=1.8,
        hz=4,
    )

    def capture(pose_def: dict) -> Image.Image | None:
        p = copy.deepcopy(pose_def)
        want_grip = str(p.get("gripper_orientation") or p.get("pitch") or "").strip().lower()

        buf_o = io.StringIO()
        buf_e = io.StringIO()
        with (
            contextlib.redirect_stdout(buf_o),
            contextlib.redirect_stderr(buf_e),
        ):
            matches = mg._find_matching_poses(p)
        if want_grip in {"horizontal", "vertical"}:
            filt = [
                row
                for row in matches
                if str(row.get("gripper_orientation") or "").strip().lower() == want_grip
            ]
            if filt:
                matches = filt
        if not matches:
            return None
        picks = _select_initial_poses(matches, p, top_k=1)
        if not picks:
            return None
        pose_row = picks[0]
        buf2_o = io.StringIO()
        buf2_e = io.StringIO()
        with (
            contextlib.redirect_stdout(buf2_o),
            contextlib.redirect_stderr(buf2_e),
        ):
            mg.jacobian_calculator._set_pose_from_data(pose_row)
            arr = mg._capture_image()
        return Image.fromarray(np.asarray(arr))

    return capture


def _ensure_rgba(im: Image.Image) -> Image.Image:
    return im.convert("RGBA") if im.mode != "RGBA" else im


def _failure_tile(msg: str, w: int = 512, h: int = 512) -> Image.Image:
    im = Image.new("RGBA", (w, h), (32, 32, 42, 255))
    try:
        from PIL import ImageDraw, ImageFont
    except Exception:
        return im
    draw = ImageDraw.Draw(im)
    text = msg[:500]
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None  # type: ignore[assignment]
    draw.text((12, h // 2 - 10), text, fill=(255, 107, 107, 255), font=font)
    return im


def _write_html(
    *,
    html_path: Path,
    frames_rel: Path,
    rows: list[dict[str, Any]],
) -> None:
    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8"/>',
        '<meta name="viewport" content="width=device-width,initial-scale=1"/>',
        "<title>Pose review grid</title>",
        "<style>",
        "body{font-family:system-ui,Segoe UI,sans-serif;background:#12141a;color:#e8eaf0;"
        "margin:16px;line-height:1.35;font-size:14px;}",
        "h1{font-size:1.25rem;font-weight:650;margin:0 0 .6rem;}",
        "table{border-collapse:collapse;width:100%;table-layout:fixed;background:#161922;}",
        "th,td{border:1px solid #282d3c;padding:10px 8px;vertical-align:top;}",
        "th{background:#222633;font-weight:600;font-size:.78rem;color:#cfd6ee;}",
        "td.idx{width:132px;background:#161b27;font-size:.8rem;}",
        "td.cell{text-align:center;}",
        ".lab{font-weight:680;font-size:.8rem;color:#cfd6ee;margin:.25rem 0 .15rem;}",
        ".meta{color:#8b93b0;font-size:.72rem;margin:0;}",
        "img{width:min(340px,100%);max-width:340px;height:auto;border-radius:4px;display:block;"
        "margin:0 auto;border:1px solid #363c50;}",
        ".hdr{color:#8b93b0;font-size:.74rem;margin:0 0 .8rem max(0,.2rem);}",
        "</style>",
        "</head><body>",
        "<h1>Pose review grid: GIF · original pose · Δdir · Δgrip · Δxyz</h1>",
        f'<p class="hdr">{html.escape(html_path.parent.name)} / {frames_rel}</p>',
        '<p class="hdr">GIF = full-motion render beside the authored pose PNGs; Δgrip enforces authored H/V '
        '(Tiago elbow-roll + filtered IIWA poses).</p>',
        "<table>",
        "<thead><tr>",
        "<th>#</th><th class='idx'>Index &amp; meta</th>"
        "<th>GIF<br/><small>(cue)</small></th>"
        "<th>Original</th><th>Variation: direction</th><th>Variation: grip</th>"
        "<th>Variation: xyz</th>",
        "</tr></thead>",
        "<tbody>",
    ]

    specs = (
        FrameSpec("orig", "Original", ""),
        FrameSpec("dir", "Δ direction", ""),
        FrameSpec("grip", "Δ gripper orient", ""),
        FrameSpec("xyz", "Δ xyz (percent)", ""),
    )

    for r in rows:
        idx = int(r["global_index"])
        meta_lines = [
            f"<b>{html.escape(str(r['embodiment']))}</b>",
            f"<span class='meta'>cue: {html.escape(str(r['cue']))}</span>",
            f"<span class='meta'>config idx field: {r.get('motion_idx')} | pose step #{r['pose_step_seq']}</span>",
        ]
        parts.append("<tr>")
        parts.append(f"<td>{idx}</td>")
        parts.append("<td class='idx'>" + "<br/>".join(meta_lines) + "</td>")
        mgif = r.get("motion_gif") or {}
        gif_ok = bool(mgif.get("ok"))
        gif_rel = str(mgif.get("relpath") or "").replace("\\", "/")
        parts.append('<td class="cell">')
        if gif_ok and gif_rel:
            gif_b = "#363c50"
            parts.append(
                f'<img src="{html.escape(gif_rel)}" alt="" loading="lazy" '
                f'style="border-color:{gif_b};"/>'
            )
            if mgif.get("subtitle"):
                parts.append(f'<p class="meta">{html.escape(str(mgif["subtitle"]))}</p>')
        else:
            note = mgif.get("note") or "GIF 없음"
            parts.append(f'<p class="meta">{html.escape(str(note))}</p>')
        parts.append("</td>")
        for sp in specs:
            rel = str(r["cells"][sp.key]["relpath"]).replace("\\", "/")
            sub = html.escape(str(r["cells"][sp.key].get("subtitle") or ""))
            lbl = html.escape(sp.label)
            ok = bool(r["cells"][sp.key].get("ok", True))
            border = "#363c50" if ok else "#8b3e3e"
            parts.append('<td class="cell">')
            parts.append(f'<img src="{html.escape(rel)}" style="border-color:{border};"/>')
            parts.append(f'<p class="lab">{lbl}</p>')
            if sub:
                parts.append(f'<p class="meta">{sub}</p>')
            parts.append("</td>")
        parts.append("</tr>")

    parts.extend(["</tbody></table>", "</body></html>"])
    html_path.write_text("\n".join(parts), encoding="utf-8")


def _collect_rows_google(
    configs: list[dict],
    cap_g: Callable[[dict], Image.Image],
    *,
    frames_dir: Path,
    global_start: int,
    max_take: int,
    motion_gif_bundle: Callable[[str, Any, Any], dict],
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    gidx = global_start
    consumed = 0
    def write_cell(key: str, im: Image.Image, fn: Path, subtitle: str, ok: bool) -> dict:
        im.save(fn, optimize=True)
        return {
            "key": key,
            "relpath": str(Path("frames") / fn.name).replace("\\", "/"),
            "subtitle": subtitle,
            "ok": ok,
        }

    for ci, cue_cfg, pseq, pose in _iter_pose_steps(configs):
        if consumed >= max_take:
            break
        pv_dir, pv_grip, pv_xyz = _variation_google(pose)

        cue = cue_cfg.get("cue", "?")
        mid = cue_cfg.get("idx", ci)
        base = (
            _slug(cue, 48)
            + f"_g{gidx:04d}_c{mid}_pi{pseq}"

        )

        originals = []
        errs: dict[str, str] = {}

        originals.append(("orig", pose, _summarize_google(pose)))
        originals.extend(
            [
                ("dir", pv_dir, _summarize_google(pv_dir)),
                ("grip", pv_grip, _summarize_google(pv_grip)),
                ("xyz", pv_xyz, _summarize_google(pv_xyz)),
            ]
        )

        cells: dict[str, dict] = {}
        for tag, pst, subtitle in originals:
            fn = frames_dir / f"{base}_{tag}.png"
            try:
                im = _ensure_rgba(cap_g(pst))
                cells[tag] = write_cell(tag, im, fn, subtitle, ok=True)
            except Exception as e:  # noqa: BLE001
                tiles = _failure_tile(f"{tag}: {e}")
                cells[tag] = write_cell(tag, tiles, fn, subtitle, ok=False)

        rows.append(
            {
                "global_index": gidx,
                "embodiment": "tiago",
                "cue": cue,
                "motion_idx": mid,
                "pose_step_seq": pseq,
                "config_list_index": ci,
                "motion_gif": motion_gif_bundle("tiago", cue, mid),
                "cells": cells,
            }
        )

        consumed += 1
        gidx += 1
    return rows, gidx


def _collect_rows_manipulator(
    configs: list[dict],
    cap_m: Callable[[dict], Image.Image | None],
    *,
    frames_dir: Path,
    global_start: int,
    max_take: int,
    motion_gif_bundle: Callable[[str, Any, Any], dict],
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    gidx = global_start
    consumed = 0

    def render_pair(tag: str, pst: dict, fn: Path, subtitle: str) -> dict:
        try:
            im = cap_m(pst)
            ok = im is not None
            tgt = _ensure_rgba(im) if ok else _failure_tile("no matching pose DB row")
            tgt.save(fn, optimize=True)
        except Exception as e:  # noqa: BLE001
            ok = False
            tgt = _failure_tile(str(e))
            tgt.save(fn, optimize=True)
        return {
            "relpath": str(Path("frames") / fn.name).replace("\\", "/"),
            "subtitle": subtitle,
            "ok": ok,
        }

    for ci, cue_cfg, pseq, pose in _iter_pose_steps(configs):
        if consumed >= max_take:
            break
        pv_dir, pv_grip, pv_xyz = _variation_manip(pose)

        cue = cue_cfg.get("cue", "?")
        mid = cue_cfg.get("idx", ci)

        base = _slug(cue, 48) + f"_g{gidx:04d}_c{mid}_pi{pseq}"
        originals = (
            ("orig", pose, _summarize_manip(pose)),
            ("dir", pv_dir, _summarize_manip(pv_dir)),
            ("grip", pv_grip, _summarize_manip(pv_grip)),
            ("xyz", pv_xyz, _summarize_manip(pv_xyz)),
        )
        cells: dict[str, dict] = {}

        for tag, pst, subtitle in originals:
            fn = frames_dir / f"{base}_{tag}.png"
            cells[tag] = render_pair(tag, pst, fn, subtitle)

        rows.append(
            {
                "global_index": gidx,
                "embodiment": "manipulator",
                "cue": cue,
                "motion_idx": mid,
                "pose_step_seq": pseq,
                "config_list_index": ci,
                "motion_gif": motion_gif_bundle("manipulator", cue, mid),
                "cells": cells,
            }
        )

        consumed += 1
        gidx += 1
    return rows, gidx


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="HTML grid for pose snapshots + variants")
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/results/html/pose_review_grid",
        help="Output folder (writes index.html + frames/) under repo root if relative",
    )
    p.add_argument(
        "--embodiments",
        type=str,
        default="tiago,manipulator",
        help="Comma list: tiago, manipulator",
    )
    p.add_argument(
        "--sources-yml",
        type=str,
        default="",
        help="Optional override path to see_html_sources.yml",
    )
    p.add_argument(
        "--motion-json-google",
        type=str,
        default="",
        help="Optional override motion config JSON for Tiago/mobile",
    )
    p.add_argument(
        "--motion-json-manip",
        type=str,
        default="",
        help="Optional override motion config JSON for manipulator",
    )
    p.add_argument(
        "--shots-only",
        action="store_true",
        help=(
            "Restrict to cues present in embodiment shots catalogs (YAML shots_json paths): "
            "manipulator matches merged cue ``idx`` to shot rows; tiago matches ``cue`` string."
        ),
    )
    p.add_argument(
        "--shots-json-google",
        type=str,
        default="",
        help="With --shots-only: override repo-relative shots JSON for tiago",
    )
    p.add_argument(
        "--shots-json-manipulator",
        type=str,
        default="",
        help="With --shots-only: override repo-relative shots JSON for manipulator",
    )
    p.add_argument(
        "--skip-motion-gifs",
        action="store_true",
        help="Do not bundle cue GIFs into out-dir/gifs/ or fill motion_gif cells (PNG columns only).",
    )
    p.add_argument(
        "--max-google-rows",
        type=int,
        default=0,
        help="Cap Tiago/mobile pose captures (0 = no cap)",
    )
    p.add_argument(
        "--max-manip-rows",
        type=int,
        default=0,
        help="Cap manipulator pose captures (0 = no cap)",
    )
    p.add_argument(
        "--max-total-rows",
        type=int,
        default=0,
        help="Cap rows across embodiments combined (0 = no cap)",
    )
    ns = p.parse_args(argv)

    root = dev_robosuite_root()
    sources_path = Path(ns.sources_yml).resolve() if ns.sources_yml else None

    want = [x.strip() for x in ns.embodiments.split(",") if x.strip()]
    for slug in want:
        if slug not in {"tiago", "manipulator"}:
            raise SystemExit(
                "--embodiments accepts only tiago and manipulator for this script"
            )

    motion_g: Path | None = (
        repo_rel_to_path(ns.motion_json_google, root) if ns.motion_json_google else None
    )
    motion_m: Path | None = (
        repo_rel_to_path(ns.motion_json_manip, root) if ns.motion_json_manip else None
    )

    if motion_g is None and "tiago" in want:
        _, _, mp, _ = resolve_embodiment_paths("tiago", sources_path=sources_path)
        motion_g = mp
    if motion_m is None and "manipulator" in want:
        _, _, mp, _ = resolve_embodiment_paths("manipulator", sources_path=sources_path)
        motion_m = mp

    out_abs = Path(ns.out_dir)
    if not out_abs.is_absolute():
        out_abs = (root / out_abs).resolve()
    frames_dir = out_abs / "frames"
    out_abs.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    include_motion_gifs = not bool(ns.skip_motion_gifs)
    out_gifs_dir = out_abs / "gifs"
    copy_by_canon: dict[str, str] = {}
    render_abs_google: Path | None = None
    render_abs_manipulator: Path | None = None
    if include_motion_gifs:
        out_gifs_dir.mkdir(parents=True, exist_ok=True)
        if "tiago" in want:
            _, _, _, blk_git_g = resolve_embodiment_paths(
                "tiago", sources_path=sources_path
            )
            render_abs_google = repo_rel_to_path(blk_git_g["render_dir"], root)
        if "manipulator" in want:
            _, _, _, blk_git_m = resolve_embodiment_paths(
                "manipulator", sources_path=sources_path
            )
            render_abs_manipulator = repo_rel_to_path(blk_git_m["render_dir"], root)

    def motion_gif_bundle(slug: str, cue_any: Any, mid_any: Any) -> dict:
        rd = render_abs_google if slug == "tiago" else render_abs_manipulator
        return _bundle_motion_gif(
            embodiment_slug=slug,
            cue=str(cue_any),
            motion_idx=mid_any,
            render_dir_abs=rd,
            out_gifs_dir=out_gifs_dir,
            copy_by_canon=copy_by_canon,
            enabled=include_motion_gifs,
        )

    meta_path = out_abs / "review_rows_meta.json"

    configs_g = (
        json.loads(motion_g.read_text(encoding="utf-8"))
        if motion_g and motion_g.is_file()
        else []
    )
    configs_m = (
        json.loads(motion_m.read_text(encoding="utf-8"))
        if motion_m and motion_m.is_file()
        else []
    )

    if ns.shots_only:
        if "tiago" in want:
            if str(ns.shots_json_google).strip():
                sj_g = repo_rel_to_path(ns.shots_json_google.strip(), root)
            else:
                _, sj_g, _, _ = resolve_embodiment_paths("tiago", sources_path=sources_path)
            if not sj_g.is_file():
                raise SystemExit(f"--shots-only: tiago shots file missing: {sj_g}")
            shots_g_rows = json.loads(sj_g.read_text(encoding="utf-8"))
            if not isinstance(shots_g_rows, list):
                raise SystemExit("--shots-only: google shots JSON must be a list")
            configs_g, note_g = filter_motion_configs_by_shots_only(
                configs_g, slug="tiago", shots_rows=shots_g_rows
            )
            print(f"[shots_only] shots file: {sj_g}")
            print(f"[shots_only] {note_g}")

        if "manipulator" in want:
            if str(ns.shots_json_manipulator).strip():
                sj_m = repo_rel_to_path(ns.shots_json_manipulator.strip(), root)
            else:
                _, sj_m, _, _ = resolve_embodiment_paths("manipulator", sources_path=sources_path)
            if not sj_m.is_file():
                raise SystemExit(f"--shots-only: manipulator shots file missing: {sj_m}")
            shots_m_rows = json.loads(sj_m.read_text(encoding="utf-8"))
            if not isinstance(shots_m_rows, list):
                raise SystemExit("--shots-only: manipulator shots JSON must be a list")
            configs_m, note_m = filter_motion_configs_by_shots_only(
                configs_m, slug="manipulator", shots_rows=shots_m_rows
            )
            print(f"[shots_only] shots file: {sj_m}")
            print(f"[shots_only] {note_m}")

        if configs_g == [] and "tiago" in want:
            print("[shots_only][warn] tiago: zero cue configs after filter")
        if configs_m == [] and "manipulator" in want:
            print("[shots_only][warn] manipulator: zero cue configs after filter")

    n_pose_g = _count_pose_steps(configs_g)
    n_pose_m = _count_pose_steps(configs_m)
    print(
        f"Pose steps in inputs: tiago={n_pose_g}, manipulator={n_pose_m} "
        f"(estimated PNGs rows×4={(n_pose_g + n_pose_m) * 4} if uncapped)."
    )

    cap_g: Callable[[dict], Image.Image] | None = None
    cap_m: Callable[[dict], Image.Image | None] | None = None

    rows_all: list[dict[str, Any]] = []
    gcursor = 0
    remaining = _budget(int(ns.max_total_rows))

    if "tiago" in want and remaining > 0:
        take = min(_budget(int(ns.max_google_rows)), remaining)
        if take <= 0:
            pass
        else:
            cap_g = _make_google_renderer()
            rg, _next = _collect_rows_google(
                configs_g,
                cap_g,
                frames_dir=frames_dir,
                global_start=gcursor,
                max_take=take,
                motion_gif_bundle=motion_gif_bundle,
            )
            rows_all.extend(rg)
            gcursor = _next
            remaining -= len(rg)

    if "manipulator" in want and remaining > 0:
        take = min(_budget(int(ns.max_manip_rows)), remaining)
        if take <= 0:
            pass
        else:
            cap_m = _make_manip_renderer()
            rm, _ = _collect_rows_manipulator(
                configs_m,
                cap_m,
                frames_dir=frames_dir,
                global_start=gcursor,
                max_take=take,
                motion_gif_bundle=motion_gif_bundle,
            )
            rows_all.extend(rm)

    meta_path.write_text(json.dumps(rows_all, indent=2), encoding="utf-8")

    html_abs = out_abs / "index.html"
    _write_html(
        html_path=html_abs,
        frames_rel=Path("frames"),
        rows=rows_all,
    )

    print(f"Wrote {len(rows_all)} rows → {html_abs}")
    print(f"Frames dir: {frames_dir}")
    print(f"Row metadata JSON: {meta_path}")

    if motion_g:
        print(f"Google motion configs: {motion_g}")
    if motion_m:
        print(f"Manipulator motion configs: {motion_m}")


if __name__ == "__main__":
    os.chdir(dev_robosuite_root())
    try:
        main()
    finally:
        sys.stdout.flush()
