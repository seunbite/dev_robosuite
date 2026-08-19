"""Render top-K pose-bank candidates (dir+orient → xyz distance) as a tile image."""
from __future__ import annotations

import copy
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from legacy.render_mobile_config import (  # noqa: E402
    HEAD_REF_BODY,
    L_EE_BODY,
    QI_HEAD_PAN,
    QI_HEAD_TILT,
    QI_L_ARM,
    QI_R_ARM,
    QI_TORSO,
    HEAD_PRESETS,
    L_ARM_REST,
    R_EE_BODY,
    R_ARM_REST,
    TORSO_MAP,
    _arm_rad,
    _capture,
    _default_qpos,
    _gesture_ee_body,
    _mirror_offset,
    rank_pose_bank_topk,
    list_pose_bank_candidates,
    render_config,
    pose_xyz_axes_specified,
    _resolve_level,
)

_MOTION_GRID_FPS_MS = 50
_MAX_MOTION_POOL_TILES = 15


def _pose_entry_xyz_deltas(pose: dict[str, Any], ent: dict[str, Any]) -> dict[str, float | None]:
    """Per-axis sel−req for axes explicitly set in pose; ``None`` when axis omitted."""
    out: dict[str, float | None] = {}
    for axis in "xyz":
        if axis not in pose_xyz_axes_specified(pose):
            out[axis] = None
            continue
        req = _resolve_level(pose[axis], axis)
        out[axis] = float(ent.get(axis, 50)) - req
    return out


def _format_xyz_delta_line(pose: dict[str, Any], ent: dict[str, Any]) -> str:
    """One-line xyz delta summary for a pool tile."""
    deltas = _pose_entry_xyz_deltas(pose, ent)
    parts: list[str] = []
    for axis in "xyz":
        d = deltas[axis]
        if d is None:
            continue
        parts.append(f"Δ{axis}{d:+.0f}")
    if parts:
        return " ".join(parts)
    return f"x{float(ent.get('x', 50)):.0f} y{float(ent.get('y', 50)):.0f} z{float(ent.get('z', 50)):.0f}"


def _tile_label_lines(pose: dict[str, Any], ent: dict[str, Any]) -> list[str]:
    """Header lines for a pose-pool tile (rank, tile id, xyz deltas)."""
    rank = int(ent.get("rank", 0))
    tile = ent.get("tile_idx", -1)
    tile_s = f"tile={tile}" if int(tile) >= 0 else "tile=?"
    line1 = f"#{rank} {tile_s}"
    line2 = (
        f"x{float(ent.get('x', 50)):.0f} y{float(ent.get('y', 50)):.0f} "
        f"z{float(ent.get('z', 50)):.0f}"
    )
    delta_s = _format_xyz_delta_line(pose, ent)
    if delta_s.startswith("Δ"):
        line3 = delta_s
    else:
        line3 = f"(no xyz req) {delta_s}"
    return [line1, line2, line3]


def _stamp_tile_label(im: Image.Image, lines: list[str], *, bar_at_top: bool = True) -> Image.Image:
    """Draw semi-opaque label bar on a tile image."""
    im = im.copy().convert("RGB")
    if not lines:
        return im
    dr = ImageDraw.Draw(im)
    line_h = 13
    bar_h = line_h * len(lines) + 8
    if bar_at_top:
        y0, y_text = 0, 4
        dr.rectangle([0, y0, im.width, y0 + bar_h], fill=(0, 0, 0))
    else:
        y0 = im.height - bar_h
        y_text = y0 + 4
        dr.rectangle([0, y0, im.width, im.height], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        dr.text((4, y_text + i * line_h), line, fill="white", font=_font(10))
    return im


def first_pose_from_row(row: dict[str, Any]) -> dict[str, Any]:
    for step in row.get("movements", []):
        if step.get("type") == "pose":
            return (step.get("parameters") or {}).get("pose") or {}
    return {}


def first_pool_pose_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """First non-still pose step, else the first pose (e.g. nod_yes → still)."""
    first: dict[str, Any] | None = None
    for step in row.get("movements") or []:
        if step.get("type") != "pose":
            continue
        pose = (step.get("parameters") or {}).get("pose") or {}
        if not (pose.get("arm_position") and pose.get("gripper_orientation")):
            continue
        if first is None:
            first = pose
        if str(pose.get("arm_position", "")).strip().lower() != "still":
            return pose
    return first or {}


def _pool_uses_main_gif_copy(row: dict[str, Any]) -> bool:
    return str(first_pool_pose_from_row(row).get("arm_position", "")).strip().lower() == "still"


def _apply_pose_template(q, pose: dict[str, Any], r_arm_rad: list[float]) -> None:
    from adhoc.generation.tiago.mobile_pose_normalize import normalize_pose_dict  # noqa: WPS433

    pose = normalize_pose_dict(pose)
    torso = pose.get("torso_height", "mid")
    q[QI_TORSO] = TORSO_MAP.get(torso, 0.18)
    q[QI_R_ARM] = r_arm_rad

    arm = pose.get("arm_position", "down+right")
    grip = pose.get("gripper_orientation", "vertical")
    left = pose.get("left_arm", "still")
    if left == "still" and str(arm).strip().lower() == "still":
        q[QI_R_ARM] = list(r_arm_rad)
        q[QI_L_ARM] = L_ARM_REST[:]
    elif left == "still":
        q[QI_L_ARM] = L_ARM_REST[:]
    elif left == "mirror":
        l_rad = list(r_arm_rad)
        _mirror_offset(l_rad, arm)
        q[QI_L_ARM] = l_rad
    else:
        q[QI_L_ARM] = _arm_rad(left, grip)

    head = pose.get("head", "center")
    hp = HEAD_PRESETS.get(head, [0, 0])
    q[QI_HEAD_PAN] = hp[0]
    q[QI_HEAD_TILT] = hp[1]


def _hand_head_world_x(env, pose: dict[str, Any]) -> tuple[float, float, str]:
    """Return (hand_x, head_x, ee_body) after current sim state."""
    sim = env.sim
    ee_body = _gesture_ee_body(pose)
    head_x = float(sim.data.body_xpos[sim.model.body_name2id(HEAD_REF_BODY)][0])
    hand_x = float(sim.data.body_xpos[sim.model.body_name2id(ee_body)][0])
    return hand_x, head_x, ee_body


def _font(size: int = 14) -> ImageFont.ImageFont:
    for name in ("/System/Library/Fonts/Helvetica.ttc", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _crop_square_bottom_viewport(img: Image.Image, *, top_frac: float) -> Image.Image:
    """Trim sky, then take a bottom-anchored centered square (Task A paper-figure style)."""
    w, h = img.size
    y_min = max(0, min(h - 1, int(h * top_frac)))
    side = min(w, h - y_min)
    if side < 2:
        return img
    x0 = (w - side) // 2
    y0 = max(y_min, h - side)
    return img.crop((x0, y0, x0 + side, y0 + side))


def _fit_panel_thumb(im: Image.Image, cell: int, *, crop_top_frac: float = 0.3) -> Image.Image:
    """Crop top margin, center square, resize to cell (matches Task A panel geometry)."""
    return _crop_square_bottom_viewport(im, top_frac=crop_top_frac).resize(
        (cell, cell), Image.Resampling.LANCZOS
    )


def _cue_footer_font(size: int = 20) -> ImageFont.ImageFont:
    return _font(size)


def _draw_centered_cue_label(
    dr: ImageDraw.ImageDraw,
    *,
    cue: str,
    x0: int,
    col_w: int,
    y: int,
    font: ImageFont.ImageFont,
) -> None:
    text = cue
    while len(text) > 3 and dr.textbbox((0, 0), text, font=font)[2] > col_w:
        text = text[:-4] + "…"
    bbox = dr.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    dr.text((x0 + (col_w - tw) // 2, y), text, fill=(0, 0, 0), font=font)


def _joint_dist2(a: list[float], b: list[float]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))


def _choose_most_different_non_pick(
    entries: list[dict[str, Any]],
    pick_ent: dict[str, Any],
    pick_rank: int,
) -> dict[str, Any] | None:
    """Pick the non-selected pool pose farthest from the VLM pick in joint space."""
    pick_rad = pick_ent.get("r_arm_rad")
    if not pick_rad:
        for ent in entries:
            if int(ent.get("rank", -1)) != pick_rank:
                return ent
        return None
    best: dict[str, Any] | None = None
    best_d = -1.0
    for ent in entries:
        if int(ent.get("rank", -1)) == pick_rank:
            continue
        rad = ent.get("r_arm_rad")
        if not rad:
            continue
        d = _joint_dist2(pick_rad, rad)
        d += (
            (float(ent.get("x", 0)) - float(pick_ent.get("x", 0))) ** 2
            + (float(ent.get("y", 0)) - float(pick_ent.get("y", 0))) ** 2
            + (float(ent.get("z", 0)) - float(pick_ent.get("z", 0))) ** 2
        ) * 0.02
        if d > best_d:
            best_d = d
            best = ent
    return best


SCORE_MARK_COLORS: dict[str, tuple[int, int, int]] = {
    "ok": (34, 180, 34),
    "warn": (234, 179, 8),
    "bad": (210, 35, 35),
}


def _draw_corner_score_mark(im: Image.Image, *, mark: str) -> Image.Image:
    """Top-right score circle (Task A / qual_generation style): ok=green, warn=yellow, bad=red."""
    color = SCORE_MARK_COLORS.get(mark)
    if not color:
        return im
    out = im.copy()
    dr = ImageDraw.Draw(out)
    side = min(out.width, out.height)
    r = max(6, int(side * 0.045))
    margin = max(5, int(side * 0.03))
    cx = out.width - margin - r
    cy = margin + r
    dr.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    return out


def render_uniform_pose_thumb(
    env,
    pose: dict[str, Any],
    entry: dict[str, Any],
    *,
    score_mark: str | None = None,
    rank_label: int | None = None,
) -> Image.Image:
    """Single pool pose frame with optional rank label and top-right score circle."""
    q = _default_qpos(env)
    _apply_pose_template(q, pose, entry["r_arm_rad"])
    im = _capture(env, q).convert("RGB")
    if rank_label is not None:
        dr = ImageDraw.Draw(im)
        label = str(rank_label)
        tw, th = dr.textbbox((0, 0), label, font=_font(16))[2:]
        pad = 4
        x0, y0 = im.width - tw - pad * 2 - 4, 4
        dr.rectangle([x0, y0, x0 + tw + pad * 2, y0 + th + pad * 2], fill=(0, 0, 0))
        dr.text((x0 + pad, y0 + pad), label, fill="white", font=_font(16))
    if score_mark:
        im = _draw_corner_score_mark(im, mark=score_mark)
    return im


def build_pick_vs_nonpick_montage(
    env,
    *,
    results: list[dict[str, Any]],
    rows_by_cue: dict[str, dict[str, Any]],
    out_path: Path,
    cell_size: int = 256,
    pad: int = 4,
    crop_top_frac: float = 0.3,
    footer_h: int = 32,
    cue_font_size: int = 20,
    pick_score_marks: dict[str, str] | None = None,
    nonpick_score_marks: dict[str, str] | None = None,
    subset: list[str] | None = None,
    group_order: tuple[str, ...] = ("ok", "warn", "bad"),
) -> Path:
    """n columns × 2 rows: row1 = VLM pick, row2 = maximally different non-pick."""
    cols: list[dict[str, Any]] = []
    for rec in sorted(results, key=lambda r: int(r.get("idx", 0))):
        cue = str(rec.get("cue", ""))
        if rec.get("error") or not cue:
            continue
        n_tiles = int(rec.get("n_tiles", 0))
        if n_tiles <= 1 or rec.get("vlm_skipped") == "single_candidate":
            continue
        row = rows_by_cue.get(cue)
        if not row:
            continue
        pose = first_pose_from_row(row)
        if not pose:
            continue
        pick_rank = rec.get("vlm_pick_index")
        if pick_rank is None:
            continue
        pick_rank = int(pick_rank)
        pool_meta = rec.get("pool_meta") or {}
        ranked = list_pose_bank_candidates(
            env,
            pose,
            top_k=0,
            plausible_only=True,
            relax_elbow_if_empty=bool(pool_meta.get("elbow_bended_relaxed")),
        )
        entries = ranked.get("entries") or []
        if len(entries) <= 1:
            continue
        pick_ent = rec.get("picked_entry")
        if not pick_ent:
            pick_ent = next((e for e in entries if int(e.get("rank", -1)) == pick_rank), entries[0])
        non_ent = _choose_most_different_non_pick(entries, pick_ent, pick_rank)
        if not pick_ent or not non_ent:
            continue
        cols.append(
            {
                "cue": cue,
                "pose": pose,
                "pick_ent": pick_ent,
                "non_ent": non_ent,
                "pick_rank": pick_rank,
            }
        )

    if not cols:
        raise RuntimeError("No Task B results with multi-tile pools to montage")

    if subset is not None:
        want = {str(c) for c in subset}
        cols = [c for c in cols if str(c.get("cue", "")) in want]
        if not cols:
            raise RuntimeError("Subset selection removed all montage columns")

    # order by traffic-light group: ok > warn > bad (stable within each group)
    if pick_score_marks:
        pri = {k: i for i, k in enumerate(group_order)}

        def _key(c: dict[str, Any]) -> tuple[int, int]:
            cue = str(c.get("cue", ""))
            mark = str(pick_score_marks.get(cue, "ok"))
            return (pri.get(mark, 999), int(c.get("_orig_i", 0)))

        for i, c in enumerate(cols):
            c["_orig_i"] = i
        cols = sorted(cols, key=_key)

    n = len(cols)
    cw = ch = cell_size
    cue_font = _cue_footer_font(cue_font_size)
    canvas_w = pad + n * (cw + pad)
    canvas_h = pad + 2 * (ch + pad) + footer_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)

    for i, col in enumerate(cols):
        cue = str(col["cue"])
        pick_mark = (pick_score_marks or {}).get(cue, "ok")
        pick_im = render_uniform_pose_thumb(env, col["pose"], col["pick_ent"])
        non_im = render_uniform_pose_thumb(env, col["pose"], col["non_ent"])
        x0 = pad + i * (cw + pad)
        y_top = pad
        y_bot = y_top + ch + pad
        pick_cell = _fit_panel_thumb(pick_im, cw, crop_top_frac=crop_top_frac)
        if pick_mark:
            pick_cell = _draw_corner_score_mark(pick_cell, mark=pick_mark)
        canvas.paste(pick_cell, (x0, y_top))
        non_cell = _fit_panel_thumb(non_im, cw, crop_top_frac=crop_top_frac)
        non_mark = (nonpick_score_marks or {}).get(cue)
        if non_mark:
            non_cell = _draw_corner_score_mark(non_cell, mark=non_mark)
        canvas.paste(non_cell, (x0, y_bot))
        _draw_centered_cue_label(
            dr, cue=cue, x0=x0, col_w=cw, y=y_bot + ch + 6, font=cue_font
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def render_topk_tile(
    env,
    pose: dict[str, Any],
    out_path: Path,
    *,
    top_k: int = 30,
    title: str | None = None,
    gemini_rank: int | None = None,
) -> dict[str, Any]:
    """Save a tile grid PNG; return rank_pose_bank_topk metadata."""
    ranked = rank_pose_bank_topk(env, pose, top_k=top_k)
    return _render_pose_candidate_grid(
        env,
        pose,
        out_path,
        ranked,
        title=title,
        gemini_rank=gemini_rank,
    )


def render_plausible_pool_tile(
    env,
    pose: dict[str, Any],
    out_path: Path,
    *,
    title: str | None = None,
    gemini_rank: int | None = None,
    uniform_tiles: bool = False,
    no_header: bool = False,
    plausible_only: bool = False,
    relax_elbow_if_empty: bool = False,
    ignore_xyz: bool = False,
    topk: int | None = None,
    row: dict[str, Any] | None = None,
    overlay_progress_bar: bool = True,
    main_gif_path: Path | None = None,
) -> dict[str, Any]:
    """Render pose pool: static PNG grid, or full-motion tiled GIF when ``row`` + ``.gif``."""
    if out_path.suffix.lower() == ".gif" and row is not None:
        if _pool_uses_main_gif_copy(row):
            if main_gif_path is not None and main_gif_path.is_file():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(main_gif_path, out_path)
                return {"mode": "copy_main_gif", "arm_position": "still"}
            raise FileNotFoundError(f"main GIF missing for still pose pool: {main_gif_path}")
        return render_pose_pool_motion_grid(
            env,
            row,
            out_path,
            ignore_xyz=ignore_xyz,
            topk=topk,
            overlay_progress_bar=overlay_progress_bar,
            main_gif_path=main_gif_path,
        )
    pool_top_k = 0 if ignore_xyz and topk is None else topk
    ranked = list_pose_bank_candidates(
        env,
        pose,
        top_k=pool_top_k,
        plausible_only=plausible_only,
        relax_elbow_if_empty=relax_elbow_if_empty,
        ignore_xyz=ignore_xyz,
    )
    return _render_pose_candidate_grid(
        env,
        pose,
        out_path,
        ranked,
        title=title,
        gemini_rank=gemini_rank,
        pool_mode=True,
        uniform_tiles=uniform_tiles,
        no_header=no_header,
    )


def _index_of_pool_pose_step(row: dict[str, Any]) -> int | None:
    """Movement index of the pose step used for pose-bank pool filtering."""
    first_pose_idx: int | None = None
    for i, step in enumerate(row.get("movements") or []):
        if step.get("type") != "pose":
            continue
        pose = (step.get("parameters") or {}).get("pose") or {}
        if not (pose.get("arm_position") and pose.get("gripper_orientation")):
            continue
        if first_pose_idx is None:
            first_pose_idx = i
        if str(pose.get("arm_position", "")).strip().lower() != "still":
            return i
    return first_pose_idx


def _config_with_pinned_pose_entry(row: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(row)
    target_i = _index_of_pool_pose_step(row)
    if target_i is None:
        return cfg
    for i, step in enumerate(cfg.get("movements") or []):
        if i != target_i or step.get("type") != "pose":
            continue
        pose = (step.setdefault("parameters", {})).setdefault("pose", {})
        tile_idx = entry.get("tile_idx")
        if tile_idx is not None and int(tile_idx) >= 0:
            pose["plausible_tile_idx"] = int(tile_idx)
        else:
            pose["pinned_r_arm_rad"] = [float(x) for x in entry["r_arm_rad"]]
        break
    return cfg


def _compose_plain_motion_strip(thumbs: list[Image.Image]) -> Image.Image:
    """Horizontally concatenate motion frames with no labels, borders, or padding."""
    if not thumbs:
        return Image.new("RGB", (1, 1))
    thumbs = [t.convert("RGB") for t in thumbs]
    cw, ch = thumbs[0].size
    canvas = Image.new("RGB", (cw * len(thumbs), ch))
    for i, im in enumerate(thumbs):
        canvas.paste(im, (i * cw, 0))
    return canvas


def _load_gif_rgb_frames(gif_path: Path) -> list[Image.Image]:
    """Load all frames from a GIF as RGB images."""
    out: list[Image.Image] = []
    with Image.open(gif_path) as im:
        for i in range(getattr(im, "n_frames", 1)):
            im.seek(i)
            out.append(im.convert("RGB"))
    return out


def render_pose_pool_motion_grid(
    env,
    row: dict[str, Any],
    out_path: Path,
    *,
    ignore_xyz: bool = True,
    topk: int | None = None,
    overlay_progress_bar: bool = False,
    max_tiles: int = _MAX_MOTION_POOL_TILES,
    main_gif_path: Path | None = None,
) -> dict[str, Any]:
    """Render full cue motion once per pose-bank candidate; sync-play in a tile grid GIF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pose = first_pool_pose_from_row(row)
    pool_top_k = 0 if topk is None else topk
    ranked = list_pose_bank_candidates(env, pose, top_k=pool_top_k, ignore_xyz=ignore_xyz)
    entries = list(ranked.get("entries") or [])
    if len(entries) == 1 and main_gif_path is not None and main_gif_path.is_file():
        shutil.copy2(main_gif_path, out_path)
        return {**ranked, "mode": "copy_main_gif", "reason": "single_tile"}
    if not entries:
        if main_gif_path is not None and main_gif_path.is_file():
            shutil.copy2(main_gif_path, out_path)
            return {**ranked, "mode": "copy_main_gif", "reason": "empty_pool"}
        img = Image.new("RGB", (640, 360), (0, 0, 0))
        _save_image_or_gif([img], out_path)
        return ranked

    if len(entries) > max_tiles:
        ranked = {**ranked, "entries": entries[:max_tiles], "n_truncated": len(entries) - max_tiles}
        entries = entries[:max_tiles]

    main_frames: list[Image.Image] | None = None
    if main_gif_path is not None and main_gif_path.is_file():
        try:
            main_frames = _load_gif_rgb_frames(main_gif_path)
        except Exception:
            main_frames = None

    cell_frames: list[list[Image.Image]] = []
    for rank_i, ent in enumerate(entries):
        if rank_i == 0 and main_frames:
            cell_frames.append(main_frames)
            continue
        cfg = _config_with_pinned_pose_entry(row, ent)
        raw = render_config(
            cfg,
            env=env,
            overlay_progress_bar=overlay_progress_bar,
            ignore_xyz=ignore_xyz,
        )
        cell_frames.append(
            [_stamp_tile_label(f, _tile_label_lines(pose, ent)) for f in raw]
        )

    max_len = max(len(fs) for fs in cell_frames)
    grid_frames: list[Image.Image] = []
    for t in range(max_len):
        thumbs = [fs[min(t, len(fs) - 1)] for fs in cell_frames]
        grid_frames.append(_compose_plain_motion_strip(thumbs))

    _save_image_or_gif(grid_frames, out_path, duration_ms=_MOTION_GRID_FPS_MS)
    ranked["motion_grid_frames"] = max_len
    ranked["motion_grid_tiles"] = len(entries)
    if main_frames is not None and entries:
        ranked["main_gif_tile0"] = True
    return ranked


def _save_image_or_gif(
    images: list[Image.Image],
    out_path: Path,
    *,
    duration_ms: int = 500,
) -> None:
    if not images:
        return
    if out_path.suffix.lower() == ".gif":
        if str(_HERE.parents[1]) not in sys.path:
            sys.path.insert(0, str(_HERE.parents[1]))
        from gif_encode import save_gif_frames

        save_gif_frames(images, out_path, duration_ms=duration_ms)
    else:
        images[0].save(out_path)


def _compose_grid_canvas(
    thumbs: list[Image.Image],
    ranked: dict[str, Any],
    *,
    title: str | None,
    gemini_rank: int | None,
    pool_mode: bool,
    no_header: bool,
    highlight_idx: int | None = None,
) -> Image.Image:
    cols = min(6, max(1, len(thumbs))) if len(thumbs) <= 6 else 6
    rows = math.ceil(len(thumbs) / cols)
    cw, ch = thumbs[0].size
    pad = 6
    header = 0 if no_header else 88
    canvas_w = pad + cols * (cw + pad)
    canvas_h = header + pad + rows * (ch + pad)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (248, 248, 252))
    if not no_header:
        dr = ImageDraw.Draw(canvas)
        tgt = ranked.get("target_xyz")
        if tgt:
            tgt_s = "(" + ",".join("—" if v is None else f"{v:g}" for v in tgt) + ")"
        else:
            tgt_s = "—"
        eb_filter = ranked.get("elbow_bended")
        eb_hdr = (
            f"  elbow_bended={eb_filter}"
            if eb_filter is not None
            else "  elbow_bended=any"
        )
        if pool_mode:
            hdr = title or (
                f"plausible pool ({len(thumbs)}/{ranked.get('n_pool', len(thumbs))})  "
                f"{ranked.get('arm_position')}+{ranked.get('gripper_orientation')}{eb_hdr}  "
                f"target xyz={tgt_s}  source={ranked.get('source', '?')}"
            )
            legend = "★xyz = closest to target xyz (used for motion render)  |  sorted by Σ|Δ|"
        else:
            hdr = title or (
                f"top-{len(thumbs)}  bank {ranked.get('bank_dir')}+{ranked.get('bank_orient')}  "
                f"target xyz={tgt_s}  pool={ranked.get('n_pool')} gap_ok={ranked.get('n_hand_ok')}"
            )
            legend = "★xyz = rank #1  |  ◆VLM = Gemini pick  |  filtered: hand_x-head_x ≥ 0.2 only"
        dr.text((10, 8), hdr, fill=(20, 20, 40), font=_font(14))
        if pool_mode:
            dr.text((10, 32), legend, fill=(60, 60, 80), font=_font(12))
            dr.text(
                (10, 54),
                "All tiles match dir + orient (+ elbow_bended when set in config pose).",
                fill=(60, 60, 80),
                font=_font(12),
            )
        else:
            dr.text((10, 32), legend, fill=(60, 60, 80), font=_font(12))
            if gemini_rank is not None:
                dr.text(
                    (10, 54),
                    f"Gemini selected tile #{gemini_rank}",
                    fill=(180, 40, 140),
                    font=_font(12),
                )

    dr = ImageDraw.Draw(canvas)
    for i, im in enumerate(thumbs):
        rr, cc = divmod(i, cols)
        x0 = pad + cc * (cw + pad)
        y0 = header + pad + rr * (ch + pad)
        canvas.paste(im, (x0, y0))
        if highlight_idx is not None and i == highlight_idx:
            dr.rectangle(
                [x0 - 2, y0 - 2, x0 + cw + 1, y0 + ch + 1],
                outline=(255, 200, 40),
                width=5,
            )
    return canvas


def _render_pose_candidate_grid(
    env,
    pose: dict[str, Any],
    out_path: Path,
    ranked: dict[str, Any],
    *,
    title: str | None = None,
    gemini_rank: int | None = None,
    pool_mode: bool = False,
    uniform_tiles: bool = False,
    no_header: bool = False,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not ranked.get("entries"):
        note = ranked.get("arm_position") == "still" and "arm_position=still" or "no bank matches"
        img = Image.new("RGB", (640, 120), (248, 248, 252))
        dr = ImageDraw.Draw(img)
        dr.text((12, 12), title or "pose bank top-K", fill=(20, 20, 40), font=_font(16))
        dr.text((12, 48), f"({note})", fill=(80, 80, 100), font=_font(13))
        _save_image_or_gif([img], out_path)
        return ranked

    thumbs: list[Image.Image] = []
    for ent in ranked["entries"]:
        q = _default_qpos(env)
        _apply_pose_template(q, pose, ent["r_arm_rad"])
        im = _capture(env, q).convert("RGB")
        rank = int(ent["rank"])
        if uniform_tiles:
            dr = ImageDraw.Draw(im)
            label = str(rank)
            tw, th = dr.textbbox((0, 0), label, font=_font(16))[2:]
            pad = 4
            x0, y0 = im.width - tw - pad * 2 - 4, 4
            dr.rectangle([x0, y0, x0 + tw + pad * 2, y0 + th + pad * 2], fill=(0, 0, 0))
            dr.text((x0 + pad, y0 + pad), label, fill="white", font=_font(16))
            if gemini_rank is not None and rank == gemini_rank:
                dr.rectangle([2, 2, im.width - 3, im.height - 3], outline=(255, 80, 220), width=5)
            thumbs.append(im)
            continue
        hand_x, head_x, ee_body = _hand_head_world_x(env, pose)
        gap_min = float(os.environ.get("MOBILE_POSE_HEAD_HAND_X_GAP", "0.2"))
        gap = hand_x - head_x
        gap_ok = gap >= gap_min
        dr = ImageDraw.Draw(im)
        bar_h = 70
        dr.rectangle([0, 0, im.width, bar_h], fill=(0, 0, 0))
        is_xyz1 = bool(ent.get("selected"))
        is_vlm = gemini_rank is not None and rank == gemini_rank
        tags = []
        if is_xyz1:
            tags.append("★xyz")
        if is_vlm:
            tags.append("◆VLM")
        tag_s = (" " + " ".join(tags)) if tags else ""
        label_color = (255, 120, 255) if is_vlm else ((120, 255, 120) if is_xyz1 else "white")
        side = "R" if ee_body == R_EE_BODY else "L"
        tile_s = f" tile={ent['tile_idx']}" if ent.get("tile_idx", -1) >= 0 else ""
        eb_s = (
            f" eb={'T' if ent.get('elbow_bended') else 'F'}"
            if ent.get("elbow_bended") is not None
            else ""
        )
        dist = ent.get("dist")
        dist_s = f"  Σ|Δ|={float(dist):.1f}" if dist is not None else ""
        dr.text(
            (6, 4),
            f"#{rank}{tag_s}{tile_s}{eb_s}{dist_s}",
            fill=label_color,
            font=_font(13),
        )
        dr.text(
            (6, 22),
            f"bank x{ent['x']:.0f} y{ent['y']:.0f} z{ent['z']:.0f}  {_format_xyz_delta_line(pose, ent)}",
            fill=(200, 200, 200),
            font=_font(11),
        )
        hx_color = (255, 100, 100) if not gap_ok else (140, 255, 160)
        dr.text(
            (6, 40),
            f"{side}_hand_x={hand_x:.3f}  head_x={head_x:.3f}  Δ={gap:.3f}"
            + ("" if gap_ok else "  FAIL"),
            fill=hx_color,
            font=_font(11),
        )
        if is_xyz1:
            dr.rectangle([0, 0, im.width - 1, im.height - 1], outline=(80, 220, 80), width=3)
        if is_vlm:
            dr.rectangle([2, 2, im.width - 3, im.height - 3], outline=(255, 80, 220), width=5)
        thumbs.append(im)

    as_gif = out_path.suffix.lower() == ".gif"
    if as_gif:
        frames = [
            _compose_grid_canvas(
                thumbs,
                ranked,
                title=title,
                gemini_rank=gemini_rank,
                pool_mode=pool_mode,
                no_header=no_header,
                highlight_idx=i,
            )
            for i in range(len(thumbs))
        ]
        _save_image_or_gif(frames, out_path, duration_ms=500)
        return ranked

    canvas = _compose_grid_canvas(
        thumbs,
        ranked,
        title=title,
        gemini_rank=gemini_rank,
        pool_mode=pool_mode,
        no_header=no_header,
        highlight_idx=None,
    )
    canvas.save(out_path)
    return ranked
