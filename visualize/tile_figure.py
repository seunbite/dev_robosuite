#!/usr/bin/env python3
"""
Build a tiled PNG figure: one panel per cue using ``adhoc/vlm_test/testset_utils.py``
visualization (e.g. ``alpha_frame_trajectory``).

Edit ``CUE_NAMES`` at the top: non-empty = only those cues (order preserved); empty = all cues
from the motion config(s) resolved for this embodiment.

**Default config + GIF roots** match ``see_html.py``: read
``data/seed/yml/see_html_sources.yml`` → embodiment block (``manipulator`` |
``google_robot`` | ``quadruped``) for ``motion_config_json`` (string or list)
and ``render_dir``. That file is the canonical pointer to “current” bundles —
not a glob by mtime under ``motion_configs/``.

Per-cue panel PNGs (after ``prepare_test_media`` for non-trajectory modes; optional
horizontal top_k tile crop **only** for ``alpha_frame``) are cached under
``data/results/visualize/_panel_cache/`` so re-runs skip work unless inputs change
(see ``--force``, ``--no-cache``).

Trajectory types (``alpha_frame_trajectory``, ``first_frame_trajectory``,
``middle_frame_trajectory``) always use the real simulator (default sim robots:
IIWA / Tiago / Go2 by embodiment; override with ``--sim-robot``): ``get_sim_bundle``
numpy frames, alpha motion stack, and ``_image_with_ee_path`` (yellow→purple) in
``testset_utils``—no GIF-based stack, no silent fallback to plain ``alpha_frame``.
Quadruped runs use ``QuadrupedMotionGenerator``; arm-style runs use ``MotionGenerator``.
Sim reruns are reduced by ``_sim_bundles`` disk cache and the panel PNG cache; optional
``--save-sim-meta`` writes screen-space path x/y and frame indices next to the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# -----------------------------------------------------------------------------
# Cue filter: leave empty to include every cue in the latest motion_configs file.
# If non-empty, only these exact ``cue`` strings are included (YAML cue keys).
# -----------------------------------------------------------------------------
CUE_NAMES: list[str] = [
    # "wave_hand",
]

# PNG-friendly ``prepare_test_media`` types (see testset_utils.normalize_test_media_type).
_DEFAULT_METHOD = "alpha_frame_trajectory"

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# motion_generation / vlm_pose_benchmark live under adhoc/generation/robotarm (not adhoc/robotarm).
_ARM = _REPO / "adhoc" / "generation" / "robotarm"
if _ARM.is_dir():
    sys.path.insert(0, str(_ARM))

from adhoc.generation.embodiment_sources import load_embodiment_block  # noqa: E402
from adhoc.generation.see_html import (  # noqa: E402
    _default_config,
    _find_gif,
    _load_see_html_sources,
    _motion_json_rel_list,
    _repo_rel_to_path,
    _rk_for_emb_slug,
    _see_html_sources_path,
)
from adhoc.utils.repo_paths import dev_robosuite_root, results_subdir  # noqa: E402
from adhoc.vlm_test import testset_utils  # noqa: E402

_EMBODIMENTS = frozenset({"manipulator", "google_robot", "quadruped"})

# Methods that produce a single PNG panel (not raw gif/mp4 paths).
_PNG_TILE_METHODS = frozenset(
    {
        "first_frame_only",
        "first_frame_trajectory",
        "middle_frame_only",
        "middle_frame_trajectory",
        "alpha_frame",
        "alpha_frame_trajectory",
    }
)

# Trajectory panel types: sim bundle only (embodiment selects default sim robot; override with --sim-robot).
_TRAJECTORY_PANEL_METHODS = frozenset(
    {
        "alpha_frame_trajectory",
        "first_frame_trajectory",
        "middle_frame_trajectory",
    }
)

# Bump when cache key semantics change (invalidates old PNGs under _panel_cache).
_PANEL_CACHE_VERSION = "5"


def _gif_tile_crop_effective(requested_canonical: str, use_first_tile_only: bool) -> bool:
    """Horizontal top_k tile crop applies only to ``alpha_frame`` (stacked GIF without sim)."""
    if requested_canonical != "alpha_frame":
        return False
    return bool(use_first_tile_only)


def _normalize_embodiment(robot: str) -> str:
    s = (robot or "").strip().lower().replace("-", "_")
    aliases = {
        "robotarm": "manipulator",
        "arm": "manipulator",
        "iiwa": "manipulator",
        "bimanual": "google_robot",
        "google": "google_robot",
        "tiago": "google_robot",
        "mobile": "google_robot",
    }
    s = aliases.get(s, s)
    if s not in _EMBODIMENTS:
        raise SystemExit(f"robot must be one of {sorted(_EMBODIMENTS)} (bimanual → google_robot), got {robot!r}")
    return s


def _sim_robot_for_embodiment(slug: str, override: str | None) -> str:
    if override:
        return override
    # Defaults match robosuite MotionGenerator / QuadrupedMotionGenerator names used in render pipelines.
    if slug == "manipulator":
        return "IIWA"
    if slug == "google_robot":
        return "Tiago"
    if slug == "quadruped":
        return "Go2"
    return "IIWA"


def _load_motion_rows(config_path: Path) -> list[dict[str, Any]]:
    import json

    if not config_path.is_file():
        raise FileNotFoundError(f"Motion config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {config_path}")
    return data


def _rows_with_config_paths(cfg_paths: list[Path]) -> list[tuple[dict[str, Any], Path]]:
    """One (row, source_json) per entry; preserves file boundaries for ``testset_utils`` lookup."""
    out: list[tuple[dict[str, Any], Path]] = []
    for p in cfg_paths:
        if not p.is_file():
            print(f"[tile_figure] warn: skip missing config — {p}", file=sys.stderr)
            continue
        for r in _load_motion_rows(p):
            out.append((r, p))
    return out


def _resolve_from_see_html_sources(
    slug: str,
    *,
    sources_path: Path | None,
) -> tuple[list[tuple[dict[str, Any], Path]], Path, str]:
    """
    Paths from ``see_html_sources.yml`` embodiment block (motion_config_json + render_dir).
    Returns (row, config_path) pairs, gif_dir, human-readable config summary.
    """
    root = dev_robosuite_root()
    sources = _load_see_html_sources(sources_path)
    block = load_embodiment_block(sources, slug)
    rels = _motion_json_rel_list(block["motion_config_json"])
    cfg_paths = [_repo_rel_to_path(r, root) for r in rels]
    gif_dir = _repo_rel_to_path(str(block["render_dir"]), root)
    pairs = _rows_with_config_paths(cfg_paths)
    title = " + ".join(p.name for p in cfg_paths if p.is_file()) or "merged configs"
    return pairs, gif_dir, title


def _filter_and_order_pairs(
    pairs: list[tuple[dict[str, Any], Path]], cue_filter: list[str]
) -> list[tuple[dict[str, Any], Path]]:
    if not cue_filter:
        return sorted(pairs, key=lambda x: int(x[0].get("idx", 0)))
    by_cue: dict[str, tuple[dict[str, Any], Path]] = {}
    for r, p in pairs:
        c = str(r.get("cue", ""))
        if c:
            by_cue[c] = (r, p)
    out: list[tuple[dict[str, Any], Path]] = []
    for name in cue_filter:
        if name not in by_cue:
            print(f"[tile_figure] skip: cue not in motion_configs — {name!r}", file=sys.stderr)
            continue
        out.append(by_cue[name])
    return out


def _build_sample(
    *,
    row: dict[str, Any],
    config_path: Path,
    gif_path: Path,
    slug: str,
) -> dict[str, Any]:
    cue = str(row["cue"])
    idx = int(row.get("idx", -1))
    safe_id = testset_utils._safe_name(f"{slug}_c{idx}_{cue}")
    return {
        "sample_id": safe_id,
        "testset": "iconic",
        "cue_idx": idx,
        "cue": cue,
        "gif_path": str(gif_path),
        "config_path": str(config_path),
        "meta": {},
    }


def _is_tiled_topk_frame(w: int, h: int, *, tiled_columns: int, min_tile_w_px: int) -> bool:
    """Wide composite (e.g. top_k pose variants in one row), same heuristic as compare_prompts."""
    return w >= tiled_columns * min_tile_w_px


def _file_fingerprint(path: Path) -> str:
    """Stable id for cache invalidation when inputs change."""
    try:
        st = os.stat(path)
        return f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}"
    except OSError:
        return str(path.resolve())


def _panel_cache_digest(
    sample: dict[str, Any],
    *,
    requested_canonical: str,
    sim_robot: str,
    hz: int,
    tiled_columns: int,
    min_tile_w_px: int,
    crop_top_ratio: float,
    use_first_tile_only: bool,
) -> str:
    gif_p = Path(sample["gif_path"])
    cfg_p = Path(sample["config_path"])
    # Sim-only trajectory panels ignore GIF path (placeholder may equal config).
    gif_key = (
        "sim_bundle_only"
        if requested_canonical in _TRAJECTORY_PANEL_METHODS
        else _file_fingerprint(gif_p)
    )
    parts = [
        _PANEL_CACHE_VERSION,
        requested_canonical,
        sim_robot,
        str(hz),
        str(sample.get("cue", "")),
        str(int(sample.get("cue_idx", -1))),
        gif_key,
        _file_fingerprint(cfg_p),
        str(tiled_columns),
        str(min_tile_w_px),
        f"{crop_top_ratio:.8f}",
        "1" if _gif_tile_crop_effective(requested_canonical, use_first_tile_only) else "0",
    ]
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return h[:40]


def _crop_tiled_first_tile_only(
    img: Image.Image,
    *,
    tiled_columns: int,
    min_tile_w_px: int,
    crop_top_ratio: float,
) -> Image.Image:
    """
    If the frame looks like a horizontal strip of ``tiled_columns`` tiles (top_k render),
    keep only the leftmost tile. Optionally strip a top band (caption bar) by ratio of tile height.
    Bottom caption: ``tile_h = min(frame_h, tile_w)`` matches compare_prompts._build_top1_checkpoint_tile.
    """
    img = img.convert("RGB")
    w, h = img.size
    if not _is_tiled_topk_frame(w, h, tiled_columns=tiled_columns, min_tile_w_px=min_tile_w_px):
        return img
    tile_w = w // tiled_columns
    tile_h = min(h, tile_w)
    crop = img.crop((0, 0, tile_w, tile_h))
    if crop_top_ratio > 0:
        y0 = min(int(tile_h * crop_top_ratio), tile_h - 2)
        if y0 > 0:
            crop = crop.crop((0, y0, tile_w, tile_h - y0))
    return crop


def _panel_to_pil(prepared: dict[str, Any]) -> Image.Image:
    mime = (prepared.get("media_mime") or "").lower()
    path = Path(prepared["media_path"])
    if mime == "image/png" or path.suffix.lower() == ".png":
        return Image.open(path).convert("RGB")
    if mime == "image/gif" or path.suffix.lower() == ".gif":
        return Image.open(path).convert("RGB")
    if mime == "video/mp4" or path.suffix.lower() == ".mp4":
        raise RuntimeError(f"Tiling mp4 not supported here: {path}")
    return Image.open(path).convert("RGB")


def _try_prepare_panel(
    sample: dict[str, Any],
    *,
    visualize_method: str,
    sim_robot: str,
    hz: int,
    force: bool,
    tmp_media_root: Path,
    tiled_columns: int,
    min_tile_w_px: int,
    crop_top_ratio: float,
    use_first_tile_only: bool,
) -> Image.Image:
    """GIF-based ``prepare_test_media`` only (not ``*_trajectory`` sim modes)."""
    canonical = testset_utils.normalize_test_media_type(visualize_method)
    rel_out = f"tile_figure/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{canonical}"
    out_root = tmp_media_root / rel_out
    prepared_list = testset_utils.prepare_test_media(
        [sample],
        test_type=canonical,
        robot=sim_robot,
        hz=hz,
        output_dir=str(out_root),
        force=force,
    )
    img = _panel_to_pil(prepared_list[0])
    if _gif_tile_crop_effective(canonical, use_first_tile_only):
        img = _crop_tiled_first_tile_only(
            img,
            tiled_columns=tiled_columns,
            min_tile_w_px=min_tile_w_px,
            crop_top_ratio=crop_top_ratio,
        )
    return img


def _prepare_panel(
    sample: dict[str, Any],
    *,
    visualize_method: str,
    sim_robot: str,
    hz: int,
    force: bool,
    tmp_media_root: Path,
    tiled_columns: int,
    min_tile_w_px: int,
    crop_top_ratio: float,
    use_first_tile_only: bool,
    cache_dir: Path | None,
    use_cache: bool,
    save_sim_meta: bool,
    sim_meta_dir: Path,
) -> Image.Image:
    requested_canonical = testset_utils.normalize_test_media_type(visualize_method)
    digest = _panel_cache_digest(
        sample,
        requested_canonical=requested_canonical,
        sim_robot=sim_robot,
        hz=hz,
        tiled_columns=tiled_columns,
        min_tile_w_px=min_tile_w_px,
        crop_top_ratio=crop_top_ratio,
        use_first_tile_only=use_first_tile_only,
    )
    cache_path = (cache_dir / f"{digest}.png") if cache_dir is not None else None

    if use_cache and cache_path and not force and cache_path.is_file():
        return Image.open(cache_path).convert("RGB")

    kw = dict(
        sim_robot=sim_robot,
        hz=hz,
        force=force,
        tmp_media_root=tmp_media_root,
        tiled_columns=tiled_columns,
        min_tile_w_px=min_tile_w_px,
        crop_top_ratio=crop_top_ratio,
        use_first_tile_only=use_first_tile_only,
    )
    if requested_canonical in _TRAJECTORY_PANEL_METHODS:
        img, meta = testset_utils.build_tile_figure_sim_trajectory_panel(
            sample,
            sim_robot,
            hz,
            canonical=requested_canonical,
            force=force,
        )
        if save_sim_meta:
            sim_meta_dir.mkdir(parents=True, exist_ok=True)
            meta_path = sim_meta_dir / f"{digest}.json"
            payload = {
                "cue": sample.get("cue"),
                "cue_idx": sample.get("cue_idx"),
                "sample_id": sample.get("sample_id"),
                "canonical": requested_canonical,
                **meta,
            }
            meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        img = _try_prepare_panel(sample, visualize_method=visualize_method, **kw)

    if use_cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(cache_path, format="PNG", optimize=True)

    return img


def _fit_cell(img: Image.Image, cell_w: int, cell_h: int) -> Image.Image:
    """Uniform scale to **cover** ``cell_w`` × ``cell_h`` (center crop). No letterbox bars."""
    img = img.convert("RGB")
    iw, ih = img.size
    if iw < 1 or ih < 1:
        return Image.new("RGB", (cell_w, cell_h), (0, 0, 0))
    scale = max(cell_w / iw, cell_h / ih)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    resized = img.resize((nw, nh), Image.LANCZOS)
    x0 = max(0, (nw - cell_w) // 2)
    y0 = max(0, (nh - cell_h) // 2)
    return resized.crop((x0, y0, x0 + cell_w, y0 + cell_h))


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def _compose_grid(
    panels: list[tuple[str, Image.Image]],
    *,
    col_n: int,
    cell_w: int,
    cell_h: int,
    footer_h: int,
    pad: int,
    cue_font_size: int,
) -> Image.Image:
    """One image per cell; cue id centered below the image only (no top bar, no method suffix)."""
    n = len(panels)
    if n == 0:
        raise ValueError("no panels")
    col_n = max(1, int(col_n))
    n_rows = (n + col_n - 1) // col_n

    font_cue = _font(cue_font_size)

    cell_total_h = cell_h + footer_h
    grid_w = col_n * (cell_w + pad) + pad
    grid_h = n_rows * (cell_total_h + pad) + pad

    out = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)

    for i, (cue_label, img) in enumerate(panels):
        r, c = divmod(i, col_n)
        x0 = pad + c * (cell_w + pad)
        y0 = pad + r * (cell_total_h + pad)

        cell = _fit_cell(img, cell_w, cell_h)
        out.paste(cell, (x0, y0))

        foot = cue_label[:120]
        bbox = draw.textbbox((0, 0), foot, font=font_cue)
        tw = bbox[2] - bbox[0]
        tx = x0 + max(0, (cell_w - int(tw)) // 2)
        ty = y0 + cell_h + 6
        draw.text((tx, ty), foot, fill=(25, 25, 25), font=font_cue)

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Tile cue-wise PNG panels (testset_utils visualization).")
    p.add_argument(
        "robot",
        help="Embodiment slug: manipulator | google_robot (bimanual alias) | quadruped",
    )
    p.add_argument(
        "visualize_method",
        nargs="?",
        default=_DEFAULT_METHOD,
        help=f"prepare_test_media test_type (default: {_DEFAULT_METHOD})",
    )
    p.add_argument("col_n", type=int, help="Number of columns in the figure grid.")
    p.add_argument(
        "--sources-yml",
        type=str,
        default=None,
        help="Path to see_html_sources.yml (default: data/seed/yml/see_html_sources.yml).",
    )
    p.add_argument(
        "--fallback-latest-glob",
        action="store_true",
        help="Ignore see_html_sources.yml; use newest motion_configs*.json under results/motion_configs/<robot>/ (legacy).",
    )
    p.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="Override motion config: single JSON file (still use --gif-dir if set).",
    )
    p.add_argument(
        "--gif-dir",
        type=str,
        default=None,
        help="Override render/GIF search root (default: embodiment render_dir from yml, or data/results/render/<robot>).",
    )
    p.add_argument("--hz", type=int, default=8, help="Sim / GIF hz for trajectory overlays.")
    p.add_argument("--sim-robot", type=str, default=None, help="MotionGenerator robot name (default: IIWA for all).")
    p.add_argument("--force", action="store_true", help="Regenerate panel PNGs (ignore cache read; refresh cache files).")
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable panel PNG cache (always run prepare_test_media + sim).",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Panel cache directory (default: data/results/visualize/_panel_cache).",
    )
    p.add_argument("--cell-w", type=int, default=420, help="Panel width (px).")
    p.add_argument("--cell-h", type=int, default=320, help="Panel image height (px).")
    p.add_argument(
        "--grid-pad",
        type=int,
        default=0,
        help="Pixels between subplot columns/rows (default: 0, no white gutters).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of cues to include (after filter); default = all.",
    )
    p.add_argument(
        "--tiled-columns",
        type=int,
        default=5,
        help="When the source GIF frame is a horizontal top_k strip, number of tiles (default: 5).",
    )
    p.add_argument(
        "--tiled-min-width",
        type=int,
        default=200,
        help="Min pixel width per tile to treat the frame as tiled (see compare_prompts heuristic).",
    )
    p.add_argument(
        "--crop-top-ratio",
        type=float,
        default=0.06,
        help="After taking the left tile, crop this fraction from the top (removes GIF title bar). 0 disables.",
    )
    p.add_argument(
        "--no-first-tile-only",
        action="store_true",
        help="Do not crop wide GIFs to the leftmost top_k tile. Only applies to visualize_method=alpha_frame.",
    )
    p.add_argument(
        "--save-sim-meta",
        action="store_true",
        help="For *trajectory modes, write JSON (EE screen x,y, frame indices) to data/results/visualize/_tile_sim_meta/.",
    )
    p.add_argument(
        "--cue-font-size",
        type=int,
        default=22,
        help="Font size for the cue label below each panel.",
    )
    args = p.parse_args()

    slug = _normalize_embodiment(args.robot)
    method = args.visualize_method.strip()
    canonical = testset_utils.normalize_test_media_type(method)
    if canonical not in _PNG_TILE_METHODS:
        raise SystemExit(
            f"visualize_method {canonical!r} is not tiled as PNG here "
            f"(use one of {sorted(_PNG_TILE_METHODS)})"
        )

    col_n = max(1, int(args.col_n))
    sim_robot = _sim_robot_for_embodiment(slug, args.sim_robot)

    sources_yml = Path(args.sources_yml).resolve() if args.sources_yml else None

    if args.config_json:
        cfg_one = Path(args.config_json).resolve()
        row_pairs = [(r, cfg_one) for r in _load_motion_rows(cfg_one)]
        gif_dir = Path(args.gif_dir).resolve() if args.gif_dir else results_subdir("render") / slug
        cfg_title = cfg_one.name
    elif args.fallback_latest_glob:
        cfg_one = _default_config(slug)
        row_pairs = [(r, cfg_one) for r in _load_motion_rows(cfg_one)]
        gif_dir = Path(args.gif_dir).resolve() if args.gif_dir else results_subdir("render") / slug
        cfg_title = cfg_one.name
    else:
        row_pairs, gif_dir, cfg_title = _resolve_from_see_html_sources(slug, sources_path=sources_yml)
        if args.gif_dir:
            gif_dir = Path(args.gif_dir).resolve()

    cue_filter = [c for c in CUE_NAMES if str(c).strip()]
    row_pairs = _filter_and_order_pairs(row_pairs, cue_filter)
    if args.limit is not None and int(args.limit) > 0:
        row_pairs = row_pairs[: int(args.limit)]
    if not row_pairs:
        raise SystemExit("No motion config rows after cue filter.")

    root = dev_robosuite_root()
    if args.config_json:
        print("[tile_figure] source: --config-json override", file=sys.stderr)
    elif args.fallback_latest_glob:
        print("[tile_figure] source: --fallback-latest-glob (newest motion_configs*.json by mtime)", file=sys.stderr)
    else:
        yml_ref = _see_html_sources_path(sources_yml).relative_to(root)
        print(f"[tile_figure] source: see_html_sources.yml → {slug} ({yml_ref})", file=sys.stderr)
    print(f"[tile_figure] motion_configs: {cfg_title}", file=sys.stderr)
    try:
        gd_rel = gif_dir.relative_to(root)
    except ValueError:
        gd_rel = gif_dir
    print(f"[tile_figure] render_dir (GIF search): {gd_rel}", file=sys.stderr)

    rk = _rk_for_emb_slug(slug)

    tmp_root = results_subdir("visualize") / "_tile_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else results_subdir("visualize") / "_panel_cache"
    use_panel_cache = not bool(args.no_cache)
    if use_panel_cache:
        try:
            cr = cache_dir.relative_to(root)
        except ValueError:
            cr = cache_dir
        print(f"[tile_figure] panel cache: {cr} (disable with --no-cache)", file=sys.stderr)
    if canonical in _TRAJECTORY_PANEL_METHODS:
        print(
            "[tile_figure] trajectory panels: sim-only (numpy alpha-stack + EE path); "
            "sim bundle disk cache → data/results/visualize/_sim_bundles",
            file=sys.stderr,
        )

    panels: list[tuple[str, Image.Image]] = []
    use_tile = not bool(args.no_first_tile_only)
    crop_top = max(0.0, float(args.crop_top_ratio))
    sim_meta_dir = results_subdir("visualize") / "_tile_sim_meta"

    for row, cfg_path in row_pairs:
        cue = str(row["cue"])
        idx = int(row.get("idx", -1))
        if canonical in _TRAJECTORY_PANEL_METHODS:
            sample = _build_sample(row=row, config_path=cfg_path, gif_path=cfg_path, slug=slug)
            img = _prepare_panel(
                sample,
                visualize_method=method,
                sim_robot=sim_robot,
                hz=int(args.hz),
                force=bool(args.force),
                tmp_media_root=tmp_root,
                tiled_columns=max(1, int(args.tiled_columns)),
                min_tile_w_px=max(1, int(args.tiled_min_width)),
                crop_top_ratio=crop_top,
                use_first_tile_only=use_tile,
                cache_dir=cache_dir,
                use_cache=use_panel_cache,
                save_sim_meta=bool(args.save_sim_meta),
                sim_meta_dir=sim_meta_dir,
            )
            panels.append((cue, img))
            continue

        hit = _find_gif(rk, gif_dir, idx, cue, row)
        if hit is None:
            print(f"[tile_figure] skip (no gif): cue={cue!r} idx={idx}", file=sys.stderr)
            continue
        sample = _build_sample(row=row, config_path=cfg_path, gif_path=hit, slug=slug)
        img = _prepare_panel(
            sample,
            visualize_method=method,
            sim_robot=sim_robot,
            hz=int(args.hz),
            force=bool(args.force),
            tmp_media_root=tmp_root,
            tiled_columns=max(1, int(args.tiled_columns)),
            min_tile_w_px=max(1, int(args.tiled_min_width)),
            crop_top_ratio=crop_top,
            use_first_tile_only=use_tile,
            cache_dir=cache_dir,
            use_cache=use_panel_cache,
            save_sim_meta=False,
            sim_meta_dir=sim_meta_dir,
        )
        panels.append((cue, img))

    if not panels:
        if canonical in _TRAJECTORY_PANEL_METHODS:
            raise SystemExit("No panels produced (check motion config rows / sim bundle errors).")
        raise SystemExit("No panels produced (missing GIFs or all failed).")

    fs = max(10, int(args.cue_font_size))
    footer_h = max(32, fs + 14)
    grid = _compose_grid(
        panels,
        col_n=col_n,
        cell_w=int(args.cell_w),
        cell_h=int(args.cell_h),
        footer_h=footer_h,
        pad=max(0, int(args.grid_pad)),
        cue_font_size=fs,
    )

    out_dir = results_subdir("visualize")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"tile_{slug}_{canonical}_{len(panels)}cues_cols{col_n}_{stamp}.png"
    out_path = out_dir / stem
    grid.save(out_path, format="PNG", optimize=True)
    rel = out_path.relative_to(dev_robosuite_root())
    print(f"Wrote {rel}")


if __name__ == "__main__":
    main()
