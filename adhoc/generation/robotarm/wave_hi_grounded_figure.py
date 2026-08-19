#!/usr/bin/env python3
"""
Render **six** ``wave_hi`` customer-context variants. Every pose uses ``gripper_orientation:
"horizontal"``; the salient motion is wrist **y** (left–right paddle). Order: distant → nearby →
first-time customer → returning customer → looking → not looking.

GIFs land under ``data/results/render/manipulator/wave_hi_grounded_<timestamp>/<Robot>/``, copied as
``wave_hi_v<N>_<slug>_<Robot>.gif``, plus a stitched row GIF and PNG ``alpha_frame_trajectory``.

Default robot: **IIWA** only (``--robots`` to override).
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adhoc.generation.embodiment_sources import load_embodiment_block  # noqa: E402
from adhoc.generation.see_html import (  # noqa: E402
    _load_see_html_sources,
    _motion_json_rel_list,
    _repo_rel_to_path,
)
from adhoc.utils.repo_paths import (  # noqa: E402
    dev_robosuite_root,
    motion_configs_results_dir,
    resolve_seed_shots_json,
    results_render_dir,
    results_subdir,
)
from adhoc.vlm_test import testset_utils  # noqa: E402

_ARM = Path(__file__).resolve().parent
if str(_ARM) not in sys.path:
    sys.path.insert(0, str(_ARM))
from legacy.motion_generation_core import generate  # noqa: E402


_DEFAULT_MANIPULATORS: tuple[str, ...] = ("IIWA",)

# Six customer-facing contexts. Tuning: (1) reach x/z, (2) wrist y (deg_y) + reps + timing,
# (3) light render_modifiers. Shoulder = one light bob per scenario (not looking = stronger bob).
_WAVE_SCENARIOS: list[dict[str, Any]] = [
    {
        "slug": "customer_distant",
        "caption": "Distant",
        "poses": [
            {"dir": "up", "x": 74, "y": 50, "z": 72, "gripper_orientation": "horizontal"},
            {"dir": "up", "x": 76, "y": 50, "z": 74, "gripper_orientation": "horizontal"},
        ],
        "pose_params": [
            {"speed": 1.35, "hold_time": 0.0},
            {"speed": 1.25, "hold_time": 0.05},
        ],
        "shoulder": {"repetition": 1, "degrees": {"z": 5}, "speed": 1.15, "hold_time": 0.0},
        "wrist": {"repetition": 7, "deg_y_plus": 34.0, "deg_y_minus": -34.0, "speed": 2.6, "hold_time": 0.08},
        "render_modifiers": {"hesitation": 0.0, "elegant_curve": 0.4, "zittering": 0.1},
        "stack_count": 20,
    },
    {
        "slug": "customer_nearby",
        "caption": "Nearby",
        "poses": [
            {"dir": "up", "x": 50, "y": 50, "z": 52, "gripper_orientation": "horizontal"},
            {"dir": "up", "x": 50, "y": 50, "z": 54, "gripper_orientation": "horizontal"},
        ],
        "pose_params": [
            {"speed": 1.05, "hold_time": 0.08},
            {"speed": 1.05, "hold_time": 0.12},
        ],
        "shoulder": {"repetition": 1, "degrees": {"z": 5}, "speed": 1.15, "hold_time": 0.0},
        "wrist": {"repetition": 5, "deg_y_plus": 22.0, "deg_y_minus": -22.0, "speed": 1.9, "hold_time": 0.12},
        "render_modifiers": {"hesitation": 0.06, "elegant_curve": 0.55, "zittering": 0.0},
        "stack_count": 16,
    },
    {
        "slug": "waves_first_reactive",
        "caption": "First-time customer",
        "poses": [
            {"dir": "up", "x": 52, "y": 50, "z": 56, "gripper_orientation": "horizontal"},
            {"dir": "up", "x": 52, "y": 50, "z": 58, "gripper_orientation": "horizontal"},
        ],
        "pose_params": [
            {"speed": 1.35, "hold_time": 0.0},
            {"speed": 1.5, "hold_time": 0.02},
        ],
        "shoulder": {"repetition": 1, "degrees": {"z": 5}, "speed": 1.15, "hold_time": 0.0},
        "wrist": {"repetition": 4, "deg_y_plus": 16.0, "deg_y_minus": -16.0, "speed": 3.6, "hold_time": 0.03},
        "render_modifiers": {"hesitation": 0.12, "elegant_curve": 0.25, "zittering": 0.05},
        "stack_count": 12,
    },
    {
        "slug": "customer_returning",
        "caption": "Returning customer",
        "poses": [
            {"dir": "up", "x": 52, "y": 52, "z": 58, "gripper_orientation": "horizontal"},
            {"dir": "up", "x": 54, "y": 52, "z": 60, "gripper_orientation": "horizontal"},
        ],
        "pose_params": [
            {"speed": 1.25, "hold_time": 0.02},
            {"speed": 1.2, "hold_time": 0.06},
        ],
        "shoulder": {"repetition": 1, "degrees": {"z": 5}, "speed": 1.15, "hold_time": 0.0},
        "wrist": {"repetition": 8, "deg_y_plus": 38.0, "deg_y_minus": -38.0, "speed": 3.0, "hold_time": 0.06},
        "render_modifiers": {"hesitation": 0.05, "elegant_curve": 0.35, "zittering": 0.22},
        "stack_count": 22,
    },
    {
        "slug": "customer_looking",
        "caption": "Looking",
        "poses": [
            {"dir": "up", "x": 50, "y": 50, "z": 50, "gripper_orientation": "horizontal"},
            {"dir": "up", "x": 50, "y": 50, "z": 52, "gripper_orientation": "horizontal"},
        ],
        "pose_params": [
            {"speed": 1.0, "hold_time": 0.1},
            {"speed": 1.05, "hold_time": 0.14},
        ],
        "shoulder": {"repetition": 1, "degrees": {"z": 5}, "speed": 1.15, "hold_time": 0.0},
        "wrist": {"repetition": 5, "deg_y_plus": 26.0, "deg_y_minus": -26.0, "speed": 2.1, "hold_time": 0.11},
        "render_modifiers": {"hesitation": 0.04, "elegant_curve": 0.62, "zittering": 0.0},
        "stack_count": 16,
    },
    {
        "slug": "customer_not_looking",
        "caption": "Not looking",
        "poses": [
            {"dir": "up", "x": 48, "y": 50, "z": 78, "gripper_orientation": "horizontal"},
            {"dir": "up", "x": 52, "y": 50, "z": 82, "gripper_orientation": "horizontal"},
        ],
        "pose_params": [
            {"speed": 1.1, "hold_time": 0.02},
            {"speed": 1.75, "hold_time": 0.06},
        ],
        "shoulder": {
            "repetition": 7,
            "deg_y_plus": 32.0,
            "deg_y_minus": -32.0,
            "speed": 2.35,
            "hold_time": 0.055,
        },
        "wrist": {"repetition": 10, "deg_y_plus": 50.0, "deg_y_minus": -48.0, "speed": 3.4, "hold_time": 0.038},
        "render_modifiers": {"hesitation": 0.08, "elegant_curve": 0.1, "zittering": 0.58},
        "stack_count": 26,
    },
]


def _merge_pose(dst: dict[str, Any], patch: dict[str, Any]) -> None:
    for k, v in patch.items():
        dst[k] = copy.deepcopy(v)


def _apply_wave_scenario(base_row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base_row)
    out["render_modifiers"] = copy.deepcopy(spec.get("render_modifiers", {}))
    moves = out.setdefault("movements", [])
    pose_specs = spec.get("poses") or []
    pose_param_specs = spec.get("pose_params") or []
    pi = 0
    for mv in moves:
        if mv.get("type") != "pose":
            continue
        if pi >= len(pose_specs):
            break
        params = mv.setdefault("parameters", {})
        pose = params.setdefault("pose", {})
        if isinstance(pose, dict):
            _merge_pose(pose, pose_specs[pi])
        if pi < len(pose_param_specs):
            for k, v in pose_param_specs[pi].items():
                params[k] = v
        pi += 1

    for mv in moves:
        if mv.get("type") != "movement":
            continue
        params = mv.setdefault("parameters", {})
        joint = str(params.get("joint", "")).lower()
        if joint == "shoulder" and "shoulder" in spec:
            sk = spec["shoulder"]
            params["repetition"] = int(sk.get("repetition", 1))
            if "deg_y_plus" in sk and "deg_y_minus" in sk:
                plus = float(sk["deg_y_plus"])
                minus = float(sk["deg_y_minus"])
                spd = float(sk["speed"])
                hld = float(sk["hold_time"])
                params["directions"] = [
                    {"degrees": {"y": plus}, "speed": spd, "hold_time": hld},
                    {"degrees": {"y": minus}, "speed": spd, "hold_time": hld},
                ]
            else:
                dirs = params.setdefault("directions", [{}])
                if dirs and isinstance(dirs[0], dict):
                    d0 = dirs[0]
                    if "degrees" in sk:
                        d0["degrees"] = copy.deepcopy(sk["degrees"])
                    if "speed" in sk:
                        d0["speed"] = float(sk["speed"])
                    if "hold_time" in sk:
                        d0["hold_time"] = float(sk["hold_time"])
        elif joint == "wrist" and "wrist" in spec:
            wk = spec["wrist"]
            params["repetition"] = max(1, int(wk.get("repetition", 3)))
            dirs = params.setdefault("directions", [{}, {}])
            plus = float(wk.get("deg_y_plus", 25))
            minus = float(wk.get("deg_y_minus", -25))
            if len(dirs) >= 1 and isinstance(dirs[0], dict):
                dirs[0].setdefault("degrees", {})["y"] = plus
                if "speed" in wk:
                    dirs[0]["speed"] = float(wk["speed"])
                if "hold_time" in wk:
                    dirs[0]["hold_time"] = float(wk["hold_time"])
            if len(dirs) >= 2 and isinstance(dirs[1], dict):
                dirs[1].setdefault("degrees", {})["y"] = minus
                if "speed" in wk:
                    dirs[1]["speed"] = float(wk["speed"])
                if "hold_time" in wk:
                    dirs[1]["hold_time"] = float(wk["hold_time"])
    return out


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {path}")
    return data


def _merged_cue_map(cfg_paths: list[Path]) -> dict[str, tuple[dict[str, Any], Path]]:
    """Later files overwrite earlier entries with the same ``cue``."""
    out: dict[str, tuple[dict[str, Any], Path]] = {}
    for p in cfg_paths:
        if not p.is_file():
            print(f"[wave_hi_grounded] skip missing config: {p}", file=sys.stderr)
            continue
        for row in _load_json_array(p):
            c = row.get("cue")
            if c:
                out[str(c)] = (row, p)
    return out


def _resolve_manipulator_configs(
    *,
    sources_yml: Path | None,
    config_json: Path | None,
) -> tuple[list[Path], str]:
    root = dev_robosuite_root()
    if config_json is not None:
        p = config_json.resolve()
        return [p], p.name
    sources = _load_see_html_sources(sources_yml)
    blk = load_embodiment_block(sources, "manipulator")
    rels = _motion_json_rel_list(blk["motion_config_json"])
    paths = [_repo_rel_to_path(r, root) for r in rels]
    title = " + ".join(p.name for p in paths if p.is_file()) or "configs"
    return paths, title


def _default_config_newest_glob() -> Path:
    d = motion_configs_results_dir("manipulator")
    cands = sorted(d.glob("motion_configs*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        return cands[0]
    raise SystemExit(f"No motion_configs*.json under {d}")


def _shot_row_for_cue(cue: str, *, shots_json: Path | None = None) -> dict[str, Any] | None:
    p = shots_json if shots_json is not None else resolve_seed_shots_json("manipulator")
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        shots = json.load(f)
    if not isinstance(shots, list):
        return None
    for s in reversed(shots):
        if isinstance(s, dict) and str(s.get("cue", "")) == cue:
            return s
    return None


def _pick_base_row(
    cue: str,
    cfg_paths: list[Path],
    *,
    base_from_shots: bool,
    shots_json: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    by_cue = _merged_cue_map(cfg_paths)
    if cue not in by_cue:
        names = sorted(by_cue.keys())
        preview = ", ".join(names[:40]) + ("…" if len(names) > 40 else "")
        raise SystemExit(f"Cue {cue!r} not found in merged motion configs. Some cues: {preview}")
    motion_row, source_path = by_cue[cue]
    if not base_from_shots:
        return copy.deepcopy(motion_row), source_path
    shot = _shot_row_for_cue(cue, shots_json=shots_json)
    if not shot:
        print("[wave_hi_grounded] --base-from-shots: no matching shot; using motion config row", file=sys.stderr)
        return copy.deepcopy(motion_row), source_path
    merged = copy.deepcopy(shot)
    merged["cue"] = cue
    if "idx" not in merged and motion_row.get("idx") is not None:
        merged["idx"] = motion_row["idx"]
    print(f"[wave_hi_grounded] base movements from shots.json; idx={merged.get('idx')}", file=sys.stderr)
    return merged, source_path


def _resolve_written_gif(
    generate_ret: Any,
    *,
    batch_dir: Path,
    robot: str,
    suffix_tag: str,
) -> Path | None:
    """Locate the GIF ``generate`` just wrote (return path, glob by suffix, or newest in robot dir)."""
    if isinstance(generate_ret, str):
        p = Path(generate_ret)
        if p.is_file():
            return p.resolve()
    d = (batch_dir / robot).resolve()
    if d.is_dir():
        tagged = sorted(d.glob(f"*{suffix_tag}.gif"), key=lambda p: p.stat().st_mtime)
        if tagged:
            return tagged[-1]
        any_gif = sorted(d.glob("*.gif"), key=lambda p: p.stat().st_mtime)
        if any_gif:
            return any_gif[-1]
    return None


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


# Pixels between cell bottom and caption baseline; tight band under single-line labels.
_CAPTION_GAP_PX = 4
_CAPTION_BOTTOM_PAD_PX = 8


def _measure_caption_footer_height(font: ImageFont.ImageFont, labels: list[str]) -> int:
    """Footer strip height = gap below cell + tallest caption + small padding below glyphs."""
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_h = 0
    for s in labels:
        bb = draw.textbbox((0, 0), s[:220], font=font)
        max_h = max(max_h, int(bb[3] - bb[1]))
    return _CAPTION_GAP_PX + max(1, max_h) + _CAPTION_BOTTOM_PAD_PX


# After scale-to-cover fit, bias the crop upward (fraction of letterboxed height) so panels
# emphasize the upper arm / end-effector instead of the base.
_CELL_CROP_BIAS_Y_FRAC = 0.13


def _fit_cell(img: Image.Image, cell_w: int, cell_h: int, *, crop_bias_y_frac: float = 0.0) -> Image.Image:
    img = img.convert("RGB")
    iw, ih = img.size
    if iw < 1 or ih < 1:
        return Image.new("RGB", (cell_w, cell_h), (0, 0, 0))
    scale = max(cell_w / iw, cell_h / ih)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    resized = img.resize((nw, nh), Image.LANCZOS)
    x0 = max(0, (nw - cell_w) // 2)
    frac = float(max(0.0, min(0.45, crop_bias_y_frac)))
    y0 = max(0, (nh - cell_h) // 2 - int(frac * nh))
    return resized.crop((x0, y0, x0 + cell_w, y0 + cell_h))


def _gif_all_frames(path: Path) -> list[Image.Image]:
    img = Image.open(path)
    out: list[Image.Image] = []
    try:
        n = int(getattr(img, "n_frames", 1))
        for i in range(n):
            img.seek(i)
            out.append(img.convert("RGB").copy())
    finally:
        img.close()
    return out


def _save_row_gif_from_variants(
    *,
    gif_paths: list[Path],
    captions: list[str],
    out_path: Path,
    cell_w: int,
    cell_h: int,
    footer_h: int,
    pad: int,
    font_size: int,
    frame_duration_ms: int,
) -> None:
    """One animated GIF: same grid as the PNG figure (panels + captions), per-frame stitched."""
    if len(gif_paths) != len(captions):
        raise ValueError("gif_paths and captions length mismatch")
    frame_lists = [_gif_all_frames(p) for p in gif_paths]
    if any(len(fl) == 0 for fl in frame_lists):
        raise RuntimeError(f"Empty GIF input in {gif_paths}")
    n_frames = min(len(fl) for fl in frame_lists)
    font = _font(font_size)

    composed: list[Image.Image] = []
    for t in range(n_frames):
        n = len(captions)
        cell_total_h = cell_h + footer_h
        grid_w = n * (cell_w + pad) + pad
        grid_h = cell_total_h + 2 * pad
        slab = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
        draw = ImageDraw.Draw(slab)
        for i, cap in enumerate(captions):
            x0 = pad + i * (cell_w + pad)
            y0 = pad
            cell_img = _fit_cell(frame_lists[i][t], cell_w, cell_h, crop_bias_y_frac=_CELL_CROP_BIAS_Y_FRAC)
            slab.paste(cell_img, (x0, y0))
            foot = cap[:220]
            bbox = draw.textbbox((0, 0), foot, font=font)
            tw = bbox[2] - bbox[0]
            tx = x0 + max(0, (cell_w - int(tw)) // 2)
            ty = y0 + cell_h + _CAPTION_GAP_PX
            draw.text((tx, ty), foot, fill=(25, 25, 25), font=font)
        composed.append(slab)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    composed[0].save(
        out_path,
        save_all=True,
        append_images=composed[1:],
        duration=max(12, int(frame_duration_ms)),
        loop=0,
        optimize=False,
    )


def _compose_horizontal(
    panels: list[tuple[str, Image.Image]],
    *,
    cell_w: int,
    cell_h: int,
    footer_h: int,
    pad: int,
    font_size: int,
) -> Image.Image:
    font = _font(font_size)
    n = len(panels)
    cell_total_h = cell_h + footer_h
    grid_w = n * (cell_w + pad) + pad
    grid_h = cell_total_h + 2 * pad
    out = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    for i, (label, img) in enumerate(panels):
        x0 = pad + i * (cell_w + pad)
        y0 = pad
        out.paste(_fit_cell(img, cell_w, cell_h, crop_bias_y_frac=_CELL_CROP_BIAS_Y_FRAC), (x0, y0))
        foot = label[:200]
        bbox = draw.textbbox((0, 0), foot, font=font)
        tw = bbox[2] - bbox[0]
        tx = x0 + max(0, (cell_w - int(tw)) // 2)
        ty = y0 + cell_h + _CAPTION_GAP_PX
        draw.text((tx, ty), foot, fill=(25, 25, 25), font=font)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Six wave_hi scenarios → PNG row (alpha_frame_trajectory) + single stitched row GIF."
    )
    ap.add_argument("--cue", default="wave_hi", help="Motion cue name in JSON (default: wave_hi)")
    ap.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="Single motion config JSON (overrides see_html_sources.yml merge).",
    )
    ap.add_argument(
        "--sources-yml",
        type=str,
        default=None,
        help="Override path to see_html_sources.yml.",
    )
    ap.add_argument(
        "--fallback-latest-config",
        action="store_true",
        help="Use newest motion_configs*.json under results/motion_configs/manipulator/ (ignores yml).",
    )
    ap.add_argument(
        "--base-from-shots",
        action="store_true",
        help="Use seed shots.json row as movement template when cue matches.",
    )
    ap.add_argument(
        "--robots",
        nargs="+",
        default=None,
        help="Sim robot names (default: IIWA).",
    )
    ap.add_argument("--hz", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=1, help="Initial poses (1 = single GIF, not tiled strip)")
    ap.add_argument("--force-sim", action="store_true", help="Bypass sim bundle disk cache for trajectory panels")
    ap.add_argument("--cell-w", type=int, default=300)
    ap.add_argument("--cell-h", type=int, default=280)
    ap.add_argument(
        "--footer-h",
        type=int,
        default=0,
        help="Min caption strip height (px). 0 = auto from font + longest caption (tight).",
    )
    ap.add_argument("--grid-pad", type=int, default=6)
    ap.add_argument("--caption-font-size", type=int, default=26)
    args = ap.parse_args()
    robots = tuple(args.robots) if args.robots else _DEFAULT_MANIPULATORS
    sources_yml = Path(args.sources_yml).resolve() if args.sources_yml else None
    if args.fallback_latest_config:
        cfg_paths = [_default_config_newest_glob()]
        cfg_title = cfg_paths[0].name
    elif args.config_json:
        cfg_paths = [Path(args.config_json).resolve()]
        cfg_title = cfg_paths[0].name
    else:
        cfg_paths, cfg_title = _resolve_manipulator_configs(sources_yml=sources_yml, config_json=None)

    shots_json_for_cue: Path | None = None
    if not args.fallback_latest_config:
        src_blk = _load_see_html_sources(sources_yml)
        blk = load_embodiment_block(src_blk, "manipulator")
        shots_json_for_cue = _repo_rel_to_path(blk["shots_json"], dev_robosuite_root())

    cue = str(args.cue).strip()
    base_row, _src = _pick_base_row(cue, cfg_paths, base_from_shots=bool(args.base_from_shots), shots_json=shots_json_for_cue)

    jsonl_path = str(dev_robosuite_root() / "data" / "seed" / "_remainder" / "closest_poses_results.jsonl")
    render_root = results_render_dir("manipulator")
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    batch_dir = (render_root / f"wave_hi_grounded_{run_stamp}").resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)

    print(f"[wave_hi_grounded] motion_configs: {cfg_title}", file=sys.stderr)
    print(f"[wave_hi_grounded] batch: {batch_dir.relative_to(dev_robosuite_root())}", file=sys.stderr)
    print(f"[wave_hi_grounded] robots: {', '.join(robots)}", file=sys.stderr)
    print(f"[wave_hi_grounded] variants: {len(_WAVE_SCENARIOS)}", file=sys.stderr)

    canonical = testset_utils.normalize_test_media_type("alpha_frame_trajectory")
    out_dir = results_subdir("visualize")
    out_dir.mkdir(parents=True, exist_ok=True)

    for sim_robot in robots:
        sim_robot = str(sim_robot).strip()
        robot_generator_dir = batch_dir / sim_robot
        print(
            f"[wave_hi_grounded] {sim_robot} → {robot_generator_dir.relative_to(dev_robosuite_root())}/",
            file=sys.stderr,
        )
        panels: list[tuple[str, Image.Image]] = []
        row_variant_gifs: list[Path] = []
        row_variant_captions: list[str] = []
        try:
            for i, spec in enumerate(_WAVE_SCENARIOS):
                slug = str(spec["slug"])
                caption = str(spec["caption"])
                row = _apply_wave_scenario(base_row, spec)
                synth_idx = 9500 + i
                row["idx"] = synth_idx
                row["cue"] = cue

                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    prefix=f"wave_hi_{sim_robot}_{slug}_",
                    delete=False,
                    encoding="utf-8",
                ) as tmp:
                    json.dump([row], tmp, indent=2, ensure_ascii=False)
                    one_cfg = Path(tmp.name)

                gif_tag = f"w{i:02d}_{slug}"
                try:
                    ret = generate(
                        robot=sim_robot,
                        cue=cue,
                        cue_idx=synth_idx,
                        jsonl_path=jsonl_path,
                        config_path=str(one_cfg),
                        output_dir=str(batch_dir),
                        hz=int(args.hz),
                        top_k=int(args.top_k),
                        overlay_progress_bar=False,
                        gif_filename_suffix=gif_tag,
                    )
                finally:
                    one_cfg.unlink(missing_ok=True)

                src = _resolve_written_gif(
                    ret,
                    batch_dir=batch_dir,
                    robot=sim_robot,
                    suffix_tag=gif_tag,
                )
                if src is not None:
                    dest = batch_dir / f"wave_hi_v{i:02d}_{slug}_{sim_robot}.gif"
                    shutil.copy2(src, dest)
                    row_variant_gifs.append(dest)
                    row_variant_captions.append(caption)
                    print(
                        f"[wave_hi_grounded] variant GIF → {dest.relative_to(dev_robosuite_root())}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[wave_hi_grounded] no GIF found for {sim_robot} {slug!r} (tag {gif_tag!r})",
                        file=sys.stderr,
                    )

                sample_id = testset_utils._safe_name(f"manipulator_{sim_robot}_{slug}_{cue}")
                sample = {
                    "sample_id": sample_id,
                    "testset": "iconic",
                    "cue_idx": synth_idx,
                    "cue": cue,
                    "gif_path": "",
                    "config_path": "",
                    "meta": {},
                }
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    prefix=f"wave_hi_sample_cfg_{sim_robot}_{i}_",
                    delete=False,
                    encoding="utf-8",
                ) as tmp2:
                    json.dump([row], tmp2, indent=2, ensure_ascii=False)
                    sample_cfg = Path(tmp2.name)
                sample["config_path"] = str(sample_cfg)
                sample["gif_path"] = str(sample_cfg)
                try:
                    stack_count = int(spec.get("stack_count", 16))
                    img, _meta = testset_utils.build_tile_figure_sim_trajectory_panel(
                        sample,
                        sim_robot,
                        int(args.hz),
                        canonical=canonical,
                        force=bool(args.force_sim),
                        stack_count=min(28, max(8, stack_count)),
                    )
                finally:
                    sample_cfg.unlink(missing_ok=True)
                panels.append((caption, img))

            fs = max(10, int(args.caption_font_size))
            font = _font(fs)
            cap_labels = [str(s.get("caption", s["slug"])) for s in _WAVE_SCENARIOS]
            measured_footer = _measure_caption_footer_height(font, cap_labels)
            if int(args.footer_h) <= 0:
                footer_h = measured_footer
            else:
                footer_h = max(measured_footer, int(args.footer_h))
            grid = _compose_horizontal(
                panels,
                cell_w=int(args.cell_w),
                cell_h=int(args.cell_h),
                footer_h=footer_h,
                pad=max(0, int(args.grid_pad)),
                font_size=fs,
            )
            safe_bot = testset_utils._safe_name(sim_robot)
            out_path = out_dir / f"wave_hi_grounded_alpha_frame_trajectory_{safe_bot}_{run_stamp}.png"
            grid.save(out_path, format="PNG", optimize=True)
            print(f"Wrote {out_path.relative_to(dev_robosuite_root())}")

            if len(row_variant_gifs) == len(_WAVE_SCENARIOS):
                gif_out = out_dir / f"wave_hi_grounded_row_{safe_bot}_{run_stamp}.gif"
                frame_ms = int(round(1000 / max(1, int(args.hz))))
                try:
                    _save_row_gif_from_variants(
                        gif_paths=row_variant_gifs,
                        captions=row_variant_captions,
                        out_path=gif_out,
                        cell_w=int(args.cell_w),
                        cell_h=int(args.cell_h),
                        footer_h=footer_h,
                        pad=max(0, int(args.grid_pad)),
                        font_size=fs,
                        frame_duration_ms=frame_ms,
                    )
                    print(f"Wrote {gif_out.relative_to(dev_robosuite_root())}")
                except Exception as ex:
                    print(f"[wave_hi_grounded] row GIF failed: {ex}", file=sys.stderr)
            else:
                print(
                    f"[wave_hi_grounded] skip row GIF (expected {len(_WAVE_SCENARIOS)} variant GIFs, "
                    f"got {len(row_variant_gifs)})",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"[wave_hi_grounded] FAILED robot={sim_robot!r}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
