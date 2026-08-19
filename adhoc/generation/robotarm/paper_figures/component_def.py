"""Pose / movement / path definition figure (3 subplots) per embodiment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
_ROBOTARM = _HERE.parent
_REPO = _ROBOTARM.parents[2]
for p in (_REPO, _ROBOTARM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from paper_figures._media import (  # noqa: E402
    alpha_trajectory_for_cfg,
    cfg_by_idx,
    pose_png_for_row,
    pose_sim_plain_png_for_cfg,
    pose_sim_png_for_cfg,
    repo,
)

DEFAULT_POSE_IDX = 16  # stop_palm_out — pose-primary

FONT_SIZE = 24
PANEL_W = 300
PANEL_H = 300
FOOTER_H = 28
LABEL_GAP = 4
CROP_TOP_FRAC = 0.10
CROP_SIDE_FRAC = 0.14
CROP_BOTTOM_FRAC = 0.04

# Manipulator (IIWA / Panda)
_DEMO_POSE: dict[str, Any] = {
    "dir": "front",
    "gripper_orientation": "horizontal",
    "x": 58,
    "y": 48,
    "z": 74,
    "pose_id": 27,
}

_PATH_POSE: dict[str, Any] = {
    "dir": "front",
    "gripper_orientation": "horizontal",
    "x": 52,
    "y": 42,
    "z": 68,
    "pose_id": 27,
}

# Tiago (TIAGo mobile)
_MOBILE_POSE: dict[str, Any] = {
    "torso_height": "mid",
    "arm_position": "front",
    "gripper_orientation": "horizontal",
    "head": "center",
    "left_arm": "still",
}

# Quadruped (Go2)
_QUAD_POSE: dict[str, Any] = {
    "dir": "front",
    "front_height": "mid",
    "back_height": "mid",
    "three_legs": True,
    "three_leg_pose": {"hand_position": "up"},
}


def _movement_demo_cfg_manipulator() -> dict[str, Any]:
    pose = dict(_DEMO_POSE)
    return {
        "idx": 99001,
        "cue": "paper_fig_movement_demo",
        "description": "Large shoulder sweep for component-definition figure.",
        "gt_fixed_first_pose": pose,
        "movements": [
            {"type": "pose", "parameters": {"pose": pose, "speed": 1.0, "hold_time": 0.0}},
            {
                "type": "movement",
                "parameters": {
                    "joint": "shoulder",
                    "repetition": 1,
                    "directions": [
                        {"degrees": {"z": -28}, "speed": 1.4, "hold_time": 0.04},
                        {"degrees": {"z": 28}, "speed": 1.4, "hold_time": 0.04},
                    ],
                },
            },
        ],
    }


def _path_demo_cfg_manipulator(*, variant: str = "y") -> dict[str, Any]:
    segments = (
        {"axis": "y", "distance": -0.5},
        {"axis": "y", "distance": 0.5},
    ) if variant == "y" else (
        {"axis": "x", "distance": 0.5},
        {"axis": "x", "distance": -0.5},
    )
    pose = dict(_PATH_POSE)
    path_steps: list[dict[str, Any]] = []
    for i, seg in enumerate(segments):
        path_steps.append(
            {
                "type": "path",
                "parameters": {
                    "shape": "line",
                    "axis": seg["axis"],
                    "distance": float(seg["distance"]),
                    "speed": 0.5,
                    "hold_time": 0.05 if i == len(segments) - 1 else 0.0,
                },
            }
        )
    return {
        "idx": 99002 if variant == "y" else 99003,
        "cue": f"paper_fig_path_demo_{variant}",
        "gt_fixed_first_pose": pose,
        "movements": [
            {"type": "pose", "parameters": {"pose": pose, "speed": 1.0, "hold_time": 0.0}},
            *path_steps,
        ],
    }


def _pose_demo_cfg_mobile() -> dict[str, Any]:
    pose = dict(_MOBILE_POSE)
    return {
        "idx": 99010,
        "cue": "paper_fig_pose_demo_mobile",
        "movements": [
            {"type": "pose", "duration": 0.5, "parameters": {"pose": pose}},
        ],
    }


def _movement_demo_cfg_mobile() -> dict[str, Any]:
    pose = dict(_MOBILE_POSE)
    return {
        "idx": 99011,
        "cue": "paper_fig_movement_demo_mobile",
        "movements": [
            {"type": "pose", "duration": 0.5, "parameters": {"pose": pose}},
            {
                "type": "movement",
                "duration": 2.0,
                "parameters": {
                    "movement": {
                        "joints": [
                            {"joint": "shoulder", "axis": "pitch", "degrees": [-35, 35], "speed": 1.2},
                            {"joint": "head", "axis": "tilt", "degrees": [-14, 14], "speed": 0.9},
                        ],
                        "repetition": 1,
                        "speed": 1.0,
                    }
                },
            },
        ],
    }


def _path_demo_cfg_mobile() -> dict[str, Any]:
    pose = dict(_MOBILE_POSE)
    return {
        "idx": 99012,
        "cue": "paper_fig_path_demo_mobile",
        "movements": [
            {"type": "pose", "duration": 0.5, "parameters": {"pose": pose}},
            {
                "type": "path",
                "parameters": {
                    "path": {
                        "shape": "line",
                        "mode": "base",
                        "x": 0.55,
                        "y": 0.38,
                        "speed": 0.5,
                    },
                },
            },
            {
                "type": "path",
                "parameters": {
                    "path": {
                        "shape": "line",
                        "mode": "base",
                        "x": -0.55,
                        "y": -0.38,
                        "speed": 0.5,
                    },
                },
            },
        ],
    }


# Paper-figure only: pull frontview back; no progress bar; trim empty sky in panel.
_TIAGO_RENDER_META: dict[str, Any] = {
    "cam_pos_scale": 3.05,
    "cam_fovy": 40.0,
    "overlay_progress_bar": False,
    "panel_crop_top_frac": 0.20,
}

# Paper-figure only: front view; trim top margin in panel.
_QUADRUPED_RENDER_META: dict[str, Any] = {
    "camera_spherical_m": 1.65,
    "camera_distance": 0.95,
    "camera_elevation_deg": 17.0,
    "camera_azimuth_deg": 0.0,
    "panel_crop_top_frac": 0.14,
}


def _pose_demo_cfg_quadruped() -> dict[str, Any]:
    pose = dict(_QUAD_POSE)
    return {
        "idx": 99020,
        "cue": "paper_fig_pose_demo_quad",
        "movements": [
            {"type": "pose", "duration": 0.5, "parameters": {"pose": pose}},
        ],
    }


def _movement_demo_cfg_quadruped() -> dict[str, Any]:
    pose = {
        "dir": "front",
        "front_height": "high",
        "back_height": "mid",
        "three_legs": True,
        "three_leg_pose": {"hand_position": "up"},
    }
    return {
        "idx": 99021,
        "cue": "paper_fig_movement_demo_quad",
        "movements": [
            {"type": "pose", "duration": 0.5, "parameters": {"pose": pose}},
            {
                "type": "movement",
                "duration": 2.0,
                "parameters": {
                    "movement": {
                        "joints": [{"joint": "hip", "degrees": {"y": [-22, -6]}}],
                        "repetition": 1,
                        "speed": 1.0,
                    }
                },
            },
        ],
    }


def _path_demo_cfg_quadruped() -> dict[str, Any]:
    pose = dict(_QUAD_POSE)
    pose["three_legs"] = False
    return {
        "idx": 99022,
        "cue": "paper_fig_path_demo_quad",
        "movements": [
            {"type": "pose", "duration": 0.4, "parameters": {"pose": pose}},
            {
                "type": "path",
                "parameters": {"path": {"shape": "arc", "degrees": 360, "radius": 0, "speed": 1.0}},
            },
        ],
    }


EMBODIMENTS: dict[str, dict[str, Any]] = {
    "IIWA": {
        "sim_robot": "IIWA",
        "label": "IIWA",
        "out_name": "component_def_pose_movement_path_iiwa.png",
        "caption_name": "component_def_caption_iiwa.txt",
        "use_pose_tiles": True,
        "pose_idx": DEFAULT_POSE_IDX,
        "path_caption": "a long world-frame Y line path (−Y→+Y)",
        "movement_caption": "joint-space shoulder sweep",
    },
    "Panda": {
        "sim_robot": "Panda",
        "label": "Panda",
        "out_name": "component_def_pose_movement_path_panda.png",
        "caption_name": "component_def_caption_panda.txt",
        "use_pose_tiles": False,
        "pose_idx": DEFAULT_POSE_IDX,
        "path_caption": "a long world-frame Y line path (−Y→+Y)",
        "movement_caption": "joint-space shoulder sweep",
    },
    "Tiago": {
        "sim_robot": "Tiago",
        "label": "Tiago",
        "out_name": "component_def_pose_movement_path_tiago.png",
        "caption_name": "component_def_caption_tiago.txt",
        "use_pose_tiles": False,
        "path_caption": "base holonomic diagonal floor path (+X+Y→−X−Y)",
        "movement_caption": "right-arm shoulder pitch sweep + head tilt",
    },
    "Quadruped": {
        "sim_robot": "Go2",
        "label": "Quadruped",
        "out_name": "component_def_pose_movement_path_quadruped.png",
        "caption_name": "component_def_caption_quadruped.txt",
        "use_pose_tiles": False,
        "path_caption": "in-place 360° yaw arc (spin)",
        "movement_caption": "lifted paw hip abduction to the left",
    },
}

_PATH_VARIANTS = frozenset({"y", "x-pos"})


def _font(size: int = FONT_SIZE) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        for root in ("/System/Library/Fonts/Supplemental", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"):
            p = Path(root) / name
            if p.is_file():
                try:
                    return ImageFont.truetype(str(p), size)
                except OSError:
                    pass
    return ImageFont.load_default()


def _crop_margins(
    img: Image.Image,
    *,
    top: float = CROP_TOP_FRAC,
    sides: float = CROP_SIDE_FRAC,
    bottom: float = CROP_BOTTOM_FRAC,
) -> Image.Image:
    w, h = img.size
    x0 = max(0, int(w * sides))
    x1 = min(w, int(w * (1.0 - sides)))
    y0 = max(0, int(h * top))
    y1 = min(h, int(h * (1.0 - bottom)))
    if x1 <= x0 or y1 <= y0:
        return img
    return img.crop((x0, y0, x1, y1))


def _fit_cover(img: Image.Image, w: int, h: int) -> Image.Image:
    iw, ih = img.size
    scale = max(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - w) // 2
    top = (nh - h) // 2
    return resized.crop((left, top, left + w, top + h))


def _crop_square_bottom_viewport(img: Image.Image, *, top_frac: float) -> Image.Image:
    """Trim sky, then take a bottom-anchored square (no stretch; fills panel when resized)."""
    w, h = img.size
    y_min = max(0, min(h - 1, int(h * top_frac)))
    side = min(w, h - y_min)
    if side < 2:
        return img
    x0 = (w - side) // 2
    y0 = max(y_min, h - side)
    return img.crop((x0, y0, x0 + side, y0 + side))


def _fit_square(img: Image.Image, side: int) -> Image.Image:
    return img.resize((side, side), Image.Resampling.LANCZOS)


def _fit_contain(
    img: Image.Image,
    w: int,
    h: int,
    bg: tuple[int, int, int] = (255, 255, 255),
    *,
    anchor: str = "center",
) -> Image.Image:
    iw, ih = img.size
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), bg)
    x0 = (w - nw) // 2
    if anchor == "bottom":
        y0 = h - nh
    elif anchor == "top":
        y0 = 0
    else:
        y0 = (h - nh) // 2
    canvas.paste(resized, (x0, y0))
    return canvas


def _panel(
    img: Image.Image,
    label: str,
    *,
    fit_contain: bool = False,
    fit_mode: str = "cover",
    crop_top_frac: float = 0.0,
    anchor_bottom: bool = False,
    anchor_top: bool = False,
    skip_margin_crop: bool = False,
) -> Image.Image:
    base = img.convert("RGB")
    if crop_top_frac > 0 and fit_mode != "square_bottom":
        base = _crop_margins(base, top=crop_top_frac, sides=0.0, bottom=0.0)
    if fit_mode == "square_bottom":
        content = _fit_square(
            _crop_square_bottom_viewport(base, top_frac=crop_top_frac),
            PANEL_W,
        )
    elif fit_contain:
        if anchor_top:
            anchor = "top"
        elif anchor_bottom:
            anchor = "bottom"
        else:
            anchor = "center"
        content = _fit_contain(
            base,
            PANEL_W,
            PANEL_H,
            anchor=anchor,
        )
    else:
        src = base if skip_margin_crop else _crop_margins(base)
        content = _fit_cover(src, PANEL_W, PANEL_H)
    canvas = Image.new("RGB", (PANEL_W, PANEL_H + FOOTER_H), (255, 255, 255))
    canvas.paste(content, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = _font()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    ty = PANEL_H + LABEL_GAP
    draw.text(((PANEL_W - tw) // 2, ty), label, fill=(0, 0, 0), font=font)
    return canvas


def _configs_for_embodiment(embodiment: str, *, path_variant: str = "y") -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    key = embodiment.strip()
    if key in ("IIWA", "Panda"):
        if key == "IIWA":
            by_idx = cfg_by_idx()
            pose_idx = int(EMBODIMENTS[key].get("pose_idx", DEFAULT_POSE_IDX))
            pose_row = by_idx[pose_idx]
            pose_cfg = {
                "idx": pose_idx,
                "cue": str(pose_row.get("cue", "pose_demo")),
                "gt_fixed_first_pose": pose_row.get("gt_fixed_first_pose") or _first_pose_from_row(pose_row),
                "movements": [s for s in (pose_row.get("movements") or []) if s.get("type") == "pose"][:1],
            }
            if not pose_cfg["movements"]:
                pose = dict(_DEMO_POSE)
                pose_cfg["movements"] = [{"type": "pose", "parameters": {"pose": pose}}]
        else:
            pose = dict(_DEMO_POSE)
            pose_cfg = {
                "idx": DEFAULT_POSE_IDX,
                "cue": "paper_fig_pose_demo_panda",
                "gt_fixed_first_pose": pose,
                "movements": [{"type": "pose", "parameters": {"pose": pose, "speed": 1.0, "hold_time": 0.3}}],
            }
        return pose_cfg, _movement_demo_cfg_manipulator(), _path_demo_cfg_manipulator(variant=path_variant)
    if key == "Tiago":
        return _pose_demo_cfg_mobile(), _movement_demo_cfg_mobile(), _path_demo_cfg_mobile()
    if key == "Quadruped":
        return _pose_demo_cfg_quadruped(), _movement_demo_cfg_quadruped(), _path_demo_cfg_quadruped()
    raise SystemExit(f"Unknown embodiment {embodiment!r}; choose from {sorted(EMBODIMENTS)}")


def _first_pose_from_row(row: dict[str, Any]) -> dict[str, Any]:
    for step in row.get("movements") or []:
        if step.get("type") == "pose":
            return step.get("parameters", {}).get("pose") or {}
    return {}


def build(
    *,
    embodiment: str = "IIWA",
    pose_idx: int | None = None,
    path_variant: str = "y",
    out: str | None = None,
    force: bool = False,
) -> Path:
    if path_variant not in _PATH_VARIANTS:
        raise SystemExit(f"--path-variant must be one of {sorted(_PATH_VARIANTS)}")

    spec = EMBODIMENTS.get(embodiment)
    if not spec:
        raise SystemExit(f"--embodiment must be one of {sorted(EMBODIMENTS)}")

    if pose_idx is not None and embodiment in ("IIWA", "Panda"):
        EMBODIMENTS[embodiment] = {**spec, "pose_idx": pose_idx}

    robot = str(spec["sim_robot"])
    cache = repo() / "data/results/paper_figures/cache/component_def" / embodiment.lower()
    cache.mkdir(parents=True, exist_ok=True)

    pose_cfg, mov_cfg, path_cfg = _configs_for_embodiment(embodiment, path_variant=path_variant)
    render_meta = (
        _TIAGO_RENDER_META
        if embodiment == "Tiago"
        else _QUADRUPED_RENDER_META
        if embodiment == "Quadruped"
        else None
    )
    # Tiago: bottom-anchored square viewport → 300×300 (no letterbox, no side chop).
    panel_fit_mode = "square_bottom" if embodiment == "Tiago" else "cover"
    fit_contain = False
    skip_margin_crop = False
    if embodiment in ("Tiago", "Quadruped"):
        crop_top = float((render_meta or {}).get("panel_crop_top_frac", CROP_TOP_FRAC))
    else:
        crop_top = 0.0
    anchor_top = False
    anchor_bottom = False
    panels: list[Image.Image] = []

    pose_png: Path | None = None
    if spec.get("use_pose_tiles") and embodiment == "IIWA":
        by_idx = cfg_by_idx()
        pose_row = by_idx[int(spec.get("pose_idx", DEFAULT_POSE_IDX))]
        pose_png = pose_png_for_row(pose_row, cache / "pose.png")
    if not pose_png or not pose_png.is_file():
        pose_fn = pose_sim_plain_png_for_cfg if embodiment == "Quadruped" else pose_sim_png_for_cfg
        pose_png = pose_fn(
            pose_cfg, robot, cache / "pose_demo.json", cache / "pose.png", force=force, meta=render_meta
        )

    if pose_png and pose_png.is_file():
        panels.append(
            _panel(
                Image.open(pose_png),
                "Pose",
                fit_contain=fit_contain,
                fit_mode=panel_fit_mode,
                crop_top_frac=crop_top,
                anchor_bottom=anchor_bottom,
                anchor_top=anchor_top,
                skip_margin_crop=skip_margin_crop,
            )
        )
    else:
        panels.append(
            _panel(
                Image.new("RGB", (PANEL_W, PANEL_H), (230, 230, 230)),
                "Pose",
                fit_contain=fit_contain,
                fit_mode=panel_fit_mode,
                crop_top_frac=crop_top,
                anchor_bottom=anchor_bottom,
                anchor_top=anchor_top,
                skip_margin_crop=skip_margin_crop,
            )
        )

    mov_traj = alpha_trajectory_for_cfg(
        mov_cfg,
        cache / "movement_demo.json",
        cache / "movement_traj.png",
        robot=robot,
        force=force,
        stack_count=20,
        blend_weight=0.48,
        annotate="movement",
        meta=render_meta,
    )
    if mov_traj and mov_traj.is_file():
        panels.append(
            _panel(
                Image.open(mov_traj),
                "Movement",
                fit_contain=fit_contain,
                fit_mode=panel_fit_mode,
                crop_top_frac=crop_top,
                anchor_bottom=anchor_bottom,
                anchor_top=anchor_top,
                skip_margin_crop=skip_margin_crop,
            )
        )
    else:
        panels.append(
            _panel(
                Image.new("RGB", (PANEL_W, PANEL_H), (230, 230, 230)),
                "Movement",
                fit_contain=fit_contain,
                fit_mode=panel_fit_mode,
                crop_top_frac=crop_top,
                anchor_bottom=anchor_bottom,
                anchor_top=anchor_top,
                skip_margin_crop=skip_margin_crop,
            )
        )

    path_traj = alpha_trajectory_for_cfg(
        path_cfg,
        cache / f"path_demo_{path_variant}.json",
        cache / f"path_traj_{path_variant}.png",
        robot=robot,
        force=force,
        stack_count=44,
        blend_weight=0.30,
        annotate="path",
        meta=render_meta,
    )
    if path_traj and path_traj.is_file():
        panels.append(
            _panel(
                Image.open(path_traj),
                "Path",
                fit_contain=fit_contain,
                fit_mode=panel_fit_mode,
                crop_top_frac=crop_top,
                anchor_bottom=anchor_bottom,
                anchor_top=anchor_top,
                skip_margin_crop=skip_margin_crop,
            )
        )
    else:
        panels.append(
            _panel(
                Image.new("RGB", (PANEL_W, PANEL_H), (230, 230, 230)),
                "Path",
                fit_contain=fit_contain,
                fit_mode=panel_fit_mode,
                crop_top_frac=crop_top,
                anchor_bottom=anchor_bottom,
                anchor_top=anchor_top,
                skip_margin_crop=skip_margin_crop,
            )
        )

    pw, ph = panels[0].size
    fig = Image.new("RGB", (pw * len(panels), ph), (255, 255, 255))
    for i, p in enumerate(panels):
        fig.paste(p, (i * pw, 0))

    od = repo() / "data/results/paper_figures"
    od.mkdir(parents=True, exist_ok=True)
    out_path = Path(out) if out else od / str(spec["out_name"])
    fig.save(out_path)
    caption_path = od / str(spec["caption_name"])
    caption_path.write_text(
        f"Motion primitives on {spec['label']}: static pose (Pose), "
        f"{spec['movement_caption']} (Movement; synthetic config), and "
        f"{spec['path_caption']} (Path; synthetic config; alpha-frame trajectory).\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")
    return out_path


def build_all(*, path_variant: str = "y", force: bool = False) -> list[Path]:
    return [build(embodiment=k, path_variant=path_variant, force=force) for k in EMBODIMENTS]


def main() -> None:
    p = argparse.ArgumentParser(description="Pose / movement / path definition figure")
    p.add_argument(
        "--embodiment",
        choices=sorted(EMBODIMENTS),
        default="IIWA",
        help="Robot embodiment (default: IIWA). Use --all-embodiments for Panda, Tiago, Quadruped too.",
    )
    p.add_argument("--all-embodiments", action="store_true", help="Build IIWA + Panda + Tiago + Quadruped")
    p.add_argument("--pose-idx", type=int, default=None, help="Manipulator cue idx for pose panel (IIWA/Panda)")
    p.add_argument(
        "--path-variant",
        choices=sorted(_PATH_VARIANTS),
        default="y",
        help="Path panel: y (horizontal trail) or x-pos (+X forward); manipulator only",
    )
    p.add_argument(
        "--all-path-variants",
        action="store_true",
        help="Build both Y and +X path versions (manipulator embodiments)",
    )
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true", help="Re-render trajectory panels")
    args = p.parse_args()

    if args.all_embodiments:
        for emb in EMBODIMENTS:
            if args.all_path_variants and emb in ("IIWA", "Panda"):
                for variant in sorted(_PATH_VARIANTS):
                    build(embodiment=emb, pose_idx=args.pose_idx, path_variant=variant, force=args.force)
            else:
                build(
                    embodiment=emb,
                    pose_idx=args.pose_idx,
                    path_variant=args.path_variant,
                    force=args.force,
                )
    elif args.all_path_variants and args.embodiment in ("IIWA", "Panda"):
        for variant in sorted(_PATH_VARIANTS):
            build(pose_idx=args.pose_idx, embodiment=args.embodiment, path_variant=variant, force=args.force)
    else:
        build(
            embodiment=args.embodiment,
            pose_idx=args.pose_idx,
            path_variant=args.path_variant,
            out=args.out,
            force=args.force,
        )


if __name__ == "__main__":
    main()
