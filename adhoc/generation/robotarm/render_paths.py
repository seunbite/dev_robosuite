"""Canonical render output paths: render/{robot}/{llm}/exp{N}/<role>_g{idx}_{cue}[...].gif"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

_ROBOTARM = Path(__file__).resolve().parent
_REPO = _ROBOTARM.parents[2]


def _safe_cue(cue: str) -> str:
    return cue.replace("/", "_").replace("\\", "_").replace(" ", "_")


ExpId = str
RENDER_ROOT = _REPO / "data/results/render"

Domain = Literal["tiago", "manipulator"]
RenderRole = str

# exp0 / exp1 / exp7 / exp15 always use xyz-trim renderer (no legacy ignore_xyz renders).
NEW_RENDER_EXPS = frozenset({"0", "1", "7", "15"})


def robot_slug(domain: Domain) -> str:
    return domain


def render_exp_dir(domain: Domain, model_tag: str, exp_id: ExpId | str) -> Path:
    return RENDER_ROOT / robot_slug(domain) / model_tag / f"exp{exp_id}"


def render_role_for_round(round_n: int) -> RenderRole:
    if round_n <= 0:
        return "gen"
    if round_n == 1:
        return "corr"
    return f"corr_r{round_n}"


def gt_render_role(variant: str = "gt", round_n: int = 0) -> RenderRole:
    base = "gt" if variant == "gt" else f"gt_{variant}"
    if round_n <= 0:
        return base
    if round_n == 1:
        return f"{base}_corr"
    return f"{base}_corr_r{round_n}"


def render_role_subdir(role: RenderRole) -> str | None:
    """Subfolder under exp{N}/ for non-generation roles (matches tiago layout)."""
    if role == "gen":
        return None
    if role == "corr":
        return "corr1"
    if role == "corr_r2":
        return "corr2"
    if role == "corr_r3":
        return "corr3"
    return None


def render_gif_basename(idx: int, cue: str, role: RenderRole) -> str:
    safe = _safe_cue(cue)
    g = f"g{idx:02d}_{safe}"
    if role == "gen":
        return f"generation_{g}.gif"
    suffix = {
        "corr": "",
        "corr_r2": "_r2",
        "corr_r3": "_r3",
        "gt": "_gt",
        "gt_y": "_gt_y",
        "gt_corr": "_gt_corr",
        "gt_corr_r2": "_gt_corr_r2",
        "gt_y_corr": "_gt_y_corr",
        "gt_y_corr_r2": "_gt_y_corr_r2",
        "soph": "_soph",
    }.get(role)
    if suffix is None:
        suffix = f"_{role}"
    return f"correction_{g}{suffix}.gif"


def render_gif_path(
    domain: Domain,
    model_tag: str,
    exp_id: ExpId | str,
    idx: int,
    cue: str,
    role: RenderRole,
) -> Path:
    base = render_exp_dir(domain, model_tag, exp_id)
    sub = render_role_subdir(role)
    if sub:
        base = base / sub
    return base / render_gif_basename(idx, cue, role)


def render_media_paths(
    domain: Domain,
    model_tag: str,
    exp_id: ExpId | str,
    idx: int,
    cue: str,
    role: RenderRole,
) -> tuple[Path, Path, Path]:
    """Return (gif, mp4, pose_png) paths in the flat exp directory."""
    gif = render_gif_path(domain, model_tag, exp_id, idx, cue, role)
    stem = gif.stem
    return gif, gif.with_suffix(".mp4"), gif.parent / f"{stem}_pose.png"


def cumulative_tile_path(
    domain: Domain,
    model_tag: str,
    exp_id: ExpId | str,
    idx: int,
    cue: str,
    n_tiles: int,
) -> Path:
    safe = _safe_cue(cue)
    return render_exp_dir(domain, model_tag, exp_id) / f"cumulative_{n_tiles}tiles_g{idx:02d}_{safe}.gif"


def uses_new_renderer(domain: Domain, exp_id: ExpId | str) -> bool:
    return str(exp_id) in NEW_RENDER_EXPS


MANIPULATOR_CORRECTION_ROOT = RENDER_ROOT / "manipulator" / "correction"
MANIPULATOR_RENDER_HZ = 20


def manipulator_cue_render_dir(
    model_tag: str,
    exp_id: ExpId | str,
    cue: str,
    variant: Literal["new_render", "correction"],
) -> Path:
    return MANIPULATOR_CORRECTION_ROOT / f"exp{exp_id}" / model_tag / _safe_cue(cue) / variant
