"""Paths for workspace tooling (adhoc scripts, humaneval, vlm_test).

Seed layout (``data/seed``)::

  yml/                  — ``cues.yml``, ``cues_new.yml``, …
  prompt/<robot>/       — ``manipulator`` | ``google_robot`` | ``quadruped``
  shots/<robot>/        — canonical ``shots.json`` (list of shot configs); legacy ``shot_*.json`` ok
  prompt/<robot>/       — canonical ``prompt.txt``; legacy ``prompt_*.txt`` ok

Generated motion configs live under ``data/results/motion_configs/<robot>/``.
GIF renders live under ``data/results/render/<robot>/``.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

_ROBOTS = frozenset({"manipulator", "google_robot", "quadruped"})


def dev_robosuite_root() -> Path:
    """Directory that contains ``robosuite/``, ``adhoc/``, ``data/``."""
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    """Parent of ``dev_robosuite`` (sibling to ``unitree_rl_mjlab``, etc.)."""
    return dev_robosuite_root().parent


def unitree_rl_mjlab_root() -> Path:
    return workspace_root() / "unitree_rl_mjlab"


def seed_dir() -> Path:
    return dev_robosuite_root() / "data" / "seed"


def seed_yml_dir() -> Path:
    return seed_dir() / "yml"


def seed_prompt_dir(robot: str) -> Path:
    if robot not in _ROBOTS:
        raise ValueError(f"robot must be one of {sorted(_ROBOTS)}, got {robot!r}")
    return seed_dir() / "prompt" / robot


def seed_shots_dir(robot: str) -> Path:
    if robot not in _ROBOTS:
        raise ValueError(f"robot must be one of {sorted(_ROBOTS)}, got {robot!r}")
    return seed_dir() / "shots" / robot


def motion_configs_results_dir(robot: str) -> Path:
    """Robot-named tree under ``data/results/motion_configs`` (any slug, e.g. ``manipulator``)."""
    return dev_robosuite_root() / "data" / "results" / "motion_configs" / robot


def results_subdir(name: str) -> Path:
    """name: ``vlm``, ``render``, ``html``, ``pptx``, ``human_eval``, …"""
    return dev_robosuite_root() / "data" / "results" / name


def results_render_dir(robot: str) -> Path:
    """GIF / frame output root for a robot slug (e.g. ``manipulator``)."""
    if robot not in _ROBOTS:
        raise ValueError(f"robot must be one of {sorted(_ROBOTS)}, got {robot!r}")
    return results_subdir("render") / robot


def resolve_seed_shots_json(robot: str) -> Path:
    """Prefer ``shots/<robot>/shots.json``; else newest ``shot*.json`` in that directory."""
    d = seed_shots_dir(robot)
    canonical = d / "shots.json"
    if canonical.exists():
        return canonical
    candidates = sorted(d.glob("shot*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    return canonical


def resolve_seed_prompt_txt(robot: str) -> Path:
    """Prefer ``prompt/<robot>/prompt.txt``; else newest ``prompt*.txt``."""
    d = seed_prompt_dir(robot)
    canonical = d / "prompt.txt"
    if canonical.exists():
        return canonical
    candidates = sorted(d.glob("prompt*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    return canonical


def infer_robot_from_prompt_path(prompt_file: str | Path) -> str | None:
    """Return ``manipulator`` / ``google_robot`` / ``quadruped`` if path contains ``.../prompt/<slug>/``."""
    parts = Path(prompt_file).parts
    for i, p in enumerate(parts):
        if p == "prompt" and i + 1 < len(parts) and parts[i + 1] in _ROBOTS:
            return parts[i + 1]
    return None


def caches_dir() -> Path:
    return dev_robosuite_root() / "data" / "caches"


def logs_dir() -> Path:
    return dev_robosuite_root() / "data" / "logs"


def dated_filename(stem: str, ext: str, *, date: str | None = None) -> str:
    """Return ``YYYYMMDD_stem.ext`` (no double date prefix)."""
    d = date or datetime.now().strftime("%Y%m%d")
    ext = ext.lstrip(".")
    if stem.startswith(f"{d}_"):
        return f"{stem}.{ext}"
    return f"{d}_{stem}.{ext}"


# --- backwards-compatible names (prefer seed_prompt_dir / seed_shots_dir above) ---


def seed_robot(robot: str, kind: str) -> Path:
    """Deprecated layout: ``data/seed/<robot>/shot`` — use ``seed_shots_dir`` etc."""
    return dev_robosuite_root() / "data" / "seed" / robot / kind
