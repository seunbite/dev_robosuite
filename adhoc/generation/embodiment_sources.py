"""Load per-embodiment paths from ``data/seed/yml/see_html_sources.yml``."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from adhoc.utils.repo_paths import dev_robosuite_root, seed_yml_dir

SEE_HTML_SOURCES_FILENAME = "see_html_sources.yml"

_EMBODIMENT_KEYS = frozenset({"manipulator", "tiago", "quadruped"})

REQUIRED_BLOCK_KEYS = frozenset(
    {
        "prompt_file",
        "shots_json",
        "motion_config_json",
        "render_dir",
        "see_html_subdir",
        "see_html_stem",
        "render_html_subdir",
        "render_html_stem",
    }
)


def see_html_sources_path(explicit: Path | None = None) -> Path:
    return explicit if explicit else (seed_yml_dir() / SEE_HTML_SOURCES_FILENAME)


def load_see_html_sources(sources_path: Path | None = None) -> dict[str, Any]:
    p = see_html_sources_path(sources_path)
    if not p.is_file():
        raise SystemExit(f"Missing {p} — add it or pass explicit path")
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML required for see_html_sources.yml (pip install pyyaml)")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid YAML root in {p}")
    return data


def motion_json_rel_list(val: Any) -> list[str]:
    if isinstance(val, str):
        return [val]
    if isinstance(val, list):
        return [str(x) for x in val]
    raise SystemExit(f"motion_config_json must be string or list, got {type(val).__name__}")


def repo_rel_to_path(rel: str, root: Path | None = None) -> Path:
    r = root or dev_robosuite_root()
    return (r / str(rel).strip().lstrip("/")).resolve()


def robot_kw_to_embodiment_slug(robot: str) -> str:
    """Map CLI ``--robot`` aliases to YAML top-level embodiment key."""
    rk = robot.strip().lower().replace("-", "_")
    aliases = {
        "manipulator": "manipulator",
        "robotarm": "manipulator",
        "arm": "manipulator",
        "iiwa": "manipulator",
        "tiago": "tiago",
        "bimanual": "tiago",
        "mobile": "tiago",
        # Legacy see_html trio still routes "google" → Tiago package paths.
        "google": "tiago",
        # see_html trio has no stretch block yet; Everyday proxy routes to tiago.
        "stretch": "tiago",
        "everyday": "tiago",
        "quadruped": "quadruped",
    }
    slug = aliases.get(rk, rk)
    if slug not in _EMBODIMENT_KEYS:
        raise SystemExit(f"robot must resolve to embodiment slug {_EMBODIMENT_KEYS}, got {robot!r}")
    return slug


def embodiment_slugs_set() -> frozenset[str]:
    return frozenset(_EMBODIMENT_KEYS)


def load_embodiment_block(sources: dict[str, Any], slug: str) -> dict[str, Any]:
    if slug not in _EMBODIMENT_KEYS:
        raise SystemExit(f"Unknown embodiment slug {slug!r}; expected one of {_EMBODIMENT_KEYS}")
    blk = sources.get(slug)
    if not isinstance(blk, dict):
        raise SystemExit(f"see_html_sources.yml: missing or invalid block for {slug!r}")
    missing = sorted(REQUIRED_BLOCK_KEYS - set(blk))
    if missing:
        raise SystemExit(f"see_html_sources.yml: {slug!r} missing keys: {missing}")
    for k in REQUIRED_BLOCK_KEYS:
        v = blk.get(k)
        if not isinstance(v, str) or not str(v).strip():
            raise SystemExit(f"see_html_sources.yml: {slug}.{k} must be a non-empty string")
    return blk


def first_motion_config_path_rel(block: dict[str, Any]) -> str:
    """First path string from ``motion_config_json`` (str or list)."""
    parts = motion_json_rel_list(block["motion_config_json"])
    if not parts:
        raise SystemExit("motion_config_json resolved to empty list")
    return parts[0]


def resolve_embodiment_paths(
    slug: str,
    sources_path: Path | None = None,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Canonical prompt path, shots path, primary motion-config JSON path, and raw YAML block."""
    root = dev_robosuite_root()
    sources = load_see_html_sources(sources_path)
    blk = load_embodiment_block(sources, slug)
    prompt = repo_rel_to_path(blk["prompt_file"], root)
    shots = repo_rel_to_path(blk["shots_json"], root)
    motion = repo_rel_to_path(first_motion_config_path_rel(blk), root)
    return prompt, shots, motion, blk


def load_cross_config(sources: dict[str, Any]) -> dict[str, Any]:
    cross = sources.get("cross")
    if not isinstance(cross, dict):
        raise SystemExit("see_html_sources.yml: missing or invalid cross mapping")
    for k in ("html_dir", "html_stem", "column_order"):
        if k not in cross:
            raise SystemExit(f"see_html_sources.yml: cross must include {k!r}")
    order = cross["column_order"]
    if not isinstance(order, list) or not all(isinstance(x, str) for x in order):
        raise SystemExit("see_html_sources.yml: cross.column_order must be a list of strings")
    return cross
