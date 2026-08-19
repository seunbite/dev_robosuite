"""Render GT fixed start pose to PNG for exp7.1 VLM input."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adhoc.generation.tiago.legacy import _make_env, render_config  # noqa: E402

DEFAULT_POSE_PNG_DIR = _REPO / "data/results/render/tiago/fixed_pose_png"


def _pose_cache_key(pose: dict[str, Any], *, cue: str = "", cue_idx: int | None = None) -> str:
    if cue_idx is not None:
        return f"g{int(cue_idx):02d}_{cue.replace('/', '_').replace(' ', '_')}"
    blob = json.dumps(pose, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def render_fixed_pose_png(
    fixed_pose: dict[str, Any],
    out_path: Path,
    *,
    cue: str = "",
    cue_idx: int | None = None,
    env=None,
) -> Path:
    """Render one hold frame of the fixed pose; return PNG path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "cue": cue or "fixed_pose",
        "idx": cue_idx if cue_idx is not None else -1,
        "movements": [
            {
                "type": "pose",
                "duration": 0.5,
                "parameters": {"pose": dict(fixed_pose)},
            }
        ],
    }
    own_env = env is None
    if own_env:
        env = _make_env()
    try:
        frames = render_config(cfg, env=env)
        if not frames:
            raise RuntimeError(f"No frames rendered for cue={cue!r}")
        frames[0].convert("RGB").save(out_path)
    finally:
        if own_env:
            closer = getattr(env, "close", None)
            if callable(closer):
                closer()
    return out_path


def fixed_pose_png_path(
    fixed_pose: dict[str, Any],
    *,
    cue: str,
    cue_idx: int,
    png_dir: Path | None = None,
) -> Path:
    root = png_dir or DEFAULT_POSE_PNG_DIR
    stem = _pose_cache_key(fixed_pose, cue=cue, cue_idx=cue_idx)
    return root / f"{stem}.png"


def ensure_fixed_pose_png(
    fixed_pose: dict[str, Any],
    *,
    cue: str,
    cue_idx: int,
    png_dir: Path | None = None,
    env=None,
    force: bool = False,
) -> tuple[Path, Image.Image]:
    path = fixed_pose_png_path(fixed_pose, cue=cue, cue_idx=cue_idx, png_dir=png_dir)
    if force or not path.is_file():
        render_fixed_pose_png(fixed_pose, path, cue=cue, cue_idx=cue_idx, env=env)
    return path, Image.open(path).convert("RGB")
