#!/usr/bin/env python3
"""Render tiago few-shot examples (legacy + new xyz-trim renderer)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

from adhoc.generation.tiago.build_shots_review_html import (  # noqa: E402
    SHOTS_PATH,
    build_html,
)
from adhoc.generation.tiago.legacy import _make_env, render_config  # noqa: E402

OUT_DIR = _REPO / "data/results/visualize/tiago/new_shots_render"
LEGACY_DIR = OUT_DIR
NEW_DIR = OUT_DIR / "new_render"


def _save_gif(frames, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0)


def _render_shot(env, shot: dict, *, ignore_xyz: bool) -> Path | None:
    cue = shot["cue"]
    sub = LEGACY_DIR if ignore_xyz else NEW_DIR
    out_path = sub / f"{cue}.gif"
    frames = render_config(
        shot,
        env=env,
        overlay_progress_bar=True,
        progress_bar_style="typed",
        ignore_xyz=ignore_xyz,
    )
    if not frames:
        return None
    _save_gif(frames, out_path)
    return out_path


def render_shots(*, legacy: bool = True, new_render: bool = True) -> None:
    os.environ.setdefault("MOBILE_POSE_XYZ_MODE", "trim")
    shots = json.loads(SHOTS_PATH.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NEW_DIR.mkdir(parents=True, exist_ok=True)

    env = _make_env()
    try:
        for shot in shots:
            cue = shot["cue"]
            if legacy:
                print(f"Rendering legacy {cue}...", flush=True)
                p = _render_shot(env, shot, ignore_xyz=True)
                print(f"  → {p}", flush=True)
            if new_render:
                print(f"Rendering new_render {cue}...", flush=True)
                p = _render_shot(env, shot, ignore_xyz=False)
                print(f"  → {p}", flush=True)
    finally:
        env.close()

    html_path = build_html()
    print(f"Review page: {html_path}")


if __name__ == "__main__":
    render_shots()
