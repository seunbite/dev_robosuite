#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adhoc.utils.repo_paths import motion_configs_results_dir, results_render_dir  # noqa: E402
from legacy.render_mobile_config import _make_env, render_config  # noqa: E402


def _default_config() -> Path:
    d = motion_configs_results_dir("google_robot")
    preferred = d / "motion_configs_iconic.json"
    if preferred.exists():
        return preferred
    cands = sorted(d.glob("motion_configs*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else (d / "motion_configs_19_mobile.json")


def run(config_json: str | None = None, output_dir: str | None = None, auto_generate_if_missing: bool = True) -> None:
    cfg = Path(config_json) if config_json else _default_config()
    if auto_generate_if_missing and not cfg.exists():
        from motion_generation import run as motion_run

        motion_run(config_json=str(cfg), run_render=False)
    gif_dir = Path(output_dir) if output_dir else results_render_dir("google_robot")
    gif_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg, "r", encoding="utf-8") as f:
        configs = json.load(f)
    env = _make_env()
    try:
        for row in tqdm(configs, desc="render[google_robot]"):
            idx = int(row.get("idx", -1))
            cue = row.get("cue", "?")
            safe = cue.replace("/", "_").replace("\\", "_").replace(" ", "_")
            gif_path = gif_dir / f"mm19_g{idx:02d}_{safe}.gif"
            try:
                frames = render_config(row, env=env)
                if frames:
                    frames[0].save(str(gif_path), save_all=True, append_images=frames[1:], duration=50, loop=0)
            except Exception as e:
                tqdm.write(f"skip g{idx}: {e}")
    finally:
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()


if __name__ == "__main__":
    fire.Fire(run)
