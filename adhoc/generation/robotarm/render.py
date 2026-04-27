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

from adhoc.utils.repo_paths import dev_robosuite_root, motion_configs_results_dir, results_render_dir  # noqa: E402
from legacy.motion_generation_core import generate  # noqa: E402


def _default_config() -> Path:
    d = motion_configs_results_dir("manipulator")
    preferred = d / "motion_configs_iconic.json"
    if preferred.exists():
        return preferred
    cands = sorted(d.glob("motion_configs*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else preferred


def run(
    config_json: str | None = None,
    output_dir: str | None = None,
    sim_robot: str = "IIWA",
    jsonl_path: str | None = None,
    hz: int = 4,
    top_k: int = 5,
    cue_indices: list[int] | None = None,
    auto_generate_if_missing: bool = True,
) -> None:
    cfg = Path(config_json) if config_json else _default_config()
    if auto_generate_if_missing and not cfg.exists():
        from motion_generation import run as motion_run

        motion_run(config_json=str(cfg), run_render=False)
    out = Path(output_dir) if output_dir else results_render_dir("manipulator")
    out.mkdir(parents=True, exist_ok=True)
    jpath = jsonl_path or str(dev_robosuite_root() / "data" / "seed" / "_remainder" / "closest_poses_results.jsonl")

    with open(cfg, "r", encoding="utf-8") as f:
        rows = json.load(f)
    todo = rows if cue_indices is None else [r for r in rows if int(r.get("idx", -1)) in set(cue_indices)]

    for row in tqdm(todo, desc="render[robotarm]"):
        idx = row.get("idx")
        cue = row.get("cue")
        if idx is None or not cue:
            continue
        try:
            generate(
                robot=sim_robot,
                cue=str(cue),
                cue_idx=int(idx),
                jsonl_path=jpath,
                config_path=str(cfg),
                output_dir=str(out),
                hz=hz,
                top_k=top_k,
            )
        except Exception as e:
            tqdm.write(f"skip c{idx} {cue}: {e}")


if __name__ == "__main__":
    fire.Fire(run)
