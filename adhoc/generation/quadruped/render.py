#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import fire
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adhoc.utils.repo_paths import motion_configs_results_dir, results_render_dir, unitree_rl_mjlab_root  # noqa: E402


def _default_config() -> Path:
    d = motion_configs_results_dir("quadruped")
    preferred = d / "motion_configs_iconic.json"
    if preferred.exists():
        return preferred
    cands = sorted(d.glob("motion_configs*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else (d / "motion_configs_prompt_19_locomotion.json")


def run(
    config_json: str | None = None,
    output_dir: str | None = None,
    python_bin: str | None = None,
    auto_generate_if_missing: bool = True,
) -> None:
    cfg = Path(config_json) if config_json else _default_config()
    if auto_generate_if_missing and not cfg.exists():
        from motion_generation import run as motion_run

        motion_run(config_json=str(cfg), run_render=False)
    gif_dir = Path(output_dir) if output_dir else results_render_dir("quadruped")
    cfg_dir = gif_dir / "_per_cue_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)
    py = python_bin or sys.executable
    mjlab = unitree_rl_mjlab_root() / "scripts" / "record_go2_locomotion_config_mjlab_gif.py"

    with open(cfg, "r", encoding="utf-8") as f:
        configs = json.load(f)
    for row in tqdm(configs, desc="render[quadruped]"):
        idx = int(row.get("idx", -1))
        cue = row.get("cue", "?")
        safe = cue.replace("/", "_").replace("\\", "_").replace(" ", "_")
        cpath = cfg_dir / f"lc19_g{idx:02d}_{safe}.json"
        gpath = gif_dir / f"lc19_g{idx:02d}_{safe}_mjlab_earth.gif"
        with open(cpath, "w", encoding="utf-8") as wf:
            json.dump({"cue": cue, "movements": row.get("movements", [])}, wf, indent=2, ensure_ascii=False)
        cmd = [
            py,
            str(mjlab),
            "--config",
            str(cpath),
            "--out",
            str(gpath),
            "--earth-gravity",
            "--hold-base",
            "--pd-scale",
            "2.0",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(unitree_rl_mjlab_root()))
        if r.returncode != 0 or not gpath.exists():
            tqdm.write(f"skip g{idx}: rc={r.returncode} {(r.stderr or '')[-200:]}")


if __name__ == "__main__":
    fire.Fire(run)
