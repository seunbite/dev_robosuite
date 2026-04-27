#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import fire
import yaml
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adhoc.utils.repo_paths import motion_configs_results_dir, resolve_seed_prompt_txt, resolve_seed_shots_json, seed_yml_dir  # noqa: E402
from legacy.config_gen_single import generate_motion_config  # noqa: E402
from legacy.motion_generation_core import MotionGenerator, _select_initial_poses, generate  # noqa: E402,F401


def run(
    yaml_path: str | None = None,
    cue_group: str = "iconic",
    prompt_file: str | None = None,
    shots_json: str | None = None,
    config_json: str | None = None,
    model: str = "gemini-2.5-pro",
    delay: float = 2.0,
    max_retries: int = 2,
    generate_only: list[int] | None = None,
    skip_fewshot_cues: bool = False,
    run_render: bool = True,
    sim_robot: str = "IIWA",
    hz: int = 4,
    top_k: int = 5,
) -> None:
    ypath = Path(yaml_path) if yaml_path else seed_yml_dir() / "cues_new.yml"
    prompt = str(Path(prompt_file)) if prompt_file else str(resolve_seed_prompt_txt("manipulator"))
    shots = str(Path(shots_json)) if shots_json else str(resolve_seed_shots_json("manipulator"))
    out = Path(config_json) if config_json else motion_configs_results_dir("manipulator") / f"motion_configs_{cue_group}.json"
    os.makedirs(out.parent, exist_ok=True)

    with open(ypath, "r", encoding="utf-8") as f:
        cues = yaml.safe_load(f)
    if cue_group not in cues:
        raise ValueError(f"cue_group={cue_group!r} not in {ypath}")
    indexed = list(enumerate(cues[cue_group].items()))
    if generate_only:
        indexed = [(i, kv) for i, kv in indexed if i in set(generate_only)]

    if skip_fewshot_cues and Path(shots).exists():
        import json

        with open(shots, "r", encoding="utf-8") as f:
            shot_list = json.load(f)
        few = {s.get("cue") for s in shot_list if isinstance(s, dict)}
        indexed = [(i, (k, d)) for i, (k, d) in indexed if k not in few]

    failed = 0
    for cue_idx, (cue_name, _desc) in tqdm(indexed, desc="motion_generation[robotarm]"):
        ok = False
        last_err = ""
        for attempt in range(max_retries + 1):
            try:
                generate_motion_config(
                    cue_name=cue_name,
                    cue_idx=cue_idx,
                    model_name=model,
                    prompt_file=prompt,
                    shots_json=shots,
                    config_json=str(out),
                )
                ok = True
                break
            except Exception as e:
                last_err = str(e)
                if attempt < max_retries:
                    time.sleep(delay * (2**attempt))
        if not ok:
            failed += 1
            tqdm.write(f"FAILED c{cue_idx} {cue_name}: {last_err[:300]}")

    print(f"Done. failures={failed} -> {out}")
    if run_render:
        from render import run as render_run  # local stage

        render_run(
            config_json=str(out),
            sim_robot=sim_robot,
            hz=hz,
            top_k=top_k,
        )


if __name__ == "__main__":
    fire.Fire(run)
