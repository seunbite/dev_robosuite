#!/usr/bin/env python3
"""Generate ~10 manipulator motion configs with updated prompt_v19_sophisticated + existing shots.

Skips cues that appear in shot_configs (by idx → canonical cue from merged reference).
Writes to a dedicated JSON (does not append to merged_104 by default).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import fire

_ROBOTARM = Path(__file__).resolve().parent
_REPO = _ROBOTARM.parents[2]
for p in (_REPO, _ROBOTARM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from adhoc.generation.embodiment_sources import resolve_embodiment_paths  # noqa: E402
from legacy.config_gen_single import generate_motion_config  # noqa: E402

DEFAULT_MERGED = (
    _REPO
    / "data/results/motion_configs/manipulator"
    / "motion_configs_prompt_v19_sophisticated_no_reasoning_merged_104.json"
)

# Cues with clear pose semantics; none of the 7 shot idx-mapped cues.
DEFAULT_CUES: list[tuple[int, str]] = [
    (61, "air_quotes"),
    (35, "salute"),
    (36, "visor_search"),
    (15, "beckon_come_here"),
    (8, "fist_pump"),
    (14, "point_you"),
    (66, "shh_gesture"),
    (64, "tilt_head_curious"),
    (12, "facepalm"),
    (25, "cheers_toast"),
]


def _shot_canonical_cues(shots_path: Path, merged_path: Path) -> set[str]:
    shots = json.loads(shots_path.read_text(encoding="utf-8"))
    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    by_idx = {int(r["idx"]): str(r["cue"]) for r in merged if "idx" in r and r.get("cue")}
    out: set[str] = set()
    for s in shots:
        if not isinstance(s, dict) or "idx" not in s:
            continue
        idx = int(s["idx"])
        if idx in by_idx:
            out.add(by_idx[idx])
    return out


def run(
    out_json: str | None = None,
    merged_json: str | None = None,
    model: str = "gemini-2.5-pro",
    delay: float = 2.0,
    max_retries: int = 2,
    cues: str | None = None,
    run_render: bool = False,
) -> None:
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY (or GEMINI_API_KEY) before running.")

    prompt_path, shots_path, _motion_default, _blk = resolve_embodiment_paths("manipulator")
    merged_path = Path(merged_json) if merged_json else DEFAULT_MERGED
    if not merged_path.is_file():
        raise SystemExit(f"Missing merged reference: {merged_path}")

    out_path = Path(out_json) if out_json else (
        _REPO
        / "data/results/motion_configs/manipulator"
        / "motion_configs_prompt_v19_sophisticated_ee_pilot10.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.is_file():
        out_path.write_text("[]\n", encoding="utf-8")

    shot_cues = _shot_canonical_cues(shots_path, merged_path)
    if cues:
        work: list[tuple[int, str]] = []
        for part in cues.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                idx_s, name = part.split(":", 1)
                work.append((int(idx_s), name.strip()))
            else:
                row = next(
                    (r for r in json.loads(merged_path.read_text()) if r.get("cue") == part),
                    None,
                )
                if row is None:
                    raise SystemExit(f"Unknown cue {part!r} in {merged_path}")
                work.append((int(row["idx"]), part))
    else:
        work = [(i, c) for i, c in DEFAULT_CUES if c not in shot_cues]

    if not work:
        raise SystemExit("No cues to generate (all in shots?).")

    print(f"Prompt: {prompt_path}")
    print(f"Shots:  {shots_path}")
    print(f"Out:    {out_path}")
    print(f"Skip shot cues: {sorted(shot_cues)}")
    print(f"Generating {len(work)}: {[c for _, c in work]}")

    failed: list[str] = []
    for idx, cue_name in work:
        ok = False
        last_err = ""
        for attempt in range(max_retries + 1):
            try:
                generate_motion_config(
                    cue_name=cue_name,
                    cue_idx=idx,
                    model_name=model,
                    prompt_file=str(prompt_path),
                    shots_json=str(shots_path),
                    config_json=str(out_path),
                    require_reasoning=True,
                )
                ok = True
                break
            except Exception as e:
                last_err = str(e)
                if attempt < max_retries:
                    time.sleep(delay * (2**attempt))
        if not ok:
            failed.append(f"{cue_name}: {last_err[:200]}")
            print(f"FAILED {cue_name}: {last_err[:300]}")
        else:
            print(f"OK {cue_name}")

    print(f"Done -> {out_path}  failures={len(failed)}")
    if failed:
        for line in failed:
            print(" ", line)

    if run_render:
        from render import run as render_run  # noqa: E402

        render_run(config_json=str(out_path), run_render=True, do_html=True)


if __name__ == "__main__":
    fire.Fire(run)
