#!/usr/bin/env python3
"""One-off: normalize data/seed + move motion JSON to data/results/motion_configs."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "data" / "seed"
RES = ROOT / "data" / "results"
MC_MANIP = RES / "motion_configs" / "manipulator"
MC_GOOGLE = RES / "motion_configs" / "google_robot"
MC_QUAD = RES / "motion_configs" / "quadruped"
BUNDLES = MC_MANIP / "bundles"
REMAINDER = SEED / "_remainder"


def move_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.move(str(src), str(dst))


def move_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    if not any(src.iterdir()):
        try:
            src.rmdir()
        except OSError:
            pass
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.move(str(src), str(dst))


def main() -> int:
    (SEED / "yml").mkdir(parents=True, exist_ok=True)
    REMAINDER.mkdir(parents=True, exist_ok=True)
    MC_MANIP.mkdir(parents=True, exist_ok=True)
    MC_GOOGLE.mkdir(parents=True, exist_ok=True)
    MC_QUAD.mkdir(parents=True, exist_ok=True)
    BUNDLES.mkdir(parents=True, exist_ok=True)

    # --- flatten old seed/prompt into temp then rebuild ---
    flat_prompt = SEED / "prompt"
    backup = SEED / "_prompt_flat_backup"
    if flat_prompt.is_dir() and not (SEED / "prompt" / "manipulator").exists():
        flat_prompt.rename(backup)
        (SEED / "prompt" / "manipulator").mkdir(parents=True, exist_ok=True)
        (SEED / "prompt" / "google_robot").mkdir(parents=True, exist_ok=True)
        (SEED / "prompt" / "quadruped").mkdir(parents=True, exist_ok=True)
        if backup.is_dir():
            for f in backup.iterdir():
                if f.is_file():
                    move_file(f, SEED / "prompt" / "manipulator" / f.name)
                elif f.is_dir():
                    move_tree(f, REMAINDER / "prompt_subdirs" / f.name)
            try:
                backup.rmdir()
            except OSError:
                shutil.rmtree(backup, ignore_errors=True)
    else:
        (SEED / "prompt" / "manipulator").mkdir(parents=True, exist_ok=True)
        (SEED / "prompt" / "google_robot").mkdir(parents=True, exist_ok=True)
        (SEED / "prompt" / "quadruped").mkdir(parents=True, exist_ok=True)

    (SEED / "shots" / "manipulator").mkdir(parents=True, exist_ok=True)
    (SEED / "shots" / "google_robot").mkdir(parents=True, exist_ok=True)
    (SEED / "shots" / "quadruped").mkdir(parents=True, exist_ok=True)

    for name in (
        "cues.yml",
        "cues_new.yml",
        "locomotion_cues.yml",
        "persona_cue_subset_v1.yml",
        "persona_tag_catalog.yml",
    ):
        move_file(SEED / name, SEED / "yml" / name)

    for src_dir, robot in (
        (SEED / "google_robot" / "prompt", "google_robot"),
        (SEED / "quadruped_go2" / "prompt", "quadruped"),
    ):
        if src_dir.is_dir():
            for f in src_dir.iterdir():
                if f.is_file():
                    move_file(f, SEED / "prompt" / robot / f.name)

    for src_dir, robot in (
        (SEED / "google_robot" / "shot", "google_robot"),
        (SEED / "quadruped_go2" / "shot", "quadruped"),
    ):
        if src_dir.is_dir():
            for f in src_dir.iterdir():
                if f.is_file():
                    move_file(f, SEED / "shots" / robot / f.name)

    qcfg = SEED / "quadruped_go2" / "config"
    if qcfg.is_dir():
        for f in sorted(qcfg.glob("*.json")):
            move_file(f, MC_QUAD / f.name)
        p19 = qcfg / "prompt19_locomotion_configs"
        if p19.is_dir():
            move_tree(p19, MC_QUAD / "prompt19_locomotion_configs")

    google_cfg = SEED / "google_robot" / "config"
    if google_cfg.is_dir():
        for f in sorted(google_cfg.glob("*.json")):
            move_file(f, MC_GOOGLE / f.name)

    for f in sorted(SEED.glob("shot_configs*.json")):
        move_file(f, SEED / "shots" / "manipulator" / f.name)
    move_file(SEED / "shot_configs.json", SEED / "shots" / "manipulator" / "shot_configs.json")

    for pat in ("motion_configs*.json", "motion_config*.json"):
        for f in sorted(SEED.glob(pat)):
            move_file(f, MC_MANIP / f.name)

    bundle_prefixes = ("baseline_", "q4_", "path_")
    bundle_exact = {
        "gif_compression_probe",
        "goodset_human_labeling",
        "motion_configs_prompt_v18_anxious_var_compare_renders",
        "prompt19_sophisticated_alpha_frames",
        "prompt19_sophisticated_path_trajectory_compare",
        "vlm_pairwise_prompt19_compare",
        "q4_analysis",
        "q4_experiment_combined",
        "q4_fill_sophisticated_contextual",
        "q4_generic_recoil_hold_experiment",
        "baseline_prompt19_combined30",
        "_legacy_flat",
    }
    for d in sorted(SEED.iterdir()):
        if not d.is_dir():
            continue
        if d.name in ("yml", "prompt", "shots", "_remainder", "_prompt_flat_backup"):
            continue
        if d.name.startswith(bundle_prefixes) or d.name in bundle_exact:
            move_tree(d, BUNDLES / d.name)

    for name in (
        "best_configs.json",
        "closest_poses_results.jsonl",
        "contextual.pptx",
        "contextual_nr.pptx",
        "iconic.pptx",
        "iconic_nr.pptx",
        "cues.txt",
        "motion_list.jsonl",
        "personality_list.json",
        "persona_triptych_debug_anxious.json",
        "persona_triptych_debug_joyful.json",
        "persona_triptych_eval_anxious.json",
        "persona_triptych_eval_joyful.json",
        "prompt19_contextual.json",
        "prompt19_iconic.json",
        "prompt19_sophisticated_contextual.json",
        "prompt19_sophisticated_iconic.json",
        "prompt19_contextual_multirobot_no_reasoning_only_20260411_ko.pptx",
    ):
        move_file(SEED / name, REMAINDER / name)

    for f in list(SEED.glob("~$*")):
        try:
            f.unlink()
        except OSError:
            pass

    for d in (SEED / "google_robot", SEED / "quadruped_go2", SEED / "manipulator"):
        if not d.is_dir():
            continue
        remainder_sub = REMAINDER / d.name
        for f in list(d.rglob("*")):
            if f.is_file():
                rel = f.relative_to(d)
                move_file(f, remainder_sub / rel)
        shutil.rmtree(d, ignore_errors=True)

    print("OK: seed layout migration", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
