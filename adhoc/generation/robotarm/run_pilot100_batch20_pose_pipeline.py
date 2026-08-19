#!/usr/bin/env python3
"""Generate + VLM verify + Text verify for pilot100 batch20 pose pipeline.

Usage (from dev_robosuite root; use micromamba robosuite env):
  source APIKEY.sh
  /Users/sb/micromamba/envs/robosuite/bin/python \\
    adhoc/generation/robotarm/run_pilot100_batch20_pose_pipeline.py --step all
  python adhoc/generation/robotarm/run_pilot100_batch20_pose_pipeline.py --step generate
  python adhoc/generation/robotarm/run_pilot100_batch20_pose_pipeline.py --step verify
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE, _HERE / "legacy"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CFG = _REPO / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot100_batch20.json"
VLM_OUT = _REPO / "data/results/verify/pose_tile_verify_pilot100_batch20_single_gemini.json"
TEXT_OUT = _REPO / "data/results/verify/pose_textonly_verify_pilot100_batch20_gemini.json"
MERGED = _REPO / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_no_reasoning_merged_104.json"

BATCH20 = [
    "hesitant_accept_half_reach",
    "cancel_previous_offer_retract",
    "conditional_accept_pause_then_reach",
    "final_decision_down_press",
    "resume_speaking_pull",
    "emphasize_point_downbeat",
    "prepare_action_raise_hold",
    "exploration_reach",
    "uncertainty_wave",
    "draw_circle_repeat",
    "clarification_request_palm_tilt",
    "speed_up_circular_cue",
    "polite_decline_small_push",
    "unsure_response_hand_hover_stop",
    "raise_index_one_moment",
    "continue_on_forward_roll",
    "start_action_forward_snap",
    "retry_action_reset_circle",
    "hesitation_pause_hold",
    "interrupt_hard",
]


def _existing_cues() -> set[str]:
    if not CFG.is_file():
        return set()
    return {r["cue"] for r in json.loads(CFG.read_text(encoding="utf-8"))}


def step_generate(model: str, delay: float) -> None:
    from generate_pilot_not_in_shots import run

    missing = [c for c in BATCH20 if c not in _existing_cues()]
    if not missing:
        print("generate: all 20 cues already in", CFG)
        return
    print(f"generate: {len(missing)} remaining -> {missing}")
    run(
        out_json=str(CFG.relative_to(_REPO)),
        model=model,
        delay=delay,
        cues=",".join(missing),
    )


def step_verify(model: str) -> None:
    import verify_pose_textonly_gemini as text_mod
    import verify_pose_tiles_gemini as vlm_mod

    if not CFG.is_file():
        raise SystemExit(f"Missing config: {CFG}")

    class Args:
        pass

    vlm = Args()
    vlm.config_json = CFG
    vlm.shots_json = _REPO / "data/seed/shots/manipulator/shot_configs_v19_sophisticated.json"
    vlm.tile_dir = _REPO / "data/results/visualize/pose_groups_12"
    vlm.tile_pick_json = _REPO / "data/results/verify/pose_tile_pick_by_group.json"
    vlm.selected_tile_dir = _REPO / "data/results/visualize/pose_groups_12_selected"
    vlm.export_selected = False
    vlm.model = model
    vlm.fewshot_n = 4
    vlm.out_json = VLM_OUT
    vlm.out_md = VLM_OUT.with_suffix(".md")
    vlm.no_checkpoint = True

    txt = Args()
    txt.config_json = CFG
    txt.shots_json = vlm.shots_json
    txt.model = model
    txt.fewshot_n = 4
    txt.out_json = TEXT_OUT

    print("verify VLM ->", VLM_OUT)
    vlm_mod.run(vlm)
    print("verify Text ->", TEXT_OUT)
    text_mod.run(txt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=("generate", "verify", "all"), default="all")
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--delay", type=float, default=5.0, help="Seconds between generation retries")
    args = ap.parse_args()

    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        key_sh = _REPO / "APIKEY.sh"
        if key_sh.is_file():
            for line in key_sh.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("export GOOGLE_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip("'\"")
                    os.environ["GOOGLE_API_KEY"] = val
                    break
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY (or add dev_robosuite/APIKEY.sh).")

    if args.step in ("generate", "all"):
        step_generate(args.model, args.delay)
    if args.step in ("verify", "all"):
        n = len(json.loads(CFG.read_text())) if CFG.is_file() else 0
        if n == 0:
            raise SystemExit("No configs to verify.")
        print(f"verify: {n} configs in {CFG}")
        step_verify(args.model)


if __name__ == "__main__":
    main()
