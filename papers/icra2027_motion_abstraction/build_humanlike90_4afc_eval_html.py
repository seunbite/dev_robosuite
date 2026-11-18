#!/usr/bin/env python3
"""Build interactive 4AFC human-eval HTML for the 90 human-like cues.

Hard negatives (hybrid):
  1× nearest Motion-Code foil (cuespec-GT pose configs, humanoid)
  2× meaning neighbors (manual map for paper30; same coarse category for rest)

Stimulus GIFs (prefer generated motion, fall back to cuespec GT pose):
  paper30  → Gemini / GT humanoid+tiago
  pilot75  → Gemini pilot75 when present, else cuespec-GT humanoid
"""
from __future__ import annotations

import hashlib
import html as html_mod
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PAPER = Path(__file__).resolve().parent
REPO = Path("/Users/sb/Downloads/workspace/dev_robosuite")
ADHOC = REPO / "adhoc/generation/robotarm"
sys.path[:0] = [str(PAPER), str(ADHOC)]

from analyze_hard_negative_strategies import MEANING_NEIGHBORS as PAPER30_NEIGHBORS  # noqa: E402
from score_embodiment_h4 import score_instance  # noqa: E402

CANVAS = Path(
    "/Users/sb/.cursor/projects/"
    "Users-sb-Downloads-workspace-dev-robosuite-papers-icra2027-motion-abstraction/"
    "canvases/gesture-critical-factors-105.canvas.tsx"
)
CAT_JSON = PAPER / "cuespec_category_accuracy.json"
CUESPEC_CFG = (
    REPO
    / "data/results/motion_configs/embodiment_motion_code_cuespec_gt"
    / "embodiment_motion_code_all.json"
)
OUT_DIR = PAPER / "humanlike90_4afc_eval"
OUT_HTML = OUT_DIR / "index.html"
OUT_TRIALS = OUT_DIR / "trials.json"
OUT_NEIGHBORS = OUT_DIR / "hard_negatives.json"
MEDIA = OUT_DIR / "media"
SEED = 20260901

GT_H = REPO / "data/results/render/embodiment_motion_code_gt/humanoid/gif"
GT_T = REPO / "data/results/render/embodiment_motion_code_gt/tiago/gif"
GEM_H = (
    REPO
    / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro/humanoid/gif"
)
GEM_T = (
    REPO
    / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro/tiago/gif"
)
PIL_H = (
    REPO
    / "data/results/render"
    / "embodiment_motion_code_gemini_gemini-2.5-pro_pilot75_humanoidlike"
    / "humanoid/gif"
)
PIL_T = (
    REPO
    / "data/results/render"
    / "embodiment_motion_code_gemini_gemini-2.5-pro_pilot75_humanoidlike"
    / "tiago/gif"
)
SPEC_H = (
    REPO
    / "data/results/render/embodiment_motion_code_cuespec_gt/humanoid/gif"
)

# Extra meaning foils for pilot/extra cues (same coarse-family confusions).
EXTRA_NEIGHBORS: dict[str, tuple[str, str, str]] = {
    "self_scratch_head_confused": ("tap_temple_think", "rub_eye_tired", "circle_temple_crazy"),
    "pat_stomach": ("hand_over_heart_sincerity", "self_hug", "pat_self_back"),
    "fan_face_hot": ("cover_yawn_sleepy", "rub_eye_tired", "cover_mouth_gasp"),
    "tap_temple_think": ("circle_temple_crazy", "self_scratch_head_confused", "uncertainty_wave"),
    "circle_temple_crazy": ("tap_temple_think", "air_quotes", "self_scratch_head_confused"),
    "lean_forward_interest": ("abandon_action_drop_turn", "apology_forward_lowered_hold", "request_turn"),
    "cover_mouth_gasp": ("cover_yawn_sleepy", "shh_gesture", "facepalm"),
    "cheers_toast": ("highfive_invite", "fistbump_invite", "thumbs_up_approval"),
    "highfive_imitation": ("highfive_invite", "fistbump_invite", "clap_applause"),
    "fistbump_invite": ("highfive_invite", "fist_pump", "thumbs_up_approval"),
    "pat_self_back": ("flex_bicep", "thumbs_up_approval", "raise_fists_victory"),
    "throat_cut_stop": ("stop_palm_out", "interrupt_hard", "shh_gesture"),
    "visor_search": ("salute", "gaze_shift_avoid", "point_you"),
    "yield_turn": ("request_turn", "polite_decline_small_push", "hesitation_pause_hold"),
    "request_turn": ("yield_turn", "raise_index_one_moment", "clarification_request_palm_tilt"),
    "guiding_arm": ("point_you", "escorting_gesture", "beckon_come_here"),
    "protective_arm_out": ("stop_palm_out", "guiding_arm", "arms_open_welcome"),
    "slow_down_request_palm_down": ("speed_up_circular_cue", "stop_palm_out", "interrupt_soft"),
    "firm_accept_forward_reach": (
        "hesitant_accept_half_reach",
        "conditional_accept_pause_then_reach",
        "exploration_reach",
    ),
    "shh_gesture": ("cover_mouth_gasp", "stop_palm_out", "raise_index_one_moment"),
    "scale_expand": ("expand_space", "depict_big_bowl", "contain_space"),
    "hesitant_accept_half_reach": (
        "firm_accept_forward_reach",
        "unsure_response_hand_hover_stop",
        "cancel_previous_offer_retract",
    ),
    "cancel_previous_offer_retract": (
        "polite_decline_small_push",
        "hesitant_accept_half_reach",
        "yield_turn",
    ),
    "conditional_accept_pause_then_reach": (
        "firm_accept_forward_reach",
        "hesitation_pause_hold",
        "hesitant_accept_half_reach",
    ),
    "final_decision_down_press": (
        "emphasize_point_downbeat",
        "stop_palm_out",
        "thumbs_up_approval",
    ),
    "resume_speaking_pull": ("request_turn", "beckon_come_here", "continue_on_forward_roll"),
    "emphasize_point_downbeat": (
        "final_decision_down_press",
        "point_you",
        "fist_pump",
    ),
    "prepare_action_raise_hold": (
        "start_action_forward_snap",
        "raise_index_one_moment",
        "highfive_invite",
    ),
    "exploration_reach": (
        "firm_accept_forward_reach",
        "hesitant_accept_half_reach",
        "point_you",
    ),
    "uncertainty_wave": ("shrug_idk", "unsure_response_hand_hover_stop", "air_quotes"),
    "clarification_request_palm_tilt": (
        "request_turn",
        "uncertainty_wave",
        "shrug_idk",
    ),
    "speed_up_circular_cue": (
        "draw_circle_repeat",
        "continue_on_forward_roll",
        "slow_down_request_palm_down",
    ),
    "polite_decline_small_push": (
        "cancel_previous_offer_retract",
        "stop_palm_out",
        "yield_turn",
    ),
    "unsure_response_hand_hover_stop": (
        "hesitation_pause_hold",
        "hesitant_accept_half_reach",
        "uncertainty_wave",
    ),
    "continue_on_forward_roll": (
        "speed_up_circular_cue",
        "beckon_come_here",
        "start_action_forward_snap",
    ),
    "start_action_forward_snap": (
        "prepare_action_raise_hold",
        "firm_accept_forward_reach",
        "emphasize_point_downbeat",
    ),
    "hesitation_pause_hold": (
        "unsure_response_hand_hover_stop",
        "interrupt_soft",
        "conditional_accept_pause_then_reach",
    ),
    "interrupt_hard": ("interrupt_soft", "stop_palm_out", "throat_cut_stop"),
    "interrupt_soft": ("interrupt_hard", "hesitation_pause_hold", "slow_down_request_palm_down"),
    "self_hug": ("cross_arms_defensive", "rub_arms_cold", "hand_over_heart_sincerity"),
    "rub_arms_cold": ("self_hug", "self_scratch_head_confused", "pat_stomach"),
    "dominance_high_open_hold": (
        "arms_open_welcome",
        "hands_up_surrender",
        "expand_space",
    ),
    "contain_space": ("compress_space", "expand_space", "scale_expand"),
    "expand_space": ("contain_space", "arms_open_welcome", "dominance_high_open_hold"),
    "compress_space": ("contain_space", "cross_arms_defensive", "scale_expand"),
    "balance_scale": ("shrug_idk", "push_pull_dual", "uncertainty_wave"),
    "push_pull_dual": ("balance_scale", "beckon_come_here", "cancel_previous_offer_retract"),
    "abandon_action_drop_turn": (
        "frustration_down_throw",
        "yield_turn",
        "lean_forward_interest",
    ),
    "escorting_gesture": ("guiding_arm", "beckon_come_here", "arms_open_welcome"),
    "wait_for_me_backward_palm": (
        "beckon_come_here",
        "stop_palm_out",
        "raise_index_one_moment",
    ),
    "prevent_fall_support_reach": (
        "firm_accept_forward_reach",
        "protective_arm_out",
        "exploration_reach",
    ),
    "frustration_down_throw": (
        "abandon_action_drop_turn",
        "final_decision_down_press",
        "fist_pump",
    ),
    "submission_lowered_open_palm": (
        "apology_forward_lowered_hold",
        "hands_up_surrender",
        "shrug_idk",
    ),
    "apology_forward_lowered_hold": (
        "submission_lowered_open_palm",
        "palms_together_thank_you",
        "hand_over_heart_sincerity",
    ),
    "chin_flick_dismiss": ("polite_decline_small_push", "stop_palm_out", "throat_cut_stop"),
    "cover_ears_too_loud": ("facepalm", "cover_mouth_gasp", "self_hug"),
    "disagreement_cross_then_release": (
        "cross_arms_defensive",
        "stop_palm_out",
        "shake_head_no",
    ),
    "thumbs_up_approval": ("fist_pump", "clap_applause", "highfive_invite"),
    "clap_applause": ("thumbs_up_approval", "highfive_imitation", "raise_fists_victory"),
    "hand_over_heart_sincerity": (
        "palms_together_thank_you",
        "pat_stomach",
        "apology_forward_lowered_hold",
    ),
}


def cue_rows() -> list[dict[str, Any]]:
    text = CANVAS.read_text(encoding="utf-8")
    cats = json.loads(CAT_JSON.read_text(encoding="utf-8"))["cue_to_cat"]
    rows = []
    for m in re.finditer(r'index:\s*(\d+),\s*cue:\s*"([^"]+)"', text):
        idx, cue = int(m.group(1)), m.group(2)
        rows.append({"idx": idx, "cue": cue, "category": cats[cue]})
    assert len(rows) == 90
    return rows


def pretty(cue: str) -> str:
    return cue.replace("_", " ")


def _stable(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def find_gif(cue: str, idx: int, plat: str) -> Path | None:
    """Prefer Gemini generation; fall back to GT / cuespec GT."""
    stem = f"emb_g{idx:02d}_{cue}.gif"
    if plat == "humanoid":
        roots = [GEM_H, GT_H, PIL_H, SPEC_H]
    else:
        roots = [GEM_T, GT_T, PIL_T, SPEC_H]  # SPEC only humanoid; last-resort
    for root in roots:
        if not root.is_dir():
            continue
        p = root / stem
        if p.is_file():
            return p
        hits = sorted(root.glob(f"*_{cue}.gif"))
        if hits:
            return hits[0]
    # humanoid cuespec always exists for all 90
    if plat == "tiago":
        return find_gif(cue, idx, "humanoid")
    return None


def link_media(src: Path | None, dest_name: str) -> str | None:
    if src is None or not src.is_file():
        return None
    MEDIA.mkdir(parents=True, exist_ok=True)
    dest = MEDIA / dest_name
    if dest.is_symlink() or dest.exists():
        try:
            dest.unlink()
        except OSError:
            pass
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        import shutil

        shutil.copy2(src, dest)
    return f"media/{dest_name}"


def meaning_neighbors(cue: str, pool: list[str], category: str, by_cat: dict[str, list[str]]) -> list[str]:
    manual = dict(PAPER30_NEIGHBORS)
    manual.update(EXTRA_NEIGHBORS)
    out: list[str] = []
    if cue in manual:
        for n in manual[cue]:
            if n in pool and n != cue and n not in out:
                out.append(n)
    # fill from same coarse category
    sibs = [c for c in by_cat.get(category, []) if c != cue]
    rng = random.Random(_stable(f"sib.{cue}"))
    rng.shuffle(sibs)
    for n in sibs:
        if n not in out:
            out.append(n)
        if len(out) >= 3:
            break
    # last resort: any other cue
    if len(out) < 3:
        others = [c for c in pool if c != cue and c not in out]
        rng.shuffle(others)
        out.extend(others[: 3 - len(out)])
    return out[:3]


def motion_rank(
    target_cfg: dict[str, Any],
    others: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked = []
    for foil in others:
        if foil["cue"] == target_cfg["cue"]:
            continue
        scored = score_instance(
            target_cfg["platforms"]["humanoid"],
            foil["platforms"]["humanoid"],
        )
        attr = scored["attribute"]
        inst = scored["instance"]
        ranked.append(
            {
                "cue": foil["cue"],
                "f1": float(attr["f1"]),
                "exact": bool(inst["exact_match"]),
                "type_seq": bool(inst["component_type_sequence_exact"]),
                "n_edit": len(attr["missing"]) + len(attr["extra"]),
                "_s": _stable(f"{target_cfg['cue']}->{foil['cue']}"),
            }
        )
    ranked.sort(
        key=lambda x: (x["f1"], x["exact"], x["type_seq"], -x["n_edit"], -x["_s"]),
        reverse=True,
    )
    for x in ranked:
        x.pop("_s")
    return ranked


def hybrid(motion: list[dict[str, Any]], semantic: list[str]) -> list[str]:
    selected: list[str] = []
    if motion:
        selected.append(motion[0]["cue"])
    for cue in semantic:
        if cue not in selected:
            selected.append(cue)
        if len(selected) == 3:
            return selected
    for item in motion[1:]:
        if item["cue"] not in selected:
            selected.append(item["cue"])
        if len(selected) == 3:
            break
    return selected[:3]


def build_trials() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = cue_rows()
    pool = [r["cue"] for r in rows]
    by_cat: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["cue"])

    cfgs_all = {e["cue"]: e for e in json.loads(CUESPEC_CFG.read_text(encoding="utf-8"))}
    # Restrict motion bank to the 90-cue eval pool (drop extras like retry_action_reset_circle).
    cfgs = {c: cfgs_all[c] for c in pool if c in cfgs_all}
    neg_payload: dict[str, Any] = {"strategy": "hybrid", "rows": []}
    trials: list[dict[str, Any]] = []
    rng = random.Random(SEED)

    for r in rows:
        cue, idx = r["cue"], r["idx"]
        semantic = meaning_neighbors(cue, pool, r["category"], by_cat)
        motion = motion_rank(cfgs[cue], list(cfgs.values())) if cue in cfgs else []
        foils = hybrid(motion, semantic)
        foils = [f for f in foils if f in pool and f != cue][:3]
        if len(foils) < 3:
            for n in semantic + [m["cue"] for m in motion]:
                if n in pool and n != cue and n not in foils:
                    foils.append(n)
                if len(foils) == 3:
                    break
        if len(foils) != 3:
            raise RuntimeError(f"bad foils for {cue}: {foils}")

        choices = [cue, *foils]
        rng.shuffle(choices)
        answer_index = choices.index(cue)

        h_src = find_gif(cue, idx, "humanoid")
        t_src = find_gif(cue, idx, "tiago")
        # Opaque filenames — cue names must not appear in the URL bar / img src.
        h_rel = link_media(h_src, f"stim_h_{idx:03d}.gif")
        t_rel = (
            link_media(t_src, f"stim_t_{idx:03d}.gif")
            if t_src and t_src != h_src
            else h_rel
        )

        trial = {
            "trial_id": f"c{idx:02d}_{cue}",
            "idx": idx,
            "cue": cue,
            "label": pretty(cue),
            "category": r["category"],
            "negatives": foils,
            "choices": [{"id": c, "label": pretty(c)} for c in choices],
            "answer_index": answer_index,
            "answer_id": cue,
            "gifs": {"humanoid": h_rel, "tiago": t_rel},
            "gif_source": {
                "humanoid": str(h_src) if h_src else None,
                "tiago": str(t_src) if t_src else None,
            },
        }
        trials.append(trial)
        neg_payload["rows"].append(
            {
                "idx": idx,
                "cue": cue,
                "category": r["category"],
                "hybrid": foils,
                "meaning3": semantic,
                "motion_top3": [m["cue"] for m in motion[:3]],
            }
        )

    return trials, neg_payload


def build_html(trials: list[dict[str, Any]]) -> str:
    """Emit trials.js + copy static template/app into OUT_DIR."""
    slim = []
    for t in trials:
        u = {k: v for k, v in t.items() if k != "gif_source"}
        slim.append(u)
    (OUT_DIR / "trials.js").write_text(
        "window.TRIALS = " + json.dumps(slim, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    template = OUT_DIR / "index.template.html"
    html = template.read_text(encoding="utf-8")
    # app.js already lives beside the template
    return html



def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trials, neg = build_trials()
    OUT_TRIALS.write_text(json.dumps(trials, indent=2, ensure_ascii=False) + "\n")
    OUT_NEIGHBORS.write_text(json.dumps(neg, indent=2, ensure_ascii=False) + "\n")
    OUT_HTML.write_text(build_html(trials), encoding="utf-8")

    n_h = sum(1 for t in trials if t["gifs"]["humanoid"])
    n_t = sum(1 for t in trials if t["gifs"]["tiago"])
    print(f"wrote {OUT_HTML}")
    print(f"trials={len(trials)} humanoid_gif={n_h} tiago_gif={n_t}")
    print(f"negatives {OUT_NEIGHBORS}")
    # sample
    s = trials[0]
    print("sample", s["trial_id"], "choices", [c["id"] for c in s["choices"]], "ans", s["answer_id"])


if __name__ == "__main__":
    main()
