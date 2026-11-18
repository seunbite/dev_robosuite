#!/usr/bin/env python3
"""Score LLM Motion Programs against discrete cue-spec tokens from the canvas.

Orient / Loc come from required ``pose(...)`` atoms; Movement from ``move``/``path``.
A cue attribute is N/A when the spec is ``—`` / empty / None.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO = Path("/Users/sb/Downloads/workspace/dev_robosuite")
OUT_JSON = REPO / "data/results/evaluation/humanlike_cuespec_scores.json"
OUT_MD = REPO / "papers/icra2027_motion_abstraction/humanlike_cuespec_scores.md"

PAPER_LLM = (
    REPO
    / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro"
    / "embodiment_motion_code_all.json"
)
PILOT_LLM = (
    REPO
    / "data/results/motion_configs"
    / "embodiment_motion_code_gemini_gemini-2.5-pro_pilot75_humanoidlike"
    / "embodiment_motion_code_all.json"
)

# Discrete required components (from gesture-critical-factors canvas).
# pose/movement/path use Motion-Code-like atoms; "—" = not required.
REQUIRED: dict[str, dict[str, str]] = {
    "nod_yes": {"pose": "pose(head=center, left_arm=still)", "movement": "move(head, tilt, ±, rep≥2)", "path": "—"},
    "shake_head_no": {"pose": "pose(head=center, left_arm=still)", "movement": "move(head, pan, ±, rep≥2)", "path": "—"},
    "draw_circle_repeat": {
        "pose": "pose(dir=front, grip=vert, z≈60, left_arm=still)",
        "movement": "—",
        "path": "path(ee, arc, plane=xz|yz, sweep=360°, rep≥2)",
    },
    "gaze_shift_avoid": {"pose": "pose(head=center, left_arm=still)", "movement": "move(head, pan, away)", "path": "—"},
    "fist_pump": {
        "pose": "pose(dir=up, grip=vert|fist, z≈60, x≈75, left_arm=still)",
        "movement": "move(shoulder, z, +, short thrust)",
        "path": "—",
    },
    "tilt_head_curious": {"pose": "pose(head=_, left_arm=still)", "movement": "move(head, tilt/roll, to side, hold)", "path": "—"},
    "cover_yawn_sleepy": {
        # in|horiz → palmdown; up|horiz → palmback (palm toward mouth/face)
        "pose": "pose(dir=up|in, grip=horiz|palmdown|palmback, z≈65, x≈70, left_arm=still)",
        "movement": "—",
        "path": "—",
    },
    "rub_eye_tired": {
        "pose": "pose(dir=in, grip=vert, z≈80, x≈75, left_arm=still)",
        "movement": "—",
        "path": "path(ee, arc, plane=yz, r small, rep≥2)",
    },
    "point_you": {"pose": "pose(dir=front, grip=vert, z≈50, left_arm=still)", "movement": "—", "path": "—"},
    "point_self": {"pose": "pose(dir=back|in, grip=_, x≈near_chest, z≈mid, left_arm=still)", "movement": "—", "path": "—"},
    "beckon_come_here": {
        "pose": "pose(dir=front, grip=vert|palmin, z≈80, x≈60, left_arm=still)",
        "movement": "move(wrist, x, ±, rep≥2)",
        "path": "—",
    },
    # palmfront only valid with up|horiz (not front|horiz)
    "stop_palm_out": {"pose": "pose(dir=up, grip=horiz|palmfront, z≈60, left_arm=still)", "movement": "—", "path": "—"},
    "salute": {"pose": "pose(dir=in, grip=horiz|palmdown, z≈80, x≈40, left_arm=still)", "movement": "—", "path": "—"},
    "raise_index_one_moment": {
        "pose": "pose(dir=out|up, grip=vert, z≈70, left_arm=still)",
        "movement": "move(shoulder, z, +)",
        "path": "—",
    },
    "highfive_invite": {"pose": "pose(dir=up, grip=horiz|palmfront, z≈shoulder, left_arm=still)", "movement": "—", "path": "—"},
    "shrug_idk": {
        "pose": "pose(dir=out, grip=horiz|palmup, left_arm=mirror, y≈20, z≈50)",
        "movement": "move(shoulder, z, +)",
        "path": "—",
    },
    "flex_bicep": {"pose": "pose(dir=in, grip=vert, elbow=T, z≈80, left_arm=still)", "movement": "—", "path": "—"},
    "air_quotes": {
        # palms toward viewer: palmfront only with up|horiz
        "pose": "pose(dir=up, grip=horiz|palmfront, z≈70, left_arm=mirror)",
        "movement": "move(wrist, z, ±, rep≥2)",
        "path": "—",
    },
    "arms_open_welcome": {
        "pose": "pose(dir=out|front, grip=horiz|palmup, left_arm=mirror, y≈0, z≈mid)",
        "movement": "move(shoulder, y, outward)",
        "path": "—",
    },
    "hands_up_surrender": {
        "pose": "pose(dir=up, grip=horiz|palmfront, z≈75, left_arm=mirror)",
        "movement": "—",
        "path": "—",
    },
    "stretch_arms_up": {"pose": "pose(dir=up, grip=_, z≈100, left_arm=mirror)", "movement": "—", "path": "—"},
    "raise_fists_victory": {
        "pose": "pose(dir=up, grip=fist, z≈90, left_arm=mirror)",
        "movement": "move(shoulder, z, +)",
        "path": "—",
    },
    "cross_arms_defensive": {
        "pose": "pose(dir=in, grip=_, z≈60, x≈70, left_arm=mirror)",
        "movement": "—",
        "path": "—",
    },
    "palms_together_thank_you": {
        "pose": "pose(dir=up, grip=vert, x≈70, y≈70, left_arm=mirror)",
        "movement": "move(head, tilt, −)",
        "path": "—",
    },
    # palmback valid with in|vert (not in|horiz)
    "facepalm": {"pose": "pose(dir=in, grip=vert|palmback, z≈80, x≈65, left_arm=still)", "movement": "—", "path": "—"},
    "depict_tree": {
        "pose": "pose(dir=out, grip=horiz, y≈0, left_arm=mirror, torso=tall)",
        "movement": "—",
        "path": "—",
    },
    "depict_big_bowl": {
        "pose": "pose(dir=out, grip=horiz|palmup, z≈55, left_arm=mirror)",
        "movement": "—",
        "path": "path(ee, arc, plane=yz)",
    },
    "depict_lion": {"pose": "pose(dir=front, grip=claw|palmdown, z≈85, left_arm=mirror)", "movement": "—", "path": "—"},
    "depict_small_object": {"pose": "pose(dir=up, grip=pinch, z≈60, x≈65, left_arm=still)", "movement": "—", "path": "—"},
    "depict_flying_bird": {
        "pose": "pose(dir=out, grip=horiz, y≈0, left_arm=mirror)",
        "movement": "move(shoulder, z, ±, rep≥3)",
        "path": "—",
    },
    "self_scratch_head_confused": {
        "pose": "pose(dir=in, grip=vert|palmback, x=50, y=53, z=81, left_arm=still)",
        "movement": "move(wrist, z|y, ±, rep)",
        "path": "—",
    },
    "pat_stomach": {"pose": "pose(dir=in, grip=vert|palmback, x=85, y=55, z=45, left_arm=still)", "movement": "move(wrist, x, −, rep)", "path": "—"},
    "fan_face_hot": {"pose": "pose(dir=in, grip=horiz|palmdown, x=70, y=40, z=80, left_arm=still)", "movement": "move(wrist, z|y, ±, rep)", "path": "—"},
    "tap_temple_think": {"pose": "pose(dir=in, grip=vert|palmback, x=50, y=53, z=82, left_arm=still)", "movement": "move(wrist, y, ±, rep)", "path": "—"},
    "circle_temple_crazy": {
        "pose": "pose(dir=in, grip=vert|palmback, x=50, y=53, z=81, left_arm=still)",
        "movement": "—",
        "path": "path(ee, arc, plane=xz|yz, rep)",
    },
    "lean_forward_interest": {"pose": "pose(dir=front, grip=vert|palmin, x=55, y=50, z=55, left_arm=still)", "movement": "move(torso, x, +)", "path": "—"},
    "cover_mouth_gasp": {"pose": "pose(dir=in, grip=horiz|palmdown, x=70, y=50, z=65, left_arm=still)", "movement": "move(_, x, −)", "path": "—"},
    "cheers_toast": {"pose": "pose(dir=up, grip=vert|palmin, x=65, y=45, z=70, left_arm=still)", "movement": "move(_, x|z, +)", "path": "—"},
    "highfive_imitation": {"pose": "pose(dir=up, grip=horiz|palmfront, x=70, y=50, z=75, left_arm=still)", "movement": "move(_, x|z, +)", "path": "—"},
    "fistbump_invite": {"pose": "pose(dir=front, grip=horiz|palmdown, x=88, y=55, z=58, left_arm=still)", "movement": "move(_, x, +)", "path": "—"},
    # back|vert→palmin; in|vert→palmback (palm against body)
    "pat_self_back": {"pose": "pose(dir=back, grip=vert|palmin, x=55, y=60, z=72, left_arm=still)", "movement": "move(_, z, ±, rep)", "path": "—"},
    "throat_cut_stop": {"pose": "pose(dir=in, grip=horiz|palmdown, x=60, y=50, z=55, left_arm=still)", "movement": "—", "path": "path(ee, line, axis=y)"},
    "visor_search": {
        "pose": "pose(dir=front, grip=horiz|palmdown, x=45, y=50, z=85, left_arm=still)",
        "movement": "move(head, pan, ±, rep)",
        "path": "—",
    },
    "yield_turn": {"pose": "pose(dir=front, grip=horiz|palmup, x=70, y=40, z=58, left_arm=still)", "movement": "move(_, x|z, −)", "path": "—"},
    "request_turn": {"pose": "pose(dir=up, grip=vert|palmin, x=60, y=50, z=70, left_arm=still)", "movement": "move(_, z|x, +)", "path": "—"},
    "guiding_arm": {"pose": "pose(dir=out, grip=horiz|palmup, x=60, y=25, z=55, left_arm=still)", "movement": "—", "path": "path(ee, arc|line)"},
    # palmfront valid with out|vert (not out|horiz)
    "protective_arm_out": {"pose": "pose(dir=out, grip=vert|palmfront, x=50, y=0, z=55, left_arm=still)", "movement": "move(_, y|x, +)", "path": "—"},
    "slow_down_request_palm_down": {
        "pose": "pose(dir=front, grip=horiz|palmdown, x=55, y=50, z=55, left_arm=still)",
        "movement": "move(_, z, −, rep)",
        "path": "—",
    },
    "firm_accept_forward_reach": {"pose": "pose(dir=front, grip=vert|palmin, x=65, y=50, z=55, left_arm=still)", "movement": "move(_, x, +)", "path": "—"},
    "shh_gesture": {"pose": "pose(dir=up, grip=vert|palmin, x=88, y=52, z=64, left_arm=still, bank=F)", "movement": "—", "path": "—"},
    "scale_expand": {"pose": "pose(dir=out, grip=vert|palmfront, x=55, y=20, z=55, left_arm=mirror)", "movement": "—", "path": "path(ee, arc)|move_apart"},
    "hesitant_accept_half_reach": {"pose": "pose(dir=front, grip=vert|palmin, x=50, y=50, z=55, left_arm=still)", "movement": "move(_, x, +)", "path": "—"},
    "cancel_previous_offer_retract": {"pose": "pose(dir=front, grip=horiz|palmdown, x=40, y=50, z=50, left_arm=still)", "movement": "move(_, x, −)", "path": "—"},
    "conditional_accept_pause_then_reach": {"pose": "pose(dir=front, grip=vert|palmin, x=65, y=50, z=55, left_arm=still)", "movement": "move(_, x, +)", "path": "—"},
    "final_decision_down_press": {"pose": "pose(dir=down, grip=horiz|palmfront, x=55, y=50, z=40, left_arm=still)", "movement": "move(_, z, −)", "path": "—"},
    "resume_speaking_pull": {"pose": "pose(dir=in, grip=vert|palmback, x=70, y=50, z=55, left_arm=still)", "movement": "move(_, x, −)", "path": "—"},
    "emphasize_point_downbeat": {"pose": "pose(dir=down, grip=vert|palmin, x=60, y=50, z=50, left_arm=still)", "movement": "move(_, z, −)", "path": "—"},
    "prepare_action_raise_hold": {"pose": "pose(dir=up, grip=vert|palmout, x=55, y=50, z=80, left_arm=still)", "movement": "move(_, z, +)", "path": "—"},
    "exploration_reach": {"pose": "pose(dir=front, grip=vert|palmin, x=65, y=50, z=55, left_arm=still)", "movement": "move(_, x, +)", "path": "—"},
    "uncertainty_wave": {"pose": "pose(dir=up, grip=horiz|palmfront, x=55, y=40, z=55, left_arm=still)", "movement": "move(_, y, ±, rep)", "path": "—"},
    "clarification_request_palm_tilt": {
        "pose": "pose(dir=front, grip=horiz|palmup, x=60, y=50, z=55, left_arm=still)",
        "movement": "move(wrist, _, _)",
        "path": "—",
    },
    "speed_up_circular_cue": {"pose": "pose(dir=front, grip=vert|palmin, x=60, y=50, z=55, left_arm=still)", "movement": "—", "path": "path(ee, arc, plane=yz, rep)"},
    # palmfront needs up|horiz (front|horiz → palmup/palmdown only)
    "polite_decline_small_push": {
        "pose": "pose(dir=up, grip=horiz|palmfront, x=60, y=50, z=55, left_arm=still)",
        "movement": "move(_, x, +)",
        "path": "—",
    },
    "unsure_response_hand_hover_stop": {"pose": "pose(dir=front, grip=vert|palmin, x=60, y=50, z=55, left_arm=still)", "movement": "move(_, x, +)", "path": "—"},
    "continue_on_forward_roll": {
        "pose": "pose(dir=front, grip=vert|palmin, x=60, y=50, z=55, left_arm=still)",
        "movement": "—",
        "path": "path(ee, arc, plane=xz, rep)",
    },
    "start_action_forward_snap": {"pose": "pose(dir=front, grip=vert|palmin, x=60, y=50, z=55, left_arm=still)", "movement": "move(_, x, +)", "path": "—"},
    "hesitation_pause_hold": {"pose": "pose(dir=front, grip=vert|palmin, x=55, y=50, z=55, left_arm=still)", "movement": "move(_, _, _)", "path": "—"},
    "interrupt_hard": {"pose": "pose(dir=front, grip=horiz|palmdown, x=55, y=50, z=55, left_arm=still)", "movement": "move(_, _, _)", "path": "—"},
    "interrupt_soft": {"pose": "pose(dir=up, grip=horiz|palmfront, x=55, y=50, z=60, left_arm=still)", "movement": "move(_, z, +)", "path": "—"},
    "self_hug": {"pose": "pose(dir=in, grip=vert|palmback, x=85, y=70, z=55, left_arm=mirror)", "movement": "move(_, x|y, −)", "path": "—"},
    # Loc required: hands on opposite upper arms → mid height + body-near depth.
    "rub_arms_cold": {
        # in|vert → palmback (palm against arm/body); not palmin
        "pose": "pose(dir=in, grip=vert|palmback, x=70, y=70, z=55, left_arm=mirror)",
        "movement": "move(_, z|y, ±, rep)",
        "path": "—",
    },
    "dominance_high_open_hold": {
        # up|horiz→palmfront; out|vert→palmfront
        "pose": "pose(dir=up, grip=horiz|palmfront, x=40, y=0, z=90, left_arm=mirror)",
        "movement": "—",
        "path": "—",
    },
    "contain_space": {"pose": "pose(dir=up, grip=vert|palmin, x=60, y=35, z=55, left_arm=mirror)", "movement": "move(_, y, −)", "path": "—"},
    "expand_space": {"pose": "pose(dir=up, grip=vert|palmout, x=60, y=15, z=55, left_arm=mirror)", "movement": "move(_, y, +)", "path": "—"},
    "compress_space": {"pose": "pose(dir=up, grip=vert|palmin, x=60, y=35, z=55, left_arm=mirror)", "movement": "move(_, y, −)", "path": "—"},
    "balance_scale": {
        "pose": "pose(dir=front, grip=horiz|palmup, x=50, y=20, z=55, left_arm=mirror)",
        "movement": "move(_, z, ±)",
        "path": "—",
    },
    "push_pull_dual": {"pose": "pose(dir=front, grip=vert|palmin, x=60, y=50, z=55, left_arm=opposed)", "movement": "move(_, x, ±)", "path": "—"},
    "abandon_action_drop_turn": {"pose": "pose(dir=down, grip=horiz|palmback, x=45, y=50, z=35, left_arm=still)", "movement": "move(torso, _, _)", "path": "—"},
    # contrast guiding_arm (palmup): escorting uses palmdown
    "escorting_gesture": {"pose": "pose(dir=out, grip=horiz|palmdown, x=60, y=25, z=55, left_arm=still)", "movement": "—", "path": "path(ee, arc|line)"},
    # palmfront invalid for back|horiz; open-palm back signal → palmup
    "wait_for_me_backward_palm": {"pose": "pose(dir=back, grip=horiz|palmup, x=40, y=50, z=55, left_arm=still)", "movement": "move(_, x, +)", "path": "—"},
    "prevent_fall_support_reach": {"pose": "pose(dir=front, grip=horiz|palmdown, x=95, y=55, z=52, left_arm=still)", "movement": "move(_, x, +)", "path": "—"},
    "frustration_down_throw": {
        "pose": "pose(dir=down, grip=vert|palmout, x=50, y=40, z=40, left_arm=mirror)",
        "movement": "move(_, z, −)",
        "path": "—",
    },
    "submission_lowered_open_palm": {
        "pose": "pose(dir=front, grip=horiz|palmup, x=60, y=40, z=40, left_arm=mirror)",
        "movement": "move(_, z, −)",
        "path": "—",
    },
    "apology_forward_lowered_hold": {
        "pose": "pose(dir=front, grip=horiz|palmup, x=55, y=50, z=40, left_arm=mirror)",
        "movement": "move(torso, _, _)",
        "path": "—",
    },
    "chin_flick_dismiss": {"pose": "pose(dir=up, grip=vert|palmin, x=70, y=40, z=68, left_arm=still)", "movement": "move(_, x|z, +)", "path": "—"},
    "cover_ears_too_loud": {
        # beside ears: in toward head
        "pose": "pose(dir=in, grip=vert|palmback, x=85, y=70, z=80, left_arm=mirror)",
        "movement": "move(_, y|x, −)",
        "path": "—",
    },
    "disagreement_cross_then_release": {
        "pose": "pose(dir=in, grip=vert|palmback, x=85, y=70, z=55, left_arm=mirror)",
        "movement": "move(_, y, ±)",
        "path": "—",
    },
    "thumbs_up_approval": {"pose": "pose(dir=front, grip=vert|palmin, x=70, y=50, z=62, left_arm=still)", "movement": "—", "path": "—"},
    "clap_applause": {
        "pose": "pose(dir=up, grip=vert|palmin, x=82, y=79, z=65, left_arm=mirror)",
        "movement": "move(_, y, ±, rep)",
        "path": "—",
    },
    "hand_over_heart_sincerity": {
        # in|vert → palmback (palm flat on chest)
        "pose": "pose(dir=in, grip=vert|palmback, x=70, y=52, z=58, left_arm=still)",
        "movement": "move(_, x, −)",
        "path": "—",
    },
    # canvas skips 67; map retry if present in configs
    "retry_action_reset_circle": {
        "pose": "pose(dir=front, grip=vert|palmin, x=55, y=50, z=55, left_arm=still)",
        "movement": "—",
        "path": "path(ee, arc, rep)",
    },
}

LOC_TOL = 15.0  # body-% band around ≈ targets


def _na(spec: str) -> bool:
    s = (spec or "").strip()
    return s in {"", "—", "-", "None", "none", "n/a", "N/A"}


def _first_pose(row: dict[str, Any]) -> dict[str, Any]:
    for step in row.get("movements") or []:
        if step.get("type") == "pose":
            p = step.get("parameters") or {}
            if isinstance(p.get("pose"), dict):
                return p["pose"]
            return p
    return {}


def _parse_pose_spec(spec: str) -> dict[str, Any]:
    """Parse pose(dir=a|b, grip=..., z≈60, ...) into check atoms."""
    out: dict[str, Any] = {}
    if _na(spec):
        return out
    m = re.search(r"pose\((.*)\)", spec)
    body = m.group(1) if m else spec
    # strip comments
    body = body.split("#")[0]
    for part in re.split(r",\s*", body):
        part = part.strip()
        if not part:
            continue
        # Support both z=60 and z≈60 / x≈near_chest
        if "≈" in part and "=" not in part:
            k, v = part.split("≈", 1)
            k, v = k.strip(), ("≈" + v.strip())
        elif "=" in part:
            k, v = part.split("=", 1)
            k, v = k.strip(), v.strip()
        else:
            continue
        if k == "dir":
            out["dir"] = [t.strip() for t in v.split("|") if t.strip() and t.strip() != "_"]
        elif k == "grip":
            toks = []
            for t in v.split("|"):
                t = t.strip().lower()
                if not t or t == "_":
                    continue
                toks.append(t)
            out["grip"] = toks
        elif k == "head":
            out["head"] = [t.strip() for t in v.split("|") if t.strip() and t.strip() != "_"] or ["_"]
        elif k == "torso":
            out["torso"] = [t.strip() for t in v.split("|") if t.strip() and t.strip() != "_"] or ["_"]
        elif k == "elbow":
            out["elbow"] = v.strip().upper()
        elif k == "left_arm":
            out["left_arm"] = v.strip().lower()
        elif k == "bank":
            # bank=F / bank=false → skip feasible bank (procedural IK)
            out["bank"] = v.strip().lower()
        elif k in {"x", "y", "z"}:
            out.setdefault("xyz", {})[k] = v.strip()
    return out


def _xyz_target(raw: str) -> tuple[str, float | None, float | None]:
    """Return (mode, lo, hi) for a location atom."""
    v = raw.lower().replace(" ", "")
    if v in {"mid", "≈mid", "z≈mid"} or "mid" in v:
        return "band", 40.0, 70.0
    if "shoulder" in v:
        return "band", 50.0, 75.0
    if "near_chest" in v or "nearchest" in v or "chest" in v:
        return "band", 70.0, 100.0
    if "forehead" in v or "head" in v:
        return "band", 75.0, 100.0
    if "tall" in v or "overhead" in v:
        return "band", 85.0, 100.0
    # numeric ≈60 / 60
    m = re.search(r"≈?\s*(-?\d+(?:\.\d+)?)", v)
    if m:
        c = float(m.group(1))
        return "center", c - LOC_TOL, c + LOC_TOL
    return "any", None, None


def _grip_ok(got: str | None, want: list[str]) -> bool:
    if not want:
        return True
    g = (got or "").lower().replace(" ", "")
    parts = set(g.split("|")) if g else set()
    for w in want:
        w = w.lower()
        if w in {"fist", "closed"}:
            if "fist" in g or "closed" in g:
                return True
            if ("vert" in g or "vertical" in g) and ("palmin" in parts or "palmin" in g):
                return True
            continue
        if w in {"pinch"}:
            if "pinch" in g:
                return True
            continue
        if w in {"claw"}:
            if "claw" in g or "palmdown" in g:
                return True
            continue
        if w in {"vert", "vertical"}:
            if "vert" in g or "vertical" in g:
                return True
            continue
        if w in {"horiz", "horizontal"}:
            if "horiz" in g or "horizontal" in g:
                return True
            continue
        # palm tokens
        if w.startswith("palm"):
            if w in parts or w in g:
                return True
            continue
        if w in parts or w in g:
            return True
    return False


def _dir_ok(got: str | None, want: list[str]) -> bool:
    if not want:
        return True
    d = (got or "still").lower()
    return d in {w.lower() for w in want}


def score_orient(pose: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Orientation = dir / grip / head / elbow / left_arm / torso (form/facing)."""
    checks: list[tuple[str, bool]] = []
    if "dir" in spec:
        checks.append(("dir", _dir_ok(pose.get("arm_position"), spec["dir"])))
    if "grip" in spec:
        checks.append(("grip", _grip_ok(str(pose.get("gripper_orientation") or ""), spec["grip"])))
    if "head" in spec and spec["head"] != ["_"]:
        hv = str(pose.get("head") or "").lower()
        want = {h.lower() for h in spec["head"]}
        checks.append(("head", hv in want or ("away" in want and hv not in {"", "center"})))
    if "elbow" in spec:
        want_t = spec["elbow"].upper() in {"T", "TRUE", "1", "BEND", "BENDED"}
        got = bool(pose.get("elbow_bended"))
        checks.append(("elbow", got == want_t if want_t else True))
    if "left_arm" in spec:
        got = str(pose.get("left_arm") or "").lower()
        want = spec["left_arm"]
        if want == "mirror":
            checks.append(("left_arm", got in {"mirror", "same", "symmetric", "1", "true"}))
        else:
            checks.append(("left_arm", got == want or want == "_"))
    if "torso" in spec and spec["torso"] != ["_"]:
        tv = str(pose.get("torso_height") or pose.get("torso") or "").lower()
        want = {t.lower() for t in spec["torso"]}
        ok = any(
            (w in tv)
            or (w == "tall" and tv in {"tall", "high"})
            or (w == "low" and tv in {"low", "down"})
            or (w == "away" and "away" in tv)
            for w in want
        )
        checks.append(("torso", ok or tv in want))

    if not checks:
        return {"applicable": False, "ok": None, "detail": "n/a"}
    ok = all(c for _, c in checks)
    return {
        "applicable": True,
        "ok": ok,
        "detail": {k: v for k, v in checks},
        "got": {
            "dir": pose.get("arm_position"),
            "grip": pose.get("gripper_orientation"),
            "head": pose.get("head"),
            "elbow": pose.get("elbow_bended"),
            "left_arm": pose.get("left_arm"),
            "torso": pose.get("torso_height"),
        },
    }


def score_loc(pose: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    xyz_spec = spec.get("xyz") or {}
    if not xyz_spec:
        return {"applicable": False, "ok": None, "detail": "n/a"}
    detail = {}
    oks = []
    for ax, raw in xyz_spec.items():
        mode, lo, hi = _xyz_target(raw)
        if mode == "any" or lo is None:
            detail[ax] = {"required": raw, "got": pose.get(ax), "ok": ax in pose}
            oks.append(ax in pose)
            continue
        if ax not in pose or not isinstance(pose.get(ax), (int, float)):
            detail[ax] = {"required": raw, "got": None, "ok": False, "band": [lo, hi]}
            oks.append(False)
            continue
        val = float(pose[ax])
        ok = lo <= val <= hi
        detail[ax] = {"required": raw, "got": val, "ok": ok, "band": [lo, hi]}
        oks.append(ok)
    return {"applicable": True, "ok": all(oks) if oks else None, "detail": detail}


def _move_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    ev = []
    for step in row.get("movements") or []:
        t = step.get("type")
        p = step.get("parameters") or {}
        if t == "movement":
            mov = p.get("movement") if isinstance(p.get("movement"), dict) else p
            for j in mov.get("joints") or []:
                deg = j.get("degrees")
                sign = None
                if isinstance(deg, list) and len(deg) >= 2:
                    a, b = float(deg[0]), float(deg[1])
                    if a * b < 0 or (a < 0 < b) or (b < 0 < a):
                        sign = "pm"
                    elif b - a > 0 or b > 0:
                        sign = "+"
                    elif b - a < 0 or b < 0:
                        sign = "-"
                elif isinstance(deg, (int, float)):
                    sign = "+" if float(deg) >= 0 else "-"
                ev.append(
                    {
                        "kind": "move",
                        "joint": str(j.get("joint") or "").lower(),
                        "axis": str(j.get("axis") or "").lower(),
                        "sign": sign,
                        "rep": int(mov.get("repetition") or 1),
                    }
                )
        elif t == "path":
            path = p.get("path") if isinstance(p.get("path"), dict) else p
            dist = path.get("distance")
            sign = None
            if isinstance(dist, list) and len(dist) >= 2:
                sign = "pm"
            elif isinstance(dist, (int, float)):
                sign = "+" if float(dist) >= 0 else "-"
            plane = str(path.get("plane") or "").lower()
            axis = str(path.get("axis") or "").lower()
            ev.append(
                {
                    "kind": "path",
                    "shape": str(path.get("shape") or "").lower(),
                    "axis": axis,
                    "plane": plane,
                    "sign": sign,
                    "rep": int(path.get("repetition") or 1),
                }
            )
    return ev


def score_move(row: dict[str, Any], move_spec: str, path_spec: str) -> dict[str, Any]:
    if _na(move_spec) and _na(path_spec):
        return {"applicable": False, "ok": None, "detail": "n/a"}
    events = _move_events(row)
    if not events and (not _na(move_spec) or not _na(path_spec)):
        # hesitation/interrupt soft: any motion counts if move(_,_,_)
        if "move(_, _, _)" in (move_spec or "") or "pause" in (move_spec or "").lower():
            has_any = bool(row.get("movements"))
            return {"applicable": True, "ok": has_any, "detail": "any_motion"}
        return {"applicable": True, "ok": False, "detail": "no_move_or_path"}

    oks: list[bool] = []
    detail: dict[str, Any] = {}

    if not _na(path_spec) and "path(" in path_spec:
        want_arc = "arc" in path_spec
        want_line = "line" in path_spec
        planes = re.findall(r"plane=([a-z|]+)", path_spec)
        axes = re.findall(r"axis=([a-z|]+)", path_spec)
        want_planes = set()
        for p in planes:
            want_planes |= {x.strip() for x in p.split("|")}
        want_axes = set()
        for a in axes:
            want_axes |= {x.strip() for x in a.split("|")}
        path_ev = [e for e in events if e["kind"] == "path"]
        ok = False
        for e in path_ev:
            if want_arc and e["shape"] != "arc":
                continue
            if want_line and e["shape"] != "line" and not want_arc:
                continue
            if want_planes and e["plane"] not in want_planes and not any(
                ax in (e["plane"] or "") for ax in want_planes
            ):
                # plane xz matches axis hints loosely
                if not (want_axes and e["axis"] in want_axes):
                    continue
            if want_axes and e["axis"] and e["axis"] not in want_axes and not want_planes:
                continue
            ok = True
            break
        # scale_expand special
        if "move_apart" in path_spec and not ok:
            ok = any(e["kind"] == "move" and e["axis"] in {"y", "x"} for e in events)
        detail["path"] = ok
        oks.append(ok)

    if not _na(move_spec) and "move(" in move_spec:
        # move(joint|_, axis|a|b, sign, ...)
        m = re.search(r"move\(([^)]*)\)", move_spec)
        body = m.group(1) if m else ""
        parts = [p.strip() for p in body.split(",")]
        joint = parts[0].lower() if parts else "_"
        axis_raw = parts[1].lower() if len(parts) > 1 else "_"
        sign_raw = parts[2] if len(parts) > 2 else "_"
        axes = [a.strip() for a in axis_raw.split("|")] if axis_raw != "_" else ["_"]
        # normalize signs
        signs = set()
        sr = sign_raw.replace(" ", "")
        if "±" in sr or "+-" in sr or "pm" in sr or "opposed" in sr:
            signs.add("pm")
        if "+" in sr or "outward" in sr or "away" in sr:
            signs.add("+")
        if "−" in sr or "-" in sr or "inward" in sr:
            signs.add("-")
        want_rep = "rep" in move_spec

        ok = False
        for e in events:
            if e["kind"] == "path" and joint == "_":
                # path can satisfy axis motion for generic _
                if axes != ["_"] and e["axis"] not in axes and e["plane"] not in axes:
                    if not any(a in (e["plane"] or "") for a in axes):
                        continue
                if signs:
                    if "pm" in signs and e["sign"] == "pm":
                        ok = True
                        break
                    if e["sign"] in signs or ("pm" in signs and e["sign"] in {"+", "-"}):
                        ok = True
                        break
                else:
                    ok = True
                    break
            if e["kind"] != "move":
                continue
            if joint not in {"_", ""} and e["joint"] != joint and not (
                joint == "shoulder" and e["joint"] in {"shoulder", "arm"}
            ):
                # allow wrist/shoulder flex
                if joint == "wrist" and e["joint"] not in {"wrist", "hand"}:
                    continue
                if joint not in {e["joint"], "_"}:
                    continue
            if axes != ["_"] and e["axis"] not in axes and not any(
                a in e["axis"] for a in axes if a != "_"
            ):
                # tilt/roll alias
                if not ("tilt" in axes and e["axis"] in {"tilt", "pitch"}):
                    if not ("roll" in axes and e["axis"] == "roll"):
                        if not ("pan" in axes and e["axis"] in {"pan", "yaw"}):
                            continue
            if signs:
                if "pm" in signs:
                    if e["sign"] == "pm" or (want_rep and e["rep"] >= 2):
                        ok = True
                        break
                    # single signed oscillation still ok if rep>=2
                    if e["rep"] >= 2 and e["sign"] in {"+", "-", "pm"}:
                        ok = True
                        break
                if e["sign"] in signs:
                    ok = True
                    break
            else:
                ok = True
                break
        # away: head pan nonzero
        if "away" in move_spec and not ok:
            ok = any(e["kind"] == "move" and e["joint"] == "head" and e["axis"] in {"pan", "yaw"} for e in events)
        detail["move"] = ok
        oks.append(ok)

    if not oks:
        return {"applicable": False, "ok": None, "detail": "n/a"}
    return {"applicable": True, "ok": all(oks), "detail": detail, "events": events}


def score_program(cue: str, row: dict[str, Any]) -> dict[str, Any]:
    req = REQUIRED.get(cue)
    if not req:
        return {"cue": cue, "known": False}
    pose = _first_pose(row)
    pose_spec = _parse_pose_spec(req.get("pose", "—"))
    orient = score_orient(pose, pose_spec)
    loc = score_loc(pose, pose_spec)
    move = score_move(row, req.get("movement", "—"), req.get("path", "—"))
    return {
        "cue": cue,
        "known": True,
        "spec": req,
        "orientation": orient,
        "location": loc,
        "movement": move,
        "code": (row.get("source_code") or "")[:200],
    }


def _plat_rows(master: list[dict], plat: str) -> dict[str, dict]:
    out = {}
    for e in master:
        cfg = (e.get("platforms") or {}).get(plat) or {}
        if cfg.get("movements") or cfg.get("source_code"):
            out[str(e["cue"])] = cfg
    return out


def _rate(rows: list[dict], key: str) -> dict[str, Any]:
    app = [r for r in rows if r.get(key, {}).get("applicable")]
    ok = sum(1 for r in app if r[key].get("ok"))
    return {"n": len(app), "ok": ok, "rate": (100.0 * ok / len(app)) if app else None}


def _score_master(master: list[dict], source: str) -> dict[str, Any]:
    by_plat: dict[str, Any] = {}
    flat: list[dict] = []
    for plat in ("humanoid", "tiago"):
        rows_map = _plat_rows(master, plat)
        scored = []
        for cue, row in sorted(rows_map.items(), key=lambda kv: kv[0]):
            s = score_program(cue, row)
            s["source"] = source
            s["platform"] = plat
            s["idx"] = int(row.get("idx") or 0)
            scored.append(s)
            flat.append(s)
        by_plat[plat] = {
            "n": len(scored),
            "orientation": _rate(scored, "orientation"),
            "location": _rate(scored, "location"),
            "movement": _rate(scored, "movement"),
            "rows": scored,
        }
    return {"by_plat": by_plat, "flat": flat}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pred",
        type=Path,
        action="append",
        default=None,
        help="Prediction master JSON (repeatable). Default: Gemini paper30 + pilot75.",
    )
    ap.add_argument(
        "--tag",
        action="append",
        default=None,
        help="Label for each --pred (same order). Default: paper30/pilot75 or stem.",
    )
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()

    sources: list[tuple[str, list]] = []
    if args.pred:
        tags = list(args.tag or [])
        while len(tags) < len(args.pred):
            tags.append(args.pred[len(tags)].parent.name)
        for tag, path in zip(tags, args.pred):
            sources.append((tag, json.loads(path.read_text())))
    else:
        paper = json.loads(PAPER_LLM.read_text()) if PAPER_LLM.is_file() else []
        pilot = json.loads(PILOT_LLM.read_text()) if PILOT_LLM.is_file() else []
        sources = [("paper30", paper), ("pilot75", pilot)]

    results: dict[str, Any] = {"by_source": {}}
    all_flat: list[dict] = []
    for source, master in sources:
        packed = _score_master(master, source)
        results["by_source"][source] = packed["by_plat"]
        all_flat.extend(packed["flat"])

    results["rows"] = all_flat
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    lines = ["# Cue-spec discrete scores", ""]
    for source, plats in results["by_source"].items():
        lines.append(f"## {source}")
        for plat, block in plats.items():
            o, l, m = block["orientation"], block["location"], block["movement"]

            def fmt(block: dict) -> str:
                if not block["n"]:
                    return "n/a"
                r = block["rate"]
                return f"{block['ok']}/{block['n']}" + (f" ({r:.0f}%)" if r is not None else "")

            lines.append(
                f"- **{plat}** n={block['n']} · Orient {fmt(o)} · Loc {fmt(l)} · Move {fmt(m)}"
            )
        lines.append("")
    args.out_md.write_text("\n".join(lines))
    print(args.out_md.read_text())
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
