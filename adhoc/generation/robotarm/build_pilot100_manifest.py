#!/usr/bin/env python3
"""Build / refresh pilot100_manifest.json and pilot100_manifest.yml."""
from __future__ import annotations

import json
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
OUT_JSON = _REPO / "data/seed/yml/pilot100_manifest.json"
OUT_YML = _REPO / "data/seed/yml/pilot100_manifest.yml"
OUT_TSV = _REPO / "data/seed/yml/pilot100_manifest.tsv"

NEW51 = [
    "hesitant_accept_half_reach",
    "cancel_previous_offer_retract",
    "conditional_accept_pause_then_reach",
    "final_decision_down_press",
    "resume_speaking_pull",
    "emphasize_point_downbeat",
    "prepare_action_raise_hold",
    "scale_compress",
    "uncertainty_wave",
    "draw_circle_repeat",
    "clarification_request_palm_tilt",
    "speed_up_circular_cue",
    "polite_decline_small_push",
    "unsure_response_hand_hover_stop",
    "raise_index_one_moment",
    "curl_fingers_give_me",
    "start_action_forward_snap",
    "retry_action_reset_circle",
    "hesitation_pause_hold",
    "interrupt_hard",
    "interrupt_soft",
    "self_hug",
    "rub_arms_cold",
    "cross_arms_defensive",
    "palms_together_thank_you",
    "arms_open_welcome",
    "stretch_arms_up",
    "dominance_high_open_hold",
    "contain_space",
    "expand_space",
    "compress_space",
    "balance_scale",
    "push_pull_dual",
    "abandon_action_drop_turn",
    "weight_shift_forward",
    "weight_shift_backward",
    "torso_rotate_away",
    "torso_rotate_toward",
    "lean_side_doubt",
    "expand_chest",
    "escorting_gesture",
    "wait_for_me_backward_palm",
    "prevent_fall_support_reach",
    "frustration_down_throw",
    "submission_lowered_open_palm",
    "apology_forward_lowered_hold",
    "chin_flick_dismiss",
    "cover_ears_too_loud",
    "disagreement_cross_then_release",
    "eye_widen_surprise",
    "gaze_shift_avoid",
]
ESSENCE10 = [
    "tempo_cue",
    "thinking",
    "skepticism",
    "agreement",
    "shame",
    "confusion",
    "curiosity",
    "anticipation",
    "hesitation",
    "urgency",
]
NEW_CAT: dict[str, str] = {}
for c in NEW51[:9]:
    NEW_CAT[c] = "A"
for c in NEW51[9:21]:
    NEW_CAT[c] = "H"
for c in NEW51[21:33]:
    NEW_CAT[c] = "B"
for c in NEW51[33:40]:
    NEW_CAT[c] = "W"
for c in NEW51[40:46]:
    NEW_CAT[c] = "A & W"
for c in NEW51[46:49]:
    NEW_CAT[c] = "H & F"
for c in NEW51[49:51]:
    NEW_CAT[c] = "F"
for c in ESSENCE10:
    NEW_CAT[c] = "M" if c != "tempo_cue" else "H"

TOKEN_TAGS = {
    "point": ["deictic", "pointing", "reference"],
    "self": ["self_reference", "body_pointing", "first_person"],
    "you": ["other_reference", "deictic", "addressee"],
    "head": ["head_region", "face_adjacent"],
    "face": ["face_contact", "head_region"],
    "temple": ["head_region", "cognitive_gesture"],
    "eye": ["face_region", "gaze_related"],
    "mouth": ["face_region", "mouth_cover"],
    "ear": ["head_region", "sensory_block"],
    "chin": ["face_region", "dismissive"],
    "throat": ["neck_region", "stop_signal"],
    "palm": ["open_hand", "palm_visible"],
    "fist": ["closed_hand", "power_gesture"],
    "hand": ["hand_focus"],
    "arm": ["arm_trajectory"],
    "wave": ["oscillation", "lateral_motion"],
    "beckon": ["approach_invite", "repetitive_pull", "come_here"],
    "stop": ["prohibition", "blocking"],
    "salute": ["formal_greeting", "forehead_contact"],
    "highfive": ["social_contact_invite", "celebration", "joint_action"],
    "fistbump": ["social_contact_invite", "casual_greeting", "joint_action"],
    "cheers": ["celebration", "forward_reach", "social_ritual"],
    "toast": ["celebration", "forward_reach"],
    "hug": ["self_contact", "comfort", "affiliation"],
    "cross": ["defensive_posture", "barrier", "closure"],
    "shrug": ["uncertainty", "shoulder_motion", "doubt"],
    "nod": ["affirmation", "head_oscillation", "yes"],
    "shake": ["negation", "head_oscillation", "no"],
    "lean": ["torso_posture", "engagement_level"],
    "weight": ["torso_posture", "stance_shift"],
    "torso": ["torso_posture", "body_orientation"],
    "surrender": ["submission", "open_hands_up", "compliance"],
    "protect": ["shielding", "spatial_barrier"],
    "guide": ["navigation", "deictic_sweep", "guidance"],
    "yield": ["turn_taking", "retract", "conversation_yield"],
    "request": ["social_request", "turn_taking"],
    "slow": ["tempo_control", "calming"],
    "fast": ["tempo_control", "urgency"],
    "hesit": ["uncertainty", "pause_mid_action"],
    "interrupt": ["conversation_control", "flow_break"],
    "circle": ["circular_path", "repetitive_trace"],
    "repeat": ["repetition", "rhythmic"],
    "hold": ["static_hold", "pause"],
    "reach": ["forward_extension", "spatial_approach"],
    "retract": ["withdrawal", "offer_cancel"],
    "expand": ["spatial_expansion", "magnitude"],
    "compress": ["spatial_compression", "magnitude"],
    "scale": ["magnitude_metaphor"],
    "fan": ["cooling_gesture", "lateral_oscillation", "self_face"],
    "pat": ["contact_repetition", "comfort", "tapping"],
    "rub": ["contact_repetition", "fatigue"],
    "scratch": ["contact_repetition", "confusion"],
    "cover": ["face_cover", "inhibition"],
    "think": ["cognitive_gesture", "temple_contact"],
    "crazy": ["stigma_gesture", "circular_trace"],
    "shh": ["silence_request", "index_finger"],
    "quote": ["discourse_marker", "finger_pair", "irony"],
    "victory": ["celebration", "arms_raised", "triumph"],
    "flex": ["strength_display", "bicep_pose"],
    "pump": ["celebration", "vertical_thrust"],
    "visor": ["eyes_shade", "search_posture"],
    "gaze": ["gaze_shift", "attention"],
    "surprise": ["surprise", "startle"],
    "apolog": ["apology", "submission_posture"],
    "frustrat": ["frustration", "downward_throw"],
    "escort": ["guidance", "torso_arm_combo"],
    "balance": ["comparison_metaphor", "dual_arm_opposition"],
    "dual": ["bimanual", "asymmetric_hands"],
    "dominance": ["dominance_display", "open_expansive"],
    "welcome": ["receptive", "open_arms", "affiliation"],
    "thank": ["gratitude", "palms_together"],
    "cold": ["discomfort", "self_rubbing"],
    "decline": ["rejection", "small_push"],
    "curl": ["finger_motion", "come_here_variant"],
    "index": ["index_finger", "pause_request"],
    "stomach": ["abdomen_target", "torso_contact"],
    "back": ["upper_back_target", "behind_body"],
    "yawn": ["sleepiness", "mouth_opening"],
    "gasp": ["surprise_reaction", "mouth_cover"],
    "sleepy": ["fatigue", "low_energy"],
    "tired": ["fatigue"],
    "confus": ["confusion"],
    "interest": ["attention", "engagement"],
    "firm": ["commitment", "decisive"],
    "imitation": ["mirror_partner", "joint_action"],
    "invite": ["invitation", "social_offer"],
    "bicep": ["strength_display"],
    "curious": ["curiosity", "head_tilt"],
    "air": ["discourse_marker"],
    "tempo": ["timing_essence", "pacing"],
    "thinking": ["cognitive_state", "abstract_essence"],
    "skepticism": ["doubt", "abstract_essence"],
    "agreement": ["affirmation", "abstract_essence"],
    "shame": ["embarrassment", "abstract_essence"],
    "confusion": ["uncertainty", "abstract_essence"],
    "curiosity": ["interest", "abstract_essence"],
    "anticipation": ["expectation", "abstract_essence"],
    "urgency": ["haste", "abstract_essence"],
}

# Cues in pilot40 but under alternate names vs cues_component.yml
COMPONENT_ALIASES: dict[str, str] = {
    "self_scratch_head_confused": "scratch_head_confused",
}

MANUAL_COMPONENT: dict[str, dict] = {
    "pat_stomach": {
        "category": "H & F",
        "fields": {"pose", "movement"},
        "meaning": "Pat stomach lightly (Hungry / Full).",
    },
    "fan_face_hot": {
        "category": "H & F",
        "fields": {"pose", "movement"},
        "meaning": "Fan face with one hand (Hot).",
    },
    "cover_yawn_sleepy": {
        "category": "H & F",
        "fields": {"pose", "movement"},
        "meaning": "Open mouth wide and cover it (Sleepy / Yawn).",
    },
}

EXTRA_BY_CUE: dict[str, list[str]] = {
    "point_self": ["points_to_own_body", "chest_target", "first_person_deictic"],
    "point_you": ["points_to_other", "second_person_deictic", "addressee_directed"],
    "fan_face_hot": ["self_face_target", "heat_relief", "wrist_fanning"],
    "pat_stomach": ["abdomen_target", "self_petting", "satiety_or_comfort"],
    "pat_self_back": ["upper_back_target", "self_congratulation"],
    "beckon_come_here": ["inward_repetition", "social_summon"],
    "stop_palm_out": ["halt_signal", "outward_palm_barrier"],
    "raise_fists_victory": ["bimanual_celebration", "triumph_pose"],
    "hands_up_surrender": ["non_threat_posture"],
    "throat_cut_stop": ["severe_stop", "metaphoric_cut", "line_path_gesture"],
    "circle_temple_crazy": ["temple_trace", "arc_path", "stigma_insanity"],
    "air_quotes": ["bimanual_fingers", "ironic_quotation"],
    "self_scratch_head_confused": ["head_contact", "confusion_display"],
    "rub_eye_tired": ["periocular_rub", "fatigue_display"],
    "cover_yawn_sleepy": ["sleepiness_display"],
    "cover_mouth_gasp": ["surprise_display"],
    "tap_temple_think": ["temple_tap", "thinking_display"],
    "lean_forward_interest": ["prosocial_lean", "attention_forward"],
    "scale_expand": ["size_metaphor_large", "arms_outward"],
    "firm_accept_forward_reach": ["decisive_extension"],
    "slow_down_request_palm_down": ["calming_down", "downward_palm_beat"],
    "protective_arm_out": ["shield_other", "lateral_barrier_arm"],
    "guiding_arm": ["direction_indication", "sweeping_arm"],
    "highfive_imitation": ["synchronization"],
    "highfive_invite": ["contact_invitation"],
    "fistbump_invite": ["contact_invitation"],
    "facepalm": ["face_slap_cover", "embarrassment", "exasperation"],
    "flex_bicep": ["arm_flexion", "strength_pose"],
    "fist_pump": ["single_arm_triumph"],
    "shrug_idk": ["epistemic_uncertainty", "shoulder_lift"],
    "nod_yes": ["vertical_head_bob"],
    "shake_head_no": ["horizontal_head_shake"],
    "tilt_head_curious": ["lateral_head_incline"],
    "visor_search": ["forehead_salute_variant", "looking_distance"],
    "yield_turn": ["conversation_yield", "hand_lower"],
    "request_turn": ["floor_request", "hand_raise_small"],
    "cheers_toast": ["clink_motion", "celebratory_extension"],
    "hesitant_accept_half_reach": ["low_confidence_approach", "partial_extension"],
    "cancel_previous_offer_retract": ["offer_withdrawal"],
    "conditional_accept_pause_then_reach": ["delayed_commitment"],
    "final_decision_down_press": ["downward_emphasis", "commitment_signal"],
    "resume_speaking_pull": ["floor_claim", "pull_to_torso"],
    "emphasize_point_downbeat": ["downbeat_accent", "rhetorical_stress"],
    "prepare_action_raise_hold": ["readiness_pose", "pre_action"],
    "scale_compress": ["size_metaphor_small"],
    "uncertainty_wave": ["loose_lateral_wave"],
    "draw_circle_repeat": ["air_circle", "repeated_trace"],
    "clarification_request_palm_tilt": ["questioning_palm", "clarify"],
    "speed_up_circular_cue": ["hurry_up", "circular_speed_signal"],
    "polite_decline_small_push": ["soft_rejection"],
    "unsure_response_hand_hover_stop": ["midair_freeze", "indecision"],
    "raise_index_one_moment": ["wait_signal", "one_moment"],
    "curl_fingers_give_me": ["beckon_fingers", "object_request"],
    "start_action_forward_snap": ["snap_initiation"],
    "retry_action_reset_circle": ["reset_loop", "retry"],
    "hesitation_pause_hold": ["stutter_motion", "pause_resume"],
    "interrupt_hard": ["hard_cut", "abrupt_stop"],
    "interrupt_soft": ["soft_pause", "gentle_interrupt"],
    "self_hug": ["self_embrace"],
    "rub_arms_cold": ["thermoregulation_pantomime"],
    "cross_arms_defensive": ["closed_posture", "resistance"],
    "palms_together_thank_you": ["prayer_hands", "gratitude_pose"],
    "arms_open_welcome": ["hospitality"],
    "stretch_arms_up": ["stretching", "vitality"],
    "dominance_high_open_hold": ["expansive_power_pose"],
    "contain_space": ["space_bounding", "inward_envelope"],
    "expand_space": ["space_opening"],
    "compress_space": ["symmetric_inward"],
    "balance_scale": ["weighing_alternatives"],
    "push_pull_dual": ["opposing_hands", "tension_display"],
    "abandon_action_drop_turn": ["disengagement", "drop_away"],
    "weight_shift_forward": ["forward_lean_weight"],
    "weight_shift_backward": ["backward_lean_weight"],
    "torso_rotate_away": ["avoidance_orientation"],
    "torso_rotate_toward": ["approach_orientation"],
    "lean_side_doubt": ["skeptical_tilt"],
    "expand_chest": ["confidence_posture", "chest_out"],
    "escorting_gesture": ["accompanying_guide"],
    "wait_for_me_backward_palm": ["wait_for_me", "backward_motion"],
    "prevent_fall_support_reach": ["support_grasp", "emergency_help"],
    "frustration_down_throw": ["aggressive_downward"],
    "submission_lowered_open_palm": ["submissive_offer"],
    "apology_forward_lowered_hold": ["remorse_posture"],
    "chin_flick_dismiss": ["dismissal_flick"],
    "cover_ears_too_loud": ["noise_block"],
    "disagreement_cross_then_release": ["disagree_then_open"],
    "eye_widen_surprise": ["eyelid_opening", "startle_face"],
    "gaze_shift_avoid": ["aversion_gaze", "disengaged_look"],
    "tempo_cue": ["abstract_pacing", "timing_modulation"],
    "thinking": ["cognition_state"],
    "skepticism": ["doubt_state"],
    "agreement": ["consent_state"],
    "shame": ["withdrawal_affect"],
    "confusion": ["disorientation_state"],
    "curiosity": ["exploratory_affect"],
    "anticipation": ["expectant_state"],
    "hesitation": ["indecision_state"],
    "urgency": ["time_pressure_state"],
}


def _parse_yml_fields(path: Path) -> dict[str, dict]:
    text = path.read_text(encoding="utf-8")
    info: dict[str, dict] = {}
    cat: str | None = None
    cue: str | None = None
    for line in text.splitlines():
        m = re.match(r"^([A-Z][A-Z &]*):$", line.strip())
        if m:
            cat = m.group(1).strip()
            continue
        m = re.match(r"^  ([a-z0-9_]+):", line)
        if m and cat:
            cue = m.group(1)
            info[cue] = {"category": cat, "fields": set()}
            continue
        m = re.match(r"^    (pose|movement|essence):", line)
        if m and cue:
            info[cue]["fields"].add(m.group(1))
    return info


def _parse_meaning(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    cat: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z][A-Z &]*):$", line.strip())
        if m:
            cat = m.group(1).strip()
            continue
        m = re.match(r"^  ([a-z0-9_]+): (.+)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _meaning_tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z]{4,}", text.lower())
    stop = {
        "with",
        "while",
        "toward",
        "another",
        "motion",
        "movement",
        "perform",
        "briefly",
        "slightly",
        "repeatedly",
        "small",
        "quick",
        "slow",
        "hand",
        "body",
        "target",
        "interaction",
        "partner",
        "robot",
        "posture",
        "gesture",
        "express",
        "indicate",
        "signal",
        "toward",
        "towards",
        "before",
        "after",
        "during",
        "without",
        "between",
        "other",
        "their",
        "toward",
    }
    return [w for w in words if w not in stop][:12]


def _hashtags(
    cue: str,
    cat: str,
    fields: set[str],
    meaning: str,
    *,
    pilot40: bool = False,
    comp: dict | None = None,
) -> list[str]:
    tags: set[str] = set()
    low = cue.lower()
    for tok, tgs in TOKEN_TAGS.items():
        if tok in low:
            tags.update(tgs)
    tags.update(EXTRA_BY_CUE.get(cue, []))
    cat_map = {
        "A": ["interaction_arm", "social_signal"],
        "B": ["bimanual", "body_space"],
        "H": ["timing_control", "conversation_meta"],
        "W": ["torso_weight", "body_position"],
        "F": ["face_gaze", "head_motion"],
        "H & F": ["face_hand_combo", "head_contact"],
        "A & W": ["torso_arm_combo", "full_body_express"],
        "M": ["abstract_essence", "mental_state"],
    }
    tags.update(cat_map.get(cat, []))
    if fields == {"essence"}:
        tags.update(["essence_only", "non_decomposable", "qualitative_state"])
    elif "pose" in fields and "movement" not in fields:
        tags.update(["pose_primary", "static_configuration"])
    elif fields == {"movement"}:
        tags.update(["movement_primary", "dynamic_only_spec"])
    elif "pose" in fields and "movement" in fields:
        tags.update(["pose_plus_movement", "two_stage_gesture"])
    mt = meaning.lower()
    for kw, tgs in [
        ("repeated", ["repetition", "rhythmic"]),
        ("hold", ["hold", "static_segment"]),
        ("forward", ["forward_axis"]),
        ("backward", ["backward_axis"]),
        ("side", ["lateral_axis"]),
        ("upward", ["upward_axis"]),
        ("downward", ["downward_axis"]),
        ("circular", ["circular_path"]),
        ("pause", ["pause", "mid_action_freeze"]),
        ("mirror", ["synchronization"]),
        ("protect", ["shielding"]),
        ("uncertain", ["uncertainty"]),
        ("celebrat", ["celebration"]),
        ("reject", ["rejection"]),
        ("invite", ["invitation"]),
        ("scan", ["search_behavior"]),
        ("rotate", ["rotation"]),
        ("weight", ["stance_shift"]),
        ("torso", ["torso_motion"]),
        ("finger", ["finger_articulation"]),
        ("palm", ["palm_gesture"]),
        ("gaze", ["gaze"]),
        ("eye", ["eye_region"]),
    ]:
        if kw in mt:
            tags.update(tgs)
    for w in _meaning_tokens(meaning):
        tags.add(f"semantic_{w}")
    if comp:
        c = comp.get("component") or {}
        if c.get("repetition") == "rep":
            tags.update(["repetition", "oscillatory"])
        if c.get("hold"):
            tags.update(["hold_segment"])
        if c.get("kind") == "path_arc":
            tags.update(["arc_path"])
        if c.get("kind") == "path_line":
            tags.update(["line_path"])
        for ax in (c.get("axes") or {}):
            tags.add(f"axis_{ax}")
        if any(v == "+-" for v in (c.get("axes") or {}).values()):
            tags.update(["bidirectional_axis"])
    if pilot40:
        tags.update(["pilot40_evaluated", "pipeline_completed"])
    return sorted(tags)


def build() -> dict:
    comp = _parse_yml_fields(_REPO / "data/seed/yml/cues_component.yml")
    meaning = _parse_meaning(_REPO / "data/seed/yml/cues_meaning.yml")
    p40_cfg = json.loads(
        (
            _REPO
            / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_gt_fixed_pose_pilot40.json"
        ).read_text(encoding="utf-8")
    )
    merged = {
        r["cue"]: r
        for r in json.loads(
            (
                _REPO
                / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_no_reasoning_merged_104.json"
            ).read_text(encoding="utf-8")
        )
    }
    motion_gt = {
        a["cue"]: a
        for a in json.loads(
            (_REPO / "data/results/verify/pilot40_motion_component_gt.json").read_text(
                encoding="utf-8"
            )
        )["annotations"]
    }

    cues: list[dict] = []

    def _row(cue: str, status: str, *, pilot40: bool = False) -> None:
        manual = MANUAL_COMPONENT.get(cue)
        alias = COMPONENT_ALIASES.get(cue)
        ci = comp.get(cue) or (comp.get(alias) if alias else {}) or {}
        if manual:
            cat = manual["category"]
            fields = set(manual["fields"])
            mean = manual.get("meaning") or meaning.get(cue, "")
        else:
            cat = ci.get("category") or NEW_CAT.get(cue) or "?"
            fields = set(ci.get("fields") or [])
            mean = meaning.get(cue, "") or (meaning.get(alias, "") if alias else "")
        if fields == {"essence"} or cue in ESSENCE10:
            fields = {"essence"}
        ann = motion_gt.get(cue) if pilot40 else None
        cues.append(
            {
                "cue": cue,
                "idx": merged.get(cue, {}).get("idx"),
                "status": status,
                "category": cat,
                "component_fields": sorted(fields),
                "meaning": mean,
                "hashtags": _hashtags(
                    cue,
                    cat,
                    fields,
                    mean,
                    pilot40=pilot40,
                    comp=ann,
                ),
            }
        )

    for r in sorted(p40_cfg, key=lambda x: int(x.get("idx", 0))):
        _row(r["cue"], "completed_pilot40", pilot40=True)
    for cue in NEW51:
        _row(cue, "pending_new51")
    for cue in ESSENCE10:
        _row(cue, "pending_essence10")

    return {
        "version": 1,
        "description": "Frozen pilot100 benchmark: 39 evaluated (pilot40) + 51 new + 10 essence-only.",
        "selection": {
            "source": "cues_component.yml + merged_104 catalog",
            "excluded_shots": [
                "wave_hi",
                "raising_hand_greeting",
                "handshake_offer",
                "handshake_imitation",
                "blow_kiss",
                "big_heart_above_head",
                "hand_over_heart",
            ],
            "excluded_duplicate": ["scratch_head_confused"],
            "counts": {
                "completed_pilot40": 39,
                "pending_new51": 51,
                "pending_essence10": 10,
                "total": 100,
            },
        },
        "cues": cues,
    }


def _write_yml(data: dict) -> None:
    lines = [
        "# pilot100 manifest — do not reorder completed_pilot40 block without reason",
        f"version: {data['version']}",
        f"total: {data['selection']['counts']['total']}",
        "",
        "completed_pilot40:",
        f"  count: {data['selection']['counts']['completed_pilot40']}",
        "  note: Pose/motion/compare experiments already run on these cues (39 configs).",
        "  cues:",
    ]
    for c in data["cues"]:
        if c["status"] != "completed_pilot40":
            continue
        tags = ", ".join(c["hashtags"])
        lines.append(f"    - cue: {c['cue']}")
        lines.append(f"      idx: {c['idx']}")
        lines.append(f"      category: {c['category']}")
        lines.append(f"      fields: [{', '.join(c['component_fields'])}]")
        lines.append(f"      hashtags: [{tags}]")
    lines += ["", "pending_new51:", f"  count: {data['selection']['counts']['pending_new51']}", "  cues:"]
    for c in data["cues"]:
        if c["status"] != "pending_new51":
            continue
        tags = ", ".join(c["hashtags"])
        lines.append(f"    - cue: {c['cue']}")
        lines.append(f"      idx: {c['idx']}")
        lines.append(f"      category: {c['category']}")
        lines.append(f"      hashtags: [{tags}]")
    lines += [
        "",
        "pending_essence10:",
        f"  count: {data['selection']['counts']['pending_essence10']}",
        "  cues:",
    ]
    for c in data["cues"]:
        if c["status"] != "pending_essence10":
            continue
        tags = ", ".join(c["hashtags"])
        lines.append(f"    - cue: {c['cue']}")
        lines.append(f"      idx: {c['idx']}")
        lines.append(f"      hashtags: [{tags}]")
    OUT_YML.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tsv(data: dict) -> None:
    lines = ["order\tstatus\tidx\tcue\tcategory\tfields\thashtags"]
    for i, c in enumerate(data["cues"], 1):
        lines.append(
            "\t".join(
                [
                    str(i),
                    c["status"],
                    str(c.get("idx") or ""),
                    c["cue"],
                    c.get("category") or "",
                    "|".join(c.get("component_fields") or []),
                    " ".join(c.get("hashtags") or []),
                ]
            )
        )
    OUT_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = build()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_yml(data)
    _write_tsv(data)
    print(f"Wrote {OUT_JSON} ({len(data['cues'])} cues)")
    print(f"Wrote {OUT_YML}")
    print(f"Wrote {OUT_TSV}")


if __name__ == "__main__":
    main()
