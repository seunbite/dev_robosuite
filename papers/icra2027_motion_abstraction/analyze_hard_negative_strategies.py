#!/usr/bin/env python3
"""Compare three morphology-aware 4AFC hard-negative policies.

Policies:
  hybrid   = nearest GT Motion Code foil + two meaning-neighbor foils
  motion3  = three nearest GT Motion Code foils for the target morphology
  meaning3 = three cue-meaning neighbors (morphology-independent)

The meaning map is deliberately explicit and reviewable. It is a draft
experimental-design artifact, not part of the paper until manually approved.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parents[1]
ROBOTARM = REPO / "adhoc/generation/robotarm"
sys.path.insert(0, str(ROBOTARM))

from score_embodiment_h4 import score_instance  # noqa: E402

GT = (
    REPO
    / "data/results/motion_configs/embodiment_motion_code_gt/"
    "embodiment_motion_code_all.json"
)
CUES = REPO / "data/seed/agent_fill/embodiment_short/cues.json"
OUT = PAPER / "analysis_hard_negative_strategies.json"
PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "roomba")

# Draft intent-level neighbors. Order matters (most plausible semantic foil first).
# This is intentionally separate from Motion Code similarity so the three
# candidate policies can be inspected rather than silently conflated.
MEANING_NEIGHBORS: dict[str, tuple[str, str, str]] = {
    "nod_yes": ("shake_head_no", "tilt_head_curious", "gaze_shift_avoid"),
    "shake_head_no": ("nod_yes", "stop_palm_out", "gaze_shift_avoid"),
    "draw_circle_repeat": ("depict_big_bowl", "air_quotes", "fist_pump"),
    "gaze_shift_avoid": ("tilt_head_curious", "shake_head_no", "cross_arms_defensive"),
    "fist_pump": ("raise_fists_victory", "flex_bicep", "highfive_invite"),
    "tilt_head_curious": ("gaze_shift_avoid", "nod_yes", "shake_head_no"),
    "cover_yawn_sleepy": ("rub_eye_tired", "facepalm", "stretch_arms_up"),
    "rub_eye_tired": ("cover_yawn_sleepy", "facepalm", "gaze_shift_avoid"),
    "point_you": ("point_self", "beckon_come_here", "stop_palm_out"),
    "point_self": ("point_you", "shrug_idk", "beckon_come_here"),
    "beckon_come_here": ("highfive_invite", "point_you", "arms_open_welcome"),
    "stop_palm_out": ("highfive_invite", "hands_up_surrender", "point_you"),
    "salute": ("highfive_invite", "stop_palm_out", "raise_index_one_moment"),
    "raise_index_one_moment": ("stop_palm_out", "point_you", "air_quotes"),
    "highfive_invite": ("stop_palm_out", "arms_open_welcome", "salute"),
    "shrug_idk": ("hands_up_surrender", "arms_open_welcome", "palms_together_thank_you"),
    "flex_bicep": ("fist_pump", "raise_fists_victory", "stretch_arms_up"),
    "air_quotes": ("raise_index_one_moment", "draw_circle_repeat", "point_you"),
    "arms_open_welcome": ("highfive_invite", "palms_together_thank_you", "shrug_idk"),
    "hands_up_surrender": ("stop_palm_out", "shrug_idk", "raise_fists_victory"),
    "stretch_arms_up": ("cover_yawn_sleepy", "raise_fists_victory", "depict_tree"),
    "raise_fists_victory": ("fist_pump", "flex_bicep", "highfive_invite"),
    "cross_arms_defensive": ("stop_palm_out", "hands_up_surrender", "gaze_shift_avoid"),
    "palms_together_thank_you": ("salute", "arms_open_welcome", "point_self"),
    "facepalm": ("rub_eye_tired", "cover_yawn_sleepy", "shrug_idk"),
    "depict_tree": ("depict_flying_bird", "stretch_arms_up", "depict_big_bowl"),
    "depict_big_bowl": ("depict_small_object", "depict_tree", "draw_circle_repeat"),
    "depict_lion": ("depict_flying_bird", "flex_bicep", "raise_fists_victory"),
    "depict_small_object": ("depict_big_bowl", "raise_index_one_moment", "air_quotes"),
    "depict_flying_bird": ("depict_lion", "depict_tree", "arms_open_welcome"),
}


def _stable_rank(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _motion_neighbors(
    target: dict[str, Any],
    platform: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_cell = target["platforms"][platform]
    ranked: list[dict[str, Any]] = []
    for foil in rows:
        if foil["cue"] == target["cue"]:
            continue
        scored = score_instance(target_cell, foil["platforms"][platform])
        attr = scored["attribute"]
        inst = scored["instance"]
        n_edit = len(attr["missing"]) + len(attr["extra"])
        ranked.append(
            {
                "cue": foil["cue"],
                "idx": int(foil["idx"]),
                "f1": round(float(attr["f1"]), 4),
                "type_seq_match": bool(inst["component_type_sequence_exact"]),
                "exact_motion_code": bool(inst["exact_match"]),
                "n_edit": int(n_edit),
                "same_dataset_category": foil.get("category") == target.get("category"),
                "_stable": _stable_rank(
                    f"{platform}.{int(target['idx']):02d}->{int(foil['idx']):02d}"
                ),
            }
        )
    ranked.sort(
        key=lambda item: (
            item["f1"],
            item["exact_motion_code"],
            item["type_seq_match"],
            -item["n_edit"],
            item["same_dataset_category"],
            -item["_stable"],
        ),
        reverse=True,
    )
    for item in ranked:
        item.pop("_stable")
    return ranked


def _hybrid(motion: list[dict[str, Any]], semantic: tuple[str, ...]) -> list[str]:
    selected = [motion[0]["cue"]]
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
    return selected


def main() -> None:
    rows = sorted(json.loads(GT.read_text()), key=lambda row: int(row["idx"]))
    cues = {row["cue"]: row for row in json.loads(CUES.read_text())}
    missing = sorted(set(cues) - set(MEANING_NEIGHBORS))
    if missing:
        raise RuntimeError(f"missing meaning neighbors: {missing}")

    results: list[dict[str, Any]] = []
    for platform in PLATFORMS:
        for target in rows:
            cue = target["cue"]
            motion = _motion_neighbors(target, platform, rows)
            semantic = MEANING_NEIGHBORS[cue]
            score_by_cue = {item["cue"]: item for item in motion}
            strategies = {
                "hybrid": _hybrid(motion, semantic),
                "motion3": [item["cue"] for item in motion[:3]],
                "meaning3": list(semantic),
            }
            results.append(
                {
                    "platform": platform,
                    "target_idx": int(target["idx"]),
                    "target": cue,
                    "description": cues[cue]["description"],
                    "strategies": strategies,
                    "motion_top5": motion[:5],
                    "strategy_details": {
                        name: [
                            {
                                "cue": foil,
                                "motion_f1": score_by_cue[foil]["f1"],
                                "type_seq_match": score_by_cue[foil]["type_seq_match"],
                                "exact_motion_code": score_by_cue[foil][
                                    "exact_motion_code"
                                ],
                            }
                            for foil in foils
                        ]
                        for name, foils in strategies.items()
                    },
                    "hybrid_motion_semantic_overlap": len(
                        set(strategies["motion3"]) & set(strategies["meaning3"])
                    ),
                }
            )

    payload = {
        "definition": {
            "target_pool": "30 cues",
            "platforms": list(PLATFORMS),
            "motion_similarity": (
                "GT Motion Code attribute-F1 within each morphology; ties favor "
                "exact script, component-type sequence match, and fewer edits"
            ),
            "meaning_similarity": (
                "draft intent-level manual neighbor map, fixed across morphologies; "
                "requires author approval before preregistration"
            ),
            "strategies": {
                "hybrid": "nearest morphology-specific Motion Code foil + two meaning foils",
                "motion3": "three nearest morphology-specific Motion Code foils",
                "meaning3": "three intent-level meaning foils",
            },
        },
        "meaning_neighbors": {
            cue: list(neighbors) for cue, neighbors in MEANING_NEIGHBORS.items()
        },
        "rows": results,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(OUT)
    print(f"rows={len(results)}")


if __name__ == "__main__":
    main()
