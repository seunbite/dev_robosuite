#!/usr/bin/env python3
"""Misspecification vs underspecification: human GT vs Gemini 2.5 Pro SMPs.

Misspecification: an aligned same-type component exists in both programs, but
at least one attribute value differs (or a GT slot is absent on that component).
Underspecification: a GT component has no same-type aligned counterpart in the
LLM program.

Paper platforms only (no quadruped): 30 cues × 5 bodies = 150 cells.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "adhoc/generation/robotarm"))

from score_embodiment_h4 import _align, _components  # noqa: E402

PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "roomba")
GT = REPO / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
PRED = (
    REPO
    / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro"
    / "embodiment_motion_code_all.json"
)
ROLES = REPO / "data/results/evaluation/gt_component_core_submotion.json"
OUT = REPO / "data/results/evaluation/misspec_underspec_gt_vs_gemini.json"
OUT_MD = Path(__file__).resolve().parent / "analysis_misspec_underspec.md"

MEANING_ATTRS = {
    "facing": ("gripper_orientation", "orientation"),
    "location": ("x", "y", "z", "head", "torso_height", "arm_position", "height"),
    "axis": ("axis",),
    "joint": ("joint", "joints"),
    "laterality": ("left_arm", "arm"),
    "extent": ("degrees", "distance", "dist"),
    "repetition": ("repetition", "rep"),
    "hold": ("hold", "hold_time"),
    "speed": ("speed", "spd"),
    "path_shape": ("shape", "mode"),
}


def _family(key: str) -> str:
    leaf = key.rsplit(".", 1)[-1].lower()
    leaf = leaf.split("[")[0]
    kl = key.lower()
    if "gripper" in kl or "orientation" in kl:
        return "facing"
    for fam, leaves in MEANING_ATTRS.items():
        if leaf in leaves or any(tok == leaf for tok in leaves):
            return fam
    return leaf or "other"


def _role_map(raw: dict[str, Any]) -> dict[tuple[int, str, int], str]:
    out = {}
    for row in raw.get("components") or []:
        out[(int(row["cue_idx"]), str(row["platform"]), int(row["component_index"]))] = str(
            row["motion_role"]
        )
    return out


def _cell(gt_row: dict[str, Any], pred_row: dict[str, Any], roles: dict, cue_idx: int, platform: str):
    gt, pred = _components(gt_row), _components(pred_row)
    pairs, missing_gt, extra_pred = _align(gt, pred)
    misspec_slots: list[dict[str, Any]] = []
    matched_clean = 0
    for gi, pj in pairs:
        ga, pa = gt[gi]["attrs"], pred[pj]["attrs"]
        diffs = []
        keys = set(ga) | set(pa)
        for key in sorted(keys):
            gv, pv = ga.get(key, "__ABSENT__"), pa.get(key, "__ABSENT__")
            if gv == pv:
                continue
            diffs.append(
                {
                    "key": key,
                    "family": _family(key),
                    "gt": gv,
                    "llm": pv,
                    "kind": "wrong_value" if gv != "__ABSENT__" and pv != "__ABSENT__" else (
                        "gt_slot_missing_in_llm" if pv == "__ABSENT__" else "llm_extra_slot"
                    ),
                }
            )
        if diffs:
            misspec_slots.append(
                {
                    "gt_i": gi,
                    "llm_j": pj,
                    "type": gt[gi]["type"],
                    "role": roles.get((cue_idx, platform, gi), "unknown"),
                    "diffs": diffs,
                }
            )
        else:
            matched_clean += 1
    underspec = [
        {
            "gt_i": i,
            "type": gt[i]["type"],
            "role": roles.get((cue_idx, platform, i), "unknown"),
            "n_attrs": len(gt[i]["attrs"]),
        }
        for i in missing_gt
    ]
    extras = [{"llm_j": j, "type": pred[j]["type"]} for j in extra_pred]
    return {
        "n_gt": len(gt),
        "n_llm": len(pred),
        "n_aligned": len(pairs),
        "n_aligned_clean": matched_clean,
        "n_misspec_components": len(misspec_slots),
        "n_underspec_components": len(underspec),
        "n_extra_components": len(extras),
        "has_misspec": bool(misspec_slots),
        "has_underspec": bool(underspec),
        "type_sequence_exact": [x["type"] for x in gt] == [x["type"] for x in pred],
        "misspec_components": misspec_slots,
        "underspec_components": underspec,
        "extra_components": extras,
        "gt_types": [x["type"] for x in gt],
        "llm_types": [x["type"] for x in pred],
        "gt_code": (gt_row.get("source_code") or "").strip(),
        "llm_code": (pred_row.get("source_code") or "").strip(),
    }


def _pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def main() -> None:
    gt_master = json.loads(GT.read_text(encoding="utf-8"))
    pred_master = json.loads(PRED.read_text(encoding="utf-8"))
    roles = _role_map(json.loads(ROLES.read_text(encoding="utf-8")))
    pred_by = {(int(e["idx"]), plat): row for e in pred_master for plat, row in (e.get("platforms") or {}).items()}

    cells = []
    for entry in gt_master:
        idx, cue = int(entry["idx"]), str(entry["cue"])
        for platform in PLATFORMS:
            row = (entry.get("platforms") or {}).get(platform)
            pred = pred_by.get((idx, platform))
            if not row or not pred:
                raise SystemExit(f"missing cell {platform} {cue}")
            scored = _cell(row, pred, roles, idx, platform)
            cells.append({"idx": idx, "cue": cue, "platform": platform, **scored})

    n = len(cells)
    fam_wrong = Counter()
    fam_missing_slot = Counter()
    under_type = Counter()
    under_role = Counter()
    extra_type = Counter()
    misspec_type = Counter()

    for c in cells:
        for u in c["underspec_components"]:
            under_type[u["type"]] += 1
            under_role[u["role"]] += 1
        for e in c["extra_components"]:
            extra_type[e["type"]] += 1
        for m in c["misspec_components"]:
            misspec_type[m["type"]] += 1
            for d in m["diffs"]:
                if d["kind"] == "wrong_value":
                    fam_wrong[d["family"]] += 1
                elif d["kind"] == "gt_slot_missing_in_llm":
                    fam_missing_slot[d["family"]] += 1

    def plat_block(plat: str | None) -> dict[str, Any]:
        sub = cells if plat is None else [c for c in cells if c["platform"] == plat]
        nn = len(sub)
        both = sum(c["has_misspec"] and c["has_underspec"] for c in sub)
        miss_only = sum(c["has_misspec"] and not c["has_underspec"] for c in sub)
        under_only = sum(c["has_underspec"] and not c["has_misspec"] for c in sub)
        neither = sum(not c["has_misspec"] and not c["has_underspec"] for c in sub)
        return {
            "n": nn,
            "mean_n_gt": sum(c["n_gt"] for c in sub) / nn,
            "mean_n_llm": sum(c["n_llm"] for c in sub) / nn,
            "pct_cells_misspec": _pct(sum(c["has_misspec"] for c in sub), nn),
            "pct_cells_underspec": _pct(sum(c["has_underspec"] for c in sub), nn),
            "pct_misspec_only": _pct(miss_only, nn),
            "pct_underspec_only": _pct(under_only, nn),
            "pct_both": _pct(both, nn),
            "pct_neither": _pct(neither, nn),
            "mean_misspec_components": sum(c["n_misspec_components"] for c in sub) / nn,
            "mean_underspec_components": sum(c["n_underspec_components"] for c in sub) / nn,
            "mean_extra_components": sum(c["n_extra_components"] for c in sub) / nn,
            "pct_type_sequence_exact": _pct(sum(c["type_sequence_exact"] for c in sub), nn),
        }

    examples = []
    for cue in (
        "air_quotes",
        "highfive_invite",
        "stop_palm_out",
        "rub_eye_tired",
        "shake_head_no",
        "raise_fists_victory",
        "gaze_shift_avoid",
        "wave_hi",
        "hands_up_surrender",
        "cover_yawn_sleepy",
    ):
        for plat in PLATFORMS:
            hits = [c for c in cells if c["cue"] == cue and c["platform"] == plat]
            if hits:
                examples.append(hits[0])

    out = {
        "definition": {
            "misspecification": "aligned same-type component with differing attribute values or a GT slot missing on that component",
            "underspecification": "GT component with no aligned LLM counterpart",
            "platforms": list(PLATFORMS),
            "n_cells": n,
            "gt": str(GT),
            "pred": str(PRED),
        },
        "overall": plat_block(None),
        "by_platform": {p: plat_block(p) for p in PLATFORMS},
        "misspec_wrong_value_by_family": dict(fam_wrong.most_common()),
        "misspec_gt_slot_missing_in_llm_by_family": dict(fam_missing_slot.most_common()),
        "underspec_component_type": dict(under_type.most_common()),
        "underspec_motion_role": dict(under_role.most_common()),
        "extra_llm_component_type": dict(extra_type.most_common()),
        "misspec_component_type": dict(misspec_type.most_common()),
        "examples": [
            {
                "cue": c["cue"],
                "platform": c["platform"],
                "has_misspec": c["has_misspec"],
                "has_underspec": c["has_underspec"],
                "gt_types": c["gt_types"],
                "llm_types": c["llm_types"],
                "underspec": c["underspec_components"],
                "misspec": [
                    {
                        "type": m["type"],
                        "role": m["role"],
                        "diffs": [
                            {k: d[k] for k in ("family", "key", "gt", "llm", "kind")}
                            for d in m["diffs"]
                            if d["kind"] != "llm_extra_slot"
                        ],
                    }
                    for m in c["misspec_components"]
                ],
                "gt_code": c["gt_code"],
                "llm_code": c["llm_code"],
            }
            for c in examples
        ],
        "cells": [
            {k: c[k] for k in (
                "idx", "cue", "platform", "n_gt", "n_llm", "has_misspec",
                "has_underspec", "n_misspec_components", "n_underspec_components",
                "n_extra_components", "gt_types", "llm_types",
            )}
            for c in cells
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    o = out["overall"]
    lines = [
        "# Misspecification vs underspecification (GT vs Gemini 2.5 Pro SMP)",
        "",
        f"N = {n} cells (30 cues × 5 bodies).",
        "",
        "## Cell-level",
        "",
        "| Split | Misspec cells | Underspec cells | Both | Misspec only | Underspec only | Neither | GT comps | LLM comps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| all | {o['pct_cells_misspec']:.1f}% | {o['pct_cells_underspec']:.1f}% | {o['pct_both']:.1f}% | {o['pct_misspec_only']:.1f}% | {o['pct_underspec_only']:.1f}% | {o['pct_neither']:.1f}% | {o['mean_n_gt']:.2f} | {o['mean_n_llm']:.2f} |",
    ]
    for p in PLATFORMS:
        b = out["by_platform"][p]
        lines.append(
            f"| {p} | {b['pct_cells_misspec']:.1f}% | {b['pct_cells_underspec']:.1f}% | {b['pct_both']:.1f}% | {b['pct_misspec_only']:.1f}% | {b['pct_underspec_only']:.1f}% | {b['pct_neither']:.1f}% | {b['mean_n_gt']:.2f} | {b['mean_n_llm']:.2f} |"
        )
    lines += [
        "",
        "## Wrong-value attributes (misspec)",
        "",
        "| Family | Count |",
        "|---|---:|",
    ]
    for k, v in fam_wrong.most_common():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## GT slots missing on an aligned component",
        "",
        "| Family | Count |",
        "|---|---:|",
    ]
    for k, v in fam_missing_slot.most_common():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Missing GT components (underspec)",
        "",
        f"- by type: {dict(under_type.most_common())}",
        f"- by role: {dict(under_role.most_common())}",
        f"- extra LLM types: {dict(extra_type.most_common())}",
        "",
        "## Named examples",
        "",
    ]
    for c in out["examples"]:
        if c["platform"] not in ("humanoid", "roomba", "stretch") and c["cue"] not in (
            "air_quotes",
            "highfive_invite",
            "rub_eye_tired",
        ):
            continue
        lines.append(f"### `{c['cue']}` / {c['platform']}  misspec={c['has_misspec']} underspec={c['has_underspec']}")
        lines.append("")
        lines.append("GT:")
        lines.append("```")
        lines.append(c["gt_code"])
        lines.append("```")
        lines.append("LLM:")
        lines.append("```")
        lines.append(c["llm_code"])
        lines.append("```")
        if c["underspec"]:
            lines.append("underspec: " + json.dumps(c["underspec"], ensure_ascii=False))
        if c["misspec"]:
            lines.append("misspec: " + json.dumps(c["misspec"], ensure_ascii=False))
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)
    print(OUT_MD)
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
