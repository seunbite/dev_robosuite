#!/usr/bin/env python3
"""
Score VLM verify (mp4 / alpha_frame_trajectory) against generation ground truth.

Generation GT (per cue): LLM tail matches your component annotation
(``component_structural_match`` from pilot40_motion_component_gt.json rules).

VLM prediction: ``represents_target_cue`` from pilot40_motion_vlm_verify_gt_fixed_gemini.json.

This is NOT human-rated gesture quality — only structural component match as proxy
until you add manual motion_ok labels.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
VLM_JSON = _REPO / "data/results/verify/pilot40_motion_vlm_verify_gt_fixed_gemini.json"
OUT_JSON = _REPO / "data/results/verify/pilot40_motion_vlm_scored_vs_component_gt.json"
OUT_TSV = _REPO / "data/results/verify/pilot40_motion_vlm_scored_vs_component_gt.tsv"


def _confusion(rows: list[tuple[bool, bool]]) -> dict[str, Any]:
    tp = tn = fp = fn = 0
    for gold, pred in rows:
        if gold and pred:
            tp += 1
        elif not gold and not pred:
            tn += 1
        elif not gold and pred:
            fp += 1
        else:
            fn += 1
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = None
    if prec is not None and rec is not None and (prec + rec) > 0:
        f1 = 2 * prec * rec / (prec + rec)
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def _score_modality(rows_in: list[dict[str, Any]], vlm_key: str) -> dict[str, Any]:
    pairs: list[tuple[bool, bool]] = []
    detail: list[dict[str, Any]] = []
    for r in rows_in:
        gold = r.get("generation_gt_component_match")
        if gold is None:
            continue
        vlm = (r.get(vlm_key) or {}).get("represents_target_cue")
        if vlm is None:
            continue
        gold_b = bool(gold)
        pred_b = bool(vlm)
        pairs.append((gold_b, pred_b))
        detail.append(
            {
                "cue_idx": r["cue_idx"],
                "cue": r["cue"],
                "generation_gt": gold_b,
                "vlm_pred": pred_b,
                "agreement": gold_b == pred_b,
                "error_type": (
                    "TP"
                    if gold_b and pred_b
                    else "TN"
                    if not gold_b and not pred_b
                    else "FP"
                    if not gold_b and pred_b
                    else "FN"
                ),
            }
        )
    conf = _confusion(pairs)
    conf["interpretation"] = {
        "generation_gt_true": "LLM tail satisfies your component annotation (axis/joint/rep/hold/path).",
        "vlm_pred_true": "VLM says motion looks like the target cue name + description.",
        "TP": "Good generation; VLM agrees.",
        "TN": "Bad generation; VLM rejects.",
        "FP": "Bad generation; VLM still says yes (permissive / reads gesture anyway).",
        "FN": "Good generation; VLM says no (miss).",
    }
    conf["rows"] = detail
    return conf


def main() -> None:
    data = json.loads(VLM_JSON.read_text(encoding="utf-8"))
    rows_out: list[dict[str, Any]] = []
    for r in data["rows"]:
        rows_out.append(
            {
                "cue_idx": r["cue_idx"],
                "cue": r["cue"],
                "annotation_raw": r.get("annotation_raw"),
                "generation_gt_component_match": r.get("component_structural_match"),
                "vlm_mp4": r.get("vlm_mp4"),
                "vlm_alpha_frame_trajectory": r.get("vlm_alpha_frame_trajectory"),
            }
        )

    scored = {
        "groundtruth_note": (
            "generation_gt_component_match = rule-based match of LLM tail to "
            "pilot40_motion_component_gt.json (same as structural_ok in VLM verify JSON). "
            "Not human perceptual 'this gesture is correct'. guiding_arm excluded (no component)."
        ),
        "vlm_source": str(VLM_JSON),
        "modalities": {
            "mp4": _score_modality(rows_out, "vlm_mp4"),
            "alpha_frame_trajectory": _score_modality(
                rows_out, "vlm_alpha_frame_trajectory"
            ),
        },
        "rows": rows_out,
    }

    OUT_JSON.write_text(json.dumps(scored, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "cue_idx\tcue\tgeneration_gt\tvlm_mp4\tvlm_alpha\tmp4_err\talpha_err",
    ]
    mp4_by = {d["cue"]: d for d in scored["modalities"]["mp4"]["rows"]}
    al_by = {d["cue"]: d for d in scored["modalities"]["alpha_frame_trajectory"]["rows"]}
    for r in rows_out:
        cue = r["cue"]
        g = r.get("generation_gt_component_match")
        m = (mp4_by.get(cue) or {}).get("vlm_pred")
        a = (al_by.get(cue) or {}).get("vlm_pred")
        lines.append(
            f"{r['cue_idx']}\t{cue}\t{g}\t{m}\t{a}\t"
            f"{(mp4_by.get(cue) or {}).get('error_type','')}\t"
            f"{(al_by.get(cue) or {}).get('error_type','')}"
        )
    OUT_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for mod, conf in scored["modalities"].items():
        print(f"\n=== {mod} vs generation component GT ===")
        print(
            f"n={conf['n']} TP={conf['tp']} TN={conf['tn']} FP={conf['fp']} FN={conf['fn']}"
        )
        print(
            f"accuracy={conf['accuracy']:.1%}  precision={conf['precision']:.1%}  "
            f"recall={conf['recall']:.1%}  f1={conf['f1']:.1%}"
            if conf.get("precision") is not None
            else "no pairs"
        )
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_TSV}")


if __name__ == "__main__":
    main()
