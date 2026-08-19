#!/usr/bin/env python3
"""GT vs Gemini25Pro motion-code similarity (script + attribute metrics).

1) Script-level exact match after parse (canonical movements JSON)
2) Attribute-level F1 / Jaccard / EM via score_embodiment_h4
3) Extra: raw source_code normalize-EM, type-sequence EM, parse success
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from motion_code_to_config import convert_code  # noqa: E402
from score_embodiment_h4 import PLATFORMS, score_instance, score_masters  # noqa: E402


def _norm_code(code: str) -> str:
    text = (code or "").strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip().lower()


def _canonical_movements(code: str, platform: str) -> tuple[bool, Any, str]:
    try:
        cfg = convert_code(code, platform=platform)
        mov = cfg.get("movements") or []
        return True, json.loads(json.dumps(mov, sort_keys=True)), ""
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)


def _iter_cells(master: list[dict[str, Any]]):
    for cue_row in master:
        for platform in PLATFORMS:
            cell = (cue_row.get("platforms") or {}).get(platform)
            if not cell:
                continue
            yield {
                "cue_idx": cue_row.get("idx"),
                "cue": cue_row.get("cue"),
                "category": cue_row.get("category"),
                "platform": platform,
                "row": cell,
                "source_code": cell.get("source_code") or "",
            }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    gt_master = json.loads(args.gt.read_text(encoding="utf-8"))
    pred_master = json.loads(args.pred.read_text(encoding="utf-8"))
    h4 = score_masters(gt_master, pred_master)

    gt_cells = {(c["platform"], c["cue"]): c for c in _iter_cells(gt_master)}
    pred_cells = {(c["platform"], c["cue"]): c for c in _iter_cells(pred_master)}
    keys = sorted(set(gt_cells) & set(pred_cells))

    per: list[dict[str, Any]] = []
    raw_em = parsed_em = type_seq = 0
    parse_ok_both = 0
    by_plat: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"raw_em": [], "parsed_em": [], "type_seq": [], "n": []}
    )

    for platform, cue in keys:
        g, p = gt_cells[(platform, cue)], pred_cells[(platform, cue)]
        g_code, p_code = g["source_code"], p["source_code"]
        raw_match = _norm_code(g_code) == _norm_code(p_code) and bool(_norm_code(g_code))
        g_ok, g_mov, g_err = _canonical_movements(g_code, platform)
        p_ok, p_mov, p_err = _canonical_movements(p_code, platform)
        both_ok = g_ok and p_ok
        parsed_match = bool(both_ok and g_mov == p_mov)
        inst = score_instance(g["row"], p["row"])
        type_match = bool(inst["instance"]["component_type_sequence_exact"])
        if raw_match:
            raw_em += 1
        if parsed_match:
            parsed_em += 1
        if type_match:
            type_seq += 1
        if both_ok:
            parse_ok_both += 1
        by_plat[platform]["raw_em"].append(int(raw_match))
        by_plat[platform]["parsed_em"].append(int(parsed_match))
        by_plat[platform]["type_seq"].append(int(type_match))
        by_plat[platform]["n"].append(1)
        per.append(
            {
                "platform": platform,
                "cue": cue,
                "cue_idx": g["cue_idx"],
                "category": g["category"],
                "raw_source_exact_match": raw_match,
                "parsed_script_exact_match": parsed_match,
                "type_sequence_exact": type_match,
                "parse_ok_gt": g_ok,
                "parse_ok_pred": p_ok,
                "parse_error_gt": g_err,
                "parse_error_pred": p_err,
                "attribute_f1": inst["attribute"]["f1"],
                "attribute_jaccard": (
                    inst["attribute"]["matched"]
                    / (
                        inst["attribute"]["gt_required"]
                        + inst["attribute"]["predicted"]
                        - inst["attribute"]["matched"]
                    )
                    if (
                        inst["attribute"]["gt_required"]
                        + inst["attribute"]["predicted"]
                        - inst["attribute"]["matched"]
                    )
                    else 1.0
                ),
                "instance_exact_match": inst["instance"]["exact_match"],
                "core_inclusion": inst["instance"]["core_inclusion"],
            }
        )

    n = len(keys) or 1
    plat_summary = {}
    for platform in (*PLATFORMS,):
        rows = by_plat[platform]
        nn = len(rows["n"]) or 1
        plat_summary[platform] = {
            "n": len(rows["n"]),
            "raw_source_exact_match": sum(rows["raw_em"]) / nn,
            "parsed_script_exact_match": sum(rows["parsed_em"]) / nn,
            "type_sequence_exact": sum(rows["type_seq"]) / nn,
            "attribute_macro_f1": h4["aggregate"][platform]["attribute_macro_f1"],
            "attribute_micro_f1": h4["aggregate"][platform]["attribute_micro_f1"],
            "attribute_macro_jaccard": h4["aggregate"][platform]["attribute_macro_jaccard"],
            "instance_exact_match": h4["aggregate"][platform]["instance_exact_match_accuracy"],
            "core_inclusion": h4["aggregate"][platform]["instance_core_inclusion_accuracy"],
        }

    out = {
        "gt": str(args.gt),
        "pred": str(args.pred),
        "n_paired_cells": len(keys),
        "overall": {
            "raw_source_exact_match": raw_em / n,
            "parsed_script_exact_match": parsed_em / n,
            "type_sequence_exact": type_seq / n,
            "parse_ok_both": parse_ok_both / n,
            "attribute_micro_f1": h4["aggregate"]["all"]["attribute_micro_f1"],
            "attribute_macro_f1": h4["aggregate"]["all"]["attribute_macro_f1"],
            "attribute_macro_jaccard": h4["aggregate"]["all"]["attribute_macro_jaccard"],
            "attribute_micro_precision": h4["aggregate"]["all"]["attribute_micro_precision"],
            "attribute_micro_required_recall": h4["aggregate"]["all"][
                "attribute_micro_required_recall"
            ],
            "instance_exact_match": h4["aggregate"]["all"]["instance_exact_match_accuracy"],
            "core_inclusion": h4["aggregate"]["all"]["instance_core_inclusion_accuracy"],
        },
        "by_platform": plat_summary,
        "h4_metric_definition": h4["metric_definition"],
        "instances": per,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = args.out.with_suffix(".md")
    o = out["overall"]
    lines = [
        "# GT vs Gemini 2.5 Pro — config similarity",
        "",
        f"- GT: `{args.gt}`",
        f"- Pred: `{args.pred}`",
        f"- Paired cells: **{len(keys)}** (cue × embodiment)",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Script EM (raw normalized `source_code`) | {100*o['raw_source_exact_match']:.1f}% |",
        f"| Script EM (parse → canonical movements) | {100*o['parsed_script_exact_match']:.1f}% |",
        f"| Component type-sequence EM | {100*o['type_sequence_exact']:.1f}% |",
        f"| Attribute micro-F1 | {100*o['attribute_micro_f1']:.1f}% |",
        f"| Attribute macro-F1 | {100*o['attribute_macro_f1']:.1f}% |",
        f"| Attribute macro-Jaccard | {100*o['attribute_macro_jaccard']:.1f}% |",
        f"| Attribute micro-P / R | {100*o['attribute_micro_precision']:.1f}% / {100*o['attribute_micro_required_recall']:.1f}% |",
        f"| Instance exact match (H4 atoms) | {100*o['instance_exact_match']:.1f}% |",
        f"| Core inclusion | {100*o['core_inclusion']:.1f}% |",
        f"| Both parse OK | {100*o['parse_ok_both']:.1f}% |",
        "",
        "## By embodiment",
        "",
        "| Platform | parsed-script EM | type-seq EM | attr macro-F1 | Jaccard | instance EM |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for platform in PLATFORMS:
        s = plat_summary[platform]
        lines.append(
            f"| {platform} | {100*s['parsed_script_exact_match']:.1f}% | "
            f"{100*s['type_sequence_exact']:.1f}% | {100*s['attribute_macro_f1']:.1f}% | "
            f"{100*s['attribute_macro_jaccard']:.1f}% | {100*s['instance_exact_match']:.1f}% |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
