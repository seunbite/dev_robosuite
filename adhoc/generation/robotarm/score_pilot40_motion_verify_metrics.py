#!/usr/bin/env python3
"""
Compare generation vs text verify vs VLM (MP4) verify against component GT.

Metrics per channel:
- generation_accuracy / *_verifying_accuracy / *_detection_accuracy
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from motion_gt_tail_builder import build_tail_from_component  # noqa: E402
from score_pilot40_motion_gt_components import (  # noqa: E402
    _build_annotation_map,
    _tail_matches_component,
    _tail_steps,
)

BASE_CFG = (
    _REPO
    / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_gt_fixed_pose_pilot40.json"
)
TEXT_VERIFY = _REPO / "data/results/verify/pilot40_motion_component_verify_text_gemini.json"
VLM_VERIFY = _REPO / "data/results/verify/pilot40_motion_component_verify_gemini.json"
OUT_JSON = _REPO / "data/results/verify/pilot40_motion_verify_metrics.json"
OUT_TSV = _REPO / "data/results/verify/pilot40_motion_verify_metrics.tsv"


def _rate(n_ok: int, n: int) -> dict[str, Any]:
    return {
        "n_correct": n_ok,
        "n": n,
        "rate": n_ok / n if n else None,
        "pct": f"{n_ok}/{n} ({100.0 * n_ok / n:.1f}%)" if n else "n/a",
    }


def _load_verify(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {
        int(r["cue_idx"]): r
        for r in json.loads(path.read_text(encoding="utf-8")).get("rows", [])
    }


def _tail_after_verify(gen_tail: list, verify_row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not verify_row:
        return gen_tail
    if verify_row.get("movement_is_appropriate") is True:
        return gen_tail
    rec = verify_row.get("recommended_component")
    if rec:
        built = build_tail_from_component(rec)
        if built:
            return built
    return gen_tail


def _channel_metrics(
    gen_tail: list,
    comp: dict[str, Any],
    gen_match: bool,
    vrow: dict[str, Any] | None,
) -> dict[str, Any]:
    ver_tail = _tail_after_verify(gen_tail, vrow)
    ver_match, _ = _tail_matches_component(ver_tail, comp)
    appropriate = vrow.get("movement_is_appropriate") if vrow else None
    det = None
    if appropriate is not None:
        det = bool(appropriate) == bool(gen_match)
    return {
        "verifying_tail_match": ver_match,
        "movement_is_appropriate": appropriate,
        "detection_agrees_with_generation_gt": det,
        "has_recommendation": bool(vrow and vrow.get("recommended_component")),
    }


def main() -> None:
    cfg_rows = json.loads(BASE_CFG.read_text(encoding="utf-8"))
    by_idx = {int(r["idx"]): r for r in cfg_rows}
    ann_by_idx = {int(a["cue_idx"]): a for a in _build_annotation_map()}
    text_by = _load_verify(TEXT_VERIFY)
    vlm_by = _load_verify(VLM_VERIFY)

    rows_out: list[dict[str, Any]] = []
    gen_ok = 0
    text_ver_ok = text_det_ok = 0
    vlm_ver_ok = vlm_det_ok = 0
    n = 0

    for idx, ann in sorted(ann_by_idx.items()):
        comp = ann.get("component")
        if comp is None:
            continue
        cfg = by_idx.get(idx)
        if not cfg:
            continue
        gen_tail = _tail_steps(cfg.get("movements") or [])
        gen_match, _ = _tail_matches_component(gen_tail, comp)
        n += 1
        if gen_match:
            gen_ok += 1

        text_m = _channel_metrics(gen_tail, comp, gen_match, text_by.get(idx))
        vlm_m = _channel_metrics(gen_tail, comp, gen_match, vlm_by.get(idx))

        if text_m["verifying_tail_match"]:
            text_ver_ok += 1
        if text_m["detection_agrees_with_generation_gt"]:
            text_det_ok += 1
        if vlm_m["verifying_tail_match"]:
            vlm_ver_ok += 1
        if vlm_m["detection_agrees_with_generation_gt"]:
            vlm_det_ok += 1

        rows_out.append(
            {
                "cue_idx": idx,
                "cue": cfg["cue"],
                "annotation_raw": ann.get("annotation_raw"),
                "generation_tail_match": gen_match,
                "text": text_m,
                "vlm_alpha": vlm_m,
            }
        )

    gen_acc = _rate(gen_ok, n)
    text_ver = _rate(text_ver_ok, n)
    text_det = _rate(text_det_ok, n)
    vlm_ver = _rate(vlm_ver_ok, n)
    vlm_det = _rate(vlm_det_ok, n)

    def delta_pp(ver_rate: float | None) -> float | None:
        if gen_acc["rate"] is None or ver_rate is None:
            return None
        return (ver_rate - gen_acc["rate"]) * 100.0

    better_ver = "text" if (text_ver.get("rate") or 0) > (vlm_ver.get("rate") or 0) else "vlm_alpha"
    if text_ver.get("rate") == vlm_ver.get("rate"):
        better_ver = "tie"
    better_det = "text" if (text_det.get("rate") or 0) > (vlm_det.get("rate") or 0) else "vlm_alpha"
    if text_det.get("rate") == vlm_det.get("rate"):
        better_det = "tie"

    payload = {
        "groundtruth": "pilot40_motion_component_gt.json",
        "sources": {
            "generation": str(BASE_CFG),
            "text_verify": str(TEXT_VERIFY) if text_by else None,
            "vlm_verify": str(VLM_VERIFY) if vlm_by else None,
        },
        "metrics": {
            "generation_accuracy": {
                **_rate(gen_ok, n),
                "description": "LLM tail matches component GT.",
            },
            "text_verifying_accuracy": {
                **text_ver,
                "delta_vs_generation_pp": delta_pp(text_ver.get("rate")),
                "description": "Tail after text verify recommendations vs component GT.",
            },
            "text_detection_accuracy": {
                **text_det,
                "description": "text movement_is_appropriate agrees with generation match.",
            },
            "vlm_verifying_accuracy": {
                **vlm_ver,
                "delta_vs_generation_pp": delta_pp(vlm_ver.get("rate")),
                "description": "Tail after VLM (alpha) verify vs component GT.",
            },
            "vlm_detection_accuracy": {
                **vlm_det,
                "description": "VLM movement_is_appropriate agrees with generation match.",
            },
            "winner": {
                "better_verifying_accuracy": better_ver,
                "better_detection_accuracy": better_det,
            },
        },
        "rows": rows_out,
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "cue_idx\tcue\tgen\ttext_ver\ttext_det\tvlm_ver\tvlm_det",
    ]
    for r in rows_out:
        lines.append(
            f"{r['cue_idx']}\t{r['cue']}\t{r['generation_tail_match']}\t"
            f"{r['text']['verifying_tail_match']}\t{r['text']['detection_agrees_with_generation_gt']}\t"
            f"{r['vlm_alpha']['verifying_tail_match']}\t{r['vlm_alpha']['detection_agrees_with_generation_gt']}"
        )
    OUT_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")

    m = payload["metrics"]
    print("=== generation_accuracy ===")
    print(f"  {m['generation_accuracy']['pct']}")
    print("=== text ===")
    print(f"  verifying: {m['text_verifying_accuracy']['pct']}")
    print(f"  detection: {m['text_detection_accuracy']['pct']}")
    print("=== vlm (alpha_frame_trajectory) ===")
    print(f"  verifying: {m['vlm_verifying_accuracy']['pct']}")
    print(f"  detection: {m['vlm_detection_accuracy']['pct']}")
    print(f"=== winner ===")
    print(f"  verifying: {m['winner']['better_verifying_accuracy']}")
    print(f"  detection: {m['winner']['better_detection_accuracy']}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
