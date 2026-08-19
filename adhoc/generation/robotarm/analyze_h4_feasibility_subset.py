#!/usr/bin/env python3
"""Aggregate true-Short H4 metrics by feasibility and export Roomba errors."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

from openpyxl import load_workbook

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_HERE))

from score_embodiment_h4 import _align, _components  # noqa: E402

PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "quadruped", "roomba")
GREEN = "FFC6EFCE"
YELLOW = "FFFFEB9C"
RED = "FFFFC7CE"
WORKBOOK = _REPO / "data/seed/gesture_embodiment_table.xlsx"
SCORE_PATH = _REPO / "data/results/evaluation/embodiment_h4_gt_vs_short.json"
GT_PATH = (
    _REPO
    / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
)
SHORT_PATH = (
    _REPO
    / "data/results/motion_configs/embodiment_motion_code_short/embodiment_motion_code_all.json"
)
OUT_PATH = (
    _REPO / "data/results/evaluation/embodiment_h4_gt_vs_short_feasibility.json"
)


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _ci(
    rows: list[dict[str, Any]],
    fn: Callable[[list[dict[str, Any]]], float],
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    estimates = [
        fn([rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(10_000)
    ]
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def _macro_recall(rows: list[dict[str, Any]]) -> float:
    return sum(
        row["attribute"]["matched"] / row["attribute"]["gt_required"] for row in rows
    ) / len(rows)


def _macro_f1(rows: list[dict[str, Any]]) -> float:
    return sum(row["attribute"]["f1"] for row in rows) / len(rows)


def _em(rows: list[dict[str, Any]]) -> float:
    return sum(row["instance"]["exact_match"] for row in rows) / len(rows)


def _core_inclusion(rows: list[dict[str, Any]]) -> float:
    return sum(row["instance"]["core_inclusion"] for row in rows) / len(rows)


def _aggregate(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    return {
        "n": len(rows),
        "attribute_macro_required_recall": _macro_recall(rows),
        "attribute_macro_required_recall_ci95": _ci(rows, _macro_recall, seed),
        "attribute_macro_f1": _macro_f1(rows),
        "attribute_macro_f1_ci95": _ci(rows, _macro_f1, seed + 100),
        "instance_exact_match_accuracy": _em(rows),
        "instance_exact_match_accuracy_ci95": _ci(rows, _em, seed + 200),
        "instance_core_inclusion_accuracy": _core_inclusion(rows),
        "mean_gt_attribute_count": sum(
            row["attribute"]["gt_required"] for row in rows
        )
        / len(rows),
        "mean_missing": sum(len(row["attribute"]["missing"]) for row in rows)
        / len(rows),
        "mean_extra": sum(len(row["attribute"]["extra"]) for row in rows)
        / len(rows),
    }


def _cell_rgb(cell) -> str:
    """Normalize openpyxl fill rgb to 8-char AARRGGBB (alpha → FF)."""
    fg = cell.fill.fgColor if cell.fill else None
    if fg is None or not fg.rgb:
        return ""
    rgb = str(fg.rgb).upper()
    if len(rgb) == 6:
        rgb = "FF" + rgb
    if len(rgb) == 8 and rgb.startswith("00"):
        # solid theme fills often store alpha as 00
        rgb = "FF" + rgb[2:]
    return rgb


def _feasibility() -> dict[tuple[int, str], str]:
    workbook = load_workbook(WORKBOOK, data_only=True)
    sheet = workbook["Gesture × Embodiment"]
    result = {}
    label = {
        GREEN: "fully_possible",
        YELLOW: "approximate",
        RED: "not_feasible",
        # also accept un-normalized alpha variants
        "00C6EFCE": "fully_possible",
        "00FFEB9C": "approximate",
        "00FFC7CE": "not_feasible",
    }
    for row in range(2, 32):
        idx = int(sheet.cell(row, 1).value)
        for column, platform in enumerate(PLATFORMS, start=4):
            color = _cell_rgb(sheet.cell(row, column))
            if color not in label:
                raise KeyError(
                    f"unknown feasibility fill at row={row} col={column} "
                    f"platform={platform}: {color!r}"
                )
            result[(idx, platform)] = label[color]
    return result


def _brief(movement: dict[str, Any]) -> str:
    parameters = movement.get("parameters") or {}
    payload = (
        parameters.get("path")
        or parameters.get("pose")
        or parameters.get("movement")
        or parameters
    )
    return f"{movement.get('type')}: {json.dumps(payload, ensure_ascii=False, sort_keys=True)}"


def main() -> None:
    score = json.loads(SCORE_PATH.read_text(encoding="utf-8"))
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    short = json.loads(SHORT_PATH.read_text(encoding="utf-8"))
    feasibility = _feasibility()
    for row in score["instances"]:
        row["feasibility"] = feasibility[(int(row["idx"]), row["platform"])]

    result: dict[str, Any] = {
        "definition": {
            "short_source": "gesture_embodiment_table.xlsx / Motion Code (Top 5)",
            "gt_source": "gesture_embodiment_table.xlsx / Motion Code (Human)",
            "subset": ["fully_possible", "approximate"],
            "ci": "10,000 cue-level percentile bootstrap samples",
        },
        "full": {},
        "fully_possible_plus_approximate": {},
    }
    for platform_index, platform in enumerate((*PLATFORMS, "all")):
        full_rows = (
            score["instances"]
            if platform == "all"
            else [r for r in score["instances"] if r["platform"] == platform]
        )
        subset_rows = [
            r
            for r in full_rows
            if r["feasibility"] in {"fully_possible", "approximate"}
        ]
        result["full"][platform] = _aggregate(
            full_rows, 20260723 + platform_index
        )
        result["fully_possible_plus_approximate"][platform] = _aggregate(
            subset_rows, 20260823 + platform_index
        )

    gt_by_idx = {int(entry["idx"]): entry for entry in gt}
    short_by_idx = {int(entry["idx"]): entry for entry in short}
    roomba_low = []
    for row in sorted(
        (
            r
            for r in score["instances"]
            if r["platform"] == "roomba"
            and r["feasibility"] in {"fully_possible", "approximate"}
            and r["attribute"]["f1"] < 1.0
        ),
        key=lambda r: r["attribute"]["f1"],
    ):
        idx = int(row["idx"])
        gt_row = gt_by_idx[idx]["platforms"]["roomba"]
        short_row = short_by_idx[idx]["platforms"]["roomba"]
        _, _, extra_indices = _align(_components(gt_row), _components(short_row))
        roomba_low.append(
            {
                "idx": idx,
                "cue": row["cue"],
                "feasibility": row["feasibility"],
                "required_recall": row["attribute"]["required_recall"],
                "precision": row["attribute"]["precision"],
                "f1": row["attribute"]["f1"],
                "instance_em": row["instance"]["exact_match"],
                "gt_code": gt_row.get("source_code", ""),
                "short_code": short_row.get("source_code", ""),
                "extra_components": [
                    {
                        "index": index + 1,
                        "summary": _brief(short_row["movements"][index]),
                    }
                    for index in extra_indices
                ],
                "missing": row["attribute"]["missing"],
                "extra_attributes": row["attribute"]["extra"],
            }
        )
    result["roomba_low_f1_subset"] = roomba_low
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(OUT_PATH)
    for name in ("full", "fully_possible_plus_approximate"):
        print("\n", name)
        for platform, values in result[name].items():
            print(
                platform,
                values["n"],
                f"R={values['attribute_macro_required_recall']:.3f}",
                f"F1={values['attribute_macro_f1']:.3f}",
                f"EM={values['instance_exact_match_accuracy']:.3f}",
            )
    print("roomba low-F1 subset", len(roomba_low))


if __name__ == "__main__":
    main()
