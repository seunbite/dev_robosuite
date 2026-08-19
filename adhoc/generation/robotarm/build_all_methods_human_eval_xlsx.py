#!/usr/bin/env python3
"""Pivot the human yes/no review CSV into a 150-row × 5-method Excel sheet."""

from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

REPO = Path(__file__).resolve().parents[3]
IN_CSV = (
    REPO
    / "data/results/evaluation/embodiment_cue_id"
    / "all_methods_human_yesno_c1_30.csv"
)
OUT_XLSX = IN_CSV.with_suffix(".xlsx")

PLATFORMS = ("humanoid", "tiago", "stretch", "single_arm", "roomba")
METHODS = ("GT", "Joint-Space", "Cartesian-Space", "Motion Program", "Trajectory Sketching")


def main() -> None:
    with IN_CSV.open(encoding="utf-8", newline="") as handle:
        source = list(csv.DictReader(handle))

    by_cell: dict[tuple[int, str], dict[str, str]] = {}
    for row in source:
        key = (int(row["idx"]), row["platform"])
        by_cell.setdefault(key, {})[row["method"]] = row["original_human_yesno"]

    expected = 30 * len(PLATFORMS)
    if len(by_cell) != expected:
        raise ValueError(f"Expected {expected} cue×embodiment cells, found {len(by_cell)}")

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Human yes-no"

    headers = ("cue_id", "embodiment_index", *METHODS)
    sheet.append(headers)
    for cue_idx in range(1, 31):
        for embodiment_idx, platform in enumerate(PLATFORMS, start=1):
            labels = by_cell[(cue_idx, platform)]
            # Numeric index is paired with its readable embodiment name for review.
            sheet.append(
                (
                    f"c{cue_idx:02d}",
                    f"{embodiment_idx} · {platform}",
                    *(labels.get(method, "N/A") for method in METHODS),
                )
            )

    header_fill = PatternFill("solid", fgColor="1F4E78")
    yes_fill = PatternFill("solid", fgColor="E2F0D9")
    no_fill = PatternFill("solid", fgColor="FCE4D6")
    header_font = Font(color="FFFFFF", bold=True)
    for cell in sheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")

    for row in sheet.iter_rows(min_row=2, min_col=3, max_col=7):
        for cell in row:
            cell.alignment = Alignment(horizontal="center")
            if cell.value == "YES":
                cell.fill = yes_fill
            elif cell.value == "NO":
                cell.fill = no_fill

    sheet.freeze_panes = "C2"
    sheet.auto_filter.ref = f"A1:G{sheet.max_row}"
    widths = (12, 24, 14, 16, 18, 18, 23)
    for column_idx, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(column_idx)].width = width

    workbook.save(OUT_XLSX)
    print(f"{OUT_XLSX} rows={sheet.max_row - 1} method_cols={len(METHODS)}")


if __name__ == "__main__":
    main()
