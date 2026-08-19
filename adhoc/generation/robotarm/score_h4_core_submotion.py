#!/usr/bin/env python3
"""Aggregate H4 blank predictions by attribute × motion_role (core/submotion)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred",
        action="append",
        required=True,
        help="prediction jsonl (repeatable; orig+CB)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO
        / "data/results/evaluation/plots/h4_core_submotion_attr_acc_32b.json",
    )
    args = ap.parse_args()

    # (role, attr) -> [correct, n]
    cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    by_role: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for path_str in args.pred:
        path = Path(path_str)
        for row in _load_jsonl(path):
            attr = str(row.get("attribute") or "")
            if attr == "path.x_sign":
                continue
            role = str(row.get("motion_role") or "core")
            if role == "hold":
                continue
            ok = bool(row.get("correct"))
            cells[(role, attr)][0] += int(ok)
            cells[(role, attr)][1] += 1
            by_role[role][0] += int(ok)
            by_role[role][1] += 1

    attrs = sorted({a for _, a in cells}, key=lambda a: -sum(cells[(r, a)][1] for r in ("core", "submotion")))
    table = []
    for attr in attrs:
        row = {"attribute": attr}
        for role in ("core", "submotion"):
            c, n = cells[(role, attr)]
            row[f"{role}_acc"] = None if n == 0 else round(100 * c / n, 1)
            row[f"{role}_n"] = n
        table.append(row)

    payload = {
        "by_role_overall": {
            role: {
                "accuracy": round(100 * by_role[role][0] / by_role[role][1], 1),
                "n": by_role[role][1],
            }
            for role in ("core", "submotion")
            if by_role[role][1]
        },
        "by_attribute": table,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("| Attribute | Core % (N) | Submotion % (N) |")
    print("|---|---:|---:|")
    for row in table:
        c = "—" if row["core_n"] == 0 else f"{row['core_acc']} ({row['core_n']})"
        s = "—" if row["submotion_n"] == 0 else f"{row['submotion_acc']} ({row['submotion_n']})"
        print(f"| {row['attribute']} | {c} | {s} |")
    print(args.out)


if __name__ == "__main__":
    main()
