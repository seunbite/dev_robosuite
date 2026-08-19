#!/usr/bin/env python3
"""Analyze multi-variant H4 prompt-iteration predictions."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

FILTERS = {
    "all": None,
    "human_identifiable": {
        "direction",
        "left_arm",
        "path.axis",
        "path.degrees_sign",
        "path.distance_sign",
        "path.x_sign",
        "repetition",
    },
    "structural_semantic": {"direction", "path.axis", "repetition"},
    "high_semantic": {"direction", "repetition"},
}


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_responses": len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "invalid": sum(row.get("prediction") not in ("A", "B") for row in rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows) if rows else 0.0,
    }


def _original_key(row: dict[str, Any]) -> str:
    marker = f".reversed.{row['experiment_id']}"
    return row["question_id"].replace(marker, f".{row['experiment_id']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.predictions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_experiment[row["experiment_id"]].append(row)

    results = {}
    for experiment, group in sorted(by_experiment.items()):
        original = {
            row["question_id"]: row for row in group if ".reversed." not in row["question_id"]
        }
        reversed_rows = {
            _original_key(row): row for row in group if ".reversed." in row["question_id"]
        }
        if set(original) != set(reversed_rows):
            raise RuntimeError(f"Pair mismatch for {experiment}")
        pair_counts = defaultdict(int)
        for qid, first in original.items():
            second = reversed_rows[qid]
            outcome = (
                "both_correct"
                if first["correct"] and second["correct"]
                else "original_only"
                if first["correct"]
                else "reversed_only"
                if second["correct"]
                else "neither_correct"
            )
            pair_counts[outcome] += 1
        filter_results = {}
        for name, attributes in FILTERS.items():
            selected = (
                group
                if attributes is None
                else [row for row in group if row["attribute"] in attributes]
            )
            unique = {
                _original_key(row) if ".reversed." in row["question_id"] else row["question_id"]
                for row in selected
            }
            filter_results[name] = {
                "n_pairs": len(unique),
                "coverage": len(unique) / len(original),
                **_stats(selected),
            }
        by_attribute: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_attribute[row["attribute"]].append(row)
        results[experiment] = {
            "model": group[0]["model"],
            "n_pairs": len(original),
            "filters": filter_results,
            "pair_outcomes": dict(pair_counts),
            "by_attribute": {
                key: _stats(value) for key, value in sorted(by_attribute.items())
            },
        }
    output = Path(args.output)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
