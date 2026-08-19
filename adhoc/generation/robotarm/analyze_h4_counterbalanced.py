#!/usr/bin/env python3
"""Pair original and A/B-reversed H4 MCQ predictions."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": (
            sum(bool(row["correct"]) for row in rows) / len(rows) if rows else 0.0
        ),
    }


def _breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: _stats(group) for name, group in sorted(groups.items())}


def _selected_value(row: dict[str, Any]) -> Any:
    for option in row["options"]:
        if option["id"] == row["prediction"]:
            return option["value"]
    return None


def analyze(original_path: Path, reversed_path: Path) -> dict[str, Any]:
    original = _jsonl(original_path)
    reversed_rows = _jsonl(reversed_path)
    original_by_id = {row["question_id"]: row for row in original}
    reversed_by_id = {row["paired_question_id"]: row for row in reversed_rows}
    if set(original_by_id) != set(reversed_by_id):
        raise RuntimeError("Original and reversed question IDs do not match")

    pair_counts = defaultdict(int)
    same_semantic = 0
    same_position = 0
    pairs = []
    for question_id, first in original_by_id.items():
        second = reversed_by_id[question_id]
        key = (
            "both_correct"
            if first["correct"] and second["correct"]
            else "original_only"
            if first["correct"]
            else "reversed_only"
            if second["correct"]
            else "neither_correct"
        )
        pair_counts[key] += 1
        semantic_consistent = _selected_value(first) == _selected_value(second)
        position_consistent = first["prediction"] == second["prediction"]
        same_semantic += semantic_consistent
        same_position += position_consistent
        pairs.append(
            {
                "question_id": question_id,
                "platform": first["platform"],
                "attribute": first["attribute"],
                "outcome": key,
                "semantic_consistent": semantic_consistent,
                "position_consistent": position_consistent,
            }
        )

    combined = original + reversed_rows
    return {
        "model": original[0]["model"],
        "original": _stats(original),
        "reversed": _stats(reversed_rows),
        "combined": _stats(combined),
        "combined_by_platform": _breakdown(combined, "platform"),
        "combined_by_attribute": _breakdown(combined, "attribute"),
        "pairs": {
            "n": len(pairs),
            **dict(pair_counts),
            "both_correct_rate": pair_counts["both_correct"] / len(pairs),
            "semantic_consistency": same_semantic / len(pairs),
            "same_position_rate": same_position / len(pairs),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--reversed", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(Path(args.original), Path(args.reversed))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
