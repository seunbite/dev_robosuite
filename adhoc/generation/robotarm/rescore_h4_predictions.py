#!/usr/bin/env python3
"""Rescore stored H4 predictions with the current response parser."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_h4_mcq_vllm import _prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        row["prediction"] = _prediction(row["raw_output"], row)
        row["correct"] = row["prediction"] == row["correct_option"]
    Path(args.output).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"rescored={len(rows)}")


if __name__ == "__main__":
    main()
