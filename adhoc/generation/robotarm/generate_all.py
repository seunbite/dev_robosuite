#!/usr/bin/env python3
"""Exp 1: generate full pose configs (LLM → result_exp1_{tag}.json).

  python adhoc/generation/robotarm/generate_all.py --backend gemini
  python adhoc/generation/robotarm/generate_all.py --backend transformers --model Qwen/Qwen2.5-VL-32B-Instruct
"""
from __future__ import annotations

import run_pilot90_exp_generation as _gen


def main() -> None:
    import sys

    sys.argv = [sys.argv[0], "--exp", "1", *sys.argv[1:]]
    _gen.main()


if __name__ == "__main__":
    main()
