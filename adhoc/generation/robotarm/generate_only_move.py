#!/usr/bin/env python3
"""Exp 7: generate movement tail with GT-fixed first pose (result_exp7_{tag}.json).

  python adhoc/generation/robotarm/generate_only_move.py --backend gemini --resume
"""
from __future__ import annotations

import run_pilot90_exp_generation as _gen


def main() -> None:
    import sys

    sys.argv = [sys.argv[0], "--exp", "7", *sys.argv[1:]]
    _gen.main()


if __name__ == "__main__":
    main()
