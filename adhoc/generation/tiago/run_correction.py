#!/usr/bin/env python3
"""Tiago wrapper — delegates to robotarm/run_correction.py with --domain tiago."""
from __future__ import annotations

import sys
from pathlib import Path

_ROBOTARM = Path(__file__).resolve().parents[1] / "robotarm"
if str(_ROBOTARM) not in sys.path:
    sys.path.insert(0, str(_ROBOTARM))

if "--domain" not in sys.argv:
    sys.argv.extend(["--domain", "tiago"])

from run_correction import main  # noqa: E402

if __name__ == "__main__":
    main()
