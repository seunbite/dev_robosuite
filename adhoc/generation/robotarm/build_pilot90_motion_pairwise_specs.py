#!/usr/bin/env python3
"""Refresh step-10 pairwise spec JSON from built MP4 sidecars."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from motion_pairwise_media import MOTION_PAIRWISE_DIR, write_pairwise_specs  # noqa: E402

OUT_DIR = MOTION_PAIRWISE_DIR


def main() -> None:
    od = OUT_DIR
    entries: list[dict] = []
    for sidecar in sorted(od.glob("*_pair_spec.json")):
        entries.append(json.loads(sidecar.read_text(encoding="utf-8")))
    if not entries:
        print(f"No sidecars in {od} — run prepare_pilot90_motion_pairwise_mp4.py first", flush=True)
        return
    path = write_pairwise_specs(entries, od)
    print(f"Wrote {len(entries)} specs -> {path}", flush=True)


if __name__ == "__main__":
    main()
