#!/usr/bin/env python3
"""Generate-only (no MuJoCo render) Gemini Motion Codes for cues75 × humanoid,tiago.

Writes the same master JSON layout as run_embodiment_exp0_gemini.py so a later
render pass can consume it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_ADHOC = _REPO / "adhoc/generation/robotarm"
for p in (_REPO, _ADHOC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from motion_code_to_config import write_json_files  # noqa: E402
from run_embodiment_exp0_gemini import (  # noqa: E402
    generate_one,
    write_fill_md,
)

DEFAULT_CUES = _HERE / "data/cues75_not_in_paper30.json"
DEFAULT_OUT = (
    _REPO
    / "data/results/motion_configs"
    / "embodiment_motion_code_gemini_gemini-2.5-pro_pilot75_humanoidlike"
)
PLATFORMS = ("humanoid", "tiago")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cues", type=Path, default=DEFAULT_CUES)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-cues", type=int, default=0)
    ap.add_argument(
        "--platforms",
        default=",".join(PLATFORMS),
        help="Comma list (default: humanoid,tiago)",
    )
    args = ap.parse_args()

    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY (source APIKEY.sh)")

    cues = json.loads(args.cues.read_text(encoding="utf-8"))
    if args.max_cues > 0:
        cues = cues[: args.max_cues]
    platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]

    out_json = args.out_json
    out_json.mkdir(parents=True, exist_ok=True)
    fill_md = (
        _REPO
        / "data/seed/agent_fill/embodiment_short"
        / f"codes_fill_gemini_pilot75_{datetime.now().strftime('%Y%m%d')}.md"
    )
    dummy_fewshot = (
        _REPO / "data/seed/agent_fill/embodiment_short/fewshot_synthetic.md"
    )

    by_cue: dict[str, dict[str, Any]] = {}
    master = out_json / "embodiment_motion_code_all.json"
    if args.resume and master.is_file():
        for entry in json.loads(master.read_text(encoding="utf-8")):
            by_cue[entry["cue"]] = entry

    n_ok = n_fail = 0
    print(f"out-json: {out_json}", flush=True)
    try:
        for plat in platforms:
            print(f"=== {plat} ===", flush=True)
            for cue_row in cues:
                cue = cue_row["cue"]
                entry = by_cue.setdefault(
                    cue,
                    {
                        "idx": int(cue_row["idx"]),
                        "cue": cue,
                        "category": cue_row.get("category", ""),
                        "platforms": {},
                    },
                )
                prev = entry["platforms"].get(plat)
                if (
                    args.resume
                    and prev
                    and str(prev.get("source_code") or "").strip()
                    and prev.get("parse_ok")
                ):
                    print(f"  skip {cue}", flush=True)
                    continue
                print(f"  gen {cue} ...", flush=True)
                try:
                    row = generate_one(
                        platform=plat,
                        cue_row=cue_row,
                        cues=cues,
                        fewshot_path=dummy_fewshot,
                        model=args.model,
                        backend="gemini",
                        sleep_s=args.sleep,
                        prompt_style="yaml_cue",
                    )
                    n_ok += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"  FAIL {cue}: {exc}", flush=True)
                    n_fail += 1
                    msg = str(exc)
                    if "RESOURCE_EXHAUSTED" in msg or "spending cap" in msg.lower():
                        print("[abort] Gemini quota exhausted", flush=True)
                        raise SystemExit(2) from exc
                    row = {
                        "idx": int(cue_row["idx"]),
                        "cue": cue,
                        "description": str(cue_row.get("description") or ""),
                        "movements": [],
                        "state": "motion_code_gemini",
                        "model": args.model,
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "experiment": "embodiment_motion_code_gemini",
                        "generation_valid": False,
                        "platform": plat,
                        "parse_ok": False,
                        "notes": [f"error: {exc}"],
                        "source_code": "",
                        "raw_response": "",
                    }
                raw_dir = out_json / "_raw" / plat
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / f"{cue}.txt").write_text(
                    row.pop("raw_response", "") or "", encoding="utf-8"
                )
                entry["platforms"][plat] = row
                configs = [by_cue[c["cue"]] for c in cues if c["cue"] in by_cue]
                write_json_files(configs, out_json)
                write_fill_md(configs, cues, platforms, fill_md)
                print(
                    f"  saved {plat}/{cue} parse_ok={bool(row.get('parse_ok'))} "
                    f"code={str(row.get('source_code') or '')[:80]!r}",
                    flush=True,
                )
    finally:
        configs = [by_cue[c["cue"]] for c in cues if c["cue"] in by_cue]
        if configs:
            write_json_files(configs, out_json)
            write_fill_md(configs, cues, platforms, fill_md)

    print(f"done ok={n_ok} fail={n_fail} master={master}", flush=True)


if __name__ == "__main__":
    main()
