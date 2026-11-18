#!/usr/bin/env python3
"""Generate Oddkin α=1 (Gesture Tree) Motion Codes for cues75 via Gemini.

Uses the gcr_alpha1 block from motion_code_synthetic_continuum.yaml and the same
yaml_cue prompting style as run_embodiment_exp0_gemini.py. Stores Motion Code
strings only (no MuJoCo convert/render — Oddkin uses a separate executor).

Example::

    source APIKEY.sh
    python papers/icra2027_motion_abstraction/synthetic_morphology/\\
      generate_cues75_oddkin_alpha1_gemini.py --model gemini-2.5-pro --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_ADHOC = _REPO / "adhoc/generation/robotarm"
for p in (_REPO, _ADHOC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_embodiment_exp0_gemini import (  # noqa: E402
    _extract_json_obj,
    extract_motion_code,
    gemini_generate,
)

SCHEMA_YAML = _HERE / "yml/motion_code_synthetic_continuum.yaml"
DEFAULT_CUES = _HERE / "data/cues75_not_in_paper30.json"
DEFAULT_OUT = (
    _REPO
    / "data/results/motion_configs"
    / "oddkin_alpha1_motion_code_gemini_gemini-2.5-pro_pilot75"
)
PLATFORM = "gcr_alpha1"
FEWSHOT = """
Examples (do not copy verbatim; rematerialize onto α=1 channels):
- nod_yes: pose(head=center, torso=mid) + move(head, tilt, [-12,12], rep=3, spd=1.5)
- stop_palm_out: pose(paddle=front, palm=palmfront, claw=still, torso=mid) + hold(t=1.5)
- beckon_come_here: pose(claw=front, grip=open, elbow_c=T) + move(claw, pitch, ±25, rep=3)
- shrug_idk: pose(paddle=out, claw=out, palm=palmup) + move(torso, pitch, +15, hold=0.4)
""".strip()


def schema_block() -> str:
    root = yaml.safe_load(SCHEMA_YAML.read_text(encoding="utf-8"))
    if PLATFORM not in root:
        raise KeyError(f"{PLATFORM} missing in {SCHEMA_YAML}")
    return yaml.safe_dump(
        {PLATFORM: root[PLATFORM]},
        sort_keys=False,
        allow_unicode=True,
        width=100,
        default_flow_style=False,
    ).rstrip()


def build_prompt(*, cue: str, description: str) -> str:
    robot = "Oddkin α=1 Gesture Tree"
    schema = schema_block()
    return f"""You are an expert in {robot} gesture design using Motion Code.

Goal:
- Generate ONE complete short cue motion using Motion Code primitives that this
  embodiment supports (`pose`, `move`, `path`, `hold` as listed in the schema).
- Prefer starting with `pose(...)` when the schema allows a pose silhouette.
- Total steps: 2~8. Be expressive; rematerialize the cue onto available channels
  (head, torso, claw/paddle arms, base) — do not invent joints not in the schema.
- Join steps with `+`.

[FEW-SHOT STYLE]
{FEWSHOT}

[CUE]
name: {cue}
description: {description}

[EMBODIMENT SCHEMA — {PLATFORM}]
{schema}

[Output]
Output ONE Motion Code line only (steps joined by +). No planning comments, no JSON, no explanation.
Do not invent attributes or component forms outside the embodiment schema above.
""".strip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cues", type=Path, default=DEFAULT_CUES)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-cues", type=int, default=0)
    args = ap.parse_args()

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY (source APIKEY.sh)")

    cues = json.loads(args.cues.read_text(encoding="utf-8"))
    if args.max_cues > 0:
        cues = cues[: args.max_cues]

    out_json = args.out_json
    out_json.mkdir(parents=True, exist_ok=True)
    master_path = out_json / "oddkin_alpha1_motion_code_all.json"
    raw_dir = out_json / "_raw" / PLATFORM
    raw_dir.mkdir(parents=True, exist_ok=True)

    by_cue: dict[str, dict[str, Any]] = {}
    if args.resume and master_path.is_file():
        for entry in json.loads(master_path.read_text(encoding="utf-8")):
            by_cue[entry["cue"]] = entry

    n_ok = n_fail = 0
    for cue_row in cues:
        cue = cue_row["cue"]
        entry = by_cue.setdefault(
            cue,
            {
                "idx": int(cue_row["idx"]),
                "cue": cue,
                "category": cue_row.get("category", ""),
                "description": cue_row.get("description", ""),
                "platforms": {},
            },
        )
        prev = entry["platforms"].get(PLATFORM)
        if args.resume and prev and str(prev.get("source_code") or "").strip():
            print(f"  skip {cue}", flush=True)
            continue

        print(f"  gen {PLATFORM}/{cue} ...", flush=True)
        prompt = build_prompt(
            cue=cue, description=str(cue_row.get("description") or "")
        )
        try:
            raw = gemini_generate(prompt, model=args.model)
            obj = _extract_json_obj(raw)
            code = extract_motion_code(raw, obj)
            if not code:
                raise RuntimeError("empty motion code")
            row = {
                "idx": int(cue_row["idx"]),
                "cue": cue,
                "description": str(cue_row.get("description") or ""),
                "category": cue_row.get("category", ""),
                "state": "motion_code_gemini",
                "model": args.model,
                "time": datetime.now().isoformat(timespec="seconds"),
                "experiment": "oddkin_alpha1_motion_code_gemini",
                "platform": PLATFORM,
                "parse_ok": True,
                "source_code": code,
                "notes": [],
            }
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {cue}: {exc}", flush=True)
            n_fail += 1
            msg = str(exc)
            if "RESOURCE_EXHAUSTED" in msg or "spending cap" in msg.lower():
                print("[abort] Gemini quota exhausted", flush=True)
                raise SystemExit(2) from exc
            raw = getattr(exc, "raw", "") or ""
            row = {
                "idx": int(cue_row["idx"]),
                "cue": cue,
                "description": str(cue_row.get("description") or ""),
                "category": cue_row.get("category", ""),
                "state": "motion_code_gemini",
                "model": args.model,
                "time": datetime.now().isoformat(timespec="seconds"),
                "experiment": "oddkin_alpha1_motion_code_gemini",
                "platform": PLATFORM,
                "parse_ok": False,
                "source_code": "",
                "notes": [f"error: {exc}"],
            }

        (raw_dir / f"{cue}.txt").write_text(
            raw if isinstance(raw, str) else "", encoding="utf-8"
        )
        # keep raw off the master blob
        entry["platforms"][PLATFORM] = {k: v for k, v in row.items() if k != "raw"}
        configs = [by_cue[c["cue"]] for c in cues if c["cue"] in by_cue]
        master_path.write_text(
            json.dumps(configs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            f"  saved {cue} parse_ok={row.get('parse_ok')} "
            f"code={str(row.get('source_code') or '')[:90]!r}",
            flush=True,
        )
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"done ok={n_ok} fail={n_fail} out={master_path}", flush=True)


if __name__ == "__main__":
    main()
