#!/usr/bin/env python3
"""Fill manipulator motion JSON with cues listed in cues_3.yml that are absent from merged config.

Writes into the canonical JSON from ``see_html_sources.yml`` (manipulator) unless overridden.
Passes ``cue_idx=None`` so new rows get idx = max(existing)+1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_ARM = Path(__file__).resolve().parent
if str(_ARM) not in sys.path:
    sys.path.insert(0, str(_ARM))

from adhoc.generation.embodiment_sources import resolve_embodiment_paths  # noqa: E402
from legacy.config_gen_single import generate_motion_config  # noqa: E402


def _palette_cues(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected YAML mapping root: {path}")
    ordered: list[str] = []
    seen: set[str] = set()
    for section, blob in data.items():
        if not isinstance(blob, str):
            continue
        for cue in blob.split():
            if cue not in seen:
                seen.add(cue)
                ordered.append(cue)
    return ordered


def _merged_cues(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return {str(r["cue"]) for r in rows if isinstance(r, dict) and r.get("cue")}


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-generate manipulator configs for cues_3 ∪ merged gaps.")
    ap.add_argument(
        "--palette-yml",
        type=str,
        default=str(_REPO / "data" / "seed" / "yml" / "cues_3.yml"),
    )
    ap.add_argument(
        "--sources-yml",
        type=str,
        default=None,
        help="Optional see_html_sources.yml (default seed/yml path).",
    )
    ap.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="Override merged JSON path (else manipulator.motion_config_json from YAML).",
    )
    ap.add_argument(
        "--prompt-file",
        type=str,
        default=None,
    )
    ap.add_argument(
        "--shots-json",
        type=str,
        default=None,
    )
    ap.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",
    )
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print missing cues and exit.",
    )
    ap.add_argument(
        "--require-reasoning",
        action="store_true",
        help="Turn on reasoning validation (merged file historically used no-validation style).",
    )
    args = ap.parse_args()

    if not args.dry_run and not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY.")

    pale = Path(args.palette_yml)
    merged = Path(args.config_json) if args.config_json else resolve_embodiment_paths("manipulator", Path(args.sources_yml) if args.sources_yml else None)[2]

    palette_cues = _palette_cues(pale.resolve())
    have = _merged_cues(merged.resolve())
    missing = [c for c in palette_cues if c not in have]

    print(f"[cues_3 gap] palette={pale.relative_to(_REPO)} merged={merged.relative_to(_REPO)} missing={len(missing)}")
    if not missing:
        print("Nothing to do.")
        return
    if args.dry_run:
        for c in missing:
            print(c)
        return

    src_opt = Path(args.sources_yml) if args.sources_yml else None
    prompt_p, shots_p, _, _ = resolve_embodiment_paths("manipulator", src_opt)
    prompt = str(Path(args.prompt_file)) if args.prompt_file else str(prompt_p)
    shots = str(Path(args.shots_json)) if args.shots_json else str(shots_p)
    out = str(merged.resolve())

    failed = 0
    for cue in missing:
        ok = False
        last_err = ""
        for attempt in range(args.max_retries + 1):
            try:
                generate_motion_config(
                    cue_name=cue,
                    cue_idx=None,
                    model_name=args.model,
                    prompt_file=prompt,
                    shots_json=shots,
                    config_json=out,
                    require_reasoning=bool(args.require_reasoning),
                )
                ok = True
                break
            except Exception as e:
                last_err = str(e)
                if attempt < args.max_retries:
                    time.sleep(args.delay * (2**attempt))
        if not ok:
            failed += 1
            print(f"FAILED {cue}: {last_err[:400]}")

    print(f"Done. failures={failed} output={merged}")


if __name__ == "__main__":
    main()
