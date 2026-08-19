#!/usr/bin/env python3
"""Single-subprocess Gemini upserts for one embodiment (avoids ``legacy.*`` import clashes)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_generate(slug: str):
    slug = slug.strip().lower().replace("-", "_")
    if slug in ("manipulator", "robotarm", "arm", "iiwa"):
        rd = _REPO / "adhoc" / "generation" / "robotarm"
        sys.path.insert(0, str(rd))
        from legacy.config_gen_single import generate_motion_config  # noqa: E402

        return generate_motion_config
    if slug in ("tiago", "google", "mobile", "bimanual"):
        rd = _REPO / "adhoc" / "generation" / "tiago"
        sys.path.insert(0, str(rd))
        from legacy.config_gen_single_mobile import generate_motion_config  # noqa: E402

        return generate_motion_config
    if slug == "quadruped":
        rd = _REPO / "adhoc" / "generation" / "quadruped"
        sys.path.insert(0, str(rd))
        from legacy.config_gen_single import generate_motion_config  # noqa: E402

        return generate_motion_config
    raise SystemExit(f"Unknown slug {slug!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", help="manipulator | tiago | quadruped")
    ap.add_argument("--manifest", required=True, help="JSON dict: cues, config_json, prompt_file, shots_json, yaml_path (may be \"\"), delay, max_retries, model, require_reasoning")
    args = ap.parse_args()
    raw = Path(args.manifest).read_text(encoding="utf-8")
    m = json.loads(raw)

    cues: list[str] = m["cues"]
    config_json = m["config_json"]
    prompt_file = m["prompt_file"]
    shots_json = m["shots_json"]
    yaml_path = m.get("yaml_path") or None
    delay = float(m.get("delay", 2.0))
    max_retries = int(m.get("max_retries", 2))
    model = str(m.get("model", "gemini-2.5-pro"))
    require_reasoning = bool(m.get("require_reasoning", False))

    if not cues:
        print("nothing to generate")
        return

    gen = _load_generate(args.slug)
    failures = []
    kw: dict = {
        "cue_idx": None,
        "model_name": model,
        "prompt_file": prompt_file,
        "shots_json": shots_json,
        "config_json": config_json,
        "require_reasoning": require_reasoning,
    }
    if yaml_path and args.slug.lower() not in ("manipulator", "robotarm", "arm", "iiwa"):
        kw["yaml_path"] = yaml_path

    for cue in cues:
        ok = False
        last_err = ""
        for attempt in range(max_retries + 1):
            try:
                gen(cue_name=cue, **kw)
                ok = True
                break
            except Exception as e:
                last_err = str(e)
                if attempt < max_retries:
                    time.sleep(delay * (2**attempt))
        if not ok:
            failures.append((cue, last_err))
            print(f"FAILED {cue}: {last_err[:500]}")

    if failures:
        raise SystemExit(f"failures={len(failures)}")


if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY.")
    main()
