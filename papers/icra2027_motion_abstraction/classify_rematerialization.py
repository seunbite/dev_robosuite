#!/usr/bin/env python3
"""Code every non-humanoid Motion Program into one rematerialization strategy.

Strategies (Sec. VI, Fig. 7):
  carrier  -- carrier transfer: the reference signed stroke (axis, sign) survives
              on a different carrier (joint or base).
  trace    -- shape tracing: the lost pose/handshape/envelope is redrawn as an
              arc or a multi-segment path.
  additive -- additive carrier motion: a stroke absent from the reference is
              added on whatever carrier remains.
  none     -- no rematerialization: pose-only, or a single advance/hold.

Reference = human-authored GT humanoid program for the same cue.
Rule order is none -> trace -> carrier -> additive; the first match wins.
Tracing is tested before carrier transfer because drawing a spatial envelope
is the more specific act: depict_big_bowl keeps the reference arc axis on
every body, yet what it rematerialises is the lost cupped handshape.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "motion_code"))

from score_nonhumanoid_olm import (  # noqa: E402
    _code,
    _sign_compat,
    load_by_cue,
    parse_moves,
    parse_paths,
    parse_poses,
)

REPO = HERE.parents[1]
MC = REPO / "data/results/motion_configs"
GT = MC / "embodiment_motion_code_gt/embodiment_motion_code_all.json"
GEMINI = MC / "embodiment_motion_code_gemini_gemini-2.5-pro/embodiment_motion_code_all.json"
STRAT = {
    "roomba_strategy": REPO / "data/results/roomba_strategy_fewshot/results_all30.json",
    "stretch_strategy": REPO / "data/results/stretch_strategy_fewshot/results_all30.json",
    "single_arm_strategy": REPO / "data/results/single_arm_strategy_fewshot/results_all30.json",
}

BODIES = ("tiago", "single_arm", "stretch", "roomba")
BODY_NAME = {
    "tiago": "Tiago",
    "single_arm": "IIWA",
    "stretch": "Stretch",
    "roomba": "Roomba",
    "roomba_strategy": "Roomba (strategy-cond.)",
    "stretch_strategy": "Stretch (strategy-cond.)",
    "single_arm_strategy": "IIWA (strategy-cond.)",
}
STRAT_ORDER = ("single_arm_strategy", "stretch_strategy", "roomba_strategy")
STRATEGIES = ("carrier", "trace", "additive", "none")


ROT_TOKENS = {"tilt", "pitch", "nod", "pan", "yaw", "roll", "twist", "shake"}


def _move_axis(mv: dict) -> str:
    """Axis token as written, before parse_moves' joint-rotation normalisation.

    parse_moves maps tilt->z and pan->y, i.e. it names the joint's rotation
    axis, whereas parse_paths' z is a vertical translation.  Mixing the two
    conventions would call a head nod and an arm lift the same stroke, so the
    rotation name is recovered from the raw token here.
    """
    toks = [t.strip().lower() for t in str(mv.get("raw", "")).split(",")]
    for t in toks[1:3]:
        if t in ROT_TOKENS or t in {"x", "y", "z", "height"}:
            return t
    return str(mv.get("axis") or "_").lower()


def dyn_keys(code: str) -> list[tuple[str, str]]:
    """Canonical (axis, sign) strokes, carrier-agnostic."""
    keys = []
    for mv in parse_moves(code):
        keys.append((_move_axis(mv), mv.get("sign") or "pm"))
    for ph in parse_paths(code):
        axis = ph.get("axis") or ph.get("plane") or ph.get("cls") or ph.get("shape")
        keys.append((str(axis).lower(), ph.get("sign") or "pm"))
    return keys


def axis_alias(a: str) -> set[str]:
    """Stroke names grouped by the plane the stroke is *seen* in.

    Carrier transfer in the sense of Sec. VI preserves the apparent signed
    stroke, so a head pitch and a fore-aft base rock belong together even
    though one is a rotation and the other a translation.  A yaw is grouped
    with lateral strokes (head shake, strafe) and NOT with a vertical lift.
    """
    table = {
        "sagittal": {"x", "pitch", "tilt", "nod", "rock", "advance", "retreat", "fore"},
        "lateral": {"y", "yaw", "pan", "shake", "strafe", "lateral"},
        "vertical": {"z", "lift", "raise", "height", "vertical"},
        "roll": {"roll", "twist"},
    }
    for canon, names in table.items():
        if a in names:
            return names | {canon}
    return {a}


def stroke_match(ref: list[tuple[str, str]], tgt: list[tuple[str, str]]) -> bool:
    for ra, rs in ref:
        for ta, ts in tgt:
            if (axis_alias(ra) & axis_alias(ta)) and _sign_compat(rs, ts):
                return True
    return False


def is_envelope(code: str) -> bool:
    """A traced spatial envelope, not an in-place rotation.

    An arc counts only when it has a real radius; r<0.08 is the in-place yaw
    threshold already used by score_nonhumanoid_olm._parse_paths, and such a
    spin is an added rotation (additive), not a redrawn shape.  A polyline
    counts only when it changes direction at least three times.
    """
    paths = parse_paths(code)
    if any(p.get("shape") == "arc" and (p.get("r") or 0.0) >= 0.08 for p in paths):
        return True
    lines = [p for p in paths if p.get("shape") == "line"]
    return len({(p.get("axis"), p.get("sign")) for p in lines}) >= 3


def has_dynamics(code: str) -> bool:
    if parse_moves(code):
        return True
    for p in parse_paths(code):
        if p.get("cls") not in (None, "advance") or p.get("sign") == "pm":
            return True
    return len(parse_paths(code)) >= 2


def classify(ref_code: str, tgt_code: str) -> str:
    if not has_dynamics(tgt_code):
        return "none"
    if is_envelope(tgt_code):
        return "trace"
    if stroke_match(dyn_keys(ref_code), dyn_keys(tgt_code)):
        return "carrier"
    return "additive"


def main() -> None:
    gt = load_by_cue(GT)
    gem = load_by_cue(GEMINI)
    strat = {}
    for name, path in STRAT.items():
        if path.is_file():
            strat[name] = {r["cue"]: r for r in json.loads(path.read_text(encoding="utf-8"))}

    counts: dict[str, Counter] = defaultdict(Counter)
    cells = []

    for cue, grow in gt.items():
        ref = _code(grow["platforms"]["humanoid"])
        for body in BODIES:
            tgt = _code(gem[cue]["platforms"][body])
            s = classify(ref, tgt)
            counts[body][s] += 1
            cells.append({"cue": cue, "body": body, "strategy": s, "code": tgt})
        for name, rows in strat.items():
            sc = (rows.get(cue, {}).get("motion_code") or "")
            if not sc:
                continue
            s = classify(ref, sc)
            counts[name][s] += 1
            cells.append({"cue": cue, "body": name, "strategy": s, "code": sc})

    payload = {
        "n_cues": len(gt),
        "bodies": list(counts),
        "counts": {b: dict(c) for b, c in counts.items()},
        "cells": cells,
    }
    (HERE / "rematerialization_strategy_counts.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8"
    )

    hdr = f"{'Body':<24}" + "".join(f"{s:>10}" for s in STRATEGIES) + f"{'n':>6}"
    print(hdr)
    print("-" * len(hdr))
    for b in list(BODIES) + [k for k in STRAT_ORDER if counts[k]]:
        c = counts[b]
        n = sum(c.values())
        row = f"{BODY_NAME[b]:<24}"
        for s in STRATEGIES:
            row += f"{c[s]:>4} ({100*c[s]/n:>3.0f}%)".rjust(10)
        print(row + f"{n:>6}")


if __name__ == "__main__":
    main()
