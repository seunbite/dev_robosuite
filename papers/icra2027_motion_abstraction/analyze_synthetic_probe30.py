#!/usr/bin/env python3
"""Structural probe of the 30-cue x 3 synthetic-morphology Motion Programs.

Reads synthetic_morphology/yml/cues30_oddkin.yaml and checks every program
against motion_code_synthetic_continuum.yaml. Reports, per body:
  - carrier usage (which carriers each cue activates),
  - motion structure (static pose / joint oscillation / path line / path arc),
  - within-body collapse (cues sharing the same structural signature),
  - cross-body change (does the motion structure change with the body?),
  - schema violations (keys, values, joints, axes, or modes outside the YAML).

These are program-level observations, not recognition measurements.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PROGRAMS = HERE / "synthetic_morphology/yml/cues30_oddkin.yaml"
SCHEMA = HERE / "synthetic_morphology/yml/motion_code_synthetic_continuum.yaml"
OUT_JSON = HERE / "synthetic_probe30.json"
OUT_MD = HERE / "synthetic_probe30.md"

BODIES = ("gcr_alpha1", "gcr_alpha05", "gcr_alpha0")
DISPLAY = {"gcr_alpha1": "PaddleClaw", "gcr_alpha05": "SpineCrab", "gcr_alpha0": "TrilegCrown"}

# Pose key / move joint / path mode -> physical carrier family.
CARRIER = {
    "gcr_alpha0": {
        "dir": "base", "leg0": "leg0", "leg1": "leg1", "leg2": "leg2", "crown": "crown",
        "hip0": "leg0", "knee0": "leg0", "hip1": "leg1", "knee1": "leg1",
        "hip2": "leg2", "knee2": "leg2", "base": "base",
    },
    "gcr_alpha05": {
        "dir": "base", "torso": "spine", "long": "long_pincer", "long_grip": "long_pincer",
        "short": "short_pincer", "antenna": "antenna", "long_pincer": "long_pincer",
        "short_pincer": "short_pincer", "collar": "collar", "flap_l": "flaps", "flap_r": "flaps",
        "ee_long": "long_pincer", "ee_short": "short_pincer", "base": "base",
    },
    "gcr_alpha1": {
        "dir": "base", "torso": "torso", "paddle": "paddle", "palm": "paddle", "elbow_p": "paddle",
        "wrist_p": "paddle", "ee_paddle": "paddle", "claw": "claw", "grip": "claw", "elbow_c": "claw",
        "wrist_c": "claw", "ee_claw": "claw", "head": "head", "sat": "sat", "base": "base",
        "x": "ee", "y": "ee", "z": "ee",
    },
}
NEUTRAL = {"still", "rest", "center", "front", "mid"}
TIMING = {"spd", "rep", "hold"}


def split_top(text: str, sep: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in text:
        depth += ch in "([" and 1 or (ch in ")]" and -1 or 0)
        if ch == sep and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    out.append(cur.strip())
    return [s for s in out if s]


def parse(code: str) -> list[dict]:
    steps = []
    for raw in re.split(r"\s\+\s", code):
        m = re.fullmatch(r"(\w+)\((.*)\)", raw.strip())
        if not m:
            steps.append({"prim": "?", "raw": raw})
            continue
        pos, kw = [], {}
        for arg in split_top(m.group(2), ","):
            if "=" in arg and not arg.startswith("["):
                k, v = arg.split("=", 1)
                kw[k.strip()] = v.strip()
            else:
                pos.append(arg)
        steps.append({"prim": m.group(1), "pos": pos, "kw": kw, "raw": raw.strip()})
    return steps


def schema_vocab(block: dict) -> dict:
    pose = {v["dsl_key"]: v.get("values") for v in block["pose"].values()
            if isinstance(v, dict) and v.get("dsl_key")}
    move = block["move"]["joint"]
    path = block["path"]
    path_kw = {"r", "deg", "sweep", "dist", "axis", "plane", "x", "y", "z"} | TIMING
    return {
        "pose": pose,
        "joints": set(move["values"]),
        "axis_by_joint": move.get("axis_by_joint", {}),
        "modes": set(path["mode"]["values"]),
        "shapes": set(path["shape"]["values"]),
        "path_kw": path_kw,
    }


def check(step: dict, vocab: dict) -> list[str]:
    prim, errs = step["prim"], []
    if prim == "pose":
        for k, v in step["kw"].items():
            if k in TIMING:
                continue
            if k not in vocab["pose"]:
                errs.append(f"pose key `{k}` not in schema")
            elif vocab["pose"][k] is not None:
                allowed = {str(x).lower() for x in vocab["pose"][k]}
                if v.lower() not in allowed:
                    errs.append(f"pose `{k}={v}` not in {sorted(allowed)}")
    elif prim == "move":
        joint, axis = (step["pos"] + ["", ""])[:2]
        amount = step["pos"][2] if len(step["pos"]) > 2 else ""
        if joint not in vocab["joints"]:
            errs.append(f"move joint `{joint}` not in schema")
        elif axis not in vocab["axis_by_joint"].get(joint, []):
            errs.append(f"move `{joint}` axis `{axis}` not in {vocab['axis_by_joint'].get(joint)}")
        if amount and not re.fullmatch(r"[±+-]?\d+(\.\d+)?|\[[-+\d., ]+\]", amount):
            errs.append(f"move amount `{amount}` is not a degree value")
        elif amount in {"0", "+0", "-0"}:
            errs.append("move with zero amplitude (no-op)")
    elif prim == "path":
        mode, shape = (step["pos"] + ["", ""])[:2]
        if mode not in vocab["modes"]:
            errs.append(f"path mode `{mode}` not in schema")
        if shape not in vocab["shapes"]:
            errs.append(f"path shape `{shape}` not in schema")
    elif prim not in {"hold"}:
        errs.append(f"unknown primitive `{prim}`")
    return errs


def oscillates(step: dict) -> bool:
    amount = step["pos"][2] if step["prim"] == "move" and len(step["pos"]) > 2 else step["kw"].get("deg", "")
    rep = int(step["kw"].get("rep", "1") or 1)
    return rep > 1 or "±" in amount or amount.startswith("[")


def describe(steps: list[dict], body: str) -> dict:
    carriers, kinds, pose_sig, motion_sig = set(), set(), [], []
    cmap = CARRIER[body]
    for s in steps:
        if s["prim"] == "pose":
            for k, v in s["kw"].items():
                if k in TIMING:
                    continue
                if v not in NEUTRAL:
                    carriers.add(cmap.get(k, k))
                    pose_sig.append(f"{k}={v}")
        elif s["prim"] == "move":
            joint, axis = (s["pos"] + ["", ""])[:2]
            carrier = cmap.get(joint, joint)
            carriers.add(carrier)
            kind = f"osc:{axis}" if oscillates(s) else f"stroke:{axis}"
            kinds.add(kind)
            motion_sig.append(f"move({carrier},{kind})")
        elif s["prim"] == "path":
            mode, shape = (s["pos"] + ["", ""])[:2]
            carrier = cmap.get(mode, mode)
            carriers.add(carrier)
            kind = f"{shape}{'(osc)' if oscillates(s) else ''}"
            kinds.add(f"path:{kind}")
            motion_sig.append(f"path({carrier},{kind})")
    if not kinds:
        kinds.add("static")
    return {
        "carriers": sorted(carriers),
        "kinds": sorted(kinds),
        "structure": "static" if kinds == {"static"} else ("dynamic" if all(not k.startswith("static") for k in kinds) else "mixed"),
        "exact_signature": " | ".join(sorted(pose_sig) + motion_sig) or "(neutral)",
        "coarse_signature": " | ".join(sorted(carriers)) + " :: " + " | ".join(sorted(kinds)),
    }


def main() -> None:
    programs = yaml.safe_load(PROGRAMS.read_text())["cues"]
    schema = yaml.safe_load(SCHEMA.read_text())
    vocab = {b: schema_vocab(schema[b]) for b in BODIES}
    prescribed = schema.get("rematerialize", {}) or {}

    rows, violations = {}, []
    for cue, per_body in programs.items():
        rows[cue] = {}
        for body in BODIES:
            code = per_body[body]
            steps = parse(code)
            d = describe(steps, body)
            errs = [e for s in steps for e in check(s, vocab[body])]
            d.update(code=code, violations=errs)
            rows[cue][body] = d
            violations += [(cue, body, e) for e in errs]

    carrier_usage = {b: Counter(c for cue in rows for c in rows[cue][b]["carriers"]) for b in BODIES}
    structure = {b: Counter(rows[cue][b]["structure"] for cue in rows) for b in BODIES}
    collapse = {}
    for level in ("exact_signature", "coarse_signature"):
        collapse[level] = {}
        for b in BODIES:
            groups = defaultdict(list)
            for cue in rows:
                groups[rows[cue][b][level]].append(cue)
            collapse[level][b] = {sig: cues for sig, cues in groups.items() if len(cues) > 1}

    cross = {}
    for cue in rows:
        kinds = {b: tuple(rows[cue][b]["kinds"]) for b in BODIES}
        cross[cue] = {
            "same_motion_structure": len(set(kinds.values())) == 1,
            "kinds": {DISPLAY[b]: list(k) for b, k in kinds.items()},
        }
    n_changed = sum(not v["same_motion_structure"] for v in cross.values())
    verbatim = sorted(
        cue for cue, per in prescribed.items()
        if isinstance(per, dict) and cue in programs
        and all(str(per.get(b, "")).split("#")[0].strip().replace(" ", "") in programs[cue][b].replace(" ", "")
                for b in BODIES if per.get(b))
    )

    payload = {
        "status": "program-level structural probe; not a recognition result",
        "provenance_note": "Programs are hard-coded in synthetic_morphology/render_cues30.py (put(...)) together with joint-level render plans; they are not outputs of the Gemini authoring pipeline.",
        "n_cues": len(rows),
        "carrier_usage": {DISPLAY[b]: dict(carrier_usage[b].most_common()) for b in BODIES},
        "structure": {DISPLAY[b]: dict(structure[b]) for b in BODIES},
        "collapse": {lvl: {DISPLAY[b]: g for b, g in per.items()} for lvl, per in collapse.items()},
        "cross_body_motion_structure_changed": n_changed,
        "schema_prescribed_cues_matching_programs": verbatim,
        "violations": [{"cue": c, "body": DISPLAY[b], "issue": e} for c, b, e in violations],
        "per_cue": {cue: {DISPLAY[b]: rows[cue][b] for b in BODIES} for cue in rows},
        "cross_body": cross,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    md = ["# Synthetic-morphology probe (30 cues x 3 bodies)", "",
          "Program-level structural observations; not recognition measurements.", "",
          f"**Provenance:** {payload['provenance_note']}", "",
          "## Carrier usage (number of cues activating each carrier)", ""]
    for b in BODIES:
        md.append(f"- **{DISPLAY[b]}**: " + ", ".join(f"{c} {n}" for c, n in carrier_usage[b].most_common()))
    md += ["", "## Motion structure", ""]
    for b in BODIES:
        md.append(f"- **{DISPLAY[b]}**: " + ", ".join(f"{k} {v}" for k, v in structure[b].items()))
    md += ["", f"Cues whose motion structure differs across the three bodies: **{n_changed}/{len(rows)}**", ""]
    for level, title in (("exact_signature", "Exact collapse (identical non-neutral pose values and motions)"),
                         ("coarse_signature", "Coarse collapse (same carriers and motion kinds)")):
        md += [f"## {title}", ""]
        for b in BODIES:
            groups = collapse[level][b]
            n = sum(len(v) for v in groups.values())
            md.append(f"- **{DISPLAY[b]}** ({n} cues in {len(groups)} groups)")
            for sig, cues in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                md.append(f"  - `{sig}` <- {', '.join(cues)}")
        md.append("")
    md += ["## Schema violations", ""]
    md += [f"- {c} / {DISPLAY[b]}: {e}" for c, b, e in violations] or ["- none"]
    md += ["", "## Schema `rematerialize` examples reproduced verbatim by the 30-cue programs", "",
           ", ".join(verbatim) or "none", "", "## Per-cue programs", "",
           "| Cue | PaddleClaw | SpineCrab | TrilegCrown | Same structure |", "|---|---|---|---|:---:|"]
    for cue in rows:
        cells = [rows[cue][b]["code"].replace("|", "\\|") for b in BODIES]
        md.append(f"| `{cue}` | `{cells[0]}` | `{cells[1]}` | `{cells[2]}` | {'Y' if cross[cue]['same_motion_structure'] else 'N'} |")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(OUT_MD.read_text()[:6000])


if __name__ == "__main__":
    main()
