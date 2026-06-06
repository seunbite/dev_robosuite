#!/usr/bin/env python3
"""Parse human movement-component GT annotations and score GT-fixed generation tails."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent


def _dev_root() -> Path:
    for anc in _HERE.parents:
        if (anc / "data" / "results" / "verify").is_dir():
            return anc
    raise SystemExit("cannot find repo root")


_REPO = _dev_root()
MOTION_JSON = _REPO / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_gt_fixed_pose_pilot40.json"
MOTION_EVAL = _REPO / "data/results/verify/pilot40_motion_gt_fixed_eval_consolidated.json"
OUT_GT = _REPO / "data/results/verify/pilot40_motion_component_gt.json"
OUT_SCORE = _REPO / "data/results/verify/pilot40_motion_component_scored.json"
OUT_TSV = _REPO / "data/results/verify/pilot40_motion_component_scored.tsv"

# User annotations in consolidated row order (first 13 sequential), then explicit cue_idx.
RAW_LINES = """
y + non hold
x + non hold
z + non shoulder hold
z - non shoulder
z +- rep wrist
x - non hold
x - non/rep
x +
z - rep wrist
x + non hold
x - rep wrist
z +- rep wrist
19 x - non hold
20 y +- rep
21 y +- rep
22 arc xz
23 x + non hold
24 y - non
25 z + non hold
26 z +
27 x + non hold
28 x + non hold
30 z +- rep
32 line y
35 z + non hold
36 y +- rep
37 x +
39 x + non
40
41 x + non
45 x -
50 x + non hold
59 z +- rep
60 y +- rep
61 x +- rep
64 y + non hold
66 x + non hold
74 y - non hold
104 y + non
""".strip().splitlines()

SEQUENTIAL_CUE_ORDER = [
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
]


def _parse_annotation_line(line: str) -> tuple[int | None, str]:
    line = line.strip()
    if not line:
        return None, ""
    if re.fullmatch(r"\d+", line):
        return int(line), ""
    m = re.match(r"^(\d+)\s+(.+)$", line)
    if m:
        return int(m.group(1)), m.group(2).strip()
    return None, line


def _parse_component(spec: str) -> dict[str, Any] | None:
    spec = spec.strip()
    if not spec:
        return None

    m = re.match(
        r"^arc\s+(xy|yz|xz|zx)"
        r"(?:\s+plane)?"
        r"(?:\s+(\d+(?:\.\d+)?)\s*(?:deg|degrees?)?)?"
        r"(?:\s+(cw|ccw|counterclockwise|counterclock|clockwise))?\s*$",
        spec,
        re.I,
    )
    if m:
        plane = m.group(1).lower()
        if plane == "zx":
            plane = "xz"
        out: dict[str, Any] = {"kind": "path_arc", "plane": plane}
        if m.group(2) is not None:
            out["sweep"] = float(m.group(2))
        if m.group(3):
            d = m.group(3).lower()
            if d in ("cw", "clockwise"):
                out["direction"] = "cw"
            else:
                out["direction"] = "ccw"
        return out

    m = re.match(r"^line\s+([xyz])$", spec, re.I)
    if m:
        return {"kind": "path_line", "axis": m.group(1).lower()}

    joint = None
    for j in ("shoulder", "elbow", "wrist"):
        if re.search(rf"\b{j}\b", spec, re.I):
            joint = j
            spec = re.sub(rf"\b{j}\b", "", spec, flags=re.I).strip()

    repetition = None
    if re.search(r"\bnon/rep\b", spec, re.I):
        repetition = "any"
        spec = re.sub(r"\bnon/rep\b", "", spec, flags=re.I).strip()
    elif re.search(r"\bnon\b", spec, re.I):
        repetition = "non"
        spec = re.sub(r"\bnon\b", "", spec, flags=re.I).strip()
    elif re.search(r"\brep\b", spec, re.I):
        repetition = "rep"
        spec = re.sub(r"\brep\b", "", spec, flags=re.I).strip()

    hold = None
    if re.search(r"\bhold\b", spec, re.I):
        hold = True
        spec = re.sub(r"\bhold\b", "", spec, flags=re.I).strip()

    axes: dict[str, str] = {}
    for ax in "xyz":
        m = re.search(rf"(?<![a-z]){ax}\s*(\+-|[+-])", spec, re.I)
        if m:
            sign = m.group(1)
            axes[ax] = "+-" if sign == "+-" else sign
            spec = spec[: m.start()] + spec[m.end() :]
            spec = spec.strip()

    # fallback: lone "x +" style
    if not axes:
        m = re.match(r"^([xyz])\s*([+-]|[+-])$", spec.strip(), re.I)
        if m:
            axes[m.group(1).lower()] = m.group(2).replace("±", "+-")

    if not axes and joint is None:
        return None

    return {
        "kind": "movement",
        "axes": axes,
        "joint": joint,
        "repetition": repetition,
        "hold": hold,
        "raw": spec,
    }


def _tail_steps(movements: list) -> list[dict]:
    out = []
    seen_pose = False
    for step in movements:
        if step.get("type") == "pose":
            seen_pose = True
            continue
        if seen_pose:
            out.append(step)
    return out


def _deg_sign(v: float) -> str:
    if v > 0:
        return "+"
    if v < 0:
        return "-"
    return "0"


def _collect_axis_signs(step: dict) -> dict[str, set[str]]:
    """Per step: axis -> signs seen in degrees."""
    signs: dict[str, set[str]] = {a: set() for a in "xyz"}
    p = step.get("parameters") or {}
    t = step.get("type")
    if t == "movement":
        for d in p.get("directions") or []:
            deg = d.get("degrees") or {}
            for ax, val in deg.items():
                if ax in signs and isinstance(val, (int, float)):
                    signs[ax].add(_deg_sign(float(val)))
    return signs


def _step_repetition(step: dict) -> int:
    p = step.get("parameters") or {}
    return int(p.get("repetition", 1) or 1)


def _step_has_hold(step: dict) -> bool:
    p = step.get("parameters") or {}
    if step.get("type") == "path":
        return float(p.get("hold_time", 0) or 0) > 0
    for d in p.get("directions") or []:
        if float(d.get("hold_time", 0) or 0) > 0:
            return True
    return False


def _step_joint(step: dict) -> str | None:
    p = step.get("parameters") or {}
    return (p.get("joint") or "").lower() or None


def _match_movement_component(step: dict, comp: dict[str, Any]) -> bool:
    if step.get("type") != "movement":
        return False
    joint = comp.get("joint")
    if joint and _step_joint(step) != joint:
        return False

    rep_rule = comp.get("repetition")
    rep = _step_repetition(step)
    if rep_rule == "non" and rep != 1:
        return False
    if rep_rule == "rep" and rep < 2:
        # also allow rep=1 with 2+ directions as repeated beat
        dirs = (step.get("parameters") or {}).get("directions") or []
        if len(dirs) < 2:
            return False

    if comp.get("hold") is True and not _step_has_hold(step):
        return False

    signs = _collect_axis_signs(step)
    for ax, rule in (comp.get("axes") or {}).items():
        s = signs.get(ax, set()) - {"0"}
        if rule == "+":
            if "+" not in s:
                return False
        elif rule == "-":
            if "-" not in s:
                return False
        elif rule == "+-":
            if "+" not in s or "-" not in s:
                # allow across multiple directions in one step
                all_s: set[str] = set()
                for d in (step.get("parameters") or {}).get("directions") or []:
                    deg = d.get("degrees") or {}
                    if ax in deg and isinstance(deg[ax], (int, float)):
                        all_s.add(_deg_sign(float(deg[ax])))
                if "+" not in all_s or "-" not in all_s:
                    return False
    return True


def _match_path_component(step: dict, comp: dict[str, Any]) -> bool:
    if step.get("type") != "path":
        return False
    p = step.get("parameters") or {}
    if comp["kind"] == "path_line":
        return p.get("shape") == "line" and (p.get("axis") or "").lower() == comp["axis"]
    if comp["kind"] == "path_arc":
        plane = (p.get("plane") or "").lower()
        if plane == "zx":
            plane = "xz"
        return p.get("shape") in ("arc", "circle") and plane == comp["plane"]
    return False


def _line_signs_and_steps(tail: list, axis: str) -> tuple[set[str], list[dict]]:
    """Collect signs and candidate line-path steps for one axis."""
    signs: set[str] = set()
    steps: list[dict] = []
    for step in tail:
        if step.get("type") != "path":
            continue
        p = step.get("parameters") or {}
        if p.get("shape") != "line":
            continue
        if (p.get("axis") or "").lower() != axis:
            continue
        dist = p.get("distance")
        if not isinstance(dist, (int, float)) or float(dist) == 0.0:
            continue
        steps.append(step)
        signs.add(_deg_sign(float(dist)))
    return signs - {"0"}, steps


def _match_line_path_as_movement(tail: list, comp: dict[str, Any]) -> bool:
    """
    Accept line-path as movement-equivalent when axis/sign, repetition, hold match.
    This applies only when GT does not constrain a specific joint.
    """
    if comp.get("kind") != "movement":
        return False
    if comp.get("joint"):
        return False

    axes = comp.get("axes") or {}
    if not axes:
        return False

    # Evaluate each required axis against line-path distance signs.
    all_candidate_steps: list[dict] = []
    for ax, rule in axes.items():
        signs, steps = _line_signs_and_steps(tail, ax)
        if not steps:
            return False
        all_candidate_steps.extend(steps)
        if rule == "+" and "+" not in signs:
            return False
        if rule == "-" and "-" not in signs:
            return False
        if rule == "+-" and ("+" not in signs or "-" not in signs):
            return False

    rep_rule = comp.get("repetition")
    hold_rule = comp.get("hold")

    reps = [int((s.get("parameters") or {}).get("repetition", 1) or 1) for s in all_candidate_steps]
    # If repetition missing, treat each line step as one event.
    events = sum(max(1, r) for r in reps) if reps else 0

    if rep_rule == "non":
        # Non-repeated: at least one candidate step with repetition 1.
        if not any(r == 1 for r in reps):
            return False
    elif rep_rule == "rep":
        if events < 2:
            return False

    if hold_rule is True:
        if not any(_step_has_hold(s) for s in all_candidate_steps):
            return False

    return True


def _tail_matches_component(tail: list, comp: dict[str, Any]) -> tuple[bool, str | None]:
    kind = comp["kind"]
    for step in tail:
        if kind.startswith("path"):
            if _match_path_component(step, comp):
                return True, step.get("type")
        elif _match_movement_component(step, comp):
            return True, step.get("type")
    if kind == "movement" and _match_line_path_as_movement(tail, comp):
        return True, "path_line_as_movement"
    return False, None


def _build_annotation_map() -> list[dict[str, Any]]:
    # If user-edited GT JSON exists, use it as the source of truth.
    if OUT_GT.is_file():
        try:
            data = json.loads(OUT_GT.read_text(encoding="utf-8"))
            anns = data.get("annotations")
            if isinstance(anns, list) and anns:
                out: list[dict[str, Any]] = []
                for a in anns:
                    raw = str(a.get("annotation_raw", ""))
                    # Re-parse from edited annotation text so user changes apply immediately.
                    comp = _parse_component(raw) if raw else a.get("component")
                    out.append(
                        {
                            "cue_idx": int(a.get("cue_idx")),
                            "cue": a.get("cue"),
                            "annotation_raw": raw,
                            "component": comp,
                        }
                    )
                return out
        except Exception:
            # Fall back to RAW_LINES parser if JSON is malformed.
            pass

    seq_i = 0
    entries = []
    for raw in RAW_LINES:
        raw = raw.strip()
        if not raw and "empty" not in raw:
            continue
        cue_idx, spec = _parse_annotation_line(raw)
        if cue_idx is None:
            if seq_i >= len(SEQUENTIAL_CUE_ORDER):
                raise ValueError(f"Too many sequential lines: {raw!r}")
            cue_idx = SEQUENTIAL_CUE_ORDER[seq_i]
            seq_i += 1
        comp = _parse_component(spec) if spec else None
        entries.append(
            {
                "cue_idx": cue_idx,
                "annotation_raw": spec,
                "component": comp,
            }
        )
    return entries


def main() -> None:
    annotations = _build_annotation_map()
    cfg_rows = json.loads(MOTION_JSON.read_text(encoding="utf-8"))
    by_idx = {int(r["idx"]): r for r in cfg_rows}
    by_cue = {r["cue"]: r for r in cfg_rows}

    eval_rows = json.loads(MOTION_EVAL.read_text(encoding="utf-8"))["rows"]
    eval_by_idx = {int(r["cue_idx"]): r for r in eval_rows}

    gt_payload = {
        "groundtruth_note": (
            "Per-cue movement/path component requirement. "
            "non=repetition 1; rep=repetition>1 or 2+ directions; hold=hold_time>0; "
            "axis +=positive degree only, -=negative only, +-=both signs required. "
            "Unspecified fields are not checked."
        ),
        "n": len(annotations),
        "annotations": [],
    }

    scored = []
    ok_n = 0
    scored_n = 0

    for ann in annotations:
        idx = ann["cue_idx"]
        cfg = by_idx.get(idx)
        if not cfg:
            # duplicate 11: scratch
            cfg = next((r for r in cfg_rows if int(r["idx"]) == idx), None)
        cue = cfg["cue"] if cfg else f"idx_{idx}"
        comp = ann["component"]
        tail = _tail_steps(cfg.get("movements") or []) if cfg else []

        row = {
            "cue_idx": idx,
            "cue": cue,
            "annotation_raw": ann["annotation_raw"],
            "component_gt": comp,
            "generation_tail": copy.deepcopy(tail),
            "n_tail_steps": len(tail),
        }

        if comp is None:
            row["match"] = None
            row["skipped"] = True
            row["note"] = "no component requirement (empty annotation)"
        else:
            matched, via = _tail_matches_component(tail, comp)
            row["match"] = matched
            row["matched_via"] = via
            row["skipped"] = False
            scored_n += 1
            if matched:
                ok_n += 1

        ev = eval_by_idx.get(idx) or {}
        row["human_pose_gt"] = ev.get("groundtruth")
        gt_payload["annotations"].append(
            {
                "cue_idx": idx,
                "cue": cue,
                "annotation_raw": ann["annotation_raw"],
                "component": comp,
            }
        )
        scored.append(row)

    accuracy = ok_n / scored_n if scored_n else None
    OUT_GT.write_text(json.dumps(gt_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    score_payload = {
        "n_annotated": len(annotations),
        "n_scored": scored_n,
        "n_correct": ok_n,
        "accuracy": accuracy,
        "rows": scored,
    }
    OUT_SCORE.write_text(json.dumps(score_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    tsv = ["cue_idx\tcue\tannotation\tmatch\tn_tail\tcomponent_kind"]
    for r in scored:
        comp = r.get("component_gt") or {}
        kind = comp.get("kind", "-") if comp else "-"
        m = r.get("match")
        mtxt = "" if m is None else str(m)
        tsv.append(
            f"{r['cue_idx']}\t{r['cue']}\t{r.get('annotation_raw','')}\t{mtxt}\t"
            f"{r.get('n_tail_steps',0)}\t{kind}"
        )
    OUT_TSV.write_text("\n".join(tsv) + "\n", encoding="utf-8")

    # Merge into motion eval consolidated
    if MOTION_EVAL.is_file():
        ev = json.loads(MOTION_EVAL.read_text(encoding="utf-8"))
        by_idx = {int(a["cue_idx"]): a for a in gt_payload["annotations"]}
        score_by_idx = {int(r["cue_idx"]): r for r in scored}
        for row in ev.get("rows", []):
            idx = int(row["cue_idx"])
            if idx in by_idx:
                row["groundtruth_motion_component"] = by_idx[idx].get("component")
                row["groundtruth_motion_annotation"] = by_idx[idx].get("annotation_raw")
            if idx in score_by_idx:
                row["generation_motion_component_match"] = score_by_idx[idx].get("match")
        ev["motion_component_gt_json"] = str(OUT_GT.relative_to(_REPO))
        ev["motion_component_scored_json"] = str(OUT_SCORE.relative_to(_REPO))
        MOTION_EVAL.write_text(json.dumps(ev, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Updated {MOTION_EVAL}")

    print(f"Wrote {OUT_GT}")
    print(f"Wrote {OUT_SCORE}")
    print(f"Wrote {OUT_TSV}")
    print(f"Accuracy: {ok_n}/{scored_n} = {accuracy*100:.1f}%" if accuracy is not None else "No scored cues")

    print("\nMisses:")
    for r in scored:
        if r.get("match") is False:
            print(f"  {r['cue_idx']} {r['cue']}: {r.get('annotation_raw')}")


if __name__ == "__main__":
    main()
