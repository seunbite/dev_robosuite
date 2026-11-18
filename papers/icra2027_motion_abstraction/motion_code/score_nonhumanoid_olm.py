#!/usr/bin/env python3
"""Score Gemini Motion Programs vs human GT on 30 cues × IIWA/Stretch/Roomba.

Orientation / Location / Movement are scored only when the GT program actually
specifies that axis (N/A otherwise). Roomba orientation and location are always
N/A: heading tokens in LLM programs are not recognition-critical on a disk.
Movement ignores speed/rep/hold and matches signed axis / path class.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

MC = Path(__file__).resolve().parent
PAPER = MC.parent
REPO = PAPER.parent.parent

GT_PATH = REPO / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
PRED_PATH = (
    REPO
    / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro"
    / "embodiment_motion_code_all.json"
)
PLATFORMS = ("single_arm", "stretch", "roomba")
PLAT_NAME = {"single_arm": "IIWA", "stretch": "Stretch", "roomba": "Roomba"}
LOC_TOL = 20.0

OUT_JSON = MC / "nonhumanoid30_olm_scores.json"
OUT_MD = MC / "nonhumanoid30_olm_scores.md"


def _code(row: dict[str, Any]) -> str:
    return (row.get("source_code") or "").replace("\n", " ")


def _norm_dir(raw: str | None) -> str | None:
    if not raw:
        return None
    d = raw.strip().lower()
    aliases = {
        "still": "still",
        "center": "still",
        "front": "front",
        "back": "back",
        "in": "in",
        "out": "out",
        "up": "up",
        "down": "down",
        "left": "in",
        "right": "out",
    }
    return aliases.get(d)


def _norm_grip(raw: str | None) -> str | None:
    if not raw:
        return None
    g = raw.strip().lower().replace("horizontal", "horiz").replace("vertical", "vert")
    if g in {"horiz", "horz"}:
        return "horiz"
    if g in {"vert"}:
        return "vert"
    if g in {"close", "closed", "fist", "open", "pinch"}:
        return g
    parts = re.split(r"[|/]", g)
    for p in parts:
        p = p.strip()
        if p in {"horiz", "vert"}:
            return p
    return g or None


def _sign_from_values(text: str) -> str | None:
    t = text.replace(" ", "")
    if "±" in t or "+-" in t or "-+" in t:
        return "pm"
    nums = [float(x) for x in re.findall(r"[+-]?\d+(?:\.\d+)?", t)]
    if len(nums) >= 2 and ((nums[0] > 0 and nums[1] < 0) or (nums[0] < 0 and nums[1] > 0)):
        return "pm"
    if re.search(r"dist=\+", t) or re.search(r"x=\+", t):
        return "plus"
    if re.search(r"dist=-", t) or re.search(r"x=-", t):
        return "minus"
    if nums:
        if all(n >= 0 for n in nums) and any(n > 0 for n in nums):
            return "plus"
        if all(n <= 0 for n in nums) and any(n < 0 for n in nums):
            return "minus"
    return None


def _sign_compat(gt: str | None, pred: str | None) -> bool:
    if gt is None or pred is None:
        return gt == pred
    if gt == "pm" or pred == "pm":
        return True
    return gt == pred


def parse_poses(code: str) -> list[dict[str, Any]]:
    poses = []
    for m in re.finditer(r"pose\(([^)]*)\)", code, re.I):
        body = m.group(1)
        p: dict[str, Any] = {}
        for part in re.split(r",\s*", body):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            k, v = k.strip().lower(), v.strip()
            if k == "dir":
                p["dir"] = _norm_dir(v)
            elif k == "grip":
                p["grip"] = _norm_grip(v)
            elif k == "head":
                p["head"] = v.strip().lower()
            elif k == "torso":
                p["torso"] = v.strip().lower()
            elif k in {"x", "y", "z"}:
                try:
                    p[k] = float(re.findall(r"-?\d+(?:\.\d+)?", v)[0])
                except IndexError:
                    pass
        poses.append(p)
    return poses


def parse_moves(code: str) -> list[dict[str, Any]]:
    out = []
    for m in re.finditer(r"move\(([^)]*)\)", code, re.I):
        body = m.group(1)
        toks = [t.strip() for t in body.split(",")]
        joint = (toks[0] if toks else "").lower()
        axis = None
        rest = body
        am = re.search(r"axis\s*=\s*([a-zA-Z]+)", body, re.I)
        if am:
            axis = am.group(1).lower()
        elif len(toks) >= 2 and toks[1].lower() in {"x", "y", "z", "tilt", "pan", "roll", "height", "yaw", "pitch"}:
            axis = toks[1].lower()
        if axis == "tilt":
            axis = "z"
        if axis == "pan":
            axis = "y"
        if axis == "height":
            axis = "z"
        sign = _sign_from_values(rest)
        out.append({"kind": "move", "joint": joint, "axis": axis, "sign": sign, "raw": body})
    return out


def parse_paths(code: str) -> list[dict[str, Any]]:
    out = []
    for m in re.finditer(r"path\(([^)]*)\)", code, re.I):
        body = m.group(1)
        toks = [t.strip() for t in body.split(",")]
        carrier = (toks[0] if toks else "ee").lower()
        shape = "line"
        if any(t.lower() == "arc" for t in toks) or "sweep=" in body.lower() or re.search(r"\br\s*=", body):
            if "line" in [t.lower() for t in toks[1:3]]:
                shape = "line"
            else:
                shape = "arc" if ("arc" in body.lower() or "sweep=" in body.lower() or re.search(r"\br\s*=", body)) else "line"
        if "arc" in body.lower():
            shape = "arc"
        if re.search(r"\bline\b", body.lower()) and "arc" not in body.lower():
            shape = "line"
        axis = None
        am = re.search(r"axis\s*=\s*([a-zA-Z]+)", body, re.I)
        if am:
            axis = am.group(1).lower()
        elif re.search(r"\bx\s*=", body):
            axis = "x"
        elif re.search(r"\by\s*=", body):
            axis = "y"
        elif re.search(r"\bz\s*=", body):
            axis = "z"
        plane = None
        pm = re.search(r"plane\s*=\s*([a-zA-Z]+)", body, re.I)
        if pm:
            plane = pm.group(1).lower()
        r = None
        rm = re.search(r"\br\s*=\s*([0-9.]+)", body, re.I)
        if rm:
            r = float(rm.group(1))
        sign = _sign_from_values(body)
        cls = None
        if carrier == "base":
            if shape == "arc" and (r is None or r < 0.08) and (not plane):
                cls = "yaw"
            elif shape == "arc":
                cls = "circle"
            elif axis == "y" or re.search(r"\by\s*=", body):
                cls = "strafe"
            else:
                if sign == "pm":
                    cls = "rock"
                elif sign == "minus":
                    cls = "retreat"
                else:
                    cls = "advance"
        out.append(
            {
                "kind": "path",
                "carrier": carrier,
                "shape": shape,
                "axis": axis,
                "plane": plane,
                "r": r,
                "sign": sign,
                "cls": cls,
                "raw": body,
            }
        )
    return out


def last_pose_with(poses: list[dict[str, Any]], *keys: str) -> dict[str, Any]:
    for p in reversed(poses):
        if any(k in p for k in keys):
            return p
    return poses[-1] if poses else {}


def score_orientation(plat: str, gt_code: str, pred_code: str) -> dict[str, Any]:
    if plat == "roomba":
        return {"applicable": False, "ok": None, "reason": "N/A (no palm/head/gripper)"}
    gp, pp = parse_poses(gt_code), parse_poses(pred_code)
    gt_dirs = {p["dir"] for p in gp if p.get("dir") and p["dir"] != "still"}
    gt_grips = {p["grip"] for p in gp if p.get("grip")}
    gt_heads = {p["head"] for p in gp if p.get("head")}
    if not gt_dirs and not gt_grips and not gt_heads:
        return {"applicable": False, "ok": None, "reason": "N/A (GT has no ori tokens)"}
    pred_dirs = {p["dir"] for p in pp if p.get("dir") and p["dir"] != "still"}
    pred_grips = {p["grip"] for p in pp if p.get("grip")}
    pred_heads = {p["head"] for p in pp if p.get("head")}
    reasons = []
    ok = True
    if gt_dirs:
        # Last distinctive GT dir should appear in pred (or pred last dir).
        want = last_pose_with(gp, "dir").get("dir")
        got = last_pose_with(pp, "dir").get("dir")
        if want == "still":
            pass
        elif want not in pred_dirs and want != got:
            ok = False
            reasons.append(f"dir {want}→{got or pred_dirs or '∅'}")
    if gt_grips:
        want_g = last_pose_with(gp, "grip").get("grip")
        got_g = last_pose_with(pp, "grip").get("grip")
        # HV only; open/close/fist compared exactly if that is what GT used
        if want_g in {"horiz", "vert"}:
            if got_g != want_g:
                ok = False
                reasons.append(f"grip {want_g}→{got_g}")
        elif want_g and want_g not in pred_grips and want_g != got_g:
            ok = False
            reasons.append(f"grip {want_g}→{got_g}")
    if gt_heads:
        want_h = last_pose_with(gp, "head").get("head")
        got_h = last_pose_with(pp, "head").get("head")
        if want_h and want_h != got_h and want_h not in pred_heads:
            ok = False
            reasons.append(f"head {want_h}→{got_h}")
    return {"applicable": True, "ok": ok, "reason": "; ".join(reasons) or "match"}


def score_location(plat: str, gt_code: str, pred_code: str) -> dict[str, Any]:
    if plat == "roomba":
        return {"applicable": False, "ok": None, "reason": "N/A (no body landmarks)"}
    gp, pp = parse_poses(gt_code), parse_poses(pred_code)
    keys = []
    for k in ("x", "y", "z", "torso"):
        if any(k in p for p in gp):
            keys.append(k)
    if not keys:
        return {"applicable": False, "ok": None, "reason": "N/A (GT has no loc tokens)"}
    g = last_pose_with(gp, *keys)
    p = last_pose_with(pp, *keys) if pp else {}
    reasons = []
    ok = True
    for k in ("x", "y", "z"):
        if k not in g:
            continue
        if k not in p:
            ok = False
            reasons.append(f"{k} {g[k]}→∅")
            continue
        if abs(float(g[k]) - float(p[k])) > LOC_TOL:
            ok = False
            reasons.append(f"{k} {g[k]:.0f}→{p[k]:.0f}")
    if "torso" in g:
        if p.get("torso") != g["torso"]:
            ok = False
            reasons.append(f"torso {g['torso']}→{p.get('torso')}")
    return {"applicable": True, "ok": ok, "reason": "; ".join(reasons) or "match"}


def _dyn_keys(code: str, plat: str) -> list[tuple]:
    keys = []
    for mv in parse_moves(code):
        axis = mv.get("axis") or "_"
        keys.append(("move", axis, mv.get("sign") or "pm"))
    for ph in parse_paths(code):
        if plat == "roomba" or ph.get("carrier") == "base":
            keys.append(("path", ph.get("cls") or ph.get("shape"), ph.get("sign")))
        else:
            axis = ph.get("axis")
            if not axis and ph.get("plane"):
                axis = ph["plane"]
            keys.append(("path", axis or ph.get("shape"), ph.get("sign")))
    if not keys and re.search(r"hold\(", code):
        keys.append(("hold", "static", None))
    return keys


def score_movement(plat: str, gt_code: str, pred_code: str) -> dict[str, Any]:
    gk = _dyn_keys(gt_code, plat)
    pk = _dyn_keys(pred_code, plat)
    if not gk or gk == [("hold", "static", None)]:
        # Static GT: pred must not introduce a strong locomotion/stroke.
        if not pk or pk == [("hold", "static", None)]:
            return {"applicable": True, "ok": True, "reason": "static"}
        # extra path/move is a miss
        return {"applicable": True, "ok": False, "reason": f"static→{pk[0]}"}
    g0 = gk[0]
    ok = False
    for p0 in pk:
        if g0[0] == "hold":
            continue
        if plat == "roomba":
            if g0[1] == p0[1]:
                ok = True
                break
            # rock vs advance: both line-x, require sign class
            if g0[1] in {"rock", "advance", "retreat"} and p0[1] in {"rock", "advance", "retreat"}:
                if g0[1] == p0[1]:
                    ok = True
                    break
            continue
        # axis match
        ga, pa = str(g0[1]), str(p0[1])
        if ga in pa or pa in ga or ga == pa:
            if _sign_compat(g0[2], p0[2]):
                ok = True
                break
        # yz plane vs z/y
        if set(ga) & set(pa) and len(ga) > 1:
            if _sign_compat(g0[2], p0[2]):
                ok = True
                break
    return {
        "applicable": True,
        "ok": ok,
        "reason": "match" if ok else f"{g0} vs {pk[:2]}",
    }


def load_by_cue(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {r["cue"]: r for r in rows}


def main() -> None:
    gt = load_by_cue(GT_PATH)
    pred = load_by_cue(PRED_PATH)
    cues = list(gt.keys())
    cells = []
    agg: dict[str, dict[str, list[int]]] = {p: defaultdict(list) for p in PLATFORMS}

    for cue in cues:
        idx = gt[cue]["idx"]
        for plat in PLATFORMS:
            gc = _code(gt[cue]["platforms"][plat])
            pc = _code(pred[cue]["platforms"][plat])
            o = score_orientation(plat, gc, pc)
            l = score_location(plat, gc, pc)
            m = score_movement(plat, gc, pc)
            cell = {
                "idx": idx,
                "cue": cue,
                "platform": plat,
                "gt": gc,
                "pred": pc,
                "orientation": o,
                "location": l,
                "movement": m,
            }
            cells.append(cell)
            for name, sc in (("ori", o), ("loc", l), ("mov", m)):
                if sc["applicable"]:
                    agg[plat][name].append(int(bool(sc["ok"])))
            apps = [sc for sc in (o, l, m) if sc["applicable"]]
            if apps:
                agg[plat]["all"].append(int(all(sc["ok"] for sc in apps)))

    summary = {}
    for plat in PLATFORMS:
        summary[plat] = {}
        for k in ("ori", "loc", "mov", "all"):
            xs = agg[plat][k]
            summary[plat][k] = {
                "n": len(xs),
                "n_ok": int(sum(xs)),
                "pct": (100.0 * sum(xs) / len(xs)) if xs else None,
            }

    payload = {"summary": summary, "loc_tol": LOC_TOL, "n_cues": len(cues), "cells": cells}
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Non-humanoid O/L/M agreement (30 cues, Gemini 2.5 Pro vs GT)",
        "",
        f"Location band = ±{LOC_TOL:.0f} body-%. Roomba O/L always N/A.",
        "",
        "| Robot | Ori % (n) | Loc % (n) | Mov % (n) | All applicable % (n) |",
        "|---|---:|---:|---:|---:|",
    ]
    for plat in PLATFORMS:
        s = summary[plat]
        def fmt(key):
            d = s[key]
            if not d["n"]:
                return "—"
            return f"{d['pct']:.1f} ({d['n_ok']}/{d['n']})"
        lines.append(f"| {PLAT_NAME[plat]} | {fmt('ori')} | {fmt('loc')} | {fmt('mov')} | {fmt('all')} |")
    lines += ["", "## Per cue", ""]
    lines.append("| # | Cue | IIWA O L M | Stretch O L M | Roomba O L M |")
    lines.append("|---:|---|---|---|---|")

    def mark(sc):
        if not sc["applicable"]:
            return "—"
        return "Y" if sc["ok"] else "N"

    by_cue: dict[str, dict[str, Any]] = defaultdict(dict)
    for c in cells:
        by_cue[c["cue"]][c["platform"]] = c
    for cue in cues:
        idx = gt[cue]["idx"]
        bits = []
        for plat in PLATFORMS:
            cell = by_cue[cue][plat]
            bits.append(f"{mark(cell['orientation'])} {mark(cell['location'])} {mark(cell['movement'])}")
        lines.append(f"| {idx} | `{cue}` | {bits[0]} | {bits[1]} | {bits[2]} |")

    lines += ["", "## Misses", ""]
    for c in cells:
        misses = []
        for name in ("orientation", "location", "movement"):
            sc = c[name]
            if sc["applicable"] and not sc["ok"]:
                misses.append(f"{name[:3]}: {sc['reason']}")
        if misses:
            lines.append(f"- **{PLAT_NAME[c['platform']]}** `{c['cue']}` — " + "; ".join(misses))

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
