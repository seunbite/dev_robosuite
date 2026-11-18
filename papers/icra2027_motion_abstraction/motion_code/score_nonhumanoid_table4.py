#!/usr/bin/env python3
"""30-cue IIWA / Stretch / Roomba vs GT, Table-IV columns.

x/y/z: mean |Δ| body-% on GT-specified pose axes (missing pred axis → 50).
Rch: last non-still reach dir (6-way).
Plm: last H/V gripper (2-way; Tiago-style).
Mov: signed axis / Roomba path class (same as score_nonhumanoid_olm.py).
Roomba x/y/z/Rch/Plm are N/A.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

MC = Path(__file__).resolve().parent
PAPER = MC.parent
REPO = PAPER.parent.parent
sys.path.insert(0, str(MC))

from score_nonhumanoid_olm import (  # noqa: E402
    PLAT_NAME,
    PLATFORMS,
    _code,
    last_pose_with,
    load_by_cue,
    parse_poses,
    score_movement,
)

GT_PATH = REPO / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
MISS_FILL = 50.0

GENERATORS = {
    "gemini": {
        "label": "Gemini",
        "path": REPO
        / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro"
        / "embodiment_motion_code_all.json",
    },
    "qwen32b": {
        "label": "Qwen-32B",
        "path": REPO
        / "data/results/motion_configs/embodiment_motion_code_qwen_qwen32b_paper30"
        / "embodiment_motion_code_all.json",
    },
    "qwen7b": {
        "label": "Qwen-7B",
        "path": REPO
        / "data/results/motion_configs/embodiment_motion_code_qwen_qwen7b_paper30"
        / "embodiment_motion_code_all.json",
    },
}

OUT_JSON = MC / "nonhumanoid30_table4_scores.json"
OUT_MD = MC / "nonhumanoid30_table4_scores.md"


def _hv(grip: str | None) -> str | None:
    if grip in {"horiz", "vert"}:
        return grip
    return None


def _e_unif_abs(g: float) -> float:
    """E[|U-g|] for U~Unif[0,100]."""
    return (g * g + (100.0 - g) * (100.0 - g)) / 200.0


def score_xyz(gt_code: str, pred_code: str) -> dict[str, float | None]:
    gp, pp = parse_poses(gt_code), parse_poses(pred_code)
    g = last_pose_with(gp, "x", "y", "z")
    p = last_pose_with(pp, "x", "y", "z") if pp else {}
    out: dict[str, float | None] = {"x": None, "y": None, "z": None}
    for ax in ("x", "y", "z"):
        if ax not in g:
            continue
        gv = float(g[ax])
        pv = float(p[ax]) if ax in p else MISS_FILL
        out[ax] = abs(pv - gv)
    return out


def score_reach(gt_code: str, pred_code: str) -> bool | None:
    gp, pp = parse_poses(gt_code), parse_poses(pred_code)
    want = last_pose_with(gp, "dir").get("dir")
    if not want or want == "still":
        return None
    got = last_pose_with(pp, "dir").get("dir") if pp else None
    pred_dirs = {q["dir"] for q in pp if q.get("dir") and q["dir"] != "still"}
    return want == got or want in pred_dirs


def score_palm(gt_code: str, pred_code: str) -> bool | None:
    gp, pp = parse_poses(gt_code), parse_poses(pred_code)
    want = _hv(last_pose_with(gp, "grip").get("grip"))
    if not want:
        return None
    # Strict: the gripper at the scored (last) pose, matching the documented
    # "last H/V gripper" rule and the Rch column's last-non-still convention.
    # Accepting the token from any earlier pose over-counts programs that
    # switch aperture mid-script (4/28 on IIWA).
    got = _hv(last_pose_with(pp, "grip").get("grip") if pp else None)
    return got == want


def _mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def _pct(oks: list[bool]) -> float | None:
    return (100.0 * sum(1 for x in oks if x) / len(oks)) if oks else None


def score_generator(gt: dict[str, dict], pred: dict[str, dict]) -> dict[str, Any]:
    cues = list(gt.keys())
    out: dict[str, Any] = {}
    for plat in PLATFORMS:
        xs: dict[str, list] = {k: [] for k in ("x", "y", "z", "rch", "plm", "mov", "rand_x", "rand_y", "rand_z")}
        cells = []
        for cue in cues:
            gc = _code(gt[cue]["platforms"][plat])
            pc = _code((pred.get(cue) or {}).get("platforms", {}).get(plat) or {})
            cell: dict[str, Any] = {"cue": cue, "gt": gc, "pred": pc}
            if plat == "roomba":
                cell["x"] = cell["y"] = cell["z"] = None
                cell["rch"] = cell["plm"] = None
            else:
                xyz = score_xyz(gc, pc)
                cell.update(xyz)
                gp = last_pose_with(parse_poses(gc), "x", "y", "z")
                for ax in ("x", "y", "z"):
                    if xyz[ax] is not None:
                        xs[ax].append(xyz[ax])
                        xs[f"rand_{ax}"].append(_e_unif_abs(float(gp[ax])))
                rch = score_reach(gc, pc)
                plm = score_palm(gc, pc)
                cell["rch"] = rch
                cell["plm"] = plm
                if rch is not None:
                    xs["rch"].append(bool(rch))
                if plm is not None:
                    xs["plm"].append(bool(plm))
            mov = score_movement(plat, gc, pc)
            cell["mov"] = bool(mov["ok"]) if mov["applicable"] else None
            cell["mov_reason"] = mov.get("reason")
            if cell["mov"] is not None:
                xs["mov"].append(bool(cell["mov"]))
            cells.append(cell)
        out[plat] = {
            "n": {k: len(xs[k]) for k in ("x", "y", "z", "rch", "plm", "mov")},
            "x": _mean(xs["x"]),
            "y": _mean(xs["y"]),
            "z": _mean(xs["z"]),
            "rch": _pct(xs["rch"]),
            "plm": _pct(xs["plm"]),
            "mov": _pct(xs["mov"]),
            "random": {
                "x": _mean(xs["rand_x"]),
                "y": _mean(xs["rand_y"]),
                "z": _mean(xs["rand_z"]),
                "rch": 100.0 / 6.0 if plat != "roomba" else None,
                "plm": 50.0 if plat != "roomba" else None,
                "mov": 100.0 / 9.0,
            },
            "cells": cells,
        }
    return out


def _fmt(v: float | None, *, pct: bool = False) -> str:
    if v is None:
        return "---"
    if pct:
        return f"{v:.1f}"
    return f"{v:.1f}"


def latex_table(payload: dict[str, Any]) -> str:
    gens = payload["generators"]
    order = [("gemini", "Gemini"), ("qwen32b", "Qwen-32B"), ("qwen7b", "Qwen-7B")]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\renewcommand{\arraystretch}{0.98}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}lllrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Robot} & \textbf{Generator} & \textbf{Category} &",
        r"$x{\downarrow}$ & $y{\downarrow}$ & $z{\downarrow}$ &",
        r"Rch${\uparrow}$ & Plm${\uparrow}$ & Mov${\uparrow}$ \\",
        r"\midrule",
    ]
    plat_rows = 1 + sum(1 for k, _ in order if k in gens)  # + Random
    for i, plat in enumerate(PLATFORMS):
        first = True
        n_gen = sum(1 for k, _ in order if k in gens) + 1
        for key, label in order:
            if key not in gens:
                continue
            s = gens[key][plat]
            robot = rf"\multirow{{{n_gen}}}{{*}}{{{PLAT_NAME[plat]}}}" if first else ""
            first = False
            lines.append(
                f" {robot} & {label} & All & "
                f"{_fmt(s['x'])} & {_fmt(s['y'])} & {_fmt(s['z'])} & "
                f"{_fmt(s['rch'], pct=True)} & {_fmt(s['plm'], pct=True)} & "
                f"{_fmt(s['mov'], pct=True)} \\\\"
            )
        # Random from first available generator (same GT)
        src = gens[next(k for k, _ in order if k in gens)][plat]
        rnd = src["random"]
        robot = "" if not first else rf"\multirow{{{n_gen}}}{{*}}{{{PLAT_NAME[plat]}}}"
        lines.append(
            f" {robot} & Random & All & "
            f"{_fmt(rnd['x'])} & {_fmt(rnd['y'])} & {_fmt(rnd['z'])} & "
            f"{_fmt(rnd['rch'], pct=True)} & {_fmt(rnd['plm'], pct=True)} & "
            f"{_fmt(rnd['mov'], pct=True)} \\\\"
        )
        if i < len(PLATFORMS) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{Attribute agreement vs.\ human GT on the 30-cue inventory (no category split).",
        r"$x$/$y$/$z$: mean $|\Delta|$ body-\% on GT-specified pose axes ($\downarrow$; omitted predicted axes filled at $50$).",
        r"Rch/Plm/Mov: agreement \% ($\uparrow$). Roomba has no recognition-critical location, reach, or palm.",
        r"Random: $x$/$y$/$z$ $=$ $\mathbb{E}[|U{-}g|]$ for $U{\sim}\mathrm{Unif}[0,100]$; Rch $=$ $1/6$; Plm $=$ $1/2$ (H/V); Mov $=$ $1/9$.",
        r"Gemini $=$ Gemini~2.5~Pro; Qwen-32B/7B $=$ Qwen2.5-VL-32B/7B.}",
        r"\label{tab:nonhumanoid-olm}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    gt = load_by_cue(GT_PATH)
    generators: dict[str, Any] = {}
    missing = []
    for key, meta in GENERATORS.items():
        path = Path(meta["path"])
        if not path.is_file():
            missing.append(f"{key}: {path}")
            continue
        pred = load_by_cue(path)
        generators[key] = score_generator(gt, pred)
        generators[key]["_path"] = str(path)
        generators[key]["_label"] = meta["label"]

    payload = {
        "miss_fill": MISS_FILL,
        "n_cues": len(gt),
        "missing_preds": missing,
        "generators": generators,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = ["# Non-humanoid Table-IV columns (30 cues)", ""]
    if missing:
        lines.append("Missing predictions:")
        lines += [f"- {m}" for m in missing]
        lines.append("")
    header = "| Robot | Generator | x | y | z | Rch | Plm | Mov |"
    lines += [header, "|---|---|---:|---:|---:|---:|---:|---:|"]
    order = [("gemini", "Gemini"), ("qwen32b", "Qwen-32B"), ("qwen7b", "Qwen-7B")]
    for plat in PLATFORMS:
        for key, label in order:
            if key not in generators:
                continue
            s = generators[key][plat]
            lines.append(
                f"| {PLAT_NAME[plat]} | {label} | {_fmt(s['x'])} | {_fmt(s['y'])} | {_fmt(s['z'])} | "
                f"{_fmt(s['rch'])} | {_fmt(s['plm'])} | {_fmt(s['mov'])} |"
            )
        if generators:
            src = generators[next(k for k, _ in order if k in generators)][plat]
            rnd = src["random"]
            lines.append(
                f"| {PLAT_NAME[plat]} | Random | {_fmt(rnd['x'])} | {_fmt(rnd['y'])} | {_fmt(rnd['z'])} | "
                f"{_fmt(rnd['rch'])} | {_fmt(rnd['plm'])} | {_fmt(rnd['mov'])} |"
            )
    lines += ["", "## n (applicable cues)", ""]
    for key, label in order:
        if key not in generators:
            continue
        lines.append(f"### {label}")
        for plat in PLATFORMS:
            n = generators[key][plat]["n"]
            lines.append(f"- {PLAT_NAME[plat]}: {n}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (MC / "nonhumanoid30_table4.tex").write_text(latex_table(payload), encoding="utf-8")
    print(json.dumps({k: {p: {kk: generators[k][p][kk] for kk in ('x','y','z','rch','plm','mov','n')} for p in PLATFORMS} for k in generators}, indent=2))
    print("missing", missing)
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
