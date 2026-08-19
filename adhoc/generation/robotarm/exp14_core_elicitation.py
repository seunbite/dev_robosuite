#!/usr/bin/env python3
"""Exp14: cue name → essential pose/movement attributes (vs human GT)."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pilot90_paths import GT_PATH, PROMPT_EXP_DIR, load_gt_by_cue  # noqa: E402

PROMPT_EXP14 = PROMPT_EXP_DIR / "prompt_exp14.txt"
OUT_DIR = _REPO / "data/results/motion_configs/manipulator/exp"
SCORE_DIR = _REPO / "data/results/verify/manipulator/exp"


def _parse_gt_poses(groundtruth: str) -> list[tuple[str, str]]:
    return [(a.strip(), b.strip()) for a, b in re.findall(r"\(([^,]+),\s*([^)]+)\)", groundtruth or "")]


def _llm_generate(prompt: str, *, model: str) -> str:
    from google import genai

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(model=model, contents=prompt)
    return (resp.text or "").strip()


def _extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])


def _validate_pred(pred: dict[str, Any], cue: str) -> list[str]:
    errs: list[str] = []
    if pred.get("cue") != cue:
        errs.append(f"cue mismatch: {pred.get('cue')!r}")
    for key in (
        "essential_pose",
        "essential_movement",
        "pose_irrelevant",
        "movement_irrelevant",
        "reasoning",
    ):
        if key not in pred:
            errs.append(f"missing key: {key}")
    ep = pred.get("essential_pose") or {}
    if not isinstance(ep, dict):
        errs.append("essential_pose must be object")
    em = pred.get("essential_movement") or {}
    if not isinstance(em, dict):
        errs.append("essential_movement must be object")
    return errs


def _pred_pose_pairs(pred: dict[str, Any]) -> set[tuple[str, str]]:
    if pred.get("pose_irrelevant"):
        return set()
    ep = pred.get("essential_pose") or {}
    if not ep.get("required"):
        return set()
    out: set[tuple[str, str]] = set()
    for p in ep.get("pairs") or []:
        if isinstance(p, dict):
            d = str(p.get("dir", "")).strip()
            g = str(p.get("gripper_orientation", "")).strip()
            if d and g:
                out.add((d, g))
    return out


def _gt_pose_pairs(gt_row: dict[str, Any]) -> set[tuple[str, str]]:
    s = str(gt_row.get("pose_gt") or gt_row.get("groundtruth") or "")
    return set(_parse_gt_poses(s))


def _movement_fields(comp: dict[str, Any] | None) -> dict[str, Any]:
    if not comp:
        return {}
    kind = comp.get("kind") or "movement"
    out: dict[str, Any] = {"kind": kind}
    if kind == "movement":
        out["axes"] = dict(comp.get("axes") or {})
        out["repetition"] = comp.get("repetition")
        out["hold"] = comp.get("hold")
        out["joint"] = comp.get("joint")
    elif kind == "path_line":
        out["axis"] = comp.get("axis")
    elif kind == "path_arc":
        out["plane"] = comp.get("plane")
    return out


def _pred_movement_fields(pred: dict[str, Any]) -> dict[str, Any]:
    if pred.get("movement_irrelevant"):
        return {}
    em = pred.get("essential_movement") or {}
    if not em.get("required"):
        return {}
    kind = em.get("kind") or "movement"
    out: dict[str, Any] = {"kind": kind}
    if kind == "movement":
        out["axes"] = {str(k): str(v) for k, v in (em.get("axes") or {}).items()}
        out["repetition"] = em.get("repetition")
        out["hold"] = em.get("hold")
        out["joint"] = em.get("joint")
    elif kind == "path_line":
        out["axis"] = em.get("axis")
    elif kind == "path_arc":
        out["plane"] = em.get("plane")
    return out


def _match_repetition(pred: str | None, gt: str | None) -> bool:
    if pred is None or gt is None:
        return pred == gt
    if gt == "any":
        return pred in ("non", "rep", "any")
    if pred == "any":
        return True
    return pred == gt


def _score_row(pred: dict[str, Any], gt_row: dict[str, Any]) -> dict[str, Any]:
    mg = gt_row.get("movement_gt") or {}
    gt_comp = mg.get("component") or {}
    ann = str(mg.get("annotation_raw") or "")

    gt_mov = _movement_fields(gt_comp)
    pr_mov = _pred_movement_fields(pred)
    gt_pairs = _gt_pose_pairs(gt_row)
    pr_pairs = _pred_pose_pairs(pred)

    pose_notes: list[str] = []
    mov_notes: list[str] = []

    # Pose scoring
    if pred.get("pose_irrelevant"):
        pose_ok = len(gt_pairs) >= 3  # loose GT — shrug/wave style
        pose_notes.append("pred pose_irrelevant")
    elif not pr_pairs and gt_pairs:
        pose_ok = False
        pose_notes.append("pred missing pose pairs")
    elif pr_pairs and gt_pairs:
        overlap = pr_pairs & gt_pairs
        if overlap:
            pose_ok = True
            pose_notes.append(f"overlap {overlap}")
        elif str(gt_row.get("pose_gt", "")).startswith("x") and pr_pairs <= gt_pairs:
            pose_ok = True
            pose_notes.append("pred subset of x-GT options")
        else:
            pose_ok = False
            pose_notes.append(f"no overlap pred={pr_pairs} gt={gt_pairs}")
    else:
        pose_ok = True
        pose_notes.append("no pose constraint")

    # Movement scoring
    if mg.get("always_correct") or ann == "none":
        mov_ok = pred.get("movement_irrelevant") or not pr_mov
        mov_notes.append("GT movement loose")
    elif pred.get("movement_irrelevant") and gt_mov:
        mov_ok = False
        mov_notes.append("pred movement_irrelevant but GT has component")
    elif not pr_mov and gt_mov:
        mov_ok = False
        mov_notes.append("pred missing movement")
    elif pr_mov and gt_mov:
        mov_ok = True
        if pr_mov.get("kind") != gt_mov.get("kind"):
            mov_ok = False
            mov_notes.append(f"kind {pr_mov.get('kind')} vs {gt_mov.get('kind')}")
        if pr_mov.get("kind") == "movement":
            gt_ax = gt_mov.get("axes") or {}
            pr_ax = pr_mov.get("axes") or {}
            for ax, sign in gt_ax.items():
                if pr_ax.get(ax) != sign:
                    mov_ok = False
                    mov_notes.append(f"axis {ax}: pred {pr_ax.get(ax)} vs gt {sign}")
            for ax in pr_ax:
                if ax not in gt_ax:
                    mov_notes.append(f"extra axis {ax}")
            if not _match_repetition(pr_mov.get("repetition"), gt_mov.get("repetition")):
                mov_ok = False
                mov_notes.append(
                    f"rep {pr_mov.get('repetition')} vs {gt_mov.get('repetition')}"
                )
            if pr_mov.get("hold") is not None and gt_mov.get("hold") is not None:
                if bool(pr_mov.get("hold")) != bool(gt_mov.get("hold")):
                    mov_ok = False
                    mov_notes.append(f"hold {pr_mov.get('hold')} vs {gt_mov.get('hold')}")
            if gt_mov.get("joint") and pr_mov.get("joint") != gt_mov.get("joint"):
                mov_notes.append(
                    f"joint pred {pr_mov.get('joint')} vs gt {gt_mov.get('joint')} (soft)"
                )
    else:
        mov_ok = not gt_mov
        mov_notes.append("no GT movement")

    return {
        "cue": gt_row["cue"],
        "cue_idx": gt_row.get("cue_idx"),
        "pose_gt": gt_row.get("pose_gt"),
        "movement_gt": ann,
        "pose_ok": pose_ok,
        "movement_ok": mov_ok,
        "both_ok": pose_ok and mov_ok,
        "pose_notes": pose_notes,
        "movement_notes": mov_notes,
        "prediction": pred,
    }


def generate_one(*, cue: str, gt_row: dict[str, Any], model: str, prompt_path: Path) -> dict[str, Any]:
    template = prompt_path.read_text(encoding="utf-8")
    prompt = template.replace("{{CUE_NAME}}", cue)
    raw = _llm_generate(prompt, model=model)
    pred = _extract_json(raw)
    errs = _validate_pred(pred, cue)
    if errs:
        raise ValueError(f"validation: {errs}")
    pred["idx"] = gt_row.get("cue_idx")
    pred["model"] = model
    pred["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pred["experiment"] = "exp14"
    pred["raw_response"] = raw
    return pred


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp14 core attribute elicitation")
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--prompt", type=Path, default=PROMPT_EXP14)
    ap.add_argument(
        "--cues",
        default="uncertainty_wave,nod_yes,stop_palm_out,beckon_come_here,shrug_idk",
        help="Comma-separated cue names",
    )
    ap.add_argument("--out-json", type=Path, default=OUT_DIR / "result_exp14_gemini-2.5-pro.json")
    ap.add_argument("--score-json", type=Path, default=SCORE_DIR / "score_exp14_gemini-2.5-pro.json")
    ap.add_argument("--delay", type=float, default=2.0)
    args = ap.parse_args()

    gt = load_gt_by_cue()
    cues = [c.strip() for c in args.cues.split(",") if c.strip()]
    results: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []

    for cue in cues:
        row = gt.get(cue)
        if not row:
            print(f"[skip] no GT for {cue}", flush=True)
            continue
        print(f"[exp14] {cue} ...", flush=True)
        try:
            pred = generate_one(cue=cue, gt_row=row, model=args.model, prompt_path=args.prompt)
            sc = _score_row(pred, row)
            results.append(pred)
            scores.append(sc)
            flag = "OK" if sc["both_ok"] else "MISS"
            print(
                f"  [{flag}] pose={sc['pose_ok']} mov={sc['movement_ok']} | "
                f"GT mov={sc['movement_gt']}",
                flush=True,
            )
        except Exception as exc:
            print(f"  [FAIL] {exc}", flush=True)
            scores.append({"cue": cue, "error": str(exc)})
        time.sleep(args.delay)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    n = sum(1 for s in scores if s.get("both_ok"))
    n_pose = sum(1 for s in scores if s.get("pose_ok"))
    n_mov = sum(1 for s in scores if s.get("movement_ok"))
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "n": len([s for s in scores if "error" not in s]),
        "n_pose_ok": n_pose,
        "n_movement_ok": n_mov,
        "n_both_ok": n,
        "rows": scores,
    }
    args.score_json.parent.mkdir(parents=True, exist_ok=True)
    args.score_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nWrote {args.out_json}", flush=True)
    print(f"Wrote {args.score_json}", flush=True)
    print(f"Summary: pose {n_pose}/{payload['n']}  movement {n_mov}/{payload['n']}  both {n}/{payload['n']}", flush=True)


if __name__ == "__main__":
    main()
