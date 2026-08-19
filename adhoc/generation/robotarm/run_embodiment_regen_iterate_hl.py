#!/usr/bin/env python3
"""Iterate regen on HL (humanoid/tiago) Human-NO cells: R2 and R3 from prior round video+code.

Measures Table-1 GenDiff vs GT at gemini / R1 / R2 / R3.

Example::

    source APIKEY.sh
    python adhoc/generation/robotarm/run_embodiment_regen_iterate_humanoid.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_HERE))

from motion_code_to_config import convert_code  # noqa: E402
from run_embodiment_exp0_gemini import render_pending_once  # noqa: E402
from run_embodiment_regen_gemini import (  # noqa: E402
    CUES_JSON,
    GEMINI_MASTER,
    _call_regen,
    _client,
    _extract_revised_code,
    _index_master,
    _parse_meta,
    build_regen_prompt,
)
from run_embodiment_exp0_gemini import _extract_json_obj  # noqa: E402


R1_DIR = (
    _REPO
    / "data/results/evaluation/embodiment_cue_id"
    / "results_gemini25pro_regen_human_no"
)
OUT_ROOT = (
    _REPO
    / "data/results/evaluation/embodiment_cue_id"
    / "results_gemini25pro_regen_hl_iterate"
)


PLATFORMS_DEFAULT = ("humanoid", "tiago")

def _load_r1(platforms: set[str] | None = None) -> dict[str, dict[str, Any]]:
    allow = platforms or set(PLATFORMS_DEFAULT)
    out: dict[str, dict[str, Any]] = {}
    for line in (R1_DIR / "predictions.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("platform") not in allow or r.get("error"):
            continue
        out[r["job_id"]] = r
    return out


def _find_media(root: Path, platform: str, idx: int, cue: str) -> Path:
    """Regen renders often put mp4 next to gif under gif/ (not mp4/)."""
    name = f"emb_g{idx:02d}_{cue}.mp4"
    candidates = [
        root / "render" / platform / "gif" / name,
        root / "render" / platform / "mp4" / name,
        root / platform / "gif" / name,
        root / platform / "mp4" / name,
    ]
    # tiago sometimes under gif/ without render/
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


def _r1_mp4(platform: str, idx: int, cue: str) -> Path:
    return _find_media(R1_DIR, platform, idx, cue)


def _round_mp4(round_dir: Path, platform: str, idx: int, cue: str) -> Path:
    return _find_media(round_dir, platform, idx, cue)


def build_iter_prompt(
    *,
    platform: str,
    cue: str,
    description: str,
    motion_code: str,
    round_n: int,
) -> str:
    """Same as default regen, plus a light iterative nudge."""
    base = build_regen_prompt(
        platform=platform,
        cue=cue,
        description=description,
        motion_code=motion_code,
        strategy="default",
    )
    nudge = f"""
[ITERATION]
This is revision round {round_n} (of up to 3). A prior revision already ran; the attached
video is that revised render, and [CURRENT MOTION CODE] is the revised code.
Human judges still did not accept the original Gemini video; refine further so the
TARGET cue is clearer. Prefer small, targeted fixes to remaining mismatches over a
full rewrite — keep what already reads correctly.
""".strip()
    # insert nudge after Task block
    return base.replace("[REVISION STRATEGY]", nudge + "\n\n[REVISION STRATEGY]")


def run_round(
    *,
    round_n: int,
    prev_rows: dict[str, dict[str, Any]],
    prev_video_fn,
    out_dir: Path,
    model: str,
    workers: int,
    retries: int,
    sleep: float,
    skip_render: bool,
) -> dict[str, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    master_out = out_dir / "regen_motion_code_all.json"
    render_root = out_dir / "render"

    done: dict[str, dict[str, Any]] = {}
    if pred_path.exists():
        for line in pred_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["job_id"]] = r

    jobs = [jid for jid in sorted(prev_rows) if jid not in done]
    print(
        f"[R{round_n}] resume={len(done)} pending={len(jobs)}/{len(prev_rows)} model={model}",
        flush=True,
    )
    client = _client() if jobs else None

    def work(job_id: str) -> dict[str, Any]:
        prev = prev_rows[job_id]
        plat = prev["platform"]
        idx = int(prev["idx"])
        cue = prev["cue"]
        desc = prev.get("description") or ""
        old_code = str(prev.get("new_code") or prev.get("old_code") or "").strip()
        video = prev_video_fn(plat, idx, cue)
        if not video.is_file():
            return {
                **{k: prev[k] for k in ("platform", "idx", "cue", "description") if k in prev},
                "job_id": job_id,
                "old_code": old_code,
                "error": f"missing_video:{video}",
                "round": round_n,
            }
        prompt = build_iter_prompt(
            platform=plat,
            cue=cue,
            description=desc,
            motion_code=old_code,
            round_n=round_n,
        )
        last_err = None
        for attempt in range(retries + 1):
            try:
                raw, usage = _call_regen(
                    client=client, model=model, prompt=prompt, video=video
                )
                meta = _parse_meta(raw)
                obj = _extract_json_obj(raw)
                new_code = _extract_revised_code(raw, obj)
                converted = (
                    convert_code(new_code, platform=plat)
                    if new_code
                    else {
                        "movements": [],
                        "parse_ok": False,
                        "notes": ["empty_extract"],
                        "source_code": "",
                    }
                )
                if sleep > 0:
                    time.sleep(sleep)
                return {
                    "platform": plat,
                    "idx": idx,
                    "cue": cue,
                    "description": desc,
                    "job_id": job_id,
                    "old_code": old_code,
                    "new_code": converted.get("source_code") or new_code,
                    "movements": converted.get("movements") or [],
                    "parse_ok": bool(converted.get("parse_ok")),
                    "notes": converted.get("notes") or [],
                    "meta": meta,
                    "raw_output": raw,
                    "usage": usage,
                    "model": model,
                    "strategy": f"iterate_r{round_n}",
                    "round": round_n,
                    "mp4": str(video.relative_to(_REPO)),
                    "prev_round": round_n - 1,
                }
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                time.sleep(min(2**attempt, 20))
        return {
            "platform": plat,
            "idx": idx,
            "cue": cue,
            "job_id": job_id,
            "old_code": old_code,
            "error": last_err or "unknown",
            "round": round_n,
        }

    completed = 0
    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = {pool.submit(work, jid): jid for jid in jobs}
            for fut in as_completed(futs):
                row = fut.result()
                with pred_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[row["job_id"]] = row
                completed += 1
                if completed % 3 == 0 or completed == len(jobs):
                    print(
                        f"[R{round_n}] {len(done)}/{len(prev_rows)} (+{completed}/{len(jobs)})",
                        flush=True,
                    )

    # master for render
    by_cue: dict[int, dict[str, Any]] = {}
    for jid in sorted(prev_rows):
        row = done.get(jid) or {"job_id": jid, "error": "missing"}
        if row.get("error") or not row.get("parse_ok"):
            continue
        idx = int(row["idx"])
        entry = by_cue.setdefault(
            idx,
            {
                "idx": idx,
                "cue": row["cue"],
                "category": "",
                "description": row.get("description") or "",
                "platforms": {},
            },
        )
        plat = row["platform"]
        domain = {"humanoid": "unitree_g1", "tiago": "tiago"}.get(plat, plat)
        entry["platforms"][plat] = {
            "idx": idx,
            "cue": row["cue"],
            "description": row.get("description") or "",
            "movements": row.get("movements") or [],
            "state": f"motion_code_regen_r{round_n}",
            "model": model,
            "time": datetime.now().isoformat(timespec="seconds"),
            "experiment": f"embodiment_regen_hl_iterate_r{round_n}",
            "generation_valid": bool(row.get("movements")),
            "platform": plat,
            "domain": domain,
            "parse_ok": True,
            "notes": row.get("notes") or [],
            "source_code": row.get("new_code") or "",
        }
    configs = [by_cue[i] for i in sorted(by_cue)]
    master_out.write_text(json.dumps(configs, indent=2, ensure_ascii=False) + "\n")

    if not skip_render and configs:
        print(f"[R{round_n}] render…", flush=True)
        render_pending_once._skip = set()  # type: ignore[attr-defined]
        for _ in range(40):
            n = render_pending_once(
                configs,
                render_root=render_root,
                out_json=master_out,
                platforms=sorted({r["platform"] for r in done.values() if not r.get("error")}),
                force=False,
                max_items=6,
            )
            if n == 0:
                break
            print(f"[R{round_n}] render attempted {n}", flush=True)

    return done


def score_gendiff(code_by_job: dict[str, str], gt_index, label: str) -> dict[str, Any]:
    # import helpers from gendiff_core by loading functions
    import copy
    import re
    from collections import Counter

    from score_embodiment_h4 import _align

    DYN = {"movement", "path", "move", "hold"}
    TOL = 0.5
    CONTINUOUS = {"speed", "hold_time", "hold", "t"}

    def leaf(key):
        k = key.split(".")[-1]
        k = re.sub(r"\[\d+\]", "", k)
        return k.split("#")[0].lower()

    def fnum(v):
        try:
            return float(v)
        except Exception:
            return None

    def flatten_raw(value, prefix=""):
        out = {}
        if isinstance(value, dict):
            for key in sorted(value):
                child = f"{prefix}.{key}" if prefix else key
                out.update(flatten_raw(value[key], child))
        elif isinstance(value, list):
            if all(not isinstance(item, (dict, list)) for item in value):
                out[prefix] = tuple(value)
            else:
                for i, item in enumerate(value):
                    out.update(flatten_raw(item, f"{prefix}[{i}]"))
        else:
            lf = leaf(prefix) if prefix else ""
            if lf in ("repetition", "rep"):
                try:
                    out[prefix] = "single" if float(value) <= 1 else "plural"
                except Exception:
                    out[prefix] = str(value).lower()
            elif isinstance(value, str):
                out[prefix] = value.strip().lower()
            elif isinstance(value, float):
                out[prefix] = round(value, 6)
            else:
                out[prefix] = value
        return out

    def components_raw(row):
        result = []
        for movement in row.get("movements") or []:
            kind = str(movement.get("type", "unknown")).lower()
            if kind == "move":
                kind = "movement"
            result.append({"type": kind, "attrs": flatten_raw(movement.get("parameters") or {})})
        return result

    def is_outer_hold(mov):
        t = str(mov.get("type") or "").lower()
        params = mov.get("parameters") or {}
        if t == "hold":
            return True
        if set(params.keys()) <= {"t", "hold_time", "hold"} and any(
            k in params for k in ("t", "hold_time", "hold")
        ):
            return True
        path = None
        if t == "path":
            path = params.get("path") if isinstance(params.get("path"), dict) else params
        elif isinstance(params.get("path"), dict):
            path = params["path"]
        if isinstance(path, dict):
            ht = path.get("hold_time", params.get("hold_time"))
            dist = path.get("distance")
            try:
                if ht is not None and dist is not None and float(dist) == 0.0:
                    return True
            except Exception:
                pass
            axes = [path[k] for k in ("x", "y", "z") if k in path]
            try:
                if ht is not None and axes and all(float(a) == 0.0 for a in axes):
                    return True
            except Exception:
                pass
        return False

    def outer_hold_value(mov):
        params = mov.get("parameters") or {}
        for k in ("hold_time", "t", "hold"):
            if fnum(params.get(k)) is not None:
                return fnum(params[k])
        path = params.get("path") if isinstance(params.get("path"), dict) else None
        if path and fnum(path.get("hold_time")) is not None:
            return fnum(path["hold_time"])
        if fnum(mov.get("duration")) is not None:
            return fnum(mov["duration"])
        return 0.0

    def get_inside_hold(mov):
        params = mov.get("parameters") or {}
        if fnum(params.get("hold_time")) is not None:
            return fnum(params["hold_time"])
        path = params.get("path")
        if isinstance(path, dict) and fnum(path.get("hold_time")) is not None:
            return fnum(path["hold_time"])
        movement = params.get("movement")
        if isinstance(movement, dict):
            if fnum(movement.get("hold_time")) is not None:
                return fnum(movement["hold_time"])
            dirs = movement.get("directions")
            if isinstance(dirs, list):
                vals = [fnum(d.get("hold_time")) for d in dirs if isinstance(d, dict)]
                vals = [v for v in vals if v is not None]
                if vals:
                    return sum(vals)
        return 0.0

    def add_hold(mov, val):
        params = mov.setdefault("parameters", {})
        t = str(mov.get("type") or "").lower()
        new = get_inside_hold(mov) + val
        if t == "pose" or isinstance(params.get("pose"), dict):
            params["hold_time"] = new
            return
        if t == "path" or isinstance(params.get("path"), dict):
            path = params.setdefault("path", {})
            if isinstance(path, dict):
                path["hold_time"] = new
            else:
                params["hold_time"] = new
            return
        if t in ("movement", "move") or isinstance(params.get("movement"), dict):
            movement = params.setdefault("movement", {})
            if isinstance(movement, dict):
                movement["hold_time"] = new
            else:
                params["hold_time"] = new
            return
        params["hold_time"] = new

    def fold_cell(cell):
        cell = copy.deepcopy(cell)
        out = []
        for mov in cell.get("movements") or []:
            if is_outer_hold(mov):
                val = outer_hold_value(mov)
                if out:
                    add_hold(out[-1], val)
                else:
                    out.append({"type": "pose", "parameters": {"hold_time": val}})
                continue
            out.append(mov)
        cell["movements"] = out
        return cell

    def bag_attrs(comps):
        bag = {}
        counts = Counter()
        for c in comps:
            if c["type"] not in DYN:
                continue
            for k, v in c["attrs"].items():
                lf = leaf(k)
                n = counts[lf]
                counts[lf] += 1
                bag[lf if n == 0 else f"{lf}#{n}"] = v
        return bag

    def classify_union(ga, pa):
        keys = set(ga) | set(pa)
        four = Counter()
        for k in keys:
            lf = leaf(k)
            if lf in CONTINUOUS:
                gn = fnum(ga[k]) if k in ga else 0.0
                pn = fnum(pa[k]) if k in pa else 0.0
                if gn is None:
                    gn = 0.0
                if pn is None:
                    pn = 0.0
                out = "accurate" if abs(pn - gn) < TOL else "wrong"
            else:
                if k in ga and k in pa:
                    out = "accurate" if ga[k] == pa[k] else "wrong"
                elif k in ga:
                    out = "missing"
                else:
                    out = "extra"
            four[out] += 1
        return four

    def rates_of(four):
        n = sum(four.values())
        if n == 0:
            return {"accurate": 1.0, "wrong": 0.0, "missing": 0.0, "extra": 0.0, "diff": 0.0}
        r = {k: four[k] / n for k in ("accurate", "wrong", "missing", "extra")}
        r["diff"] = 1 - r["accurate"]
        return r

    def score_pair(gt_cell, pred_cell):
        gf, pf = fold_cell(gt_cell), fold_cell(pred_cell)
        gc, pc = components_raw(gf), components_raw(pf)
        pairs, _, _ = _align(gc, pc)
        four = Counter()
        for i, j in pairs:
            if gc[i]["type"] != "pose" or pc[j]["type"] != "pose":
                continue
            four.update(classify_union(gc[i]["attrs"], pc[j]["attrs"]))
        static = rates_of(four) if four else None
        dyn = rates_of(classify_union(bag_attrs(gc), bag_attrs(pc)))
        return static, dyn

    statics, dyns = [], []
    per = []
    for jid, code in code_by_job.items():
        plat, _, g = jid.partition(".")
        idx = int(g[1:3])
        gt = gt_index.get((idx, plat))
        if not gt:
            continue
        converted = convert_code(code, platform=plat)
        pred_cell = {
            "movements": converted.get("movements") or [],
            "source_code": converted.get("source_code") or code,
        }
        st, dy = score_pair(gt, pred_cell)
        if st:
            statics.append(st)
        dyns.append(dy)
        per.append(
            {
                "job_id": jid,
                "idx": idx,
                "static_acc": None if not st else 100 * st["accurate"],
                "dynamic_acc": 100 * dy["accurate"],
            }
        )

    def avg(rs, key):
        return 100 * sum(r[key] for r in rs) / len(rs) if rs else float("nan")

    return {
        "label": label,
        "n": len(dyns),
        "Static": {
            "n": len(statics),
            "accurate": round(avg(statics, "accurate"), 4),
            "diff": round(avg(statics, "diff"), 4),
            "wrong": round(avg(statics, "wrong"), 4),
            "missing": round(avg(statics, "missing"), 4),
            "extra": round(avg(statics, "extra"), 4),
        },
        "Dynamic": {
            "n": len(dyns),
            "accurate": round(avg(dyns, "accurate"), 4),
            "diff": round(avg(dyns, "diff"), 4),
            "wrong": round(avg(dyns, "wrong"), 4),
            "missing": round(avg(dyns, "missing"), 4),
            "extra": round(avg(dyns, "extra"), 4),
        },
        "per_cell": per,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--platforms", default="humanoid,tiago")
    args = ap.parse_args()

    allow = {p.strip() for p in args.platforms.split(",") if p.strip()}
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    r1 = _load_r1(allow)
    print(f"HL R1 cells: {len(r1)} platforms={sorted(allow)}", flush=True)
    if len(r1) == 0:
        raise SystemExit("no R1 predictions")

    # ensure cues/descriptions filled
    cues = {int(c["idx"]): c for c in json.loads(CUES_JSON.read_text())}
    gem_idx = _index_master(json.loads(GEMINI_MASTER.read_text()))
    for jid, r in r1.items():
        idx = int(r["idx"])
        if not r.get("description"):
            r["description"] = str(cues.get(idx, {}).get("description") or "")
        if not r.get("cue"):
            info = gem_idx.get((idx, r.get("platform") or "humanoid")) or {}
            r["cue"] = info.get("cue") or r.get("cue")

    r2_dir = OUT_ROOT / "r2"
    r3_dir = OUT_ROOT / "r3"

    if not args.score_only:
        r2 = run_round(
            round_n=2,
            prev_rows=r1,
            prev_video_fn=lambda plat, idx, cue: _r1_mp4(plat, idx, cue),
            out_dir=r2_dir,
            model=args.model,
            workers=args.workers,
            retries=args.retries,
            sleep=args.sleep,
            skip_render=args.skip_render,
        )
        # check R2 videos exist; if missing, fail soft
        missing = [
            jid
            for jid, r in r2.items()
            if not r.get("error")
            and not _round_mp4(r2_dir, r["platform"], int(r["idx"]), r["cue"]).is_file()
        ]
        if missing and not args.skip_render:
            print(f"[warn] R2 missing mp4 for {len(missing)} cells; will still try R3 with available", flush=True)

        def r2_video(plat: str, idx: int, cue: str) -> Path:
            p = _round_mp4(r2_dir, plat, idx, cue)
            if p.is_file():
                return p
            return _r1_mp4(plat, idx, cue)

        r3 = run_round(
            round_n=3,
            prev_rows={jid: r for jid, r in r2.items() if not r.get("error") and r.get("parse_ok")},
            prev_video_fn=r2_video,
            out_dir=r3_dir,
            model=args.model,
            workers=args.workers,
            retries=args.retries,
            sleep=args.sleep,
            skip_render=args.skip_render,
        )
    else:
        r2 = {}
        r3 = {}
        for line in (r2_dir / "predictions.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r2[r["job_id"]] = r
        for line in (r3_dir / "predictions.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r3[r["job_id"]] = r

    # score
    from pathlib import Path as P

    GT = json.loads(
        (
            _REPO
            / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
        ).read_text()
    )
    gt_index = {}
    for row in GT:
        for p, cell in (row.get("platforms") or {}).items():
            gt_index[(int(row["idx"]), p)] = cell

    gem_codes = {}
    for jid, r in r1.items():
        idx = int(r["idx"])
        plat = r["platform"]
        cell = gem_idx.get((idx, plat), {}).get("cell") or {}
        gem_codes[jid] = str(cell.get("source_code") or r.get("old_code") or "")

    r1_codes = {jid: str(r.get("new_code") or "") for jid, r in r1.items()}
    r2_codes = {
        jid: str(r.get("new_code") or "")
        for jid, r in r2.items()
        if not r.get("error") and r.get("parse_ok")
    }
    r3_codes = {
        jid: str(r.get("new_code") or "")
        for jid, r in r3.items()
        if not r.get("error") and r.get("parse_ok")
    }

    # align to common keys present in all rounds
    common = set(gem_codes) & set(r1_codes) & set(r2_codes) & set(r3_codes)
    if not common:
        common = set(gem_codes) & set(r1_codes)
        print(f"[score] no full common set; scoring available rounds on {len(common)}", flush=True)

    report = {
        "platforms": sorted(allow),
        "n_r1": len(r1),
        "n_r2": len(r2_codes),
        "n_r3": len(r3_codes),
        "n_common": len(common),
        "rounds": {},
    }
    for label, codes in [
        ("gemini", gem_codes),
        ("r1", r1_codes),
        ("r2", r2_codes),
        ("r3", r3_codes),
    ]:
        use = {k: codes[k] for k in common if k in codes} if common else codes
        if not use:
            continue
        report["rounds"][label] = score_gendiff(use, gt_index, label)
        s = report["rounds"][label]["Static"]
        d = report["rounds"][label]["Dynamic"]
        print(
            f"[score] {label:6} n={report['rounds'][label]['n']}  "
            f"Static acc={s['accurate']:.1f} (diff {s['diff']:.1f})  "
            f"Dynamic acc={d['accurate']:.1f} (diff {d['diff']:.1f})",
            flush=True,
        )

    out_json = OUT_ROOT / "gendiff_iterate_r1_r3.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
