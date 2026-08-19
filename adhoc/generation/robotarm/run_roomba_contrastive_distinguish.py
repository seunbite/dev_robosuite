#!/usr/bin/env python3
"""Roomba contrastive generation: distinguish target cue from a named foil.

Runs all four interfaces (Motion Program, Joint, Cartesian, Trajectory) on ~5 pairs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from google import genai
from google.genai import types

from embodiment_renderers import render_roomba  # noqa: E402
from motion_code_to_config import convert_code  # noqa: E402
import generate_roomba_sophisticated_fewshot_pilot as mp_pilot  # noqa: E402
import run_embodiment_baseline_traj as baseline  # noqa: E402
import run_drag_draw_gemini as drag  # noqa: E402
from drag_draw import program_to_xyzrpy  # noqa: E402

CUES_PATH = REPO / "data/seed/agent_fill/embodiment_short/cues.json"
MP_SHOTS = REPO / "data/results/shots/embodiment_unified/roomba_sophisticated_cues_yml_5.json"
BASE_SHOTS = REPO / "data/seed/shots/embodiment_unified/roomba_sophisticated_baseline_shots.yaml"
TRAJ_SHOTS = REPO / "data/seed/shots/drag_draw/roomba_sophisticated_fewshots.yaml"
OUT = REPO / "data/results/roomba_contrastive_distinguish"

# target must look unlike foil
PAIRS = [
    {"target_idx": 15, "foil_idx": 20},  # highfive_invite ≠ hands_up_surrender
    {"target_idx": 20, "foil_idx": 15},  # hands_up_surrender ≠ highfive_invite
    {"target_idx": 22, "foil_idx": 5},  # raise_fists_victory ≠ fist_pump
    {"target_idx": 21, "foil_idx": 20},  # stretch_arms_up ≠ hands_up_surrender
    {"target_idx": 5, "foil_idx": 22},  # fist_pump ≠ raise_fists_victory
]


def load_cues() -> dict[int, dict[str, Any]]:
    return {int(r["idx"]): r for r in json.loads(CUES_PATH.read_text(encoding="utf-8"))}


def contrastive_block(target: dict[str, Any], foil: dict[str, Any]) -> str:
    return (
        f"Contrastive constraint: make the motion for \"{target['cue']}\" clearly "
        f"distinguishable from the foil cue \"{foil['cue']}\" "
        f"({foil.get('description') or ''}). "
        "A human viewer should not confuse your output with that foil. "
        "Emphasize the distinctive timing, path shape, and staging of the target cue; "
        "do not produce a generic tip-up / small-arc that could read as either cue."
    )


def mp_prompt(shots: list[dict[str, Any]], target: dict[str, Any], foil: dict[str, Any]) -> str:
    base = mp_pilot._prompt(shots, target)
    return base.replace(
        "- Return exactly ONE Motion Code expression, no prose or markdown.",
        contrastive_block(target, foil)
        + "\n- Return exactly ONE Motion Code expression, no prose or markdown.",
    )


def run_motion_program(client: Any, model: str, pairs: list[dict], cues: dict[int, dict]) -> list[dict]:
    shots = json.loads(MP_SHOTS.read_text(encoding="utf-8"))
    out_dir = OUT / "motion_program"
    render_dir = out_dir / "render"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in pairs:
        t = cues[pair["target_idx"]]
        f = cues[pair["foil_idx"]]
        target = {
            "idx": int(t["idx"]),
            "cue": t["cue"],
            "description": t.get("description") or "",
        }
        prompt = mp_prompt(shots, target, f)
        resp = client.models.generate_content(model=model, contents=prompt)
        raw = (resp.text or "").strip()
        code = mp_pilot._clean_code(raw)
        converted = convert_code(code, platform="roomba")
        row = {
            **target,
            "foil_idx": int(f["idx"]),
            "foil": f["cue"],
            "method": "motion_program",
            "motion_code": code,
            "raw_output": raw,
            "parse_ok": bool(converted.get("parse_ok")),
            "prompt": prompt,
        }
        if not row["parse_ok"]:
            row["error"] = str(converted.get("error") or "parse failed")
            rows.append(row)
            print(f"[mp] {target['cue']} vs {f['cue']}: PARSE FAIL", flush=True)
            continue
        render_row = {
            "idx": target["idx"],
            "cue": target["cue"],
            "description": target["description"],
            "movements": converted["movements"],
            "source_code": code,
            "parse_ok": True,
            "platform": "roomba",
        }
        try:
            render_roomba([render_row], render_dir, force=True, fps=8)
            gif = render_dir / f"emb_g{target['idx']:02d}_{target['cue']}.gif"
            # also save foil-tagged name
            tagged = render_dir / f"emb_g{target['idx']:02d}_{target['cue']}__vs_{f['cue']}.gif"
            if gif.is_file():
                tagged.write_bytes(gif.read_bytes())
            row["render_ok"] = tagged.is_file()
            row["render_gif"] = str(tagged)
        except Exception as exc:  # noqa: BLE001
            row["render_ok"] = False
            row["error"] = str(exc)
        rows.append(row)
        print(
            f"[mp] {target['cue']} vs {f['cue']}: parse={row['parse_ok']} "
            f"render={row.get('render_ok')}",
            flush=True,
        )
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    return rows


def run_baseline_mode(
    client: Any,
    model: str,
    mode: str,
    pairs: list[dict],
    cues: dict[int, dict],
) -> list[dict]:
    out_dir = OUT / f"baseline_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = baseline._index_gt()
    shots = baseline._fewshot_artifacts(out_dir, "roomba", gt, shots_path=BASE_SHOTS)
    pred_path = out_dir / f"predictions_{mode}.jsonl"
    if pred_path.exists():
        pred_path.unlink()
    rows = []
    for pair in pairs:
        t = cues[pair["target_idx"]]
        f = cues[pair["foil_idx"]]
        idx = int(t["idx"])
        cell = gt[(idx, "roomba")]
        desc = (cell.get("description") or "") + "\n\n" + contrastive_block(t, f)
        prompt = baseline._build_prompt(
            mode,  # type: ignore[arg-type]
            plat="roomba",
            target={"cue": t["cue"], "description": desc},
            shots=shots,
        )
        job_id = f"{mode}.roomba.g{idx:02d}.vs{int(f['idx']):02d}"
        last_err = None
        row: dict[str, Any] = {
            "job_id": job_id,
            "mode": mode,
            "platform": "roomba",
            "idx": idx,
            "cue": t["cue"],
            "foil_idx": int(f["idx"]),
            "foil": f["cue"],
            "method": mode,
            "description": desc,
            "prompt": prompt,
            "model": model,
        }
        for attempt in range(3):
            try:
                cfg = {
                    "temperature": 0.3,
                    "max_output_tokens": 16384,
                    "thinking_config": types.ThinkingConfig(thinking_budget=128),
                }
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**cfg),
                )
                raw = (getattr(resp, "text", None) or "").strip()
                schema_joints = shots[0]["joint_names"]
                schema_links = shots[0].get("link_names") or ["end_effector"]
                payload = baseline._validate(
                    mode,  # type: ignore[arg-type]
                    baseline._extract_json(raw),
                    cue=t["cue"],
                    joint_names=schema_joints,
                    link_names=schema_links,
                    plat="roomba",
                )
                row.update({"payload": payload, "raw_output": raw, "parse_ok": True})
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                time.sleep(min(2**attempt, 10))
        if last_err and not row.get("parse_ok"):
            row.update({"parse_ok": False, "error": last_err})
            print(f"[{mode}] {t['cue']} vs {f['cue']}: FAIL {last_err}", flush=True)
        else:
            print(f"[{mode}] {t['cue']} vs {f['cue']}: parse=PASS", flush=True)
        with pred_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)

    # render via baseline renderer
    render_dir = out_dir / f"render_{mode}" / "roomba"
    render_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if not row.get("parse_ok"):
            continue
        idx = int(row["idx"])
        cue = row["cue"]
        foil = row["foil"]
        gif = render_dir / f"emb_g{idx:02d}_{cue}__vs_{foil}_gemini-2.5-pro.gif"
        try:
            payload = json.loads(json.dumps(row["payload"]))
            payload["skip_capture"] = False
            baseline.RENDERERS["roomba"](
                payload,
                "xyzrpy" if mode == "xyzrpy" else "joint",
                gif,
                payload,
            )
            row["render_ok"] = gif.is_file()
            row["render_gif"] = str(gif)
        except Exception as exc:  # noqa: BLE001
            row["render_ok"] = False
            row["error"] = str(exc)
            print(f"[{mode}] render fail {cue}: {exc}", flush=True)
        print(f"[{mode}] render {cue} vs {foil}: {row.get('render_ok')}", flush=True)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    return rows


def run_trajectory(client: Any, model: str, pairs: list[dict], cues: dict[int, dict]) -> list[dict]:
    import yaml

    out_dir = OUT / "trajectory"
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = drag.load_schema()
    fewshots = yaml.safe_load(TRAJ_SHOTS.read_text(encoding="utf-8"))
    pred_path = out_dir / "predictions.jsonl"
    if pred_path.exists():
        pred_path.unlink()
    rows = []
    for pair in pairs:
        t = dict(cues[pair["target_idx"]])
        f = cues[pair["foil_idx"]]
        t["description"] = (t.get("description") or "") + "\n\n" + contrastive_block(t, f)
        prompt = drag.build_prompt("roomba", t, fewshots)
        # append contrastive again at end for emphasis
        prompt = prompt.replace(
            "Return exactly one valid YAML program and nothing else.",
            contrastive_block(t, f) + "\n\nReturn exactly one valid YAML program and nothing else.",
        )
        job_id = f"drag_draw.roomba.c{int(t['idx']):02d}.vs{int(f['idx']):02d}"
        row: dict[str, Any] = {
            "job_id": job_id,
            "platform": "roomba",
            "idx": int(t["idx"]),
            "cue": t["cue"],
            "foil_idx": int(f["idx"]),
            "foil": f["cue"],
            "method": "trajectory",
            "prompt": prompt,
            "model": model,
        }
        last_err = None
        for attempt in range(3):
            try:
                cfg = {
                    "temperature": 0.3,
                    "max_output_tokens": 8192,
                    "thinking_config": types.ThinkingConfig(thinking_budget=128),
                }
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**cfg),
                )
                raw = drag.extract_text(resp)
                obj = drag.extract_yaml(raw)
                program = drag.validate_target(obj, platform="roomba", cue=t, schema=schema)
                row.update({"program": program, "raw_output": raw, "parse_ok": True})
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                time.sleep(min(2**attempt, 10))
        if last_err and not row.get("parse_ok"):
            row.update({"parse_ok": False, "error": last_err})
            print(f"[traj] {t['cue']} vs {f['cue']}: FAIL {last_err}", flush=True)
        else:
            print(f"[traj] {t['cue']} vs {f['cue']}: parse=PASS", flush=True)
        with pred_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)

    render_dir = out_dir / "render" / "roomba"
    render_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if not row.get("parse_ok"):
            continue
        idx = int(row["idx"])
        cue = row["cue"]
        foil = row["foil"]
        gif = render_dir / f"c{idx:02d}_{cue}__vs_{foil}_gemini-2.5-pro.gif"
        try:
            payload = program_to_xyzrpy(row["program"])
            baseline.RENDERERS["roomba"](payload, "xyzrpy", gif, payload)
            row["render_ok"] = gif.is_file()
            row["render_gif"] = str(gif)
        except Exception as exc:  # noqa: BLE001
            row["render_ok"] = False
            row["error"] = str(exc)
            print(f"[traj] render fail {cue}: {exc}", flush=True)
        print(f"[traj] render {cue} vs {foil}: {row.get('render_ok')}", flush=True)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    return rows


def write_gallery(all_rows: dict[str, list[dict]]) -> Path:
    import html as H

    cards = []
    # group by target
    keys = [(p["target_idx"], p["foil_idx"]) for p in PAIRS]
    cues = load_cues()
    for ti, fi in keys:
        t, f = cues[ti], cues[fi]
        cells = []
        for method in ("motion_program", "joint", "xyzrpy", "trajectory"):
            rows = all_rows.get(method, [])
            hit = next((r for r in rows if int(r["idx"]) == ti and r.get("foil") == f["cue"]), None)
            if hit and hit.get("render_ok") and Path(hit.get("render_gif") or "").is_file():
                rel = os.path.relpath(hit["render_gif"], OUT)
                cells.append(
                    f"<div class='cell'><div class='m'>{H.escape(method)}</div>"
                    f"<img src='{H.escape(rel)}?v=1'></div>"
                )
            else:
                err = H.escape(str((hit or {}).get("error") or "missing"))
                cells.append(f"<div class='cell'><div class='m'>{H.escape(method)}</div><p class='fail'>{err}</p></div>")
        cards.append(
            f"<section><h2>{H.escape(t['cue'])} <span>≠</span> {H.escape(f['cue'])}</h2>"
            f"<div class='row'>{''.join(cells)}</div></section>"
        )
    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>Roomba contrastive distinguish</title>
<style>
body{{margin:24px;background:#111;color:#eee;font-family:system-ui}}
h1{{font-size:1.2rem}} h2{{font-size:1rem;margin:1.2rem 0 .4rem}} h2 span{{color:#f59e0b}}
.row{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}
.cell{{background:#1a1a1a;border:1px solid #333;padding:6px}}
.cell img{{width:100%;display:block;background:#000}}
.m{{font-size:.75rem;color:#9aa;margin-bottom:4px}}
.fail{{color:#f88;font-size:.8rem}}
</style></head><body>
<h1>Roomba contrastive generation — distinguish target from foil</h1>
<p>Each row: generate target while explicitly avoiding the foil cue. Methods: MP · Joint · Cartesian · Trajectory.</p>
{''.join(cards)}
</body></html>"""
    path = OUT / "gallery.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument(
        "--methods",
        default="motion_program,joint,xyzrpy,trajectory",
        help="Comma list of methods to run",
    )
    args = ap.parse_args()
    OUT = args.out_dir.expanduser().resolve()
    OUT.mkdir(parents=True, exist_ok=True)

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
    client = genai.Client(api_key=key)
    cues = load_cues()
    (OUT / "pairs.json").write_text(
        json.dumps(
            [
                {
                    **p,
                    "target": cues[p["target_idx"]]["cue"],
                    "foil": cues[p["foil_idx"]]["cue"],
                }
                for p in PAIRS
            ],
            indent=2,
        )
        + "\n"
    )

    methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    all_rows: dict[str, list[dict]] = {}
    if "motion_program" in methods:
        all_rows["motion_program"] = run_motion_program(client, args.model, PAIRS, cues)
    if "joint" in methods:
        all_rows["joint"] = run_baseline_mode(client, args.model, "joint", PAIRS, cues)
    if "xyzrpy" in methods:
        all_rows["xyzrpy"] = run_baseline_mode(client, args.model, "xyzrpy", PAIRS, cues)
    if "trajectory" in methods:
        all_rows["trajectory"] = run_trajectory(client, args.model, PAIRS, cues)

    gal = write_gallery(all_rows)
    print("DONE", gal, flush=True)


if __name__ == "__main__":
    main()
