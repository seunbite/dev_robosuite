#!/usr/bin/env python3
"""Contrastive Roomba Motion Programs from nav-eval situations (not cue names).

For each pair: same scene + A vs B meanings. Generate a program that conveys
TARGET and would not be confused with the FOIL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "adhoc/generation/robotarm"))

from motion_code_to_config import convert_code  # noqa: E402
from run_embodiment_exp0_gemini import (  # noqa: E402
    embodiment_schema_yaml,
    extract_motion_code,
    gemini_generate,
    _extract_json_obj,
)

EVAL_JSON = (
    REPO
    / "papers/icra2027_motion_abstraction/roomba_nav_pair_sample/roomba_nav_eval.json"
)
OUT_JSON = REPO / "data/results/motion_configs/roomba_nav_contrastive_gemini"
OUT_RENDER = REPO / "data/results/render/roomba_nav_contrastive_gemini/roomba/gif"


def _pair_questions(pair: dict[str, Any]) -> list[dict[str, Any]]:
    if "question" in pair:
        return [pair["question"]]
    return list(pair["questions"])


def _meaning_block(questions: list[dict[str, Any]], which: str) -> str:
    key = "opt1" if which == "A" else "opt2"
    lines = []
    for q in questions:
        kind = {
            "intent": "what the Roomba means",
            "you": "what the person should do",
            "robot": "what the Roomba will do",
        }[q["kind"]]
        lines.append(f"- {kind}: {q['prompt']} → {q[key]}")
    return "\n".join(lines)


def build_contrastive_prompt(
    *,
    scene: str,
    questions: list[dict[str, Any]],
    target: str,
    schema: str,
) -> str:
    foil = "B" if target == "A" else "A"
    return f"""You are designing a Roomba-style mobile-base gesture with Motion Code.

A person and a Roomba share this situation:
{scene}

Two readings are possible. You must convey reading {target} so a viewer would NOT
choose reading {foil}.

[Reading A]
{_meaning_block(questions, "A")}

[Reading B]
{_meaning_block(questions, "B")}

[Task]
Write ONE Motion Code that communicates Reading {target}.
Use only Roomba primitives in the schema (pose facing, path line/arc, hold).
No arms, hands, head joints, or invented attributes.
Do not name a cue. Do not copy a stock emblem label.
Make the path class, timing, and staging distinctive vs the other reading
(e.g. rock vs yaw vs advance-and-hold vs retreat vs in-place loop).
Prefer 2–6 steps joined by +. pose(...) is optional on Roomba; path/hold may start.

[EMBODIMENT SCHEMA — roomba]
{schema}

[Output]
ONE Motion Code line only (steps joined by +). No comments, no JSON, no markdown.
""".strip() + "\n"


def _gif_name(pair_id: int, ab: str, cue: str) -> str:
    return f"p{pair_id:02d}_{ab}_{cue}.gif"


def generate_all(*, model: str, render: bool, retries: int) -> dict[str, Any]:
    spec = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    schema = embodiment_schema_yaml("roomba")
    OUT_JSON.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for pair in spec["pairs"]:
        for ab, side in (("A", pair["a"]), ("B", pair["b"])):
            prompt = build_contrastive_prompt(
                scene=pair["scene"],
                questions=_pair_questions(pair),
                target=ab,
                schema=schema,
            )
            raw = ""
            code = ""
            converted: dict[str, Any] = {"movements": [], "parse_ok": False, "notes": []}
            last_err = None
            for attempt in range(retries + 1):
                try:
                    raw = gemini_generate(prompt, model=model)
                    obj = _extract_json_obj(raw)
                    code = extract_motion_code(raw, obj) or ""
                    converted = convert_code(code, platform="roomba")
                    if converted.get("parse_ok") and converted.get("movements"):
                        last_err = None
                        break
                    last_err = "empty parse"
                except Exception as exc:  # noqa: BLE001
                    last_err = str(exc)
                    time.sleep(1.0)
            row = {
                "pair_id": int(pair["id"]),
                "arm": ab,
                "idx": int(side["idx"]),
                "cue": side["cue"],
                "foil_cue": pair["b" if ab == "A" else "a"]["cue"],
                "scene": pair["scene"],
                "target_reading": ab,
                "model": model,
                "time": datetime.now().isoformat(timespec="seconds"),
                "prompt": prompt,
                "raw_response": raw,
                "source_code": converted.get("source_code") or code,
                "movements": converted.get("movements") or [],
                "parse_ok": bool(converted.get("parse_ok") and converted.get("movements")),
                "notes": converted.get("notes") or [],
                "error": last_err,
                "platform": "roomba",
            }
            rows.append(row)
            print(
                f"[{ab}] p{pair['id']:02d} {side['cue']}: parse={row['parse_ok']} "
                f"code={(row['source_code'] or '')[:80]}",
                flush=True,
            )

    payload = {"model": model, "n": len(rows), "rows": rows}
    (OUT_JSON / "contrastive_programs.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if render:
        from embodiment_renderers import render_roomba  # noqa: E402

        OUT_RENDER.mkdir(parents=True, exist_ok=True)
        render_rows = []
        for row in rows:
            if not row["parse_ok"]:
                continue
            render_rows.append(
                {
                    "idx": int(row["pair_id"]) * 10 + (1 if row["arm"] == "A" else 2),
                    "cue": f"{row['arm'].lower()}_{row['cue']}",
                    "description": row["scene"],
                    "movements": row["movements"],
                    "source_code": row["source_code"],
                    "parse_ok": True,
                    "platform": "roomba",
                    "_pair_id": row["pair_id"],
                    "_arm": row["arm"],
                    "_cue": row["cue"],
                }
            )
        if render_rows:
            render_roomba(render_rows, OUT_RENDER, force=True, fps=8)
        for rr in render_rows:
            src = OUT_RENDER / f"emb_g{rr['idx']:02d}_{rr['cue']}.gif"
            dst = OUT_RENDER / _gif_name(rr["_pair_id"], rr["_arm"].lower(), rr["_cue"])
            if src.is_file():
                dst.write_bytes(src.read_bytes())
                print(f"[render] {dst.name}", flush=True)

    print(f"wrote {OUT_JSON / 'contrastive_programs.json'}", flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()
    generate_all(model=args.model, render=args.render, retries=args.retries)


if __name__ == "__main__":
    main()
