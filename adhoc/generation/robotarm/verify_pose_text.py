#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _gemini_client():
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY (or GEMINI_API_KEY).")
    return genai.Client(api_key=api_key)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _first_pose(row: dict[str, Any]) -> dict[str, Any]:
    for step in row.get("movements", []):
        if step.get("type") == "pose":
            return step.get("parameters", {}).get("pose", {}) or {}
    return {}


def _extract_json(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        s = m.group(0)
    return json.loads(s)


def _fewshot_block(shots: list[dict[str, Any]], n: int = 4) -> str:
    out: list[str] = []
    picked = [r for r in shots if r.get("state") in ("handmade", "choreography")][:n]
    for i, r in enumerate(picked, 1):
        p = _first_pose(r)
        out.append(
            f"[EX{i}] cue={r.get('cue')} | pose(dir={p.get('dir')}, grip={p.get('gripper_orientation')})\n"
            f"description={r.get('description','')}"
        )
    return "\n\n".join(out)


def _prompt(cue_row: dict[str, Any], fewshot_text: str) -> str:
    from prompt_loader import fill_template  # noqa: WPS433

    p = _first_pose(cue_row)
    return fill_template(
        "exp03_pose_verify_text.txt",
        {
            "FEWSHOT": fewshot_text,
            "CUE": str(cue_row.get("cue", "")),
            "DESCRIPTION": str(cue_row.get("description", "")),
            "DIR": str(p.get("dir", "")),
            "GRIPPER_ORIENTATION": str(p.get("gripper_orientation", "")),
        },
    )


def run(args: argparse.Namespace) -> None:
    rows = _load_json(args.config_json)
    shots = _load_json(args.shots_json)
    fewshot_text = _fewshot_block(shots, n=args.fewshot_n)

    backend = (
        getattr(args, "vlm_backend", None)
        or os.getenv("VLM_BACKEND")
        or "transformers"
    ).lower()
    vlm = None
    client = None
    if backend == "gemini":
        client = _gemini_client()
    else:
        _here = Path(__file__).resolve().parent
        if str(_here) not in sys.path:
            sys.path.insert(0, str(_here))
        from vlm_client import (  # noqa: WPS433
            VLMClient,
            init_inprocess_engine,
            is_inprocess_backend,
            is_vllm_http_backend,
            require_vllm_server,
        )

        if is_vllm_http_backend(backend):
            require_vllm_server()
        elif is_inprocess_backend(backend):
            init_inprocess_engine(backend, args.model)
        vlm = VLMClient(backend=backend, model=args.model)

    out: list[dict[str, Any]] = []
    done_idx: set[int] = set()
    if getattr(args, "resume", False) and args.out_json.is_file():
        prev = json.loads(args.out_json.read_text(encoding="utf-8"))
        out = list(prev.get("results") or [])
        done_idx = {
            int(r["idx"])
            for r in out
            if r.get("idx") is not None and "error" not in r and isinstance(r.get("result"), dict)
        }
        if done_idx:
            print(f"[resume] skipping {len(done_idx)} cues already in {args.out_json.name}", flush=True)

    for r in sorted(rows, key=lambda x: int(x.get("idx", 0))):
        idx = int(r.get("idx", 0))
        if idx in done_idx:
            continue
        pose = _first_pose(r)
        prompt = _prompt(r, fewshot_text)
        if vlm is not None:
            text = vlm.generate(prompt)
        else:
            resp = client.models.generate_content(model=args.model, contents=[prompt])
            text = (resp.text or "").strip()
        try:
            parsed = _extract_json(text)
        except Exception as e:
            parsed = {"parse_error": str(e), "raw_text": text}
        out = [x for x in out if int(x.get("idx", -1)) != idx] + [
            {
                "idx": r.get("idx"),
                "cue": r.get("cue"),
                "current_dir": pose.get("dir"),
                "current_gripper_orientation": pose.get("gripper_orientation"),
                "result": parsed,
            }
        ]
        print(f"[ok] idx={r.get('idx')} cue={r.get('cue')}")
        if not getattr(args, "dry_run", False):
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(
                json.dumps(
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "mode": "text_only_pose_verification",
                        "vlm_backend": backend,
                        "model": args.model,
                        "config_json": str(args.config_json),
                        "partial": True,
                        "total": len(out),
                        "results": out,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "text_only_pose_verification",
        "vlm_backend": backend,
        "model": args.model,
        "config_json": str(args.config_json),
        "total": len(out),
        "results": out,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out_json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config-json",
        type=Path,
        default=Path("data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot10.json"),
    )
    ap.add_argument(
        "--shots-json",
        type=Path,
        default=Path("data/seed/shots/manipulator/shot_configs_v19_sophisticated.json"),
    )
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument(
        "--vlm-backend",
        default=os.getenv("VLM_BACKEND", "transformers"),
        choices=["transformers", "hf", "local", "vllm-local", "vllm", "openai", "qwen", "gemini"],
    )
    ap.add_argument("--fewshot-n", type=int, default=4)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/results/verify/pose_textonly_verify_pilot20_gemini.json"),
    )
    ap.add_argument("--resume", action="store_true", help="Skip cues already scored in out-json")
    args = ap.parse_args()
    if args.model is None:
        args.model = (
            os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")
            if args.vlm_backend != "gemini"
            else "gemini-2.5-pro"
        )
    run(args)


if __name__ == "__main__":
    main()
