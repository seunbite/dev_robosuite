#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai


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
    p = _first_pose(cue_row)
    return f"""
You are verifying robot cue pose suitability using ONLY text (no image).

Definitions (must be strictly followed):
- World frame: +x = forward toward human viewer, +y = robot left, +z = up.
- End-effector (EE) pointing axis: wrist -> fingertips (approach direction).
- Direction `dir` is the dominant world direction of this pointing axis:
  up/down/front/back/left/right.
- `gripper_orientation`:
  Let end line = jaw opening line between fingertip tips.
  Project that line orthogonally onto the plane perpendicular to `dir`.
  Observer faces that plane.
  If projected line is left-right (ㅡ) => horizontal.
  If projected line is up-down (|) => vertical.

Task:
1) Q1: judge if current pose labels (dir + gripper_orientation) are appropriate for the cue intent.
   "Appropriate" means: for this robot to perform the cue, if it starts from this pose (these dir and gripper_orientation labels), can a motion that conveys the cue's meaning be created using simple subsequent movements?
2) Q3: if appropriate, propose movement plan; if not appropriate, propose corrected pose and movement plan.

Few-shot style hints:
{fewshot_text}

Target:
- cue: {cue_row.get("cue")}
- description: {cue_row.get("description","")}
- current_pose: dir={p.get("dir")}, gripper_orientation={p.get("gripper_orientation")}

Return ONLY strict JSON:
{{
  "pose_is_appropriate": true/false,
  "direction_orientation_assessment": "string",
  "if_appropriate": {{
    "recommended_movement_plan": [
      "step guidance 1",
      "step guidance 2",
      "step guidance 3"
    ]
  }},
  "if_not_appropriate": {{
    "recommended_dir": "front|back|left|right|up|down",
    "recommended_gripper_orientation": "horizontal|vertical",
    "why_change": "string",
    "recommended_movement_plan_after_change": [
      "step guidance 1",
      "step guidance 2",
      "step guidance 3"
    ]
  }},
  "confidence": 0.0
}}
""".strip()


def run(args: argparse.Namespace) -> None:
    rows = _load_json(args.config_json)
    shots = _load_json(args.shots_json)
    fewshot_text = _fewshot_block(shots, n=args.fewshot_n)

    backend = getattr(args, "vlm_backend", "gemini") or "gemini"
    vlm = None
    client = None
    if backend == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("Set GOOGLE_API_KEY (or GEMINI_API_KEY).")
        client = genai.Client(api_key=api_key)
    else:
        import sys
        from pathlib import Path

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
    for r in sorted(rows, key=lambda x: int(x.get("idx", 0))):
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
        out.append(
            {
                "idx": r.get("idx"),
                "cue": r.get("cue"),
                "current_dir": pose.get("dir"),
                "current_gripper_orientation": pose.get("gripper_orientation"),
                "result": parsed,
            }
        )
        print(f"[ok] idx={r.get('idx')} cue={r.get('cue')}")

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
        default=os.getenv("VLM_BACKEND", "gemini"),
        choices=["transformers", "hf", "local", "vllm-local", "vllm", "openai", "qwen", "gemini"],
    )
    ap.add_argument("--fewshot-n", type=int, default=4)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/results/verify/pose_textonly_verify_pilot20_gemini.json"),
    )
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
