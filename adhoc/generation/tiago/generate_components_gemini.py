#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from google import genai


_PROMPT_DIR = Path("data/seed/prompt/tiago")
_DEFAULT_CUES = Path("data/seed/yml/cues_new.yml")


def _extract_json(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        s = m.group(0)
    return json.loads(s)


def _load_cues(yaml_path: Path) -> dict[str, str]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for grp in ("iconic", "contextual"):
        if grp in data and isinstance(data[grp], dict):
            for k, v in data[grp].items():
                out[str(k)] = str(v)
    return out


def _prompt_path(component: str, prompt_file: Path | None) -> Path:
    if prompt_file:
        return prompt_file
    if component == "pose":
        return _PROMPT_DIR / "prompt_generation_pose_component.txt"
    return _PROMPT_DIR / "prompt_generation_movement_component.txt"


def _fill_prompt(template: str, component: str, cue: str, desc: str, cues: dict[str, str], fixed_pose: dict[str, Any] | None) -> str:
    catalog = "\n".join(f"- {k}: {v}" for k, v in cues.items())
    out = (
        template.replace("{{CUE_NAME}}", cue)
        .replace("{{CUE_DESCRIPTION}}", desc)
        .replace("{{CUE_CATALOG}}", catalog)
        .replace("{{FEW_SHOT_EXAMPLES}}", "")
    )
    if component == "movement":
        out = out.replace("{{FIXED_START_POSE_JSON}}", json.dumps(fixed_pose or {}, ensure_ascii=False, indent=2))
    return out


def run(args: argparse.Namespace) -> None:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY (or GEMINI_API_KEY).")
    client = genai.Client(api_key=api_key)

    cues = _load_cues(args.cues_yaml)
    if args.cues:
        selected = [c.strip() for c in args.cues.split(",") if c.strip()]
    else:
        selected = list(cues.keys())
    if args.limit:
        selected = selected[: args.limit]

    fixed_pose_map: dict[str, dict[str, Any]] = {}
    if args.fixed_pose_config and args.fixed_pose_config.is_file():
        rows = json.loads(args.fixed_pose_config.read_text(encoding="utf-8"))
        for r in rows:
            cue = str(r.get("cue"))
            pose = {}
            for step in r.get("movements", []):
                if step.get("type") == "pose":
                    pose = step
                    break
            fixed_pose_map[cue] = pose

    template = _prompt_path(args.component, args.prompt_file).read_text(encoding="utf-8")
    out_rows: list[dict[str, Any]] = []
    for cue in selected:
        desc = cues.get(cue, "")
        fixed = fixed_pose_map.get(cue)
        prompt = _fill_prompt(template, args.component, cue, desc, cues, fixed)
        resp = client.models.generate_content(model=args.model, contents=[prompt])
        text = (resp.text or "").strip()
        try:
            parsed = _extract_json(text)
        except Exception as e:
            parsed = {"parse_error": str(e), "raw_text": text}
        out_rows.append({"cue": cue, "description": desc, "component": args.component, "result": parsed})
        print(f"[ok] cue={cue}")

    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "component": args.component,
        "total": len(out_rows),
        "results": out_rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out_json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", choices=("pose", "movement"), required=True)
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--cues-yaml", type=Path, default=_DEFAULT_CUES)
    ap.add_argument("--cues", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt-file", type=Path, default=None)
    ap.add_argument("--fixed-pose-config", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
