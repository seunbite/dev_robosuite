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
from google.genai import types


_PROMPT_DIR = Path("data/seed/prompt/google_robot")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_json(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        s = m.group(0)
    return json.loads(s)


def _safe_cue(cue: str) -> str:
    return cue.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _gif_for_row(render_dir: Path, row: dict[str, Any]) -> Path | None:
    idx = int(row.get("idx", -1))
    cue = _safe_cue(str(row.get("cue", "")))
    exact = render_dir / f"mm19_g{idx:02d}_{cue}.gif"
    if exact.is_file():
        return exact
    cands = sorted(render_dir.glob(f"*g{idx:02d}*{cue}*.gif"))
    return cands[0] if cands else None


def _first_pose_component(row: dict[str, Any]) -> dict[str, Any]:
    for step in row.get("movements", []):
        if step.get("type") == "pose":
            return step
    return {}


def _movement_tail_component(row: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    consumed_first_pose = False
    for step in row.get("movements", []):
        if step.get("type") == "pose" and not consumed_first_pose:
            consumed_first_pose = True
            continue
        out.append(step)
    return out


def _prompt_template(component: str, prompt_file: Path | None) -> str:
    if prompt_file:
        return prompt_file.read_text(encoding="utf-8")
    if component == "pose":
        path = _PROMPT_DIR / "prompt_compare_pose_vlm_component.txt"
    else:
        path = _PROMPT_DIR / "prompt_compare_movement_vlm_component.txt"
    return path.read_text(encoding="utf-8")


def _fill_prompt(template: str, component: str, ra: dict[str, Any], rb: dict[str, Any]) -> str:
    out = (
        template.replace("{{CUE_NAME}}", str(ra.get("cue", "")))
        .replace("{{CUE_DESCRIPTION}}", str(ra.get("description", "")))
    )
    if component == "pose":
        out = out.replace("{{POSE_A_JSON}}", json.dumps(_first_pose_component(ra), ensure_ascii=False, indent=2))
        out = out.replace("{{POSE_B_JSON}}", json.dumps(_first_pose_component(rb), ensure_ascii=False, indent=2))
    else:
        out = out.replace(
            "{{MOVEMENT_A_JSON}}",
            json.dumps(_movement_tail_component(ra), ensure_ascii=False, indent=2),
        )
        out = out.replace(
            "{{MOVEMENT_B_JSON}}",
            json.dumps(_movement_tail_component(rb), ensure_ascii=False, indent=2),
        )
    return out


def run(args: argparse.Namespace) -> None:
    vlm = getattr(args, "vlm", None)
    client = None
    if vlm is None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("Set GOOGLE_API_KEY (or GEMINI_API_KEY).")
        client = genai.Client(api_key=api_key)

    rows_a = sorted(_load_json(args.config_a), key=lambda r: int(r.get("idx", 0)))
    rows_b = sorted(_load_json(args.config_b), key=lambda r: int(r.get("idx", 0)))
    by_cue_a = {str(r.get("cue")): r for r in rows_a}
    by_cue_b = {str(r.get("cue")): r for r in rows_b}
    cues = sorted(set(by_cue_a.keys()) & set(by_cue_b.keys()))
    if args.limit:
        cues = cues[: args.limit]

    template = _prompt_template(args.component, args.prompt_file)
    results: list[dict[str, Any]] = []

    for cue in cues:
        ra = by_cue_a[cue]
        rb = by_cue_b[cue]
        ga = _gif_for_row(args.render_dir_a, ra)
        gb = _gif_for_row(args.render_dir_b, rb)
        prompt = _fill_prompt(template, args.component, ra, rb)
        if not ga or not gb:
            parsed: dict[str, Any] = {"error": "missing_gif", "gif_a": str(ga) if ga else None, "gif_b": str(gb) if gb else None}
        elif vlm is not None:
            from vlm_infer_shared import load_vlm_image, parse_json_response  # noqa: WPS433

            text = vlm.generate(prompt, images=[load_vlm_image(ga), load_vlm_image(gb)])
            parsed = parse_json_response(text)
        else:
            pa = types.Part.from_bytes(data=ga.read_bytes(), mime_type="image/gif")
            pb = types.Part.from_bytes(data=gb.read_bytes(), mime_type="image/gif")
            resp = client.models.generate_content(model=args.model, contents=[pa, pb, prompt])
            txt = (resp.text or "").strip()
            try:
                parsed = _extract_json(txt)
            except Exception as e:
                parsed = {"parse_error": str(e), "raw_text": txt}

        results.append(
            {
                "cue": cue,
                "idx_a": ra.get("idx"),
                "idx_b": rb.get("idx"),
                "gif_a": str(ga) if ga else None,
                "gif_b": str(gb) if gb else None,
                "result": parsed,
            }
        )
        print(f"[ok] cue={cue}")

    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "component": args.component,
        "config_a": str(args.config_a),
        "config_b": str(args.config_b),
        "render_dir_a": str(args.render_dir_a),
        "render_dir_b": str(args.render_dir_b),
        "total": len(results),
        "results": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out_json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", choices=("pose", "movement"), required=True)
    ap.add_argument("--config-a", type=Path, required=True)
    ap.add_argument("--config-b", type=Path, required=True)
    ap.add_argument("--render-dir-a", type=Path, required=True)
    ap.add_argument("--render-dir-b", type=Path, required=True)
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--prompt-file", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
