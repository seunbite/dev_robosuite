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


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
import sys

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from prompt_loader import exp_prompt_path, load_snippet  # noqa: E402

_PROMPT_DIR = _REPO / "data/seed/prompt/google_robot"
_RENDER_DIR = _REPO / "data/results/render/google_robot"
_MEDIA_DIR = _REPO / "data/results/render/google_robot/pilot40_media"
_DEFAULT_CFG = _REPO / "data/seed/shots/google_robot/shot_configs_pilot40_mobile.json"
_FEWSHOT_SHOTS = _REPO / "data/seed/shots/google_robot/diverse_shots_mobile.json"


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


def _first_pose_component(row: dict[str, Any]) -> dict[str, Any]:
    for step in row.get("movements", []):
        if step.get("type") == "pose":
            return step
    return {}


def _movement_tail_component(row: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_pose = False
    for step in row.get("movements", []):
        if step.get("type") == "pose" and not seen_pose:
            seen_pose = True
            continue
        out.append(step)
    return out


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


def _movement_summary(tail: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for step in tail[:4]:
        parts.append(str(step.get("type", "?")))
    return ", ".join(parts) if parts else "none"


def _fewshot_text(n: int = 2) -> str:
    if not _FEWSHOT_SHOTS.is_file():
        return ""
    rows = _load_json(_FEWSHOT_SHOTS)[:n]
    lines: list[str] = []
    for r in rows:
        cue = r.get("cue", "?")
        moves = [s.get("type") for s in (r.get("movements") or [])[1:4]]
        lines.append(f"- {cue}: tail steps {', '.join(moves)}")
    return "\n".join(lines)


def _prompt_path(component: str, modality: str, prompt_file: Path | None, exp_id: str | None = None) -> Path:
    if prompt_file:
        return prompt_file
    if exp_id:
        p = exp_prompt_path(exp_id)
        if p.is_file():
            return p
    if component == "pose" and modality == "text":
        return exp_prompt_path(3) if exp_prompt_path(3).is_file() else _PROMPT_DIR / "prompt_verify_pose_text_component.txt"
    if component == "pose" and modality == "vlm":
        return exp_prompt_path(2) if exp_prompt_path(2).is_file() else _PROMPT_DIR / "prompt_verify_pose_vlm_component.txt"
    if component == "movement" and modality == "text":
        return exp_prompt_path(9) if exp_prompt_path(9).is_file() else _PROMPT_DIR / "prompt_verify_movement_text_component.txt"
    if component == "movement" and modality == "vlm":
        return exp_prompt_path(8) if exp_prompt_path(8).is_file() else _PROMPT_DIR / "prompt_verify_movement_vlm_component.txt"
    raise ValueError(f"Unsupported component/modality: {component}/{modality}")


def _fill_prompt(template: str, row: dict[str, Any]) -> str:
    pose_comp = _first_pose_component(row)
    move_comp = _movement_tail_component(row)
    pose_fields = (pose_comp.get("parameters") or {}).get("pose") or {}
    return (
        template.replace("{{CUE_NAME}}", str(row.get("cue", "")))
        .replace("{{CUE}}", str(row.get("cue", "")))
        .replace("{{CUE_DESCRIPTION}}", str(row.get("description", "")))
        .replace("{{DESCRIPTION}}", str(row.get("description", "")))
        .replace("{{POSE_COMPONENT_JSON}}", json.dumps(pose_comp, ensure_ascii=False, indent=2))
        .replace("{{MOVEMENT_COMPONENT_JSON}}", json.dumps(move_comp, ensure_ascii=False, indent=2))
        .replace("{{TAIL_JSON}}", json.dumps(move_comp, ensure_ascii=False, indent=2))
        .replace("{{MOVEMENT_SUMMARY}}", _movement_summary(move_comp))
        .replace("{{TAIL_SUMMARY}}", _movement_summary(move_comp))
        .replace("{{APPROPRIATE_MEANS}}", load_snippet("_shared_appropriate_means.txt"))
        .replace("{{POSE_DEFINITIONS}}", load_snippet("_shared_pose_definitions.txt"))
        .replace("{{FEWSHOT}}", _fewshot_text())
        .replace(
            "{{FIXED_ARM_POSITION}}",
            str(pose_fields.get("arm_position", "")),
        )
        .replace(
            "{{FIXED_GRIPPER_ORIENTATION}}",
            str(pose_fields.get("gripper_orientation", "")),
        )
    )


def _call_text(client: genai.Client, model: str, prompt: str) -> dict[str, Any]:
    resp = client.models.generate_content(model=model, contents=[prompt])
    txt = (resp.text or "").strip()
    try:
        return _extract_json(txt)
    except Exception as e:
        return {"parse_error": str(e), "raw_text": txt}


def _media_stem(row: dict[str, Any]) -> str:
    idx = int(row.get("idx", -1))
    cue = _safe_cue(str(row.get("cue", "")))
    return f"mm19_g{idx:02d}_{cue}"


def _pose_png_for_row(media_dir: Path, row: dict[str, Any]) -> Path | None:
    p = media_dir / "pose" / f"{_media_stem(row)}_pose.png"
    return p if p.is_file() else None


def _mp4_for_row(media_dir: Path, row: dict[str, Any]) -> Path | None:
    p = media_dir / "mp4" / f"{_media_stem(row)}.mp4"
    return p if p.is_file() else None


def _call_vlm_image(client: genai.Client, model: str, prompt: str, img_path: Path) -> dict[str, Any]:
    part = types.Part.from_bytes(data=img_path.read_bytes(), mime_type="image/png")
    resp = client.models.generate_content(model=model, contents=[part, prompt])
    txt = (resp.text or "").strip()
    try:
        return _extract_json(txt)
    except Exception as e:
        return {"parse_error": str(e), "raw_text": txt}


def _call_vlm_video(client: genai.Client, model: str, prompt: str, mp4_path: Path) -> dict[str, Any]:
    part = types.Part.from_bytes(data=mp4_path.read_bytes(), mime_type="video/mp4")
    resp = client.models.generate_content(model=model, contents=[part, prompt])
    txt = (resp.text or "").strip()
    try:
        return _extract_json(txt)
    except Exception as e:
        return {"parse_error": str(e), "raw_text": txt}


def _call_vlm(client: genai.Client, model: str, prompt: str, gif_path: Path) -> dict[str, Any]:
    part = types.Part.from_bytes(data=gif_path.read_bytes(), mime_type="image/gif")
    resp = client.models.generate_content(model=model, contents=[part, prompt])
    txt = (resp.text or "").strip()
    try:
        return _extract_json(txt)
    except Exception as e:
        return {"parse_error": str(e), "raw_text": txt}


def run(args: argparse.Namespace) -> None:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY (or GEMINI_API_KEY).")
    client = genai.Client(api_key=api_key)

    rows = sorted(_load_json(args.config_json), key=lambda r: int(r.get("idx", 0)))
    if args.limit:
        rows = rows[: args.limit]

    template = _prompt_path(args.component, args.modality, args.prompt_file, args.exp_id).read_text(encoding="utf-8")
    media_dir = Path(args.media_dir)
    out_rows: list[dict[str, Any]] = []

    for row in rows:
        prompt = _fill_prompt(template, row)
        media_path: Path | None = None
        if args.modality == "text":
            parsed = _call_text(client, args.model, prompt)
        elif args.component == "pose" and args.modality == "vlm":
            media_path = _pose_png_for_row(media_dir, row)
            if not media_path:
                media_path = _gif_for_row(args.render_dir, row)
                if media_path:
                    parsed = _call_vlm(client, args.model, prompt, media_path)
                else:
                    parsed = {"error": "missing_pose_png_or_gif"}
            else:
                parsed = _call_vlm_image(client, args.model, prompt, media_path)
        elif args.component == "movement" and args.modality == "vlm":
            media_path = _mp4_for_row(media_dir, row)
            if not media_path:
                gif_path = _gif_for_row(args.render_dir, row)
                if gif_path:
                    media_path = gif_path
                    parsed = _call_vlm(client, args.model, prompt, gif_path)
                else:
                    parsed = {"error": "missing_mp4_or_gif"}
            else:
                parsed = _call_vlm_video(client, args.model, prompt, media_path)
        else:
            gif_path = _gif_for_row(args.render_dir, row)
            if not gif_path:
                parsed = {"error": "missing_gif_for_vlm"}
            else:
                media_path = gif_path
                parsed = _call_vlm(client, args.model, prompt, gif_path)

        out_rows.append(
            {
                "idx": row.get("idx"),
                "cue": row.get("cue"),
                "component": args.component,
                "modality": args.modality,
                "media": str(media_path) if media_path else None,
                "result": parsed,
            }
        )
        ok = parsed.get("pose_is_appropriate") if args.component == "pose" else parsed.get(
            "movement_is_appropriate"
        )
        print(f"[ok] idx={row.get('idx')} cue={row.get('cue')} ok={ok}")

    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "component": args.component,
        "modality": args.modality,
        "config_json": str(args.config_json),
        "render_dir": str(args.render_dir),
        "media_dir": str(media_dir),
        "total": len(out_rows),
        "results": out_rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out_json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-json", type=Path, default=_DEFAULT_CFG)
    ap.add_argument("--component", choices=("pose", "movement"), required=True)
    ap.add_argument("--modality", choices=("text", "vlm"), required=True)
    ap.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    ap.add_argument("--exp-id", type=str, default=None, help="Use prompt_exp{N}.txt when set")
    ap.add_argument("--prompt-file", type=Path, default=None)
    ap.add_argument("--render-dir", type=Path, default=_RENDER_DIR)
    ap.add_argument("--media-dir", type=Path, default=_MEDIA_DIR)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
