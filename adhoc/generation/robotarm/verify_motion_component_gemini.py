#!/usr/bin/env python3
"""
Motion verify (pose-style): is the generated tail appropriate for the cue?
If not, recommend component-level fixes (axis/joint/rep/hold/path).

Input: rendered motion MP4 + text (fixed pose + tail summary).
Scores vs component GT in score_pilot40_motion_verify_metrics.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from motion_media_paths import prepare_pilot40_motion_mp4s, resolve_mp4, write_pilot40_manifest  # noqa: E402
from verify_pose_tiles_gemini import (  # noqa: E402
    APPROPRIATE_MEANS_LINE,
    _fewshot_block,
    _first_pose,
    _movement_summary,
)
from vlm_client import setup_vlm_from_args  # noqa: E402
from vlm_json import extract_json  # noqa: E402

BASE_CFG = (
    _REPO
    / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_gt_fixed_pose_pilot40.json"
)
MP4_DIR = _REPO / "data/results/render/manipulator/motion_vlm_verify_pilot40/mp4"
MANIFEST = _REPO / "data/results/render/manipulator/motion_vlm_verify_pilot40/manifest_pilot40.json"
SHOTS = _REPO / "data/seed/shots/manipulator/shot_configs_v19_sophisticated.json"
OUT_JSON = _REPO / "data/results/verify/pilot40_motion_component_verify_gemini.json"


def _gemini_client():
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY.")
    return genai.Client(api_key=api_key)


def _vlm_prompt(row: dict[str, Any], fewshot_text: str) -> str:
    p = _first_pose(row)
    fixed = row.get("gt_fixed_first_pose") or p
    return f"""
You are verifying a robot-arm motion (IIWA) for a social gesture cue.
You see a short video of the simulated robot executing the motion.

Context:
- The **first pose is fixed** (human GT); only the **tail movement** after that pose was generated.
- World frame: +x forward toward viewer, +y robot left, +z up.
- Movement uses joint rotations (shoulder/elbow/wrist) and/or Cartesian paths (line/arc).

Task:
1) Q1: Is the **current tail movement** appropriate for conveying this cue, given the fixed start pose?
{APPROPRIATE_MEANS_LINE.replace("this pose", "this fixed start pose").replace("subsequent movements", "the shown tail movement")}
2) Q2: If appropriate, note small optional refinements (short bullets).
3) Q3: If **not** appropriate, recommend how to **change the movement** using the component vocabulary below
   (same style as human motion annotations: e.g. "z +- rep wrist", "x + non hold", "arc xz", "line y").

Component vocabulary for recommendations:
- movement: axes x/y/z each +, -, or +- ; optional joint shoulder|elbow|wrist ; repetition non|rep|any ; optional hold
- path_arc: plane xy|yz|xz
- path_line: axis x|y|z

Few-shot examples (pose + movement style):
{fewshot_text}

Target:
- cue: {row.get("cue")}
- description: {row.get("description", "")}
- fixed_start_pose: dir={fixed.get("dir")}, gripper_orientation={fixed.get("gripper_orientation")}
- current_tail_summary: {_movement_summary(row)}

Return ONLY strict JSON:
{{
  "movement_is_appropriate": true/false,
  "movement_assessment": "string",
  "if_appropriate": {{
    "optional_refinements": ["string", "string"]
  }},
  "if_not_appropriate": {{
    "why_not": "string",
    "recommended_component": {{
      "kind": "movement|path_arc|path_line",
      "axes": {{"x": "+", "y": "-", "z": "+-"}},
      "joint": "shoulder|elbow|wrist|null",
      "repetition": "non|rep|any|null",
      "hold": true|null,
      "plane": "xy|yz|xz|null",
      "axis": "x|y|z|null"
    }},
    "recommended_tail_guidance": ["step 1", "step 2", "step 3"]
  }},
  "confidence": 0.0
}}
""".strip()


def _vlm_mp4(
    model_id: str,
    prompt: str,
    mp4: Path,
    *,
    vlm_backend: str = "transformers",
    vlm: Any | None = None,
) -> dict[str, Any]:
    if vlm is not None:
        text = vlm.generate(prompt, videos=[str(mp4.resolve())])
    elif vlm_backend == "gemini":
        from google.genai import types

        client = _gemini_client()
        part = types.Part.from_bytes(data=mp4.read_bytes(), mime_type="video/mp4")
        resp = client.models.generate_content(model=model_id, contents=[part, prompt])
        text = (resp.text or "").strip()
    else:
        raise RuntimeError(
            f"Non-gemini backend {vlm_backend!r} requires an in-process VLMClient (got vlm=None)."
        )
    try:
        return extract_json(text)
    except Exception as e:
        return {"parse_error": str(e), "raw_text": text}


def _normalize_component(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if not raw or not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    if kind not in ("movement", "path_arc", "path_line"):
        return None
    out: dict[str, Any] = {"kind": kind}
    if kind == "path_arc":
        plane = (raw.get("plane") or "xz")
        if plane == "null":
            return None
        out["plane"] = str(plane).lower()
        return out
    if kind == "path_line":
        axis = raw.get("axis")
        if not axis or axis == "null":
            return None
        out["axis"] = str(axis).lower()
        return out
    axes = raw.get("axes") or {}
    if isinstance(axes, dict):
        clean = {}
        for ax in "xyz":
            if ax in axes and axes[ax] in ("+", "-", "+-"):
                clean[ax] = axes[ax]
        if clean:
            out["axes"] = clean
    j = raw.get("joint")
    if j and j != "null":
        out["joint"] = str(j).lower()
    rep = raw.get("repetition")
    if rep and rep != "null":
        out["repetition"] = str(rep).lower()
    if raw.get("hold") is True:
        out["hold"] = True
    return out if len(out) > 1 else None


def run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    out_path = Path(args.out_json) if getattr(args, "out_json", None) else OUT_JSON
    rows = sorted(json.loads(BASE_CFG.read_text(encoding="utf-8")), key=lambda r: int(r["idx"]))
    by_idx = {int(r["idx"]): r for r in rows}
    if manifest_path.is_file():
        manifest_rows = json.loads(manifest_path.read_text(encoding="utf-8"))["rows"]
    else:
        print(f"[warn] manifest missing ({manifest_path}); using motion config rows", flush=True)
        manifest_rows = [{"cue_idx": int(r["idx"]), "cue": r["cue"]} for r in rows]
    shots = json.loads(SHOTS.read_text(encoding="utf-8"))
    fewshot = _fewshot_block(shots, n=args.fewshot_n)

    backend, vlm = setup_vlm_from_args(args)
    print(
        f"[motion-vlm] backend={backend} modality=mp4 shared_vlm={'yes' if vlm else 'no'}",
        flush=True,
    )

    out_rows: list[dict[str, Any]] = []
    if args.resume and out_path.is_file():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        out_rows = prev.get("rows") or []
        done = {int(r["cue_idx"]) for r in out_rows}
    else:
        done = set()

    if args.limit:
        manifest_rows = manifest_rows[: args.limit]

    prepare_media = getattr(args, "prepare_media", True)
    if prepare_media and not args.dry_run:
        todo = [
            (int(item["cue_idx"]), str(item["cue"]))
            for item in manifest_rows
            if int(item["cue_idx"]) not in done or args.force
        ]
        ready, failures = prepare_pilot40_motion_mp4s(_REPO, _HERE, todo, config_json=BASE_CFG)
        print(f"[prepare] {ready}/{len(todo)} mp4 ready", flush=True)
        if failures and ready < len(todo):
            print(f"[prepare warn] {len(failures)} issues (first 3):", flush=True)
            for line in failures[:3]:
                print(f"  {line}", flush=True)
        write_pilot40_manifest(_REPO, [by_idx[i] for i, _ in todo if i in by_idx])

    for item in manifest_rows:
        idx = int(item["cue_idx"])
        if idx in done and not args.force:
            continue
        row = by_idx.get(idx)
        if not row:
            continue
        mp4_path, skip_reason = resolve_mp4(_REPO, item, idx, str(item["cue"]))
        if not mp4_path:
            print(f"[skip] {item['cue']}: {skip_reason or 'no mp4'}", flush=True)
            continue

        parsed = _vlm_mp4(
            args.model,
            _vlm_prompt(row, fewshot),
            mp4_path,
            vlm_backend=backend,
            vlm=vlm,
        )
        rec_comp = None
        if not parsed.get("movement_is_appropriate"):
            rec_comp = _normalize_component(
                (parsed.get("if_not_appropriate") or {}).get("recommended_component")
            )

        record = {
            "cue_idx": idx,
            "cue": item["cue"],
            "mp4": str(mp4_path),
            "verify_result": parsed,
            "movement_is_appropriate": parsed.get("movement_is_appropriate"),
            "recommended_component": rec_comp,
        }
        out_rows = [r for r in out_rows if int(r["cue_idx"]) != idx] + [record]
        done.add(idx)
        print(
            f"[verify] {item['cue']} appropriate={parsed.get('movement_is_appropriate')} "
            f"rec={rec_comp is not None}",
            flush=True,
        )
        if not args.dry_run:
            _checkpoint(out_rows, args.model, out_path)

    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "vlm_backend": backend,
        "model": args.model,
        "mode": "motion_component_verify_mp4",
        "config": str(BASE_CFG),
        "n": len(out_rows),
        "rows": sorted(out_rows, key=lambda r: int(r["cue_idx"])),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")


def _checkpoint(rows: list[dict[str, Any]], model: str, out_path: Path) -> None:
    out_path.write_text(
        json.dumps(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "model": model,
                "partial": True,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--vlm-backend",
        default=os.getenv("VLM_BACKEND", "transformers"),
        choices=["transformers", "hf", "local", "vllm-local", "vllm", "openai", "qwen", "gemini"],
    )
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--fewshot-n", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--prepare-media",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("MOTION_PREPARE_MP4", "1") != "0",
        help="Render missing GIFs + build MP4 before VLM (default: on)",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST,
        help="Optional JSON manifest with mp4 paths (default: motion_vlm_verify_pilot40)",
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
