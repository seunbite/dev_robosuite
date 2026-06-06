#!/usr/bin/env python3
"""Motion GT vs negative pairwise compare using side-by-side MP4 (Qwen-VL / Gemini)."""
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

from pilot40_experiment_suite import (  # noqa: E402
    MOTION_CFG,
    MOTION_PAIRWISE_DIR,
    PROMPT_MOTION_PAIRWISE,
)
from render_and_verify_o_picked_motions import _extract_json  # noqa: E402
from verify_pose_tiles_gemini import _first_pose, _movement_summary  # noqa: E402
from vlm_client import (  # noqa: E402
    VLMClient,
    init_inprocess_engine,
    is_inprocess_backend,
    is_vllm_http_backend,
    require_vllm_server,
)

DEFAULT_PAIRWISE_JSONS = [
    MOTION_PAIRWISE_DIR / "pairwise_eval_results.json",
    MOTION_PAIRWISE_DIR / "pairwise_eval_results_extra7.json",
    MOTION_PAIRWISE_DIR / "pairwise_eval_results_extra10.json",
    MOTION_PAIRWISE_DIR / "pairwise_eval_results_remaining_mp4.json",
]
DEFAULT_OUT = _REPO / "data/results/verify/pilot40_motion_pairwise_mp4_qwen.json"

NEG_LABELS = {
    "gt": "component GT positive tail",
    "axis": "negative: wrong axis",
    "joint": "negative: wrong joint",
    "direction": "negative: wrong direction",
}


def _load_pair_specs(json_paths: list[Path]) -> list[dict[str, Any]]:
    by_idx: dict[int, dict[str, Any]] = {}
    for fp in json_paths:
        if not fp.is_file():
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        for e in data.get("mp4", []):
            idx = int(e["idx"])
            by_idx[idx] = dict(e)
    return [by_idx[k] for k in sorted(by_idx)]


def _tail_label(kind: str, row: dict[str, Any]) -> str:
    if kind == "gt":
        return _movement_summary(row)
    return NEG_LABELS.get(kind, kind)


def _fill_prompt(
    template: str,
    *,
    cue: str,
    description: str,
    fixed: dict[str, Any],
    left_kind: str,
    right_kind: str,
    row: dict[str, Any],
) -> str:
    return (
        template.replace("{{CUE}}", cue)
        .replace("{{DESCRIPTION}}", description)
        .replace("{{FIXED_DIR}}", str(fixed.get("dir", "?")))
        .replace("{{FIXED_GRIPPER_ORIENTATION}}", str(fixed.get("gripper_orientation", "?")))
        .replace("{{LEFT_TAIL_SUMMARY}}", _tail_label(left_kind, row))
        .replace("{{RIGHT_TAIL_SUMMARY}}", _tail_label(right_kind, row))
    )


def run(args: argparse.Namespace) -> None:
    backend = args.vlm_backend
    if backend == "gemini":
        vlm = None
        from google import genai

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("Set GOOGLE_API_KEY for gemini backend.")
        client = genai.Client(api_key=api_key)
    else:
        if is_vllm_http_backend(backend):
            require_vllm_server()
        elif is_inprocess_backend(backend):
            init_inprocess_engine(backend, args.model)
        vlm = VLMClient(backend=backend, model=args.model)
        client = None

    template = PROMPT_MOTION_PAIRWISE.read_text(encoding="utf-8")
    cfg_rows = {int(r["idx"]): r for r in json.loads(MOTION_CFG.read_text(encoding="utf-8"))}
    json_paths = args.pairwise_jsons or DEFAULT_PAIRWISE_JSONS
    specs = _load_pair_specs(json_paths)
    if args.limit:
        specs = specs[: args.limit]

    out_path = Path(args.out_json)
    existing: list[dict[str, Any]] = []
    done: set[int] = set()
    if args.resume and out_path.is_file():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        existing = prev.get("mp4") or []
        done = {int(r["idx"]) for r in existing if "vlm_correct" in r or "correct" in r}

    results: list[dict[str, Any]] = list(existing)
    for spec in specs:
        idx = int(spec["idx"])
        if idx in done and not args.force:
            continue
        cue = str(spec.get("cue", ""))
        row = cfg_rows.get(idx)
        if not row:
            print(f"[skip] {cue}: no motion config", flush=True)
            continue
        mp4_rel = spec.get("pair_mp4")
        if not mp4_rel:
            hits = sorted(MOTION_PAIRWISE_DIR.glob(f"{idx:03d}_{cue}_pair*.mp4"))
            mp4_rel = str(hits[0].relative_to(_REPO)) if hits else None
        mp4_path = _REPO / mp4_rel if mp4_rel else None
        if not mp4_path or not mp4_path.is_file():
            print(f"[skip] {cue}: missing mp4 {mp4_rel}", flush=True)
            continue

        gt_side = str(spec.get("gt_side", "left")).lower()
        left_kind = str(spec.get("left", "left"))
        right_kind = str(spec.get("right", "right"))
        fixed = row.get("gt_fixed_first_pose") or _first_pose(row)
        prompt = spec.get("prompt")
        if not prompt:
            prompt = _fill_prompt(
                template,
                cue=cue,
                description=str(row.get("description", "")),
                fixed=fixed,
                left_kind=left_kind,
                right_kind=right_kind,
                row=row,
            )

        if vlm is not None:
            text = vlm.generate(prompt, videos=[str(mp4_path.resolve())])
        else:
            from google.genai import types

            part = types.Part.from_bytes(data=mp4_path.read_bytes(), mime_type="video/mp4")
            resp = client.models.generate_content(model=args.model, contents=[part, prompt])
            text = (resp.text or "").strip()

        try:
            parsed = _extract_json(text)
        except Exception as e:
            parsed = {"parse_error": str(e), "raw_text": text}

        better = str(parsed.get("better_side", "")).lower().strip()
        vlm_correct = better == gt_side
        record: dict[str, Any] = {
            "idx": idx,
            "cue": cue,
            "gt_side": gt_side,
            "left": left_kind,
            "right": right_kind,
            "pair_mp4": str(mp4_rel),
            "prompt": prompt,
            "parsed": parsed,
            "pred": better,
            "better_side": better,
            "vlm_correct": vlm_correct,
            "correct": vlm_correct,
        }
        results = [r for r in results if int(r.get("idx", -1)) != idx] + [record]
        mark = "OK" if vlm_correct else "MISS"
        print(f"[{mark}] {cue} gt={gt_side} vlm={better}", flush=True)
        if not args.dry_run:
            _checkpoint(out_path, args, results)

    ok = sum(1 for r in results if r.get("vlm_correct"))
    scored = sum(1 for r in results if "vlm_correct" in r)
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "motion_gt_neg_pairwise_mp4",
        "vlm_backend": backend,
        "model": args.model,
        "n": len(results),
        "n_scored": scored,
        "accuracy": ok / scored if scored else None,
        "mp4": sorted(results, key=lambda r: int(r["idx"])),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    acc = payload["accuracy"]
    print(f"\nWrote {out_path} ({scored} scored, accuracy={acc})", flush=True)


def _checkpoint(out_path: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    out_path.write_text(
        json.dumps(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "partial": True,
                "model": args.model,
                "mp4": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Motion pairwise MP4 eval (pilot40)")
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model", default=None)
    p.add_argument(
        "--vlm-backend",
        default=os.getenv("VLM_BACKEND", "transformers"),
        choices=["transformers", "hf", "local", "vllm-local", "vllm", "openai", "qwen", "gemini"],
    )
    p.add_argument("--pairwise-jsons", nargs="*", type=Path, default=DEFAULT_PAIRWISE_JSONS)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.model is None:
        args.model = (
            os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")
            if args.vlm_backend != "gemini"
            else "gemini-2.5-pro"
        )
    run(args)


if __name__ == "__main__":
    main()
