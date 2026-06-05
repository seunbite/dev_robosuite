"""Google Robot component compare via in-process VLM (GIF inputs)."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vlm_client import VLMClient, init_inprocess_engine  # noqa: E402

_PROMPT_DIR = _REPO / "data/seed/prompt/google_robot"


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


def _gif_for_row(render_dir: Path, row: dict[str, Any]) -> Path | None:
    idx = int(row.get("idx", -1))
    cue = str(row.get("cue", "")).replace("/", "_").replace(" ", "_")
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


def _gif_first_frame(path: Path) -> Image.Image:
    im = Image.open(path)
    try:
        im.seek(0)
    except EOFError:
        pass
    return im.convert("RGB")


def run(args: argparse.Namespace, *, vlm: VLMClient | None = None) -> dict[str, Any]:
    if not args.render_dir_a.is_dir() or not args.render_dir_b.is_dir():
        payload = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "status": "skipped",
            "note": f"render dirs missing: {args.render_dir_a} / {args.render_dir_b}",
            "headline": "skipped (no GIF renders in repo — rsync data/results/render/google_robot/)",
            "results": [],
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[google_robot] skipped — wrote {args.out_json}", flush=True)
        return payload

    if vlm is None:
        init_inprocess_engine(args.vlm_backend, args.model)
        vlm = VLMClient(backend=args.vlm_backend, model=args.model)

    template = (args.prompt_file or _PROMPT_DIR / "prompt_compare_pose_vlm_component.txt").read_text(
        encoding="utf-8"
    )
    rows_a = sorted(_load_json(args.config_a), key=lambda r: int(r.get("idx", 0)))
    rows_b = sorted(_load_json(args.config_b), key=lambda r: int(r.get("idx", 0)))
    by_a = {str(r.get("cue")): r for r in rows_a}
    by_b = {str(r.get("cue")): r for r in rows_b}
    cues = sorted(set(by_a.keys()) & set(by_b.keys()))
    if args.limit:
        cues = cues[: int(args.limit)]

    results: list[dict[str, Any]] = []
    for cue in cues:
        ra, rb = by_a[cue], by_b[cue]
        ga, gb = _gif_for_row(args.render_dir_a, ra), _gif_for_row(args.render_dir_b, rb)
        if not ga or not gb:
            results.append({"cue": cue, "skipped": True, "error": "missing_gif"})
            continue
        prompt = (
            template.replace("{{CUE_NAME}}", cue)
            .replace("{{CUE_DESCRIPTION}}", str(ra.get("description", "")))
            .replace("{{POSE_A_JSON}}", json.dumps(_first_pose_component(ra), ensure_ascii=False, indent=2))
            .replace("{{POSE_B_JSON}}", json.dumps(_first_pose_component(rb), ensure_ascii=False, indent=2))
        )
        text = vlm.generate(prompt, images=[_gif_first_frame(ga), _gif_first_frame(gb)])
        try:
            parsed = _extract_json(text)
        except Exception as e:
            parsed = {"parse_error": str(e), "raw_text": text}
        winner = str(parsed.get("winner", "")).upper().strip()
        results.append(
            {
                "cue": cue,
                "gif_a": str(ga),
                "gif_b": str(gb),
                "vlm_result": parsed,
                "vlm_winner": winner if winner in {"A", "B", "TIE"} else None,
            }
        )
        print(f"[google_robot] {cue} -> {winner}", flush=True)

    scored = sum(1 for r in results if r.get("vlm_winner"))
    skipped = sum(1 for r in results if r.get("skipped"))
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "status": "ok",
        "model": args.model,
        "vlm_backend": args.vlm_backend,
        "component": "pose",
        "config_a": str(args.config_a),
        "config_b": str(args.config_b),
        "render_dir_a": str(args.render_dir_a),
        "render_dir_b": str(args.render_dir_b),
        "n_total": len(results),
        "n_scored": scored,
        "n_skipped": skipped,
        "headline": f"scored {scored}, skipped {skipped} (no GT acc — compare only)",
        "results": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out_json}", flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-a", type=Path, required=True)
    ap.add_argument("--config-b", type=Path, required=True)
    ap.add_argument("--render-dir-a", type=Path, required=True)
    ap.add_argument("--render-dir-b", type=Path, required=True)
    ap.add_argument("--model", default=os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"))
    ap.add_argument("--vlm-backend", default=os.getenv("VLM_BACKEND", "transformers"))
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--prompt-file", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
