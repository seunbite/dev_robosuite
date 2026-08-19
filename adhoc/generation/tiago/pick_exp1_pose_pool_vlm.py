#!/usr/bin/env python3
"""Pick best pose-pool motion tile per cue (exp1 HTML pose pool → per-tile GIFs → VLM).

Uses exp4-style representative framing; only cues with 2+ pool tiles are sent to the VLM.
"""
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
_ROBOTARM = _REPO / "adhoc/generation/robotarm"
for p in (_REPO, _ROBOTARM, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build_exp1_review_html import exp1_pose_tiles_dir, exp1_render_dir  # noqa: E402
from pilot40_paths import load_config_list, result_config_path  # noqa: E402
from prompt_loader import load_snippet  # noqa: E402

PROMPT_PATH = _REPO / "data/seed/prompt/tiago/prompt_exp1_pose_pool_motion_pick.txt"
DEFAULT_OUT = _REPO / "data/results/verify/tiago/exp1_pose_pool_pick_gemini-2.5-pro.json"
DEFAULT_TILE_EXPORT = _REPO / "data/results/visualize/tiago/exp1_pose_pool_tiles"
_MAX_MOTION_POOL_TILES = 15


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


def _parse_idx_only(raw: list[str] | None) -> set[int] | None:
    if not raw:
        return None
    joined = " ".join(raw).strip()
    if not joined or joined.lower() in {"none", "null"}:
        return None
    if joined.startswith("["):
        vals = json.loads(joined)
        return {int(x) for x in vals}
    out: set[int] = set()
    for tok in re.split(r"[\s,]+", joined):
        tok = tok.strip("[]")
        if tok:
            out.add(int(tok))
    return out


def _pool_gif_path(model_tag: str, idx: int, cue: str) -> Path:
    safe = _safe_cue(cue)
    return exp1_pose_tiles_dir(model_tag) / f"mm19_g{idx:02d}_{safe}_pose_pool.gif"


def _main_gif_path(model_tag: str, idx: int, cue: str) -> Path:
    safe = _safe_cue(cue)
    return exp1_render_dir(model_tag) / f"mm19_g{idx:02d}_{safe}.gif"


def _infer_n_tiles(pool_path: Path, main_gif_path: Path) -> int:
    with Image.open(pool_path) as pool, Image.open(main_gif_path) as main:
        pw = pool.size[0]
        mw = main.size[0]
        if mw <= 0:
            raise ValueError(f"invalid main gif width: {main_gif_path}")
        if pw % mw != 0:
            n = max(1, round(pw / mw))
        else:
            n = pw // mw
        return int(n)


def _first_pool_pose_from_row(row: dict[str, Any]) -> dict[str, Any]:
    first: dict[str, Any] | None = None
    for step in row.get("movements") or []:
        if step.get("type") != "pose":
            continue
        pose = (step.get("parameters") or {}).get("pose") or {}
        if not (pose.get("arm_position") and pose.get("gripper_orientation")):
            continue
        if first is None:
            first = pose
        if str(pose.get("arm_position", "")).strip().lower() != "still":
            return pose
    return first or {}


def _pool_entries(env, row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from legacy.render_mobile_config import list_pose_bank_candidates  # noqa: WPS433

    pose = _first_pool_pose_from_row(row)
    ranked = list_pose_bank_candidates(env, pose, top_k=0, ignore_xyz=True)
    entries = list(ranked.get("entries") or [])
    if len(entries) > _MAX_MOTION_POOL_TILES:
        entries = entries[:_MAX_MOTION_POOL_TILES]
        ranked = {**ranked, "entries": entries, "n_truncated": len(ranked.get("entries") or []) - len(entries)}
    return entries, ranked


def _fallback_entries(n_tiles: int) -> list[dict[str, Any]]:
    return [{"rank": i, "tile_idx": i - 1} for i in range(1, n_tiles + 1)]


def _split_pool_gif(pool_path: Path, n_tiles: int, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with Image.open(pool_path) as im:
        duration = im.info.get("duration", 50)
        n_frames = getattr(im, "n_frames", 1)
        w, h = im.size
        if n_tiles < 1:
            raise ValueError(f"n_tiles must be >= 1, got {n_tiles}")
        cw = w // n_tiles
        if cw < 1:
            raise ValueError(f"cannot split {pool_path} ({w}x{h}) into {n_tiles} tiles")

        for t in range(n_tiles):
            frames: list[Image.Image] = []
            x0 = t * cw
            for fi in range(n_frames):
                im.seek(fi)
                frame = im.convert("RGB")
                frames.append(frame.crop((x0, 0, x0 + cw, h)))
            out = out_dir / f"tile_{t + 1:02d}.gif"
            frames[0].save(
                out,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
            )
            paths.append(out)
    return paths


def _tile_labels(entries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, ent in enumerate(entries, 1):
        rank = ent.get("rank", i)
        tile_idx = ent.get("tile_idx", "?")
        x, y, z = ent.get("x"), ent.get("y"), ent.get("z")
        xyz = f"x={x:g} y={y:g} z={z:g}" if all(v is not None for v in (x, y, z)) else "xyz=?"
        lines.append(f"- tile {i}: bank_rank={rank}, plausible_tile={tile_idx}, {xyz}")
    return "\n".join(lines)


def _fill_prompt(template: str, row: dict[str, Any], n_tiles: int, entries: list[dict[str, Any]]) -> str:
    rep = load_snippet("_shared_representative_means.txt")
    return (
        template.replace("{{N_TILES}}", str(n_tiles))
        .replace("{{REPRESENTATIVE_MEANS}}", rep)
        .replace("{{CUE}}", str(row.get("cue", "")))
        .replace("{{DESCRIPTION}}", str(row.get("description", "")))
        .replace("{{TILE_LABELS}}", _tile_labels(entries))
    )


def _gemini_client():
    from google import genai

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY.")
    return genai.Client(api_key=key)


def _vlm_pick_tiles(
    client,
    *,
    model: str,
    prompt: str,
    tile_gifs: list[Path],
) -> dict[str, Any]:
    from google.genai import types

    parts: list[Any] = []
    for p in tile_gifs:
        parts.append(types.Part.from_bytes(data=p.read_bytes(), mime_type="image/gif"))
    parts.append(prompt)
    resp = client.models.generate_content(model=model, contents=parts)
    try:
        return _extract_json(resp.text or "")
    except Exception as e:
        return {"parse_error": str(e), "raw_text": (resp.text or "")[:2000]}


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_tag = args.model_tag
    config_path = Path(args.config_json)
    rows = sorted(load_config_list(config_path), key=lambda r: int(r.get("idx", 0)))
    idx_only = _parse_idx_only(args.idx_only)
    if idx_only is not None:
        rows = [r for r in rows if int(r.get("idx", -1)) in idx_only]

    template = Path(args.prompt).read_text(encoding="utf-8")
    tile_root = Path(args.tile_export_dir)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    client = None if args.dry_run else _gemini_client()
    env = None
    if not args.no_pool_meta:
        try:
            from legacy.render_mobile_config import _make_env, list_pose_bank_candidates  # noqa: WPS433

            env = _make_env()
        except Exception as e:
            print(f"[warn] mujoco env unavailable — tile labels will be minimal: {e}", flush=True)

    results: list[dict[str, Any]] = []
    try:
        for row in rows:
            idx = int(row.get("idx", -1))
            cue = str(row.get("cue", "?"))
            safe = _safe_cue(cue)
            pool_path = _pool_gif_path(model_tag, idx, cue)
            main_path = _main_gif_path(model_tag, idx, cue)
            rec: dict[str, Any] = {
                "idx": idx,
                "cue": cue,
                "description": row.get("description"),
                "pool_gif": str(pool_path),
            }

            if not pool_path.is_file():
                rec["error"] = "missing_pose_pool_gif"
                results.append(rec)
                print(f"[skip] g{idx} {cue}: no pool gif", flush=True)
                continue

            if not main_path.is_file():
                rec["error"] = "missing_main_gif"
                results.append(rec)
                print(f"[skip] g{idx} {cue}: no main gif", flush=True)
                continue

            entries: list[dict[str, Any]]
            pool_meta: dict[str, Any] = {}
            if env is not None:
                entries, pool_meta = _pool_entries(env, row)
                n_tiles = len(entries)
            else:
                n_tiles = _infer_n_tiles(pool_path, main_path)
                entries = _fallback_entries(n_tiles)
                pool_meta = {"inferred_from_gif_width": True}
            rec["n_tiles"] = n_tiles
            rec["pool_meta"] = pool_meta

            if n_tiles < 2:
                rec["vlm_skipped"] = "single_candidate" if n_tiles == 1 else "empty_pool"
                results.append(rec)
                print(f"[skip] g{idx} {cue}: {n_tiles} tile(s) — need >= 2", flush=True)
                continue

            export_dir = tile_root / f"mm19_g{idx:02d}_{safe}"
            try:
                tile_gifs = _split_pool_gif(pool_path, n_tiles, export_dir)
            except Exception as e:
                rec["error"] = f"split_failed: {e}"
                results.append(rec)
                print(f"[skip] g{idx} {cue}: split failed — {e}", flush=True)
                continue

            rec["tile_gifs"] = [str(p.relative_to(_REPO)) for p in tile_gifs]
            rec["tile_entries"] = entries

            if args.dry_run:
                rec["dry_run"] = True
                results.append(rec)
                print(f"[dry] g{idx} {cue}: {n_tiles} tiles exported", flush=True)
                continue

            prompt = _fill_prompt(template, row, n_tiles, entries)
            parsed = _vlm_pick_tiles(client, model=args.model, prompt=prompt, tile_gifs=tile_gifs)
            pick_raw = parsed.get("best_tile_index")
            try:
                pick = int(pick_raw)
            except (TypeError, ValueError):
                pick = None
            if pick is not None:
                pick = max(1, min(pick, n_tiles))
            picked = entries[pick - 1] if pick is not None and 1 <= pick <= n_tiles else None
            rec.update(
                {
                    "vlm_result": parsed,
                    "vlm_pick_index": pick,
                    "picked_entry": picked,
                }
            )
            results.append(rec)
            print(
                f"[ok] g{idx} {cue}: pick={pick}/{n_tiles} "
                f"— {str(parsed.get('direction_orientation_assessment', ''))[:80]}",
                flush=True,
            )
    finally:
        if env is not None:
            closer = getattr(env, "close", None)
            if callable(closer):
                closer()

    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "experiment": "exp1_pose_pool_pick",
        "model": args.model,
        "model_tag": model_tag,
        "prompt": str(Path(args.prompt).relative_to(_REPO)),
        "config_json": str(config_path),
        "pose_tiles_dir": str(exp1_pose_tiles_dir(model_tag)),
        "render_dir": str(exp1_render_dir(model_tag)),
        "idx_only": sorted(idx_only) if idx_only is not None else None,
        "n_results": len(results),
        "n_vlm": sum(1 for r in results if r.get("vlm_pick_index") is not None),
        "n_skipped_single": sum(1 for r in results if r.get("vlm_skipped")),
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_json}", flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description="VLM pick best exp1 pose-pool motion tile (2+ tiles only; per-tile GIF inputs)"
    )
    ap.add_argument("--model-tag", default="gemini-2.5-pro")
    ap.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
    ap.add_argument("--config-json", type=Path, default=None)
    ap.add_argument("--prompt", type=Path, default=PROMPT_PATH)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tile-export-dir", type=Path, default=DEFAULT_TILE_EXPORT)
    ap.add_argument(
        "--idx-only",
        nargs="*",
        default=None,
        help="Limit to cue idx list, e.g. --idx-only 18 26 16 or --idx-only '[18, 26, 16]' (default: all multi-tile)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Split tiles only; skip VLM")
    ap.add_argument(
        "--no-pool-meta",
        action="store_true",
        help="Skip mujoco pose-bank lookup; infer tile count from gif widths only",
    )
    args = ap.parse_args()
    if args.config_json is None:
        args.config_json = result_config_path("1", args.model_tag)
    run(args)


if __name__ == "__main__":
    main()
