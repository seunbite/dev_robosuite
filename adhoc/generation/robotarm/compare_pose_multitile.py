#!/usr/bin/env python3
"""
N-way pose tile GT identification: among 6 (3x2) or 12 (4x3) cropped tiles, can VLM pick GT?

Each tile is the human-picked representative from one (dir, gripper_orientation) group.
Tile order is shuffled per cue (seeded). Task matches pairwise: iconic/representative pose.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from compare_pose_2 import (  # noqa: E402
    ALL_GROUPS,
    REPRESENTATIVE_MEANS_LINE,
    _configs_by_cue,
    _dedupe_rows_by_cue,
    _parse_gt_poses,
)
from verify_pose_vlm import (  # noqa: E402
    _extract_json,
    _load_json,
    _load_tile_pick,
    _resolve_pose_image,
)
from vlm_batch_util import vlm_generate_texts  # noqa: E402
from vlm_client import (  # noqa: E402
    VLMClient,
    init_inprocess_engine,
    is_inprocess_backend,
    is_vllm_http_backend,
    require_vllm_server,
    vlm_batch_size,
)

CONSOLIDATED = _REPO / "data/results/verify/pilot40_pose_eval_consolidated.json"
TILE_DIR = _REPO / "data/results/visualize/pose_groups_12"
TILE_PICK = _REPO / "data/results/verify/pose_tile_pick_by_group.json"
DEFAULT_OUT = _REPO / "data/results/verify/pilot20_pose_multitile_gt_gemini.json"
DEFAULT_IMG_DIR = _REPO / "data/results/visualize/pose_multitile_gt"

GRID_LAYOUTS: dict[int, tuple[int, int]] = {
    6: (3, 2),
    12: (4, 3),
}


def _font(size: int = 13) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load_tile(
    tile_dir: Path,
    tile_pick: dict[tuple[str, str], int],
    d: str,
    g: str,
) -> Image.Image:
    group_path = tile_dir / f"group_{d}_{g}.png"
    if not group_path.is_file():
        raise FileNotFoundError(group_path)
    idx = tile_pick.get((d, g))
    if idx is None:
        raise KeyError(f"no tile pick for ({d}, {g})")
    img, _ = _resolve_pose_image(group_path, d, g, idx, None, False)
    return img


def _pick_six_groups(
    gt_set: set[tuple[str, str]],
    gt_primary: tuple[str, str],
    *,
    seed: int,
) -> list[tuple[str, str]]:
    """GT primary + 5 distractor groups not in human GT set."""
    distractors = [dg for dg in ALL_GROUPS if dg not in gt_set]
    rng = random.Random(seed)
    rng.shuffle(distractors)
    picked = [gt_primary] + distractors[:5]
    rng.shuffle(picked)
    return picked


def _stitch_grid(
    tiles: list[Image.Image],
    labels: list[str],
    *,
    cue: str,
    cols: int,
    rows: int,
    pad: int = 8,
    header_h: int = 40,
    footer_h: int = 36,
) -> Image.Image:
    if len(tiles) != len(labels):
        raise ValueError("tiles and labels length mismatch")
    tw = max(t.width for t in tiles)
    th = max(t.height for t in tiles)
    body_w = pad + cols * (tw + pad)
    body_h = header_h + pad + rows * (th + footer_h + pad)
    canvas = Image.new("RGB", (body_w, body_h), (248, 248, 252))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(15)
    label_font = _font(12)

    draw.text((pad, 10), f"cue: {cue}", fill=(20, 20, 40), font=title_font)

    for i, (tile, label) in enumerate(zip(tiles, labels)):
        rr, cc = divmod(i, cols)
        x = pad + cc * (tw + pad)
        y = header_h + pad + rr * (th + footer_h + pad)
        canvas.paste(tile, (x + (tw - tile.width) // 2, y + (th - tile.height) // 2))
        draw.rectangle(
            [x, y, x + tw, y + th],
            outline=(180, 180, 200),
            width=2,
        )
        draw.text((x + 4, y + th + 4), label, fill=(40, 40, 60), font=label_font)

    return canvas


def _grid_prompt(
    *,
    cue: str,
    description: str,
    n_tiles: int,
    cols: int,
    rows: int,
    tile_labels: list[str],
    temporal_prompt: bool = False,
) -> str:
    temporal_line = (
        "\nThis cue is TEMPORAL / rhythmic — weight whether the static pose suggests the cue's "
        "motion tempo and repetition pattern, not only final pose shape.\n"
        if temporal_prompt
        else ""
    )
    from prompt_loader import fill_template  # noqa: WPS433

    numbered = "\n".join(f"  {i + 1}. {tile_labels[i]}" for i in range(n_tiles))
    return fill_template(
        "exp05_pose_multitile_grid.txt",
        {
            "TEMPORAL_LINE": temporal_line,
            "N_TILES": str(n_tiles),
            "COLS": str(cols),
            "ROWS": str(rows),
            "REPRESENTATIVE_MEANS": REPRESENTATIVE_MEANS_LINE,
            "CUE": cue,
            "DESCRIPTION": description,
            "TILE_LABELS": numbered,
        },
    )


def _apply_multitile_vlm_result(record: dict[str, Any], text: str) -> dict[str, Any]:
    try:
        parsed = _extract_json(text)
    except Exception as e:
        parsed = {"parse_error": str(e), "raw_text": text}

    pick_raw = parsed.get("best_tile_index")
    try:
        pick_index = int(pick_raw)
    except (TypeError, ValueError):
        pick_index = None

    gt_indices = record.get("gt_indices") or []
    vlm_correct = pick_index in gt_indices if pick_index is not None else False
    picked_meta = next(
        (t for t in (record.get("tiles") or []) if t.get("display_index") == pick_index),
        None,
    )

    record["vlm_result"] = parsed
    record["vlm_pick_index"] = pick_index
    record["vlm_correct"] = vlm_correct
    if picked_meta:
        record["vlm_pick_pose"] = {
            "dir": picked_meta["dir"],
            "gripper_orientation": picked_meta["gripper_orientation"],
        }
    return record


def _flush_multitile_batch(
    batch_state: dict[str, Any],
    *,
    out_json: Path,
    args: argparse.Namespace,
    results: list[dict[str, Any]],
) -> None:
    pending = batch_state.get("pending") or []
    vlm = batch_state.get("vlm")
    if not pending or vlm is None:
        batch_state["pending"] = []
        return

    texts = vlm_generate_texts(
        vlm,
        batch_state["backend"],
        [{"prompt": item["prompt"], "images": [item["grid_img"]]} for item in pending],
    )
    for item, text in zip(pending, texts):
        rec = _apply_multitile_vlm_result(item["record"], text.strip())
        results.append(rec)
        mark = "OK" if rec.get("vlm_correct") else "MISS"
        print(
            f"[{mark}] c{rec.get('cue_idx')} {rec.get('cue')} grid{rec.get('grid_n')} "
            f"pick={rec.get('vlm_pick_index')} gt={rec.get('gt_indices')}",
            flush=True,
        )
    if not args.dry_run:
        _write_checkpoint(out_json, args, results, batch_state["rows"], batch_state["grid_sizes"])
    batch_state["pending"] = []


def _evaluate_one_grid(
    *,
    ev: dict[str, Any],
    n_tiles: int,
    tile_dir: Path,
    tile_pick: dict[tuple[str, str], int],
    cfg_by_cue: dict[str, dict[str, Any]],
    img_dir: Path,
    vlm: VLMClient | None,
    dry_run: bool,
    temporal_prompt: bool = False,
    batch_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    cue = ev["cue"]
    gt_poses = _parse_gt_poses(ev["groundtruth"])
    if not gt_poses:
        return {"cue_idx": ev.get("cue_idx"), "cue": cue, "error": "no GT poses parsed"}
    gt_set = set(gt_poses)
    gt_primary = gt_poses[0]
    cfg = cfg_by_cue.get(cue, {})
    description = cfg.get("description", ev.get("description", ""))
    cols, rows = GRID_LAYOUTS[n_tiles]
    seed = int(ev.get("cue_idx", 0)) * 10007 + n_tiles * 13 + hash(cue) % 100000

    if n_tiles == 12:
        groups = list(ALL_GROUPS)
        rng = random.Random(seed)
        rng.shuffle(groups)
    else:
        groups = _pick_six_groups(gt_set, gt_primary, seed=seed)

    tiles: list[Image.Image] = []
    tile_meta: list[dict[str, Any]] = []
    for i, (d, g) in enumerate(groups):
        try:
            img = _load_tile(tile_dir, tile_pick, d, g)
        except Exception as e:
            return {
                "cue_idx": ev.get("cue_idx"),
                "cue": cue,
                "grid_n": n_tiles,
                "error": str(e),
                "failed_group": {"dir": d, "gripper_orientation": g},
            }
        tiles.append(img)
        is_gt = (d, g) in gt_set
        tile_meta.append(
            {
                "display_index": i + 1,
                "dir": d,
                "gripper_orientation": g,
                "tile_pick": tile_pick.get((d, g)),
                "is_gt": is_gt,
            }
        )

    gt_indices = [t["display_index"] for t in tile_meta if t["is_gt"]]
    labels = [
        f"#{t['display_index']}: dir={t['dir']}, grip={t['gripper_orientation']}"
        for t in tile_meta
    ]
    grid_img = _stitch_grid(tiles, labels, cue=cue, cols=cols, rows=rows)
    img_name = f"{int(ev.get('cue_idx', 0)):03d}_{cue}_grid{n_tiles}.png"
    img_path = img_dir / img_name
    grid_img.save(img_path)

    record: dict[str, Any] = {
        "cue_idx": ev.get("cue_idx"),
        "cue": cue,
        "groundtruth": ev["groundtruth"],
        "gt_poses": [{"dir": d, "gripper_orientation": g} for d, g in gt_poses],
        "grid_n": n_tiles,
        "grid_cols": cols,
        "grid_rows": rows,
        "shuffle_seed": seed,
        "tiles": tile_meta,
        "gt_indices": gt_indices,
        "grid_image": str(img_path),
        "prompt_task": "multitile_gt_identification",
    }

    if dry_run:
        record["dry_run"] = True
        return record

    prompt = _grid_prompt(
        cue=cue,
        description=description,
        n_tiles=n_tiles,
        cols=cols,
        rows=rows,
        tile_labels=labels,
        temporal_prompt=temporal_prompt,
    )
    if batch_state is not None and vlm is not None:
        batch_state["pending"].append(
            {"record": record, "prompt": prompt, "grid_img": grid_img}
        )
        if len(batch_state["pending"]) >= int(batch_state["batch_size"]):
            _flush_multitile_batch(
                batch_state,
                out_json=batch_state["out_json"],
                args=batch_state["args"],
                results=batch_state["results"],
            )
        return None

    text = vlm.generate(prompt, images=[grid_img]) if vlm is not None else ""
    return _apply_multitile_vlm_result(record, text)


def run(args: argparse.Namespace) -> None:
    consolidated = _load_json(args.consolidated_json)
    cfg_by_cue = _configs_by_cue()
    tile_dir = Path(args.tile_dir)
    tile_pick = _load_tile_pick(args.tile_pick_json)
    img_dir = Path(args.image_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    rows = _dedupe_rows_by_cue(consolidated["rows"])
    rows = sorted(rows, key=lambda r: int(r.get("cue_idx", 0)))
    if args.cues:
        want = {x.strip() for x in args.cues.split(",") if x.strip()}
        rows = [r for r in rows if r.get("cue") in want]
    elif args.cue_indices:
        want = {int(x) for x in args.cue_indices.split(",") if x.strip()}
        rows = [r for r in rows if int(r.get("cue_idx", -1)) in want]
    if args.max_cues:
        rows = rows[: int(args.max_cues)]

    grid_sizes = [int(x) for x in args.grid_sizes.split(",") if x.strip()]
    for n in grid_sizes:
        if n not in GRID_LAYOUTS:
            raise SystemExit(f"Unsupported grid size {n}; use 6 or 12")

    existing: list[dict[str, Any]] = []
    if args.resume and out_json.is_file():
        prev = _load_json(out_json)
        existing = [
            r for r in (prev.get("results") or [])
            if r.get("vlm_correct") is not None or r.get("vlm_pick_index") is not None
        ]

    done_keys = {(r.get("cue"), r.get("grid_n")) for r in existing}

    vlm: VLMClient | None = None
    if not args.dry_run:
        if is_vllm_http_backend(args.vlm_backend):
            require_vllm_server()
        elif is_inprocess_backend(args.vlm_backend):
            init_inprocess_engine(args.vlm_backend, args.model)
        vlm = VLMClient(backend=args.vlm_backend, model=args.model)

    results: list[dict[str, Any]] = list(existing)
    batch_state: dict[str, Any] | None = None
    if vlm is not None and vlm_batch_size(args.vlm_backend) > 1:
        batch_state = {
            "pending": [],
            "batch_size": vlm_batch_size(args.vlm_backend),
            "vlm": vlm,
            "backend": args.vlm_backend,
            "out_json": out_json,
            "args": args,
            "results": results,
            "rows": rows,
            "grid_sizes": grid_sizes,
        }

    for ev in rows:
        for n_tiles in grid_sizes:
            if (ev.get("cue"), n_tiles) in done_keys:
                continue
            rec = _evaluate_one_grid(
                ev=ev,
                n_tiles=n_tiles,
                tile_dir=tile_dir,
                tile_pick=tile_pick,
                cfg_by_cue=cfg_by_cue,
                img_dir=img_dir,
                vlm=vlm,
                dry_run=args.dry_run,
                temporal_prompt=getattr(args, "temporal_prompt", False),
                batch_state=batch_state,
            )
            if rec is None:
                continue
            results.append(rec)
            if args.dry_run:
                print(f"[dry] c{rec.get('cue_idx')} {rec.get('cue')} grid{n_tiles}", flush=True)
            elif rec.get("vlm_correct") is not None:
                mark = "OK" if rec["vlm_correct"] else "MISS"
                print(
                    f"[{mark}] c{rec.get('cue_idx')} {rec.get('cue')} grid{n_tiles} "
                    f"pick={rec.get('vlm_pick_index')} gt={rec.get('gt_indices')}",
                    flush=True,
                )
            else:
                print(f"[err] c{rec.get('cue_idx')} {rec.get('cue')} grid{n_tiles}: {rec.get('error')}", flush=True)
            if not args.dry_run and batch_state is None:
                _write_checkpoint(out_json, args, results, rows, grid_sizes)

    if batch_state is not None:
        _flush_multitile_batch(
            batch_state,
            out_json=out_json,
            args=args,
            results=results,
        )

    summary: dict[str, Any] = {}
    for n in grid_sizes:
        scored = [r for r in results if r.get("grid_n") == n and "vlm_correct" in r]
        ok = sum(1 for r in scored if r.get("vlm_correct"))
        summary[f"grid_{n}"] = {
            "ok": ok,
            "n": len(scored),
            "accuracy": ok / len(scored) if scored else None,
            "random_baseline": 1 / n,
        }

    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "multitile_gt_identification",
        "vlm_backend": args.vlm_backend,
        "model": args.model,
        "dry_run": args.dry_run,
        "prompt_task": "representative_gt_tile_pick",
        "consolidated_json": str(args.consolidated_json),
        "tile_dir": str(tile_dir),
        "tile_pick_json": str(args.tile_pick_json),
        "image_dir": str(img_dir),
        "grid_sizes": grid_sizes,
        "n_cues": len(rows),
        "n_results": len(results),
        "summary": summary,
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for n in grid_sizes:
        s = summary[f"grid_{n}"]
        acc_txt = f"{100 * s['accuracy']:.1f}%" if s["accuracy"] is not None else "n/a (dry-run)"
        print(
            f"grid {n}: {s['ok']}/{s['n']} = {acc_txt} (random {100 / n:.1f}%)",
            flush=True,
        )
    print(f"\nWrote {out_json}", flush=True)


def _write_checkpoint(
    out_json: Path,
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    grid_sizes: list[int],
) -> None:
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "multitile_gt_identification",
        "vlm_backend": args.vlm_backend,
        "model": args.model,
        "partial": True,
        "grid_sizes": grid_sizes,
        "n_cues": len(rows),
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="N-way GT tile identification (6 or 12 tiles, Gemini)")
    p.add_argument("--consolidated-json", type=Path, default=CONSOLIDATED)
    p.add_argument("--tile-dir", type=Path, default=TILE_DIR)
    p.add_argument("--tile-pick-json", type=Path, default=TILE_PICK)
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMG_DIR)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model", default=None, help="Override VLM_MODEL env")
    p.add_argument(
        "--vlm-backend",
        default=os.getenv("VLM_BACKEND", "transformers"),
        choices=["transformers", "hf", "local", "vllm-local", "vllm", "openai", "qwen", "gemini"],
        help="Default transformers (HF). local=vLLM in-process; vllm/openai=HTTP server.",
    )
    p.add_argument("--grid-sizes", default="6,12", help="Comma-separated: 6 and/or 12")
    p.add_argument("--max-cues", type=int, default=20)
    p.add_argument("--cue-indices", type=str, default=None)
    p.add_argument("--cues", type=str, default=None, help="Comma-separated cue names")
    p.add_argument(
        "--temporal-prompt",
        action="store_true",
        help="Add tempo/rhythm emphasis for dynamic_temporal cues",
    )
    p.add_argument("--dry-run", action="store_true", help="Only save grid PNGs, no API")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip (cue, grid_n) pairs already scored in --out-json",
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
