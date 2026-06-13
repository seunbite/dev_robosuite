#!/usr/bin/env python3
"""N-way mobile pose tile GT identification (pilot-40 Google Robot)."""
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

from prompt_loader import exp_prompt_path, load_snippet  # noqa: E402
from pilot40_paths import GT_CONSOLIDATED, MULTITILE_IMG_DIR, TILE_DIR  # noqa: E402

DIR_TO_ARM = {
    "front": "front",
    "back": "back",
    "left": "out",
    "right": "in",
    "up": "up",
    "down": "down",
}
ALL_GROUPS = [(d, g) for d in ("front", "back", "in", "out", "up", "down") for g in ("horizontal", "vertical")]
GRID_LAYOUTS: dict[int, tuple[int, int]] = {6: (3, 2), 12: (4, 3)}


def _parse_gt_poses(groundtruth: str) -> list[tuple[str, str]]:
    return [(a.strip(), b.strip()) for a, b in re.findall(r"\(([^,]+),\s*([^)]+)\)", groundtruth)]


def _gt_arm_grip(d: str, g: str) -> tuple[str, str]:
    return DIR_TO_ARM.get(d.strip().lower(), d), g.strip().lower()


def _font(size: int = 13) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _extract_json(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        s = m.group(0)
    return json.loads(s)


def _load_group_tile(tile_dir: Path, arm: str, grip: str, tile_index: int = 1) -> Image.Image:
    path = tile_dir / f"group_{arm}_{grip}.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    img = Image.open(path).convert("RGB")
    # group images are 5x5 grids; crop tile `tile_index` (1-based, row-major)
    cols = 5
    rows = 5
    w, h = img.size
    cw, ch = w // cols, h // rows
    idx = max(0, min(tile_index - 1, cols * rows - 1))
    r, c = divmod(idx, cols)
    return img.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch))


def _stitch_grid(tiles: list[Image.Image], labels: list[str], *, cols: int, rows: int, cue: str) -> Image.Image:
    tw, th = tiles[0].size
    pad = 28
    out_w = cols * tw
    out_h = rows * th + 36
    canvas = Image.new("RGB", (out_w, out_h), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    font = _font(12)
    draw.text((8, 4), cue, fill=(230, 230, 230), font=font)
    for i, (tile, label) in enumerate(zip(tiles, labels)):
        r, c = divmod(i, cols)
        x, y = c * tw, 36 + r * th
        canvas.paste(tile, (x, y))
        draw.text((x + 4, y + 4), label, fill=(255, 220, 80), font=font)
    return canvas


def _grid_prompt(*, cue: str, description: str, n_tiles: int, cols: int, rows: int, labels: list[str]) -> str:
    exp_id = 5 if n_tiles == 6 else 6
    template = exp_prompt_path(exp_id).read_text(encoding="utf-8")
    return (
        template.replace("{{TEMPORAL_LINE}}", "")
        .replace("{{N_TILES}}", str(n_tiles))
        .replace("{{COLS}}", str(cols))
        .replace("{{ROWS}}", str(rows))
        .replace("{{CUE}}", cue)
        .replace("{{DESCRIPTION}}", description)
        .replace("{{TILE_LABELS}}", "\n".join(labels))
        .replace("{{REPRESENTATIVE_MEANS}}", load_snippet("_shared_representative_means.txt"))
    )


def _pick_groups(gt_set: set[tuple[str, str]], gt_primary: tuple[str, str], *, n: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    pool = [g for g in ALL_GROUPS if g not in gt_set]
    rng.shuffle(pool)
    chosen = list(gt_set) + pool[: max(0, n - len(gt_set))]
    if gt_primary in chosen:
        chosen.remove(gt_primary)
    chosen = [gt_primary] + chosen
    return chosen[:n]


def run(args: argparse.Namespace) -> None:
    vlm = getattr(args, "vlm", None)
    client = None
    if vlm is None:
        from google import genai

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("Set GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)

    consolidated = json.loads(Path(args.consolidated_json).read_text(encoding="utf-8"))
    rows = sorted(consolidated.get("rows") or [], key=lambda r: int(r.get("cue_idx", 0)))
    if args.cues:
        want = {c.strip() for c in args.cues.split(",") if c.strip()}
        rows = [r for r in rows if r.get("cue") in want]
    if args.max_cues:
        rows = rows[: int(args.max_cues)]

    tile_dir = Path(args.tile_dir)
    img_dir = Path(args.image_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    n_tiles = int(args.grid_sizes)
    cols, rows_n = GRID_LAYOUTS[n_tiles]

    existing: list[dict[str, Any]] = []
    if args.resume and args.out_json.is_file():
        prev = json.loads(args.out_json.read_text(encoding="utf-8"))
        existing = list(prev.get("results") or [])
    done = {r.get("cue") for r in existing if r.get("vlm_pick_index") is not None}

    results = list(existing)
    for ev in rows:
        cue = str(ev.get("cue", ""))
        if cue in done:
            continue
        gt_pairs = [_gt_arm_grip(d, g) for d, g in _parse_gt_poses(str(ev.get("groundtruth", "")))]
        if not gt_pairs:
            continue
        gt_set = set(gt_pairs)
        gt_primary = gt_pairs[0]
        seed = int(ev.get("cue_idx", 0)) * 1009 + n_tiles
        groups = _pick_groups(gt_set, gt_primary, n=n_tiles, seed=seed)

        tiles: list[Image.Image] = []
        tile_meta: list[dict[str, Any]] = []
        for i, (arm, grip) in enumerate(groups):
            try:
                tiles.append(_load_group_tile(tile_dir, arm, grip, tile_index=1))
            except Exception as e:
                results.append({"cue": cue, "grid_n": n_tiles, "error": str(e)})
                break
            tile_meta.append(
                {
                    "display_index": i + 1,
                    "arm_position": arm,
                    "gripper_orientation": grip,
                    "is_gt": (arm, grip) in gt_set,
                }
            )
        else:
            labels = [
                f"#{t['display_index']}: arm={t['arm_position']}, grip={t['gripper_orientation']}"
                for t in tile_meta
            ]
            grid_img = _stitch_grid(tiles, labels, cols=cols, rows=rows_n, cue=cue)
            img_path = img_dir / f"{int(ev.get('cue_idx', 0)):03d}_{cue}_grid{n_tiles}.png"
            grid_img.save(img_path)
            prompt = _grid_prompt(
                cue=cue,
                description=str(ev.get("description") or ""),
                n_tiles=n_tiles,
                cols=cols,
                rows=rows_n,
                labels=labels,
            )
            if vlm is not None:
                from vlm_infer_shared import parse_json_response  # noqa: WPS433

                text = vlm.generate(prompt, images=[grid_img])
                parsed = parse_json_response(text)
            else:
                resp = client.models.generate_content(model=args.model, contents=[grid_img, prompt])
                try:
                    parsed = _extract_json(resp.text or "")
                except Exception as e:
                    parsed = {"parse_error": str(e), "raw_text": (resp.text or "")[:500]}
            pick = parsed.get("best_tile_index")
            try:
                pick_i = int(pick)
            except (TypeError, ValueError):
                pick_i = None
            gt_idx = [t["display_index"] for t in tile_meta if t["is_gt"]]
            results.append(
                {
                    "cue_idx": ev.get("cue_idx"),
                    "cue": cue,
                    "groundtruth": ev.get("groundtruth"),
                    "grid_n": n_tiles,
                    "gt_indices": gt_idx,
                    "grid_image": str(img_path),
                    "vlm_result": parsed,
                    "vlm_pick_index": pick_i,
                    "vlm_correct": pick_i in gt_idx if pick_i is not None else False,
                    "tiles": tile_meta,
                }
            )
            print(f"[ok] {cue} grid{n_tiles} pick={pick_i} gt={gt_idx}")

    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "grid_sizes": n_tiles,
        "tile_dir": str(tile_dir),
        "image_dir": str(img_dir),
        "results": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out_json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--consolidated-json", type=Path, default=GT_CONSOLIDATED)
    ap.add_argument("--tile-dir", type=Path, default=TILE_DIR)
    ap.add_argument("--image-dir", type=Path, default=MULTITILE_IMG_DIR)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
    ap.add_argument("--grid-sizes", default="6")
    ap.add_argument("--cues", default=None)
    ap.add_argument("--max-cues", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
