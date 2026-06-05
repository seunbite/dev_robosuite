#!/usr/bin/env python3
"""
Pairwise pose comparison: human GT pose (specified tile) vs each of the other 11 groups.

For each cue, crop the selected tile from the GT (dir, orient) group and from one distractor
group, stitch side-by-side, and ask Gemini which side is a better starting pose for the cue.
Left/right order is randomized per pair (seeded) to avoid position bias.
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

from verify_pose_tiles_gemini import (  # noqa: E402
    _extract_json,
    _load_json,
    _load_tile_pick,
    _resolve_pose_image,
)
from vlm_client import VLMClient, is_local_backend, is_vllm_http_backend, require_vllm_server  # noqa: E402

REPRESENTATIVE_MEANS_LINE = (
    '   "More representative" means: which side\'s static pose (dir + gripper_orientation, and the '
    "arm configuration shown) best matches what a viewer would recognize as embodying this cue's "
    "meaning — as an iconic snapshot of the gesture, NOT whether it is a convenient starting pose "
    "for a follow-on motion sequence."
)

CONSOLIDATED = _REPO / "data/results/verify/pilot40_pose_eval_consolidated.json"
CONFIG_PATHS = [
    _REPO / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot10.json",
    _REPO / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot20_more.json",
]
TILE_DIR = _REPO / "data/results/visualize/pose_groups_12"
TILE_PICK = _REPO / "data/results/verify/pose_tile_pick_by_group.json"
DEFAULT_OUT = _REPO / "data/results/verify/pilot40_pose_pairwise_12_gemini.json"
DEFAULT_IMG_DIR = _REPO / "data/results/visualize/pose_pairwise_12"

ALL_GROUPS: list[tuple[str, str]] = [
    (d, g)
    for d in ("front", "back", "left", "right", "up", "down")
    for g in ("horizontal", "vertical")
]


def _dedupe_rows_by_cue(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        c = r.get("cue", "")
        if c in seen:
            continue
        seen.add(c)
        out.append(r)
    return out


def _pick_distractor_both_differ(
    gd: str,
    gg: str,
    gt_set: set[tuple[str, str]],
    *,
    seed: int,
) -> tuple[str, str] | None:
    """One distractor: dir and gripper_orientation both differ from GT primary."""
    candidates = [
        (d, g)
        for d, g in ALL_GROUPS
        if (d, g) not in gt_set and d != gd and g != gg
    ]
    if not candidates:
        return None
    rng = random.Random(seed)
    return rng.choice(candidates)


def _parse_gt_poses(groundtruth: str) -> list[tuple[str, str]]:
    return [(a.strip(), b.strip()) for a, b in re.findall(r"\(([^,]+),\s*([^)]+)\)", groundtruth)]


def _configs_by_cue() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in CONFIG_PATHS:
        for row in _load_json(p):
            out[row["cue"]] = row
    return out


def _pair_prompt(
    *,
    cue: str,
    description: str,
    left_d: str,
    left_g: str,
    right_d: str,
    right_g: str,
) -> str:
    return f"""
You are comparing two robot poses for the SAME iconic gesture cue.
The image shows two cropped robot renders side by side (LEFT and RIGHT). Each side is one
sample pose from a (dir, gripper_orientation) group. Labels below describe geometry only.

Task: Which side is MORE REPRESENTATIVE of this cue — which static pose would a human viewer
most readily associate with the gesture's meaning?
{REPRESENTATIVE_MEANS_LINE}
Do NOT judge based on which pose is easier to start a motion from, or what movement should come next.

Definitions (same for both sides):
Coordinate frame: +x forward (viewer), +y robot left, +z up. EE pointing axis = wrist → fingertips.
- dir: dominant world direction of pointing axis (front|back|left|right|up|down).
- gripper_orientation: projected jaw-opening line on plane ⊥ to dir — horizontal (ㅡ) or vertical (|).

Target cue:
- cue: {cue}
- description: {description}

Side labels (geometry only — do NOT assume either is "ground truth"):
- LEFT: dir={left_d}, gripper_orientation={left_g}
- RIGHT: dir={right_d}, gripper_orientation={right_g}

Return ONLY strict JSON:
{{
  "better_side": "left" or "right",
  "direction_orientation_assessment": "string: which side is more iconic/representative of the cue and why",
  "confidence": 0.0
}}
""".strip()


def _font(size: int = 14) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _stitch_pair(
    left_img: Image.Image,
    right_img: Image.Image,
    *,
    cue: str,
    left_label: str,
    right_label: str,
    pad: int = 8,
    header_h: int = 36,
    footer_h: int = 44,
) -> Image.Image:
    lh, rh = left_img.height, right_img.height
    h = max(lh, rh)
    lw, rw = left_img.width, right_img.width
    body_w = pad + lw + pad + rw + pad
    canvas_h = header_h + pad + h + pad + footer_h
    canvas = Image.new("RGB", (body_w, canvas_h), (248, 248, 252))
    draw = ImageDraw.Draw(canvas)
    font = _font(13)
    title_font = _font(15)

    draw.text((pad, 8), f"cue: {cue}", fill=(20, 20, 40), font=title_font)

    y0 = header_h + pad
    canvas.paste(left_img, (pad, y0 + (h - lh) // 2))
    canvas.paste(right_img, (pad + lw + pad, y0 + (h - rh) // 2))

    mid_x = pad + lw + pad // 2
    draw.line([(mid_x, y0), (mid_x, y0 + h)], fill=(180, 180, 200), width=2)

    fy = header_h + pad + h + pad + 6
    draw.text((pad, fy), f"LEFT — {left_label}", fill=(30, 90, 30), font=font)
    draw.text((pad + lw + pad, fy), f"RIGHT — {right_label}", fill=(120, 40, 40), font=font)

    return canvas


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
    if args.cues:
        want_cues = {x.strip() for x in args.cues.split(",") if x.strip()}
        rows = [r for r in rows if r.get("cue") in want_cues]
    elif args.cue_indices:
        want = {int(x) for x in args.cue_indices.split(",") if x.strip()}
        rows = [r for r in rows if int(r.get("cue_idx", -1)) in want]
    if args.exclude_cues:
        skip = {x.strip() for x in args.exclude_cues.split(",") if x.strip()}
        rows = [r for r in rows if r.get("cue") not in skip]
    if args.max_cues:
        rows = rows[: int(args.max_cues)]

    replace_set: set[str] = set()
    if args.replace_cues and args.cues:
        replace_set = {x.strip() for x in args.cues.split(",") if x.strip()}

    existing: list[dict[str, Any]] = []
    if out_json.is_file() and (args.append_results or replace_set):
        prev = _load_json(out_json)
        existing = list(prev.get("comparisons") or [])
        if replace_set:
            existing = [c for c in existing if c.get("cue") not in replace_set]
        elif args.append_results and not args.cues and not args.cue_indices:
            have = {c["cue"] for c in existing if c.get("cue")}
            rows = [r for r in rows if r.get("cue") not in have]

    vlm: VLMClient | None = None
    if not args.dry_run:
        if is_vllm_http_backend(args.vlm_backend):
            require_vllm_server()
        elif is_local_backend(args.vlm_backend):
            from vllm_local import get_vllm_engine

            get_vllm_engine(model=args.model)
        vlm = VLMClient(backend=args.vlm_backend, model=args.model)

    comparisons: list[dict[str, Any]] = list(existing)
    one_pair = args.one_pair_per_cue or (args.max_pairs_per_cue == 1)

    for ev in rows:
        cue = ev["cue"]
        gt_poses = _parse_gt_poses(ev["groundtruth"])
        if not gt_poses:
            continue
        gt_primary = gt_poses[0]
        gt_set = set(gt_poses)
        cfg = cfg_by_cue.get(cue, {})
        description = cfg.get("description", ev.get("description", ""))

        if one_pair:
            picked = _pick_distractor_both_differ(
                gt_primary[0],
                gt_primary[1],
                gt_set,
                seed=int(ev.get("cue_idx", 0)) * 9973 + hash(cue) % 100000,
            )
            distractors = [picked] if picked else []
        else:
            distractors = [dg for dg in ALL_GROUPS if dg not in gt_set]
            if args.max_pairs_per_cue:
                distractors = distractors[: int(args.max_pairs_per_cue)]

        for wrong_dg in distractors:
            if wrong_dg is None:
                comparisons.append(
                    {
                        "cue_idx": ev.get("cue_idx"),
                        "cue": cue,
                        "gt_pose": {"dir": gt_primary[0], "gripper_orientation": gt_primary[1]},
                        "error": "no distractor with both dir and grip different from GT",
                    }
                )
                continue
            wd, wg = wrong_dg
            gd, gg = gt_primary
            rng = random.Random(int(ev.get("cue_idx", 0)) * 1000 + hash(wrong_dg) % 10000)
            gt_on_left = rng.random() < 0.5

            try:
                gt_img = _load_tile(tile_dir, tile_pick, gd, gg)
                wrong_img = _load_tile(tile_dir, tile_pick, wd, wg)
            except Exception as e:
                comparisons.append(
                    {
                        "cue_idx": ev.get("cue_idx"),
                        "cue": cue,
                        "gt_pose": {"dir": gd, "gripper_orientation": gg},
                        "wrong_pose": {"dir": wd, "gripper_orientation": wg},
                        "error": str(e),
                    }
                )
                continue

            if gt_on_left:
                left_img, right_img = gt_img, wrong_img
                left_d, left_g, right_d, right_g = gd, gg, wd, wg
                gt_side = "left"
            else:
                left_img, right_img = wrong_img, gt_img
                left_d, left_g, right_d, right_g = wd, wg, gd, gg
                gt_side = "right"

            pair_img = _stitch_pair(
                left_img,
                right_img,
                cue=cue,
                left_label=f"dir={left_d}, grip={left_g}",
                right_label=f"dir={right_d}, grip={right_g}",
            )
            img_name = f"{int(ev.get('cue_idx', 0)):03d}_{cue}_gt{gd}_{gg}_vs_{wd}_{wg}.png"
            img_path = img_dir / img_name
            pair_img.save(img_path)

            record: dict[str, Any] = {
                "cue_idx": ev.get("cue_idx"),
                "cue": cue,
                "groundtruth": ev["groundtruth"],
                "gt_pose": {"dir": gd, "gripper_orientation": gg},
                "wrong_pose": {"dir": wd, "gripper_orientation": wg},
                "gt_side": gt_side,
                "left_pose": {"dir": left_d, "gripper_orientation": left_g},
                "right_pose": {"dir": right_d, "gripper_orientation": right_g},
                "pair_image": str(img_path),
                "tile_gt": tile_pick.get((gd, gg)),
                "tile_wrong": tile_pick.get((wd, wg)),
                "prompt_task": "representative",
            }

            if args.dry_run:
                record["dry_run"] = True
                comparisons.append(record)
                print(f"[dry] {cue} gt={gd},{gg} vs {wd},{wg} -> {img_path.name}", flush=True)
                continue

            prompt = _pair_prompt(
                cue=cue,
                description=description,
                left_d=left_d,
                left_g=left_g,
                right_d=right_d,
                right_g=right_g,
            )
            resp_text = vlm.generate(prompt, images=[pair_img])
            text = resp_text.strip()
            try:
                parsed = _extract_json(text)
            except Exception as e:
                parsed = {"parse_error": str(e), "raw_text": text}

            better = str(parsed.get("better_side", "")).lower().strip()
            vlm_correct = better == gt_side
            record["vlm_result"] = parsed
            record["vlm_better_side"] = better
            record["vlm_correct"] = vlm_correct
            comparisons.append(record)
            mark = "OK" if vlm_correct else "MISS"
            print(
                f"[{mark}] {cue} gt={gd},{gg} vs {wd},{wg} "
                f"gt_side={gt_side} vlm={better}",
                flush=True,
            )
            _write_checkpoint(out_json, args, comparisons)

    ok = sum(1 for c in comparisons if c.get("vlm_correct"))
    scored = sum(1 for c in comparisons if "vlm_correct" in c)
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": (
            "pairwise_one_distractor_both_differ"
            if one_pair
            else "pairwise_12groups_gt_tile_vs_distractor"
        ),
        "vlm_backend": args.vlm_backend,
        "model": args.model or os.getenv("VLM_MODEL", "gemini-2.5-pro"),
        "dry_run": args.dry_run,
        "one_pair_per_cue": one_pair,
        "distractor_rule": "dir and gripper_orientation both differ from GT primary"
        if one_pair
        else "all non-GT groups",
        "prompt_task": "representative",
        "consolidated_json": str(args.consolidated_json),
        "tile_dir": str(tile_dir),
        "tile_pick_json": str(args.tile_pick_json),
        "image_dir": str(img_dir),
        "n_cues": len(rows),
        "n_comparisons": len(comparisons),
        "n_scored": scored,
        "accuracy": ok / scored if scored else None,
        "comparisons": comparisons,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_json} ({len(comparisons)} pairs, accuracy={payload['accuracy']})", flush=True)


def _write_checkpoint(out_json: Path, args: argparse.Namespace, comparisons: list) -> None:
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "pairwise_12groups_gt_tile_vs_distractor",
        "model": args.model,
        "partial": True,
        "n_comparisons": len(comparisons),
        "comparisons": comparisons,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Pairwise GT pose vs 11 distractors (Gemini VLM)")
    p.add_argument("--consolidated-json", type=Path, default=CONSOLIDATED)
    p.add_argument("--tile-dir", type=Path, default=TILE_DIR)
    p.add_argument("--tile-pick-json", type=Path, default=TILE_PICK)
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMG_DIR)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model", default=None, help="Override VLM_MODEL env")
    p.add_argument(
        "--vlm-backend",
        default=os.getenv("VLM_BACKEND", "local"),
        choices=["local", "vllm-local", "vllm", "openai", "qwen", "gemini"],
    )
    p.add_argument("--dry-run", action="store_true", help="Only save stitched PNGs, no API")
    p.add_argument("--max-cues", type=int, default=None)
    p.add_argument("--max-pairs-per-cue", type=int, default=None)
    p.add_argument(
        "--one-pair-per-cue",
        action="store_true",
        help="At most one comparison per cue; distractor must differ in BOTH dir and grip",
    )
    p.add_argument(
        "--append-results",
        action="store_true",
        help="Merge with existing --out-json and skip cues already present",
    )
    p.add_argument(
        "--replace-cues",
        action="store_true",
        help="With --cues: drop prior comparisons for those cue names before adding new ones",
    )
    p.add_argument(
        "--exclude-cues",
        type=str,
        default=None,
        help="Comma-separated cue names to skip",
    )
    p.add_argument("--cue-indices", type=str, default=None, help="Comma-separated cue_idx filter")
    p.add_argument("--cues", type=str, default=None, help="Comma-separated cue names (overrides --cue-indices)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
