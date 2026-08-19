"""Verification qualitative grids: task-4 pose pairwise / task-10 motion pairwise."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
_ROBOTARM = _HERE.parent
_REPO = _ROBOTARM.parents[2]
for p in (_REPO, _ROBOTARM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from paper_figures._media import (  # noqa: E402
    CACHE,
    manifest90_rows,
    mp4_last_frame_png,
    repo,
)
from paper_figures.qual_generation import (  # noqa: E402
    VERIFY_CELL_SIZE,
    _cell_overlay,
    _cell_selective_verify,
    _parse_csv_ints,
    _stack,
    _stitch,
)
from pilot90_experiment_suite import MOTION_PAIRWISE_DIR, PAIRWISE_IMG_DIR  # noqa: E402

POSE_PAIRWISE_FALLBACK = _REPO / "data/results/visualize/pose_pairwise_12"
POSE_JSON_CANDIDATES = (
    _REPO / "data/results/verify/pilot90_gemini/exp04_pose_pairwise_2way.json",
    _REPO / "data/results/verify/pilot40_pose_pairwise_12_gemini.json",
)
MOTION_SPECS = MOTION_PAIRWISE_DIR / "pairwise_specs_motion_gt_correct.json"
MOTION_RESULTS = _REPO / "data/results/verify/pilot90_gemini/exp10_motion_pairwise_mp4.json"


def _font(size: int = 18) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _pilot90_idxs() -> set[int]:
    return {int(r["idx"]) for r in manifest90_rows()}


def _resolve_pair_image(path: str | Path) -> Path | None:
    p = Path(path)
    if p.is_file():
        return p
    name = p.name
    for d in (PAIRWISE_IMG_DIR, POSE_PAIRWISE_FALLBACK):
        hit = d / name
        if hit.is_file():
            return hit
    return None


def _load_pose_comparisons(json_path: Path | None = None) -> list[dict[str, Any]]:
    path = json_path
    if path is None:
        for cand in POSE_JSON_CANDIDATES:
            if cand.is_file():
                path = cand
                break
    if path is None or not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    pool = list(data.get("comparisons") or [])
    idx90 = _pilot90_idxs()
    return [c for c in pool if int(c.get("cue_idx", -1)) in idx90]


def _pose_by_cue(comparisons: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for c in comparisons:
        ci = int(c.get("cue_idx", -1))
        out.setdefault(ci, []).append(c)
    return out


def _load_motion_specs() -> list[dict[str, Any]]:
    if not MOTION_SPECS.is_file():
        return []
    return list(json.loads(MOTION_SPECS.read_text(encoding="utf-8")).get("mp4") or [])


def _motion_by_idx(specs: list[dict]) -> dict[int, dict]:
    return {int(s["idx"]): s for s in specs}


def _pose_state_label(entry: dict) -> str | None:
    gt = entry.get("gt_pose") or {}
    d, g = gt.get("dir"), gt.get("gripper_orientation")
    if d and g:
        return f"({d}, {g})"
    return None


def _motion_state_label(spec: dict) -> str | None:
    summary = str(spec.get("gt_tail_summary") or "")
    m = re.search(r"movement\s+(\w+)\s+\(([^)]+)\)", summary)
    if m:
        joint, moves = m.group(1), m.group(2)
        axes = re.findall(r"([xyz])", moves)
        axis = axes[0] if axes else "line"
        parts = [p.strip() for p in moves.split(",") if p.strip()]
        rep = "rep" if len(parts) >= 2 else "single"
        return f"({axis}, {joint}, {rep})"
    m = re.search(r"path\s+(\w+)\s+\(([^)]+)\)", summary)
    if m:
        shape, detail = m.group(1), m.group(2)
        kind = "arc" if shape in ("arc", "circle") else "line"
        plane = detail.split(",")[0].strip() if "," in detail else detail.strip()
        nums = [int(x) for x in re.findall(r"\d+", detail)]
        rep = "rep" if (nums and max(nums) >= 360) or "," in detail else "single"
        return f"({plane}, {kind}, {rep})"
    return None


def _load_pose_pair_image(entry: dict, cache: Path) -> Image.Image | None:
    cache.mkdir(parents=True, exist_ok=True)
    cue_idx = int(entry.get("cue_idx", 0))
    out = cache / f"c{cue_idx}_{Path(str(entry.get('pair_image', 'x'))).stem}.png"
    if out.is_file():
        return Image.open(out).convert("RGB")
    src = _resolve_pair_image(str(entry.get("pair_image") or ""))
    if src is None:
        return None
    img = Image.open(src).convert("RGB")
    img.save(out)
    return img


def _load_motion_pair_image(spec: dict, cache: Path) -> Image.Image | None:
    cache.mkdir(parents=True, exist_ok=True)
    idx = int(spec["idx"])
    out = cache / f"c{idx}_pair.png"
    if out.is_file():
        return Image.open(out).convert("RGB")
    rel = str(spec.get("pair_mp4") or "")
    mp4 = repo() / rel if rel else MOTION_PAIRWISE_DIR / f"{idx:03d}_{spec['cue']}_pair_axis.mp4"
    if not mp4.is_file():
        mp4 = MOTION_PAIRWISE_DIR / Path(rel).name
    frame = mp4_last_frame_png(mp4, cache / f"c{idx}_frame.png")
    if frame is None:
        return None
    img = Image.open(frame).convert("RGB")
    img.save(out)
    return img


def _pick_pose_entry(by_cue: dict[int, list[dict]], cue_idx: int, *, want_correct: bool | None) -> dict | None:
    entries = by_cue.get(cue_idx) or []
    if not entries:
        return None
    if want_correct is None:
        return entries[0]
    for e in entries:
        if bool(e.get("vlm_correct")) == want_correct:
            return e
    return entries[0]


def _pose_good_pool(by_cue: dict[int, list[dict]]) -> list[int]:
    return sorted(
        ci for ci, entries in by_cue.items() if any(e.get("vlm_correct") for e in entries)
    )


def _pose_bad_pool(by_cue: dict[int, list[dict]]) -> list[int]:
    return sorted(
        ci for ci, entries in by_cue.items() if any(not e.get("vlm_correct") for e in entries)
    )


def _motion_results_by_idx() -> dict[int, bool]:
    if not MOTION_RESULTS.is_file():
        return {}
    data = json.loads(MOTION_RESULTS.read_text(encoding="utf-8"))
    return {int(row["idx"]): bool(row.get("correct")) for row in data.get("mp4") or []}


def _motion_good_pool(specs: list[dict]) -> list[int]:
    results = _motion_results_by_idx()
    return sorted(int(s["idx"]) for s in specs if results.get(int(s["idx"])) is True)


def _motion_bad_pool(specs: list[dict]) -> list[int]:
    results = _motion_results_by_idx()
    return sorted(int(s["idx"]) for s in specs if results.get(int(s["idx"])) is False)


def _auto_pose_splits(by_cue: dict[int, list[dict]], n: int = 4) -> tuple[list[int], list[int]]:
    return _pose_good_pool(by_cue)[:n], _pose_bad_pool(by_cue)[:n]


def _auto_motion_splits(specs: list[dict], n: int = 4) -> tuple[list[int], list[int]]:
    return _motion_good_pool(specs)[:n], _motion_bad_pool(specs)[:n]


def _sample_idxs(pool: list[int], n: int, rng: random.Random) -> list[int]:
    if not pool:
        return []
    return rng.sample(pool, min(n, len(pool)))


def _random_pose_splits(by_cue: dict[int, list[dict]], n: int, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    return _sample_idxs(_pose_good_pool(by_cue), n, rng), _sample_idxs(_pose_bad_pool(by_cue), n, rng)


def _random_motion_splits(specs: list[dict], n: int, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    return _sample_idxs(_motion_good_pool(specs), n, rng), _sample_idxs(_motion_bad_pool(specs), n, rng)


def build_grid(
    *,
    mode: str,
    idx_arg: str = "all",
    ncols: int = 4,
    good_idxs: list[int] | None = None,
    bad_idxs: list[int] | None = None,
    pose_json: Path | None = None,
    out_path: Path | None = None,
    cell_size: int = VERIFY_CELL_SIZE,
    seed: int | None = None,
) -> Path:
    auto_sel = idx_arg.strip().lower() == "sel"
    selective = auto_sel or bool(good_idxs or bad_idxs)
    tag = "pose" if mode == "pose" else "motion"
    od = repo() / "data/results/paper_figures"
    od.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"verify_{tag}"

    if mode == "pose":
        comparisons = _load_pose_comparisons(pose_json)
        by_cue = _pose_by_cue(comparisons)
        if selective:
            good_idxs = good_idxs or []
            bad_idxs = bad_idxs or []
        elif idx_arg.strip().lower() == "all":
            good_idxs, bad_idxs = [], []
            idxs = sorted(by_cue)
        else:
            idxs = _parse_csv_ints(idx_arg)
            good_idxs, bad_idxs = idxs, []
            selective = False
        if selective and not good_idxs and not bad_idxs:
            if seed is not None:
                good_idxs, bad_idxs = _random_pose_splits(by_cue, ncols, seed)
            else:
                good_idxs, bad_idxs = _auto_pose_splits(by_cue, ncols)

        def _cell_for(
            cue_idx: int,
            *,
            browse: bool,
            want_correct: bool | None = None,
            mark_wrong: bool = False,
        ) -> Image.Image | None:
            entry = _pick_pose_entry(by_cue, cue_idx, want_correct=want_correct)
            if entry is None:
                print(f"[skip] pose verify missing cue_idx={cue_idx}", flush=True)
                return None
            img = _load_pose_pair_image(entry, cache)
            if img is None:
                print(f"[skip] pose pair image missing cue_idx={cue_idx}", flush=True)
                return None
            cue = str(entry.get("cue", "?"))
            if browse:
                return _cell_overlay(img, idx=cue_idx, cue=cue, size=cell_size)
            return _cell_selective_verify(
                img,
                cue=cue,
                width=cell_size,
                gt_side=str(entry.get("gt_side", "left")),
                mark_wrong=mark_wrong,
            )

        if not selective and idx_arg.strip().lower() == "all":
            if seed is not None:
                rng = random.Random(seed)
                idxs = rng.sample(idxs, len(idxs))
            cells = [_cell_for(i, browse=True) for i in idxs]
            cells = [c for c in cells if c is not None]
            grid = _stitch(cells, ncols)
            out_path = out_path or od / f"qual_{tag}_verify_all.png"
        else:
            top_cells = [
                _cell_for(
                    i,
                    browse=False,
                    want_correct=True if len(by_cue.get(i, [])) > 1 else None,
                    mark_wrong=False,
                )
                for i in (good_idxs or [])
            ]
            bot_cells = [
                _cell_for(
                    i,
                    browse=False,
                    want_correct=False if len(by_cue.get(i, [])) > 1 else None,
                    mark_wrong=True,
                )
                for i in (bad_idxs or [])
            ]
            top_cells = [c for c in top_cells if c is not None]
            bot_cells = [c for c in bot_cells if c is not None]
            top = _stitch(top_cells, ncols)
            bottom = _stitch(bot_cells, ncols)
            grid = _stack(top, bottom)
            out_path = out_path or od / f"qual_{tag}_verify_sel.png"
            if top is not None:
                top.save(od / f"good_{tag}_verify_example.png")
            if bottom is not None:
                bottom.save(od / f"bad_{tag}_verify_example.png")

    elif mode == "motion":
        specs = _load_motion_specs()
        by_idx = _motion_by_idx(specs)
        if selective:
            good_idxs = good_idxs or []
            bad_idxs = bad_idxs or []
        elif idx_arg.strip().lower() == "all":
            good_idxs, bad_idxs = [], []
            idxs = sorted(by_idx)
        else:
            idxs = _parse_csv_ints(idx_arg)
            good_idxs, bad_idxs = idxs, []
            selective = False
        if selective and not good_idxs and not bad_idxs:
            if seed is not None:
                good_idxs, bad_idxs = _random_motion_splits(specs, ncols, seed)
            else:
                good_idxs, bad_idxs = _auto_motion_splits(specs, ncols)

        def _cell_for(cue_idx: int, *, browse: bool, mark_wrong: bool = False) -> Image.Image | None:
            spec = by_idx.get(cue_idx)
            if spec is None:
                print(f"[skip] motion verify missing idx={cue_idx}", flush=True)
                return None
            img = _load_motion_pair_image(spec, cache)
            if img is None:
                print(f"[skip] motion pair mp4 missing idx={cue_idx}", flush=True)
                return None
            cue = str(spec.get("cue", "?"))
            if browse:
                return _cell_overlay(img, idx=cue_idx, cue=cue, size=cell_size)
            return _cell_selective_verify(
                img,
                cue=cue,
                width=cell_size,
                gt_side=str(spec.get("gt_side", "left")),
                mark_wrong=mark_wrong,
            )

        if not selective and idx_arg.strip().lower() == "all":
            if seed is not None:
                rng = random.Random(seed)
                idxs = rng.sample(idxs, len(idxs))
            cells = [_cell_for(i, browse=True) for i in idxs]
            cells = [c for c in cells if c is not None]
            grid = _stitch(cells, ncols)
            out_path = out_path or od / f"qual_{tag}_verify_all.png"
        else:
            top_cells = [_cell_for(i, browse=False, mark_wrong=False) for i in (good_idxs or [])]
            bot_cells = [_cell_for(i, browse=False, mark_wrong=True) for i in (bad_idxs or [])]
            top_cells = [c for c in top_cells if c is not None]
            bot_cells = [c for c in bot_cells if c is not None]
            top = _stitch(top_cells, ncols)
            bottom = _stitch(bot_cells, ncols)
            grid = _stack(top, bottom)
            out_path = out_path or od / f"qual_{tag}_verify_sel.png"
            if top is not None:
                top.save(od / f"good_{tag}_verify_example.png")
            if bottom is not None:
                bottom.save(od / f"bad_{tag}_verify_example.png")
    else:
        raise ValueError(mode)

    if grid is None:
        raise SystemExit("No panels rendered")
    grid.save(out_path)
    written = [out_path.name]
    if selective:
        written.extend([f"good_{tag}_verify_example.png", f"bad_{tag}_verify_example.png"])
    seed_note = f", seed={seed}" if seed is not None else ""
    pick_note = ""
    if selective and (good_idxs or bad_idxs):
        pick_note = f", good_idx={good_idxs or []}, bad_idx={bad_idxs or []}"
    print(
        f"Wrote {', '.join(written)} "
        f"(mode={mode}, good={len(good_idxs or [])}, bad={len(bad_idxs or [])}, "
        f"ncols={ncols}, selective={selective}{seed_note}{pick_note})",
        flush=True,
    )
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Verification qualitative grid (pose task4 / motion task10)")
    p.add_argument("--mode", choices=("pose", "motion"), required=True)
    p.add_argument("--idx", default="all", help='Cue idx list or "all" (browse grid with #idx)')
    p.add_argument("--good-idx", default=None, help="Comma-separated good examples (top row, 4 cols)")
    p.add_argument("--bad-idx", default=None, help="Comma-separated bad examples (bottom row)")
    p.add_argument("--pose-json", default=None)
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--cell-size", type=int, default=VERIFY_CELL_SIZE)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for --idx sel (or shuffled --idx all). Change seed to browse different picks.",
    )
    p.add_argument("--out", default=None)
    args = p.parse_args()
    build_grid(
        mode=args.mode,
        idx_arg=args.idx,
        ncols=args.ncols,
        good_idxs=_parse_csv_ints(args.good_idx),
        bad_idxs=_parse_csv_ints(args.bad_idx),
        pose_json=Path(args.pose_json) if args.pose_json else None,
        out_path=Path(args.out) if args.out else None,
        cell_size=args.cell_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
