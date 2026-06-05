#!/usr/bin/env python3
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
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from vlm_client import VLMClient, is_local_backend, is_vllm_http_backend, require_vllm_server  # noqa: E402

# Must match generate_pose_group_tiles.py layout
_GRID_PAD = 6
_GRID_HEADER = 58
_GRID_COLS = 3

# Default human picks (1-based); override via --tile-pick-json
APPROPRIATE_MEANS_LINE = (
    '   "Appropriate" means: for this robot to perform the cue, if it starts from this pose '
    "(these dir and gripper_orientation labels), can a motion that conveys the cue's meaning "
    "be created using simple subsequent movements?"
)

DEFAULT_TILE_PICK: dict[tuple[str, str], int] = {
    ("front", "horizontal"): 5,
    ("front", "vertical"): 3,
    ("back", "horizontal"): 4,
    ("back", "vertical"): 5,
    ("down", "horizontal"): 6,
    ("down", "vertical"): 2,
    ("left", "horizontal"): 2,
    ("left", "vertical"): 3,
    ("right", "horizontal"): 5,
    ("right", "vertical"): 4,
    ("up", "horizontal"): 9,
    ("up", "vertical"): 2,
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_tile_pick(path: Path | None) -> dict[tuple[str, str], int]:
    if path is None:
        return dict(DEFAULT_TILE_PICK)
    data = _load_json(path)
    picks = data.get("picks", data)
    out: dict[tuple[str, str], int] = {}
    for key, idx in picks.items():
        if isinstance(key, str) and "_" in key:
            d, g = key.split("_", 1)
            out[(d, g)] = int(idx)
    if not out:
        raise SystemExit(f"No picks found in {path}")
    return out


def _first_pose(row: dict[str, Any]) -> dict[str, Any]:
    for step in row.get("movements", []):
        if step.get("type") == "pose":
            return step.get("parameters", {}).get("pose", {}) or {}
    return {}


def _movement_summary(row: dict[str, Any]) -> str:
    chunks: list[str] = []
    for step in row.get("movements", []):
        t = step.get("type")
        p = step.get("parameters", {})
        if t == "movement":
            joint = p.get("joint", "?")
            rep = p.get("repetition", 1)
            dparts = []
            for d in p.get("directions", []):
                deg = d.get("degrees", {}) or {}
                if deg:
                    dparts.append(",".join(f"{k}:{v}" for k, v in deg.items()))
            deg_txt = " | ".join(dparts) if dparts else "-"
            chunks.append(f"movement(joint={joint}, rep={rep}, deg={deg_txt})")
        elif t == "path":
            chunks.append(
                "path("
                f"shape={p.get('shape', '?')}, joint={p.get('joint', '?')}, "
                f"axis={p.get('axis', p.get('plane', '?'))}, speed={p.get('speed', '?')}"
                ")"
            )
    return " -> ".join(chunks) if chunks else "(no movement)"


def _fewshot_block(shots: list[dict[str, Any]], n: int = 4) -> str:
    picked = [r for r in shots if r.get("state") in ("handmade", "choreography")][:n]
    lines: list[str] = []
    for i, r in enumerate(picked, 1):
        p = _first_pose(r)
        lines.append(
            f"[EX{i}] cue={r.get('cue')} | pose(dir={p.get('dir')}, grip={p.get('gripper_orientation')})\n"
            f"desc={r.get('description', '')}\n"
            f"movement={_movement_summary(r)}"
        )
    return "\n\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        s = m.group(0)
    return json.loads(s)


def _infer_cell_size(canvas_w: int, canvas_h: int, n_cells: int = 9) -> tuple[int, int]:
    cols = _GRID_COLS
    rows = (n_cells + cols - 1) // cols
    for cw in range(200, 900):
        if _GRID_PAD + cols * (cw + _GRID_PAD) == canvas_w:
            ch = (canvas_h - _GRID_HEADER - _GRID_PAD - rows * _GRID_PAD) // rows
            if ch > 0:
                return cw, ch
    cw = (canvas_w - _GRID_PAD * (cols + 1)) // cols
    y0 = _GRID_HEADER + _GRID_PAD
    ch = (canvas_h - y0 - _GRID_PAD * (rows + 1)) // rows
    return cw, ch


def _crop_tile_from_group(group_img: Image.Image, tile_index_1based: int) -> Image.Image:
    if not 1 <= tile_index_1based <= 9:
        raise ValueError(f"tile_index must be 1..9, got {tile_index_1based}")
    im = group_img.convert("RGB")
    cw, ch = _infer_cell_size(im.width, im.height)
    i = tile_index_1based - 1
    rr, cc = divmod(i, _GRID_COLS)
    x = _GRID_PAD + cc * (cw + _GRID_PAD)
    y = _GRID_HEADER + _GRID_PAD + rr * (ch + _GRID_PAD)
    return im.crop((x, y, x + cw, y + ch))


def _selected_tile_path(
    selected_dir: Path,
    d: str,
    g: str,
    tile_index: int,
) -> Path:
    return selected_dir / f"group_{d}_{g}_tile{tile_index:02d}.png"


def _resolve_pose_image(
    group_path: Path,
    d: str,
    g: str,
    tile_index: int,
    selected_dir: Path | None,
    export_selected: bool,
) -> tuple[Image.Image, Path | None]:
    group_img = Image.open(group_path).convert("RGB")
    cell = _crop_tile_from_group(group_img, tile_index)
    saved: Path | None = None
    if export_selected and selected_dir is not None:
        selected_dir.mkdir(parents=True, exist_ok=True)
        saved = _selected_tile_path(selected_dir, d, g, tile_index)
        if not saved.is_file():
            cell.save(saved)
    return cell, saved


def _prompt(cue_row: dict[str, Any], fewshot_text: str) -> str:
    p = _first_pose(cue_row)
    return f"""
You are verifying robot gesture pose suitability from a single rendered robot image.
The image shows one representative pose from group (dir={p.get('dir')}, gripper_orientation={p.get('gripper_orientation')}).

Task:
1) Judge whether current pose labels (dir + gripper_orientation) are appropriate for this cue.
{APPROPRIATE_MEANS_LINE}
2) If appropriate: propose next movement sequence guidance.
3) If not appropriate: propose corrected dir + gripper_orientation and explain.

Definitions:
Coordinate frame (world): +x = forward toward the human viewer, +y = robot left, +z = up (ceiling). The EE is the parallel-jaw gripper. Its **pointing axis** runs wrist → fingertips (palm normal / approach direction).
1. Direction (`dir`) — where the EE **points** in 3D (NOT gesture category):
Choose `dir` only from the dominant world direction of that pointing axis:
- **up**: pointing toward the ceiling (+z). Example: fingertips up, “air quotes” beside the head.
- **down**: pointing toward the floor (−z). Example: palm-down press, point at the ground.
- **front**: pointing toward the viewer (+x). Example: reach to shake hands, offer object forward.
- **back**: pointing toward the robot body (−x). Example: hand on own chest, retract toward torso.
- **left** / **right**: pointing toward the robot’s left (+y) / right (−y). Example: temple tap on one side.

2. Gripper orientation (`gripper_orientation`):
The **end line** is the jaw-opening line (between the two fingertip tips). Take the plane **perpendicular** to the pointing axis (`dir`). **Orthogonally project** the end line onto that plane. An observer stands **facing that plane** (looking straight at it). Read the projected line:
- **horizontal**: projected line is a **ㅡ** (left–right).
- **vertical**: projected line is a **|** (up–down).


Few-shot movement style examples:
{fewshot_text}

Target cue:
- cue: {cue_row.get("cue")}
- description: {cue_row.get("description", "")}
- current pose: dir={p.get("dir")}, gripper_orientation={p.get("gripper_orientation")}
- current movement summary: {_movement_summary(cue_row)}

Return ONLY strict JSON:
{{
  "pose_is_appropriate": true/false,
  "direction_orientation_assessment": "string",
  "if_appropriate": {{
    "recommended_movement_plan": [
      "step guidance 1",
      "step guidance 2",
      "step guidance 3"
    ]
  }},
  "if_not_appropriate": {{
    "recommended_dir": "front|back|left|right|up|down",
    "recommended_gripper_orientation": "horizontal|vertical",
    "why_change": "string",
    "recommended_movement_plan_after_change": [
      "step guidance 1",
      "step guidance 2",
      "step guidance 3"
    ]
  }},
  "confidence": 0.0
}}
""".strip()


def _write_checkpoint(
    out_json: Path,
    args: argparse.Namespace,
    tile_pick: dict[tuple[str, str], int],
    selected_dir: Path | None,
    results: list[dict[str, Any]],
) -> None:
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "single_selected_tile",
        "vlm_backend": args.vlm_backend,
        "model": args.model,
        "config_json": str(args.config_json),
        "tile_dir": str(args.tile_dir),
        "tile_pick_json": str(args.tile_pick_json) if args.tile_pick_json else None,
        "tile_pick": {f"{d}_{g}": i for (d, g), i in sorted(tile_pick.items())},
        "selected_tile_dir": str(selected_dir) if selected_dir else None,
        "total": len(results),
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if is_vllm_http_backend(args.vlm_backend):
        require_vllm_server()
    elif is_local_backend(args.vlm_backend):
        from vllm_local import get_vllm_engine

        get_vllm_engine(model=args.model)
    vlm = VLMClient(backend=args.vlm_backend, model=args.model)

    cfg_rows = _load_json(args.config_json)
    shots = _load_json(args.shots_json)
    tile_dir = Path(args.tile_dir)
    tile_pick = _load_tile_pick(args.tile_pick_json)
    selected_dir = Path(args.selected_tile_dir) if args.export_selected else None
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    fewshot_text = _fewshot_block(shots, n=args.fewshot_n)

    results: list[dict[str, Any]] = []
    for row in sorted(cfg_rows, key=lambda x: int(x.get("idx", 0))):
        pose = _first_pose(row)
        d = pose.get("dir")
        g = pose.get("gripper_orientation")
        key_dg = (d, g)
        if key_dg not in tile_pick:
            results.append(
                {
                    "idx": row.get("idx"),
                    "cue": row.get("cue"),
                    "error": f"no tile pick for ({d}, {g})",
                }
            )
            continue

        tile_index = tile_pick[key_dg]
        group_path = tile_dir / f"group_{d}_{g}.png"
        if not group_path.is_file():
            results.append(
                {
                    "idx": row.get("idx"),
                    "cue": row.get("cue"),
                    "error": f"missing group tile image: {group_path}",
                }
            )
            continue

        try:
            img, saved_path = _resolve_pose_image(
                group_path,
                d,
                g,
                tile_index,
                selected_dir,
                export_selected=args.export_selected,
            )
        except Exception as e:
            results.append(
                {
                    "idx": row.get("idx"),
                    "cue": row.get("cue"),
                    "error": f"crop failed: {e}",
                }
            )
            continue

        prompt = _prompt(row, fewshot_text)
        text = vlm.generate(prompt, images=[img])
        try:
            parsed = _extract_json(text)
        except Exception as e:
            parsed = {"parse_error": str(e), "raw_text": text}

        results.append(
            {
                "idx": row.get("idx"),
                "cue": row.get("cue"),
                "current_dir": d,
                "current_gripper_orientation": g,
                "group_tile_image": str(group_path),
                "tile_index": tile_index,
                "selected_tile_image": str(saved_path) if saved_path else None,
                "result": parsed,
            }
        )
        print(f"[ok] idx={row.get('idx')} cue={row.get('cue')} tile={tile_index}", flush=True)
        if not getattr(args, "no_checkpoint", False):
            _write_checkpoint(out_json, args, tile_pick, selected_dir, results)

    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "single_selected_tile",
        "vlm_backend": args.vlm_backend,
        "model": args.model,
        "config_json": str(args.config_json),
        "tile_dir": str(args.tile_dir),
        "tile_pick_json": str(args.tile_pick_json) if args.tile_pick_json else None,
        "tile_pick": {f"{d}_{g}": i for (d, g), i in sorted(tile_pick.items())},
        "selected_tile_dir": str(selected_dir) if selected_dir else None,
        "total": len(results),
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Pose Verification — single selected tile",
        "",
        f"- backend: `{args.vlm_backend}`",
        f"- model: `{args.model}`",
        f"- config: `{args.config_json}`",
        f"- group tiles: `{args.tile_dir}`",
        f"- tile picks: `{args.tile_pick_json or 'built-in defaults'}`",
        "",
        "| idx | cue | tile | pose ok? | note |",
        "|---:|---|---:|---|---|",
    ]
    for r in results:
        if "error" in r:
            md_lines.append(f"| {r.get('idx')} | {r.get('cue')} | - | - | {r['error']} |")
            continue
        rr = r.get("result", {})
        ok = rr.get("pose_is_appropriate")
        note = rr.get("direction_orientation_assessment", "")
        note = str(note).replace("|", "/")
        md_lines.append(
            f"| {r.get('idx')} | {r.get('cue')} | {r.get('tile_index')} | {ok} | {note[:140]} |"
        )

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verify poses with one human-selected tile per (dir, gripper_orientation) group.",
    )
    ap.add_argument(
        "--config-json",
        type=Path,
        default=Path("data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot10.json"),
    )
    ap.add_argument(
        "--shots-json",
        type=Path,
        default=Path("data/seed/shots/manipulator/shot_configs_v19_sophisticated.json"),
    )
    ap.add_argument(
        "--tile-dir",
        type=Path,
        default=Path("data/results/visualize/pose_groups_12"),
    )
    ap.add_argument(
        "--tile-pick-json",
        type=Path,
        default=Path("data/results/verify/pose_tile_pick_by_group.json"),
    )
    ap.add_argument(
        "--selected-tile-dir",
        type=Path,
        default=Path("data/results/visualize/pose_groups_12_selected"),
        help="Where to export cropped single-tile PNGs (with --export-selected).",
    )
    ap.add_argument(
        "--export-selected",
        action="store_true",
        help="Save cropped single-tile PNGs under --selected-tile-dir.",
    )
    ap.add_argument(
        "--vlm-backend",
        default=os.getenv("VLM_BACKEND", "local"),
        choices=["local", "vllm-local", "vllm", "openai", "qwen", "gemini"],
        help="Default local (in-process vLLM). gemini=Google API.",
    )
    ap.add_argument("--model", type=str, default=None, help="Override VLM_MODEL env")
    ap.add_argument("--fewshot-n", type=int, default=4)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/results/verify/pose_tile_verify_single_gemini.json"),
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=Path("data/results/verify/pose_tile_verify_single_gemini.md"),
    )
    ap.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Skip per-cue JSON checkpoint writes (useful when disk is tight).",
    )
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
