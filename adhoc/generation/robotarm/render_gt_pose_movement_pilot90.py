#!/usr/bin/env python3
"""Render all 90 manipulator cues: human GT tile-pick pose → component-GT movement tail."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from motion_gt_tail_builder import (  # noqa: E402
    build_config_from_resolved_pose_and_tail,
    build_tail_from_component,
    resolve_first_pose_step,
)
from pilot90_paths import POSE_CFG_LEGACY, load_gt_rows  # noqa: E402

OUT_ROOT = _REPO / "data/results/render/manipulator/gt_pose_movement_pilot90"
CFG_DIR = OUT_ROOT / "configs"
GIF_DIR = OUT_ROOT / "gif"
PNG_DIR = OUT_ROOT / "png"
OUT_HTML = _REPO / "data/results/html/manipulator/gt_pose_movement_pilot90.html"
JSONL = _REPO / "data/seed/_remainder/closest_poses_results.jsonl"
ROBOT = "IIWA"
HZ = 10


def _esc(x: object) -> str:
    import html

    return html.escape(str(x) if x is not None else "")


def _rel(path: Path | None, html_path: Path) -> str:
    if not path or not path.is_file():
        return ""
    return os.path.relpath(path, html_path.parent).replace(os.sep, "/")


def _short_pose_gt(s: str) -> str:
    import re

    s = re.sub(r"\bhorizontal\b", "h", str(s or ""))
    s = re.sub(r"\bvertical\b", "v", s)
    return re.sub(r"\s+", " ", s).strip()


def build_all_configs(motion_cfg: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    motion_by_cue = {str(r["cue"]): r for r in json.loads(motion_cfg.read_text(encoding="utf-8"))}
    configs: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for gt_row in sorted(load_gt_rows(), key=lambda r: int(r.get("cue_idx", 0))):
        cue = str(gt_row["cue"])
        idx = int(gt_row["cue_idx"])
        base = copy.deepcopy(motion_by_cue.get(cue) or {})
        if not base:
            skipped.append({"cue": cue, "reason": "missing_motion_config"})
            continue

        base["idx"] = idx
        base["cue"] = cue
        base["groundtruth"] = str(gt_row.get("pose_gt") or gt_row.get("groundtruth") or "")
        if gt_row.get("description"):
            base["description"] = gt_row["description"]

        pose_step, start_src = resolve_first_pose_step(base)
        if not pose_step:
            skipped.append({"cue": cue, "reason": "no_gt_pose"})
            continue
        start_pose = copy.deepcopy((pose_step.get("parameters") or {}).get("pose") or {})
        base["gt_fixed_first_pose"] = start_pose

        mg = gt_row.get("movement_gt") or {}
        ann = str(mg.get("annotation_raw") or "").strip()
        comp = mg.get("component")
        pose_only = bool(mg.get("always_correct") or ann == "none" or (comp or {}).get("kind") == "any")

        if pose_only:
            hold_step = copy.deepcopy(pose_step)
            hold_step.setdefault("parameters", {})["hold_time"] = 0.5
            cfg = {
                "idx": idx,
                "cue": cue,
                "description": base.get("description", ""),
                "groundtruth": base["groundtruth"],
                "gt_fixed_first_pose": start_pose,
                "start_pose_source": start_src,
                "movements": [hold_step],
                "state": "gt_pose_only",
                "generation_mode": "gt_pose_movement_pilot90",
                "movement_gt_raw": ann or "none",
            }
        else:
            tail = build_tail_from_component(comp)
            cfg = build_config_from_resolved_pose_and_tail(
                base,
                tail,
                state_tag="gt_pose_movement_pilot90",
            )
            if not cfg:
                skipped.append({"cue": cue, "reason": "build_failed"})
                continue
            cfg["movement_gt_raw"] = ann

        configs.append(cfg)

    return configs, skipped


def _pose_index(cfg: dict[str, Any]) -> int | None:
    gfp = cfg.get("gt_fixed_first_pose") or {}
    if gfp.get("pose_id") is not None:
        return int(gfp["pose_id"])
    for step in cfg.get("movements") or []:
        if step.get("type") != "pose":
            continue
        pose = (step.get("parameters") or {}).get("pose") or {}
        if pose.get("pose_id") is not None:
            return int(pose["pose_id"])
        break
    return None


def render_one(cfg: dict[str, Any], *, force: bool = False) -> tuple[Path | None, Path | None]:
    from legacy.motion_generation_core import generate  # noqa: WPS433
    from PIL import Image  # noqa: WPS433

    idx = int(cfg["idx"])
    cue = str(cfg["cue"])
    out_gif = GIF_DIR / f"{idx:03d}_{cue}.gif"
    out_png = PNG_DIR / f"{idx:03d}_{cue}.png"
    if out_gif.is_file() and out_png.is_file() and not force:
        return out_gif, out_png

    cfg_path = CFG_DIR / f"{idx:03d}_{cue}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps([cfg], indent=2, ensure_ascii=False), encoding="utf-8")

    scratch = OUT_ROOT / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    gif = generate(
        robot=ROBOT,
        cue=cue,
        cue_idx=idx,
        pose_index=_pose_index(cfg),
        jsonl_path=str(JSONL),
        config_path=str(cfg_path),
        output_dir=str(scratch),
        hz=HZ,
        top_k=1,
    )
    if not gif:
        return None, None

    src = Path(gif)
    if not src.is_file():
        return None, None

    GIF_DIR.mkdir(parents=True, exist_ok=True)
    out_gif.write_bytes(src.read_bytes())

    PNG_DIR.mkdir(parents=True, exist_ok=True)
    with Image.open(out_gif) as im:
        n = getattr(im, "n_frames", 1)
        im.seek(n // 2 if n > 1 else 0)
        im.convert("RGB").save(out_png)
    return out_gif, out_png


def build_html(configs: list[dict[str, Any]], media: dict[str, dict[str, Path | None]]) -> Path:
    cards: list[str] = []
    for cfg in configs:
        idx = int(cfg["idx"])
        cue = str(cfg["cue"])
        key = f"{idx}:{cue}"
        gif = media.get(key, {}).get("gif")
        png = media.get(key, {}).get("png")
        src = _rel(gif, OUT_HTML) if isinstance(gif, Path) else ""
        img = (
            f'<img src="{_esc(src)}" loading="lazy" alt="{_esc(cue)}" />'
            if src
            else '<div class="missing">missing</div>'
        )
        pose_gt = _short_pose_gt(str(cfg.get("groundtruth") or ""))
        mov_gt = _esc(cfg.get("movement_gt_raw") or "")
        cards.append(
            f'<article class="card">'
            f"<h2>c{idx} {_esc(cue)}</h2>"
            f'<p class="meta">pose GT: <code>{_esc(pose_gt)}</code> · mov GT: <code>{mov_gt}</code></p>'
            f"{img}"
            f"</article>"
        )

    css = """
body{font-family:system-ui,sans-serif;margin:20px;background:#f5f7fb}
header{background:#fff;border:1px solid #dce1ea;border-radius:10px;padding:14px 18px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.card{background:#fff;border:1px solid #dce1ea;border-radius:10px;padding:10px}
.card h2{font-size:13px;margin:0 0 6px}
.meta{font-size:11px;color:#555;margin:0 0 8px}
.meta code{background:#eef2f8;padding:1px 4px;border-radius:4px;font-size:10px}
img{width:100%;height:auto;border-radius:6px;background:#111;border:1px solid #ccc}
.missing{border:1px dashed #bbb;border-radius:6px;padding:24px;text-align:center;font-size:11px;color:#666}
"""
    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>GT pose + movement — manipulator pilot90</title>"
        f"<style>{css}</style></head><body>"
        "<header>"
        "<h1>GT pose → movement (manipulator pilot90)</h1>"
        f"<p>{len(configs)} cues · human tile-pick pose, then component-GT tail</p>"
        "</header>"
        f'<div class="grid">{"".join(cards)}</div>'
        "</body></html>"
    )
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(doc, encoding="utf-8")
    return OUT_HTML


def main() -> None:
    ap = argparse.ArgumentParser(description="Render GT pose + movement for pilot90")
    ap.add_argument("--motion-config", type=Path, default=POSE_CFG_LEGACY)
    ap.add_argument("--force-render", action="store_true")
    ap.add_argument("--html-only", action="store_true")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--max-cues", type=int, default=0, help="0 = all (debug only)")
    ap.add_argument("--cues", type=str, default="", help="Comma-separated cue names to render (default: all)")
    args = ap.parse_args()

    configs, skipped = build_all_configs(args.motion_config)
    all_configs = configs
    cue_filter = {c.strip() for c in args.cues.split(",") if c.strip()}
    if cue_filter:
        configs = [c for c in configs if str(c["cue"]) in cue_filter]
    if args.max_cues > 0:
        configs = configs[: args.max_cues]
    print(f"[gt render] {len(configs)} to render / {len(all_configs)} total", flush=True)
    if skipped:
        for s in skipped:
            print(f"  skip {s['cue']}: {s['reason']}", flush=True)

    manifest_path = OUT_ROOT / "manifest.json"
    media: dict[str, dict[str, Path | None]] = {}

    if not args.html_only:
        CFG_DIR.mkdir(parents=True, exist_ok=True)
        all_cfg_path = CFG_DIR / "all_pilot90.json"
        all_cfg_path.write_text(json.dumps(all_configs, indent=2, ensure_ascii=False), encoding="utf-8")

        ok = 0
        for cfg in configs:
            idx, cue = int(cfg["idx"]), str(cfg["cue"])
            key = f"{idx}:{cue}"
            gif, png = render_one(cfg, force=args.force_render)
            media[key] = {"gif": gif, "png": png}
            status = "ok" if gif else "fail"
            print(f"[{status}] {idx:03d} {cue} -> {gif}", flush=True)
            if gif:
                ok += 1

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "n_configs": len(configs),
                    "n_rendered": ok,
                    "skipped": skipped,
                    "media": {
                        k: {kk: str(vv) if vv else None for kk, vv in vv.items()}
                        for k, vv in media.items()
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[done] {ok}/{len(configs)} rendered", flush=True)

    # HTML always lists all configs; media from disk for cues not rendered this run.
    media: dict[str, dict[str, Path | None]] = {}
    for cfg in all_configs:
        idx, cue = int(cfg["idx"]), str(cfg["cue"])
        key = f"{idx}:{cue}"
        gif = GIF_DIR / f"{idx:03d}_{cue}.gif"
        png = PNG_DIR / f"{idx:03d}_{cue}.png"
        media[key] = {
            "gif": gif if gif.is_file() else None,
            "png": png if png.is_file() else None,
        }

    html = build_html(all_configs, media)
    print(f"Wrote {html}", flush=True)
    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(html)], check=False)


if __name__ == "__main__":
    main()
