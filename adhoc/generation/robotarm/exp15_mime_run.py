#!/usr/bin/env python3
"""Exp15 — mime20 illustrator cues: exp1-style full config generation + render."""
from __future__ import annotations

import argparse
import copy
import html
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_ROBOTARM = _HERE
for p in (_REPO, _ROBOTARM, _ROBOTARM / "legacy"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from render_paths import MANIPULATOR_RENDER_HZ  # noqa: E402, F401 — re-export for scripts

MIME_MANIFEST = _REPO / "data/seed/yml/mime20_manifest.json"
PROMPT_MANIP = _REPO / "data/seed/prompt/manipulator/exp/prompt_exp15.txt"
PROMPT_MOBILE = _REPO / "data/seed/prompt/tiago/exp/prompt_exp15.txt"
OUT_CFG_MANIP = _REPO / "data/results/motion_configs/manipulator/exp/result_exp15_gemini-2.5-pro.json"
OUT_CFG_MOBILE = _REPO / "data/results/motion_configs/tiago/exp/result_exp15_gemini-2.5-pro.json"
RENDER_MANIP = _REPO / "data/results/render/manipulator/exp15_mime20"
RENDER_MOBILE = _REPO / "data/results/render/tiago/exp15_mime20"
HTML_OUT = _REPO / "data/results/html/exp15_mime20_pilot.html"

# Faster exp15 preview renders (override via env or CLI).
EXP15_RENDER_HZ = int(os.getenv("EXP15_RENDER_HZ", "12"))
EXP15_MOBILE_FPS = int(os.getenv("EXP15_MOBILE_FPS", "12"))
EXP15_GIF_MS = int(os.getenv("EXP15_GIF_MS", "100"))
EXP15_SPEED_SCALE = float(os.getenv("EXP15_SPEED_SCALE", "1.25"))

Domain = Literal["manipulator", "tiago"]
Variant = Literal["orig", "corr"]


def _safe_cue(cue: str) -> str:
    return cue.replace("/", "_")


def mime_gif_path(domain: Domain, idx: int, cue: str, *, variant: Variant = "orig") -> Path:
    sub = "gif" if variant == "orig" else "correction/gif"
    root = RENDER_MANIP if domain == "manipulator" else RENDER_MOBILE
    if domain == "manipulator":
        return root / sub / "IIWA" / f"mime_g{idx:02d}_{cue}.gif"
    return root / sub / f"mime_g{idx:02d}_{_safe_cue(cue)}.gif"


def mime_media_path(domain: Domain, idx: int, cue: str, *, variant: Variant = "orig") -> Path:
    gif = mime_gif_path(domain, idx, cue, variant=variant)
    mp4 = gif.with_suffix(".mp4")
    if mp4.is_file():
        return mp4
    return gif


def _attach_mp4(gif_path: Path) -> Path | None:
    if not gif_path.is_file():
        return None
    from correction_lib import _ensure_mp4  # noqa: WPS433

    return _ensure_mp4(gif_path, gif_path.with_suffix(".mp4"))


def load_mime_rows(*, category: str | None = None) -> list[dict[str, Any]]:
    data = json.loads(MIME_MANIFEST.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    cats = data.get("categories") or {}
    keys = [category] if category else list(cats.keys())
    for key in keys:
        for row in cats.get(key) or []:
            item = dict(row)
            item["category"] = key
            rows.append(item)
    rows.sort(key=lambda r: int(r.get("idx", 0)))
    return rows


def _upsert_json(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if path.is_file():
        rows = json.loads(path.read_text(encoding="utf-8"))
    cue = str(row.get("cue", ""))
    rows = [r for r in rows if str(r.get("cue", "")) != cue]
    rows.append(row)
    rows.sort(key=lambda r: int(r.get("idx", 0)))
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def generate_manipulator(
    *,
    rows: list[dict[str, Any]],
    model: str,
    out_path: Path,
    delay: float,
) -> list[dict[str, Any]]:
    from config_gen_vlm import generate_exp1_row  # noqa: WPS433

    os.environ.setdefault("EXP1_REQUIRE_REASONING", "0")
    out: list[dict[str, Any]] = []
    for row in rows:
        cue = str(row["cue"])
        print(f"[exp15/manipulator] {cue} ...", flush=True)
        try:
            pred = generate_exp1_row(
                cue=cue,
                cue_idx=int(row["idx"]),
                description=str(row.get("prompt_line") or ""),
                model=model,
                backend="gemini",
                max_attempts=4,
                out_path=out_path,
                prompt_path=PROMPT_MANIP,
            )
            pred["experiment"] = "exp15"
            pred["category"] = row.get("category")
            pred["mime_prompt_line"] = row.get("prompt_line")
            _upsert_json(out_path, pred)
            out.append(pred)
            print(f"  OK ({len(pred.get('movements') or [])} steps)", flush=True)
        except Exception as exc:
            print(f"  FAIL {exc}", flush=True)
        time.sleep(delay)
    return out


def generate_tiago(
    *,
    rows: list[dict[str, Any]],
    model: str,
    out_path: Path,
    delay: float,
) -> list[dict[str, Any]]:
    _google = _REPO / "adhoc/generation/tiago"
    for p in (_google,):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from config_gen_mobile_vlm import generate_exp1_row  # noqa: WPS433

    out: list[dict[str, Any]] = []
    for row in rows:
        cue = str(row["cue"])
        print(f"[exp15/tiago] {cue} ...", flush=True)
        try:
            pred = generate_exp1_row(
                cue=cue,
                cue_idx=int(row["idx"]),
                description=str(row.get("prompt_line") or ""),
                model=model,
                backend="gemini",
                out_path=out_path,
                prompt_path=PROMPT_MOBILE,
            )
            pred["experiment"] = "exp15"
            pred["category"] = row.get("category")
            pred["mime_prompt_line"] = row.get("prompt_line")
            _upsert_json(out_path, pred)
            out.append(pred)
            print(f"  OK ({len(pred.get('movements') or [])} steps)", flush=True)
        except Exception as exc:
            print(f"  FAIL {exc}", flush=True)
        time.sleep(delay)
    return out


def _scale_row_speeds(row: dict[str, Any], scale: float) -> dict[str, Any]:
    if scale <= 1.0:
        return row
    out = copy.deepcopy(row)
    for step in out.get("movements") or []:
        params = step.get("parameters") or {}
        if "speed" in params and params["speed"] is not None:
            params["speed"] = round(float(params["speed"]) * scale, 3)
        for key in ("directions", "direction"):
            block = params.get(key)
            if isinstance(block, list):
                for ent in block:
                    if isinstance(ent, dict) and ent.get("speed") is not None:
                        ent["speed"] = round(float(ent["speed"]) * scale, 3)
            elif isinstance(block, dict) and block.get("speed") is not None:
                block["speed"] = round(float(block["speed"]) * scale, 3)
    return out


def render_manipulator(
    rows: list[dict[str, Any]],
    *,
    out_root: Path,
    cfg_path: Path,
    force: bool = False,
    hz: int = EXP15_RENDER_HZ,
    speed_scale: float = EXP15_SPEED_SCALE,
    variant: Variant = "orig",
) -> dict[str, Path]:
    import shutil

    from legacy.motion_generation_core import generate  # noqa: WPS433

    os.environ.setdefault("MANIPULATOR_POSE_XYZ_MODE", "trim")
    gif_sub = "gif" if variant == "orig" else "correction/gif"
    gif_dir = out_root / gif_sub / "IIWA"
    gif_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg = out_root / "configs" / ("_render_tmp_corr.json" if variant == "corr" else "_render_tmp.json")
    jpath = str(_REPO / "data/seed/_remainder/closest_poses_results.jsonl")
    out: dict[str, Path] = {}
    for row in rows:
        cue = str(row["cue"])
        idx = int(row["idx"])
        gif_path = gif_dir / f"mime_g{idx:02d}_{cue}.gif"
        if gif_path.is_file() and not force:
            _attach_mp4(gif_path)
            out[cue] = gif_path
            continue
        if gif_path.is_file() and force:
            gif_path.unlink()
            mp4 = gif_path.with_suffix(".mp4")
            if mp4.is_file():
                mp4.unlink()
        tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
        tmp_cfg.write_text(
            json.dumps([_scale_row_speeds(row, speed_scale)], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        scratch = out_root / "_scratch" / (f"corr_{cue}" if variant == "corr" else cue)
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            generate(
                robot="IIWA",
                cue=cue,
                cue_idx=idx,
                pose_index=None,
                jsonl_path=jpath,
                config_path=str(tmp_cfg),
                output_dir=str(scratch),
                hz=hz,
                top_k=1,
                ignore_xyz=False,
            )
            robot_dir = scratch / "IIWA"
            gifs = sorted(robot_dir.glob("*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)
            if gifs:
                shutil.copy2(gifs[0], gif_path)
                _attach_mp4(gif_path)
                out[cue] = gif_path
                tag = "corr" if variant == "corr" else "new"
                print(f"[render/manipulator/{tag}] {cue} -> {gif_path.name} (hz={hz})", flush=True)
        except Exception as exc:
            print(f"[render/manipulator] FAIL {cue}: {exc}", flush=True)
    if variant == "orig":
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def _first_pose_lacking_xyz(row: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
    for i, step in enumerate(row.get("movements") or []):
        if step.get("type") != "pose":
            continue
        pose = (step.get("parameters") or {}).get("pose") or {}
        from adhoc.generation.tiago.legacy.render_mobile_config import (  # noqa: WPS433
            pose_has_explicit_xyz,
        )

        if not pose_has_explicit_xyz(pose):
            return i, pose
    return None


def _pin_plausible_tile(row: dict[str, Any], *, pose_step_idx: int, tile_idx: int) -> dict[str, Any]:
    cfg = copy.deepcopy(row)
    pose = cfg["movements"][pose_step_idx]["parameters"]["pose"]
    pose["plausible_tile_idx"] = int(tile_idx)
    return cfg


def _save_tiled_motion_gif(
    variations: list[tuple[list[Any], str]],
    out_gif: Path,
    *,
    frame_duration_ms: int = 80,
    max_cols: int = 5,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not variations:
        return
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except OSError:
        font = ImageFont.load_default()

    label_h = 22
    tile_w, tile_h = variations[0][0][0].size
    n = len(variations)
    cols = min(n, max_cols)
    rows_n = (n + cols - 1) // cols
    total_w = cols * tile_w
    total_h = rows_n * (tile_h + label_h)
    max_frames = max(len(v[0]) for v in variations)

    combined: list[Image.Image] = []
    for fi in range(max_frames):
        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for vi, (frames, label) in enumerate(variations):
            col = vi % cols
            row_i = vi // cols
            x0 = col * tile_w
            y0 = row_i * (tile_h + label_h)
            frame = frames[fi % len(frames)]
            if frame.mode != "RGB":
                frame = frame.convert("RGB")
            if frame.size != (tile_w, tile_h):
                frame = frame.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            canvas.paste(frame, (x0, y0))
            tw = draw.textlength(label, font=font) if hasattr(draw, "textlength") else 40
            draw.text((x0 + (tile_w - tw) / 2, y0 + tile_h + 3), label, fill=(20, 20, 20), font=font)
        combined.append(canvas)

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    combined[0].save(
        str(out_gif),
        save_all=True,
        append_images=combined[1:],
        duration=frame_duration_ms,
        loop=0,
        disposal=2,
    )


def render_tiago_robot(
    rows: list[dict[str, Any]],
    *,
    out_root: Path,
    force: bool = False,
    mobile_fps: int = EXP15_MOBILE_FPS,
    gif_ms: int = EXP15_GIF_MS,
    speed_scale: float = EXP15_SPEED_SCALE,
    variant: Variant = "orig",
) -> dict[str, Path]:
    from adhoc.generation.tiago.legacy import _make_env, render_config  # noqa: WPS433
    import adhoc.generation.tiago.legacy.render_mobile_config as rmc  # noqa: WPS433

    gen = _REPO / "adhoc/generation"
    if str(gen) not in sys.path:
        sys.path.insert(0, str(gen))
    from gif_encode import save_gif_frames  # noqa: WPS433

    gif_sub = "gif" if variant == "orig" else "correction/gif"
    gif_dir = out_root / gif_sub
    gif_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    env = _make_env()
    prev_fps = rmc.FPS
    rmc.FPS = int(mobile_fps)
    try:
        for row in rows:
            cue = str(row["cue"])
            idx = int(row["idx"])
            safe = _safe_cue(cue)
            gif_path = gif_dir / f"mime_g{idx:02d}_{safe}.gif"
            if gif_path.is_file() and not force:
                _attach_mp4(gif_path)
                out[cue] = gif_path
                continue
            if gif_path.is_file() and force:
                gif_path.unlink()
                mp4 = gif_path.with_suffix(".mp4")
                if mp4.is_file():
                    mp4.unlink()
            try:
                cfg = _scale_row_speeds(row, speed_scale)
                frames = render_config(
                    cfg,
                    env=env,
                    overlay_progress_bar=True,
                    progress_bar_style="typed",
                    ignore_xyz=False,
                )
                if frames:
                    save_gif_frames(frames, gif_path, duration_ms=gif_ms)
                    _attach_mp4(gif_path)
                    out[cue] = gif_path
                    tag = "corr" if variant == "corr" else "new"
                    print(
                        f"[render/tiago/{tag}] {cue} -> {gif_path.name} "
                        f"(fps={mobile_fps}, gif_ms={gif_ms})",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[render/tiago] FAIL {cue}: {exc}", flush=True)
    finally:
        rmc.FPS = prev_fps
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()
    return out


def _gif_map_from_disk(
    rows: list[dict[str, Any]],
    *,
    domain: Domain,
    variant: Variant = "orig",
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for row in rows:
        cue = str(row["cue"])
        idx = int(row["idx"])
        p = mime_gif_path(domain, idx, cue, variant=variant)
        if p.is_file():
            out[cue] = p
    return out


def _rows_for_html(
    all_rows: list[dict[str, Any]],
    *,
    category: str | None,
    max_cues: int,
) -> list[dict[str, Any]]:
    """Pilot HTML scope (category + max_cues), independent of --cues render subset."""
    if category is None:
        return all_rows[:max_cues]
    scoped = [r for r in all_rows if r.get("category") == category]
    return scoped[:max_cues]


def _load_cfg_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {str(r["cue"]): r for r in json.loads(path.read_text(encoding="utf-8")) if r.get("cue")}


def _config_toggle(cfg: dict[str, Any] | None, *, label: str) -> str:
    if not cfg:
        body = "(missing)"
    else:
        slim = {
            k: cfg[k]
            for k in ("idx", "cue", "description", "movements", "experiment", "model", "time")
            if k in cfg
        }
        body = json.dumps(slim, indent=2, ensure_ascii=False)
    return (
        f'<details class="cfg-toggle"><summary>{html.escape(label)}</summary>'
        f'<pre class="cfg-pre">{html.escape(body)}</pre></details>'
    )


def _media_html(path: Path | None, *, html_parent: Path) -> str:
    if path is None or not path.is_file():
        return '<p class="miss">no media</p>'
    rel = html.escape(os.path.relpath(path, html_parent))
    if path.suffix.lower() == ".mp4":
        return (
            f'<video src="{rel}" controls muted loop playsinline preload="metadata"></video>'
        )
    return f'<img src="{rel}" loading="lazy">'


def _corr_summary(record: dict[str, Any] | None) -> str:
    if not record:
        return '<p class="miss">no correction</p>'
    action = str(record.get("action") or "")
    corr = record.get("correction") or {}
    q1 = html.escape(str(corr.get("correction_q1") or ""))
    q2 = html.escape(str(corr.get("correction_q2") or ""))
    badge_cls = "ok" if action == "verified_ok" else "fix" if action == "corrected" else "unk"
    badge = html.escape(action or "?")
    return (
        f'<p class="corr-badge {badge_cls}">{badge}</p>'
        f'<p class="corr-q"><b>Q1</b> {q1}</p>'
        f'<p class="corr-q"><b>Q2</b> {q2}</p>'
    )


def _load_corr_record(domain: Domain, cue: str, *, model_tag: str) -> dict[str, Any] | None:
    from correction_lib import correction_cue_dir, correction_record_path  # noqa: WPS433

    cue_dir = correction_cue_dir(domain, "15", model_tag, cue)
    path = correction_record_path(cue_dir, 1)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_one_exp15_correction(
    domain: Domain,
    row: dict[str, Any],
    *,
    model_tag: str,
    force: bool = False,
    hz: int = EXP15_RENDER_HZ,
    mobile_fps: int = EXP15_MOBILE_FPS,
    gif_ms: int = EXP15_GIF_MS,
    speed_scale: float = EXP15_SPEED_SCALE,
    force_corr_render: bool = False,
) -> dict[str, Any]:
    from correction_lib import (  # noqa: WPS433
        CorrectionResult,
        RenderArtifacts,
        _ensure_mp4,
        config_snapshot_path,
        correction_cue_dir,
        correction_indicates_verified_ok,
        correction_record_path,
        next_correction_round,
        run_correction_vlm,
        save_correction_record,
        write_config_snapshot,
    )
    from correction_prompts import row_from_parsed_config  # noqa: WPS433

    cue = str(row["cue"])
    idx = int(row["idx"])
    cfg_path = OUT_CFG_MANIP if domain == "manipulator" else OUT_CFG_MOBILE
    cue_dir = correction_cue_dir(domain, "15", model_tag, cue)
    rec_path = correction_record_path(cue_dir, 1)
    if rec_path.is_file() and not force:
        return json.loads(rec_path.read_text(encoding="utf-8"))

    orig_gif = mime_gif_path(domain, idx, cue, variant="orig")
    if not orig_gif.is_file():
        raise FileNotFoundError(f"Original render missing: {orig_gif}")
    orig_mp4 = _ensure_mp4(orig_gif, orig_gif.with_suffix(".mp4"))
    media = RenderArtifacts(gif=orig_gif, mp4=orig_mp4)

    correction = run_correction_vlm(
        domain,
        "15",
        row,
        media,
        model=model_tag,
        backend="gemini",
        inner_exp_id="15",
    )
    is_correct = correction.get("is_correct")
    if correction_indicates_verified_ok(correction):
        is_correct = True
        correction["is_correct"] = True

    round_n = next_correction_round(cue_dir)
    action = "none"
    output_row: dict[str, Any] | None = None
    config_snapshot_json: Path | None = None
    corr_gif = mime_gif_path(domain, idx, cue, variant="corr")

    if is_correct is True:
        action = "verified_ok"
        output_row = row
        corr_gif.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(orig_gif, corr_gif)
        media = RenderArtifacts(
            gif=corr_gif,
            mp4=_ensure_mp4(corr_gif, corr_gif.with_suffix(".mp4")),
        )
    else:
        config_parsed = correction.get("config_parsed")
        if not isinstance(config_parsed, dict) or not config_parsed.get("movements"):
            raise ValueError(
                f"Correction failed parse for {cue} "
                f"(parse_error={correction.get('parse_error')!r})"
            )
        output_row = row_from_parsed_config(
            domain,
            "15",
            row,
            config_parsed,
            model=model_tag,
        )
        output_row["correction_q1"] = correction.get("correction_q1")
        output_row["correction_q2"] = correction.get("correction_q2")
        if correction.get("inner_reasoning"):
            output_row["reasoning"] = correction["inner_reasoning"]
        action = "corrected"
        config_snapshot_json = config_snapshot_path(domain, "15", model_tag, cue, round_n)
        write_config_snapshot(config_snapshot_json, output_row)
        if domain == "manipulator":
            render_manipulator(
                [output_row],
                out_root=RENDER_MANIP,
                cfg_path=RENDER_MANIP / "configs" / "corr_pilot.json",
                force=force_corr_render,
                hz=hz,
                speed_scale=speed_scale,
                variant="corr",
            )
        else:
            render_tiago(
                [output_row],
                out_root=RENDER_MOBILE,
                force=force_corr_render,
                mobile_fps=mobile_fps,
                gif_ms=gif_ms,
                speed_scale=speed_scale,
                variant="corr",
            )
        if corr_gif.is_file():
            media = RenderArtifacts(
                gif=corr_gif,
                mp4=_ensure_mp4(corr_gif, corr_gif.with_suffix(".mp4")),
            )

    result = CorrectionResult(
        cue=cue,
        domain=domain,
        exp_id="15",
        model_tag=model_tag,
        round=round_n,
        parent_round=0,
        config_json=cfg_path,
        input_row=row,
        correction=correction,
        is_correct=is_correct if isinstance(is_correct, bool) else None,
        action=action,
        output_row=output_row,
        media=media,
        config_snapshot_json=config_snapshot_json,
    )
    save_correction_record(result)
    return json.loads(correction_record_path(cue_dir, round_n).read_text(encoding="utf-8"))


def run_exp15_correction_batch(
    rows: list[dict[str, Any]],
    *,
    robot: str,
    model_tag: str,
    force: bool = False,
    delay: float = 2.0,
    hz: int = EXP15_RENDER_HZ,
    mobile_fps: int = EXP15_MOBILE_FPS,
    gif_ms: int = EXP15_GIF_MS,
    speed_scale: float = EXP15_SPEED_SCALE,
    force_corr_render: bool = False,
) -> list[dict[str, Any]]:
    domains: list[Domain] = []
    if robot in ("manipulator", "both"):
        domains.append("manipulator")
    if robot in ("tiago", "both"):
        domains.append("tiago")
    cfg_maps = {
        "manipulator": _load_cfg_map(OUT_CFG_MANIP),
        "tiago": _load_cfg_map(OUT_CFG_MOBILE),
    }
    results: list[dict[str, Any]] = []
    todo: list[tuple[Domain, dict[str, Any]]] = []
    for domain in domains:
        cmap = cfg_maps[domain]
        for row in rows:
            cue = str(row["cue"])
            if cue not in cmap:
                continue
            todo.append((domain, cmap[cue]))

    print(f"[corr/exp15] {len(todo)} correction jobs", flush=True)
    for n, (domain, row) in enumerate(todo, start=1):
        cue = str(row["cue"])
        print(f"[corr/exp15 {n}/{len(todo)}] {domain} {cue} ...", flush=True)
        t0 = time.time()
        try:
            rec = run_one_exp15_correction(
                domain,
                row,
                model_tag=model_tag,
                force=force,
                hz=hz,
                mobile_fps=mobile_fps,
                gif_ms=gif_ms,
                speed_scale=speed_scale,
                force_corr_render=force_corr_render,
            )
            entry = {
                "domain": domain,
                "cue": cue,
                "ok": True,
                "action": rec.get("action"),
                "elapsed_s": round(time.time() - t0, 1),
            }
            print(f"  -> {entry['action']} ({entry['elapsed_s']}s)", flush=True)
        except Exception as exc:
            entry = {
                "domain": domain,
                "cue": cue,
                "ok": False,
                "error": str(exc),
                "elapsed_s": round(time.time() - t0, 1),
            }
            print(f"  ERROR: {exc}", flush=True)
        results.append(entry)
        time.sleep(delay)
    return results


def build_html(
    *,
    rows: list[dict[str, Any]],
    manip_gifs: dict[str, Path],
    mobile_gifs: dict[str, Path],
    manip_cfgs: dict[str, dict[str, Any]],
    mobile_cfgs: dict[str, dict[str, Any]],
    out_html: Path,
    model_tag: str = "gemini-2.5-pro",
    manip_corr_gifs: dict[str, Path] | None = None,
    mobile_corr_gifs: dict[str, Path] | None = None,
) -> Path:
    manip_corr_gifs = manip_corr_gifs or {}
    mobile_corr_gifs = mobile_corr_gifs or {}
    out_html.parent.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []
    n_ok = n_fix = 0
    for row in rows:
        cue = str(row["cue"])
        idx = int(row["idx"])
        cat = html.escape(str(row.get("category", "")))
        line = html.escape(str(row.get("prompt_line") or row.get("mime_prompt_line") or ""))
        m_gif = manip_gifs.get(cue)
        g_gif = mobile_gifs.get(cue)
        m_corr_gif = manip_corr_gifs.get(cue)
        g_corr_gif = mobile_corr_gifs.get(cue)
        m_orig = mime_media_path("manipulator", idx, cue, variant="orig") if m_gif else None
        g_orig = mime_media_path("tiago", idx, cue, variant="orig") if g_gif else None
        m_corr = mime_media_path("manipulator", idx, cue, variant="corr") if m_corr_gif else None
        g_corr = mime_media_path("tiago", idx, cue, variant="corr") if g_corr_gif else None
        m_rec = _load_corr_record("manipulator", cue, model_tag=model_tag)
        g_rec = _load_corr_record("tiago", cue, model_tag=model_tag)
        for rec in (m_rec, g_rec):
            if not rec:
                continue
            if rec.get("action") == "verified_ok":
                n_ok += 1
            elif rec.get("action") == "corrected":
                n_fix += 1
        m_cfg = _config_toggle(manip_cfgs.get(cue), label="Generated config (IIWA)")
        g_cfg = _config_toggle(mobile_cfgs.get(cue), label="Generated config (TIAGo)")
        m_corr_cfg = _config_toggle(
            (m_rec or {}).get("output_row") if m_rec and m_rec.get("action") == "corrected" else None,
            label="Corrected config (IIWA)",
        )
        g_corr_cfg = _config_toggle(
            (g_rec or {}).get("output_row") if g_rec and g_rec.get("action") == "corrected" else None,
            label="Corrected config (TIAGo)",
        )
        sections.append(
            f'<section class="cue"><h2>{idx:02d}. {html.escape(cue)} <span class="tag">{cat}</span></h2>'
            f'<p class="line">{line}</p>'
            f'<div class="grid">'
            f'<div class="col"><h3>Manipulator (IIWA)</h3>'
            f'<div class="pair"><div class="sub"><h4>Original</h4>{_media_html(m_orig if m_orig and m_orig.is_file() else m_gif, html_parent=out_html.parent)}</div>'
            f'<div class="sub"><h4>Correction</h4>{_media_html(m_corr if m_corr and m_corr.is_file() else m_corr_gif, html_parent=out_html.parent)}{_corr_summary(m_rec)}{m_corr_cfg}</div></div>'
            f"{m_cfg}</div>"
            f'<div class="col"><h3>Tiago (TIAGo)</h3>'
            f'<div class="pair"><div class="sub"><h4>Original</h4>{_media_html(g_orig if g_orig and g_orig.is_file() else g_gif, html_parent=out_html.parent)}</div>'
            f'<div class="sub"><h4>Correction</h4>{_media_html(g_corr if g_corr and g_corr.is_file() else g_corr_gif, html_parent=out_html.parent)}{_corr_summary(g_rec)}{g_corr_cfg}</div></div>'
            f"{g_cfg}</div>"
            f"</div></section>"
        )
    corr_note = f" · corr1: {n_ok} verified_ok · {n_fix} corrected" if (n_ok or n_fix) else ""
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Exp15 mime20 pilot</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e8eaed}}
h1{{padding:16px 24px;margin:0;border-bottom:1px solid #333}}
.meta{{padding:10px 24px;color:#9aa0a6;font-size:0.9rem;border-bottom:1px solid #222}}
.cue{{margin:20px 24px;padding:16px;border:1px solid #333;border-radius:10px;background:#171a21}}
.cue h2{{margin:0 0 8px;font-size:1.1rem}}
.tag{{font-size:0.75rem;color:#8ab4f8;font-weight:normal}}
.line{{margin:0 0 12px;color:#9aa0a6;font-size:0.9rem}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.col{{background:#0f1115;border-radius:8px;padding:10px}}
.col h3{{margin:0 0 8px;font-size:0.95rem;color:#bdc1c6}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:8px}}
.sub h4{{margin:0 0 6px;font-size:0.8rem;color:#9aa0a6;font-weight:600}}
img,video{{width:100%;height:auto;border-radius:6px;background:#000;margin-bottom:8px}}
.miss{{color:#f28b82;font-size:0.85rem}}
.corr-badge{{font-size:0.8rem;font-weight:600;margin:4px 0}}
.corr-badge.ok{{color:#81c995}}
.corr-badge.fix{{color:#fdd663}}
.corr-badge.unk{{color:#9aa0a6}}
.corr-q{{font-size:0.75rem;color:#9aa0a6;margin:2px 0 6px;line-height:1.35}}
.cfg-toggle{{margin-top:8px;border:1px solid #333;border-radius:8px;background:#12151c}}
.cfg-toggle summary{{cursor:pointer;padding:8px 10px;font-size:12px;font-weight:600;color:#bdc1c6;user-select:none}}
.cfg-toggle summary:hover{{background:#1a1d24}}
.cfg-pre{{margin:0;padding:10px;font-size:11px;line-height:1.4;overflow:auto;max-height:360px;background:#0f1115;border-top:1px solid #333;color:#c4c7c5;white-space:pre-wrap;word-break:break-word}}
</style></head><body>
<h1>Exp15 — Mime illustrator pilot</h1>
<p class="meta">gemini-2.5-pro · new render (xyz-trim) · MP4→VLM corr1 · {len(rows)} mime cues × 2 robots · hz/fps≈{EXP15_RENDER_HZ} · speed×{EXP15_SPEED_SCALE}{corr_note}</p>
{"".join(sections)}
</body></html>"""
    out_html.write_text(page, encoding="utf-8")
    return out_html


def _rows_missing_cfg(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    existing = _load_cfg_map(path)
    return [r for r in rows if str(r["cue"]) not in existing]


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp15 mime20 generation + render")
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--category", default="all", help="object|profession|all")
    ap.add_argument("--max-cues", type=int, default=20)
    ap.add_argument("--cues", default="", help="Comma-separated cue override")
    ap.add_argument("--robot", choices=("manipulator", "tiago", "both"), default="both")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="Skip cues already in config JSON")
    ap.add_argument("--force-render", action="store_true")
    ap.add_argument("--render-hz", type=int, default=EXP15_RENDER_HZ, help="Manipulator capture FPS")
    ap.add_argument("--mobile-fps", type=int, default=EXP15_MOBILE_FPS, help="Tiago capture FPS")
    ap.add_argument("--gif-ms", type=int, default=EXP15_GIF_MS, help="GIF frame duration (playback)")
    ap.add_argument(
        "--speed-scale",
        type=float,
        default=EXP15_SPEED_SCALE,
        help="Multiply motion speeds during render only (default 1.25)",
    )
    ap.add_argument("--top-k", type=int, default=10, help="Plausible-pool tiled GIF columns (Tiago)")
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--html", type=Path, default=HTML_OUT)
    ap.add_argument("--run-correction", action="store_true", help="Run corr1 VLM (MP4 input) + render")
    ap.add_argument("--skip-correction", action="store_true")
    ap.add_argument("--correction-force", action="store_true", help="Re-run corr1 even if record exists")
    ap.add_argument("--correction-delay", type=float, default=2.0)
    ap.add_argument("--force-corr-render", action="store_true", help="Re-render correction GIFs/MP4")
    args = ap.parse_args()

    cat = None if args.category == "all" else args.category
    all_rows = load_mime_rows(category=cat)
    if args.cues.strip():
        want = {c.strip() for c in args.cues.split(",") if c.strip()}
        rows = [r for r in all_rows if str(r["cue"]) in want]
    else:
        rows = all_rows[: args.max_cues]
    if not rows:
        raise SystemExit("No mime cues selected")

    gen_rows = rows
    if args.skip_existing:
        gen_rows = [
            r
            for r in rows
            if str(r["cue"]) not in _load_cfg_map(OUT_CFG_MANIP)
            or str(r["cue"]) not in _load_cfg_map(OUT_CFG_MOBILE)
        ]
        if not gen_rows:
            print("All selected cues already have configs.", flush=True)
        else:
            print(
                f"Generating {len(gen_rows)} cue(s) missing config "
                f"(of {len(rows)} selected).",
                flush=True,
            )

    manip_rows: list[dict[str, Any]] = []
    mobile_rows: list[dict[str, Any]] = []
    manip_cfgs: dict[str, dict[str, Any]] = {}
    mobile_cfgs: dict[str, dict[str, Any]] = {}

    if not args.skip_generate:
        if args.robot in ("manipulator", "both"):
            m_gen = (
                _rows_missing_cfg(gen_rows, OUT_CFG_MANIP)
                if args.skip_existing
                else gen_rows
            )
            if m_gen:
                manip_rows = generate_manipulator(
                    rows=m_gen, model=args.model, out_path=OUT_CFG_MANIP, delay=args.delay
                )
            manip_cfgs = _load_cfg_map(OUT_CFG_MANIP)
        if args.robot in ("tiago", "both"):
            g_gen = (
                _rows_missing_cfg(gen_rows, OUT_CFG_MOBILE)
                if args.skip_existing
                else gen_rows
            )
            if g_gen:
                mobile_rows = generate_tiago(
                    rows=g_gen, model=args.model, out_path=OUT_CFG_MOBILE, delay=args.delay
                )
            mobile_cfgs = _load_cfg_map(OUT_CFG_MOBILE)
    else:
        manip_cfgs = _load_cfg_map(OUT_CFG_MANIP)
        mobile_cfgs = _load_cfg_map(OUT_CFG_MOBILE)
        if args.robot in ("manipulator", "both"):
            manip_rows = [manip_cfgs[str(r["cue"])] for r in rows if str(r["cue"]) in manip_cfgs]
        if args.robot in ("tiago", "both"):
            mobile_rows = [mobile_cfgs[str(r["cue"])] for r in rows if str(r["cue"]) in mobile_cfgs]

    html_rows = _rows_for_html(all_rows, category=cat, max_cues=args.max_cues)
    manip_gifs = _gif_map_from_disk(html_rows, domain="manipulator", variant="orig")
    mobile_gifs = _gif_map_from_disk(html_rows, domain="tiago", variant="orig")
    if not args.skip_render:
        if manip_rows:
            manip_gifs = {
                **manip_gifs,
                **render_manipulator(
                    manip_rows,
                    out_root=RENDER_MANIP,
                    cfg_path=RENDER_MANIP / "configs" / "pilot.json",
                    force=args.force_render,
                    hz=args.render_hz,
                    speed_scale=args.speed_scale,
                    variant="orig",
                ),
            }
        if mobile_rows:
            mobile_gifs = {
                **mobile_gifs,
                **render_tiago(
                    mobile_rows,
                    out_root=RENDER_MOBILE,
                    force=args.force_render,
                    mobile_fps=args.mobile_fps,
                    gif_ms=args.gif_ms,
                    speed_scale=args.speed_scale,
                    variant="orig",
                ),
            }

    if args.run_correction and not args.skip_correction:
        run_exp15_correction_batch(
            rows,
            robot=args.robot,
            model_tag=args.model,
            force=args.correction_force,
            delay=args.correction_delay,
            hz=args.render_hz,
            mobile_fps=args.mobile_fps,
            gif_ms=args.gif_ms,
            speed_scale=args.speed_scale,
            force_corr_render=args.force_corr_render,
        )

    manip_corr_gifs = _gif_map_from_disk(html_rows, domain="manipulator", variant="corr")
    mobile_corr_gifs = _gif_map_from_disk(html_rows, domain="tiago", variant="corr")

    if html_rows and (manip_gifs or mobile_gifs):
        manip_cfgs = _load_cfg_map(OUT_CFG_MANIP)
        mobile_cfgs = _load_cfg_map(OUT_CFG_MOBILE)
        html_path = build_html(
            rows=html_rows,
            manip_gifs=manip_gifs,
            mobile_gifs=mobile_gifs,
            manip_cfgs=manip_cfgs,
            mobile_cfgs=mobile_cfgs,
            out_html=args.html,
            model_tag=args.model,
            manip_corr_gifs=manip_corr_gifs,
            mobile_corr_gifs=mobile_corr_gifs,
        )
        print(f"\nHTML: {html_path}", flush=True)
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(html_path)])
        except Exception:
            pass

    print(f"Configs: {OUT_CFG_MANIP} | {OUT_CFG_MOBILE}", flush=True)


if __name__ == "__main__":
    main()
