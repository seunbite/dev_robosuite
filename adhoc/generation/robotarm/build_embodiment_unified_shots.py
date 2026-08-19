#!/usr/bin/env python3
"""Build unified shot.yaml for 5 Fig1 robots from existing shot / fewshot sources.

Structure::

    humanoid:
      - cue: depict_lion
        shot_id: ...
        motion_code: "pose(...)+..."
        xyzrpy: [[x,y,z,r,p,y], ...]
        joint_angles: {joint_names: [...], joints_rad: [[...], ...]}
        render_gif: relative/path.gif
        source: path/to/origin

Also renders Motion Codes and writes an HTML gallery.
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_HERE))

from motion_code_to_config import convert_code  # noqa: E402
from run_embodiment_baseline_traj import (  # noqa: E402
    EXTRACTORS,
    RENDERERS,
    HZ,
    MAX_FRAMES,
    _round_rows,
    _subsample,
)

PLATS = ["humanoid", "tiago", "stretch", "single_arm", "roomba"]
OUT_YAML = _REPO / "data/seed/shots/embodiment_unified/shot.yaml"
OUT_DIR = _REPO / "data/results/shots/embodiment_unified"
INV_JSON = _REPO / "data/results/baseline_traj/ours_fewshot_inventory.json"


def _slug(s: str) -> str:
    s = re.sub(r"[^\w.-]+", "_", (s or "").strip()).strip("_").lower()
    return s[:80] or "shot"


def _add(
    bag: dict[str, list[dict[str, Any]]],
    *,
    platform: str,
    cue: str,
    motion_code: str,
    source: str,
    role: str = "",
    shot_id: str = "",
    notes: str = "",
) -> None:
    code = (motion_code or "").strip()
    if not code:
        return
    # normalize whitespace
    code = re.sub(r"\s+", " ", code.replace("\n", " ")).strip()
    converted = convert_code(code, platform=platform)
    if not converted.get("parse_ok"):
        # still keep, mark parse fail
        movements = []
        parse_ok = False
        notes = (notes + " parse_fail").strip()
    else:
        movements = converted.get("movements") or []
        parse_ok = True
        code = str(converted.get("source_code") or code)
    sid = shot_id or f"{platform}__{_slug(cue)}__{_slug(role or 'main')}"
    bag.setdefault(platform, []).append(
        {
            "cue": cue,
            "shot_id": sid,
            "role": role or "shot",
            "motion_code": code,
            "movements": movements,
            "parse_ok": parse_ok,
            "source": source,
            "notes": notes,
        }
    )


def collect_shots() -> dict[str, list[dict[str, Any]]]:
    bag: dict[str, list[dict[str, Any]]] = {p: [] for p in PLATS}

    # 1) fewshot_synthetic.md
    fs = (_REPO / "data/seed/agent_fill/embodiment_short/fewshot_synthetic.md").read_text(
        encoding="utf-8"
    )
    core_m = re.search(r"\*\*core:\*\*\s*`([^`]+)`", fs)
    soph_m = re.search(r"\*\*sophisticated:\*\*\s*`([^`]+)`", fs)
    if core_m:
        _add(
            bag,
            platform="roomba",
            cue="rub_eye_tired",
            motion_code=core_m.group(1),
            source="data/seed/agent_fill/embodiment_short/fewshot_synthetic.md",
            role="core",
        )
    if soph_m:
        _add(
            bag,
            platform="roomba",
            cue="rub_eye_tired",
            motion_code=soph_m.group(1),
            source="data/seed/agent_fill/embodiment_short/fewshot_synthetic.md",
            role="sophisticated",
        )

    # 2) sophistication json
    soph = json.loads(
        (
            _REPO
            / "data/seed/shots/embodiment_sophistication/roomba_xy_sway_sophistication_fewshot.json"
        ).read_text(encoding="utf-8")
    )
    for role in ("core", "sophistication"):
        block = soph.get(role) or {}
        if block.get("source_code"):
            _add(
                bag,
                platform="roomba",
                cue="rub_eye_tired",
                motion_code=str(block["source_code"]),
                source="data/seed/shots/embodiment_sophistication/roomba_xy_sway_sophistication_fewshot.json",
                role=f"sophistication_{role}",
            )

    # 3) long 1-4
    long = json.loads(
        (
            _REPO / "data/seed/shots/embodiment_long/roomba_rub_eye_tired_long_1_4.json"
        ).read_text(encoding="utf-8")
    )
    for ex in long.get("examples") or []:
        level = ex.get("level") or ex.get("id") or "long"
        cfgs = ex.get("configs") or {}
        for which in ("original", "result"):
            block = cfgs.get(which) or {}
            if block.get("source_code"):
                _add(
                    bag,
                    platform="roomba",
                    cue=str(ex.get("cue") or "rub_eye_tired"),
                    motion_code=str(block["source_code"]),
                    source="data/seed/shots/embodiment_long/roomba_rub_eye_tired_long_1_4.json",
                    role=f"{level}_{which}",
                    shot_id=str(ex.get("id") or "") + f"_{which}",
                )

    # 4) humanoid correction oneshot
    lion = json.loads(
        (
            _REPO / "data/seed/shots/embodiment_correction/humanoid_depict_lion_oneshot.json"
        ).read_text(encoding="utf-8")
    )
    for role in ("origin", "corrected"):
        block = lion.get(role) or {}
        if block.get("source_code"):
            _add(
                bag,
                platform="humanoid",
                cue=str(lion.get("cue") or "depict_lion"),
                motion_code=str(block["source_code"]),
                source="data/seed/shots/embodiment_correction/humanoid_depict_lion_oneshot.json",
                role=role,
            )

    # 5) prompt_exp0 synthetic sketches (per platform) — from inventory or parse
    if INV_JSON.is_file():
        inv = json.loads(INV_JSON.read_text(encoding="utf-8"))
        sketches = inv.get("prompt_exp0_sketches") or {}
    else:
        sketches = {}
        sketch_re = re.compile(r"^-\s+([^:]+):\s*`([^`]+)`", re.M)
        for plat in PLATS:
            text = (_REPO / f"data/seed/prompt/{plat}/exp/prompt_exp0.txt").read_text(
                encoding="utf-8"
            )
            m = re.search(
                r"Synthetic few-shot sketches.*?(?:\n\[|\nOutput |\nCue catalog|\nConstraints)",
                text,
                re.S,
            )
            block = m.group(0) if m else text
            sketches[plat] = [
                {"name": n.strip(), "source_code": c.strip()} for n, c in sketch_re.findall(block)
            ]
    for plat, items in sketches.items():
        if plat not in PLATS:
            continue
        for it in items:
            name = str(it.get("name") or "sketch")
            _add(
                bag,
                platform=plat,
                cue=_slug(name),  # sketch label as cue-ish id
                motion_code=str(it.get("source_code") or ""),
                source=f"data/seed/prompt/{plat}/exp/prompt_exp0.txt#Synthetic_few-shot_sketches",
                role="schema_sketch",
                notes=f"sketch_name={name}",
            )

    # 6) tiago diverse_shots — modern movements; recover DSL best-effort from GT if same cue, else serialize lightly
    diverse = json.loads(
        (_REPO / "data/seed/shots/tiago/diverse_shots_mobile.json").read_text(encoding="utf-8")
    )
    gt_by_cue = {}
    gt_path = (
        _REPO
        / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
    )
    if gt_path.is_file():
        for row in json.loads(gt_path.read_text(encoding="utf-8")):
            gt_by_cue[row["cue"]] = row
    for r in diverse:
        cue = str(r.get("cue") or "")
        code = ""
        g = gt_by_cue.get(cue)
        if g and (g.get("platforms") or {}).get("tiago", {}).get("source_code"):
            code = str(g["platforms"]["tiago"]["source_code"])
            src = "tiago/diverse_shots_mobile.json + GT source_code match"
        else:
            code = _movements_to_dsl_approx(r.get("movements") or [])
            src = "data/seed/shots/tiago/diverse_shots_mobile.json"
        _add(
            bag,
            platform="tiago",
            cue=cue,
            motion_code=code,
            source=src,
            role="diverse_shot",
        )

    # dedupe by (cue, motion_code)
    for plat in PLATS:
        seen = set()
        uniq = []
        for e in bag[plat]:
            key = (e["cue"], e["motion_code"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(e)
        bag[plat] = uniq
    return bag


def _movements_to_dsl_approx(movements: list[dict[str, Any]]) -> str:
    """Best-effort DSL for modern pose/move/path/hold movement dicts."""
    parts: list[str] = []
    for m in movements:
        t = str(m.get("type") or "").lower()
        params = m.get("parameters") or {}
        if t == "pose":
            pose = params.get("pose") or {}
            args = []
            mapping = {
                "arm_position": "dir",
                "dir": "dir",
                "gripper_orientation": "grip",
                "grip": "grip",
                "head": "head",
                "torso_height": "torso",
                "torso": "torso",
                "left_arm": "left_arm",
                "elbow_bended": "elbow",
                "elbow": "elbow",
                "x": "x",
                "y": "y",
                "z": "z",
            }
            for k, dsl in mapping.items():
                if k in pose and pose[k] is not None:
                    v = pose[k]
                    if isinstance(v, bool):
                        v = "T" if v else "F"
                    if isinstance(v, str) and not re.fullmatch(r"-?\d+(\.\d+)?", v):
                        args.append(f"{dsl}={v}")
                    else:
                        args.append(f"{dsl}={v}")
            parts.append(f"pose({', '.join(args)})" if args else "pose()")
        elif t in ("movement", "move"):
            mv = params.get("movement") or params
            joints = mv.get("joints") or []
            if joints and isinstance(joints[0], dict):
                j0 = joints[0]
                joint = j0.get("joint", "shoulder")
                axis = j0.get("axis", "z")
                deg = j0.get("degrees", 0)
            else:
                joint = mv.get("joint", "shoulder")
                axis = "z"
                deg = 0
                dirs = mv.get("directions") or []
                if dirs and isinstance(dirs[0], dict):
                    d0 = dirs[0].get("degrees") or {}
                    if isinstance(d0, dict) and d0:
                        axis, deg = next(iter(d0.items()))
                    elif isinstance(d0, (int, float, list)):
                        deg = d0
            rep = mv.get("repetition", params.get("repetition", 1))
            spd = mv.get("speed", params.get("speed", 1.0))
            hold = mv.get("hold_time", params.get("hold_time", 0))
            deg_s = json.dumps(deg) if isinstance(deg, list) else str(deg)
            tail = f", rep={rep}, spd={spd}"
            if hold:
                tail += f", hold={hold}"
            parts.append(f"move({joint}, {axis}, {deg_s}{tail})")
        elif t == "path":
            path = params.get("path") or params
            mode = path.get("mode", "ee")
            shape = path.get("shape", "line")
            bits = [str(mode), str(shape)]
            for k, dsl in [
                ("axis", "axis"),
                ("distance", "dist"),
                ("plane", "plane"),
                ("radius", "r"),
                ("degrees", "deg"),
                ("sweep", "sweep"),
                ("x", "x"),
                ("y", "y"),
                ("speed", "spd"),
                ("repetition", "rep"),
                ("hold_time", "hold"),
            ]:
                if k in path and path[k] is not None:
                    v = path[k]
                    bits.append(f"{dsl}={json.dumps(v) if isinstance(v, list) else v}")
            parts.append(f"path({', '.join(bits)})")
        elif t == "hold":
            dur = (
                m.get("duration")
                or params.get("t")
                or params.get("hold_time")
                or params.get("duration")
                or 1.0
            )
            parts.append(f"hold(t={dur})")
    return " + ".join(parts)


def enrich_and_render(bag: dict[str, list[dict[str, Any]]], *, force: bool = False) -> dict[str, list[dict[str, Any]]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for plat, items in bag.items():
        for e in items:
            cell = {
                "idx": 0,
                "cue": e["cue"],
                "description": e.get("notes") or e["cue"],
                "movements": e.get("movements") or [],
                "source_code": e["motion_code"],
                "parse_ok": e.get("parse_ok", True),
                "platform": plat,
            }
            # extract traj
            try:
                art = EXTRACTORS[plat](cell, hz=HZ)
                e["xyzrpy"] = art.get("xyzrpy") or []
                e["joint_angles"] = {
                    "joint_names": art.get("joint_names") or [],
                    "joints_rad": art.get("joints_rad") or [],
                }
                e["hz"] = art.get("hz", HZ)
                e["extract_ok"] = True
            except Exception as exc:  # noqa: BLE001
                e["xyzrpy"] = []
                e["joint_angles"] = {"joint_names": [], "joints_rad": []}
                e["extract_ok"] = False
                e["extract_error"] = str(exc)
                print(f"[extract] FAIL {plat}/{e['shot_id']}: {exc}", flush=True)

            # render motion_code via platform renderer using movements
            gif = OUT_DIR / "render" / plat / f"{e['shot_id']}.gif"
            if gif.is_file() and not force:
                e["render_gif"] = str(gif.relative_to(_REPO))
                e["render_ok"] = True
                continue
            if not e.get("movements"):
                e["render_ok"] = False
                e["render_error"] = "no_movements"
                continue
            try:
                # Prefer official embodiment renderers for fidelity
                rows = [
                    {
                        "idx": abs(hash(e["shot_id"])) % 90 + 1,
                        "cue": e["shot_id"][:40],
                        "description": e["cue"],
                        "movements": e["movements"],
                        "source_code": e["motion_code"],
                        "parse_ok": True,
                        "platform": plat,
                    }
                ]
                if plat == "roomba":
                    from embodiment_renderers import render_roomba

                    render_roomba(rows, OUT_DIR / "render" / "roomba", force=True, fps=HZ)
                    src = OUT_DIR / "render" / "roomba" / f"emb_g{rows[0]['idx']:02d}_{rows[0]['cue']}.gif"
                elif plat == "stretch":
                    from embodiment_renderers import render_stretch

                    render_stretch(rows, OUT_DIR / "render" / "stretch", force=True, fps=HZ)
                    src = OUT_DIR / "render" / "stretch" / f"emb_g{rows[0]['idx']:02d}_{rows[0]['cue']}.gif"
                elif plat == "humanoid":
                    from embodiment_renderers import render_unitree_g1

                    render_unitree_g1(rows, OUT_DIR / "render" / "humanoid", force=True, fps=20)
                    src = OUT_DIR / "render" / "humanoid" / f"emb_g{rows[0]['idx']:02d}_{rows[0]['cue']}.gif"
                elif plat == "single_arm":
                    from motion_code_to_config import render_manipulator

                    render_manipulator(rows, out_root=OUT_DIR / "render", force=True)
                    src = (
                        OUT_DIR
                        / "render"
                        / "manipulator"
                        / "gif"
                        / "IIWA"
                        / f"emb_g{rows[0]['idx']:02d}_{rows[0]['cue']}.gif"
                    )
                elif plat == "tiago":
                    from motion_code_to_config import render_tiago

                    render_tiago(rows, out_root=OUT_DIR / "render", force=True)
                    src = OUT_DIR / "render" / "tiago" / "gif" / f"emb_g{rows[0]['idx']:02d}_{rows[0]['cue']}.gif"
                else:
                    src = None
                if src is not None and src.is_file():
                    gif.parent.mkdir(parents=True, exist_ok=True)
                    if src.resolve() != gif.resolve():
                        gif.write_bytes(src.read_bytes())
                    e["render_gif"] = str(gif.relative_to(_REPO))
                    e["render_ok"] = True
                    print(f"[render] {plat}/{e['shot_id']}", flush=True)
                else:
                    e["render_ok"] = False
                    e["render_error"] = "missing_gif"
            except Exception as exc:  # noqa: BLE001
                e["render_ok"] = False
                e["render_error"] = str(exc)
                print(f"[render] FAIL {plat}/{e['shot_id']}: {exc}", flush=True)
    return bag


def write_yaml(bag: dict[str, list[dict[str, Any]]]) -> Path:
    out: dict[str, Any] = {
        "meta": {
            "description": "Unified embodiment shots for Fig1 robots (humanoid/tiago/stretch/single_arm/roomba)",
            "fields": ["cue", "shot_id", "role", "motion_code", "xyzrpy", "joint_angles", "render_gif", "source"],
            "hz": HZ,
            "platforms": PLATS,
        }
    }
    for plat in PLATS:
        rows = []
        for e in bag.get(plat) or []:
            rows.append(
                {
                    "cue": e["cue"],
                    "shot_id": e["shot_id"],
                    "role": e.get("role"),
                    "motion_code": e["motion_code"],
                    "xyzrpy": e.get("xyzrpy") or [],
                    "joint_angles": e.get("joint_angles")
                    or {"joint_names": [], "joints_rad": []},
                    "render_gif": e.get("render_gif") or "",
                    "source": e.get("source") or "",
                    "parse_ok": bool(e.get("parse_ok")),
                    "extract_ok": bool(e.get("extract_ok")),
                    "render_ok": bool(e.get("render_ok")),
                    "notes": e.get("notes") or "",
                }
            )
        out[plat] = rows
    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    OUT_YAML.write_text(
        yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )
    # also json sidecar for easy loading
    side = OUT_DIR / "shot.json"
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT_YAML}", flush=True)
    print(f"wrote {side}", flush=True)
    return OUT_YAML


def write_gallery(bag: dict[str, list[dict[str, Any]]]) -> Path:
    sections = []
    for plat in PLATS:
        for e in bag.get(plat) or []:
            gif = e.get("render_gif") or ""
            gif_path = _REPO / gif if gif else None
            if gif_path and gif_path.is_file():
                rel = os.path.relpath(gif_path, OUT_DIR)
                bust = int(gif_path.stat().st_mtime)
                media = f'<img src="{html_mod.escape(rel)}?v={bust}" loading="lazy">'
            else:
                media = '<p class="miss">missing render</p>'
            ja = e.get("joint_angles") or {}
            n_j = len(ja.get("joints_rad") or [])
            n_x = len(e.get("xyzrpy") or [])
            sections.append(
                f'<section class="cue"><h2>{plat} · {html_mod.escape(e["cue"])} '
                f'<span class="role">{html_mod.escape(e.get("role") or "")}</span></h2>'
                f'<p class="meta">shot_id={html_mod.escape(e["shot_id"])} · '
                f'parse={e.get("parse_ok")} extract={e.get("extract_ok")} render={e.get("render_ok")} · '
                f'frames xyzrpy={n_x} joints={n_j}</p>'
                f'<div class="grid"><div class="col"><h3>Render</h3>{media}</div>'
                f'<div class="col"><h3>Motion Code</h3><pre>{html_mod.escape(e["motion_code"])}</pre>'
                f'<p class="src">{html_mod.escape(e.get("source") or "")}</p></div></div></section>'
            )
    html_path = OUT_DIR / "gallery.html"
    n = sum(len(bag.get(p) or []) for p in PLATS)
    html_path.write_text(
        f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Unified embodiment shots</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:#111;color:#eee}}
h1{{padding:16px 24px;margin:0;border-bottom:1px solid #333}}
.meta{{padding:8px 24px;color:#9aa0a6;font-size:.85rem}}
.cue{{margin:12px 16px;padding:12px;border:1px solid #333;border-radius:10px;background:#1a1d24}}
.grid{{display:grid;grid-template-columns:1fr 1.4fr;gap:12px}}
.col h3{{font-size:.8rem;margin:0 0 6px;color:#c4c7ce}}
.col img{{width:100%;max-width:420px;aspect-ratio:1;object-fit:cover;border-radius:8px;background:#000}}
pre{{font-size:11px;white-space:pre-wrap;background:#0a0c10;padding:10px;border-radius:8px}}
.role{{color:#9ab;font-size:.85rem;font-weight:500}}
.src,.miss{{color:#f28b82;font-size:.8rem}}
.src{{color:#9aa0a6}}
@media (max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Unified embodiment shots</h1>
<p class="meta">n={n} · robots={', '.join(PLATS)} · motion_code + xyzrpy + joint_angles · YAML: data/seed/shots/embodiment_unified/shot.yaml</p>
{''.join(sections)}
</body></html>
""",
        encoding="utf-8",
    )
    print(f"gallery → {html_path}", flush=True)
    return html_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force-render", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    args = ap.parse_args()

    bag = collect_shots()
    for plat in PLATS:
        print(f"collected {plat}: {len(bag[plat])}", flush=True)
    if not args.skip_render:
        bag = enrich_and_render(bag, force=args.force_render)
    else:
        # extract only
        for plat, items in bag.items():
            for e in items:
                cell = {
                    "idx": 0,
                    "cue": e["cue"],
                    "description": e["cue"],
                    "movements": e.get("movements") or [],
                    "source_code": e["motion_code"],
                    "parse_ok": e.get("parse_ok", True),
                    "platform": plat,
                }
                try:
                    art = EXTRACTORS[plat](cell, hz=HZ)
                    e["xyzrpy"] = art.get("xyzrpy") or []
                    e["joint_angles"] = {
                        "joint_names": art.get("joint_names") or [],
                        "joints_rad": art.get("joints_rad") or [],
                    }
                    e["extract_ok"] = True
                except Exception as exc:  # noqa: BLE001
                    e["xyzrpy"] = []
                    e["joint_angles"] = {"joint_names": [], "joints_rad": []}
                    e["extract_ok"] = False
                    e["extract_error"] = str(exc)
    write_yaml(bag)
    write_gallery(bag)


if __name__ == "__main__":
    main()
