#!/usr/bin/env python3
"""Synth60 Motion Codes → Tiago + Paddlebot (gcr_alpha1) GIFs.

Motion Code recipe (user):
  - orientation: dir + grip HV only (no palm direction)
  - location: Tiago xyz from cue_xyz_60_humanoid_tiago.json
  - movement: same concrete exec tails as synth60 G1 (auto_exec_move)

Outputs:
  data/results/motion_configs/embodiment_motion_code_synth60_tiago60/
  data/results/render/embodiment_motion_code_synth60_tiago60/{tiago,gcr_alpha1}/gif/
  motion_code/synth60_tiago60_gallery.html
  motion_code/synth60_paddlebot60_gallery.html
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

os.environ.setdefault("MUJOCO_GL", "glfw")

MC = Path(__file__).resolve().parent
PAPER = MC.parent
REPO = PAPER.parent.parent
ADHOC = REPO / "adhoc/generation/robotarm"
ODDKIN = PAPER / "synthetic_morphology"
sys.path[:0] = [str(MC), str(PAPER), str(REPO), str(ADHOC), str(ODDKIN)]

from render_synth60_motion60 import (  # noqa: E402
    CUES60,
    auto_exec_move,
    cue_indices,
    semantic_blurb,
)
from score_humanlike_cuespec import REQUIRED  # noqa: E402
from motion_code_to_config import convert_code  # noqa: E402
from run_embodiment_exp0_gemini import gif_path_for, render_pending_once  # noqa: E402

OUT_TAG = "embodiment_motion_code_synth60_tiago60"
OUT_CFG = REPO / "data/results/motion_configs" / OUT_TAG
OUT_RENDER = REPO / "data/results/render" / OUT_TAG
OUT_HTML_TIAGO = MC / "synth60_tiago60_gallery.html"
OUT_HTML_PADDLE = MC / "synth60_paddlebot60_gallery.html"
OUT_MANIFEST = OUT_CFG / "manifest.json"
XYZ_JSON = MC / "data" / "cue_xyz_60_humanoid_tiago.json"

_HV = {"horiz", "horizontal", "vert", "vertical"}
_PALM = {"palmfront", "palmback", "palmup", "palmdown", "palmin", "palmout"}


def _strip_palm_grip(grip_raw: str) -> str:
    """Keep only HV gripper token; drop palm direction."""
    parts = [p.strip().lower() for p in re.split(r"[|/]", str(grip_raw or "")) if p.strip()]
    hv = next((p for p in parts if p in _HV), None)
    if hv is None:
        return "vert"
    return "horiz" if hv.startswith("horiz") else "vert"


def _parse_pose_fields(pose_spec: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = re.search(r"dir=([a-zA-Z_]+)", pose_spec or "")
    if m:
        out["dir"] = m.group(1).lower()
    m = re.search(r"grip=([^,\)]+)", pose_spec or "")
    if m:
        out["grip_raw"] = m.group(1).strip()
        out["grip"] = _strip_palm_grip(m.group(1))
    m = re.search(r"left_arm=([a-zA-Z_]+)", pose_spec or "")
    if m:
        out["left_arm"] = m.group(1).lower()
    m = re.search(r"head=([a-zA-Z_]+)", pose_spec or "")
    if m:
        out["head"] = m.group(1).lower()
    m = re.search(r"torso=([a-zA-Z_]+)", pose_spec or "")
    if m:
        out["torso"] = m.group(1).lower()
    return out


def _tiago_xyz(cue: str, xyz_db: dict[str, Any]) -> dict[str, int]:
    row = (xyz_db.get("cues") or xyz_db).get(cue) or {}
    t = row.get("tiago") if isinstance(row, dict) else None
    out: dict[str, int] = {}
    if isinstance(t, dict):
        for a in ("x", "y", "z"):
            if isinstance(t.get(a), (int, float)):
                out[a] = int(round(float(t[a])))
    # fill missing from REQUIRED humanoid pose numbers if any
    if len(out) < 3:
        pose = str(REQUIRED[cue].get("pose") or "")
        for a in ("x", "y", "z"):
            if a in out:
                continue
            m = re.search(rf"{a}=([-\d.]+)", pose)
            if m:
                out[a] = int(round(float(m.group(1))))
    return out


# Tiago-only motion overrides (keep G1 auto_exec_move unchanged).
TIAGO_EXEC_OVERRIDES: dict[str, str] = {
    # longer throat-cut sweep across the neck line
    "throat_cut_stop": "path(ee, line, axis=y, dist=0.34, spd=3.0)\n+ hold(t=0.6)",
    # lean: drive base forward (torso-x maps poorly on Tiago)
    "lean_forward_interest": "path(base, line, axis=x, dist=0.28, spd=0.85)\n+ hold(t=1.0)",
    # frustration: drop torso then throw hands down
    "frustration_down_throw": (
        "move(torso, height, low, spd=0.85)\n"
        "+ move(shoulder, z, [0,-28], rep=1, spd=2.0, hold=0.1)\n"
        "+ hold(t=0.8)"
    ),
    # rub arms: lateral scrub along upper arms (y), not vertical z
    "rub_arms_cold": "move(shoulder, y, [-22,22], rep=4, spd=2.0, hold=0.06)\n+ hold(t=0.8)",
    # Tiago EE arc path is unstable → approximate “crazy temple” with wrist yz scrub
    "circle_temple_crazy": (
        "move(wrist, y, [-14,14], rep=3, spd=2.2, hold=0.05)\n"
        "+ move(wrist, z, [-10,10], rep=3, spd=2.2, hold=0.05)\n"
        "+ hold(t=0.4)"
    ),
}


def build_tiago_source_code(cue: str, xyz_db: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    fields = _parse_pose_fields(str(REQUIRED[cue].get("pose") or ""))
    xyz = _tiago_xyz(cue, xyz_db)
    parts: list[str] = []
    if fields.get("dir"):
        parts.append(f"dir={fields['dir']}")
    parts.append(f"grip={fields.get('grip', 'vert')}")
    for a in ("x", "y", "z"):
        if a in xyz:
            parts.append(f"{a}={xyz[a]}")
    if fields.get("left_arm"):
        parts.append(f"left_arm={fields['left_arm']}")
    if fields.get("head"):
        parts.append(f"head={fields['head']}")
    if fields.get("torso"):
        parts.append(f"torso={fields['torso']}")
    else:
        parts.append("torso=mid")

    pose_code = f"pose({', '.join(parts)})"
    exec_tail = TIAGO_EXEC_OVERRIDES.get(cue) or auto_exec_move(cue)
    chunks = [pose_code]
    if exec_tail:
        chunks.append(exec_tail)
    if "hold(" not in exec_tail:
        chunks.append("hold(t=0.8)")
    code = "\n+ ".join(chunks)
    meta = {
        "dir": fields.get("dir"),
        "grip": fields.get("grip"),
        "grip_raw": fields.get("grip_raw"),
        "left_arm": fields.get("left_arm"),
        "xyz": xyz,
        "exec_move": exec_tail,
        "semantic": semantic_blurb(cue),
        "platform": "tiago",
    }
    return code, meta


def build_paddle_source_code(cue: str, xyz_db: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Rematerialize same attributes onto Paddlebot (gcr_alpha1).

    - arm dir → claw=dir (primary); left_arm=mirror → paddle=dir too
    - grip HV → palm=edge|flat (schema aliases for vert|horiz); no palmfront/…
    - xyz = Tiago location percentiles
    - movement: map shoulder/wrist → claw (and paddle if mirrored)
    """
    fields = _parse_pose_fields(str(REQUIRED[cue].get("pose") or ""))
    xyz = _tiago_xyz(cue, xyz_db)
    arm_dir = fields.get("dir") or "front"
    hv = fields.get("grip") or "vert"
    palm = "edge" if hv == "vert" else "flat"
    mirror = fields.get("left_arm") == "mirror"

    parts = [f"dir=front", f"claw={arm_dir}", f"palm={palm}", "grip=open", "torso=mid"]
    if mirror:
        parts.append(f"paddle={arm_dir}")
    else:
        parts.append("paddle=still")
    for a in ("x", "y", "z"):
        if a in xyz:
            parts.append(f"{a}={xyz[a]}")

    # Adapt movement tokens onto claw/paddle channels
    exec_tail = auto_exec_move(cue)
    adapted = _adapt_move_for_paddle(exec_tail, mirror=mirror)
    chunks = [f"pose({', '.join(parts)})"]
    if adapted:
        chunks.append(adapted)
    if "hold(" not in adapted:
        chunks.append("hold(t=0.8)")
    code = "\n+ ".join(chunks)
    meta = {
        "dir": arm_dir,
        "palm_hv": palm,
        "mirror": mirror,
        "xyz": xyz,
        "exec_move": adapted,
        "semantic": semantic_blurb(cue),
        "platform": "gcr_alpha1",
    }
    return code, meta


def _adapt_move_for_paddle(exec_tail: str, *, mirror: bool) -> str:
    if not exec_tail or exec_tail.strip().startswith("hold("):
        return exec_tail
    lines = []
    for line in exec_tail.split("\n"):
        s = line.strip().lstrip("+").strip()
        if not s:
            continue
        if s.startswith("hold(") or s.startswith("path(base"):
            lines.append(s)
            continue
        if s.startswith("move(head"):
            lines.append(s)  # head exists on α1
            continue
        if s.startswith("move(torso"):
            # torso x/z → pitch; keep magnitude loosely
            if "z" in s and ("-" in s or "−" in s):
                lines.append("move(torso, pitch, [0,20], rep=1, spd=1.4, hold=0.2)")
            else:
                lines.append("move(torso, pitch, [0,-15], rep=1, spd=1.2, hold=0.3)")
            continue
        if s.startswith("path(ee"):
            if "arc" in s:
                lines.append("path(ee_claw, arc, plane=yz, r=0.06, sweep=360, rep=2, spd=1.6)")
            elif "axis=y" in s:
                lines.append("path(ee_claw, line, axis=y, dist=0.12, spd=2.0)")
            elif "axis=z" in s:
                lines.append("path(ee_claw, line, axis=z, dist=-0.08, spd=2.0)")
            else:
                lines.append("path(ee_claw, line, axis=x, dist=0.10, spd=2.2)")
            continue
        # joint moves → claw (and paddle if mirror)
        m = re.match(r"move\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*(.+)\)$", s)
        if m:
            joint, axis, rest = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            # wrist → wrist_c roll/yaw; shoulder/elbow → claw pitch/yaw
            if joint == "wrist":
                ch = "wrist_c"
                ax = "roll" if axis in ("z", "x") else "yaw"
            else:
                ch = "claw"
                ax = "yaw" if axis == "y" else "pitch"
            lines.append(f"move({ch}, {ax}, {rest}")
            if mirror:
                ch2 = "wrist_p" if joint == "wrist" else "paddle"
                lines.append(f"move({ch2}, {ax}, {rest}")
            continue
        lines.append(s)
    return "\n+ ".join(lines) if lines else "hold(t=1.0)"


def build_configs() -> list[dict[str, Any]]:
    xyz_db = json.loads(XYZ_JSON.read_text(encoding="utf-8"))
    idxs = cue_indices()
    configs: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for cue in CUES60:
        idx = int(idxs.get(cue, 0))
        t_code, t_meta = build_tiago_source_code(cue, xyz_db)
        p_code, p_meta = build_paddle_source_code(cue, xyz_db)
        t_frag = convert_code(t_code, platform="tiago")
        t_row = {
            "idx": idx,
            "cue": cue,
            "description": f"Synth60 tiago60 for {cue}",
            "movements": t_frag.get("movements") or [],
            "state": "synth60_tiago60",
            "model": "fewshot_xyz_pose_move_nopalm",
            "experiment": OUT_TAG,
            "generation_valid": bool(t_frag.get("parse_ok")),
            "platform": "tiago",
            "source_code": t_code,
            "parse_ok": bool(t_frag.get("parse_ok")),
            "notes": list(t_frag.get("notes") or []),
        }
        # paddle stored as source_code only (custom executor, not convert_code)
        p_row = {
            "idx": idx,
            "cue": cue,
            "description": f"Synth60 paddlebot60 for {cue}",
            "movements": [],  # filled by paddle renderer via plan
            "state": "synth60_paddlebot60",
            "model": "rematerialize_nopalm_tiagoxyz",
            "experiment": OUT_TAG,
            "generation_valid": True,
            "platform": "gcr_alpha1",
            "source_code": p_code,
            "parse_ok": True,
            "notes": [],
            "meta": p_meta,
        }
        configs.append(
            {"idx": idx, "cue": cue, "platforms": {"tiago": t_row, "gcr_alpha1": p_row}}
        )
        manifest.append(
            {
                "idx": idx,
                "cue": cue,
                "tiago_code": t_code,
                "paddle_code": p_code,
                "tiago_meta": t_meta,
                "paddle_meta": p_meta,
                "tiago_parse_ok": t_row["parse_ok"],
            }
        )
        print(f"[code] {idx} {cue}\n  TIAGO: {t_code.splitlines()[0]}\n  PADDLE: {p_code.splitlines()[0]}", flush=True)

    OUT_CFG.mkdir(parents=True, exist_ok=True)
    (OUT_CFG / "embodiment_motion_code_all.json").write_text(
        json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    bad = [m for m in manifest if not m["tiago_parse_ok"]]
    print(f"[cfg] {len(configs)} configs; tiago_parse_fail={len(bad)}", flush=True)
    return configs


# ----- Paddlebot plan + render (reuse Oddkin α1 executor) -----

def _paddle_plan_for(meta: dict[str, Any], exec_move: str) -> list[dict[str, Any]]:
    import render_cues30 as rc

    arm = meta.get("dir") or "front"
    mirror = bool(meta.get("mirror"))
    palm = meta.get("palm_hv") or "flat"

    claw_key = {
        "front": "claw_front",
        "back": "claw_back",
        "in": "claw_in",
        "out": "claw_out",
        "up": "claw_up",
        "down": "claw_front",  # soft fallback
        "still": "still",
    }.get(arm, "claw_front")
    paddle_key = {
        "front": "paddle_front",
        "back": "paddle_out",
        "in": "paddle_in",
        "out": "paddle_out",
        "up": "paddle_up",
        "down": "paddle_front",
        "still": "still",
    }.get(arm if mirror else "still", "still")

    joints = {}
    if claw_key != "still":
        joints.update(rc.A1.get(claw_key, {}))
    if paddle_key != "still":
        joints.update(rc.A1.get(paddle_key, {}))
    joints.update(rc.A1["grip_open"])
    # palm HV → bl2 roll (edge vs flat)
    if palm == "edge":
        joints["bl2"] = joints.get("bl2", 0.0) + 0.55
    else:
        joints["bl2"] = joints.get("bl2", 0.9) * 0.55

    # z percentile → slight trunk / arm raise bias
    z = float((meta.get("xyz") or {}).get("z") or 55)
    if z >= 70:
        joints["trunk3_pitch"] = -0.15
    elif z <= 45:
        joints["trunk3_pitch"] = 0.25

    plan: list[dict[str, Any]] = [
        {"op": "pose", "home": "1", "joints": joints, "n": 14},
    ]

    mv = exec_move or ""
    if "move(head, pan" in mv:
        plan.append({"op": "osc", "joints": {"neck_yaw": 0.4}, "rep": 3, "n": 30, "home": "1"})
    elif "move(head, tilt" in mv:
        plan.append({"op": "osc", "joints": {"neck_pitch": 0.3}, "rep": 3, "n": 28, "home": "1"})
    elif "move(torso" in mv:
        plan.append(
            {
                "op": "osc",
                "joints": {"trunk1_pitch": 0.25, "trunk2_pitch": 0.2},
                "rep": 1,
                "n": 24,
                "home": "1",
                "base_joints": joints,
            }
        )
    elif "path(ee_claw, arc" in mv or "path(ee, arc" in mv:
        plan.append(
            {
                "op": "osc",
                "joints": {"br0": 0.4, "br3": 0.45},
                "rep": 2,
                "n": 36,
                "home": "1",
                "base_joints": joints,
            }
        )
    elif "path(ee_claw, line" in mv or "path(ee, line" in mv:
        amp = {"br0": 0.35} if "axis=y" in mv else ({"br1": 0.3} if "axis=z" in mv else {"br0": 0.25, "br1": -0.2})
        plan.append(
            {
                "op": "osc",
                "joints": amp,
                "rep": 1,
                "n": 22,
                "home": "1",
                "base_joints": joints,
                "phase": "half",
            }
        )
    elif "move(wrist" in mv or "move(wrist_c" in mv:
        amp = {"br3": 0.55}
        if mirror:
            amp["bl2"] = 0.45
        rep = 4 if "rep=4" in mv or "rep=3" in mv else 2
        plan.append(
            {"op": "osc", "joints": amp, "rep": rep, "n": 32, "home": "1", "base_joints": joints}
        )
    elif "move(claw" in mv or "move(shoulder" in mv or "move(elbow" in mv or "move(paddle" in mv:
        amp = {"br0": 0.35, "br1": 0.25}
        if mirror:
            amp.update({"bl0": 0.35, "bl1": 0.25})
        rep = 3 if "rep=" in mv else 1
        plan.append(
            {"op": "osc", "joints": amp, "rep": max(rep, 1), "n": 28, "home": "1", "base_joints": joints}
        )
    else:
        # hold peak
        plan.append({"op": "pose", "home": "1", "joints": joints, "n": 18})

    # final hold
    plan.append({"op": "pose", "home": "1", "joints": joints, "n": 10})
    return plan


def render_paddlebot(configs: list[dict[str, Any]], *, force: bool = False) -> int:
    import mujoco
    import render_cues30 as rc

    gif_dir = OUT_RENDER / "gcr_alpha1" / "gif"
    gif_dir.mkdir(parents=True, exist_ok=True)
    xml_path = ODDKIN / "xml" / "gcr_alpha1.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    lookat, distance, base_z = (0.0, 0.0, 0.40), 2.05, 0.13
    n_done = 0
    for entry in configs:
        cue = entry["cue"]
        idx = int(entry["idx"])
        row = (entry.get("platforms") or {}).get("gcr_alpha1") or {}
        out_gif = gif_dir / f"emb_g{idx:02d}_{cue}.gif"
        if out_gif.is_file() and not force:
            n_done += 1
            continue
        meta = row.get("meta") or {}
        plan = _paddle_plan_for(meta, str(meta.get("exec_move") or "hold(t=1.0)"))
        frames = rc.execute_plan(plan)
        rc.render_clip(
            model,
            frames,
            out_gif,
            title=f"{idx}. {cue}",
            subtitle=(row.get("source_code") or "")[:90],
            lookat=lookat,
            distance=distance,
            base_z=base_z,
        )
        n_done += 1
        print(f"[render/gcr_alpha1] {cue} -> {out_gif.name} ({len(frames)}f)", flush=True)
    return n_done


def _plat_gif(platform: str, idx: int, cue: str) -> Path:
    if platform == "tiago":
        return gif_path_for(OUT_RENDER, "tiago", idx, cue)
    return OUT_RENDER / "gcr_alpha1" / "gif" / f"emb_g{idx:02d}_{cue}.gif"


def build_html(configs: list[dict[str, Any]], platform: str, out_html: Path, title: str) -> None:
    manifest = json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))
    by_cue = {m["cue"]: m for m in manifest}
    gif_dir = OUT_RENDER / ("tiago" if platform == "tiago" else "gcr_alpha1") / "gif"
    n_gif = sum(1 for p in gif_dir.glob("emb_g*.gif") if p.is_file() and p.stat().st_size > 1000) if gif_dir.is_dir() else 0
    code_key = "tiago_code" if platform == "tiago" else "paddle_code"
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html_mod.escape(title)}</title>",
        "<style>",
        "body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:20px;background:#f4f4f4;color:#222}",
        "h1{font-size:22px;margin:0 0 6px} .sub{color:#555;max-width:1100px;line-height:1.45;margin-bottom:16px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}",
        ".card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:10px}",
        ".card img{width:100%;background:#0d0d0d;border-radius:6px;aspect-ratio:1;object-fit:contain}",
        ".meta{font-size:12px;margin-top:8px} .tok{font-family:ui-monospace,Menlo,monospace;font-size:10.5px;",
        "background:#f0f3f8;padding:6px 7px;border-radius:5px;margin-top:5px;white-space:pre-wrap;word-break:break-word}",
        ".pill{display:inline-block;background:#e8f5e9;border:1px solid #81c784;color:#1b5e20;",
        "border-radius:999px;padding:2px 10px;font-size:12px;margin-right:6px}",
        "</style></head><body>",
        f"<h1>{html_mod.escape(title)}</h1>",
        f"<p class='sub'><span class='pill'>{n_gif}/{len(configs)} GIFs</span>"
        "Orientation = <b>dir + grip HV</b> (no palm). Location = <b>Tiago xyz</b>. "
        "Movement = synth60 exec tails.</p><div class='grid'>",
    ]
    for entry in sorted(configs, key=lambda e: e["idx"]):
        cue, idx = entry["cue"], entry["idx"]
        gif = _plat_gif(platform, idx, cue)
        rel = os.path.relpath(gif, out_html.parent) if gif.is_file() else ""
        man = by_cue.get(cue) or {}
        code = man.get(code_key) or ""
        parts.append(
            f"<div class='card'>"
            + (f"<img src='{rel}?v=7' alt='{html_mod.escape(cue)}'>" if rel else "<div class='miss'>missing</div>")
            + f"<div class='meta'><b>{idx}. {html_mod.escape(cue)}</b></div>"
            f"<div class='tok'>{html_mod.escape(code)}</div></div>"
        )
    parts.append("</div></body></html>")
    out_html.write_text("\n".join(parts), encoding="utf-8")
    print(f"[html] {out_html} ({n_gif}/{len(configs)})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--skip-tiago", action="store_true")
    ap.add_argument("--skip-paddle", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max", type=int, default=0, help="Limit cues (debug)")
    ap.add_argument(
        "--cues",
        default="",
        help="Comma-separated cue IDs to render; configs/gallery still contain all 60",
    )
    args = ap.parse_args()

    all_configs = build_configs()
    configs = all_configs
    if args.cues:
        selected = {cue.strip() for cue in args.cues.split(",") if cue.strip()}
        unknown = selected - {entry["cue"] for entry in all_configs}
        if unknown:
            raise SystemExit(f"Unknown cue IDs: {sorted(unknown)}")
        configs = [entry for entry in all_configs if entry["cue"] in selected]
    if args.max > 0:
        configs = configs[: args.max]

    if args.build_only:
        return

    if not args.skip_tiago:
        if hasattr(render_pending_once, "_skip"):
            delattr(render_pending_once, "_skip")
        # render one-by-one so HTML can refresh
        for i, entry in enumerate(configs, 1):
            render_pending_once(
                [entry],
                render_root=OUT_RENDER,
                out_json=OUT_CFG,
                platforms=["tiago"],
                max_items=1,
                force=args.force,
            )
            if i % 10 == 0 or i == len(configs):
                build_html(all_configs, "tiago", OUT_HTML_TIAGO, "Synth60 × Tiago (no-palm ori + Tiago xyz)")
            print(f"[tiago] progress {i}/{len(configs)}", flush=True)
        build_html(all_configs, "tiago", OUT_HTML_TIAGO, "Synth60 × Tiago (no-palm ori + Tiago xyz)")

    if not args.skip_paddle:
        render_paddlebot(configs, force=args.force)
        build_html(
            all_configs,
            "gcr_alpha1",
            OUT_HTML_PADDLE,
            "Synth60 × Paddlebot / gcr_alpha1 (rematerialized, no palm-dir)",
        )


if __name__ == "__main__":
    main()
