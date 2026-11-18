#!/usr/bin/env python3
"""Paddlebot (gcr_alpha1) five-shot Motion Codes → GIFs + gallery.

Matches the paddlebot90 dialect:
  pose(dir=front, claw=…, paddle=…, palm=…, x/y/z) + move/hold
Start already in pose (snap). Used as few-shots for Gemini/Qwen generation.

Outputs:
  data/results/motion_configs/embodiment_motion_code_paddlebot_fewshot/
  data/results/render/embodiment_motion_code_paddlebot_fewshot/gcr_alpha1/gif/
  motion_code/paddlebot_fewshot_gallery.html
  motion_code/data/paddlebot_fewshot.json
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "glfw")

MC = Path(__file__).resolve().parent
PAPER = MC.parent
REPO = PAPER.parent.parent
ODDKIN = PAPER / "synthetic_morphology"
sys.path[:0] = [str(MC), str(ODDKIN), str(PAPER), str(REPO)]

import mujoco  # noqa: E402
import render_cues30 as rc  # noqa: E402
import render_paddlebot90 as pb  # noqa: E402

OUT_TAG = "embodiment_motion_code_paddlebot_fewshot"
OUT_CFG = REPO / "data/results/motion_configs" / OUT_TAG
OUT_RENDER = REPO / "data/results/render" / OUT_TAG
OUT_HTML = MC / "paddlebot_fewshot_gallery.html"
OUT_JSON = MC / "data" / "paddlebot_fewshot.json"
YML = ODDKIN / "yml" / "fewshot_cues_oddkin.yaml"

# Five unified few-shot cues (same names as other embodiments).
FEWSHOTS: list[dict[str, Any]] = [
    {
        "cue": "raising_hand_greeting",
        "idx": 1,
        "blurb": "Raise paddle arm as a greeting hold",
        "code": (
            "pose(dir=front, torso=mid, claw=still, paddle=up, palm=flat, "
            "x=55, y=48, z=88, grip=open)\n+ hold(t=0.9)"
        ),
        "kind": "raise_paddle",
    },
    {
        "cue": "wave_hi",
        "idx": 2,
        "blurb": "Claw raised; side-to-side wave",
        "code": (
            "pose(dir=front, torso=mid, claw=up, paddle=still, palm=edge, "
            "x=55, y=40, z=90, grip=open)\n"
            "+ move(claw, y, ±, rep=3, spd=1.8, hold=0.12)"
        ),
        "kind": "wave_claw",
    },
    {
        "cue": "bow_greeting",
        "idx": 3,
        "blurb": "Trunk/head bow then recover",
        "code": (
            "pose(dir=front, torso=mid, claw=down, paddle=down, head=center)\n"
            "+ move(torso, pitch, +, spd=0.85, hold=0.35)\n"
            "+ move(torso, pitch, 0, spd=0.95)"
        ),
        "kind": "bow",
    },
    {
        "cue": "blow_kiss",
        "idx": 4,
        "blurb": "Claw to mouth then toss forward",
        "code": (
            "pose(dir=front, torso=mid, claw=back, paddle=still, palm=edge, "
            "grip=open, x=23, y=43, z=63)\n"
            "+ pose(dir=front, torso=mid, claw=front, paddle=still, palm=edge, "
            "grip=open, x=78, y=42, z=55)\n"
            "+ hold(t=0.5)"
        ),
        "kind": "blow_kiss",
    },
    {
        "cue": "circle_temple_crazy",
        "idx": 5,
        "blurb": "Claw at temple; EE arc circle",
        "code": (
            "pose(dir=front, torso=mid, claw=up, paddle=still, palm=edge, "
            "grip=open, x=25, y=29, z=78)\n"
            "+ path(ee_claw, arc, plane=yz, r=0.06, sweep=540, spd=1.2)"
        ),
        "kind": "circle_temple",
    },
]


def _pose_joints(
    *,
    claw: str,
    paddle: str,
    xyz: dict[str, int] | None = None,
    xyz_arm: str = "both",
) -> dict[str, float]:
    joints = dict(rc.home1())
    joints.update(pb._preset(f"claw_{claw}" if f"claw_{claw}" in pb.A1_DIRS else "claw_still"))
    joints.update(pb._preset(f"paddle_{paddle}" if f"paddle_{paddle}" in pb.A1_DIRS else "paddle_still"))
    joints.update(rc.A1.get("grip_open", {}))
    if xyz:
        before = {k: joints.get(k, 0.0) for k in ("br0", "br1", "br2", "br3", "bl0", "bl1", "bl2")}
        pb._apply_xyz_offsets(joints, xyz, gain=0.85)
        # Keep the idle arm on its dir preset (xyz otherwise lifts both).
        if xyz_arm == "paddle":
            for k in ("br0", "br1", "br2", "br3"):
                joints[k] = before[k]
        elif xyz_arm == "claw":
            for k in ("bl0", "bl1", "bl2"):
                joints[k] = before[k]
    for tn in ("trunk1_yaw", "trunk1_pitch", "trunk2_yaw", "trunk2_pitch", "trunk3_pitch", "trunk3_roll", "sat_spin"):
        joints[tn] = 0.0
    return joints


def _plan_for(row: dict[str, Any]) -> list[dict[str, Any]]:
    kind = row["kind"]
    if kind == "raise_paddle":
        j0 = _pose_joints(claw="still", paddle="still")
        j = _pose_joints(claw="still", paddle="up", xyz={"x": 55, "y": 48, "z": 88}, xyz_arm="paddle")
        return [
            {"op": "snap", "joints": j0, "n": 6},
            {"op": "stroke", "from": j0, "to": j, "n": 14},
            {"op": "hold", "joints": j, "n": 16},
        ]

    if kind == "wave_claw":
        # Do NOT apply xyz offsets here: high-z nudges fold claw_up into claw_back.
        j = _pose_joints(claw="up", paddle="still")
        return [
            {"op": "snap", "joints": j, "n": 8},
            {"op": "osc", "joints": {"br0": 0.45, "br3": 0.35}, "rep": 3, "n": 40, "base_joints": j},
            {"op": "hold", "joints": j, "n": 8},
        ]

    if kind == "bow":
        j0 = _pose_joints(claw="down", paddle="down")
        # +pitch leans crown toward camera (viewer bow); same sign as oddkin fewshot renderer.
        j1 = dict(j0)
        j1.update({"trunk1_pitch": 0.22, "trunk2_pitch": 0.28, "trunk3_pitch": 0.32, "neck_pitch": 0.28})
        j2 = dict(j0)
        j2.update({"trunk1_pitch": 0.50, "trunk2_pitch": 0.55, "trunk3_pitch": 0.48, "neck_pitch": 0.40})
        return [
            {"op": "snap", "joints": j0, "n": 6},
            {"op": "stroke", "from": j0, "to": j1, "n": 10},
            {"op": "stroke", "from": j1, "to": j2, "n": 12},
            {"op": "hold", "joints": j2, "n": 10},
            {"op": "stroke", "from": j2, "to": j0, "n": 14},
            {"op": "hold", "joints": j0, "n": 6},
        ]

    if kind == "blow_kiss":
        ik = pb._load_ik_joints("shh_gesture") or pb._load_ik_joints("cover_mouth_gasp")
        if ik:
            j_mouth = dict(rc.home1())
            j_mouth.update(ik)
            j_mouth.update(rc.A1.get("grip_open", {}))
            j_mouth.update(pb._preset("paddle_still"))
        else:
            j_mouth = _pose_joints(claw="back", paddle="still", xyz={"x": 23, "y": 43, "z": 63}, xyz_arm="claw")
        j_front = _pose_joints(claw="front", paddle="still")  # pure front toss, no xyz lift
        return [
            {"op": "snap", "joints": j_mouth, "n": 10},
            {"op": "hold", "joints": j_mouth, "n": 8},
            {"op": "stroke", "from": j_mouth, "to": j_front, "n": 14},
            {"op": "hold", "joints": j_front, "n": 10},
        ]

    if kind == "circle_temple":
        ik = pb._load_ik_joints("circle_temple_crazy") or pb._load_ik_joints("self_scratch_head_confused")
        if ik:
            j = dict(rc.home1())
            j.update(ik)
            j.update(rc.A1.get("grip_open", {}))
        else:
            j = _pose_joints(claw="up", paddle="still", xyz={"x": 25, "y": 29, "z": 78}, xyz_arm="claw")
            j["br0"] = 0.55
            j["br1"] = -1.15
            j["br2"] = -0.45
        return [
            {"op": "snap", "joints": j, "n": 8},
            {
                "op": "arc",
                "base_joints": j,
                # 1.5 turns (≈540°) in yz via phase-offset claw joints
                "cos": {"br0": 0.38, "br3": 0.28},
                "sin": {"br1": 0.32, "br2": 0.16},
                "turns": 1.5,
                "n": 48,
            },
            {"op": "hold", "joints": j, "n": 8},
        ]

    raise KeyError(kind)


def _execute_plan(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Like paddlebot90 snap executor, plus arc (phase-offset cos/sin)."""
    if not any(s.get("op") == "arc" for s in plan):
        return pb._execute_plan_snap(plan)

    frames: list[dict[str, Any]] = []
    cur_base = (0.0, 0.0, 0.0)
    cur_joints: dict[str, float] = dict(rc.home1())

    for step in plan:
        op = step["op"]
        if op == "arc":
            root = dict(rc.home1(step.get("base_joints") or cur_joints))
            cos_amp = step.get("cos") or {}
            sin_amp = step.get("sin") or {}
            turns = float(step.get("turns") or 1.0)
            n = max(int(step.get("n", 36)), 2)
            last = None
            for i in range(n):
                ang = 2.0 * math.pi * turns * (i / (n - 1))
                j = dict(root)
                for k, a in cos_amp.items():
                    j[k] = root.get(k, 0.0) + float(a) * math.cos(ang)
                for k, a in sin_amp.items():
                    j[k] = root.get(k, 0.0) + float(a) * math.sin(ang)
                last = {"base": cur_base, "joints": j}
                frames.append(last)
            if last:
                cur_joints = dict(last["joints"])
            continue
        # reuse paddlebot90 for other ops by wrapping a one-step plan
        sub = pb._execute_plan_snap([step])
        frames.extend(sub)
        if sub:
            cur_joints = dict(sub[-1]["joints"])
            cur_base = sub[-1].get("base", cur_base)
    return frames


def _sync_oddkin_yaml(rows: list[dict[str, Any]]) -> None:
    """YAML is maintained alongside this file; no runtime rewrite."""
    _ = rows
    return


def build_configs() -> list[dict[str, Any]]:
    configs = []
    for row in FEWSHOTS:
        configs.append(
            {
                "idx": row["idx"],
                "cue": row["cue"],
                "description": row["blurb"],
                "source_code": row["code"],
                "platform": "gcr_alpha1",
                "model": "paddlebot_fewshot",
                "experiment": OUT_TAG,
                "meta": {"kind": row["kind"], "blurb": row["blurb"], "fewshot": True},
                "parse_ok": True,
            }
        )
    return configs


def render_all(configs: list[dict[str, Any]], *, force: bool = False) -> int:
    gif_dir = OUT_RENDER / "gcr_alpha1" / "gif"
    gif_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(ODDKIN / "xml" / "gcr_alpha1.xml"))
    lookat, distance, base_z = (0.0, 0.0, 0.40), 2.05, 0.13
    n = 0
    by_cue = {r["cue"]: r for r in FEWSHOTS}
    for row in configs:
        cue, idx = row["cue"], int(row["idx"])
        out_gif = gif_dir / f"fs_{idx:02d}_{cue}.gif"
        if out_gif.is_file() and out_gif.stat().st_size > 1000 and not force:
            n += 1
            continue
        plan = _plan_for(by_cue[cue])
        frames = _execute_plan(plan)
        rc.render_clip(
            model,
            frames,
            out_gif,
            title=f"FS{idx}. {cue}",
            subtitle=(row.get("source_code") or "")[:90],
            lookat=lookat,
            distance=distance,
            base_z=base_z,
        )
        n += 1
        print(f"[render] {cue} -> {out_gif.name} ({len(frames)}f)", flush=True)
    return n


def build_html(configs: list[dict[str, Any]]) -> None:
    gif_dir = OUT_RENDER / "gcr_alpha1" / "gif"
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Paddlebot · 5 few-shots</title>",
        "<style>",
        "body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:20px;background:#f4f4f4;color:#222}",
        "h1{font-size:22px;margin:0 0 6px}.sub{color:#555;max-width:900px;line-height:1.45;margin-bottom:16px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}",
        ".card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:10px}",
        ".card img{width:100%;background:#0d0d0d;border-radius:6px;aspect-ratio:1;object-fit:contain}",
        ".meta{font-size:13px;margin-top:8px}.tok{font-family:ui-monospace,Menlo,monospace;font-size:11px;",
        "background:#f0f3f8;padding:7px;border-radius:5px;margin-top:6px;white-space:pre-wrap;word-break:break-word}",
        ".pill{display:inline-block;background:#fff3e0;border:1px solid #ffb74d;color:#e65100;",
        "border-radius:999px;padding:2px 10px;font-size:12px;margin-right:6px}",
        "</style></head><body>",
        "<h1>Paddlebot · five-shot Motion Codes</h1>",
        f"<p class='sub'><span class='pill'>{len(configs)} few-shots</span>"
        "Same cue names as other embodiments. Dialect matches paddlebot90 "
        "(<code>dir=front</code> body facing; <code>claw=</code>/<code>paddle=</code>; xyz). "
        "Ready for Gemini / Qwen prompts.</p><div class='grid'>",
    ]
    for row in configs:
        cue, idx = row["cue"], row["idx"]
        gif = gif_dir / f"fs_{idx:02d}_{cue}.gif"
        rel = os.path.relpath(gif, OUT_HTML.parent) if gif.is_file() else ""
        code = row.get("source_code") or ""
        parts.append(
            "<div class='card'>"
            + (
                f"<img src='{html_mod.escape(rel)}?v=4' alt='{html_mod.escape(cue)}'>"
                if rel
                else "<div style='aspect-ratio:1;background:#111'></div>"
            )
            + f"<div class='meta'><b>FS{idx}. {html_mod.escape(cue)}</b> — "
            + f"{html_mod.escape(row.get('description') or '')}</div>"
            + f"<div class='tok'>{html_mod.escape(code)}</div></div>"
        )
    parts.append("</div></body></html>")
    OUT_HTML.write_text("".join(parts), encoding="utf-8")
    print(f"[html] {OUT_HTML}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--codes-only", action="store_true")
    args = ap.parse_args()

    configs = build_configs()
    OUT_CFG.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "platform": "gcr_alpha1",
        "n": len(configs),
        "cues": [
            {
                "idx": r["idx"],
                "cue": r["cue"],
                "description": r["description"],
                "source_code": r["source_code"],
            }
            for r in configs
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT_CFG / "embodiment_motion_code_all.json").write_text(
        json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _sync_oddkin_yaml(FEWSHOTS)
    print(f"[cfg] {len(configs)} fewshots → {OUT_CFG}", flush=True)
    print(f"[json] {OUT_JSON}", flush=True)
    for r in configs:
        print(f"  FS{r['idx']} {r['cue']}\n    {r['source_code'].splitlines()[0]}", flush=True)

    if not args.codes_only:
        render_all(configs, force=args.force)
    build_html(configs)


if __name__ == "__main__":
    main()
