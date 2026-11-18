#!/usr/bin/env python3
"""Build Motion Codes for 30 synth60 cues and render G1 GIFs (wrist-preserving EE translate)."""
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
sys.path[:0] = [str(MC), str(PAPER), str(REPO), str(ADHOC)]

from render_cuespec_gt_poses import normalize_pose_code  # noqa: E402
from score_humanlike_cuespec import REQUIRED, _na  # noqa: E402
from motion_code_to_config import convert_code  # noqa: E402
from run_embodiment_exp0_gemini import gif_path_for, render_pending_once  # noqa: E402
import embodiment_renderers as er  # noqa: E402

OUT_TAG = "embodiment_motion_code_synth60_motion30"
OUT_CFG = REPO / "data/results/motion_configs" / OUT_TAG
OUT_RENDER = REPO / "data/results/render" / OUT_TAG
OUT_HTML = MC / "archive" / "synth60_motion30_gallery.html"
OUT_MANIFEST = OUT_CFG / "manifest.json"

# First 29 non-paper30 cues + clap_applause (index 90) so both throat_cut and clap are in.
_SYNTH = [c for c in REQUIRED if c not in list(REQUIRED)[:30]]
CUES30 = _SYNTH[:29] + ["clap_applause"]


def _first_axis(token: str, default: str = "x") -> str:
    parts = re.split(r"[|/]", token.strip())
    for p in parts:
        if p in ("x", "y", "z"):
            return p
    return default


def _joint_name(raw: str) -> str:
    j = (raw or "_").strip().lower()
    if j in ("wrist", "shoulder", "elbow", "head", "torso"):
        return j
    return "shoulder"


def auto_exec_move(cue: str) -> str:
    """Concrete move/path from REQUIRED axis/sign, with visible amplitudes."""
    # Hand-tuned overrides for clarity / known-good amplitudes.
    overrides = {
        "self_scratch_head_confused": "move(wrist, z, [-18,18], rep=4, spd=2.2, hold=0.05)",
        "tap_temple_think": "move(wrist, y, [-16,16], rep=4, spd=2.2, hold=0.05)",
        "pat_stomach": "move(wrist, x, [-22,0], rep=4, spd=2.5, hold=0.08)",
        "fan_face_hot": "move(wrist, y, [-28,28], rep=4, spd=2.4, hold=0.05)",
        "cover_mouth_gasp": "path(ee, line, axis=x, dist=-0.10, spd=3.0)\n+ hold(t=0.9)",
        "fistbump_invite": "path(ee, line, axis=x, dist=0.12, spd=3.2)\n+ hold(t=0.7)",
        "throat_cut_stop": "path(ee, line, axis=y, dist=0.18, spd=3.0)\n+ hold(t=0.6)",
        "uncertainty_wave": "move(wrist, y, [-30,30], rep=4, spd=2.0, hold=0.05)",
        "clap_applause": "move(shoulder, y, [-4,4], rep=5, spd=2.4, hold=0.05)",
        "protective_arm_out": "move(shoulder, y, [0,-35], rep=1, spd=1.6, hold=0.9)",
        "slow_down_request_palm_down": "move(shoulder, z, [0,-22], rep=3, spd=1.4, hold=0.15)",
        "visor_search": "move(head, pan, [-25,25], rep=3, spd=1.6, hold=0.1)",
        "lean_forward_interest": "move(torso, x, 0.08, spd=0.9)\n+ hold(t=1.0)",
        "shh_gesture": "hold(t=1.2)",
        "cover_ears_too_loud": "move(shoulder, y, [0,6], rep=2, spd=1.4, hold=0.15)",
        "circle_temple_crazy": "path(ee, arc, plane=yz, r=0.04, sweep=360, rep=3, spd=1.8)\n+ hold(t=0.5)",
        "guiding_arm": "path(ee, line, axis=y, dist=-0.15, spd=1.4)\n+ hold(t=0.8)",
        "scale_expand": "path(ee, line, axis=y, dist=-0.12, spd=1.2)\n+ hold(t=0.8)",
    }
    if cue in overrides:
        return overrides[cue]

    req = REQUIRED[cue]
    mv = str(req.get("movement") or "—").strip()
    path = str(req.get("path") or "—").strip()

    if path and not _na(path) and path != "—":
        if "arc" in path:
            plane = "yz"
            m = re.search(r"plane=([a-z|]+)", path)
            if m:
                plane = m.group(1).split("|")[0]
            rep = 3 if "rep" in path else 1
            return f"path(ee, arc, plane={plane}, r=0.07, sweep=360, rep={rep}, spd=1.6)\n+ hold(t=0.6)"
        axis = "x"
        m = re.search(r"axis=([xyz])", path)
        if m:
            axis = m.group(1)
        elif "y" in path:
            axis = "y"
        elif "z" in path:
            axis = "z"
        dist = 0.12 if axis == "x" else (0.15 if axis == "y" else -0.10)
        if "−" in path or "-" in path.split("axis")[-1][:20]:
            dist = -abs(dist)
        return f"path(ee, line, axis={axis}, dist={dist:.2f}, spd=2.2)\n+ hold(t=0.7)"

    if not mv or _na(mv) or mv == "—":
        return "hold(t=1.0)"

    # move(joint, axis, sign/±, ...)
    m = re.match(
        r"move\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,\)]+)",
        mv.replace(" ", ""),
    )
    if not m:
        return "hold(t=1.0)"
    joint = _joint_name(m.group(1))
    axis_tok = m.group(2)
    sign_tok = m.group(3)
    axis = _first_axis(axis_tok, "y" if joint == "wrist" else "x")
    rep = 3 if "rep" in mv else 1
    spd = 2.0

    if joint == "head":
        if "pan" in axis_tok or axis == "y":
            return f"move(head, pan, [-22,22], rep={max(rep,2)}, spd=1.6, hold=0.1)"
        return f"move(head, tilt, [-15,15], rep={max(rep,2)}, spd=1.4, hold=0.1)"
    if joint == "torso":
        if axis == "x":
            return "move(torso, x, 0.08, spd=0.9)\n+ hold(t=1.0)"
        if axis == "z":
            return "move(torso, z, -0.05, spd=0.9)\n+ hold(t=1.0)"
        return "move(torso, x, 0.06, spd=0.9)\n+ hold(t=1.0)"

    # arm / wrist / shoulder / elbow / _
    if "±" in sign_tok or "+-" in sign_tok or "-+" in sign_tok:
        amp = 22 if joint == "wrist" else 28
        return f"move({joint}, {axis}, [-{amp},{amp}], rep={max(rep,3)}, spd={spd}, hold=0.06)"
    if "−" in sign_tok or sign_tok.strip().startswith("-"):
        amp = 20 if joint == "wrist" else 25
        return f"move({joint}, {axis}, [0,-{amp}], rep={max(rep,1)}, spd={spd}, hold=0.1)"
    # default +
    amp = 20 if joint == "wrist" else 25
    return f"move({joint}, {axis}, [0,{amp}], rep={max(rep,1)}, spd={spd}, hold=0.1)"


def cue_indices() -> dict[str, int]:
    canvas = Path(
        "/Users/sb/.cursor/projects/"
        "Users-sb-Downloads-workspace-dev-robosuite-papers-icra2027-motion-abstraction/"
        "canvases/gesture-critical-factors-105.canvas.tsx"
    )
    out: dict[str, int] = {}
    if canvas.is_file():
        for m in re.finditer(r"index:\s*(\d+),\s*cue:\s*\"([^\"]+)\"", canvas.read_text()):
            out[m.group(2)] = int(m.group(1))
    for i, cue in enumerate(REQUIRED, start=1):
        out.setdefault(cue, i)
    return out


def semantic_blurb(cue: str) -> str:
    # Prefer canvas movement semantics when available via REQUIRED movement string.
    req = REQUIRED[cue]
    bits = [b for b in (req.get("movement"), req.get("path")) if b and not _na(str(b)) and str(b) != "—"]
    return " · ".join(str(b) for b in bits) if bits else "pose hold"


def build_source_code(cue: str) -> tuple[str, dict[str, Any]]:
    req = REQUIRED[cue]
    pose_code, meta = normalize_pose_code(req.get("pose", "—"))
    exec_tail = auto_exec_move(cue)
    parts = [pose_code]
    if exec_tail:
        parts.append(exec_tail)
    if "hold(" not in exec_tail:
        parts.append("hold(t=0.8)")
    code = "\n+ ".join(parts)
    meta.update(
        {
            "movement_raw": req.get("movement", "—"),
            "path_raw": req.get("path", "—"),
            "exec_move": exec_tail,
            "semantic": semantic_blurb(cue),
        }
    )
    return code, meta


def build_configs() -> list[dict[str, Any]]:
    idxs = cue_indices()
    configs: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for cue in CUES30:
        idx = int(idxs.get(cue, 0))
        source_code, meta = build_source_code(cue)
        frag = convert_code(source_code, platform="humanoid")
        row = {
            "idx": idx,
            "cue": cue,
            "description": f"Synth60 motion30 for {cue}",
            "movements": frag.get("movements") or [],
            "state": "synth60_motion30",
            "model": "fewshot_xyz_pose_move",
            "experiment": OUT_TAG,
            "generation_valid": bool(frag.get("parse_ok")),
            "platform": "humanoid",
            "source_code": source_code,
            "parse_ok": bool(frag.get("parse_ok")),
            "notes": list(frag.get("notes") or []),
        }
        configs.append({"idx": idx, "cue": cue, "platforms": {"humanoid": row}})
        manifest.append(
            {
                "idx": idx,
                "cue": cue,
                "required_pose": REQUIRED[cue].get("pose"),
                "required_move": REQUIRED[cue].get("movement"),
                "required_path": REQUIRED[cue].get("path"),
                "semantic": meta.get("semantic"),
                "exec_code": source_code,
                "meta": meta,
                "parse_ok": row["parse_ok"],
                "notes": row["notes"],
            }
        )
        print(
            f"[code] {cue}\n{source_code}\n  parse_ok={row['parse_ok']} notes={row['notes'][:2]}",
            flush=True,
        )

    OUT_CFG.mkdir(parents=True, exist_ok=True)
    (OUT_CFG / "embodiment_motion_code_all.json").write_text(
        json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    bad = [m for m in manifest if not m["parse_ok"]]
    print(f"[cfg] {len(configs)} configs; parse_fail={len(bad)}", flush=True)
    return configs


def drain_renders(configs: list[dict[str, Any]], *, batch: int = 2) -> None:
    OUT_RENDER.mkdir(parents=True, exist_ok=True)
    if hasattr(render_pending_once, "_skip"):
        delattr(render_pending_once, "_skip")
    total = 0
    while True:
        n = render_pending_once(
            configs,
            render_root=OUT_RENDER,
            platforms=["humanoid"],
            max_items=batch,
            force=False,
        )
        total += n
        if n == 0:
            break
        print(f"[drain] +{n} (attempted {total})", flush=True)
    print(f"[drain] done attempted={total}", flush=True)


def _first_pose_params(plat: dict[str, Any]) -> dict[str, Any] | None:
    for mv in plat.get("movements") or []:
        if mv.get("type") == "pose":
            pose = (mv.get("parameters") or {}).get("pose")
            if isinstance(pose, dict):
                return pose
    return None


def resolve_bank_poses(configs: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    er._G1_FEASIBLE_POSE_CACHE = None
    out: dict[str, dict[str, Any] | None] = {}
    for entry in configs:
        cue = entry["cue"]
        pose = _first_pose_params(entry["platforms"]["humanoid"])
        sel = er._g1_select_feasible_pose(pose) if pose else None
        if sel is None:
            out[cue] = None
            continue
        targets = {
            a: float(pose[a])
            for a in ("x", "y", "z")
            if isinstance(pose.get(a), (int, float))
        }
        l1 = sum(abs(float(sel[a]) - v) for a, v in targets.items())
        out[cue] = {
            "pose_id": int(sel["pose_id"]),
            "label": (
                f"{sel.get('direction')}|{sel.get('gripper_orientation')}|"
                f"{sel.get('palm_semantic')}"
            ),
            "xyz": (
                round(float(sel["x"])),
                round(float(sel["y"])),
                round(float(sel["z"])),
            ),
            "l1": round(l1, 1),
        }
    return out


def build_html(configs: list[dict[str, Any]]) -> Path:
    manifest = json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))
    by_cue = {m["cue"]: m for m in manifest}
    bank = resolve_bank_poses(configs)
    for m in manifest:
        m["bank_pose"] = bank.get(m["cue"])
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Synth60 ×30 Motion Codes (humanoid)</title>",
        "<style>",
        "body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:20px;background:#f4f4f4;color:#222}",
        "h1{font-size:22px;margin:0 0 6px} .sub{color:#555;max-width:1100px;line-height:1.45;margin-bottom:16px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}",
        ".card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:10px}",
        ".card img{width:100%;background:#0d0d0d;border-radius:6px;aspect-ratio:1;object-fit:contain}",
        ".meta{font-size:12px;margin-top:8px} .meta b{font-size:13px}",
        ".tok{font-family:ui-monospace,Menlo,monospace;font-size:10.5px;background:#f0f3f8;",
        "padding:6px 7px;border-radius:5px;margin-top:5px;white-space:pre-wrap;word-break:break-word}",
        ".bank{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:#0d47a1;margin-top:6px;",
        "background:#e3f2fd;padding:5px 7px;border-radius:5px}",
        ".miss{color:#a33}",
        "</style></head><body>",
        "<h1>Synth60 Motion Codes → G1 GIF (30 cues)</h1>",
        "<p class='sub'>Wrist-preserving EE parallel translate. Req. Pose includes "
        "<b>left_arm</b>. Bank pose_id shown per card. "
        "clap=<code>up|vert|palmin</code>, throat_cut=<code>in|horiz|palmdown</code>.</p>",
        "<div class='grid'>",
    ]
    for entry in sorted(configs, key=lambda e: int(e["idx"])):
        cue = entry["cue"]
        idx = int(entry["idx"])
        m = by_cue.get(cue) or {}
        gif = gif_path_for(OUT_RENDER, "humanoid", idx, cue)
        rel = os.path.relpath(gif, OUT_HTML.parent) if gif.is_file() else ""
        code = html_mod.escape(m.get("exec_code") or "")
        sem = html_mod.escape(m.get("semantic") or "")
        req_p = html_mod.escape(str(m.get("required_pose") or ""))
        req_m = html_mod.escape(str(m.get("required_move") or m.get("required_path") or ""))
        bp = bank.get(cue)
        if bp:
            bank_line = (
                f"bank <b>#{bp['pose_id']}</b>  {html_mod.escape(bp['label'])}  "
                f"xyz=({bp['xyz'][0]},{bp['xyz'][1]},{bp['xyz'][2]})  "
                f"L1={bp['l1']:.0f}"
            )
        else:
            bank_line = "<span class='miss'>bank: no hit</span>"
        bust = "?v=m30"
        img = (
            f"<img src='{html_mod.escape(rel)}{bust}' alt='{html_mod.escape(cue)}'>"
            if rel
            else "<div class='miss'>GIF missing</div>"
        )
        parts.append(
            f"<div class='card'>{img}<div class='meta'><b>{idx}. {html_mod.escape(cue)}</b>"
            f"<div>{sem}</div>"
            f"<div class='bank'>{bank_line}</div>"
            f"<div class='tok'>Req: {req_p}\nMove: {req_m}</div>"
            f"<div class='tok'>{code}</div></div></div>"
        )
    parts.append("</div></body></html>")
    OUT_HTML.write_text("\n".join(parts), encoding="utf-8")
    print(f"[html] {OUT_HTML}", flush=True)
    return OUT_HTML


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()
    print(f"[cues] n={len(CUES30)}", flush=True)
    configs = build_configs()
    if not args.skip_render:
        gif_dir = OUT_RENDER / "humanoid" / "gif"
        if gif_dir.is_dir():
            for p in gif_dir.glob("emb_g*"):
                p.unlink(missing_ok=True)
        drain_renders(configs, batch=args.batch)
    build_html(configs)
    gifs = (
        list((OUT_RENDER / "humanoid" / "gif").glob("emb_g*.gif"))
        if OUT_RENDER.exists()
        else []
    )
    print(f"[done] gifs={len(gifs)} → {OUT_RENDER / 'humanoid' / 'gif'}", flush=True)


if __name__ == "__main__":
    main()
