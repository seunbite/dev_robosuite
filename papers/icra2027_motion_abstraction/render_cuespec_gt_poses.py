#!/usr/bin/env python3
"""Render cue-spec Loc/Orient tokens as humanoid EE poses for GT confirmation.

Takes REQUIRED pose(...) atoms from score_humanlike_cuespec.py, normalizes
≈ / symbolic bands / dir|grip alternatives into executable Motion Code, then
renders Unitree G1 pose+hold GIFs + an HTML gallery.
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

PAPER = Path(__file__).resolve().parent
REPO = Path("/Users/sb/Downloads/workspace/dev_robosuite")
ADHOC = REPO / "adhoc/generation/robotarm"
sys.path[:0] = [str(PAPER), str(REPO), str(ADHOC)]

from score_humanlike_cuespec import REQUIRED, _na, _parse_pose_spec  # noqa: E402
from motion_code_to_config import convert_code  # noqa: E402
from run_embodiment_exp0_gemini import gif_path_for, render_pending_once  # noqa: E402

CANVAS = Path(
    "/Users/sb/.cursor/projects/"
    "Users-sb-Downloads-workspace-dev-robosuite-papers-icra2027-motion-abstraction/"
    "canvases/gesture-critical-factors-105.canvas.tsx"
)
OUT_CFG = (
    REPO
    / "data/results/motion_configs"
    / "embodiment_motion_code_cuespec_gt"
)
OUT_RENDER = (
    REPO
    / "data/results/render"
    / "embodiment_motion_code_cuespec_gt"
)
OUT_HTML = PAPER / "cuespec_gt_pose_gallery.html"
OUT_MANIFEST = OUT_CFG / "cuespec_gt_manifest.json"

# Band centers aligned with score_humanlike_cuespec._xyz_target
SYM_XYZ = {
    "mid": 55,
    "shoulder": 62,
    "near_chest": 85,
    "nearchest": 85,
    "chest": 85,
    "forehead": 88,
    "head": 88,
    "tall": 95,
    "overhead": 95,
}

_PALM = {"palmfront", "palmback", "palmup", "palmdown", "palmin", "palmout"}
_HV = {"horiz", "horizontal", "vert", "vertical"}
_SPECIFIC = {"fist", "closed", "pinch", "claw"}


def cue_indices() -> dict[str, int]:
    text = CANVAS.read_text(encoding="utf-8")
    out: dict[str, int] = {}
    for m in re.finditer(r'index:\s*(\d+),\s*cue:\s*"([^"]+)"', text):
        out[m.group(2)] = int(m.group(1))
    # fallback: REQUIRED insertion order starting at 1
    for i, cue in enumerate(REQUIRED, start=1):
        out.setdefault(cue, i)
    return out


def _first_alt(raw: str) -> str:
    parts = [p.strip() for p in str(raw).split("|") if p.strip() and p.strip() != "_"]
    return parts[0] if parts else ""


def _pick_dir(raw: str) -> str | None:
    v = _first_alt(raw)
    return v or None


def _pick_grip(raw: str) -> str | None:
    s = str(raw or "").strip().lower()
    if not s or s == "_":
        return None
    parts = [p.strip() for p in re.split(r"[|/]", s) if p.strip() and p.strip() != "_"]
    if not parts:
        return None
    # compound: hv + palm  → keep both (first of each class)
    hv = next((p for p in parts if p in _HV), None)
    palm = next((p for p in parts if p in _PALM or p.startswith("palm")), None)
    if hv and palm:
        hv_n = "horiz" if hv.startswith("horiz") else "vert"
        return f"{hv_n}|{palm}"
    # OR alternatives: prefer specific hand shape
    for p in parts:
        if p in _SPECIFIC:
            return p
    if hv:
        return "horiz" if hv.startswith("horiz") else "vert"
    return parts[0]


def _resolve_xyz(raw: str) -> int | None:
    v = str(raw).lower().replace(" ", "").lstrip("≈=")
    if not v or v == "_":
        return None
    for key, num in SYM_XYZ.items():
        if key in v:
            return num
    m = re.search(r"-?\d+(?:\.\d+)?", v)
    if m:
        return int(round(float(m.group(0))))
    return None


def normalize_pose_code(pose_spec: str) -> tuple[str, dict[str, Any]]:
    """Turn scoring pose(...) into executable pose(...)= numeric Motion Code."""
    meta: dict[str, Any] = {"raw": pose_spec, "skipped": False}
    if _na(pose_spec):
        meta["skipped"] = True
        return "pose()", meta

    parsed = _parse_pose_spec(pose_spec)
    parts: list[str] = []

    if "dir" in parsed:
        d = _pick_dir("|".join(parsed["dir"]))
        if d:
            parts.append(f"dir={d}")
            meta["dir"] = d
            meta["dir_alts"] = parsed["dir"]
    if "grip" in parsed:
        g = _pick_grip("|".join(parsed["grip"]))
        if g:
            parts.append(f"grip={g}")
            meta["grip"] = g
            meta["grip_alts"] = parsed["grip"]
    if "head" in parsed:
        h = _first_alt("|".join(parsed["head"])) or "center"
        if h != "_":
            parts.append(f"head={h}")
            meta["head"] = h
    if "torso" in parsed:
        t = _first_alt("|".join(parsed["torso"])) or "mid"
        if t != "_":
            parts.append(f"torso={t}")
            meta["torso"] = t
    if "elbow" in parsed:
        parts.append("elbow=T")
        meta["elbow"] = True
    if parsed.get("left_arm") == "mirror":
        parts.append("left_arm=mirror")
        meta["left_arm"] = "mirror"

    xyz_out: dict[str, int] = {}
    for axis, raw in (parsed.get("xyz") or {}).items():
        num = _resolve_xyz(raw)
        if num is not None:
            parts.append(f"{axis}={num}")
            xyz_out[axis] = num
    if xyz_out:
        meta["xyz"] = xyz_out

    # Preserve bank=F/T so procedural IK can skip the discrete feasible bank.
    bank_raw = str(parsed.get("bank") or "").strip().lower()
    if bank_raw in {"f", "false", "0", "no", "off"}:
        parts.append("bank=F")
        meta["bank"] = False
    elif bank_raw in {"t", "true", "1", "yes", "on"}:
        parts.append("bank=T")
        meta["bank"] = True

    code = f"pose({', '.join(parts)})" if parts else "pose()"
    meta["code"] = code
    return code, meta


def build_source_code(req: dict[str, str]) -> tuple[str, dict[str, Any]]:
    pose_code, meta = normalize_pose_code(req.get("pose", "—"))
    # Hold so the peak pose is readable; skip complex move/path for confirm.
    code = f"{pose_code}\n+ hold(t=1.5)"
    meta["movement_raw"] = req.get("movement", "—")
    meta["path_raw"] = req.get("path", "—")
    return code, meta


def build_configs() -> list[dict[str, Any]]:
    idxs = cue_indices()
    configs: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for cue, req in REQUIRED.items():
        idx = int(idxs.get(cue, 0))
        source_code, meta = build_source_code(req)
        frag = convert_code(source_code, platform="humanoid")
        row = {
            "idx": idx,
            "cue": cue,
            "description": f"Cue-spec GT pose confirm for {cue}",
            "movements": frag.get("movements") or [],
            "state": "cuespec_gt",
            "model": "cuespec_token_gt",
            "experiment": "embodiment_motion_code_cuespec_gt",
            "generation_valid": bool(frag.get("parse_ok")),
            "platform": "humanoid",
            "source_code": source_code,
            "parse_ok": bool(frag.get("parse_ok")),
            "notes": list(frag.get("notes") or []),
        }
        entry = {
            "idx": idx,
            "cue": cue,
            "platforms": {"humanoid": row},
        }
        configs.append(entry)
        manifest.append(
            {
                "idx": idx,
                "cue": cue,
                "required_pose": req.get("pose"),
                "required_move": req.get("movement"),
                "required_path": req.get("path"),
                "exec_code": source_code,
                "meta": meta,
                "parse_ok": row["parse_ok"],
                "pose_params": (
                    (row["movements"][0].get("parameters") or {}).get("pose")
                    if row["movements"]
                    else None
                ),
            }
        )
    OUT_CFG.mkdir(parents=True, exist_ok=True)
    (OUT_CFG / "embodiment_motion_code_all.json").write_text(
        json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_CFG / "humanoid").mkdir(parents=True, exist_ok=True)
    for e in configs:
        cue = e["cue"]
        (OUT_CFG / "humanoid" / f"{cue}.json").write_text(
            json.dumps([e["platforms"]["humanoid"]], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    OUT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    bad = [m for m in manifest if not m["parse_ok"]]
    print(f"[cfg] wrote {len(configs)} configs; parse_fail={len(bad)}", flush=True)
    for m in bad[:10]:
        print(f"  fail {m['cue']}: {m['exec_code']!r}", flush=True)
    return configs


def drain_renders(configs: list[dict[str, Any]], *, batch: int = 4, force: bool = False) -> None:
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
            force=force,
        )
        total += n
        if n == 0:
            break
        print(f"[drain] +{n} (total attempted {total})", flush=True)
    print(f"[drain] done attempted={total}", flush=True)


def build_html(configs: list[dict[str, Any]]) -> Path:
    manifest = json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))
    by_cue = {m["cue"]: m for m in manifest}
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Cue-spec GT poses (Loc/Orient → EE)</title>",
        "<style>",
        "body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:20px;background:#f5f5f5;color:#222}",
        "h1{font-size:22px;margin:0 0 6px} .sub{color:#555;max-width:980px;margin-bottom:16px;line-height:1.45}",
        ".filt{margin:10px 0 16px;display:flex;gap:8px;flex-wrap:wrap}",
        ".filt button{border:1px solid #ccc;background:#fff;padding:4px 10px;border-radius:999px;cursor:pointer;font-size:12px}",
        ".filt button.on{background:#1f6feb;color:#fff;border-color:#1f6feb}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}",
        ".card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px}",
        ".card img{width:100%;background:#111;border-radius:4px;aspect-ratio:1;object-fit:contain}",
        ".meta{font-size:12px;margin-top:6px} .meta b{font-size:13px}",
        ".tok{font-family:ui-monospace,Menlo,monospace;font-size:10.5px;color:#333;background:#f0f3f8;",
        "padding:4px 6px;border-radius:4px;margin-top:4px;white-space:pre-wrap;word-break:break-word}",
        ".tok .k{color:#6a1a9a} .miss{color:#a33} .ok{color:#1a7a3a}",
        ".badge{display:inline-block;font-size:10px;padding:1px 6px;border-radius:999px;background:#eee;margin-right:4px}",
        ".badge.loc{background:#e8f5e9;color:#1b5e20} .badge.orient{background:#e3f2fd;color:#0d47a1}",
        ".badge.move{background:#fff3e0;color:#e65100} .badge.na{background:#eceff1;color:#546e7a}",
        ".badge.fail{background:#ffebee;color:#b71c1c}",
        "</style></head><body>",
        "<h1>Cue-spec GT → humanoid EE pose</h1>",
        "<p class='sub'>Each cell is the <b>token GT</b> (Loc/Orient) converted to a single "
        "<code>pose(...)+hold</code> and rendered on Unitree G1. "
        "Alternatives (<code>dir=up|front</code>) pick the <b>first</b> option; "
        "symbolic Loc (<code>mid</code>/<code>shoulder</code>/<code>near_chest</code>) "
        "map to band centers (55/62/85). Confirm whether the EE pose matches the cue intent. "
        "Movement/path tokens are shown as text only (not animated here).</p>",
        "<div class='filt' id='filt'>",
        "<button class='on' data-f='all'>all</button>",
        "<button data-f='arm'>arm Loc/Orient</button>",
        "<button data-f='head'>head-only</button>",
        "<button data-f='bimanual'>bimanual</button>",
        "<button data-f='fail'>missing gif</button>",
        "</div>",
        "<div class='grid' id='grid'>",
    ]

    for entry in sorted(configs, key=lambda e: int(e["idx"])):
        cue = entry["cue"]
        idx = int(entry["idx"])
        m = by_cue.get(cue) or {}
        meta = m.get("meta") or {}
        gif = gif_path_for(OUT_RENDER, "humanoid", idx, cue)
        rel = os.path.relpath(gif, OUT_HTML.parent) if gif.is_file() else ""
        has_xyz = bool(meta.get("xyz"))
        has_dir = bool(meta.get("dir"))
        has_grip = bool(meta.get("grip"))
        head_only = bool(meta.get("head")) and not has_dir and not has_xyz
        bimanual = meta.get("left_arm") == "mirror"
        tags = ["all"]
        if head_only:
            tags.append("head")
        if has_dir or has_xyz or has_grip:
            tags.append("arm")
        if bimanual:
            tags.append("bimanual")
        if not gif.is_file():
            tags.append("fail")

        badges = []
        if has_dir or has_grip:
            badges.append("<span class='badge orient'>Orient</span>")
        if has_xyz:
            badges.append("<span class='badge loc'>Loc</span>")
        if not _na(str(m.get("required_move") or "—")) or not _na(
            str(m.get("required_path") or "—")
        ):
            badges.append("<span class='badge move'>Move/Path (text)</span>")
        if head_only:
            badges.append("<span class='badge na'>head</span>")
        if not gif.is_file():
            badges.append("<span class='badge fail'>no gif</span>")

        img = (
            f"<img src='{html_mod.escape(rel)}' alt='{html_mod.escape(cue)}'/>"
            if rel
            else "<div style='aspect-ratio:1;background:#ddd;display:flex;align-items:center;"
            "justify-content:center;color:#666;font-size:12px'>missing</div>"
        )
        req_pose = html_mod.escape(str(m.get("required_pose") or ""))
        exec_code = html_mod.escape(str(m.get("exec_code") or "").replace("\n", " "))
        move = html_mod.escape(str(m.get("required_move") or "—"))
        path = html_mod.escape(str(m.get("required_path") or "—"))
        xyz = meta.get("xyz") or {}
        xyz_s = ", ".join(f"{k}={v}" for k, v in xyz.items()) or "—"
        parts.append(
            f"<div class='card' data-tags='{' '.join(tags)}'>"
            f"{img}"
            f"<div class='meta'><b>{idx}. {html_mod.escape(cue)}</b><br/>"
            f"{''.join(badges)}</div>"
            f"<div class='tok'><span class='k'>REQUIRED</span> {req_pose}\n"
            f"<span class='k'>EXEC</span>    {exec_code}\n"
            f"<span class='k'>XYZ</span>     {html_mod.escape(xyz_s)}\n"
            f"<span class='k'>MOVE</span>    {move}\n"
            f"<span class='k'>PATH</span>    {path}</div>"
            f"</div>"
        )

    parts += [
        "</div>",
        "<script>",
        "const btns=[...document.querySelectorAll('#filt button')];",
        "const cards=[...document.querySelectorAll('.card')];",
        "btns.forEach(b=>b.onclick=()=>{",
        "  btns.forEach(x=>x.classList.toggle('on',x===b));",
        "  const f=b.dataset.f;",
        "  cards.forEach(c=>{c.style.display=(f==='all'||c.dataset.tags.split(' ').includes(f))?'':'none';});",
        "});",
        "</script></body></html>",
    ]
    OUT_HTML.write_text("\n".join(parts), encoding="utf-8")
    n_gif = sum(
        1
        for e in configs
        if gif_path_for(OUT_RENDER, "humanoid", int(e["idx"]), e["cue"]).is_file()
    )
    print(f"[html] {OUT_HTML}  gifs={n_gif}/{len(configs)}", flush=True)
    return OUT_HTML


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="optional cue limit for smoke test")
    args = ap.parse_args()

    configs = build_configs()
    if args.limit > 0:
        configs = sorted(configs, key=lambda e: int(e["idx"]))[: args.limit]
        print(f"[limit] using first {len(configs)} cues", flush=True)
    if not args.skip_render:
        drain_renders(configs, batch=args.batch, force=args.force)
    build_html(configs if args.limit else json.loads(
        (OUT_CFG / "embodiment_motion_code_all.json").read_text(encoding="utf-8")
    ))


if __name__ == "__main__":
    main()
