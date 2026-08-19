#!/usr/bin/env python3
"""Build preview HTML for motion verification experiments (40 GT-fixed cues)."""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]

from legacy.motion_generation_core import generate  # noqa: E402

CFG = (
    _REPO
    / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_gt_fixed_pose_pilot40.json"
)
OUT_ROOT = _REPO / "data/results/render/manipulator/verify_preview_40"
MEDIA_DIR = OUT_ROOT / "media"
TMP_CFG_DIR = OUT_ROOT / "tmp_configs"
OUT_HTML = _REPO / "data/results/html/manipulator/motion_verify_preview_40.html"
MANIFEST = OUT_ROOT / "manifest.json"


def _load_rows() -> list[dict[str, Any]]:
    rows = json.loads(CFG.read_text(encoding="utf-8"))
    return sorted(rows, key=lambda r: int(r.get("idx", 0)))


def _first_tail_step(row: dict[str, Any]) -> dict[str, Any] | None:
    for st in row.get("movements", []):
        if st.get("type") != "pose":
            return st
    return None


def _alt_axis(ax: str) -> str:
    for cand in ("x", "y", "z"):
        if cand != ax:
            return cand
    return ax


def _alt_joint(joint: str) -> str:
    for cand in ("shoulder", "elbow", "wrist"):
        if cand != joint:
            return cand
    return joint


def _negative_variant(row: dict[str, Any], kind: str) -> dict[str, Any] | None:
    v = copy.deepcopy(row)
    st = _first_tail_step(v)
    if st is None:
        return None
    p = st.get("parameters") or {}
    t = st.get("type")

    if kind == "axis":
        if t == "movement":
            dirs = p.get("directions") or []
            if not dirs:
                return None
            deg = dirs[0].get("degrees") or {}
            if not deg:
                return None
            old_ax = next(iter(deg.keys()))
            new_ax = _alt_axis(old_ax)
            deg[new_ax] = deg.pop(old_ax)
            dirs[0]["degrees"] = deg
            return v
        if t == "path":
            shape = p.get("shape")
            if shape == "line":
                p["axis"] = _alt_axis(str(p.get("axis", "x")))
                st["parameters"] = p
                return v
            if shape in ("arc", "circle"):
                plane = str(p.get("plane", "xy")).lower()
                p["plane"] = {"xy": "yz", "yz": "xz", "xz": "xy", "zx": "xy"}.get(plane, "yz")
                st["parameters"] = p
                return v
        return None

    if kind == "joint":
        if t != "movement":
            return None
        p["joint"] = _alt_joint(str(p.get("joint", "shoulder")))
        st["parameters"] = p
        return v

    if kind == "direction":
        if t == "movement":
            dirs = p.get("directions") or []
            if not dirs:
                return None
            deg = dirs[0].get("degrees") or {}
            if not deg:
                return None
            ax = next(iter(deg.keys()))
            val = deg[ax]
            if isinstance(val, (int, float)):
                deg[ax] = -val
                dirs[0]["degrees"] = deg
                return v
            return None
        if t == "path":
            shape = p.get("shape")
            if shape == "line" and isinstance(p.get("distance"), (int, float)):
                p["distance"] = -float(p["distance"])
                st["parameters"] = p
                return v
            if shape in ("arc", "circle"):
                d = str(p.get("direction", "ccw")).lower()
                p["direction"] = "cw" if d == "ccw" else "ccw"
                st["parameters"] = p
                return v
        return None

    return None


def _render_row(row: dict[str, Any], tag: str, *, robot: str = "IIWA", hz: int = 10) -> Path | None:
    cue = row["cue"]
    idx = int(row["idx"])
    pose_id = int((row["movements"][0]["parameters"]["pose"]).get("pose_id", 0))
    cfg_path = TMP_CFG_DIR / f"{idx:03d}_{cue}_{tag}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps([row], indent=2, ensure_ascii=False), encoding="utf-8")
    suffix = f"{tag}_p{pose_id}"
    out_dir = MEDIA_DIR / "gif"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        gif = generate(
            robot=robot,
            cue=cue,
            cue_idx=idx,
            pose_index=pose_id,
            jsonl_path=str(_REPO / "data/seed/_remainder/closest_poses_results.jsonl"),
            config_path=str(cfg_path),
            output_dir=str(out_dir),
            hz=hz,
            top_k=1,
            gif_filename_suffix=suffix,
        )
    except Exception:
        return None
    return Path(gif) if gif else None


def _gif_to_mp4(gif: Path, mp4: Path) -> bool:
    ff = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(gif), "-pix_fmt", "yuv420p", str(mp4)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def _alpha_from_gif(gif: Path, out_png: Path) -> bool:
    try:
        im = Image.open(gif).convert("RGBA")
    except Exception:
        return False
    frames: list[Image.Image] = []
    try:
        i = 0
        while True:
            im.seek(i)
            frames.append(im.convert("RGBA").copy())
            i += 1
    except EOFError:
        pass
    if not frames:
        return False

    base = Image.new("RGBA", frames[0].size, (255, 255, 255, 255))
    n = len(frames)
    step = max(1, n // 12)
    sampled = [frames[i] for i in range(0, n, step)]
    for i, fr in enumerate(sampled):
        a = int(45 + (180 * (i + 1) / len(sampled)))
        over = fr.copy()
        over.putalpha(a)
        base.alpha_composite(over)

    # simple trajectory from image center-of-mass proxy (non-white bbox center)
    draw = ImageDraw.Draw(base)
    pts = []
    for fr in sampled:
        bbox = fr.getbbox()
        if bbox:
            x = (bbox[0] + bbox[2]) // 2
            y = (bbox[1] + bbox[3]) // 2
            pts.append((x, y))
    if len(pts) > 1:
        draw.line(pts, fill=(255, 215, 0, 220), width=3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(out_png, format="PNG", optimize=True)
    return True


def _build_media_for_row(row: dict[str, Any], tag: str) -> dict[str, Any]:
    idx = int(row["idx"])
    cue = row["cue"]
    base = f"{idx:03d}_{cue}_{tag}"
    gif_out = MEDIA_DIR / "gif" / f"{base}.gif"
    mp4_out = MEDIA_DIR / "mp4" / f"{base}.mp4"
    alpha_out = MEDIA_DIR / "alpha" / f"{base}.png"

    if not gif_out.is_file():
        rendered = _render_row(row, tag)
        if rendered and rendered.is_file():
            rendered.replace(gif_out)

    ok_gif = gif_out.is_file()
    ok_mp4 = mp4_out.is_file() or (ok_gif and _gif_to_mp4(gif_out, mp4_out))
    ok_alpha = alpha_out.is_file() or (ok_gif and _alpha_from_gif(gif_out, alpha_out))
    return {
        "gif": str(gif_out) if ok_gif else None,
        "mp4": str(mp4_out) if ok_mp4 else None,
        "alpha": str(alpha_out) if ok_alpha else None,
    }


def _rel(p: str | None) -> str:
    if not p:
        return ""
    return Path(p).relative_to(OUT_HTML.parent).as_posix()


def build_preview() -> None:
    rows = _load_rows()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    data = []
    for row in rows:
        pos = _build_media_for_row(row, "pos")
        axis_row = _negative_variant(row, "axis")
        joint_row = _negative_variant(row, "joint")
        dir_row = _negative_variant(row, "direction")
        axis = _build_media_for_row(axis_row, "neg_axis") if axis_row else {"gif": None, "mp4": None, "alpha": None}
        joint = _build_media_for_row(joint_row, "neg_joint") if joint_row else {"gif": None, "mp4": None, "alpha": None}
        direc = _build_media_for_row(dir_row, "neg_direction") if dir_row else {"gif": None, "mp4": None, "alpha": None}
        data.append(
            {
                "idx": row["idx"],
                "cue": row["cue"],
                "description": row.get("description", ""),
                "positive": pos,
                "neg_axis": axis,
                "neg_joint": joint,
                "neg_direction": direc,
            }
        )
        print(f"[done] {row['idx']} {row['cue']}", flush=True)

    MANIFEST.write_text(json.dumps({"rows": data}, indent=2, ensure_ascii=False), encoding="utf-8")

    cards = []
    for r in data:
        cards.append(
            f"""
<article class="card">
  <h2>{r['idx']}. {r['cue']}</h2>
  <p class="desc">{r['description']}</p>
  <div class="exp">
    <h3>1) VLM verify (mp4 / alpha_frame_trajectory)</h3>
    <div class="grid2">
      <video src="{_rel(r['positive']['mp4'])}" controls muted loop></video>
      <img src="{_rel(r['positive']['alpha'])}" alt="alpha trajectory"/>
    </div>
  </div>
  <div class="exp">
    <h3>2) text verify</h3>
    <pre>Input: config JSON (groundtruth start pose + generated tail)\nQuestions: correctness / improvements / reasoning quality</pre>
  </div>
  <div class="exp">
    <h3>3) Compare (positive vs negative)</h3>
    <div class="cmp">
      <div><h4>axis different</h4><video src="{_rel(r['neg_axis']['mp4'])}" controls muted loop></video><img src="{_rel(r['neg_axis']['alpha'])}" /></div>
      <div><h4>joint different</h4><video src="{_rel(r['neg_joint']['mp4'])}" controls muted loop></video><img src="{_rel(r['neg_joint']['alpha'])}" /></div>
      <div><h4>direction different</h4><video src="{_rel(r['neg_direction']['mp4'])}" controls muted loop></video><img src="{_rel(r['neg_direction']['alpha'])}" /></div>
    </div>
  </div>
</article>
"""
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>Motion verify preview 40</title>
<style>
body{{font-family:system-ui,sans-serif;background:#f5f7fb;margin:0;padding:20px}}
.card{{background:#fff;border:1px solid #dfe3ec;border-radius:10px;padding:14px;margin-bottom:16px}}
h2{{margin:0 0 6px}} .desc{{color:#444}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.cmp{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}
video,img{{width:100%;border:1px solid #ddd;border-radius:8px;background:#fafafa}}
pre{{background:#f2f2f2;padding:10px;border-radius:8px;white-space:pre-wrap}}
</style></head><body>
<h1>Verification preview (40 cues)</h1>
<p>Start pose is GT-fixed. Compare negatives: axis-only / joint-only / direction-only.</p>
{''.join(cards)}
</body></html>"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_HTML}")


if __name__ == "__main__":
    build_preview()

