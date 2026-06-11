#!/usr/bin/env python3
"""Render GT component tail + GT neg-axis; build mp4/alpha; review HTML."""
from __future__ import annotations

import html as html_module
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _REPO / "adhoc" / "vlm_test", _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import testset_utils  # noqa: E402
from motion_gt_tail_builder import (  # noqa: E402
    _first_tail_step,
    apply_single_element_variant,
    build_config_from_gt_pose_and_component,
)
from motion_neg_axis_pick import primary_axis_from_component  # noqa: E402
from score_pilot40_motion_gt_components import _build_annotation_map  # noqa: E402
from verify_pose_vlm import _movement_summary  # noqa: E402

BASE_CFG = (
    _REPO
    / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_gt_fixed_pose_pilot40.json"
)
GEN_GIF_DIR = _REPO / "data/results/visualize/gt_fixed_pose_pilot20_hz10/IIWA"
OUT_ROOT = _REPO / "data/results/render/manipulator/motion_gt_compare"
CFG_DIR = OUT_ROOT / "configs"
OUT_HTML = _REPO / "data/results/html/manipulator/motion_gt_compare_review.html"
GEN_MANIFEST = OUT_ROOT / "manifest_generation_pilot40.json"
METRICS_JSON = _REPO / "data/results/verify/pilot40_motion_verify_metrics.json"
ROBOT = "IIWA"
HZ = 10


def _rows() -> list[dict[str, Any]]:
    return sorted(json.loads(BASE_CFG.read_text(encoding="utf-8")), key=lambda r: int(r["idx"]))


def _annotation_for_row(r: dict[str, Any], anns: list[dict[str, Any]]) -> dict[str, Any] | None:
    cue = r.get("cue")
    idx = int(r["idx"])
    for a in anns:
        if a.get("cue") and cue and a.get("cue") == cue:
            return a
    for a in anns:
        if int(a["cue_idx"]) == idx and not a.get("cue"):
            return a
    for a in anns:
        if int(a["cue_idx"]) == idx:
            return a
    return None


def _write_configs(rows: list[dict[str, Any]]) -> dict[str, Path]:
    anns = _build_annotation_map()
    positive: list[dict[str, Any]] = []
    neg_axis: list[dict[str, Any]] = []
    for r in rows:
        idx = int(r["idx"])
        ann = _annotation_for_row(r, anns)
        comp = (ann or {}).get("component")
        pos = build_config_from_gt_pose_and_component(r, comp, state_tag="gt_component_positive")
        if not pos:
            continue
        positive.append(pos)
        pax = primary_axis_from_component(comp)
        v = apply_single_element_variant(pos, "axis", primary_axis=pax)
        if v:
            v["state"] = "gt_component_neg_axis"
            neg_axis.append(v)
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "gt": CFG_DIR / "pilot40_gt_component.json",
        "neg_axis": CFG_DIR / "pilot40_gt_neg_axis.json",
    }
    paths["gt"].write_text(json.dumps(positive, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["neg_axis"].write_text(json.dumps(neg_axis, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


def _render(cfg: Path, out_dir: Path, indices: list[int], *, skip_existing: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "micromamba",
        "run",
        "-n",
        "robosuite",
        "python",
        str(_HERE / "render.py"),
        f"--config_json={cfg}",
        f"--output_dir={out_dir}",
        f"--sim_robot={ROBOT}",
        f"--hz={HZ}",
        f"--cue_indices={','.join(str(i) for i in indices)}",
        f"--skip_existing={skip_existing}",
        "--do_html=False",
        "--auto_generate_if_missing=False",
    ]
    subprocess.run(cmd, check=False)


def _pose_id_for_row(row: dict[str, Any]) -> int | None:
    gfp = row.get("gt_fixed_first_pose") or {}
    pid = gfp.get("pose_id")
    return int(pid) if pid is not None else None


def _built_tail_summary(st: dict[str, Any] | None) -> str | None:
    """Human-readable summary of the config step we actually render."""
    if not st:
        return None
    p = st.get("parameters") or {}
    t = st.get("type")
    if t == "movement":
        dirs = p.get("directions") or []
        if not dirs:
            return None
        deg = dirs[0].get("degrees") or {}
        if not deg:
            return None
        parts = []
        for ax, val in deg.items():
            if isinstance(val, (int, float)):
                parts.append(f"{ax}{'+' if val > 0 else '-'}")
        j = p.get("joint", "?")
        return f"movement {j} ({', '.join(parts)})"
    if t == "path":
        if p.get("shape") == "line":
            ax = str(p.get("axis", "?"))
            dist = p.get("distance")
            if isinstance(dist, (int, float)):
                return f"path line {ax} distance {float(dist):+.2f}m"
            return f"path line {ax}"
        plane = str(p.get("plane", "?")).lower()
        sweep = p.get("sweep")
        direction = p.get("direction")
        extra = ""
        if isinstance(sweep, (int, float)):
            extra += f" sweep {float(sweep):g}°"
        if direction:
            extra += f" {direction}"
        return f"path arc plane {plane}{extra}"
    return None


def _neg_axis_flip_label(pos_cfg: dict[str, Any], neg_cfg: dict[str, Any] | None) -> str:
    if not neg_cfg:
        return "—"
    meta = neg_cfg.get("neg_axis_meta") or {}
    if meta.get("neg_axis") and meta.get("true_axis"):
        parts = [f"{meta['true_axis']} → {meta['neg_axis']}"]
        if meta.get("neg_joint"):
            parts.append(f"j={meta['neg_joint']}")
        leak = meta.get("leak_true_m")
        if leak is not None:
            parts.append(f"parallel_to_gt={leak}m")
        perp = meta.get("gain_neg_m")
        if perp is not None:
            parts.append(f"perp={perp}m")
        return " ".join(parts)
    if meta.get("neg_plane"):
        avoid = meta.get("avoid_axis")
        avoid_s = f" (avoid {avoid})" if avoid else ""
        return f"arc {meta.get('true_plane')} → {meta['neg_plane']}{avoid_s}"
    pst = _first_tail_step(pos_cfg)
    nst = _first_tail_step(neg_cfg)
    if not pst or not nst:
        return "—"
    pp = pst.get("parameters") or {}
    np_ = nst.get("parameters") or {}
    if pst.get("type") == "movement" and nst.get("type") == "movement":
        pdeg = (pp.get("directions") or [{}])[0].get("degrees") or {}
        ndeg = (np_.get("directions") or [{}])[0].get("degrees") or {}
        if pdeg and ndeg:
            old_ax = next(iter(pdeg.keys()))
            new_ax = next(iter(ndeg.keys()))
            return f"{old_ax} → {new_ax}"
    if pst.get("type") == "path" and nst.get("type") == "path":
        if pp.get("shape") == "line" and np_.get("shape") == "line":
            old_ax = str(pp.get("axis", "?"))
            new_ax = str(np_.get("axis", "?"))
            return f"line {old_ax} → line {new_ax}"
        old_pl = str(pp.get("plane", "?")).lower()
        new_pl = str(np_.get("plane", "?")).lower()
        return f"arc {old_pl} → arc {new_pl}"
    return "—"


def _tail_steps_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_pose = False
    for st in row.get("movements") or []:
        if st.get("type") == "pose":
            seen_pose = True
            continue
        if seen_pose:
            out.append(st)
    return out


def _llm_config_note(row: dict[str, Any]) -> str:
    tail = _tail_steps_from_row(row)
    if not tail:
        return "config <code>(no tail steps)</code>"
    parts = [_movement_summary(row)]
    for i, st in enumerate(tail, 1):
        s = _built_tail_summary(st)
        if s:
            parts.append(f"step{i}: <code>{html_module.escape(s)}</code>")
    body = "<br/>".join(parts)
    tail_json = html_module.escape(json.dumps(tail, indent=2, ensure_ascii=False))
    return (
        f'<p class="axis-note">{body}</p>'
        f'<details class="cfg-detail"><summary>tail JSON</summary>'
        f"<pre>{tail_json}</pre></details>"
    )


def _cfg_by_idx(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {int(r["idx"]): r for r in json.loads(path.read_text(encoding="utf-8"))}


def _pick_gif(d: Path, cue: str) -> Path | None:
    if not d.is_dir():
        return None
    c = sorted(
        [p for p in d.glob("*.gif") if f"_{ROBOT}_{cue}_" in p.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return c[0] if c else None


def _gif_to_mp4(gif: Path, mp4: Path) -> None:
    mp4.parent.mkdir(parents=True, exist_ok=True)
    ff = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    subprocess.run(
        [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(gif), "-pix_fmt", "yuv420p", str(mp4)],
        check=False,
    )


def _alpha(cfg_path: Path, idx: int, cue: str, out_png: Path, *, force: bool = False) -> None:
    pose_id = 0
    for row in json.loads(cfg_path.read_text(encoding="utf-8")):
        if int(row["idx"]) == idx:
            pose_id = int((row.get("gt_fixed_first_pose") or {}).get("pose_id") or 0)
            break
    sample = {
        "sample_id": testset_utils._safe_name(f"gtcmp_{idx}_{cue}"),
        "testset": "iconic",
        "cue_idx": idx,
        "cue": cue,
        "config_path": str(cfg_path),
        "gif_path": str(cfg_path),
        "selected_pose_id": pose_id,
        "meta": {},
    }
    img, _ = testset_utils.build_tile_figure_sim_trajectory_panel(
        sample, ROBOT, HZ, canonical="alpha_frame_trajectory", force=force
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png, format="PNG", optimize=False)


def _media_pack(
    gif: Path | None,
    cfg_path: Path,
    idx: int,
    cue: str,
    sub: Path,
    key: str,
    *,
    refresh_mp4: bool = False,
    force_alpha: bool = False,
) -> dict[str, str | None]:
    out: dict[str, str | None] = {"gif": str(gif) if gif else None, "mp4": None, "alpha": None}
    if not gif or not gif.is_file():
        return out
    mp4 = sub / "mp4" / f"{key}.mp4"
    alpha = sub / "alpha_frame_trajectory" / f"{key}.png"
    mp4.parent.mkdir(parents=True, exist_ok=True)
    alpha.parent.mkdir(parents=True, exist_ok=True)
    if refresh_mp4 or not mp4.is_file():
        _gif_to_mp4(gif, mp4)
    if force_alpha or not alpha.is_file():
        try:
            _alpha(cfg_path, idx, cue, alpha, force=force_alpha)
        except Exception as e:
            print(f"[alpha fail] {cue} {key}: {e}", flush=True)
    out["mp4"] = str(mp4) if mp4.is_file() else None
    out["alpha"] = str(alpha) if alpha.is_file() else None
    return out


def _rel(p: str | None) -> str:
    if not p:
        return ""
    return Path(os.path.relpath(p, OUT_HTML.parent)).as_posix()


def write_generation_manifest(rows: list[dict[str, Any]], anns: list[dict[str, Any]]) -> Path:
    """
    Manifest for ``verify_motion_component_gemini.py``: alpha/mp4 under
    ``motion_gt_compare/media/generation`` (sim trajectory from LLM config).
    """
    media = OUT_ROOT / "media" / "generation"
    manifest_rows: list[dict[str, Any]] = []
    for r in rows:
        idx = int(r["idx"])
        cue = r["cue"]
        ann = _annotation_for_row(r, anns) or {}
        comp = ann.get("component")
        if not comp:
            continue
        key = f"{idx:03d}_{cue}_gen"
        alpha = media / "alpha_frame_trajectory" / f"{key}.png"
        mp4 = media / "mp4" / f"{key}.mp4"
        gif = _pick_gif(GEN_GIF_DIR, cue)
        manifest_rows.append(
            {
                "cue_idx": idx,
                "cue": cue,
                "description": r.get("description", ""),
                "annotation_raw": ann.get("annotation_raw", ""),
                "component_gt": comp,
                "config_path": str(BASE_CFG),
                "gif": str(gif) if gif else None,
                "mp4": str(mp4) if mp4.is_file() else None,
                "alpha_frame_trajectory": str(alpha) if alpha.is_file() else None,
            }
        )
    GEN_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    GEN_MANIFEST.write_text(
        json.dumps({"rows": manifest_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return GEN_MANIFEST


def _metrics_html() -> str:
    if not METRICS_JSON.is_file():
        return "<p>Run score_pilot40_motion_verify_metrics.py after text verify.</p>"
    m = json.loads(METRICS_JSON.read_text(encoding="utf-8"))["metrics"]
    return f"""
<div class="summary">
  <h2>Accuracy vs component GT</h2>
  <table>
    <tr><th>Method</th><th>Generation / verifying tail</th><th>Detection (appropriate ↔ gen match)</th></tr>
    <tr><td>Generation only</td><td>{m['generation_accuracy']['pct']}</td><td>—</td></tr>
    <tr><td>Text verify</td><td>{m['text_verifying_accuracy']['pct']}</td><td>{m['text_detection_accuracy']['pct']}</td></tr>
    <tr><td>VLM verify (alpha_frame_trajectory)</td><td>{m['vlm_verifying_accuracy']['pct']}</td><td>{m['vlm_detection_accuracy']['pct']}</td></tr>
  </table>
  <p>Winner verifying: <b>{m['winner']['better_verifying_accuracy']}</b> |
     Winner detection: <b>{m['winner']['better_detection_accuracy']}</b></p>
</div>"""


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--html-only", action="store_true", help="Skip MuJoCo render; rebuild HTML from existing GIFs")
    ap.add_argument(
        "--force-rerender",
        action="store_true",
        help="Re-render all GIFs with fixed pose_id (skip_existing=False) and refresh mp4",
    )
    ap.add_argument(
        "--rerender-gt-only",
        action="store_true",
        help="Re-render only GT + neg-axis columns (after tail-builder fix)",
    )
    ap.add_argument(
        "--regenerate-alpha",
        action="store_true",
        help="Rebuild all alpha_frame_trajectory PNGs (matches current GIF/config)",
    )
    ap.add_argument(
        "--cue-indices",
        default=None,
        help="Comma-separated cue idx to render (default: all rows in base config)",
    )
    args = ap.parse_args()

    rows = _rows()
    render_indices = [int(r["idx"]) for r in rows]
    if args.cue_indices:
        render_indices = [int(x.strip()) for x in args.cue_indices.split(",") if x.strip()]
    indices = render_indices
    anns = _build_annotation_map()
    ann_by_idx = {int(a["cue_idx"]): a for a in anns}
    cfgs = _write_configs(_rows())
    gt_cfg_by_idx = _cfg_by_idx(cfgs["gt"])
    neg_cfg_by_idx = _cfg_by_idx(cfgs["neg_axis"])
    skip_existing = not (args.force_rerender or args.rerender_gt_only)
    refresh_mp4 = args.force_rerender or args.rerender_gt_only
    force_alpha = args.regenerate_alpha or args.force_rerender or args.rerender_gt_only

    gen_out = GEN_GIF_DIR.parent
    gt_dir = OUT_ROOT / "gt_positive"
    ax_dir = OUT_ROOT / "gt_neg_axis"
    if not args.html_only:
        if not args.rerender_gt_only:
            print("[render] LLM generation (pose_id fixed) ...", flush=True)
            _render(BASE_CFG, gen_out, indices, skip_existing=skip_existing)
        print("[render] gt_positive ...", flush=True)
        _render(cfgs["gt"], gt_dir, indices, skip_existing=skip_existing)
        print("[render] gt_neg_axis ...", flush=True)
        _render(cfgs["neg_axis"], ax_dir, indices, skip_existing=skip_existing)

    media = OUT_ROOT / "media"
    cards = []
    for r in rows:
        idx = int(r["idx"])
        cue = r["cue"]
        gen_gif_path = _pick_gif(GEN_GIF_DIR, cue)
        gen = _media_pack(
            gen_gif_path,
            BASE_CFG,
            idx,
            cue,
            media / "generation",
            f"{idx:03d}_{cue}_gen",
            refresh_mp4=refresh_mp4,
            force_alpha=force_alpha,
        )
        gt = _media_pack(
            _pick_gif(gt_dir / ROBOT, cue),
            cfgs["gt"],
            idx,
            cue,
            media / "gt",
            f"{idx:03d}_{cue}_gt",
            refresh_mp4=refresh_mp4,
            force_alpha=force_alpha,
        )
        ax = _media_pack(
            _pick_gif(ax_dir / ROBOT, cue),
            cfgs["neg_axis"],
            idx,
            cue,
            media / "neg_axis",
            f"{idx:03d}_{cue}_axis",
            refresh_mp4=refresh_mp4,
            force_alpha=force_alpha,
        )
        pid = _pose_id_for_row(r)
        gfp = r.get("gt_fixed_first_pose") or {}
        pose_lbl = (
            f"pose_id={pid} · {gfp.get('dir')},{gfp.get('gripper_orientation')}"
            if pid is not None
            else ""
        )
        ann = _annotation_for_row(r, anns) or ann_by_idx.get(idx) or {}
        gt_ann = str(ann.get("annotation_raw") or "").strip() or "—"
        pos_cfg = gt_cfg_by_idx.get(idx)
        neg_cfg = neg_cfg_by_idx.get(idx)
        built = _built_tail_summary(_first_tail_step(pos_cfg)) if pos_cfg else None
        gt_axis_lbl = f"annotation <code>{gt_ann}</code>"
        if built:
            gt_axis_lbl += f"<br/>built <code>{built}</code>"
        neg_flip = _neg_axis_flip_label(pos_cfg, neg_cfg) if pos_cfg else "—"
        llm_cfg = _llm_config_note(r)
        print(f"[media] {idx} {cue}", flush=True)
        cards.append(
            f"""
<article class="card">
  <h2>{idx}. {cue}</h2>
  <p class="meta">{pose_lbl}</p>
  <p class="desc">{r.get('description','')}</p>
  <div class="grid3">
    <section>
      <h3>LLM generation</h3>
      {llm_cfg}
      <video src="{_rel(gen['mp4'])}" controls loop muted playsinline></video>
      <img src="{_rel(gen['alpha'])}" alt="gen alpha"/>
    </section>
    <section>
      <h3>GT component tail</h3>
      <p class="axis-note">GT axis: {gt_axis_lbl}</p>
      <video src="{_rel(gt['mp4'])}" controls loop muted playsinline></video>
      <img src="{_rel(gt['alpha'])}" alt="gt alpha"/>
    </section>
    <section>
      <h3>GT neg axis only</h3>
      <p class="axis-note">neg axis: <code>{neg_flip}</code></p>
      <video src="{_rel(ax['mp4'])}" controls loop muted playsinline></video>
      <img src="{_rel(ax['alpha'])}" alt="neg axis alpha"/>
    </section>
  </div>
</article>"""
        )

    html = f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>Motion GT compare review</title>
<style>
body{{font-family:system-ui,sans-serif;background:#f5f7fb;margin:0;padding:20px}}
.summary{{background:#fff;border:1px solid #dce1ea;border-radius:10px;padding:14px;margin-bottom:16px}}
.summary table{{border-collapse:collapse;width:100%}}
.summary th,.summary td{{border:1px solid #ddd;padding:8px;text-align:left}}
.card{{background:#fff;border:1px solid #dce1ea;border-radius:10px;padding:14px;margin-bottom:14px}}
.grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}
video,img{{width:100%;border:1px solid #ddd;border-radius:8px;background:#fafafa}}
.desc{{font-size:13px;color:#555}}
.meta{{font-size:12px;color:#666;font-family:ui-monospace,monospace}}
.axis-note{{font-size:12px;color:#333;margin:0 0 6px;line-height:1.4}}
.axis-note code{{background:#f0f4ff;padding:1px 5px;border-radius:4px}}
.cfg-detail{{font-size:12px;margin:0 0 8px}}
.cfg-detail pre{{background:#f4f4f5;padding:8px;border-radius:6px;overflow:auto;max-height:220px;font-size:11px;margin:6px 0 0}}
.dim{{color:#888}}
h3{{font-size:14px;margin:8px 0 4px}}
</style></head><body>
<h1>Motion: LLM gen vs GT component vs GT−axis</h1>
<p class="meta">GIF/mp4 use fixed GT tile-pick <code>pose_id</code> (same as alpha_frame_trajectory).</p>
{_metrics_html()}
{''.join(cards)}
</body></html>"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    manifest_path = write_generation_manifest(rows, anns)
    print(f"wrote {OUT_HTML}")
    print(f"wrote {manifest_path} ({len(json.loads(manifest_path.read_text())['rows'])} cues)")


if __name__ == "__main__":
    main()
