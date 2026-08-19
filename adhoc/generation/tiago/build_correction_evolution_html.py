#!/usr/bin/env python3
"""Build / refresh exp1 correction evolution HTML (original + corr1 + corr2 per cue)."""
from __future__ import annotations

import html
import json
import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_ROBOTARM = _REPO / "adhoc/generation/robotarm"
for p in (_REPO, _ROBOTARM, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from correction_lib import (  # noqa: E402
    correction_cue_dir,
    correction_record_path,
    correction_stem,
    config_snapshot_path,
    default_config_path,
    list_correction_rounds,
    load_config_row,
    load_correction_record,
)
from render_paths import (  # noqa: E402
    gt_render_role,
    render_exp_dir,
    render_gif_path,
    render_media_paths,
    render_role_for_round,
)

_ENV = None
_RENDER_MOD = None


def _get_render_mod():
    global _RENDER_MOD
    if _RENDER_MOD is None:
        import importlib.util

        mod_path = _HERE / "legacy" / "render_mobile_config.py"
        spec = importlib.util.spec_from_file_location("tiago_render_mobile_config", mod_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {mod_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _RENDER_MOD = mod
    return _RENDER_MOD


def _get_render_env():
    global _ENV
    if _ENV is None:
        _ENV = _get_render_mod()._make_env()
    return _ENV


def _close_render_env() -> None:
    global _ENV
    if _ENV is not None:
        try:
            _ENV.close()
        except Exception:
            pass
        _ENV = None


def _pool_pose_from_row(row: dict) -> dict:
    first: dict | None = None
    for step in row.get("movements") or []:
        if step.get("type") != "pose":
            continue
        pose = (step.get("parameters") or {}).get("pose") or {}
        if not (pose.get("arm_position") and pose.get("gripper_orientation")):
            continue
        if first is None:
            first = pose
        if str(pose.get("arm_position", "")).strip().lower() != "still":
            return pose
    return first or {}


def _pose_pool_xyz_html(row: dict, *, ignore_xyz: bool) -> str:
    pose = _pool_pose_from_row(row)
    if not pose or str(pose.get("arm_position", "")).strip().lower() == "still":
        return ""
    try:
        env = _get_render_env()
        rmc = _get_render_mod()
    except Exception as exc:
        return f'<div class="pool-xyz ee-missing">Pool xyz: env unavailable ({_esc(exc)})</div>'
    ranked = rmc.list_pose_bank_candidates(env, pose, top_k=0, ignore_xyz=ignore_xyz)
    entries = ranked.get("entries") or []
    if len(entries) <= 1:
        return ""
    axes = rmc.pose_xyz_axes_specified(pose)
    rows_html: list[str] = []
    for ent in entries:
        cells: list[str] = [f"<td>#{ent.get('rank')}</td>", f"<td>{ent.get('tile_idx', '—')}</td>"]
        for axis in "xyz":
            sel = float(ent.get(axis, 50))
            if axis in axes:
                req = float(rmc._resolve_level(pose[axis], axis))
                delta = sel - req
                warn = ' class="ee-warn"' if abs(delta) > 5 else ""
                cells.append(
                    f"<td{warn}>{axis} {sel:.0f} (req {req:.0f}, Δ{delta:+.0f})</td>"
                )
            else:
                cells.append(f"<td>{axis} {sel:.0f}</td>")
        dist = ent.get("dist")
        dist_cell = f"{float(dist):.1f}" if dist is not None else "—"
        cells.append(f"<td>{dist_cell}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    hdr = (
        "<tr><th>#</th><th>tile</th>"
        + "".join(f"<th>{a}</th>" for a in "xyz")
        + "<th>Σ|Δ|</th></tr>"
    )
    return (
        '<div class="pool-xyz"><b>Pose pool xyz</b>'
        '<table class="pool-xyz-table"><thead>'
        + hdr
        + "</thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table></div>"
    )


def _pose_steps(row: dict) -> list[tuple[int, dict]]:
    out: list[tuple[int, dict]] = []
    for i, step in enumerate(row.get("movements") or []):
        if step.get("type") == "pose":
            out.append((i, (step.get("parameters") or {}).get("pose") or {}))
    return out


def _pose_ee_xyz_report(env, pose: dict, *, ignore_xyz: bool) -> dict:
    rmc = _get_render_mod()

    axes: dict[str, dict[str, object]] = {}
    for axis in "xyz":
        if axis not in pose:
            axes[axis] = {"raw": "—", "req": None}
        else:
            raw = pose.get(axis, "—")
            req = rmc._resolve_level(pose[axis], axis)
            axes[axis] = {"raw": raw, "req": req}

    ranked = rmc.rank_pose_bank_topk(env, pose, top_k=1, ignore_xyz=ignore_xyz)
    ent = (ranked.get("entries") or [None])[0] or {}
    sel = {a: ent.get(a) for a in "xyz"}
    rank_mode = ranked.get("xyz_rank_mode") or rmc.pose_xyz_rank_mode(pose, ignore_xyz=ignore_xyz)
    return {
        "arm": pose.get("arm_position"),
        "grip": pose.get("gripper_orientation"),
        "explicit_xyz": rmc.pose_has_explicit_xyz(pose),
        "xyz_rank_mode": rank_mode,
        "ignore_xyz": ignore_xyz,
        "axes": axes,
        "sel": sel,
        "dist": ent.get("dist"),
        "tile_idx": ent.get("tile_idx"),
        "n_pool": ranked.get("n_pool", 0),
        "source": ranked.get("source", "?"),
    }


def _ee_xyz_html(reports: list[tuple[str, dict]]) -> str:
    if not reports:
        return ""
    parts = ['<div class="ee-xyz">']
    for label, r in reports:
        axis_lines: list[str] = []
        for axis in "xyz":
            ax = r["axes"][axis]
            sel = r["sel"].get(axis)
            req = ax["req"]
            raw = ax["raw"]
            if req is None:
                axis_lines.append(f"{axis}: (not specified) → sel {sel if sel is None else f'{float(sel):.0f}'}")
                continue
            req_f = float(req)  # type: ignore[arg-type]
            if sel is None:
                axis_lines.append(f"{axis}: req {raw} ({req_f:.0f}) → sel —")
                continue
            delta = float(sel) - req_f
            warn = ' class="ee-warn"' if abs(delta) > 5 else ""
            axis_lines.append(
                f'<span{warn}>{axis}: req {raw} ({req_f:.0f}) → sel {float(sel):.0f} '
                f'(Δ{delta:+.0f})</span>'
            )
        if r["ignore_xyz"]:
            mode = "ignore_xyz"
        else:
            mode = str(r.get("xyz_rank_mode") or "xyz-rank")
        exp = "explicit xyz" if r["explicit_xyz"] else "no xyz"
        dist = r.get("dist")
        dist_s = f" Σ|Δ|={float(dist):.1f}" if dist is not None else ""
        meta = (
            f'{r.get("arm", "?")}/{r.get("grip", "?")} · pool={r.get("n_pool", 0)} '
            f'tile={r.get("tile_idx", "—")} · {r.get("source", "?")} · {mode} · {exp}{dist_s}'
        )
        parts.append(
            f'<div class="ee-pose"><b>{_esc(label)}</b> <span class="ee-meta">{_esc(meta)}</span><br>'
            + "<br>".join(axis_lines)
            + "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _ee_xyz_for_row(row: dict, *, ignore_xyz: bool) -> str:
    poses = _pose_steps(row)
    if not poses:
        return ""
    try:
        env = _get_render_env()
    except Exception as exc:
        return f'<div class="ee-xyz ee-missing">EE xyz: env unavailable ({_esc(exc)})</div>'
    reports: list[tuple[str, dict]] = []
    for n, (_idx, pose) in enumerate(poses, start=1):
        if str(pose.get("arm_position", "")).strip().lower() == "still":
            continue
        label = f"Pose {n}" if len(poses) > 1 else "First pose"
        reports.append((label, _pose_ee_xyz_report(env, pose, ignore_xyz=ignore_xyz)))
    return _ee_xyz_html(reports)

DEFAULT_MODEL_TAG = "gemini-2.5-pro"
DEFAULT_EXP_ID = "1"
DEFAULT_DOMAIN = "tiago"
OUT_DIR = _REPO / "data/results/visualize/tiago/correction/exp1"


def _esc(s: object) -> str:
    return html.escape(str(s))


def _rel(out: Path, p: Path) -> str:
    if not p.is_file():
        return ""
    return _esc(os.path.relpath(p, out.parent).replace("\\", "/"))


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


ADHOC_CORRECTION_SPECS: list[tuple[str, str, str, str]] = [
    ("gt_correction", "GT → correction", "gt-corr", "gt/gt_corr"),
    ("orig_gt_correction", "Original + GT → correction", "orig-gt-corr", "orig_gt_corr"),
    (
        "orig_gt_triple_correction",
        "Original + GT + Orig+GT corr → correction",
        "orig-gt-triple-corr",
        "orig_gt_triple_corr",
    ),
    (
        "orig_gt_triple_correction_shuf",
        "Original + GT + Orig+GT corr → correction (shuffled)",
        "orig-gt-triple-corr-shuf",
        "orig_gt_triple_corr_shuf",
    ),
]


def _is_correction_q_prompt_echo(q_num: int, text: str) -> bool:
    low = (text or "").lower()
    if q_num == 1:
        if "appropriate and representative" in low:
            return True
        if re.search(r'start with\s+["\']?yes', low):
            return True
    if q_num == 2:
        if "entire" in low and "movements" in low:
            return True
        if "cover pose" in low and "path/movement" in low:
            return True
        if re.search(r"if yes,\s*write\s+[\"']?n/a", low):
            return True
    return False


def _extract_correction_q_line(raw_text: str, q_num: int) -> str | None:
    lines = (raw_text or "").splitlines()
    label = rf"^\s*#\s*Correction\s*Q{q_num}\s*:\s*(.*)$"
    candidates: list[str] = []
    prompt_pat = re.compile(label, re.IGNORECASE)
    for i, line in enumerate(lines):
        m = prompt_pat.match(line)
        if not m:
            continue
        ans = m.group(1).strip()
        if _is_correction_q_prompt_echo(q_num, ans):
            for j in range(i + 1, min(i + 4, len(lines))):
                nxt = lines[j].strip()
                if not nxt.startswith("#"):
                    continue
                alt = nxt.lstrip("#").strip()
                if alt and not _is_correction_q_prompt_echo(q_num, alt):
                    candidates.append(alt)
                    break
            continue
        if ans:
            candidates.append(ans)
    return candidates[0] if candidates else None


def _correction_qs(corr: dict) -> tuple[str | None, str | None]:
    q1 = corr.get("correction_q1")
    q2 = corr.get("correction_q2")
    raw = corr.get("raw_text") or corr.get("inner_reasoning") or ""
    if not q1:
        q1 = _extract_correction_q_line(raw, 1)
    if not q2:
        q2 = _extract_correction_q_line(raw, 2)
    return q1, q2


def adhoc_config_path(
    domain: str,
    exp_id: str,
    model_tag: str,
    cue: str,
    tag: str,
) -> Path:
    return (
        _REPO
        / "data/results/motion_configs"
        / domain
        / "exp/correction"
        / model_tag
        / f"exp{exp_id}"
        / cue
        / f"{tag}_config.json"
    )


def adhoc_render_dir(
    model_tag: str,
    exp_id: str,
    cue: str,
    render_subdir: str,
) -> Path:
    del cue, render_subdir
    return render_exp_dir("tiago", model_tag, exp_id)


def gt_config_path(
    domain: str,
    exp_id: str,
    model_tag: str,
    cue: str,
) -> Path:
    return (
        _REPO
        / "data/results/motion_configs"
        / domain
        / "exp/correction"
        / model_tag
        / f"exp{exp_id}"
        / cue
        / "gt_config.json"
    )


def _gt_render_media_paths(
    idx: int,
    cue: str,
    model_tag: str,
    exp_id: str,
) -> tuple[Path, Path, Path | None]:
    gif, mp4, _ = render_media_paths("tiago", model_tag, exp_id, idx, cue, "gt")
    pool = gif.parent / f"{gif.stem}_pose_pool.gif"
    return gif, mp4, pool if pool.is_file() else None


def discover_corrected_cues(
    *,
    domain: str = DEFAULT_DOMAIN,
    exp_id: str = DEFAULT_EXP_ID,
    model_tag: str = DEFAULT_MODEL_TAG,
) -> list[tuple[int, str]]:
    verify_root = _REPO / "data/results/verify" / domain / "correction" / f"exp{exp_id}" / model_tag
    if not verify_root.is_dir():
        return []
    cfg_path = default_config_path(domain, model_tag, exp_id)  # type: ignore[arg-type]
    out: list[tuple[int, str]] = []
    for cue_dir in sorted(verify_root.iterdir()):
        if not cue_dir.is_dir():
            continue
        if not correction_record_path(cue_dir, 1).is_file():
            continue
        cue = cue_dir.name
        try:
            row = load_config_row(cfg_path, cue=cue)
            idx = int(row.get("idx", -1))
        except KeyError:
            idx = -1
        out.append((idx, cue))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def _panel_html(
    out: Path,
    *,
    title: str,
    gif: Path,
    mp4: Path | None,
    cfg: dict,
    q1: str | None = None,
    q2: str | None = None,
    ignore_xyz: bool = False,
    pool_gif: Path | None = None,
    panel_class: str = "panel",
    input_gif: Path | None = None,
) -> str:
    media = ""
    if input_gif and input_gif.is_file():
        in_r = _rel(out, input_gif)
        media += (
            f'<h4 class="pool-title">VLM input (cumulative tiles)</h4>'
            f'<img class="gif input" src="{in_r}" alt="VLM input" loading="lazy">'
        )
    gif_r = _rel(out, gif)
    media += (
        f'<img class="gif" src="{gif_r}" alt="{_esc(title)}" loading="lazy">'
        if gif_r
        else '<p class="missing">GIF missing</p>'
    )
    if pool_gif and pool_gif.is_file():
        pool_r = _rel(out, pool_gif)
        media += (
            f'<h4 class="pool-title">Pose pool (ignore_xyz)</h4>'
            f'<img class="gif pool" src="{pool_r}" alt="pose pool" loading="lazy">'
        )
        media += _pose_pool_xyz_html(cfg, ignore_xyz=True)
    if mp4 and mp4.is_file():
        media += f'<p class="mp4-link"><a href="{_rel(out, mp4)}">MP4</a></p>'
    media += _ee_xyz_for_row(cfg, ignore_xyz=ignore_xyz)
    q_html = ""
    if q1:
        q_html = (
            f'<div class="vq"><strong>Correction Q1:</strong> {_esc(q1)}</div>'
            f'<div class="vq"><strong>Correction Q2:</strong> {_esc(q2 or "")}</div>'
        )
    cfg_body = _esc(json.dumps(cfg, indent=2, ensure_ascii=False))
    return (
        f'<article class="{panel_class}"><h2>{_esc(title)}</h2>{media}{q_html}'
        f'<details><summary>Config</summary><pre>{cfg_body}</pre></details></article>'
    )


def _render_media_paths(
    idx: int,
    cue: str,
    model_tag: str,
    exp_id: str,
    round_n: int | None,
) -> tuple[Path, Path, Path | None]:
    if round_n is None:
        role = "gen"
    else:
        role = render_role_for_round(round_n)
    gif, mp4, _ = render_media_paths("tiago", model_tag, exp_id, idx, cue, role)
    pool = gif.parent / f"{gif.stem}_pose_pool.gif"
    return gif, mp4, pool if pool.is_file() else None


def build_evolution_html(
    *,
    domain: str = DEFAULT_DOMAIN,
    exp_id: str = DEFAULT_EXP_ID,
    model_tag: str = DEFAULT_MODEL_TAG,
    cues: list[str] | None = None,
    out_path: Path | None = None,
) -> Path:
    """Write evolution HTML; auto-discovers all cues with round-1 correction unless ``cues`` set."""
    out = out_path or (OUT_DIR / model_tag / "correction_evolution.html")
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg_path = default_config_path(domain, model_tag, exp_id)  # type: ignore[arg-type]
    if cues is None:
        cue_list = [c for _, c in discover_corrected_cues(domain=domain, exp_id=exp_id, model_tag=model_tag)]
    else:
        cue_list = list(cues)

    toc_links: list[str] = []
    sections: list[str] = []

    try:
        for cue in cue_list:
            base_row = load_config_row(cfg_path, cue=cue)
            idx = int(base_row.get("idx", -1))
            cue_dir = correction_cue_dir(domain, exp_id, model_tag, cue)  # type: ignore[arg-type]
            rounds = list_correction_rounds(cue_dir)

            panels: list[str] = []
            og_gif, og_mp4, _ = _render_media_paths(idx, cue, model_tag, exp_id, None)
            panels.append(
                _panel_html(
                    out,
                    title="Original (exp1)",
                    gif=og_gif,
                    mp4=og_mp4,
                    cfg=base_row,
                    ignore_xyz=False,
                )
            )

            for rn in rounds:
                rec = load_correction_record(cue_dir, rn)
                cfg_path_r = config_snapshot_path(domain, exp_id, model_tag, cue, rn)  # type: ignore[arg-type]
                cfg_row = _load_json(cfg_path_r)[0]
                gif, mp4, pool = _render_media_paths(idx, cue, model_tag, exp_id, rn)
                corr = rec.get("correction") or {}
                q1, q2 = _correction_qs(corr)
                panels.append(
                    _panel_html(
                        out,
                        title=f"Correction {rn}",
                        gif=gif,
                        mp4=mp4,
                        cfg=cfg_row,
                        q1=q1,
                        q2=q2,
                        ignore_xyz=True,
                        pool_gif=pool,
                    )
                )

            gt_cfg_path = gt_config_path(domain, exp_id, model_tag, cue)
            if gt_cfg_path.is_file():
                gt_row = _load_json(gt_cfg_path)[0]
                gt_gif, gt_mp4, gt_pool = _gt_render_media_paths(idx, cue, model_tag, exp_id)
                panels.append(
                    _panel_html(
                        out,
                        title="GT",
                        gif=gt_gif,
                        mp4=gt_mp4,
                        cfg=gt_row,
                        ignore_xyz=True,
                        pool_gif=gt_pool,
                        panel_class="panel gt",
                    )
                )

            for tag, title, panel_cls, render_subdir in ADHOC_CORRECTION_SPECS:
                verify_path = cue_dir / f"{tag}.json"
                cfg_snap = adhoc_config_path(domain, exp_id, model_tag, cue, tag)
                if not verify_path.is_file() or not cfg_snap.is_file():
                    continue
                adhoc_rec = _load_json(verify_path)
                if not isinstance(adhoc_rec, dict):
                    continue
                cfg_row = _load_json(cfg_snap)[0]
                render_base = adhoc_render_dir(model_tag, exp_id, cue, render_subdir)
                safe = cue.replace("/", "_").replace("\\", "_").replace(" ", "_")
                stem = f"mm19_g{idx:02d}_{safe}"
                gif = render_base / f"{stem}.gif"
                mp4 = render_base / f"{stem}.mp4"
                pool = render_base / f"{stem}_pose_pool.gif"
                corr = adhoc_rec.get("correction") or {}
                q1, q2 = _correction_qs(corr)
                input_gif = None
                if tag == "orig_gt_correction":
                    tiled = render_base / "cumulative_2tiles.gif"
                    if tiled.is_file():
                        input_gif = tiled
                if tag == "orig_gt_triple_correction":
                    tiled = render_base / "cumulative_3tiles.gif"
                    if tiled.is_file():
                        input_gif = tiled
                if tag == "orig_gt_triple_correction_shuf":
                    tiled = render_base / "cumulative_3tiles_shuf.gif"
                    if tiled.is_file():
                        input_gif = tiled
                panels.append(
                    _panel_html(
                        out,
                        title=title,
                        gif=gif,
                        mp4=mp4,
                        cfg=cfg_row,
                        q1=q1,
                        q2=q2,
                        ignore_xyz=True,
                        pool_gif=pool if pool.is_file() else None,
                        panel_class=f"panel {panel_cls}",
                        input_gif=input_gif,
                    )
                )

            toc_links.append(f'<a href="#g{idx:02d}">g{idx:02d} {cue}</a>')
            sections.append(
                f'<section class="cue-block" id="g{idx:02d}">'
                f"<h1>g{idx:02d} <code>{_esc(cue)}</code></h1>"
                f'<p class="sub">Original exp1 vs correction rounds + GT experiments ({model_tag})</p>'
                f'<div class="grid">{"".join(panels)}</div></section>'
            )
    finally:
        _close_render_env()

    page = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Correction evolution — exp1 {model_tag}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #fafafa; color: #222; }}
nav.toc {{ margin-bottom: 28px; font-size: 0.95rem; line-height: 1.8; }}
nav.toc a {{ margin-right: 16px; }}
.cue-block {{ margin-bottom: 48px; padding-bottom: 32px; border-bottom: 2px solid #ddd; }}
.cue-block:last-child {{ border-bottom: none; }}
h1 {{ margin-bottom: 8px; }}
.sub {{ color: #666; margin-bottom: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; align-items: start; }}
.panel {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 14px; }}
.panel.gt {{ border-color: #7eb8da; background: #f4faff; }}
.panel.gt-corr {{ border-color: #9ec9e8; background: #eef7ff; }}
.panel.orig-gt-corr {{ border-color: #b8a0e0; background: #f6f2ff; }}
.panel.orig-gt-triple-corr {{ border-color: #6bb89a; background: #f0faf5; }}
.panel.orig-gt-triple-corr-shuf {{ border-color: #d4a86a; background: #fffaf0; }}
.panel h2 {{ margin: 0 0 12px; font-size: 1.05rem; }}
img.gif.input {{ margin-bottom: 10px; opacity: 0.95; }}
img.gif {{ width: 100%; height: auto; display: block; border: 1px solid #eee; border-radius: 4px; }}
img.gif.pool {{ margin-top: 6px; }}
.pool-title {{ margin: 12px 0 6px; font-size: 0.85rem; color: #555; }}
.pool-xyz {{ margin: 8px 0 0; font-size: 0.75rem; }}
.pool-xyz-table {{ width: 100%; border-collapse: collapse; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.72rem; }}
.pool-xyz-table th, .pool-xyz-table td {{ border: 1px solid #e0e4ec; padding: 3px 5px; text-align: left; }}
.pool-xyz-table th {{ background: #f0f2f8; }}
.mp4-link {{ margin: 8px 0 0; font-size: 0.85rem; }}
.ee-xyz {{ margin: 10px 0 0; padding: 8px 10px; background: #f8f9fc; border: 1px solid #e4e8f0; border-radius: 6px; font-size: 0.78rem; line-height: 1.5; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
.ee-pose + .ee-pose {{ margin-top: 8px; padding-top: 8px; border-top: 1px dashed #dde3ee; }}
.ee-meta {{ color: #666; font-family: system-ui, sans-serif; font-size: 0.72rem; }}
.ee-warn {{ color: #b45309; font-weight: 600; }}
.ee-missing {{ color: #c00; }}
.vq {{ font-size: 0.88rem; margin: 10px 0; line-height: 1.45; color: #333; }}
details {{ margin-top: 12px; }}
pre {{ white-space: pre-wrap; font-size: 0.75rem; background: #f4f4f8; padding: 10px; border-radius: 4px; max-height: 320px; overflow: auto; }}
.missing {{ color: #c00; }}
@media (max-width: 1100px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style></head>
<body>
<h1>Correction evolution (exp{exp_id}, {model_tag})</h1>
<nav class="toc">{"".join(toc_links)}</nav>
{"".join(sections)}
</body></html>
"""
    out.write_text(page, encoding="utf-8")

    # Legacy filename (bookmark compat)
    legacy = out.parent / "g24_cover_mouth_gasp.html"
    legacy.write_text(page, encoding="utf-8")
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build correction evolution HTML")
    ap.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    ap.add_argument("--cues", default=None, help="Comma-separated cues (default: auto-discover)")
    args = ap.parse_args()
    cues = [c.strip() for c in args.cues.split(",") if c.strip()] if args.cues else None
    path = build_evolution_html(model_tag=args.model_tag, cues=cues)
    print(f"HTML saved: {path}")


if __name__ == "__main__":
    main()
