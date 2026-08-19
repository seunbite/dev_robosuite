#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import fire

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adhoc.generation.embodiment_sources import (  # noqa: E402
    load_embodiment_block,
    load_cross_config,
    load_see_html_sources as _load_see_html_sources,
    motion_json_rel_list as _motion_json_rel_list,
    repo_rel_to_path as _repo_rel_to_path,
    see_html_sources_path as _see_html_sources_path,
)
from adhoc.generation.utils import announce_output, dated_results_html_path  # noqa: E402
from adhoc.utils.repo_paths import (  # noqa: E402
    dev_robosuite_root,
    motion_configs_results_dir,
    results_render_dir,
    results_subdir,
    seed_yml_dir,
)


def _safe(name: str) -> str:
    return str(name).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _default_config(robot: str) -> Path:
    d = motion_configs_results_dir(robot)
    cands = sorted(d.glob("motion_configs*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        return cands[0]
    return d / "motion_configs.json"


def _load_merged_json_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        if not p.is_file():
            continue
        with open(p, encoding="utf-8") as f:
            chunk = json.load(f)
        if not isinstance(chunk, list):
            raise SystemExit(f"Expected a JSON array in {p}")
        rows.extend(chunk)
    return rows


def _html_slug_for_robot_rk(rk: str) -> str:
    return "manipulator" if rk == "robotarm" else rk


def _newest_path(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _find_gif(robot: str, gif_dir: Path, idx: int, cue: str, row: dict | None = None) -> Path | None:
    """Resolve a render gif; searches subfolders and falls back to cue-based names when idx naming drifts."""
    safe = _safe(cue)
    render_root = results_subdir("render")

    if robot == "robotarm":
        if idx >= 0:
            idx_pats = [f"*_c{idx}_tiled.gif"]
            if idx < 1000:
                idx_pats.append(f"*_c{idx:02d}_tiled.gif")
            for pat in idx_pats:
                cands = sorted(gif_dir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
                if cands:
                    return cands[0]
                cands = sorted(gif_dir.rglob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
                if cands:
                    return cands[0]
        cue_pat = f"*_{safe}_*_tiled.gif"
        cands = sorted(gif_dir.rglob(cue_pat), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            return cands[0]
        # top_k=1 / preview pipeline: ``{ts}_IIWA_{cue}_p{pose}.gif`` (no _tiled suffix)
        for pat in (f"*IIWA_{safe}_*.gif", f"*_IIWA_{safe}_*.gif"):
            cands = sorted(gif_dir.rglob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
            if cands:
                return cands[0]
        for pat in (cue_pat, f"*IIWA_{safe}_*.gif", f"*_IIWA_{safe}_*.gif"):
            cands = list(render_root.rglob(pat))
            hit = _newest_path(cands)
            if hit:
                return hit
        return None

    if robot == "tiago":
        if idx >= 0:
            exact = gif_dir / f"mm19_g{idx:02d}_{safe}.gif"
            if exact.is_file():
                return exact
            pool = list(gif_dir.rglob(f"mm19_g{idx:02d}_{safe}.gif"))
            hit = _newest_path(pool)
            if hit:
                return hit
        pool = list(render_root.rglob(f"mm19_*_{safe}.gif"))
        return _newest_path(pool)

    if robot == "quadruped":
        r = row or {}
        has_gc = str(r.get("group", "")) in ("iconic", "contextual") and "group_idx" in r
        if has_gc:
            pfx = "g" if r.get("group") == "iconic" else "c"
            n = int(r["group_idx"])
            exact_fn = f"lc19_{pfx}{n:02d}_{safe}_mjlab_earth.gif"

            def pick_in_tree(root: Path) -> Path | None:
                if not root.is_dir():
                    return None
                pool = list(root.rglob(exact_fn))
                hit = _newest_path(pool)
                if hit:
                    return hit
                wild = f"lc19_{pfx}{n:02d}_*_mjlab_earth.gif"
                matches = list(root.rglob(wild))
                if matches:
                    preferred = [x for x in matches if safe in x.stem]
                    return _newest_path(preferred if preferred else matches)
                return None

            hit = pick_in_tree(gif_dir)
            if hit:
                return hit
            pool = list(render_root.rglob(exact_fn))
            hit = _newest_path(pool)
            if hit:
                return hit
        pool = list(render_root.rglob(f"lc19_*_{safe}_mjlab_earth.gif"))
        hit = _newest_path(pool)
        if hit:
            return hit
        if gif_dir.is_dir():
            return _newest_path(list(gif_dir.rglob(f"lc19_*_{safe}_mjlab_earth.gif")))
        return None

    return None


def _load_cues_3_sections_ordered() -> list[tuple[str, list[str]]] | None:
    """Return ``(section_title, [cue, ...])`` rows from ``cues_3.yml`` in file order.

    Each top-level key (e.g. ``A``, ``H``, ``M``, ``H & F``) is one section; cue tokens
    preserve order and duplicates (same cue in two sections → two rows on the cross page).
    """
    yml_path = seed_yml_dir() / "cues_3.yml"
    if not yml_path.is_file():
        return None
    try:
        import yaml
    except ImportError:
        return None
    try:
        with open(yml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    out: list[tuple[str, list[str]]] = []
    for sec_title, blob in data.items():
        title = str(sec_title).strip()
        if not title:
            continue
        cues: list[str] = []
        if isinstance(blob, str):
            cues = [t for t in blob.split() if t]
        elif isinstance(blob, list):
            cues = [str(x).strip() for x in blob if str(x).strip()]
        else:
            continue
        if cues:
            out.append((title, cues))
    return out if out else None


def _cross_section_element_id(sec_title: str) -> str:
    """Stable id for ``cues_3`` section keys (A, H, ``A & W``, …) used by filters + anchors."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(sec_title).strip()).strip("-").lower()
    return f"cross-sec-{slug or 'section'}"


def _load_cue_group_keys() -> tuple[set[str], set[str]] | None:
    yml_path = seed_yml_dir() / "cues_new.yml"
    if not yml_path.is_file():
        return None
    try:
        import yaml
    except ImportError:
        return None
    with open(yml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return None
    i = data.get("iconic")
    c = data.get("contextual")
    if not isinstance(i, dict) or not isinstance(c, dict):
        return None
    return set(i), set(c)


def _probe_badges_html(cue: str) -> str:
    """Small labels for pose_combo_probes cue naming convention."""
    parts: list[str] = []
    if cue.startswith("static_"):
        parts.append('<span class="badge b-static">static</span>')
    m = re.search(r"p2p_midmid_to_flow_bhigh_(.+)$", cue)
    if m:
        tag = m.group(1)
        parts.append(f'<span class="badge b-vel">{tag}</span>')
    m = re.match(r"p2p_repeat_(\d+)_", cue)
    if m:
        parts.append(f'<span class="badge b-rep">rep×{m.group(1)}</span>')
    if cue.startswith("bow_cycle_"):
        tag = "still" if "still" in cue else ("walk" if "walk" in cue else "")
        if tag:
            parts.append(f'<span class="badge b-bow">{tag}</span>')
    if cue.startswith("nod_like"):
        parts.append('<span class="badge b-nod">nod-like</span>')
    return (" " + " ".join(parts)) if parts else ""


def _card_html(
    row: dict,
    robot: str,
    gif_dir: Path,
    out_dir: Path,
    *,
    probe_mode: bool = False,
) -> tuple[str, bool]:
    idx = int(row.get("idx", -1))
    cue = str(row.get("cue", "?"))
    gif = _find_gif(robot, gif_dir, idx, cue, row)
    rel = os.path.relpath(gif, out_dir) if gif else ""
    has = gif is not None
    img = f'<img src="{rel}" loading="lazy" alt="">' if gif else "<div class='na'>No gif</div>"
    sub = f"c{idx}"
    if robot == "quadruped" and str(row.get("group", "")) in ("iconic", "contextual") and "group_idx" in row:
        pfx = "g" if row.get("group") == "iconic" else "c"
        sub = f"{pfx}{int(row['group_idx']):02d} · {sub}"
    badges = _probe_badges_html(cue) if probe_mode else ""
    tag = (
        f"<article><h3><span class='sub'>{sub}</span> <span class='cue'>{cue}</span>{badges}</h3>{img}"
        f"<pre>{json.dumps(row, indent=2, ensure_ascii=False)}</pre></article>"
    )
    return tag, has


def _split_pose_combo_probe_rows(rows: list[dict]) -> list[tuple[str, str | None, list[dict]]]:
    """Sections for pose_combo_probes.json: (title, optional description html, rows)."""
    static: list[dict] = []
    vel_sweep: list[dict] = []
    rep: list[dict] = []
    bow: list[dict] = []
    nod: list[dict] = []
    other: list[dict] = []
    for r in rows:
        cue = str(r.get("cue", ""))
        if cue.startswith("static_"):
            static.append(r)
        elif cue.startswith("p2p_midmid_to_flow_bhigh_"):
            vel_sweep.append(r)
        elif cue.startswith("p2p_repeat_"):
            rep.append(r)
        elif cue.startswith("bow_cycle_"):
            bow.append(r)
        elif cue.startswith("nod_like"):
            nod.append(r)
        else:
            other.append(r)
    for lst in (static, vel_sweep, rep, bow, nod, other):
        lst.sort(key=lambda x: int(x.get("idx", -1)))

    sections: list[tuple[str, str | None, list[dict]]] = []
    if static:
        sections.append(
            (
                "1 · Static 9-grid (front × back height)",
                "<p>순수 <code>pose</code>만. velocity/path 없음 → height 매핑만 확인.</p>",
                static,
            )
        )
    if vel_sweep:
        sections.append(
            (
                "2 · pose_to_pose + velocity sweep",
                "<p>동일 전환(mid/mid→low/high)인데 <code>velocity</code> 태그만 변경. "
                "gait 블렌딩 때문에 잔동작·전진·회전 차이가 나는지 비교.</p>",
                vel_sweep,
            )
        )
    if rep:
        sections.append(
            (
                "3 · pose_to_pose + repetition",
                "<p><code>velocity=0</code> 고정, <code>repetition</code>만 1/2/3. 왕복 횟수 차이.</p>",
                rep,
            )
        )
    if bow:
        sections.append(
            (
                "4 · Bow-like cycle",
                "<p>mid→bow 성격 자세를 <code>repetition=2</code>로 왕복. "
                "<code>still</code> vs <code>walk</code>(약한 vx) 비교.</p>",
                bow,
            )
        )
    if nod:
        sections.append(
            (
                "5 · Nod-like (small oscillation)",
                "<p>작은 각도 변화 + repetition으로 nod 느낌 프로브.</p>",
                nod,
            )
        )
    if other:
        sections.append(("Other", None, other))
    return sections


def _split_rows(
    rows: list[dict],
    keys: tuple[set[str], set[str]] | None,
) -> list[tuple[str, list[dict]]]:
    if keys is None:
        sorted_rows = sorted(rows, key=lambda r: int(r.get("idx", -1)))
        return [("Motion configs", sorted_rows)]
    iconic_keys, contextual_keys = keys
    iconic: list[dict] = []
    contextual: list[dict] = []
    other: list[dict] = []
    for row in rows:
        cue = str(row.get("cue", ""))
        if cue in iconic_keys:
            iconic.append(row)
        elif cue in contextual_keys:
            contextual.append(row)
        else:
            other.append(row)
    for lst in (iconic, contextual, other):
        lst.sort(key=lambda r: int(r.get("idx", -1)))
    sections: list[tuple[str, list[dict]]] = []
    if iconic:
        sections.append(("Iconic", iconic))
    if contextual:
        sections.append(("Contextual", contextual))
    if other:
        sections.append(("Other (not in cues_new.yml)", other))
    if not sections:
        sections.append(("Motion configs", sorted(rows, key=lambda r: int(r.get("idx", -1)))))
    return sections


def _probe_intro_html() -> str:
    return (
        '<div class="probe-intro">'
        "<h2>Pose combo probes — 읽는 법</h2>"
        "<ul>"
        "<li><strong>static_f*_b*</strong>: front/back height 9조합 정적 포즈.</li>"
        "<li><strong>p2p_midmid_to_flow_bhigh_*</strong>: 같은 pose_to_pose 전환 + velocity 태그만 변경 "
        "(<code>v0</code>, <code>vx12</code>, <code>vx28</code>, <code>wz09</code>; 차이가 잘 보이게 값↑).</li>"
        "<li><strong>p2p_repeat_*</strong>: velocity 0, repetition만 변경.</li>"
        "<li><strong>bow_cycle_*</strong>: repetition 2 왕복 + still/walk.</li>"
        "<li><strong>nod_like_*</strong>: 작은 진동 + repetition.</li>"
        "</ul>"
        "</div>"
    )


def _rk_for_emb_slug(slug: str) -> str:
    """Map motion_config / render folder slug to gif lookup kind used in _find_gif."""
    s = slug.strip().lower().replace("-", "_")
    if s == "quadruped":
        return "quadruped"
    if s == "manipulator":
        return "robotarm"
    if s in ("tiago", "bimanual"):
        return "tiago"
    return "robotarm"


def _rows_by_cue(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        cue = str(r.get("cue", ""))
        if cue:
            out[cue] = r
    return out


def _config_path_row_in(config_paths: list[Path], idx: int) -> Path | None:
    """Pick a motion JSON file that contains ``idx`` (merged bundles may span several files)."""
    for p in config_paths:
        if not p.is_file():
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, list):
            continue
        for c in data:
            if isinstance(c, dict) and int(c.get("idx", -1)) == int(idx):
                return p
    return config_paths[0] if config_paths and config_paths[0].is_file() else None


def _cross_traj_png_path(out_dir: Path, emb_slug: str, idx: int, cue: str) -> Path:
    d = out_dir / "_cross_traj_png"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{emb_slug}_i{idx:04d}_{_safe(cue)}.png"


def _ensure_cross_alpha_trajectory_png(
    *,
    emb_slug: str,
    row: dict | None,
    cue: str,
    gif_path: Path | None,
    config_paths: list[Path],
    out_dir: Path,
    force: bool = False,
) -> Path | None:
    """Sim-run panel (alpha motion stack + EE path); cached next to cross HTML."""
    if row is None:
        return None
    idx = int(row.get("idx", -1))
    cfg_p = _config_path_row_in(list(config_paths), idx)
    if cfg_p is None:
        return None
    tgt = _cross_traj_png_path(out_dir, emb_slug, idx, cue)
    if (
        not force
        and tgt.is_file()
        and gif_path
        and gif_path.is_file()
        and tgt.stat().st_mtime >= gif_path.stat().st_mtime
        and tgt.stat().st_mtime >= cfg_p.stat().st_mtime
    ):
        return tgt
    robot_hz = {
        "manipulator": ("IIWA", 8),
        "tiago": ("Tiago", 20),
        "quadruped": ("Go2", 12),
    }
    rb = robot_hz.get(emb_slug)
    if rb is None:
        return None
    robot, hz = rb
    sample = {
        "sample_id": f"cross_{emb_slug}_{_safe(cue)}_{idx}",
        "cue": str(row.get("cue", cue)),
        "cue_idx": idx,
        "config_path": str(cfg_p),
        "testset": "motion",
    }
    try:
        from adhoc.vlm_test import testset_utils as _tu

        panel, _meta = _tu.build_tile_figure_sim_trajectory_panel(
            sample,
            robot,
            hz,
            canonical="alpha_frame_trajectory",
            force=force,
        )
        panel.save(tgt)
        return tgt
    except Exception:
        return None


def _cross_cell_html(
    emb_label: str,
    row: dict | None,
    cue: str,
    emb_slug: str,
    gif_dir: Path,
    out_dir: Path,
    *,
    config_paths: list[Path],
) -> tuple[str, bool]:
    rk = _rk_for_emb_slug(emb_slug)
    cue_key = str(row.get("cue", cue)) if row else str(cue)
    idx = int(row.get("idx", -1)) if row else -1
    gif = _find_gif(rk, gif_dir, idx, cue_key, row)
    rel = os.path.relpath(gif, out_dir) if gif else ""
    has_gif = gif is not None
    traj_png = (
        _ensure_cross_alpha_trajectory_png(
            emb_slug=emb_slug,
            row=row,
            cue=cue_key,
            gif_path=gif,
            config_paths=config_paths,
            out_dir=out_dir,
        )
        if row is not None
        else None
    )
    rel_traj = os.path.relpath(traj_png, out_dir) if traj_png and traj_png.is_file() else ""
    traj_slot = (
        f'<img src="{rel_traj}" loading="lazy" alt="">'
        if rel_traj
        else "<div class='na traj-miss'>sim panel n/a</div>"
    )
    img_block = (
        f'<div class="emb-vis"><figure><figcaption>gif</figcaption>'
        f'<img src="{rel}" loading="lazy" alt=""></figure>'
        f'<figure><figcaption class="traj-cap">alpha + traj (sim)</figcaption>'
        f"{traj_slot}</figure></div>"
        if gif
        else "<div class='na'>No gif</div>"
    )
    if row is None:
        sub = "gif only · not in latest motion_configs" if has_gif else "no config row · no gif"
    else:
        sub = f"c{idx}"
        if rk == "quadruped" and str(row.get("group", "")) in ("iconic", "contextual") and "group_idx" in row:
            pfx = "g" if row.get("group") == "iconic" else "c"
            sub = f"{pfx}{int(row['group_idx']):02d} · {sub}"
    miss = row is None and not has_gif
    cell_cls = "emb-cell missing" if miss else "emb-cell"
    if row is not None:
        slim = {k: v for k, v in row.items() if k not in ("reasoning", "validation_warnings")}
        cfg_txt = json.dumps(slim, indent=2, ensure_ascii=False)
        cfg_block = f'<pre class="cfg">{cfg_txt}</pre>'
    else:
        cfg_block = '<pre class="cfg empty">— no merged row —</pre>'
    return (
        f'<div class="{cell_cls}"><span class="emb-label">{emb_label}</span> '
        f"<span class='sub'>{sub}</span>{img_block}{cfg_block}</div>",
        has_gif,
    )


def _run_cross_embodiment(*, open_html: bool, sources_path: Path | None = None) -> str:
    """Cross page from see_html_sources.yml (paths + column order); HTML name is {stem}_YYYYMMDD."""
    sources = _load_see_html_sources(sources_path)
    root = dev_robosuite_root()
    ce = load_cross_config(sources)
    order = ce["column_order"]
    out_dir = _repo_rel_to_path(str(ce.get("html_dir", "data/results/html/cross_embodiment")), root)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(ce.get("html_stem", "see_cross_embodiment"))
    day = datetime.now().strftime("%Y%m%d")
    html_path = out_dir / f"{stem}_{day}.html"

    cross_cols: list[tuple[str, str]] = []
    per_emb: dict[str, tuple[list[Path], dict[str, dict], Path]] = {}
    meta_lines: list[str] = []
    for key in order:
        block = load_embodiment_block(sources, key)
        label = str(block.get("label", key))
        cross_cols.append((key, label))
        rels = _motion_json_rel_list(block["motion_config_json"])
        abs_paths = [_repo_rel_to_path(r, root) for r in rels]
        gdir = _repo_rel_to_path(str(block["render_dir"]), root)
        rows = _load_merged_json_rows(abs_paths)
        by_cue = _rows_by_cue(rows)
        per_emb[key] = (abs_paths, by_cue, gdir)
        rel_meta = [str(p.relative_to(root)) if p.is_file() else f"{p.relative_to(root)} (missing)" for p in abs_paths]
        meta_lines.append(
            f"<li><strong>{label}</strong> <code>({key})</code>: configs <code>{' + '.join(rel_meta)}</code> · "
            f"{len(rows)} merged rows · render <code>{gdir.relative_to(root)}</code></li>"
        )

    sections = _load_cues_3_sections_ordered()
    if sections is None:
        raise SystemExit("cues_3.yml not found or invalid; cannot build cross-embodiment page.")

    n_cue_rows = sum(len(cues) for _, cues in sections)
    announce_output(out_dir, n_cue_rows, kind="html cross-row")

    filter_labels: list[str] = []
    row_blocks: list[str] = []
    total_cells = 0
    found_cells = 0
    for sec_title, cues in sections:
        el_id = _cross_section_element_id(sec_title)
        filter_labels.append(
            f'<label class="cross-filter-item"><input type="checkbox" checked data-target="{el_id}"> '
            f"{html.escape(sec_title)}</label>"
        )
        blocks: list[str] = []
        for cue in cues:
            cells: list[str] = []
            for path_slug, col_label in cross_cols:
                paths_list, by_cue, gdir = per_emb[path_slug]
                row = by_cue.get(cue)
                h, ok = _cross_cell_html(
                    col_label, row, cue, path_slug, gdir, out_dir, config_paths=paths_list
                )
                cells.append(h)
                total_cells += 1
                if ok:
                    found_cells += 1
            badges = _probe_badges_html(cue)
            blocks.append(
                f'<div class="cue-row"><h3 class="cue-title"><span class="cue">{cue}</span>{badges}</h3>'
                f'<div class="emb-triplet">{"".join(cells)}</div></div>'
            )
        row_blocks.append(
            f'<section id="{el_id}" class="cross-section-wrap" data-category="{html.escape(sec_title)}">'
            f"<details open class=\"cross-details\"><summary>{html.escape(sec_title)} "
            f'<span class="n">({len(cues)} cues)</span></summary>'
            f'<div class="cross-section-body">{"".join(blocks)}</div></details></section>'
        )

    filters_bar = (
        '<div class="cross-filters" role="group" aria-label="Section visibility">'
        '<span class="cross-filters-title">Show sections</span>'
        '<span class="cross-filters-actions">'
        '<button type="button" class="cross-filters-btn" data-cross-filters-all>All</button>'
        '<button type="button" class="cross-filters-btn" data-cross-filters-none>None</button>'
        "</span>"
        f'{"".join(filter_labels)}</div>'
    )
    script = """<script>
(function(){
  function sync(cb){
    var id = cb.getAttribute("data-target");
    var el = id && document.getElementById(id);
    if (el) { el.style.display = cb.checked ? "" : "none"; }
  }
  function init(){
    document.querySelectorAll(".cross-filters input[type=checkbox]").forEach(function(cb){
      cb.addEventListener("change", function(){ sync(this); });
    });
    var bar = document.querySelector(".cross-filters");
    if (!bar) return;
    bar.querySelector("[data-cross-filters-all]")?.addEventListener("click", function(){
      document.querySelectorAll(".cross-filters input[type=checkbox]").forEach(function(cb){
        cb.checked = true; sync(cb);
      });
    });
    bar.querySelector("[data-cross-filters-none]")?.addEventListener("click", function(){
      document.querySelectorAll(".cross-filters input[type=checkbox]").forEach(function(cb){
        cb.checked = false; sync(cb);
      });
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
</script>"""
    src_rel = _see_html_sources_path(sources_path).relative_to(root)
    style = (
        "body{font-family:system-ui,sans-serif;padding:16px;margin:0;background:#fafafa;color:#111}"
        ".meta{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px 14px;margin:0 0 14px;font-size:13px}"
        ".meta ul{margin:6px 0 0;padding-left:1.2rem}"
        ".cross-filters{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;padding:12px 14px;"
        "background:#fff;border:1px solid #ddd;border-radius:8px;margin:0 0 18px;position:sticky;top:0;z-index:3;"
        "box-shadow:0 1px 3px rgba(0,0,0,.06)}"
        ".cross-filters-title{font-weight:600;margin-right:8px;font-size:13px;color:#333}"
        ".cross-filters-actions{display:inline-flex;gap:6px;margin-right:12px}"
        ".cross-filters-btn{font-size:12px;padding:4px 10px;border:1px solid #ccc;border-radius:6px;"
        "background:#f8f8f9;cursor:pointer}"
        ".cross-filters-btn:hover{background:#eee}"
        ".cross-filter-item{font-size:13px;cursor:pointer;user-select:none;white-space:nowrap}"
        ".cross-section-wrap{margin-bottom:14px}"
        ".cross-details{border:1px solid #ddd;border-radius:8px;background:#fff;overflow:hidden}"
        ".cross-details>summary{cursor:pointer;padding:10px 14px;background:#f0f0f3;font-weight:600;font-size:1.05rem;"
        "list-style:none;border-bottom:1px solid #e4e4e7}"
        ".cross-details>summary::-webkit-details-marker{display:none}"
        ".cross-section-body{padding:10px 8px 8px;background:#fafafa}"
        "section .n{color:#666;font-weight:normal}"
        ".cue-row{border:1px solid #ddd;border-radius:8px;padding:10px;margin-bottom:12px;background:#fff;"
        "box-shadow:0 1px 2px rgba(0,0,0,.04)}"
        ".cue-row h3.cue-title{font-size:14px;margin:0 0 8px;font-weight:600}"
        ".cue-row .cue{font-weight:600;color:#222}"
        ".emb-triplet{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;align-items:start}"
        ".emb-cell{border:1px solid #e5e5e5;border-radius:6px;padding:6px;min-width:0;background:#fafafa}"
        ".emb-cell.missing{background:#fff8f8}"
        ".emb-label{display:block;font-size:11px;font-weight:700;text-transform:uppercase;color:#555;margin-bottom:2px}"
        ".emb-cell .sub{font-size:10px;color:#888}"
        ".emb-vis{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;align-items:start}"
        ".emb-vis figure{margin:0;min-width:0}"
        ".emb-vis figcaption{font-size:10px;font-weight:600;color:#647;margin:0 0 4px}"
        ".na.traj-miss{padding:10px;font-size:11px}"
        "img{width:100%;height:auto;display:block;border-radius:4px;vertical-align:middle}"
        ".emb-cell pre.cfg{margin:8px 0 0;font-size:10px;line-height:1.3;white-space:pre-wrap;"
        "background:#f4f4f5;padding:8px;border-radius:4px;max-height:240px;overflow:auto;border:1px solid #e4e4e7;color:#222}"
        ".emb-cell pre.cfg.empty{color:#999;font-style:italic}"
        ".na{color:#999;font-style:italic;padding:16px;text-align:center;font-size:12px}"
        ".badge{font-size:10px;margin-left:6px;padding:2px 6px;border-radius:4px;font-weight:600;vertical-align:middle}"
        ".b-static{background:#e8f4ff;color:#064}"
        ".b-vel{background:#fff3cd;color:#664}"
        ".b-rep{background:#e7f6e7;color:#162}"
        ".b-bow{background:#f3e8ff;color:#326}"
        ".b-nod{background:#fdeef4;color:#612}"
    )
    meta = (
        '<div class="meta"><p><strong>Cross-embodiment</strong> · sources <code>'
        f"{src_rel}</code> · gifs {found_cells}/{total_cells} · "
        f'<code>{html_path.relative_to(root)}</code></p><ul>{"".join(meta_lines)}</ul></div>'
    )
    page_html = (
        "<html><head><meta charset='utf-8'><title>cross embodiment</title>"
        f"<style>{style}</style></head><body>"
        "<h1>Cross-embodiment · cues_3.yml (sections)</h1>"
        f"{meta}{filters_bar}{''.join(row_blocks)}{script}</body></html>"
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(page_html)
    print(f"HTML saved: {html_path}")
    if open_html:
        _open_file(html_path)
    return str(html_path)


def _open_file(path: Path) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        print(f"Could not open {path}: {e}")


def run(
    robot: str = "robotarm",
    config_json: str | None = None,
    gif_dir: str | None = None,
    open_html: bool = True,
    cross_embodiment: bool = False,
    sources_yml: str | None = None,
) -> str:
    src_opt = Path(sources_yml) if sources_yml else None
    if cross_embodiment:
        return _run_cross_embodiment(open_html=open_html, sources_path=src_opt)

    aliases = {"manipulator": "robotarm", "robotarm": "robotarm", "tiago": "tiago", "quadruped": "quadruped"}
    rk = aliases.get(robot.strip().lower().replace("-", "_"), robot)
    yml_key = "manipulator" if rk == "robotarm" else rk
    root = dev_robosuite_root()
    cfg_title: str
    cfg_paths: list[Path] = []
    if config_json:
        cfg_one = Path(config_json)
        cfg_paths = [cfg_one]
        with open(cfg_one, encoding="utf-8") as f:
            rows = json.load(f)
        cfg_title = cfg_one.name
        gdir = Path(gif_dir) if gif_dir else results_render_dir("manipulator" if rk == "robotarm" else rk)
        html_subdir = _html_slug_for_robot_rk(rk)
        html_stem = f"see_{cfg_one.stem}"
        probe_stem = cfg_one.stem
    else:
        sources = _load_see_html_sources(src_opt)
        block = load_embodiment_block(sources, yml_key)
        rels = _motion_json_rel_list(block["motion_config_json"])
        cfg_paths = [_repo_rel_to_path(r, root) for r in rels]
        rows = _load_merged_json_rows(cfg_paths)
        cfg_title = " + ".join(p.name for p in cfg_paths if p.is_file()) or "merged configs"
        gdir = Path(gif_dir) if gif_dir else _repo_rel_to_path(str(block["render_dir"]), root)
        html_subdir = str(block["see_html_subdir"])
        html_stem = str(block["see_html_stem"])
        probe_stem = cfg_paths[0].stem if cfg_paths else ""

    out_dir = results_subdir("html") / html_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = dated_results_html_path(html_subdir, html_stem)
    probe_mode = probe_stem == "pose_combo_probes" or any(str(r.get("group")) == "probe" for r in rows)
    if probe_mode:
        probe_sections = _split_pose_combo_probe_rows(rows)
        announce_output(out_dir, len(rows), kind="html-card")
        found = 0
        section_blocks: list[str] = []
        for title, desc, sec_rows in probe_sections:
            cards: list[str] = []
            for row in sec_rows:
                html, has = _card_html(row, rk, gdir, out_dir, probe_mode=True)
                cards.append(html)
                if has:
                    found += 1
            desc_html = f'<div class="sec-desc">{desc}</div>' if desc else ""
            section_blocks.append(
                f'<section class="cue-section probe"><h2>{title} <span class="n">({len(sec_rows)})</span></h2>'
                f"{desc_html}"
                f'<div class="cards">{"".join(cards)}</div></section>'
            )
    else:
        group_keys = _load_cue_group_keys()
        sections = _split_rows(rows, group_keys)
        announce_output(out_dir, len(rows), kind="html-card")

        found = 0
        section_blocks = []
        for title, sec_rows in sections:
            cards: list[str] = []
            for row in sec_rows:
                html, has = _card_html(row, rk, gdir, out_dir, probe_mode=False)
                cards.append(html)
                if has:
                    found += 1
            section_blocks.append(
                f'<section class="cue-section"><h2>{title} <span class="n">({len(sec_rows)})</span></h2>'
                f'<div class="cards">{"".join(cards)}</div></section>'
            )

    style = (
        "body{font-family:system-ui,sans-serif;padding:16px;margin:0;background:#fafafa;color:#111}"
        ".cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;align-items:start}"
        "section.cue-section{margin-bottom:28px}"
        "section.cue-section>h2{font-size:1.1rem;margin:0 0 10px;padding-bottom:6px;border-bottom:1px solid #ccc}"
        "section .n{color:#666;font-weight:normal}"
        ".probe-intro{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 14px;margin:0 0 20px}"
        ".probe-intro h2{font-size:1rem;margin:0 0 8px}"
        ".probe-intro ul{margin:0;padding-left:1.2rem;font-size:13px;line-height:1.45}"
        ".sec-desc{font-size:12px;color:#444;margin:0 0 10px;line-height:1.5}"
        ".sec-desc code{font-size:11px;background:#eee;padding:1px 4px;border-radius:3px}"
        ".badge{font-size:10px;margin-left:6px;padding:2px 6px;border-radius:4px;font-weight:600;vertical-align:middle}"
        ".b-static{background:#e8f4ff;color:#064}"
        ".b-vel{background:#fff3cd;color:#664}"
        ".b-rep{background:#e7f6e7;color:#162}"
        ".b-bow{background:#f3e8ff;color:#326}"
        ".b-nod{background:#fdeef4;color:#612}"
        "article{border:1px solid #ddd;border-radius:8px;padding:8px;min-width:0;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04)}"
        "article h3{font-size:12px;margin:0 0 6px;font-weight:600}"
        "article .cue{font-weight:normal;color:#333}"
        "img{width:100%;height:auto;display:block;border-radius:4px;vertical-align:middle}"
        "pre{font-size:10px;line-height:1.25;white-space:pre-wrap;background:#f5f5f5;padding:6px;margin:6px 0 0;border-radius:4px;"
        "max-height:200px;overflow:auto}.na{color:#999;font-style:italic;padding:24px;text-align:center}"
    )
    intro = _probe_intro_html() if probe_mode else ""
    html = (
        "<html><head><meta charset='utf-8'><title>see html</title>"
        f"<style>{style}</style>"
        f"</head><body><h1>{yml_key} · {cfg_title} · gifs {found}/{len(rows)}</h1>"
        f"{intro}"
        f"{''.join(section_blocks)}</body></html>"
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML saved: {html_path}")
    if open_html:
        _open_file(html_path)
    return str(html_path)


if __name__ == "__main__":
    fire.Fire(run)
