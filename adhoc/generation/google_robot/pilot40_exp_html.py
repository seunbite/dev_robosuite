"""Per-experiment review HTML for pilot-40 Google Robot (with media)."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

from google_robot_experiment_suite import metrics_from_json
from pilot40_paths import (
    DEFAULT_GEN_TAG,
    DEFAULT_VERIFY_TAG,
    EXPERIMENT_TITLES,
    GT_CONSOLIDATED,
    MEDIA_DIR,
    N_CUES,
    SHOTS,
    TOPK_GRID_DIR,
    html_result_path,
    load_config_list,
    score_result_path,
    verify_result_path,
)

_REPO = Path(__file__).resolve().parents[3]


def _esc(x: object) -> str:
    return html.escape(str(x) if x is not None else "")


def _rel_href(out_html: Path, asset: Path) -> str:
    if not asset.is_file():
        return ""
    return Path(os.path.relpath(asset.resolve(), out_html.parent.resolve())).as_posix()


def _cue_idx(row: dict[str, Any], *, gt: dict[str, dict[str, Any]] | None = None) -> int:
    if row.get("idx") is not None:
        return int(row["idx"])
    cue = str(row.get("cue", ""))
    if gt and cue in gt:
        ev = gt[cue]
        if ev.get("cue_idx") is not None:
            return int(ev["cue_idx"])
    return 0


def _stem(row: dict[str, Any]) -> str:
    cue = str(row["cue"]).replace("/", "_").replace("\\", "_").replace(" ", "_")
    return f"mm19_g{_cue_idx(row):02d}_{cue}"


def _badge(val: bool | None) -> str:
    if val is True:
        return '<span class="ok">OK</span>'
    if val is False:
        return '<span class="miss">NO</span>'
    return '<span class="na">—</span>'


def _write_page(out: Path, *, title: str, meta: str, body: str) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    css = """
body{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e8eaed}
h1{padding:16px 24px;margin:0;font-size:1.25rem;border-bottom:1px solid #333}
.meta{padding:12px 24px;color:#9aa0a6;font-size:0.9rem;border-bottom:1px solid #222}
.card{margin:20px 24px;padding:14px;border:1px solid #333;border-radius:10px;background:#171a21}
.card h2{margin:0 0 6px;font-size:1rem}
.card p.desc{margin:0 0 10px;color:#9aa0a6;font-size:0.85rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}
.col{background:#0f1115;border-radius:8px;padding:8px}
video,img{width:100%;height:auto;border-radius:4px;background:#000;display:block}
.missing{color:#f28b82;font-size:0.8rem}
table{border-collapse:collapse;width:calc(100% - 48px);margin:16px 24px;font-size:0.85rem}
th,td{border:1px solid #333;padding:6px 8px}
th{background:#1a1d24}
.ok{color:#81c995;font-weight:600}.miss{color:#f28b82;font-weight:600}.na{color:#888}
"""
    out.write_text(
        f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<title>{_esc(title)}</title><style>{css}</style></head><body>
<h1>{_esc(title)}</h1>
<div class="meta">{meta}</div>
{body}
</body></html>""",
        encoding="utf-8",
    )
    print(f"[html] wrote {out}")
    return out


def _shots_lookup() -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    shots = load_config_list(SHOTS)
    by_idx = {_cue_idx(r): r for r in shots}
    by_cue = {str(r.get("cue", "")): r for r in shots if r.get("cue")}
    return by_idx, by_cue


def _cfg_for_row(
    by_idx: dict[int, dict[str, Any]],
    by_cue: dict[str, dict[str, Any]],
    *,
    idx: int,
    cue: str,
) -> dict[str, Any]:
    return by_idx.get(idx) or by_cue.get(cue) or {"idx": idx, "cue": cue}


def write_exp1_html(model_tag: str = DEFAULT_GEN_TAG) -> Path | None:
    score = score_result_path(1, model_tag)
    if not score.is_file():
        return None
    data = json.loads(score.read_text(encoding="utf-8"))
    rows_cfg, rows_cfg_by_cue = _shots_lookup()
    out = html_result_path(1, model_tag)
    cards = []
    for r in data.get("rows") or []:
        idx = int(r.get("cue_idx", 0))
        cue = str(r.get("cue", ""))
        cfg = _cfg_for_row(rows_cfg, rows_cfg_by_cue, idx=idx, cue=cue)
        stem = _stem(cfg)
        mp4 = MEDIA_DIR / "mp4" / f"{stem}.mp4"
        href = _rel_href(out, mp4)
        vid = f'<video controls loop muted playsinline src="{_esc(href)}"></video>' if href else '<p class="missing">missing mp4</p>'
        cards.append(
            f'<section class="card"><h2>g{idx} · {_esc(r.get("cue"))} {_badge(r.get("generation_correct"))}</h2>'
            f'<p class="desc">GT: {_esc(r.get("pose_gt"))}</p>{vid}</section>'
        )
    meta = f"score: {data.get('n_correct')}/{data.get('n')} · config: {_esc(SHOTS)}"
    return _write_page(out, title=f"Exp1 Google Robot — {EXPERIMENT_TITLES['1']}", meta=meta, body="".join(cards))


def write_exp7_html(model_tag: str = DEFAULT_GEN_TAG) -> Path | None:
    score = score_result_path(7, model_tag)
    if not score.is_file():
        return None
    data = json.loads(score.read_text(encoding="utf-8"))
    rows_cfg, rows_cfg_by_cue = _shots_lookup()
    out = html_result_path(7, model_tag)
    cards = []
    for r in data.get("rows") or []:
        idx = int(r.get("cue_idx", 0))
        cue = str(r.get("cue", ""))
        cfg = _cfg_for_row(rows_cfg, rows_cfg_by_cue, idx=idx, cue=cue)
        stem = _stem(cfg)
        mp4 = MEDIA_DIR / "mp4" / f"{stem}.mp4"
        href = _rel_href(out, mp4)
        vid = f'<video controls loop muted playsinline src="{_esc(href)}"></video>' if href else '<p class="missing">missing mp4</p>'
        cards.append(
            f'<section class="card"><h2>g{idx} · {_esc(r.get("cue"))} {_badge(r.get("component_match"))}</h2>'
            f'<p class="desc">component: {_esc(r.get("motion_component_gt"))}</p>{vid}</section>'
        )
    meta = f"score: {data.get('n_correct')}/{data.get('n')}"
    return _write_page(out, title=f"Exp7 Google Robot — {EXPERIMENT_TITLES['7']}", meta=meta, body="".join(cards))


def _verify_cards(
    data: dict[str, Any],
    *,
    out: Path,
    pose_key: str,
    media: str,
) -> str:
    rows_cfg, rows_cfg_by_cue = _shots_lookup()
    cards = []
    for r in data.get("results") or []:
        idx = int(r.get("idx", 0))
        cue = str(r.get("cue", ""))
        cfg = _cfg_for_row(rows_cfg, rows_cfg_by_cue, idx=idx, cue=cue)
        stem = _stem(cfg)
        res = r.get("result") or {}
        ok = res.get(pose_key)
        if media == "png":
            asset = MEDIA_DIR / "pose" / f"{stem}_pose.png"
            href = _rel_href(out, asset)
            media_html = f'<img src="{_esc(href)}" alt="pose"/>' if href else '<p class="missing">missing png</p>'
        else:
            asset = MEDIA_DIR / "mp4" / f"{stem}.mp4"
            href = _rel_href(out, asset)
            media_html = f'<video controls loop muted playsinline src="{_esc(href)}"></video>' if href else '<p class="missing">missing mp4</p>'
        note = _esc(res.get("visual_assessment") or res.get("text_assessment") or "")[:400]
        cards.append(
            f'<section class="card"><h2>g{idx} · {_esc(cue)} {_badge(ok if isinstance(ok, bool) else None)}</h2>'
            f'{media_html}<p class="desc">{note}</p></section>'
        )
    return "".join(cards)


def write_exp2_html(model_tag: str = DEFAULT_VERIFY_TAG) -> Path | None:
    path = verify_result_path(2, model_tag)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    out = html_result_path(2, model_tag)
    met = metrics_from_json(path, "pose_verify_vlm")
    body = _verify_cards(data, out=out, pose_key="pose_is_appropriate", media="png")
    return _write_page(out, title=f"Exp2 Google Robot — {EXPERIMENT_TITLES['2']}", meta=_esc(met.get("headline", "")), body=body)


def write_exp3_html(model_tag: str = DEFAULT_VERIFY_TAG) -> Path | None:
    path = verify_result_path(3, model_tag)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    out = html_result_path(3, model_tag)
    met = metrics_from_json(path, "pose_verify_text")
    body = _verify_cards(data, out=out, pose_key="pose_is_appropriate", media="png")
    return _write_page(out, title=f"Exp3 Google Robot — {EXPERIMENT_TITLES['3']}", meta=_esc(met.get("headline", "")), body=body)


def write_exp8_html(model_tag: str = DEFAULT_VERIFY_TAG) -> Path | None:
    path = verify_result_path(8, model_tag)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    out = html_result_path(8, model_tag)
    met = metrics_from_json(path, "motion_verify_vlm")
    body = _verify_cards(data, out=out, pose_key="movement_is_appropriate", media="mp4")
    return _write_page(out, title=f"Exp8 Google Robot — {EXPERIMENT_TITLES['8']}", meta=_esc(met.get("headline", "")), body=body)


def write_exp9_html(model_tag: str = DEFAULT_VERIFY_TAG) -> Path | None:
    path = verify_result_path(9, model_tag)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    out = html_result_path(9, model_tag)
    met = metrics_from_json(path, "motion_verify_text")
    body = _verify_cards(data, out=out, pose_key="movement_is_appropriate", media="mp4")
    return _write_page(out, title=f"Exp9 Google Robot — {EXPERIMENT_TITLES['9']}", meta=_esc(met.get("headline", "")), body=body)


def write_exp5_html(model_tag: str = DEFAULT_VERIFY_TAG) -> Path | None:
    path = verify_result_path(5, model_tag)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    picks = data.get("picks") or data.get("results") or []
    out = html_result_path(5, model_tag)
    cards = []
    for r in picks:
        idx = int(r.get("idx", r.get("cue_idx", 0)))
        cue = str(r.get("cue", ""))
        grid_rel = r.get("grid_image")
        if grid_rel:
            grid = _REPO / str(grid_rel)
        else:
            safe = cue.replace("/", "_").replace(" ", "_")
            grid = TOPK_GRID_DIR / f"mm19_g{idx:02d}_{safe}_top30_gemini.png"
        href = _rel_href(out, grid)
        img = f'<img src="{_esc(href)}" alt="topk"/>' if href else '<p class="missing">missing grid</p>'
        gem = r.get("gemini") or r.get("pick") or r.get("result") or {}
        pick_txt = gem.get("selected_rank") or gem.get("reasoning") or str(gem)[:200]
        cards.append(
            f'<section class="card"><h2>g{idx} · {_esc(cue)}</h2>{img}'
            f'<p class="desc">pick: {_esc(pick_txt)}</p></section>'
        )
    meta = f"partial: {len(picks)} cues (target {N_CUES})"
    return _write_page(out, title=f"Exp5 Google Robot — {EXPERIMENT_TITLES['5']}", meta=meta, body="".join(cards))


def write_combined_verify_html() -> Path | None:
    try:
        from adhoc.generation.google_robot.build_pilot40_verify_review_html import (  # noqa: WPS433
            build as build_combined,
        )
    except ImportError:
        try:
            from build_pilot40_verify_review_html import build as build_combined  # noqa: WPS433
        except ImportError as e:
            print(f"[html] skip combined verify page: {e}", flush=True)
            return None
    import argparse
    from pilot40_paths import LEGACY_VERIFY_DIR

    out = _REPO / "data/results/html/google_robot/pilot40_verify_review.html"
    build_combined(
        argparse.Namespace(
            config_json=str(SHOTS),
            media_dir=str(MEDIA_DIR),
            verify_dir=str(LEGACY_VERIFY_DIR),
            out_html=str(out),
        )
    )
    return out


def write_index_html() -> Path:
    out = _REPO / "data/results/html/google_robot/index.html"
    links = [
        ("Combined verify (5-col)", "pilot40_verify_review.html"),
        (f"Exp1 {DEFAULT_GEN_TAG}", f"exp1_{DEFAULT_GEN_TAG}.html"),
        (f"Exp2 {DEFAULT_VERIFY_TAG}", f"exp2_{DEFAULT_VERIFY_TAG}.html"),
        (f"Exp3 {DEFAULT_VERIFY_TAG}", f"exp3_{DEFAULT_VERIFY_TAG}.html"),
        (f"Exp5 topk (partial)", f"exp5_{DEFAULT_VERIFY_TAG}.html"),
        (f"Exp7 {DEFAULT_GEN_TAG}", f"exp7_{DEFAULT_GEN_TAG}.html"),
        (f"Exp8 {DEFAULT_VERIFY_TAG}", f"exp8_{DEFAULT_VERIFY_TAG}.html"),
        (f"Exp9 {DEFAULT_VERIFY_TAG}", f"exp9_{DEFAULT_VERIFY_TAG}.html"),
    ]
    items = "".join(
        f'<li><a href="{_esc(href)}">{_esc(label)}</a></li>' for label, href in links
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>Google Robot pilot-40</title>
<style>body{{font-family:system-ui;max-width:720px;margin:2rem auto;padding:0 1rem}}
a{{color:#4c8bf5}}</style></head><body>
<h1>Google Robot pilot-40 — experiment HTML</h1>
<p>GT: <code>{_esc(GT_CONSOLIDATED)}</code></p>
<ul>{items}</ul>
</body></html>""",
        encoding="utf-8",
    )
    print(f"[html] wrote {out}")
    return out


def write_exp_html(exp_id: str | int, model_tag: str, out_json: Path | None = None) -> Path | None:
    """Write per-experiment HTML when a writer exists."""
    eid = str(exp_id)
    writers = {
        "1": write_exp1_html,
        "2": write_exp2_html,
        "3": write_exp3_html,
        "5": write_exp5_html,
        "7": write_exp7_html,
        "8": write_exp8_html,
        "9": write_exp9_html,
    }
    fn = writers.get(eid)
    if fn is None:
        return None
    tag = DEFAULT_GEN_TAG if eid in {"1", "7"} else model_tag
    return fn(tag)


def write_all_html() -> list[Path]:
    paths: list[Path] = []
    for fn in (
        write_exp1_html,
        write_exp2_html,
        write_exp3_html,
        write_exp5_html,
        write_exp7_html,
        write_exp8_html,
        write_exp9_html,
    ):
        p = fn()
        if p:
            paths.append(p)
    combined = write_combined_verify_html()
    if combined:
        paths.append(combined)
    paths.append(write_index_html())
    return [p for p in paths if p]
