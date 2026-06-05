#!/usr/bin/env python3
"""Build wrong-answer notebook HTML for pilot40 seven evaluation settings."""
from __future__ import annotations

import ast
import csv
import html
import json
import os
import re
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]

OUT_HTML = _REPO / "data/results/html/manipulator/pilot40_wrong_answer_notebook.html"

POSE_SCORED = _REPO / "data/results/verify/pilot40_pose_eval_consolidated_scored.tsv"
POSE_CONSOLIDATED = _REPO / "data/results/verify/pilot40_pose_eval_consolidated.json"
POSE_VLM_VERIFY = [
    _REPO / "data/results/verify/pose_tile_verify_pilot10_gemini.json",
    _REPO / "data/results/verify/pose_tile_verify_pilot20_gemini.json",
    _REPO / "data/results/verify/pose_tile_verify_pilot20_more_gemini.json",
]
POSE_TEXT_VERIFY = [
    _REPO / "data/results/verify/pose_textonly_verify_pilot10_gemini.json",
    _REPO / "data/results/verify/pose_textonly_verify_pilot20_gemini.json",
    _REPO / "data/results/verify/pose_textonly_verify_pilot20_more_gemini.json",
]
PAIRWISE = _REPO / "data/results/verify/pilot40_pose_pairwise_12_gemini.json"

MOTION_CFG = (
    _REPO
    / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_gt_fixed_pose_pilot40.json"
)
MOTION_METRICS = _REPO / "data/results/verify/pilot40_motion_verify_metrics.json"
MOTION_VLM_VERIFY = _REPO / "data/results/verify/pilot40_motion_component_verify_gemini.json"
MOTION_TEXT_VERIFY = _REPO / "data/results/verify/pilot40_motion_component_verify_text_gemini.json"
MOTION_MANIFEST = _REPO / "data/results/render/manipulator/motion_gt_compare/manifest_generation_pilot40.json"
MOTION_PAIRWISE_DIR = _REPO / "data/results/verify/samples/motion_gt_neg_pairwise"
MOTION_PAIRWISE_JSONS = [
    MOTION_PAIRWISE_DIR / "pairwise_eval_results.json",
    MOTION_PAIRWISE_DIR / "pairwise_eval_results_extra7.json",
    MOTION_PAIRWISE_DIR / "pairwise_eval_results_extra10.json",
    MOTION_PAIRWISE_DIR / "pairwise_eval_results_remaining_mp4.json",
]

POSE_GIF_DIRS = [
    _REPO / "data/results/visualize/pose_generation_pilot40_hz10/IIWA",
    _REPO / "data/results/visualize/gt_fixed_pose_pilot20_hz10/IIWA",
]

CUE_ALIAS = {"scratch_head_confused": "self_scratch_head_confused"}

SETTINGS = [
    ("pose_generation", "Pose — generation vs human GT"),
    ("pose_verify_vlm", "Pose — VLM verify / regenerate"),
    ("pose_verify_text", "Pose — text verify / regenerate"),
    ("pose_compare_vlm", "Pose — VLM pairwise compare"),
    ("motion_generation", "Motion — generation vs component GT"),
    ("motion_verify_vlm", "Motion — VLM verify (alpha trajectory)"),
    ("motion_verify_text", "Motion — text verify"),
    ("motion_compare_mp4", "Motion — VLM pairwise compare (MP4)"),
]


def _esc(x: object) -> str:
    return html.escape(str(x) if x is not None else "")


def _rel(path: Path | str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        return os.path.relpath(p, OUT_HTML.parent)
    except ValueError:
        return str(p)


def _to_bool(v: object) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}


def _parse_pose_dict(s: str) -> dict[str, Any]:
    if not s:
        return {}
    try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, dict) else {}
    except (SyntaxError, ValueError):
        return {}


def _pose_str(d: dict[str, Any] | None) -> str:
    if not d:
        return "—"
    return f"dir={d.get('dir')}, grip={d.get('gripper_orientation')}"


def _latest_gif(cue: str) -> Path | None:
    cue = CUE_ALIAS.get(cue, cue)
    hits: list[Path] = []
    for d in POSE_GIF_DIRS:
        if d.is_dir():
            hits.extend(d.glob(f"*_IIWA_{cue}_*.gif"))
    if not hits:
        return None
    return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _merge_pose_verify(files: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for fp in files:
        if not fp.is_file():
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        for r in data.get("results", []):
            cue = str(r.get("cue", ""))
            out[cue] = r
            alias = CUE_ALIAS.get(cue)
            if alias:
                out[alias] = r
    return out


def _img(path: Path | str | None, *, cap: str = "") -> str:
    rel = _rel(path)
    if not rel:
        return f'<div class="missing">No image ({_esc(cap)})</div>'
    return (
        f'<figure><figcaption>{_esc(cap)}</figcaption>'
        f'<img src="{_esc(rel)}" loading="lazy" alt="{_esc(cap)}"/></figure>'
    )


def _video(path: Path | str | None, *, cap: str = "") -> str:
    rel = _rel(path)
    if not rel:
        return f'<div class="missing">No video ({_esc(cap)})</div>'
    return (
        f'<figure><figcaption>{_esc(cap)}</figcaption>'
        f'<video src="{_esc(rel)}" controls muted loop playsinline></video></figure>'
    )


def _parse_side_labels(prompt: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for side in ("LEFT", "RIGHT"):
        m = re.search(rf"- {side}: movement (.+)", prompt or "")
        if m:
            out[side.lower()] = m.group(1).strip()
    return out


def _find_pair_mp4(idx: int, cue: str, explicit: str | None = None) -> Path | None:
    if explicit:
        p = _REPO / explicit if not Path(explicit).is_absolute() else Path(explicit)
        if p.is_file():
            return p
    hits = sorted(MOTION_PAIRWISE_DIR.glob(f"{idx:03d}_{cue}_pair*.mp4"))
    return hits[0] if hits else None


def _load_motion_pairwise_mp4() -> list[dict[str, Any]]:
    by_idx: dict[int, dict[str, Any]] = {}
    for fp in MOTION_PAIRWISE_JSONS:
        if not fp.is_file():
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        for e in data.get("mp4", []):
            idx = int(e["idx"])
            pred = e.get("pred") or e.get("better_side") or (e.get("raw") or {}).get("better_side")
            correct = e.get("correct")
            if correct is None:
                correct = pred == e.get("gt_side")
            parsed = e.get("parsed") or e.get("raw") or {}
            assessment = parsed.get("motion_assessment")
            by_idx[idx] = {
                "idx": idx,
                "cue": e.get("cue", ""),
                "gt_side": e.get("gt_side", "?"),
                "pred": pred or "?",
                "correct": bool(correct),
                "left": e.get("left", "?"),
                "right": e.get("right", "?"),
                "assessment": assessment,
                "pair_mp4": e.get("pair_mp4"),
                "side_labels": _parse_side_labels(e.get("prompt", "")),
            }
    return [by_idx[k] for k in sorted(by_idx)]


def _pair_ab_summary(row: dict[str, Any]) -> str:
    left = row.get("left", "?")
    right = row.get("right", "?")
    gt_side = row.get("gt_side", "?")
    labels = row.get("side_labels") or {}
    left_hint = labels.get("left", "")
    right_hint = labels.get("right", "")
    left_txt = f"LEFT = {left}" + (f" ({left_hint})" if left_hint else "")
    right_txt = f"RIGHT = {right}" + (f" ({right_hint})" if right_hint else "")
    gt_on = f"component GT on {gt_side.upper()}"
    picked = f"VLM picked {str(row.get('pred', '?')).upper()}"
    return f"{left_txt} | {right_txt}; {gt_on}; {picked}"


def _gif_img(cue: str, cap: str = "render") -> str:
    return _img(_latest_gif(cue), cap=cap)


def _blockquote(lines: list[str]) -> str:
    parts = [f"<p>{_esc(ln)}</p>" for ln in lines if ln and str(ln).strip()]
    return "".join(parts) if parts else ""


def _extract_pose_reason(vr: dict[str, Any] | None) -> dict[str, Any]:
    if not vr:
        return {}
    res = vr.get("result") or {}
    out: dict[str, Any] = {
        "assessment": res.get("direction_orientation_assessment") or res.get("best_tile_reason"),
        "why": None,
        "plan": [],
    }
    if res.get("pose_is_appropriate"):
        out["plan"] = (res.get("if_appropriate") or {}).get("recommended_movement_plan") or []
    else:
        block = res.get("if_not_appropriate") or {}
        out["why"] = block.get("why_change")
        out["plan"] = block.get("recommended_movement_plan_after_change") or []
    return out


def _tail_summary(movements: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    seen_pose = False
    for step in movements:
        if step.get("type") == "pose":
            seen_pose = True
            continue
        if not seen_pose:
            continue
        st = step.get("type")
        params = step.get("parameters") or {}
        if st == "movement":
            axes = params.get("directions", [{}])[0].get("degrees", {})
            parts.append(
                f"movement joint={params.get('joint')} rep={params.get('repetition')} axes={axes}"
            )
        elif st == "path":
            parts.append(f"path {params.get('path_type')} axis/plane={params.get('axis') or params.get('plane')}")
        else:
            parts.append(str(st))
    return "; ".join(parts[:6]) or "—"


def _load_motion_by_idx() -> dict[int, dict[str, Any]]:
    cfg = {int(r["idx"]): r for r in json.loads(MOTION_CFG.read_text(encoding="utf-8"))}
    manifest = {
        int(r["cue_idx"]): r
        for r in json.loads(MOTION_MANIFEST.read_text(encoding="utf-8")).get("rows", [])
    }
    vlm = {int(r["cue_idx"]): r for r in json.loads(MOTION_VLM_VERIFY.read_text(encoding="utf-8")).get("rows", [])}
    text = {int(r["cue_idx"]): r for r in json.loads(MOTION_TEXT_VERIFY.read_text(encoding="utf-8")).get("rows", [])}
    return {"cfg": cfg, "manifest": manifest, "vlm": vlm, "text": text}


def _motion_reason(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    vr = row.get("verify_result") or {}
    block = vr.get("if_not_appropriate") or {}
    return {
        "assessment": vr.get("movement_assessment"),
        "why": block.get("why_not"),
        "recommended": block.get("recommended_component"),
        "guidance": block.get("recommended_tail_guidance") or [],
        "appropriate": vr.get("movement_is_appropriate"),
    }


def _card(
    *,
    title: str,
    gt: str,
    chosen: str,
    reason: dict[str, Any],
    media_html: str,
    extra: str = "",
) -> str:
    reason_html = []
    if reason.get("assessment"):
        reason_html.append(f"<p><b>Assessment</b>: {_esc(reason['assessment'])}</p>")
    if reason.get("why"):
        reason_html.append(f"<p><b>Why</b>: {_esc(reason['why'])}</p>")
    if reason.get("recommended"):
        reason_html.append(f"<p><b>Recommended</b>: <code>{_esc(reason['recommended'])}</code></p>")
    if reason.get("guidance"):
        reason_html.append("<p><b>Guidance</b>:</p><ul>")
        for g in reason["guidance"]:
            reason_html.append(f"<li>{_esc(g)}</li>")
        reason_html.append("</ul>")
    if reason.get("plan"):
        reason_html.append("<p><b>Movement plan</b>:</p><ul>")
        for g in reason["plan"]:
            reason_html.append(f"<li>{_esc(g)}</li>")
        reason_html.append("</ul>")
    return (
        '<article class="card">'
        f'<h3>{_esc(title)}</h3>'
        f'<div class="kv"><span class="k">Ground truth</span><span class="v">{_esc(gt)}</span></div>'
        f'<div class="kv"><span class="k">Chose / output</span><span class="v">{_esc(chosen)}</span></div>'
        f'{"".join(reason_html)}'
        f'{extra}'
        f'<div class="media">{media_html}</div>'
        "</article>"
    )


def build_pose_generation(rows: list[dict[str, str]]) -> list[str]:
    cards = []
    for row in rows:
        if _to_bool(row.get("generation_correct", "")):
            continue
        cue = row["cue"]
        resolved = CUE_ALIAS.get(cue, cue)
        gen = _parse_pose_dict(row.get("generation", ""))
        cards.append(
            _card(
                title=f"c{row['cue_idx']} {resolved}",
                gt=row.get("groundtruth", ""),
                chosen=_pose_str(gen),
                reason={},
                media_html=_gif_img(resolved, "generated motion (GT-fixed pose pipeline GIF if any)"),
            )
        )
    return cards


def build_pose_verify(
    rows: list[dict[str, str]],
    *,
    bool_col: str,
    fix_col: str,
    correct_col: str,
    verify_map: dict[str, dict[str, Any]],
    include_tile: bool,
) -> list[str]:

    cards = []
    for row in rows:
        if _to_bool(row.get(correct_col, "")):
            continue
        cue = row["cue"]
        resolved = CUE_ALIAS.get(cue, cue)
        gen = _parse_pose_dict(row.get("generation", ""))
        appropriate = _to_bool(row.get(bool_col, ""))
        fix = _parse_pose_dict(row.get(fix_col, "")) if row.get(fix_col) else {}
        if appropriate:
            chosen = f"OK (appropriate): {_pose_str(gen)}"
        elif fix:
            chosen = f"Fix → {_pose_str(fix)}"
        else:
            chosen = f"Rejected current: {_pose_str(gen)} (no fix recorded)"

        vr = verify_map.get(resolved) or verify_map.get(cue)
        reason = _extract_pose_reason(vr)
        media_parts = []
        if include_tile:
            tile = (vr or {}).get("tile_image")
            media_parts.append(_img(_REPO / tile if tile else None, cap="VLM tile group"))
        media_parts.append(_gif_img(resolved, "cue render"))
        media = "".join(media_parts)
        cards.append(
            _card(
                title=f"c{row['cue_idx']} {resolved}",
                gt=row.get("groundtruth", ""),
                chosen=chosen,
                reason=reason,
                media_html=media,
                extra=f'<p><b>pose_is_appropriate</b>: {_esc(row.get(bool_col, ""))}</p>',
            )
        )
    return cards


def build_pose_pairwise() -> list[str]:
    if not PAIRWISE.is_file():
        return []
    data = json.loads(PAIRWISE.read_text(encoding="utf-8"))
    cards = []
    for comp in data.get("comparisons", []):
        if comp.get("vlm_correct"):
            continue
        gt_side = comp.get("gt_side", "?")
        better = (comp.get("vlm_result") or {}).get("better_side", "?")
        left = comp.get("left_pose") or {}
        right = comp.get("right_pose") or {}
        gt_pose = comp.get("gt_pose") or {}
        cards.append(
            _card(
                title=f"c{comp.get('cue_idx')} {comp.get('cue')} (pair)",
                gt=f"GT pose {_pose_str(gt_pose)} on side {gt_side}",
                chosen=f"Picked side {better} — left={_pose_str(left)} | right={_pose_str(right)}",
                reason={
                    "assessment": (comp.get("vlm_result") or {}).get("direction_orientation_assessment"),
                },
                media_html=_img(comp.get("pair_image"), cap="pairwise A|B"),
            )
        )
    return cards


def build_motion_generation(motion: dict[str, Any], metric_rows: list[dict[str, Any]]) -> list[str]:
    cards = []
    cfg_by = motion["cfg"]
    man_by = motion["manifest"]
    for mr in metric_rows:
        if mr.get("generation_tail_match"):
            continue
        idx = int(mr["cue_idx"])
        cfg = cfg_by.get(idx, {})
        man = man_by.get(idx, {})
        gt = mr.get("annotation_raw", "")
        tail = _tail_summary(cfg.get("movements") or [])
        media = _img(man.get("alpha_frame_trajectory"), cap="alpha + EE trajectory") + _img(
            man.get("gif"), cap="render GIF"
        )
        cards.append(
            _card(
                title=f"c{idx} {mr.get('cue')}",
                gt=f"component GT: {gt}",
                chosen=f"generated tail: {tail}",
                reason={},
                media_html=media,
            )
        )
    return cards


def build_motion_compare_mp4(rows: list[dict[str, Any]]) -> list[str]:
    cards = []
    for row in rows:
        if row.get("correct"):
            continue
        idx = int(row["idx"])
        cue = row["cue"]
        mp4 = _find_pair_mp4(idx, cue, row.get("pair_mp4"))
        media = _video(mp4, cap="pairwise A | B (alpha trajectory MP4)")
        cards.append(
            _card(
                title=f"c{idx} {cue}",
                gt=f"component GT tail on side {row.get('gt_side', '?')}",
                chosen=_pair_ab_summary(row),
                reason={"assessment": row.get("assessment")},
                media_html=f'<div class="media wide">{media}</div>',
            )
        )
    return cards


def build_motion_verify(
    metric_rows: list[dict[str, Any]],
    motion: dict[str, Any],
    *,
    channel: str,
) -> list[str]:
    key = "vlm_alpha" if channel == "vlm" else "text"
    verify_key = "vlm" if channel == "vlm" else "text"
    cards = []
    for mr in metric_rows:
        ch = mr.get(key) or {}
        if ch.get("verifying_tail_match"):
            continue
        idx = int(mr["cue_idx"])
        cfg = motion["cfg"].get(idx, {})
        man = motion["manifest"].get(idx, {})
        vrow = motion[verify_key].get(idx)
        reason = _motion_reason(vrow)
        gt = mr.get("annotation_raw", "")
        gen_tail = _tail_summary(cfg.get("movements") or [])
        appropriate = ch.get("movement_is_appropriate")
        cards.append(
            _card(
                title=f"c{idx} {mr.get('cue')}",
                gt=f"component GT: {gt}",
                chosen=(
                    f"movement_is_appropriate={appropriate}; "
                    f"after-verify tail match={ch.get('verifying_tail_match')}; "
                    f"gen tail: {gen_tail}"
                ),
                reason=reason,
                media_html=_img(man.get("alpha_frame_trajectory"), cap="alpha trajectory"),
                extra=(
                    f"<p><b>has_recommendation</b>: {_esc(ch.get('has_recommendation'))}</p>"
                    if channel == "vlm"
                    else ""
                ),
            )
        )
    return cards


def build() -> Path:
    pose_rows: list[dict[str, str]] = []
    with POSE_SCORED.open(encoding="utf-8") as f:
        pose_rows.extend(csv.DictReader(f, delimiter="\t"))

    motion_payload = json.loads(MOTION_METRICS.read_text(encoding="utf-8"))
    metric_rows = motion_payload.get("rows", [])
    motion = _load_motion_by_idx()
    motion_pairwise = _load_motion_pairwise_mp4()
    motion_pairwise_ok = sum(1 for r in motion_pairwise if r.get("correct"))

    vlm_verify = _merge_pose_verify(POSE_VLM_VERIFY)
    text_verify = _merge_pose_verify(POSE_TEXT_VERIFY)

    motion_compare_label = (
        f"{SETTINGS[7][1]} — {motion_pairwise_ok}/{len(motion_pairwise)} "
        f"({100 * motion_pairwise_ok / len(motion_pairwise):.1f}%)"
        if motion_pairwise
        else SETTINGS[7][1]
    )

    sections: list[tuple[str, str, list[str]]] = [
        ("pose_generation", SETTINGS[0][1], build_pose_generation(pose_rows)),
        (
            "pose_verify_vlm",
            SETTINGS[1][1],
            build_pose_verify(
                pose_rows,
                bool_col="verify_VLM_bool",
                fix_col="verify_VLM_fix",
                correct_col="VLM_correct",
                verify_map=vlm_verify,
                include_tile=True,
            ),
        ),
        (
            "pose_verify_text",
            SETTINGS[2][1],
            build_pose_verify(
                pose_rows,
                bool_col="verify_Text_bool",
                fix_col="verify_Text_fix",
                correct_col="Text_correct",
                verify_map=text_verify,
                include_tile=False,
            ),
        ),
        ("pose_compare_vlm", SETTINGS[3][1], build_pose_pairwise()),
        ("motion_generation", SETTINGS[4][1], build_motion_generation(motion, metric_rows)),
        ("motion_verify_vlm", SETTINGS[5][1], build_motion_verify(metric_rows, motion, channel="vlm")),
        ("motion_verify_text", SETTINGS[6][1], build_motion_verify(metric_rows, motion, channel="text")),
        ("motion_compare_mp4", motion_compare_label, build_motion_compare_mp4(motion_pairwise)),
    ]

    nav = "".join(
        f'<a href="#{sid}">{_esc(label)} ({len(cards)})</a>' for sid, label, cards in sections
    )

    body_parts = [
        "<header>",
        "<h1>Pilot40 wrong-answer notebook</h1>",
        "<p>Eight settings vs human pose GT / motion component GT. Only incorrect cases.</p>",
        f"<nav class=\"toc\">{nav}</nav>",
        "</header>",
    ]

    for sid, label, cards in sections:
        body_parts.append(f'<section id="{sid}" class="setting">')
        body_parts.append(f"<h2>{_esc(label)} <span class=\"count\">{len(cards)} wrong</span></h2>")
        if cards:
            body_parts.append('<div class="grid">')
            body_parts.extend(cards)
            body_parts.append("</div>")
        else:
            body_parts.append('<p class="empty">No wrong cases recorded.</p>')
        body_parts.append("</section>")

    css = """
body{font-family:system-ui,-apple-system,sans-serif;margin:0;background:#f4f6fa;color:#1a1a2e}
header{background:#fff;border-bottom:1px solid #dde3ef;padding:20px 24px;position:sticky;top:0;z-index:2}
h1{margin:0 0 8px;font-size:1.45rem}
.toc{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.toc a{font-size:.85rem;padding:6px 10px;background:#eef2ff;border-radius:8px;text-decoration:none;color:#243b7a}
.setting{padding:20px 24px 32px;border-bottom:1px solid #e2e8f0}
.setting h2{font-size:1.15rem;margin:0 0 14px}
.count{font-size:.85rem;color:#b45309;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}
.card{background:#fff;border:1px solid #d8dee9;border-radius:12px;padding:14px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.card h3{margin:0 0 10px;font-size:1rem}
.kv{display:grid;grid-template-columns:110px 1fr;gap:4px 8px;font-size:.9rem;margin:4px 0}
.k{font-weight:600;color:#475569}
.v{font-family:ui-monospace,Menlo,monospace;font-size:.82rem;word-break:break-word}
.media{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}
.media figure{margin:0}
.media img,.media video{max-width:100%;border-radius:8px;border:1px solid #e5e7eb;background:#fafafa}
.media.wide{grid-template-columns:1fr}
.media figcaption{font-size:.72rem;color:#64748b;margin-bottom:4px}
.missing{font-size:.8rem;color:#94a3b8;padding:24px;text-align:center;border:1px dashed #cbd5e1;border-radius:8px}
.empty{color:#64748b;font-style:italic}
p,li{font-size:.88rem;line-height:1.45}
ul{margin:6px 0;padding-left:18px}
code{font-size:.8rem;background:#f1f5f9;padding:2px 4px;border-radius:4px}
"""

    doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Pilot40 wrong-answer notebook</title><style>{css}</style></head><body>"
        + "".join(body_parts)
        + "</body></html>"
    )

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(doc, encoding="utf-8")
    return OUT_HTML


def main() -> None:
    out = build()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
