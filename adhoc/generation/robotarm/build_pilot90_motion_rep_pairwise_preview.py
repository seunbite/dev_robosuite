#!/usr/bin/env python3
"""Build exp11 repetition-flip pairwise specs + review HTML (pilot-90)."""
from __future__ import annotations

import argparse
import copy
import html
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = Path(__file__).resolve().parents[3]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from motion_gt_tail_builder import (  # noqa: E402
    build_config_from_gt_pose_and_rep_variant,
    human_gt_fixed_pose_from_row,
)
from motion_pairwise_media import compact_tail_label, pairwise_gt_side  # noqa: E402
from pilot90_experiment_suite import MOTION_CFG  # noqa: E402
from pilot90_paths import (  # noqa: E402
    GT_PATH,
    MOTION_REP_PAIRWISE_DIR,
    REP_PAIRWISE_INPUT_HTML,
    REP_PAIRWISE_SPECS_NAME,
    VERIFY_EXP_DIR,
    load_gt_by_cue,
    prompt_exp_path,
)
from score_pilot40_motion_gt_components import _build_annotation_map  # noqa: E402

REP_PAIRWISE_DIR = MOTION_REP_PAIRWISE_DIR
SPECS_NAME = REP_PAIRWISE_SPECS_NAME
PROMPT_EXP11 = prompt_exp_path(11)
OUT_HTML = REP_PAIRWISE_INPUT_HTML


def _esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def _is_single_direction_movement(comp: dict[str, Any]) -> bool:
    """True when every annotated axis is a lone + or − (not +- oscillation)."""
    axes = comp.get("axes") or {}
    return bool(axes) and all(rule in ("+", "-") for rule in axes.values())


def _neg_rep_rule(gt_rep: str | None, comp: dict[str, Any]) -> str | None:
    """Return neg rep_override token, or None to skip cue."""
    if gt_rep == "rep":
        return "non"
    if gt_rep in ("non", None):
        if _is_single_direction_movement(comp):
            return "osc_repeat3"
        return "repeat3"
    return None


def _gt_rep_override(gt_rep: str | None) -> str:
    if gt_rep == "rep":
        return "rep"
    return "non"


def _movement_rep_param(cfg: dict[str, Any]) -> int | None:
    for step in cfg.get("movements") or []:
        if step.get("type") == "movement":
            return (step.get("parameters") or {}).get("repetition")
    return None


def _pairwise_rep_prompt(*, row: dict[str, Any]) -> str:
    tpl = PROMPT_EXP11.read_text(encoding="utf-8")
    return tpl.replace("{{CUE}}", str(row.get("cue", "")))


def build_rep_pairwise_specs(*, motion_cfg: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ann = {int(a["cue_idx"]): a for a in _build_annotation_map()}
    gt_by = load_gt_by_cue()
    rows = json.loads((motion_cfg or MOTION_CFG).read_text(encoding="utf-8"))
    by_idx = {int(r["idx"]): r for r in rows}

    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for idx, row in sorted(by_idx.items()):
        cue = str(row["cue"])
        entry = ann.get(idx) or {}
        if entry.get("always_correct"):
            skipped.append({"cue": cue, "reason": "always_correct"})
            continue
        comp = entry.get("component")
        if not comp or comp.get("kind") in ("any",):
            skipped.append({"cue": cue, "reason": "no_component_or_any"})
            continue
        if comp.get("kind") not in ("movement",):
            skipped.append({"cue": cue, "reason": f"kind={comp.get('kind')}"})
            continue

        gt_rep = comp.get("repetition")
        neg_override = _neg_rep_rule(gt_rep, comp)
        if neg_override is None:
            skipped.append({"cue": cue, "reason": f"rep={gt_rep}"})
            continue

        base = copy.deepcopy(row)
        if not base.get("groundtruth"):
            ev = gt_by.get(cue) or {}
            base["groundtruth"] = ev.get("pose_gt") or ""
        human_pose = human_gt_fixed_pose_from_row(base)
        if not human_pose:
            skipped.append({"cue": cue, "reason": "no_human_pose"})
            continue
        base["gt_fixed_first_pose"] = human_pose

        gt_cfg = build_config_from_gt_pose_and_rep_variant(
            base, comp, rep_override=_gt_rep_override(gt_rep), state_tag="pair_rep_gt"
        )
        neg_cfg = build_config_from_gt_pose_and_rep_variant(
            base, comp, rep_override=neg_override, state_tag="pair_rep_neg"
        )
        if not gt_cfg or not neg_cfg:
            skipped.append({"cue": cue, "reason": "build_failed"})
            continue

        gt_side = pairwise_gt_side(idx, cue)
        left_cfg, right_cfg = (gt_cfg, neg_cfg) if gt_side == "left" else (neg_cfg, gt_cfg)
        left_summary = compact_tail_label(left_cfg)
        right_summary = compact_tail_label(right_cfg)
        prompt = _pairwise_rep_prompt(row=base)

        gt_rep_n = _movement_rep_param(gt_cfg)
        neg_rep_n = _movement_rep_param(neg_cfg)
        mp4_name = f"{idx:03d}_{cue}_pair_rep.mp4"

        entries.append(
            {
                "idx": idx,
                "cue": cue,
                "description": base.get("description", ""),
                "component": comp,
                "gt_repetition_label": gt_rep,
                "gt_rep_override": _gt_rep_override(gt_rep),
                "neg_rep_override": neg_override,
                "gt_side": gt_side,
                "gt_movement_repetition_param": gt_rep_n,
                "neg_movement_repetition_param": neg_rep_n,
                "left": "gt" if gt_side == "left" else "neg",
                "right": "gt" if gt_side == "right" else "neg",
                "left_tail_summary": left_summary,
                "right_tail_summary": right_summary,
                "pair_mp4": mp4_name,
                "prompt": prompt,
                "gt_cfg": gt_cfg,
                "neg_cfg": neg_cfg,
            }
        )

    n_rep_gt = sum(1 for e in entries if e.get("gt_repetition_label") == "rep")
    n_non_gt = len(entries) - n_rep_gt
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "exp11_motion_pairwise_rep_flip",
        "version": "v1",
        "layout": "left_right_plain_mp4_stitched",
        "prompt_template": str(PROMPT_EXP11.relative_to(_REPO)),
        "motion_cfg": str((motion_cfg or MOTION_CFG).relative_to(_REPO)),
        "groundtruth": str(GT_PATH.relative_to(_REPO)),
        "n_specs": len(entries),
        "n_skipped": len(skipped),
        "n_gt_rep": n_rep_gt,
        "n_gt_non": n_non_gt,
        "skipped": skipped,
        "mp4": entries,
    }
    return entries, payload


def load_vlm_results(*json_paths: Path) -> tuple[dict[int, dict[str, Any]], str | None]:
    """Merge exp11 verify JSON rows keyed by cue idx (later files override)."""
    by_idx: dict[int, dict[str, Any]] = {}
    model: str | None = None
    for path in json_paths:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        model = str(data.get("model") or model or path.stem.replace("result_exp11_", ""))
        for row in data.get("mp4") or data.get("rows") or []:
            idx = int(row.get("idx") or row.get("cue_idx") or 0)
            if idx:
                by_idx[idx] = row
    return by_idx, model


def _vlm_section(e: dict[str, Any], vlm: dict[str, Any] | None) -> str:
    if not vlm:
        return '<div class="vlm-out vlm-missing"><p><b>VLM</b> — no scored run for this cue</p></div>'
    gt_side = str(e.get("gt_side") or vlm.get("gt_side") or "?")
    vlm_side = str(vlm.get("better_side") or vlm.get("pred") or "?")
    correct = vlm.get("vlm_correct")
    if correct is True:
        badge_cls, badge_txt = "vlm-ok", "✓ GT match"
    elif correct is False:
        badge_cls, badge_txt = "vlm-miss", f"✗ picked {vlm_side.upper()}, GT={gt_side.upper()}"
    else:
        badge_cls, badge_txt = "vlm-pending", f"VLM={vlm_side.upper()}"
    parsed = vlm.get("parsed") or {}
    assess = parsed.get("motion_assessment") or ""
    conf = parsed.get("confidence")
    conf_s = f"{float(conf):.2f}" if conf is not None else "—"
    raw = vlm.get("raw") or vlm.get("raw_text") or ""
    raw_block = (
        f'<details><summary>raw response</summary><pre>{_esc(raw)}</pre></details>'
        if raw
        else ""
    )
    return f"""<div class="vlm-out {badge_cls}">
  <p class="vlm-verdict"><b>VLM (Gemini)</b> · chose <b>{_esc(vlm_side.upper())}</b> · GT={_esc(gt_side.upper())} · <span class="badge">{_esc(badge_txt)}</span> · conf={_esc(conf_s)}</p>
  <p class="vlm-assess">{_esc(assess)}</p>
  {raw_block}
</div>"""


def _resolve_pair_mp4(e: dict[str, Any]) -> tuple[Path, str]:
    """Resolve on-disk stitched MP4 path and basename for <video src>."""
    idx = int(e["idx"])
    cue = str(e["cue"])
    ref = str(e.get("pair_mp4") or f"{idx:03d}_{cue}_pair_rep.mp4")
    name = Path(ref).name
    for candidate in (Path(ref), _REPO / ref, REP_PAIRWISE_DIR / name):
        if candidate.is_file():
            return candidate.resolve(), name
    return REP_PAIRWISE_DIR / name, name


def _display_entry(e: dict[str, Any], vlm: dict[str, Any] | None) -> dict[str, Any]:
    """Align GT left/right labels with the layout used for MP4 + VLM scoring."""
    out = dict(e)
    if not vlm:
        return out
    vlm_gt = str(vlm.get("gt_side") or out.get("gt_side") or "?")
    if str(out.get("gt_side")) != vlm_gt:
        out["gt_side"] = vlm_gt
        out["left"], out["right"] = out.get("right"), out.get("left")
        out["left_tail_summary"], out["right_tail_summary"] = (
            out.get("right_tail_summary"),
            out.get("left_tail_summary"),
        )
    else:
        out["gt_side"] = vlm_gt
    return out


def _render_cue_section(e: dict[str, Any], vlm_by_idx: dict[int, dict[str, Any]]) -> str:
    idx = int(e["idx"])
    cue = str(e["cue"])
    vlm = vlm_by_idx.get(idx)
    disp = _display_entry(e, vlm)
    local, href = _resolve_pair_mp4(e)
    gt_side = str(disp.get("gt_side", "?"))
    media = (
        f'<video class="media" src="{_esc(href)}" controls loop muted playsinline></video>'
        if local.is_file()
        else '<div class="media missing">pair MP4 not rendered</div>'
    )
    comp = e.get("component") or {}
    vlm_html = _vlm_section(disp, vlm)
    return f"""
<section id="c{idx}" class="cue-card">
  <h3>{idx}. {_esc(cue)}</h3>
  <p class="answer">GT side = {_esc(gt_side.upper())} · human GT rep label = {_esc(e.get('gt_repetition_label'))}</p>
  {vlm_html}
  <table>
    <tr><th>GT tail</th><td>override={_esc(e.get('gt_rep_override'))} · repetition param={_esc(e.get('gt_movement_repetition_param'))}</td></tr>
    <tr><th>NEG tail</th><td>override={_esc(e.get('neg_rep_override'))} · repetition param={_esc(e.get('neg_movement_repetition_param'))}</td></tr>
    <tr><th>component</th><td colspan="1">{_esc(comp.get('kind'))} axes={_esc(comp.get('axes'))} joint={_esc(comp.get('joint'))}</td></tr>
    <tr><th>LEFT ({_esc(disp.get('left'))})</th><td>{_esc(disp.get('left_tail_summary'))}</td></tr>
    <tr><th>RIGHT ({_esc(disp.get('right'))})</th><td>{_esc(disp.get('right_tail_summary'))}</td></tr>
  </table>
  <div class="grid2">
    <div>{media}</div>
    <div><details open><summary>prompt_exp11.txt</summary><pre>{_esc(e.get('prompt',''))}</pre></details></div>
  </div>
</section>"""


def _nav_links(entries: list[dict[str, Any]], *, prefix: str) -> str:
    if not entries:
        return "<p class=\"dim\">—</p>"
    return " ".join(
        f'<a href="#{prefix}c{int(e["idx"])}">{int(e["idx"])}</a>' for e in entries
    )


def write_rep_pairwise_html(
    payload: dict[str, Any],
    out_path: Path,
    *,
    vlm_by_idx: dict[int, dict[str, Any]] | None = None,
    vlm_model: str | None = None,
) -> Path:
    rows = payload.get("mp4") or []
    vlm_by_idx = vlm_by_idx or {}

    correct: list[dict[str, Any]] = []
    wrong: list[dict[str, Any]] = []
    unscored: list[dict[str, Any]] = []
    for e in rows:
        idx = int(e["idx"])
        vlm = vlm_by_idx.get(idx)
        if not vlm or "vlm_correct" not in vlm:
            unscored.append(e)
        elif vlm.get("vlm_correct"):
            correct.append(e)
        else:
            wrong.append(e)

    n_scored = len(correct) + len(wrong)
    acc_pct = f"{100 * len(correct) / n_scored:.1f}%" if n_scored else "n/a"

    blocks: list[str] = []
    if correct:
        cards = "".join(_render_cue_section(e, vlm_by_idx) for e in correct)
        blocks.append(
            f"""<section class="group group-ok" id="group-correct">
  <h2>✓ Correct — VLM picked GT side ({len(correct)}/{n_scored})</h2>
  <nav class="jump"><b>Jump:</b> {_nav_links(correct, prefix="")}</nav>
  {cards}
</section>"""
        )
    if wrong:
        cards = "".join(_render_cue_section(e, vlm_by_idx) for e in wrong)
        blocks.append(
            f"""<section class="group group-miss" id="group-wrong">
  <h2>✗ Wrong — VLM missed GT side ({len(wrong)}/{n_scored})</h2>
  <nav class="jump"><b>Jump:</b> {_nav_links(wrong, prefix="")}</nav>
  {cards}
</section>"""
        )
    if unscored:
        cards = "".join(_render_cue_section(e, vlm_by_idx) for e in unscored)
        blocks.append(
            f"""<section class="group group-pending" id="group-unscored">
  <h2>— Not scored yet ({len(unscored)})</h2>
  <nav class="jump"><b>Jump:</b> {_nav_links(unscored[:40], prefix="")}{' …' if len(unscored) > 40 else ''}</nav>
  {cards}
</section>"""
        )

    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>Exp11 repetition pairwise — pilot-90 preview</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:20px auto;padding:0 16px 48px;background:#fafafa}}
h1{{font-size:22px}} h2{{font-size:19px;margin-top:40px;padding-bottom:6px;border-bottom:2px solid #ccc}}
h3{{font-size:16px;margin:24px 0 8px}}
.summary{{background:#fff;border:1px solid #ccc;border-radius:10px;padding:14px;margin:16px 0}}
.stats{{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}}
.stat{{flex:1;min-width:140px;border-radius:10px;padding:12px 14px;font-size:14px}}
.stat b{{display:block;font-size:22px;margin-bottom:4px}}
.stat-ok{{background:#e8f5e9;border:1px solid #81c784}}
.stat-miss{{background:#ffebee;border:1px solid #e57373}}
.stat-all{{background:#e3f2fd;border:1px solid #64b5f6}}
.topnav{{margin:16px 0;font-size:14px}}
.topnav a{{margin-right:14px;font-weight:600}}
.group-ok h2{{color:#2e7d32;border-color:#81c784}}
.group-miss h2{{color:#c62828;border-color:#e57373}}
.group-pending h2{{color:#666;border-color:#bbb}}
.cue-card{{background:#fff;border:1px solid #ddd;border-radius:10px;padding:14px 16px 18px;margin:18px 0}}
.group-ok .cue-card{{border-left:4px solid #43a047}}
.group-miss .cue-card{{border-left:4px solid #e53935}}
.answer{{background:#e8f5e9;border:1px solid #81c784;border-radius:8px;padding:8px 12px;font-weight:600;margin:8px 0}}
.vlm-out{{border-radius:8px;padding:10px 12px;margin:10px 0;font-size:13px}}
.vlm-ok{{background:#e8f5e9;border:1px solid #81c784}}
.vlm-miss{{background:#ffebee;border:1px solid #e57373}}
.vlm-pending{{background:#fff8e1;border:1px solid #ffb74d}}
.vlm-missing{{background:#f5f5f5;border:1px dashed #bbb;color:#666}}
.vlm-verdict{{margin:0 0 6px}}
.vlm-assess{{margin:0;color:#333;line-height:1.45}}
.badge{{font-weight:700}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px}}
.media{{width:100%;border:1px solid #ccc;border-radius:8px;background:#000;min-height:160px}}
.missing{{background:#eee;color:#666;padding:40px;text-align:center;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}}
pre{{background:#f0f0f2;padding:12px;border-radius:8px;font-size:11px;white-space:pre-wrap;max-height:360px;overflow:auto}}
.jump a{{margin-right:8px;font-size:12px}}
.dim{{color:#888}}
</style></head><body>
<h1>Exp11 — Movement pairwise (repetition flip)</h1>
<div class="summary">
  <p>n={len(rows)} specs · GT rep={payload.get('n_gt_rep')} · GT non={payload.get('n_gt_non')} · skipped={payload.get('n_skipped')} (of pilot-90)</p>
  <p><b>Layout:</b> one stitched left|right plain MP4 per cue (no <code>alpha_frame_trajectory</code> overlay).</p>
  <p><b>Neg rule:</b> human GT <code>rep</code> → single cycle (repetition=1); human GT <code>non</code> with lone <code>+</code>/<code>−</code> axis → <code>+-</code> oscillation ×3; other <code>non</code> → same direction ×3.</p>
  <p>Prompt: <code>{_esc(payload.get('prompt_template'))}</code> · Specs: <code>{_esc(REP_PAIRWISE_SPECS_NAME)}</code></p>
  <p><b>VLM:</b> {_esc(vlm_model or 'exp11 verify JSON')} · scored {n_scored} cues · accuracy <b>{acc_pct}</b></p>
  <div class="stats">
    <div class="stat stat-ok"><b>{len(correct)}</b>correct</div>
    <div class="stat stat-miss"><b>{len(wrong)}</b>wrong</div>
    <div class="stat stat-all"><b>{len(unscored)}</b>not scored</div>
  </div>
  <nav class="topnav">
    <a href="#group-correct">✓ Correct ({len(correct)})</a>
    <a href="#group-wrong">✗ Wrong ({len(wrong)})</a>
    <a href="#group-unscored">— Unscored ({len(unscored)})</a>
  </nav>
</div>
{''.join(blocks)}
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Build exp11 rep-flip pairwise specs + HTML")
    p.add_argument("--motion-cfg", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=REP_PAIRWISE_DIR)
    p.add_argument(
        "--refresh-specs",
        action="store_true",
        help="Rebuild pairwise_specs_rep_pilot90.json (default: reuse existing specs, HTML only)",
    )
    p.add_argument(
        "--vlm-json",
        type=Path,
        action="append",
        default=None,
        help="Exp11 verify JSON(s) to embed VLM answers (default: pilot30 + full gemini runs)",
    )
    p.add_argument("--open", action="store_true")
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / REP_PAIRWISE_SPECS_NAME
    if args.refresh_specs or not spec_path.is_file():
        entries, payload = build_rep_pairwise_specs(motion_cfg=args.motion_cfg)
        spec_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(entries)} specs → {spec_path}", flush=True)
    else:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        print(f"Reusing specs → {spec_path} (n={payload.get('n_specs')})", flush=True)
    vlm_paths = args.vlm_json or [
        VERIFY_EXP_DIR / "result_exp11_gemini-2.5-pro_pilot30.json",
        VERIFY_EXP_DIR / "result_exp11_gemini-2.5-pro.json",
    ]
    vlm_by_idx, vlm_model = load_vlm_results(*vlm_paths)
    html_path = write_rep_pairwise_html(
        payload,
        out_dir / REP_PAIRWISE_INPUT_HTML.name,
        vlm_by_idx=vlm_by_idx,
        vlm_model=vlm_model,
    )
    print(f"Wrote HTML → {html_path}")
    if args.open:
        import subprocess

        if sys.platform == "darwin":
            subprocess.run(["open", str(html_path)], check=False)


if __name__ == "__main__":
    main()
