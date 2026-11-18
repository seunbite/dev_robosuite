#!/usr/bin/env python3
"""Build a 150-row × 5-method human-eval HTML with randomized 4AFC choices.

Rows: 30 cues × 5 morphologies.
Columns: GT | Joint | Cartesian | Motion Program | Trajectory.
Choices: target + hybrid 3 negatives, shuffled with a fixed seed.
"""

from __future__ import annotations

import html
import json
import os
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT_HTML = REPO / "data/results/html/human_eval_4afc_150x5.html"
OUT_JSON = REPO / "data/results/html/human_eval_4afc_150x5_trials.json"
HARD_NEG = (
    REPO
    / "papers/icra2027_motion_abstraction/analysis_hard_negative_strategies.json"
)
CUES = REPO / "data/seed/agent_fill/embodiment_short/cues.json"

PLATFORMS = ("humanoid", "tiago", "stretch", "single_arm", "roomba")
METHODS = (
    ("gt", "GT / Human SMP"),
    ("joint", "Joint"),
    ("xyzrpy", "Cartesian"),
    ("motion_program", "Motion Program (LLM)"),
    ("trajectory", "Selective Cartesian"),
)
SEED = 20260825


def cue_map() -> dict[int, str]:
    rows = json.loads(CUES.read_text(encoding="utf-8"))
    return {int(row["idx"]): str(row["cue"]) for row in rows}


def hybrid_foils() -> dict[tuple[str, int], list[str]]:
    payload = json.loads(HARD_NEG.read_text(encoding="utf-8"))
    out: dict[tuple[str, int], list[str]] = {}
    for row in payload["rows"]:
        out[(row["platform"], int(row["target_idx"]))] = list(
            row["strategies"]["hybrid"]
        )
    return out


def trajectory_renders() -> dict[tuple[str, int], Path]:
    sources = (
        REPO / "data/results/drag_draw/gemini25pro_c1_25/predictions_rendered.json",
        REPO / "data/results/drag_draw/gemini25pro_c1_5_c26_30/predictions_rendered.json",
        REPO / "data/results/drag_draw/roomba_sophisticated_full30/predictions_rendered.json",
    )
    renders: dict[tuple[str, int], Path] = {}
    for source in sources:
        if not source.is_file():
            continue
        for row in json.loads(source.read_text(encoding="utf-8")):
            if row.get("parse_ok") and row.get("render_gif"):
                renders[(row["platform"], int(row["idx"]))] = REPO / row["render_gif"]
    return renders


def render_path(
    method: str,
    platform: str,
    idx: int,
    cue: str,
    trajectory: dict[tuple[str, int], Path],
) -> Path | None:
    if method == "trajectory":
        return trajectory.get((platform, idx))
    stem = f"emb_g{idx:02d}_{cue}"
    if method in ("joint", "xyzrpy"):
        if platform == "roomba":
            base = (
                REPO
                / "data/results/baseline_traj/roomba_sophisticated_full30"
                / f"render_{method}/roomba"
            )
            candidate = base / f"{stem}_gemini-2.5-pro.gif"
            return candidate if candidate.is_file() else None
        base = REPO / f"data/results/baseline_traj/ours5/render_{method}/{platform}"
        for candidate in (base / f"{stem}_gemini-2.5-pro.gif", base / f"{stem}.gif"):
            if candidate.is_file():
                return candidate
        return None
    if method == "motion_program" and platform == "roomba":
        path = REPO / f"data/results/roomba_sophisticated_full30_motion_code/render/{stem}.gif"
        return path if path.is_file() else None
    root = (
        "embodiment_motion_code_gt"
        if method == "gt"
        else "embodiment_motion_code_gemini_gemini-2.5-pro"
    )
    render_platform = "manipulator" if platform == "single_arm" else platform
    if method == "motion_program" and platform in ("humanoid", "tiago"):
        ee_path = (
            REPO
            / "data/results/render"
            / f"{root}_ee_translate"
            / render_platform
            / "gif"
            / f"{stem}.gif"
        )
        if ee_path.is_file():
            return ee_path
    base = REPO / f"data/results/render/{root}/{render_platform}/gif"
    if platform == "single_arm":
        base /= "IIWA"
    path = base / f"{stem}.gif"
    return path if path.is_file() else None


def method_cell(
    method: str,
    platform: str,
    idx: int,
    cue: str,
    trajectory: dict[tuple[str, int], Path],
) -> str:
    rendered = render_path(method, platform, idx, cue, trajectory)
    if not rendered:
        return "<div class='cell missing'>render unavailable</div>"
    rel = html.escape(os.path.relpath(rendered, OUT_HTML.parent))
    return (
        f"<div class='cell'>"
        f"<img loading='lazy' src='{rel}?v={rendered.stat().st_mtime_ns}' "
        f"alt='{html.escape(method)} {html.escape(platform)} c{idx:02d}'>"
        f"</div>"
    )


def main() -> None:
    cues = cue_map()
    foils = hybrid_foils()
    trajectory = trajectory_renders()
    rng = random.Random(SEED)

    trials = []
    row_html = []
    missing = 0
    for idx in range(1, 31):
        cue = cues[idx]
        for platform in PLATFORMS:
            negatives = foils[(platform, idx)]
            if len(negatives) != 3:
                raise ValueError(f"expected 3 foils for {platform} c{idx:02d}")
            choices = [cue, *negatives]
            rng.shuffle(choices)
            answer_index = choices.index(cue)
            trial = {
                "trial_id": f"{platform}.c{idx:02d}",
                "platform": platform,
                "idx": idx,
                "target": cue,
                "negatives": negatives,
                "choices_randomized": choices,
                "answer_index": answer_index,
                "seed": SEED,
            }
            trials.append(trial)

            cells = []
            for method, _label in METHODS:
                path = render_path(method, platform, idx, cue, trajectory)
                if path is None:
                    missing += 1
                cells.append(method_cell(method, platform, idx, cue, trajectory))

            choice_spans = []
            for i, label in enumerate(choices):
                mark = " ★" if i == answer_index else ""
                choice_spans.append(
                    f"<span class='choice{' ans' if i == answer_index else ''}'>"
                    f"{chr(65 + i)}. {html.escape(label)}{mark}</span>"
                )

            row_html.append(
                "<tr>"
                f"<td class='meta'><div class='id'>{html.escape(trial['trial_id'])}</div>"
                f"<div class='cue'>{html.escape(cue)}</div>"
                f"<div class='plat'>{html.escape(platform)}</div>"
                f"<div class='choices'>{''.join(choice_spans)}</div>"
                f"<div class='neg'>foils: {html.escape(', '.join(negatives))}</div></td>"
                + "".join(f"<td>{c}</td>" for c in cells)
                + "</tr>"
            )

    headers = "".join(f"<th>{html.escape(label)}</th>" for _, label in METHODS)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Human eval set — 150 × 5 methods (4AFC)</title>
<style>
:root {{ color-scheme: dark; }}
body {{ margin: 0; font-family: ui-sans-serif, system-ui, sans-serif; background:#0b0d10; color:#e8eaed; }}
header {{ position:sticky; top:0; z-index:5; background:#12151aee; backdrop-filter:blur(8px);
  border-bottom:1px solid #2a3038; padding:12px 16px; }}
header h1 {{ margin:0; font-size:1.05rem; font-weight:650; }}
header p {{ margin:4px 0 0; font-size:.82rem; color:#9aa3ad; }}
.wrap {{ overflow:auto; max-height: calc(100vh - 72px); }}
table {{ border-collapse: separate; border-spacing:0; width:max-content; min-width:100%; }}
th, td {{ border-bottom:1px solid #232830; vertical-align:top; padding:8px; }}
th {{ position:sticky; top:0; background:#1a1f26; font-size:.75rem; text-transform:uppercase;
  letter-spacing:.04em; color:#b6bec8; z-index:2; }}
td.meta {{ position:sticky; left:0; background:#12151a; min-width:240px; max-width:280px; z-index:1;
  border-right:1px solid #2a3038; }}
.id {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.78rem; color:#8ab4f8; }}
.cue {{ font-weight:600; margin-top:2px; }}
.plat {{ font-size:.75rem; color:#9aa3ad; margin-bottom:6px; }}
.choices {{ display:flex; flex-direction:column; gap:2px; font-size:.72rem; }}
.choice {{ color:#c5cbd3; }}
.choice.ans {{ color:#81c995; font-weight:600; }}
.neg {{ margin-top:6px; font-size:.68rem; color:#7d8692; }}
.cell img {{ width:180px; height:auto; display:block; background:#000; border-radius:4px; }}
.cell.missing {{ width:180px; height:100px; display:flex; align-items:center; justify-content:center;
  background:#1a1f26; color:#f28b82; font-size:.75rem; border-radius:4px; }}
tr:hover td {{ background:#161a20; }}
tr:hover td.meta {{ background:#1a2030; }}
</style></head><body>
<header>
  <h1>Human eval set — 150 trials × 5 methods</h1>
  <p>Rows = cue × morphology (30×5). Columns = methods. Choices = target + hybrid 3 negatives, shuffled (seed={SEED}). Answer marked ★ for experimenters only.</p>
  <p>Missing renders: {missing} / {150 * len(METHODS)}. Trials JSON: {html.escape(OUT_JSON.name)}</p>
</header>
<div class="wrap">
<table>
<thead><tr><th>Trial / 4AFC</th>{headers}</tr></thead>
<tbody>
{''.join(row_html)}
</tbody>
</table>
</div>
</body></html>
"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(page, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(trials, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_HTML}")
    print(f"wrote {OUT_JSON} ({len(trials)} trials)")
    print(f"missing render cells: {missing}")


if __name__ == "__main__":
    main()
