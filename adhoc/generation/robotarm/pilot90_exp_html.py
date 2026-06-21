"""Build per-experiment review HTML for pilot-90 (data/results/html/exp{N}_{tag}.html)."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from pilot90_paths import GT_PATH, html_result_path


def _esc(x: object) -> str:
    return html.escape(str(x) if x is not None else "")


def _badge(ok: bool | None) -> str:
    if ok is True:
        return '<span class="ok">OK</span>'
    if ok is False:
        return '<span class="miss">MISS</span>'
    return '<span class="na">—</span>'


def _headline(data: dict[str, Any], kind: str) -> str:
    if kind in {"pose_generation_score", "motion_generation_score"}:
        ok, n = int(data.get("n_correct", 0)), int(data.get("n", 0))
        acc = data.get("accuracy")
        pct = f"{100 * acc:.1f}%" if acc is not None else "n/a"
        return f"{ok}/{n} = {pct}"
    if kind in {"pose_verify_vlm", "pose_verify_text"}:
        ok, n = int(data.get("n_correct", 0)), int(data.get("n", 0))
        acc = data.get("accuracy")
        pct = f"{100 * acc:.1f}%" if acc is not None else "n/a"
        return f"verify-gt {ok}/{n} = {pct}" if n else str(data.get("total", len(data.get("results") or [])))
    if kind == "pose_pairwise":
        comps = data.get("comparisons") or data.get("results") or []
        scored = [c for c in comps if "vlm_correct" in c]
        ok = sum(1 for c in scored if c.get("vlm_correct"))
        return f"{ok}/{len(scored)}" if scored else "n/a"
    if kind == "multitile":
        summary = data.get("summary") or {}
        parts = []
        for k, s in sorted(summary.items()):
            if isinstance(s, dict) and s.get("n"):
                parts.append(f"{k}: {s.get('ok')}/{s.get('n')}")
        return "; ".join(parts) if parts else "n/a"
    if kind in {"motion_verify_vlm", "motion_verify_text"}:
        ok, n = int(data.get("n_correct", 0)), int(data.get("n", 0))
        acc = data.get("accuracy")
        pct = f"{100 * acc:.1f}%" if acc is not None else "n/a"
        rows = data.get("rows") or data.get("results") or []
        return f"verify-gt {ok}/{n} = {pct}" if n else str(len(rows))
    if kind == "motion_pairwise_mp4":
        rows = data.get("rows") or data.get("results") or []
        ok = sum(1 for r in rows if r.get("vlm_correct"))
        scored = [r for r in rows if "vlm_correct" in r]
        return f"{ok}/{len(scored)}" if scored else str(len(rows))
    return ""


def _rows_for_table(data: dict[str, Any], kind: str) -> tuple[list[str], list[list[str]]]:
    headers: list[str] = ["idx", "cue"]
    body: list[list[str]] = []

    if kind in {"pose_generation_score", "motion_generation_score"}:
        headers += ["verdict", "detail"]
        key = "generation_correct" if kind == "pose_generation_score" else "component_match"
        for r in data.get("rows") or []:
            detail = r.get("pose_gt") or r.get("annotation_raw") or ""
            body.append(
                [
                    _esc(r.get("cue_idx")),
                    _esc(r.get("cue")),
                    _badge(r.get(key)),
                    _esc(detail),
                ]
            )
        return headers, body

    if kind in {"pose_verify_vlm", "pose_verify_text"}:
        headers += ["appropriate", "gen_ok", "rec_ok", "verify"]
        for r in data.get("results") or []:
            result = r.get("result") or {}
            scored = r.get("verify_scoring") or {}
            body.append(
                [
                    _esc(r.get("idx")),
                    _esc(r.get("cue")),
                    _badge(result.get("pose_is_appropriate") if isinstance(result.get("pose_is_appropriate"), bool) else None),
                    _badge(scored.get("generation_correct")),
                    _badge(scored.get("recommended_matches_gt")),
                    _badge(scored.get("verify_correct")),
                ]
            )
        return headers, body

    if kind == "pose_pairwise":
        headers += ["vlm_correct", "groundtruth"]
        for r in data.get("comparisons") or data.get("results") or []:
            body.append(
                [
                    _esc(r.get("cue_idx")),
                    _esc(r.get("cue")),
                    _badge(r.get("vlm_correct")),
                    _esc(r.get("groundtruth")),
                ]
            )
        return headers, body

    if kind == "multitile":
        headers += ["grid", "vlm_correct", "pick", "gt"]
        for r in data.get("results") or []:
            body.append(
                [
                    _esc(r.get("cue_idx")),
                    _esc(r.get("cue")),
                    _esc(r.get("grid_n")),
                    _badge(r.get("vlm_correct")),
                    _esc(r.get("vlm_pick_index")),
                    _esc(r.get("gt_indices")),
                ]
            )
        return headers, body

    if kind in {"motion_verify_vlm", "motion_verify_text"}:
        headers += ["appropriate", "gen_ok", "rec_ok", "verify"]
        for r in data.get("rows") or data.get("results") or []:
            parsed = r.get("parsed") or r.get("result") or r.get("verify_result") or {}
            scored = r.get("verify_scoring") or {}
            body.append(
                [
                    _esc(r.get("cue_idx")),
                    _esc(r.get("cue")),
                    _badge(
                        r.get("movement_is_appropriate")
                        if isinstance(r.get("movement_is_appropriate"), bool)
                        else parsed.get("movement_is_appropriate")
                    ),
                    _badge(scored.get("generation_match")),
                    _badge(scored.get("recommended_matches_gt")),
                    _badge(scored.get("verify_correct")),
                ]
            )
        return headers, body

    if kind == "motion_pairwise_mp4":
        headers += ["vlm_correct", "pair"]
        for r in data.get("rows") or data.get("results") or []:
            body.append(
                [
                    _esc(r.get("cue_idx") or r.get("idx")),
                    _esc(r.get("cue")),
                    _badge(r.get("vlm_correct")),
                    _esc(r.get("pair_mp4") or r.get("pair_label")),
                ]
            )
        return headers, body

    return headers, body


def write_exp_review_html(
    exp_id: str | int,
    model_tag: str,
    result_path: Path,
    *,
    title: str,
    kind: str,
) -> Path | None:
    if not result_path.is_file():
        return None
    data = json.loads(result_path.read_text(encoding="utf-8"))
    headers, rows = _rows_for_table(data, kind)
    out = html_result_path(exp_id, model_tag)
    out.parent.mkdir(parents=True, exist_ok=True)

    table_rows = "\n".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    page = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>{_esc(title)} — {_esc(model_tag)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1.5rem; }}
h1 {{ font-size: 1.25rem; }}
.meta {{ color: #444; margin-bottom: 1rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; }}
th {{ background: #f4f4f4; }}
.ok {{ color: #0a7; font-weight: 600; }}
.miss {{ color: #c33; font-weight: 600; }}
.na {{ color: #888; }}
</style></head><body>
<h1>Exp {int(exp_id)}: {_esc(title)}</h1>
<p class="meta">model_tag={_esc(model_tag)} · headline={_esc(_headline(data, kind))}<br/>
json: <code>{_esc(result_path)}</code><br/>
gt: <code>{_esc(GT_PATH)}</code></p>
<table>
<thead><tr>{"".join(f"<th>{_esc(h)}</th>" for h in headers)}</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</body></html>
"""
    out.write_text(page, encoding="utf-8")
    print(f"[html] wrote {out}", flush=True)
    return out
