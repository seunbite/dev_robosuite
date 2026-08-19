#!/usr/bin/env python3
"""Task A — HTML review for labeled pos/neg plausibility scoring vs Gemini."""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from generate_pose_group_tiles import ORDER_DIR, ORDER_GRIP  # noqa: E402
from pilot40_paths import HTML_EXP_DIR  # noqa: E402

DEFAULT_JSON = _REPO / "data/results/verify/tiago/pose_plausibility_labeled_12groups.json"
DEFAULT_TILE_ROOT = _REPO / "data/results/visualize/tiago/pose_plausibility_labeled"
DEFAULT_PROMPT = _REPO / "data/seed/prompt/tiago/exp/prompt_expA.txt"
DEFAULT_HTML = HTML_EXP_DIR / "expA_labeled_scoring_gemini-2.5-pro.html"


def _all_group_keys() -> list[str]:
    return [f"{d}_{g}" for d in ORDER_DIR for g in ORDER_GRIP]


def _rel_href(out_html: Path, asset: Path) -> str:
    if not asset.is_file():
        return ""
    return Path(os.path.relpath(asset.resolve(), out_html.parent.resolve())).as_posix()


def _score_tile(*, gt: str, pred_plausible: bool) -> str:
    pred = "pos" if pred_plausible else "neg"
    if pred == gt:
        return "correct"
    if pred == "pos":
        return "fp"
    return "fn"


def _badge(score: str) -> str:
    if score == "correct":
        return '<span class="badge ok">✓</span>'
    if score == "fp":
        return '<span class="badge fp">FP</span>'
    return '<span class="badge fn">FN</span>'


def _gt_badge(gt: str) -> str:
    if gt == "pos":
        return '<span class="gt pos">pos</span>'
    return '<span class="gt neg">neg</span>'


def _pred_badge(plausible: bool) -> str:
    if plausible:
        return '<span class="pred plaus">plausible</span>'
    return '<span class="pred implaus">implausible</span>'


def build(
    *,
    json_path: Path,
    tile_root: Path,
    prompt_path: Path,
    html_out: Path,
) -> Path:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    groups_data = payload.get("groups") or {}
    user_neg = payload.get("user_neg") or {}
    user_pos = payload.get("user_pos") or {}
    metrics = payload.get("metrics_all_12groups") or {}
    prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else ""

    group_sections: list[str] = []
    summary_rows: list[str] = []

    for group in _all_group_keys():
        neg_set = set(user_neg.get(group, []))
        pos_set = set(user_pos.get(group, []))
        labels = {i: "neg" for i in neg_set}
        for i in pos_set:
            labels[i] = "pos"
        rows = groups_data.get(group, [])
        by_idx = {int(r["tile_idx"]): r for r in rows}

        gc = gfp = gfn = 0
        table_rows: list[str] = []
        cards: list[str] = []

        for idx in sorted(labels):
            gt = labels[idx]
            rec = by_idx.get(idx)
            if not rec:
                continue
            plausible = bool(rec.get("is_plausible"))
            score = _score_tile(gt=gt, pred_plausible=plausible)
            if score == "correct":
                gc += 1
            elif score == "fp":
                gfp += 1
            else:
                gfn += 1

            reason = html.escape(str(rec.get("reason", "")))
            tile_path = tile_root / group / f"tile_{idx:02d}.png"
            img_href = _rel_href(html_out, tile_path)
            img = (
                f'<img src="{html.escape(img_href)}" loading="lazy" alt="tile {idx}"/>'
                if img_href
                else '<p class="missing">tile image missing</p>'
            )
            table_rows.append(
                f"<tr class='{score}'><td>#{idx}</td>"
                f"<td>{_gt_badge(gt)}</td>"
                f"<td>{_pred_badge(plausible)}</td>"
                f"<td>{_badge(score)}</td>"
                f"<td class='reason'>{reason}</td></tr>"
            )
            cards.append(
                f"<article class='tile {score}'><div class='head'>"
                f"<b>#{idx}</b> {_gt_badge(gt)} → {_pred_badge(plausible)} {_badge(score)}"
                f"</div>{img}<p class='reason'>{reason}</p></article>"
            )

        gn = gc + gfp + gfn
        acc = gc / gn if gn else 0.0
        summary_rows.append(
            f"<tr><td>{html.escape(group)}</td><td>{gc}/{gn}</td>"
            f"<td>{acc:.3f}</td><td>{gfp}</td><td>{gfn}</td></tr>"
        )
        group_sections.append(
            f"<section class='group' id='{html.escape(group)}'>"
            f"<h2>{html.escape(group)} "
            f"<span class='sub'>{gc}/{gn} correct · FP {gfp} · FN {gfn}</span></h2>"
            f"<table class='score-table'><thead><tr>"
            f"<th>tile</th><th>GT</th><th>pred</th><th>score</th><th>reason</th>"
            f"</tr></thead><tbody>{''.join(table_rows)}</tbody></table>"
            f"<div class='tile-grid'>{''.join(cards)}</div></section>"
        )

    n = int(metrics.get("n", 0))
    correct = int(metrics.get("correct", 0))
    fp = int(metrics.get("fp", 0))
    fn = int(metrics.get("fn", 0))
    acc_all = correct / n if n else 0.0

    css = """
body{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e8eaed}
.wrap{padding:16px 24px 48px;max-width:1400px;margin:0 auto}
h1{font-size:1.35rem;margin:0 0 8px}
h2{font-size:1.05rem;margin:24px 0 10px;padding-top:8px;border-top:1px solid #333}
h2 .sub{font-weight:400;color:#9aa0a6;font-size:0.9rem}
.summary{background:#171a21;border:1px solid #333;border-radius:10px;padding:14px;margin:12px 0}
.metrics{display:flex;flex-wrap:wrap;gap:12px;margin:8px 0}
.metric{background:#1a1d24;border:1px solid #3c4043;border-radius:8px;padding:8px 14px;font-size:14px}
.metric b{font-size:1.15rem;color:#a5d6ff}
.nav{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0}
.nav a{font-size:12px;color:#8ab4f8;text-decoration:none;padding:4px 8px;border:1px solid #3c4043;border-radius:6px}
.nav a:hover{background:#1a1d24}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 16px}
th,td{border:1px solid #333;padding:6px 8px;text-align:left;vertical-align:top}
th{background:#1a1d24}
tr.fp{background:rgba(242,139,130,0.08)}
tr.fn{background:rgba(253,214,99,0.08)}
tr.correct{background:rgba(129,201,149,0.06)}
.reason{color:#cbd5e1;font-size:12px;max-width:480px}
.tile-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.tile{background:#171a21;border:1px solid #333;border-radius:10px;padding:8px;overflow:hidden}
.tile.fp{border-color:#f28b82}
.tile.fn{border-color:#fdd663}
.tile.correct{border-color:#3c4043}
.tile img{width:100%;height:auto;border-radius:6px;display:block;background:#000}
.tile .head{font-size:13px;margin-bottom:6px}
.badge{font-weight:700;font-size:11px;padding:2px 6px;border-radius:4px}
.badge.ok{background:#1e3a2f;color:#81c995}
.badge.fp{background:#3d1f1f;color:#f28b82}
.badge.fn{background:#3d3419;color:#fdd663}
.gt,.pred{font-size:11px;font-weight:600;padding:1px 6px;border-radius:4px}
.gt.pos{background:#1e3a2f;color:#81c995}
.gt.neg{background:#2d1f3d;color:#c58af9}
.pred.plaus{background:#1e3a2f;color:#81c995}
.pred.implaus{background:#2d1f3d;color:#c58af9}
.missing{color:#9aa0a6;font-size:12px;padding:24px;text-align:center}
details{margin-top:12px}
details summary{cursor:pointer;color:#8ab4f8;font-size:13px}
pre{background:#0d0f12;border:1px solid #333;border-radius:8px;padding:12px;font-size:11px;white-space:pre-wrap;max-height:320px;overflow:auto;color:#cbd5e1}
"""
    nav = "".join(
        f'<a href="#{html.escape(g)}">{html.escape(g)}</a>' for g in _all_group_keys()
    )
    summary_table = (
        "<table><thead><tr><th>group</th><th>correct</th><th>acc</th><th>FP</th><th>FN</th>"
        "</tr></thead><tbody>"
        + "".join(summary_rows)
        + "</tbody></table>"
    )
    prompt_block = ""
    if prompt_text:
        prompt_block = (
            "<details><summary>prompt_expA.txt (VLM input text)</summary>"
            f"<pre>{html.escape(prompt_text)}</pre></details>"
        )

    body = (
        f"<h1>Task A — labeled pos/neg vs Gemini plausibility</h1>"
        f"<div class='summary'>"
        f"<p><b>Input:</b> one rendered PNG tile per labeled pose + <code>prompt_expA.txt</code></p>"
        f"<p><b>GT:</b> user <code>pos</code> = USER_SELECTION · user <code>neg</code> = manual picker</p>"
        f"<p><b>Model:</b> gemini-2.5-pro · <b>JSON:</b> "
        f"<code>{html.escape(json_path.relative_to(_REPO).as_posix())}</code></p>"
        f"<div class='metrics'>"
        f"<div class='metric'><b>{correct}/{n}</b><br/>accuracy ({acc_all:.3f})</div>"
        f"<div class='metric'><b>{fp}</b><br/>FP (neg→plausible)</div>"
        f"<div class='metric'><b>{fn}</b><br/>FN (pos→implausible)</div>"
        f"</div>"
        f"{summary_table}{prompt_block}</div>"
        f"<nav class='nav'>{nav}</nav>"
        + "".join(group_sections)
    )
    doc = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Task A — labeled scoring review</title><style>{css}</style></head>"
        f"<body><div class='wrap'>{body}</div></body></html>"
    )
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(doc, encoding="utf-8")
    return html_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--tile-root", type=Path, default=DEFAULT_TILE_ROOT)
    ap.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    ap.add_argument("--html-out", type=Path, default=DEFAULT_HTML)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    out = build(
        json_path=args.json.resolve(),
        tile_root=args.tile_root.resolve(),
        prompt_path=args.prompt.resolve(),
        html_out=args.html_out.resolve(),
    )
    print(f"[html] wrote {out}")
    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    main()
