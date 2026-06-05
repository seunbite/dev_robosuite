#!/usr/bin/env python3
"""HTML review for N-way pose tile GT identification (grid 6 / 12)."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]

IN_JSON = _REPO / "data/results/verify/pilot20_pose_multitile_gt_gemini.json"
OUT_HTML = _REPO / "data/results/html/manipulator/pose_multitile_gt_review.html"


def _esc(x: object) -> str:
    return html.escape(str(x) if x is not None else "")


def _rel(path: str | None, out_html: Path) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        return os.path.relpath(p, out_html.parent)
    except ValueError:
        return str(p)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--in-json", type=Path, default=IN_JSON)
    p.add_argument("--out-html", type=Path, default=OUT_HTML)
    args = p.parse_args()

    data = json.loads(args.in_json.read_text(encoding="utf-8"))
    results = data.get("results", [])
    summary = data.get("summary", {})

    by_cue: dict[str, list[dict]] = {}
    for r in results:
        by_cue.setdefault(r.get("cue", "?"), []).append(r)

    summary_bits = []
    for key in sorted(summary):
        s = summary[key]
        acc = s.get("accuracy")
        acc_txt = f"{100 * acc:.1f}%" if acc is not None else "n/a"
        n = key.replace("grid_", "")
        summary_bits.append(f"grid {n}: {s.get('ok')}/{s.get('n')} = {acc_txt} (random {100/int(n):.1f}%)")

    rows_html: list[str] = []
    for cue in sorted(by_cue, key=lambda c: int(by_cue[c][0].get("cue_idx", 0))):
        for r in sorted(by_cue[cue], key=lambda x: x.get("grid_n", 0)):
            if r.get("error"):
                continue
            pick = r.get("vlm_pick_index")
            gt = r.get("gt_indices") or []
            ok = r.get("vlm_correct")
            if pick is None and ok is None:
                cls, verdict = "pending", "PENDING"
            else:
                cls = "ok" if ok else "miss"
                verdict = "OK" if ok else "MISS"
            img = _rel(r.get("grid_image"), args.out_html)
            assess = (r.get("vlm_result") or {}).get("direction_orientation_assessment", "")
            tile_lines = "<br>".join(
                _esc(
                    f"#{t['display_index']}: dir={t['dir']}, grip={t['gripper_orientation']}"
                    + (" [GT]" if t.get("is_gt") else "")
                )
                for t in r.get("tiles", [])
            )
            rows_html.append(
                "<tr>"
                f"<td>c{r.get('cue_idx')}</td><td>{_esc(cue)}</td><td>{r.get('grid_n')}</td>"
                f"<td><code>{_esc(gt)}</code></td><td><code>{_esc(pick)}</code></td>"
                f'<td class="{cls}">{verdict}</td>'
                f'<td><img src="{_esc(img)}" style="max-width:520px;border:1px solid #ccc;border-radius:6px"/></td>'
                f"<td>{tile_lines}</td>"
                f"<td><details><summary>assessment</summary><pre>{_esc(assess)}</pre></details></td>"
                "</tr>"
            )

    doc = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Pose multitile GT review</title>
<style>
body{{font-family:system-ui,sans-serif;margin:20px}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ddd;padding:8px;vertical-align:top}}
th{{background:#f3f5f8}}
.ok{{color:#1b5e20;font-weight:700}}
.miss{{color:#b71c1c;font-weight:700}}
.pending{{color:#9a6700;font-weight:700}}
pre{{white-space:pre-wrap;background:#f4f4f5;padding:8px;border-radius:6px;font-size:12px}}
</style></head><body>
<h1>Pose multitile GT identification (pilot 20)</h1>
<p>{_esc(' | '.join(summary_bits))}</p>
<p>Task: pick GT iconic pose tile among shuffled (dir, grip) representatives.</p>
<table>
<thead><tr>
<th>idx</th><th>cue</th><th>grid</th><th>GT indices</th><th>pick</th><th>result</th>
<th>input</th><th>tiles</th><th>assessment</th>
</tr></thead>
<tbody>{''.join(rows_html)}</tbody>
</table></body></html>"""

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(doc, encoding="utf-8")
    print(f"Wrote {args.out_html}")


if __name__ == "__main__":
    main()
