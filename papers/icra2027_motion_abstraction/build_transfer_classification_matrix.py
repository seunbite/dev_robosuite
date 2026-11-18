#!/usr/bin/env python3
"""Render a cue-by-robot explanation matrix for transfer classifications."""

from __future__ import annotations

import csv
import ast
from collections import defaultdict
from html import escape
from pathlib import Path


PAPER = Path(__file__).resolve().parent
INPUT = PAPER / "analysis/nonanthropomorphic_transfer_rule_labels.csv"
OUTPUT = PAPER / "analysis/nonanthropomorphic_transfer_matrix.html"
ROBOTS = [("single_arm", "IIWA"), ("stretch", "Stretch"), ("roomba", "Roomba")]


def tokens(text: str) -> set[str]:
    return set(part.strip() for part in text.split("|") if part.strip())


def pretty_tokens(text: str) -> str:
    """Render stored tuple tokens as named Motion-Type fields."""
    shown = []
    for raw in sorted(tokens(text)):
        try:
            family, axis, sign, where = ast.literal_eval(raw)
            shown.append(f"{family}(axis={axis}, sign={sign}, where={where})")
        except (ValueError, SyntaxError):
            shown.append(raw)
    return "<br/>".join(escape(item) for item in shown) or "--"


def explanation(row: dict[str, str]) -> str:
    ref, tgt = tokens(row["reference_signature"]), tokens(row["target_signature"])
    if row["label"] == "retargeting":
        return "Same family / axis / sign; carrier location is ignored."
    missing, added = ref - tgt, tgt - ref
    bits = []
    if missing:
        bits.append("changes " + ", ".join(sorted(missing)))
    if added:
        bits.append("adds " + ", ".join(sorted(added)))
    return "; ".join(bits) or "Dynamic token mismatch."


def cell(row: dict[str, str]) -> str:
    label = "RT" if row["label"] == "retargeting" else "RM"
    cls = "rt" if label == "RT" else "rm"
    return f"""<td class='{cls}'>
      <b>{label}</b><span class='why'>{escape(explanation(row))}</span>
      <code>{pretty_tokens(row['target_signature'])}</code>
    </td>"""


def reference_cell(row: dict[str, str]) -> str:
    return f"<td class='reference'><code>{pretty_tokens(row['reference_signature'])}</code></td>"


def table(author: str, cue_order: list[str], by_key: dict[tuple[str, str, str], dict[str, str]]) -> str:
    rows = []
    for idx, cue in enumerate(cue_order, 1):
        reference = by_key[(author, ROBOTS[0][0], cue)]
        cells = [cell(by_key[(author, platform, cue)]) for platform, _ in ROBOTS]
        rows.append(f"<tr><th>{idx:02d}<br/><code>{escape(cue)}</code></th>{reference_cell(reference)}{''.join(cells)}</tr>")
    headers = "".join(f"<th>{label}</th>" for _, label in ROBOTS)
    return f"""<section><h2>{author}-authored programs</h2>
      <table><thead><tr><th>Cue</th><th>Human humanoid<br/>reference (G1)</th>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></section>"""


def main() -> None:
    rows = list(csv.DictReader(INPUT.open()))
    by_key = {(r["author"], r["platform"], r["cue"]): r for r in rows}
    cue_order = []
    for row in rows:
        if row["cue"] not in cue_order:
            cue_order.append(row["cue"])
    content = "\n".join(table(author, cue_order, by_key) for author in ("Human", "LLM"))
    OUTPUT.write_text(f"""<!doctype html><html lang='en'><head><meta charset='utf-8'/>
<title>Retargeting vs. Rematerialization · 30 cues × 3 robots</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--muted:#8b949e;--rt:#1f6f4a;--rm:#7c3030}}
body{{margin:0;padding:26px;background:var(--bg);color:var(--text);font:14px/1.35 Arial,sans-serif}}
h1{{margin:0 0 6px}} .lede{{max-width:1050px;color:var(--muted)}} section{{margin-top:30px}}
table{{border-collapse:separate;border-spacing:5px;width:100%;min-width:1100px}} th{{text-align:left;vertical-align:top;color:#c9d1d9;padding:8px;position:sticky;top:0;background:var(--bg)}}
td{{width:24%;vertical-align:top;padding:8px;border-radius:7px;border:1px solid var(--line)}} td.reference{{background:#202833}} td.rt{{background:var(--rt)}} td.rm{{background:var(--rm)}}
b{{display:inline-block;margin-right:7px}} .why{{display:block;font-size:12px;margin-bottom:6px}} code{{font:11px/1.35 ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-word}}
</style></head><body><h1>Retargeting vs. rematerialization</h1>
<p class='lede'>Each dynamic action is tokenized as <code>(family, axis, sign, where)</code>; static 
<code>pose</code> and <code>hold</code> primitives are excluded. A target is <b>RT</b> when its 
dynamic-token multiset matches the human-authored humanoid reference after ignoring <code>where</code>; otherwise it is <b>RM</b>. Both tables use the same human-authored humanoid reference.</p>
{content}</body></html>""")
    print(OUTPUT)


if __name__ == "__main__":
    main()
