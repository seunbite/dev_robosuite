#!/usr/bin/env python3
"""Stacked bar: orig pass vs corr pass (o→o black + x→o white). SVG → PNG/PDF via rsvg-convert."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
LABELS_PATH = (
    _REPO
    / "data/results/visualize/tiago/correction/exp1/gemini-2.5-pro/visual_outcome_labels.json"
)
OUT_DIR = _REPO / "data/results/paper_figures/correction"


def _counts(labels: dict) -> tuple[int, int, int, int]:
    oo = xo = 0
    for v in labels.values():
        o, c = v.get("orig", ""), v.get("corr", "")
        if o == "o" and c == "o":
            oo += 1
        elif o == "x" and c == "o":
            xo += 1
    n = len(labels)
    orig_pct = round((sum(1 for v in labels.values() if v.get("orig") == "o")) / n * 100)
    oo_pct = round(oo / n * 100)
    xo_pct = round(xo / n * 100)
    return orig_pct, oo_pct, xo_pct, n


def render_svg(orig_pct: int, oo_pct: int, xo_pct: int, n: int) -> str:
    h, w = 360, 280
    pad_l, pad_b, pad_t = 48, 56, 52
    chart_h = h - pad_b - pad_t
    bar_w = 72
    x0, x1 = 90, 220
    ymax = 100

    def y(v: int) -> float:
        return pad_t + chart_h * (1 - v / ymax)

    oo_h = chart_h * oo_pct / ymax
    xo_h = chart_h * xo_pct / ymax
    orig_h = chart_h * orig_pct / ymax

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <style>
    text {{ font-family: Helvetica, Arial, sans-serif; fill: #111; }}
    .label {{ font-size: 13px; text-anchor: middle; }}
    .pct {{ font-size: 18px; font-weight: bold; text-anchor: middle; dominant-baseline: middle; }}
    .title {{ font-size: 14px; font-weight: bold; text-anchor: middle; }}
    .axis {{ stroke: #333; stroke-width: 1; }}
    .grid {{ stroke: #ddd; stroke-width: 1; }}
  </style>
  <rect x="0" y="0" width="{w}" height="{h}" fill="#fff"/>
  <text class="title" x="{w/2}" y="28">Tiago exp1 — visual pass (n={n})</text>
  <line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + chart_h}"/>
  <line class="axis" x1="{pad_l}" y1="{pad_t + chart_h}" x2="{w - 20}" y2="{pad_t + chart_h}"/>
  <text x="{w - 12}" y="{pad_t + chart_h/2}" transform="rotate(90 {w - 12} {pad_t + chart_h/2})" class="label">Script-level</text>
  <line class="grid" x1="{pad_l}" y1="{y(25)}" x2="{w-20}" y2="{y(25)}"/>
  <line class="grid" x1="{pad_l}" y1="{y(50)}" x2="{w-20}" y2="{y(50)}"/>
  <line class="grid" x1="{pad_l}" y1="{y(75)}" x2="{w-20}" y2="{y(75)}"/>

  <rect x="{x0 - bar_w/2}" y="{y(orig_pct)}" width="{bar_w}" height="{orig_h}" fill="#111" stroke="#111"/>
  <text class="pct" x="{x0}" y="{y(orig_pct) + orig_h/2}" fill="#fff">{orig_pct}</text>

  <rect x="{x1 - bar_w/2}" y="{y(oo_pct + xo_pct)}" width="{bar_w}" height="{xo_h}" fill="#fff" stroke="#111" stroke-width="1.5"/>
  <rect x="{x1 - bar_w/2}" y="{y(oo_pct)}" width="{bar_w}" height="{oo_h}" fill="#111" stroke="#111"/>
  <text class="pct" x="{x1}" y="{y(oo_pct) + oo_h/2}" fill="#fff">{oo_pct}</text>
  <text class="pct" x="{x1}" y="{y(oo_pct + xo_pct) + xo_h/2}" fill="#111">{xo_pct}</text>

  <text class="label" x="{x0}" y="{h - 28}">Try 1</text>
  <text class="label" x="{x0}" y="{h - 12}">(Original)</text>
  <text class="label" x="{x1}" y="{h - 28}">Try 2</text>
  <text class="label" x="{x1}" y="{h - 12}">(Correction)</text>
</svg>"""


def _export_png_pdf(svg: str, png: Path, pdf: Path, *, dpi: int = 300) -> None:
    rsvg = shutil.which("rsvg-convert")
    if not rsvg:
        raise RuntimeError("rsvg-convert not found (install librsvg)")
    png.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
        f.write(svg.encode("utf-8"))
        tmp = Path(f.name)
    try:
        subprocess.run([rsvg, "-b", "white", "-d", str(dpi), "-p", str(dpi), "-o", str(png), str(tmp)], check=True)
        subprocess.run([rsvg, "-f", "pdf", "-o", str(pdf), str(tmp)], check=True)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    orig_pct, oo_pct, xo_pct, n = _counts(labels)
    svg = render_svg(orig_pct, oo_pct, xo_pct, n)
    stem = Path(sys.argv[1]).stem if len(sys.argv) > 1 else "orig_corr_pass_bar"
    out_dir = Path(sys.argv[1]).parent if len(sys.argv) > 1 else OUT_DIR
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    _export_png_pdf(svg, png, pdf)
    print(f"saved {png}")
    print(f"saved {pdf}")
    print(f"orig={orig_pct}%  o→o={oo_pct}%  x→o={xo_pct}%  corr_total={oo_pct+xo_pct}%")


if __name__ == "__main__":
    main()
