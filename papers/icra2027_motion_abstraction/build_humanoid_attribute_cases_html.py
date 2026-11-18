#!/usr/bin/env python3
"""HTML gallery: humanoid GT vs Gemini attribute score extremes."""
from __future__ import annotations

import html
import json
from pathlib import Path

REPO = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SCORE = REPO / "data/results/evaluation/humanoid_gt_vs_gemini_loc_dir_move.json"
GT_GIF = REPO / "data/results/render/embodiment_motion_code_gt/humanoid/gif"
LLM_GIF = REPO / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro/humanoid/gif"
# Prefer ee-translate LLM renders when present (closer to paper visuals).
LLM_GIF_EE = REPO / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro_ee_translate/humanoid/gif"
OUT = REPO / "data/results/html/humanoid_gt_vs_gemini_attribute_cases.html"
K = 4  # examples per bucket


def gif_paths(idx: int, cue: str) -> tuple[Path | None, Path | None]:
    stem = f"emb_g{idx:02d}_{cue}.gif"
    gt = GT_GIF / stem
    llm = LLM_GIF_EE / stem
    if not llm.is_file():
        llm = LLM_GIF / stem
    return (gt if gt.is_file() else None, llm if llm.is_file() else None)


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    import os

    return html.escape(os.path.relpath(path, OUT.parent))


def card(row: dict, metric_label: str, metric_value: str) -> str:
    gt, llm = gif_paths(row["idx"], row["cue"])
    gt_src = rel(gt)
    llm_src = rel(llm)
    gt_code = html.escape(row.get("gt_code") or "")
    llm_code = html.escape(row.get("llm_code") or "")
    cue = html.escape(row["cue"])
    missing = ""
    if row["location"].get("missing_axes"):
        missing = f" <span class='warn'>LLM missing axes: {', '.join(row['location']['missing_axes'])}</span>"
    return f"""
    <div class="card">
      <div class="meta">
        <div class="cue">c{row['idx']:02d} · {cue}</div>
        <div class="metric">{html.escape(metric_label)}: <b>{html.escape(metric_value)}</b>{missing}</div>
      </div>
      <div class="pair">
        <div class="col">
          <div class="label">GT</div>
          {"<img loading='lazy' src='" + gt_src + "'>" if gt_src else "<div class='miss'>missing gif</div>"}
          <pre>{gt_code}</pre>
        </div>
        <div class="col">
          <div class="label">Gemini</div>
          {"<img loading='lazy' src='" + llm_src + "'>" if llm_src else "<div class='miss'>missing gif</div>"}
          <pre>{llm_code}</pre>
        </div>
      </div>
    </div>
    """


def section(title: str, blurb: str, rows: list[tuple[dict, str, str]]) -> str:
    body = "\n".join(card(r, lab, val) for r, lab, val in rows)
    return f"""
    <section>
      <h2>{html.escape(title)}</h2>
      <p class="blurb">{html.escape(blurb)}</p>
      <div class="grid">{body}</div>
    </section>
    """


def main() -> None:
    payload = json.loads(SCORE.read_text())
    rows = payload["rows"]
    summary = payload["summary"]

    # Location: applicable with L2
    loc_rows = [r for r in rows if r["location"].get("l2") is not None]
    loc_sorted = sorted(loc_rows, key=lambda r: r["location"]["l2"])
    loc_small = loc_sorted[:K]
    loc_large = list(reversed(loc_sorted[-K:]))

    # Direction: geodesic preferred; else reach angle
    def dir_err(r):
        g = r["direction"].get("geodesic_deg")
        if isinstance(g, (int, float)):
            return float(g)
        a = r["direction"].get("reach_angle_deg")
        return float(a) if isinstance(a, (int, float)) else None

    dir_rows = [r for r in rows if dir_err(r) is not None]
    dir_sorted = sorted(dir_rows, key=dir_err)
    dir_small = dir_sorted[:K]
    dir_large = list(reversed(dir_sorted[-K:]))

    move_ok = [r for r in rows if r["movement"].get("applicable") and r["movement"].get("ok")]
    move_bad = [r for r in rows if r["movement"].get("applicable") and not r["movement"].get("ok")]

    def pack(rs, label_fn):
        return [(r, *label_fn(r)) for r in rs]

    def loc_lab(r):
        loc = r["location"]
        d = loc.get("delta") or {}
        parts = []
        for ax in ("x", "y", "z"):
            if ax in (loc.get("gt") or {}):
                v = d.get(ax)
                parts.append(f"Δ{ax}={'—' if v is None else f'{v:.0f}'}")
        return (
            "Location L2 (body %)",
            f"{loc['l2']:.1f}  ({', '.join(parts)}; mean|Δ|={loc['mean_abs']:.1f})",
        )

    def dir_lab(r):
        d = r["direction"]
        geo = d.get("geodesic_deg")
        reach = d.get("reach_angle_deg")
        palm = d.get("palm_angle_deg")
        rpy = d.get("delta_rpy_deg") or {}
        rpy_s = (
            f"Δrpy=({rpy.get('roll', 0):.0f},{rpy.get('pitch', 0):.0f},{rpy.get('yaw', 0):.0f})"
            if rpy
            else ""
        )
        return (
            "Direction error (°)",
            f"geo={geo:.0f}°  reach={reach if reach is not None else '—'}°  "
            f"palm={palm if palm is not None else '—'}°  {rpy_s}  "
            f"[{d.get('gt_dir')}/{d.get('gt_grip')} → {d.get('llm_dir')}/{d.get('llm_grip')}]",
        )

    def move_lab(r):
        m = r["movement"]
        return (
            "Movement (binary)",
            "OK ✓" if m.get("ok") else "FAIL ✗",
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Humanoid GT vs Gemini — attribute cases</title>
<style>
  :root {{
    --bg: #f6f7f9;
    --card: #fff;
    --ink: #1f2328;
    --muted: #5b6570;
    --line: #d8dee4;
    --good: #1a7f37;
    --bad: #cf222e;
    --warn: #9a6700;
  }}
  body {{
    margin: 0; padding: 24px 28px 64px;
    font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    color: var(--ink); background: var(--bg);
  }}
  h1 {{ font-size: 22px; margin: 0 0 8px; }}
  h2 {{ font-size: 18px; margin: 28px 0 6px; border-bottom: 1px solid var(--line); padding-bottom: 6px; }}
  .blurb {{ color: var(--muted); margin: 0 0 14px; max-width: 900px; }}
  .summary {{
    display: flex; flex-wrap: wrap; gap: 10px; margin: 12px 0 8px;
  }}
  .pill {{
    background: #fff; border: 1px solid var(--line); border-radius: 999px;
    padding: 6px 12px; color: var(--muted); font-size: 13px;
  }}
  .grid {{ display: grid; gap: 14px; }}
  .card {{
    background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 12px 14px 14px; box-shadow: 0 1px 0 rgba(0,0,0,.02);
  }}
  .meta {{ display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: baseline; margin-bottom: 10px; }}
  .cue {{ font-weight: 700; font-size: 15px; }}
  .metric {{ color: var(--muted); }}
  .metric b {{ color: var(--ink); }}
  .warn {{ color: var(--warn); font-size: 12px; }}
  .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .col {{ min-width: 0; }}
  .label {{
    font-size: 12px; font-weight: 700; letter-spacing: .04em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 6px;
  }}
  img {{
    width: 100%; max-height: 280px; object-fit: contain;
    background: #111; border-radius: 8px; display: block;
  }}
  pre {{
    margin: 8px 0 0; padding: 8px 10px; background: #f0f3f6;
    border-radius: 6px; white-space: pre-wrap; word-break: break-word;
    font: 11px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    color: #333; max-height: 90px; overflow: auto;
  }}
  .miss {{
    height: 180px; display: grid; place-items: center;
    background: #eee; border-radius: 8px; color: var(--muted);
  }}
  .tag-good {{ color: var(--good); font-weight: 700; }}
  .tag-bad {{ color: var(--bad); font-weight: 700; }}
  @media (max-width: 900px) {{
    .pair {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
  <h1>Humanoid attribute scoring cases — GT vs Gemini</h1>
  <p class="blurb">
    Continuous metrics: location L2 (body-relative %) and direction geodesic/reach/palm (°).
    Movement is binary (GT joint+axis / path shape ⊆ LLM).
    Each card shows GT (left) and Gemini (right) GIFs with Motion Codes.
  </p>
  <div class="summary">
    <div class="pill">Location mean L2: {summary['location']['mean_l2']:.1f} (n={summary['location']['n_with_xyz']})</div>
    <div class="pill">Direction mean geodesic: {summary['direction']['mean_geodesic_deg']:.1f}° (n={summary['direction']['n_rpy']})</div>
    <div class="pill">Movement acc: {summary['movement']['ok']}/{summary['movement']['n']} ({100*summary['movement']['acc']:.0f}%)</div>
  </div>

  {section(
      "Location — small error",
      "Lowest L2 among cues with GT xyz (missing LLM axes penalized).",
      pack(loc_small, loc_lab),
  )}
  {section(
      "Location — large error",
      "Highest L2 (often wrong coords or omitted axes).",
      pack(loc_large, loc_lab),
  )}
  {section(
      "Direction — small error",
      "Lowest rotation/reach error (geodesic when available).",
      pack(dir_small, dir_lab),
  )}
  {section(
      "Direction — large error",
      "Highest orientation mismatch (often ~90–180° category flips).",
      pack(dir_large, dir_lab),
  )}
  {section(
      "Movement — correct",
      "GT movement/path keys ⊆ LLM keys.",
      pack(move_ok[: max(K, 6)], move_lab),
  )}
  {section(
      "Movement — incorrect",
      "GT joint+axis or path shape not covered by LLM.",
      pack(move_bad[: max(K, 6)], move_lab),
  )}
</body>
</html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html_doc)
    print("wrote", OUT)
    print("loc small", [r["cue"] for r in loc_small])
    print("loc large", [r["cue"] for r in loc_large])
    print("dir small", [r["cue"] for r in dir_small])
    print("dir large", [r["cue"] for r in dir_large])
    print("move ok", len(move_ok), "bad", len(move_bad))


if __name__ == "__main__":
    main()
