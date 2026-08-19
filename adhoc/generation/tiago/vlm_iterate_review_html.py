#!/usr/bin/env python3
"""Write a simple gallery HTML for a ``rub_eye_tired_*`` VLM iterate run folder and open it."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = _REPO / "data/results/vlm_pose_iterate"


def _latest_run() -> Path:
    cands = sorted(OUT_ROOT.glob("rub_eye_tired_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise SystemExit(f"No runs under {OUT_ROOT}")
    return cands[0]


def build_html(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    manifest = run_dir / "manifest.jsonl"
    rows: list[dict] = []
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    parts: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>VLM pose iterate — rub_eye_tired</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:960px;margin:24px auto;padding:0 16px}"
        "section{border:1px solid #ddd;border-radius:8px;padding:16px;margin:20px 0;background:#fafafa}"
        "img{max-width:100%;height:auto;border:1px solid #ccc}"
        "pre{white-space:pre-wrap;background:#fff;padding:12px;border-radius:6px;font-size:13px}</style>",
        "</head><body>",
        f"<h1>rub_eye_tired — {run_dir.name}</h1>",
        f"<p><code>{run_dir}</code></p>",
    ]

    for row in rows:
        it = int(row["iter"])
        sub = run_dir / f"iter_{it:03d}"
        gif = sub / "pose_preview.gif"
        judge = sub / "vlm_judge.txt"
        cfg = sub / "config_pose_only.json"
        parts.append(f"<section><h2>Iteration {it} — {row.get('verdict', '?')}</h2>")
        if gif.is_file():
            rel = gif.relative_to(run_dir).as_posix()
            parts.append(f'<p><img src="{rel}" alt="iter {it}"></p>')
        if judge.is_file():
            parts.append(f"<h3>VLM</h3><pre>{judge.read_text(encoding='utf-8')}</pre>")
        if cfg.is_file():
            parts.append(f"<h3>config_pose_only.json</h3><pre>{cfg.read_text(encoding='utf-8')}</pre>")
        parts.append("</section>")

    fin_gif = run_dir / "final_full_motion.gif"
    fin_cfg = run_dir / "final_full_config.json"
    if fin_gif.is_file():
        parts.append("<section><h2>Final full motion (after YES)</h2>")
        parts.append(f'<p><img src="{fin_gif.relative_to(run_dir).as_posix()}" alt="final"></p>')
        if fin_cfg.is_file():
            parts.append(f"<h3>final_full_config.json</h3><pre>{fin_cfg.read_text(encoding='utf-8')}</pre>")
        parts.append("</section>")

    parts.append("</body></html>")
    out = run_dir / "review.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def main() -> None:
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else _latest_run()
    if not run.is_dir():
        raise SystemExit(f"Not a directory: {run}")
    html = build_html(run)
    print(html)


if __name__ == "__main__":
    main()
