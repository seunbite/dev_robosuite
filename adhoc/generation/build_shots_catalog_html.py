#!/usr/bin/env python3
"""Build ``data/results/html/shots_from_see_sources/shots_catalog.html`` from see_html_sources.yml shots paths.

- **manipulator / tiago**: GIFs under ``data/results/render/shots_from_see_sources/`` (render there first).
- **quadruped**: resolve ``data/results/render/quadruped/*{cue}*.gif`` (mjlab naming). Missing files require
  mjlab + Python &lt; 3.14 (see ``adhoc/generation/quadruped/render.py`` or render single cues via unitree env).

Open the HTML with ``python adhoc/generation/build_shots_catalog_html.py --open``."""
from __future__ import annotations

import argparse
import html as html_module
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adhoc.utils.repo_paths import dev_robosuite_root  # noqa: E402


def _rel(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target, from_dir)


def _escape_cfg(cfg: dict) -> str:
    return html_module.escape(json.dumps(cfg, indent=2, ensure_ascii=False))


def _quadruped_gif(repo: Path, cue: str) -> Path | None:
    d = repo / "data/results/render/quadruped"
    if not d.is_dir():
        return None
    needle = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(cue))
    cands: list[Path] = []
    cands.extend(d.glob(f"lc19*{needle}*_mjlab_earth.gif"))
    cands.extend(d.glob(f"*{needle}*mjlab*.gif"))
    if not cands:
        return None
    return max(set(cands), key=lambda p: p.stat().st_mtime)


def build(*, open_browser: bool = False) -> Path:
    repo = dev_robosuite_root()
    html_dir = repo / "data/results/html/shots_from_see_sources"
    html_dir.mkdir(parents=True, exist_ok=True)
    gif_google = repo / "data/results/render/shots_from_see_sources/tiago"
    gif_manip = repo / "data/results/render/shots_from_see_sources/manipulator" / "IIWA"

    paths = {
        "manipulator": repo / "data/seed/shots/manipulator/shot_configs_v19_sophisticated.json",
        "tiago": repo / "data/seed/shots/tiago/shot_configs_19_mobile.json",
        "quadruped": repo / "data/seed/shots/quadruped/shot_configs_19_locomotion.json",
    }
    yaml_hint = "\n".join(
        f"• {k}: {p.relative_to(repo)} (see data/seed/yml/see_html_sources.yml → shots_json)"
        for k, p in paths.items()
    )

    blocks: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>",
        "<title>Few-shot catalogs (see_html_sources.yml)</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px}",
        "h1,h2{border-bottom:1px solid #ccc}",
        "section{border:1px solid #ddd;border-radius:10px;padding:16px;margin:20px 0;background:#fafafa}",
        ".meta{color:#555;font-size:14px}article{border-top:1px solid #eee;padding:12px 0}",
        "img{max-width:100%;height:auto;border:1px solid #ccc}",
        "pre{font-size:12px;white-space:pre-wrap;background:#fff;padding:10px;border-radius:6px}</style>",
        "</head><body>",
        "<h1>Few-shot shot configs</h1>",
        f"<p class='meta'><pre>{html_module.escape(yaml_hint)}</pre></p>",
    ]

    grow = json.loads(paths["tiago"].read_text(encoding="utf-8"))
    sec = ["<section><h2>tiago (TIAGo / bimanual)</h2>"]
    for row in grow:
        idx = int(row.get("idx", -1))
        cue = row.get("cue", "?")
        safe = str(cue).replace("/", "_").replace("\\", "_").replace(" ", "_")
        gif = gif_google / f"mm19_g{idx:02d}_{safe}.gif"
        rel = _rel(html_dir, gif) if gif.is_file() else ""
        ig = f'<p><img src="{html_module.escape(rel)}" alt="" loading="lazy"/></p>' if rel else "<p><em>GIF missing — run tiago/render.py with this JSON and --output_dir shots_from_see_sources/tiago</em></p>"
        sec.append(
            f"<article><h3>g{idx:02d} {html_module.escape(str(cue))}</h3>{ig}<pre>{_escape_cfg(row)}</pre></article>"
        )
    sec.append("</section>")
    blocks.append("\n".join(sec))

    mrows = json.loads(paths["manipulator"].read_text(encoding="utf-8"))
    mans = sorted(gif_manip.glob("*.gif"), key=lambda p: p.stat().st_mtime) if gif_manip.is_dir() else []
    sec = [
        "<section><h2>manipulator (IIWA)</h2>",
        "<p class='meta'>GIFs in <code>data/results/render/shots_from_see_sources/manipulator/IIWA/</code>, "
        "paired with sophisticated shot JSON rows in scan order vs file mtime (same pipeline as robotarm/render).</p>",
    ]
    if len(mans) >= len(mrows):
        for i, (row, gif) in enumerate(zip(mrows, mans)):
            cue = row.get("cue", "?")
            rel = _rel(html_dir, gif)
            sec.append(
                f"<article><h3>shot {i} {html_module.escape(str(cue))}</h3>"
                f"<p><small>{html_module.escape(gif.name)}</small></p>"
                f'<p><img src="{html_module.escape(rel)}" alt="" loading="lazy"/></p>'
                f"<pre>{_escape_cfg(row)}</pre></article>"
            )
    else:
        sec.append(f"<p><em>Expected {len(mrows)} GIFs under IIWA/, found {len(mans)}.</em></p>")
    sec.append("</section>")
    blocks.append("\n".join(sec))

    qrows = json.loads(paths["quadruped"].read_text(encoding="utf-8"))
    sec = [
        "<section><h2>quadruped (GO2 / mjlab)</h2>",
        "<p class='meta'>GIFs resolved from <code>data/results/render/quadruped/</code> by <code>cue</code> "
        "(e.g. <code>lc19_*{cue}*_mjlab_earth.gif</code>). If missing, run "
        "<code>adhoc/generation/quadruped/render.py</code> with a Python that has <code>torch</code>+"
        "<code>mjlab</code> (Python 3.10–3.13), or set <code>MOTION_RENDER_PYTHON</code>.</p>",
    ]
    for row in qrows:
        idx = int(row.get("idx", -1))
        cue = row.get("cue", "?")
        gif = _quadruped_gif(repo, str(cue))
        if gif and gif.is_file():
            rel = _rel(html_dir, gif)
            ig = (
                f'<p><img src="{html_module.escape(rel)}" alt="" loading="lazy"/>'
                f"<br/><small>{html_module.escape(gif.name)}</small></p>"
            )
        else:
            ig = "<p><em>No GIF under render/quadruped for this cue — render missing shots.</em></p>"
        sec.append(
            f"<article><h3>g{idx:02d} {html_module.escape(str(cue))}</h3>{ig}<pre>{_escape_cfg(row)}</pre></article>"
        )
    sec.append("</section>")
    blocks.append("\n".join(sec))

    blocks.append("</body></html>")
    out = html_dir / "shots_catalog.html"
    out.write_text("\n".join(blocks), encoding="utf-8")
    print(out)
    if open_browser:
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(out)])
            elif os.name == "nt":
                os.startfile(str(out))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(out)])
        except OSError as e:
            print("Could not open browser:", e)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--open", action="store_true", help="open shots_catalog.html in default browser")
    args = p.parse_args()
    build(open_browser=args.open)


if __name__ == "__main__":
    main()
