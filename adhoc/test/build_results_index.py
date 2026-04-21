import glob
import os
from pathlib import Path

import fire


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main(results_dir: str = "adhoc/test/results", output_name: str = "index.html"):
    results_path = Path(results_dir).resolve()
    html_files = sorted(results_path.glob("*_report.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    out_path = results_path / output_name

    parts = ["""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VLM Eval Results</title>
<style>
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #f5f7fb; color: #17202a; }
.wrap { max-width: 1100px; margin: 0 auto; padding: 24px; }
.card { background: white; border: 1px solid #d7dee8; border-radius: 14px; padding: 16px; margin-bottom: 14px; }
a { color: #0969da; text-decoration: none; }
a:hover { text-decoration: underline; }
.meta { color: #667382; font-size: 13px; margin-top: 4px; }
</style>
</head>
<body><div class="wrap"><h1>VLM Eval Results</h1>"""]

    for path in html_files:
        rel = os.path.relpath(path, out_path.parent)
        parts.append(
            f'<section class="card"><div><a href="{_esc(rel)}">{_esc(path.name)}</a></div>'
            f'<div class="meta">{_esc(path.stat().st_mtime_ns)}</div></section>'
        )

    parts.append("</div></body></html>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))

    print(f"Index HTML: {out_path}")
    print(f"Index URL: file://{out_path}")


if __name__ == "__main__":
    fire.Fire(main)
