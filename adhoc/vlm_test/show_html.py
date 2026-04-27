#!/usr/bin/env python3
"""
Build a static HTML preview under adhoc/vlm_test/: humaneval task + prompt + per-input_type media
(without VLM). Defaults match run_exp: sample_n=20; robot=all (binary) or robot=manipulator (compare).
"""
from __future__ import annotations

import argparse
import base64
import html
import sys
from pathlib import Path

_DEV_R = Path(__file__).resolve().parents[2]
_TEST = _DEV_R / "adhoc" / "test"
if str(_TEST) not in sys.path:
    sys.path.insert(0, str(_TEST))

from heval_data import (  # noqa: E402
    load_binary_instances,
    load_compare_instances,
)
from testset_utils import (  # noqa: E402
    expand_input_types,
    prepare_test_media,
    normalize_test_media_type,
)


def _embed_file(path: str, mime: str) -> str:
    p = Path(path)
    if not p.is_file():
        return f'<p class="err">Missing: {html.escape(str(p))}</p>'
    b = p.read_bytes()
    b64 = base64.standard_b64encode(b).decode("ascii")
    if mime == "image/gif" or (mime or "").endswith("gif"):
        return f'<img class="g" src="data:{mime};base64,{b64}" alt="" />'
    if (mime or "").startswith("image/") or p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif"):
        return f'<img class="g" src="data:{mime or "image/png"};base64,{b64}" alt="" />'
    if (mime or "").startswith("video/") or p.suffix.lower() == ".mp4":
        return f'<video class="g" controls src="data:{mime or "video/mp4"};base64,{b64}"></video>'
    return f'<p>binary {html.escape(mime or "?")} ({len(b)} B)</p>'


def _rows_for_instance(inst, input_types: list[str], *, hz: int) -> str:
    sample = inst.meta.get("_sample")
    if not sample:
        return "<tr><td>— (no _sample; compare or missing binary meta)</td></tr>"
    blocks = []
    for it in input_types:
        t = normalize_test_media_type(it)
        sim_robot = (sample.get("sim_robot") or "IIWA").strip()
        try:
            prepped = prepare_test_media([dict(sample)], test_type=t, robot=sim_robot, hz=hz, force=False)
        except Exception as e:
            blocks.append(
                f'<tr><td><code>{html.escape(t)}</code></td><td class="err">{html.escape(str(e))}</td></tr>'
            )
            continue
        for row in prepped:
            mpath = row.get("media_path") or row.get("gif_path")
            mm = row.get("media_mime") or "image/png"
            cell = _embed_file(str(mpath), mm)
            blocks.append(
                f'<tr><td><code>{html.escape(t)}</code></td><td>{cell}</td></tr>'
            )
    return "\n".join(blocks) if blocks else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("binary_classification", "compare_baseline"), default="binary_classification")
    ap.add_argument("--robot", type=str, default=None)
    ap.add_argument("--sample_n", type=int, default=None)
    ap.add_argument(
        "--first_n",
        type=int,
        default=None,
        help="After sample_n shuffle, keep only the first N items per robot.",
    )
    ap.add_argument(
        "--input_type",
        type=str,
        default="all",
        help="For binary: testset_utils names; `all` expands to static types (no sim).",
    )
    ap.add_argument("--position_seed", type=int, default=20260424)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", type=str, default="", help=f"default: {_TEST / 'heval_vlm_preview.html'}")
    ap.add_argument("--hz", type=int, default=8)
    args = ap.parse_args()

    if args.sample_n is None:
        args.sample_n = 20
    if args.robot is None:
        args.robot = "manipulator,tiago" if args.task == "binary_classification" else "manipulator"

    if args.task == "binary_classification":
        insts = load_binary_instances(
            robot=args.robot, sample_n=args.sample_n, seed=args.seed, first_n=args.first_n
        )
    else:
        insts = load_compare_instances(
            robot=args.robot, sample_n=args.sample_n, position_seed=args.position_seed, seed=args.seed
        )
    itypes = expand_input_types(args.input_type)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>heval VLM preview</title>",
        "<style>body{font-family:system-ui,Segoe UI,Arial;margin:1.2rem;max-width:1200px} .box{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0} pre{white-space:pre-wrap} img.g,video.g{max-width:420px;max-height:360px} .err{color:#a00} table{border-collapse:collapse} td{padding:0.4rem 0.6rem;border:1px solid #ddd;vertical-align:top}</style>",
        "</head><body>",
        f"<h1>{html.escape(args.task)} · robot={html.escape(args.robot)} · input_type={html.escape(args.input_type)}</h1>",
        f"<p>Instances: {len(insts)} (sample_n={args.sample_n})</p>",
    ]
    for inst in insts:
        parts.append('<div class="box">')
        parts.append(f"<h2>{html.escape(inst.instance_id)}</h2>")
        parts.append(f"<h3>Prompt</h3><pre>{html.escape(inst.prompt)}</pre>")
        parts.append(f"<h3>Raw media ({len(inst.media)})</h3>")
        for role, p, m in inst.media:
            parts.append(f"<p><b>{html.escape(role)}</b> ({html.escape(m)})</p>{_embed_file(str(p), m)}")
        if args.task == "binary_classification":
            tbl = _rows_for_instance(inst, itypes, hz=args.hz)
            if tbl:
                parts.append("<h3>Prepared by input type</h3><table><tr><th>type</th><th>media</th></tr>")
                parts.append(tbl)
                parts.append("</table>")
        parts.append("</div>")

    parts.append("</body></html>")
    html_s = "\n".join(parts)
    out = Path(args.out) if args.out else _TEST / "heval_vlm_preview.html"
    out.write_text(html_s, encoding="utf-8")
    print(str(out.resolve()))


if __name__ == "__main__":
    main()
