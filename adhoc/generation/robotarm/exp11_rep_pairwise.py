#!/usr/bin/env python3
"""Exp11 — temporal pairwise (repetition-flip) input specs + review HTML."""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from build_pilot90_motion_rep_pairwise_preview import (  # noqa: E402
    build_rep_pairwise_specs,
    load_vlm_results,
    write_rep_pairwise_html,
)
from pilot90_paths import (  # noqa: E402
    HTML_EXP_DIR,
    MOTION_REP_PAIRWISE_DIR,
    REP_PAIRWISE_INPUT_HTML,
    REP_PAIRWISE_SPECS_NAME,
    VERIFY_EXP_DIR,
)


def _esc(x: object) -> str:
    return html.escape(str(x) if x is not None else "")


def _pair_rep_href(mp4_dir: Path, pair_mp4: str, *, from_dir: Path) -> tuple[str, bool]:
    name = Path(pair_mp4).name
    local = mp4_dir / name
    if not local.is_file():
        for candidate in (Path(pair_mp4), mp4_dir.parent.parent.parent / pair_mp4):
            if candidate.is_file():
                local = candidate.resolve()
                break
    href = os.path.relpath(local, from_dir) if local.is_file() else f"../../verify/samples/motion_gt_neg_rep_pairwise_pilot90/{name}"
    return href.replace("\\", "/"), local.is_file()


def write_exp11_result_html(
    result_json: Path,
    *,
    mp4_dir: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    """Build result review HTML; video src points at stitched *_pair_rep.mp4 inputs."""
    data = json.loads(result_json.read_text(encoding="utf-8"))
    rows = data.get("mp4") or data.get("rows") or []
    model = str(data.get("model") or result_json.stem.replace("result_exp11_", ""))
    tag = model.replace("/", "-")
    mp4_root = mp4_dir or MOTION_REP_PAIRWISE_DIR
    out = out_path or (HTML_EXP_DIR / "manipulator" / f"exp11_{tag}.html")
    out.parent.mkdir(parents=True, exist_ok=True)

    scored = [r for r in rows if "vlm_correct" in r]
    ok = sum(1 for r in scored if r.get("vlm_correct"))
    acc = data.get("accuracy")
    acc_s = f"{100 * acc:.1f}%" if acc is not None else (f"{100 * ok / len(scored):.1f}%" if scored else "n/a")

    cards: list[str] = []
    for r in rows:
        idx = int(r.get("idx") or r.get("cue_idx") or 0)
        cue = str(r.get("cue") or "")
        gt_side = str(r.get("gt_side") or "?")
        vlm_side = str(r.get("better_side") or r.get("pred") or "?")
        correct = r.get("vlm_correct")
        badge = "ok" if correct else "miss"
        label = "GT" if correct else "MISS"
        assess = (r.get("parsed") or {}).get("motion_assessment") or ""
        pair_mp4 = str(r.get("pair_mp4") or f"{idx:03d}_{cue}_pair_rep.mp4")
        href, exists = _pair_rep_href(mp4_root, pair_mp4, from_dir=out.parent)
        video = (
            f'<video src="{_esc(href)}" controls loop muted playsinline></video>'
            if exists
            else f'<div class="missing">missing: {_esc(Path(pair_mp4).name)}</div>'
        )
        cards.append(
            f"""<section class="card {badge}">
<h3>{idx}. {_esc(cue)} <span class="{badge}">{label}</span></h3>
<p>GT={_esc(gt_side)} · VLM={_esc(vlm_side)}</p>
{video}
<p class="a">{_esc(assess)}</p>
</section>"""
        )

    doc = f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>Exp11 — {_esc(model)}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1000px;margin:20px auto;padding:0 16px 40px;background:#0f1115;color:#e8eaed}}
h1{{font-size:22px}}
.meta{{color:#9aa0a6;margin:12px 0 20px}}
.card{{border:1px solid #333;border-radius:10px;padding:14px;margin:16px 0;background:#171a21}}
.card.ok{{border-color:#2e7d32}} .card.miss{{border-color:#c62828}}
.ok{{color:#81c995;font-weight:700}} .miss{{color:#f28b82;font-weight:700}}
video{{width:100%;max-width:640px;border-radius:6px;background:#000}}
.a{{font-size:13px;color:#9aa0a6;margin-top:8px}}
.missing{{color:#888;padding:24px;background:#222;border-radius:6px}}
</style></head><body>
<h1>Exp11 repetition pairwise — {_esc(model)}</h1>
<div class="meta">GT pick <b>{ok}/{len(scored)} ({acc_s})</b> · scored {len(scored)} · layout=plain pair_rep MP4 · inputs <code>{_esc(mp4_root.name)}</code></div>
{''.join(cards)}
</body></html>"""
    out.write_text(doc, encoding="utf-8")
    print(f"Wrote result HTML → {out}")
    return out


def build_input(*, motion_cfg: Path | None = None, out_dir: Path | None = None) -> tuple[Path, Path]:
    od = out_dir or MOTION_REP_PAIRWISE_DIR
    od.mkdir(parents=True, exist_ok=True)
    entries, payload = build_rep_pairwise_specs(motion_cfg=motion_cfg)
    spec_path = od / REP_PAIRWISE_SPECS_NAME
    spec_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    vlm_by_idx, vlm_model = load_vlm_results(
        VERIFY_EXP_DIR / "result_exp11_gemini-2.5-pro_pilot30.json",
        VERIFY_EXP_DIR / "result_exp11_gemini-2.5-pro.json",
    )
    html_path = write_rep_pairwise_html(
        payload,
        od / REP_PAIRWISE_INPUT_HTML.name,
        vlm_by_idx=vlm_by_idx,
        vlm_model=vlm_model,
    )
    print(f"Wrote {len(entries)} specs → {spec_path}")
    print(f"Wrote HTML → {html_path}")
    return spec_path, html_path


def main() -> None:
    p = argparse.ArgumentParser(description="Exp11 repetition-flip pairwise — build specs + input HTML")
    p.add_argument("--motion-cfg", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=MOTION_REP_PAIRWISE_DIR)
    p.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help="Write result review HTML from verify JSON (uses *_pair_rep.mp4 inputs)",
    )
    p.add_argument("--open", action="store_true")
    args = p.parse_args()

    html_path: Path
    if args.result_json:
        html_path = write_exp11_result_html(args.result_json, mp4_dir=args.out_dir)
    else:
        _, html_path = build_input(motion_cfg=args.motion_cfg, out_dir=args.out_dir)

    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(html_path)], check=False)


if __name__ == "__main__":
    main()
