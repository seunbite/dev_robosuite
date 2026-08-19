#!/usr/bin/env python3
"""HTML review for N-way pose tile GT identification (grid 6 / 12)."""
from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from verify_pose_vlm import _load_tile_pick, _resolve_pose_image  # noqa: E402

IN_JSON = _REPO / "data/results/verify/pilot20_pose_multitile_gt_gemini.json"
OUT_HTML = _REPO / "data/results/html/manipulator/pose_multitile_gt_review.html"
TILE_CACHE = _REPO / "data/results/visualize/_multitile_review_tiles"


def _esc(x: object) -> str:
    return html.escape(str(x) if x is not None else "")


def _rel(path: Path | str | None, out_html: Path) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.is_file():
        p = _REPO / path
    if not p.is_file():
        return ""
    try:
        return os.path.relpath(p, out_html.parent)
    except ValueError:
        return str(p)


def _resolve_dir(raw: str | Path | None, fallback: Path) -> Path:
    if raw:
        p = Path(str(raw))
        if p.is_dir():
            return p
        local = _REPO / p
        if local.is_dir():
            return local
    return fallback


def _resolve_file(raw: str | Path | None, fallback: Path) -> Path:
    if raw:
        p = Path(str(raw))
        if p.is_file():
            return p
        local = _REPO / p
        if local.is_file():
            return local
    return fallback


def _export_tile_image(
    *,
    tile: dict,
    tile_dir: Path,
    tile_pick: dict[tuple[str, str], int],
    thumb_dir: Path,
) -> Path | None:
    d = str(tile.get("dir") or "")
    g = str(tile.get("gripper_orientation") or "")
    if not d or not g:
        return None
    idx = tile.get("tile_pick")
    if idx is None:
        idx = tile_pick.get((d, g))
    if idx is None:
        return None
    group_path = tile_dir / f"group_{d}_{g}.png"
    if not group_path.is_file():
        return None
    out_path = thumb_dir / f"group_{d}_{g}_t{int(idx):02d}.png"
    if out_path.is_file():
        return out_path
    try:
        cell, _ = _resolve_pose_image(group_path, d, g, int(idx), None, False)
    except Exception:
        return None
    thumb_dir.mkdir(parents=True, exist_ok=True)
    cell.save(out_path)
    return out_path


def _tile_thumbs_html(
    r: dict,
    *,
    tile_dir: Path,
    tile_pick: dict[tuple[str, str], int],
    thumb_dir: Path,
    out_html: Path,
    pick: int | None,
) -> str:
    parts: list[str] = []
    for tile in r.get("tiles") or []:
        disp = int(tile.get("display_index", 0))
        is_gt = bool(tile.get("is_gt"))
        is_pick = pick is not None and disp == int(pick)
        classes = ["tile-thumb"]
        if is_gt:
            classes.append("gt")
        if is_pick:
            classes.append("pick")
        if is_pick and is_gt:
            classes.append("pick-ok")
        elif is_pick and not is_gt:
            classes.append("pick-miss")

        img_path = _export_tile_image(
            tile=tile,
            tile_dir=tile_dir,
            tile_pick=tile_pick,
            thumb_dir=thumb_dir,
        )
        src = _rel(img_path, out_html) if img_path else ""
        img_tag = (
            f'<img src="{_esc(src)}" alt="#{disp}"/>'
            if src
            else '<div class="no-img">no image</div>'
        )
        tags = []
        if is_gt:
            tags.append("GT")
        if is_pick:
            tags.append("PICK")
        tag_html = f'<span class="tags">{_esc(" · ".join(tags))}</span>' if tags else ""
        parts.append(
            f'<div class="{" ".join(classes)}">'
            f'<div class="tile-lbl">#{disp}</div>'
            f"{img_tag}"
            f'<div class="tile-cap">dir={_esc(tile.get("dir"))}<br>grip={_esc(tile.get("gripper_orientation"))}</div>'
            f"{tag_html}"
            f"</div>"
        )
    return f'<div class="tile-strip">{"".join(parts)}</div>'


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--in-json", type=Path, default=IN_JSON)
    p.add_argument("--out-html", type=Path, default=OUT_HTML)
    args = p.parse_args()

    data = json.loads(args.in_json.read_text(encoding="utf-8"))
    results = data.get("results", [])
    summary = data.get("summary", {})
    out_html: Path = args.out_html
    thumb_dir = TILE_CACHE

    tile_dir = _resolve_dir(
        data.get("tile_dir"),
        _REPO / "data/results/visualize/pose_groups_12",
    )
    tile_pick_file = _resolve_file(
        data.get("tile_pick_json"),
        _REPO / "data/results/verify/pose_tile_pick_by_group.json",
    )
    tile_pick = _load_tile_pick(tile_pick_file)

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
            grid_img = _rel(r.get("grid_image"), out_html)
            assess = (r.get("vlm_result") or {}).get("direction_orientation_assessment", "")
            tile_thumbs = _tile_thumbs_html(
                r,
                tile_dir=tile_dir,
                tile_pick=tile_pick,
                thumb_dir=thumb_dir,
                out_html=out_html,
                pick=int(pick) if pick is not None else None,
            )
            rows_html.append(
                "<tr>"
                f"<td>c{r.get('cue_idx')}</td><td>{_esc(cue)}</td><td>{r.get('grid_n')}</td>"
                f"<td><code>{_esc(gt)}</code></td><td><code>{_esc(pick)}</code></td>"
                f'<td class="{cls}">{verdict}</td>'
                f"<td><div class='gt-line'><b>Human GT:</b> {_esc(r.get('groundtruth', ''))}</div>"
                f'<img class="grid-img" src="{_esc(grid_img)}" alt="grid input"/></td>'
                f"<td>{tile_thumbs}</td>"
                f"<td><details><summary>assessment</summary><pre>{_esc(assess)}</pre></details></td>"
                "</tr>"
            )

    model = data.get("model", "")
    backend = data.get("vlm_backend", "")
    title_bits = [x for x in (model, backend) if x]
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
.gt-line{{font-size:13px;margin:0 0 8px;color:#333}}
.grid-img{{max-width:520px;border:1px solid #ccc;border-radius:6px;display:block}}
.tile-strip{{display:flex;flex-wrap:wrap;gap:8px;max-width:640px}}
.tile-thumb{{width:108px;border:1px solid #ddd;border-radius:8px;padding:6px;background:#fff;text-align:center}}
.tile-thumb.gt{{border-color:#86efac;background:#f0fdf4}}
.tile-thumb.pick{{box-shadow:0 0 0 2px #2563eb}}
.tile-thumb.pick-ok{{box-shadow:0 0 0 2px #16a34a}}
.tile-thumb.pick-miss{{box-shadow:0 0 0 2px #dc2626}}
.tile-thumb img{{width:100%;height:auto;display:block;border-radius:4px}}
.tile-lbl{{font-weight:700;font-size:12px;margin-bottom:4px}}
.tile-cap{{font-size:10px;color:#555;margin-top:4px;line-height:1.3}}
.tags{{display:block;font-size:10px;font-weight:700;color:#1d4ed8;margin-top:4px}}
.no-img{{height:72px;display:flex;align-items:center;justify-content:center;font-size:11px;color:#888;background:#f4f4f5;border-radius:4px}}
</style></head><body>
<h1>Pose multitile GT identification — pilot-90</h1>
<p><b>Model:</b> <code>{_esc(' · '.join(title_bits) or 'n/a')}</code></p>
<p>{_esc(' | '.join(summary_bits))}</p>
<p>Task: pick GT iconic pose tile among shuffled (dir, grip) representatives.
   Green border = GT tile; blue ring = model pick; green ring = correct pick.</p>
<table>
<thead><tr>
<th>idx</th><th>cue</th><th>grid</th><th>GT indices</th><th>pick</th><th>result</th>
<th>grid input</th><th>input tiles</th><th>assessment</th>
</tr></thead>
<tbody>{''.join(rows_html)}</tbody>
</table></body></html>"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(doc, encoding="utf-8")
    print(f"Wrote {out_html}")


if __name__ == "__main__":
    main()
