#!/usr/bin/env python3
"""HTML review: VLM prompt, input PNGs, choice, correct/wrong per pairwise comparison."""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _dev_root() -> Path:
    for anc in _HERE.parents:
        if (anc / "data" / "results" / "verify").is_dir():
            return anc
    raise SystemExit(f"Cannot find dev_robosuite root from {_HERE}")


DEV_ROOT = _dev_root()
for p in (DEV_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from verify_pose_tiles_gemini import (  # noqa: E402
    _load_json,
    _load_tile_pick,
    _resolve_pose_image,
)
from verify_pose_pairwise_12_gemini import _pair_prompt  # noqa: E402

IN_JSON = DEV_ROOT / "data/results/verify/pilot40_pose_pairwise_12_gemini.json"
IMG_REL = "../../visualize/pose_pairwise_12"
TILE_REL = "../../visualize/pose_groups_12"
OUT_HTML = DEV_ROOT / "data/results/html/manipulator/pose_pairwise_12_review.html"
THUMB_DIR = DEV_ROOT / "data/results/visualize/pose_pairwise_12/_review_tiles"
CONFIG_PATHS = [
    DEV_ROOT / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot10.json",
    DEV_ROOT / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot20_more.json",
]


def _configs_by_cue() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in CONFIG_PATHS:
        for row in _load_json(p):
            out[row["cue"]] = row
    return out


def _esc(s: object) -> str:
    return html.escape(str(s))


def _pick_one_per_cue(comps: list[dict]) -> dict[str, dict]:
    """Prefer single-pair run: distractor differs in both dir and grip from GT."""
    by_cue: dict[str, list[dict]] = {}
    for c in comps:
        by_cue.setdefault(c["cue"], []).append(c)

    out: dict[str, dict] = {}
    for cue, items in by_cue.items():
        gt = items[0].get("gt_pose") or {}
        gd, gg = gt.get("dir"), gt.get("gripper_orientation")
        chosen = None
        for c in items:
            w = c.get("wrong_pose") or {}
            if w.get("dir") != gd and w.get("gripper_orientation") != gg:
                chosen = c
                break
        out[cue] = chosen or items[-1]
    return out


def _rel_from_repo(abs_path: str | None) -> str:
    if not abs_path:
        return ""
    p = Path(abs_path)
    try:
        return str(p.relative_to(DEV_ROOT))
    except ValueError:
        return p.name


def _export_side_tiles(
    c: dict,
    tile_dir: Path,
    tile_pick: dict[tuple[str, str], int],
) -> tuple[str, str]:
    """Save LEFT/RIGHT crop PNGs for review; return relative paths under visualize/."""
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    cue = c["cue"]
    left = c.get("left_pose") or {}
    right = c.get("right_pose") or {}
    paths: list[str] = []

    for side, pose in (("left", left), ("right", right)):
        d, g = pose.get("dir"), pose.get("gripper_orientation")
        if not d or not g:
            paths.append("")
            continue
        group_path = tile_dir / f"group_{d}_{g}.png"
        idx = tile_pick.get((d, g), 1)
        out_name = f"{cue}_{side}_{d}_{g}_t{idx:02d}.png"
        out_path = THUMB_DIR / out_name
        rel = f"{IMG_REL}/_review_tiles/{out_name}"
        if not out_path.is_file() and group_path.is_file():
            cell, _ = _resolve_pose_image(group_path, d, g, idx, None, False)
            cell.save(out_path)
        paths.append(rel if out_path.is_file() else "")

    return paths[0], paths[1]


def _card(
    c: dict,
    *,
    cfg_by_cue: dict,
    tile_dir: Path,
    tile_pick: dict[tuple[str, str], int],
) -> str:
    cue = c["cue"]
    gt = c.get("gt_pose") or {}
    wr = c.get("wrong_pose") or {}
    left = c.get("left_pose") or {}
    right = c.get("right_pose") or {}
    gt_side = c.get("gt_side", "?")
    vlm_side = c.get("vlm_better_side", "—")
    ok = c.get("vlm_correct")
    err = c.get("error")

    if err:
        return (
            f'<article class="card err"><h3>{_esc(cue)}</h3>'
            f'<p class="errmsg">{_esc(err)}</p></article>'
        )

    pair_rel = _rel_from_repo(c.get("pair_image"))
    pair_src = f"{IMG_REL}/{Path(pair_rel).name}" if pair_rel else ""

    left_rel, right_rel = _export_side_tiles(c, tile_dir, tile_pick)

    if ok is True:
        verdict = '<span class="verdict ok">CORRECT</span>'
        verdict_detail = f"VLM chose <b>{_esc(vlm_side)}</b> — GT was on <b>{_esc(gt_side)}</b>."
    elif ok is False:
        verdict = '<span class="verdict miss">WRONG</span>'
        verdict_detail = (
            f"VLM chose <b>{_esc(vlm_side)}</b> — GT was on <b>{_esc(gt_side)}</b> "
            f"(should have picked <b>{_esc(gt_side)}</b>)."
        )
    else:
        verdict = '<span class="verdict dry">not scored</span>'
        verdict_detail = ""

    desc = (cfg_by_cue.get(cue) or {}).get("description", "")
    prompt = _pair_prompt(
        cue=cue,
        description=desc,
        left_d=left.get("dir", "?"),
        left_g=left.get("gripper_orientation", "?"),
        right_d=right.get("dir", "?"),
        right_g=right.get("gripper_orientation", "?"),
    )
    vlm = c.get("vlm_result") or {}
    assessment = vlm.get("direction_orientation_assessment", "")
    confidence = vlm.get("confidence", "")

    def thumb(label: str, rel: str, pose: dict, tile_n: int | None, is_gt: bool) -> str:
        tag = "gt" if is_gt else "dist"
        if not rel:
            return f'<div class="thumb missing"><div class="lbl">{_esc(label)}</div><p>no image</p></div>'
        return (
            f'<div class="thumb {tag}">'
            f'<div class="lbl">{_esc(label)}</div>'
            f'<img src="{_esc(rel)}" alt="{_esc(label)}"/>'
            f'<div class="cap">dir={_esc(pose.get("dir"))}, grip={_esc(pose.get("gripper_orientation"))}'
            + (f' · tile #{tile_n}' if tile_n else "")
            + (" · <b>GT</b>" if is_gt else " · distractor")
            + "</div></div>"
        )

    gt_on = gt_side
    left_is_gt = gt_on == "left"
    tile_gt = c.get("tile_gt")
    tile_wr = c.get("tile_wrong")

    return f"""
<article class="card">
  <header class="card-h">
    <h3>{_esc(cue)} <span class="idx">idx {c.get('cue_idx','')}</span></h3>
    {verdict}
  </header>
  <p class="gt-line"><b>Human GT:</b> {_esc(c.get('groundtruth',''))}</p>
  <p class="gt-line"><b>GT pose (specified):</b> dir={_esc(gt.get('dir'))}, grip={_esc(gt.get('gripper_orientation'))}
     · distractor: dir={_esc(wr.get('dir'))}, grip={_esc(wr.get('gripper_orientation'))}</p>
  <p class="verdict-line">{verdict_detail}</p>

  <h4>VLM input (stitched pair sent to model)</h4>
  <div class="pair-input">
    <img src="{_esc(pair_src)}" alt="pair input"/>
    <p class="cap">Randomized layout: GT on <b>{_esc(gt_side)}</b>.
       Labels on image: LEFT = dir={_esc(left.get('dir'))}, grip={_esc(left.get('gripper_orientation'))};
       RIGHT = dir={_esc(right.get('dir'))}, grip={_esc(right.get('gripper_orientation'))}.</p>
  </div>

  <h4>Input tiles (cropped from 12-group grids)</h4>
  <div class="thumbs">
    {thumb("LEFT", left_rel, left, tile_gt if left_is_gt else tile_wr, left_is_gt)}
    {thumb("RIGHT", right_rel, right, tile_gt if not left_is_gt else tile_wr, not left_is_gt)}
  </div>

  <details class="prompt-box">
    <summary>Prompt sent to VLM (this trial)</summary>
    <pre>{_esc(prompt)}</pre>
  </details>

  <div class="vlm-out">
    <h4>VLM output</h4>
    <table class="kv">
      <tr><th>better_side</th><td><code>{_esc(vlm_side)}</code></td></tr>
      <tr><th>confidence</th><td>{_esc(confidence)}</td></tr>
      <tr><th>assessment</th><td>{_esc(assessment)}</td></tr>
    </table>
  </div>
</article>
"""


def main() -> None:
    data = json.loads(IN_JSON.read_text(encoding="utf-8"))
    comps = data.get("comparisons", [])
    picked = _pick_one_per_cue(comps)
    cfg_by_cue = _configs_by_cue()
    tile_dir = Path(data.get("tile_dir", DEV_ROOT / "data/results/visualize/pose_groups_12"))
    tile_pick = _load_tile_pick(DEV_ROOT / "data/results/verify/pose_tile_pick_by_group.json")

    scored = [c for c in picked.values() if "vlm_correct" in c]
    ok = sum(1 for c in scored if c["vlm_correct"])
    acc = ok / len(scored) if scored else None

    example = next(iter(picked.values()))
    ex_left = example.get("left_pose") or {}
    ex_right = example.get("right_pose") or {}
    template_prompt = _pair_prompt(
        cue="(example cue)",
        description="(cue description from motion config)",
        left_d=ex_left.get("dir", "left"),
        left_g=ex_left.get("gripper_orientation", "horizontal"),
        right_d=ex_right.get("dir", "right"),
        right_g=ex_right.get("gripper_orientation", "vertical"),
    )

    cards = "".join(
        _card(c, cfg_by_cue=cfg_by_cue, tile_dir=tile_dir, tile_pick=tile_pick)
        for cue in sorted(picked.keys())
        for c in [picked[cue]]
    )

    doc = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Pose pairwise VLM review</title>
<style>
:root {{
  --bg: #f4f5f8; --card: #fff; --border: #d8dce6;
  --ok: #0a7a32; --miss: #b91c1c; --muted: #555;
}}
* {{ box-sizing: border-box; }}
body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; background: var(--bg); color: #111; }}
.wrap {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}
.top {{ background: linear-gradient(135deg,#e8eeff,#f0f4ff); border: 1px solid var(--border);
  border-radius: 10px; padding: 20px 24px; margin-bottom: 28px; }}
.top h1 {{ margin: 0 0 8px; font-size: 1.35rem; }}
.top .meta {{ color: var(--muted); font-size: 14px; margin: 6px 0; }}
.prompt-top {{ margin-top: 16px; }}
.prompt-top summary {{ cursor: pointer; font-weight: 600; }}
.prompt-top pre {{
  white-space: pre-wrap; word-break: break-word;
  background: #1e1e2e; color: #e8e8f0; padding: 14px; border-radius: 8px;
  font-size: 12px; line-height: 1.45; max-height: 420px; overflow: auto;
}}
.stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 12px; }}
.stat {{ background: #fff; border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; }}
.stat b {{ font-size: 1.2rem; }}
.card {{
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 18px 20px; margin-bottom: 24px;
}}
.card.err {{ border-color: #f5c2c2; }}
.card-h {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
.card-h h3 {{ margin: 0; font-size: 1.1rem; }}
.idx {{ color: var(--muted); font-weight: normal; font-size: 0.85rem; }}
.verdict {{ font-weight: 700; padding: 4px 10px; border-radius: 6px; font-size: 0.9rem; }}
.verdict.ok {{ background: #d1fae5; color: var(--ok); }}
.verdict.miss {{ background: #fee2e2; color: var(--miss); }}
.verdict.dry {{ background: #eee; color: #666; }}
.gt-line, .verdict-line {{ font-size: 14px; margin: 8px 0; color: #333; }}
h4 {{ margin: 16px 0 8px; font-size: 0.95rem; color: #334; }}
.pair-input img {{ max-width: 100%; height: auto; border: 2px solid #333; border-radius: 4px; }}
.cap {{ font-size: 13px; color: var(--muted); margin: 8px 0 0; }}
.thumbs {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.thumb {{ flex: 1; min-width: 200px; max-width: 320px; border: 1px solid var(--border); border-radius: 8px; padding: 8px; }}
.thumb.gt {{ border-color: #86efac; background: #f0fdf4; }}
.thumb.dist {{ border-color: #fecaca; background: #fff5f5; }}
.thumb img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
.thumb .lbl {{ font-weight: 600; font-size: 13px; margin-bottom: 6px; }}
.prompt-box {{ margin-top: 12px; }}
.prompt-box pre {{
  white-space: pre-wrap; font-size: 11px; background: #f8f9fc; padding: 10px;
  border-radius: 6px; border: 1px solid var(--border); max-height: 280px; overflow: auto;
}}
.kv {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
.kv th {{ text-align: left; width: 120px; padding: 6px 8px; vertical-align: top; color: var(--muted); }}
.kv td {{ padding: 6px 8px; border-top: 1px solid #eee; }}
.errmsg {{ color: var(--miss); }}
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <h1>Pose pairwise VLM — review</h1>
    <p class="meta">Task: pick the side that is <b>more representative</b> of the cue (iconic static pose),
       not the better starting pose for follow-on motion.</p>
    <p class="meta">Model: <code>{_esc(data.get('model',''))}</code> ·
       Mode: <code>{_esc(data.get('mode',''))}</code> ·
       Rule: <code>{_esc(data.get('distractor_rule',''))}</code></p>
    <div class="stats">
      <div class="stat"><div>cues shown</div><b>{len(picked)}</b></div>
      <div class="stat"><div>1-pair accuracy</div><b>{acc*100:.1f}%</b> ({ok}/{len(scored)})</div>
    </div>
    <details class="prompt-top" open>
      <summary>VLM prompt template — representative pose (per trial: cue + LEFT/RIGHT labels filled in)</summary>
      <pre>{_esc(template_prompt)}</pre>
    </details>
  </header>
  {cards}
</div>
</body></html>"""

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(doc, encoding="utf-8")
    print(f"Wrote {OUT_HTML} ({len(picked)} cues)")


if __name__ == "__main__":
    main()
