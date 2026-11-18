#!/usr/bin/env python3
"""video_olm correction on 9 GT-handle negatives (3 rows × loc/ori/move)."""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_PAPER = _REPO / "papers/icra2027_motion_abstraction"
for p in (_REPO, _HERE, _PAPER):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from motion_code_to_config import convert_code  # noqa: E402
from run_embodiment_exp0_gemini import (  # noqa: E402
    _extract_json_obj,
    gif_path_for,
    render_pending_once,
)
from run_embodiment_regen_gemini import (  # noqa: E402
    _call_regen,
    _extract_revised_code,
    _parse_meta,
)
from run_gt_olm_negatives import KIND_LABEL, _cell_cfg, _merge_platform_cfgs, media_tag  # noqa: E402
from run_olm_state_regen_ablation import (  # noqa: E402
    _client,
    build_olm_prompt,
    format_sim_olm,
)
from score_humanoid_loc_dir_move import _move_keys, score_pair  # noqa: E402

NEG_JSON = (
    _REPO
    / "data/results/evaluation/embodiment_cue_id"
    / "results_gt_olm_negatives_pilot20"
    / "negatives.json"
)
NEG_RENDER = NEG_JSON.parent / "render"
OUT = NEG_JSON.parent / "video_olm_pilot9"
HTML_COPY = _PAPER / "olm_gt_negatives_pilot" / "correction_pilot9.html"

TARGETS = (
    ("tiago", "shh_gesture"),
    ("tiago", "cover_ears_too_loud"),
    ("humanoid", "stop_palm_out"),
)
DESCS = {
    "shh_gesture": "Hold a vertical finger or hand at the lips as if telling someone to be quiet.",
    "cover_ears_too_loud": "Cover both ears with the hands as if the sound is too loud.",
    "stop_palm_out": "Raise one hand with open palm facing outward toward the addressee and hold.",
}
LOC_OK = 10.0  # body-% max |Δ|


def _mp4(platform: str, idx: int, file_cue: str, kind: str) -> Path | None:
    gif = gif_path_for(NEG_RENDER / kind, platform, idx, file_cue)
    for p in (gif.with_suffix(".mp4"), gif):
        if p.is_file() and p.suffix.lower() == ".mp4":
            return p
    return gif.with_suffix(".mp4") if gif.with_suffix(".mp4").is_file() else None


def _flags(score: dict[str, Any], llm_row: dict[str, Any]) -> dict[str, Any]:
    loc = score.get("location") or {}
    direc = score.get("direction") or {}
    move = score.get("movement") or {}
    max_abs = loc.get("max_abs")
    loc_ok = isinstance(max_abs, (int, float)) and max_abs <= LOC_OK
    reach_ok = bool(direc.get("reach_ok"))
    palm_ok = bool(direc.get("palm_ok"))
    ori_ok = reach_ok and palm_ok
    gt_keys = move.get("gt_keys")
    if move.get("applicable"):
        move_ok = bool(move.get("ok"))
    else:
        llm_keys = _move_keys(llm_row)
        move_ok = not llm_keys
    return {
        "loc_ok": loc_ok,
        "ori_ok": ori_ok,
        "reach_ok": reach_ok,
        "palm_ok": palm_ok,
        "move_ok": move_ok,
        "l2": loc.get("l2"),
        "max_abs": max_abs,
        "reach_deg": direc.get("reach_angle_deg"),
        "palm_deg": direc.get("palm_angle_deg"),
        "gt_dir": direc.get("gt_dir"),
        "llm_dir": direc.get("llm_dir"),
        "gt_grip": direc.get("gt_grip"),
        "llm_grip": direc.get("llm_grip"),
        "gt_xyz": loc.get("gt_effective"),
        "llm_xyz": loc.get("llm_effective"),
        "gt_move": move.get("gt_keys") or list(_move_keys({"movements": []})),
        "llm_move": move.get("llm_keys") if move.get("applicable") else sorted(
            list(_move_keys(llm_row))
        ),
    }


def _target_ok(kind: str, flags: dict[str, Any]) -> bool:
    return {"loc": flags["loc_ok"], "ori": flags["ori_ok"], "move": flags["move_ok"]}[kind]


def write_html(jobs: list[dict[str, Any]], out_html: Path) -> None:
    import html as html_mod

    out_html.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    n_ok = sum(1 for j in jobs if j.get("target_ok"))
    for j in jobs:
        plat, cue, kind = j["platform"], j["cue"], j["kind"]
        idx = int(j["idx"])
        flags = j.get("flags") or {}
        gt_gif = gif_path_for(NEG_RENDER / "gt", plat, idx, cue)
        neg_gif = gif_path_for(NEG_RENDER / kind, plat, idx, f"{cue}__{kind}")
        corr_gif = gif_path_for(OUT / "render" / kind, plat, idx, f"{cue}__corr_{kind}")
        mark = "YES" if j.get("target_ok") else "NO"
        sections.append(
            f'<section class="row {"ok" if j.get("target_ok") else "bad"}">'
            f"<h2>{plat} · g{idx:02d} {html_mod.escape(cue)} · {kind}  "
            f'<span class="mark">{mark}</span></h2>'
            f'<p class="sum">{html_mod.escape((j.get("meta") or {}).get("change_summary") or "")}</p>'
            f'<p class="flags">loc {flags.get("max_abs")} (ok={flags.get("loc_ok")}) · '
            f'reach {flags.get("reach_deg")}° / palm {flags.get("palm_deg")}° '
            f'(ok={flags.get("ori_ok")}) · move ok={flags.get("move_ok")}</p>'
            f'<div class="grid">'
            f'<div class="col"><h3>GT</h3>{media_tag(gt_gif if gt_gif.is_file() else None, out_html.parent)}'
            f'<pre>{html_mod.escape(j.get("gt_code") or "")}</pre></div>'
            f'<div class="col"><h3>NEG {kind}</h3>{media_tag(neg_gif if neg_gif.is_file() else None, out_html.parent)}'
            f'<pre>{html_mod.escape(j.get("neg_code") or "")}</pre></div>'
            f'<div class="col"><h3>corrected (video_olm)</h3>{media_tag(corr_gif if corr_gif.is_file() else None, out_html.parent)}'
            f'<pre>{html_mod.escape(j.get("new_code") or j.get("error") or "")}</pre></div>'
            f"</div></section>"
        )
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>video_olm correction pilot 9</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:#111;color:#eee}}
h1{{padding:16px 24px;margin:0;border-bottom:1px solid #333}}
.meta{{padding:8px 24px;color:#9aa0a6}}
.row{{margin:12px 16px;padding:12px;border-radius:12px;border:1px solid #333;background:#1a1d24}}
.row.ok{{border-color:#3aa76d}}
.row.bad{{border-color:#c5221f}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}
img,video{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;background:#000;display:block}}
pre{{font-size:11px;white-space:pre-wrap;word-break:break-word;background:#0a0c10;padding:8px;border-radius:6px;max-height:110px;overflow:auto}}
.mark{{font-size:.85rem;padding:2px 8px;border-radius:999px;background:#333}}
.ok .mark{{background:#3aa76d;color:#041}}
.bad .mark{{background:#c5221f}}
.flags,.sum{{font-size:.85rem;color:#c4c7ce}}
</style></head>
<body>
<h1>video_olm correction · 9 negatives</h1>
<p class="meta">Target-handle recovery {n_ok}/9 · loc |Δ|max≤{LOC_OK:g}% · reach+palm ≤1° · movement keys cover GT (static GT → no extra axis travel)</p>
{"".join(sections)}
</body></html>
"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads(NEG_JSON.read_text(encoding="utf-8"))
    wanted = {t: None for t in TARGETS}
    for r in rows:
        key = (r["platform"], r["cue"])
        if key in wanted:
            wanted[key] = r
    missing = [k for k, v in wanted.items() if v is None]
    if missing:
        raise SystemExit(f"missing negatives.json rows: {missing}")

    pred_path = OUT / "predictions.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["job_id"]] = row

    jobs: list[dict[str, Any]] = []
    for key, row in wanted.items():
        plat, cue = key
        idx = int(row["idx"])
        gt_code = row["gt_code"]
        gt_conv = convert_code(gt_code, platform=plat)
        for kind in ("loc", "ori", "move"):
            job_id = f"{plat}.{cue}.{kind}"
            spec = {
                "job_id": job_id,
                "platform": plat,
                "cue": cue,
                "idx": idx,
                "kind": kind,
                "gt_code": gt_code,
                "neg_code": row["neg"][kind]["code"],
                "olm_text": row["neg"][kind]["olm"],
                "description": DESCS[cue],
                "file_cue": row["neg"][kind]["file_cue"],
            }
            jobs.append(spec)

    pending = [j for j in jobs if j["job_id"] not in done]
    print(f"pilot9 resume={len(done)} pending={len(pending)}", flush=True)
    client = _client() if pending else None
    for spec in pending:
        plat, cue, kind = spec["platform"], spec["cue"], spec["kind"]
        video = _mp4(plat, int(spec["idx"]), spec["file_cue"], kind)
        prompt = build_olm_prompt(
            platform=plat,
            cue=cue,
            description=spec["description"],
            motion_code=spec["neg_code"],
            olm_text=spec["olm_text"],
            include_video=True,
            include_code=True,
        )
        last_err = None
        for attempt in range(3):
            try:
                if video is None or not video.is_file():
                    raise FileNotFoundError(f"missing mp4 {spec['job_id']}")
                raw, usage = _call_regen(
                    client=client, model="gemini-2.5-pro", prompt=prompt, video=video
                )
                meta = _parse_meta(raw)
                new_code = _extract_revised_code(raw, _extract_json_obj(raw))
                conv = convert_code(new_code, platform=plat) if new_code else {
                    "movements": [],
                    "parse_ok": False,
                    "source_code": "",
                }
                row_out = {
                    **spec,
                    "new_code": conv.get("source_code") or new_code,
                    "movements": conv.get("movements") or [],
                    "parse_ok": bool(conv.get("parse_ok")),
                    "meta": meta,
                    "raw_output": raw,
                    "usage": usage,
                    "mp4": str(video),
                }
                with pred_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                done[spec["job_id"]] = row_out
                print(f"done {spec['job_id']} {row_out['new_code'][:90]}", flush=True)
                time.sleep(0.3)
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                time.sleep(min(2**attempt, 12))
        if last_err:
            row_out = {**spec, "error": last_err}
            with pred_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            done[spec["job_id"]] = row_out
            print(f"fail {spec['job_id']} {last_err}", flush=True)

    scored: list[dict[str, Any]] = []
    render_cfgs: dict[str, list[dict[str, Any]]] = {k: [] for k in ("loc", "ori", "move")}
    for spec in jobs:
        row = done[spec["job_id"]]
        plat = row["platform"]
        gt_conv = convert_code(row["gt_code"], platform=plat)
        llm_conv = {
            "movements": row.get("movements") or [],
            "source_code": row.get("new_code") or "",
        }
        if row.get("parse_ok"):
            score = score_pair(
                gt_conv, llm_conv, int(row["idx"]), row["cue"], platform=plat
            )
            flags = _flags(score, llm_conv)
            target_ok = _target_ok(row["kind"], flags)
        else:
            score, flags, target_ok = {}, {}, False
        packed = {
            **row,
            "score": score,
            "flags": flags,
            "target_ok": target_ok,
        }
        scored.append(packed)
        if row.get("parse_ok") and row.get("movements"):
            file_cue = f"{row['cue']}__corr_{row['kind']}"
            render_cfgs[row["kind"]].append(
                {
                    "idx": int(row["idx"]),
                    "cue": file_cue,
                    "category": "",
                    "description": row["cue"],
                    "platforms": {
                        plat: _cell_cfg(
                            idx=int(row["idx"]),
                            cue=file_cue,
                            platform=plat,
                            code=row["new_code"],
                            converted={
                                "movements": row["movements"],
                                "parse_ok": True,
                                "notes": [],
                                "source_code": row["new_code"],
                            },
                            experiment="video_olm_corr",
                        )
                    },
                }
            )

    (OUT / "scored.json").write_text(
        json.dumps(scored, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    for kind, items in render_cfgs.items():
        configs = _merge_platform_cfgs(items)
        if not configs:
            continue
        render_root = OUT / "render" / kind
        master_out = OUT / f"corr_{kind}.json"
        master_out.write_text(json.dumps(configs, indent=2, ensure_ascii=False) + "\n")
        render_pending_once._skip = set()  # type: ignore[attr-defined]
        for _ in range(12):
            n = render_pending_once(
                configs,
                render_root=render_root,
                out_json=master_out,
                platforms=["humanoid", "tiago"],
                force=False,
                max_items=6,
            )
            if n == 0:
                break

    write_html(scored, OUT / "gallery.html")
    write_html(scored, HTML_COPY)
    n_ok = sum(1 for j in scored if j.get("target_ok"))
    print(f"target-handle recovery {n_ok}/9", flush=True)
    print(f"html {HTML_COPY}", flush=True)
    for j in scored:
        f = j.get("flags") or {}
        print(
            f"{j['platform']:9} {j['cue']:26} {j['kind']:4} target={j.get('target_ok')} "
            f"loc={f.get('loc_ok')} ori={f.get('ori_ok')} move={f.get('move_ok')} "
            f"Δmax={f.get('max_abs')} rch={f.get('reach_deg')} plm={f.get('palm_deg')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
