#!/usr/bin/env python3
"""Convert 5 easy→hard cues from _cues.yml using tiago/shot_configs_19_mobile.json."""
from __future__ import annotations

import html as H
import json
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_HERE))

from build_embodiment_unified_shots import (  # noqa: E402
    PLATS,
    _add,
    _movements_to_dsl_approx,
    enrich_and_render,
)

SRC = _REPO / "data/seed/shots/tiago/shot_configs_19_mobile.json"
OUT = _REPO / "data/results/shots/embodiment_unified"
SEED = _REPO / "data/seed/shots/embodiment_unified"
CUES30 = {
    c["cue"]
    for c in json.loads((_REPO / "data/seed/agent_fill/embodiment_short/cues.json").read_text())
}

SELECTION = [
    {
        "rank": 1,
        "difficulty": "Easy",
        "cue": "raising_hand_greeting",
        "why": "단순 올리기 — pose + short lift (hold-ish)",
    },
    {
        "rank": 2,
        "difficulty": "Easy–Mid",
        "cue": "wave_hi",
        "why": "반복 측방 — raised pose + EE y oscillation",
    },
    {
        "rank": 3,
        "difficulty": "Mid",
        "cue": "bow_greeting",
        "why": "몸통 굴신 — torso height movement",
    },
    {
        "rank": 4,
        "difficulty": "Mid–Hard",
        "cue": "blow_kiss",
        "why": "얼굴→전방 릴리스 — face-band pose + forward puff path",
    },
    {
        "rank": 5,
        "difficulty": "Hard",
        "cue": "circle_temple_crazy",
        "why": "정밀 path figure — temple-local arc sweep",
    },
]

PLAT_LAB = {
    "humanoid": "Humanoid",
    "tiago": "Tiago",
    "stretch": "Stretch",
    "single_arm": "Single Arm",
    "roomba": "Roomba",
}


def slim(e: dict) -> dict:
    keep = {
        "cue",
        "shot_id",
        "role",
        "motion_code",
        "xyzrpy",
        "joint_angles",
        "render_gif",
        "source",
        "notes",
        "parse_ok",
        "extract_ok",
        "render_ok",
        "hz",
        "extract_error",
        "render_error",
    }
    return {k: e[k] for k in keep if k in e}


def write_gallery(bag: dict) -> Path:
    idx = {p: {e["cue"]: e for e in bag[p]} for p in PLATS}
    rows_html = []
    for s in SELECTION:
        cells = []
        for plat in PLATS:
            e = idx[plat][s["cue"]]
            gif = e.get("render_gif") or ""
            gif_url = ("../../../../" + gif) if gif else ""
            mc = H.escape((e.get("motion_code") or "")[:180])
            ok = "ok" if e.get("render_ok") else "fail"
            img = (
                f'<img src="{H.escape(gif_url)}" alt="{H.escape(s["cue"])}"/>'
                if gif_url
                else "<div class='miss'>no gif</div>"
            )
            cells.append(
                f"<td class='{ok}'><div class='plat'>{PLAT_LAB[plat]}</div>{img}"
                f"<pre class='mc'>{mc}</pre></td>"
            )
        rows_html.append(
            f"<tr><th><div class='rank'>#{s['rank']} · {H.escape(s['difficulty'])}</div>"
            f"<div class='cue'>{H.escape(s['cue'])}</div>"
            f"<div class='why'>{H.escape(s['why'])}</div></th>{''.join(cells)}</tr>"
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>Unified shots · _cues.yml easy→hard (outside 30)</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:24px;background:#0f1115;color:#e8eaed}}
h1{{font-size:1.25rem;margin:0 0 6px}}
.sub{{color:#9aa0a6;margin:0 0 18px;font-size:.9rem;max-width:90ch}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #2a2f3a;padding:8px;vertical-align:top;width:18%}}
th:first-child{{width:14%;text-align:left;background:#161a22}}
img{{width:100%;max-width:220px;background:#000;border-radius:4px;display:block}}
.plat{{font-size:.75rem;color:#9aa0a6;margin-bottom:4px}}
.cue{{font-family:ui-monospace,monospace;font-size:.95rem;margin:4px 0}}
.rank{{color:#8ab4f8;font-size:.8rem}}
.why{{color:#9aa0a6;font-size:.75rem;margin-top:6px}}
.mc{{font-size:.65rem;white-space:pre-wrap;color:#bdc1c6;margin:6px 0 0;max-height:4.8em;overflow:auto}}
.fail{{outline:1px solid #c5221f}}
.miss{{color:#c5221f;padding:40px 0;text-align:center}}
code{{background:#1e2430;padding:1px 4px;border-radius:3px}}
</style></head><body>
<h1>5 columns × 5 cues from <code>data/seed/yml/_cues.yml</code> (easy → hard)</h1>
<p class="sub">Source converted: <code>{SRC.relative_to(_REPO)}</code> · outside the Fig1 30-cue set ·
Motion Code / xyzrpy / joint_angles → <code>shot_cues_yml_5.json</code>.</p>
<table>
<thead><tr><th>Cue</th>{''.join(f'<th>{PLAT_LAB[p]}</th>' for p in PLATS)}</tr></thead>
<tbody>
{''.join(rows_html)}
</tbody></table>
</body></html>
"""
    gpath = OUT / "gallery_5col_cues_yml.html"
    gpath.write_text(html)
    return gpath


def main() -> None:
    tiago = {r["cue"]: r for r in json.loads(SRC.read_text())}
    for s in SELECTION:
        assert s["cue"] in tiago, s["cue"]
        assert s["cue"] not in CUES30, s["cue"]

    bag = {p: [] for p in PLATS}
    for s in SELECTION:
        row = tiago[s["cue"]]
        code = _movements_to_dsl_approx(row.get("movements") or [])
        print(f"[dsl] {s['cue']}: {code}", flush=True)
        for plat in PLATS:
            _add(
                bag,
                platform=plat,
                cue=s["cue"],
                motion_code=code,
                source=str(SRC.relative_to(_REPO)),
                role=f"cues_yml_e2h_r{s['rank']}",
                shot_id=f"{plat}__{s['cue']}__cues_yml",
                notes=s["why"],
            )

    # Skip existing GIFs (force=False)
    print("[enrich] start", flush=True)
    bag = enrich_and_render(bag, force=False)
    print("[enrich] done", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    SEED.mkdir(parents=True, exist_ok=True)

    focus = {p: [slim(e) for e in bag[p]] for p in PLATS}
    (OUT / "shot_cues_yml_5.json").write_text(json.dumps(focus, indent=2) + "\n")
    (SEED / "shot_cues_yml_5.yaml").write_text(
        yaml.safe_dump(focus, sort_keys=False, allow_unicode=True)
    )
    (OUT / "gallery_5col_cues_yml_selection.json").write_text(
        json.dumps(SELECTION, indent=2) + "\n"
    )

    # merge into unified shot.json
    existing = OUT / "shot.json"
    if existing.is_file():
        old = json.loads(existing.read_text())
    else:
        old = {p: [] for p in PLATS}
    merged = {}
    for plat in PLATS:
        by_id = {e["shot_id"]: e for e in old.get(plat, []) if isinstance(e, dict) and "shot_id" in e}
        # old may have meta key when written by write_yaml
        for e in bag[plat]:
            by_id[e["shot_id"]] = slim(e)
        merged[plat] = list(by_id.values())
    # preserve meta if present
    out_obj = {"meta": old.get("meta")} if isinstance(old, dict) and "meta" in old else {}
    out_obj.update(merged)
    (OUT / "shot.json").write_text(json.dumps(out_obj, indent=2, ensure_ascii=False) + "\n")
    (SEED / "shot.yaml").write_text(yaml.safe_dump(out_obj, sort_keys=False, allow_unicode=True))

    gpath = write_gallery(bag)
    print(f"wrote {gpath}", flush=True)

    n_ok = sum(1 for p in PLATS for e in bag[p] if e.get("render_ok"))
    n_ex = sum(1 for p in PLATS for e in bag[p] if e.get("extract_ok"))
    print(f"render_ok {n_ok}/25 extract_ok {n_ex}/25", flush=True)
    for s in SELECTION:
        for plat in PLATS:
            e = next(x for x in bag[plat] if x["cue"] == s["cue"])
            print(
                f"  {plat}/{s['cue']}: r={e.get('render_ok')} e={e.get('extract_ok')} "
                f"{e.get('render_error') or e.get('extract_error') or ''}",
                flush=True,
            )


if __name__ == "__main__":
    main()
