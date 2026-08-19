#!/usr/bin/env python3
"""HTML from cues_component.yml: pose (gif + alpha traj + highlighted config) / movement blocks."""
from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import fire

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adhoc.generation.embodiment_sources import (  # noqa: E402
    motion_json_rel_list,
    repo_rel_to_path,
    robot_kw_to_embodiment_slug,
)
from adhoc.generation.see_html import (  # noqa: E402
    _ensure_cross_alpha_trajectory_png,
    _find_gif,
    _load_merged_json_rows,
    _rk_for_emb_slug,
    _rows_by_cue,
    _safe,
)
from adhoc.generation.utils import announce_output, dated_results_html_path  # noqa: E402
from adhoc.utils.repo_paths import dev_robosuite_root, results_subdir, seed_yml_dir


def _load_simple_yaml(path: Path) -> dict:
    """Minimal YAML loader for see_html_sources / cues_component (no PyYAML)."""
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]

    def cur() -> dict:
        return stack[-1][1]

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            text = line.strip()
            while len(stack) > 1 and indent <= stack[-1][0]:
                stack.pop()
            if text.endswith(":") and ":" not in text[:-1]:
                key = text[:-1].strip()
                nxt: dict = {}
                cur()[key] = nxt
                stack.append((indent, nxt))
                continue
            if ":" not in text:
                continue
            key, val = text.split(":", 1)
            key, val = key.strip(), val.strip().strip("'\"")
            if val == "":
                nxt = {}
                cur()[key] = nxt
                stack.append((indent, nxt))
            else:
                cur()[key] = val
    return root


def _emb_block(sources: dict, slug: str) -> dict:
    block = sources.get(slug)
    if not isinstance(block, dict):
        raise SystemExit(f"Missing embodiment {slug!r} in see_html_sources.yml")
    return block


def _load_shot_cues(emb_slug: str, block: dict, merged_rows: list[dict], root: Path) -> set[str]:
    """Canonical cue names listed in this embodiment's shots JSON."""
    rel = block.get("shots_json")
    if not rel:
        return set()
    p = repo_rel_to_path(str(rel), root)
    if not p.is_file():
        return set()
    with open(p, encoding="utf-8") as f:
        shots = json.load(f)
    if not isinstance(shots, list):
        return set()
    out: set[str] = set()
    if emb_slug == "manipulator":
        by_idx = {
            int(r.get("idx", -1)): str(r.get("cue", "")).strip()
            for r in merged_rows
            if r.get("cue") is not None and "idx" in r
        }
        for s in shots:
            if not isinstance(s, dict):
                continue
            if "idx" in s and int(s["idx"]) in by_idx:
                out.add(by_idx[int(s["idx"])])
                continue
            c = str(s.get("cue", "")).strip()
            if c and "_" in c and " " not in c:
                out.add(c)
    else:
        for s in shots:
            if isinstance(s, dict) and s.get("cue"):
                out.add(str(s["cue"]).strip())
    return out


def _load_sections() -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
    yml_path = seed_yml_dir() / "cues_component.yml"
    data = _load_simple_yaml(yml_path)
    out: list[tuple[str, list[tuple[str, dict[str, str]]]]] = []
    for sec_title, blob in data.items():
        if not isinstance(blob, dict):
            continue
        cues: list[tuple[str, dict[str, str]]] = []
        for cue_name, fields in blob.items():
            if not isinstance(fields, dict):
                continue
            meta = {
                k: str(fields[k]).strip()
                for k in ("pose", "movement", "essence")
                if fields.get(k) and str(fields[k]).strip()
            }
            if meta:
                cues.append((str(cue_name).strip(), meta))
        if cues:
            out.append((str(sec_title).strip(), cues))
    return out


def _sec_id(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title.strip()).strip("-").lower()
    return f"comp-sec-{slug or 'x'}"


def _mv_matches(mv_type: str, focus: str) -> bool:
    t = (mv_type or "").lower()
    return t == "pose" if focus == "pose" else t in ("path", "movement")


def _cfg_html(row: dict, focus: str) -> str:
    slim = {k: v for k, v in row.items() if k not in ("reasoning", "validation_warnings")}
    lines = ["{"]
    first = True
    for key, val in slim.items():
        if key == "movements":
            continue
        sep = "" if first else ","
        first = False
        chunk = json.dumps({key: val}, indent=2, ensure_ascii=False)[1:-1]
        lines.append(f"{sep}\n{html.escape(chunk)}")
    mvs = slim.get("movements")
    if isinstance(mvs, list) and mvs:
        sep = "" if first else ","
        lines.append(f'{sep}\n  "movements": [')
        for i, mv in enumerate(mvs):
            if not isinstance(mv, dict):
                continue
            body = "\n".join("    " + ln for ln in json.dumps(mv, indent=2, ensure_ascii=False).splitlines())
            comma = "," if i else ""
            if _mv_matches(str(mv.get("type", "")), focus):
                lines.append(f'{comma}\n<span class="hl-{focus}">{html.escape(body)}</span>')
            else:
                lines.append(f"{comma}\n{html.escape(body)}")
        lines.append("\n  ]")
    lines.append("\n}")
    return "".join(lines)


def _vis(focus: str, gif: Path | None, traj: Path | None, out_dir: Path) -> str:
    if not gif:
        return "<div class='na'>No gif</div>"
    rel = os.path.relpath(gif, out_dir)
    if focus == "pose":
        if traj and traj.is_file():
            tr = os.path.relpath(traj, out_dir)
            traj_h = f'<img src="{tr}" loading="lazy" alt="">'
        else:
            traj_h = "<div class='na'>alpha + traj n/a</div>"
        return (
            f'<div class="comp-vis"><figure><figcaption>gif</figcaption>'
            f'<img src="{rel}" loading="lazy"></figure>'
            f'<figure><figcaption>alpha + traj</figcaption>{traj_h}</figure></div>'
        )
    return (
        f'<div class="comp-vis single"><figure><figcaption>gif</figcaption>'
        f'<img src="{rel}" loading="lazy"></figure></div>'
    )


def _block(focus: str, hint: str, row: dict | None, gif: Path | None, traj: Path | None, out_dir: Path) -> str:
    cfg = (
        f'<pre class="cfg">{_cfg_html(row, focus)}</pre>'
        if row
        else '<pre class="cfg empty">— no config —</pre>'
    )
    return (
        f'<div class="comp-block comp-{focus}"><h4>{focus}</h4>'
        f'<p class="hint">{html.escape(hint)}</p>{_vis(focus, gif, traj, out_dir)}{cfg}</div>'
    )


def build(emb_slug: str, *, open_html: bool = True, skip_traj: bool = False) -> str:
    root = dev_robosuite_root()
    sources = _load_simple_yaml(seed_yml_dir() / "see_html_sources.yml")
    block = _emb_block(sources, emb_slug)
    label = str(block.get("label", emb_slug))
    config_paths = [repo_rel_to_path(r, root) for r in motion_json_rel_list(block["motion_config_json"])]
    gdir = repo_rel_to_path(str(block["render_dir"]), root)
    merged_rows = _load_merged_json_rows(config_paths)
    by_cue = _rows_by_cue(merged_rows)
    shot_cues = _load_shot_cues(emb_slug, block, merged_rows, root)
    shots_rel = str(block.get("shots_json", ""))
    rk = _rk_for_emb_slug(emb_slug)
    html_sub = str(block["see_html_subdir"])
    stem = f"{block['see_html_stem']}_component"
    out_dir = results_subdir("html") / html_sub
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = dated_results_html_path(html_sub, stem)
    sections = _load_sections()
    announce_output(out_dir, sum(len(c) for _, c in sections), kind="component html")

    filters, sec_html = [], []
    for sec_title, cues in sections:
        el = _sec_id(sec_title)
        filters.append(
            f'<label><input type="checkbox" checked data-target="{el}"> {html.escape(sec_title)}</label>'
        )
        rows_h = []
        for cue, meta in cues:
            row = by_cue.get(cue)
            idx = int(row.get("idx", -1)) if row else -1
            gif = _find_gif(rk, gdir, idx, cue, row)
            traj = None
            if row and "pose" in meta and not skip_traj:
                traj = _ensure_cross_alpha_trajectory_png(
                    emb_slug=emb_slug,
                    row=row,
                    cue=cue,
                    gif_path=gif,
                    config_paths=config_paths,
                    out_dir=out_dir,
                )
            parts = []
            if "pose" in meta:
                parts.append(_block("pose", meta["pose"], row, gif, traj, out_dir))
            if "movement" in meta:
                parts.append(_block("movement", meta["movement"], row, gif, None, out_dir))
            if not parts:
                parts.append(
                    f'<div class="comp-block"><p class="hint">{html.escape(meta.get("essence", ""))}</p>'
                    "<p class='na'>essence only</p></div>"
                )
            sub = f"c{idx}" if row else "—"
            cue_cls = "cue-name in-shot" if cue in shot_cues else "cue-name"
            rows_h.append(
                f'<div class="cue-row"><h3><span class="{cue_cls}">{html.escape(cue)}</span> '
                f'<span class="sub">{html.escape(sub)}</span></h3>{"".join(parts)}</div>'
            )
        sec_html.append(
            f'<section id="{el}"><details open><summary>{html.escape(sec_title)} ({len(cues)})</summary>'
            f'<div class="body">{"".join(rows_h)}</div></details></section>'
        )

    css = (
        "body{font-family:system-ui,sans-serif;padding:16px;background:#fafafa}"
        ".meta,.comp-filters{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px 14px;margin-bottom:14px;font-size:13px}"
        ".comp-filters{position:sticky;top:0;z-index:2;display:flex;flex-wrap:wrap;gap:8px}"
        "details{border:1px solid #ddd;border-radius:8px;margin-bottom:12px;background:#fff}"
        "summary{padding:10px 14px;font-weight:600;cursor:pointer;background:#f0f0f3}"
        ".body{padding:8px}.cue-row{border:1px solid #e5e5e5;border-radius:6px;padding:10px;margin:8px}"
        ".comp-block{border:1px solid #ececf0;border-radius:6px;padding:8px;margin:8px 0;background:#fafafa}"
        ".hint{font-size:12px;color:#444;margin:0 0 8px}.comp-vis{display:grid;grid-template-columns:1fr 1fr;gap:8px}"
        ".comp-vis.single{grid-template-columns:1fr;max-width:480px} img{width:100%;border-radius:4px}"
        "pre.cfg{font-size:10px;white-space:pre-wrap;background:#f4f4f5;padding:8px;max-height:260px;overflow:auto}"
        ".hl-pose{background:#dff0ff;display:block;padding:2px 4px}.hl-movement{background:#fff3cd;display:block;padding:2px 4px}"
        ".na{color:#999;font-style:italic;text-align:center;padding:12px;font-size:12px} h4{margin:0 0 6px;text-transform:uppercase;font-size:11px}"
        ".cue-name{font-weight:600;color:#222}.cue-name.in-shot{color:#888}"
        ".sub{font-size:11px;color:#888;font-weight:normal}"
    )
    script = """<script>(function(){function s(c){var e=document.getElementById(c.getAttribute("data-target"));if(e)e.style.display=c.checked?"":"none";}
document.querySelectorAll(".comp-filters input").forEach(function(c){c.onchange=function(){s(this);};});
document.querySelector("[data-all]")?.addEventListener("click",function(){document.querySelectorAll(".comp-filters input").forEach(function(c){c.checked=true;s(c);});});
document.querySelector("[data-none]")?.addEventListener("click",function(){document.querySelectorAll(".comp-filters input").forEach(function(c){c.checked=false;s(c);});});
})();</script>"""
    yml_rel = (seed_yml_dir() / "cues_component.yml").relative_to(root)
    page = (
        f"<!doctype html><html><head><meta charset=utf-8><title>{html.escape(label)} component</title>"
        f"<style>{css}</style></head><body>"
        f"<h1>{html.escape(label)} · cues_component.yml</h1>"
        f'<div class="meta">YAML <code>{html.escape(str(yml_rel))}</code> · '
        f"shots <code>{html.escape(shots_rel)}</code> ({len(shot_cues)} cues, gray name) · "
        f'<code>{html_path.relative_to(root)}</code></div>'
        f'<div class="comp-filters"><button data-all>All</button><button data-none>None</button>'
        f'{"".join(filters)}</div>{"".join(sec_html)}{script}</body></html>'
    )
    html_path.write_text(page, encoding="utf-8")
    print(html_path)
    if open_html and sys.platform == "darwin":
        subprocess.Popen(["open", str(html_path)])
    return str(html_path)


def run(
    robots: str | list | tuple = "manipulator,bimanual",
    open_html: bool = True,
    skip_traj: bool = False,
) -> str:
    if isinstance(robots, (list, tuple)):
        tokens = [str(t).strip() for t in robots if str(t).strip()]
    else:
        tokens = [t.strip() for t in str(robots).split(",") if t.strip()]
    out: list[str] = []
    for tok in tokens:
        slug = robot_kw_to_embodiment_slug(tok)
        out.append(build(slug, open_html=open_html, skip_traj=skip_traj))
    return out[-1] if len(out) == 1 else "\n".join(out)


if __name__ == "__main__":
    fire.Fire(run)
