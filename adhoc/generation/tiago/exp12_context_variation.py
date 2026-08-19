#!/usr/bin/env python3
"""Exp12 Tiago — context variation (13 shot cues × 20 contexts)."""
from __future__ import annotations

import argparse
import copy
import html
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pilot40_paths import (  # noqa: E402
    CONTEXT_VARIATION_INPUT_HTML,
    CONTEXT_VARIATION_OUT,
    CONTEXT_VARIATION_YML,
    SHOT_CUE_SET,
    prompt_exp_path,
)

PROMPT_EXP12 = prompt_exp_path(12)


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML required: pip install pyyaml")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_context_catalog(path: Path | None = None) -> dict[str, dict[str, str]]:
    data = _load_yaml(path or CONTEXT_VARIATION_YML)
    out: dict[str, dict[str, str]] = {}
    for key, desc in (data.get("personas") or {}).items():
        out[str(key)] = {"type": "persona", "desc": str(desc).strip()}
    for key, desc in (data.get("scenarios") or {}).items():
        out[str(key)] = {"type": "scenario", "desc": str(desc).strip()}
    return out


def load_run_pairs(*, catalog_yml: Path | None = None) -> list[tuple[int, str, str, str]]:
    path = catalog_yml or CONTEXT_VARIATION_YML
    data = _load_yaml(path)
    catalog = load_context_catalog(path)
    pairs: list[tuple[int, str, str, str]] = []
    for row in data.get("runs") or []:
        idx = int(row["idx"])
        for key in row.get("contexts") or []:
            ckey = str(key)
            if ckey not in catalog:
                raise KeyError(f"context {ckey!r} missing from catalog in {path}")
            meta = catalog[ckey]
            pairs.append((idx, ckey, meta["type"], meta["desc"]))
    return pairs


def load_shot_bases(path: Path | None = None) -> dict[int, dict[str, Any]]:
    sp = path or SHOT_CUE_SET
    rows = json.loads(sp.read_text(encoding="utf-8"))
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        idx = int(row["idx"])
        base = {k: row[k] for k in row if k not in ("time", "model", "validation_warnings")}
        base["cue_slug"] = str(row.get("cue") or f"shot_{idx}")
        base["base_source"] = "tiago_shots"
        out[idx] = base
    return out


def base_for_idx(idx: int, *, shot_bases: dict[int, dict[str, Any]] | None = None) -> dict[str, Any] | None:
    shots = shot_bases if shot_bases is not None else load_shot_bases()
    if idx in shots:
        return copy.deepcopy(shots[idx])
    return None


def _gif_name(idx: int, base: dict[str, Any], context_key: str) -> str:
    slug = str(base.get("cue_slug") or base.get("cue") or f"shot_{idx}")
    return f"c{idx}_{slug}_{context_key}.gif"


def _fill_prompt(*, cue: str, context_name: str, context_desc: str, base: dict[str, Any]) -> str:
    tpl = PROMPT_EXP12.read_text(encoding="utf-8")
    return (
        tpl.replace("{{CUE}}", cue)
        .replace("{{CONTEXT_NAME}}", context_name)
        .replace("{{CONTEXT_DESC}}", context_desc)
        .replace("{{BASE_JSON}}", json.dumps(base, indent=2, ensure_ascii=False))
    )


def _extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("no JSON in response")
    raw = m.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*([\]}])", r"\1", raw)
        return json.loads(fixed)


def _gemini_context_variation(
    base: dict[str, Any],
    context_key: str,
    context_type: str,
    context_desc: str,
    model: str,
    *,
    max_retries: int = 3,
) -> dict[str, Any]:
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    prompt = _fill_prompt(
        cue=str(base.get("cue", "")),
        context_name=context_key,
        context_desc=context_desc,
        base=base,
    )
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            text = resp.text or ""
            out = _extract_json(text)
            out["idx"] = base.get("idx")
            out["cue"] = base.get("cue")
            if base.get("cue_slug"):
                out["cue_slug"] = base.get("cue_slug")
            out["context"] = context_key
            out["context_type"] = context_type
            out["context_desc"] = context_desc
            out["base_model"] = model
            out["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return out
        except Exception as e:
            last_err = e
            print(f"    [retry {attempt + 1}/{max_retries}] {e}", flush=True)
    raise RuntimeError(f"Gemini context variation failed: {last_err}")


def _render_gif(cfg: dict[str, Any], out_gif: Path, *, force: bool = False) -> bool:
    if out_gif.is_file() and not force:
        return True
    if force and out_gif.is_file():
        out_gif.unlink()
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    ctx = str(cfg.get("context", "ctx"))
    render_cfg = dict(cfg)
    render_cfg["persona"] = ctx
    try:
        from adhoc.generation.tiago.legacy import (  # noqa: WPS433
            _make_env,
            render_config,
        )

        env = _make_env()
        try:
            frames = render_config(render_cfg, env=env)
            if not frames:
                return False
            frames[0].save(
                out_gif,
                save_all=True,
                append_images=frames[1:],
                duration=50,
                loop=0,
            )
            return out_gif.is_file()
        finally:
            env.close()
    except Exception as e:
        print(f"[render fail] {cfg.get('cue')}:{ctx}: {e}", flush=True)
        return False


def _summarize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    speeds: list[float] = []
    holds: list[float] = []
    n_steps = len(cfg.get("movements") or [])
    for step in cfg.get("movements") or []:
        p = step.get("parameters") or {}
        if isinstance(p.get("speed"), (int, float)):
            speeds.append(float(p["speed"]))
        path = p.get("path") or {}
        if isinstance(path.get("speed"), (int, float)):
            speeds.append(float(path["speed"]))
        for j in p.get("joints") or []:
            if isinstance(j.get("speed"), (int, float)):
                speeds.append(float(j["speed"]))
            if isinstance(j.get("hold_time"), (int, float)):
                holds.append(float(j["hold_time"]))
    return {
        "n_steps": n_steps,
        "speed_min": min(speeds) if speeds else None,
        "speed_max": max(speeds) if speeds else None,
        "hold_sum": round(sum(holds), 2) if holds else None,
    }


def build_review_html(items: list[tuple[dict, Path | None]], out_html: Path) -> None:
    by_cue: dict[tuple[int, str], list[tuple[dict, Path | None]]] = {}
    for cfg, gif in items:
        key = (int(cfg["idx"]), str(cfg.get("cue", "")))
        by_cue.setdefault(key, []).append((cfg, gif))

    sections: list[str] = []
    for (idx, cue), group in sorted(by_cue.items(), key=lambda x: x[0][0]):
        cards = []
        for cfg, gif in sorted(group, key=lambda x: str(x[0].get("context", ""))):
            ctx = html.escape(str(cfg.get("context", "?")))
            ctype = html.escape(str(cfg.get("context_type", "")))
            summary = _summarize_config(cfg)
            media = (
                f'<img class="gif" src="{html.escape(gif.name)}" loading="lazy">'
                if gif and gif.is_file()
                else '<div class="missing">no gif</div>'
            )
            cards.append(
                f"""<article class="card">
  <h4>{ctx} <span class="tag {ctype}">{ctype}</span></h4>
  <p class="desc">{html.escape(str(cfg.get('context_desc',''))[:140])}</p>
  <p class="meta">steps={summary['n_steps']} speed={summary['speed_min']}–{summary['speed_max']} holdΣ={summary['hold_sum']}</p>
  {media}
</article>"""
            )
        sections.append(
            f"""<section id="c{idx}">
  <h2>{idx}. {html.escape(cue)}</h2>
  <div class="grid">{''.join(cards)}</div>
</section>"""
        )

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Exp12 Tiago — context variation</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1400px;margin:20px auto;padding:0 16px 48px;background:#fafafa}}
h1{{font-size:22px}} h2{{font-size:17px;margin-top:32px}}
.summary{{background:#fff;border:1px solid #ccc;border-radius:10px;padding:14px;margin:16px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:10px;padding:10px}}
.card h4{{margin:0 0 6px;font-size:13px}}
.tag{{font-size:10px;padding:2px 6px;border-radius:4px;background:#eee}}
.tag.persona{{background:#e3f2fd;color:#1565c0}}
.tag.scenario{{background:#fff3e0;color:#e65100}}
.desc{{font-size:11px;color:#555;margin:0 0 6px}}
.meta{{font-size:10px;color:#777;margin:0 0 8px}}
.gif{{width:100%;border-radius:6px;border:1px solid #ccc}}
.missing{{color:#999;font-style:italic;font-size:12px;padding:24px 0;text-align:center}}
</style></head><body>
<h1>Exp12 — Tiago context variation</h1>
<div class="summary">
  <p>{len(items)} jobs · queue=<code>{html.escape(str(SHOT_CUE_SET.name))}</code></p>
  <p>Output: <code>{html.escape(str(CONTEXT_VARIATION_OUT.relative_to(_REPO)))}</code></p>
</div>
{''.join(sections)}
</body></html>"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(doc, encoding="utf-8")


def build_input_html(*, catalog_yml: Path | None = None, out_html: Path | None = None) -> Path:
    path = catalog_yml or CONTEXT_VARIATION_YML
    catalog = load_context_catalog(path)
    pairs = load_run_pairs(catalog_yml=path)
    shot_bases = load_shot_bases()
    out = out_html or CONTEXT_VARIATION_INPUT_HTML
    out.parent.mkdir(parents=True, exist_ok=True)

    n_cues = len(shot_bases)
    n_ctx = len(catalog)
    n_jobs = len(pairs)

    run_rows = []
    for idx, ckey, ctype, cdesc in pairs:
        base = base_for_idx(idx, shot_bases=shot_bases) or {}
        cue = str(base.get("cue", "?"))
        run_rows.append(
            f"<tr><td>c{idx}</td><td><code>{html.escape(cue)}</code></td>"
            f"<td><code>{html.escape(ckey)}</code></td><td>{html.escape(ctype)}</td>"
            f"<td class='desc'>{html.escape(cdesc[:100])}</td></tr>"
        )

    cue_rows = []
    for idx in sorted(shot_bases):
        b = shot_bases[idx]
        cue_rows.append(
            f"<tr><td>{idx}</td><td><code>{html.escape(str(b.get('cue','?')))}</code></td>"
            f"<td>{html.escape(str(b.get('cue_text') or b.get('description',''))[:80])}</td></tr>"
        )

    yml_rel = os.path.relpath(path.resolve(), _REPO)
    shots_rel = os.path.relpath(SHOT_CUE_SET.resolve(), _REPO)
    out_rel = os.path.relpath(CONTEXT_VARIATION_OUT.resolve(), _REPO)
    cmd = "python adhoc/generation/tiago/exp12_context_variation.py --model gemini-2.5-pro"

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Exp12 Tiago — input plan</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:20px auto;padding:0 16px 48px;background:#fafafa}}
h1{{font-size:22px}} h2{{font-size:17px;margin-top:28px}}
.summary{{background:#fff;border:1px solid #ccc;border-radius:10px;padding:14px;margin:16px 0;font-size:15px}}
table{{border-collapse:collapse;width:100%;font-size:13px;background:#fff;margin:10px 0}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#f0f0f0}}
td.desc{{color:#444;font-size:12px}}
pre.cmd{{background:#1e1e2e;color:#e8e8f0;padding:12px;border-radius:8px;font-size:12px}}
.num{{font-size:28px;font-weight:700;color:#0969da}}
</style></head><body>
<h1>Exp12 — Tiago context variation</h1>
<div class="summary">
  <p><span class="num">{n_cues}</span> shot cues × <span class="num">{n_ctx}</span> contexts =
    <span class="num">{n_jobs}</span> jobs (Gemini config + GIF each)</p>
  <p><b>Cue queue:</b> <code>{html.escape(shots_rel)}</code></p>
  <p><b>Catalog:</b> <code>{html.escape(yml_rel)}</code> (10 personas + 10 scenarios)</p>
  <p><b>Output:</b> <code>{html.escape(out_rel)}</code></p>
</div>
<h2>How to run</h2>
<pre class="cmd">source APIKEY.sh
{html.escape(cmd)}</pre>
<h2>Cue queue ({n_cues})</h2>
<table><thead><tr><th>idx</th><th>cue</th><th>description</th></tr></thead><tbody>{''.join(cue_rows)}</tbody></table>
<h2>Run matrix ({n_jobs} jobs)</h2>
<table><thead><tr><th>idx</th><th>cue</th><th>context</th><th>type</th><th>description</th></tr></thead><tbody>{''.join(run_rows)}</tbody></table>
</body></html>"""
    out.write_text(doc, encoding="utf-8")
    print(f"Wrote input HTML → {out}")
    return out


def run(
    *,
    catalog_yml: str | None = None,
    model: str = "gemini-2.5-pro",
    dry_run: bool = False,
    skip_render: bool = False,
    force_render: bool = False,
    html_only: bool = False,
    fresh: bool = False,
    out_dir: str | None = None,
) -> Path:
    yml_path = Path(catalog_yml) if catalog_yml else CONTEXT_VARIATION_YML
    if html_only and not force_render:
        build_input_html(catalog_yml=yml_path)
        od = Path(out_dir) if out_dir else CONTEXT_VARIATION_OUT
        cfg_out = od / "context_configs.json"
        if not cfg_out.is_file():
            return CONTEXT_VARIATION_INPUT_HTML
        configs = json.loads(cfg_out.read_text(encoding="utf-8"))
        gif_dir = od / "gifs"
        items = []
        for cfg in configs:
            gif = gif_dir / _gif_name(int(cfg["idx"]), cfg, str(cfg.get("context", "")))
            items.append((cfg, gif if gif.is_file() else None))
        review = od / "context_variations.html"
        build_review_html(items, review)
        return review

    pairs = load_run_pairs(catalog_yml=yml_path)
    shot_bases = load_shot_bases()
    od = Path(out_dir) if out_dir else CONTEXT_VARIATION_OUT
    if fresh:
        cfg_p = od / "context_configs.json"
        if cfg_p.is_file():
            cfg_p.unlink()
        gif_d = od / "gifs"
        if gif_d.is_dir():
            for g in gif_d.glob("*.gif"):
                g.unlink()

    cfg_out = od / "context_configs.json"
    gif_dir = od / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    configs: list[dict[str, Any]] = []
    if cfg_out.is_file():
        try:
            configs = json.loads(cfg_out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            configs = []
    done_keys = {(int(c["idx"]), c.get("context")) for c in configs if c.get("idx") is not None}

    render_items: list[tuple[dict, Path | None]] = []
    print(
        f"[exp12-mobile] cues={len(shot_bases)} contexts={len(load_context_catalog(yml_path))} "
        f"jobs={len(pairs)} → {od}",
        flush=True,
    )

    for idx, ckey, ctype, cdesc in pairs:
        base = base_for_idx(idx, shot_bases=shot_bases)
        if not base:
            print(f"[skip] idx {idx} not in {SHOT_CUE_SET.name}", flush=True)
            continue
        gif_path = gif_dir / _gif_name(idx, base, ckey)
        if (idx, ckey) in done_keys and gif_path.is_file() and not force_render:
            cfg = next(c for c in configs if int(c["idx"]) == idx and c.get("context") == ckey)
            render_items.append((cfg, gif_path))
            continue
        if (idx, ckey) not in done_keys:
            print(f"  {ctype} {ckey} @ c{idx} {base.get('cue')}", flush=True)
            if dry_run:
                cfg = dict(base)
                cfg["context"] = ckey
                cfg["context_type"] = ctype
                cfg["context_desc"] = cdesc
            else:
                cfg = _gemini_context_variation(base, ckey, ctype, cdesc, model)
            configs.append(cfg)
            done_keys.add((idx, ckey))
            cfg_out.write_text(json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            cfg = next(c for c in configs if int(c["idx"]) == idx and c.get("context") == ckey)

        if not skip_render:
            ok = _render_gif(cfg, gif_path, force=force_render)
            render_items.append((cfg, gif_path if ok else None))
        else:
            render_items.append((cfg, gif_path if gif_path.is_file() else None))

    review = od / "context_variations.html"
    build_review_html(render_items, review)
    print(f"Wrote {review}")
    print(f"Configs: {cfg_out} ({len(configs)} entries)")
    return review


def main() -> None:
    p = argparse.ArgumentParser(description="Exp12 Tiago context variation")
    p.add_argument("--catalog-yml", default=None)
    p.add_argument("--model", default="gemini-2.5-pro")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-render", action="store_true")
    p.add_argument("--force-render", action="store_true")
    p.add_argument("--html-only", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--open", action="store_true")
    args = p.parse_args()

    if args.html_only and not (Path(args.out_dir or CONTEXT_VARIATION_OUT) / "context_configs.json").is_file():
        out = build_input_html(catalog_yml=Path(args.catalog_yml) if args.catalog_yml else None)
    else:
        out = run(
            catalog_yml=args.catalog_yml,
            model=args.model,
            dry_run=args.dry_run,
            skip_render=args.skip_render,
            force_render=args.force_render,
            html_only=args.html_only,
            fresh=args.fresh,
            out_dir=args.out_dir,
        )
    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    main()
