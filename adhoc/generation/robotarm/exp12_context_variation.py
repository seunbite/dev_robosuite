#!/usr/bin/env python3
"""Exp12 — context variation (10 personas + 10 scenarios, scenario-only prompts)."""
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

from paper_figures._media import cfg_by_idx, repo  # noqa: E402
from paper_figures.persona_change import (  # noqa: E402
    _extract_json,
    _render_gif,
    build_html as build_review_html,
    summarize_config,
)
from pilot90_paths import (  # noqa: E402
    CONTEXT_VARIATION_INPUT_HTML,
    CONTEXT_VARIATION_OUT,
    CONTEXT_VARIATION_YML,
    SHOTS,
    prompt_exp_path,
)

PROMPT_EXP12 = prompt_exp_path(12)

_SHOT_SLUGS: dict[int, str] = {
    0: "wave_hello",
    1: "raise_brief",
    2: "bow",
    3: "salute",
    4: "crazy_temple",
    5: "rub_eye",
    6: "tilt_curious",
}


def load_shot_bases(path: Path | None = None) -> dict[int, dict[str, Any]]:
    """Few-shot shot_configs entries keyed by shot idx (not pilot-90 idx)."""
    sp = path or SHOTS
    rows = json.loads(sp.read_text(encoding="utf-8"))
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        idx = int(row["idx"])
        base = {k: row[k] for k in ("idx", "cue", "description", "movements", "state") if k in row}
        base["cue_slug"] = _SHOT_SLUGS.get(idx, f"shot_{idx}")
        base["base_source"] = "shots"
        out[idx] = base
    return out


def base_for_idx(idx: int, *, shot_bases: dict[int, dict[str, Any]] | None = None) -> dict[str, Any] | None:
    shots = shot_bases if shot_bases is not None else load_shot_bases()
    if idx in shots:
        return copy.deepcopy(shots[idx])
    return cfg_by_idx().get(idx)


def _gif_name(idx: int, base: dict[str, Any], context_key: str) -> str:
    slug = str(base.get("cue_slug") or re.sub(r"[^\w]+", "_", str(base.get("cue", "cue"))).strip("_")[:48])
    return f"c{idx}_{slug}_{context_key}.gif"


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML required: pip install pyyaml")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_context_catalog(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Return {key: {type, desc}} for all 20 contexts."""
    data = _load_yaml(path or CONTEXT_VARIATION_YML)
    out: dict[str, dict[str, str]] = {}
    for key, desc in (data.get("personas") or {}).items():
        out[str(key)] = {"type": "persona", "desc": str(desc).strip()}
    for key, desc in (data.get("scenarios") or {}).items():
        out[str(key)] = {"type": "scenario", "desc": str(desc).strip()}
    return out


def load_run_pairs(*, catalog_yml: Path | None = None) -> list[tuple[int, str, str, str]]:
    """Return [(cue_idx, context_key, context_type, context_desc), ...]."""
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


def _fill_prompt(*, cue: str, context_name: str, context_desc: str, base: dict[str, Any]) -> str:
    tpl = PROMPT_EXP12.read_text(encoding="utf-8")
    return (
        tpl.replace("{{CUE}}", cue)
        .replace("{{CONTEXT_NAME}}", context_name)
        .replace("{{CONTEXT_DESC}}", context_desc)
        .replace("{{BASE_JSON}}", json.dumps(base, indent=2, ensure_ascii=False))
    )


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


def build_input_html(
    *,
    catalog_yml: Path | None = None,
    out_html: Path | None = None,
) -> Path:
    """HTML plan: 20 contexts, run matrix, shell commands (no generation)."""
    path = catalog_yml or CONTEXT_VARIATION_YML
    catalog = load_context_catalog(path)
    pairs = load_run_pairs(catalog_yml=path)
    shot_bases = load_shot_bases()
    out = out_html or CONTEXT_VARIATION_INPUT_HTML
    out.parent.mkdir(parents=True, exist_ok=True)

    persona_rows = []
    scenario_rows = []
    for key, meta in sorted(catalog.items()):
        row = f"<tr><td><code>{html.escape(key)}</code></td><td>{html.escape(meta['type'])}</td><td>{html.escape(meta['desc'])}</td></tr>"
        if meta["type"] == "persona":
            persona_rows.append(row)
        else:
            scenario_rows.append(row)

    run_rows = []
    for idx, ckey, ctype, cdesc in pairs:
        base = base_for_idx(idx, shot_bases=shot_bases) or {}
        cue = str(base.get("cue", "?"))
        run_rows.append(
            f"<tr><td>c{idx}</td><td><code>{html.escape(cue)}</code></td>"
            f"<td><code>{html.escape(ckey)}</code></td><td>{html.escape(ctype)}</td>"
            f"<td class='desc'>{html.escape(cdesc[:120])}</td></tr>"
        )

    yml_rel = os.path.relpath(path.resolve(), _REPO)
    out_rel = os.path.relpath(CONTEXT_VARIATION_OUT.resolve(), _REPO)
    prompt_rel = os.path.relpath(PROMPT_EXP12.resolve(), _REPO)

    cmd_html = f"python adhoc/generation/robotarm/exp12_context_variation.py --html-only"
    cmd_gen = f"python adhoc/generation/robotarm/exp12_context_variation.py --model gemini-2.5-pro"
    cmd_render = f"python adhoc/generation/robotarm/exp12_context_variation.py --html-only --force-render"
    cmd_cli = (
        "python adhoc/generation/robotarm/paper_figures/cli.py context "
        f"--catalog-yml {yml_rel}"
    )

    sample = pairs[0] if pairs else None
    sample_prompt = ""
    if sample:
        idx, ckey, _, cdesc = sample
        base = base_for_idx(idx, shot_bases=shot_bases) or {"cue": "?", "idx": idx, "movements": []}
        sample_prompt = _fill_prompt(cue=str(base.get("cue")), context_name=ckey, context_desc=cdesc, base=base)

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Exp12 — context variation input</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:20px auto;padding:0 16px 48px;background:#fafafa;color:#111}}
h1{{font-size:22px}} h2{{font-size:17px;margin-top:28px}}
.summary{{background:#fff;border:1px solid #ccc;border-radius:10px;padding:14px;margin:16px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0;background:#fff}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#f0f0f0}}
td.desc{{color:#444;font-size:12px}}
pre.cmd{{background:#1e1e2e;color:#e8e8f0;padding:12px;border-radius:8px;font-size:12px;overflow:auto}}
pre.sample{{background:#f5f5f8;padding:12px;border-radius:8px;font-size:11px;white-space:pre-wrap;max-height:420px;overflow:auto}}
.tag{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}}
.tag.persona{{background:#e3f2fd;color:#1565c0}}
.tag.scenario{{background:#fff3e0;color:#e65100}}
</style></head><body>
<h1>Exp12 — Context variation (20 contexts)</h1>
<div class="summary">
  <p><b>Catalog:</b> <code>{html.escape(yml_rel)}</code> — 10 personas + 10 scenarios.</p>
  <p><b>Prompt:</b> <code>{html.escape(prompt_rel)}</code> — scenario/context only; no component GT.</p>
  <p><b>Run matrix:</b> {len(pairs)} (cue × context) jobs → <code>{html.escape(out_rel)}</code></p>
  <p><span class="tag persona">persona</span> character style &nbsp; <span class="tag scenario">scenario</span> situational context</p>
</div>

<h2>How to run</h2>
<pre class="cmd"># 1) Refresh this input page
{html.escape(cmd_html)}

# 2) Generate configs (Gemini) + render GIFs (GPU)
source APIKEY.sh
{html.escape(cmd_gen)}

# 3) Rebuild review HTML from existing outputs
{html.escape(cmd_render)}

# CLI alias
{html.escape(cmd_cli)}</pre>

<h2>20 contexts</h2>
<h3>Personas (10)</h3>
<table><thead><tr><th>key</th><th>type</th><th>description</th></tr></thead><tbody>{''.join(persona_rows)}</tbody></table>
<h3>Scenarios (10)</h3>
<table><thead><tr><th>key</th><th>type</th><th>description</th></tr></thead><tbody>{''.join(scenario_rows)}</tbody></table>

<h2>Run matrix ({len(pairs)} jobs)</h2>
<table><thead><tr><th>idx</th><th>cue</th><th>context</th><th>type</th><th>description</th></tr></thead><tbody>{''.join(run_rows)}</tbody></table>

<h2>Sample filled prompt (first job)</h2>
<pre class="sample">{html.escape(sample_prompt)}</pre>
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
    out_dir: str | None = None,
) -> Path:
    yml_path = Path(catalog_yml) if catalog_yml else CONTEXT_VARIATION_YML
    if html_only and not force_render:
        build_input_html(catalog_yml=yml_path)
        pairs = load_run_pairs(catalog_yml=yml_path)
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
        build_review_html(_adapt_for_persona_html(items), review)
        return review

    pairs = load_run_pairs(catalog_yml=yml_path)
    shot_bases = load_shot_bases()
    od = Path(out_dir) if out_dir else CONTEXT_VARIATION_OUT
    cfg_out = od / "context_configs.json"
    gif_dir = od / "gifs"
    tmp_cfg = od / "_tmp_render.json"
    gif_dir.mkdir(parents=True, exist_ok=True)

    configs: list[dict[str, Any]] = []
    if cfg_out.is_file():
        try:
            configs = json.loads(cfg_out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            configs = []
    done_keys = {(int(c["idx"]), c.get("context")) for c in configs if c.get("idx") is not None}

    render_items: list[tuple[dict, Path | None]] = []
    print(f"[exp12] contexts={len(load_context_catalog(yml_path))} jobs={len(pairs)} → {od}", flush=True)

    for idx, ckey, ctype, cdesc in pairs:
        base = base_for_idx(idx, shot_bases=shot_bases)
        if not base:
            print(f"[skip] idx {idx} not in shot bases or pilot90 config", flush=True)
            continue
        gif_path = gif_dir / _gif_name(idx, base, ckey)
        if (idx, ckey) in done_keys and gif_path.is_file() and not force_render:
            cfg = next(c for c in configs if int(c["idx"]) == idx and c.get("context") == ckey)
            render_items.append((cfg, gif_path))
            continue
        if (idx, ckey) not in done_keys:
            print(f"  {ctype} {ckey} @ c{idx} {base['cue']}", flush=True)
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
            ok = _render_gif(_for_render(cfg), gif_path, tmp_cfg, force=force_render)
            render_items.append((cfg, gif_path if ok else None))
        else:
            render_items.append((cfg, gif_path if gif_path.is_file() else None))

    review = od / "context_variations.html"
    build_review_html(_adapt_for_persona_html(render_items), review)
    print(f"Wrote {review}")
    print(f"Configs: {cfg_out} ({len(configs)} entries)")
    return review


def _for_render(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    out["persona"] = str(cfg.get("context", "ctx"))
    return out


def _adapt_for_persona_html(items: list[tuple[dict, Path | None]]) -> list[tuple[dict, Path | None]]:
    adapted = []
    for cfg, gif in items:
        row = dict(cfg)
        row["persona"] = str(cfg.get("context", cfg.get("persona", "?")))
        row["persona_desc"] = str(cfg.get("context_desc", cfg.get("persona_desc", "")))
        adapted.append((row, gif))
    return adapted


def main() -> None:
    p = argparse.ArgumentParser(description="Exp12 context variation (persona + scenario)")
    p.add_argument("--catalog-yml", default=None)
    p.add_argument("--model", default="gemini-2.5-pro")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-render", action="store_true")
    p.add_argument("--force-render", action="store_true")
    p.add_argument("--html-only", action="store_true", help="Input plan HTML, or review HTML if configs exist")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--fresh", action="store_true", help="Delete prior context_configs.json and GIFs in out-dir")
    p.add_argument("--open", action="store_true")
    args = p.parse_args()

    if args.html_only and not (Path(args.out_dir or CONTEXT_VARIATION_OUT) / "context_configs.json").is_file():
        out = build_input_html(catalog_yml=Path(args.catalog_yml) if args.catalog_yml else None)
    else:
        if args.fresh:
            od = Path(args.out_dir or CONTEXT_VARIATION_OUT)
            cfg_p = od / "context_configs.json"
            if cfg_p.is_file():
                cfg_p.unlink()
            gif_d = od / "gifs"
            if gif_d.is_dir():
                for g in gif_d.glob("*.gif"):
                    g.unlink()
        out = run(
            catalog_yml=args.catalog_yml,
            model=args.model,
            dry_run=args.dry_run,
            skip_render=args.skip_render,
            force_render=args.force_render,
            html_only=args.html_only,
            out_dir=args.out_dir,
        )
    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    main()
