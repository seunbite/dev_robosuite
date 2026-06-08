"""Persona-driven config variation: curated cue×persona matrix → render HTML."""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

_HERE = Path(__file__).resolve().parent
_ROBOTARM = _HERE.parent
_REPO = _ROBOTARM.parents[2]
for p in (_REPO, _ROBOTARM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from paper_figures._media import cfg_by_idx, repo  # noqa: E402

DEFAULT_CUE_IDXS = (7, 15, 21, 28, 35)
MATRIX_V20_YML = _REPO / "data/seed/yml/persona_paper_runs_v20.yml"

# legacy 5×5 emotion grid
PERSONAS_LEGACY: tuple[tuple[str, str], ...] = (
    ("sad", "Subdued, slow, low energy; longer holds, smaller amplitude."),
    ("aggressive", "Sharp, fast, emphatic; short holds, large angles, decisive beats."),
    ("happy", "Buoyant, bouncy rhythm; slightly faster speed, upward flourishes."),
    ("anxious", "Hesitant micro-pauses, uneven timing, reduced reach, jitter."),
    ("calm", "Smooth, steady, minimal overshoot; moderate speed and clean holds."),
)

PERSONA_PROMPT = """You are refining a robot arm motion JSON for a specific persona/character.

Base cue: {cue}
Persona: {persona_name} — {persona_desc}

Base config (JSON):
{base_json}

Rewrite the JSON to express this persona by adjusting hold_time, speed, angle/degrees, repetition,
and optionally adding short preparatory or follow-through motion steps before/after the core gesture.
The timing differences MUST be visible: vary speed (0.5–4.0) and hold_time meaningfully vs the base.
Keep the same cue name and idx. Preserve valid schema (pose > movement/path). Output ONLY valid JSON.
"""


def load_persona_catalog(path: Path | None = None) -> dict[str, str]:
    path = path or MATRIX_V20_YML
    if yaml is None:
        raise RuntimeError("PyYAML required: pip install pyyaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    personas = data.get("personas") or {}
    return {str(k): str(v).strip() for k, v in personas.items()}


def load_run_pairs(matrix: str = "v20", *, matrix_yml: Path | None = None) -> list[tuple[int, str, str]]:
    """Return [(cue_idx, persona_key, persona_desc), ...]."""
    if matrix == "legacy":
        out: list[tuple[int, str, str]] = []
        for idx in DEFAULT_CUE_IDXS:
            for pname, pdesc in PERSONAS_LEGACY:
                out.append((idx, pname, pdesc))
        return out

    path = matrix_yml or MATRIX_V20_YML
    if yaml is None:
        raise RuntimeError("PyYAML required: pip install pyyaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    catalog = load_persona_catalog(path)
    pairs: list[tuple[int, str, str]] = []
    for row in data.get("runs") or []:
        idx = int(row["idx"])
        for pname in row.get("personas") or []:
            key = str(pname)
            if key not in catalog:
                raise KeyError(f"persona {key!r} missing from catalog in {path}")
            pairs.append((idx, key, catalog[key]))
    return pairs


def matrix_out_dir(matrix: str, out_dir: str | None) -> Path:
    if out_dir:
        return Path(out_dir)
    sub = "persona" if matrix == "legacy" else f"persona_{matrix}"
    return repo() / "data/results/paper_figures" / sub


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


def summarize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Compact numeric summary for cross-persona comparison."""
    speeds: list[float] = []
    holds: list[float] = []
    degs: list[float] = []
    reps: list[int] = []
    step_lines: list[str] = []

    for i, step in enumerate(cfg.get("movements") or [], 1):
        stype = step.get("type", "?")
        p = step.get("parameters") or {}

        if stype == "pose":
            pose = p.get("pose") or {}
            sp = p.get("speed")
            ht = p.get("hold_time", 0)
            if sp is not None:
                speeds.append(float(sp))
            if ht:
                holds.append(float(ht))
            step_lines.append(
                f"{i}:{stype} {pose.get('dir','?')}+{pose.get('gripper_orientation','?')} "
                f"sp={sp} hold={ht}"
            )
        elif stype == "movement":
            rep = int(p.get("repetition", 1))
            reps.append(rep)
            joint = p.get("joint", "?")
            parts: list[str] = []
            for d in p.get("directions") or []:
                sp = d.get("speed")
                ht = d.get("hold_time", 0)
                if sp is not None:
                    speeds.append(float(sp))
                if ht:
                    holds.append(float(ht))
                deg = d.get("degrees") or {}
                for ax, val in deg.items():
                    degs.append(abs(float(val)))
                    parts.append(f"{ax}{val:+.0f}°@sp{sp}")
            step_lines.append(f"{i}:{stype} {joint}×{rep} {'; '.join(parts)}")
        elif stype == "path":
            sp = p.get("speed")
            if sp is not None:
                speeds.append(float(sp))
            step_lines.append(
                f"{i}:{stype} {p.get('shape','?')} {p.get('axis','?')} "
                f"dist={p.get('distance','?')} sp={sp}"
            )
        else:
            step_lines.append(f"{i}:{stype}")

    return {
        "n_steps": len(cfg.get("movements") or []),
        "speed_min": min(speeds) if speeds else None,
        "speed_max": max(speeds) if speeds else None,
        "speed_avg": round(sum(speeds) / len(speeds), 2) if speeds else None,
        "hold_sum": round(sum(holds), 2) if holds else 0.0,
        "max_deg": max(degs) if degs else None,
        "max_rep": max(reps) if reps else None,
        "sequence": " → ".join(stype for stype in [s.get("type", "?") for s in cfg.get("movements") or []]),
        "step_lines": step_lines,
    }


def _fmt_num(v: Any, *, suffix: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.1f}{suffix}"
    return f"{v}{suffix}"


def _summary_block(cfg: dict[str, Any]) -> str:
    s = summarize_config(cfg)
    rows = [
        ("steps", str(s["n_steps"])),
        ("sequence", s["sequence"]),
        ("speed", f"{_fmt_num(s['speed_min'])} – {_fmt_num(s['speed_max'])} (avg {_fmt_num(s['speed_avg'])})"),
        ("hold Σ", f"{s['hold_sum']:.1f}s"),
        ("max |°|", _fmt_num(s["max_deg"])),
        ("max rep", _fmt_num(s["max_rep"])),
    ]
    inner = "".join(
        f'<div class="kv"><span class="k">{html.escape(k)}</span><span class="v">{html.escape(v)}</span></div>'
        for k, v in rows
    )
    steps_detail = "<br>".join(html.escape(line) for line in s["step_lines"])
    return f'<div class="summary">{inner}<div class="steps">{steps_detail}</div></div>'


def _gemini_variation(
    base: dict[str, Any],
    persona_name: str,
    persona_desc: str,
    model: str,
    *,
    max_retries: int = 3,
) -> dict[str, Any]:
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    prompt = PERSONA_PROMPT.format(
        cue=base.get("cue"),
        persona_name=persona_name,
        persona_desc=persona_desc,
        base_json=json.dumps(base, indent=2, ensure_ascii=False),
    )
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            text = resp.text or ""
            out = _extract_json(text)
            out["idx"] = base.get("idx")
            out["cue"] = base.get("cue")
            out["persona"] = persona_name
            out["base_model"] = model
            out["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return out
        except Exception as e:
            last_err = e
            print(f"    [retry {attempt + 1}/{max_retries}] {e}", flush=True)
    raise RuntimeError(f"Gemini persona failed after {max_retries} tries: {last_err}")


def _render_gif(cfg: dict[str, Any], out_gif: Path, tmp_cfg: Path, *, force: bool = False) -> bool:
    import shutil

    if out_gif.is_file() and not force:
        return True
    if force and out_gif.is_file():
        out_gif.unlink()

    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg.write_text(json.dumps([cfg], indent=2, ensure_ascii=False), encoding="utf-8")
    persona = str(cfg.get("persona", "base"))
    try:
        from legacy.motion_generation_core import generate

        scratch = out_gif.parent / "_render_scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        jpath = str(repo() / "data/seed/_remainder/closest_poses_results.jsonl")
        generate(
            robot="IIWA",
            cue=str(cfg["cue"]),
            cue_idx=int(cfg["idx"]),
            pose_index=None,
            jsonl_path=jpath,
            config_path=str(tmp_cfg),
            output_dir=str(scratch),
            hz=10,
            top_k=1,
            gif_filename_suffix=f"_{persona}",
        )
        robot_dir = scratch / "IIWA"
        pattern = f"*_{cfg['cue']}_*{persona}*.gif"
        gifs = sorted(robot_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not gifs:
            gifs = sorted(robot_dir.glob("*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)
        if gifs:
            shutil.copy2(gifs[0], out_gif)
            return True
    except Exception as e:
        print(f"[render fail] {cfg.get('cue')}:{persona}: {e}", flush=True)
    return False


def build_html(items: list[tuple[dict, Path | None]], out_html: Path) -> None:
    by_cue: dict[str, list[tuple[dict, Path | None]]] = {}
    for cfg, gif in items:
        by_cue.setdefault(str(cfg.get("cue", "?")), []).append((cfg, gif))

    sections: list[str] = []
    for cue, group in sorted(by_cue.items(), key=lambda x: int(x[1][0][0].get("idx", 0))):
        idx = group[0][0].get("idx")
        # comparison table
        thead = (
            "<tr><th>persona</th><th>steps</th><th>speed</th><th>hold Σ</th>"
            "<th>max °</th><th>max rep</th><th>sequence</th></tr>"
        )
        trows = []
        for cfg, _ in sorted(group, key=lambda x: str(x[0].get("persona", ""))):
            s = summarize_config(cfg)
            trows.append(
                f"<tr><td><b>{html.escape(str(cfg.get('persona')))}</b></td>"
                f"<td>{s['n_steps']}</td>"
                f"<td>{_fmt_num(s['speed_min'])}–{_fmt_num(s['speed_max'])}</td>"
                f"<td>{s['hold_sum']:.1f}s</td>"
                f"<td>{_fmt_num(s['max_deg'])}</td>"
                f"<td>{_fmt_num(s['max_rep'])}</td>"
                f"<td class='seq'>{html.escape(s['sequence'])}</td></tr>"
            )
        cards = []
        for cfg, gif in sorted(group, key=lambda x: str(x[0].get("persona", ""))):
            persona = html.escape(str(cfg.get("persona", "?")))
            rel = os.path.relpath(gif, out_html.parent) if gif and gif.is_file() else ""
            media = f'<img src="{rel}" loading="lazy">' if rel else "<div class='missing'>No render</div>"
            raw = html.escape(json.dumps(cfg, indent=2, ensure_ascii=False))
            pdesc = html.escape(str(cfg.get("persona_desc", ""))[:120])
            desc_html = f'<p class="pdesc">{pdesc}…</p>' if pdesc else ""
            cards.append(
                f"<article><h4>{persona}</h4>{desc_html}{media}"
                f"{_summary_block(cfg)}"
                f"<details><summary>raw JSON</summary><pre>{raw}</pre></details></article>"
            )
        sections.append(
            f"<section class='cue-group'><h2>c{idx} {html.escape(cue)}</h2>"
            f"<table class='compare'><thead>{thead}</thead><tbody>{''.join(trows)}</tbody></table>"
            f"<div class='grid'>{''.join(cards)}</div></section>"
        )

    doc = (
        "<html><head><meta charset='utf-8'><title>Persona variations</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:16px;max-width:1400px}"
        ".cue-group{margin:28px 0;border-top:2px solid #222;padding-top:12px}"
        "table.compare{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0 16px}"
        "table.compare th,table.compare td{border:1px solid #ccc;padding:4px 8px;text-align:left}"
        "table.compare th{background:#f0f0f0}"
        "td.seq{font-size:11px;color:#444}"
        ".grid{display:flex;flex-wrap:wrap;gap:10px}"
        "article{border:1px solid #ddd;padding:8px;width:240px;vertical-align:top}"
        "article h4{margin:0 0 4px}"
        ".pdesc{font-size:10px;color:#555;margin:0 0 6px;line-height:1.3}"
        "img{max-width:220px;height:auto;display:block}"
        ".summary{background:#f8f9fc;border:1px solid #e0e4ef;border-radius:4px;padding:6px;margin:6px 0;font-size:11px}"
        ".kv{display:flex;gap:6px;margin:2px 0}"
        ".k{color:#555;min-width:52px;font-weight:600}"
        ".v{color:#111;flex:1}"
        ".steps{margin-top:6px;padding-top:4px;border-top:1px dashed #ccd;color:#333;line-height:1.35}"
        "details pre{font-size:9px;max-height:160px;overflow:auto;background:#f5f5f5;margin:4px 0}"
        ".missing{color:#999;font-size:12px}"
        "</style></head><body>"
        f"<h1>Persona variations ({len(items)} configs)</h1>"
        f"<p>Grouped by cue — compare <b>speed / hold / amplitude</b> in table + summary blocks.</p>"
        f"{''.join(sections)}</body></html>"
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(doc, encoding="utf-8")


def run(
    *,
    matrix: str = "v20",
    matrix_yml: str | None = None,
    cue_idxs: str | None = None,
    model: str = "gemini-2.5-pro",
    dry_run: bool = False,
    skip_render: bool = False,
    force_render: bool = False,
    html_only: bool = False,
    out_dir: str | None = None,
) -> Path:
    yml_path = Path(matrix_yml) if matrix_yml else MATRIX_V20_YML
    pairs = load_run_pairs(matrix, matrix_yml=yml_path)
    if cue_idxs:
        want = {int(x) for x in cue_idxs.split(",") if x.strip()}
        pairs = [(i, p, d) for i, p, d in pairs if i in want]

    by_idx = cfg_by_idx()
    od = matrix_out_dir(matrix, out_dir)
    cfg_out = od / "persona_configs.json"
    gif_dir = od / "gifs"
    tmp_cfg = od / "_tmp_render.json"
    gif_dir.mkdir(parents=True, exist_ok=True)

    configs: list[dict[str, Any]] = []
    if cfg_out.is_file():
        try:
            configs = json.loads(cfg_out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            configs = []
    done_keys = {(int(c["idx"]), c.get("persona")) for c in configs if c.get("idx") is not None}

    render_items: list[tuple[dict, Path | None]] = []
    print(f"[persona] matrix={matrix} runs={len(pairs)} → {od}", flush=True)

    if html_only:
        for cfg in configs:
            gif_path = gif_dir / f"c{cfg['idx']}_{cfg['cue']}_{cfg['persona']}.gif"
            render_items.append((cfg, gif_path if gif_path.is_file() else None))
    else:
        for idx, pname, pdesc in pairs:
            base = by_idx.get(idx)
            if not base:
                print(f"[skip] idx {idx} not in pilot90 config", flush=True)
                continue
            gif_path = gif_dir / f"c{idx}_{base['cue']}_{pname}.gif"
            if (idx, pname) in done_keys and gif_path.is_file() and not force_render:
                cfg = next(c for c in configs if int(c["idx"]) == idx and c.get("persona") == pname)
                render_items.append((cfg, gif_path))
                continue
            if (idx, pname) not in done_keys:
                print(f"  persona {pname} @ c{idx} {base['cue']}", flush=True)
                if dry_run:
                    cfg = dict(base)
                    cfg["persona"] = pname
                    cfg["persona_desc"] = pdesc
                else:
                    cfg = _gemini_variation(base, pname, pdesc, model)
                    cfg["persona_desc"] = pdesc
                configs.append(cfg)
                done_keys.add((idx, pname))
                cfg_out.write_text(json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8")
            else:
                cfg = next(c for c in configs if int(c["idx"]) == idx and c.get("persona") == pname)

            if not skip_render:
                ok = _render_gif(cfg, gif_path, tmp_cfg, force=force_render)
                render_items.append((cfg, gif_path if ok else None))
                if not ok:
                    print(f"  [warn] render failed for {pname}@{base['cue']}", flush=True)
            else:
                render_items.append((cfg, gif_path if gif_path.is_file() else None))

    html_path = od / "persona_variations.html"
    build_html(render_items, html_path)
    print(f"Wrote {html_path}")
    if not html_only:
        print(f"Configs: {cfg_out} ({len(configs)} entries)")
    return html_path


def main() -> None:
    p = argparse.ArgumentParser(description="Persona-driven motion config variations")
    p.add_argument("--matrix", default="v20", choices=("legacy", "v20"), help="Run matrix (default: v20 = 20 pairs)")
    p.add_argument("--matrix-yml", default=None, help="Override persona_paper_runs YAML")
    p.add_argument("--cue-idxs", default=None, help="Filter matrix to these cue idx only")
    p.add_argument("--model", default="gemini-2.5-pro")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-render", action="store_true")
    p.add_argument("--force-render", action="store_true", help="Re-render all GIFs from persona configs")
    p.add_argument("--html-only", action="store_true", help="Rebuild HTML from existing configs/GIFs")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    run(
        matrix=args.matrix,
        matrix_yml=args.matrix_yml,
        cue_idxs=args.cue_idxs,
        model=args.model,
        dry_run=args.dry_run,
        skip_render=args.skip_render,
        force_render=args.force_render,
        html_only=args.html_only,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
