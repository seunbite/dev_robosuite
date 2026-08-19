#!/usr/bin/env python3
"""Generate Short Motion Codes via Gemini for all 5 embodiments.

Default prompt (``yaml_cue``):
  - only the target cue name + description (no Available-cues catalog)
  - only that embodiment's block from ``motion_code_embodiment.yaml``

Legacy ``prompt_exp0`` style remains available via ``--prompt-style exp0``.

Example::

    export GOOGLE_API_KEY=...
    python adhoc/generation/robotarm/run_embodiment_exp0_gemini.py \\
      --platforms humanoid,roomba --max-cues 2 --model gemini-2.5-flash

Does not open Human GT motion codes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE, _REPO / "adhoc/generation", _REPO / "adhoc/generation/tiago"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from motion_code_to_config import (  # noqa: E402
    convert_code,
    render_tiago,
    render_manipulator,
    render_quadruped,
    write_json_files,
)

PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "quadruped", "roomba")
PROMPT_PATHS = {
    p: _REPO / "data/seed/prompt" / p / "exp" / "prompt_exp0.txt" for p in PLATFORMS
}
SCHEMA_YAML = _REPO / "data/seed/yml/motion_code_embodiment.yaml"
DEFAULT_CUES = _REPO / "data/seed/agent_fill/embodiment_short/cues.json"
DEFAULT_FEWSHOT = _REPO / "data/seed/agent_fill/embodiment_short/fewshot_synthetic.md"
DEFAULT_OUT = _REPO / "data/results/motion_configs/embodiment_motion_code_gemini"
DEFAULT_OUT_RENDER = _REPO / "data/results/render/embodiment_motion_code_gemini"
DEFAULT_HTML = _REPO / "data/results/html/embodiment_motion_code_gemini.html"

_GIF_REL = {
    "humanoid": ("humanoid", "gif"),
    "tiago": ("tiago", "gif"),
    "single_arm": ("manipulator", "gif", "IIWA"),
    "stretch": ("stretch", "gif"),
    "quadruped": ("quadruped", "gif"),
    "roomba": ("roomba", "gif"),
}

_MOTION_LINE_RE = re.compile(
    r"(?m)^(?:Motion Code|source_code)\s*:\s*(.+)$",
    re.IGNORECASE,
)
_CODE_FENCE_RE = re.compile(r"```(?:motion|text)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_POSE_START_RE = re.compile(
    r"(?:^|\n)\s*((?:pose|path|move|hold|movement)\s*\(.*)",
    re.IGNORECASE,
)


def load_cues(path: Path) -> list[dict[str, Any]]:
    return sorted(json.loads(path.read_text(encoding="utf-8")), key=lambda r: int(r["idx"]))


@lru_cache(maxsize=1)
def _schema_root() -> dict[str, Any]:
    return yaml.safe_load(SCHEMA_YAML.read_text(encoding="utf-8"))


def embodiment_schema_yaml(platform: str) -> str:
    """Dump only the requested embodiment block (anchors already resolved)."""
    root = _schema_root()
    if platform not in root:
        raise KeyError(f"unknown platform in schema YAML: {platform}")
    return yaml.safe_dump(
        {platform: root[platform]},
        sort_keys=False,
        allow_unicode=True,
        width=100,
        default_flow_style=False,
    ).rstrip()


def load_fewshot_for_platform(path: Path, platform: str) -> str:
    if not path.is_file():
        return "(none)"
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^##\s+", text)
    for part in parts:
        head, _, body = part.partition("\n")
        if head.strip().lower() == platform:
            return body.strip() or "(none)"
    return "(see fewshot_synthetic.md)"


def cue_catalog(cues: list[dict[str, Any]], *, target: str) -> str:
    lines = ["[Available cues]"]
    for c in cues:
        mark = " <<" if c["cue"] == target else ""
        lines.append(f"- {c['cue']}: {c.get('description', '')}{mark}")
    return "\n".join(lines)


def build_prompt_yaml_cue(
    *,
    platform: str,
    cue: str,
    description: str,
    category: str = "",  # unused; kept for call-site compat
) -> str:
    """Short generation: one cue definition + one embodiment YAML block only."""
    del category
    root = _schema_root()
    robot = str(root.get(platform, {}).get("robot") or platform)
    schema = embodiment_schema_yaml(platform)
    return f"""You are an expert in {robot} gesture design using Motion Code.

Goal:
- Generate ONE complete short cue motion using Motion Code primitives that this
  embodiment supports (`pose`, `move`, `path`, `hold` as listed in the schema).
- Every Motion Code MUST start with `pose(...)`.
- Total steps: 2~8. Be expressive; match the cue's emotion and rhythm.
- Join steps with `+`.

[CUE]
name: {cue}
description: {description}

[EMBODIMENT SCHEMA — {platform}]
# Source: data/seed/yml/motion_code_embodiment.yaml ({platform} only, anchors resolved)
{schema}

[Output]
Output ONE Motion Code line only (steps joined by +). No planning comments, no JSON, no explanation.
Do not invent attributes or component forms outside the embodiment schema above.
""".strip() + "\n"


def build_prompt_exp0(
    *,
    platform: str,
    cue: str,
    description: str,
    cues: list[dict[str, Any]],
    fewshot_path: Path,
) -> str:
    template = PROMPT_PATHS[platform].read_text(encoding="utf-8")
    return (
        template.replace("{{CUE_CATALOG}}", cue_catalog(cues, target=cue))
        .replace("{{FEW_SHOT_EXAMPLES}}", load_fewshot_for_platform(fewshot_path, platform))
        .replace("{{CUE_NAME}}", cue)
        .replace("{{DESCRIPTION}}", description)
        + "\n\n# Output requirement:\n"
        "# After the four # Q lines, print ONE Motion Code line (steps joined by +).\n"
        '# Then optionally a JSON object with \"source_code\" and \"movements\".\n'
    )


def build_prompt(
    *,
    platform: str,
    cue: str,
    description: str,
    cues: list[dict[str, Any]] | None = None,
    fewshot_path: Path | None = None,
    category: str = "",
    prompt_style: str = "yaml_cue",
) -> str:
    if prompt_style == "yaml_cue":
        return build_prompt_yaml_cue(
            platform=platform,
            cue=cue,
            description=description,
            category=category,
        )
    if prompt_style == "exp0":
        if cues is None or fewshot_path is None:
            raise ValueError("exp0 style requires cues + fewshot_path")
        return build_prompt_exp0(
            platform=platform,
            cue=cue,
            description=description,
            cues=cues,
            fewshot_path=fewshot_path,
        )
    raise ValueError(f"unknown prompt_style={prompt_style!r}")


def _extract_json_obj(raw: str) -> dict[str, Any] | None:
    start = raw.find("{")
    if start < 0:
        return None
    # brace scan
    depth = 0
    for i, ch in enumerate(raw[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = raw[start : i + 1]
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def extract_motion_code(raw: str, obj: dict[str, Any] | None = None) -> str:
    if obj:
        sc = str(obj.get("source_code") or "").strip()
        if sc:
            return sc
    m = _MOTION_LINE_RE.search(raw)
    if m:
        return m.group(1).strip().strip("`")
    fence = _CODE_FENCE_RE.search(raw)
    if fence:
        body = fence.group(1).strip()
        for line in body.splitlines():
            if re.match(r"^(pose|path|move|hold|movement)\s*\(", line.strip(), re.I):
                return " ".join(body.split())
    # last non-comment line that looks like motion code
    for line in reversed(raw.splitlines()):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if re.match(r"^(pose|path|move|hold|movement)\s*\(", s, re.I):
            return s
    m2 = _POSE_START_RE.search(raw)
    if m2:
        # take until blank line / JSON
        chunk = m2.group(1)
        rest = raw[m2.end() :]
        for line in rest.splitlines():
            if line.strip().startswith("{") or line.strip().startswith("#"):
                break
            if not line.strip():
                if "+" in chunk:
                    break
                continue
            chunk += " " + line.strip()
        return " ".join(chunk.split())
    return ""


def gemini_generate(prompt: str, *, model: str) -> str:
    from google import genai

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY (source APIKEY.sh)")
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(model=model, contents=prompt)
    return (resp.text or "").strip()


def _model_tag(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model)


def write_fill_md(
    configs: list[dict[str, Any]],
    cues: list[dict[str, Any]],
    platforms: list[str],
    out_path: Path,
) -> None:
    """Checkpoint Motion Codes as agent-fill style markdown."""
    by_cue = {e["cue"]: e for e in configs}
    lines = [
        f"# Gemini short fill checkpoint ({datetime.now().isoformat(timespec='seconds')})",
        "",
    ]
    for plat in platforms:
        lines.append(f"## {plat}")
        lines.append("")
        for c in cues:
            entry = by_cue.get(c["cue"], {})
            row = (entry.get("platforms") or {}).get(plat) or {}
            code = str(row.get("source_code") or "").strip()
            lines.append(f"{int(c['idx']):02d} {c['cue']} | {code}")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def gif_path_for(render_root: Path, platform: str, idx: int, cue: str) -> Path:
    parts = _GIF_REL[platform]
    return render_root.joinpath(*parts) / f"emb_g{idx:02d}_{cue}.gif"


def build_live_html(
    configs: list[dict[str, Any]],
    *,
    render_root: Path,
    out_html: Path,
    title: str,
    model: str,
) -> Path:
    """Partial-safe gallery (missing platforms / gifs OK) with auto-refresh."""
    import html as html_mod

    done_gif = 0
    done_parse = 0
    total_slots = 0
    sections: list[str] = []
    for entry in sorted(configs, key=lambda e: int(e["idx"])):
        cue = entry["cue"]
        idx = int(entry["idx"])
        cols: list[str] = []
        for plat, label in [
            ("humanoid", "Humanoid"),
            ("tiago", "Tiago"),
            ("single_arm", "Single-arm"),
            ("stretch", "Stretch"),
            ("quadruped", "Quadruped"),
            ("roomba", "Roomba"),
        ]:
            total_slots += 1
            cfg = (entry.get("platforms") or {}).get(plat)
            gif = gif_path_for(render_root, plat, idx, cue)
            if cfg is None:
                media = '<p class="miss">pending gen</p>'
                source_code = "(not generated)"
                movements_txt = "[]"
            else:
                if cfg.get("parse_ok"):
                    done_parse += 1
                source_code = str(cfg.get("source_code") or "").strip() or "(empty)"
                movements_txt = json.dumps(cfg.get("movements") or [], indent=2, ensure_ascii=False)
                if gif.is_file():
                    done_gif += 1
                    rel = os.path.relpath(gif, out_html.parent)
                    bust = int(gif.stat().st_mtime)
                    media = (
                        f'<img src="{html_mod.escape(rel)}?v={bust}" loading="lazy">'
                    )
                elif cfg.get("parse_ok"):
                    media = '<p class="miss">render queued</p>'
                else:
                    media = '<p class="miss">parse fail</p>'
            cols.append(
                f'<div class="col"><h3>{label}</h3>{media}'
                f'<div class="mc"><div class="mc-label">Motion Code</div>'
                f'<pre class="mc-pre">{html_mod.escape(source_code)}</pre></div>'
                f'<div class="mc"><div class="mc-label">Config</div>'
                f'<pre class="mc-pre">{html_mod.escape(movements_txt)}</pre></div></div>'
            )
        sections.append(
            f'<section class="cue"><h2>{idx:02d}. {html_mod.escape(cue)}</h2>'
            f'<div class="grid">{"".join(cols)}</div></section>'
        )

    now = datetime.now().isoformat(timespec="seconds")
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html_mod.escape(title)}</title>
<meta http-equiv="refresh" content="20">
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e8eaed}}
h1{{padding:16px 24px;margin:0;border-bottom:1px solid #333}}
.meta{{padding:10px 24px;color:#9aa0a6;font-size:0.9rem}}
.cue{{margin:12px 16px;padding:12px;border:1px solid #333;border-radius:10px;background:#171a21}}
.grid{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px}}
.col{{background:#0f1115;border-radius:8px;padding:8px;min-width:0}}
.col h3{{margin:0 0 6px;font-size:0.78rem;font-weight:600;color:#c4c7ce}}
img{{width:100%;height:auto;aspect-ratio:1;object-fit:cover;border-radius:6px;background:#000;display:block}}
.miss{{color:#f28b82;font-size:0.8rem}}
.mc{{margin-top:8px}}
.mc-label{{font-size:0.68rem;color:#9aa0a6;margin-bottom:3px;text-transform:uppercase}}
.mc-pre{{font-size:10px;line-height:1.35;overflow:auto;max-height:120px;background:#0a0c10;padding:6px;border-radius:4px;white-space:pre-wrap;word-break:break-word;margin:0;color:#d7dbe2}}
@media (max-width:1300px){{.grid{{grid-template-columns:repeat(6,minmax(140px,1fr));overflow-x:auto}}}}
</style></head><body>
<h1>{html_mod.escape(title)}</h1>
<p class="meta">model={html_mod.escape(model)} · parse_ok={done_parse} · gifs={done_gif} · slots={total_slots} · updated {html_mod.escape(now)} · auto-refresh 20s</p>
{"".join(sections)}
</body></html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding="utf-8")
    return out_html


def render_pending_once(
    configs: list[dict[str, Any]],
    *,
    render_root: Path,
    out_json: Path | None = None,
    platforms: list[str] | None = None,
    force: bool = False,
    max_items: int = 3,
) -> int:
    """Render up to ``max_items`` missing GIFs. Returns how many were attempted."""
    from embodiment_renderers import render_stretch, render_roomba, render_unitree_g1

    plats = platforms or list(PLATFORMS)
    skip = getattr(render_pending_once, "_skip", None)
    if skip is None:
        skip = set()
        render_pending_once._skip = skip  # type: ignore[attr-defined]
    pending: list[tuple[str, dict[str, Any]]] = []
    for entry in configs:
        for plat in plats:
            cfg = (entry.get("platforms") or {}).get(plat)
            if not cfg or not cfg.get("parse_ok") or not (cfg.get("movements") or []):
                continue
            cue = str(entry["cue"])
            if (plat, cue) in skip:
                continue
            gif = gif_path_for(render_root, plat, int(entry["idx"]), cue)
            if gif.is_file() and not force:
                continue
            pending.append((plat, cfg))
    if not pending:
        return 0

    batch = pending[: max(1, max_items)]
    by_plat: dict[str, list[dict[str, Any]]] = {}
    for plat, cfg in batch:
        by_plat.setdefault(plat, []).append(cfg)

    for plat, rows in by_plat.items():
        print(f"[render] {plat} ×{len(rows)}", flush=True)
        before = {str(r.get("cue")): int(r["idx"]) for r in rows}
        try:
            if plat == "humanoid":
                render_unitree_g1(rows, render_root / "humanoid" / "gif", force=force)
            elif plat == "tiago":
                render_tiago(rows, out_root=render_root, force=force)
            elif plat == "single_arm":
                render_manipulator(rows, out_root=render_root, force=force)
            elif plat == "quadruped":
                cfg_path = (out_json or DEFAULT_OUT) / "result_quadruped.json"
                if not cfg_path.is_file():
                    q_rows = []
                    for entry in configs:
                        q = (entry.get("platforms") or {}).get("quadruped")
                        if q:
                            q_rows.append(q)
                    cfg_path.parent.mkdir(parents=True, exist_ok=True)
                    cfg_path.write_text(
                        json.dumps(q_rows, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                render_quadruped(
                    rows,
                    config_path=cfg_path,
                    out_root=render_root,
                    force=force,
                )
            elif plat == "stretch":
                render_stretch(
                    rows, render_root / "stretch" / "gif", force=force
                )
            elif plat == "roomba":
                render_roomba(rows, render_root / "roomba" / "gif", force=force)
        except Exception as exc:  # noqa: BLE001
            print(f"[render] {plat} error: {exc}", flush=True)
        for cue, idx in before.items():
            gif = gif_path_for(render_root, plat, idx, cue)
            if not gif.is_file():
                skip.add((plat, cue))
                print(f"[render] skip further retries for {plat}/{cue}", flush=True)
    return len(batch)


def render_watch_loop(
    *,
    out_json: Path,
    out_render: Path,
    html_out: Path,
    model: str,
    stop_path: Path,
    poll_s: float = 4.0,
) -> None:
    """Background process: drain pending renders + refresh HTML until stop file."""
    master = out_json / "embodiment_motion_code_all.json"
    title = f"Embodiment Motion Code — Gemini ({model})"
    while not stop_path.is_file():
        if master.is_file():
            try:
                configs = json.loads(master.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                time.sleep(poll_s)
                continue
            try:
                n = render_pending_once(
                    configs,
                    render_root=out_render,
                    out_json=out_json,
                    max_items=2,
                )
                build_live_html(
                    configs,
                    render_root=out_render,
                    out_html=html_out,
                    title=title,
                    model=model,
                )
                if n:
                    print(f"[render] wrote html gifs+={n} → {html_out}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[render] error: {exc}", flush=True)
        time.sleep(poll_s)
    # final pass
    if master.is_file():
        configs = json.loads(master.read_text(encoding="utf-8"))
        while (
            render_pending_once(
                configs, render_root=out_render, out_json=out_json, max_items=8
            )
            > 0
        ):
            configs = json.loads(master.read_text(encoding="utf-8"))
        build_live_html(
            configs,
            render_root=out_render,
            out_html=html_out,
            title=title,
            model=model,
        )
    print("[render] watcher stop", flush=True)


def generate_one(
    *,
    platform: str,
    cue_row: dict[str, Any],
    cues: list[dict[str, Any]],
    fewshot_path: Path,
    model: str,
    backend: str,
    sleep_s: float,
    prompt_style: str = "yaml_cue",
) -> dict[str, Any]:
    cue = cue_row["cue"]
    description = str(cue_row.get("description") or "")
    prompt = build_prompt(
        platform=platform,
        cue=cue,
        description=description,
        cues=cues,
        fewshot_path=fewshot_path,
        category=str(cue_row.get("category") or ""),
        prompt_style=prompt_style,
    )
    if backend == "gemini":
        raw = gemini_generate(prompt, model=model)
    else:
        raise ValueError(f"unsupported backend={backend!r} (use gemini)")

    obj = _extract_json_obj(raw)
    code = extract_motion_code(raw, obj)
    if not code and obj and isinstance(obj.get("movements"), list):
        # JSON-only fallback: keep movements, synthesize empty source
        converted = {
            "movements": obj["movements"],
            "parse_ok": True,
            "notes": ["json_movements_only"],
            "source_code": "",
        }
    else:
        converted = convert_code(code, platform=platform)

    now = datetime.now().isoformat(timespec="seconds")
    domain = {
        "single_arm": "manipulator",
        "tiago": "tiago",
        "stretch": "stretch",
        "humanoid": "unitree_g1",
        "quadruped": "unitree_go2",
        "roomba": "roomba",
    }[platform]
    row = {
        "idx": int(cue_row["idx"]),
        "cue": cue,
        "description": description or f"Gemini embodiment ({platform}) for {cue}",
        "movements": converted["movements"],
        "state": "motion_code_gemini",
        "model": model,
        "time": now,
        "experiment": "embodiment_motion_code_gemini",
        "generation_valid": bool(converted["movements"]),
        "platform": platform,
        "domain": domain,
        "parse_ok": converted["parse_ok"],
        "notes": converted["notes"],
        "source_code": converted.get("source_code") or code,
        "raw_response": raw,
    }
    if sleep_s > 0:
        time.sleep(sleep_s)
    return row


def main() -> None:
    import subprocess

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cues", type=Path, default=DEFAULT_CUES)
    ap.add_argument("--fewshot", type=Path, default=DEFAULT_FEWSHOT)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-render", type=Path, default=None)
    ap.add_argument("--html-out", type=Path, default=None)
    ap.add_argument("--fill-md", type=Path, default=None)
    ap.add_argument(
        "--platforms",
        default=",".join(PLATFORMS),
        help="Comma list of platforms (canonical order by default)",
    )
    ap.add_argument("--max-cues", type=int, default=0)
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--backend", default="gemini", choices=["gemini"])
    ap.add_argument(
        "--prompt-style",
        default="yaml_cue",
        choices=["yaml_cue", "exp0"],
        help="yaml_cue: embodiment YAML + single cue only; exp0: legacy prompt_exp0 + catalog",
    )
    ap.add_argument("--sleep", type=float, default=0.35, help="Seconds between API calls")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip cues that already exist with non-empty source_code in out-json",
    )
    ap.add_argument(
        "--live-render",
        action="store_true",
        help="Spawn background render+HTML watcher while generating",
    )
    ap.add_argument(
        "--no-live-render",
        action="store_true",
        help="Disable background render watcher",
    )
    ap.add_argument(
        "--render-watch-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = ap.parse_args()

    tag = _model_tag(args.model)
    out_json = args.out_json or (DEFAULT_OUT.parent / f"embodiment_motion_code_gemini_{tag}")
    out_render = args.out_render or (
        DEFAULT_OUT_RENDER.parent / f"embodiment_motion_code_gemini_{tag}"
    )
    html_out = args.html_out or (
        DEFAULT_HTML.parent / f"embodiment_motion_code_gemini_{tag}.html"
    )

    if args.render_watch_only:
        stop_path = out_json / "_render_stop"
        render_watch_loop(
            out_json=out_json,
            out_render=out_render,
            html_out=html_out,
            model=args.model,
            stop_path=stop_path,
            poll_s=3.0,
        )
        return

    fill_md = args.fill_md or (
        _REPO
        / "data/seed/agent_fill/embodiment_short"
        / f"codes_fill_gemini_{tag}_{datetime.now().strftime('%Y%m%d')}.md"
    )

    cues = load_cues(args.cues)
    if args.max_cues > 0:
        cues = cues[: args.max_cues]
    platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]
    for p in platforms:
        if p not in PLATFORMS:
            raise SystemExit(f"unknown platform: {p}")
        if args.prompt_style == "exp0" and not PROMPT_PATHS[p].is_file():
            raise SystemExit(f"missing prompt: {PROMPT_PATHS[p]}")
    if args.prompt_style == "yaml_cue" and not SCHEMA_YAML.is_file():
        raise SystemExit(f"missing schema YAML: {SCHEMA_YAML}")

    out_json.mkdir(parents=True, exist_ok=True)
    out_render.mkdir(parents=True, exist_ok=True)
    print(f"out-json: {out_json}", flush=True)
    print(f"out-render: {out_render}", flush=True)
    print(f"html: {html_out}", flush=True)
    print(f"fill-md: {fill_md}", flush=True)

    live = args.live_render or not args.no_live_render
    stop_path = out_json / "_render_stop"
    if stop_path.is_file():
        stop_path.unlink()
    render_proc: subprocess.Popen | None = None
    if live:
        # Separate interpreter (not multiprocessing spawn) so macOS does not
        # re-enter this script's ``main()`` and duplicate API calls.
        render_proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--render-watch-only",
                "--out-json",
                str(out_json),
                "--out-render",
                str(out_render),
                "--html-out",
                str(html_out),
                "--model",
                args.model,
            ],
            env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "glfw")},
        )
        print(f"[render] watcher pid={render_proc.pid}", flush=True)

    by_cue: dict[str, dict[str, Any]] = {}
    existing_master = out_json / "embodiment_motion_code_all.json"
    if args.resume and existing_master.is_file():
        for entry in json.loads(existing_master.read_text(encoding="utf-8")):
            by_cue[entry["cue"]] = entry

    n_done = 0
    n_fail = 0
    try:
        for plat in platforms:
            print(f"=== {plat} ===", flush=True)
            for cue_row in cues:
                cue = cue_row["cue"]
                entry = by_cue.setdefault(
                    cue,
                    {
                        "idx": int(cue_row["idx"]),
                        "cue": cue,
                        "category": cue_row.get("category", ""),
                        "platforms": {},
                    },
                )
                prev = entry["platforms"].get(plat)
                if (
                    args.resume
                    and prev
                    and str(prev.get("source_code") or "").strip()
                    and prev.get("parse_ok")
                ):
                    print(f"  skip {cue}", flush=True)
                    continue
                print(f"  gen {cue} ...", flush=True)
                try:
                    row = generate_one(
                        platform=plat,
                        cue_row=cue_row,
                        cues=cues,
                        fewshot_path=args.fewshot,
                        model=args.model,
                        backend=args.backend,
                        sleep_s=args.sleep,
                        prompt_style=args.prompt_style,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  FAIL {cue}: {exc}", flush=True)
                    n_fail += 1
                    msg = str(exc)
                    if "RESOURCE_EXHAUSTED" in msg or "spending cap" in msg.lower():
                        print(
                            "[abort] Gemini spend/quota exhausted — stopping remaining calls.",
                            flush=True,
                        )
                        # do not write empty fail row for this cue; keep prior state
                        raise SystemExit(2) from exc
                    row = {
                        "idx": int(cue_row["idx"]),
                        "cue": cue,
                        "description": str(cue_row.get("description") or ""),
                        "movements": [],
                        "state": "motion_code_gemini",
                        "model": args.model,
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "experiment": "embodiment_motion_code_gemini",
                        "generation_valid": False,
                        "platform": plat,
                        "parse_ok": False,
                        "notes": [f"error: {exc}"],
                        "source_code": "",
                        "raw_response": "",
                    }
                raw_dir = out_json / "_raw" / plat
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / f"{cue}.txt").write_text(
                    row.pop("raw_response", "") or "", encoding="utf-8"
                )
                entry["platforms"][plat] = row
                configs = [by_cue[c["cue"]] for c in cues if c["cue"] in by_cue]
                write_json_files(configs, out_json)
                write_fill_md(configs, cues, platforms, fill_md)
                build_live_html(
                    configs,
                    render_root=out_render,
                    out_html=html_out,
                    title=f"Embodiment Motion Code — Gemini ({args.model})",
                    model=args.model,
                )
                n_done += 1
                ok = bool(row.get("parse_ok"))
                print(
                    f"  saved {plat}/{cue} parse_ok={ok} code={str(row.get('source_code') or '')[:80]!r}",
                    flush=True,
                )
    finally:
        if live:
            stop_path.write_text("stop\n", encoding="utf-8")
            if render_proc is not None:
                try:
                    render_proc.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    render_proc.terminate()
                    render_proc.wait(timeout=30)

    configs = [by_cue[c["cue"]] for c in cues if c["cue"] in by_cue]
    master = write_json_files(configs, out_json)
    write_fill_md(configs, cues, platforms, fill_md)
    # drain remaining renders in-process
    while (
        render_pending_once(
            configs, render_root=out_render, out_json=out_json, max_items=10
        )
        > 0
    ):
        pass
    build_live_html(
        configs,
        render_root=out_render,
        out_html=html_out,
        title=f"Embodiment Motion Code — Gemini ({args.model})",
        model=args.model,
    )
    print(f"JSON: {master}", flush=True)
    print(f"HTML: {html_out}", flush=True)
    print(f"done_calls={n_done} fail={n_fail}", flush=True)
    for plat in platforms:
        ok = sum(
            1
            for e in configs
            if e.get("platforms", {}).get(plat, {}).get("parse_ok")
        )
        print(f"  {plat}: parse_ok={ok}/{len(configs)}")


if __name__ == "__main__":
    main()
