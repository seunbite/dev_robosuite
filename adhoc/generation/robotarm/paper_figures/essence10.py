"""Essence-10: reasoning about expressible actions, generate configs, render HTML."""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROBOTARM = _HERE.parent
_REPO = _ROBOTARM.parents[2]
for p in (_REPO, _ROBOTARM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from paper_figures._media import pick_gif, repo  # noqa: E402

MANIFEST = _REPO / "data/seed/yml/pilot100_manifest.tsv"
ESSENCE_PROMPT = _REPO / "data/seed/prompt/manipulator/prompt_essence10.txt"
ESSENCE_SHOTS = _REPO / "data/seed/shots/manipulator/shot_configs_v19_sophisticated.json"
OUT_CFG = _REPO / "data/results/motion_configs/manipulator/motion_configs_essence10_pilot100.json"
OUT_DIR = _REPO / "data/results/paper_figures/essence10"


def essence_cues() -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 4 and parts[1] == "pending_essence10":
            out.append((int(parts[2]), parts[3]))
    return out


def generate_one(
    cue: str,
    cue_idx: int,
    model: str,
    config_json: Path,
    *,
    max_tries: int = 3,
) -> dict[str, Any] | None:
    from legacy.config_gen_single import generate_motion_config

    last_err: Exception | None = None
    for try_i in range(max_tries):
        try:
            generate_motion_config(
                cue_name=cue,
                cue_idx=cue_idx,
                model_name=model,
                prompt_file=str(ESSENCE_PROMPT),
                config_json=str(config_json),
                shots_json=str(ESSENCE_SHOTS),
                use_shots=True,
                require_reasoning=True,
                max_attempts=4,
            )
            rows = json.loads(config_json.read_text(encoding="utf-8"))
            row = next(r for r in rows if r.get("cue") == cue and int(r.get("idx", -1)) == cue_idx)
            if row.get("movements"):
                return row
            raise ValueError("saved config has empty movements")
        except Exception as e:
            last_err = e
            print(f"  [essence retry {try_i + 1}/{max_tries}] {cue}: {e}", flush=True)
    print(f"[essence FAIL] {cue}: {last_err}", flush=True)
    return None


def _render_row(cfg: dict[str, Any], gif_dir: Path) -> Path | None:
    import shutil

    gif_dir.mkdir(parents=True, exist_ok=True)
    out = gif_dir / f"c{cfg['idx']}_{cfg['cue']}.gif"
    if out.is_file():
        return out
    tmp_cfg = gif_dir / "_tmp_render.json"
    tmp_cfg.write_text(json.dumps([cfg], indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        from legacy.motion_generation_core import generate

        scratch = gif_dir / "_render_scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        jpath = str(_REPO / "data/seed/_remainder/closest_poses_results.jsonl")
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
        )
        robot_dir = scratch / "IIWA"
        gifs = sorted(robot_dir.glob("*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)
        if gifs:
            shutil.copy2(gifs[0], out)
            return out
    except Exception as e:
        print(f"[render] {cfg['cue']}: {e}", flush=True)
    return None


def build_html(configs: list[dict[str, Any]], gifs: dict[str, Path | None], out_html: Path) -> None:
    cards = []
    for cfg in configs:
        cue = html.escape(str(cfg.get("cue", "?")))
        idx = cfg.get("idx")
        reasoning = html.escape(str(cfg.get("reasoning", "")))
        gif = gifs.get(str(cfg["cue"]))
        rel = os.path.relpath(gif, out_html.parent) if gif and gif.is_file() else ""
        media = (
            f'<img src="{rel}" loading="lazy">' if rel else "<div>No render</div>"
        )
        cards.append(
            f"<article><h3>c{idx} {cue}</h3>{media}"
            f"<pre class='reason'>{reasoning}</pre>"
            f"<pre class='cfg'>{html.escape(json.dumps(cfg.get('movements'), indent=2, ensure_ascii=False))}</pre></article>"
        )
    doc = (
        "<html><head><meta charset='utf-8'><title>Essence-10</title>"
        "<style>body{font-family:sans-serif;margin:16px}article{border:1px solid #ccc;margin:12px 0;padding:12px}"
        "img{max-width:480px;height:auto;display:block;margin:8px 0}"
        "pre.reason{background:#f0f4ff;padding:8px;white-space:pre-wrap;font-size:11px}"
        "pre.cfg{background:#f7f7f7;padding:8px;font-size:10px;max-height:240px;overflow:auto}</style></head><body>"
        f"<h1>Essence-10 pilot cues ({len(configs)}/{len(essence_cues())})</h1>{''.join(cards)}</body></html>"
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(doc, encoding="utf-8")


def run(
    *,
    model: str = "gemini-2.5-pro",
    skip_generate: bool = False,
    skip_render: bool = False,
    open_html: bool = True,
    cues_filter: str | None = None,
) -> Path:
    if not os.getenv("GOOGLE_API_KEY") and not skip_generate:
        raise SystemExit("Set GOOGLE_API_KEY for generation")

    cues = essence_cues()
    if cues_filter:
        want = {c.strip() for c in cues_filter.split(",") if c.strip()}
        cues = [(i, c) for i, c in cues if c in want]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not OUT_CFG.is_file():
        OUT_CFG.write_text("[]", encoding="utf-8")

    configs: list[dict[str, Any]] = []
    existing: dict[str, dict[str, Any]] = {}
    if OUT_CFG.is_file():
        for r in json.loads(OUT_CFG.read_text(encoding="utf-8")):
            if r.get("cue") and r.get("movements"):
                existing[str(r["cue"])] = r

    if skip_generate:
        want = {c for _, c in cues}
        configs = [r for r in existing.values() if r.get("cue") in want]
    else:
        for idx, cue in cues:
            if cue in existing:
                print(f"[essence] skip existing c{idx} {cue}", flush=True)
                configs.append(existing[cue])
                continue
            print(f"[essence] generate c{idx} {cue} (model={model})", flush=True)
            cfg = generate_one(cue, idx, model, OUT_CFG)
            if cfg:
                configs.append(cfg)
                existing[cue] = cfg

    gif_dir = OUT_DIR / "gifs"
    gifs: dict[str, Path | None] = {}
    if not skip_render:
        for cfg in configs:
            gifs[str(cfg["cue"])] = _render_row(cfg, gif_dir)
    else:
        for cfg in configs:
            gifs[str(cfg["cue"])] = pick_gif(str(cfg["cue"]))

    html_path = OUT_DIR / "essence10_review.html"
    build_html(configs, gifs, html_path)
    print(f"Wrote {html_path} ({len(configs)} configs)")
    if open_html:
        import webbrowser

        webbrowser.open(html_path.resolve().as_uri())
    return html_path


def main() -> None:
    p = argparse.ArgumentParser(description="Essence-10 generation + HTML review")
    p.add_argument("--model", default="gemini-2.5-pro")
    p.add_argument("--cues", default=None, help="Comma-separated cue names to run (default: all 10)")
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--skip-render", action="store_true")
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()
    run(
        model=args.model,
        skip_generate=args.skip_generate,
        skip_render=args.skip_render,
        open_html=not args.no_open,
        cues_filter=args.cues,
    )


if __name__ == "__main__":
    main()
