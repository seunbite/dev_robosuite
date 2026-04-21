import json
import os
from datetime import datetime
from typing import Any, Dict, List

import fire
from google import genai

from config_persona_variation import (
    _esc,
    _load_json_list,
    _open_preview,
    _render_config,
    _select_configs,
    apply_variation_spec,
    design_variation_spec,
)


def _write_triptych_html(
    *,
    output_path: str,
    title: str,
    rows: List[Dict[str, Any]],
    persona_a_name: str,
    persona_b_name: str,
) -> str:
    html_dir = os.path.dirname(output_path)
    parts = [f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_esc(title)}</title>
<style>
:root {{
  --bg: #f5f6f8; --surface: #ffffff; --surface2: #eef1f5;
  --border: #d5d9e0; --text: #1f2328; --text2: #5b6472; --accent: #0969da;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, 'SF Pro Text', 'Segoe UI', sans-serif; }}
.wrap {{ max-width: 1800px; margin: 0 auto; padding: 24px; }}
.hero {{ margin-bottom: 18px; }}
.hero h1 {{ margin: 0 0 8px; font-size: 28px; }}
.hero p {{ margin: 0; color: var(--text2); }}
.row {{ margin-bottom: 22px; background: var(--surface); border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }}
.cue-header {{ padding: 14px 16px; border-bottom: 1px solid var(--border); background: var(--surface2); font-weight: 700; font-size: 18px; }}
.cue-header .idx {{ color: var(--accent); margin-right: 8px; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; }}
.col {{ padding: 14px; border-right: 1px solid var(--border); }}
.col:last-child {{ border-right: 0; }}
.label {{ margin-bottom: 10px; font-size: 15px; font-weight: 700; color: var(--text2); text-transform: uppercase; letter-spacing: 0.04em; }}
.gif {{ min-height: 220px; display: flex; align-items: center; justify-content: center; margin-bottom: 12px; background: var(--surface2); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
.gif img {{ display: block; max-width: 100%; }}
.na {{ color: var(--text2); font-style: italic; }}
.block-title {{ margin: 12px 0 6px; font-size: 12px; font-weight: 700; color: var(--text2); text-transform: uppercase; letter-spacing: 0.05em; }}
pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; background: var(--surface2); border-radius: 8px; padding: 10px; font-size: 12px; border: 1px solid var(--border); }}
@media (max-width: 1180px) {{
  .grid {{ grid-template-columns: 1fr; }}
  .col {{ border-right: 0; border-bottom: 1px solid var(--border); }}
  .col:last-child {{ border-bottom: 0; }}
}}
</style>
</head>
<body>
<div class="wrap">
  <section class="hero">
    <h1>{_esc(title)}</h1>
    <p>Columns: base vs { _esc(persona_a_name) } vs { _esc(persona_b_name) }. Rendered at higher hz to make timing/persona differences easier to see.</p>
  </section>
"""]

    for row in rows:
        parts.append(f'<section class="row"><div class="cue-header"><span class="idx">c{row["idx"]}</span>{_esc(row["cue"])}</div><div class="grid">')
        for col in row["cols"]:
            gif = col.get("gif")
            rel = _esc(os.path.relpath(gif, html_dir)) if gif else ""
            parts.append(f"""
  <div class="col">
    <div class="label">{_esc(col['label'])}</div>
    <div class="gif">{f'<img src="{rel}" alt="{_esc(col["label"])} c{row["idx"]}">' if gif else '<span class="na">No render found</span>'}</div>
    <div class="block-title">Description</div>
    <pre>{_esc(col['cfg'].get('description', ''))}</pre>
    <div class="block-title">Movements</div>
    <pre>{_esc(json.dumps(col['cfg'].get('movements', []), indent=2, ensure_ascii=False))}</pre>
    {f'<div class="block-title">Variation Spec</div><pre>{_esc(json.dumps(col["spec"], indent=2, ensure_ascii=False))}</pre>' if col.get('spec') else ''}
  </div>
""")
        parts.append("</div></section>")
    parts.append("</div></body></html>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    return output_path


def main(
    input_json: str,
    cue_idxs: List[int] | None = None,
    output_prefix: str | None = None,
    persona_a_name: str = "anxious",
    persona_a_description: str = "slightly hesitant, unstable, a bit fidgety, with small wavering motion and uneven timing",
    persona_b_name: str = "joyful",
    persona_b_description: str = "energetic, buoyant, expressive, upbeat, with larger amplitude, brighter timing, and confident flourish",
    model_name: str = "gemini-3.1-flash-lite-preview",
    edit_strength: str = "medium",
    temperature: float = 0.3,
    render_robot: str = "IIWA",
    render_top_k: int = 1,
    render_hz: int = 10,
    render_speed_scale: float = 0.8,
    render_hold_scale: float = 1.0,
    render_max_hold_time: float | None = None,
    open_html: bool = True,
):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable.")

    base_data = _load_json_list(input_json)
    selected = _select_configs(base_data, cue_idxs)
    if not selected:
        raise ValueError("No configs selected.")

    if output_prefix is None:
        stem, _ = os.path.splitext(os.path.abspath(input_json))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"{stem}_triptych_{ts}"

    client = genai.Client(api_key=api_key)
    persona_a_by_idx: Dict[int, Dict[str, Any]] = {}
    persona_b_by_idx: Dict[int, Dict[str, Any]] = {}

    print(f"selected_cues={len(selected)}")
    print(f"personas=[{persona_a_name}, {persona_b_name}] render_hz={render_hz} speed_scale={render_speed_scale}")

    for cfg in selected:
        idx = int(cfg["idx"])
        cue = cfg.get("cue", f"cue_{idx}")
        print(f"\nDesigning c{idx}: {cue}")

        spec_a = design_variation_spec(
            cfg,
            persona_name=persona_a_name,
            persona_description=persona_a_description,
            edit_strength=edit_strength,
            client=client,
            model_name=model_name,
            temperature=temperature,
        )
        edited_a = apply_variation_spec(cfg, spec_a)
        edited_a["persona_variation_spec"] = spec_a
        persona_a_by_idx[idx] = edited_a

        spec_b = design_variation_spec(
            cfg,
            persona_name=persona_b_name,
            persona_description=persona_b_description,
            edit_strength=edit_strength,
            client=client,
            model_name=model_name,
            temperature=temperature,
        )
        edited_b = apply_variation_spec(cfg, spec_b)
        edited_b["persona_variation_spec"] = spec_b
        persona_b_by_idx[idx] = edited_b

        print(f"  {persona_a_name}: {spec_a['path_variation']['style']}")
        print(f"  {persona_b_name}: {spec_b['path_variation']['style']}")

    persona_a_json = f"{output_prefix}_{persona_a_name}.json"
    persona_b_json = f"{output_prefix}_{persona_b_name}.json"
    with open(persona_a_json, "w", encoding="utf-8") as f:
        json.dump([{**cfg, **persona_a_by_idx.get(cfg.get('idx'), {})} if cfg.get('idx') in persona_a_by_idx else cfg for cfg in base_data], f, indent=2, ensure_ascii=False)
    with open(persona_b_json, "w", encoding="utf-8") as f:
        json.dump([{**cfg, **persona_b_by_idx.get(cfg.get('idx'), {})} if cfg.get('idx') in persona_b_by_idx else cfg for cfg in base_data], f, indent=2, ensure_ascii=False)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    render_root = f"{output_prefix}_renders"
    rows = []
    for cfg in selected:
        idx = int(cfg["idx"])
        cue = cfg["cue"]
        print(f"\nRendering c{idx}: {cue}")
        base_gif = _render_config(
            script_dir=script_dir,
            config_path=os.path.abspath(input_json),
            cue_idx=idx,
            cue_name=cue,
            robot=render_robot,
            output_dir=os.path.join(render_root, "base"),
            top_k=render_top_k,
            hz=render_hz,
            preview_speed_scale=render_speed_scale,
            preview_hold_scale=render_hold_scale,
            preview_max_hold_time=render_max_hold_time,
        )
        anxious_gif = _render_config(
            script_dir=script_dir,
            config_path=os.path.abspath(persona_a_json),
            cue_idx=idx,
            cue_name=cue,
            robot=render_robot,
            output_dir=os.path.join(render_root, persona_a_name),
            top_k=render_top_k,
            hz=render_hz,
            preview_speed_scale=render_speed_scale,
            preview_hold_scale=render_hold_scale,
            preview_max_hold_time=render_max_hold_time,
        )
        joyful_gif = _render_config(
            script_dir=script_dir,
            config_path=os.path.abspath(persona_b_json),
            cue_idx=idx,
            cue_name=cue,
            robot=render_robot,
            output_dir=os.path.join(render_root, persona_b_name),
            top_k=render_top_k,
            hz=render_hz,
            preview_speed_scale=render_speed_scale,
            preview_hold_scale=render_hold_scale,
            preview_max_hold_time=render_max_hold_time,
        )
        rows.append({
            "idx": idx,
            "cue": cue,
            "cols": [
                {"label": "Base", "cfg": cfg, "gif": base_gif, "spec": None},
                {"label": persona_a_name, "cfg": persona_a_by_idx[idx], "gif": anxious_gif, "spec": persona_a_by_idx[idx]["persona_variation_spec"]},
                {"label": persona_b_name, "cfg": persona_b_by_idx[idx], "gif": joyful_gif, "spec": persona_b_by_idx[idx]["persona_variation_spec"]},
            ],
        })

    html_path = f"{output_prefix}_triptych.html"
    _write_triptych_html(
        output_path=html_path,
        title=f"Persona Triptych: {persona_a_name} vs {persona_b_name}",
        rows=rows,
        persona_a_name=persona_a_name,
        persona_b_name=persona_b_name,
    )
    html_abs = os.path.abspath(html_path)
    print(f"\nTriptych HTML: {html_abs}")
    print(f"Triptych URL: file://{html_abs}")
    if open_html:
        _open_preview(html_abs)


if __name__ == "__main__":
    fire.Fire(main)
