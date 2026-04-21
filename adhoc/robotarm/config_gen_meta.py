import fire
import glob
import json
import os
import re
import sys
import time
import yaml
import subprocess
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


def _prompt_suffix(prompt_file: str) -> str | None:
    name = os.path.basename(prompt_file)
    if not name.endswith(".txt"):
        return None
    stem = name[:-4]
    if stem.startswith("prompt_"):
        return stem[len("prompt_"):]
    return stem


def _resolve_preview_output_dir(prompt_file: str, cue_group: str = "iconic") -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prompt_suffix = _prompt_suffix(prompt_file)
    if prompt_suffix is not None:
        suffix = prompt_suffix if cue_group == "iconic" else f"{prompt_suffix}_{cue_group}"
        return os.path.join(repo_root, "data", "motions", suffix)
    return os.path.join(repo_root, "data", "motions", "generate_first")


def _safe_cue_name(cue: str) -> str:
    return cue.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _find_latest_tiled_gif(render_dir: str, cue: str, cue_idx: int | None = None) -> str | None:
    matches = []
    if cue_idx is not None:
        matches.extend(glob.glob(os.path.join(render_dir, f"*_c{cue_idx}_tiled.gif")))
    safe_cue = _safe_cue_name(cue)
    matches.extend(glob.glob(os.path.join(render_dir, f"*_{safe_cue}_tiled.gif")))
    matches.extend(glob.glob(os.path.join(render_dir, f"*_{safe_cue}_*_tiled.gif")))
    matches.extend(glob.glob(os.path.join(render_dir, f"*{safe_cue}*tiled.gif")))
    if not matches:
        return None
    matches = sorted(set(matches), key=os.path.getmtime, reverse=True)
    return matches[0]


def _open_preview(path: str) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        print(f"  ⚠ Preview popup failed: {e}")


def _combine_tiled_gif_preview(entries: list[tuple[int, str, str]], output_path: str, hz: int = 8) -> str | None:
    """Stack existing per-cue tiled GIFs vertically into one preview GIF."""
    if not entries:
        return None

    opened = []
    try:
        for cue_idx, cue, gif_path in entries:
            img = Image.open(gif_path)
            opened.append((cue_idx, cue, img))

        tile_w = max(img.size[0] for _, _, img in opened)
        tile_h = max(img.size[1] for _, _, img in opened)
        label_w = 260
        total_w = label_w + tile_w
        total_h = tile_h * len(opened)
        max_frames = max(getattr(img, "n_frames", 1) for _, _, img in opened)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except Exception:
            font = ImageFont.load_default()

        frames = []
        for frame_i in range(max_frames):
            canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)

            for row, (cue_idx, cue, gif) in enumerate(opened):
                y = row * tile_h
                label = f"c{cue_idx} {cue}"
                draw.text((12, y + max(0, (tile_h // 2) - 12)), label, fill="black", font=font)
                draw.line([(0, y), (total_w, y)], fill=(220, 220, 220), width=1)

                n_frames = getattr(gif, "n_frames", 1)
                gif.seek(frame_i % n_frames)
                frame = gif.copy().convert("RGB")
                if frame.size != (tile_w, tile_h):
                    frame = frame.resize((tile_w, tile_h))
                canvas.paste(frame, (label_w, y))

            frames.append(canvas)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / hz),
            loop=0,
            disposal=1,
        )
        return output_path
    finally:
        for _, _, img in opened:
            try:
                img.close()
            except Exception:
                pass


def _render_preview_batch(
    preview_items: list[tuple[int, str]],
    *,
    prompt_file: str,
    config_json: str,
    cue_group: str,
    python_bin: str,
    top_k: int,
    robot: str,
    batch_label: str,
) -> str | None:
    """Render a preview batch as tiled per-cue GIFs plus one combined preview GIF."""
    if not preview_items:
        return None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    motion_script = os.path.join(script_dir, "motion_generation.py")
    output_dir = _resolve_preview_output_dir(prompt_file, cue_group=cue_group)
    render_dir = os.path.join(output_dir, robot)
    os.makedirs(render_dir, exist_ok=True)

    rendered_entries: list[tuple[int, str, str]] = []
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", batch_label).strip("_") or "preview"
    print(f"\n── {batch_label} preview ({robot}, top_k={top_k}) ──")

    for cue_idx, cue in preview_items:
        print(f"  ▶ c{cue_idx}: {cue}")
        cmd = [
            python_bin,
            motion_script,
            f"--robot={robot}",
            f"--cue_idx={cue_idx}",
            f"--config_path={config_json}",
            f"--output_dir={output_dir}",
            f"--top_k={top_k}",
        ]
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=600)
        if result.returncode != 0:
            stderr_tail = result.stderr.strip().splitlines()[-3:] if result.stderr else []
            print(f"    ✗ render failed")
            for line in stderr_tail:
                print(f"      {line}")
            continue

        gif_path = _find_latest_tiled_gif(render_dir, cue, cue_idx=cue_idx)
        if not gif_path:
            print(f"    ✗ tiled GIF not found")
            continue

        print(f"    ✓ {os.path.basename(gif_path)}")
        rendered_entries.append((cue_idx, cue, gif_path))

    if not rendered_entries:
        print("  ⚠ No preview GIFs rendered.")
        return None

    ts = time.strftime("%Y%m%d_%H%M%S")
    preview_path = os.path.join(render_dir, f"{ts}_{robot}_{safe_label}_preview.gif")
    combined = _combine_tiled_gif_preview(rendered_entries, preview_path, hz=8)
    if combined:
        print(f"  ✓ combined preview: {combined}")
        _open_preview(combined)
    return combined


def _load_configs_by_idx(config_json: str) -> dict[int, dict]:
    if not os.path.exists(config_json):
        return {}
    with open(config_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        cfg["idx"]: cfg
        for cfg in data
        if isinstance(cfg, dict) and isinstance(cfg.get("idx"), int)
    }


def _write_generated_html_dashboard(
    *,
    config_json: str,
    prompt_file: str,
    cue_group: str,
    cue_items: list[tuple[int, str]],
    robot: str,
    output_path: str | None = None,
) -> str | None:
    if not cue_items:
        return None

    configs_by_idx = _load_configs_by_idx(config_json)
    selected = []
    for cue_idx, cue_name in cue_items:
        cfg = configs_by_idx.get(cue_idx)
        if cfg is None:
            continue
        selected.append((cue_idx, cue_name, cfg))

    if not selected:
        return None

    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except Exception:
        prompt_text = ""

    if output_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        config_stem = os.path.splitext(os.path.basename(config_json))[0]
        output_path = os.path.join(
            os.path.dirname(config_json),
            f"{config_stem}_generated_{cue_group}_{robot}_{ts}.html",
        )

    def esc(text: str) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    prompt_display = prompt_text.split("{{FEW_SHOT_EXAMPLES}}")[0] if "{{FEW_SHOT_EXAMPLES}}" in prompt_text else prompt_text
    html_parts = [f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Generated Motion Configs</title>
<style>
:root {{
  --bg: #f6f8fb; --surface: #ffffff; --surface2: #eef2f7;
  --border: #d0d7de; --text: #1f2328; --text2: #59636e;
  --accent: #0969da; --accent2: #1a7f37; --purple: #8250df;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: -apple-system, 'SF Pro Text', 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); }}
.wrap {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
.hero {{ margin-bottom: 20px; }}
.hero h1 {{ margin: 0 0 8px; font-size: 28px; }}
.hero p {{ margin: 0 0 12px; color: var(--text2); }}
.chips {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }}
.chip {{ display: inline-block; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--border); background: var(--surface2); color: var(--text2); font-size: 14px; }}
.prompt {{ margin-bottom: 24px; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
.prompt summary {{ cursor: pointer; padding: 14px 16px; font-weight: 600; background: var(--surface2); }}
.prompt pre {{ margin: 0; padding: 16px; white-space: pre-wrap; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }}
.card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
.card-header {{ padding: 12px 14px; border-bottom: 1px solid var(--border); background: var(--surface2); font-weight: 600; }}
.card-header .idx {{ color: var(--accent); margin-right: 8px; }}
.card-body {{ padding: 14px; }}
.gif {{ margin-bottom: 12px; text-align: center; }}
.gif img {{ display: block; max-width: 100%; border: 1px solid var(--border); border-radius: 8px; background: var(--surface2); }}
.na {{ color: var(--text2); font-style: italic; }}
.steps {{ display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin: 0 0 12px; }}
.step {{ padding: 3px 8px; border-radius: 999px; font-size: 13px; background: #1f6feb22; color: var(--accent); }}
.step.movement {{ background: #2ea04322; color: var(--accent2); }}
.step.path {{ background: #bc8cff22; color: var(--purple); }}
.arrow {{ color: var(--text2); }}
.label {{ margin: 12px 0 4px; font-size: 12px; font-weight: 700; letter-spacing: 0.05em; color: var(--text2); text-transform: uppercase; }}
.cot {{ padding: 10px; border-left: 3px solid var(--purple); background: var(--surface2); border-radius: 6px; white-space: pre-wrap; font-size: 13px; }}
.json {{ margin: 0; padding: 10px; background: var(--surface2); border-radius: 6px; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; white-space: pre-wrap; overflow: auto; max-height: 320px; }}
</style>
</head>
<body>
<div class="wrap">
  <section class="hero">
    <h1>Generated Motion Configs</h1>
    <p>Current run output for the newly generated cues.</p>
    <div class="chips">
      <span class="chip">cue_group: {esc(cue_group)}</span>
      <span class="chip">robot: {esc(robot)}</span>
      <span class="chip">cues: {len(selected)}</span>
      <span class="chip">config: {esc(config_json)}</span>
      <span class="chip">prompt: {esc(prompt_file)}</span>
    </div>
  </section>
  <details class="prompt">
    <summary>Prompt</summary>
    <pre>{esc(prompt_display or "No prompt file found")}</pre>
  </details>
  <section class="grid">
"""]

    html_dir = os.path.dirname(output_path)
    for cue_idx, cue_name, cfg in selected:
        gif_path = _find_latest_tiled_gif(os.path.join(_resolve_preview_output_dir(prompt_file, cue_group=cue_group), robot), cue_name, cue_idx=cue_idx)
        gif_html = f'<img src="{esc(os.path.relpath(gif_path, html_dir))}" loading="lazy" alt="c{cue_idx} preview">' if gif_path else '<div class="na">No render found</div>'
        movements = cfg.get("movements", [])
        steps = []
        for mi, movement in enumerate(movements):
            mtype = movement.get("type", "?")
            label = mtype
            params = movement.get("parameters", {})
            if mtype == "pose":
                label = f"pose({params.get('pose', {}).get('dir', '?')})"
            elif mtype == "movement":
                label = f"move({params.get('joint', '?')})"
            elif mtype == "path":
                label = f"path({params.get('shape', '?')})"
            if mi > 0:
                steps.append('<span class="arrow">→</span>')
            steps.append(f'<span class="step {esc(mtype)}">{esc(label)}</span>')

        reasoning = cfg.get("reasoning") or "No CoT saved"
        display_cfg = {k: v for k, v in cfg.items() if k not in ("state", "time")}
        html_parts.append(f"""
    <article class="card">
      <div class="card-header"><span class="idx">c{cue_idx}</span>{esc(cue_name)}</div>
      <div class="card-body">
        <div class="gif">{gif_html}</div>
        <div class="steps">{''.join(steps) if steps else '<span class="na">No movements</span>'}</div>
        <div class="label">Chain of Thought</div>
        <div class="cot">{esc(reasoning)}</div>
        <div class="label">Config JSON</div>
        <pre class="json">{esc(json.dumps(display_cfg, indent=2, ensure_ascii=False))}</pre>
      </div>
    </article>
""")

    html_parts.append("""
  </section>
</div>
</body>
</html>
""")

    os.makedirs(html_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))
    return output_path


def main(
    model: str = "gemini-3.1-pro-preview",
    yaml_path: str = "data/seed/cues.yml",
    cue_group: str = "iconic",
    prompt_file: str = "data/seed/prompt/prompt_v18.txt",
    config_json: str | None = None,
    shots_json: str = "data/seed/shot_configs_v18.json",
    verbose: bool = False,
    delay: float = 2.0,
    max_retries: int = 2,
    generate_first: list[int] = [6, 31, 44],
    generate_only: list[int] = None,
    top_k: int = 5,
    preview_robot: str = "IIWA",
    preview_every: int = 5,
    show_html: bool = True,
):
    if config_json is None:
        prompt_name = os.path.basename(prompt_file).replace('.txt', '')
        suffix = "" if cue_group == "iconic" else f"_{cue_group}"
        config_json = f"data/seed/motion_configs_{prompt_name}{suffix}.json"

    with open(yaml_path, 'r') as f:
        cues_dict = yaml.safe_load(f)

    if cue_group not in cues_dict:
        raise ValueError(f"cue_group='{cue_group}' not found in {yaml_path}. Available: {sorted(cues_dict.keys())}")

    items = list(cues_dict[cue_group].items())
    indexed_items = list(enumerate(items))
    prompt_name = os.path.basename(prompt_file).replace('.txt', '')
    python_bin = sys.executable
    print(f"Prompt: {prompt_file} | Group: {cue_group} | Output: {config_json} | Cues: {len(items)}")

    generate_first = generate_first or []
    generate_first = list(dict.fromkeys(generate_first))
    valid_first = [idx for idx in generate_first if 0 <= idx < len(indexed_items)]
    invalid_first = [idx for idx in generate_first if idx not in valid_first]
    if invalid_first:
        print(f"Warning: ignoring out-of-range generate_first indexes: {invalid_first}")

    generate_only = generate_only or []
    generate_only = list(dict.fromkeys(generate_only))
    valid_only = [idx for idx in generate_only if 0 <= idx < len(indexed_items)]
    invalid_only = [idx for idx in generate_only if idx not in valid_only]
    if invalid_only:
        print(f"Warning: ignoring out-of-range generate_only indexes: {invalid_only}")

    index_map = {idx: (key, desc) for idx, (key, desc) in indexed_items}
    if valid_only:
        allowed_set = set(valid_only)
        indexed_items = [(idx, item) for idx, item in indexed_items if idx in allowed_set]
        valid_first = [idx for idx in valid_first if idx in allowed_set]
        print(f"generate_only: {valid_only}")
    first_set = set(valid_first)
    first_batch = [(idx, *index_map[idx]) for idx in valid_first]
    rest_batch = [(idx, key, desc) for idx, (key, desc) in indexed_items if idx not in first_set]
    ordered_items = first_batch + rest_batch
    preview_generated: list[tuple[int, str]] = []
    rolling_preview_items: list[tuple[int, str]] = []
    generated_success_items: list[tuple[int, str]] = []
    preview_done = False
    preview_boundary = len(first_batch)
    if first_batch:
        print(f"generate_first: {valid_first} | preview_robot={preview_robot} | top_k={top_k}")

    failed_count = 0
    for order_i, (cue_idx, cue_name, cue_description) in enumerate(
        tqdm(ordered_items, desc=f"[{prompt_name}]", disable=verbose)
    ):
        prompt_cue_name = cue_name
        cmd = [
            python_bin,
            "adhoc/robotarm/config_gen_single.py",
            f"--cue_name={prompt_cue_name}",
            f"--cue_idx={cue_idx}",
            f"--prompt_file={prompt_file}",
            f"--config_json={config_json}",
            f"--shots_json={shots_json}",
            f"--model_name={model}",
        ]

        success = False
        last_err = ""
        for attempt in range(max_retries + 1):
            try:
                if verbose:
                    print(" ".join(cmd))
                    result = subprocess.run(cmd, timeout=180)
                    success = result.returncode == 0
                    last_err = ""
                else:
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, timeout=180,
                    )
                    success = result.returncode == 0
                    last_err = result.stderr.strip() if result.stderr else ""
            except subprocess.TimeoutExpired:
                success = False
                last_err = "Timed out (180s)"

            if success:
                break
            if attempt < max_retries:
                backoff = delay * (2 ** attempt)
                tqdm.write(f"  Retry {attempt+1}/{max_retries} for '{cue_description[:40]}...' (wait {backoff:.0f}s)")
                time.sleep(backoff)

        if not success:
            failed_count += 1
            tqdm.write(f"FAILED: {cue_description}\n{last_err[-200:] if last_err else ''}")
        else:
            generated_success_items.append((cue_idx, cue_description))
            if cue_idx in first_set:
                preview_generated.append((cue_idx, cue_description))
            else:
                rolling_preview_items.append((cue_idx, cue_description))

        if not preview_done and preview_boundary and order_i + 1 == preview_boundary:
            _render_preview_batch(
                preview_generated,
                prompt_file=prompt_file,
                config_json=config_json,
                cue_group=cue_group,
                python_bin=python_bin,
                top_k=top_k,
                robot=preview_robot,
                batch_label="generate_first",
            )
            preview_done = True
        elif success and cue_idx not in first_set and preview_every > 0 and len(rolling_preview_items) >= preview_every:
            batch_items = rolling_preview_items[:preview_every]
            start_idx = batch_items[0][0]
            end_idx = batch_items[-1][0]
            _render_preview_batch(
                batch_items,
                prompt_file=prompt_file,
                config_json=config_json,
                cue_group=cue_group,
                python_bin=python_bin,
                top_k=top_k,
                robot=preview_robot,
                batch_label=f"batch_c{start_idx}_c{end_idx}",
            )
            rolling_preview_items = rolling_preview_items[preview_every:]

        time.sleep(delay)

    if rolling_preview_items:
        start_idx = rolling_preview_items[0][0]
        end_idx = rolling_preview_items[-1][0]
        _render_preview_batch(
            rolling_preview_items,
            prompt_file=prompt_file,
            config_json=config_json,
            cue_group=cue_group,
            python_bin=python_bin,
            top_k=top_k,
            robot=preview_robot,
            batch_label=f"batch_c{start_idx}_c{end_idx}",
        )

    total_attempted = len(ordered_items)
    if failed_count:
        print(f"\n{failed_count}/{total_attempted} cues failed.")
    if show_html and generated_success_items:
        html_path = _write_generated_html_dashboard(
            config_json=config_json,
            prompt_file=prompt_file,
            cue_group=cue_group,
            cue_items=generated_success_items,
            robot=preview_robot,
        )
        if html_path:
            print(f"\nHTML dashboard: {html_path}")
            _open_preview(html_path)


if __name__ == "__main__":
    fire.Fire(main)
