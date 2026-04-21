import json
import re
from html import escape
from pathlib import Path


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SEED = ROOT / "data" / "seed"
OUT = SEED / "prompt19_sophisticated_integrated_20260402_ko.html"
ALPHA_ROOT = SEED / "prompt19_sophisticated_alpha_frames"

SPECS = [
    (
        "iconic",
        "Sophisticated Iconic",
        SEED / "motion_configs_prompt_v19_sophisticated.json",
        ROOT / "data" / "motions" / "v19_sophisticated" / "IIWA",
        "../motions/v19_sophisticated/IIWA",
    ),
    (
        "contextual",
        "Sophisticated Contextual",
        SEED / "motion_configs_prompt_v19_sophisticated_contextual.json",
        ROOT / "data" / "motions" / "v19_sophisticated_contextual" / "IIWA",
        "../motions/v19_sophisticated_contextual/IIWA",
    ),
]


def latest_gif(motion_dir: Path, cue: str):
    matches = [
        p
        for p in motion_dir.glob(f"*_{cue}_*.gif")
        if "batch_" not in p.name and "generate_first_preview" not in p.name
    ]
    if not matches:
        matches = [
            p
            for p in motion_dir.glob(f"*{cue}*.gif")
            if "batch_" not in p.name and "generate_first_preview" not in p.name
        ]
    return sorted(matches, key=lambda p: p.name)[-1] if matches else None


def compact_json_list(items):
    if not items:
        return "[]"
    lines = ["["]
    for i, item in enumerate(items):
        suffix = "," if i < len(items) - 1 else ""
        lines.append(
            f"  {json.dumps(item, ensure_ascii=False, separators=(', ', ': '))}{suffix}"
        )
    lines.append("]")
    return "\n".join(lines)


def colorize_movements(text):
    escaped = escape(text)

    # Bold only the values for selected movement fields.
    value_keys = [
        "dir",
        "x",
        "y",
        "z",
        "gripper_orientation",
        "shape",
        "joint",
        "plane",
        "radius",
        "sweep",
        "speed",
        "direction",
        "repetition",
        "degrees",
        "hold_time",
    ]
    for key in value_keys:
        pattern = rf'(&quot;{key}&quot;: )(&quot;.*?&quot;|-?\d+(?:\.\d+)?|\{{.*?\}})'
        escaped = re.sub(
            pattern,
            lambda m: f'{m.group(1)}<span class="tok-value">{m.group(2)}</span>',
            escaped,
        )

    escaped = escaped.replace(
        '&quot;pose&quot;', '<span class="tok-pose">&quot;pose&quot;</span>'
    )
    escaped = escaped.replace(
        '&quot;movement&quot;', '<span class="tok-movement">&quot;movement&quot;</span>'
    )
    escaped = escaped.replace(
        '&quot;path&quot;', '<span class="tok-path">&quot;path&quot;</span>'
    )
    escaped = escaped.replace(
        '&quot;gripper&quot;', '<span class="tok-gripper">&quot;gripper&quot;</span>'
    )
    return escaped


def summarize_step(step):
    step_type = step.get("type", "?")
    params = step.get("parameters", {})

    if step_type == "pose":
        direction = params.get("pose", {}).get("dir", "?")
        return {"label": f"pose({direction})", "class_name": "pose"}

    if step_type == "movement":
        joint = params.get("joint", "?")
        return {"label": f"move({joint})", "class_name": "movement"}

    if step_type == "path":
        shape = params.get("shape", "?")
        return {"label": f"path({shape})", "class_name": "path"}

    if step_type == "gripper":
        action = params.get("action")
        label = f"gripper({action})" if action else "gripper"
        return {"label": label, "class_name": "gripper"}

    return {"label": step_type, "class_name": ""}


def load_sections():
    sections = []
    all_cards = []
    for kind, label, config_path, motion_dir, motion_rel in SPECS:
        items = json.loads(config_path.read_text())
        items.sort(key=lambda x: x["idx"])
        cards = []
        for item in items:
            cue = item["cue"]
            gif = latest_gif(motion_dir, cue)
            alpha_png = ALPHA_ROOT / kind / f"c{item['idx']}_{cue}.png"
            cards.append(
                {
                    "group": kind,
                    "group_label": label,
                    "idx": item["idx"],
                    "cue": cue,
                    "steps": [summarize_step(step) for step in item.get("movements", [])],
                    "reasoning": item.get("reasoning") or "",
                    "movements": compact_json_list(item.get("movements", [])),
                    "image": (
                        f"./prompt19_sophisticated_alpha_frames/{kind}/c{item['idx']}_{cue}.png"
                        if alpha_png.exists()
                        else (f"{motion_rel}/{gif.name}" if gif else None)
                    ),
                }
            )
        sections.append((label, kind, cards))
        all_cards.extend(cards)
    return sections, all_cards


def build_picker_html(sections):
    parts = []
    for label, kind, cards in sections:
        for card in cards:
            card_id = f"{kind}-c{card['idx']}"
            parts.append(
                f'<label class="cue-pill"><input type="checkbox" data-filter-card="{card_id}">'
                f'{escape(label)} c{card["idx"]} {escape(card["cue"])}</label>'
            )
    return "".join(parts)


def build_card_html(card, kind):
    card_id = f"{kind}-c{card['idx']}"
    media = (
        f'<img src="{escape(card["image"])}" loading="lazy" alt="c{card["idx"]} preview">'
        if card["image"]
        else '<div class="na">GIF not found</div>'
    )
    step_html = []
    for i, step in enumerate(card["steps"]):
        step_html.append(
            f'<span class="step {step["class_name"]}">{escape(step["label"])}</span>'
        )
        if i != len(card["steps"]) - 1:
            step_html.append('<span class="arrow">→</span>')
    steps_block = "".join(step_html) if step_html else '<span class="na">No steps</span>'
    prefix = "I" if kind == "iconic" else "C"
    return (
        f'<article class="card" data-card-id="{card_id}" data-group="{kind}" '
        f'data-cue="{escape(card["cue"])}" data-idx="{card["idx"]}">'
        f'<div class="card-header"><span class="idx">{prefix}c{card["idx"]}</span>{escape(card["cue"])}</div>'
        f'<div class="card-body">'
        f'<div class="gif">{media}</div>'
        f'<div class="steps">{steps_block}</div>'
        f'<div class="label">Chain of Thought</div>'
        f'<div class="cot">{escape(card["reasoning"])}</div>'
        f'<div class="label">Movements</div>'
        f'<pre class="json">{colorize_movements(card["movements"])}</pre>'
        f"</div>"
        f"</article>"
    )


def build_catalog_html(sections):
    parts = []
    for label, kind, cards in sections:
        parts.append(
            f'<section class="catalog-section"><h2 class="catalog-title">{escape(label)}</h2>'
            f'<div class="grid">'
        )
        for card in cards:
            parts.append(build_card_html(card, kind))
        parts.append("</div></section>")
    return "".join(parts)


def build_page():
    sections, all_cards = load_sections()
    picker_html = build_picker_html(sections)
    catalog_html = build_catalog_html(sections)
    iconic_count = sum(1 for c in all_cards if c["group"] == "iconic")
    contextual_count = sum(1 for c in all_cards if c["group"] == "contextual")

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Generated Motion Configs</title>
<style>
:root {{
  --bg: #ffffff; --surface: #fcfcfc; --surface2: #f7f8f9;
  --border: #e3e7eb; --text: #1f2328; --text2: #59636e;
  --accent: #0969da; --accent2: #1a7f37; --purple: #8250df;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: -apple-system, 'SF Pro Text', 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); }}
.wrap {{ max-width: 1440px; margin: 0 auto; padding: 20px; }}
.hero {{ margin-bottom: 16px; }}
.hero h1 {{ margin: 0 0 8px; font-size: 28px; }}
.hero p {{ margin: 0 0 12px; color: var(--text2); }}
.chips {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 12px; }}
.chip {{ display: inline-block; padding: 3px 9px; border-radius: 999px; border: 1px solid var(--border); background: transparent; color: var(--text2); font-size: 13px; }}
.prompt {{ margin-bottom: 18px; background: transparent; border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
.prompt summary {{ cursor: pointer; padding: 12px 14px; font-weight: 600; background: transparent; }}
.prompt pre {{ margin: 0; padding: 16px; white-space: pre-wrap; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; }}
.controls {{ display:flex; gap:10px; align-items:flex-start; margin: 0 0 14px; flex-wrap:wrap; }}
.filter-toggle {{ display:flex; gap:8px; align-items:center; background: transparent; border:1px solid #e8ebef; border-radius: 999px; padding: 7px 11px; }}
.cue-picker {{ background: transparent; border:1px solid #eaedf1; border-radius: 10px; padding: 10px; display:flex; gap:8px; flex-wrap:wrap; max-width: 100%; }}
.cue-pill {{ display:flex; gap:6px; align-items:center; padding: 5px 9px; border-radius: 999px; background: transparent; border:1px solid #e8ebef; font-size: 12px; }}
.selected-zone {{ margin: 0 0 16px; }}
.selected-title {{ margin: 0 0 10px; color: var(--text2); font-size: 14px; font-weight: 600; }}
.selected-row {{ display:flex; gap:12px; overflow:auto; padding-bottom: 2px; }}
.selected-row .card {{ min-width: 320px; max-width: 320px; flex: 0 0 auto; }}
.empty-picked {{ color: var(--text2); font-style: italic; padding: 6px 0; }}
.catalog-section {{ margin: 0 0 16px; }}
.catalog-title {{ margin: 0 0 8px; font-size: 16px; font-weight: 600; color: #39414a; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
.card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; box-shadow: none; }}
.card-header {{ padding: 10px 12px; border-bottom: 1px solid #edf0f2; background: #fcfcfc; font-weight: 600; }}
.card-header .idx {{ color: var(--accent); margin-right: 8px; }}
.card-body {{ padding: 12px; }}
.gif {{ margin-bottom: 10px; text-align: center; }}
.gif img {{ display: block; max-width: 100%; border: 1px solid #eceff2; border-radius: 6px; background: #fafafa; }}
.na {{ color: var(--text2); font-style: italic; }}
.steps {{ display: flex; gap: 5px; align-items: center; flex-wrap: wrap; margin: 0 0 10px; }}
.step {{ padding: 2px 7px; border-radius: 999px; font-size: 12px; background: #1f6feb16; color: var(--accent); }}
.step.movement {{ background: #2ea04316; color: var(--accent2); }}
.step.path {{ background: #bc8cff16; color: var(--purple); }}
.arrow {{ color: var(--text2); }}
.label {{ margin: 10px 0 4px; font-size: 11px; font-weight: 700; letter-spacing: 0.05em; color: var(--text2); text-transform: uppercase; }}
.cot {{ padding: 9px; border-left: 1px solid #ddd8e8; background: #f7f8fa; border-radius: 5px; white-space: pre-wrap; font-size: 12px; }}
.json {{ margin: 0; padding: 9px; background: #f7f8fa; border-radius: 5px; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 11px; white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; overflow-x: hidden; max-height: none; }}
.tok-pose {{ color: var(--accent); font-weight: 600; }}
.tok-movement {{ color: var(--accent2); font-weight: 600; }}
.tok-path {{ color: var(--purple); font-weight: 600; }}
.tok-gripper {{ color: #b42318; font-weight: 600; }}
.tok-value {{ color: #111111; font-weight: 700; }}
.hidden-card {{ display:none !important; }}
@media (max-width: 960px) {{
  .wrap {{ padding: 14px; }}
  .selected-row .card {{ min-width: min(92vw, 320px); max-width: min(92vw, 320px); }}
}}
</style>
</head>
<body>
<div class="wrap">
  <section class="hero">
    <h1>Generated Motion Configs</h1>
    <p>Integrated sophisticated run output for iconic and contextual cues.</p>
    <div class="chips">
      <span class="chip">cue_group: sophisticated integrated</span>
      <span class="chip">robot: IIWA</span>
      <span class="chip">iconic cues: {iconic_count}</span>
      <span class="chip">contextual cues: {contextual_count}</span>
      <span class="chip">config: data/seed/motion_configs_prompt_v19_sophisticated*.json</span>
    </div>
  </section>
  <details class="prompt">
    <summary>Sources</summary>
    <pre>Iconic config: data/seed/motion_configs_prompt_v19_sophisticated.json
Contextual config: data/seed/motion_configs_prompt_v19_sophisticated_contextual.json
Iconic render dir: data/motions/v19_sophisticated/IIWA
Contextual render dir: data/motions/v19_sophisticated_contextual/IIWA

This page follows the v18 card format:
- GIF
- step chips
- Chain of Thought
- Movements

The cue picker supports mixed selection across both groups.
Selected cards are mirrored in a horizontal strip below.</pre>
  </details>
  <section class="controls">
    <label class="filter-toggle"><input id="show-picked-only" type="checkbox"> 선택한 것만 보기</label>
    <span class="chip"><span id="picked-count">0</span> selected</span>
    <div class="cue-picker">{picker_html}</div>
  </section>
  <section class="selected-zone">
    <div class="selected-title">Selected Motions</div>
    <div id="selected-strip" class="selected-row">
      <div id="empty-state" class="empty-picked">위 체크박스에서 cue를 고르면 여기에 가로로 붙습니다.</div>
    </div>
  </section>
  <section class="catalog">{catalog_html}</section>
</div>
<script>
(() => {{
  const STORAGE_KEY = "prompt19_sophisticated_integrated_checks_v1";
  const TOGGLE_KEY = "prompt19_sophisticated_integrated_toggle_v1";
  const checkboxes = Array.from(document.querySelectorAll("input[data-filter-card]"));
  const cards = Array.from(document.querySelectorAll(".card[data-card-id]"));
  const selectedStrip = document.getElementById("selected-strip");
  const emptyState = document.getElementById("empty-state");
  const pickedCount = document.getElementById("picked-count");
  const showPickedOnly = document.getElementById("show-picked-only");

  function loadState() {{
    let picked = [];
    let only = false;
    try {{ picked = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); }} catch (_) {{ picked = []; }}
    try {{ only = JSON.parse(localStorage.getItem(TOGGLE_KEY) || "false"); }} catch (_) {{ only = false; }}
    return {{ picked, only }};
  }}

  function saveState(picked) {{
    localStorage.setItem(STORAGE_KEY, JSON.stringify(picked));
  }}

  function saveToggle(value) {{
    localStorage.setItem(TOGGLE_KEY, JSON.stringify(value));
  }}

  function syncCheckboxes(pickedSet) {{
    checkboxes.forEach((cb) => {{
      cb.checked = pickedSet.has(cb.dataset.filterCard);
    }});
  }}

  function refreshCatalog(picked, onlyPicked) {{
    const pickedSet = new Set(picked);
    cards.forEach((card) => {{
      const active = pickedSet.has(card.dataset.cardId);
      card.classList.toggle("hidden-card", onlyPicked && !active);
    }});
  }}

  function refreshSelectedStrip(picked) {{
    const ordered = picked
      .map((id) => document.querySelector(`.card[data-card-id="${{CSS.escape(id)}}"]`))
      .filter(Boolean);
    selectedStrip.querySelectorAll(".card").forEach((node) => node.remove());
    if (ordered.length === 0) {{
      emptyState.style.display = "block";
      return;
    }}
    emptyState.style.display = "none";
    ordered.forEach((card) => selectedStrip.appendChild(card.cloneNode(true)));
  }}

  function render(picked, onlyPicked) {{
    pickedCount.textContent = String(picked.length);
    syncCheckboxes(new Set(picked));
    refreshCatalog(picked, onlyPicked);
    refreshSelectedStrip(picked);
    showPickedOnly.checked = onlyPicked;
  }}

  function getPicked() {{
    return checkboxes.filter((cb) => cb.checked).map((cb) => cb.dataset.filterCard);
  }}

  let state = loadState();
  render(state.picked, state.only);

  checkboxes.forEach((cb) => {{
    cb.addEventListener("change", () => {{
      state = {{ ...state, picked: getPicked() }};
      saveState(state.picked);
      render(state.picked, state.only);
    }});
  }});

  showPickedOnly.addEventListener("change", () => {{
    state = {{ ...state, only: showPickedOnly.checked }};
    saveToggle(state.only);
    render(state.picked, state.only);
  }});
}})();
</script>
</body>
</html>
"""


def main():
    OUT.write_text(build_page())
    print(OUT)


if __name__ == "__main__":
    main()
