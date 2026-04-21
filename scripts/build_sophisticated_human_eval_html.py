import json
from html import escape
from pathlib import Path


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
EVAL_DIR = ROOT / "data" / "human_eval" / "sophisticated_v1"
OUT_PATH = EVAL_DIR / "index_v2.html"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    manifest = _load_json(EVAL_DIR / "manifest.json")
    prompt_text = (EVAL_DIR / "labeler_prompt_ko.md").read_text(encoding="utf-8")
    batches = []
    for path in sorted(EVAL_DIR.glob("sophisticated_eval_batch_*.json")):
        batches.append(_load_json(path))

    data_blob = json.dumps(
        {
            "manifest": manifest,
            "prompt": prompt_text,
            "batches": batches,
        },
        ensure_ascii=False,
    )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sophisticated Human Eval</title>
<style>
:root {{
  --bg:#ffffff;
  --surface:#fbfbfc;
  --surface-2:#f4f6f8;
  --border:#dde3e8;
  --text:#171b21;
  --muted:#66707b;
  --yes:#2563eb;
  --no:#d97706;
  --ok:#0f766e;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text','Segoe UI',sans-serif; }}
.page {{ max-width:1600px; margin:0 auto; padding:20px; }}
.hero {{ display:grid; gap:12px; margin-bottom:16px; }}
.title-row {{ display:flex; gap:12px; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; }}
h1 {{ margin:0; font-size:28px; }}
.meta {{ color:var(--muted); font-size:14px; }}
.chips {{ display:flex; gap:8px; flex-wrap:wrap; }}
.chip {{ border:1px solid var(--border); background:var(--surface); border-radius:999px; padding:6px 10px; font-size:13px; color:var(--muted); }}
.toolbar {{ position:sticky; top:0; z-index:10; background:rgba(255,255,255,.96); backdrop-filter:blur(8px); border:1px solid var(--border); border-radius:14px; padding:12px; display:grid; gap:10px; margin-bottom:16px; }}
.toolbar-top {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
select, button {{ font:inherit; }}
select {{ border:1px solid var(--border); background:#fff; border-radius:10px; padding:8px 10px; }}
button {{ border:1px solid var(--border); background:#fff; border-radius:10px; padding:8px 12px; cursor:pointer; }}
.progress {{ color:var(--muted); font-size:13px; }}
.prompt {{ border:1px solid var(--border); background:var(--surface); border-radius:14px; padding:14px; white-space:pre-wrap; line-height:1.45; color:#28323c; margin-bottom:16px; }}
.cards {{ display:grid; gap:16px; }}
.card {{ border:1px solid var(--border); border-radius:16px; overflow:hidden; background:var(--surface); }}
.card-head {{ padding:14px 16px; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap; align-items:flex-end; }}
.cue {{ font-size:20px; font-weight:700; }}
.sub {{ color:var(--muted); font-size:13px; }}
.truth {{ color:#0f766e; font-size:13px; font-weight:600; }}
.compare {{ display:grid; grid-template-columns:1fr; gap:12px; padding:14px; }}
.panel {{ display:grid; gap:8px; }}
.panel-head {{ display:flex; justify-content:space-between; align-items:center; }}
.panel-title {{ font-size:12px; text-transform:uppercase; letter-spacing:.06em; font-weight:700; }}
.variant {{ color:var(--muted); font-size:12px; }}
img {{ width:100%; display:block; border:1px solid var(--border); border-radius:12px; background:#fff; }}
.actions {{ display:flex; gap:8px; padding:0 14px 14px; flex-wrap:wrap; }}
.pick {{ border:1px solid var(--border); background:#fff; border-radius:12px; padding:10px 14px; min-width:88px; }}
.pick.active-yes {{ background:#eff6ff; border-color:#93c5fd; color:#1d4ed8; }}
.pick.active-no {{ background:#fff7ed; border-color:#fdba74; color:#b45309; }}
.response {{ padding:0 14px 14px; color:var(--ok); font-size:13px; }}
@media (max-width: 960px) {{
  .page {{ padding:14px; }}
}}
</style>
</head>
<body>
<div class="page">
  <section class="hero">
    <div class="title-row">
      <div>
        <h1>Sophisticated Human Eval</h1>
        <div class="meta">104 top1 gifs, 208 binary items, 8 labelers x 26 items</div>
      </div>
      <div class="chips">
        <span class="chip">labeler_1: batch 1</span>
        <span class="chip">labeler_2: batch 2</span>
        <span class="chip">labeler_3: batch 3</span>
        <span class="chip">labeler_4: batch 4</span>
        <span class="chip">labeler_5: batch 5</span>
        <span class="chip">labeler_6: batch 6</span>
        <span class="chip">labeler_7: batch 7</span>
        <span class="chip">labeler_8: batch 8</span>
      </div>
    </div>
  </section>

  <section class="toolbar">
    <div class="toolbar-top">
      <label for="batchSelect">Batch</label>
      <select id="batchSelect"></select>
      <button id="exportBtn">Export Responses</button>
      <button id="clearBtn">Clear Current Batch</button>
      <div id="progress" class="progress"></div>
    </div>
  </section>

  <details class="prompt">
    <summary><strong>Labeler Prompt</strong></summary>
    <div id="promptBox"></div>
  </details>

  <section id="cards" class="cards"></section>
</div>

<script>
const DATA = {data_blob};
const STORAGE_PREFIX = 'sophisticated_human_eval_v1_';

const batchSelect = document.getElementById('batchSelect');
const cardsEl = document.getElementById('cards');
const progressEl = document.getElementById('progress');
const promptBox = document.getElementById('promptBox');
const exportBtn = document.getElementById('exportBtn');
const clearBtn = document.getElementById('clearBtn');

promptBox.textContent = DATA.prompt;

function normalizeGifSrc(path) {{
  if (!path) return '';
  if (path.startsWith('file://') || path.startsWith('http://') || path.startsWith('https://') || path.startsWith('data:')) {{
    return path;
  }}
  if (path.startsWith('/')) {{
    return 'file://' + path;
  }}
  return path;
}}

for (const batch of DATA.batches) {{
  const option = document.createElement('option');
  option.value = String(batch.batch_id);
  option.textContent = `Batch ${{String(batch.batch_id).padStart(2, '0')}} · ${{batch.suggested_labeler}}`;
  batchSelect.appendChild(option);
}}

function storageKey(batchId) {{
  return STORAGE_PREFIX + String(batchId).padStart(2, '0');
}}

function loadResponses(batchId) {{
  try {{
    return JSON.parse(localStorage.getItem(storageKey(batchId)) || '{{}}');
  }} catch {{
    return {{}};
  }}
}}

function saveResponses(batchId, responses) {{
  localStorage.setItem(storageKey(batchId), JSON.stringify(responses));
}}

function renderBatch(batchId) {{
  const batch = DATA.batches.find(b => b.batch_id === Number(batchId));
  const responses = loadResponses(batch.batch_id);
  cardsEl.innerHTML = '';
  let answered = 0;

  for (const item of batch.items) {{
    if (responses[item.assignment_id]) answered += 1;

    const card = document.createElement('article');
    card.className = 'card';
    const current = responses[item.assignment_id] || '';

    card.innerHTML = `
      <div class="card-head">
        <div>
          <div class="cue">${{item.order_in_batch}}. ${{item.shown_cue}}</div>
          <div class="sub">${{item.testset}} · c${{item.cue_idx}} · ${{item.assignment_id}}</div>
        </div>
        <div>
          <div class="sub">source: ${{item.source_cue}} · pair: ${{item.pair_id}}</div>
          <div class="truth">Ground Truth: ${{item.ground_truth || 'yes'}}</div>
        </div>
      </div>
      <div class="compare">
        <div class="panel">
          <div class="panel-head">
            <div class="panel-title">Motion GIF</div>
            <div class="variant">${{(item.metadata && item.metadata.task_variant) || item.ground_truth}}</div>
          </div>
          <img src="${{normalizeGifSrc(item.gif_path)}}" alt="gif ${{item.assignment_id}}">
        </div>
      </div>
      <div class="actions">
        <button class="pick ${{current === 'yes' ? 'active-yes' : ''}}" data-id="${{item.assignment_id}}" data-value="yes">Yes</button>
        <button class="pick ${{current === 'no' ? 'active-no' : ''}}" data-id="${{item.assignment_id}}" data-value="no">No</button>
      </div>
      <div class="response">${{current ? `Saved: ${{current}}` : ''}}</div>
    `;
    cardsEl.appendChild(card);
  }}

  progressEl.textContent = `Answered ${{answered}} / ${{batch.items.length}}`;

  cardsEl.querySelectorAll('.pick').forEach(btn => {{
    btn.addEventListener('click', () => {{
      const id = btn.dataset.id;
      const value = btn.dataset.value;
      const next = loadResponses(batch.batch_id);
      next[id] = value;
      saveResponses(batch.batch_id, next);
      renderBatch(batch.batch_id);
      window.scrollTo({{ top: btn.closest('.card').offsetTop - 90, behavior: 'instant' }});
    }});
  }});
}}

batchSelect.addEventListener('change', () => renderBatch(batchSelect.value));

exportBtn.addEventListener('click', () => {{
  const batchId = Number(batchSelect.value);
  const batch = DATA.batches.find(b => b.batch_id === batchId);
  const payload = {{
    batch_id: batchId,
    suggested_labeler: batch.suggested_labeler,
    responses: loadResponses(batchId),
  }};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: 'application/json' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `sophisticated_eval_batch_${{String(batchId).padStart(2, '0')}}_responses.json`;
  a.click();
  URL.revokeObjectURL(url);
}});

clearBtn.addEventListener('click', () => {{
  const batchId = Number(batchSelect.value);
  localStorage.removeItem(storageKey(batchId));
  renderBatch(batchId);
}});

renderBatch(DATA.batches[0].batch_id);
</script>
</body>
</html>
"""

    OUT_PATH.write_text(html, encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
