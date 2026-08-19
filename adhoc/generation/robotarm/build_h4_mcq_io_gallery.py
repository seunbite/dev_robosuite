#!/usr/bin/env python3
"""Build an interactive HTML audit of H4 MCQ inputs and model outputs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
EVAL = REPO / "data/results/evaluation"
MODEL_DIR = EVAL / "h4_mcq_models"
GT = REPO / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
QUESTIONS = EVAL / "h4_gt_binary_mcq.json"
OUT = REPO / "data/results/html/h4_binary_mcq_input_output.html"
MODELS = {
    "Qwen 32B": ("h4_qwen_32b_predictions.jsonl", "h4cb_qwen_32b_predictions.jsonl"),
    "Qwen 7B": ("h4_qwen_7b_predictions.jsonl", "h4cb_qwen_7b_predictions.jsonl"),
    "Qwen 3B": ("h4_qwen_3b_predictions.jsonl", "h4cb_qwen_3b_predictions.jsonl"),
}
SYSTEM_PROMPT = (
    "You are an expert in robot gesture motion codes. "
    "Select the correct attribute value. Answer only A or B."
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _component_payload(movement: dict[str, Any]) -> dict[str, Any]:
    parameters = movement.get("parameters") or {}
    kind = str(movement.get("type", "unknown")).lower()
    nested = parameters.get(kind)
    return nested if isinstance(nested, dict) else parameters


def _blanked_movements(platform_gt: dict[str, Any], question: dict[str, Any]) -> list[dict[str, Any]]:
    movements = copy.deepcopy(platform_gt["movements"])
    movement = movements[int(question["component_index"])]
    payload = _component_payload(movement)
    attribute = str(question["attribute"])
    if movement["type"] == "pose":
        pose = payload.get("pose") if isinstance(payload.get("pose"), dict) else payload
        key = {
            "direction": "arm_position" if "arm_position" in pose else "dir",
            "gripper_orientation": "gripper_orientation",
            "left_arm": "left_arm",
        }.get(attribute, attribute.removeprefix("pose."))
        pose[key] = "<BLANK>"
    elif attribute == "repetition":
        key = "repetition" if "repetition" in payload else "rep"
        payload[key] = "<BLANK>"
    elif attribute.startswith("path."):
        key = attribute.removeprefix("path.").removesuffix("_sign")
        payload[key] = "<BLANK>"
    return movements


def _prompt_text(question: dict[str, Any]) -> str:
    options = "\n".join(
        f"{option['id']}. {option['text']}" for option in question["options"]
    )
    return (
        f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n[USER]\n{question['prompt']}\n\n"
        f"{options}\n\nAnswer only A or B."
    )


def main() -> None:
    master = json.loads(GT.read_text(encoding="utf-8"))
    gt_by_key = {
        (int(entry["idx"]), platform): row
        for entry in master
        for platform, row in entry["platforms"].items()
    }
    question_rows = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    predictions: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for label, (original_name, reversed_name) in MODELS.items():
        original = {
            row["question_id"]: row for row in _load_jsonl(MODEL_DIR / original_name)
        }
        reversed_rows = {
            row["paired_question_id"]: row
            for row in _load_jsonl(MODEL_DIR / reversed_name)
        }
        predictions[label] = (original, reversed_rows)

    records = []
    for question in question_rows:
        qid = question["question_id"]
        platform_gt = gt_by_key[(int(question["cue_idx"]), question["platform"])]
        model_outputs = {}
        for label, (original, reversed_rows) in predictions.items():
            model_outputs[label] = {
                "original": {
                    "options": original[qid]["options"],
                    "prediction": original[qid]["prediction"],
                    "raw_output": original[qid]["raw_output"],
                    "correct": original[qid]["correct"],
                },
                "reversed": {
                    "options": reversed_rows[qid]["options"],
                    "prediction": reversed_rows[qid]["prediction"],
                    "raw_output": reversed_rows[qid]["raw_output"],
                    "correct": reversed_rows[qid]["correct"],
                },
            }
        records.append(
            {
                **question,
                "actual_model_input": _prompt_text(question),
                "gt_source_code": platform_gt.get("source_code", ""),
                "full_gt_config_not_sent": platform_gt["movements"],
                "intended_blank_config_not_sent": _blanked_movements(platform_gt, question),
                "model_outputs": model_outputs,
            }
        )

    embedded = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    html_text = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>H4 Binary MCQ Input / Output Audit</title>
<style>
:root{--bg:#f4f5f7;--panel:#fff;--ink:#1c2430;--muted:#667085;--line:#d7dce3;--accent:#2457a6;--bad:#a52a2a;--good:#17643a;--warn:#8a4b08}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{background:var(--panel);border-bottom:1px solid var(--line);padding:24px max(24px,calc((100% - 1500px)/2))}
h1{font-size:24px;margin:0 0 8px}.lead{font-size:16px;max-width:1050px;margin:0}.no{color:var(--bad);font-weight:750}
.summary{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:18px}.summary section{border:1px solid var(--line);padding:14px;background:#fafbfc}
.summary h2{font-size:15px;margin:0 0 8px}.summary ul{margin:0;padding-left:20px}
.controls{position:sticky;top:0;z-index:3;display:flex;gap:10px;flex-wrap:wrap;padding:12px max(24px,calc((100% - 1500px)/2));background:var(--panel);border-bottom:1px solid var(--line)}
input,select,button{font:inherit;border:1px solid var(--line);border-radius:4px;background:white;padding:8px 10px}input{min-width:260px}button{cursor:pointer}.count{padding:8px;color:var(--muted)}
main{max-width:1500px;margin:0 auto;padding:20px 24px 60px}.card{background:var(--panel);border:1px solid var(--line);margin-bottom:18px}
.card-head{display:flex;justify-content:space-between;gap:14px;padding:14px 16px;border-bottom:1px solid var(--line)}.qid{font-family:ui-monospace,monospace;color:var(--muted)}
.tags{display:flex;gap:6px;flex-wrap:wrap}.tag{border:1px solid var(--line);padding:2px 7px;border-radius:3px;color:var(--muted)}
.truth{color:var(--good);font-weight:700}.grid{display:grid;grid-template-columns:1fr 1fr}.column{padding:16px;min-width:0}.column+ .column{border-left:1px solid var(--line)}
h3{font-size:15px;margin:0 0 8px}h4{font-size:13px;margin:14px 0 6px;color:var(--muted)}
pre{white-space:pre-wrap;word-break:break-word;background:#f7f8fa;border:1px solid var(--line);padding:12px;margin:0;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;max-height:330px;overflow:auto}
.not-sent{color:var(--bad);font-weight:700}.sent{color:var(--accent);font-weight:700}.outputs{border-top:1px solid var(--line);padding:16px}
.model-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.model{border:1px solid var(--line);padding:10px}.model h4{margin:0 0 8px;color:var(--ink)}
.run{display:grid;grid-template-columns:75px 1fr auto;gap:8px;padding:7px 0;border-top:1px solid var(--line);align-items:start}.run:first-of-type{border-top:0}
.ok{color:var(--good);font-weight:700}.wrong{color:var(--bad);font-weight:700}.opts{font-family:ui-monospace,monospace;font-size:12px}
.pagination{display:flex;justify-content:center;gap:10px;align-items:center;margin:22px 0}.hidden{display:none}
@media(max-width:900px){.summary,.grid,.model-grid{grid-template-columns:1fr}.column+ .column{border-left:0;border-top:1px solid var(--line)}}
</style>
</head>
<body>
<header>
  <h1>H4 Binary MCQ Input / Output Audit</h1>
  <p class="lead"><span class="no">결론: 현재 실험은 GT config 빈칸 채우기가 아니다.</span>
  모델은 전체 GT config를 받지 않았고, cue·embodiment·component 번호/type·attribute 이름과 A/B 값만 받았다.</p>
  <div class="summary">
    <section><h2>실제로 모델에 전송됨</h2><ul><li>System instruction</li><li>Cue 이름과 embodiment</li><li>Component 번호/type과 attribute 이름</li><li>A/B 두 값</li></ul></section>
    <section><h2>모델에 전송되지 않음</h2><ul><li>전체 GT movements/config</li><li>GT source DSL</li><li>나머지 component sequence와 값</li><li>attribute가 빈칸 처리된 config</li></ul></section>
  </div>
</header>
<div class="controls">
  <input id="search" placeholder="cue, question ID 검색">
  <select id="platform"><option value="">모든 embodiment</option></select>
  <select id="attribute"><option value="">모든 attribute</option></select>
  <button id="reset">초기화</button><span class="count" id="count"></span>
</div>
<main><div id="cards"></div><div class="pagination"><button id="prev">이전</button><span id="page"></span><button id="next">다음</button></div></main>
<script id="records" type="application/json">__DATA__</script>
<script>
const all=JSON.parse(document.getElementById('records').textContent), perPage=20;
const search=document.getElementById('search'), platform=document.getElementById('platform'), attribute=document.getElementById('attribute');
let page=0, filtered=all;
function esc(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function options(run){return run.options.map(o=>`${o.id}=${JSON.stringify(o.value)}`).join(' · ')}
function populate(el,values){for(const v of [...new Set(values)].sort()){const o=document.createElement('option');o.value=v;o.textContent=v;el.appendChild(o)}}
populate(platform,all.map(x=>x.platform));populate(attribute,all.map(x=>x.attribute));
function render(){
 const term=search.value.toLowerCase().trim();
 filtered=all.filter(x=>(!term||`${x.question_id} ${x.cue}`.toLowerCase().includes(term))&&(!platform.value||x.platform===platform.value)&&(!attribute.value||x.attribute===attribute.value));
 const pages=Math.max(1,Math.ceil(filtered.length/perPage));page=Math.min(page,pages-1);
 document.getElementById('count').textContent=`${filtered.length} / ${all.length} 문항`;
 document.getElementById('page').textContent=`${page+1} / ${pages}`;
 document.getElementById('prev').disabled=page===0;document.getElementById('next').disabled=page>=pages-1;
 document.getElementById('cards').innerHTML=filtered.slice(page*perPage,(page+1)*perPage).map(x=>`
 <article class="card">
  <div class="card-head"><div><strong>${esc(x.cue)}</strong> <span class="qid">${esc(x.question_id)}</span>
   <div class="tags"><span class="tag">${esc(x.platform)}</span><span class="tag">${esc(x.component_type)} #${x.component_number}</span><span class="tag">${esc(x.attribute)}</span></div></div>
   <div class="truth">GT: ${esc(x.correct_option)} = ${esc(JSON.stringify(x.correct_value))}</div></div>
  <div class="grid">
   <section class="column"><h3 class="sent">Actual model input — SENT</h3><pre>${esc(x.actual_model_input)}</pre></section>
   <section class="column"><h3 class="not-sent">Full GT / intended blank format — NOT SENT</h3>
    <h4>GT source DSL</h4><pre>${esc(x.gt_source_code)}</pre>
    <h4>사용자가 기대한 blank-config 형태</h4><pre>${esc(JSON.stringify(x.intended_blank_config_not_sent,null,2))}</pre>
    <details><summary>원본 full GT movements 보기</summary><pre>${esc(JSON.stringify(x.full_gt_config_not_sent,null,2))}</pre></details>
   </section>
  </div>
  <section class="outputs"><h3>Actual model outputs</h3><div class="model-grid">
   ${Object.entries(x.model_outputs).map(([name,runs])=>`<div class="model"><h4>${esc(name)}</h4>
    ${[['Original',runs.original],['A/B reversed',runs.reversed]].map(([label,r])=>`<div class="run"><strong>${label}</strong><span class="opts">${esc(options(r))}<br>raw=${esc(r.raw_output)}</span><span class="${r.correct?'ok':'wrong'}">${esc(r.prediction)} ${r.correct?'correct':'wrong'}</span></div>`).join('')}
   </div>`).join('')}
  </div></section>
 </article>`).join('');
}
[search,platform,attribute].forEach(el=>el.addEventListener(el===search?'input':'change',()=>{page=0;render()}));
document.getElementById('reset').onclick=()=>{search.value='';platform.value='';attribute.value='';page=0;render()};
document.getElementById('prev').onclick=()=>{page--;render();scrollTo(0,0)};document.getElementById('next').onclick=()=>{page++;render();scrollTo(0,0)};
render();
</script>
</body></html>""".replace("__DATA__", embedded)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html_text, encoding="utf-8")
    print(OUT)
    print(f"records={len(records)} bytes={OUT.stat().st_size}")


if __name__ == "__main__":
    main()
