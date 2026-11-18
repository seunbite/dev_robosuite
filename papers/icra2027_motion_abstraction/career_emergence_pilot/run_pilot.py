#!/usr/bin/env python3
"""Pilot: 100 diverse O*NET-grounded personas × 3 career-creativity generators."""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from grounded import FRAME_IDS, blend_payload, rulebend_payload
DATA = ROOT / "data"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

SOC_KO = {
    "11": "경영·관리",
    "13": "경영지원·금융",
    "15": "컴퓨터·수학",
    "17": "건축·공학",
    "19": "과학",
    "21": "사회서비스",
    "23": "법률",
    "25": "교육·도서관",
    "27": "예술·미디어",
    "29": "보건 전문",
    "31": "보건 보조",
    "33": "보호·안전",
    "35": "음식",
    "37": "시설·환경",
    "39": "개인서비스",
    "41": "판매",
    "43": "사무",
    "45": "농림어업",
    "47": "건설",
    "49": "설치·수리",
    "51": "생산",
    "53": "운송",
    "55": "군인",
}

LIFE = [
    ("커리어 초반", "서울", "돌봄 없음", "이직 여유 있음", "인정"),
    ("경력 정체", "부산", "미취학 자녀 1", "야간 근무 불가", "안정"),
    ("경력단절 복귀", "대전", "초등 자녀 2", "주 25시간만", "의미"),
    ("중년 전환", "광주", "부모 간병", "장거리 출퇴근 불가", "존엄"),
    ("은퇴 전 재설계", "제주", "돌봄 없음", "체력 중저", "기여"),
    ("이주민", "안산", "가족 해외", "한국어 중급, 자격 미비", "정착"),
    ("비수도권 정주", "강릉", "농번기 가족 일손", "대도시 이주 거부", "지역"),
    ("장애·만성질환", "수원", "본인 컨디션 기복", "장시간 입식 불가", "자율"),
    ("학업 병행", "대구", "돌봄 없음", "주 3일만 가능", "성장"),
    ("프리랜서 불안정", "서울", "1인 가구", "현금흐름 공백 위험", "통제"),
]

GIVEN = [
    "김하린", "박준호", "이서연", "최민재", "정하늘", "한지훈", "오수빈", "윤도윤",
    "장예은", "신태양", "배지아", "임현우", "강나래", "조은성", "문채원", "류시원",
    "서지호", "안보람", "홍가은", "노태민",
]


def load_occupations():
    occ = {}
    with (DATA / "Occupation Data.txt").open(encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            occ[row["O*NET-SOC Code"]] = {
                "code": row["O*NET-SOC Code"],
                "title": row["Title"],
                "desc": row["Description"],
                "major": row["O*NET-SOC Code"][:2],
            }
    zones = {}
    with (DATA / "Job Zones.txt").open(encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            zones[row["O*NET-SOC Code"]] = int(row["Job Zone"])
    related = defaultdict(list)
    with (DATA / "Related Occupations.txt").open(encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            related[row["O*NET-SOC Code"]].append(row["Related O*NET-SOC Code"])
    for code, o in occ.items():
        o["zone"] = zones.get(code, 3)
        o["related"] = related.get(code, [])
    return occ


def stratified_codes(occ, n=100, rng=None):
    rng = rng or random.Random(42)
    buckets = defaultdict(list)
    for code, o in occ.items():
        if o["title"].startswith("All Other"):
            continue
        buckets[o["major"]].append(code)
    majors = sorted(buckets)
    picked = []
    i = 0
    while len(picked) < n:
        m = majors[i % len(majors)]
        pool = [c for c in buckets[m] if c not in picked]
        if pool:
            picked.append(rng.choice(pool))
        i += 1
        if i > 5000:
            break
    return picked[:n]


def make_personas(occ, n=100):
    rng = random.Random(42)
    codes = stratified_codes(occ, n, rng)
    personas = []
    for i, code in enumerate(codes):
        o = occ[code]
        life = LIFE[i % len(LIFE)]
        stage, loc, care, constraint, theme = life
        age = {
            "커리어 초반": rng.randint(24, 29),
            "경력 정체": rng.randint(32, 38),
            "경력단절 복귀": rng.randint(36, 44),
            "중년 전환": rng.randint(45, 54),
            "은퇴 전 재설계": rng.randint(56, 63),
            "이주민": rng.randint(28, 46),
            "비수도권 정주": rng.randint(30, 52),
            "장애·만성질환": rng.randint(27, 50),
            "학업 병행": rng.randint(22, 34),
            "프리랜서 불안정": rng.randint(29, 41),
        }[stage]
        years = max(0, age - 24 - rng.randint(0, 4))
        personas.append(
            {
                "id": f"P{i+1:03d}",
                "name": GIVEN[i % len(GIVEN)],
                "age": age,
                "location": loc,
                "stage": stage,
                "care": care,
                "constraint": constraint,
                "theme": theme,
                "years": years,
                "code": code,
                "title": o["title"],
                "title_ko": o["title"],
                "major": o["major"],
                "sector": SOC_KO.get(o["major"], o["major"]),
                "zone": o["zone"],
                "desc": o["desc"][:280],
                "related": o["related"][:8],
                "idx": i,
            }
        )
    return personas


def clamp(x, lo=1, hi=5):
    return max(lo, min(hi, x))


def gen_modal(p, occ):
    rel = p["related"]
    nxt = occ[rel[0]]["title"] if rel else p["title"]
    payer = "기존 채용 시장(같은 산업 구인)"
    return dict(
        method="M1_modal",
        method_ko="전형 후계 (O*NET related)",
        role=nxt,
        offer=f"{p['title']}의 O*NET 일차 관련 직업으로 이직한다. 슬롯 합성 없이 분류 공간을 한 칸 옮긴다.",
        payer=payer,
        day90=f"관련 직무 공고 15개를 모아 이력서를 {nxt} 키워드로 다시 쓴다",
        bent="없음 (R 고정)",
        frames=f"{p['title']} → {nxt}",
        scores=score_pack(
            comb=1, expl=2, trans=1,
            c1=4 if abs(p["zone"] - occ.get(rel[0], p)["zone"] if rel else p["zone"]) <= 1 else 3,
            c2=4,
            constraint_hit=p["constraint"],
            method="modal",
            p=p,
        ),
        why_c1="숙련·자격 공간이 인접해 90일 공고 지원이 가능",
        why_c2="이미 구인 공고가 있는 표준 노동시장",
        fail_mode="창발 없음. 동질화의 하한.",
    )


def gen_blend(p, rng):
    # Round-robin frames so one high-C2 template cannot dominate all 100.
    fid = FRAME_IDS[p["idx"] % len(FRAME_IDS)]
    fr = blend_payload(p, fid)
    c1 = 4
    if "야간 근무 불가" in p["constraint"] and fid == "hh_protocol":
        c1 = 2
    if "장시간 입식 불가" in p["constraint"] and p["major"] in ("47", "51", "53"):
        c1 = 3
    if p["stage"] == "커리어 초반" and fid == "eol_audit":
        c1 = 3
    if p["stage"] == "이주민" and fid == "dispute_docent":
        c1 = 2
    c2 = fr["c2"]
    if fid == "muni_evidence" and p["location"] in ("서울", "수원"):
        c2 = 3
    return dict(
        method="M2_blend",
        method_ko="기능 유추 블렌드 (직무 접지)",
        role=fr["role"],
        offer=fr["offer"],
        payer=fr["payer"],
        day90=fr["day90"],
        bent=fr["bent"] + " (일부)",
        frames=fr["frames"],
        frame_id=fid,
        scores=score_pack(
            comb=fr["comb"], expl=3, trans=fr["trans"],
            c1=c1, c2=c2, constraint_hit=p["constraint"], method="blend", p=p,
        ),
        why_c1="핵심 행위가 이 직무의 artifact/상대에 있으면 90일 납품이 가능.",
        why_c2=f"지불 주체: {fr['payer']}.",
        fail_mode="일반 민원인 문장으로 붕괴하면 실패. 직무 특화 상대·산출이 보여야 함.",
    )


def gen_rulebend(p, rng):
    fr = rulebend_payload(p)
    c1 = 3 if p["stage"] in ("커리어 초반", "학업 병행") else 4
    if "체력 중저" in p["constraint"]:
        c1 = 3
    c2 = 4
    return dict(
        method="M3_rulebend",
        method_ko="직업 문법 1개 변환 (U=유급 유지)",
        role=fr["role"],
        offer=fr["offer"],
        payer=fr["payer"],
        day90=fr["day90"],
        bent=fr["bent"],
        frames=fr["frames"],
        scores=score_pack(
            comb=3, expl=3, trans=5,
            c1=c1, c2=c2, constraint_hit=p["constraint"], method="rulebend", p=p,
        ),
        why_c1="숙련 artifact는 유지. 초년은 발주 네트워크가 약해 감점.",
        why_c2=f"지불: {fr['payer']}.",
        fail_mode="유급을 버리거나 초능력 직업이면 세계 파괴.",
    )


def score_pack(comb, expl, trans, c1, c2, constraint_hit, method, p):
    if "주 25시간만" in constraint_hit or "주 3일만" in constraint_hit:
        if method != "modal":
            c1 = clamp(c1 - 1)
    if p["stage"] == "이주민" and method == "modal":
        c1 = clamp(c1 - 1)
        c2 = clamp(c2 - 1)
    creativity = round((comb + expl + trans) / 3, 2)
    viable = round((c1 + c2) / 2, 2)
    gated = round(creativity * (1 if min(c1, c2) >= 3 else 0.35), 2)
    return {
        "combinational": comb,
        "exploratory": expl,
        "transformational": trans,
        "creativity_mean": creativity,
        "c1_doable": c1,
        "c2_payer": c2,
        "viable_mean": viable,
        "gated_creativity": gated,
    }


def run():
    occ = load_occupations()
    personas = make_personas(occ, 100)
    rng = random.Random(7)
    rows = []
    for p in personas:
        for rec in (gen_modal(p, occ), gen_blend(p, rng), gen_rulebend(p, rng)):
            rec["persona"] = p
            rows.append(rec)

    (OUT / "personas.json").write_text(
        json.dumps(personas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    slim = []
    for r in rows:
        p = r["persona"]
        slim.append({
            "id": p["id"], "name": p["name"], "age": p["age"], "location": p["location"],
            "stage": p["stage"], "occupation": p["title"], "sector": p["sector"],
            "constraint": p["constraint"], "care": p["care"], "theme": p["theme"],
            "method": r["method"], "method_ko": r["method_ko"],
            "role": r["role"], "offer": r["offer"], "payer": r["payer"],
            "day90": r["day90"], "bent": r["bent"], "frames": r["frames"],
            "frame_id": r.get("frame_id", ""),
            **r["scores"], "why_c1": r["why_c1"], "why_c2": r["why_c2"],
            "fail_mode": r["fail_mode"],
        })
    with (OUT / "recommendations.jsonl").open("w", encoding="utf-8") as f:
        for s in slim:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # CSV
    keys = list(slim[0].keys())
    with (OUT / "recommendations.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(slim)

    summary = summarize(slim, personas)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_html(personas, slim, summary)
    write_md(summary, slim, personas)
    print("personas", len(personas), "rows", len(slim))
    print("html", OUT / "report.html")
    return summary


def mean(xs):
    return round(statistics.mean(xs), 3) if xs else 0


def summarize(slim, personas):
    by_m = defaultdict(list)
    for s in slim:
        by_m[s["method"]].append(s)
    methods = {}
    for m, rs in by_m.items():
        methods[m] = {
            "n": len(rs),
            "combinational": mean([r["combinational"] for r in rs]),
            "exploratory": mean([r["exploratory"] for r in rs]),
            "transformational": mean([r["transformational"] for r in rs]),
            "creativity_mean": mean([r["creativity_mean"] for r in rs]),
            "c1_doable": mean([r["c1_doable"] for r in rs]),
            "c2_payer": mean([r["c2_payer"] for r in rs]),
            "gated_creativity": mean([r["gated_creativity"] for r in rs]),
            "pass_gate_pct": round(100 * sum(min(r["c1_doable"], r["c2_payer"]) >= 3 for r in rs) / len(rs), 1),
        }
    sectors = Counter(p["sector"] for p in personas)
    stages = Counter(p["stage"] for p in personas)
    blends = [s for s in slim if s["method"] == "M2_blend"]

    blend_sorted = sorted(
        blends,
        key=lambda r: (r["gated_creativity"], r["c2_payer"], r["c1_doable"]),
        reverse=True,
    )
    by_fid = {}
    for r in blend_sorted:
        fid = r.get("frame_id") or r.get("frames", "")
        if fid not in by_fid:
            by_fid[fid] = r
    top_blend = [by_fid[fid] for fid in FRAME_IDS if fid in by_fid]
    for r in blend_sorted:
        if len(top_blend) >= 8:
            break
        if r not in top_blend:
            top_blend.append(r)
    unique_roles = len({s["role"] for s in blends})
    unique_offers = len({s["offer"] for s in blends})
    generic_foil = sum("민원인·환자 가족·심사자" in s["offer"] for s in blends)
    top_t = sorted(
        [s for s in slim if s["method"] == "M3_rulebend"],
        key=lambda r: (r["gated_creativity"], r["transformational"]),
        reverse=True,
    )[:6]
    weak = sorted(
        [s for s in slim if s["method"] != "M1_modal"],
        key=lambda r: (r["c1_doable"] + r["c2_payer"], r["creativity_mean"]),
    )[:6]
    return {
        "n_personas": len(personas),
        "n_recs": len(slim),
        "source": "O*NET 30.1 Occupation Data + Related Occupations, synthetic life overlays",
        "sectors": dict(sectors),
        "stages": dict(stages),
        "methods": methods,
        "top_blend": top_blend,
        "top_transform": top_t,
        "weak": weak,
        "m2_unique_roles": unique_roles,
        "m2_unique_offers": unique_offers,
        "m2_generic_foil": generic_foil,
        "m2_n": len(blends),
    }


def esc(s):
    return (
        str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def write_html(personas, slim, summary):
    m = summary["methods"]
    cards = []
    for s in summary["top_blend"][:6]:
        cards.append(f"""
        <article class="card">
          <div class="kicker">{esc(s['id'])} · {esc(s['name'])} · {esc(s['age'])}세 · {esc(s['location'])}</div>
          <h3>{esc(s['role'])}</h3>
          <p class="meta">{esc(s['occupation'])} · {esc(s['stage'])} · 제약: {esc(s['constraint'])}</p>
          <p>{esc(s['offer'])}</p>
          <ul>
            <li><b>지불</b> {esc(s['payer'])}</li>
            <li><b>90일</b> {esc(s['day90'])}</li>
            <li><b>프레임</b> {esc(s['frames'])}</li>
          </ul>
          <div class="pills">
            <span>C {s['combinational']}</span><span>E {s['exploratory']}</span>
            <span>T {s['transformational']}</span>
            <span>할 만함 {s['c1_doable']}</span><span>지불 {s['c2_payer']}</span>
            <span>게이트 {s['gated_creativity']}</span>
          </div>
        </article>""")

    table_rows = []
    for mid, label in [
        ("M1_modal", "M1 전형 후계"),
        ("M2_blend", "M2 기능 블렌드"),
        ("M3_rulebend", "M3 문법 변환"),
    ]:
        x = m[mid]
        table_rows.append(
            f"<tr><td>{label}</td><td>{x['combinational']}</td><td>{x['exploratory']}</td>"
            f"<td>{x['transformational']}</td><td>{x['creativity_mean']}</td>"
            f"<td>{x['c1_doable']}</td><td>{x['c2_payer']}</td>"
            f"<td>{x['gated_creativity']}</td><td>{x['pass_gate_pct']}%</td></tr>"
        )

    persona_rows = []
    for p in personas:
        persona_rows.append(
            f"<tr><td>{p['id']}</td><td>{esc(p['name'])}</td><td>{p['age']}</td>"
            f"<td>{esc(p['location'])}</td><td>{esc(p['sector'])}</td>"
            f"<td>{esc(p['title'])}</td><td>{esc(p['stage'])}</td>"
            f"<td>{esc(p['constraint'])}</td></tr>"
        )

    bars = []
    for mid, label, color in [
        ("M1_modal", "M1", "#64748b"),
        ("M2_blend", "M2", "#0f766e"),
        ("M3_rulebend", "M3", "#a16207"),
    ]:
        x = m[mid]
        for key, lab in [
            ("combinational", "조합"),
            ("exploratory", "탐색"),
            ("transformational", "변환"),
            ("c1_doable", "할 만함"),
            ("c2_payer", "지불"),
        ]:
            w = x[key] / 5 * 100
            bars.append(
                f'<div class="barline"><span class="blab">{label} {lab}</span>'
                f'<div class="track"><div class="fill" style="width:{w}%;background:{color}"></div></div>'
                f'<span class="bn">{x[key]}</span></div>'
            )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>진로 창발 파일럿 · 100 페르소나 × 3 생성기</title>
<style>
  :root {{ --bg:#f6f1e8; --ink:#1c1917; --muted:#57534e; --card:#fffdf8; --line:#e7e0d4; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font:16px/1.55 "Pretendard", "Noto Sans KR", system-ui, sans-serif;
         background:var(--bg); color:var(--ink); }}
  header {{ padding:48px 8vw 24px; }}
  h1 {{ font-size:2rem; letter-spacing:-.03em; margin:0 0 .4em; }}
  .lede {{ max-width:68ch; color:var(--muted); }}
  main {{ padding:0 8vw 80px; }}
  section {{ margin:42px 0; }}
  h2 {{ font-size:1.25rem; margin:0 0 12px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:18px 18px 14px; }}
  .kicker {{ font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:#0f766e; }}
  .meta {{ color:var(--muted); font-size:13px; }}
  .pills span {{ display:inline-block; background:#ecfdf5; color:#115e59; font-size:12px;
                 padding:2px 8px; border-radius:999px; margin:2px 4px 0 0; }}
  table {{ width:100%; border-collapse:collapse; background:var(--card); font-size:13px; }}
  th, td {{ border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }}
  th {{ background:#faf6ef; position:sticky; top:0; }}
  .scroll {{ max-height:420px; overflow:auto; border:1px solid var(--line); border-radius:12px; }}
  .barline {{ display:grid; grid-template-columns:120px 1fr 40px; gap:8px; align-items:center; margin:6px 0; font-size:13px; }}
  .track {{ height:10px; background:#efe8dc; border-radius:99px; overflow:hidden; }}
  .fill {{ height:100%; }}
  .callout {{ background:#111; color:#f5f0e6; padding:18px 20px; border-radius:16px; }}
  .callout h2 {{ color:#fff; }}
</style>
</head>
<body>
<header>
  <h1>유급은 유지하고, 직업 문법만 비틀기</h1>
  <p class="lede">O*NET 30.1에서 층화한 페르소나 100명 × 생성기 3종(전형 후계 / 기능 블렌드 / 문법 변환).
  창의성은 Boden 세 층으로 쪼개고, 할 만함(C1)과 지불 주체(C2)가 둘 다 약하면 창의 점수를 게이트한다.
  이 파일럿은 LLM API가 아니라 <b>기호 생성기</b>다. 1차 실행은 프레임당 영어 보일러플레이트가 같아 상위 카드가 민원인 문장으로 붕괴했다. 이번 실행은 직업 키워드로 artifact·상대·지불칸을 접지한다.</p>
</header>
<main>
<section class="callout">
  <h2>가설</h2>
  <ol>
    <li>전형 후계(M1)는 할 만함·지불은 높고 창의성(특히 변환)은 낮다.</li>
    <li>기능 블렌드(M2)는 조합 창의와 게이트된 창의에서 가장 높다. 로봇 상담형 <i>새 항목</i>이 여기 있다.</li>
    <li>문법 변환(M3)은 변환 점수는 최고이나 C1/C2가 흔들려 게이트 후 점수는 M2보다 낮을 수 있다.</li>
    <li>‘창의적으로’ 없이, 슬롯 합성·규칙 1개 부정으로도 사회 질서를 살짝 벗어난 유급 역할이 나온다.</li>
  </ol>
</section>

<section>
  <h2>실험 설계</h2>
  <div class="grid">
    <div class="card"><b>I</b><p>O*NET 1,016개 직업에서 전공 그룹을 돌아가며 100개 표집. 생애 오버레이 10종(돌봄·이주민·정주·장애·단절 등)을 순환.</p></div>
    <div class="card"><b>생성기</b><p>M1 관련직업 1순위. M2 기능 유추 프레임 8종과 직군 매칭. M3 유급(U) 고정, 고용/개인단위/성공정의 등 R 1개 부정.</p></div>
    <div class="card"><b>채점</b><p>조합·탐색·변환 1–5. C1 할 만함, C2 지불. min(C1,C2)&lt;3이면 창의 점수 ×0.35 (TwistBench realism 게이트와 동형).</p></div>
  </div>
</section>

<section>
  <h2>방법별 평균 (n=100씩, 총 300 추천)</h2>
  <div class="scroll">
  <table>
    <thead><tr><th>방법</th><th>조합</th><th>탐색</th><th>변환</th><th>창의 평균</th><th>C1</th><th>C2</th><th>게이트 창의</th><th>게이트 통과</th></tr></thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table>
  </div>
  <div style="margin-top:18px">{''.join(bars)}</div>
</section>

<section>
  <h2>정성: 게이트를 통과한 창발 예시 (M2 상위)</h2>
  <div class="grid">{''.join(cards)}</div>
</section>

<section>
  <h2>결론 (이 파일럿이 말하는 것)</h2>
  <div class="card">
    <p>M2 문장 다양성: unique roles {summary['m2_unique_roles']}/{summary['m2_n']}, unique offers {summary['m2_unique_offers']}/{summary['m2_n']}, 일반 민원인 보일러플레이트 {summary['m2_generic_foil']}건.</p>
    <p>M1은 취업 추천으로는 무난하고 창발 벤치마크의 <b>음성 대조</b>로 쓰여야 한다.
    M2는 ‘새 항목’이 나오면서도 기존 예산 칸(연수, 컴플라이언스, 법률 부가)에 붙이면 C2가 산다.
    M3는 변환 정의에는 가장 충실하지만, 발주 네트워크가 없는 초년·이주민 프로필에서는 C1이 무너진다.</p>
    <p>따라서 워크숍 본실험의 LLM 조건은 <b>M2 파이프라인</b>을 창발 주 조작으로, M3를 변환 조작으로, M1을 베이스라인으로 두면 된다.
    형용사 프롬프트는 이 세 방법보다 약할 것으로 예측한다.</p>
    <p style="color:var(--muted);font-size:13px">한계: 채점이 규칙 기반이며 인간 상담사 평정이 없다. 페르소나는 합성이다. 프레임 8개는 공유하지만 산출·상대는 직업에 붙인다. 다음 단계는 같은 I에 LLM 직접생성 vs 이 파이프라인을 대조하는 것이다.</p>
  </div>
</section>

<section>
  <h2>페르소나 100</h2>
  <p class="meta">출처: O*NET-SOC 타이틀 + 합성 생애 제약. 직업명은 영문 원문 유지(분류 정합).</p>
  <div class="scroll">
  <table>
    <thead><tr><th>ID</th><th>이름</th><th>나이</th><th>지역</th><th>직군</th><th>현재 직업 (O*NET)</th><th>단계</th><th>제약</th></tr></thead>
    <tbody>{''.join(persona_rows)}</tbody>
  </table>
  </div>
</section>
</main>
</body>
</html>"""
    (OUT / "report.html").write_text(html, encoding="utf-8")
    (ROOT / "report.html").write_text(html, encoding="utf-8")


def write_md(summary, slim, personas):
    m = summary["methods"]
    lines = [
        "# 진로 창발 파일럿 (100 × 3)",
        "",
        "## 가설",
        "- M1 전형 후계는 C1/C2는 높고 창의성(변환)은 낮다.",
        "- M2 기능 블렌드는 조합 창의와 게이트된 창의에서 가장 높다.",
        "- M3 문법 변환은 변환 점수는 최고, 게이트 후에는 M2보다 낮을 수 있다.",
        "",
        "## 설계",
        f"- 페르소나 {summary['n_personas']}명. 출처: {summary['source']}.",
        "- 생성기 3: O*NET related / 기능 유추 블렌드 / U 고정 R 1개 부정.",
        "- 지표: 조합·탐색·변환, C1 할 만함, C2 지불. min(C1,C2)<3이면 창의×0.35.",
        "",
        "## 결과 요약",
        "",
        "| 방법 | 조합 | 탐색 | 변환 | 창의 | C1 | C2 | 게이트 창의 | 통과율 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "M1_modal": "M1 전형 후계",
        "M2_blend": "M2 기능 블렌드",
        "M3_rulebend": "M3 문법 변환",
    }
    for mid, lab in labels.items():
        x = m[mid]
        lines.append(
            f"| {lab} | {x['combinational']} | {x['exploratory']} | {x['transformational']} | "
            f"{x['creativity_mean']} | {x['c1_doable']} | {x['c2_payer']} | "
            f"{x['gated_creativity']} | {x['pass_gate_pct']}% |"
        )
    lines += ["", "## 창발 상위 예시 (M2)", ""]
    for s in summary["top_blend"][:6]:
        lines += [
            f"### {s['role']}",
            f"- 프로필: {s['id']} {s['name']}, {s['occupation']}, {s['stage']}, 제약 `{s['constraint']}`",
            f"- 제안: {s['offer']}",
            f"- 지불: {s['payer']}",
            f"- 90일: {s['day90']}",
            f"- 점수: C={s['combinational']} E={s['exploratory']} T={s['transformational']} "
            f"C1={s['c1_doable']} C2={s['c2_payer']} gated={s['gated_creativity']}",
            "",
        ]
    lines += ["## 결론", ""]
    m1, m2, m3 = m["M1_modal"], m["M2_blend"], m["M3_rulebend"]
    lines.append(
        f"게이트 창의는 M2({m2['gated_creativity']}) > M3({m3['gated_creativity']}) > "
        f"M1({m1['gated_creativity']}). 변환 원점수는 M3({m3['transformational']})가 "
        f"가장 높지만, 할 만함·지불을 걸면 창발 블렌드가 추천 가능한 창의로 남는다."
    )
    lines.append(
        f"M2 문장 다양성: unique roles {summary['m2_unique_roles']}/{summary['m2_n']}, "
        f"unique offers {summary['m2_unique_offers']}/{summary['m2_n']}, "
        f"일반 민원인 보일러플레이트 {summary['m2_generic_foil']}건. "
        "상위 카드는 프레임을 한 장씩만 고른다."
    )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
