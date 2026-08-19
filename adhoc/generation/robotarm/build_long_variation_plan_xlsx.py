#!/usr/bin/env python3
"""Build a one-sheet Long 1–4 variation plan for ten Roomba cues."""
from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

_REPO = Path(__file__).resolve().parents[3]
_OUT = _REPO / "data/results/plans/long_1_4_variation_plan_10cues.xlsx"

HEADERS = [
    "Cue #",
    "Cue",
    "Core 의미",
    "Core config",
    "Long 1 variation",
    "Long 1 config",
    "Long 2 variation",
    "Long 2 config",
    "Long 3 variation",
    "Long 3 config",
    "Long 4 variation",
    "Long 4 config",
    "설계 메모",
]

ROWS = [
    [
        1,
        "nod_yes",
        "앞뒤로 짧고 빠르게 3번 움직여 고개 끄덕임을 대체한다.",
        "path(base, line, x=[-0.15,0.15], rep=3, spd=2.0)",
        "Core 뒤에 0.8초 정지해 ‘알겠다’는 확정 상태를 만든다.",
        "CORE + hold(t=0.8)",
        "Core 앞에 상대에게 살짝 다가간 뒤 끄덕여 수락 대상을 분명히 한다.",
        "path(base,line,x=+0.10,spd=0.5) + CORE",
        "Core 앞에 좌우로 한 번 천천히 살핀 뒤, 상대를 발견하고 빠르게 끄덕인다.",
        "path(base,arc,r=0,deg=[-25,25],rep=1,spd=0.4) + CORE",
        "Core 뒤에 180도 회전하고 앞으로 이동해, 동의한 내용을 실행하러 가는 서사를 만든다.",
        "CORE + path(base,arc,r=0,deg=180,spd=0.8) + path(base,line,x=+0.30,spd=1.2)",
        "수락의 core는 항상 동일하게 유지하고 앞뒤 맥락만 바꾼다.",
    ],
    [
        2,
        "shake_head_no",
        "제자리에서 좌우로 빠르게 3번 회전해 고개 젓기를 대체한다.",
        "path(base, arc, r=0, deg=[-30,30], rep=3, spd=2.0)",
        "Core 뒤에 살짝 뒤로 물러나 멈추며 거절의 거리를 표현한다.",
        "CORE + path(base,line,x=-0.12,spd=0.5) + hold(t=0.6)",
        "앞으로 다가오다 갑자기 멈춘 뒤 core를 수행해 제안을 거절하는 장면을 만든다.",
        "path(base,line,x=+0.12,spd=0.6) + hold(t=0.3) + CORE",
        "Core 앞에 더 크고 느린 좌우 흔들기 [-45,45]를 넣어 망설이다가 확실히 거절하는 흐름을 만든다.",
        "path(base,arc,r=0,deg=[-45,45],rep=1,spd=0.35) + CORE",
        "Core 뒤에 반대 방향으로 180도 돌고 빠르게 멀어져 강한 거절과 이탈을 표현한다.",
        "CORE + path(base,arc,r=0,deg=180,spd=1.0) + path(base,line,x=+0.35,spd=1.5)",
        "Long 4는 의미가 과해지는 over-context 후보로 사용 가능하다.",
    ],
    [
        7,
        "cover_yawn_sleepy",
        "느린 반원 이동 후 조금 전진하고 오래 멈춰 큰 하품과 졸림을 표현한다.",
        "path(base,arc,r=0.3,deg=180,spd=0.5) + path(base,line,x=+0.05,spd=0.3) + hold(t=2.0)",
        "Core 뒤에 아주 천천히 뒤로 밀리며 하품 후 힘이 빠지는 모습을 만든다.",
        "CORE + path(base,line,x=-0.10,spd=0.2)",
        "Core 앞에 짧게 휘청이는 좌우 회전을 넣고 하품 동작으로 연결한다.",
        "path(base,arc,r=0,deg=[-20,20],rep=1,spd=0.35) + CORE",
        "Core 앞에 더 큰 [-45,45] 느린 흔들기와 0.7초 정지를 넣어 꾸벅꾸벅 조는 상태를 강조한다.",
        "path(base,arc,r=0,deg=[-45,45],rep=1,spd=0.25) + hold(t=0.7) + CORE",
        "Core 뒤에 180도 돌아 천천히 이동한 뒤 멈춰 잠자리로 가는 서사를 표현한다.",
        "CORE + path(base,arc,r=0,deg=180,spd=0.5) + path(base,line,x=+0.30,spd=0.45) + hold(t=1.5)",
        "rub_eye_tired와 구분되도록 하품 후 긴 정지를 핵심으로 둔다.",
    ],
    [
        8,
        "rub_eye_tired",
        "왼쪽·오른쪽으로 작은 도리도리 [-15,15]를 반복해 눈을 비비는 미세 왕복을 대체한다.",
        "path(base, arc, r=0, deg=[-15,15], rep=3, spd=0.4)",
        "Core 뒤에 살짝 앞으로 나와 멈추는 모습을 통해 하품하는 모습을 표현한다.",
        "CORE + path(base,line,x=+0.05,spd=0.3) + hold(t=1.2)",
        "살짝 휘청이며 뒤로 가서 피곤해 중심을 잃는 모습을 표현한다.",
        "CORE + path(base,arc,r=0,deg=[-20,20],rep=1,spd=0.3) + path(base,line,x=-0.15,spd=0.3)",
        "Core 앞에 왼쪽·오른쪽으로 더 크게 느리게 움직여 꾸벅꾸벅 조는 모습을 표현한다.",
        "path(base,arc,r=0,deg=[-45,45],rep=1,spd=0.25) + CORE",
        "Long 1 뒤에 반대로 180도 돌아 총총총 이동해 어딘가 자러 가는 모습을 표현한다.",
        "CORE + path(base,line,x=+0.05,spd=0.3) + hold(t=1.2) + path(base,arc,r=0,deg=180,spd=0.8) + path(base,line,x=+0.35,spd=1.5)",
        "사용자 제안 예시를 그대로 구조화했다.",
    ],
    [
        9,
        "point_you",
        "상대를 향해 짧게 전진한 뒤 정지해 ‘당신’을 지목한다.",
        "path(base,line,x=+0.12,spd=1.2) + hold(t=0.7)",
        "Core 앞에 좌우를 살피다 정면으로 맞춘 뒤 지목해 대상 선택을 표현한다.",
        "path(base,arc,r=0,deg=[-25,25],rep=1,spd=0.5) + CORE",
        "Core 뒤에 조금 후퇴해 지목한 상대와의 공간적 대비를 만든다.",
        "CORE + path(base,line,x=-0.10,spd=0.5)",
        "Core 앞에 큰 탐색 회전 후 빠르게 상대에게 접근해 ‘바로 너’라는 강조를 만든다.",
        "path(base,arc,r=0,deg=90,spd=0.5) + path(base,arc,r=0,deg=-90,spd=0.8) + CORE",
        "Core 뒤에 반원으로 상대 주위를 돌며 계속 대상을 가리키는 과도한 맥락을 만든다.",
        "CORE + path(base,arc,r=0.25,deg=180,spd=0.6) + hold(t=0.5)",
        "현재 Human GT의 hold-only보다 지목 의미가 드러나는 core 후보를 사용했다.",
    ],
    [
        10,
        "point_self",
        "180도 회전해 자신의 방향을 가리키는 상태를 만들고 정지한다.",
        "path(base,arc,r=0,deg=180,spd=0.5) + hold(t=1.5)",
        "Core 앞에 상대를 향해 잠시 정지한 뒤 자신 쪽으로 회전해 ‘나’를 대비시킨다.",
        "hold(t=0.6) + CORE",
        "Core 뒤에 아주 짧게 앞뒤 왕복해 자기 위치를 두 번 강조한다.",
        "CORE + path(base,line,x=[-0.06,0.06],rep=2,spd=0.7)",
        "Core 앞에 상대 쪽으로 한 번 전진했다가 원위치한 뒤 자신을 가리켜 you→me 대비를 만든다.",
        "path(base,line,x=+0.12,spd=0.7) + path(base,line,x=-0.12,spd=0.7) + CORE",
        "Core 뒤에 다시 정면으로 돌아 상대에게 다가가며 ‘내가 할게’라는 수행 서사를 붙인다.",
        "CORE + path(base,arc,r=0,deg=-180,spd=0.7) + path(base,line,x=+0.20,spd=0.8)",
        "point_you와 paired contrast로 평가하기 좋다.",
    ],
    [
        11,
        "beckon_come_here",
        "앞뒤로 3번 왕복해 손짓하듯 상대를 부른다.",
        "path(base,line,x=[-0.20,0.20],rep=3,spd=1.5)",
        "Core 앞에 상대 방향으로 45도 정렬한 뒤 부르는 동작을 수행한다.",
        "path(base,arc,r=0,deg=45,spd=0.8) + CORE",
        "Core 뒤에 1초 기다려 상대가 오는지 확인하는 모습을 표현한다.",
        "CORE + hold(t=1.0)",
        "Core 앞에 멀리서 상대를 찾는 큰 좌우 탐색을 넣고 발견 후 빠르게 부른다.",
        "path(base,arc,r=0,deg=[-60,60],rep=1,spd=0.45) + CORE",
        "Core 뒤에 직접 상대에게 다가갔다가 다시 뒤로 움직여 따라오라는 경로를 보여준다.",
        "CORE + path(base,line,x=+0.25,spd=0.8) + path(base,line,x=-0.25,spd=0.8) + CORE",
        "Long 4의 core 반복은 길이 증가가 의미를 강화하는지 시험한다.",
    ],
    [
        13,
        "salute",
        "빠르게 45도 회전한 뒤 2초간 각을 유지해 경례 자세를 대체한다.",
        "path(base,arc,r=0,deg=45,spd=3.5) + hold(t=2.0)",
        "Core 앞에 짧게 전진해 상대 앞에 정렬한 후 경례한다.",
        "path(base,line,x=+0.10,spd=0.8) + CORE",
        "Core 뒤에 -45도로 원위치해 경례 시작과 종료를 분명히 한다.",
        "CORE + path(base,arc,r=0,deg=-45,spd=1.0)",
        "Core 앞에 천천히 정렬하고 잠시 멈춘 후 매우 빠르게 경례해 군기 든 대비를 만든다.",
        "path(base,arc,r=0,deg=45,spd=0.4) + hold(t=0.4) + path(base,arc,r=0,deg=-45,spd=0.4) + CORE",
        "Core 뒤에 180도 돌아 직선으로 이동해 임무를 수행하러 떠나는 모습을 표현한다.",
        "CORE + path(base,arc,r=0,deg=180,spd=1.0) + path(base,line,x=+0.30,spd=1.0)",
        "빠른 core와 느린 context의 속도 대비를 유지한다.",
    ],
    [
        25,
        "facepalm",
        "앞으로 짧게 움직이다 급정지하고 오래 멈춰 당혹감에 굳는 모습을 표현한다.",
        "path(base,line,x=+0.08,spd=1.0) + hold(t=2.5)",
        "Core 앞에 작은 좌우 흔들기를 넣어 믿기 어렵다는 반응 후 굳어버리는 모습을 만든다.",
        "path(base,arc,r=0,deg=[-15,15],rep=1,spd=0.5) + CORE",
        "Core 뒤에 천천히 뒤로 물러나며 실망과 거리두기를 표현한다.",
        "CORE + path(base,line,x=-0.15,spd=0.25)",
        "Core 앞에 크게 좌우로 한 번 흔들고 짧은 정지 후 급정지 core로 연결한다.",
        "path(base,arc,r=0,deg=[-45,45],rep=1,spd=0.35) + hold(t=0.5) + CORE",
        "Core 뒤에 반대로 돌아 느리게 떠나며 ‘더는 못 보겠다’는 과장된 반응을 만든다.",
        "CORE + path(base,arc,r=0,deg=180,spd=0.5) + path(base,line,x=+0.25,spd=0.35)",
        "Roomba에는 얼굴/손이 없어 context 의존성이 큰 hard case다.",
    ],
    [
        28,
        "depict_lion",
        "빠르게 0.30m 돌진한 뒤 짧게 멈춰 사자의 포효·공격을 대체한다.",
        "path(base,line,x=+0.30,spd=3.5) + hold(t=0.3)",
        "Core 앞에 작은 좌우 스텝을 넣어 웅크려 노려보다 돌진하는 모습을 표현한다.",
        "path(base,arc,r=0,deg=[-20,20],rep=1,spd=0.5) + CORE",
        "Core 앞에 뒤로 0.15m 물러나 도움닫기 후 더 강하게 돌진한다.",
        "path(base,line,x=-0.15,spd=0.5) + CORE",
        "Core 앞에 넓은 반원을 천천히 돌며 먹잇감을 스토킹한 뒤 정면 돌진한다.",
        "path(base,arc,r=0.35,deg=180,spd=0.4) + hold(t=0.4) + CORE",
        "Core 뒤에 제자리에서 큰 좌우 회전을 반복하고 전진해 영역을 과시하는 포효 후속 동작을 만든다.",
        "CORE + path(base,arc,r=0,deg=[-60,60],rep=2,spd=1.4) + path(base,line,x=+0.15,spd=1.0)",
        "공격적 속도와 큰 공간 범위가 lion cue의 핵심이다.",
    ],
]


def build(path: Path = _OUT) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "Long1-4 Variation Plan"
    ws.append(HEADERS)
    for row in ROWS:
        ws.append(row)

    header_fill = PatternFill("solid", fgColor="1F4E78")
    long_fills = {
        5: PatternFill("solid", fgColor="D9EAD3"),
        7: PatternFill("solid", fgColor="FFF2CC"),
        9: PatternFill("solid", fgColor="FCE5CD"),
        11: PatternFill("solid", fgColor="EAD1DC"),
    }
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        for col, fill in long_fills.items():
            row[col - 1].fill = fill
            row[col].fill = fill

    widths = [8, 24, 34, 43, 44, 50, 44, 50, 48, 58, 48, 68, 38]
    for idx, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = width
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.row_dimensions[1].height = 34
    for idx in range(2, ws.max_row + 1):
        ws.row_dimensions[idx].height = 130
    ws.sheet_view.showGridLines = False
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


if __name__ == "__main__":
    print(build())
