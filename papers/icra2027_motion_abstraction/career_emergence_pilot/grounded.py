"""Occupation-grounded blend payloads so offers are not one boilerplate."""

import re

FRAME_IDS = [
    "eol_audit",
    "train_foil",
    "hh_protocol",
    "muni_evidence",
    "dispute_docent",
    "tandem",
    "dropout",
    "signal_license",
]


def noun_from_title(title: str) -> str:
    t = title.split(",")[0].strip()
    for noise in (
        "Workers", "Specialists", "Scientists", "Managers", "Officers",
        "Technicians", "Technologists", "Assistants", "Teachers", "Directors",
    ):
        t = t.replace(noise, "").strip()
    return t or title.split()[0]


def _hits(blob: str, keys: tuple[str, ...]) -> bool:
    for k in keys:
        if " " in k:
            if k in blob:
                return True
        elif re.search(rf"(?<![a-z]){re.escape(k)}", blob):
            return True
    return False


def ground_occupation(p: dict) -> dict:
    title_blob = p["title"].lower()
    desc_blob = (p.get("desc") or "").lower()
    noun = noun_from_title(p["title"])
    g = {
        "artifact": noun,
        "dead_thing": f"중단된 {noun} 과제",
        "counterpart": f"{noun} 업무에서 실제로 막히는 상대",
        "buyer": "해당 기관 운영·품질 예산",
        "complement": "일정·장부 파트너",
        "scene": "현장",
        "signal": "작업 리듬",
        "evidence": "기록",
        "noun": noun,
    }
    rules = [
        (("software", "computer", "informatic", "programmer", "database", "web developer"), dict(
            artifact="모델·로그·권한",
            dead_thing="폐기 모델과 미회수 접근권",
            counterpart="장애 난 배포를 재연해야 하는 상대",
            buyer="보안·컴플라이언스 예산",
            complement="도메인 현장 전문가",
            scene="스테이징",
            signal="온콜 알람",
            evidence="접근 로그",
        )),
        (("clergy", "counsel", "social work", "psycholog"), dict(
            artifact="사례 판단",
            dead_thing="종결하지 못한 사례",
            counterpart="침묵하거나 폭발하는 내담자",
            buyer="슈퍼비전·수련 예산",
            complement="행정",
            scene="상담실·가정",
            signal="침묵의 길이",
            evidence="사례 노트",
        )),
        (("nurse", "medical", "health", "dental", "physician", "clinical", "therapist"), dict(
            artifact="인수인계·퇴원 판단",
            dead_thing="끝나지 않은 케어 플랜",
            counterpart="설명이 안 되는 보호자",
            buyer="병원 환자경험·교육 예산",
            complement="가사·이동 도우미",
            scene="병동·가정",
            signal="바이탈 알람",
            evidence="차트",
        )),
        (("teacher", "instructor", "professor", "librarian", "curriculum"), dict(
            artifact="커리큘럼 구멍",
            dead_thing="중도 포기한 코스",
            counterpart="수업을 멈추게 하는 학부모·학생",
            buyer="학교·부트캠프 HRD",
            complement="특수교육 보조",
            scene="교실",
            signal="집중이 깨지는 소음",
            evidence="출결·과제 이탈",
        )),
        (("animal", "veterinar", "zookeeper", "wildlife", "groomer"), dict(
            artifact="사육·훈련 루틴",
            dead_thing="포기한 훈련 과제",
            counterpart="물거나 멈추는 동물·보호자",
            buyer="보호소·농장 교육 예산",
            complement="수의·운송",
            scene="켄넬·목장",
            signal="급여·운동 주기",
            evidence="행동 로그",
        )),
        (("police", "fire", "detective", "correction", "security", "gambling", "sheriff"), dict(
            artifact="조서·식별 기록",
            dead_thing="종결 안 된 사건 파일",
            counterpart="진술이 엇갈리는 목격자",
            buyer="훈련기관·사업장 컴플라이언스 교육",
            complement="통역·외상 상담",
            scene="지구대·사업장",
            signal="교대 사이 대기",
            evidence="CCTV·지문",
        )),
        (("finance", "account", "auditor", "bank", "tax", "examiner", "actuary", "credit"), dict(
            artifact="장부·위반 흔적",
            dead_thing="닫지 못한 감사 이슈",
            counterpart="증빙을 안 내놓는 담당자",
            buyer="내부통제·컴플라이언스 교육",
            complement="현장 실사",
            scene="장부·현장",
            signal="마감 직전 침묵",
            evidence="분개·증빙",
        )),
        (("engineer", "architect", "surveyor", "civil", "mechanical", "electro"), dict(
            artifact="도면·계측값",
            dead_thing="중단된 시공 도면",
            counterpart="감리와 충돌하는 시공사",
            buyer="지자체 안전·하자 예산",
            complement="행정 서류",
            scene="현장 컨테이너",
            signal="장비 소음 주기",
            evidence="측량 포인트",
        )),
        (("art", "artist", "arts", "designer", "media", "film", "music", "writer", "animat", "photographer", "graphic"), dict(
            artifact="시안·원본 파일",
            dead_thing="죽은 캠페인 시안",
            counterpart="승인만 미루는 클라이언트",
            buyer="광고주 교육·IP 분쟁 부가",
            complement="세무·계약",
            scene="스튜디오·창고",
            signal="편집 타임라인",
            evidence="버전 히스토리",
        )),
        (("lawyer", "legal", "judge", "court", "paralegal"), dict(
            artifact="조항과 당사자 서사",
            dead_thing="합의 결렬 사건",
            counterpart="서류를 읽지 못하는 의뢰인",
            buyer="로펌 클라이언트 교육 부가",
            complement="해설·통역",
            scene="상담실",
            signal="기일 전 침묵",
            evidence="녹취·계약",
        )),
        (("farm", "agricult", "forest", "fish", "soil", "crop"), dict(
            artifact="작기·토양 기록",
            dead_thing="포기한 작기",
            counterpart="수매 기준을 들이미는 유통",
            buyer="농협·군 농업 예산",
            complement="직판·물류",
            scene="밭·위판장",
            signal="물주기·수확",
            evidence="작황",
        )),
        (("cook", "chef", "food", "waiter", "bartender"), dict(
            artifact="레시피·동선",
            dead_thing="메뉴에서 내린 요리",
            counterpart="위생 점검관",
            buyer="프랜차이즈·보건 교육",
            complement="회계",
            scene="키친",
            signal="피크 타임",
            evidence="온도 로그",
        )),
        (("truck", "driver", "locomotive", "pilot", "transport", "bus"), dict(
            artifact="대기·배차",
            dead_thing="공차 운행",
            counterpart="하차를 늦추는 화주",
            buyer="물류 교육·마을 교통 예산",
            complement="창고 피킹",
            scene="차고지",
            signal="공회전",
            evidence="타코그래프",
        )),
        (("repair", "install", "mechanic", "maintenance", "plumber", "electric"), dict(
            artifact="고장 계보",
            dead_thing="반복 고장 설비",
            counterpart="‘일단 막아 달라’는 건물주",
            buyer="관리사무소 유지비",
            complement="세무·견적",
            scene="기계실",
            signal="이상 진동",
            evidence="A/S 이력",
        )),
        (("sales", "retail", "cashier", "telemarketer"), dict(
            artifact="거절·환불 문장",
            dead_thing="팔다 만 프로모션",
            counterpart="환불을 요구하는 고객",
            buyer="본사 교육 예산",
            complement="물류",
            scene="매장",
            signal="피크 줄",
            evidence="CS 티켓",
        )),
        (("clerk", "secretary", "receptionist", "typist", "correspondence"), dict(
            artifact="문서 공백·회신",
            dead_thing="닫지 못한 민원 철",
            counterpart="기한을 밀어붙이는 발신자",
            buyer="민원·총무 교육 예산",
            complement="현장 확인",
            scene="민원창구",
            signal="미회신 쌓임",
            evidence="접수 대장",
        )),
        (("compliance", "environment", "occupational health"), dict(
            artifact="규제 위반 흔적",
            dead_thing="시정명령을 못 닫은 건",
            counterpart="현장 핑계를 대는 관리자",
            buyer="사업장 환경·안전 교육",
            complement="공정 오퍼레이터",
            scene="사업장",
            signal="점검 전 침묵",
            evidence="측정 대장",
        )),
        (("production", "machin", "weld", "assembl", "operator", "inspector"), dict(
            artifact="공정 리듬",
            dead_thing="스크랩 난 로트",
            counterpart="라인을 멈추는 품질 검사",
            buyer="공장 안전·민원 예산",
            complement="보전",
            scene="라인",
            signal="프레스 주기",
            evidence="불량률",
        )),
        (("manager", "chief", "director"), dict(
            artifact="결정 잔여·미결 안건",
            dead_thing="폐기된 이니셔티브",
            counterpart="이사회에서 되묻는 질문자",
            buyer="거버넌스·감사 예산",
            complement="현장 실무",
            scene="회의실 밖 현장",
            signal="에스컬레이션",
            evidence="의사록",
        )),
    ]
    matched = False
    for keys, upd in rules:
        if _hits(title_blob, keys):
            g.update(upd)
            matched = True
            break
    if not matched:
        for keys, upd in rules:
            if _hits(desc_blob, keys):
                g.update(upd)
                break
    g["noun"] = noun
    return g


def blend_payload(p: dict, frame_id: str) -> dict:
    g = ground_occupation(p)
    loc = p["location"]
    noun = g["noun"]
    if frame_id == "eol_audit":
        return dict(
            id=frame_id, ko="종료·부검", bent="성공=성장",
            role=f"{noun} 종료 부검가",
            offer=(
                f"{p['title']}가 다루는 {g['artifact']} 중 {g['dead_thing']}을 유료로 부검한다. "
                f"산출은 새 프로젝트가 아니라 ‘왜 끝났는지’와 잔여 위험({g['evidence']})이다."
            ),
            payer=g["buyer"],
            day90=f"{g['dead_thing']} 2건을 {g['scene']} 기준으로 익명 부검해 전 동료에게 보낸다",
            frames=f"[{p['title']}] ⊗ [종료 부검 of {g['artifact']}]",
            comb=5, trans=3, c2=5,
        )
    if frame_id == "train_foil":
        return dict(
            id=frame_id, ko="직무 특화 상대역", bent="자격자만 그 일을 한다",
            role=f"{g['counterpart']} 재연 훈련",
            offer=(
                f"{p['title']} 신입이 실제로 막히는 상대는 일반 민원인이 아니라 {g['counterpart']}다. "
                f"{g['scene']}에서 그 마찰만 재연하는 유료 훈련으로 판다."
            ),
            payer=g["buyer"],
            day90=f"{g['counterpart']} 대본 5개를 {g['buyer']} 담당에게 파일럿으로 제안한다",
            frames=f"[{p['title']}] ⊗ [직무 특화 상대역]",
            comb=5, trans=4, c2=5,
        )
    if frame_id == "hh_protocol":
        return dict(
            id=frame_id, ko="구독 프로토콜", bent="고용주=직장",
            role=f"{noun} 판단 문장 구독",
            offer=(
                f"{p['title']}의 {g['artifact']} 판단 문장을 여러 {g['scene']}이 구독한다. "
                f"{loc}에서 상주 고용 없이 설계와 짧은 온콜만 판다."
            ),
            payer=f"{g['scene']} 운영비",
            day90=f"한 {g['scene']}의 {g['artifact']} 문장 세트를 유료 납품한다",
            frames=f"[{p['title']}] ⊗ [다현장 구독]",
            comb=4, trans=4, c2=4,
        )
    if frame_id == "muni_evidence":
        return dict(
            id=frame_id, ko="지역 증거 임대", bent="고객=원청",
            role=f"{loc} {g['evidence']} 임차인",
            offer=(
                f"{p['title']}의 {g['evidence']} 잡는 법을 {loc} 명의 데이터로 임대한다. "
                f"원청 보고서가 아니라 공청회·존치용 한 장이다."
            ),
            payer=f"{loc} 지자체·협의체 용역",
            day90=f"{loc} 협의체와 2주 파일럿으로 {g['evidence']} 1장을 만든다",
            frames=f"[{p['title']}] ⊗ [지역 증거]",
            comb=4, trans=4, c2=4,
        )
    if frame_id == "dispute_docent":
        return dict(
            id=frame_id, ko="분쟁 도슨트", bent="해설은 본업 장소에서만",
            role=f"{g['artifact']} 분쟁 도슨트",
            offer=(
                f"법률가가 {g['artifact']}의 내력을 설명하지 못한다. "
                f"{p['title']}가 {g['scene']}에서 {g['evidence']}를 90분 투어로 해설하고 수임은 부가로 붙는다."
            ),
            payer="법률·조정 부가 수임",
            day90="사무소 1곳에 부가 상품 안내를 넣고 모의 투어 1회",
            frames=f"[{p['title']}] ⊗ [분쟁 해설]",
            comb=5, trans=3, c2=4,
        )
    if frame_id == "tandem":
        return dict(
            id=frame_id, ko="탠덤 청구", bent="직업=개인 단위",
            role=f"{noun} + {g['complement']} 한 청구서",
            offer=(
                f"고객은 {p['title']}만 사면 {g['complement']} 구간에서 끊긴다. "
                f"둘을 한 계약으로 팔아 {g['scene']}의 빈틈을 판다."
            ),
            payer=f"{g['scene']} 기존 외주를 합친 청구",
            day90=f"{g['complement']} 파트너와 공동 견적을 {g['scene']} 3곳에 보낸다",
            frames=f"[{p['title']}] ⊗ [{g['complement']}]",
            comb=4, trans=5, c2=4,
        )
    if frame_id == "dropout":
        return dict(
            id=frame_id, ko="이탈 설계", bent="상품=완주·합격",
            role=f"{noun} 이탈 지점 설계자",
            offer=(
                f"{p['title']} 경로에서 사람들이 놓는 지점({g['dead_thing']})을 기관이 사게 한다. "
                f"성공 후기가 아니라 이탈 맵이다."
            ),
            payer="교육·수련 기관의 이탈 방지 예산",
            day90=f"{g['dead_thing']} 관련 인터뷰 4건을 맵으로 만들어 교육담당에게 판다",
            frames=f"[{p['title']}] ⊗ [이탈 맵]",
            comb=4, trans=4, c2=4,
        )
    return dict(
        id=frame_id, ko="현장 신호 라이선스", bent="측정은 리포트, 리듬은 예술",
        role=f"{g['signal']} 라이선스",
        offer=(
            f"{p['title']}가 이미 듣는 {g['signal']}를 {g['scene']}의 주의·휴식 신호로 바꿔 라이선스한다. "
            f"공연 티켓이 아니라 안전·민원 예산이다."
        ),
        payer="안전·환경·민원 예산",
        day90=f"{g['scene']}에서 {g['signal']} 30분을 채집해 프로토타입 1개를 납품한다",
        frames=f"[{p['title']}] ⊗ [신호 라이선스]",
        comb=5, trans=3, c2=4,
    )


def rulebend_payload(p: dict) -> dict:
    g = ground_occupation(p)
    rules = [
        ("고용주=직장", f"여러 {g['scene']}이 {g['artifact']} 구독으로 공동 발주하게 한다"),
        ("직업=개인 단위", f"{g['complement']}와 한 청구서로만 {g['scene']}에 판다"),
        ("성공=승진·성장", f"{g['dead_thing']}을 유료 산출로 판다"),
        ("일은 자격 정점만", f"{g['counterpart']} 경험을 기관 훈련으로 판다"),
        ("고객=원청", f"{g['buyer']}처럼 이미 싸는 칸으로 판다"),
    ]
    bent, tactic = rules[hash(p["id"] + p["code"]) % len(rules)]
    return dict(
        bent=bent,
        tactic=tactic,
        role=f"{g['noun']} 문법 전환 ({bent.split('=')[0]})",
        offer=(
            f"유급은 유지한다. 깨는 규칙: {bent}. "
            f"{p['title']}의 {g['artifact']}는 그대로 두고, {tactic}."
        ),
        payer=g["buyer"],
        day90=f"{tactic}는 문장 상품으로 잠재 발주자 4명에게 보낸다",
        frames=f"R' = R − {{{bent}}}; artifact={g['artifact']}",
    )
