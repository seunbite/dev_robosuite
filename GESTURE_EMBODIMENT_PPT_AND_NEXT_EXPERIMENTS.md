# PPT 구성 + 다음 실험 저장본

> 실행하지 않음. 휴먼이밸 점검, 발표 스토리보드, 필요 실험만 저장한다.
> Framing 원본: `GESTURE_EMBODIMENT_FRAMING_v2.md` · 수치 v1: `GESTURE_EMBODIMENT_HYPOTHESES.md`

---

## 0. 용어 (발표용) — 스펙트럼이 아니라 직교 축

상세 정의: `GESTURE_EMBODIMENT_FRAMING_v2.md` §0.

| 용어 | 종류 | 의미 | 측정 |
|---|---|---|---|
| **Core composition** (Approximation) | 생성 조작 A | cue identity를 담는 **core**의 필수 slot을 조합·기입 | Short/core attribute recall·F1·EM |
| **Contextual elaboration** (Sophistication) | 생성 조작 B | core **바깥**에 PRE/POST context를 붙임 (+ body channel 선택) | non-core component/channel 통계 |
| **Cue identification** | 결과 판정 | 클립을 보고 “그 제스처인지” 알아채는가 | human/VLM identification accuracy |

구분 한 줄: composition = core **안**을 채움 / elaboration = core **밖**을 붙임 / cue identification = 보고 “그 제스처인가?”를 맞히는가.

Embodiment 양극:
- **Humanoid**: composition 부담 큼 (attribute 많음). cue identification 잠재력은 큼.
- **Roomba**: composition은 쉬움. identification ceiling이 낮음.

---

## 1. 휴먼이밸 점검 — 완벽한가?

**결론: 완벽하지 않다. L2 cue identification용 human score가 없다.**

### 있는 것

| 자산 | 경로 | 상태 |
|---|---|---|
| Quadruped binary HEval pack | `data/results/human_eval/quadruped_v1/` | stimuli + protocol 준비 (63 motions → 126 items). **rater 응답/정확도 없음** |
| Mobile manip binary HEval pack | `data/results/human_eval/mobile_mani_v1/` | stimuli + protocol 준비 (110 → 220 items). **rater 응답/정확도 없음** |
| HEval 툴링 / PPTX builder | `adhoc/humaneval/` | 코드만 있음 |
| Human-authored GT motion | `render/embodiment_motion_code_human/`, `embodiment_human.html` | 5 embodiment × 30 cues GT는 있음 |
| Expert pose annotation | `verify/pilot40_pose_eval_*` | generation correctness (o/x), cue identification 아님 |

### 없는 것 / 구멍

1. **저장된 human recognition accuracy가 없음**  
   pack은 있어도 labeler 응답 sheet / aggregated accuracy / kappa가 없음.
2. **Humanoid · Roomba · Single-arm용 binary HEval pack 없음**  
   발표 스토리의 양극(humanoid–roomba)에 바로 쓸 human L2 숫자가 없음.
3. **Sophistication / Long 조건 human eval 없음**  
   `render/embodiment_long_1_4/` 렌더는 있으나 HEval item화·응답 수집 없음.
4. **Manipulator `sophisticated_v1` pack 경로 깨짐**  
   코드가 참조하지만 디스크에 패키지가 없음.
5. **Human–VLM agreement (H6) 미실행**

### 발표에 쓸 때 주의

- 지금 자신 있게 말할 수 있는 human 쪽 숫자는 **H1 attribute burden**, **Human GT 존재**, (있다면) pose expert o/x 정도.
- “cue identification 성능 차이” 슬라이드는 **현재 VLM proxy 또는 ‘예정 실험’** 으로 표시하거나, 먼저 humanoid–roomba HEval을 돌려야 함.
- Approximation 슬라이드의 “꽤 잘한다”는 **H4 attribute metrics (자동평가)** 로 지지 가능.

---

## 2. PPT 스토리보드

권장 흐름: **원형 다이어그램 → Approximation → (실패 분석) → Cue identification gap → Sophistication → 못 하는 것**.

### Slide 1. Title / claim

- Gesture embodiment: Approximation vs Cue identification
- One-liner: LLMs approximate across embodiments; they fail differently on the way to recognition.

### Slide 2. 원형 다이어그램 매핑 피규어

중앙/링 구조로 세 층을 그림:

```text
          [ Cue identification ]
         /         |         \
        /    sophistication   \
       /     (context before/  \
      /       after + body     \
     /        channel)          \
    [ -------- Approximation -------- ]
              (core motion)
```

또는 동심원:
1. 안쪽: **Approximation** = core
2. 중간: **Sophistication** = PRE/POST context + body-part channel
3. 바깥: **Cue identification** = human/VLM이 cue로 읽음

옆에 embodiment 화살표:
- Humanoid: core로도 바깥 원에 닿기 쉬움 / 안쪽 원 채우기는 비쌈
- Roomba: 안쪽 원은 쉽게 채움 / 바깥 원에 잘 못 닿음

### Slide 3. Approximation 정의

- “가장 핵심적인 동작을 하는 것”
- Short / core only
- Metric: required-attribute recall, F1, EM

### Slide 4. Humanoid vs Roomba — Approximation 성능

표 (v1 H4, Top-5 Short):

| Embodiment | Required recall | Attr F1 | Instance EM |
|---|---:|---:|---:|
| Humanoid | 80.1% | 83.0% | 36.7% |
| Roomba | 83.9% (subset 89.5%) | 83.5% (89.4%) | 53.3% (64.7%) |

메시지: **둘 다 approximation은 꽤 잘한다 (recall/F1 높음).**

### Slide 5. Approximation에서 못 하는 것

Humanoid 중심:
- EM 붕괴 → exactness 실패
- 취약 attribute: gripper, path.axis, pose xyz (MCQ/H4 오류 분석)
- 해석: slot이 많아서 precision erosion

Roomba 쪽 한 줄:
- L1 지표는 괜찮지만 fully possible 셀이 적음 → “맞게 채움”과 “형태적으로 충분한 표현”은 다름

### Slide 6. Cue identification 기준이면 격차가 이렇게 남

- 같은 approximation 성공 cell이라도 식별률이 embodiment별로 갈라짐
- **예상 패턴 (검증 필요):**
  - Humanoid: A↑ → R도 비교적 따라옴
  - Roomba: A↑ 여도 R 낮음
- 현재 상태 뱃지: `HUMAN EVAL INCOMPLETE` / VLM proxy only if used

### Slide 7. Sophistication 정의

- “앞뒤로 context를 부여하는 것”
- 반드시 **신체부위/채널 정의**를 포함
  - Humanoid: head / arm / torso / bimanual …
  - Roomba: translate / rotate / arc / hold …
- Long1–4 또는 PRE–CORE–POST 구조로 시각화

### Slide 8. Sophistication 진행 → Cue identification 상승?

| Condition | Humanoid I | Roomba I | ΔI |
|---|---:|---:|---:|
| Approx (Short) | TBD | TBD | — |
| Soph (Long/scaffold) | TBD | TBD | TBD |

예측 메시지:
- Humanoid: 소폭~중간 상승 (이미 core만으로도 많이 식별됨)
- Roomba: 상승은 하지만 ceiling이 낮음 / 또는 기대보다 안 오름

### Slide 9. Sophistication 후에도 못 하는 것

분석 축:
- 어떤 body channel을 잘못 고르는지
- context가 cue와 causal하게 무관한 경우
- 길이만 늘고 identity가 안 사는 경우
- evaluate는 맞추는데 correct로 연결 못 하는 경우 (C4)

### Slide 10. Takeaway

1. Approximation competence is shared  
2. Humanoid fails by imprecision  
3. Roomba fails by weak cue identification / under-development  
4. Sophistication is the proposed bridge — but must be measured on R, not only on attribute F1  
5. Human eval for R is the missing pillar

---

## 3. 필요한 실험 (저장만 · 실행 금지)

우선순위 순.

### P0. Humanoid–Roomba Cue identification HEval (발표 핵심)

**목적:** Slide 6 숫자를 인간 기준으로 채움.

**설계**
- Stimuli: Human GT Short + LLM Short (동일 cue set)
- Embodiments: **Humanoid, Roomba** (최소). 여유 있으면 Tiago.
- Task: binary cue–video match (기존 quadruped/mobile pack 프로토콜 재사용)
- Raters: ≥ N human (배치 분할)
- 지표: accuracy, yes/no bias, cue-wise, embodiment-wise
- 부가: 가능하면 동일 item을 VLM에도 돌려 Human–VLM agreement

**산출**
- `P(R | HumanGT)`, `P(R | LLM-Approx)` by embodiment
- A-score×I-score contingency (E3)

**의존성**
- humanoid/roomba용 binary pack 생성 (현재 없음)
- 렌더 GIF 경로 정리

### P1. Sophistication → ΔCue identification

**목적:** Slide 8–9.

**설계**
- 동일 cue × embodiment에서 Short vs Soph(Long-best 또는 Long2/3) 쌍
- Body-part/channel 라벨을 config에서 추출해 기록
- Human(+VLM) recognition
- 지표: ΔI, channel별 성공/실패, “길이↑ but R↓” 사례

**예측 기록용**
- Roomba: ΔI 작음 또는 불안정
- Humanoid: ΔI 양수이나 Short 대비 한계이득 감소

### P2. Approximation failure anatomy (이미 데이터 재집계 가능)

**목적:** Slide 5를 더 날카롭게.

**실행 없이 준비만**
- H4 + English glossary MCQ를 Humanoid vs Roomba로
  - recall / precision / EM
  - attribute-type error ranking
- “못 하는 attribute” Top-k 표 고정

### P3. Evaluate vs Correct (C4) — PPT 후반 또는 다음 버전

**목적:** sophistication 실패를 repair로 못 고친다는 주장.

**설계**
- evaluate → localize → correct → re-eval
- 지금은 PPT 필수 슬라이드보다 appendix로 둠

### Explicitly deferred

- 전체 5 embodiment × Long1–4 full factorial human study  
- Manipulator sophisticated_v1 pack 복구  
- Gemini MCQ 행 채우기  

---

## 4. 당장 PPT에 넣을 수 있는 숫자 vs 구멍

| 슬라이드 | 지금 가능 | 막힌 것 |
|---|---|---|
| 원형 다이어그램 | 개념도 | — |
| Approx 성능 | H4 recall/F1/EM | — |
| Approx 실패 | H4 EM + MCQ attribute 취약점 | precision을 H4에서 재표로 뽑으면 더 좋음 |
| Cue identification 격차 | 개념 + 예측 | **human identification 숫자 없음 (P0)** |
| Soph → ΔI | 개념 + 렌더 존재 | **human/VLM R before-after 없음 (P1)** |
| Soph 실패 분석 | 프레임만 | P1 결과 필요 |

---

## 5. 다음 액션 체크리스트 (아직 실행 안 함)

- [ ] P0: humanoid/roomba binary HEval pack 생성 스크립트 계획
- [ ] P0: rater 배치·프로토콜 확정 (기존 `EVAL_PROTOCOL.md` 이식)
- [ ] P1: Short vs Soph 쌍 stimulus list (cue × embodiment)
- [ ] P2: H4 precision 재집계 표 초안
- [ ] PPT 피규어: 원형 다이어그램 1장 먼저 고정
- [ ] 휴먼이밸 완료 전까지 Slide 6에 `preliminary / to-collect` 라벨 유지

---

## 6. 한 줄 상태

> Approximation 이야기는 지금 숫자로 가능하고, Cue identification·Sophistication 효과 이야기는
> **humanoid–roomba human eval (P0/P1)이 끝나야** 슬라이드가 닫힌다.
