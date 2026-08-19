# Gesture × Embodiment Framing v2

> `GESTURE_EMBODIMENT_HYPOTHESES.md`는 v1으로 유지한다. 이 문서는 결론 프레이밍을
> 다시 잡고, 그 결론을 검증하기 위한 실험 설계를 정리한다.

## 0. 용어 정의 (직교 분리)

스펙트럼처럼 느껴지는 이유는 Approximation / Sophistication을
“덜 완성 ↔ 더 완성”의 **정도**로 쓰기 때문이다. 아래처럼 **서로 다른 축**으로 고정한다.

### 0.1 세 개념은 한 줄의 다른 종류의 것이다

| 용어 | 종류 | 한 줄 정의 | 무엇으로 판정하지 않는가 |
|---|---|---|---|
| **Core composition** (≈ Approximation) | **생성 조작 A** | cue identity를 담는 **core unit**의 필수 control slot을 조합·기입하는 것 | 인식률이 높다고 core가 맞은 것은 아님 |
| **Contextual elaboration** (≈ Sophistication) | **생성 조작 B** | 이미 정해진 core **바깥**에 PRE/POST(또는 보조 채널) component를 **구조적으로 추가**하는 것 | component가 많다고 더 “좋은 approximation”은 아님 |
| **Cue identification** | **결과 지표** | 사람이(또는 VLM이) 최종 렌더를 보고 “이 cue인가?”를 맞히는지 | 생성 전략의 이름이 아님 |

핵심 문장:

> Approximation과 Sophistication은 cue identification의 약한/강한 버전이 아니다.
> 전자는 **core 안의 slot filling**, 후자는 **core 밖의 structural expansion**,
> Cue identification은 관찰자가 그 결과물을 보고 “이 제스처인가?”를 맞히는 **판정**이다.

### 0.2 조작 정의 (operational)

**Core composition / Approximation**
- 대상: `core` 구간의 pose/move/path 등
- 하는 일: required semantic slots 선택·조합·값 기입
  (body channel, direction, repetition, shape, sign, …)
- 단위: Short motion = core-only config
- 측정: core required-attribute recall / precision / F1 / EM
- 포함하지 않음: PRE/POST 추가, 길이 늘리기

**Contextual elaboration / Sophistication**
- 대상: core 밖의 component (setup / emphasis / settle / recovery 등)
- 하는 일: 어느 **신체부위·채널**로 context를 붙일지 정하고 앞뒤에 붙임
- 단위: Long / scaffolded config; core span은 별도로 표시
- 측정: non-core component 수, role 분포, channel 분포, core 대비 길이
- 포함하지 않음: core slot을 더 정확히 맞추는 행위
  (그건 여전히 approximation / core refinement)

**Cue identification**
- 대상: 렌더된 전체 클립
- 측정: human/VLM cue recognition accuracy
- Short여도 R이 높을 수 있고, Long이어도 R이 낮을 수 있음

### 0.3 왜 스펙트럼이 아닌가 (2×2)

|  | Elaboration = 없음 (Short) | Elaboration = 있음 (Long) |
|---|---|---|
| **Core composition 성공** | 최소 유효 제스처 | scaffolded 제스처 |
| **Core composition 실패** | 빈약한/틀린 core | 바쁜데 identity는 약한 동작 |

 Cue identification은 이 표의 칸마다 달라질 수 있는 **별도 축**이다.
 따라서 “살을 붙인다 = approximation을 더 한다”가 아니다.
 살을 붙이는 것은 elaboration이고, core를 고치는 것은 composition이다.

### 0.4 이름 대응

발표/기존 문서와의 호환:

| 이 문서의 기본 용어 | 동의어로 허용 | 피해야 할 쓰임 |
|---|---|---|
| Core composition | Approximation, core encoding | “대충 알아보임”, “약한 cue identification” |
| Contextual elaboration | Sophistication, scaffolding | “더 잘 알아보임”, “높은 cue identification” |
| Cue identification | identification / “이 제스처인가?” 판정; (구) recognizability | 생성 전략 이름; “알아챌 수 있는 정도” 같은 모호어 |

권장: 논문·PPT에서는 **Composition / Elaboration / Cue identification** 三联을 쓰고,
Approximation / Sophistication은 괄호 병기만 한다.

## 1. 핵심 프레이밍

기존의 “잘한다 / 못한다”를 embodiment별로 흩뿌리지 말고,
**생성 조작(composition vs elaboration)** 과 **결과(cue identification)** 를 나눈다.

| Layer | 이름 | 질문 | 주로 보는 것 |
|---|---|---|---|
| Gen-A | **Core composition** | cue의 핵심 semantic slots를 core에 빠짐없이, 대체로 맞게 채웠는가? | attribute recall / F1 / EM |
| Gen-B | **Contextual elaboration** | core 밖에 cue-supporting context를 구조적으로 붙였는가? | non-core components, channels |
| Out | **Cue identification** | 사람이 클립을 보고 그 cue인지 아닌지 알아채는가? | forced-choice / yes-no identification accuracy |

이 구분이 필요한 이유는, embodiment마다 composition 부담과 cue identification 가능성이
반대 방향으로 움직이기 때문이다.

```text
High-resolution embodiment (Humanoid)
  Core composition     : hard   ← 채워야 할 attribute가 많음
  Cue identification      : easier ← 보고 그 cue인지 맞히기 쉬움
  Elaboration 이득     : 상대적 작을 수 있음 (core만으로도 식별됨)

Low-resolution embodiment (Roomba)
  Core composition     : easy   ← attribute 공간이 작음
  Cue identification      : hard   ← 보고도 그 cue인지 맞히기 어려움
  Elaboration 이득     : 필요하지만 ceiling이 낮을 수 있음
```

한 문장으로 쓰면:

> Embodiment semantic resolution은 **core composition 부담**을 키우고, **cue identification 가능성**을 키운다.
> Contextual elaboration은 둘을 잇는 별도 조작이지, composition의 “더 강한 버전”이 아니다.

## 2. 결론 프레이밍 (수정안)

원안의 네 문장을 아래처럼 다듬는 것을 제안한다.

### C1. Approximation competence is shared

Humanoid든 Roomba든, LLM은 Short/core motion의 **required attribute를 채우는 L1 근사**는
비교적 잘한다.

- 지지 신호(v1): H4 required recall이 전 embodiment에서 높음
  (Humanoid 80.1%, Roomba 83.9–89.5%).
- 해석: “못 만든다”가 아니라, “대략 맞는 skeleton은 만든다”.

### C2. High-resolution failure mode = precision erosion

Humanoid에서는 L1이 완벽하지 않다. slot이 많아서 **precision이 깎이고**, 일부 attribute는
체계적으로 틀린다. 그 결과 instance exact match가 낮아진다.

- 지지 신호(v1): Humanoid recall은 높지만 EM이 36.7%로 낮음.
- Binary MCQ에서도 gripper / path.axis / pose 계열이 취약.
- 해석: Humanoid 병목은 “무엇을 써야 하는지 모름”보다
  “많은 자유도 중 정확한 값을 고르지 못함”.

### C3. Low-resolution failure mode = cue identification ceiling

Roomba는 L1 근사가 쉽고 실제로도 잘한다. 그러나 그 결과가 **cue identification으로 잘 승격되지
않는다**. 간단한 plausible motion은 만들지만, “그 제스처다”라고 식별될 만큼 develop하지 못한다.

- 지지 신호(v1): Roomba의 attribute 지표는 높은 편이나, fully possible 셀이 적고
  (4/30), approximate 비율이 큼.
- 해석: Roomba 병목은 attribute filling이 아니라,
  low-resolution morphology 위에서 cue identity를 살리는 expressive development.

### C4. Critique ≠ Repair

LLM은 motion을 **평가(evaluate)** 하는 데는 강하고, 그 평가를 **수정(correct/repair)** 으로
연결하는 데는 약하다.

- 잠정 주장: cue identification / binary attribute judgment / “wrong vs right” 판정은 가능해도,
  틀린 slot을 찾아 고친 config를 안정적으로 재작성하지는 못함.
- 이 부분은 v1에 확정 숫자가 부족하므로, 아래 실험군 E4가 핵심 confirmatory test다.

### 한 장 요약

| Embodiment | L1 Approximation | L2 Cue identification | Dominant LLM failure |
|---|---|---|---|
| Humanoid | 가능하지만 비용 큼 | 잠재적으로 높음 | precision / exactness |
| Roomba | 쉽고 잘함 | 낮게 천장 형성 | under-development / weak cue identity |
| 공통 | attribute skeleton 생성 OK | — | evaluate > correct |

## 3. 기존 주장과의 정렬

이 프레이밍은 v1 가설을 폐기하지 않고 재배치한다.

| v1 | v2에서의 위치 |
|---|---|
| H1 attribute burden | L1 난이도의 embodiment 비대칭을 설명하는 capacity claim |
| H2 cue identification | L2 outcome |
| H3 / H3a length | Roomba의 L1→L2 승격 수단 후보 (scaffolding) |
| H4 short generation | C1/C2의 주 증거 |
| H5 long asymmetry | C3를 component 수준으로 분해 |
| H6 Human–VLM agreement | L2 평가 신뢰도 |
| binary MCQ | C2의 attribute-level failure anatomy |

## 4. 실험 설계

목표: C1–C4를 각각 독립적으로 falsifiable하게 만든다.

### E1. L1 Approximation benchmark (이미 상당 부분 있음)

**목적:** C1/C2 확인.

**설계**
- 입력: cue + embodiment + Short generation prompt
- 출력: motion code / config
- 지표:
  - required-attribute recall
  - precision
  - F1
  - instance exact match
  - attribute-type별 error rate

**예측**
- Recall: Humanoid ≈ Roomba로 높음 (C1)
- Precision / EM: Humanoid < Roomba (C2)
- Humanoid 오류는 high-cardinality slot
  (`pose.x/y/z`, `gripper_orientation`, `path.axis`)에 집중

**현재 상태:** H4 + binary MCQ가 초안 증거. confirmatory를 위해
required/optional slot dual coding과 사전등록된 성공 기준이 필요.

### E2. L2 Cue identification benchmark

**목적:** C3와 “L1 성공 ≠ L2 성공”을 분리.

**설계**
- 동일 cue × embodiment에 대해 세 종류의 렌더를 만듦:
  1. Human GT
  2. LLM Short (L1-strong sample)
  3. LLM Long-best (scaffolded sample)
- 평가자: human + VLM
- 과제: forced-choice cue recognition 또는 open-set cue ID

**예측**
- Humanoid: L1이 어느 정도만 되어도 L2가 비교적 빨리 올라감
- Roomba: L1 metric이 좋아도 L2가 낮게 유지됨
- 따라서 `cue identification | high attribute F1` 조건에서 Roomba << Humanoid

**핵심 분석**
- `P(identify | approximate)` 를 embodiment별로 비교
- 이것이 C3의 직접 검정이다.

### E3. Approximation-to-Cue identification gap

**목적:** “근사까지는 되는데 인식까지는 안 된다”를 수치로 고정.

각 cell에 두 점수를 붙인다.

| Score | 정의 |
|---|---|
| A-score | attribute F1 또는 required recall이 임계값 이상이면 1 |
| I-score | human/VLM가 해당 cue로 식별하면 1 |

그다음 contingency를 embodiment별로 본다.

| | I=1 | I=0 |
|---|---|---|
| A=1 | 성공적 승격 | **근사만 된 실패** |
| A=0 | 드물게 운이 좋은 인식 | 완전한 실패 |

**예측**
- Humanoid: A=1,I=0 비율이 상대적으로 낮음
- Roomba: A=1,I=0 비율이 높음 ← C3의 중심 cell

### E4. Evaluate vs Correct asymmetry

**목적:** C4 확인.

같은 LLM에게 세 작업을 연쇄적으로 준다.

1. **Evaluate:** 렌더 또는 config가 cue에 적절한가? (yes/no + 이유)
2. **Localize:** 틀린 attribute / component를 지적
3. **Correct:** 수정된 full config 생성
4. **Re-evaluate:** 수정본을 다시 평가

**지표**
- Evaluate accuracy
- Localization accuracy (attribute-level)
- Correction success = 수정 후 A-score 또는 I-score 상승
- Self-consistency: “wrong”이라고 판정한 뒤 고친 결과가 실제로 나아졌는지

**예측**
- Evaluate >> Correct
- Localization은 중간
- “wrong을 맞힌 뒤에도 correction이 A/R를 올리지 못하는” 경우가 많음

이 실험이 C4를 단독으로 세운다.

### E5. Roomba development test

**목적:** C3가 “못 만듦”이 아니라 “develop을 못 함”인지 구분.

Roomba에 대해 Short 대비 의도적 development 조건을 둔다.

| Condition | 내용 |
|---|---|
| Short | core only |
| Long-scaffold | setup / emphasis / settle를 명시적으로 요청 |
| Contrastive | “같은 trajectory라도 cue identity가 보이게” 제약 |
| Feedback-repair | E4의 evaluate 결과를 주고 재작성 |

**예측**
- Attribute F1는 조건 간 비슷하거나 Long에서 약간 상승
- Cue identification은 Humanoid보다 훨씬 덜 상승
- 즉 development prompt가 L1은 바꿔도 L2 ceiling을 잘 못 뚫음

### E6. Humanoid precision stress test

**목적:** C2를 attribute family별로 고정.

Humanoid만 따로, high-cardinality slot ablation:

- direction / repetition / left_arm 같은 low-entropy slot
- pose xyz / gripper / path axis 같은 high-entropy slot

**예측**
- low-entropy: 높은 accuracy
- high-entropy: precision 붕괴의 주원인
- 따라서 Humanoid의 낮은 EM은 “전체를 못해서”가 아니라
  “어려운 slot 몇 개가 exactness를 무너뜨려서”

## 5. 추천 실험 순서

현실적인 진행 순서:

1. **E1 정리** — 기존 H4/MCQ를 L1 표로 재집계 (바로 가능)
2. **E2 + E3** — GT vs LLM Short 렌더 recognition으로 gap 측정 (가장 중요한 새 실험)
3. **E4** — evaluate/correct asymmetry (C4를 새로 세우는 실험)
4. **E5 / E6** — Roomba development와 Humanoid precision을 깊게 분해

최소 논문 스토리라인은 E1–E4만으로도 가능하다.

```text
Capacity:
  Humanoid = high L1 burden, high L2 potential
  Roomba   = low L1 burden, low L2 ceiling

Capability:
  LLM approximates both
  but fails differently:
    Humanoid by imprecision
    Roomba by unrecognizable under-development
  and can critique more than it can repair
```

## 6. 피해야 할 프레이밍

- “Roomba를 못한다”고 단정하지 말 것.  
  더 정확히는 **L1은 하고 L2로 못 간다**.
- “Humanoid를 잘한다”고 단정하지 말 것.  
  더 정확히는 **L2 잠재력은 있으나 L1 exactness가 약하다**.
- Evaluation 강함을 generation 강함과 동일시하지 말 것.  
  C4가 이 혼동을 끊는 장치다.
- Approximation metric만으로 embodiment 순위를 매기지 말 것.  
  Roomba가 L1에서 이겨도 L2에서 질 수 있다.

## 7. 즉시 쓸 수 있는 claim 문장

논문/토의용 초안:

1. **Shared approximation:** LLMs can fill the core semantic skeleton of a gesture across embodiments.
2. **Asymmetric ceilings:** High-resolution embodiments lose exactness; low-resolution embodiments lose cue identification.
3. **Gap claim:** Attribute-level approximation is not sufficient for cue identification, especially on low-resolution platforms.
4. **Critique–repair gap:** Models diagnose gesture failures more reliably than they repair them.

## 8. 다음 액션 체크리스트

- [ ] E1: H4 표를 recall / precision / EM으로 Humanoid vs Roomba 대비 재정렬
- [ ] E2: Human GT / LLM Short 렌더 recognition pilot (소수 cue로)
- [ ] E3: A-score × I-score contingency 초안
- [ ] E4: evaluate → localize → correct → re-evaluate 프로토콜 작성
- [ ] E5/E6는 pilot 결과에 따라 우선순위 결정
