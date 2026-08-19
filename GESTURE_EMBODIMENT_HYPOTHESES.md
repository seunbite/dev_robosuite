# Gesture × Embodiment 연구 요인 및 가설

## H4 binary attribute MCQ 결과

Protocol: blank-config + `glossary_value` (gesture meaning + motion-code glossary).
Prompts are fully English. Each of 519 questions is evaluated twice (original A/B order and
exact reversal). Accuracy is the combined accuracy over both orders.

### 모델별 정확도

| Model | N pairs | Accuracy |
|---|---:|---:|
| Gemini | 519 | — |
| Qwen2.5-VL-32B | 519 | **69.7%** |
| Qwen2.5-VL-7B | 519 | 60.3% |
| Qwen2.5-VL-3B | 519 | 61.8% |

### Embodiment별 정확도

| Embodiment | N pairs | Qwen 32B | Qwen 7B | Qwen 3B |
|---|---:|---:|---:|---:|
| Humanoid | 159 | 68.9% | 62.9% | 59.7% |
| Tiago | 156 | 67.0% | 62.2% | 58.7% |
| Single-arm | 137 | 69.0% | 59.1% | 62.8% |
| Quadruped | 29 | 82.8% | 50.0% | 74.1% |
| Roomba | 38 | 76.3% | 53.9% | 71.1% |

### Attribute 타입별 정확도

| Attribute | N pairs | Qwen 32B | Qwen 7B | Qwen 3B |
|---|---:|---:|---:|---:|
| Repetition | 91 | 90.1% | 55.5% | 68.1% |
| Path x sign | 9 | 77.8% | 50.0% | 72.2% |
| Direction | 77 | 77.3% | 77.9% | 81.2% |
| Left arm (`mirror/still`) | 23 | 60.9% | 71.7% | 54.3% |
| Path degree sign | 19 | 76.3% | 44.7% | 68.4% |
| Path distance sign | 21 | 76.2% | 47.6% | 61.9% |
| Gripper orientation | 85 | 50.6% | 49.4% | 47.1% |
| Path axis | 58 | 43.1% | 53.4% | 74.1% |
| Pose x | 28 | 73.2% | 78.6% | 48.2% |
| Pose z | 79 | 75.9% | 67.7% | 45.6% |
| Pose y | 29 | 69.0% | 50.0% | 65.5% |

## 현재 검증 결과

> 아래 결과는 현재 데이터에 대한 기술통계적 지지이다. 사전등록된 판정 기준과 독립 표본을
> 사용한 confirmatory test가 완료된 것은 아니다.

| ID | 가설 | 현재 결과 | 판정 | 남은 조건 |
|---|---|---|---|---|
| H1 | Embodiment의 의미 해상도가 높을수록 cue 표현에 필요한 필수 attribute가 많다. | Human GT의 명시적 semantic argument 수가 Humanoid에서 Roomba 방향으로 감소했다. | 기술통계 지지 | Platform-independent semantic atom으로 required/optional을 이중 코딩하고 mixed model로 검정 |
| H4 | Short/core motion은 비인간형 embodiment에서도 LLM이 비교적 잘 생성한다. | 전체 150 cells: required recall 86.4%, F1 87.4%, instance exact match 58.0%. Fully possible + approximate subset: recall 87.0%, F1 88.5%, exact match 59.7%. | 기술통계 지지 | 성공 기준 사전등록, 동일 model·prompt·sampling 조건의 반복 실험 |

## 연구 요인

### Axis A. Embodiment semantic resolution

Cue의 의미를 로봇의 형태와 자유도로 얼마나 직접적이고 세밀하게 표현할 수 있는지를 뜻한다.

의미 해상도의 사전등록 순서는 다음과 같다.

1. Humanoid
2. Tiago
3. Single-arm
4. Quadruped
5. Roomba

조작점검에는 다음 항목을 사용한다.

- 사용 가능한 표현 채널: 손, 팔, 머리, 몸통, 이동, 회전 등
- Cue를 원형에 가깝게 표현할 수 있는 정도
- Fully possible / approximate / not feasible 비율
- Human GT의 필수 semantic attribute 실현 가능성

### Axis B. Expression length and contextual scaffolding

하나의 cue를 핵심 동작만으로 간결하게 표현할지, 전후에 의미를 보조하는 submotion을 추가할지를
뜻한다.

| Level | 정의 |
|---|---|
| Short | Original core motion만 사용 |
| Long1 | 자연스러운 최소 contextual motion을 추가한 버전 |
| Long2 | 동일한 original에서 시작하여 submotion 1–2개 추가 |
| Long3 | 동일한 original에서 시작하여 submotion 3–4개 추가 |
| Long4 | 동일한 original에서 시작하여 submotion 5–6개 추가 |

모든 Long level은 직전 level을 확장하지 않고 같은 Short/original에서 독립적으로 생성한다.
각 데이터에는 추가된 submotion의 설명, 전체 config, core component의 1-based inclusive range를
기록한다.

## Gesture 및 embodiment 가설

### H1. Semantic attribute burden

Embodiment의 의미 해상도가 높을수록 cue를 정확히 표현하기 위해 명시되어야 하는 필수
attribute가 많다.

- 주요 attribute: body part, direction, shape, repetition, spatial relation, temporal profile
- 독립변수: embodiment semantic resolution
- 종속변수: cue × embodiment별 required attribute 수
- 권장 분석: count mixed model with cue random intercept

#### H1 결과: Human GT의 distinct semantic control slots

| Embodiment | 전체 N | 전체 평균 | Fully possible N | Fully possible 평균 | Possible + approximate N | Possible + approximate 평균 |
|---|---:|---:|---:|---:|---:|---:|
| Humanoid | 30 | 9.07 | 30 | 9.07 | 30 | 9.07 |
| Tiago | 30 | 8.77 | 27 | 8.59 | 30 | 8.77 |
| Single-arm | 30 | 7.77 | 12 | 7.25 | 30 | 7.77 |
| Quadruped | 30 | 5.43 | 10 | 6.00 | 27 | 5.56 |
| Roomba | 30 | 5.03 | 4 | 5.75 | 18 | 4.94 |

측정값은 각 Human GT cell에서 `pose`, `move`, `path`, `hold`에 명시된 서로 다른 semantic
control slot의 수이다. 동일 slot이 한 cell에서 반복되어도 한 번만 센다.

### H2. Embodiment recognizability

Embodiment의 의미 해상도가 높을수록 사람과 VLM의 cue recognizability가 높다.

- 입력: 동일 조건으로 렌더된 motion video
- 종속변수: cue recognition의 binary accuracy
- 권장 분석: `recognized ~ embodiment + (1|cue) + (1|rater)`

### H3. Inverted-U effect of expression length

Expression length가 증가할수록 recognizability는 처음에는 향상되지만, 지나치게 길어지면 의미가
희석되어 감소한다.

- Short 단계에서는 필요한 context가 부족할 수 있다.
- 중간 Long 단계에서는 setup, core, settle 등의 조합이 cue 해석을 돕는다.
- 과도한 Long 단계에서는 불필요한 동작이 cue를 모호하게 만들 수 있다.
- 권장 분석: length의 linear term과 quadratic term을 포함한 mixed model

### H3a. Embodiment × length interaction

의미 해상도가 낮은 embodiment일수록 contextual scaffolding의 이득이 크고 recognizability가
최대가 되는 최적 길이도 더 길다.

- Humanoid처럼 의미 해상도가 높은 embodiment는 짧은 core motion만으로도 충분할 가능성이 크다.
- Roomba와 Quadruped처럼 의미 해상도가 낮은 embodiment는 전후 context의 도움을 더 많이 받을
  가능성이 크다.
- 단, 지나치게 긴 context는 모든 embodiment에서 recognizability를 낮출 수 있다.

## LLM 및 AI 가설

### H4. Short/core motion generation

LLM은 embodiment가 인간과 유사하지 않더라도 핵심적인 Short motion code를 비교적 잘 생성한다.

- 비교 단위: Human GT와 LLM-generated Short config
- 주요 지표: script similarity, required-attribute recall, precision, F1, instance exact match
- Embodiment별 성능 차이와 feasibility subset별 성능을 함께 보고한다.

#### H4 결과: Human GT vs Top-5 Short

| Embodiment | 전체 N | Required recall | Attribute F1 | Instance EM | Possible + approximate N | Subset recall | Subset F1 | Subset EM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Humanoid | 30 | 80.1% | 83.0% | 36.7% | 30 | 80.1% | 83.0% | 36.7% |
| Tiago | 30 | 82.0% | 84.6% | 46.7% | 30 | 82.0% | 84.6% | 46.7% |
| Single-arm | 30 | 85.8% | 87.0% | 56.7% | 30 | 85.8% | 87.0% | 56.7% |
| Quadruped | 30 | 100.0% | 98.6% | 96.7% | 27 | 100.0% | 100.0% | 100.0% |
| Roomba | 30 | 83.9% | 83.5% | 53.3% | 17 | 89.5% | 89.4% | 64.7% |
| **Overall** | **150** | **86.4%** | **87.4%** | **58.0%** | **134** | **87.0%** | **88.5%** | **59.7%** |

#### H4 결과: Fully possible + approximate subset만

| Embodiment | N | Required recall | Attribute F1 | Instance EM |
|---|---:|---:|---:|---:|
| Humanoid | 30 | 80.1% | 83.0% | 36.7% |
| Tiago | 30 | 82.0% | 84.6% | 46.7% |
| Single-arm | 30 | 85.8% | 87.0% | 56.7% |
| Quadruped | 27 | 100.0% | 100.0% | 100.0% |
| Roomba | 17 | 89.5% | 89.4% | 64.7% |
| **Overall** | **134** | **87.0%** | **88.5%** | **59.7%** |

#### H4 보조 검증: GT attribute 이지선다

Human GT에서 519개의 deterministic binary-choice 문항을 생성한다.

| Mutation target | 이지선다 구성 |
|---|---|
| Repetition | `1` ↔ `plural` |
| Direction | 정답 ↔ 나머지 5방향 중 하나 |
| Gripper orientation | `horizontal` ↔ `vertical` |
| Arm relation | `left_arm=mirror` ↔ `still` |
| Pose x/y/z | GT 값 ↔ 범위 `[0,100]` 안의 `GT ± 30` |
| Path axis | 정답 축 ↔ 다른 Cartesian 축 |
| Line/path sign | GT displacement 또는 angle ↔ 부호 반전 |


### H5. Asymmetric knowledge in long-motion generation

LLM의 long-motion 생성 능력은 component 종류에 따라 비대칭적이다. 즉, 잘 생성하는 contextual
component와 반복적으로 실패하는 component가 구분될 것이다.

- component 위치: PRE / CORE / POST
- component 역할: setup, emphasis, transition, settle, recovery
- 평가: component precision·recall, 누락·과잉·순서 오류, cue와의 causal relevance

### H6. Human–VLM evaluation agreement

동일한 motion video에 대한 Human과 VLM의 cue 평가가 유의하게 일치한다.

- 동일 video와 동일 질문을 사용한 paired evaluation
- 주요 지표: raw agreement, Cohen's kappa 또는 Gwet's AC1
- Cue와 embodiment에 따른 agreement 차이도 분석한다.

## 평가 체계

### Automatic evaluation

1. **Script–text:** 생성된 script가 Human GT script와 얼마나 유사한지 측정한다.
2. **Attribute–text:** Human GT의 필수 attribute 중 일치, 누락, 과잉 생성된 항목을 측정한다.

### Human / VLM video evaluation

1. **Render–video:** 영상이 목표 cue를 드러내는지 binary recognition으로 평가한다.
2. **Attribute / fix-level–video:** 목표 cue에 가까워지기 위해 필요한 수정 내용과 수정량을
   평가한다.

## 핵심 분석 구조

- Main effect A: embodiment semantic resolution
- Main effect B: expression length
- Interaction: embodiment semantic resolution × expression length
- Random effects: cue, human rater 또는 VLM repeat
- 주요 outcome: recognizability, attribute recall/F1, exact match, fix magnitude,
  Human–VLM agreement
