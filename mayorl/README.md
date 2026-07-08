# mayorl — 금전 제약이 있는 도시 운영 RL 환경

원본 ralph-city (Micropolis/SimCity RL)의 환경을 확장하여,
**에이전트가 제한된 예산 안에서 도시를 운영**하도록 하는 프로젝트.

원본에서는 매 스텝마다 자금이 리셋(무한 자금)되지만, mayorl에서는
건설 비용이 실제로 차감되고 세수만이 수입원이다.

## 코드 구조

```
mayorl/
├── config.py        # 모든 하이퍼파라미터의 중앙 관리 (MayorlConfig)
├── env.py           # BudgetCityEnv — 자금 리셋 제거, action masking
├── reward.py        # RewardCalculator — 6개 컴포넌트 합산 리워드
├── curriculum.py    # BudgetCurriculum — 자금을 점진적으로 줄이는 커리큘럼
├── wrapper.py       # ActionMaskWrapper, BudgetMonitor, make_mayorl_env
├── utils.py         # EpisodeTracker, 로깅 포맷터, seed 설정
├── train.py         # 학습 진입점 (python -m mayorl.train)
├── evaluate.py      # 평가 스크립트 (python -m mayorl.evaluate)
└── __init__.py      # 패키지 API (env/wrapper는 lazy import)
```

### 의존성 흐름

```
config.py  ← (독립, 의존성 없음)
    ↑
reward.py  ← config
curriculum.py ← config
    ↑
env.py     ← config, reward, gym_city.envs.env.MicropolisEnv
    ↑
wrapper.py ← config, curriculum, env
    ↑
train.py   ← 전체 + 원본 model.py, algo/, storage.py
evaluate.py ← 전체 + 원본 model.py
```

> `env.py`, `wrapper.py`, `train.py`, `evaluate.py`는 Micropolis 엔진(GTK/gi)이
> 필요하므로 Docker 컨테이너에서 실행해야 한다.
> `config.py`, `reward.py`, `curriculum.py`, `utils.py`는 로컬에서도 사용 가능.

---

## 주요 하이퍼파라미터

모든 설정은 `config.py`의 `MayorlConfig` dataclass에 집중되어 있다.
아래는 사용자가 **직접 조절해야 할 가능성이 높은** 파라미터들이다.

### 커리큘럼 관련

| 파라미터 | 기본값 | 위치 | 설명 |
|---------|--------|------|------|
| `pop_target` | `600.0` | `MayorlConfig` | **커리큘럼 승격 기준이 되는 목표 인구.** 각 Phase의 `pop_threshold_ratio`와 곱해져서 실제 승격 기준이 된다. 예: Phase 0의 threshold=0.6이면, `600 * 0.6 = 360` 이상의 평균 인구가 필요. 맵 크기에 따라 달라져야 한다 — 16x16 맵에서 600은 높은 편이고, 20x20이면 적절하다. |
| `curriculum_window` | `100` | `MayorlConfig` | 승격 판단에 사용하는 최근 에피소드 수. 너무 작으면 불안정하게 승격, 너무 크면 승격이 늦다. |
| `curriculum_phases` | 4단계 | `MayorlConfig` | Phase별 초기 자금과 승격 조건 리스트. 아래 [커리큘럼 Phase](#커리큘럼-phase-상세) 참조. |

#### `pop_target`을 어떻게 설정하나?

`pop_target`은 "이 맵에서 잘 하면 도달할 수 있는 인구"의 대략적인 추정치다.
커리큘럼의 각 Phase는 이 값의 **일정 비율**을 달성해야 다음 단계로 넘어간다.

- 맵이 작으면(12x12) → `pop_target`을 낮춰야 (예: 200~300)
- 맵이 크면(20x20) → 기본값 600 또는 그 이상
- 확신이 없으면 Phase 0에서 자금 무제한으로 몇 번 돌려보고, 달성 가능한 인구를 측정

### 리워드 가중치

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `reward_weights.population` | `1.0` | 인구 증가분(res+com+ind delta)에 곱해지는 가중치. 가장 핵심적인 보상 신호. |
| `reward_weights.budget_health` | `0.3` | 자금 변화율 보상. 높이면 에이전트가 보수적(건설 적게)으로 행동. |
| `reward_weights.mayor_rating` | `0.2` | 시장 지지율 변화 보상. 범죄/오염 관리를 유도. |
| `reward_weights.invalid_action_penalty` | `-0.1` | 자금 부족한 건설 시도 시 페널티. 너무 크면 탐색 억제. |
| `reward_weights.bankruptcy_penalty` | `-10.0` | 파산 시 큰 음수 보상. 에피소드도 종료됨. |
| `reward_weights.infrastructure_shaping` | `0.1` | 전력망/도로 커버리지 기반 potential shaping. 초반 인프라 투자 유도. |

#### 튜닝 가이드

- 에이전트가 **건설을 안 하고 돈만 모으면** → `population` ↑, `budget_health` ↓
- 에이전트가 **무작위로 비싼 건물만 지으면** → `invalid_action_penalty`를 약간 더 키우거나, `infrastructure_shaping` ↑
- 에이전트가 **도로/전력 없이 구역만 지으면** → `infrastructure_shaping` ↑
- 에이전트가 **너무 빨리 파산하면** → Phase 0에서 더 오래 학습시키거나, `bankruptcy_penalty` ↑

### 환경 설정

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `map_x`, `map_y` | `20` | 맵 크기. 작을수록 학습 빠름. 추천: 실험용 12~16, 본격 학습 16~20. |
| `max_steps` | `None` (= map_x * map_y) | 에피소드 최대 스텝. `None`이면 맵 면적만큼. |
| `init_budget` | `2,000,000` | 초기 자금. 커리큘럼 사용 시 Phase 0의 값으로 덮어씌워짐. |
| `min_funds` | `0` | 이 이하로 떨어지면 파산. |
| `empty_start` | `True` | 빈 맵에서 시작할지 여부. |
| `random_builds` | `False` | 랜덤 초기 건물 배치 여부. |

### 학습 파라미터 (train.py CLI)

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--algo` | `ppo` | 알고리즘. `ppo` 또는 `a2c`. |
| `--lr` | `7e-4` | 학습률. |
| `--num-processes` | `4` | 병렬 환경 수. GPU 메모리와 CPU에 따라 조절. |
| `--num-steps` | `5` | 업데이트 당 rollout 길이. |
| `--num-frames` | `5,000,000` | 총 학습 프레임. |
| `--model` | `FullyConv` | 모델 아키텍처. `FullyConv`, `FractalNet` 등. |
| `--no-curriculum` | `False` | 커리큘럼 비활성화 (고정 자금). |
| `--init-budget` | (Phase 0) | 커리큘럼 무시하고 고정 자금 설정. |

---

## 커리큘럼 Phase 상세

에이전트가 처음부터 적은 돈으로 시작하면 의미있는 탐색을 못 한다.
따라서 **넉넉한 자금 → 점진적 축소** 전략을 사용한다.

| Phase | 이름 | 초기 자금 | 승격 조건 | 의미 |
|-------|------|-----------|-----------|------|
| 0 | `free_build` | 2,000,000 | avg_pop >= `pop_target` * 0.6 | 돈 걱정 없이 도시 건설 패턴 학습 |
| 1 | `moderate` | 500,000 | avg_pop >= `pop_target` * 0.5 AND 파산율 < 20% | 비용 인식 시작 |
| 2 | `tight` | 100,000 | avg_pop >= `pop_target` * 0.4 AND 파산율 < 10% | 효율적 건설 순서 학습 |
| 3 | `realistic` | 20,000 | (최종 단계) | 최소 자금으로 최적 전략 |

### 승격 메커니즘

```
rolling window (최근 100 에피소드)
    ↓
avg_pop >= pop_target * threshold  AND  bankruptcy_rate < max_rate
    ↓  (두 조건 모두 충족)
다음 Phase로 승격, window 초기화, 모델 가중치 유지
```

- Phase 0에서는 파산율 조건 없음 (자금이 넉넉하므로 파산 자체가 거의 불가)
- Phase 간 **인구 기준을 점진적으로 낮춤** (0.6 → 0.5 → 0.4): 자금이 줄수록 같은 인구를 달성하기 어렵기 때문
- 승격 시 모델 가중치는 유지됨 (이전 Phase에서 배운 건설 패턴 재활용)

### 커리큘럼 커스터마이징

```python
from mayorl.config import MayorlConfig, CurriculumPhase

config = MayorlConfig(
    pop_target=300,  # 작은 맵용
    curriculum_window=50,  # 빠른 승격 판단
    curriculum_phases=[
        CurriculumPhase("easy", 1_000_000, 0.5, None),
        CurriculumPhase("hard", 50_000, 0.3, 0.15),
        CurriculumPhase("final", 10_000, 0.0, None),
    ],
)
```

---

## 리워드 구조

총 리워드 = 6개 컴포넌트의 가중합:

```
R(t) = w_pop   * (인구 변화량)
     + w_budget * clamp(자금 변화율, -1, 1)
     + w_rating * (시장 지지율 변화량)
     + w_infra  * (Phi(s') - Phi(s))       ← potential-based shaping
     + penalty_invalid                      ← 자금 부족 건설 시도
     + penalty_bankrupt                     ← 파산 시
```

- **인구 변화량**: `(res+com+ind)_now - (res+com+ind)_prev`. 핵심 보상.
- **자금 변화율**: `(funds_now - funds_prev) / init_budget`. 세수 > 지출이면 양수.
- **시장 지지율**: `rating_now - rating_prev`. 범죄/오염 관리 유도.
- **인프라 shaping**: `Phi(s) = power_coverage + road_coverage`. 인프라 투자에 대한 중간 보상.
- **무효 액션 페널티**: 못 짓는 건물을 시도하면 -0.1.
- **파산 페널티**: 자금 0 이하 도달 시 -10.0 + 에피소드 종료.

---

## 건설 비용표

| Tool | 비용 | 용도 |
|------|------|------|
| Residential | 100 | 주거 구역 (3x3) |
| Commercial | 100 | 상업 구역 (3x3) |
| Industrial | 100 | 산업 구역 (3x3) |
| Road | 10 | 도로 (교통, 구역 연결) |
| Wire | 5 | 전력선 (발전소↔구역 연결) |
| Rail | 20 | 철도 |
| Park | 10 | 공원 (토지 가치 상승) |
| Clear | 1 | 철거 |
| PoliceDept | 500 | 경찰서 (범죄 감소) |
| FireDept | 500 | 소방서 (화재 대응) |
| Stadium | 3,000 | 경기장 |
| CoalPowerPlant | 3,000 | 석탄 발전소 |
| NuclearPowerPlant | 5,000 | 원자력 발전소 |
| Seaport | 5,000 | 항구 |
| Airport | 10,000 | 공항 |
| Net / Water / Land / Forest | 1 | 기타 지형 |
| Nil | 0 | 아무것도 안 함 |

자금이 적은 Phase에서는 Road(10) + Wire(5) + Residential(100)처럼
저렴한 조합으로 효율적인 도시를 만드는 전략이 필요하다.

---

## 실행 예시

```bash
# Docker 컨테이너 안에서
python -m mayorl.train --algo ppo --map-width 16 --num-processes 4

# 커리큘럼 없이 고정 자금으로 학습
python -m mayorl.train --no-curriculum --init-budget 100000

# 학습된 모델 평가
python -m mayorl.evaluate --load-dir trained_models/mayorl/ppo_FullyConv_w16_...

# 특정 Phase의 자금으로 평가
python -m mayorl.evaluate --load-dir ... --phase 2
```

---

## Observation 구조

원본 관측(맵 피처 + 밀도맵 + 스칼라)에 **budget 채널 2개가 추가**된다:

| 채널 | 값 범위 | 설명 |
|------|---------|------|
| `normalized_funds` | [-1, 1] | `current_funds / (init_budget * 2)`. 맵 전체에 broadcast. |
| `affordable_ratio` | [-1, 1] | 현재 자금으로 건설 가능한 tool의 비율. -1이면 아무것도 못 짓고, 1이면 전부 가능. |

이를 통해 에이전트는 현재 재정 상태를 시각적(공간적) 관측의 일부로 인식한다.
