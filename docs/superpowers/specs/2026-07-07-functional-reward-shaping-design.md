# 설계: 기능적 구조 보상 shaping (raw coverage 교체)

- 날짜: 2026-07-07
- 상태: 설계 승인됨 (구현 전)
- 관련: CLAUDE.md 학습 이슈, `mayorl/reward.py`, `mayorl/env.py`, `mayorl/config.py`

## 1. 배경 / 문제

세금 제약 학습에서 에이전트가 **인구가 자라는 도시를 전혀 못 짓는다**(pop≈0, entropy≈max=거의 랜덤). 근본 원인은 보상의 오도(misleading proxy)다:

- 현재 인프라 shaping의 잠재함수 `Φ(s) = power_coverage + road_coverage`는 **연결 여부와 무관하게** 발전소·도로 타일을 아무 데나 깔기만 해도 오른다.
- 그래서 에이전트는 인프라를 흩뿌려 쉬운 shaping 보상(≈0.2)을 챙기고, **zone과 전력·도로를 연결하는 법은 배우지 못한다.**
- 진짜 인구 보상은 발전소–전력–도로–zone이 **연결돼야** 나오는데(별도 진단 `_diag_power.py`에서 확인: 인접 보장 도시는 pop 0→32 성장), 랜덤 탐색으로 그 조합을 맞출 확률은 거의 0.

즉 보상이 "기능하는 도시" 대신 "흩뿌린 인프라"를 보상한다.

## 2. 목표

인구가 자라기 **전에도**, 발전소–전력–도로가 실제로 연결된 **기능적 zone**에 밀도 있는 gradient를 주어 탐색을 부트스트랩한다. 흩뿌리기로는 얻을 수 없어야 한다(오도 제거).

비목표: pop/budget/mayor_rating/penalty 항은 건드리지 않는다. 커리큘럼·prebuild·VLM은 이 스펙 범위 밖.

## 3. 설계 개요

인프라 shaping의 잠재함수 Φ **하나만** 교체한다. 형태는 potential-based 유지(정책 최적점 불변, 학습만 가속, 진동 해킹 방지).

```
기존:  Φ(s) = power_coverage + road_coverage           (raw coverage — 제거)
신규:  Φ_func(s) = functional_zone_tiles(s) / N_norm    (전력+도로 충족 zone 비율)

reward 항:  w_func · ( Φ_func(s') − Φ_func(s) )
```

`functional_zone_tiles` = 다음을 **모두** 만족하는 RCI zone 타일 수:
1. zone 타일이다 (`zoneFromInt(tile) ∈ {Residential, Commercial, Industrial}`)
2. **전력 공급됨** (`engine.getPowerGrid(...) > 0`)
3. **도로 접근 있음** (Chebyshev 거리 ≤ `road_access_radius` 내에 Road/RoadWire 타일 존재)

전체 리워드(VLM off), 변경 후:
```
R = 1.0 · Δpop
  + 0.3 · budget_health
  + 0.2 · Δmayor_rating           (= 엔진 cityYes 변화; 그대로)
  + w_func · (Φ_func(s') − Φ_func(s))   ★ raw coverage 제거, 기능적 zone으로 교체
  + (-0.1 if invalid) + (-10 if 파산)
```

동작 원리: 발전소 옆 전선망에 연결되고 도로에 접한 zone이 하나 생기는 **순간** Φ_func이 오른다 — 인구 발전(수 게임년) 이전에 이미 gradient 발생. 인구가 자라면 pop 보상(1.0)이 자연히 지배하고 shaping은 부트스트랩 역할만 한다(potential-based의 표준 동작).

## 4. 상세: 기능적 zone 탐지 (`env.py`)

신규 메서드 `_calc_functional_zone_fraction() -> float` 를 `BudgetCityEnv`에 추가한다. **맵 1회 스캔**으로 배열 3개를 만들고 벡터 연산으로 계산한다(스텝당 비용은 기존 `getDensityMaps` 스캔과 동급).

```
MX, MY = MAP_X, MAP_Y; xs, ys = MAP_XS, MAP_YS
for i in range(MX), j in range(MY):
    t = engine.getTile(i+xs, j+ys) & 1023         # getTile과 동일 좌표 규약
    cls[i,j]     = zoneFromInt(t)                  # 타일 분류
    power[i,j]   = engine.getPowerGrid(i+xs, j+ys) # 전력 (getTile과 동일 좌표 → 정합)
is_zone = cls ∈ {Residential, Commercial, Industrial}
is_road = cls ∈ {Road, RoadWire}
road_access[i,j] = is_road 가 (i,j) 중심 (2R+1)² 창 안에 하나라도 존재  # R = road_access_radius
functional = is_zone & (power > 0) & road_access
return functional.sum() / (MX * MY)               # Φ_func ∈ [0, 1]
```

- **좌표 정합성**: `getTile`과 `getPowerGrid`를 동일한 `(i+MAP_XS, j+MAP_YS)`로 질의해 타일↔전력 정합을 보장한다(`_diag_power.py`에서 검증한 규약; 원본 `getDensityMaps`의 i/j 전치 이슈를 피함).
- **도로 접근 근사**: 실제 Micropolis는 zone 발전에 도로 근접을 요구한다. 여기서는 반경 `road_access_radius`(기본 2) 내 도로 존재로 근사한다.
- 예외 시 0.0 반환(기존 `_calc_*_coverage`와 동일한 방어).

기존 `_calc_power_coverage` / `_calc_road_coverage`는 **유지**한다 — info dict(`power_coverage`, `road_coverage`)와 VLM 브리핑에서 계속 쓰이므로. 리워드 경로에서만 빠진다.

## 5. 리워드 변경 (`reward.py`, `config.py`)

### 5.1 `reward.py`
- `infrastructure_potential(power_coverage, road_coverage)` → **`functional_potential(functional_fraction) -> float`** 로 교체(단순 반환이지만 명시적 함수로 유지해 유닛테스트/의미 부여).
- `StepInfo`: `power_coverage/last_power_coverage/road_coverage/last_road_coverage` 4필드를 **`functional_fraction/last_functional_fraction` 2필드로 교체**(리워드는 더 이상 coverage를 안 씀).
- `RewardCalculator.compute` / `decompose`: 인프라 shaping 블록을 기능적 shaping으로 교체:
  ```
  reward += w_func * (functional_potential(info.functional_fraction)
                      - functional_potential(info.last_functional_fraction))
  ```
- `decompose`의 키 `"infra_shaping"` → `"functional_shaping"`.

### 5.2 `config.py` (`RewardWeights`)
- `infrastructure_shaping: float = 0.1` → **`functional_shaping: float = 2.0`** 로 교체(rename + 기본값↑).
  - 근거: Φ_func은 맵 대비 비율이라 값이 작다(예: 기능적 zone 9타일 / 256 ≈ 0.035). w_func=2.0이면 그런 클러스터 1개 생성 시 ΔR ≈ 0.07 — pop 보상과 같은 자릿수. **정확한 값은 짧은 학습으로 튜닝**(§9).

## 6. env 변경 (`_postact_budget`)

```
# 기존
power_coverage = self._calc_power_coverage()
road_coverage  = self._calc_road_coverage()
... StepInfo(power_coverage=..., last_power_coverage=self._last_power_coverage, road_coverage=..., ...)
self._last_power_coverage = power_coverage; self._last_road_coverage = road_coverage

# 변경
functional = self._calc_functional_zone_fraction()
... StepInfo(functional_fraction=functional, last_functional_fraction=self._last_functional)
self._last_functional = functional
```

- 상태 변수 `self._last_functional`(초기 0.0)을 `_reset_episode_stats`에 추가.
- info dict에는 관측용으로 `functional_fraction`을 **추가**한다(기존 `power_coverage/road_coverage`도 유지 — info/VLM용).

## 7. 관측 / 로깅 (observability)

- info dict에 `functional_fraction` 추가.
- 학습 로그(`train.py` `_log_progress` 또는 `EpisodeTracker`)에 평균 `functional_fraction`을 컬럼으로 노출 → **pop이 오르기 전에 functional이 먼저 오르는지** 관찰 가능(설계 의도의 직접 검증 지표).

## 8. 테스트 계획

1. **유닛테스트 (`reward.py`, stateless)**:
   - `functional_potential(x)` 반환값 검증.
   - `RewardCalculator.compute`가 기능적 shaping 항을 정확히 포함하고, **raw coverage에는 더 이상 반응하지 않음**을 검증.
2. **env 탐지 테스트** (엔진 필요, Docker):
   - `_diag_power.py`의 클린 클러스터(전력+도로+zone 인접) → `_calc_functional_zone_fraction() > 0`.
   - `layGrid`의 흩뿌린 인프라(전력·도로 커버리지는 높지만 zone 무연결) → **functional ≈ 0**. "흩뿌리기로는 못 얻는다"는 핵심 속성 직접 검증.
3. **통합(짧은 학습)**: 커리큘럼 free_build로 짧게 학습 → `functional_fraction`이 0 위로 오르고 그 뒤 `pop`이 따라 오르는지 관찰.

## 9. 튜닝 파라미터 (config, 경험적 조정)

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `RewardWeights.functional_shaping` (w_func) | 2.0 | 기능적 shaping 가중치 |
| `road_access_radius` (R) | 2 | zone의 도로 접근 판정 반경 (config에 추가) |
| `N_norm` | `MAP_X*MAP_Y` | Φ_func 정규화 분모 (맵 면적 고정) |

## 10. 범위 밖 (Deferred)

- **사용자의 2단계 게이팅**("기능적 클러스터가 생긴 뒤 raw coverage도 보상"): 보류. 기능적 shaping이 이미 "더 많은 zone 연결"을 위해 전력·도로 확장을 내포 유도하므로 raw coverage 추가는 중복(YAGNI). 필요 시 나중에 config 플래그로 추가.
- **prebuild 격리 테스트**(옵션 2): 별도 실험. 코드 변경 없이 기존 `--prebuild` 경로로 검증 예정.
- pop/budget/mayor_rating/penalty 항, 커리큘럼, VLM: 불변.

## 11. 리스크 / 엣지케이스

- **파괴 시 음보상**: potential-based라 기능적 zone이 Rubble로 파괴되면 Φ 하락→음보상. `auto_bulldoze=False`(mayorl 기본)라 드물다. 정상 동작.
- **도로 접근 근사 오차**: 반경 근사가 엔진의 실제 traffic 접근성과 미세하게 다를 수 있음 — 인구가 실제로 자라는지(통합 테스트)로 교차검증.
- **가중치 스케일**: Φ_func이 작아 w_func 튜닝 민감. §9로 조정, functional_fraction 로깅으로 관찰.
- **성능**: 스텝당 맵 1회 스캔(256타일 getTile+getPowerGrid). 기존 coverage 계산과 동급, 허용.
