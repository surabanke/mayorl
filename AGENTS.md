# 금전(세금) 제약 도시 운영 RL — `mayorl`

## 프로젝트 방향 (2026-07 변경됨)

**이전 원칙("원본 불변")은 폐기되었다.**

과거에는 "원본은 그대로 유지한 채 `mayorl/` 안에서만 학습환경·MDP를 재구성한다"가
지침이었다. 그러나 이 제약 때문에 `mayorl/problem.md`에 정리된 여러 문제
(action-space 하드코딩, 사후(post-hoc) action masking, 죽은 invalid 페널티, 관측
정규화 불일치 등)를 우회하기 어려웠고, 실제로는 원본(`corecontrol.py` 등)이 이미
수정되어 원칙이 깨진 상태였다.

이제 방향은 다음과 같다:

- **`gym-city-origin/`** = 박제된 진짜 원본 (읽기 전용 기준점). 절대 수정하지 않는다.
  최종적으로 `mayorl`을 이 원본과 비교하는 기준으로 쓴다.
- **`ralph-city/`** (이 저장소) = **변경 가능한 작업본**. 자유롭게 수정한다.
- **목표**: `mayorl`이 필요로 하는 기능을 모두 `mayorl/` 안으로 이동시켜
  **self-contained**하게 만든다. 그 후 `ralph-city`의 나머지(원본에서 상속만
  하던 부분)를 정리·삭제하고, 최종적으로 `mayorl` ↔ `gym-city-origin`을 비교한다.
- **VLM 부분**(`mayorl/vlm_mayor.py` 및 관련 경로)은 **현재 사용하지 않지만 살려둔다.**
  삭제·리팩터 대상이 아니다. 나머지 코드는 변경 가능하다.

원본과의 diff는 이제 "위반"이 아니라 정상적인 작업 산출물이다. 다만 변경 시
`gym-city-origin/`과 대조해 무엇을 왜 바꿨는지는 추적 가능하게 남긴다.

---

## 현재 우선순위 (먼저 할 것)

**VLM 시장을 추가하기 전에, 세금(예산) 제약이 들어갔을 때 에이전트가 제대로
학습되는지를 먼저 검증한다.**

### ✅ #1 블로커 — 해결됨 (2026-07-07): 원인 = `layGrid` 빌더 결함 (엔진/콜백 정상)

**경제 루프는 정상이었다.** "인구 0" 증상은 엔진/mayorl 문제가 아니라, 검증
스크립트가 도시 생성에 쓰던 `layGrid`(corecontrol.py) 헬퍼가 **Rubble·틈투성이
도시를 만들어 zone이 전력·도로에 연결된 적이 없었기 때문**이다. `layGrid`는 RL
학습엔 안 쓰이고 `test.py` 스크래치에서만 쓰이던 **검증 안 된 방치 헬퍼** — 그래서
실제 학습 경로엔 무영향, 세금 검증 스크립트만 이 함정에 빠져 있었다.

증거 (`mayorl/_diag_power.py`, Docker 실측 2026-07-07):
- **A(layGrid 재현)**: zone 0개(전부 Rubble), 인구 0. layGrid 산출물이 깨짐.
- **C(손으로 짠 클린 도시: 도로/주거/전선/발전소 인접 보장, Rubble 0)**: zone 급전
  **100%(36/36)**, 인구 **0→32 성장**, demands `[+1,-1,+1]`→`[+1,+1,+1]`(주민 발생→
  상업수요 생성). **엔진·전력전파·콜백·census 모두 정상 작동 확정.**

즉 `problem.md`의 항목들(작동하는 경제의 튜닝 문제)로 정상 복귀. 세금·학습 검증에는
`layGrid` 대신 **클린 빌더(C 방식)** 를 쓴다 — `verify_tax.py`에 이식 완료
(`build_powered_city`). 상세 근거: 메모리 `econ-loop-rootcause`.

**실행 환경 주의(중요)**: 레포 전체를 `-v repo:/usr/src/app`로 마운트하면 이미지에
컴파일된 엔진(.so)이 덮여 사라져 ImportError로 죽는다(로그엔 CUDA 배너만 남고 멈춘
것처럼 보임 — 과거 "배너 뒤 무응답" 증상의 정체). → `mayorl/`와 수정 파일
(`corecontrol.py`/`tilemap.py`)만 **개별 마운트**할 것. Git Bash(MSYS)는 경로를
오변환하므로 **WSL2**에서 실행:
```bash
docker run -d --name mayorl-diag \
  -v "$(pwd)/mayorl:/usr/src/app/mayorl" \
  -v "$(pwd)/gym_city/envs/corecontrol.py:/usr/src/app/gym_city/envs/corecontrol.py" \
  -v "$(pwd)/gym_city/envs/tilemap.py:/usr/src/app/gym_city/envs/tilemap.py" \
  mayorl:latest tail -f /dev/null
docker exec mayorl-diag sh -c 'cd /usr/src/app && xvfb-run -a python3 -u -m mayorl.verify_tax'
```

### 검증 순서

1. ✅ **세수 재검증 완료 (2026-07-07 PASS)** — 클린 도시로 `verify_tax.py` 실행:
   세율 0%→누적세수 $0, 세율 20%→$9,378(인구 32, 자금 순증). 세율↑ 시 세수↑ 확인.
2. ⬅ **다음**: 짧은 학습을 돌려 보상/파산율/인구 곡선이 합리적으로 움직이는지 확인.
   (학습 시 아래 '알려진 학습 영향 이슈' — 특히 model.py num_actions=19 하드코딩,
   사후 action masking — 이 실제로 곡선에 영향을 주는지 관찰.)

VLM 시장(`--use-vlm`, 전략 목표·세율 권고)은 이 검증이 끝난 뒤 순위다.

---

## 시뮬레이션 콜백 정책 (세금 제약 실현에 필요)

Micropolis 엔진의 시뮬레이션 콜백은 원본(`gym-city-origin`)에서 대부분 주석 처리되어
있었다. 세금·재정 메커니즘이 의미를 가지려면 아래가 켜져 있어야 한다.
`ralph-city/gym_city/envs/corecontrol.py` `simTick()` 기준 현재 상태:

| 콜백 | 상태 | 역할 | 판단 |
|------|------|------|------|
| `tickEngine`, `simTick` | ✅ on | 코어 시뮬레이션 스텝 | 유지 |
| `cityEvaluation` | ✅ on | `cityYes`→`mayor_rating`, 자산가치 평가, **세금 assessment** | **필요** |
| `updateDate` | ✅ on | 도시 시간 진행 — **연간 세금 징수가 날짜에 게이팅됨** | **필요** |
| `changeCensus` | ✅ on | 인구 census 갱신 (RCI 가치평가가 의존) | **필요** |
| `updateHeads`, `simUpdate` | ❌ off | GUI 히스토그램/리드로우 전용 | headless라 불필요, 유지 |
| `doTimeStuff` | ❌ off | **재난**(화재·홍수·토네이도·멜트다운) + 시간 이벤트 | **끈 채로 유지 — 아래 참조** |

**`doTimeStuff`(재난)를 켜지 않는 이유**: 예산 결정과 무관한 고분산 음(-)보상을
주입해 "세금 제약 하에서의 학습" 신호를 오염시킨다. 세금 제약 학습이 안정화된
후에 커리큘럼의 한 단계로 의도적으로 켜는 것을 고려한다.

> **실측 상태 (2026-07-02, Docker `test_tax.py`)**: 자금 **차감(지출)은 확인됨**
> (발전소 건설 시 $3,000 정상 차감). 그러나 **세수(수입)는 아직 미확인** — 테스트
> 도시의 인구가 0으로 유지되어(주거지가 무전력·미연결) 걷을 세금이 없었음. 이는
> 세금 콜백이 고장났다는 증거가 아니라 `test_tax.py`가 "작동하는 도시"를 짓지
> 못한다는 뜻(발전소↔주거지 Wire 없음). → 전력·도로로 제대로 연결된 클러스터를
> 짓고 충분한 시뮬 시간 동안 돌려 인구 성장→세수를 재검증해야 한다. "알려진 학습
> 영향 이슈 4번(연간 세금 희소성)"과 직결되는 핵심 미해결 항목.
> 향후 mayorl self-contained화 시, 이 콜백 활성화 로직도 공유 `corecontrol.simTick`
> 대신 mayorl 쪽으로 옮기는 것을 고려한다.

---

## `autoBulldoze = False`

`corecontrol.py`의 엔진 기본값은 `autoBulldoze = True`(기존 건물/잔해/숲 위에 철거
없이 바로 건설)였다. **mayorl에서는 `False`로 둔다** — 재개발 시 철거 비용까지
고려하게 하여 경제 시스템의 전략적 깊이를 살린다.

- 구현: 공유 `corecontrol.py`의 전역 기본값(True)은 건드리지 않는다(원본 대비
  비교/비-mayorl 학습을 위해). 대신 **`MayorlConfig.auto_bulldoze`(기본 False)**로
  두고 `BudgetCityEnv`가 매 에피소드 엔진에 적용한다.
- **주의(학습 영향)**: `False`는 철거 경제(Clear→build 2단계)를 세금 신호와 결합시켜
  "세금 제약만의 학습 효과"를 분리 관찰하기 어렵게 만든다. 학습 곡선 해석 시
  이 결합을 감안한다. 검증이 어려우면 일시적으로 `True`로 두고 A/B 비교한다.

---

## 알려진 학습 영향 이슈 (다음 이터레이션 — 지금은 보고만)

이번 이터레이션 범위 밖이지만, 학습에 영향을 주는 것으로 파악된 항목들.
자세한 근거는 `mayorl/problem.md` 참조. 이제 `ralph-city`가 변경 가능하므로
아래는 직접 수정할 수 있다.

1. **[High] `model.py:37` `num_actions=19` 하드코딩** — `'Micropolis'` 환경이면 무조건
   19로 덮어써서, 정책이 20번째 툴 `Nil`(no-op)을 **절대 샘플링 못 함**. 액션 공간이
   off-by-one. → `model.py`를 직접 고쳐 env가 넘긴 `len(TOOLS)=20`을 쓰게 한다.
2. **[High] 사후 action masking + 죽은 invalid 페널티** — `ActionMaskWrapper`가 샘플링
   *후* 무효 액션을 Nil로 치환하므로, (a) 로짓 마스킹의 탐색효율 이득이 없고
   (b) `invalid_action_penalty`가 기본 설정에서 한 번도 발동 안 함(죽은 코드).
   → 진짜 로짓 마스킹 도입 또는 페널티 경로 활성화.
3. **[Med] 관측 정규화 불일치** — 원본 RCI 스칼라 채널은 정규화 안 됨(원값), mayorl이
   추가한 budget/goal 채널만 `[-1,1]`. `FullyConv`(dirac init)에서 스케일 혼재가
   안정성을 해칠 수 있음.
4. **[Med, 검증 필요] 연간 세금 징수 vs 에피소드 길이** — 세금은 도시 1년마다 징수.
   `max_steps ≈ map_x*map_y`(=400) 스텝이 도시 몇 년에 해당하는지에 따라, 에피소드
   내 세수 이벤트가 1~2회뿐이면 세수 신호가 극히 희소해지고 `budget_health`가 지출로만
   지배됨. **"세금 제약 하 학습"의 성패를 가를 수 있는 항목** — `test_tax.py`로 확인.
5. **[Low] 엔진 실제 비용 vs `config.TOOL_COSTS` 불일치 가능성** — affordability 체크/
   action mask는 `TOOL_COSTS`(문서 기반 하드코딩)를 쓰지만 실제 자금 차감은 엔진 내부
   비용. 둘이 다르면 마스크가 실제와 어긋남. 검증 권장.

---

## 실행 (Docker)

GTK/컴파일된 Micropolis 엔진 때문에 Windows 로컬에선 못 돌리고 Docker에서 실행한다.

- 이미지: `mayorl:latest` (`Dockerfile.mayorl`)
- 세금 검증: `python -m mayorl.test_tax`
- 학습: `python -m mayorl.train --algo ppo --map-width 16 --num-processes 1 [--use-vlm]`
- 필요 의존성/shim: `mayorl/_shim/baselines`(OpenAI baselines 대체),
  `opencv-python-headless`, `torchsummary`, `shimmy>=2.0` (`Dockerfile.mayorl`에 반영됨).
- 원본 `storage.py`의 `value_preds` 크기 버그는 `train.py`의
  `MayorlRolloutStorage` 서브클래스로 교정됨(GAE 정상화). 상세는 `mayorl/log.md` 참조.

---

## 핵심 변경점: 금전(예산) 제약

원본 `gym_city/envs/env.py`는 매 스텝 `setFunds(init_funds)`로 자금을 리셋한다(무한 건설).
mayorl은 이를 제거한다:

- 자금은 리셋되지 않고 건설 비용이 실제 차감된다.
- 세수가 시뮬레이션 틱마다(연간 징수 게이팅) 유입된다.
- 자금 부족 시 건설 불가 (invalid → no-op + 페널티).
- 파산(자금 ≤ `min_funds`) 시 에피소드 조기 종료.

에이전트는 "언제/어디에/무엇을" 뿐 아니라 "지금 지을 여유가 있는가"를 함께 판단한다.

---

## 구조

```
mayorl/
├── config.py       # MayorlConfig, TOOL_COSTS, 커리큘럼 phase, 리워드 가중치
├── env.py          # BudgetCityEnv — 자금 관리, obs 확장, action_masks
├── reward.py       # 인구+재정건전성+지지율+인프라 shaping+페널티
├── curriculum.py   # BudgetCurriculum — 자금 단계별 승격
├── wrapper.py      # ActionMaskWrapper, BudgetMonitor
├── train.py        # 학습 루프 (원본 Policy/PPO/RolloutStorage 재활용)
├── evaluate.py     # 평가·시각화
├── vlm_mayor.py    # (미사용, 보존) VLM 시장 — 전략 목표·세율 권고
└── test_tax.py     # 세금 동작 검증 스크립트
```

## 리워드 (구현됨, `reward.py`)

```
R = w_pop*Δpop + w_budget*budget_health + w_rating*Δrating
  + w_infra*(Φ(s')−Φ(s)) + invalid_penalty + bankruptcy_penalty
Φ(s) = power_coverage + road_coverage
```

## 커리큘럼 (구현됨, `curriculum.py`)

넉넉한 자금 → 점진 축소. Phase 승격은 "최근 N 에피소드 평균 인구 ≥ 목표비율" AND
"파산율 < 상한". 모델 가중치는 phase 간 유지(transfer).

| Phase | 초기 자금 | 인구 임계 | 파산율 상한 |
|-------|-----------|-----------|-------------|
| free_build | 2,000,000 | 60% | — |
| moderate | 500,000 | 50% | 20% |
| tight | 100,000 | 40% | 10% |
| realistic | 20,000 | (최종) | — |

---

## 원본(`ralph-city`)에서 import하여 재활용

```python
from model import Policy, FractalNet, FullyConv
from algo.ppo import PPO
from storage import RolloutStorage
from envs import make_vec_envs
from distributions import Categorical2D
from gym_city.envs.env import MicropolisEnv
from gym_city.envs.corecontrol import MicropolisControl
from utils import update_linear_schedule
```

(장기적으로 이들 의존성은 `mayorl/`로 흡수하여 self-contained화한다.)
