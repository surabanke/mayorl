# 기능적 구조 보상 shaping — 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 인프라 shaping의 잠재함수를 `power_coverage + road_coverage`(흩뿌리기로 게임됨)에서 `전력+도로 충족 zone 비율`로 교체해, 인구가 자라기 전에도 기능적 도시 구조에 밀도 있는 gradient를 준다.

**Architecture:** potential-based shaping 형태 유지, Φ만 교체. reward 레이어(config+reward.py)는 엔진 없이 유닛테스트, env 탐지(`_calc_functional_zone_fraction`)는 Docker 프로브로 검증, 학습 관측성(functional_fraction 로깅)은 짧은 학습으로 확인.

**Tech Stack:** Python 3.10, numpy, PyTorch(PPO), Micropolis(GTK, Docker 전용). 스펙: `docs/superpowers/specs/2026-07-07-functional-reward-shaping-design.md`.

## Global Constraints

- reward/config 유닛테스트는 **conda isaac** 환경에서 실행(numpy 필요, 엔진 불필요): `conda activate isaac && python -m mayorl.tests.test_functional_reward`.
- env/학습 검증은 **Docker**(`mayorl-diag` 컨테이너, WSL2)에서 실행. 로컬(Windows/base python)은 엔진·numpy 부재로 불가.
- **커밋 보류**: 현재 `master` 브랜치이고 사용자가 커밋 보류 요청. 각 Task 마지막의 커밋 스텝은 **검증 체크포인트**로 간주하고, 실제 `git commit`은 사용자 승인 후 별도 브랜치에서 일괄 수행한다.
- 좌표 규약: 타일·전력은 반드시 `(i+MAP_XS, j+MAP_YS)` 동일 좌표로 질의(정합 보장, `_diag_power.py`에서 검증됨).
- 불변: pop/budget/mayor_rating/penalty 항, `_calc_power_coverage`/`_calc_road_coverage`(info dict/VLM용 존치), 커리큘럼/VLM.

---

### Task 1: Reward 레이어 (config 가중치 + reward.py)

**Files:**
- Create: `mayorl/tests/__init__.py`
- Create: `mayorl/tests/test_functional_reward.py`
- Modify: `mayorl/config.py:130` (RewardWeights), `mayorl/config.py` MayorlConfig misc 섹션
- Modify: `mayorl/reward.py` (StepInfo 필드, `functional_potential`, `RewardCalculator.compute`/`decompose`)

**Interfaces:**
- Produces:
  - `RewardWeights.functional_shaping: float = 2.0` (기존 `infrastructure_shaping` 제거)
  - `MayorlConfig.road_access_radius: int = 2`
  - `mayorl.reward.functional_potential(functional_fraction: float) -> float`
  - `StepInfo(..., functional_fraction: float = 0.0, last_functional_fraction: float = 0.0)` (기존 `power_coverage/last_power_coverage/road_coverage/last_road_coverage` 4필드 제거)
  - `RewardCalculator.compute(info: StepInfo) -> float` 는 shaping 항으로 `functional_shaping * (functional_potential(info.functional_fraction) - functional_potential(info.last_functional_fraction))` 를 더한다.

- [ ] **Step 1: 실패 테스트 작성**

`mayorl/tests/__init__.py` (빈 파일):
```python
```

`mayorl/tests/test_functional_reward.py`:
```python
"""functional shaping 리워드 유닛테스트 (엔진 불필요, conda isaac에서 실행)."""
from mayorl.config import RewardWeights, MayorlConfig
from mayorl.reward import RewardCalculator, StepInfo, functional_potential

_ZERO_METRICS = {'res_pop': 0, 'com_pop': 0, 'ind_pop': 0, 'mayor_rating': 0}


def _base_kwargs():
    return dict(
        city_metrics=dict(_ZERO_METRICS),
        last_city_metrics=dict(_ZERO_METRICS),
        current_funds=100.0, previous_funds=100.0, init_budget=100.0,
    )


def test_weight_renamed():
    w = RewardWeights()
    assert w.functional_shaping == 2.0
    assert not hasattr(w, 'infrastructure_shaping')


def test_road_access_radius_default():
    assert MayorlConfig().road_access_radius == 2


def test_functional_potential_identity():
    assert functional_potential(0.0) == 0.0
    assert functional_potential(0.25) == 0.25


def test_compute_rewards_functional_progress():
    cfg = MayorlConfig()
    calc = RewardCalculator(cfg)
    info = StepInfo(**_base_kwargs(),
                    functional_fraction=0.5, last_functional_fraction=0.0)
    r = calc.compute(info)
    assert abs(r - cfg.reward_weights.functional_shaping * 0.5) < 1e-9


def test_compute_no_shaping_when_flat():
    cfg = MayorlConfig()
    calc = RewardCalculator(cfg)
    info = StepInfo(**_base_kwargs(),
                    functional_fraction=0.3, last_functional_fraction=0.3)
    assert calc.compute(info) == 0.0


def test_stepinfo_has_no_coverage_fields():
    info = StepInfo(**_base_kwargs())
    for dead in ('power_coverage', 'road_coverage',
                 'last_power_coverage', 'last_road_coverage'):
        assert not hasattr(info, dead), f'{dead} should be removed'


def main():
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
            passed += 1
    print(f'ALL PASS ({passed})')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `conda activate isaac && python -m mayorl.tests.test_functional_reward`
Expected: FAIL — `ImportError: cannot import name 'functional_potential'` (또는 `AttributeError: functional_shaping`).

- [ ] **Step 3: config.py 수정**

`mayorl/config.py` — `RewardWeights` 안 `infrastructure_shaping: float = 0.1` 줄을 교체:
```python
    functional_shaping: float = 2.0
```
`mayorl/config.py` — `MayorlConfig`의 `auto_bulldoze: bool = False` 다음(VLM/Observation 섹션 근처)에 추가:
```python
    # 기능적 zone 판정 시 zone의 "도로 접근" 반경(Chebyshev). 반경 내 Road/RoadWire 존재.
    road_access_radius: int = 2
```

- [ ] **Step 4: reward.py 수정**

`mayorl/reward.py` — `StepInfo`의 Infrastructure 4필드를 교체:
```python
    # Functional structure (for potential-based shaping)
    # 전력+도로가 모두 충족된 zone 타일 비율 [0,1]
    functional_fraction: float = 0.0
    last_functional_fraction: float = 0.0
```
`infrastructure_potential` 함수를 교체:
```python
def functional_potential(functional_fraction: float) -> float:
    """Potential Phi(s) = 전력+도로 충족 zone 비율.

    흩뿌린 인프라로는 오르지 않고, 발전소-전력-도로가 zone에 실제로 연결돼야만
    오른다. potential-based shaping의 잠재함수로 쓰인다.
    """
    return functional_fraction
```
`RewardCalculator.compute` 의 인프라 shaping 블록(주석 `# 4. Infrastructure ...`)을 교체:
```python
        # 4. Functional-structure potential-based shaping (gamma=1 simplification)
        phi_new = functional_potential(info.functional_fraction)
        phi_old = functional_potential(info.last_functional_fraction)
        reward += self._w.functional_shaping * (phi_new - phi_old)
```
`RewardCalculator.decompose` 의 phi 계산과 반환 dict 항목 교체:
```python
        phi_new = functional_potential(info.functional_fraction)
        phi_old = functional_potential(info.last_functional_fraction)
```
그리고 decompose 반환 dict의 `"infra_shaping"` 항목을 교체:
```python
            "functional_shaping": self._w.functional_shaping * (
                phi_new - phi_old
            ),
```

- [ ] **Step 5: 테스트 실행 → 통과 확인**

Run: `conda activate isaac && python -m mayorl.tests.test_functional_reward`
Expected: PASS — `ALL PASS (6)`.

- [ ] **Step 6: 커밋 (보류 — 체크포인트)**

실제 커밋은 사용자 승인 후. 검증만 확인:
```bash
git add mayorl/tests/ mayorl/config.py mayorl/reward.py
# git commit -m "feat(reward): functional-structure shaping (config+reward layer)"  # 보류
```

---

### Task 2: env 기능적 zone 탐지 + 배선

**Files:**
- Modify: `mayorl/env.py` (import 추가, `_reset_episode_stats`, `_calc_functional_zone_fraction` 신규, `_postact_budget` 배선, info dict)
- Create: `mayorl/_diag_functional.py` (Docker 프로브 — 검증용)

**Interfaces:**
- Consumes: Task 1의 `StepInfo(functional_fraction=..., last_functional_fraction=...)`.
- Produces:
  - `BudgetCityEnv._calc_functional_zone_fraction() -> float` (전력+도로 충족 zone 비율 [0,1])
  - info dict에 `'functional_fraction'` 키 추가.

- [ ] **Step 1: 검증 프로브 작성 (실패 상태 확인용)**

`mayorl/_diag_functional.py`:
```python
"""기능적 zone 탐지 검증 (Docker). 클린 클러스터는 functional>0, 흩뿌린 layGrid는 ~0.

Docker:
    docker exec -e PYTHONPATH=/usr/src/app/mayorl/_shim mayorl-diag \\
      sh -c 'cd /usr/src/app && xvfb-run -a python3 -u -m mayorl._diag_functional'
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mayorl.config import MayorlConfig
from mayorl.env import BudgetCityEnv


def _make_env(map_w=16):
    cfg = MayorlConfig(map_x=map_w, map_y=map_w, init_budget=2_000_000,
                       max_steps=100000, empty_start=True, render_gui=False,
                       auto_bulldoze=True)
    env = BudgetCityEnv(MAP_X=map_w, MAP_Y=map_w, config=cfg)
    env.setMapSize((map_w, map_w), rank=0, max_step=100000, render_gui=False)
    env.reset()
    return env, map_w


def _place(env, tool, x, y, map_w):
    idx = env.micro.tools.index(tool)
    env.step(idx * (map_w * map_w) + x * map_w + y)


def build_clean(env, map_w):
    m = env.micro
    def b(x, y, t):
        m.simTick(); m.doTool(x, y, t)
    for x in range(2, 14): b(x, 3, "Road")
    for cx in (3, 6, 9, 12): b(cx, 5, "Residential")
    for x in range(2, 14): b(x, 7, "Wire")
    for cx in (3, 10): _place(env, "CoalPowerPlant", cx, 9, map_w)
    for _ in range(20): m.simTick()


def build_laygrid(env, map_w):
    env.micro.layGrid(5, 5)
    for (x, y) in [(3, 3), (3, 11), (11, 3), (11, 11)]:
        _place(env, "CoalPowerPlant", x, y, map_w)


def main():
    print("=" * 60)
    env, map_w = _make_env()
    build_clean(env, map_w)
    f_clean = env._calc_functional_zone_fraction()
    print(f"클린 클러스터  functional_fraction = {f_clean:.4f}  (기대: > 0)")

    env, map_w = _make_env()
    build_laygrid(env, map_w)
    f_lay = env._calc_functional_zone_fraction()
    print(f"layGrid(흩뿌림) functional_fraction = {f_lay:.4f}  (기대: ~0)")
    print("=" * 60)
    ok = (f_clean > 0.0) and (f_lay < f_clean)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 프로브 실행 → 실패 확인**

Run (Docker/WSL2):
```bash
docker exec -e PYTHONPATH=/usr/src/app/mayorl/_shim mayorl-diag \
  sh -c 'cd /usr/src/app && timeout 120 xvfb-run -a python3 -u -m mayorl._diag_functional; echo EXITCODE=$?'
```
Expected: FAIL — `AttributeError: 'BudgetCityEnv' object has no attribute '_calc_functional_zone_fraction'`.

- [ ] **Step 3: env.py import 추가**

`mayorl/env.py` 상단 import(예: `from gym_city.envs.env import MicropolisEnv` 다음 줄)에 추가:
```python
from gym_city.envs.tilemap import zoneFromInt
```

- [ ] **Step 4: `_reset_episode_stats`에 상태 변수 추가**

`mayorl/env.py` `_reset_episode_stats` 내 `self._last_road_coverage = 0.0` 다음 줄에 추가:
```python
        self._last_functional = 0.0
```

- [ ] **Step 5: `_calc_functional_zone_fraction` 신규 메서드 추가**

`mayorl/env.py` `_calc_road_coverage` 메서드 바로 다음에 추가:
```python
    def _calc_functional_zone_fraction(self):
        """전력+도로가 모두 충족된 RCI zone 타일 비율 [0, 1].

        흩뿌린 인프라로는 오르지 않는다(연결돼야 점수). getTile과 getPowerGrid를
        동일 좌표 (i+MAP_XS, j+MAP_YS)로 질의해 정합을 보장한다.
        """
        try:
            MX, MY = self.MAP_X, self.MAP_Y
            xs, ys = self.micro.MAP_XS, self.micro.MAP_YS
            eng = self.micro.engine
            R = getattr(self.config, 'road_access_radius', 2)

            is_zone = np.zeros((MX, MY), dtype=bool)
            is_road = np.zeros((MX, MY), dtype=bool)
            powered = np.zeros((MX, MY), dtype=bool)
            for i in range(MX):
                for j in range(MY):
                    cls = zoneFromInt(eng.getTile(i + xs, j + ys) & 1023)
                    if cls in ('Residential', 'Commercial', 'Industrial'):
                        is_zone[i, j] = True
                    elif cls in ('Road', 'RoadWire'):
                        is_road[i, j] = True
                    if eng.getPowerGrid(i + xs, j + ys) > 0:
                        powered[i, j] = True

            functional_count = 0
            for i in range(MX):
                for j in range(MY):
                    if is_zone[i, j] and powered[i, j]:
                        lo_i, hi_i = max(0, i - R), min(MX, i + R + 1)
                        lo_j, hi_j = max(0, j - R), min(MY, j + R + 1)
                        if is_road[lo_i:hi_i, lo_j:hi_j].any():
                            functional_count += 1
            return functional_count / max(MX * MY, 1)
        except Exception:
            return 0.0
```

- [ ] **Step 6: `_postact_budget` 배선 교체**

`mayorl/env.py` `_postact_budget` 내 인프라 커버리지/StepInfo 블록을 교체.
기존:
```python
        # 인프라 커버리지 계산 (reward shaping용)
        power_coverage = self._calc_power_coverage()
        road_coverage = self._calc_road_coverage()

        # RewardCalculator를 통한 리워드 계산
        bankrupt = self._current_funds <= self.config.min_funds
        step_info = StepInfo(
            city_metrics=self.city_metrics,
            last_city_metrics=self.last_city_metrics,
            current_funds=self._current_funds,
            previous_funds=self._previous_funds,
            init_budget=self.config.init_budget,
            action_was_invalid=action_failed,
            went_bankrupt=bankrupt,
            power_coverage=power_coverage,
            last_power_coverage=self._last_power_coverage,
            road_coverage=road_coverage,
            last_road_coverage=self._last_road_coverage,
        )
        reward = self._reward_calc.compute(step_info)

        # 인프라 커버리지 기록 (다음 스텝용)
        self._last_power_coverage = power_coverage
        self._last_road_coverage = road_coverage
```
교체 후:
```python
        # 인프라 커버리지(info/VLM용 유지) + 기능적 zone 비율(reward shaping용)
        power_coverage = self._calc_power_coverage()
        road_coverage = self._calc_road_coverage()
        functional = self._calc_functional_zone_fraction()

        # RewardCalculator를 통한 리워드 계산
        bankrupt = self._current_funds <= self.config.min_funds
        step_info = StepInfo(
            city_metrics=self.city_metrics,
            last_city_metrics=self.last_city_metrics,
            current_funds=self._current_funds,
            previous_funds=self._previous_funds,
            init_budget=self.config.init_budget,
            action_was_invalid=action_failed,
            went_bankrupt=bankrupt,
            functional_fraction=functional,
            last_functional_fraction=self._last_functional,
        )
        reward = self._reward_calc.compute(step_info)

        # 다음 스텝용 기록
        self._last_power_coverage = power_coverage
        self._last_road_coverage = road_coverage
        self._last_functional = functional
```

- [ ] **Step 7: info dict에 functional_fraction 추가**

`mayorl/env.py` `_postact_budget` 의 info dict에서 `'road_coverage': road_coverage,` 다음 줄에 추가:
```python
            'functional_fraction': functional,
```

- [ ] **Step 8: 프로브 실행 → 통과 확인**

Run (Docker/WSL2):
```bash
docker exec -e PYTHONPATH=/usr/src/app/mayorl/_shim mayorl-diag \
  sh -c 'cd /usr/src/app && timeout 120 xvfb-run -a python3 -u -m mayorl._diag_functional; echo EXITCODE=$?'
```
Expected: PASS — `클린 클러스터 functional_fraction > 0`, `layGrid ~0`, 마지막 줄 `PASS`.

- [ ] **Step 9: 커밋 (보류 — 체크포인트)**

```bash
git add mayorl/env.py mayorl/_diag_functional.py
# git commit -m "feat(env): functional zone detection + reward wiring"  # 보류
```

---

### Task 3: 학습 관측성 (functional_fraction 로깅)

**Files:**
- Modify: `mayorl/train.py` (`__init__` deque 추가, `_collect_step` 수집, `_log_progress` 출력)

**Interfaces:**
- Consumes: Task 2의 info dict `'functional_fraction'`.
- Produces: 학습 로그 각 줄에 `func=<평균 functional_fraction>` 컬럼.

- [ ] **Step 1: 최근 functional 추적 deque 추가**

`mayorl/train.py` `MayorlTrainer.__init__` 의 `self.episode_rewards = deque(maxlen=100)` 다음 줄에 추가:
```python
        self._functional_recent = deque(maxlen=100)
```

- [ ] **Step 2: `_collect_step`에서 functional 수집**

`mayorl/train.py` `_collect_step` 의 `obs, reward, done, infos = self.envs.step(actions_np)` 다음에 추가:
```python
        # 관측성: 스텝별 functional_fraction 평균을 최근 버퍼에 기록
        fr = [info.get('functional_fraction', 0.0) for info in infos]
        if fr:
            self._functional_recent.append(sum(fr) / len(fr))
```

- [ ] **Step 3: `_log_progress`에 func 컬럼 출력**

`mayorl/train.py` `_log_progress` 의 `summary = self.episode_tracker.summary()` 블록 다음, `msg_parts.extend([... v_loss ...])` 앞에 추가:
```python
        if self._functional_recent:
            msg_parts.append(
                f'func={np.mean(self._functional_recent):.4f}'
            )
```

- [ ] **Step 4: 짧은 학습으로 관측 확인**

Run (Docker/WSL2):
```bash
docker exec -e PYTHONPATH=/usr/src/app/mayorl/_shim mayorl-diag \
  sh -c 'cd /usr/src/app && timeout 400 xvfb-run -a python3 -u -m mayorl.train --algo ppo --map-width 16 --num-processes 1 --num-frames 500000 --log-interval 50; echo EXITCODE=$?'
```
Expected: 크래시 없이 로그 각 줄에 `func=...` 표시. **`func`이 0 위로 오른 뒤 `pop`이 따라 오르는지** 관찰(설계 의도 검증). 초기 몇 분으로는 미세할 수 있음 — 추세만 확인.

- [ ] **Step 5: 커밋 (보류 — 체크포인트)**

```bash
git add mayorl/train.py
# git commit -m "feat(train): log functional_fraction"  # 보류
```

---

## Self-Review

**Spec coverage:**
- §3 Φ 교체 → Task 1 Step 4. ✓
- §4 탐지(_calc_functional_zone_fraction, 좌표 정합, 반경 도로접근) → Task 2 Step 5. ✓
- §5 reward.py/config(StepInfo, functional_potential, weight rename, decompose) → Task 1. ✓
- §6 env `_postact_budget` 배선 + `_last_functional` → Task 2 Step 4,6. ✓
- §7 관측성(info + 로그) → Task 2 Step 7 + Task 3. ✓
- §8 테스트(reward 유닛 / env 탐지 클린vs layGrid / 짧은 학습) → Task 1 Step1, Task 2, Task 3 Step4. ✓
- §9 튜닝 파라미터(functional_shaping, road_access_radius, N_norm=map면적) → Task 1(가중치/반경), Task 2(map면적 정규화). ✓
- §10 Deferred(게이팅/prebuild) → 계획에 미포함(의도). ✓

**Placeholder scan:** 모든 코드 스텝에 실제 코드 포함, "TODO/적절히 처리" 없음. ✓

**Type consistency:** `functional_fraction`/`last_functional_fraction`(reward StepInfo, env, info dict), `functional_potential(float)->float`(reward, decompose), `functional_shaping`(config, reward compute/decompose), `_calc_functional_zone_fraction()`(env 정의/호출), `_last_functional`(env reset/postact) — 명명 일관. ✓
