# mayorl 파악된 문제점 정리

CLAUDE.md의 "원본은 그대로 유지한 채 mayorl 폴더에서 학습환경과 MDP를 재구성한다"는
지침을 기준으로, 실제로 지켜지지 않았거나 계획과 다르게(약화되어) 구현된 부분,
그리고 그로 인해 파생된 버그를 정리한다. VLM Mayor 관련 사항은 제외(별도 논외).

각 항목은 "Evidence"(직접 확인한 코드/명령 결과)와 "Impact"(실제로 학습에 미치는 영향)를
구분해서 적는다 — 짐작인 부분은 명시.

---

## 1. [High] 원본 파일(`gym_city/envs/corecontrol.py`)이 실제로 직접 수정됨

**CLAUDE.md 위반**: "원본은 그대로 유지한 채" 라는 명시적 지침을 어기고 원본 소스가
uncommitted 상태로 수정되어 있다.

**Evidence** (`git diff -- gym_city/envs/corecontrol.py`):
```diff
 def simTick(self):
    #self.engine.resume()
-   #self.engine.cityEvaluation()
     self.engine.tickEngine()
     self.engine.simTick()
+    self.engine.cityEvaluation()
+    self.engine.updateDate()
+    self.engine.changeCensus()
    #self.engine.updateHeads()
-   #self.engine.updateDate()
-   #self.engine.changeCensus()
    #self.engine.simUpdate()
    #self.engine.doTimeStuff()
```
그리고 `MicropolisControl` 클래스에 새 메서드 3개가 통째로 추가됨:
```python
def getTaxRate(self): ...
def setTaxRate(self, rate): ...
def render_to_file(self, path, overlay='all'): ...
```

**경위 추정**: `mayorl/claude's_memo.md`에서 "cityEvaluation/updateDate/changeCensus가
꺼져 있으면 세수 징수가 안 돌아가서 mayorl의 예산 메커니즘 자체가 무의미해질 수 있다"는
우려를 제기했고, 이후 세션에서 이를 직접 활성화한 것으로 보인다. `setTaxRate`/`getTaxRate`도
`mayorl/env.py`(`self.micro.setTaxRate(...)`)가 필요로 해서 추가된 것.

**대안이 있었음**: `BudgetCityEnv`가 `MicropolisEnv`는 서브클래싱하지만 `self.micro`
(`MicropolisControl` 인스턴스)는 서브클래싱하지 않으므로, 원본을 건드리지 않고도
monkeypatch로 해결할 수 있었다 (예: `mayorl/env.py` 안에서
`MicropolisControl.setTaxRate = ...`를 런타임에 주입). 더 쉬운 직접 수정 경로를 택함.

**Impact**: 기능적으로는 오히려 "고쳐진" 상태(세수가 정상 작동하게 됨)라 당장 학습을
막지는 않는다. 다만 원본 repo와의 diff가 uncommitted로 방치되어 있어 pull/rebase 시
충돌 위험이 있고, "원본 불변" 원칙이 이미 깨져 있다는 사실 자체를 인지하고 있어야 한다.

---

## 2. [High] `num_actions` 하드코딩으로 인해 정책이 마지막 tool("Nil")을 절대 선택 못 함

**Evidence** (`model.py:30-43`, `Policy.__init__`):
```python
elif 'Micropolis' in args.env_name:
    if args.power_puzzle:
        num_actions = 1
    else:
        num_actions = 19          # 하드코딩 (TODO 주석: "have this already from env")
...
base_kwargs = {**base_kwargs, **{'num_actions': num_actions}}   # mayorl이 넘긴 값을 덮어씀
```
`mayorl/train.py:256-260`은 `num_actions = len(TOOLS)`(=20, `mayorl/config.py`의
`TOOLS` 리스트 = 엔진의 실제 `MicropolisControl.tools`와 정확히 동일한 20개 항목,
`corecontrol.py:85-106`에서 확인)를 계산해서 `base_kwargs`에 넣지만, `Policy.__init__`이
`args.env_name`(mayorl 기본값 `'MicropolisEnv-v0'`)에 `'Micropolis'`가 포함된다는
이유만으로 이 값을 무조건 19로 덮어쓴다.

**Impact**: `FullyConv`의 action head(`self.act = Conv2d(num_chan, num_actions, ...)`)가
19채널만 출력 → `Categorical2D`가 만드는 분포는 `19*H*W`개 카테고리뿐 →
`TOOLS` 리스트의 마지막 항목인 `"Nil"`(no-op 툴)은 정책이 **절대 샘플링할 수 없다**.
(참고: env 자체의 `action_space`는 `Discrete(20*H*W)`로 정의돼 있어 크래시는 안 나고,
그냥 조용히 마지막 20*H*W~19*H*W 구간이 미사용 상태로 남는다.)

---

## 3. [High] Action masking이 계획(로짓 마스킹)과 다르게 구현됐고, invalid 페널티가 사실상 죽은 코드가 됨

**CLAUDE.md 위반**: Step 2-3/5-1은 "invalid action의 logit을 -inf로 설정 → 학습 효율을
크게 높임"을 명시했다. 실제로는:

**Evidence 1 — 로짓 마스킹 미구현**: `model.py:121`의 `Policy.act(self, inputs, rnn_hxs, masks, ...)`는
mask 인자를 받지 않는 구조(원본 불변 원칙 때문에 수정 불가). 대신 `mayorl/wrapper.py:62-67`의
`ActionMaskWrapper.step()`이 **정책이 이미 샘플링한 액션을 env에 넘기기 직전에 사후 치환**한다:
```python
def step(self, action):
    if self._replace_invalid:
        masks = self.action_masks()
        if not masks[action]:
            action = self._noop_action   # Nil @ (0,0)
    return self.env.step(action)
```
정책은 여전히 전체 액션 공간에서 균등하게(혹은 학습된 분포로) 샘플링하며, 무효 액션에
확률질량을 계속 낭비한다 — pre-sampling 마스킹의 "탐색 효율 향상" 효과가 실현되지 않음.

**Evidence 2 — invalid_action_penalty가 사실상 발동 안 함**: 래퍼 순서는
`BudgetCityEnv -> BudgetMonitor -> ActionMaskWrapper`(`wrapper.py:280` 주석,
`ActionMaskWrapper`가 가장 바깥). `envs.step(action)` 호출 시 `ActionMaskWrapper`가
가장 먼저 가로채서 무효 액션을 Nil(비용 0)로 치환한 뒤에야 `BudgetCityEnv.step()`으로
넘어간다. 그런데 `BudgetCityEnv.step()`(`mayorl/env.py:105-128`) 자체도 독립적으로
비용 체크를 한다:
```python
cost = self._tool_costs[tool_idx]
if cost > self._previous_funds:
    self._episode_stats['failed_builds'] += 1
    return self._postact_budget(action_failed=True)   # invalid 페널티로 이어짐
```
`ActionMaskWrapper`가 무효 액션을 이미 Nil로 바꿔서 넘기기 때문에, `BudgetCityEnv`가
받는 액션은 (기본 설정 `action_mask=True`, `mayorl/train.py:185`에서) 거의 항상 이미
"합법적인" 액션이다. 따라서:
- `cost > self._previous_funds`가 참이 될 일이 사실상 없음 (Nil 비용 = 0)
- `action_failed`는 거의 항상 `False`
- `reward.py:169-170`의 `if info.action_was_invalid: reward += invalid_action_penalty`가
  **사실상 한 번도 발동하지 않는다**
- `episode_stats['failed_builds']`도 항상 0으로 남음

`mayorl/log.md`에 기록된 실제 학습 로그(update 665~795 내내 `invalid=0.0`)가 이 현상과
정확히 일치한다 — 에이전트가 무효 액션을 안 뽑아서가 아니라, 페널티 체크 지점에
도달하기도 전에 `ActionMaskWrapper`가 이미 지워버렸기 때문에 항상 0으로 보이는 것.

**Impact**: mayorl 리워드의 6개 컴포넌트 중 `invalid_action_penalty`가 기본 설정에서는
죽은 코드다. `RewardWeights.invalid_action_penalty`를 아무리 조정해도 학습에 영향을
주지 못한다. `README.md`의 튜닝 가이드("무작위로 비싼 건물만 지으면 →
`invalid_action_penalty` ↑")는 실제로는 작동하지 않는 조언이다.

**주의**: `action_mask=False`로 학습하면(`make_mayorl_vec_envs`에 하드코딩된
`action_mask=True`를 바꿔야 함) 이 문제는 사라지지만, 대신 문제 2(19채널 한계)와
겹쳐서 다른 형태의 혼란이 생길 수 있음 — 검증 필요.

---

## 4. [Medium] 기본 모델을 FractalNet 대신 FullyConv로 바꾼 근거가 문서화되어 있지 않음

**Evidence**: CLAUDE.md, `mayorl/README.md`, 코드 주석 어디에도 선택 이유가 없음.
`mayorl/log.md`(도커 파이프라인 디버깅 세션 기록)를 보면, 이 세션의 목표는 성능/구조
검토가 아니라 "파이프라인이 끝까지 도는지" 검증이었다 — `baselines` shim 작성,
`opencv-python-headless`/`torchsummary`/`shimmy` 설치, `RolloutStorage`의
`value_preds` 크기 버그 수정 등이 그 세션의 실제 작업 내용.

**추정**: `FractalNet`은 `n_recs`/`rule`/`intra_shr`/`inter_shr`/`drop_path` 등
하이퍼파라미터가 많고 디버깅 표면이 넓어서, 일단 파이프라인 검증용으로 더 단순한
`FullyConv`를 선택했을 가능성이 높다. 의도적 아키텍처 결정이라기보단 실용적 임시
선택으로 보인다. `--model FractalNet`으로 전환은 여전히 가능.

---

## 5. [Medium] Observation 정규화 불일치 (원본에서 물려받음)

**Evidence**: `post_gui()`에서 `observation_space`는 `[-1, 1]`로 선언되지만
(`low_obs`/`high_obs` = -1/1), 원본 RCI 스칼라 채널(`res_pop`/`com_pop`/`ind_pop`/
`resDemand`/`comDemand`/`indDemand`, `gym_city/envs/env.py:301-312`)은
`poet=False`(mayorl 기본값)일 때 **전혀 정규화되지 않고 원값 그대로** 맵에 broadcast된다
(`env.py:324-328`). 즉 선언된 관측 범위를 원본부터 위반하고 있었음.

mayorl은 이 문제를 고치지 않고 그대로 물려받았고, 자기가 새로 추가한 budget/affordable/
goal 채널(`mayorl/env.py:279-306`)만 `[-1, 1]`로 꼼꼼히 정규화했다. 결과적으로 같은
관측 텐서 안에 "정규화된 채널"과 "정규화 안 된 채널"이 섞여 있는 불균형이 존재한다.

**Impact**: 신경망 학습 시 스케일이 크게 다른 입력 채널이 섞이면(특히 `dirac_` 초기화를
쓰는 `FullyConv`처럼 입력 스케일에 민감한 초기화) 학습 안정성에 부정적 영향을 줄 수
있음 — 직접 측정하지 않았으므로 확정적 영향은 아니고 리스크로만 기록.

---

## 6. [Low] `autoBulldoze = True` 상시 활성화로 철거 전략이 배제됨

**Evidence**: `corecontrol.py:116`, 원본 그대로 유지된 설정.
`self.engine.autoBulldoze = True` — 기존 건물 위에 철거 절차 없이 바로 새 건물을
지을 수 있다.

**Impact**: mayorl의 경제 시스템(예산 제약 하에서 "언제, 어디에, 무엇을 지을지" 전략을
배우게 하는 것)이 의도한 전략적 깊이 중 "재개발 시 철거 비용까지 고려"하는 부분이
원천적으로 배제된다. `claude's_memo.md`도 이를 "mayorl에서 추가로 활성화할 만한 것"
중 하나로 언급했으나 아직 반영되지 않음.

---

## 7. [Info] `Makefile`도 uncommitted 상태로 변경되어 있음 (mayorl 작업과 무관해 보임)

**Evidence**: `git diff -- Makefile` — `sudo make install` → `make install` 변경,
그리고 `enjoy.py` 기반의 각종 데모 타겟(`MP_res_FC`, `nice_mix`, `GoL_SC` 등) 다수 추가.
mayorl 세션 로그(`log.md`, `claude's_memo.md`)에는 이 변경에 대한 언급이 없어
mayorl 작업과 직접 관련은 없어 보이지만, 원본 파일이 uncommitted로 남아있다는 점에서
1번 항목과 같은 맥락의 "원본 불변 원칙 이탈" 사례로 기록해둔다.

---

## 요약 표

| # | 심각도 | 문제 | 원본 불변 원칙 위반 여부 |
|---|--------|------|--------------------------|
| 1 | High | `corecontrol.py` 직접 수정 | 예 (명백) |
| 2 | High | `num_actions` 하드코딩 → Nil 선택 불가 | 아니오 (원본 버그가 mayorl에 전이) |
| 3 | High | Action masking 약화 + invalid penalty 죽은 코드 | 아니오 (설계와 실제 구현의 괴리) |
| 4 | Medium | FullyConv 선택 근거 미문서화 | 아니오 |
| 5 | Medium | Observation 정규화 불일치 | 아니오 (원본 버그 계승) |
| 6 | Low | autoBulldoze 상시 활성화 | 아니오 (원본 설정 유지) |
| 7 | Info | `Makefile` uncommitted 변경 | 예 (mayorl과 무관해 보임) |

---

## 8. [2026-07-08 점검] VLM 시장 역할 분석 — 중복/실효성 판정과 권고

`--use-vlm` 경로(`vlm_mayor.py` + `train.py:_run_vlm_evaluation`)의 세 출력이
실제로 학습에 기여하는지 점검한 결과:

| VLM 출력 | 현재 적용 | 판정 |
|---|---|---|
| `score` → `_vlm_bonus` | 다음 50 update 동안 **모든 스텝 reward에 상수 가산** (train.py `_collect_step`의 `reward_np + self._vlm_bonus`) | **사실상 무효 + 역할 중복**. 상수 오프셋은 rollout 내 모든 액션에 동일하게 붙어 advantage 계산에서 상쇄(value function이 흡수) → 정책 gradient 기여 ≈ 0. 하는 일(도시상태 평가)도 `RewardCalculator`와 중복 |
| `strategic_goal` → obs goal 채널 | 관측에 one-hot broadcast (`set_strategic_goal`) | **미접지(장식적)**. goal을 따를 유인(보상 연결)이 전혀 없어 정책이 무시해도 무방한 입력. RL과 중복은 아니나 현재로선 무의미 |
| `tax_recommendation` → 세율 | `--vlm-apply-tax`일 때만 (`set_tax_rate_via_vlm`) | **유일한 진짜 보완 역할**. RL 액션 공간에 세율이 없으므로 VLM만의 고유 레버. "VLM=시장(재정 거시), RL=공무원(건설 미시)" 역할분리 컨셉과 정확히 부합 |

추가 문제:
- 평가 입력이 **직전 rollout(num_steps=5 step)치 stats만** 집계 — 250 step 주기 대비
  시야가 지나치게 좁음(myopic). 에피소드 윈도우 집계로 넓혀야 함.
- pop=0인 현재 상태에서는 VLM이 매번 동일한 "전부 0" 도시를 보므로 정보량 0.
  → **VLM 개선은 pop 학습(탐색 문제) 해결 이후가 순서.**

**권고 역할 (우선순위순, 후속 작업):**
1. score 상수 보너스 **제거 또는 차분화**(Δscore를 평가 시점 1회만 가산). 현 구현은 무효.
2. **goal 접지**: strategic_goal이 shaping 가중치를 실제로 바꾸게 (예: `power_first` →
   `functional_shaping` 일시↑). 그래야 goal 채널이 학습 가능한 조건 입력이 됨.
3. **세율 통제를 VLM 기본 역할로** 승격 (`--vlm-apply-tax`를 VLM 사용 시 기본 on).
4. VLM을 **커리큘럼 심판**으로: 승격 판단/프리빌드 템플릿 선택에 score·issues 활용
   (보상 경로 밖이라 안전).
5. (로짓 마스킹 인프라 도입 후) 공간적 매크로 조언 → 툴/지역 logit prior 주입.
