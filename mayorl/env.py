"""
BudgetCityEnv — 금전 제약이 있는 Micropolis RL 환경.

원본 MicropolisEnv를 상속하여 다음을 변경:
- 매 스텝 자금 리셋 제거 (예산이 실제로 차감됨)
- 건설 비용 체크 및 자금 부족 시 no-op
- observation에 budget 채널 추가
- action masking 지원
- 파산 시 에피소드 조기 종료
"""

import numpy as np
from gym import spaces

import sys
import os

# 원본 환경 import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gym_city.envs.env import MicropolisEnv
from gym_city.envs.tilemap import zoneFromInt

from .config import MayorlConfig, TOOL_COSTS, TOOL_COST_ARRAY
from .reward import RewardCalculator, StepInfo
from .vlm_mayor import NUM_GOALS


class BudgetCityEnv(MicropolisEnv):
    """금전 제약이 있는 도시 운영 환경.

    원본 MicropolisEnv와의 핵심 차이:
    1. 자금이 매 스텝 리셋되지 않음
    2. 건설 전 비용 확인 → 자금 부족 시 no-op + 페널티
    3. observation에 정규화된 잔액 채널 추가
    4. 파산(자금 <= 0) 시 에피소드 종료
    """

    def __init__(self, MAP_X=20, MAP_Y=20, PADDING=0, config=None):
        super().__init__(MAP_X=MAP_X, MAP_Y=MAP_Y, PADDING=PADDING)
        self.config = config or MayorlConfig()
        self._reward_calc = RewardCalculator(self.config)

        # 에피소드 통계 추적
        self._reset_episode_stats()

    def _reset_episode_stats(self):
        self._episode_stats = {
            'total_builds': 0,
            'failed_builds': 0,
            'total_spent': 0,
            'went_bankrupt': False,
            'tax_revenue': 0,
        }
        self._previous_funds = 0.0
        self._last_power_coverage = 0.0
        self._last_road_coverage = 0.0
        self._last_functional = 0.0
        # Phase 3: VLM 전략 목표 벡터 (기본값: all zeros = 목표 없음)
        self._strategic_goal = np.zeros(NUM_GOALS, dtype=np.float32)

    def post_gui(self):
        """원본 post_gui 호출 후 budget 관련 observation space 확장."""
        super().post_gui()

        # budget 채널 2개 (+ VLM일 때만 전략 목표 채널 NUM_GOALS개)
        # 비-VLM 학습에서는 goal 채널이 상수 0(죽은 입력)이므로 제외한다.
        cfg = getattr(self, 'config', None)
        use_vlm = bool(cfg and cfg.use_vlm)
        self._num_budget_channels = 2
        self._num_goal_channels = NUM_GOALS if use_vlm else 0
        self.num_obs_channels += self._num_budget_channels + self._num_goal_channels

        # observation space 재정의
        low_obs = np.full(
            (self.num_obs_channels, self.MAP_X, self.MAP_Y), fill_value=-1
        )
        high_obs = np.full(
            (self.num_obs_channels, self.MAP_X, self.MAP_Y), fill_value=1
        )
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, dtype=float
        )

        # tool → cost 매핑 구축 (self.micro.tools 기준)
        self._tool_costs = []
        for tool_name in self.micro.tools:
            cost = TOOL_COSTS.get(tool_name, 0)
            self._tool_costs.append(cost)

    def reset(self):
        """환경 리셋. 초기 자금과 세율을 config에서 가져옴."""
        self._reset_episode_stats()

        # 원본 reset 호출 전 init_funds를 config 값으로 덮어쓰기
        self.micro.init_funds = self.config.init_budget
        state = super().reset()

        # 탐색 부트스트랩: 검증된 백본(도로줄+전선줄+발전소 2기)을 static 프리빌드.
        # 에이전트 과제가 "전선 옆에 zone 놓기"로 축소된다 (_diag_power 시나리오 C 레이아웃).
        if getattr(self.config, 'prebuild_backbone', False):
            self._prebuild_backbone()
            state = self.getState()  # 백본이 반영된 관측으로 갱신

        # 세율 적용 (engine.cityTax = 0~20 정수)
        self.micro.setTaxRate(self.config.tax_rate)

        # autoBulldoze 정책 적용. 엔진 기본값(corecontrol=True)을 mayorl config로
        # 덮어쓴다. False면 기존 건물/잔해/숲 위 건설 시 Clear가 선행되어야 하며,
        # 재개발에 철거 비용이 든다. (공유 corecontrol의 전역 기본값은 건드리지 않음)
        self.micro.engine.autoBulldoze = bool(self.config.auto_bulldoze)

        # budget 상태 초기화
        self._current_funds = self.micro.getFunds()
        self._previous_funds = self._current_funds
        return self._append_budget_obs(state)

    def set_budget(self, budget):
        """커리큘럼 학습에서 동적으로 초기 자금 변경."""
        self.config.init_budget = budget

    def step(self, a, static_build=False):
        """자금 확인 후 액션 실행. 자금 리셋 없음."""
        if self.player_step:
            if self.static_player_builds:
                static_build = True
            a = self.player_step
            self.player_step = False

        action = self.intsToActions[a]
        tool_idx = action[0]
        cost = self._tool_costs[tool_idx]
        self._previous_funds = self.micro.getFunds()

        # 자금 확인
        if cost > self._previous_funds:
            # 자금 부족 → no-op (건설 안 함)
            self._episode_stats['failed_builds'] += 1
            return self._postact_budget(action_failed=True)
        else:
            # 건설 실행
            self.micro.takeAction(action, static_build)
            self._episode_stats['total_builds'] += 1
            self._episode_stats['total_spent'] += cost
            return self._postact_budget(action_failed=False)

    def _postact_budget(self, action_failed=False):
        """원본 postact를 대체. 자금 리셋을 제거한 버전."""
        # 핵심: self.micro.setFunds(self.micro.init_funds) 호출하지 않음!

        funds_before_tick = self.micro.getFunds()

        # 시뮬레이션 틱
        self.micro.simTick()

        funds_after_tick = self.micro.getFunds()
        tick_revenue = funds_after_tick - funds_before_tick
        if tick_revenue > 0:
            self._episode_stats['tax_revenue'] += tick_revenue

        # 상태 업데이트
        self.state = self.getState()
        self.curr_pop = self.getPop()
        self.last_city_metrics = self.city_metrics
        self.city_metrics = self.get_city_metrics()

        if self.render_gui:
            self.display_city_metrics()

        # 현재 자금
        self._current_funds = self.micro.getFunds()

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

        # 종료 조건
        self._episode_stats['went_bankrupt'] = bankrupt
        terminal = (bankrupt or self.num_step >= self.max_step) and self.auto_reset

        if self.print_map:
            self.printMap()
        if self.render_gui:
            self.micro.render()

        # info dict
        # 시장(VLM/LLM)이 종합 평가에 쓸 수 있도록 전체 도시 수치를 매 step 기록한다.
        # SubprocVecEnv는 이 dict를 그대로 전달하므로, trainer가 벡터 환경에서도
        # 실제 com/ind_pop·수요·커버리지·지지율을 집계할 수 있다.
        infos = {
            'funds': self._current_funds,
            'tax_rate': self.micro.getTaxRate(),
            'tick_revenue': tick_revenue,
            'bankrupt': bankrupt,
            'failed_build': action_failed,
            # 시장 평가용 풍부한 도시 수치
            'res_pop': self.city_metrics.get('res_pop', 0),
            'com_pop': self.city_metrics.get('com_pop', 0),
            'ind_pop': self.city_metrics.get('ind_pop', 0),
            'mayor_rating': self.city_metrics.get('mayor_rating', 0),
            'power_coverage': power_coverage,
            'road_coverage': road_coverage,
            'functional_fraction': functional,
        }
        if terminal:
            infos['episode_stats'] = self._episode_stats.copy()
            infos['reward_decomposition'] = self._reward_calc.decompose(step_info)

        # 플레이어 빌드 큐 처리
        if self.micro.player_builds:
            b = self.micro.player_builds[0]
            a_int = self.actionsToInts[b]
            infos['player_move'] = int(a_int)
            self.micro.player_builds = self.micro.player_builds[1:]
            self.player_step = a_int

        self.num_step += 1

        # budget 채널 추가한 state 반환
        state_with_budget = self._append_budget_obs(self.state)
        return (state_with_budget, reward, terminal, infos)

    def _calc_power_coverage(self):
        """전력 그리드 커버리지 비율 [0, 1]."""
        try:
            density_maps = self.micro.getDensityMaps()
            power_map = density_maps[0]  # power grid
            total_cells = self.MAP_X * self.MAP_Y
            powered_cells = np.count_nonzero(power_map)
            return powered_cells / max(total_cells, 1)
        except Exception:
            return 0.0

    def _calc_road_coverage(self):
        """도로 커버리지 비율 [0, 1]."""
        try:
            num_roads = getattr(self.micro, 'num_roads', 0)
            total_cells = self.MAP_X * self.MAP_Y
            return min(num_roads / max(total_cells, 1), 1.0)
        except Exception:
            return 0.0

    def _prebuild_backbone(self):
        """검증된 백본을 static으로 프리빌드하고 자금을 복원한다(백본은 무료 제공).

        레이아웃 (_diag_power 시나리오 C에서 zone만 제외):
            y=3      : 도로줄   (x=2..13) — 이후 zone의 도로 접근 제공
            y=7      : 전선줄   (x=2..13) — zone 급전용
            y=8..11  : 발전소 2기 (센터 x=3,10, y=9; 상단이 전선줄에 인접)
        static_build=True 라 에이전트가 파괴/덮어쓰기 불가.
        """
        m = self.micro
        for x in range(2, 14):
            m.doBotTool(x, 3, 'Road', static_build=True)
        for x in range(2, 14):
            m.doBotTool(x, 7, 'Wire', static_build=True)
        for cx in (3, 10):
            m.doBotTool(cx, 9, 'CoalPowerPlant', static_build=True)
        # 전력 스캔이 전선망을 에너지화하도록 몇 틱 진행
        for _ in range(8):
            m.simTick()
        # 백본 비용은 에이전트 예산에서 차감하지 않는다
        m.setFunds(self.config.init_budget)

    def _calc_functional_zone_fraction(self):
        """기능적 RCI zone 타일의 가중 비율 [0, 1].

        2단계 부분점수 (엔진 성장 조건 근거, zone.cpp:550-583):
        - 전력만 충족(도로 없음) zone → ``functional_partial_credit``(기본 0.5).
          초기 인구 성장은 전력만 필수(빈 zone은 트래픽 검사 스킵).
        - 전력+도로 모두 충족 zone → 1.0.
          인구가 생긴 뒤 도로 없으면 유출로 전환되므로 완전점수는 도로 포함.
        - 무전력 zone → 0 (엔진상 성장 완전 차단: zscore=-500).

        흩뿌린 인프라로는 오르지 않는다(연결돼야 점수). getTile과 getPowerGrid를
        동일 좌표 (i+MAP_XS, j+MAP_YS)로 질의해 정합을 보장한다.
        """
        try:
            MX, MY = self.MAP_X, self.MAP_Y
            xs, ys = self.micro.MAP_XS, self.micro.MAP_YS
            eng = self.micro.engine
            R = getattr(self.config, 'road_access_radius', 2)
            partial = getattr(self.config, 'functional_partial_credit', 0.5)

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

            weighted = 0.0
            for i in range(MX):
                for j in range(MY):
                    if is_zone[i, j] and powered[i, j]:
                        lo_i, hi_i = max(0, i - R), min(MX, i + R + 1)
                        lo_j, hi_j = max(0, j - R), min(MY, j + R + 1)
                        if is_road[lo_i:hi_i, lo_j:hi_j].any():
                            weighted += 1.0          # 전력+도로
                        else:
                            weighted += partial      # 전력만
            return weighted / max(MX * MY, 1)
        except Exception:
            return 0.0

    def set_strategic_goal(self, goal: np.ndarray) -> None:
        """외부(VLM 시장)에서 전략 목표 벡터를 주입한다.

        Parameters
        ----------
        goal : np.ndarray, shape (NUM_GOALS,)
            one-hot 또는 soft 벡터. FractalNet observation에 broadcast된다.
        """
        self._strategic_goal = np.asarray(goal, dtype=np.float32)

    def set_tax_rate_via_vlm(self, rate: int) -> None:
        """VLM 시장이 권장한 세율을 엔진에 적용한다.

        train.py의 ``--vlm-apply-tax`` 경로에서 ``env_method``로 호출된다.
        엔진 내부에서 0~20으로 clamp된다 (corecontrol.setTaxRate).

        Parameters
        ----------
        rate : int
            권장 세율 (0~20).
        """
        self.micro.setTaxRate(int(rate))

    def _get_demands(self):
        """현재 R/C/I 수요를 반환한다. 엔진 미지원 시 (0, 0, 0).

        VLM 시장 평가에서 ``env_method('_get_demands')``로 호출된다.
        """
        try:
            return tuple(self.micro.engine.getDemands())
        except Exception:
            return (0.0, 0.0, 0.0)

    def _append_budget_obs(self, state):
        """기존 observation에 budget 채널 2개 + goal 채널 NUM_GOALS개를 추가."""
        budget_channels = np.zeros(
            (self._num_budget_channels, self.MAP_X, self.MAP_Y)
        )

        # 채널 0: 정규화된 잔액 [-1, 1] 범위
        max_funds = self.config.init_budget * 2  # 세수로 초기값 이상 가능
        normalized_funds = np.clip(
            self._current_funds / max(max_funds, 1), -1.0, 1.0
        )
        budget_channels[0].fill(normalized_funds)

        # 채널 1: affordable ratio (건설 가능한 tool 비율)
        if self.num_tools > 0:
            affordable_count = sum(
                1 for cost in self._tool_costs if cost <= self._current_funds
            )
            affordable_ratio = affordable_count / self.num_tools
            # [0, 1] → [-1, 1] 로 스케일링
            budget_channels[1].fill(affordable_ratio * 2 - 1)

        parts = [state, budget_channels]

        # goal 채널: VLM 전략 목표를 맵 전체에 broadcast (VLM 사용 시에만 존재)
        if self._num_goal_channels > 0:
            goal_channels = np.zeros(
                (self._num_goal_channels, self.MAP_X, self.MAP_Y)
            )
            for i, g in enumerate(self._strategic_goal[: self._num_goal_channels]):
                goal_channels[i].fill(float(g))
            parts.append(goal_channels)

        return np.concatenate(parts, axis=0)

    def getState(self):
        """원본 getState를 그대로 사용 (budget 채널은 _append_budget_obs에서 추가)."""
        return super().getState()

    def action_masks(self):
        """각 액션의 유효성 마스크 반환. True = 실행 가능.

        shape: (num_tools * MAP_X * MAP_Y,)
        """
        masks = np.ones(self.num_tools * self.MAP_X * self.MAP_Y, dtype=bool)
        funds = self.micro.getFunds()

        for tool_idx, cost in enumerate(self._tool_costs):
            if cost > funds:
                # 이 tool의 모든 위치를 마스킹
                start = tool_idx * self.MAP_X * self.MAP_Y
                end = start + self.MAP_X * self.MAP_Y
                masks[start:end] = False

        return masks

    def get_episode_stats(self):
        """현재 에피소드의 통계 반환."""
        return self._episode_stats.copy()
