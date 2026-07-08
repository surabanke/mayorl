"""mayorl/config.py -- Configuration management for budget-constrained city RL.

Defines all hyperparameters, tool costs, reward weights, curriculum phases,
and environment settings as frozen dataclasses for reproducibility.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Tool cost table (Micropolis reference documentation values)
# ---------------------------------------------------------------------------

TOOL_COSTS: Dict[str, int] = {
    "Residential": 100,
    "Commercial": 100,
    "Industrial": 100,
    "Road": 10,
    "Wire": 5,
    "Rail": 20,
    "Park": 10,
    "Clear": 1,
    "PoliceDept": 500,
    "FireDept": 500,
    "Stadium": 3000,
    "CoalPowerPlant": 3000,
    "NuclearPowerPlant": 5000,
    "Seaport": 5000,
    "Airport": 10000,
    "Net": 1,
    "Water": 1,
    "Land": 1,
    "Forest": 1,
    "Nil": 0,
}

# Canonical tool ordering -- must match corecontrol.py ``self.tools``
TOOLS: List[str] = [
    "Residential",
    "Commercial",
    "Industrial",
    "FireDept",
    "PoliceDept",
    "Clear",
    "Wire",
    "Rail",
    "Road",
    "Stadium",
    "Park",
    "Seaport",
    "CoalPowerPlant",
    "NuclearPowerPlant",
    "Airport",
    "Net",
    "Water",
    "Land",
    "Forest",
    "Nil",
]

# Ordered list of costs matching ``TOOLS`` ordering for fast lookup.
TOOL_COST_ARRAY: List[int] = [TOOL_COSTS[t] for t in TOOLS]


# ---------------------------------------------------------------------------
# City metric definitions (mirrors env.py city_trgs / city_metrics)
# ---------------------------------------------------------------------------

CITY_METRIC_NAMES: List[str] = [
    "res_pop",
    "com_pop",
    "ind_pop",
    "traffic",
    "num_plants",
    "mayor_rating",
]

DEFAULT_CITY_TARGETS: OrderedDict[str, float] = OrderedDict(
    [
        ("res_pop", 500),
        ("com_pop", 50),
        ("ind_pop", 50),
        ("traffic", 2000),
        ("num_plants", 14),
        ("mayor_rating", 100),
    ]
)

DEFAULT_PARAM_BOUNDS: OrderedDict[str, Tuple[float, float]] = OrderedDict(
    [
        ("res_pop", (0, 750)),
        ("com_pop", (0, 100)),
        ("ind_pop", (0, 100)),
        ("traffic", (0, 2000)),
        ("num_plants", (0, 100)),
        ("mayor_rating", (0, 100)),
    ]
)


# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewardWeights:
    """Weights for each reward component.

    Attributes:
        population: weight for normalised population growth signal.
        budget_health: weight for fiscal-health shaping term.
        mayor_rating: weight for mayor-rating delta.
        invalid_action_penalty: flat penalty when agent attempts an
            unaffordable action.
        bankruptcy_penalty: large negative reward on bankruptcy
            (triggers episode termination).
        functional_shaping: weight for potential-based shaping on the
            fraction of zones that are both powered and road-connected
            (functional structure). Replaces the old raw power/road coverage.
    """

    population: float = 1.0
    budget_health: float = 0.3
    mayor_rating: float = 0.2
    invalid_action_penalty: float = -0.1
    bankruptcy_penalty: float = -10.0
    functional_shaping: float = 2.0


# ---------------------------------------------------------------------------
# Curriculum phase definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CurriculumPhase:
    """A single phase in the budget curriculum.

    Attributes:
        name: human-readable phase label.
        init_budget: starting funds for this phase.
        pop_threshold_ratio: fraction of ``pop_target`` the agent must
            exceed (rolling average) to graduate.
        max_bankruptcy_rate: maximum bankruptcy rate allowed for
            graduation.  ``None`` means no bankruptcy constraint (Phase 0).
    """

    name: str
    init_budget: int
    pop_threshold_ratio: float
    max_bankruptcy_rate: float | None


DEFAULT_CURRICULUM_PHASES: List[CurriculumPhase] = [
    CurriculumPhase(
        name="free_build",
        init_budget=2_000_000,
        pop_threshold_ratio=0.6,
        max_bankruptcy_rate=None,
    ),
    CurriculumPhase(
        name="moderate",
        init_budget=500_000,
        pop_threshold_ratio=0.5,
        max_bankruptcy_rate=0.2,
    ),
    CurriculumPhase(
        name="tight",
        init_budget=100_000,
        pop_threshold_ratio=0.4,
        max_bankruptcy_rate=0.1,
    ),
    CurriculumPhase(
        name="realistic",
        init_budget=20_000,
        pop_threshold_ratio=0.0,  # final phase -- no graduation
        max_bankruptcy_rate=None,
    ),
]


# ---------------------------------------------------------------------------
# Top-level environment configuration
# ---------------------------------------------------------------------------

@dataclass
class MayorlConfig:
    """Central configuration object for the mayorl environment.

    All fields have sensible defaults derived from the original
    ``MicropolisEnv`` and the CLAUDE.md implementation plan.
    """

    # -- Map --
    map_x: int = 20
    map_y: int = 20
    padding: int = 0

    # -- Budget --
    init_budget: int = 2_000_000
    min_funds: int = 0  # funds <= this triggers bankruptcy
    tax_rate: int = 7   # 0~20 정수; reset()에서 engine.cityTax에 직접 설정됨

    # -- Episode --
    max_steps: int | None = None  # ``None`` -> map_x * map_y

    # -- Rewards --
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    # -- City targets (POET-style) --
    city_targets: OrderedDict[str, float] = field(
        default_factory=lambda: OrderedDict(DEFAULT_CITY_TARGETS)
    )
    param_bounds: OrderedDict[str, Tuple[float, float]] = field(
        default_factory=lambda: OrderedDict(DEFAULT_PARAM_BOUNDS)
    )

    # -- Curriculum --
    curriculum_phases: List[CurriculumPhase] = field(
        default_factory=lambda: list(DEFAULT_CURRICULUM_PHASES)
    )
    curriculum_window: int = 100  # rolling window for promotion check

    # -- Population target used by curriculum thresholds --
    pop_target: float = 600.0

    # -- Misc --
    empty_start: bool = True
    render_gui: bool = False
    power_puzzle: bool = False
    random_builds: bool = False
    # 재개발 시 철거 비용까지 고려하게 하려면 False (엔진 기본값은 True).
    # False = 기존 건물/잔해/숲 위에 지으려면 Clear가 선행되어야 함.
    auto_bulldoze: bool = False

    # -- VLM / Observation --
    # VLM 시장을 쓸 때만 goal 채널(NUM_GOALS개)을 관측에 포함한다. 비-VLM 학습에서는
    # 이 채널들이 상수 0(죽은 입력)이 되므로 제외해 관측을 슬림화한다.
    # train.py가 args.use_vlm 값으로 덮어쓴다.
    use_vlm: bool = False
    # 관측 채널 스케일 정규화(ObservationNormWrapper). 원본 RCI/밀도 채널은 원값이고
    # mayorl budget 채널만 [-1,1]이라 스케일이 혼재됨 → running mean/std로 통일해
    # FullyConv(dirac init)의 학습 안정성을 확보한다.
    obs_norm: bool = True
    # 기능적 zone 판정 시 zone의 "도로 접근" 반경(Chebyshev). 반경 내 Road/RoadWire 존재.
    road_access_radius: int = 2
    # 전력만 충족(도로 없음)한 zone의 부분점수 가중. 엔진상 초기 인구 성장은 전력만
    # 필수(빈 zone은 트래픽 검사 스킵, zone.cpp:550-561)이므로 0이 아닌 gradient를 준다.
    # 도로까지 충족하면 1.0 (인구가 생긴 뒤 도로 없으면 유출로 전환되므로 완전점수는 도로 포함).
    functional_partial_credit: float = 0.5
    # 에피소드 시작 시 검증된 백본(도로줄+전선줄+발전소 2기)을 static으로 프리빌드.
    # 에이전트 과제를 "전선 옆에 zone 놓기"로 축소하는 탐색 부트스트랩. 비용은 무료(자금 복원).
    prebuild_backbone: bool = False

    # -- Derived ----------------------------------------------------------
    @property
    def effective_max_steps(self) -> int:
        if self.max_steps is not None:
            return self.max_steps
        return self.map_x * self.map_y

    def tool_cost(self, tool_name: str) -> int:
        """Return the build cost for *tool_name*."""
        return TOOL_COSTS[tool_name]

    def tool_cost_by_index(self, tool_idx: int) -> int:
        """Return the build cost for the tool at *tool_idx*."""
        return TOOL_COST_ARRAY[tool_idx]
