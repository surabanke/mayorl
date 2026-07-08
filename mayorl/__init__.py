"""mayorl -- Budget-constrained Micropolis RL environment.

This package wraps the original ``gym_city`` Micropolis environment with
realistic budget constraints: funds are *not* reset each step, tool costs
are deducted, and tax revenue is the only income source.

Note: BudgetCityEnv, wrappers, and make_mayorl_env require the Micropolis
engine (GTK/gi). They are lazily imported to allow using config/reward/
curriculum/utils modules without GTK.
"""

from .config import (
    TOOL_COSTS,
    TOOL_COST_ARRAY,
    TOOLS,
    CITY_METRIC_NAMES,
    CurriculumPhase,
    MayorlConfig,
    RewardWeights,
)
from .curriculum import BudgetCurriculum, EpisodeStats
from .reward import RewardCalculator, StepInfo
from .utils import (
    EpisodeTracker,
    RunningMeanStd,
    compute_build_efficiency,
    format_episode_log,
    format_curriculum_log,
    seed_everything,
    flat_action_to_tuple,
    tuple_to_flat_action,
)


def __getattr__(name):
    """Lazy imports for modules that require GTK/Micropolis engine."""
    if name == "BudgetCityEnv":
        from .env import BudgetCityEnv
        return BudgetCityEnv
    if name in ("ActionMaskWrapper", "BudgetMonitor", "ObservationNormWrapper", "make_mayorl_env"):
        from . import wrapper
        return getattr(wrapper, name)
    if name in ("VLMMayor", "CityStats", "MayorEvaluation", "STRATEGIC_GOALS"):
        from . import vlm_mayor
        return getattr(vlm_mayor, name)
    raise AttributeError(f"module 'mayorl' has no attribute {name!r}")


__all__ = [
    # config
    "TOOL_COSTS",
    "TOOL_COST_ARRAY",
    "TOOLS",
    "CITY_METRIC_NAMES",
    "CurriculumPhase",
    "MayorlConfig",
    "RewardWeights",
    # reward
    "RewardCalculator",
    "StepInfo",
    # curriculum
    "BudgetCurriculum",
    "EpisodeStats",
    # wrapper (lazy)
    "ActionMaskWrapper",
    "BudgetMonitor",
    "ObservationNormWrapper",
    "make_mayorl_env",
    # env (lazy)
    "BudgetCityEnv",
    # utils
    "EpisodeTracker",
    "RunningMeanStd",
    "compute_build_efficiency",
    "format_episode_log",
    "format_curriculum_log",
    "seed_everything",
    "flat_action_to_tuple",
    "tuple_to_flat_action",
]
