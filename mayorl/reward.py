"""mayorl/reward.py -- Reward functions for budget-constrained city RL.

The reward integrates:
  1. Population growth (res + com + ind delta)
  2. Budget / fiscal health shaping
  3. Mayor-rating delta
  4. Invalid-action penalty (unaffordable builds)
  5. Bankruptcy penalty (large negative, triggers termination)
  6. Infrastructure potential-based shaping (power + roads)

All component functions are stateless and accept plain dicts / scalars so
they can be unit-tested independently of the Micropolis engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .config import MayorlConfig, RewardWeights


# ---------------------------------------------------------------------------
# Episode step info container
# ---------------------------------------------------------------------------

@dataclass
class StepInfo:
    """Snapshot of a single environment transition.

    ``RewardCalculator.compute`` consumes this to produce a scalar reward.
    All population / metric fields should already be absolute values
    (not deltas).
    """

    # Current city metrics (after step)
    city_metrics: Dict[str, float]
    # Previous city metrics (before step)
    last_city_metrics: Dict[str, float]

    # Budget
    current_funds: float
    previous_funds: float
    init_budget: float

    # Flags
    action_was_invalid: bool = False
    went_bankrupt: bool = False

    # Functional structure (for potential-based shaping)
    # 전력+도로가 모두 충족된 zone 타일 비율 [0,1]
    functional_fraction: float = 0.0
    last_functional_fraction: float = 0.0


# ---------------------------------------------------------------------------
# Individual reward components
# ---------------------------------------------------------------------------

def population_reward(
    metrics: Dict[str, float],
    last_metrics: Dict[str, float],
) -> float:
    """Delta in total population (res + com + ind), normalised."""
    pop_keys = ("res_pop", "com_pop", "ind_pop")
    curr = sum(metrics.get(k, 0.0) for k in pop_keys)
    prev = sum(last_metrics.get(k, 0.0) for k in pop_keys)
    return curr - prev


def budget_health_reward(
    current_funds: float,
    previous_funds: float,
    init_budget: float,
) -> float:
    """Change in normalised budget level.

    Positive when funds grow (e.g. via tax revenue), negative when the
    agent overspends relative to income.  Clamped to [-1, 1].
    """
    if init_budget <= 0:
        return 0.0
    delta = (current_funds - previous_funds) / init_budget
    return max(-1.0, min(1.0, delta))


def mayor_rating_reward(
    metrics: Dict[str, float],
    last_metrics: Dict[str, float],
) -> float:
    """Delta in mayor approval rating (0-100 scale)."""
    curr = metrics.get("mayor_rating", 0.0)
    prev = last_metrics.get("mayor_rating", 0.0)
    return curr - prev


def functional_potential(functional_fraction: float) -> float:
    """Potential Phi(s) = 전력+도로 충족 zone 비율.

    흩뿌린 인프라로는 오르지 않고, 발전소-전력-도로가 zone에 실제로 연결돼야만
    오른다. potential-based shaping의 잠재함수로 쓰인다.
    """
    return functional_fraction


# ---------------------------------------------------------------------------
# Main reward calculator
# ---------------------------------------------------------------------------

class RewardCalculator:
    """Computes the composite reward for a single environment step.

    Parameters
    ----------
    config : MayorlConfig
        Central configuration (provides ``reward_weights``).
    """

    def __init__(self, config: Optional[MayorlConfig] = None) -> None:
        if config is None:
            config = MayorlConfig()
        self._w: RewardWeights = config.reward_weights

    # -- public API --------------------------------------------------------

    def compute(self, info: StepInfo) -> float:
        """Return the scalar reward for the transition described by *info*.

        The reward decomposes as::

            R = w_pop   * delta_population
              + w_budget * budget_health
              + w_rating * delta_mayor_rating
              + w_infra  * (Phi(s') - Phi(s))   # potential-based shaping
              + penalty_invalid
              + penalty_bankrupt
        """
        reward = 0.0

        # 1. Population growth
        reward += self._w.population * population_reward(
            info.city_metrics, info.last_city_metrics
        )

        # 2. Fiscal health
        reward += self._w.budget_health * budget_health_reward(
            info.current_funds, info.previous_funds, info.init_budget
        )

        # 3. Mayor rating
        reward += self._w.mayor_rating * mayor_rating_reward(
            info.city_metrics, info.last_city_metrics
        )

        # 4. Functional-structure potential-based shaping (gamma=1 simplification)
        phi_new = functional_potential(info.functional_fraction)
        phi_old = functional_potential(info.last_functional_fraction)
        reward += self._w.functional_shaping * (phi_new - phi_old)

        # 5. Invalid-action penalty
        if info.action_was_invalid:
            reward += self._w.invalid_action_penalty

        # 6. Bankruptcy penalty
        if info.went_bankrupt:
            reward += self._w.bankruptcy_penalty

        return reward

    # -- convenience -------------------------------------------------------

    def decompose(self, info: StepInfo) -> Dict[str, float]:
        """Return a dict of individual reward components (for logging)."""
        phi_new = functional_potential(info.functional_fraction)
        phi_old = functional_potential(info.last_functional_fraction)
        return {
            "population": self._w.population * population_reward(
                info.city_metrics, info.last_city_metrics
            ),
            "budget_health": self._w.budget_health * budget_health_reward(
                info.current_funds, info.previous_funds, info.init_budget
            ),
            "mayor_rating": self._w.mayor_rating * mayor_rating_reward(
                info.city_metrics, info.last_city_metrics
            ),
            "functional_shaping": self._w.functional_shaping * (
                phi_new - phi_old
            ),
            "invalid_penalty": (
                self._w.invalid_action_penalty
                if info.action_was_invalid
                else 0.0
            ),
            "bankruptcy_penalty": (
                self._w.bankruptcy_penalty if info.went_bankrupt else 0.0
            ),
        }
