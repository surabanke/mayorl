"""mayorl/utils.py -- Utility functions for budget-constrained city RL.

Provides:
  - EpisodeTracker: accumulates episode statistics across training
  - format_episode_log: human-readable single-line episode summary
  - format_curriculum_log: curriculum phase summary for logging
  - compute_build_efficiency: population-per-dollar metric
  - RunningMeanStd: incremental mean/std tracker (generic)
  - seed_everything: reproducibility helper
"""

from __future__ import annotations

import logging
import random
from collections import deque
from typing import Any, Deque, Dict, Optional, Sequence

import numpy as np

from .config import TOOL_COSTS, TOOLS
from .curriculum import EpisodeStats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Running mean / std tracker
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Welford's online algorithm for incremental mean and variance.

    Parameters
    ----------
    shape : tuple
        Shape of the values being tracked (scalar = ``()``).
    """

    def __init__(self, shape: tuple = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count: int = 0

    def update(self, x: np.ndarray) -> None:
        """Update statistics with a new sample (or batch along axis 0)."""
        batch = np.asarray(x, dtype=np.float64)
        if batch.ndim == len(self.mean.shape):
            batch = batch[np.newaxis, ...]  # single sample -> batch of 1

        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        total = self.count + batch_count

        new_mean = self.mean + delta * batch_count / max(total, 1)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / max(total, 1)

        self.mean = new_mean
        self.var = m2 / max(total, 1)
        self.count = total

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)


# ---------------------------------------------------------------------------
# Episode tracker
# ---------------------------------------------------------------------------

class EpisodeTracker:
    """Accumulates episode-level statistics for logging and evaluation.

    Maintains a bounded deque of recent episodes plus lifetime counters.

    Parameters
    ----------
    window_size : int
        Number of recent episodes kept for rolling averages.
    """

    def __init__(self, window_size: int = 100) -> None:
        self._window_size = window_size
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=window_size)
        self._total_episodes: int = 0
        self._total_steps: int = 0

    def record(self, stats: Dict[str, Any]) -> None:
        """Record one completed episode's summary dict.

        Accepts the ``budget_monitor`` dict produced by
        ``BudgetMonitor`` or any dict with compatible keys.
        """
        self._buffer.append(stats)
        self._total_episodes += 1
        self._total_steps += stats.get("num_steps", 0)

    def record_from_episode_stats(self, es: EpisodeStats) -> None:
        """Convenience: record from an ``EpisodeStats`` dataclass."""
        self.record({
            "final_population": es.final_population,
            "went_bankrupt": es.went_bankrupt,
            "final_funds": es.final_funds,
            "num_steps": es.num_steps,
            "total_reward": es.total_reward,
            "invalid_action_count": es.invalid_action_count,
        })

    # -- Aggregated statistics ------------------------------------------------

    def rolling_mean(self, key: str, default: float = 0.0) -> float:
        """Mean of *key* over the rolling window."""
        if not self._buffer:
            return default
        values = [ep.get(key, default) for ep in self._buffer]
        return sum(values) / len(values)

    def rolling_rate(self, key: str) -> float:
        """Fraction of episodes where ``key`` is truthy."""
        if not self._buffer:
            return 0.0
        return sum(1 for ep in self._buffer if ep.get(key)) / len(self._buffer)

    @property
    def total_episodes(self) -> int:
        return self._total_episodes

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def summary(self) -> Dict[str, float]:
        """Return a dict of rolling-window summary statistics."""
        return {
            "total_episodes": self._total_episodes,
            "total_steps": self._total_steps,
            "avg_population": self.rolling_mean("final_population"),
            "avg_reward": self.rolling_mean("total_reward"),
            "avg_funds": self.rolling_mean("final_funds"),
            "bankruptcy_rate": self.rolling_rate("went_bankrupt"),
            "avg_invalid_actions": self.rolling_mean("invalid_action_count"),
            "avg_episode_length": self.rolling_mean("num_steps"),
        }


# ---------------------------------------------------------------------------
# Logging formatters
# ---------------------------------------------------------------------------

def format_episode_log(
    episode_num: int,
    stats: Dict[str, Any],
    curriculum_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Format a single-line episode summary for console logging.

    Parameters
    ----------
    episode_num : int
        Global episode counter.
    stats : dict
        Episode summary (e.g. from ``BudgetMonitor``).
    curriculum_info : dict or None
        Curriculum summary dict (from ``BudgetCurriculum.get_stats_summary``).

    Returns
    -------
    str
        Human-readable log line.
    """
    parts = [
        f"ep={episode_num}",
        f"pop={stats.get('final_population', 0):.0f}",
        f"reward={stats.get('total_reward', 0):.2f}",
        f"funds={stats.get('final_funds', 0):.0f}",
        f"steps={stats.get('num_steps', 0)}",
        f"invalid={stats.get('invalid_action_count', 0)}",
    ]
    if stats.get("went_bankrupt"):
        parts.append("BANKRUPT")

    if curriculum_info is not None:
        parts.append(
            f"phase={curriculum_info.get('phase_name', '?')}"
            f"(budget={curriculum_info.get('budget', 0)})"
        )

    return " | ".join(parts)


def format_curriculum_log(curriculum_summary: Dict[str, Any]) -> str:
    """Format a multi-line curriculum status for periodic logging.

    Parameters
    ----------
    curriculum_summary : dict
        From ``BudgetCurriculum.get_stats_summary()``.
    """
    lines = [
        "--- Curriculum Status ---",
        f"  Phase: {curriculum_summary.get('phase_idx', 0)} "
        f"({curriculum_summary.get('phase_name', '?')})",
        f"  Budget: {curriculum_summary.get('budget', 0):,}",
        f"  Avg Population: {curriculum_summary.get('avg_pop', 0):.1f}",
        f"  Bankruptcy Rate: {curriculum_summary.get('bankruptcy_rate', 0):.1%}",
        f"  Avg Reward: {curriculum_summary.get('avg_reward', 0):.2f}",
        f"  Avg Invalid Actions: {curriculum_summary.get('avg_invalid_actions', 0):.1f}",
        f"  Buffer Size: {curriculum_summary.get('buffer_size', 0)}",
        "-------------------------",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build efficiency metrics
# ---------------------------------------------------------------------------

def compute_build_efficiency(
    final_population: float,
    total_spent: float,
) -> float:
    """Population per unit of money spent.

    Returns 0.0 if nothing was spent (avoids division by zero).
    """
    if total_spent <= 0:
        return 0.0
    return final_population / total_spent


def compute_tool_usage_summary(
    action_history: Sequence[int],
    num_tools: int,
    map_x: int,
    map_y: int,
) -> Dict[str, int]:
    """Count how many times each tool was used.

    Parameters
    ----------
    action_history : sequence of int
        Flat action indices from the episode.
    num_tools : int
        Number of tools in the environment.
    map_x, map_y : int
        Map dimensions.

    Returns
    -------
    dict
        Mapping from tool name to usage count.
    """
    cells_per_tool = map_x * map_y
    counts: Dict[str, int] = {name: 0 for name in TOOLS[:num_tools]}

    for action in action_history:
        tool_idx = action // cells_per_tool
        if 0 <= tool_idx < num_tools:
            counts[TOOLS[tool_idx]] += 1

    return counts


def estimate_total_cost(
    action_history: Sequence[int],
    num_tools: int,
    map_x: int,
    map_y: int,
) -> int:
    """Estimate total build cost from an action history.

    This does not account for failed builds (insufficient funds) -- it
    assumes every action was executed.
    """
    from .config import TOOL_COST_ARRAY

    cells_per_tool = map_x * map_y
    total = 0
    for action in action_history:
        tool_idx = action // cells_per_tool
        if 0 <= tool_idx < len(TOOL_COST_ARRAY):
            total += TOOL_COST_ARRAY[tool_idx]
    return total


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility across numpy and stdlib."""
    random.seed(seed)
    np.random.seed(seed)

    # Optional: torch seeding (only if torch is available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def flat_action_to_tuple(
    action: int,
    num_tools: int,
    map_x: int,
    map_y: int,
) -> tuple[int, int, int]:
    """Convert flat action index to (tool_idx, x, y) tuple.

    The flat index layout is: action = tool_idx * (map_x * map_y) + x * map_y + y
    """
    cells = map_x * map_y
    tool_idx = action // cells
    remainder = action % cells
    x = remainder // map_y
    y = remainder % map_y
    return (tool_idx, x, y)


def tuple_to_flat_action(
    tool_idx: int,
    x: int,
    y: int,
    map_x: int,
    map_y: int,
) -> int:
    """Convert (tool_idx, x, y) to flat action index."""
    return tool_idx * (map_x * map_y) + x * map_y + y
