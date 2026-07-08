"""mayorl/curriculum.py -- Budget curriculum for progressive difficulty.

The idea: start training with a generous budget so the agent can learn
spatial city-building patterns (roads, power, zoning) without worrying
about money.  Then progressively tighten the budget so the agent also
learns fiscal discipline.

Phase promotion is decided by a rolling window of episode statistics.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from .config import CurriculumPhase, MayorlConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Episode statistics (lightweight record kept per episode)
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStats:
    """Summary of a completed episode, used by the curriculum scheduler."""

    final_population: float = 0.0
    went_bankrupt: bool = False
    final_funds: float = 0.0
    num_steps: int = 0
    total_reward: float = 0.0
    invalid_action_count: int = 0


# ---------------------------------------------------------------------------
# Curriculum scheduler
# ---------------------------------------------------------------------------

class BudgetCurriculum:
    """Manages progressive budget reduction across training.

    Parameters
    ----------
    config : MayorlConfig
        Central configuration.  ``curriculum_phases``, ``curriculum_window``,
        and ``pop_target`` are read from here.

    Usage
    -----
    >>> curriculum = BudgetCurriculum(config)
    >>> budget = curriculum.current_budget  # use for env reset
    >>> # ... run episode ...
    >>> curriculum.report_episode(stats)
    >>> if curriculum.check_promotion():
    ...     new_budget = curriculum.current_budget
    """

    def __init__(self, config: Optional[MayorlConfig] = None) -> None:
        if config is None:
            config = MayorlConfig()

        self._phases: List[CurriculumPhase] = list(config.curriculum_phases)
        self._window_size: int = config.curriculum_window
        self._pop_target: float = config.pop_target

        self._current_phase_idx: int = 0
        self._buffer: Deque[EpisodeStats] = deque(maxlen=self._window_size)
        self._total_episodes: int = 0

        if not self._phases:
            raise ValueError("curriculum_phases must contain at least one phase")

    # -- Properties --------------------------------------------------------

    @property
    def current_phase(self) -> CurriculumPhase:
        return self._phases[self._current_phase_idx]

    @property
    def current_phase_index(self) -> int:
        return self._current_phase_idx

    @property
    def current_budget(self) -> int:
        return self.current_phase.init_budget

    @property
    def num_phases(self) -> int:
        return len(self._phases)

    @property
    def is_final_phase(self) -> bool:
        return self._current_phase_idx >= len(self._phases) - 1

    @property
    def total_episodes(self) -> int:
        return self._total_episodes

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    # -- Core API ----------------------------------------------------------

    def report_episode(self, stats: EpisodeStats) -> None:
        """Record the outcome of one completed episode."""
        self._buffer.append(stats)
        self._total_episodes += 1

    def check_promotion(self) -> bool:
        """Evaluate whether the agent should graduate to the next phase.

        Returns ``True`` if a promotion just happened (and updates internal
        state), ``False`` otherwise.
        """
        if self.is_final_phase:
            return False

        if len(self._buffer) < self._window_size:
            return False

        phase = self.current_phase

        # Average population over the window
        avg_pop = sum(s.final_population for s in self._buffer) / len(
            self._buffer
        )
        pop_ok = avg_pop >= self._pop_target * phase.pop_threshold_ratio

        # Bankruptcy rate
        if phase.max_bankruptcy_rate is not None:
            bankruptcy_rate = sum(
                1 for s in self._buffer if s.went_bankrupt
            ) / len(self._buffer)
            bankruptcy_ok = bankruptcy_rate < phase.max_bankruptcy_rate
        else:
            bankruptcy_ok = True

        if pop_ok and bankruptcy_ok:
            self._promote()
            return True

        return False

    def reset(self) -> None:
        """Reset curriculum to phase 0 and clear the stats buffer."""
        self._current_phase_idx = 0
        self._buffer.clear()
        self._total_episodes = 0

    def force_phase(self, phase_idx: int) -> None:
        """Manually set the curriculum phase (for evaluation / debugging)."""
        if phase_idx < 0 or phase_idx >= len(self._phases):
            raise IndexError(
                f"phase_idx {phase_idx} out of range [0, {len(self._phases)})"
            )
        self._current_phase_idx = phase_idx
        self._buffer.clear()
        logger.info(
            "Curriculum forced to phase %d (%s), budget=%d",
            phase_idx,
            self.current_phase.name,
            self.current_budget,
        )

    # -- Logging helpers ---------------------------------------------------

    def get_stats_summary(self) -> dict:
        """Return a dict of rolling-window statistics for logging."""
        if not self._buffer:
            return {
                "phase_idx": self._current_phase_idx,
                "phase_name": self.current_phase.name,
                "budget": self.current_budget,
                "avg_pop": 0.0,
                "bankruptcy_rate": 0.0,
                "avg_reward": 0.0,
                "avg_invalid_actions": 0.0,
                "buffer_size": 0,
            }

        n = len(self._buffer)
        return {
            "phase_idx": self._current_phase_idx,
            "phase_name": self.current_phase.name,
            "budget": self.current_budget,
            "avg_pop": sum(s.final_population for s in self._buffer) / n,
            "bankruptcy_rate": sum(
                1 for s in self._buffer if s.went_bankrupt
            ) / n,
            "avg_reward": sum(s.total_reward for s in self._buffer) / n,
            "avg_invalid_actions": sum(
                s.invalid_action_count for s in self._buffer
            ) / n,
            "buffer_size": n,
        }

    # -- Internal ----------------------------------------------------------

    def _promote(self) -> None:
        old_phase = self.current_phase
        self._current_phase_idx += 1
        new_phase = self.current_phase
        self._buffer.clear()
        logger.info(
            "Curriculum promotion: %s (budget=%d) -> %s (budget=%d) "
            "after %d total episodes",
            old_phase.name,
            old_phase.init_budget,
            new_phase.name,
            new_phase.init_budget,
            self._total_episodes,
        )
