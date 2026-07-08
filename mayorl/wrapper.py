"""mayorl/wrapper.py -- Observation and action wrappers for BudgetCityEnv.

Provides:
  - ActionMaskWrapper: exposes ``action_masks()`` for masked policy sampling
  - ObservationNormWrapper: running-mean observation normalization
  - BudgetMonitor: episode-level logging of budget/city statistics
  - make_mayorl_env: factory that composes env + wrappers in the right order
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np

from .config import MayorlConfig
from .curriculum import BudgetCurriculum, EpisodeStats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action masking wrapper
# ---------------------------------------------------------------------------

class ActionMaskWrapper(gym.Wrapper):
    """Exposes ``action_masks()`` and optionally replaces invalid actions.

    Many masked-action PPO implementations (e.g. sb3-contrib MaskablePPO)
    expect the environment to provide an ``action_masks()`` method returning
    a boolean array.  This wrapper delegates to ``BudgetCityEnv.action_masks``
    and, as a safety net, replaces any invalid action with the Nil (no-op)
    tool at position (0, 0) if the policy somehow selects a masked action.

    Parameters
    ----------
    env : gym.Env
        Must be a ``BudgetCityEnv`` (or something with ``action_masks``).
    replace_invalid : bool
        If ``True``, silently replace invalid actions with no-op instead of
        forwarding them (the env already handles this, but the wrapper
        provides an extra safety layer).
    """

    def __init__(self, env: gym.Env, replace_invalid: bool = True) -> None:
        super().__init__(env)
        self._replace_invalid = replace_invalid

        # Pre-compute the no-op action index (Nil tool at position 0,0)
        # Nil is the last tool in the canonical ordering.
        unwrapped = self.unwrapped
        num_tools = getattr(unwrapped, "num_tools", 0)
        map_x = getattr(unwrapped, "MAP_X", 1)
        map_y = getattr(unwrapped, "MAP_Y", 1)
        nil_tool_idx = num_tools - 1  # Nil is last in TOOLS list
        self._noop_action: int = nil_tool_idx * map_x * map_y  # (nil, 0, 0)

    # -- gym.Wrapper interface ------------------------------------------------

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self._replace_invalid:
            masks = self.action_masks()
            if not masks[action]:
                action = self._noop_action
        return self.env.step(action)

    def reset(self, **kwargs: Any) -> np.ndarray:
        return self.env.reset(**kwargs)

    # -- Mask API -------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Boolean mask over the flat action space. True = valid."""
        return self.env.action_masks()


# ---------------------------------------------------------------------------
# Running-mean observation normalisation
# ---------------------------------------------------------------------------

class ObservationNormWrapper(gym.ObservationWrapper):
    """Per-channel running-mean/std normalisation for spatial observations.

    Maintains an exponential moving average of channel means and variances
    and normalises observations to approximately zero-mean, unit-variance.

    Parameters
    ----------
    env : gym.Env
        Wrapped environment whose observation is (C, H, W).
    clip : float
        Clamp normalised values to [-clip, clip].
    epsilon : float
        Small constant to avoid division by zero.
    """

    def __init__(
        self,
        env: gym.Env,
        clip: float = 10.0,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(env)
        obs_shape = self.observation_space.shape  # (C, H, W)
        num_channels = obs_shape[0]

        self._clip = clip
        self._epsilon = epsilon
        self._count: int = 0

        # Per-channel running statistics
        self._mean = np.zeros(num_channels, dtype=np.float64)
        self._var = np.ones(num_channels, dtype=np.float64)

        # Update observation space bounds to reflect normalisation range
        low = np.full(obs_shape, -clip, dtype=np.float32)
        high = np.full(obs_shape, clip, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._update_stats(obs)
        return self._normalise(obs)

    def _update_stats(self, obs: np.ndarray) -> None:
        """Update running mean/var with Welford's online algorithm."""
        # obs shape: (C, H, W) -- compute per-channel mean
        channel_means = obs.mean(axis=(1, 2))  # shape (C,)
        self._count += 1
        delta = channel_means - self._mean
        self._mean += delta / self._count
        delta2 = channel_means - self._mean
        self._var += (delta * delta2 - self._var) / self._count

    def _normalise(self, obs: np.ndarray) -> np.ndarray:
        std = np.sqrt(self._var + self._epsilon)
        # Broadcast (C,) over (C, H, W)
        normed = (obs - self._mean[:, None, None]) / std[:, None, None]
        return np.clip(normed, -self._clip, self._clip).astype(np.float32)


# ---------------------------------------------------------------------------
# Budget monitor wrapper
# ---------------------------------------------------------------------------

class BudgetMonitor(gym.Wrapper):
    """Tracks per-episode budget statistics and reports to curriculum.

    On episode termination, constructs an ``EpisodeStats`` from the info
    dict and (optionally) forwards it to a ``BudgetCurriculum`` instance.

    Parameters
    ----------
    env : gym.Env
        Must be a ``BudgetCityEnv`` (or wrapped version thereof).
    curriculum : BudgetCurriculum or None
        If provided, ``report_episode`` is called automatically on
        episode end.  The curriculum's ``current_budget`` is also used
        to update the env budget on each reset.
    """

    def __init__(
        self,
        env: gym.Env,
        curriculum: Optional[BudgetCurriculum] = None,
    ) -> None:
        super().__init__(env)
        self._curriculum = curriculum

        # Accumulator for the current episode
        self._episode_reward: float = 0.0
        self._episode_steps: int = 0
        self._invalid_actions: int = 0

        # History of completed episodes (lightweight summary list)
        self._episode_history: list[Dict[str, Any]] = []

    # -- gym.Wrapper interface ------------------------------------------------

    def reset(self, **kwargs: Any) -> np.ndarray:
        # Apply curriculum budget before reset
        if self._curriculum is not None:
            self.unwrapped.set_budget(self._curriculum.current_budget)

        obs = self.env.reset(**kwargs)

        self._episode_reward = 0.0
        self._episode_steps = 0
        self._invalid_actions = 0
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)

        self._episode_reward += reward
        self._episode_steps += 1
        if info.get("failed_build", False):
            self._invalid_actions += 1

        if done:
            episode_stats = self._build_episode_stats(info)
            info["budget_monitor"] = self._stats_to_dict(episode_stats)

            # Report to curriculum
            if self._curriculum is not None:
                self._curriculum.report_episode(episode_stats)
                promoted = self._curriculum.check_promotion()
                info["curriculum_promoted"] = promoted
                info["curriculum_summary"] = self._curriculum.get_stats_summary()

            self._episode_history.append(info["budget_monitor"])

        return obs, reward, done, info

    # -- Public helpers -------------------------------------------------------

    @property
    def episode_history(self) -> list[Dict[str, Any]]:
        """List of per-episode summary dicts."""
        return self._episode_history

    @property
    def num_episodes(self) -> int:
        return len(self._episode_history)

    # -- Internal -------------------------------------------------------------

    def _build_episode_stats(self, info: Dict[str, Any]) -> EpisodeStats:
        """Construct an ``EpisodeStats`` from terminal info dict."""
        ep = info.get("episode_stats", {})

        # Total population from city metrics
        unwrapped = self.unwrapped
        metrics = getattr(unwrapped, "city_metrics", {})
        final_pop = (
            metrics.get("res_pop", 0.0)
            + metrics.get("com_pop", 0.0)
            + metrics.get("ind_pop", 0.0)
        )

        return EpisodeStats(
            final_population=final_pop,
            went_bankrupt=info.get("bankrupt", False),
            final_funds=info.get("funds", 0.0),
            num_steps=self._episode_steps,
            total_reward=self._episode_reward,
            invalid_action_count=self._invalid_actions,
        )

    @staticmethod
    def _stats_to_dict(stats: EpisodeStats) -> Dict[str, Any]:
        return {
            "final_population": stats.final_population,
            "went_bankrupt": stats.went_bankrupt,
            "final_funds": stats.final_funds,
            "num_steps": stats.num_steps,
            "total_reward": stats.total_reward,
            "invalid_action_count": stats.invalid_action_count,
        }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def make_mayorl_env(
    config: Optional[MayorlConfig] = None,
    curriculum: Optional[BudgetCurriculum] = None,
    *,
    action_mask: bool = True,
    obs_norm: bool = False,
    obs_clip: float = 10.0,
    replace_invalid: bool = True,
    rank: int = 0,
) -> gym.Env:
    """Create a fully-wrapped BudgetCityEnv.

    Wrapper ordering (inside-out):
        BudgetCityEnv -> BudgetMonitor -> ActionMaskWrapper [-> ObservationNormWrapper]

    Parameters
    ----------
    config : MayorlConfig or None
        Environment configuration. Defaults to ``MayorlConfig()``.
    curriculum : BudgetCurriculum or None
        Optional curriculum scheduler for progressive budget reduction.
    action_mask : bool
        Whether to apply the ``ActionMaskWrapper``.
    obs_norm : bool
        Whether to apply running observation normalisation.
    obs_clip : float
        Clipping range for observation normalisation.
    replace_invalid : bool
        Whether ``ActionMaskWrapper`` should replace invalid actions with
        no-op.
    rank : int
        Environment rank (for parallel environments).

    Returns
    -------
    gym.Env
        The composed environment with all requested wrappers.
    """
    from .env import BudgetCityEnv

    if config is None:
        config = MayorlConfig()

    env = BudgetCityEnv(
        MAP_X=config.map_x,
        MAP_Y=config.map_y,
        PADDING=config.padding,
        config=config,
    )

    # The env needs setMapSize to fully initialise (mirrors original pattern)
    env.setMapSize(
        (config.map_x, config.map_y),
        rank=rank,
        max_step=config.effective_max_steps,
        empty_start=config.empty_start,
        render_gui=config.render_gui,
        power_puzzle=config.power_puzzle,
        random_builds=config.random_builds,
    )

    # Budget monitor (always on -- lightweight)
    env = BudgetMonitor(env, curriculum=curriculum)

    # Action masking
    if action_mask:
        env = ActionMaskWrapper(env, replace_invalid=replace_invalid)

    # Observation normalisation
    if obs_norm:
        env = ObservationNormWrapper(env, clip=obs_clip)

    return env
