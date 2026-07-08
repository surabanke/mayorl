"""baselines.common.vec_env shim -> stable-baselines3 vec env primitives."""

from stable_baselines3.common.vec_env import (  # noqa: F401
    VecEnv,
    VecEnvWrapper,
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

# CloudpickleWrapper lives in base_vec_env in sb3.
try:
    from stable_baselines3.common.vec_env.base_vec_env import (  # noqa: F401
        CloudpickleWrapper,
    )
except Exception:  # pragma: no cover - fallback for layout differences
    from stable_baselines3.common.vec_env import CloudpickleWrapper  # noqa: F401
