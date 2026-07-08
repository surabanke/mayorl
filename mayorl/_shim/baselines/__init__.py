"""Minimal `baselines` shim.

The original ralph-city code (envs.py, dummy_vec_env.py, subproc_vec_env.py)
imports a handful of symbols from OpenAI ``baselines`` at module-load time.
mayorl never actually *calls* these (it uses its own stable-baselines3 vector
envs), but the imports and a couple of class definitions must succeed.

This shim re-exports equivalent functionality from stable-baselines3, which is
already installed, so we avoid pulling in the heavy (and py3.10-incompatible)
real ``baselines`` + TensorFlow stack.
"""
