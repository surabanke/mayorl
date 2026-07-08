"""baselines.common.atari_wrappers shim.

mayorl never builds Atari envs, so these are inert stubs that exist only to
satisfy the top-level ``from baselines.common.atari_wrappers import ...`` in
the original envs.py.
"""


def make_atari(*args, **kwargs):  # pragma: no cover - never called by mayorl
    raise NotImplementedError(
        "baselines.common.atari_wrappers.make_atari is a stub (mayorl shim); "
        "Atari envs are not used in this project."
    )


def wrap_deepmind(*args, **kwargs):  # pragma: no cover - never called by mayorl
    raise NotImplementedError(
        "baselines.common.atari_wrappers.wrap_deepmind is a stub (mayorl shim); "
        "Atari envs are not used in this project."
    )
