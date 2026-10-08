"""Gymnasium environment: BeamEnv, reward, wrappers.

``BeamEnv`` lands in Phase 3. Until then it is resolved lazily through
``importlib``, so this package imports and typechecks without a partially-built
environment module.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from flywaydqn.env.reward import make_reward_fn

__all__ = ["BeamEnv", "make_reward_fn"]


def __getattr__(name: str) -> Any:
    """Resolve ``BeamEnv`` from :mod:`flywaydqn.env.beam_env` (Phase 3)."""
    if name == "BeamEnv":
        module = import_module("flywaydqn.env.beam_env")
        return module.BeamEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
