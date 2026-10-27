"""Gymnasium environment: BeamEnv, reward, wrappers.

:class:`~flywaydqn.env.beam_env.BeamEnv` (spec Phase 3) wraps the same RF,
mobility and exposure primitives the demo twin uses, so a policy trained here
and a policy served by :mod:`flywaydqn.service.app` are scored on identical
physics. Rewards come from :mod:`flywaydqn.env.reward` (spec Section 4.4).
"""

from __future__ import annotations

from flywaydqn.env.beam_env import ENV_ID, BeamEnv, BeamEnvConfig, load_env_config
from flywaydqn.env.reward import make_reward_fn

__all__ = ["ENV_ID", "BeamEnv", "BeamEnvConfig", "load_env_config", "make_reward_fn"]
