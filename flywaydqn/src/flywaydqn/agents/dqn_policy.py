"""BeamPolicy adapter for a trained DQN checkpoint (WP-E).

Wraps :func:`flywaydqn.agents.policy_io.load_model` so the FastAPI runner can
treat a trained agent as one more
:class:`~flywaydqn.agents.baselines_demo.BeamPolicy` alongside the rule-based
baselines.

One real surface, no fabricated summaries:

1. **Decision** — ``act`` rebuilds the *training* observation from the live
   step: the runner's ``LiveState`` plus the last action the policy applied,
   assembled by :meth:`BeamEnv.observation_from`, which is the same feature
   construction, scaling and action decoding the trainer used. The loaded
   Q-network scores all actions on that observation and the argmax is served.
2. **What the frame carries** — the action's ``detail`` (rendered into the
   frame's ``explanation``) names the argmax action, its decoded beam, power
   and null flag, and the Q-value the running network produced for that exact
   observation. Nothing else is claimed: the per-action Q-vector is not on the
   schema-3 wire, and no placeholder stands in for it.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import torch

from flywaydqn.agents.baselines_demo import (
    BeamAction,
    BeamPolicy,
    LiveState,
    NullDecision,
    RadioSetup,
    link_budgets,
    peak_cell_geometry,
    sum_served_throughput_mbps,
    transmit_weights,
)
from flywaydqn.agents.policies import register_dqn
from flywaydqn.agents.policy_io import CheckpointError, load_meta
from flywaydqn.agents.policy_io import load_model as _load_model
from flywaydqn.env.beam_env import BeamEnv, BeamEnvConfig, DecodedAction
from flywaydqn.rf.array import composite_gain_dbi
from flywaydqn.rf.codebook import Beam
from flywaydqn.units import dbm_to_mw

_LOGGER = logging.getLogger(__name__)

CHECKPOINT_DIR_ENV_VAR: Final[str] = "FLYWAYDQN_CHECKPOINT_DIR"

# Default checkpoint served as the ``dqn`` controller: the first-pass run kept
# in the repo. ``FLYWAYDQN_CHECKPOINT_DIR`` points the service at another run.
_DEFAULT_CHECKPOINT: Final[tuple[str, ...]] = (
    "experiments",
    "checkpoints",
    "dqn_first_pass",
    "seed_1",
)


def default_checkpoint_dir() -> Path:
    """Directory of the checkpoint the ``dqn`` controller serves."""
    override = os.environ.get(CHECKPOINT_DIR_ENV_VAR)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3].joinpath(*_DEFAULT_CHECKPOINT)


class DQNPolicy(BeamPolicy):
    """Serve a trained Double DQN as a :class:`BeamPolicy`.

    The policy owns a :class:`BeamEnv` built from the checkpoint's recorded
    ``env_config`` (refused with :class:`CheckpointError` when the recorded
    spaces disagree), so feature scaling and action decoding are the trainer's
    own code. The scene itself comes from the runner through ``act``; the env
    is never free-run, so what the network sees always describes the frame the
    dashboard renders.
    """

    def __init__(self, checkpoint_dir: Path, *, env_config: BeamEnvConfig) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._meta = load_meta(checkpoint_dir)
        self._env = BeamEnv(env_config)
        obs_shape = self._env.observation_space.shape
        if obs_shape is None or obs_shape[0] != self._meta.observation_size:
            raise CheckpointError(
                f"checkpoint {checkpoint_dir} trained for obs="
                f"{self._meta.observation_size}; the serving env expects {obs_shape}"
            )
        if self._env.n_actions != self._meta.n_actions:
            raise CheckpointError(
                f"checkpoint {checkpoint_dir} trained for "
                f"actions={self._meta.n_actions}; the serving env expects "
                f"{self._env.n_actions}"
            )
        self._model = _load_model(
            checkpoint_dir,
            observation_size=self._meta.observation_size,
            n_actions=self._meta.n_actions,
        )
        # The previous action, carried across calls so the next observation can
        # condition on it exactly like BeamEnv.step does during training.
        self._prev_action: DecodedAction | None = None

    def act(self, state: LiveState, setup: RadioSetup) -> BeamAction:
        """Choose this step's action with the checkpoint's Q-network."""
        self._check_setup(setup)
        prev = self._prev_action
        if prev is None:
            # First frame mirrors BeamEnv.reset: beam 0 at the top power level
            # and no previous throughput to condition on (spec 4.3).
            levels = self._env._config.power_levels_dbm
            prev_beam_index = 0
            prev_power_index = len(levels) - 1
            prev_power_mw = dbm_to_mw(levels[prev_power_index])
            prev_weights = setup.codebook[0].weights
            prev_throughput_mbps = 0.0
        else:
            prev_beam_index = prev.beam_index
            prev_power_index = prev.power_index
            prev_power_mw = prev.power_mw
            prev_beam = setup.codebook[prev.beam_index]
            # Re-steer the previous null onto this step's footprint the same way
            # BeamEnv.step does before it assembles the next observation.
            prev_weights, _ = self._steered_weights(prev_beam, prev.null_active, state, setup)
            budgets = link_budgets(prev_weights, state, replace(setup, tx_power_mw=prev_power_mw))
            prev_throughput_mbps = sum_served_throughput_mbps(budgets)

        observation = self._env.observation_from(
            state,
            beam_index=prev_beam_index,
            power_index=prev_power_index,
            power_mw=prev_power_mw,
            weights=prev_weights,
            prev_throughput_mbps=prev_throughput_mbps,
            log10_w=state.log10_w,
        )
        q_values = self._q_values(observation)
        argmax = int(np.argmax(q_values))
        decoded = self._env.decode_action(np.int64(argmax))

        beam = setup.codebook[decoded.beam_index]
        weights, null_applied = self._steered_weights(beam, decoded.null_active, state, setup)
        self._prev_action = decoded

        level_db = self._env._config.power_levels_dbm[decoded.power_index]
        if decoded.null_active and not null_applied:
            null_note = "null rejected (non-finite gain)"
        else:
            null_note = "null on" if null_applied else "null off"
        detail = (
            f"DQN argmax: action {argmax} -> beam {decoded.beam_index} @ "
            f"{beam.azimuth_deg:.1f} deg, power {decoded.power_mw:.0f} mW "
            f"({level_db:.0f} dBm), {null_note}; "
            f"Q(argmax) = {float(q_values[argmax]):.3f}"
        )
        return BeamAction(
            beam=beam,
            weights=weights,
            null_active=null_applied,
            power_mw=decoded.power_mw,
            decision=(NullDecision.APPLIED if null_applied else NullDecision.NOT_APPLICABLE),
            detail=detail,
        )

    def _q_values(self, observation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Score every action for one observation (raw Q-network output)."""
        with torch.no_grad():
            q = self._model.q_net(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0))
        return np.asarray(q.squeeze(0).cpu().numpy(), dtype=np.float64)

    def _steered_weights(
        self,
        beam: Beam,
        null_active: bool,
        state: LiveState,
        setup: RadioSetup,
    ) -> tuple[npt.NDArray[np.complex128], bool]:
        """Weights one action transmits, with the env's guarded projection null.

        Returns ``(weights, null_applied)``: a projection that would radiate a
        non-finite gain is refused and the plain beam is sent, so ``null_active``
        on the wire never claims a null that is not on the air.
        """
        weights = beam.weights
        if not null_active:
            return weights, False
        peak = peak_cell_geometry(beam.weights, state.cells, setup)
        candidate = transmit_weights(
            beam,
            setup,
            null_elevation_deg=peak.elevation_deg,
            null_azimuth_deg=peak.bearing_deg,
        )
        finite = math.isfinite(
            composite_gain_dbi(
                candidate,
                beam.elevation_deg,
                beam.azimuth_deg,
                setup.n_rows,
                setup.n_cols,
                setup.d_lambda,
            )
        )
        if finite:
            return candidate, True
        return weights, False

    def _check_setup(self, setup: RadioSetup) -> None:
        """Refuse a scene the checkpoint was not trained for — loudly."""
        n_beams = len(self._env._setup.codebook)
        if len(setup.codebook) != n_beams:
            raise CheckpointError(
                f"the scene offers {len(setup.codebook)} beams; the checkpoint "
                f"was trained on {n_beams}. Serving it would mis-index every "
                "beam action."
            )
        top_level_mw = dbm_to_mw(max(self._env._config.power_levels_dbm))
        if not math.isclose(setup.tx_power_mw, top_level_mw, rel_tol=1e-9):
            raise CheckpointError(
                f"the scene transmits at {setup.tx_power_mw:.0f} mW; the "
                f"checkpoint's top power level is {top_level_mw:.0f} mW. The "
                "power grid the policy was trained on does not match the scene."
            )


def make_dqn_policy(checkpoint_dir: Path | str, *, env_config: BeamEnvConfig) -> DQNPolicy:
    """Factory for :class:`DQNPolicy` used by the policy registry (WP-E)."""
    return DQNPolicy(Path(checkpoint_dir), env_config=env_config)


def register_trained_controller() -> bool:
    """Offer the trained checkpoint as the ``dqn`` controller, by name.

    Called at service startup and by the baseline measurement script, so the
    controllers the service *serves* and the controllers an experiment
    *measures* are the same set by construction rather than two lists that can
    drift apart.

    Registration reads only the checkpoint's ``meta.json`` (cheap); the
    Q-network itself loads the first time something actually builds the policy.
    A checkout without the build artifact simply does not offer the controller
    -- and says so -- instead of listing a policy the service cannot serve.

    Returns
    -------
    bool
        ``True`` when the controller was registered, ``False`` when no
        readable checkpoint was found.
    """
    checkpoint = default_checkpoint_dir()
    try:
        meta = load_meta(checkpoint)
    except CheckpointError as exc:
        _LOGGER.warning("trained checkpoint unavailable (%s): controller 'dqn' not offered", exc)
        return False
    env_config = BeamEnvConfig.model_validate(meta.env_config)
    register_dqn(
        "dqn",
        "FlywayDQN (Trained)",
        (
            f"Double DQN checkpoint {meta.run_name} (seed {meta.seed}, "
            f"{meta.total_steps:,} steps): scored through the environment's own "
            "observation and action decoding."
        ),
        # The torch-heavy build runs only when something actually selects the
        # controller, never at registration time.
        lambda _cfg: make_dqn_policy(checkpoint, env_config=env_config),
    )
    return True


__all__ = [
    "CHECKPOINT_DIR_ENV_VAR",
    "DQNPolicy",
    "default_checkpoint_dir",
    "make_dqn_policy",
    "register_trained_controller",
]
