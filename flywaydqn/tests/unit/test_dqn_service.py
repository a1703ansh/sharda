"""WP-E: the trained checkpoint served as the ``dqn`` controller.

These tests pin the seam the dashboard's fifth button depends on: starting the
service registers ``dqn``, a session selecting it streams frames whose values
all come from the running checkpoint, and the metadata route reports exactly
what ``meta.json`` records.

The checkpoint under ``experiments/checkpoints/`` is a build artifact (see
``.gitignore``), so every test here skips with a clear reason when it is
absent instead of failing on machines that never trained.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

from flywaydqn.agents.policies import available_policies, available_policy_names, build_policy
from flywaydqn.agents.policy_io import load_meta
from flywaydqn.demo import DemoRunner, load_scenario
from flywaydqn.env.beam_env import BeamEnv, BeamEnvConfig
from flywaydqn.service.app import create_app
from flywaydqn.service.schemas import ScenarioRequest

CHECKPOINT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "checkpoints"
    / "dqn_first_pass"
    / "seed_1"
)
requires_checkpoint = pytest.mark.skipif(
    not (CHECKPOINT / "meta.json").is_file(),
    reason="trained checkpoint not present (experiments/checkpoints is a build artifact)",
)


def _request(*, policy: str | None = None, seed: int = 3) -> ScenarioRequest:
    return ScenarioRequest(key="scenario_001", seed=seed, weight=1.0, policy=policy)


def _floats(node: object) -> Iterator[float]:
    if isinstance(node, float):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _floats(value)
    elif isinstance(node, list):
        for value in node:
            yield from _floats(value)


@requires_checkpoint
def test_service_startup_registers_the_trained_controller() -> None:
    """``create_app`` offers ``dqn`` exactly once the checkpoint is readable."""
    create_app()
    names = available_policy_names()
    assert "dqn" in names, f"service startup did not register dqn: {names}"
    spec = next(spec for spec in available_policies() if spec.name == "dqn")
    assert spec.label == "FlywayDQN (Trained)"
    # Rebuilding the app must not duplicate or lose entries.
    create_app()
    assert available_policy_names() == names


@requires_checkpoint
def test_registered_dqn_builds_through_the_checkpoint() -> None:
    create_app()
    policy = build_policy("dqn", load_scenario("scenario_001").policy)
    # Building it proves the meta/space check passed; acting on a live state
    # proves the Q-network path works (covered frame-by-frame below).
    assert policy.__class__.__name__ == "DQNPolicy"


@requires_checkpoint
def test_dqn_session_serves_finite_frames_from_the_network() -> None:
    """A ``dqn`` session produces wire frames the UI can parse, no NaN/inf."""
    runner = DemoRunner(_request(policy="dqn"))
    frames = [runner.initial_frame()] + [runner.step() for _ in range(40)]
    for frame in frames:
        assert frame.policy_name == "dqn"
        assert frame.explanation is not None
        assert "DQN argmax" in frame.explanation
        assert "Q(argmax)" in frame.explanation
        dumped = frame.model_dump(mode="json")
        assert all(math.isfinite(value) for value in _floats(dumped))
        assert frame.inference_latency_ms >= 0.0
    # The decoded actions are real: beams within the codebook, powers on the
    # checkpoint's own grid.
    assert all(0 <= frame.beam.index < 16 for frame in frames)
    # The checkpoint's power grid: 44/47/50/53 dBm in mW.
    assert {round(frame.power_mw) for frame in frames} <= {25119, 50119, 100000, 199526}


@requires_checkpoint
def test_checkpoint_meta_describes_the_serving_env() -> None:
    """The recorded spaces are the spaces the env actually exposes."""
    meta = load_meta(CHECKPOINT)
    assert meta.total_steps == 40000
    assert meta.observation_size == 44
    assert meta.n_actions == 128
    assert meta.run_name == "dqn_first_pass"
    env = BeamEnv(BeamEnvConfig.model_validate(meta.env_config))
    assert env.observation_space.shape == (meta.observation_size,)
    assert env.n_actions == meta.n_actions


def test_metadata_route_is_registered_on_the_app() -> None:
    app = create_app()
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/policies/dqn/meta" in paths
    assert "/policies" in paths
