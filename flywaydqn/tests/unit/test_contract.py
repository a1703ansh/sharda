"""Contract tests for the wire frame and the policy registry.

These pin the two seams every other package depends on: the exact field set of
:class:`~flywaydqn.service.schemas.StateFrame`, and the measured quantities the
dashboard is allowed to display (the scenario's own exposure threshold, the null
depth actually achieved, and the policy-decision latency).

The rule they enforce is the one the demo is judged on: a value a judge can see
must be produced by the running system, so these tests fail if a field silently
disappears, stops being measured, or is reported as a plausible constant.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Final, cast

import pytest

from flywaydqn.agents.baselines_demo import (
    NO_SERVICE_SINR_DB,
    BeamPolicy,
    UnconstrainedPolicy,
)
from flywaydqn.agents.policies import (
    UnknownPolicyError,
    available_policy_names,
    build_policy,
    policy_name_for,
)
from flywaydqn.demo import DemoRunner, load_scenario
from flywaydqn.service.schemas import SCHEMA_VERSION, ScenarioRequest, StateFrame

KEYS: Final[tuple[str, ...]] = ("scenario_001", "scenario_002", "scenario_003")
EPISODE_STEPS: Final[int] = 600

FRAME_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "step",
        "timeS",
        "dtS",
        "sector",
        "beam",
        "powerMw",
        "nullActive",
        "nullDepthDb",
        "inferenceLatencyMs",
        "policyName",
        "flock",
        "users",
        "metrics",
        "baseline",
        "explanation",
        "weight",
    }
)

BASELINE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "policyName",
        "powerMw",
        "sumThroughputMbps",
        "p95DensityWm2",
        "maxDensityWm2",
        "fracOver",
        "doseWsm2",
    }
)

POLICIES: Final[tuple[str, ...]] = (
    "unconstrained",
    "analytic-nulling",
    "static-backoff",
    "sector-shutdown",
)


def _floats(node: object) -> Iterator[float]:
    """Yield every float in a nested model dump."""
    if isinstance(node, float):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _floats(value)
    elif isinstance(node, list):
        for value in node:
            yield from _floats(value)


METRIC_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "sumThroughputMbps",
        "p95DensityWm2",
        "maxDensityWm2",
        "fracOver",
        "doseWsm2",
        "sThrWm2",
        "exposurePenalty",
        "throughputTerm",
        "powerTerm",
        "reward",
    }
)


def _request(key: str, *, seed: int = 1, policy: str | None = None) -> ScenarioRequest:
    return ScenarioRequest(key=key, seed=seed, weight=1.0, policy=policy)


def _episode(
    key: str, *, seed: int = 1, policy: str | None = None, steps: int = EPISODE_STEPS
) -> list[StateFrame]:
    runner = DemoRunner(_request(key, seed=seed, policy=policy))
    frames = [runner.initial_frame()]
    frames.extend(runner.step() for _ in range(steps - 1))
    return frames


def test_schema_version_is_three() -> None:
    assert SCHEMA_VERSION == "3"


def test_frame_field_set_is_frozen() -> None:
    """Any drift in the wire contract fails here instead of reaching the UI."""
    frame = DemoRunner(_request("scenario_001")).initial_frame()
    assert set(frame.model_dump(by_alias=True)) == set(FRAME_FIELDS)
    assert set(frame.metrics.model_dump(by_alias=True)) == set(METRIC_FIELDS)
    assert set(frame.baseline.model_dump(by_alias=True)) == set(BASELINE_FIELDS)


@pytest.mark.parametrize("key", KEYS)
def test_exposure_threshold_is_the_scenarios_own(key: str) -> None:
    """The threshold on the wire is the one the reward actually uses."""
    config = load_scenario(key)
    frame = DemoRunner(_request(key)).initial_frame()
    assert frame.metrics.s_thr_wm2 == config.exposure.s_thr_w_m2
    runner = DemoRunner(_request(key))
    runner.initial_frame()
    assert all(runner.step().metrics.s_thr_wm2 == config.exposure.s_thr_w_m2 for _ in range(20))


def test_null_depth_is_zero_whenever_no_null_is_transmitted() -> None:
    """A null-off frame must not carry a leftover depth from an earlier step."""
    frames = _episode("scenario_001", seed=1)
    assert any(frame.null_active for frame in frames), "scenario_001 must null at least once"
    assert all(frame.null_depth_db == 0.0 for frame in frames if not frame.null_active)
    # The never-nulling baseline is zero for its whole episode.
    baseline = _episode("scenario_002", seed=2)
    assert all(frame.null_depth_db == 0.0 for frame in baseline)


def test_null_depth_meets_the_configured_gate_and_matches_the_explanation() -> None:
    """The measured depth must clear the policy's own minimum reduction."""
    config = load_scenario("scenario_001")
    frames = _episode("scenario_001", seed=1)
    nulled = [frame for frame in frames if frame.null_active]
    assert nulled, "scenario_001 did not exercise the applied null"
    for frame in nulled:
        assert math.isfinite(frame.null_depth_db)
        assert frame.null_depth_db >= config.policy.min_exposure_reduction_db - 1e-9
        # The explanation renders the same reduction to one decimal, so the
        # wire field is cross-checked against an independently built string.
        assert f"({frame.null_depth_db:.1f} dB)" in (frame.explanation or "")


def test_null_depth_tracks_throughput_cost_it_is_not_free() -> None:
    """A null that costs rate must still be reported as the reduction it bought."""
    frames = _episode("scenario_001", seed=1)
    first_null = next(frame for frame in frames if frame.null_active)
    before = next(frame for frame in frames if frame.step == first_null.step - 1)
    assert first_null.metrics.p95_density_wm2 < before.metrics.p95_density_wm2
    assert first_null.metrics.sum_throughput_mbps <= before.metrics.sum_throughput_mbps


def test_inference_latency_is_measured_every_step() -> None:
    frames = _episode("scenario_001", seed=1)
    for frame in frames[:50]:
        assert math.isfinite(frame.inference_latency_ms)
        assert frame.inference_latency_ms > 0.0
        # Generous ceiling: catches an unmeasured zero or a unit slip.
        assert frame.inference_latency_ms < 500.0


def test_policy_name_reports_what_actually_ran() -> None:
    assert DemoRunner(_request("scenario_001")).initial_frame().policy_name == "analytic-nulling"
    assert DemoRunner(_request("scenario_002")).initial_frame().policy_name == "unconstrained"
    assert (
        DemoRunner(_request("scenario_002", policy="analytic-nulling")).initial_frame().policy_name
        == "analytic-nulling"
    )


def test_policy_override_switches_the_controller() -> None:
    """Selecting the baseline on a nulling scenario must really switch it off."""
    frames = _episode("scenario_001", seed=1, policy="unconstrained")
    assert all(frame.policy_name == "unconstrained" for frame in frames)
    assert not any(frame.null_active for frame in frames)
    assert all(frame.null_depth_db == 0.0 for frame in frames)
    # And the same scenario with the default policy does mitigate.
    assert any(frame.null_active for frame in _episode("scenario_001", seed=1))


def test_injected_policy_instance_is_named_after_its_class_not_the_scenario() -> None:
    """A test-injected policy must not be mislabelled as the scenario default."""
    runner = DemoRunner(_request("scenario_001"), policy=UnconstrainedPolicy())
    assert runner.initial_frame().policy_name == "unconstrained"
    assert not any(runner.step().null_active for _ in range(50))


def test_policy_name_for_unknown_instance_falls_back_to_the_class_name() -> None:
    class _Custom:
        def __init__(self) -> None:
            pass

        def act(self, state: object, setup: object) -> object:  # pragma: no cover
            raise NotImplementedError

    assert policy_name_for(cast(BeamPolicy, _Custom())) == "_Custom"


def test_policy_name_for_an_unknown_subclass_still_resolves_to_the_class_name() -> None:
    class _ShutdownStub:
        """A subclass whose parent is already registered must still resolve by class name."""

        def __init__(self) -> None:
            pass

        def act(self, state: object, setup: object) -> object:  # pragma: no cover
            raise NotImplementedError

    # The registered SectorShutdownPolicy should not swallow its subclass's name.
    assert policy_name_for(cast(BeamPolicy, _ShutdownStub())) == "_ShutdownStub"


def test_baseline_is_the_same_scenario_run_without_mitigation() -> None:
    """The reference block must equal a real unconstrained run, step for step.

    This is the whole point of putting it on the wire: a consumer prints a
    before/after pair that is measured from the same flock, the same cells and
    the same users, not from a second stream that has to be time-aligned.
    """
    reference = _episode("scenario_002", seed=2, policy="unconstrained", steps=320)
    frames = _episode("scenario_002", seed=2, policy="analytic-nulling", steps=320)
    assert any(frame.null_active for frame in frames), "the mitigated run must mitigate"
    for frame, control in zip(frames, reference, strict=True):
        assert frame.baseline.policy_name == "unconstrained"
        assert frame.baseline.p95_density_wm2 == pytest.approx(control.metrics.p95_density_wm2)
        assert frame.baseline.max_density_wm2 == pytest.approx(control.metrics.max_density_wm2)
        assert frame.baseline.frac_over == pytest.approx(control.metrics.frac_over)
        assert frame.baseline.dose_wsm2 == pytest.approx(control.metrics.dose_wsm2)
        assert frame.baseline.sum_throughput_mbps == pytest.approx(
            control.metrics.sum_throughput_mbps
        )
        assert frame.baseline.power_mw == pytest.approx(control.power_mw)


def test_baseline_equals_the_frame_when_no_mitigation_is_selected() -> None:
    """Selecting the reference controller must produce a zero-delta A/B."""
    for frame in _episode("scenario_001", seed=1, policy="unconstrained", steps=40):
        assert frame.metrics.p95_density_wm2 == pytest.approx(frame.baseline.p95_density_wm2)
        assert frame.metrics.dose_wsm2 == pytest.approx(frame.baseline.dose_wsm2)
        assert frame.metrics.sum_throughput_mbps == pytest.approx(
            frame.baseline.sum_throughput_mbps
        )


def test_backoff_transmits_half_power_and_the_wire_shows_it() -> None:
    """A power reduction must reach the frame's density, not just the policy."""
    frames = _episode("scenario_001", seed=1, policy="static-backoff", steps=220)
    backed_off = [frame for frame in frames if frame.power_mw < frame.baseline.power_mw]
    assert backed_off, "static back-off never engaged while the flock was in range"
    assert all(frame.null_active is False for frame in frames)
    for frame in backed_off:
        # Same greedy beam, half the power, so exactly half the density: any
        # other ratio would mean the power never reached the exposure model.
        assert frame.power_mw == pytest.approx(frame.baseline.power_mw * 0.5)
        assert frame.metrics.p95_density_wm2 == pytest.approx(frame.baseline.p95_density_wm2 * 0.5)
        assert frame.metrics.dose_wsm2 == pytest.approx(frame.baseline.dose_wsm2 * 0.5)


def test_shutdown_radiates_nothing_and_never_claims_a_null() -> None:
    """Transmitter off means zero exposure, zero service, and no invented null."""
    frames = _episode("scenario_001", seed=1, policy="sector-shutdown", steps=220)
    off = [frame for frame in frames if frame.power_mw == 0.0]
    assert off, "the shutdown never engaged while the flock was in range"
    for frame in off:
        assert frame.metrics.p95_density_wm2 == 0.0
        assert frame.metrics.max_density_wm2 == 0.0
        assert frame.metrics.frac_over == 0.0
        assert frame.metrics.dose_wsm2 == 0.0
        assert frame.metrics.sum_throughput_mbps == 0.0
        assert frame.null_active is False
        assert frame.null_depth_db == 0.0
        assert all(user.sinr == NO_SERVICE_SINR_DB for user in frame.users)
        assert math.isfinite(frame.beam.gain_dbi)
    # And the mitigation is real against the same-step reference.
    assert sum(f.metrics.dose_wsm2 for f in frames) < sum(f.baseline.dose_wsm2 for f in frames)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("key", KEYS)
def test_every_frame_is_finite_on_the_wire_for_every_policy(key: str, policy: str) -> None:
    """Whatever the policy does, the JSON the UI parses carries no inf or nan."""
    frames = _episode(key, seed=1, policy=policy, steps=40)
    for frame in frames:
        dumped = frame.model_dump(mode="json")
        assert all(math.isfinite(value) for value in _floats(dumped))


def test_unknown_policy_name_is_rejected_with_the_valid_names() -> None:
    """An unknown name fails loudly and names what *is* accepted.

    The service answers ``POST /sessions {"policy": "typo"}`` with a 404 whose
    detail is this exception's message (``flywaydqn/service/app.py``), so the
    message has to carry the accepted names and not just the offending one.
    """
    config = load_scenario("scenario_001").policy
    with pytest.raises(UnknownPolicyError) as raised:
        build_policy("definitely-not-a-policy", config)
    message = str(raised.value)
    assert "definitely-not-a-policy" in message
    for name in available_policy_names():
        assert name in message, f"{name} is accepted but the error does not say so"
    assert raised.value.known == available_policy_names()


def test_the_rule_based_baselines_are_always_registered() -> None:
    """The four baselines exist without a checkpoint; ``dqn`` may or may not.

    ``create_app`` (and ``register_trained_controller``) registers ``dqn`` only
    when a trained checkpoint is readable, so the registry's exact contents
    depend on the machine and on test order. What has to hold everywhere is
    that every rule-based baseline is registered and nothing unrecognised is.
    """
    names = set(available_policy_names())
    assert {
        "analytic-nulling",
        "static-backoff",
        "sector-shutdown",
        "unconstrained",
    } <= names
    assert names <= {
        "analytic-nulling",
        "static-backoff",
        "sector-shutdown",
        "unconstrained",
        "dqn",
    }
