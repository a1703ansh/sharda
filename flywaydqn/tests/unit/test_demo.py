"""Tests for the demo simulator: config loading, determinism, and the baselines.

The three shipped scenarios are part of the contract with the dashboard, so
these tests pin their behaviour end to end: analytic nulling must actually
null when geometry allows (``scenario_001``), the unconstrained baseline must
never null (``scenario_002``), and the mirror-bearing approach must be
declined (``scenario_003``).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from flywaydqn.agents.baselines_demo import UnconstrainedPolicy
from flywaydqn.demo import (
    CONFIG_DIR_ENV_VAR,
    DemoRunner,
    ScenarioConfig,
    demo_config_dir,
    demo_runner_factory,
    list_demo_scenarios,
    load_scenario,
    scenario_path,
)
from flywaydqn.service.app import RunnerFactory, create_app
from flywaydqn.service.schemas import PaceRequest, ScenarioRequest, StateFrame

KEYS = ("scenario_001", "scenario_002", "scenario_003")
EPISODE_STEPS = 600


def _request(key: str, *, seed: int = 1, weight: float = 1.0) -> ScenarioRequest:
    return ScenarioRequest(key=key, seed=seed, weight=weight)


def _episode(key: str, *, seed: int = 1, weight: float = 1.0) -> list[StateFrame]:
    runner = DemoRunner(_request(key, seed=seed, weight=weight))
    frames = [runner.initial_frame()]
    frames.extend(runner.step() for _ in range(EPISODE_STEPS - 1))
    return frames


def _floats(node: Any) -> Iterator[float]:
    """Yield every float in a nested model dump."""
    if isinstance(node, float):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _floats(value)
    elif isinstance(node, list):
        for value in node:
            yield from _floats(value)


def test_list_demo_scenarios_matches_the_yaml_files_on_disk() -> None:
    infos = list_demo_scenarios()
    assert tuple(info.key for info in infos) == KEYS
    assert all(info.label and info.description for info in infos)
    assert infos[0].label == "Spring migration / Sector A"


def test_demo_config_dir_is_the_repository_configs_demo() -> None:
    directory = demo_config_dir()
    assert directory.name == "demo"
    assert directory.is_dir()


def test_scenario_keys_reject_path_traversal() -> None:
    for bad_key in ("../secrets", "a/b", ".hidden", "", "a b", "a.yaml"):
        with pytest.raises(ValueError, match="invalid scenario key"):
            scenario_path(bad_key)


def test_unknown_scenario_key_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="no demo scenario"):
        load_scenario("does-not-exist")


def test_load_scenario_rejects_unknown_fields() -> None:
    raw = ScenarioConfig.model_validate(
        {"label": "x", "description": "y", "flock": {"range_m": 100.0}}
    ).model_dump()
    raw["surprise"] = 1
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(raw)


def test_config_env_var_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(tmp_path))
    assert demo_config_dir() == tmp_path
    with pytest.raises(FileNotFoundError):
        load_scenario("scenario_001")


@pytest.mark.parametrize(("key", "seed"), [(key, i + 1) for i, key in enumerate(KEYS)])
def test_initial_frame_is_wire_valid_and_every_value_finite(key: str, seed: int) -> None:
    frame = DemoRunner(_request(key, seed=seed)).initial_frame()
    assert frame.step == 0
    assert frame.time_s == 0.0
    assert frame.beam.index in range(16)
    assert math.isfinite(frame.beam.gain_dbi)
    assert frame.power_mw > 0.0
    dumped = frame.model_dump()
    assert all(math.isfinite(value) for value in _floats(dumped))
    StateFrame.model_validate(dumped)  # the dump round-trips through the wire model


# Wall-clock measurement, so it is the one frame field that is not a function of
# the seed. The determinism contract covers the simulated physics; the latency
# field is asserted separately to still be a real measurement.
_NON_DETERMINISTIC_FIELDS = {"inference_latency_ms"}


@pytest.mark.parametrize("key", KEYS)
def test_frames_are_deterministic_for_a_seed(key: str) -> None:
    first = _episode(key, seed=7)
    second = _episode(key, seed=7)
    assert [f.model_dump(exclude=_NON_DETERMINISTIC_FIELDS) for f in first] == [
        f.model_dump(exclude=_NON_DETERMINISTIC_FIELDS) for f in second
    ]
    # The excluded field must still be a real measurement, not a hole.
    assert all(
        f.inference_latency_ms > 0.0 and math.isfinite(f.inference_latency_ms)
        for f in first + second
    )


def test_different_seeds_move_the_users() -> None:
    a = DemoRunner(_request("scenario_001", seed=1)).initial_frame()
    b = DemoRunner(_request("scenario_001", seed=2)).initial_frame()
    assert [u.position for u in a.users] != [u.position for u in b.users]


def test_weight_overrides_the_file_and_reweights_the_reward() -> None:
    light = _episode("scenario_001", seed=1, weight=1.0)
    heavy = _episode("scenario_001", seed=1, weight=5.0)
    assert all(frame.weight == 5.0 for frame in heavy)
    # Same geometry, same throughput term; only the exposure term moves.
    assert heavy[100].metrics.throughput_term == light[100].metrics.throughput_term
    assert heavy[500].metrics.reward != light[500].metrics.reward


def test_zero_weight_is_accepted() -> None:
    frame = DemoRunner(_request("scenario_001", weight=0.0)).initial_frame()
    assert frame.weight == 0.0
    assert math.isfinite(frame.metrics.reward)


def test_unconstrained_baseline_never_nulls() -> None:
    frames = _episode("scenario_002", seed=2)
    assert not any(frame.null_active for frame in frames)
    # The heavier load near the flock still drives real exposure.
    assert max(frame.metrics.frac_over for frame in frames) > 0.05


def test_unconstrained_policy_override_never_nulls() -> None:
    runner = DemoRunner(_request("scenario_001"), policy=UnconstrainedPolicy())
    runner.initial_frame()
    assert not any(runner.step().null_active for _ in range(200))


def test_scenario_001_nulls_when_the_flock_is_in_range() -> None:
    frames = _episode("scenario_001", seed=1)
    nulled = [frame for frame in frames if frame.null_active]
    assert nulled, "analytic-nulling baseline never engaged in scenario_001"
    # The null must actually buy exposure reduction on the reported metric.
    before = [f for f in frames if f.step == nulled[0].step - 1]
    assert before and nulled[0].metrics.p95_density_wm2 < before[0].metrics.p95_density_wm2
    assert nulled[0].metrics.frac_over <= before[0].metrics.frac_over


def test_scenario_003_declines_to_null_on_the_mirror_bearing() -> None:
    frames = _episode("scenario_003", seed=3)
    assert not any(frame.null_active for frame in frames)
    mirrored = [f for f in frames if "mirror" in (f.explanation or "").lower()]
    assert mirrored, "scenario_003 never exercised the front/back mirror guard"
    assert any("front/back mirror" in (f.explanation or "") for f in mirrored)


def test_explanation_reports_the_live_decision_and_metrics() -> None:
    frame = _episode("scenario_003", seed=3)[300]
    explanation = frame.explanation or ""
    assert f"beam {frame.beam.index}" in explanation
    assert "null off" in explanation
    assert f"{frame.metrics.p95_density_wm2:.3f}" in explanation
    assert f"{frame.metrics.frac_over:.0%}" in explanation


def test_flock_respawns_when_it_leaves_the_sector() -> None:
    config = load_scenario("scenario_001")
    config.flock = config.flock.model_copy(update={"recycle_range_m": 100.0})
    runner = DemoRunner(_request("scenario_001"), config=config)
    entry = runner.initial_frame()
    recycled = runner.step()
    assert recycled.flock.centroid == entry.flock.centroid


def test_demo_runner_factory_matches_the_service_protocol() -> None:
    factory: RunnerFactory = demo_runner_factory
    runner = factory(_request("scenario_001"))
    assert runner.initial_frame().step == 0
    assert isinstance(runner.step(), StateFrame)


def test_create_app_builds_with_the_demo_factory() -> None:
    app = create_app()
    assert app.title == "FlywayDQN demo service"


def test_pace_route_exists_and_pace_is_bounded() -> None:
    """A live session can be re-paced, and a nonsensical pace is rejected.

    Pacing is presentation only, so the route is a thin setter; what matters is
    that it is registered and that the body cannot ask for a zero, negative or
    absurd multiplier.
    """
    app = create_app()
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/sessions/{session_id}/pace" in paths
    assert PaceRequest(pace=8.0).pace == 8.0
    for bad in (0.0, -1.0, 51.0):
        with pytest.raises(ValidationError):
            PaceRequest(pace=bad)
