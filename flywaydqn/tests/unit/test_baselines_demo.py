"""Unit tests for the rule-based demo baselines and their shared helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flywaydqn.agents.baselines_demo import (
    AnalyticNullingPolicy,
    CellGeometry,
    LiveState,
    NullDecision,
    RadioSetup,
    UnconstrainedPolicy,
    UserLink,
    angular_separation_deg,
    greedy_beam,
    mirror_azimuth_deg,
    transmit_weights,
)
from flywaydqn.rf.codebook import build_dft_codebook

ELEV_DEG = math.degrees(math.atan2(25.0, 200.0))  # a 50 m flock band over a 25 m mast


def _setup() -> RadioSetup:
    beams = build_dft_codebook(16, 8, 8, 0.5)
    return RadioSetup(
        codebook=tuple(beams),
        n_rows=8,
        n_cols=8,
        d_lambda=0.5,
        tx_power_mw=200_000.0,
        bandwidth_mhz=100.0,
        fc_ghz=3.5,
        noise_power_mw=2.0e-9,
        interference_mw=1.0e-11,
        s_thr_w_m2=0.03,
        spectral_efficiency_cap=6.0,
        h_bs_m=25.0,
        h_ut_m=1.5,
    )


def _cells(bearing_deg: float, range_m: float = 200.0) -> tuple[CellGeometry, ...]:
    """A compact footprint spread a few degrees around ``bearing_deg``."""
    slant_m = math.hypot(range_m, 25.0)
    elevation_deg = math.degrees(math.atan2(25.0, range_m))
    return tuple(
        CellGeometry(
            bearing_deg=bearing_deg + offset,
            elevation_deg=elevation_deg,
            slant_range_m=slant_m,
        )
        for offset in (-3.0, -1.5, 0.0, 1.5, 3.0)
    )


def _state(
    *,
    flock_bearing_deg: float,
    flock_range_m: float = 300.0,
    user_bearing_deg: float = 0.0,
    cells: tuple[CellGeometry, ...] | None = None,
) -> LiveState:
    return LiveState(
        step=0,
        time_s=0.0,
        flock_bearing_deg=flock_bearing_deg,
        flock_elevation_deg=ELEV_DEG,
        flock_range_m=flock_range_m,
        users=(UserLink(bearing_deg=user_bearing_deg, elevation_deg=-6.0, range_m=200.0),),
        cells=cells if cells is not None else _cells(flock_bearing_deg),
    )


def test_angular_separation_matches_azimuth_distance_on_the_horizon() -> None:
    assert angular_separation_deg(0.0, 30.0, 0.0, 30.0) == pytest.approx(0.0)
    assert angular_separation_deg(0.0, 0.0, 0.0, 90.0) == pytest.approx(90.0)


def test_angular_separation_sees_elevation() -> None:
    """A horizon beam and a flock at the same bearing are separated in 3-D."""
    assert angular_separation_deg(10.0, 0.0, 0.0, 0.0) == pytest.approx(10.0)


def test_mirror_azimuth_swaps_front_and_back() -> None:
    assert mirror_azimuth_deg(30.0) == pytest.approx(150.0)
    assert mirror_azimuth_deg(-60.0) == pytest.approx(-120.0)
    assert mirror_azimuth_deg(0.0) == pytest.approx(180.0)
    assert mirror_azimuth_deg(mirror_azimuth_deg(37.5)) == pytest.approx(37.5)


def test_transmit_weights_without_a_null_is_the_beam_itself() -> None:
    setup = _setup()
    beam = setup.codebook[5]
    assert transmit_weights(beam, setup) is beam.weights


def test_transmit_weights_with_a_null_kills_the_null_direction() -> None:
    setup = _setup()
    beam = setup.codebook[8]  # boresight
    weights = transmit_weights(beam, setup, null_elevation_deg=0.0, null_azimuth_deg=20.0)
    assert not np.array_equal(weights, beam.weights)
    assert float(np.linalg.norm(weights)) == pytest.approx(1.0, rel=1e-9)


def test_greedy_beam_follows_the_users() -> None:
    setup = _setup()
    beam = greedy_beam(_state(flock_bearing_deg=10.0, user_bearing_deg=45.0), setup)
    assert beam.azimuth_deg == pytest.approx(45.0)
    beam = greedy_beam(_state(flock_bearing_deg=10.0, user_bearing_deg=-30.0), setup)
    assert beam.azimuth_deg == pytest.approx(-30.0)


def test_unconstrained_policy_never_nulls() -> None:
    setup = _setup()
    action = UnconstrainedPolicy().act(_state(flock_bearing_deg=20.0), setup)
    assert action.null_active is False
    assert action.decision is NullDecision.NOT_APPLICABLE
    assert action.weights is action.beam.weights


def test_analytic_policy_applies_a_null_when_exposure_is_real() -> None:
    setup = _setup()
    # Users at boresight, a dense footprint ~9 deg off the beam: the peak cell
    # clears the threshold and sits far enough off-axis to null.
    state = _state(
        flock_bearing_deg=8.0,
        flock_range_m=300.0,
        user_bearing_deg=0.0,
        cells=_cells(8.0, 200.0),
    )
    action = AnalyticNullingPolicy().act(state, setup)
    assert action.decision is NullDecision.APPLIED
    assert action.null_active is True


def test_analytic_policy_declines_on_the_serving_axis() -> None:
    setup = _setup()
    # A far, low footprint right on the beam's azimuth: the peak cell is only
    # a couple of degrees from the serving direction in 3-D.
    state = _state(
        flock_bearing_deg=30.0,
        flock_range_m=300.0,
        user_bearing_deg=30.0,
        cells=_cells(30.0, 800.0),
    )
    action = AnalyticNullingPolicy().act(state, setup)
    assert action.decision is NullDecision.SERVING_AXIS
    assert action.null_active is False


def test_analytic_policy_declines_on_the_mirror_bearing() -> None:
    setup = _setup()
    # Users at +30 put the serving beam at 30 deg; the footprint sits at its
    # front/back mirror, 150 deg, where the array cannot null without killing
    # the beam. The mirror guard fires even though the footprint is cold.
    state = _state(
        flock_bearing_deg=150.0,
        flock_range_m=300.0,
        user_bearing_deg=30.0,
        cells=_cells(150.0, 800.0),
    )
    action = AnalyticNullingPolicy().act(state, setup)
    assert action.decision is NullDecision.MIRROR_BEARING
    assert action.null_active is False


def test_analytic_policy_declines_out_of_range() -> None:
    setup = _setup()
    state = _state(flock_bearing_deg=12.0, flock_range_m=900.0)
    action = AnalyticNullingPolicy(null_max_range_m=500.0).act(state, setup)
    assert action.decision is NullDecision.OUT_OF_RANGE


def test_analytic_policy_declines_below_the_threshold() -> None:
    setup = _setup()
    # Inside the nulling range but far enough out that S_p95 is already tiny.
    state = _state(flock_bearing_deg=12.0, flock_range_m=480.0, cells=_cells(12.0, 480.0))
    action = AnalyticNullingPolicy(null_max_range_m=500.0).act(state, setup)
    assert action.decision is NullDecision.BELOW_THRESHOLD


def test_analytic_policy_declines_when_the_null_would_not_help() -> None:
    setup = _setup()
    # Footprint far off the beam axis: the peak cell is cold, and a null on it
    # cannot buy the required reduction because the p95 is already on the floor.
    state = _state(flock_bearing_deg=55.0, user_bearing_deg=-40.0)
    action = AnalyticNullingPolicy().act(state, setup)
    assert action.decision in {NullDecision.NOT_HELPFUL, NullDecision.BELOW_THRESHOLD}


def test_every_decline_carries_an_explanation() -> None:
    setup = _setup()
    policy = AnalyticNullingPolicy()
    states = [
        _state(
            flock_bearing_deg=150.0,
            flock_range_m=300.0,
            user_bearing_deg=30.0,
            cells=_cells(150.0, 800.0),
        ),
        _state(
            flock_bearing_deg=30.0,
            flock_range_m=300.0,
            user_bearing_deg=30.0,
            cells=_cells(30.0, 800.0),
        ),
        _state(flock_bearing_deg=12.0, flock_range_m=900.0),
    ]
    actions = [policy.act(state, setup) for state in states]
    assert all(action.detail for action in actions)
    assert len({action.decision for action in actions}) == len(actions)
