"""Free-space power density and per-step exposure metrics.

The hand calculations in this module are the "independent hand calculation on
fixed scenarios" required by the Phase 2 acceptance criteria (spec Section 6).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from flywaydqn.exposure.density import (
    ICNIRP_2020_PUBLIC_REFERENCE_W_PER_M2,
    dose_w_s_per_m2,
    fraction_over_threshold,
    power_density_grid_w_per_m2,
    power_density_w_per_m2,
    s_max,
    s_p95,
)
from flywaydqn.exposure.metrics import (
    exposure_metrics_from_density,
    time_over_threshold_fraction,
)
from flywaydqn.exposure.thresholds import (
    exposure_penalty_softplus,
    icnirp_public_reference_level_2020_w_per_m2,
)

# --- power density -----------------------------------------------------------


def test_power_density_matches_hand_calculation() -> None:
    # S = P_tx * G / (4 pi r^2); 1 W into 0 dBi at 1 m is 1/(4 pi) W/m^2.
    assert power_density_w_per_m2(tx_power_mw=1000.0, gain_dbi=0.0, range_m=1.0) == pytest.approx(
        1.0 / (4.0 * math.pi), rel=1e-12
    )


def test_power_density_decays_with_inverse_square() -> None:
    near = power_density_w_per_m2(tx_power_mw=1000.0, gain_dbi=10.0, range_m=10.0)
    far = power_density_w_per_m2(tx_power_mw=1000.0, gain_dbi=10.0, range_m=100.0)
    assert near / far == pytest.approx(100.0, rel=1e-9)


def test_power_density_scales_linearly_with_gain_and_power() -> None:
    base = power_density_w_per_m2(tx_power_mw=100.0, gain_dbi=0.0, range_m=50.0)
    assert power_density_w_per_m2(tx_power_mw=200.0, gain_dbi=0.0, range_m=50.0) == pytest.approx(
        2.0 * base, rel=1e-12
    )
    assert power_density_w_per_m2(tx_power_mw=100.0, gain_dbi=10.0, range_m=50.0) == pytest.approx(
        10.0 * base, rel=1e-12
    )


def test_power_density_is_zero_without_transmit_power() -> None:
    assert power_density_w_per_m2(tx_power_mw=0.0, gain_dbi=20.0, range_m=10.0) == 0.0


@pytest.mark.parametrize("bad_range", [0.0, -1.0])
def test_invalid_range_raises(bad_range: float) -> None:
    with pytest.raises(ValueError):
        power_density_w_per_m2(tx_power_mw=100.0, gain_dbi=0.0, range_m=bad_range)


def test_icnirp_anchor_is_ten_watts_per_square_meter() -> None:
    assert ICNIRP_2020_PUBLIC_REFERENCE_W_PER_M2 == 10.0
    assert icnirp_public_reference_level_2020_w_per_m2() == 10.0


# --- grid density ------------------------------------------------------------


def test_grid_density_matches_pointwise_calculation() -> None:
    cells = np.array([[0.0, 100.0], [100.0, 0.0], [0.0, -100.0]])
    grid = power_density_grid_w_per_m2(
        tx_power_mw=1000.0,
        cell_positions_xy_m=cells,
        sector_xy_m=(0.0, 0.0),
        antenna_height_m=25.0,
        gains_dbi=np.zeros(3),
    )
    # Slant range includes the antenna height, so compare against r^2 = 100^2 + 25^2.
    expected = 1.0 * 1.0 / (4.0 * math.pi * (100.0**2 + 25.0**2))
    assert grid.shape == (3,)
    assert np.allclose(grid, expected, rtol=1e-12)


def test_grid_density_accepts_dbm_power() -> None:
    cells = np.array([[100.0, 0.0]])
    from_mw = power_density_grid_w_per_m2(
        tx_power_mw=100.0,
        cell_positions_xy_m=cells,
        sector_xy_m=(0.0, 0.0),
        antenna_height_m=25.0,
        gains_dbi=np.zeros(1),
    )
    from_dbm = power_density_grid_w_per_m2(
        tx_power_mw=0.0,
        cell_positions_xy_m=cells,
        sector_xy_m=(0.0, 0.0),
        antenna_height_m=25.0,
        tx_power_dbm=20.0,  # 20 dBm == 100 mW
        gains_dbi=np.zeros(1),
    )
    assert np.allclose(from_mw, from_dbm, rtol=1e-12)


def test_grid_density_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        power_density_grid_w_per_m2(
            tx_power_mw=100.0,
            cell_positions_xy_m=np.zeros((4, 3)),
            sector_xy_m=(0.0, 0.0),
            antenna_height_m=25.0,
        )
    with pytest.raises(ValueError):
        power_density_grid_w_per_m2(
            tx_power_mw=100.0,
            cell_positions_xy_m=np.zeros((4, 2)),
            sector_xy_m=(0.0, 0.0),
            antenna_height_m=25.0,
            gains_dbi=np.zeros(3),
        )


# --- scalar metrics ----------------------------------------------------------


def test_scalar_metrics_on_a_known_grid() -> None:
    grid = np.array([0.1, 0.5, 2.0, 20.0])
    assert s_max(grid) == pytest.approx(20.0)
    assert fraction_over_threshold(grid, s_thr=10.0) == pytest.approx(0.25)
    assert dose_w_s_per_m2(grid, dt_seconds=0.1) == pytest.approx(22.6 * 0.1)


def test_s_p95_of_uniform_grid_is_that_value() -> None:
    grid = np.full(100, 3.0)
    assert s_p95(grid) == pytest.approx(3.0)
    assert s_max(grid) == pytest.approx(3.0)


def test_empty_inputs_return_zero_metrics() -> None:
    empty: list[float] = []
    assert s_max(empty) == 0.0
    assert s_p95(empty) == 0.0
    assert fraction_over_threshold(empty, s_thr=1.0) == 0.0
    assert dose_w_s_per_m2(empty, dt_seconds=0.1) == 0.0


def test_invalid_threshold_and_dt_raise() -> None:
    with pytest.raises(ValueError):
        fraction_over_threshold([1.0], s_thr=0.0)
    with pytest.raises(ValueError):
        dose_w_s_per_m2([1.0], dt_seconds=0.0)


# --- per-step aggregation ----------------------------------------------------


def test_single_step_metrics_have_the_four_spec_keys() -> None:
    metrics = exposure_metrics_from_density([0.5, 1.0, 40.0], s_thr=10.0, dt_seconds=0.1)
    assert set(metrics) == {"s_max", "s_p95", "frac_over", "dose"}
    assert metrics["s_max"] == pytest.approx(40.0)
    assert metrics["frac_over"] == pytest.approx(1.0 / 3.0)
    assert metrics["dose"] == pytest.approx(41.5 * 0.1)


def test_rollout_metrics_aggregate_worst_case_and_total_dose() -> None:
    rollout = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],  # nothing over threshold
            [5.0, 5.0, 50.0, 5.0],  # one cell over
        ]
    )
    metrics = exposure_metrics_from_density(rollout, s_thr=10.0, dt_seconds=0.1)
    assert metrics["s_max"] == pytest.approx(50.0)
    assert metrics["frac_over"] == pytest.approx(0.25)
    assert metrics["dose"] == pytest.approx(float(rollout.sum()) * 0.1)


def test_empty_density_grid_raises() -> None:
    with pytest.raises(ValueError):
        exposure_metrics_from_density([], s_thr=10.0)


def test_three_dimensional_input_raises() -> None:
    with pytest.raises(ValueError):
        exposure_metrics_from_density(np.zeros((2, 2, 2)), s_thr=10.0)


def test_time_over_threshold_fraction_counts_steps() -> None:
    rollout = np.array([[1.0], [20.0], [1.0], [30.0]])
    assert time_over_threshold_fraction(rollout, s_thr=10.0) == pytest.approx(0.5)
    assert time_over_threshold_fraction([1.0], s_thr=10.0) == 0.0
    assert time_over_threshold_fraction([20.0], s_thr=10.0) == 1.0
    assert time_over_threshold_fraction([], s_thr=10.0) == 0.0


# --- softplus hinge (Decision D3) -------------------------------------------


def test_softplus_penalty_is_small_below_threshold() -> None:
    # Documented in the spec: about 0.0013 at x = 0 with kappa = 5.
    assert exposure_penalty_softplus(0.0, 10.0, kappa=5.0) == pytest.approx(0.0013, abs=1e-4)


def test_softplus_penalty_is_smooth_and_monotone_in_density() -> None:
    densities = np.linspace(0.0, 40.0, 400)
    penalties = np.array([exposure_penalty_softplus(float(d), 10.0, kappa=5.0) for d in densities])
    assert np.all(np.diff(penalties) >= 0.0)  # non-decreasing


def test_softplus_penalty_grows_roughly_linearly_above_threshold() -> None:
    at_2x = exposure_penalty_softplus(20.0, 10.0, kappa=5.0)
    at_3x = exposure_penalty_softplus(30.0, 10.0, kappa=5.0)
    # For large arguments softplus(x) -> x, so the increments approach 1 per unit x.
    assert (at_3x - at_2x) == pytest.approx(1.0, abs=0.02)
    assert at_2x < at_3x


def test_softplus_penalty_is_bounded_at_the_threshold() -> None:
    assert exposure_penalty_softplus(10.0, 10.0, kappa=5.0) == pytest.approx(
        math.log(2.0) / 5.0, rel=1e-12
    )


def test_invalid_softplus_arguments_raise() -> None:
    with pytest.raises(ValueError):
        exposure_penalty_softplus(1.0, 0.0)
    with pytest.raises(ValueError):
        exposure_penalty_softplus(1.0, 10.0, kappa=0.0)
