"""SINR and throughput mapping."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flywaydqn.rf.sinr import (
    DEFAULT_SPECTRAL_EFFICIENCY_CAP,
    fifth_percentile_throughput_mbps,
    max_throughput_reference_mbps,
    min_rate_fraction,
    sinr_db,
    sinr_linear,
    sum_throughput_mbps,
    throughput_mbps_shannon,
)


def test_sinr_matches_hand_calculation() -> None:
    # signal = P*G/PL, noise+interference = N + I
    sinr = sinr_linear(
        serving_gain_dbi=10.0,  # x10
        pathloss_db=80.0,  # 1e8
        interference_mw=1e-7,
        noise_power_mw=1e-8,
        tx_power_mw=20.0,
    )
    expected = (20.0 * 10.0 / 1e8) / (1e-8 + 1e-7)
    assert sinr == pytest.approx(expected, rel=1e-12)


def test_sinr_grows_with_transmit_power_linearly() -> None:
    kwargs = dict(
        serving_gain_dbi=8.0,
        pathloss_db=100.0,
        interference_mw=1e-6,
        noise_power_mw=1e-7,
    )
    low = sinr_linear(tx_power_mw=10.0, **kwargs)
    high = sinr_linear(tx_power_mw=100.0, **kwargs)
    assert high == pytest.approx(low * 10.0, rel=1e-12)


def test_sinr_falls_as_interference_rises() -> None:
    kwargs = dict(serving_gain_dbi=8.0, pathloss_db=100.0, noise_power_mw=1e-7, tx_power_mw=20.0)
    assert sinr_linear(interference_mw=1e-6, **kwargs) > sinr_linear(interference_mw=1e-4, **kwargs)


def test_sinr_falls_as_path_loss_rises() -> None:
    kwargs = dict(serving_gain_dbi=8.0, interference_mw=1e-6, noise_power_mw=1e-7, tx_power_mw=20.0)
    near = sinr_linear(pathloss_db=80.0, **kwargs)
    far = sinr_linear(pathloss_db=110.0, **kwargs)
    assert far == pytest.approx(near / 1e3, rel=1e-9)


def test_zero_transmit_power_gives_zero_sinr() -> None:
    assert (
        sinr_linear(
            serving_gain_dbi=8.0,
            pathloss_db=90.0,
            interference_mw=1e-6,
            noise_power_mw=1e-7,
            tx_power_mw=0.0,
        )
        == 0.0
    )


def test_sinr_db_is_the_log_of_linear_sinr() -> None:
    assert sinr_db(100.0) == pytest.approx(20.0)
    assert sinr_db(1.0) == pytest.approx(0.0)
    assert sinr_db(0.0) == -math.inf


def test_throughput_is_zero_without_sinr() -> None:
    assert throughput_mbps_shannon(sinr=0.0, bandwidth_mhz=100.0) == 0.0


def test_throughput_is_scaled_by_bandwidth() -> None:
    narrow = throughput_mbps_shannon(sinr=30.0, bandwidth_mhz=20.0)
    wide = throughput_mbps_shannon(sinr=30.0, bandwidth_mhz=100.0)
    assert wide == pytest.approx(narrow * 5.0, rel=1e-12)


def test_throughput_matches_shannon_below_the_cap() -> None:
    sinr = 15.0
    expected = 100.0 * math.log2(1.0 + sinr)
    assert throughput_mbps_shannon(sinr=sinr, bandwidth_mhz=100.0) == pytest.approx(
        expected, rel=1e-12
    )


def test_throughput_is_capped_at_high_sinr() -> None:
    capped = throughput_mbps_shannon(sinr=1e6, bandwidth_mhz=100.0)
    assert capped == pytest.approx(100.0 * DEFAULT_SPECTRAL_EFFICIENCY_CAP, rel=1e-12)


def test_negative_sinr_is_clamped_to_zero() -> None:
    assert throughput_mbps_shannon(sinr=-5.0, bandwidth_mhz=100.0) == 0.0


def test_sum_throughput_handles_empty_input() -> None:
    assert sum_throughput_mbps([]) == 0.0
    assert sum_throughput_mbps([10.0, 20.0, 30.0]) == pytest.approx(60.0)


def test_fifth_percentile_throughput_matches_numpy() -> None:
    values = list(range(1, 101))
    assert fifth_percentile_throughput_mbps(values) == pytest.approx(
        float(np.percentile(values, 5.0))
    )
    assert fifth_percentile_throughput_mbps([]) == 0.0


def test_min_rate_fraction_counts_users_below_the_floor() -> None:
    throughputs = [1.0, 5.0, 10.0, 20.0]
    assert min_rate_fraction(throughputs, min_rate_mbps=6.0) == pytest.approx(0.5)
    assert min_rate_fraction(throughputs, min_rate_mbps=0.5) == pytest.approx(0.0)
    assert min_rate_fraction([], min_rate_mbps=5.0) == 0.0


def test_max_throughput_reference_scales_with_users_and_bandwidth() -> None:
    one = max_throughput_reference_mbps(n_users=1, bandwidth_mhz=100.0)
    ten = max_throughput_reference_mbps(n_users=10, bandwidth_mhz=100.0)
    assert one == pytest.approx(100.0 * DEFAULT_SPECTRAL_EFFICIENCY_CAP)
    assert ten == pytest.approx(one * 10.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tx_power_mw": -1.0},
        {"noise_power_mw": -1.0},
        {"interference_mw": -1.0},
    ],
)
def test_negative_powers_raise(kwargs: dict[str, float]) -> None:
    base = dict(
        serving_gain_dbi=8.0,
        pathloss_db=90.0,
        interference_mw=1e-6,
        noise_power_mw=1e-7,
        tx_power_mw=20.0,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        sinr_linear(**base)


def test_invalid_throughput_arguments_raise() -> None:
    with pytest.raises(ValueError):
        throughput_mbps_shannon(sinr=10.0, bandwidth_mhz=-1.0)
    with pytest.raises(ValueError):
        throughput_mbps_shannon(sinr=10.0, bandwidth_mhz=100.0, spectral_efficiency_cap=0.0)
    with pytest.raises(ValueError):
        max_throughput_reference_mbps(n_users=-1, bandwidth_mhz=100.0)
