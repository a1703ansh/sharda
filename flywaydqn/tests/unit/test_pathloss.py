"""TR 38.901 UMa path loss, breakpoint, LOS probability, and shadow fading.

Reference values are computed from the closed-form expressions in
3GPP TR 38.901 V19.4.0 Table 7.4.1-1 and Table 7.4.2-1 with the standard
default heights (h_BS = 25 m, h_UT = 1.5 m, h_E = 1 m).
"""

from __future__ import annotations

import math

import pytest

from flywaydqn.rf.pathloss import (
    DEFAULT_H_BS_M,
    DEFAULT_H_UT_M,
    SCENARIO_DEFAULT,
    TR38901_VERSION,
    apply_shadow_fading,
    breakpoint_distance_m,
    los_probability_uma,
    pathloss_uma,
    shadow_fading_std_db,
    tr38901_uma_los_pathloss_dbm,
    tr38901_uma_nlos_pathloss_dbm,
)

FC_GHZ = 3.5


def test_reference_metadata_is_pinned() -> None:
    assert TR38901_VERSION == "3GPP TR 38.901 V19.4.0 (Rel-19, 2026-07)"
    assert SCENARIO_DEFAULT == "UMa"
    assert (DEFAULT_H_BS_M, DEFAULT_H_UT_M) == (25.0, 1.5)


def test_breakpoint_distance_matches_hand_calculation() -> None:
    # d_BP = 4 * (25 - 1) * (1.5 - 1) * 3.5e9 / c
    expected = 4.0 * 24.0 * 0.5 * 3.5e9 / 299_792_458.0
    assert breakpoint_distance_m(FC_GHZ) == pytest.approx(expected, rel=1e-12)
    # About 560 m at 3.5 GHz with the standard heights.
    assert breakpoint_distance_m(FC_GHZ) == pytest.approx(560.4, abs=0.2)


def test_los_below_breakpoint_matches_spec_formula() -> None:
    d_m = 100.0
    expected = 28.0 + 22.0 * math.log10(d_m) + 20.0 * math.log10(FC_GHZ)
    assert tr38901_uma_los_pathloss_dbm(d_m, FC_GHZ) == pytest.approx(expected, rel=1e-12)
    # 28 + 44 + 10.881 = 82.881 dB
    assert tr38901_uma_los_pathloss_dbm(d_m, FC_GHZ) == pytest.approx(82.881, abs=0.01)


def test_los_above_breakpoint_uses_slope_40_with_correction() -> None:
    d_m = 1000.0
    d_bp = breakpoint_distance_m(FC_GHZ)
    expected = (
        28.0
        + 40.0 * math.log10(d_m)
        + 20.0 * math.log10(FC_GHZ)
        - 9.0 * math.log10(d_bp**2 + (DEFAULT_H_BS_M - DEFAULT_H_UT_M) ** 2)
    )
    assert tr38901_uma_los_pathloss_dbm(d_m, FC_GHZ) == pytest.approx(expected, rel=1e-12)
    assert tr38901_uma_los_pathloss_dbm(d_m, FC_GHZ) == pytest.approx(109.40, abs=0.05)


def test_los_pathloss_is_nearly_continuous_at_the_breakpoint() -> None:
    """The two UMa LOS branches meet within a documented, small step.

    The standard's PL1 (slope 22) and PL2 (slope 40, minus the
    ``9*log10(d_BP^2 + (h_BS - h_UT)^2)`` correction) are not exactly equal at
    ``d_BP``: the residual step is

        18*log10(d_BP) - 9*log10(d_BP^2 + (h_BS - h_UT)^2)
        = -9*log10(1 + ((h_BS - h_UT)/d_BP)^2)

    which is about -0.007 dB for the default heights at 3.5 GHz. This test pins
    that residual so a later "fix" for spurious discontinuity is not mistaken
    for an improvement, and so the magnitude cannot grow unnoticed.
    """
    d_bp = breakpoint_distance_m(FC_GHZ)
    below = tr38901_uma_los_pathloss_dbm(d_bp - 1e-6, FC_GHZ)
    above = tr38901_uma_los_pathloss_dbm(d_bp + 1e-6, FC_GHZ)
    expected_step = -9.0 * math.log10(1.0 + ((DEFAULT_H_BS_M - DEFAULT_H_UT_M) / d_bp) ** 2)
    assert above - below == pytest.approx(expected_step, rel=1e-3, abs=1e-9)
    assert abs(above - below) < 0.01  # sub-0.01 dB in practice


def test_los_pathloss_is_monotone_increasing_and_slope_changes() -> None:
    # 10x the distance below the breakpoint adds 22 dB.
    d1, d2 = 20.0, 200.0
    assert tr38901_uma_los_pathloss_dbm(d2, FC_GHZ) - tr38901_uma_los_pathloss_dbm(
        d1, FC_GHZ
    ) == pytest.approx(22.0, abs=1e-9)

    # 10x the distance above the breakpoint adds 40 dB.
    d3, d4 = 1000.0, 4000.0
    assert tr38901_uma_los_pathloss_dbm(d4, FC_GHZ) - tr38901_uma_los_pathloss_dbm(
        d3, FC_GHZ
    ) == pytest.approx(40.0 * math.log10(d4 / d3), abs=1e-9)


def test_los_pathloss_scales_with_frequency_as_20log10() -> None:
    d_m = 150.0
    delta = tr38901_uma_los_pathloss_dbm(d_m, 7.0) - tr38901_uma_los_pathloss_dbm(d_m, 3.5)
    assert delta == pytest.approx(20.0 * math.log10(2.0), abs=1e-9)


def test_nlos_pathloss_matches_spec_formula() -> None:
    d_m = 100.0
    expected = 13.54 + 39.08 * math.log10(d_m) + 20.0 * math.log10(FC_GHZ) - 0.6 * (1.5 - 1.5)
    assert tr38901_uma_nlos_pathloss_dbm(d_m, FC_GHZ) == pytest.approx(expected, rel=1e-12)
    assert tr38901_uma_nlos_pathloss_dbm(d_m, FC_GHZ) == pytest.approx(102.58, abs=0.01)


def test_nlos_is_worse_than_los_at_the_same_range() -> None:
    for d_m in (50.0, 200.0, 900.0):
        assert tr38901_uma_nlos_pathloss_dbm(d_m, FC_GHZ) > tr38901_uma_los_pathloss_dbm(
            d_m, FC_GHZ
        )


def test_nlos_height_correction_is_small_and_negative() -> None:
    d_m = 200.0
    base = tr38901_uma_nlos_pathloss_dbm(d_m, FC_GHZ, h_ut_m=1.5)
    taller = tr38901_uma_nlos_pathloss_dbm(d_m, FC_GHZ, h_ut_m=10.0)
    assert taller == pytest.approx(base - 0.6 * (10.0 - 1.5), abs=1e-9)


def test_los_probability_is_one_up_to_18m_then_decays() -> None:
    assert los_probability_uma(5.0) == 1.0
    assert los_probability_uma(18.0) == 1.0
    assert los_probability_uma(50.0) < 1.0
    assert los_probability_uma(500.0) < los_probability_uma(50.0)
    assert 0.0 <= los_probability_uma(1000.0) <= 1.0


def test_los_probability_matches_hand_calculation_at_50m() -> None:
    d_m = 50.0
    expected = 18.0 / d_m + math.exp(-d_m / 63.0) * (1.0 - 18.0 / d_m)  # C' = 0 for h_UT <= 13 m
    assert los_probability_uma(d_m) == pytest.approx(expected, rel=1e-12)


def test_los_probability_height_term_activates_above_13m() -> None:
    d_m = 120.0
    low = los_probability_uma(d_m, h_ut_m=13.0)
    high = los_probability_uma(d_m, h_ut_m=23.0)
    assert high > low


def test_shadow_fading_std_matches_table_values() -> None:
    assert shadow_fading_std_db(los=True) == 4.0
    assert shadow_fading_std_db(los=False) == 6.0
    assert shadow_fading_std_db(los=False, fc_ghz=28.0) == 6.0


def test_shadow_fading_is_seeded_and_centred() -> None:
    import numpy as np

    rng = np.random.default_rng(1234)
    samples = [apply_shadow_fading(100.0, False, FC_GHZ, rng) for _ in range(2000)]
    residuals = np.asarray(samples) - 100.0
    assert abs(float(residuals.mean())) < 0.5  # centred near zero
    assert float(residuals.std()) == pytest.approx(6.0, rel=0.15)

    # Same seed reproduces the same sequence.
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    seq_a = [apply_shadow_fading(100.0, True, FC_GHZ, rng_a) for _ in range(5)]
    seq_b = [apply_shadow_fading(100.0, True, FC_GHZ, rng_b) for _ in range(5)]
    assert seq_a == seq_b


def test_pathloss_uma_dispatches_on_los_flag() -> None:
    assert pathloss_uma(120.0, FC_GHZ, los=True) == tr38901_uma_los_pathloss_dbm(120.0, FC_GHZ)
    assert pathloss_uma(120.0, FC_GHZ, los=False) == tr38901_uma_nlos_pathloss_dbm(120.0, FC_GHZ)
    assert pathloss_uma(120.0, FC_GHZ) == tr38901_uma_los_pathloss_dbm(120.0, FC_GHZ)


@pytest.mark.parametrize("bad_distance", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_distances_raise(bad_distance: float) -> None:
    with pytest.raises(ValueError):
        tr38901_uma_los_pathloss_dbm(bad_distance, FC_GHZ)
    with pytest.raises(ValueError):
        tr38901_uma_nlos_pathloss_dbm(bad_distance, FC_GHZ)
    with pytest.raises(ValueError):
        los_probability_uma(bad_distance)


@pytest.mark.parametrize("bad_frequency", [0.0, -3.5, float("nan")])
def test_invalid_frequencies_raise(bad_frequency: float) -> None:
    with pytest.raises(ValueError):
        tr38901_uma_los_pathloss_dbm(100.0, bad_frequency)
    with pytest.raises(ValueError):
        shadow_fading_std_db(los=True, fc_ghz=bad_frequency)
