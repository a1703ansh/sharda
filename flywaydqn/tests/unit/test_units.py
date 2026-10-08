"""Unit conversions live in one module; verify round-trips and known anchors."""

from __future__ import annotations

import math

import pytest

from flywaydqn.units import (
    db_to_linear,
    dbi_to_linear,
    dbm_to_mw,
    ghz_to_hz,
    hz_to_ghz,
    linear_to_db,
    linear_to_dbi,
    mw_per_cm2_to_watt_per_m2,
    mw_to_dbm,
    thermal_noise_power_mw,
    watt_per_m2_to_mw_per_cm2,
)


def test_dbm_mw_round_trip() -> None:
    for dbm in (-30.0, 0.0, 10.0, 20.0, 43.0):
        assert mw_to_dbm(dbm_to_mw(dbm)) == pytest.approx(dbm, abs=1e-9)


def test_known_dbm_anchors() -> None:
    assert dbm_to_mw(0.0) == pytest.approx(1.0)
    assert dbm_to_mw(30.0) == pytest.approx(1000.0)
    assert mw_to_dbm(100.0) == pytest.approx(20.0)


def test_db_linear_round_trip() -> None:
    assert db_to_linear(10.0) == pytest.approx(10.0)
    assert linear_to_db(100.0) == pytest.approx(20.0)
    assert linear_to_db(0.0) == -math.inf


def test_dbi_helpers_match_db_helpers() -> None:
    assert dbi_to_linear(18.06) == pytest.approx(db_to_linear(18.06))
    assert linear_to_dbi(64.0) == pytest.approx(10.0 * math.log10(64.0))
    assert linear_to_dbi(0.0) == -math.inf


def test_frequency_round_trip() -> None:
    assert ghz_to_hz(3.5) == pytest.approx(3.5e9)
    assert hz_to_ghz(3.5e9) == pytest.approx(3.5)


def test_icnirp_density_conversion_anchor() -> None:
    # The ICNIRP 2020 general-public anchor is 10 W/m^2 == 1 mW/cm^2.
    assert watt_per_m2_to_mw_per_cm2(10.0) == pytest.approx(1.0)
    assert mw_per_cm2_to_watt_per_m2(1.0) == pytest.approx(10.0)
    assert mw_per_cm2_to_watt_per_m2(watt_per_m2_to_mw_per_cm2(7.3)) == pytest.approx(7.3)


def test_thermal_noise_is_about_minus_94_dbm_for_20mhz_noise_figure_7() -> None:
    # k*T*B at 290 K with B = 20 MHz is about -101 dBm; +7 dB NF gives about -94 dBm.
    noise_mw = thermal_noise_power_mw(20e6, noise_figure_db=7.0)
    assert mw_to_dbm(noise_mw) == pytest.approx(-94.0, abs=0.5)


def test_thermal_noise_scales_with_bandwidth() -> None:
    narrow = thermal_noise_power_mw(10e6)
    wide = thermal_noise_power_mw(20e6)
    assert wide == pytest.approx(2.0 * narrow, rel=1e-9)


@pytest.mark.parametrize(
    "call",
    [
        lambda: mw_to_dbm(0.0),
        lambda: mw_to_dbm(-1.0),
        lambda: ghz_to_hz(0.0),
        lambda: hz_to_ghz(-1.0),
        lambda: thermal_noise_power_mw(0.0),
    ],
)
def test_invalid_inputs_raise(call: object) -> None:
    with pytest.raises(ValueError):
        call()  # type: ignore[operator]
