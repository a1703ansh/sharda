"""Unit conversions, kept in one place.

Spec Section 7 (Engineering Standards): "Unit conversions in one module only."
Every other module should import from here rather than open-coding a ``10 **``
or ``log10`` conversion.

Canonical units used across the package
---------------------------------------
- power: milliwatt (mW) internally, dBm at configuration boundaries
- gain: dBi
- power density / exposure: watt per square meter (W/m^2)
- distance: meter (m); frequency: hertz (Hz) internally, GHz at config boundaries
- time: second (s)
- throughput: megabit per second (Mbps)
"""

from __future__ import annotations

import math

# Physical constants.
SPEED_OF_LIGHT_M_PER_S: float = 299_792_458.0
BOLTZMANN_J_PER_K: float = 1.380_649e-23
REFERENCE_TEMPERATURE_K: float = 290.0


def dbm_to_mw(power_dbm: float) -> float:
    """Convert power in dBm to milliwatt (mW)."""
    return float(10.0 ** (power_dbm / 10.0))


def mw_to_dbm(power_mw: float) -> float:
    """Convert power in milliwatt (mW) to dBm."""
    if power_mw <= 0.0:
        raise ValueError(f"power_mw must be > 0 to convert to dBm, got {power_mw}")
    return 10.0 * math.log10(power_mw)


def db_to_linear(ratio_db: float) -> float:
    """Convert a power ratio in dB to a linear factor."""
    return float(10.0 ** (ratio_db / 10.0))


def linear_to_db(ratio: float) -> float:
    """Convert a linear power ratio to dB."""
    if ratio <= 0.0:
        return -math.inf
    return 10.0 * math.log10(ratio)


def dbi_to_linear(gain_dbi: float) -> float:
    """Convert an antenna gain in dBi to a linear power factor."""
    return float(10.0 ** (gain_dbi / 10.0))


def linear_to_dbi(gain_linear: float) -> float:
    """Convert a linear antenna gain factor to dBi."""
    if gain_linear <= 0.0:
        return -math.inf
    return 10.0 * math.log10(gain_linear)


def ghz_to_hz(fc_ghz: float) -> float:
    """Convert frequency from gigahertz to hertz."""
    if fc_ghz <= 0.0:
        raise ValueError(f"fc_ghz must be > 0, got {fc_ghz}")
    return fc_ghz * 1e9


def hz_to_ghz(fc_hz: float) -> float:
    """Convert frequency from hertz to gigahertz."""
    if fc_hz <= 0.0:
        raise ValueError(f"fc_hz must be > 0, got {fc_hz}")
    return fc_hz / 1e9


def watt_per_m2_to_mw_per_cm2(density_w_per_m2: float) -> float:
    """Convert power density from W/m^2 to mW/cm^2 (1 mW/cm^2 = 10 W/m^2)."""
    return density_w_per_m2 / 10.0


def mw_per_cm2_to_watt_per_m2(density_mw_per_cm2: float) -> float:
    """Convert power density from mW/cm^2 to W/m^2 (10 W/m^2 = 1 mW/cm^2)."""
    return density_mw_per_cm2 * 10.0


def thermal_noise_power_mw(bandwidth_hz: float, noise_figure_db: float = 7.0) -> float:
    """Return thermal noise power in milliwatt (mW).

        N = k * T * B * 10**(F/10)

    Parameters
    ----------
    bandwidth_hz : float
        Receiver bandwidth in hertz.
    noise_figure_db : float
        Receiver noise figure in dB. Default 7.0 dB, a typical value for a
        3.5 GHz macro receiver.

    Returns
    -------
    float
        Noise power in mW.
    """
    if bandwidth_hz <= 0.0:
        raise ValueError(f"bandwidth_hz must be > 0, got {bandwidth_hz}")
    noise_watt = (
        BOLTZMANN_J_PER_K * REFERENCE_TEMPERATURE_K * bandwidth_hz * db_to_linear(noise_figure_db)
    )
    return noise_watt * 1e3


__all__ = [
    "SPEED_OF_LIGHT_M_PER_S",
    "BOLTZMANN_J_PER_K",
    "REFERENCE_TEMPERATURE_K",
    "dbm_to_mw",
    "mw_to_dbm",
    "db_to_linear",
    "linear_to_db",
    "dbi_to_linear",
    "linear_to_dbi",
    "ghz_to_hz",
    "hz_to_ghz",
    "watt_per_m2_to_mw_per_cm2",
    "mw_per_cm2_to_watt_per_m2",
    "thermal_noise_power_mw",
]
