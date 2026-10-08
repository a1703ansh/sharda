"""SINR and throughput mapping for user links.

Spec Section 4.1: user links get SINR from serving-beam gain, path loss,
inter-sector interference (fixed neighbours), and noise; throughput comes from
Shannon with a bounded efficiency factor (or an MCS table mapping).

All unit conversions come from :mod:`flywaydqn.units`.

Units
-----
- power: milliwatt (mW)
- gain: dBi
- path loss: dB
- SINR: linear ratio (dimensionless, >= 0)
- throughput: megabit per second (Mbps)
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from flywaydqn.units import db_to_linear, dbi_to_linear, dbm_to_mw, mw_to_dbm

# Shannon spectral efficiency is capped to model a finite modulation order.
DEFAULT_SPECTRAL_EFFICIENCY_CAP: float = 6.0

__all__ = [
    "DEFAULT_SPECTRAL_EFFICIENCY_CAP",
    "dbm_to_mw",
    "mw_to_dbm",
    "sinr_linear",
    "sinr_db",
    "throughput_mbps_shannon",
    "sum_throughput_mbps",
    "min_rate_fraction",
    "fifth_percentile_throughput_mbps",
    "max_throughput_reference_mbps",
]


def sinr_linear(
    serving_gain_dbi: float,
    pathloss_db: float,
    interference_mw: float,
    noise_power_mw: float,
    tx_power_mw: float,
) -> float:
    """Return linear SINR for one user link.

        SINR = (P_tx * G_serving / PL_linear) / (I + N)

    Parameters
    ----------
    serving_gain_dbi : float
        Serving beam gain toward the user, in dBi.
    pathloss_db : float
        Path loss in dB (positive number representing loss).
    interference_mw : float
        Aggregate inter-sector interference power in mW.
    noise_power_mw : float
        Thermal noise power in mW.
    tx_power_mw : float
        Transmit power in mW.

    Returns
    -------
    float
        Linear SINR, ``>= 0``. Returns ``inf`` only when there is no noise or
        interference, which cannot happen with a positive noise floor.
    """
    if tx_power_mw < 0.0:
        raise ValueError(f"tx_power_mw must be >= 0, got {tx_power_mw}")
    if noise_power_mw < 0.0:
        raise ValueError(f"noise_power_mw must be >= 0, got {noise_power_mw}")
    if interference_mw < 0.0:
        raise ValueError(f"interference_mw must be >= 0, got {interference_mw}")

    signal_mw = tx_power_mw * dbi_to_linear(serving_gain_dbi) / db_to_linear(pathloss_db)
    denominator_mw = noise_power_mw + interference_mw
    if denominator_mw <= 0.0:
        return math.inf
    return signal_mw / denominator_mw


def sinr_db(sinr: float) -> float:
    """Convert a linear SINR to dB."""
    if sinr <= 0.0:
        return -math.inf
    return 10.0 * math.log10(sinr)


def throughput_mbps_shannon(
    sinr: float,
    bandwidth_mhz: float,
    spectral_efficiency_cap: float = DEFAULT_SPECTRAL_EFFICIENCY_CAP,
) -> float:
    """Return single-link throughput in Mbps.

        T = B * min(log2(1 + SINR), cap)

    Parameters
    ----------
    sinr : float
        Linear SINR.
    bandwidth_mhz : float
        Allocated bandwidth in MHz.
    spectral_efficiency_cap : float
        Maximum spectral efficiency in bit/s/Hz. Default 6.0.

    Returns
    -------
    float
        Throughput in Mbps. Zero when ``bandwidth_mhz`` is zero.
    """
    if bandwidth_mhz < 0.0:
        raise ValueError(f"bandwidth_mhz must be >= 0, got {bandwidth_mhz}")
    if spectral_efficiency_cap <= 0.0:
        raise ValueError(f"spectral_efficiency_cap must be > 0, got {spectral_efficiency_cap}")

    effective_sinr = max(float(sinr), 0.0)
    spectral_efficiency = min(math.log2(1.0 + effective_sinr), spectral_efficiency_cap)
    return bandwidth_mhz * spectral_efficiency


def sum_throughput_mbps(throughputs_mbps: npt.ArrayLike) -> float:
    """Return the sum of per-user throughputs in Mbps."""
    values = np.asarray(throughputs_mbps, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.sum(values))


def fifth_percentile_throughput_mbps(throughputs_mbps: npt.ArrayLike) -> float:
    """Return the 5th-percentile user throughput in Mbps (spec Section 8)."""
    values = np.asarray(throughputs_mbps, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 5.0))


def min_rate_fraction(
    throughputs_mbps: npt.ArrayLike,
    min_rate_mbps: float,
) -> float:
    """Return the fraction of users below a minimum rate (spec Section 8)."""
    values = np.asarray(throughputs_mbps, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.mean(values < min_rate_mbps))


def max_throughput_reference_mbps(
    n_users: int,
    bandwidth_mhz: float,
    spectral_efficiency_cap: float = DEFAULT_SPECTRAL_EFFICIENCY_CAP,
) -> float:
    """Return the per-scenario max-throughput reference for reward normalization.

    The reward term ``T_hat`` (spec Section 4.4) is divided by this reference so
    that the throughput term is comparable across scenarios.
    """
    if n_users < 0:
        raise ValueError(f"n_users must be >= 0, got {n_users}")
    return throughput_mbps_shannon(
        sinr=float("inf"),
        bandwidth_mhz=bandwidth_mhz * n_users,
        spectral_efficiency_cap=spectral_efficiency_cap,
    )
