"""Power-density computation over a flock footprint.

Spec Section 4.2: the power density at a point at range ``r`` in direction
``(theta, phi)`` is

    S = P_tx * G(theta, phi) / (4 * pi * r^2)     [W/m^2]

using the free-space far-field formula. ``G`` already includes the element
pattern and the array factor (see :func:`flywaydqn.rf.array.composite_gain_dbi`).

The exposure metrics derived from a density grid are ``S_max``, ``S_p95``,
``frac_over``, and ``dose`` (spec Section 4.2 and the reward in 4.4).

Units
-----
- power density: watt per square meter (W/m^2)
- transmit power: milliwatt (mW) at the API boundary, converted internally
- gain: dBi
- range: meters (m)
- dose: W*s/m^2
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from flywaydqn.units import dbi_to_linear, dbm_to_mw

# ICNIRP 2020 general-public reference level for 2-300 GHz, in W/m^2.
ICNIRP_2020_PUBLIC_REFERENCE_W_PER_M2: float = 10.0


def power_density_w_per_m2(
    tx_power_mw: float,
    gain_dbi: float,
    range_m: float,
) -> float:
    """Return free-space power density at a point, in W/m^2.

        S = P_tx * G / (4 * pi * r^2)

    Parameters
    ----------
    tx_power_mw : float
        Transmit power in milliwatt (mW).
    gain_dbi : float
        Antenna gain toward the point, in dBi.
    range_m : float
        Slant range from the antenna to the point, in meters. Must be > 0.

    Returns
    -------
    float
        Power density in W/m^2.
    """
    if tx_power_mw < 0.0:
        raise ValueError(f"tx_power_mw must be >= 0, got {tx_power_mw}")
    if range_m <= 0.0:
        raise ValueError(f"range_m must be > 0, got {range_m}")

    tx_power_w = tx_power_mw * 1e-3
    return tx_power_w * dbi_to_linear(gain_dbi) / (4.0 * math.pi * range_m**2)


def power_density_grid_w_per_m2(
    tx_power_mw: float,
    cell_positions_xy_m: npt.NDArray[np.float64],
    sector_xy_m: tuple[float, float],
    antenna_height_m: float,
    *,
    tx_power_dbm: float | None = None,
    gains_dbi: npt.NDArray[np.float64] | None = None,
    n_rows: int = 8,
    n_cols: int = 8,
    d_lambda: float = 0.5,
) -> npt.NDArray[np.float64]:
    """Return power density at each flock cell, in W/m^2 (vectorized).

    Parameters
    ----------
    tx_power_mw : float
        Transmit power in mW. Ignored when ``tx_power_dbm`` is given.
    cell_positions_xy_m : numpy.ndarray
        Array of shape ``(n_cells, 2)`` with cell positions in meters.
    sector_xy_m : tuple of float
        Sector antenna position in the same frame, in meters.
    antenna_height_m : float
        Antenna height above the cell plane, in meters.
    tx_power_dbm : float or None
        Alternative way to pass transmit power, in dBm. When set it takes
        precedence over ``tx_power_mw``.
    gains_dbi : numpy.ndarray or None
        Precomputed per-cell gains in dBi. If omitted, a fixed 0 dBi isotropic
        assumption is used; callers with a beam should always pass gains.
    n_rows, n_cols, d_lambda
        Accepted for interface symmetry with the array module; unused when
        ``gains_dbi`` is provided. Kept so callers can pass the array geometry
        through one call site without a second import.

    Returns
    -------
    numpy.ndarray
        Power densities in W/m^2, one per cell.
    """
    del n_rows, n_cols, d_lambda  # geometry is folded into gains_dbi by the caller

    cells = np.asarray(cell_positions_xy_m, dtype=np.float64)
    if cells.ndim != 2 or cells.shape[1] != 2:
        raise ValueError(f"cell_positions_xy_m must have shape (n, 2), got {cells.shape}")
    if antenna_height_m <= 0.0:
        raise ValueError(f"antenna_height_m must be > 0, got {antenna_height_m}")

    power_mw = dbm_to_mw(tx_power_dbm) if tx_power_dbm is not None else tx_power_mw
    if power_mw < 0.0:
        raise ValueError(f"transmit power must be >= 0, got {power_mw} mW")

    sector_x, sector_y = sector_xy_m
    dx = cells[:, 0] - sector_x
    dy = cells[:, 1] - sector_y
    # np.asarray keeps the dtype concrete: numpy's stubs type the intermediate
    # elementwise arithmetic as Any, which would otherwise leak out of this
    # typed public function.
    ranges_m = np.asarray(np.sqrt(dx**2 + dy**2 + antenna_height_m**2), dtype=np.float64)

    if gains_dbi is None:
        gain_linear = np.ones_like(ranges_m)
    else:
        gains = np.asarray(gains_dbi, dtype=np.float64)
        if gains.shape != ranges_m.shape:
            raise ValueError(f"gains_dbi shape {gains.shape} != cell count {ranges_m.shape}")
        gain_linear = np.power(10.0, gains / 10.0)

    power_w = power_mw * 1e-3
    return np.asarray(power_w * gain_linear / (4.0 * math.pi * ranges_m**2), dtype=np.float64)


def s_max(density_w_per_m2: npt.ArrayLike) -> float:
    """Return the maximum power density over the cell set, in W/m^2."""
    values = np.asarray(density_w_per_m2, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.max(values))


def s_p95(density_w_per_m2: npt.ArrayLike) -> float:
    """Return the 95th-percentile power density over the cell set, in W/m^2.

    This is the quantity the exposure penalty in the reward is built on
    (spec Section 4.4, Decision D3).
    """
    values = np.asarray(density_w_per_m2, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 95.0))


def fraction_over_threshold(density_w_per_m2: npt.ArrayLike, s_thr: float) -> float:
    """Return the fraction of cells with density strictly above ``s_thr``."""
    if s_thr <= 0.0:
        raise ValueError(f"s_thr must be > 0, got {s_thr}")
    values = np.asarray(density_w_per_m2, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.mean(values > s_thr))


def dose_w_s_per_m2(density_w_per_m2: npt.ArrayLike, dt_seconds: float) -> float:
    """Return the time-integrated dose over the cell set, in W*s/m^2.

    dose = sum(S_cells) * dt
    """
    if dt_seconds <= 0.0:
        raise ValueError(f"dt_seconds must be > 0, got {dt_seconds}")
    values = np.asarray(density_w_per_m2, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.sum(values) * dt_seconds)


__all__ = [
    "ICNIRP_2020_PUBLIC_REFERENCE_W_PER_M2",
    "power_density_w_per_m2",
    "power_density_grid_w_per_m2",
    "s_max",
    "s_p95",
    "fraction_over_threshold",
    "dose_w_s_per_m2",
]
