"""Per-step exposure metrics over a flock footprint.

Spec Section 4.2 defines three metrics per step:

- ``S_max``: maximum power density over the flock footprint
- ``frac_over``: fraction of flock cells with ``S > S_thr``
- ``dose``: ``sum(S) * dt``, the time-integrated exposure over flock cells

The reward (spec Section 4.4) additionally uses ``S_p95``, so it is reported
here as well.

Units
-----
- power density: W/m^2
- dose: W*s/m^2
- ``dt``: seconds
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import numpy.typing as npt

from flywaydqn.exposure.density import (
    dose_w_s_per_m2,
    fraction_over_threshold,
    s_max,
    s_p95,
)


class ExposureMetrics(TypedDict):
    """Exposure metrics for one step (or aggregated over a rollout)."""

    s_max: float
    s_p95: float
    frac_over: float
    dose: float


def exposure_metrics_from_density(
    density_w_per_m2: npt.ArrayLike,
    s_thr: float,
    *,
    dt_seconds: float = 0.1,
) -> ExposureMetrics:
    """Compute per-step exposure metrics from a power-density grid.

    Parameters
    ----------
    density_w_per_m2 : array-like
        Power density per flock cell, in W/m^2. A 1-D array is scored as a
        single step. A 2-D array of shape ``(n_steps, n_cells)`` is scored over
        the rollout: ``s_max`` and ``s_p95`` become the largest values seen
        across steps, ``frac_over`` the worst-case fraction, and ``dose`` the
        sum over all steps and cells times ``dt``.
    s_thr : float
        Exposure threshold in W/m^2. Default scenarios use the ICNIRP 2020
        general-public reference level (10 W/m^2); this is a configurable
        parameter, not an avian-harm threshold (spec Section 2.2).
    dt_seconds : float
        Simulation time step in seconds. Default 0.1, matching the near-RT RIC
        timescale used by the environment.

    Returns
    -------
    ExposureMetrics
        Dictionary with keys ``s_max``, ``s_p95``, ``frac_over``, and ``dose``.
    """
    grid = np.asarray(density_w_per_m2, dtype=np.float64)
    if grid.size == 0:
        raise ValueError("density_w_per_m2 must be non-empty")

    if grid.ndim == 1:
        return ExposureMetrics(
            s_max=s_max(grid),
            s_p95=s_p95(grid),
            frac_over=fraction_over_threshold(grid, s_thr),
            dose=dose_w_s_per_m2(grid, dt_seconds),
        )

    if grid.ndim == 2:
        # Score each step, then aggregate. This is deliberately explicit:
        # max/percentile/fraction are not linear, so they cannot be computed on
        # the flattened array without changing their meaning.
        per_step_max = np.array([s_max(row) for row in grid], dtype=np.float64)
        per_step_p95 = np.array([s_p95(row) for row in grid], dtype=np.float64)
        per_step_frac = np.array(
            [fraction_over_threshold(row, s_thr) for row in grid], dtype=np.float64
        )
        return ExposureMetrics(
            s_max=float(np.max(per_step_max)),
            s_p95=float(np.max(per_step_p95)),
            frac_over=float(np.max(per_step_frac)),
            dose=dose_w_s_per_m2(grid, dt_seconds),
        )

    raise ValueError(f"density_w_per_m2 must be 1-D or 2-D, got ndim={grid.ndim}")


def time_over_threshold_fraction(
    density_w_per_m2: npt.ArrayLike,
    s_thr: float,
) -> float:
    """Return the fraction of steps whose max cell density exceeds ``s_thr``.

    This is the "fraction of time over threshold" metric in the evaluation
    protocol (spec Section 8).
    """
    grid = np.asarray(density_w_per_m2, dtype=np.float64)
    if grid.size == 0:
        return 0.0
    if grid.ndim == 1:
        return 1.0 if s_max(grid) > s_thr else 0.0
    if grid.ndim == 2:
        over = np.array([s_max(row) > s_thr for row in grid], dtype=np.float64)
        return float(np.mean(over))
    raise ValueError(f"density_w_per_m2 must be 1-D or 2-D, got ndim={grid.ndim}")


__all__ = [
    "ExposureMetrics",
    "exposure_metrics_from_density",
    "time_over_threshold_fraction",
]
