"""Exposure thresholds and penalty functions.

Spec (Section 4.4, Decision D3): exposure penalty is a softplus hinge on
`S_p95 / S_thr`, with kappa = 5 and per-step reward clipped to [-10, 10].

Penalty form:
    x_t = S_p95 / S_thr
    Ê_t = (1 / kappa) * log(1 + exp(kappa * (x_t - 1)))

At x = 0, Ê_t ≈ 0.0013 for kappa = 5.

Default threshold anchor: ICNIRP 2020 general-public reference level for
2–300 GHz (10 W/m² = 1 mW/cm²). The threshold is configurable; this module does
not assert avian harm thresholds.

Units:
- power density: W/m² unless noted otherwise
- threshold: same unit as the power density passed in
- penalty: dimensionless (normalized)
"""

from __future__ import annotations

import math


def exposure_penalty_softplus(
    s_p95: float,
    s_thr: float,
    *,
    kappa: float = 5.0,
) -> float:
    """Return the softplus-hinge exposure penalty for a single step.

    Parameters
    ----------
    s_p95 : float
        95th-percentile power density over flock cells at this step, in the
        same units used for `s_thr`.
    s_thr : float
        Exposure threshold (configurable). For the default anchor this is the
        ICNIRP 2020 general-public reference level for 2–300 GHz:
        10 W/m² = 1 mW/cm².
    kappa : float
        Softplus steepness. Default 5.0 per Decision D3.

    Returns
    -------
    float
        Dimensionless exposure penalty Ê_t.
    """
    if s_thr <= 0.0:
        raise ValueError(f"s_thr must be > 0, got {s_thr}")
    if kappa <= 0.0:
        raise ValueError(f"kappa must be > 0, got {kappa}")

    x = s_p95 / s_thr
    # Numerically stable softplus for the term (kappa*(x-1)).
    arg = kappa * (x - 1.0)
    if arg > 50.0:
        softplus = arg
    elif arg < -50.0:
        softplus = 0.0
    else:
        softplus = math.log(1.0 + math.exp(arg))
    return softplus / kappa


def icnirp_public_reference_level_2020_w_per_m2() -> float:
    """Return the ICNIRP 2020 general-public reference level for 2–300 GHz.

    This is the default regulatory anchor for the exposure threshold, not a claim
    of an avian-harm threshold.

    Returns
    -------
    float
        Reference level in W/m² (10 W/m² = 1 mW/cm²).
    """
    return 10.0
