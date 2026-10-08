"""3GPP TR 38.901 Urban Macro (UMa) path loss, LOS probability, shadow fading.

Reference (pinned by Decision D1 in the spec README):

    3GPP TR 38.901 V19.4.0 (Rel-19), "Study on channel model for frequencies
    from 0.5 to 100 GHz", ETSI TR 138 901 V19.4.0 (2026-07).

Scenario: Urban Macro (UMa), Tables 7.4.1-1 and 7.4.2-1. Defaults follow the
standard: ``h_BS = 25 m``, ``h_UT = 1.5 m``, effective environment height
``h_E = 1 m``.

Units
-----
- distances: meters (m)
- frequency: gigahertz (GHz) at the public API, hertz (Hz) internally
- path loss and shadow fading: decibels (dB)

Conventions
-----------
- ``PL_LOS`` is the piecewise UMa LOS path loss with the breakpoint distance
  ``d_BP``; below ``d_BP`` the slope is 22, above it 40.
- ``PL_NLOS`` is the UMa NLOS path loss (independent of distance relative to
  the breakpoint).
- The path loss returned is the median; shadow fading is a separate log-normal
  term supplied by :func:`shadow_fading_std_db`.
- These functions are deterministic. Randomness is applied by the caller so
  that seeds flow through one place (see :mod:`flywaydqn.env`).
"""

from __future__ import annotations

import math

import numpy as np

TR38901_VERSION: str = "3GPP TR 38.901 V19.4.0 (Rel-19, 2026-07)"
SCENARIO_DEFAULT: str = "UMa"

# Standard default heights, meters.
DEFAULT_H_BS_M: float = 25.0
DEFAULT_H_UT_M: float = 1.5
# Effective environment height in the UMa breakpoint formula, meters.
DEFAULT_H_E_M: float = 1.0

# Speed of light, meters per second.
_C_M_PER_S: float = 299_792_458.0

# Valid ranges from Table 7.4.1-1 for UMa, meters.
MIN_DISTANCE_M: float = 10.0
MAX_DISTANCE_M: float = 5_000.0


def _validate(d3d_m: float, fc_ghz: float) -> None:
    if not math.isfinite(d3d_m) or d3d_m <= 0.0:
        raise ValueError(f"d3d_m must be a finite value > 0, got {d3d_m}")
    if not math.isfinite(fc_ghz) or fc_ghz <= 0.0:
        raise ValueError(f"fc_ghz must be a finite value > 0, got {fc_ghz}")


def breakpoint_distance_m(
    fc_ghz: float,
    h_bs_m: float = DEFAULT_H_BS_M,
    h_ut_m: float = DEFAULT_H_UT_M,
    h_e_m: float = DEFAULT_H_E_M,
) -> float:
    """Return the UMa LOS breakpoint distance ``d_BP`` in meters.

        d_BP = 4 * h_BS' * h_UT' * f_c / c

    with ``h_BS' = h_BS - h_E`` and ``h_UT' = h_UT - h_E`` (TR 38.901
    Section 7.4.1). For the default heights at 3.5 GHz this is about 1.2 km.
    """
    _validate(1.0, fc_ghz)
    h_bs_prime = h_bs_m - h_e_m
    h_ut_prime = h_ut_m - h_e_m
    if h_bs_prime <= 0.0 or h_ut_prime <= 0.0:
        raise ValueError("h_BS and h_UT must both exceed h_E")
    return 4.0 * h_bs_prime * h_ut_prime * (fc_ghz * 1e9) / _C_M_PER_S


def tr38901_uma_los_pathloss_dbm(
    d3d_m: float,
    fc_ghz: float,
    *,
    h_bs_m: float = DEFAULT_H_BS_M,
    h_ut_m: float = DEFAULT_H_UT_M,
    h_e_m: float = DEFAULT_H_E_M,
) -> float:
    """Return UMa LOS median path loss in dB (Table 7.4.1-1).

    Below the breakpoint distance::

        PL1 = 28.0 + 22 * log10(d3D) + 20 * log10(f_c)

    At and beyond the breakpoint distance::

        PL2 = 28.0 + 40 * log10(d3D) + 20 * log10(f_c)
              - 9 * log10(d_BP**2 + (h_BS - h_UT)**2)

    The result is continuous at ``d3D == d_BP`` up to the additional
    ``- 9 * log10(...)`` correction term, which is the form given by the
    standard.

    Note: the historical name of this function ends in ``_dbm`` for
    compatibility with earlier revisions of this package. Path loss is a
    relative quantity and is reported in **dB**, not dBm.
    """
    _validate(d3d_m, fc_ghz)
    d_bp = breakpoint_distance_m(fc_ghz, h_bs_m, h_ut_m, h_e_m)

    base = 28.0 + 20.0 * math.log10(fc_ghz)
    if d3d_m <= d_bp:
        return base + 22.0 * math.log10(d3d_m)

    correction = 9.0 * math.log10(d_bp**2 + (h_bs_m - h_ut_m) ** 2)
    return base + 40.0 * math.log10(d3d_m) - correction


def tr38901_uma_nlos_pathloss_dbm(
    d3d_m: float,
    fc_ghz: float,
    *,
    h_ut_m: float = DEFAULT_H_UT_M,
) -> float:
    """Return UMa NLOS median path loss in dB (Table 7.4.1-1).

    PL = 13.54 + 39.08 * log10(d3D) + 20 * log10(f_c)
         - 0.6 * (h_UT - 1.5)
    """
    _validate(d3d_m, fc_ghz)
    return 13.54 + 39.08 * math.log10(d3d_m) + 20.0 * math.log10(fc_ghz) - 0.6 * (h_ut_m - 1.5)


def los_probability_uma(
    d3d_m: float,
    h_ut_m: float = DEFAULT_H_UT_M,
) -> float:
    """Return the UMa line-of-sight probability (Table 7.4.2-1).

        P_LOS = 1,                                          d <= 18 m
        P_LOS = [18/d + exp(-d/63) * (1 - 18/d)] *
                (1 + C'(h_UT) * (5/4) * (d/100)**3 * exp(-d/150)),  d > 18 m

    with ``C'(h_UT) = 0`` for ``h_UT <= 13 m`` and
    ``((h_UT - 13) / 10)**1.5`` otherwise.
    """
    if not math.isfinite(d3d_m) or d3d_m <= 0.0:
        raise ValueError(f"d3d_m must be a finite value > 0, got {d3d_m}")

    if d3d_m <= 18.0:
        return 1.0

    term = 18.0 / d3d_m + math.exp(-d3d_m / 63.0) * (1.0 - 18.0 / d3d_m)
    c_prime = 0.0 if h_ut_m <= 13.0 else ((h_ut_m - 13.0) / 10.0) ** 1.5
    height_factor = 1.0 + c_prime * (5.0 / 4.0) * (d3d_m / 100.0) ** 3 * math.exp(-d3d_m / 150.0)
    return min(1.0, max(0.0, term * height_factor))


def shadow_fading_std_db(los: bool, fc_ghz: float = 3.5) -> float:
    """Return the UMa log-normal shadow-fading standard deviation in dB.

    TR 38.901 Table 7.4.1-1 lists ``sigma_SF = 4.0 dB`` for UMa LOS and
    ``sigma_SF = 6.0 dB`` for UMa NLOS. ``fc_ghz`` is accepted for interface
    symmetry and future frequency-dependent extensions.
    """
    if not math.isfinite(fc_ghz) or fc_ghz <= 0.0:
        raise ValueError(f"fc_ghz must be a finite value > 0, got {fc_ghz}")
    return 4.0 if los else 6.0


def apply_shadow_fading(
    pathloss_db: float,
    los: bool,
    fc_ghz: float,
    rng: np.random.Generator,
) -> float:
    """Add a log-normal shadow-fading realisation to a median path loss.

    Parameters
    ----------
    pathloss_db : float
        Median path loss in dB.
    los : bool
        Whether the link is LOS (selects the standard deviation).
    fc_ghz : float
        Carrier frequency in GHz.
    rng : numpy.random.Generator
        Source of randomness. Passing the environment's seeded generator keeps
        all randomness on one seed (spec Section 6, Phase 0).

    Returns
    -------
    float
        Path loss in dB with shadow fading applied.
    """
    sigma = shadow_fading_std_db(los, fc_ghz)
    return pathloss_db + float(rng.normal(0.0, sigma))


def pathloss_uma(
    d3d_m: float,
    fc_ghz: float,
    *,
    los: bool | None = None,
    h_bs_m: float = DEFAULT_H_BS_M,
    h_ut_m: float = DEFAULT_H_UT_M,
    h_e_m: float = DEFAULT_H_E_M,
) -> float:
    """Return UMa path loss in dB, choosing LOS/NLOS per ``los``.

    When ``los`` is ``None`` the LOS probability is not sampled here; the LOS
    branch is used. Callers that want a probabilistic choice should sample
    against :func:`los_probability_uma` with their seeded generator and pass an
    explicit ``los``.
    """
    if los is None or los:
        return tr38901_uma_los_pathloss_dbm(
            d3d_m, fc_ghz, h_bs_m=h_bs_m, h_ut_m=h_ut_m, h_e_m=h_e_m
        )
    return tr38901_uma_nlos_pathloss_dbm(d3d_m, fc_ghz, h_ut_m=h_ut_m)
