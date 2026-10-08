"""Uniform planar array (UPA) aperture, element pattern, and array factor.

Geometry and angle convention
-----------------------------
The panel is mounted vertically with its normal (boresight) horizontal, which is
the standard macro-sector mounting. In the array-local frame:

- the panel plane is the ``y``-``z`` plane and the panel normal is ``+x``
- columns (index ``n``) run along ``y`` and are the **azimuth** axis
- rows (index ``m``) run along ``z`` and are the **elevation** axis

Angles are measured from the panel boresight, not from zenith:

- ``phi_deg``: azimuth, in the horizontal plane, 0 deg = panel boresight
- ``theta_deg``: elevation above the horizon, 0 deg = horizon, 90 deg = zenith

The direction unit vector is therefore

    u = (cos(theta) cos(phi), cos(theta) sin(phi), sin(theta))

and with uniform spacing ``d_lambda`` in wavelengths the element position is
``p_mn = lambda * d_lambda * (0, n, m)``. The propagation phase is

    k . p_mn = 2*pi * d_lambda * (n * cos(theta) * sin(phi) + m * sin(theta))

Important consequence, and a deliberate modelling decision
----------------------------------------------------------
A single planar panel with half-wavelength spacing develops a grating lobe when
scanned to end-fire (90 deg off broadside). A full-circle azimuth codebook would
therefore place a full-strength lobe directly behind the sector. Real macro
panels cover a 3-sector 120 deg slice, so the default codebook span is
``[-60, +60]`` deg about boresight: within that span the visible spatial
frequency ``|sin(phi)| <= sin(60 deg) = 0.866`` stays clear of the ``sin(phi) =
+-1`` grating-lobe condition, and the 8 dBi element pattern attenuates whatever
falls outside the sector. A wider span remains available as a config option and
is documented as incurring the back lobe.

Reference: 3GPP TR 38.901 V19.4.0 (Rel-19), Table 7.3-1 (element pattern) and
Section 7.3 (array model). Scenario default is UMa (Decision D1).

Units
-----
- angles: degrees at the public API boundary
- ``d_lambda``: element spacing in wavelengths (default 0.5)
- gains: dBi
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

# TR 38.901 Table 7.3-1: maximum directional gain of a single element.
ELEMENT_MAX_GAIN_DBI: float = 8.0
# 3 dB beamwidths in the vertical and horizontal planes, degrees.
ELEMENT_THETA_3DB_DEG: float = 65.0
ELEMENT_PHI_3DB_DEG: float = 65.0
# Front-to-back ratio and side-lobe floor, dB.
ELEMENT_FRONT_BACK_DB: float = 30.0

# Half-wavelength spacing; the standard inter-element spacing for a codebook
# without grating lobes inside a +-60 deg sector.
DEFAULT_D_LAMBDA: float = 0.5


def _validate_array(n_rows: int, n_cols: int, d_lambda: float) -> None:
    if n_rows < 1 or n_cols < 1:
        raise ValueError(f"array dimensions must be >= 1, got {n_rows}x{n_cols}")
    if d_lambda <= 0.0:
        raise ValueError(f"d_lambda must be > 0, got {d_lambda}")


def _validate_angles(theta_deg: float, phi_deg: float) -> None:
    if not math.isfinite(theta_deg) or not math.isfinite(phi_deg):
        raise ValueError(f"angles must be finite, got theta={theta_deg}, phi={phi_deg}")


def upa_steering_vector(
    n_rows: int,
    n_cols: int,
    theta_deg: float,
    phi_deg: float,
    d_lambda: float = DEFAULT_D_LAMBDA,
) -> npt.NDArray[np.complex128]:
    """Return the UPA steering vector for a direction measured from boresight.

    Parameters
    ----------
    n_rows : int
        Number of rows (elements along ``z``, the elevation axis).
    n_cols : int
        Number of columns (elements along ``y``, the azimuth axis).
    theta_deg : float
        Elevation above the horizon in degrees. 0 deg = horizon/boresight
        elevation, 90 deg = zenith.
    phi_deg : float
        Azimuth from the panel boresight in degrees. 0 deg = boresight.
    d_lambda : float
        Element spacing in wavelengths. Default 0.5.

    Returns
    -------
    numpy.ndarray
        Complex steering vector of shape ``(n_rows * n_cols,)`` with
        ``||a||**2 == n_rows * n_cols``. Index ``m * n_cols + n`` is
        ``(row=m, col=n)`` (C order).
    """
    _validate_array(n_rows, n_cols, d_lambda)
    _validate_angles(theta_deg, phi_deg)

    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)

    # Spatial frequencies along the azimuth (y) and elevation (z) axes.
    v_azimuth = math.cos(theta) * math.sin(phi)
    v_elevation = math.sin(theta)

    n_idx = np.arange(n_cols, dtype=np.float64)[None, :]
    m_idx = np.arange(n_rows, dtype=np.float64)[:, None]
    phase = 2.0 * math.pi * d_lambda * (n_idx * v_azimuth + m_idx * v_elevation)

    return np.exp(1j * phase).reshape(-1)


def matched_filter_weights(
    n_rows: int,
    n_cols: int,
    theta_deg: float,
    phi_deg: float,
    d_lambda: float = DEFAULT_D_LAMBDA,
) -> npt.NDArray[np.complex128]:
    """Return unit-norm matched-filter weights pointed at ``(theta_deg, phi_deg)``.

    ``w = a / ||a||``, so ``||w|| == 1`` and the peak array gain is
    ``10 * log10(n_rows * n_cols)`` dBi.
    """
    a = upa_steering_vector(n_rows, n_cols, theta_deg, phi_deg, d_lambda=d_lambda)
    return a / float(np.linalg.norm(a))


def _gain_dbi(
    weights: npt.NDArray[np.complex128],
    a: npt.NDArray[np.complex128],
    n_rows: int,
    n_cols: int,
) -> float:
    expected = n_rows * n_cols
    weights = np.asarray(weights, dtype=np.complex128)
    if weights.shape != (expected,):
        raise ValueError(f"weights must have shape ({expected},), got {weights.shape}")
    response_sq = float(abs(np.vdot(weights, a)) ** 2)
    if response_sq <= 0.0:
        return -math.inf
    return 10.0 * math.log10(response_sq)


def beamforming_gain_dbi(
    weights: npt.NDArray[np.complex128],
    theta_deg: float,
    phi_deg: float,
    n_rows: int,
    n_cols: int,
    d_lambda: float = DEFAULT_D_LAMBDA,
) -> float:
    """Return absolute array gain of ``weights`` toward a direction, in dBi.

        G = 10 * log10( |w^H a(theta, phi)|**2 )

    Matched unit-norm weights evaluated at their own pointing direction return
    ``10 * log10(n_rows * n_cols)`` dBi, or ``-inf`` for a zero weight vector.
    """
    a = upa_steering_vector(n_rows, n_cols, theta_deg, phi_deg, d_lambda=d_lambda)
    return _gain_dbi(weights, a, n_rows, n_cols)


def upa_array_factor_dbi(
    weights: npt.NDArray[np.complex128],
    theta_deg: float,
    phi_deg: float,
    n_rows: int,
    n_cols: int,
    d_lambda: float = DEFAULT_D_LAMBDA,
) -> float:
    """Return the normalized array-factor pattern toward a direction, in dB.

    Normalized so matched weights evaluated at their own pointing direction
    return 0 dB. Use :func:`beamforming_gain_dbi` when an absolute dBi scale is
    needed, for example in exposure computation.
    """
    gain_db = beamforming_gain_dbi(weights, theta_deg, phi_deg, n_rows, n_cols, d_lambda)
    if gain_db == -math.inf:
        return -math.inf
    return gain_db - 10.0 * math.log10(float(n_rows * n_cols))


def element_pattern_dbi(elevation_deg: float, azimuth_deg: float) -> float:
    """Return the single-element gain pattern from TR 38.901 Table 7.3-1, in dBi.

    Both angles are offsets from the panel boresight, in degrees:

        A_V = -min(12 * (elevation / 65)**2, 30)     [vertical, 65 deg HPBW]
        A_H = -min(12 * (azimuth   / 65)**2, 30)     [horizontal, 65 deg HPBW]
        A   = -min(-(A_V + A_H), 30)                 [combined, 30 dB floor]
        G_E = 8 dBi + A

    The peak is 8 dBi at boresight ``(0, 0)``; the combined pattern is floored at
    30 dB below that peak.
    """
    _validate_angles(elevation_deg, azimuth_deg)
    a_v = -min(12.0 * (elevation_deg / ELEMENT_THETA_3DB_DEG) ** 2, ELEMENT_FRONT_BACK_DB)
    a_h = -min(12.0 * (azimuth_deg / ELEMENT_PHI_3DB_DEG) ** 2, ELEMENT_FRONT_BACK_DB)
    attenuation = -min(-(a_v + a_h), ELEMENT_FRONT_BACK_DB)
    return ELEMENT_MAX_GAIN_DBI + attenuation


def composite_gain_dbi(
    weights: npt.NDArray[np.complex128],
    theta_deg: float,
    phi_deg: float,
    n_rows: int,
    n_cols: int,
    d_lambda: float = DEFAULT_D_LAMBDA,
) -> float:
    """Return array gain including the per-element pattern, in dBi.

    This is the ``G(theta, phi)`` used by the exposure model (spec Section 4.2):

        G = G_element * G_array     (in decibels: G_element_dBi + G_array_dBi)
    """
    array_gain = beamforming_gain_dbi(weights, theta_deg, phi_deg, n_rows, n_cols, d_lambda)
    if array_gain == -math.inf:
        return -math.inf
    return array_gain + element_pattern_dbi(theta_deg, phi_deg)


__all__ = [
    "ELEMENT_MAX_GAIN_DBI",
    "ELEMENT_THETA_3DB_DEG",
    "ELEMENT_PHI_3DB_DEG",
    "ELEMENT_FRONT_BACK_DB",
    "DEFAULT_D_LAMBDA",
    "upa_steering_vector",
    "matched_filter_weights",
    "beamforming_gain_dbi",
    "upa_array_factor_dbi",
    "element_pattern_dbi",
    "composite_gain_dbi",
]
