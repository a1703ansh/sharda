"""Projection null steering.

Spec Section 4.1: given codebook weights ``w0`` and the steering vector ``a``
toward the flock bearing, the nulled weights are

    w = (I - a a^H / ||a||^2) w0

renormalised to preserve transmit power. This is a linear projection, not a
learned operation: the policy chooses *when* to apply it, and the geometry is
fixed here so it can be unit- and property-tested.

Two numerical details matter and are handled explicitly:

1. The projection removes the whole component of ``w0`` along ``a``, so the
   response toward the null bearing is exactly zero in exact arithmetic.
2. If ``w0`` and ``a`` are parallel (a null on the beam axis), the projection
   annihilates the beam and the surviving norm is zero. Renormalising that
   would amplify floating-point noise into a high-gain spurious pattern, so the
   annihilated vector is returned instead and callers observe ``-inf`` gain.

Angle convention: ``elevation_deg`` is elevation above the horizon and
``azimuth_deg`` is azimuth from panel boresight (see :mod:`flywaydqn.rf.array`).

Units
-----
- angles: degrees
- weights: dimensionless complex vectors
- gains: dBi
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from flywaydqn.rf.array import (
    DEFAULT_D_LAMBDA,
    beamforming_gain_dbi,
    upa_steering_vector,
)

# Below this squared norm the steering vector is treated as degenerate and the
# input weights are returned unchanged.
_EPSILON: float = 1e-12

# When the surviving weight norm drops below this fraction of the original, the
# projection is treated as annihilating the beam rather than renormalized.
_ANNIHILATION_FRACTION: float = 1e-9


def project_null(
    w0: npt.NDArray[np.complex128],
    a: npt.NDArray[np.complex128],
    *,
    epsilon: float = _EPSILON,
    renormalize: bool = True,
) -> npt.NDArray[np.complex128]:
    """Return ``w0`` projected onto the subspace orthogonal to ``a``.

        w = (I - a a^H / ||a||^2) w0

    With ``renormalize=True`` (the default, and what the spec calls for) the
    result is scaled back to ``||w0||`` so the array does not lose transmit
    power. Note that renormalising rescales the whole pattern: it preserves the
    total norm, not the per-direction gain, so gain can *rise* in directions
    where the original had a deep null. Total gain stays physical either way,
    bounded by ``10*log10(N)`` for unit-norm weights.

    With ``renormalize=False`` the projection is applied as-is, so
    ``||w|| <= ||w0||`` and gain cannot increase in any direction. Use this when
    the analysis needs the monotonicity property.

    In the degenerate case where the null sits on the beam axis the projection
    annihilates the beam; the zero vector is returned rather than renormalising
    numerical noise into a high-gain spurious pattern.

    Parameters
    ----------
    w0 : numpy.ndarray
        Original complex weight vector.
    a : numpy.ndarray
        Steering vector toward the null direction; same shape as ``w0``.
    epsilon : float
        Degenerate-norm threshold below which ``w0`` is returned unchanged.
    renormalize : bool
        Whether to restore ``||w0||`` after projecting. Default True.

    Returns
    -------
    numpy.ndarray
        Nulled weights, or zeros when the projection removes the beam entirely.
    """
    w0 = np.asarray(w0, dtype=np.complex128)
    a = np.asarray(a, dtype=np.complex128)
    if w0.shape != a.shape:
        raise ValueError(f"w0 shape {w0.shape} != steering vector shape {a.shape}")
    if w0.ndim != 1:
        raise ValueError(f"weights must be 1-D, got ndim={w0.ndim}")

    norm_sq = float(np.vdot(a, a).real)
    if norm_sq <= epsilon:
        return w0.copy()

    # P = I - a a^H / ||a||^2, applied as P @ w0 without forming an NxN matrix.
    projection = np.vdot(a, w0) / norm_sq  # scalar a^H w0 / ||a||^2
    w = w0 - projection * a

    w0_norm = float(np.linalg.norm(w0))
    if not renormalize or w0_norm == 0.0:
        return w

    w_norm = float(np.linalg.norm(w))
    if w_norm <= _ANNIHILATION_FRACTION * w0_norm:
        # The null sits on the beam axis; the beam is gone. Returning zeros
        # (rather than renormalizing noise) is what makes the -inf gain and the
        # physical-bound property both hold.
        return np.zeros_like(w)
    return w * (w0_norm / w_norm)


def null_toward(
    w0: npt.NDArray[np.complex128],
    elevation_deg: float,
    azimuth_deg: float,
    n_rows: int,
    n_cols: int,
    d_lambda: float = DEFAULT_D_LAMBDA,
    *,
    renormalize: bool = True,
) -> npt.NDArray[np.complex128]:
    """Return ``w0`` with a projection null toward ``(elevation_deg, azimuth_deg)``.

    Convenience wrapper that builds the steering vector and calls
    :func:`project_null`.
    """
    a = upa_steering_vector(n_rows, n_cols, elevation_deg, azimuth_deg, d_lambda=d_lambda)
    return project_null(w0, a, renormalize=renormalize)


def null_depth_db(
    w0: npt.NDArray[np.complex128],
    elevation_deg: float,
    azimuth_deg: float,
    n_rows: int,
    n_cols: int,
    d_lambda: float = DEFAULT_D_LAMBDA,
) -> float:
    """Return the suppression achieved toward a bearing, in dB.

    ``G_nulled - G_unnulled``, so ``-30`` means the null suppressed that
    direction by 30 dB and ``-inf`` means the beam was annihilated.
    """
    w_nulled = null_toward(w0, elevation_deg, azimuth_deg, n_rows, n_cols, d_lambda=d_lambda)
    gain_before = beamforming_gain_dbi(
        w0, elevation_deg, azimuth_deg, n_rows, n_cols, d_lambda=d_lambda
    )
    gain_after = beamforming_gain_dbi(
        w_nulled, elevation_deg, azimuth_deg, n_rows, n_cols, d_lambda=d_lambda
    )
    if gain_after == -math.inf:
        return -math.inf if gain_before != -math.inf else math.nan
    if gain_before == -math.inf:
        return math.nan
    return float(gain_after - gain_before)


def main_lobe_loss_db(
    w0: npt.NDArray[np.complex128],
    elevation_deg: float,
    azimuth_deg: float,
    null_elevation_deg: float,
    null_azimuth_deg: float,
    n_rows: int,
    n_cols: int,
    d_lambda: float = DEFAULT_D_LAMBDA,
) -> float:
    """Return how much a null degrades gain at a direction of interest, in dB.

    Negative values mean loss; ``-inf`` means the direction of interest was
    annihilated. Used by the property tests to bound main-lobe damage from a
    null placed far from the serving direction.
    """
    w_nulled = null_toward(
        w0, null_elevation_deg, null_azimuth_deg, n_rows, n_cols, d_lambda=d_lambda
    )
    gain_before = beamforming_gain_dbi(
        w0, elevation_deg, azimuth_deg, n_rows, n_cols, d_lambda=d_lambda
    )
    gain_after = beamforming_gain_dbi(
        w_nulled, elevation_deg, azimuth_deg, n_rows, n_cols, d_lambda=d_lambda
    )
    if gain_after == -math.inf:
        return -math.inf
    if gain_before == -math.inf:
        return math.nan
    return float(gain_after - gain_before)


__all__ = [
    "project_null",
    "null_toward",
    "null_depth_db",
    "main_lobe_loss_db",
]
