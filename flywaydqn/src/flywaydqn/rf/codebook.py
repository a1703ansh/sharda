"""DFT beam codebook for azimuth beam selection.

Spec Section 4.1: a DFT codebook with ``K`` beams (default 16 azimuth x 1
elevation tilt; configurable).

Layout
------
Beams sit on an even grid across ``azimuth_span_deg``. The default span is a
3-sector macro slice, ``[-60, +60]`` deg about the panel boresight: scanning a
half-wavelength planar array to +-90 deg would put a full-strength grating lobe
behind the sector (see :mod:`flywaydqn.rf.array` for the derivation). A wider
span is available as a config option, and the element pattern attenuates
off-sector energy.

Angles follow the array convention: azimuth is measured from boresight in the
horizontal plane, elevation is measured above the horizon.

Units
-----
- angles: degrees
- ``d_lambda``: element spacing in wavelengths
- gain: dBi
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from flywaydqn.rf.array import (
    DEFAULT_D_LAMBDA,
    beamforming_gain_dbi,
    composite_gain_dbi,
    matched_filter_weights,
    upa_steering_vector,
)

# Default 3-sector macro slice: +-60 deg about boresight.
DEFAULT_AZIMUTH_SPAN_DEG: tuple[float, float] = (-60.0, 60.0)
FULL_CIRCLE_SPAN_DEG: tuple[float, float] = (-180.0, 180.0)
# Horizon; elevation is measured above the horizon, so a non-tilted beam is 0 deg.
HORIZON_ELEVATION_DEG: float = 0.0
DEFAULT_ELEVATION_DEG: float = HORIZON_ELEVATION_DEG


@dataclass(frozen=True)
class Beam:
    """One codebook entry.

    Attributes
    ----------
    index : int
        Beam index in ``[0, n_beams)``.
    azimuth_deg : float
        Azimuth pointing direction, in degrees from boresight.
    elevation_deg : float
        Elevation pointing direction, in degrees above the horizon.
    weights : numpy.ndarray
        Unit-norm complex weight vector of length ``n_rows * n_cols``.
    """

    index: int
    azimuth_deg: float
    elevation_deg: float
    weights: npt.NDArray[np.complex128]

    def gain_toward_dbi(
        self,
        elevation_deg: float,
        azimuth_deg: float,
        n_rows: int,
        n_cols: int,
        d_lambda: float = DEFAULT_D_LAMBDA,
    ) -> float:
        """Return this beam's absolute array gain toward a direction, in dBi."""
        return beamforming_gain_dbi(
            self.weights, elevation_deg, azimuth_deg, n_rows, n_cols, d_lambda=d_lambda
        )

    def composite_gain_toward_dbi(
        self,
        elevation_deg: float,
        azimuth_deg: float,
        n_rows: int,
        n_cols: int,
        d_lambda: float = DEFAULT_D_LAMBDA,
    ) -> float:
        """Return array gain including the element pattern, in dBi."""
        return composite_gain_dbi(
            self.weights, elevation_deg, azimuth_deg, n_rows, n_cols, d_lambda=d_lambda
        )


def build_dft_codebook(
    n_beams: int = 16,
    n_rows: int = 8,
    n_cols: int = 8,
    d_lambda: float = DEFAULT_D_LAMBDA,
    *,
    elevation_deg: float = DEFAULT_ELEVATION_DEG,
    azimuth_span_deg: tuple[float, float] = DEFAULT_AZIMUTH_SPAN_DEG,
) -> list[Beam]:
    """Return a DFT codebook of ``n_beams`` azimuth beams.

    Beams are laid out at ``span_start + span * i / n_beams`` for ``i`` in
    ``[0, n_beams)``; the span is half-open, so the span width divides the
    beams evenly and no two beams coincide (except for a full-circle span,
    where the grid is interpreted modulo 360 deg).

    Parameters
    ----------
    n_beams : int
        Number of beams, ``K``. Must be >= 1.
    n_rows, n_cols : int
        Array dimensions.
    d_lambda : float
        Element spacing in wavelengths.
    elevation_deg : float
        Elevation pointing angle shared by all beams. Default 0 deg (horizon).
    azimuth_span_deg : tuple of float
        ``(start, stop)`` azimuth span, in degrees. Default ``(-60, 60)``.

    Returns
    -------
    list[Beam]
        Codebook ordered by beam index.
    """
    if n_beams < 1:
        raise ValueError(f"n_beams must be >= 1, got {n_beams}")

    start, _stop = azimuth_span_deg
    span = azimuth_span_width_deg(azimuth_span_deg)

    beams: list[Beam] = []
    for i in range(n_beams):
        azimuth_deg = normalize_azimuth(start + span * i / n_beams)
        weights = matched_filter_weights(
            n_rows, n_cols, elevation_deg, azimuth_deg, d_lambda=d_lambda
        )
        beams.append(
            Beam(
                index=i,
                azimuth_deg=azimuth_deg,
                elevation_deg=elevation_deg,
                weights=weights,
            )
        )
    return beams


def azimuth_span_width_deg(azimuth_span_deg: tuple[float, float]) -> float:
    """Return the positive width of an azimuth span, in degrees.

    A span that closes on itself (for example ``(-180, 180)``) is treated as a
    full 360 deg circle rather than a zero-width span.
    """
    start, stop = azimuth_span_deg
    width = (stop - start) % 360.0
    return width if width > 0.0 else 360.0


def normalize_azimuth(azimuth_deg: float) -> float:
    """Wrap an azimuth into ``(-180, +180]`` degrees, preserving boresight at 0."""
    wrapped = (azimuth_deg + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def dft_codebook_azimuth(
    n_beams: int,
    n_rows: int = 8,
    n_cols: int = 8,
    d_lambda: float = DEFAULT_D_LAMBDA,
    *,
    azimuth_range_deg: tuple[float, float] = DEFAULT_AZIMUTH_SPAN_DEG,
) -> list[tuple[int, float, npt.NDArray[np.complex128]]]:
    """Return the codebook as ``(index, azimuth_deg, weights)`` tuples.

    Compatibility wrapper around :func:`build_dft_codebook` for callers that
    prefer plain tuples.
    """
    beams = build_dft_codebook(
        n_beams, n_rows, n_cols, d_lambda, azimuth_span_deg=azimuth_range_deg
    )
    return [(b.index, b.azimuth_deg, b.weights) for b in beams]


def beam_azimuth_deg(
    index: int,
    n_beams: int,
    azimuth_span_deg: tuple[float, float] = DEFAULT_AZIMUTH_SPAN_DEG,
) -> float:
    """Return the grid azimuth for a beam index, in degrees from boresight."""
    if n_beams < 1:
        raise ValueError(f"n_beams must be >= 1, got {n_beams}")
    if not 0 <= index < n_beams:
        raise ValueError(f"beam index {index} out of range [0, {n_beams})")
    start, _stop = azimuth_span_deg
    span = azimuth_span_width_deg(azimuth_span_deg)
    return normalize_azimuth(start + span * index / n_beams)


def beam_index_from_azimuth(
    azimuth_deg: float,
    n_beams: int,
    azimuth_span_deg: tuple[float, float] = DEFAULT_AZIMUTH_SPAN_DEG,
) -> int:
    """Return the nearest codebook beam index for a target azimuth.

    The target is compared against the same half-open grid used by
    :func:`build_dft_codebook`, with azimuths outside the span snapped to the
    nearest end beam. This is what the analytical-nulling baseline uses to pick
    a serving beam for a flock bearing.
    """
    if n_beams < 1:
        raise ValueError(f"n_beams must be >= 1, got {n_beams}")

    start, _stop = azimuth_span_deg
    span = azimuth_span_width_deg(azimuth_span_deg)
    offset = (azimuth_deg - start) % 360.0
    if span < 360.0 and offset > span:
        # Outside the sector: snap to whichever edge beam is angularly closer.
        to_start = angular_distance_deg(azimuth_deg, start)
        to_stop = angular_distance_deg(azimuth_deg, start + span)
        return 0 if to_start <= to_stop else n_beams - 1

    index = int(round(offset / span * n_beams)) % n_beams
    return index


def angular_distance_deg(a_deg: float, b_deg: float) -> float:
    """Return the shortest angular distance between two bearings, in degrees."""
    if not math.isfinite(a_deg) or not math.isfinite(b_deg):
        raise ValueError(f"angles must be finite, got {a_deg}, {b_deg}")
    return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)


def steering_vector_toward(
    elevation_deg: float,
    azimuth_deg: float,
    n_rows: int = 8,
    n_cols: int = 8,
    d_lambda: float = DEFAULT_D_LAMBDA,
) -> npt.NDArray[np.complex128]:
    """Return the raw (un-normalized) steering vector toward a direction.

    Convenience re-export so callers building nulls need only one import.
    """
    return upa_steering_vector(n_rows, n_cols, elevation_deg, azimuth_deg, d_lambda=d_lambda)


__all__ = [
    "Beam",
    "DEFAULT_AZIMUTH_SPAN_DEG",
    "FULL_CIRCLE_SPAN_DEG",
    "HORIZON_ELEVATION_DEG",
    "DEFAULT_ELEVATION_DEG",
    "build_dft_codebook",
    "dft_codebook_azimuth",
    "azimuth_span_width_deg",
    "normalize_azimuth",
    "beam_azimuth_deg",
    "beam_index_from_azimuth",
    "angular_distance_deg",
    "steering_vector_toward",
]
