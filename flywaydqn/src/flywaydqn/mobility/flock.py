"""Synthetic flock model (Phase A default; spec Section 9).

A flock is represented by a centroid, an altitude band, an extent, a heading,
and a speed. Phase A uses a synthetic generator with migratory drift; Phase B
(optional) replays radar-derived tracks through the same interface.

Units
-----
- positions and extent: meters (m)
- altitude: meters (m)
- speed: meters per second (m/s)
- heading and bearing: degrees, measured counter-clockwise from +x
- time: seconds (s)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

DEFAULT_FLOCK_SPEED_MPS: float = 12.0
DEFAULT_ALTITUDE_MIN_M: float = 20.0
DEFAULT_ALTITUDE_MAX_M: float = 80.0
DEFAULT_EXTENT_M: float = 60.0
DEFAULT_N_INDIVIDUALS: int = 40

# Maximum random heading change per step, degrees.
_MAX_HEADING_JITTER_DEG: float = 0.5


@dataclass(frozen=True)
class FlockState:
    """State of a synthetic flock at one simulation step.

    Attributes
    ----------
    centroid_x_m, centroid_y_m : float
        Flock centroid in the sector-local frame, in meters.
    centroid_heading_deg : float
        Heading of the centroid velocity, in degrees.
    speed_mps : float
        Ground speed of the centroid, in m/s.
    altitude_min_m, altitude_max_m : float
        Altitude band in meters.
    extent_m : float
        Characteristic radius of the flock footprint, in meters.
    n_individuals : int
        Number of individuals represented by this density blob.
    density_cells_w_per_m2 : numpy.ndarray
        Optional per-cell power densities populated by the exposure model. This
        field is intentionally read-only state carried alongside the geometry.
    """

    centroid_x_m: float
    centroid_y_m: float
    centroid_heading_deg: float = 45.0
    speed_mps: float = DEFAULT_FLOCK_SPEED_MPS
    altitude_min_m: float = DEFAULT_ALTITUDE_MIN_M
    altitude_max_m: float = DEFAULT_ALTITUDE_MAX_M
    extent_m: float = DEFAULT_EXTENT_M
    n_individuals: int = DEFAULT_N_INDIVIDUALS
    # compare=False: the dataclass-level equality check must not try to compare
    # numpy arrays (``arr1 == arr2`` returns an array, not a bool). Geometric
    # equality is what callers want, and the exposure model recomputes the
    # densities each step anyway.
    density_cells_w_per_m2: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), compare=False
    )

    @property
    def altitude_mean_m(self) -> float:
        """Mean altitude of the band, in meters."""
        return 0.5 * (self.altitude_min_m + self.altitude_max_m)

    @property
    def heading_rad(self) -> float:
        """Heading in radians."""
        return math.radians(self.centroid_heading_deg)

    def polar_from(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Return ``(bearing_deg, range_m)`` of the centroid seen from a point.

        Bearing is measured counter-clockwise from the +x axis and normalized
        to ``[0, 360)``.

        Parameters
        ----------
        x_m, y_m : float
            Observer position (typically the sector antenna), in meters.
        """
        dx = self.centroid_x_m - x_m
        dy = self.centroid_y_m - y_m
        range_m = math.hypot(dx, dy)
        bearing_deg = math.degrees(math.atan2(dy, dx)) % 360.0
        return bearing_deg, range_m


def generate_flock(
    rng: np.random.Generator,
    *,
    bearing_deg: float,
    range_m: float,
    speed_mps: float = DEFAULT_FLOCK_SPEED_MPS,
    altitude_min_m: float = DEFAULT_ALTITUDE_MIN_M,
    altitude_max_m: float = DEFAULT_ALTITUDE_MAX_M,
    extent_m: float = DEFAULT_EXTENT_M,
    n_individuals: int = DEFAULT_N_INDIVIDUALS,
    sector_x_m: float = 0.0,
    sector_y_m: float = 0.0,
) -> FlockState:
    """Create a flock centred at a bearing and range from the sector.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded generator; all randomness flows through it (Phase 0 requirement).
    bearing_deg : float
        Entry bearing from the sector, in degrees.
    range_m : float
        Initial range from the sector, in meters.
    speed_mps, altitude_min_m, altitude_max_m, extent_m, n_individuals
        Scenario parameters; see :class:`FlockState`.
    sector_x_m, sector_y_m : float
        Sector position, in meters.

    Returns
    -------
    FlockState
        Initialized flock. Heading is the entry bearing with a small jitter, so
        the flock drifts across the sector rather than into it.
    """
    if range_m <= 0.0:
        raise ValueError(f"range_m must be > 0, got {range_m}")
    if altitude_max_m < altitude_min_m:
        raise ValueError("altitude_max_m must be >= altitude_min_m")
    if n_individuals < 1:
        raise ValueError(f"n_individuals must be >= 1, got {n_individuals}")

    bearing_rad = math.radians(bearing_deg)
    centroid_x = sector_x_m + range_m * math.cos(bearing_rad)
    centroid_y = sector_y_m + range_m * math.sin(bearing_rad)

    heading = (bearing_deg + float(rng.uniform(-15.0, 15.0))) % 360.0

    return FlockState(
        centroid_x_m=centroid_x,
        centroid_y_m=centroid_y,
        centroid_heading_deg=heading,
        speed_mps=speed_mps,
        altitude_min_m=altitude_min_m,
        altitude_max_m=altitude_max_m,
        extent_m=extent_m,
        n_individuals=n_individuals,
    )


def generate_flock_step(
    state: FlockState,
    *,
    dt_seconds: float = 0.1,
    rng: np.random.Generator | None = None,
) -> FlockState:
    """Advance a flock by one step of migratory drift.

    The centroid translates along its heading; with a seeded generator a small
    heading jitter models natural wander. Passing ``rng=None`` makes the step
    fully deterministic, which the Phase 2 reproducibility tests rely on.

    Parameters
    ----------
    state : FlockState
        Previous flock state.
    dt_seconds : float
        Simulation time step in seconds. Default 0.1.
    rng : numpy.random.Generator or None
        Optional seeded generator for heading jitter.

    Returns
    -------
    FlockState
        Updated flock state.
    """
    if dt_seconds <= 0.0:
        raise ValueError(f"dt_seconds must be > 0, got {dt_seconds}")

    heading_deg = state.centroid_heading_deg
    if rng is not None:
        heading_deg = (
            heading_deg + float(rng.uniform(-_MAX_HEADING_JITTER_DEG, _MAX_HEADING_JITTER_DEG))
        ) % 360.0

    heading_rad = math.radians(heading_deg)
    distance_m = state.speed_mps * dt_seconds

    return replace(
        state,
        centroid_x_m=state.centroid_x_m + distance_m * math.cos(heading_rad),
        centroid_y_m=state.centroid_y_m + distance_m * math.sin(heading_rad),
        centroid_heading_deg=heading_deg,
    )


def flock_footprint_cells(
    state: FlockState,
    n_cells: int = 64,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """Return sample positions covering the flock footprint.

    Cells are drawn uniformly inside the flock extent (a disc of radius
    ``extent_m``) so the exposure model can score power density over an area
    rather than at a single point. With ``rng=None`` the cells are laid out on a
    deterministic sunflower (Vogel) spiral of the same radius, which keeps the
    footprint reproducible without a generator.

    Parameters
    ----------
    state : FlockState
        Flock whose footprint is sampled.
    n_cells : int
        Number of sample cells.
    rng : numpy.random.Generator or None
        Optional seeded generator.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_cells, 2)`` holding ``(x, y)`` positions in meters.
    """
    if n_cells < 1:
        raise ValueError(f"n_cells must be >= 1, got {n_cells}")

    if rng is None:
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        indices = np.arange(n_cells, dtype=np.float64)
        radii = state.extent_m * np.sqrt((indices + 0.5) / n_cells)
        angles = indices * golden_angle
    else:
        radii = state.extent_m * np.sqrt(rng.uniform(0.0, 1.0, size=n_cells))
        angles = rng.uniform(0.0, 2.0 * math.pi, size=n_cells)

    xs = state.centroid_x_m + radii * np.cos(angles)
    ys = state.centroid_y_m + radii * np.sin(angles)
    return np.column_stack((xs, ys))


__all__ = [
    "FlockState",
    "DEFAULT_FLOCK_SPEED_MPS",
    "DEFAULT_ALTITUDE_MIN_M",
    "DEFAULT_ALTITUDE_MAX_M",
    "DEFAULT_EXTENT_M",
    "DEFAULT_N_INDIVIDUALS",
    "generate_flock",
    "generate_flock_step",
    "flock_footprint_cells",
]
