"""User equipment (UE) mobility.

Spec Section 4.3: user mobility is part of the environment transition. This
module provides a bounded random-waypoint model with a pedestrian/vehicular
speed mix; richer vehicular traces are optional later work.

Units
-----
- positions: meters (m)
- speed: meters per second (m/s)
- heading: degrees, counter-clockwise from +x
- time: seconds (s)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

# Speed ranges by mobility class, meters per second.
PEDESTRIAN_SPEED_MPS: tuple[float, float] = (0.5, 3.0)
VEHICULAR_SPEED_MPS: tuple[float, float] = (5.0, 25.0)

DEFAULT_AREA_M: float = 500.0


@dataclass(frozen=True)
class UserState:
    """State of one user equipment at a simulation step.

    Attributes
    ----------
    x_m, y_m : float
        Position in the sector-local frame, in meters.
    heading_deg : float
        Current heading, in degrees.
    speed_mps : float
        Current speed, in m/s.
    vehicular : bool
        Whether this user follows the vehicular speed range.
    """

    x_m: float
    y_m: float
    heading_deg: float
    speed_mps: float
    vehicular: bool = False

    def polar_from(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Return ``(bearing_deg, range_m)`` seen from a reference point."""
        dx = self.x_m - x_m
        dy = self.y_m - y_m
        return math.degrees(math.atan2(dy, dx)) % 360.0, math.hypot(dx, dy)


def generate_users(
    rng: np.random.Generator,
    n_users: int,
    *,
    min_range_m: float = 50.0,
    max_range_m: float = 400.0,
    vehicular_fraction: float = 0.3,
    area_m: float = DEFAULT_AREA_M,
) -> list[UserState]:
    """Create ``n_users`` users in an annulus around the sector.

    Positions are drawn uniformly over area in the annulus
    ``[min_range_m, max_range_m]`` centred on the origin, so users are not
    clustered near the mast. Each user is assigned a heading and a speed from
    its mobility class.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded generator; all randomness flows through it.
    n_users : int
        Number of users to create.
    min_range_m, max_range_m : float
        Radial bounds of the deployment annulus, in meters.
    vehicular_fraction : float
        Fraction of users assigned the vehicular speed range.
    area_m : float
        Side length of the square area used by :func:`generate_user_step`; stored
        only so callers share one convention.

    Returns
    -------
    list[UserState]
        Created users.
    """
    if n_users < 0:
        raise ValueError(f"n_users must be >= 0, got {n_users}")
    if min_range_m <= 0.0 or max_range_m <= min_range_m:
        raise ValueError(f"require 0 < min_range_m < max_range_m, got {min_range_m}, {max_range_m}")
    if not 0.0 <= vehicular_fraction <= 1.0:
        raise ValueError(f"vehicular_fraction must be in [0, 1], got {vehicular_fraction}")

    del area_m  # documented above; step function takes the bounds explicitly
    users: list[UserState] = []
    # Uniform over the annulus area: sample r^2 uniformly between the squared bounds.
    low_sq = min_range_m**2
    high_sq = max_range_m**2
    for _ in range(n_users):
        radius_m = math.sqrt(float(rng.uniform(low_sq, high_sq)))
        bearing_rad = float(rng.uniform(0.0, 2.0 * math.pi))
        vehicular = bool(rng.uniform() < vehicular_fraction)
        speed_range = VEHICULAR_SPEED_MPS if vehicular else PEDESTRIAN_SPEED_MPS
        users.append(
            UserState(
                x_m=radius_m * math.cos(bearing_rad),
                y_m=radius_m * math.sin(bearing_rad),
                heading_deg=float(rng.uniform(0.0, 360.0)),
                speed_mps=float(rng.uniform(*speed_range)),
                vehicular=vehicular,
            )
        )
    return users


def generate_user_step_random_waypoint(
    state: UserState,
    *,
    min_x_m: float = -DEFAULT_AREA_M,
    max_x_m: float = DEFAULT_AREA_M,
    min_y_m: float = -DEFAULT_AREA_M,
    max_y_m: float = DEFAULT_AREA_M,
    dt_seconds: float = 0.1,
    rng: np.random.Generator | None = None,
) -> UserState:
    """Advance one user by a bounded random-waypoint step.

    The user moves along its heading; on hitting the bounding box it picks a new
    heading and speed (when a seeded generator is supplied) rather than
    reflecting, which keeps it inside the box without drifting along edges.
    With ``rng=None`` the user reflects off the wall deterministically.

    Parameters
    ----------
    state : UserState
        Previous user state.
    min_x_m, max_x_m, min_y_m, max_y_m : float
        Bounding box for motion, in meters.
    dt_seconds : float
        Time step in seconds. Default 0.1.
    rng : numpy.random.Generator or None
        Optional seeded generator for waypoint resampling.

    Returns
    -------
    UserState
        Updated user state.
    """
    if dt_seconds <= 0.0:
        raise ValueError(f"dt_seconds must be > 0, got {dt_seconds}")
    if not (min_x_m < max_x_m and min_y_m < max_y_m):
        raise ValueError("bounding box must satisfy min < max on both axes")

    heading_rad = math.radians(state.heading_deg)
    new_x = state.x_m + state.speed_mps * dt_seconds * math.cos(heading_rad)
    new_y = state.y_m + state.speed_mps * dt_seconds * math.sin(heading_rad)

    if min_x_m <= new_x <= max_x_m and min_y_m <= new_y <= max_y_m:
        return replace(state, x_m=new_x, y_m=new_y)

    # Out of bounds: pick a new waypoint facing back into the area.
    if rng is not None:
        speed_range = VEHICULAR_SPEED_MPS if state.vehicular else PEDESTRIAN_SPEED_MPS
        new_speed = float(rng.uniform(*speed_range))
        new_heading = float(rng.uniform(0.0, 360.0))
    else:
        new_speed = state.speed_mps
        new_heading = (state.heading_deg + 180.0) % 360.0

    clipped_x = min(max(new_x, min_x_m), max_x_m)
    clipped_y = min(max(new_y, min_y_m), max_y_m)
    return replace(
        state,
        x_m=clipped_x,
        y_m=clipped_y,
        heading_deg=new_heading,
        speed_mps=new_speed,
    )


__all__ = [
    "UserState",
    "PEDESTRIAN_SPEED_MPS",
    "VEHICULAR_SPEED_MPS",
    "DEFAULT_AREA_M",
    "generate_users",
    "generate_user_step_random_waypoint",
]
