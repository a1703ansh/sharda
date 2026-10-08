"""Mobility: user mobility, flock generator, and track replay loaders.

Phase A (default) uses synthetic flocks, which are fully controllable and
reproducible (spec Section 9). A radar track-replay loader shares the same
interface in Phase B.
"""

from flywaydqn.mobility.flock import (
    DEFAULT_ALTITUDE_MAX_M,
    DEFAULT_ALTITUDE_MIN_M,
    DEFAULT_EXTENT_M,
    DEFAULT_FLOCK_SPEED_MPS,
    DEFAULT_N_INDIVIDUALS,
    FlockState,
    flock_footprint_cells,
    generate_flock,
    generate_flock_step,
)
from flywaydqn.mobility.users import (
    DEFAULT_AREA_M,
    PEDESTRIAN_SPEED_MPS,
    VEHICULAR_SPEED_MPS,
    UserState,
    generate_user_step_random_waypoint,
    generate_users,
)

__all__ = [
    # flock
    "DEFAULT_ALTITUDE_MAX_M",
    "DEFAULT_ALTITUDE_MIN_M",
    "DEFAULT_EXTENT_M",
    "DEFAULT_FLOCK_SPEED_MPS",
    "DEFAULT_N_INDIVIDUALS",
    "FlockState",
    "flock_footprint_cells",
    "generate_flock",
    "generate_flock_step",
    # users
    "DEFAULT_AREA_M",
    "PEDESTRIAN_SPEED_MPS",
    "VEHICULAR_SPEED_MPS",
    "UserState",
    "generate_user_step_random_waypoint",
    "generate_users",
]
