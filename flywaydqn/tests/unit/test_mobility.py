"""Flock and user mobility: geometry, bounds, and seed determinism."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flywaydqn.mobility.flock import (
    FlockState,
    flock_footprint_cells,
    generate_flock,
    generate_flock_step,
)
from flywaydqn.mobility.users import (
    UserState,
    generate_user_step_random_waypoint,
    generate_users,
)

# --- flock -------------------------------------------------------------------


def test_generate_flock_places_centroid_at_the_requested_bearing_and_range() -> None:
    rng = np.random.default_rng(11)
    flock = generate_flock(rng, bearing_deg=45.0, range_m=200.0)
    bearing, rng_m = flock.polar_from(0.0, 0.0)
    assert bearing == pytest.approx(45.0, abs=1e-9)
    assert rng_m == pytest.approx(200.0, rel=1e-12)
    assert flock.centroid_x_m == pytest.approx(200.0 / math.sqrt(2.0), rel=1e-9)


def test_generate_flock_is_deterministic_for_a_seed() -> None:
    first = generate_flock(np.random.default_rng(42), bearing_deg=30.0, range_m=150.0)
    second = generate_flock(np.random.default_rng(42), bearing_deg=30.0, range_m=150.0)
    assert first == second


def test_flock_step_moves_along_heading_deterministically() -> None:
    flock = FlockState(centroid_x_m=0.0, centroid_y_m=0.0, centroid_heading_deg=0.0, speed_mps=10.0)
    after = generate_flock_step(flock, dt_seconds=0.1, rng=None)
    # 10 m/s for 0.1 s along +x is 1 m.
    assert after.centroid_x_m == pytest.approx(1.0, rel=1e-12)
    assert after.centroid_y_m == pytest.approx(0.0, abs=1e-12)
    assert after.centroid_heading_deg == pytest.approx(0.0)


def test_flock_step_with_seed_reproduces_the_same_trajectory() -> None:
    def rollout(seed: int) -> list[tuple[float, float]]:
        rng = np.random.default_rng(seed)
        state = generate_flock(rng, bearing_deg=90.0, range_m=300.0)
        track = []
        for _ in range(50):
            state = generate_flock_step(state, dt_seconds=0.1, rng=rng)
            track.append((state.centroid_x_m, state.centroid_y_m))
        return track

    assert rollout(7) == rollout(7)
    assert rollout(7) != rollout(8)


def test_flock_step_requires_positive_dt() -> None:
    flock = FlockState(centroid_x_m=0.0, centroid_y_m=0.0)
    with pytest.raises(ValueError):
        generate_flock_step(flock, dt_seconds=0.0)


def test_altitude_mean_is_the_midpoint() -> None:
    flock = FlockState(centroid_x_m=0.0, centroid_y_m=0.0, altitude_min_m=20.0, altitude_max_m=80.0)
    assert flock.altitude_mean_m == pytest.approx(50.0)


def test_generate_flock_rejects_invalid_scenarios() -> None:
    with pytest.raises(ValueError):
        generate_flock(np.random.default_rng(0), bearing_deg=0.0, range_m=0.0)
    with pytest.raises(ValueError):
        generate_flock(
            np.random.default_rng(0),
            bearing_deg=0.0,
            range_m=100.0,
            altitude_min_m=90.0,
            altitude_max_m=10.0,
        )
    with pytest.raises(ValueError):
        generate_flock(np.random.default_rng(0), bearing_deg=0.0, range_m=100.0, n_individuals=0)


# --- flock footprint ---------------------------------------------------------


def test_footprint_has_the_requested_cell_count_and_stays_inside_the_extent() -> None:
    flock = FlockState(centroid_x_m=10.0, centroid_y_m=-5.0, extent_m=50.0)
    cells = flock_footprint_cells(flock, n_cells=64)
    assert cells.shape == (64, 2)
    offsets = np.linalg.norm(cells - np.array([10.0, -5.0]), axis=1)
    assert np.all(offsets <= 50.0 + 1e-9)
    # The sunflower layout should actually fill the disc, not cluster at the centre.
    assert offsets.max() > 0.5 * 50.0


def test_footprint_is_reproducible_without_a_generator() -> None:
    flock = FlockState(centroid_x_m=0.0, centroid_y_m=0.0, extent_m=30.0)
    assert np.array_equal(flock_footprint_cells(flock, 32), flock_footprint_cells(flock, 32))


def test_footprint_with_generator_is_seed_deterministic() -> None:
    flock = FlockState(centroid_x_m=0.0, centroid_y_m=0.0, extent_m=30.0)
    a = flock_footprint_cells(flock, 32, rng=np.random.default_rng(5))
    b = flock_footprint_cells(flock, 32, rng=np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_footprint_rejects_empty_request() -> None:
    with pytest.raises(ValueError):
        flock_footprint_cells(FlockState(centroid_x_m=0.0, centroid_y_m=0.0), n_cells=0)


# --- users -------------------------------------------------------------------


def test_generate_users_respects_the_annulus_and_count() -> None:
    users = generate_users(
        np.random.default_rng(3), n_users=20, min_range_m=50.0, max_range_m=400.0
    )
    assert len(users) == 20
    ranges = [math.hypot(u.x_m, u.y_m) for u in users]
    assert min(ranges) >= 50.0 - 1e-9
    assert max(ranges) <= 400.0 + 1e-9


def test_generate_users_is_deterministic_for_a_seed() -> None:
    first = generate_users(np.random.default_rng(99), n_users=10)
    second = generate_users(np.random.default_rng(99), n_users=10)
    assert first == second


def test_generate_users_assigns_a_mobility_mix() -> None:
    users = generate_users(np.random.default_rng(1), n_users=200, vehicular_fraction=1.0)
    assert all(u.vehicular for u in users)
    assert all(u.speed_mps >= 5.0 for u in users)

    pedestrians = generate_users(np.random.default_rng(1), n_users=200, vehicular_fraction=0.0)
    assert all(not u.vehicular for u in pedestrians)
    assert all(u.speed_mps <= 3.0 for u in pedestrians)


def test_generate_users_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        generate_users(np.random.default_rng(0), n_users=-1)
    with pytest.raises(ValueError):
        generate_users(np.random.default_rng(0), n_users=1, min_range_m=400.0, max_range_m=50.0)
    with pytest.raises(ValueError):
        generate_users(np.random.default_rng(0), n_users=1, vehicular_fraction=1.5)


def test_user_step_stays_in_bounds_and_moves() -> None:
    user = UserState(x_m=0.0, y_m=0.0, heading_deg=0.0, speed_mps=10.0)
    after = generate_user_step_random_waypoint(
        user, min_x_m=-100.0, max_x_m=100.0, min_y_m=-100.0, max_y_m=100.0, dt_seconds=0.1
    )
    assert after.x_m == pytest.approx(1.0, rel=1e-12)
    assert after.y_m == pytest.approx(0.0, abs=1e-12)


def test_user_step_turns_around_at_the_wall_without_a_generator() -> None:
    user = UserState(x_m=99.99, y_m=0.0, heading_deg=0.0, speed_mps=10.0)
    after = generate_user_step_random_waypoint(
        user, min_x_m=-100.0, max_x_m=100.0, min_y_m=-100.0, max_y_m=100.0, dt_seconds=0.1
    )
    assert -100.0 <= after.x_m <= 100.0
    assert after.heading_deg == pytest.approx(180.0)


def test_user_step_is_well_behaved_over_a_long_rollout() -> None:
    rng = np.random.default_rng(2024)
    state = UserState(x_m=0.0, y_m=0.0, heading_deg=30.0, speed_mps=15.0, vehicular=True)
    for _ in range(2000):
        state = generate_user_step_random_waypoint(
            state, min_x_m=-200.0, max_x_m=200.0, min_y_m=-200.0, max_y_m=200.0, rng=rng
        )
        assert -200.0 - 1e-9 <= state.x_m <= 200.0 + 1e-9
        assert -200.0 - 1e-9 <= state.y_m <= 200.0 + 1e-9


def test_user_step_rejects_invalid_box_and_dt() -> None:
    user = UserState(x_m=0.0, y_m=0.0, heading_deg=0.0, speed_mps=10.0)
    with pytest.raises(ValueError):
        generate_user_step_random_waypoint(user, dt_seconds=0.0)
    with pytest.raises(ValueError):
        generate_user_step_random_waypoint(user, min_x_m=10.0, max_x_m=-10.0)
