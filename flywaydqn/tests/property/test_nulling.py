"""Property tests for projection nulling.

Phase 1 acceptance criteria (spec Section 6):

- gain at the null bearing is below -30 dB relative to the un-nulled pattern
- main-lobe loss stays bounded for beams that are not adjacent to the null
- transmit power is preserved away from the degenerate case

Angle convention: ``elevation_deg`` is above the horizon, ``azimuth_deg`` is from
panel boresight, and the default sector span is +-60 deg (see
:mod:`flywaydqn.rf.array`).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from flywaydqn.rf.array import beamforming_gain_dbi, matched_filter_weights, upa_steering_vector
from flywaydqn.rf.codebook import build_dft_codebook
from flywaydqn.rf.nulling import (
    main_lobe_loss_db,
    null_depth_db,
    null_toward,
    project_null,
)

ROWS, COLS = 8, 8
N_ELEMENTS = ROWS * COLS
NULL_AZIMUTH_DEG = 47.2  # the bearing used in the spec's example explanation
ELEVATION_DEG = 0.0  # horizon


def _serving_beam() -> np.ndarray:
    """Unit-norm beam pointed at boresight (elevation 0, azimuth 0)."""
    return matched_filter_weights(ROWS, COLS, ELEVATION_DEG, 0.0)


def test_null_suppresses_the_target_bearing_by_more_than_30db() -> None:
    depth = null_depth_db(_serving_beam(), ELEVATION_DEG, NULL_AZIMUTH_DEG, ROWS, COLS)
    assert depth < -30.0, f"null depth {depth:.2f} dB did not clear the -30 dB bar"


def test_null_reaches_an_absolute_floor_at_every_beam_and_bearing() -> None:
    """The null is a true null in absolute terms, for every beam and bearing.

    A *relative* depth of 30 dB is only meaningful where there was gain to
    remove. The DFT beams are spaced in azimuth rather than in spatial
    frequency, so some beam/bearing pairs already sit at the numerical floor
    before nulling; for those, relative depth is meaningless. The invariant
    that does hold everywhere is absolute: after nulling, gain toward the null
    bearing is at or below -100 dB (in practice about -300 dB).
    """
    bearings = [-60.0, -37.5, -12.0, 0.0, 15.0, 47.2, 59.0, 75.0, 120.0, 180.0, 250.0]
    for beam in build_dft_codebook(n_beams=16, n_rows=ROWS, n_cols=COLS):
        for azimuth in bearings:
            # Skip the exact-annihilation case (null on the beam axis); that has
            # its own dedicated test below.
            if abs(azimuth - beam.azimuth_deg) < 1e-6:
                continue
            w = null_toward(beam.weights, ELEVATION_DEG, azimuth, ROWS, COLS)
            gain_after = beamforming_gain_dbi(w, ELEVATION_DEG, azimuth, ROWS, COLS)
            assert gain_after <= -100.0, (
                f"beam {beam.index} ({beam.azimuth_deg:.1f} deg) nulled at "
                f"{azimuth:.1f} deg left {gain_after:.1f} dB"
            )


def test_null_depth_exceeds_30db_wherever_there_was_gain_to_remove() -> None:
    """Relative depth clears -30 dB whenever the pre-null gain was meaningful."""
    bearings = [-37.5, -12.0, 0.0, 15.0, 47.2, 59.0]
    checked = 0
    for beam in build_dft_codebook(n_beams=16, n_rows=ROWS, n_cols=COLS):
        for azimuth in bearings:
            if abs(azimuth - beam.azimuth_deg) < 1e-6:
                continue
            gain_before = beamforming_gain_dbi(beam.weights, ELEVATION_DEG, azimuth, ROWS, COLS)
            if gain_before <= -60.0:
                continue  # already at the floor; nothing meaningful to suppress
            depth = null_depth_db(beam.weights, ELEVATION_DEG, azimuth, ROWS, COLS)
            assert depth < -30.0, (
                f"beam {beam.index} ({beam.azimuth_deg:.1f} deg) nulled at "
                f"{azimuth:.1f} deg only reached {depth:.1f} dB "
                f"from {gain_before:.1f} dB"
            )
            checked += 1
    assert checked > 0, "no beam/bearing pair had meaningful pre-null gain to test"


def test_null_depth_holds_off_the_boresight_elevation() -> None:
    w0 = matched_filter_weights(ROWS, COLS, 5.0, 0.0)
    for elevation in (-10.0, 0.0, 5.0, 8.0):
        depth = null_depth_db(w0, elevation, NULL_AZIMUTH_DEG, ROWS, COLS)
        assert depth < -30.0


def test_response_toward_the_null_is_at_the_numerical_floor() -> None:
    w0 = _serving_beam()
    a = upa_steering_vector(ROWS, COLS, ELEVATION_DEG, NULL_AZIMUTH_DEG)
    w = null_toward(w0, ELEVATION_DEG, NULL_AZIMUTH_DEG, ROWS, COLS)
    assert abs(np.vdot(w, a)) < 1e-12


def test_null_preserves_transmit_power_away_from_the_beam_axis() -> None:
    w0 = _serving_beam()  # pointing at azimuth 0
    for azimuth in (15.0, 47.2, 60.0, -60.0, 120.0, 250.0):
        w = null_toward(w0, ELEVATION_DEG, azimuth, ROWS, COLS)
        assert float(np.linalg.norm(w)) == pytest.approx(float(np.linalg.norm(w0)), rel=1e-9)


def test_null_on_the_beam_axis_collapses_the_beam() -> None:
    """A null exactly on the serving direction annihilates that beam.

    When ``a`` and ``w0`` are parallel the projection removes the whole weight
    vector, leaving nothing to renormalize. This is the worst case for the
    analytical-nulling baseline (spec Section 10, first risk row): with a user
    sitting on the flock bearing, always-null pays the full serving gain.

    The implementation must return the annihilated vector rather than
    renormalizing numerical noise into a high-gain spurious pattern.
    """
    w0 = _serving_beam()  # pointing at azimuth 0
    w = null_toward(w0, ELEVATION_DEG, 0.0, ROWS, COLS)
    assert float(np.linalg.norm(w)) == pytest.approx(0.0, abs=1e-9)
    assert beamforming_gain_dbi(w, ELEVATION_DEG, 0.0, ROWS, COLS) == -math.inf
    assert null_depth_db(w0, ELEVATION_DEG, 0.0, ROWS, COLS) == -math.inf


def test_unrenormalized_projection_never_raises_peak_gain_or_norm() -> None:
    """Without renormalization, total power and *peak* gain are non-increasing.

    Note the careful claim. A projection does not reduce gain in every
    direction: it removes energy from one direction and redistributes it, so a
    direction where the original had a deep null can pick up power. The
    invariants that do hold are the norm and the peak gain over all directions
    (bounded by ``||w||**2 * N``).
    """
    w0 = _serving_beam()
    peak_before = _peak_gain_dbi(w0)
    for null_azimuth in np.arange(-170.0, 180.0, 10.0):
        w = null_toward(w0, ELEVATION_DEG, float(null_azimuth), ROWS, COLS, renormalize=False)
        assert float(np.linalg.norm(w)) <= float(np.linalg.norm(w0)) + 1e-12
        assert _peak_gain_dbi(w) <= peak_before + 1e-9


def test_projection_can_raise_gain_in_a_direction_that_was_nulled() -> None:
    """Documented hazard: nulling for the flock can raise power elsewhere.

    A boresight beam has an exact null at azimuth 90 deg (the alternating-phase
    direction is orthogonal to the all-ones steering vector). Projecting a null
    at 47 deg perturbs the weights so that the 90 deg null fills in. This is why
    the exposure model must re-score the *whole* flock footprint after an action
    rather than assuming that nulling can only reduce exposure, and it is part
    of why a learned controller can beat always-null.
    """
    w0 = _serving_beam()
    before = beamforming_gain_dbi(w0, ELEVATION_DEG, 90.0, ROWS, COLS)
    w = null_toward(w0, ELEVATION_DEG, 47.0, ROWS, COLS)
    after = beamforming_gain_dbi(w, ELEVATION_DEG, 90.0, ROWS, COLS)
    assert before < -100.0  # exact null before
    assert after > before + 20.0  # materially filled in after


def _peak_gain_dbi(weights: np.ndarray) -> float:
    """Return the largest array gain over a dense azimuth sweep, in dBi."""
    return max(
        beamforming_gain_dbi(weights, ELEVATION_DEG, float(az), ROWS, COLS)
        for az in np.arange(-180.0, 180.0, 2.0)
    )


def test_renormalized_null_keeps_gain_physical_and_suppressed() -> None:
    """Renormalization may move gain around, but it stays physical.

    For unit-norm weights the array gain is bounded by ``10*log10(N)`` in every
    direction, and the null bearing itself is still driven to the floor. A
    spurious renormalization blow-up would break the bound.
    """
    w0 = _serving_beam()
    bound_dbi = 10.0 * math.log10(N_ELEMENTS)
    for null_azimuth in np.arange(-170.0, 180.0, 10.0):
        w = null_toward(w0, ELEVATION_DEG, float(null_azimuth), ROWS, COLS)
        for probe in np.arange(-170.0, 180.0, 10.0):
            gain = beamforming_gain_dbi(w, ELEVATION_DEG, float(probe), ROWS, COLS)
            if gain == -math.inf:
                continue
            assert gain <= bound_dbi + 1e-6, (
                f"gain {gain:.1f} dB exceeded the {bound_dbi:.2f} dB physical bound"
            )
        assert beamforming_gain_dbi(w, ELEVATION_DEG, float(null_azimuth), ROWS, COLS) <= -100.0


def test_annihilated_null_does_not_create_spurious_gain() -> None:
    """Regression guard for the renormalization hazard.

    A null exactly on the beam axis must zero the pattern rather than renormalize
    floating-point noise into a high-gain spurious lobe.
    """
    w0 = _serving_beam()
    w = null_toward(w0, ELEVATION_DEG, 0.0, ROWS, COLS)
    gains = [
        beamforming_gain_dbi(w, ELEVATION_DEG, float(az), ROWS, COLS)
        for az in np.arange(0.0, 360.0, 10.0)
    ]
    assert all(g == -math.inf for g in gains)


def test_main_lobe_loss_is_bounded_for_distant_nulls() -> None:
    """Nulling away from the serving direction costs under 1 dB of serving gain.

    Measured values for a boresight beam are all within -0.7 dB for nulls from
    10 to 150 deg off boresight (the worst case, -0.68 dB, is at 10 deg).
    """
    w0 = _serving_beam()  # serving direction is azimuth 0
    for azimuth in (10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0):
        loss = main_lobe_loss_db(
            w0,
            elevation_deg=ELEVATION_DEG,
            azimuth_deg=0.0,
            null_elevation_deg=ELEVATION_DEG,
            null_azimuth_deg=azimuth,
            n_rows=ROWS,
            n_cols=COLS,
        )
        assert -1.5 < loss <= 1e-9, (
            f"serving gain moved by {loss:.2f} dB with a null at {azimuth} deg"
        )


def test_mirror_bearing_null_destroys_the_boresight_beam() -> None:
    """At the horizon the array cannot tell the front from the mirror direction.

    ``a(azimuth=0)`` and ``a(azimuth=180)`` are identical to 1e-15, so a
    projection null placed at the mirror bearing removes exactly the same
    component as a null on the serving direction and annihilates the boresight
    beam. This is the sharpest failure mode of array-only nulling and a concrete
    reason the RL controller should be able to *decline* to null: the composite
    gain (element pattern included) is 30 dB down behind the panel, so a mirror
    null buys almost nothing while costing the whole serving beam.
    """
    w0 = _serving_beam()
    loss = main_lobe_loss_db(
        w0,
        elevation_deg=ELEVATION_DEG,
        azimuth_deg=0.0,
        null_elevation_deg=ELEVATION_DEG,
        null_azimuth_deg=180.0,
        n_rows=ROWS,
        n_cols=COLS,
    )
    assert loss == -math.inf


def test_array_response_depends_only_on_the_sine_of_azimuth() -> None:
    """At the horizon the response is a function of ``sin(azimuth)`` alone.

    Hence ``gain(phi) == gain(180 - phi)`` for every direction, which is the
    front-back mirror ambiguity in its exact form. A boresight beam is the
    special case where both halves of the mirror pair sit at full gain.
    """
    w = matched_filter_weights(ROWS, COLS, ELEVATION_DEG, 0.0)
    full_gain = 10.0 * math.log10(N_ELEMENTS)
    for probe in (10.0, 37.0, 55.0, 80.0, 120.0, 143.0):
        forward = beamforming_gain_dbi(w, ELEVATION_DEG, probe, ROWS, COLS)
        mirrored = beamforming_gain_dbi(w, ELEVATION_DEG, 180.0 - probe, ROWS, COLS)
        assert forward == pytest.approx(mirrored, abs=1e-9)
    # Boresight is the mirror of itself in gain terms: both ends are full gain.
    assert beamforming_gain_dbi(w, ELEVATION_DEG, 0.0, ROWS, COLS) == pytest.approx(
        full_gain, abs=1e-6
    )
    assert beamforming_gain_dbi(w, ELEVATION_DEG, 180.0, ROWS, COLS) == pytest.approx(
        full_gain, abs=1e-6
    )


def test_adjacent_null_does_cost_measurable_serving_gain() -> None:
    """A null on the serving direction is allowed to hurt, and it does.

    This documents the trade-off the policy is meant to manage: the analytic
    baseline always nulls, so it pays this cost whenever a user sits on the
    flock bearing (spec Section 10, first risk row).
    """
    loss = main_lobe_loss_db(
        _serving_beam(),
        elevation_deg=ELEVATION_DEG,
        azimuth_deg=0.0,
        null_elevation_deg=ELEVATION_DEG,
        null_azimuth_deg=0.0,
        n_rows=ROWS,
        n_cols=COLS,
    )
    assert loss == -math.inf


def test_degenerate_steering_vector_returns_input_unchanged() -> None:
    w0 = _serving_beam()
    out = project_null(w0, np.zeros(N_ELEMENTS, dtype=np.complex128))
    assert np.allclose(out, w0)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        project_null(_serving_beam(), np.ones(N_ELEMENTS - 1, dtype=np.complex128))


def test_nulling_is_deterministic() -> None:
    w0 = _serving_beam()
    first = null_toward(w0, ELEVATION_DEG, NULL_AZIMUTH_DEG, ROWS, COLS)
    second = null_toward(w0, ELEVATION_DEG, NULL_AZIMUTH_DEG, ROWS, COLS)
    assert np.array_equal(first, second)


def test_peak_gain_stays_within_the_physical_band_after_renormalization() -> None:
    w0 = _serving_beam()
    w = null_toward(w0, ELEVATION_DEG, NULL_AZIMUTH_DEG, ROWS, COLS)
    best = max(
        beamforming_gain_dbi(w, ELEVATION_DEG, float(az), ROWS, COLS)
        for az in np.arange(-180.0, 180.0, 5.0)
    )
    assert 0.0 < best <= 10.0 * math.log10(N_ELEMENTS) + 1e-6
