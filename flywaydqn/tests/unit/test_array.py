"""UPA steering vector, array factor, and TR 38.901 element pattern.

Angle convention under test: ``theta_deg`` is elevation above the horizon and
``phi_deg`` is azimuth from the panel boresight (see :mod:`flywaydqn.rf.array`).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from flywaydqn.rf.array import (
    ELEMENT_FRONT_BACK_DB,
    ELEMENT_MAX_GAIN_DBI,
    beamforming_gain_dbi,
    composite_gain_dbi,
    element_pattern_dbi,
    matched_filter_weights,
    upa_array_factor_dbi,
    upa_steering_vector,
)

ROWS, COLS = 8, 8
N_ELEMENTS = ROWS * COLS


# --- steering vector ---------------------------------------------------------


def test_steering_vector_shape_and_unit_magnitude() -> None:
    a = upa_steering_vector(ROWS, COLS, 0.0, 0.0)
    assert a.shape == (N_ELEMENTS,)
    assert np.allclose(np.abs(a), 1.0)
    assert float(np.linalg.norm(a) ** 2) == pytest.approx(N_ELEMENTS)


def test_steering_vector_at_boresight_is_all_ones() -> None:
    # Boresight is (elevation 0, azimuth 0): no phase progression at all.
    assert np.allclose(upa_steering_vector(ROWS, COLS, 0.0, 0.0), 1.0 + 0.0j)


def test_steering_vector_phase_steps_along_the_azimuth_axis() -> None:
    # At the horizon, an azimuth offset of 30 deg steps the column phase by
    # 2*pi*0.5*sin(30 deg) = pi/2 rad at half-wavelength spacing.
    a = upa_steering_vector(ROWS, COLS, 0.0, 30.0).reshape(ROWS, COLS)
    expected = np.exp(1j * (math.pi / 2.0) * np.arange(COLS))
    assert np.allclose(a[0], expected)
    # No elevation term at theta = 0, so every row is identical.
    assert np.allclose(a, a[0][None, :])


def test_steering_vector_phase_steps_along_the_elevation_axis() -> None:
    # Pure elevation: 2*pi*0.5*sin(10 deg) per row, no azimuth term.
    a = upa_steering_vector(ROWS, COLS, 10.0, 0.0).reshape(ROWS, COLS)
    step = 2.0 * math.pi * 0.5 * math.sin(math.radians(10.0))
    expected = np.exp(1j * step * np.arange(ROWS))[:, None]
    assert np.allclose(a, expected)
    # No azimuth term, so every column is identical.
    assert np.allclose(a, a[:, 0][:, None])


def test_steering_vector_is_even_in_azimuth_at_the_boresight_plane() -> None:
    # sin(-phi) = -sin(phi): the pattern magnitude is symmetric about boresight.
    a_pos = upa_steering_vector(ROWS, COLS, 0.0, 35.0)
    a_neg = upa_steering_vector(ROWS, COLS, 0.0, -35.0)
    assert np.allclose(a_pos, np.conj(a_neg))


def test_matched_filter_weights_are_unit_norm() -> None:
    assert float(np.linalg.norm(matched_filter_weights(ROWS, COLS, 10.0, -25.0))) == pytest.approx(
        1.0
    )


def test_array_factor_has_a_mirror_lobe_at_the_horizon() -> None:
    """A planar panel has an inherent front/back ambiguity at the horizon.

    The array factor depends only on the spatial frequency
    ``sin(phi) * cos(theta)``, so at elevation 0 the directions ``phi`` and
    ``180 - phi`` are indistinguishable and ``phi = 180`` shows a full-strength
    array response. This is the classic front-back ambiguity of a planar
    aperture, not a modelling bug: resolving it is the job of the element
    pattern, which is asserted separately below.
    """
    w = matched_filter_weights(ROWS, COLS, 0.0, 0.0)
    assert upa_array_factor_dbi(w, 0.0, 180.0, ROWS, COLS) == pytest.approx(0.0, abs=1e-9)


def test_element_pattern_resolves_the_front_back_ambiguity() -> None:
    """With the element pattern included, the back hemisphere is suppressed.

    The array contributes 0 dB at ``phi = 180`` but the 8 dBi patch has a 30 dB
    front-to-back ratio, so the composite gain behind the panel is far below the
    boresight gain.
    """
    w = matched_filter_weights(ROWS, COLS, 0.0, 0.0)
    front = composite_gain_dbi(w, 0.0, 0.0, ROWS, COLS)
    back = composite_gain_dbi(w, 0.0, 180.0, ROWS, COLS)
    assert front == pytest.approx(18.06 + ELEMENT_MAX_GAIN_DBI, abs=0.01)
    # At the horizon the element pattern contributes exactly its 30 dB
    # front-to-back ratio, so the back lobe sits 30 dB below the front.
    assert back == pytest.approx(front - ELEMENT_FRONT_BACK_DB, abs=0.01)
    # Peeking slightly upward breaks the mirror symmetry entirely, because
    # sin(theta) != 0 then enters the elevation phase term.
    assert composite_gain_dbi(w, 10.0, 180.0, ROWS, COLS) < back


def test_scanned_beams_pick_up_no_extra_full_strength_lobe() -> None:
    """A +-60 deg span keeps |sin(phi)| clear of the grating-lobe condition.

    Grating lobes appear when the spatial frequency sweeps by an integer
    multiple of ``1/d_lambda = 2``. Inside the default span the largest
    ``|sin(phi)|`` is ``sin(60 deg) = 0.866``, so ``|v - v0| <= 1.73 < 2``. The
    test encodes that directly: over the whole circle, the only directions at
    full gain are the pointing direction and its horizon mirror (the inherent
    front-back ambiguity), i.e. no *extra* lobe appears.
    """
    for beam_azimuth in (-60.0, -30.0, 0.0, 30.0, 52.5):
        w = matched_filter_weights(ROWS, COLS, 0.0, beam_azimuth)
        mirror = ((180.0 - beam_azimuth + 180.0) % 360.0) - 180.0
        for probe in np.arange(-179.0, 180.0, 1.0):
            lobe = upa_array_factor_dbi(w, 0.0, float(probe), ROWS, COLS)
            # "Full strength" means within 1 dB of the peak. The skirt of the
            # main lobe and of the mirror lobe sit well below that, so only a
            # genuine grating lobe could appear here.
            if lobe <= -1.0:
                continue
            near_pointing = abs(((probe - beam_azimuth + 180.0) % 360.0) - 180.0)
            near_mirror = abs(((probe - mirror + 180.0) % 360.0) - 180.0)
            assert min(near_pointing, near_mirror) <= 10.0, (
                f"beam at {beam_azimuth:.0f} deg had an unexplained {lobe:.1f} dB "
                f"lobe at {probe:.0f} deg"
            )


def test_array_factor_is_normalized_to_zero_at_its_own_pointing() -> None:
    w = matched_filter_weights(ROWS, COLS, 10.0, -25.0)
    assert upa_array_factor_dbi(w, 10.0, -25.0, ROWS, COLS) == pytest.approx(0.0, abs=1e-9)


def test_array_factor_rolls_off_away_from_boresight() -> None:
    w = matched_filter_weights(ROWS, COLS, 0.0, 0.0)
    assert upa_array_factor_dbi(w, 0.0, 0.0, ROWS, COLS) == pytest.approx(0.0, abs=1e-9)
    assert upa_array_factor_dbi(w, 0.0, 40.0, ROWS, COLS) < -6.0


def test_sidelobe_floor_inside_the_sector_stays_below_the_uniform_aperture_bound() -> None:
    """Outside the main lobe, the peak sidelobe sits near the -13.2 dB bound.

    A uniformly weighted 8-element array has a half-power beamwidth of about
    0.886 * lambda / (N * d) = 0.886 / 4 = 12.7 deg, and its first sidelobe sits
    about 13.2 dB below the peak. Probes inside 1.5x the beamwidth are still in
    the main lobe, so they are excluded.
    """
    w = matched_filter_weights(ROWS, COLS, 0.0, 0.0)
    half_power_bw_deg = math.degrees(0.886 / (COLS * 0.5))
    probes = [float(az) for az in np.arange(-59.0, 60.0, 0.5) if abs(az) > 1.5 * half_power_bw_deg]
    sidelobes = [upa_array_factor_dbi(w, 0.0, az, ROWS, COLS) for az in probes]
    worst = max(sidelobes)
    assert -15.0 < worst < -9.0, f"worst in-sector sidelobe was {worst:.2f} dB"


def test_matched_filter_peak_gain_equals_the_array_size() -> None:
    w = matched_filter_weights(ROWS, COLS, 0.0, 0.0)
    gain = beamforming_gain_dbi(w, 0.0, 0.0, ROWS, COLS)
    assert gain == pytest.approx(10.0 * math.log10(N_ELEMENTS), abs=1e-9)
    assert gain == pytest.approx(18.06, abs=0.01)  # 10 log10(64)


def test_larger_arrays_give_more_gain() -> None:
    small = matched_filter_weights(4, 4, 0.0, 0.0)
    large = matched_filter_weights(8, 8, 0.0, 0.0)
    delta = beamforming_gain_dbi(large, 0.0, 0.0, 8, 8) - beamforming_gain_dbi(
        small, 0.0, 0.0, 4, 4
    )
    assert delta == pytest.approx(10.0 * math.log10(4.0), abs=1e-9)


# --- element pattern ---------------------------------------------------------


def test_element_pattern_peaks_at_boresight_with_table_gain() -> None:
    assert element_pattern_dbi(0.0, 0.0) == pytest.approx(ELEMENT_MAX_GAIN_DBI)
    assert element_pattern_dbi(90.0, 0.0) < 0.0


def test_element_pattern_is_clamped_by_the_front_to_back_floor() -> None:
    floor = ELEMENT_MAX_GAIN_DBI - ELEMENT_FRONT_BACK_DB
    assert element_pattern_dbi(200.0, 0.0) == pytest.approx(floor)
    assert element_pattern_dbi(0.0, 400.0) == pytest.approx(floor)
    assert element_pattern_dbi(-200.0, 200.0) == pytest.approx(floor)


def test_element_pattern_3db_point_matches_the_table_beamwidth() -> None:
    # At half the 65 deg HPBW the vertical attenuation is 12 * 0.5**2 = 3 dB.
    half_bw = 32.5
    attenuation = element_pattern_dbi(half_bw, 0.0) - ELEMENT_MAX_GAIN_DBI
    assert attenuation == pytest.approx(-3.0, abs=1e-9)
    assert element_pattern_dbi(0.0, half_bw) - ELEMENT_MAX_GAIN_DBI == pytest.approx(-3.0, abs=1e-9)


def test_element_pattern_combines_both_planes_within_the_floor() -> None:
    combined = element_pattern_dbi(40.0, 40.0)
    # -min(12*(40/65)^2, 30) each = -4.545 dB, sum -9.09 dB, floored at -30 dB.
    assert combined == pytest.approx(ELEMENT_MAX_GAIN_DBI - 2 * 12.0 * (40.0 / 65.0) ** 2, abs=1e-9)


def test_composite_gain_is_array_plus_element() -> None:
    w = matched_filter_weights(ROWS, COLS, 5.0, -20.0)
    array_gain = beamforming_gain_dbi(w, 5.0, -20.0, ROWS, COLS)
    element_gain = element_pattern_dbi(5.0, -20.0)
    assert composite_gain_dbi(w, 5.0, -20.0, ROWS, COLS) == pytest.approx(
        array_gain + element_gain, abs=1e-9
    )


# --- validation --------------------------------------------------------------


def test_wrong_weight_shape_raises() -> None:
    with pytest.raises(ValueError):
        beamforming_gain_dbi(np.ones(7, dtype=complex), 0.0, 0.0, ROWS, COLS)


@pytest.mark.parametrize(
    ("rows", "cols", "d_lambda"),
    [(0, 8, 0.5), (8, 0, 0.5), (8, 8, 0.0), (8, 8, -0.5)],
)
def test_invalid_array_geometry_raises(rows: int, cols: int, d_lambda: float) -> None:
    with pytest.raises(ValueError):
        upa_steering_vector(rows, cols, 0.0, 0.0, d_lambda=d_lambda)


def test_non_finite_angles_raise() -> None:
    with pytest.raises(ValueError):
        upa_steering_vector(ROWS, COLS, float("nan"), 0.0)
    with pytest.raises(ValueError):
        element_pattern_dbi(0.0, float("inf"))
