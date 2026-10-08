"""DFT beam codebook: layout, gain, nearest-beam lookup, and span handling."""

from __future__ import annotations

import math

import pytest

from flywaydqn.rf.codebook import (
    DEFAULT_AZIMUTH_SPAN_DEG,
    HORIZON_ELEVATION_DEG,
    angular_distance_deg,
    azimuth_span_width_deg,
    beam_azimuth_deg,
    beam_index_from_azimuth,
    build_dft_codebook,
    dft_codebook_azimuth,
    normalize_azimuth,
)

ROWS, COLS = 8, 8
PEAK_DBI = 10.0 * math.log10(ROWS * COLS)


def test_default_codebook_has_sixteen_beams() -> None:
    beams = build_dft_codebook()
    assert len(beams) == 16
    assert [b.index for b in beams] == list(range(16))


def test_default_span_is_the_three_sector_macro_slice() -> None:
    assert DEFAULT_AZIMUTH_SPAN_DEG == (-60.0, 60.0)
    beams = build_dft_codebook()
    azimuths = [b.azimuth_deg for b in beams]
    assert min(azimuths) == pytest.approx(-60.0)
    assert max(azimuths) < 60.0
    assert azimuths == sorted(azimuths)


def test_beams_are_evenly_spaced_across_the_span() -> None:
    n_beams = 16
    beams = build_dft_codebook(n_beams=n_beams)
    step = 120.0 / n_beams
    for i, beam in enumerate(beams):
        assert beam.azimuth_deg == pytest.approx(-60.0 + i * step, abs=1e-9)


def test_all_beams_point_at_the_horizon_by_default() -> None:
    assert all(b.elevation_deg == HORIZON_ELEVATION_DEG for b in build_dft_codebook(n_beams=4))


def test_each_beam_peaks_at_its_own_pointing_direction() -> None:
    for beam in build_dft_codebook(n_beams=8, n_rows=ROWS, n_cols=COLS):
        gain = beam.gain_toward_dbi(beam.elevation_deg, beam.azimuth_deg, ROWS, COLS)
        assert gain == pytest.approx(PEAK_DBI, abs=1e-6)


def test_composite_beam_gain_adds_the_element_pattern() -> None:
    beam = build_dft_codebook(n_beams=8, n_rows=ROWS, n_cols=COLS)[0]
    composite = beam.composite_gain_toward_dbi(beam.elevation_deg, beam.azimuth_deg, ROWS, COLS)
    from flywaydqn.rf.array import element_pattern_dbi

    assert composite == pytest.approx(
        PEAK_DBI + element_pattern_dbi(beam.elevation_deg, beam.azimuth_deg), abs=1e-9
    )


def test_beam_weights_are_unit_norm() -> None:
    import numpy as np

    for beam in build_dft_codebook(n_beams=4, n_rows=ROWS, n_cols=COLS):
        assert float(np.linalg.norm(beam.weights)) == pytest.approx(1.0)


# --- lookup ------------------------------------------------------------------


def test_nearest_beam_index_recovers_each_beams_own_azimuth() -> None:
    n_beams = 16
    for i in range(n_beams):
        assert beam_index_from_azimuth(beam_azimuth_deg(i, n_beams), n_beams) == i


def test_nearest_beam_index_snaps_to_the_closest_grid_point() -> None:
    n_beams = 16
    step = 120.0 / n_beams
    assert beam_index_from_azimuth(-60.0 + step * 0.4, n_beams) == 0
    assert beam_index_from_azimuth(-60.0 + step * 0.6, n_beams) == 1


def test_azimuth_outside_the_span_snaps_to_the_nearer_edge_beam() -> None:
    n_beams = 16
    # Far left of the sector.
    assert beam_index_from_azimuth(-120.0, n_beams) == 0
    # Far right of the sector.
    assert beam_index_from_azimuth(150.0, n_beams) == n_beams - 1
    # Straight behind the panel is angularly equidistant; either edge is valid,
    # but it must not raise and must stay inside the codebook.
    assert beam_index_from_azimuth(180.0, n_beams) in (0, n_beams - 1)


def test_full_circle_span_behaves_circularly() -> None:
    n_beams = 16
    span = (-180.0, 180.0)
    assert azimuth_span_width_deg(span) == pytest.approx(360.0)
    for i in range(n_beams):
        azimuth = beam_azimuth_deg(i, n_beams, span)
        assert beam_index_from_azimuth(azimuth, n_beams, span) == i


def test_single_beam_codebook_is_allowed() -> None:
    beams = build_dft_codebook(n_beams=1)
    assert len(beams) == 1
    assert beam_index_from_azimuth(123.0, 1) == 0


def test_restricted_span_codebook_stays_inside_the_span() -> None:
    span = (-30.0, 30.0)
    beams = build_dft_codebook(n_beams=5, azimuth_span_deg=span)
    azimuths = [b.azimuth_deg for b in beams]
    assert min(azimuths) >= -30.0
    assert max(azimuths) < 30.0
    assert azimuths == sorted(azimuths)


def test_tuple_wrapper_matches_dataclass_codebook() -> None:
    tuples = dft_codebook_azimuth(4, n_rows=ROWS, n_cols=COLS)
    beams = build_dft_codebook(4, n_rows=ROWS, n_cols=COLS)
    assert len(tuples) == len(beams)
    for (index, azimuth, weights), beam in zip(tuples, beams, strict=True):
        assert index == beam.index
        assert azimuth == beam.azimuth_deg
        assert (weights == beam.weights).all()


# --- helpers -----------------------------------------------------------------


def test_angular_distance_wraps_at_the_boundary() -> None:
    assert angular_distance_deg(10.0, 350.0) == pytest.approx(20.0)
    assert angular_distance_deg(0.0, 180.0) == pytest.approx(180.0)
    assert angular_distance_deg(45.0, 45.0) == pytest.approx(0.0)
    assert angular_distance_deg(179.0, -179.0) == pytest.approx(2.0)


def test_normalize_azimuth_wraps_into_the_boresight_centred_range() -> None:
    assert normalize_azimuth(0.0) == pytest.approx(0.0)
    assert normalize_azimuth(370.0) == pytest.approx(10.0)
    assert normalize_azimuth(-370.0) == pytest.approx(-10.0)
    assert normalize_azimuth(180.0) == pytest.approx(180.0)
    assert normalize_azimuth(-180.0) == pytest.approx(180.0)


def test_span_width_treats_a_closed_span_as_a_full_circle() -> None:
    assert azimuth_span_width_deg((0.0, 360.0)) == pytest.approx(360.0)
    assert azimuth_span_width_deg((-60.0, 60.0)) == pytest.approx(120.0)
    assert azimuth_span_width_deg((350.0, 10.0)) == pytest.approx(20.0)


@pytest.mark.parametrize("bad_n_beams", [0, -1])
def test_invalid_beam_counts_raise(bad_n_beams: int) -> None:
    with pytest.raises(ValueError):
        build_dft_codebook(n_beams=bad_n_beams)
    with pytest.raises(ValueError):
        beam_azimuth_deg(0, bad_n_beams)
    with pytest.raises(ValueError):
        beam_index_from_azimuth(0.0, bad_n_beams)


def test_out_of_range_beam_index_raises() -> None:
    with pytest.raises(ValueError):
        beam_azimuth_deg(16, 16)
    with pytest.raises(ValueError):
        beam_azimuth_deg(-1, 16)


def test_non_finite_angles_raise() -> None:
    with pytest.raises(ValueError):
        angular_distance_deg(float("nan"), 0.0)
    with pytest.raises(ValueError):
        angular_distance_deg(0.0, float("inf"))
