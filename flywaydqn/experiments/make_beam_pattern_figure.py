"""Reproduce beam patterns with and without a null (Phase 1 acceptance).

Spec Section 6, Phase 1: "a plotting script reproduces beam patterns (with and
without null) as a figure."

Run it with ``make figures`` or directly::

    python -m experiments.make_beam_pattern_figure

Outputs, under ``experiments/figures/``:

- ``beam_patterns_azimuth.png``: azimuth cuts of the composite gain for a
  boresight beam, a scanned beam, and the scanned beam with a projection null
  toward the flock bearing, plus the array-factor-only curves for comparison.
- ``beam_patterns_elevation.png``: elevation cuts showing the element pattern
  roll-off and the front-to-back behaviour.

All curves are computed from the library code; nothing here is hard-coded. The
script also prints a few measured numbers so the figure is self-describing.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from flywaydqn.rf.array import (  # noqa: E402
    composite_gain_dbi,
    element_pattern_dbi,
    upa_array_factor_dbi,
)
from flywaydqn.rf.codebook import (  # noqa: E402
    DEFAULT_AZIMUTH_SPAN_DEG,
    build_dft_codebook,
)
from flywaydqn.rf.nulling import null_toward  # noqa: E402

ROWS, COLS = 8, 8
FLOCK_BEARING_DEG = 47.2  # the bearing used in the spec's example explanation
ELEVATION_DEG = 0.0
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"


def _azimuth_cut(
    weights: np.ndarray, azimuths_deg: np.ndarray, *, composite: bool = True
) -> np.ndarray:
    """Return gain in dBi along an azimuth cut at zero elevation."""
    function = composite_gain_dbi if composite else upa_array_factor_dbi
    return np.array(
        [function(weights, ELEVATION_DEG, float(azimuth), ROWS, COLS) for azimuth in azimuths_deg]
    )


def make_azimuth_figure() -> tuple[plt.Figure, dict[str, float]]:
    """Render the azimuth-cut figure and return it with a small metrics dict."""
    azimuths = np.arange(-180.0, 180.0, 0.5)

    beams = build_dft_codebook(n_beams=16, n_rows=ROWS, n_cols=COLS)
    boresight = beams[8]  # the beam whose grid azimuth is 0 deg
    scanned = beams[14]  # near the positive edge of the sector

    nulled = null_toward(scanned.weights, ELEVATION_DEG, FLOCK_BEARING_DEG, ROWS, COLS)

    boresight_composite = _azimuth_cut(boresight.weights, azimuths)
    scanned_composite = _azimuth_cut(scanned.weights, azimuths)
    nulled_composite = _azimuth_cut(nulled, azimuths)
    scanned_array_only = _azimuth_cut(scanned.weights, azimuths, composite=False)
    nulled_array_only = _azimuth_cut(nulled, azimuths, composite=False)

    figure, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13.0, 5.2))

    # Left: composite gain (array + element pattern), the quantity the exposure
    # model actually uses.
    ax_left.plot(
        azimuths, boresight_composite, label=f"boresight beam ({boresight.azimuth_deg:.1f}°)"
    )
    ax_left.plot(azimuths, scanned_composite, label=f"scanned beam ({scanned.azimuth_deg:.1f}°)")
    ax_left.plot(
        azimuths,
        nulled_composite,
        label=f"scanned beam + null at {FLOCK_BEARING_DEG:.1f}°",
        linestyle="--",
    )
    ax_left.axvline(FLOCK_BEARING_DEG, color="crimson", linewidth=0.8, alpha=0.7)
    ax_left.axvspan(*DEFAULT_AZIMUTH_SPAN_DEG, color="gray", alpha=0.08)
    ax_left.set_title("Composite gain (array + TR 38.901 element pattern)")
    ax_left.set_xlabel("azimuth (deg from boresight)")
    ax_left.set_ylabel("gain (dBi)")
    ax_left.set_ylim(-25.0, 30.0)
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(fontsize=8, loc="lower center")

    # Right: array factor only, to make the mirror ambiguity visible.
    ax_right.plot(azimuths, scanned_array_only, label="scanned beam, array factor")
    ax_right.plot(
        azimuths, nulled_array_only, label="scanned beam + null, array factor", linestyle="--"
    )
    ax_right.axvline(FLOCK_BEARING_DEG, color="crimson", linewidth=0.8, alpha=0.7)
    mirror = ((180.0 - FLOCK_BEARING_DEG + 180.0) % 360.0) - 180.0
    ax_right.axvline(mirror, color="darkorange", linewidth=0.8, alpha=0.7, linestyle=":")
    ax_right.set_title("Array factor only (front/back mirror visible)")
    ax_right.set_xlabel("azimuth (deg from boresight)")
    ax_right.set_ylabel("array factor (dB)")
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(fontsize=8, loc="lower center")

    figure.suptitle(
        "FlywayDQN beam patterns: 8x8 UPA, 3.5 GHz, 0.5λ spacing, "
        "TR 38.901 UMa (element pattern 8 dBi / 65° HPBW)"
    )
    figure.tight_layout()

    # Measured numbers, so the figure caption need not hard-code anything.
    metrics = {
        "null_depth_at_flock_db": float(
            dbi_at(nulled_array_only, azimuths, FLOCK_BEARING_DEG)
            - dbi_at(scanned_array_only, azimuths, FLOCK_BEARING_DEG)
        ),
        "serving_cost_db": float(
            dbi_at(nulled_array_only, azimuths, scanned.azimuth_deg)
            - dbi_at(scanned_array_only, azimuths, scanned.azimuth_deg)
        ),
        "boresight_peak_dbi": float(np.max(boresight_composite)),
        "back_lobe_dbi": float(dbi_at(boresight_composite, azimuths, 180.0)),
        "element_peak_dbi": float(element_pattern_dbi(0.0, 0.0)),
    }
    return figure, metrics


def dbi_at(curve: np.ndarray, azimuths: np.ndarray, azimuth_deg: float) -> float:
    """Return the curve value at the sample nearest an azimuth."""
    index = int(np.argmin(np.abs(azimuths - azimuth_deg)))
    return float(curve[index])


def make_elevation_figure() -> plt.Figure:
    """Render an elevation cut showing the element pattern and array roll-off."""
    elevations = np.arange(-90.0, 90.0, 0.5)
    beams = build_dft_codebook(n_beams=16, n_rows=ROWS, n_cols=COLS)
    boresight = beams[8]

    element = np.array([element_pattern_dbi(float(e), 0.0) for e in elevations])
    boresight_array = np.array(
        [upa_array_factor_dbi(boresight.weights, float(e), 0.0, ROWS, COLS) for e in elevations]
    )
    boresight_composite = np.array(
        [composite_gain_dbi(boresight.weights, float(e), 0.0, ROWS, COLS) for e in elevations]
    )

    figure, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(elevations, element, label="element pattern (TR 38.901 Table 7.3-1)")
    ax.plot(elevations, boresight_array, label="array factor (8x8, d=0.5λ)")
    ax.plot(elevations, boresight_composite, label="composite gain")
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_title("Elevation cut at boresight azimuth")
    ax.set_xlabel("elevation above horizon (deg)")
    ax.set_ylabel("gain (dBi / dB)")
    ax.set_ylim(-40.0, 24.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    figure.tight_layout()
    return figure


def main() -> None:
    """Write both figures and print the measured numbers."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    azimuth_figure, metrics = make_azimuth_figure()
    azimuth_path = OUTPUT_DIR / "beam_patterns_azimuth.png"
    azimuth_figure.savefig(azimuth_path, dpi=140)

    elevation_figure = make_elevation_figure()
    elevation_path = OUTPUT_DIR / "beam_patterns_elevation.png"
    elevation_figure.savefig(elevation_path, dpi=140)

    print(f"wrote {azimuth_path}")
    print(f"wrote {elevation_path}")
    print("measured from the rendered curves:")
    for key, value in metrics.items():
        if math.isfinite(value):
            print(f"  {key} = {value:.2f}")
        else:
            print(f"  {key} = {value}")


if __name__ == "__main__":  # pragma: no cover
    main()
