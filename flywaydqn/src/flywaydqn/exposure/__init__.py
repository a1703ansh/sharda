"""Exposure: power density, per-step metrics, and threshold-based penalties.

Spec Section 2.2 is binding here: the exposure threshold is a configurable
parameter anchored to the ICNIRP 2020 general-public reference level for
2-300 GHz. It is *not* presented as an avian-harm threshold.
"""

from flywaydqn.exposure.density import (
    ICNIRP_2020_PUBLIC_REFERENCE_W_PER_M2,
    dose_w_s_per_m2,
    fraction_over_threshold,
    power_density_grid_w_per_m2,
    power_density_w_per_m2,
    s_max,
    s_p95,
)
from flywaydqn.exposure.metrics import (
    ExposureMetrics,
    exposure_metrics_from_density,
    time_over_threshold_fraction,
)
from flywaydqn.exposure.thresholds import (
    exposure_penalty_softplus,
    icnirp_public_reference_level_2020_w_per_m2,
)

__all__ = [
    "ICNIRP_2020_PUBLIC_REFERENCE_W_PER_M2",
    "ExposureMetrics",
    "dose_w_s_per_m2",
    "exposure_metrics_from_density",
    "exposure_penalty_softplus",
    "fraction_over_threshold",
    "icnirp_public_reference_level_2020_w_per_m2",
    "power_density_grid_w_per_m2",
    "power_density_w_per_m2",
    "s_max",
    "s_p95",
    "time_over_threshold_fraction",
]
