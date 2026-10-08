"""RF core: path loss, array, codebook, nulling, and SINR.

This package is pure math with no RL or web dependencies, so it can be unit- and
property-tested in isolation (spec Section 3 design rule and Phase 1).
"""

from flywaydqn.rf.array import (
    ELEMENT_MAX_GAIN_DBI,
    beamforming_gain_dbi,
    composite_gain_dbi,
    element_pattern_dbi,
    matched_filter_weights,
    upa_array_factor_dbi,
    upa_steering_vector,
)
from flywaydqn.rf.codebook import (
    Beam,
    angular_distance_deg,
    beam_azimuth_deg,
    beam_index_from_azimuth,
    build_dft_codebook,
    dft_codebook_azimuth,
)
from flywaydqn.rf.nulling import (
    main_lobe_loss_db,
    null_depth_db,
    null_toward,
    project_null,
)
from flywaydqn.rf.pathloss import (
    SCENARIO_DEFAULT,
    TR38901_VERSION,
    breakpoint_distance_m,
    los_probability_uma,
    pathloss_uma,
    shadow_fading_std_db,
    tr38901_uma_los_pathloss_dbm,
    tr38901_uma_nlos_pathloss_dbm,
)
from flywaydqn.rf.sinr import (
    DEFAULT_SPECTRAL_EFFICIENCY_CAP,
    fifth_percentile_throughput_mbps,
    min_rate_fraction,
    sinr_db,
    sinr_linear,
    sum_throughput_mbps,
    throughput_mbps_shannon,
)

__all__ = [
    # array
    "ELEMENT_MAX_GAIN_DBI",
    "beamforming_gain_dbi",
    "composite_gain_dbi",
    "element_pattern_dbi",
    "matched_filter_weights",
    "upa_array_factor_dbi",
    "upa_steering_vector",
    # codebook
    "Beam",
    "angular_distance_deg",
    "beam_azimuth_deg",
    "beam_index_from_azimuth",
    "build_dft_codebook",
    "dft_codebook_azimuth",
    # nulling
    "main_lobe_loss_db",
    "null_depth_db",
    "null_toward",
    "project_null",
    # path loss
    "SCENARIO_DEFAULT",
    "TR38901_VERSION",
    "breakpoint_distance_m",
    "los_probability_uma",
    "pathloss_uma",
    "shadow_fading_std_db",
    "tr38901_uma_los_pathloss_dbm",
    "tr38901_uma_nlos_pathloss_dbm",
    # sinr / throughput
    "DEFAULT_SPECTRAL_EFFICIENCY_CAP",
    "fifth_percentile_throughput_mbps",
    "min_rate_fraction",
    "sinr_db",
    "sinr_linear",
    "sum_throughput_mbps",
    "throughput_mbps_shannon",
]
