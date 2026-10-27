"""Rule-based beam policies for the demo simulator (spec Section 6).

Two baselines the learned controller is compared against:

- ``analytic-nulling`` (:class:`AnalyticNullingPolicy`): serve users with the
  greedy codebook beam, and steer a projection null toward the *peak-density
  cell* of the flock footprint whenever geometry and cost guards allow. It
  declines to null when the null direction would sit on, or mirror onto, the
  serving direction; when the flock is out of range; when the footprint is
  already below the exposure threshold; when the null would cost too much
  served throughput; or when it would not materially reduce footprint
  exposure.
- ``unconstrained`` (:class:`UnconstrainedPolicy`): greedy serving beam, full
  power, no null at all. The no-mitigation reference.

Why the peak cell rather than the centroid
------------------------------------------
The naive analytic baseline nulls toward the flock centroid, but the hottest
cells of the footprint sit closest to the *serving beam axis*, not to the
centroid, whenever the flock is off that axis by more than the footprint
half-width. A centroid null then misses the cells that actually set ``S_p95``
and ``frac_over``, and exposure barely moves (spec README, finding 3:
exposure is not monotone in nulling). Targeting the live peak cell keeps the
baseline rule-based -- still a closed-form projection, decided from the
current geometry alone -- while making its mitigation show up in the metrics.
The guards below are evaluated against the chosen null direction, so the
front/back mirror hazard (finding 2) is caught the same way.

All computation is pure: the same :class:`LiveState` yields the same
:class:`BeamAction`, so the demo runner is deterministic given its seed.

Units
-----
- angles: degrees (azimuth from panel boresight, elevation above the horizon)
- distances: meters (m)
- power: milliwatts (mW)
- power density: watts per square meter (W/m^2)
- throughput: megabits per second (Mbps)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt

from flywaydqn.exposure.density import power_density_w_per_m2, s_p95
from flywaydqn.rf.array import composite_gain_dbi
from flywaydqn.rf.codebook import Beam, beam_index_from_azimuth, normalize_azimuth
from flywaydqn.rf.nulling import null_toward
from flywaydqn.rf.pathloss import tr38901_uma_los_pathloss_dbm
from flywaydqn.rf.sinr import sinr_linear, throughput_mbps_shannon

# Fixed inter-sector interference floor from the neighbouring sectors, in dBm.
# Kept near a 7 dB-noise-figure 100 MHz receiver so it perturbs, but does not
# dominate, the demo link budgets.
DEFAULT_INTERFERENCE_DBM: Final[float] = -80.0

# dB reported for a link that is not served at all: a shut-down sector or an
# annihilated beam. The linear SINR is exactly 0 there, whose dB value is
# ``-inf``; the wire needs a finite number, so a clipped value is reported as
# this floor and never as a plausible positive one.
NO_SERVICE_SINR_DB: Final[float] = -30.0

__all__ = [
    "DEFAULT_INTERFERENCE_DBM",
    "NO_SERVICE_SINR_DB",
    "AnalyticNullingPolicy",
    "BeamAction",
    "BeamPolicy",
    "CellGeometry",
    "LiveState",
    "NullDecision",
    "RadioSetup",
    "UnconstrainedPolicy",
    "UserLink",
    "UserLinkBudget",
    "angular_separation_deg",
    "cell_densities_w_m2",
    "exposure_p95_w_m2",
    "greedy_beam",
    "link_budgets",
    "mirror_azimuth_deg",
    "peak_cell_geometry",
    "sum_served_throughput_mbps",
    "transmit_weights",
    "StaticBackoffPolicy",
    "SectorShutdownPolicy",
]


class NullDecision(StrEnum):
    """Why a policy did or did not place a null on this step."""

    APPLIED = "applied"
    OUT_OF_RANGE = "out of range"
    SERVING_AXIS = "on the serving axis"
    MIRROR_BEARING = "front/back mirror of the serving beam"
    BELOW_THRESHOLD = "exposure below threshold"
    SERVING_COST = "too costly for the served links"
    NOT_HELPFUL = "would not reduce footprint exposure"
    NOT_APPLICABLE = "no nulling in this policy"


@dataclass(frozen=True)
class CellGeometry:
    """One flock-footprint sample, seen from the sector antenna.

    Angles follow the array convention: azimuth from panel boresight,
    elevation above the horizon. ``slant_range_m`` is the 3-D antenna-to-cell
    distance.
    """

    bearing_deg: float
    elevation_deg: float
    slant_range_m: float


@dataclass(frozen=True)
class UserLink:
    """One scheduled user, seen from the sector antenna."""

    bearing_deg: float
    elevation_deg: float
    range_m: float


@dataclass(frozen=True)
class LiveState:
    """Everything a policy may look at for one step.

    The runner precomputes the per-cell and per-user geometry once and shares
    it here, so a policy's exposure estimate and the frame's reported metrics
    are always computed from the same numbers.
    """

    step: int
    time_s: float
    flock_bearing_deg: float
    flock_elevation_deg: float
    flock_range_m: float
    users: tuple[UserLink, ...]
    cells: tuple[CellGeometry, ...]
    # Flock summary on the trained observation's sector block (spec 4.3): the
    # runner and the env both fill these from their live FlockState, so a
    # policy served on the demo scene reads the same features it was trained
    # on. Defaults keep older constructions (rule-based tests) valid.
    flock_altitude_mean_m: float = 0.0
    flock_extent_m: float = 0.0
    flock_heading_deg: float = 0.0
    flock_speed_ms: float = 0.0
    # Session preference weight on the reward's log10 scale (Decision D2);
    # the runner passes ``_log10_weight()`` so the DQN's weight feature
    # tracks the live trade-off slider instead of a sampled constant.
    log10_w: float = 0.0


@dataclass(frozen=True)
class RadioSetup:
    """Static radio geometry shared by every step of one scenario."""

    codebook: tuple[Beam, ...]
    n_rows: int
    n_cols: int
    d_lambda: float
    tx_power_mw: float
    bandwidth_mhz: float
    fc_ghz: float
    noise_power_mw: float
    interference_mw: float
    s_thr_w_m2: float
    spectral_efficiency_cap: float
    h_bs_m: float
    h_ut_m: float


@dataclass(frozen=True)
class BeamAction:
    """One policy decision: which weights to transmit, and why."""

    beam: Beam
    weights: npt.NDArray[np.complex128]
    null_active: bool
    power_mw: float
    decision: NullDecision
    detail: str


@dataclass(frozen=True)
class UserLinkBudget:
    """Served link budget for one user under the transmitted weights."""

    bearing_deg: float
    range_m: float
    serving_gain_dbi: float
    pathloss_db: float
    sinr: float
    throughput_mbps: float

    @property
    def sinr_db(self) -> float:
        """SINR in dB; ``-inf`` for a zero or annihilated link."""
        if self.sinr <= 0.0:
            return -math.inf
        return 10.0 * math.log10(self.sinr)


class BeamPolicy(Protocol):
    """Structural interface every demo policy implements."""

    def act(self, state: LiveState, setup: RadioSetup) -> BeamAction:
        """Return the beam decision for one step."""
        ...


def angular_separation_deg(
    elevation_a_deg: float,
    azimuth_a_deg: float,
    elevation_b_deg: float,
    azimuth_b_deg: float,
) -> float:
    """Return the true 3-D angle between two array-frame directions, in degrees.

    Unlike :func:`flywaydqn.rf.codebook.angular_distance_deg` (azimuth only),
    this accounts for elevation as well: two directions on the same bearing but
    14 deg apart in elevation -- a horizon beam and a flock overhead -- are 14
    deg apart here, not 0.
    """
    theta_a, phi_a = math.radians(elevation_a_deg), math.radians(azimuth_a_deg)
    theta_b, phi_b = math.radians(elevation_b_deg), math.radians(azimuth_b_deg)
    cos_sep = math.sin(theta_a) * math.sin(theta_b) + math.cos(theta_a) * math.cos(
        theta_b
    ) * math.cos(phi_a - phi_b)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_sep))))


def mirror_azimuth_deg(azimuth_deg: float) -> float:
    """Return the front/back mirror bearing of an azimuth, in degrees.

    At the horizon the array response depends on ``sin(azimuth)`` alone, so
    ``a(phi)`` and ``a(180 - phi)`` are identical: a null placed at the mirror
    bearing annihilates the serving beam while the 30 dB front/back element
    attenuation keeps the real exposure from that direction negligible (spec
    README, finding 2).
    """
    return normalize_azimuth(180.0 - azimuth_deg)


def transmit_weights(
    beam: Beam,
    setup: RadioSetup,
    *,
    null_elevation_deg: float | None = None,
    null_azimuth_deg: float | None = None,
) -> npt.NDArray[np.complex128]:
    """Return the weights to transmit: ``beam`` or ``beam`` with a projection null.

    With both ``null_*`` arguments ``None`` the beam's own weights are
    returned unchanged. Otherwise a projection null is steered toward the
    given direction and renormalized to preserve transmit power.
    """
    if null_elevation_deg is None or null_azimuth_deg is None:
        return beam.weights
    return null_toward(
        beam.weights,
        null_elevation_deg,
        null_azimuth_deg,
        setup.n_rows,
        setup.n_cols,
        setup.d_lambda,
    )


def cell_densities_w_m2(
    weights: npt.NDArray[np.complex128],
    cells: tuple[CellGeometry, ...],
    setup: RadioSetup,
    *,
    power_mw: float | None = None,
) -> npt.NDArray[np.float64]:
    """Return free-space power density at each footprint cell, in W/m^2.

    Gains are composite (element pattern included), matching the exposure model
    in :mod:`flywaydqn.exposure.density`.

    ``power_mw`` overrides the setup's transmit power, so a policy that backs
    off -- or switches the sector off -- is scored on the power it actually
    transmits. A non-positive power density is zero everywhere by definition,
    which also keeps the zero-weight shutdown case finite instead of ``-inf``
    dB of gain.
    """
    tx_power_mw = setup.tx_power_mw if power_mw is None else power_mw
    if tx_power_mw <= 0.0:
        return np.zeros(len(cells), dtype=np.float64)
    densities = np.empty(len(cells), dtype=np.float64)
    for i, cell in enumerate(cells):
        gain_dbi = composite_gain_dbi(
            weights,
            cell.elevation_deg,
            cell.bearing_deg,
            setup.n_rows,
            setup.n_cols,
            setup.d_lambda,
        )
        densities[i] = power_density_w_per_m2(tx_power_mw, gain_dbi, cell.slant_range_m)
    return densities


def exposure_p95_w_m2(
    weights: npt.NDArray[np.complex128],
    cells: tuple[CellGeometry, ...],
    setup: RadioSetup,
    *,
    power_mw: float | None = None,
) -> float:
    """Return the 95th-percentile footprint density under ``weights``, in W/m^2."""
    return s_p95(cell_densities_w_m2(weights, cells, setup, power_mw=power_mw))


def link_budgets(
    weights: npt.NDArray[np.complex128],
    state: LiveState,
    setup: RadioSetup,
    *,
    power_mw: float | None = None,
) -> list[UserLinkBudget]:
    """Return the served link budget for every user under ``weights``.

    Path loss is the TR 38.901 UMa LOS median; serving gain is the composite
    (element pattern included) array gain evaluated at the user's own
    elevation, which sits below the horizon for a ground user under a tall
    macro antenna.

    ``power_mw`` overrides the setup's transmit power, so a power-reducing
    policy is scored on the power it actually transmits. A zero power leaves
    every linear SINR at exactly 0, i.e. no service.
    """
    tx_power_mw = setup.tx_power_mw if power_mw is None else power_mw
    budgets: list[UserLinkBudget] = []
    for user in state.users:
        gain_dbi = composite_gain_dbi(
            weights,
            user.elevation_deg,
            user.bearing_deg,
            setup.n_rows,
            setup.n_cols,
            setup.d_lambda,
        )
        pathloss_db = tr38901_uma_los_pathloss_dbm(
            user.range_m,
            setup.fc_ghz,
            h_bs_m=setup.h_bs_m,
            h_ut_m=setup.h_ut_m,
        )
        sinr = sinr_linear(
            gain_dbi,
            pathloss_db,
            setup.interference_mw,
            setup.noise_power_mw,
            tx_power_mw,
        )
        budgets.append(
            UserLinkBudget(
                bearing_deg=user.bearing_deg,
                range_m=user.range_m,
                serving_gain_dbi=gain_dbi,
                pathloss_db=pathloss_db,
                sinr=sinr,
                throughput_mbps=throughput_mbps_shannon(
                    sinr, setup.bandwidth_mhz, setup.spectral_efficiency_cap
                ),
            )
        )
    return budgets


def sum_served_throughput_mbps(budgets: list[UserLinkBudget]) -> float:
    """Return the sum of the served user throughputs, in Mbps."""
    return math.fsum(budget.throughput_mbps for budget in budgets)


def greedy_beam(state: LiveState, setup: RadioSetup) -> Beam:
    """Return the codebook beam that best serves the live users.

    Each beam is scored by the summed *linear* composite gain toward every
    user, so the pick is the maximum-service beam rather than the one nearest
    any single bearing. With no users on this step the fallback is the beam
    nearest the flock bearing (the tracking choice used by the spec's
    analytical baseline).
    """
    if not state.users:
        index = beam_index_from_azimuth(
            state.flock_bearing_deg,
            len(setup.codebook),
            (setup.codebook[0].azimuth_deg, setup.codebook[-1].azimuth_deg),
        )
        return setup.codebook[index]

    best = setup.codebook[0]
    best_score = -math.inf
    for beam in setup.codebook:
        score = math.fsum(
            10.0
            ** (
                beam.composite_gain_toward_dbi(
                    user.elevation_deg,
                    user.bearing_deg,
                    setup.n_rows,
                    setup.n_cols,
                    setup.d_lambda,
                )
                / 10.0
            )
            for user in state.users
        )
        if score > best_score:
            best_score = score
            best = beam
    return best


def peak_cell_geometry(
    weights: npt.NDArray[np.complex128],
    cells: tuple[CellGeometry, ...],
    setup: RadioSetup,
) -> CellGeometry:
    """Return the geometry of the densest footprint cell under ``weights``."""
    densities = cell_densities_w_m2(weights, cells, setup)
    return cells[int(np.argmax(densities))]


class UnconstrainedPolicy:
    """Greedy serving beam, full power, no null: the no-mitigation reference."""

    def act(self, state: LiveState, setup: RadioSetup) -> BeamAction:
        """Serve the users greedily and transmit the plain codebook weights."""
        beam = greedy_beam(state, setup)
        return BeamAction(
            beam=beam,
            weights=beam.weights,
            null_active=False,
            power_mw=setup.tx_power_mw,
            decision=NullDecision.NOT_APPLICABLE,
            detail="Unconstrained baseline: greedy serving beam, no null.",
        )


class StaticBackoffPolicy:
    """Fixed transmit-power reduction while the flock is in range; else full power.

    This is the simplest precautionary baseline: the operator keeps the carrier
    on but backs off by a fixed factor whenever the flock is within a configured
    range, regardless of whether the footprint is actually over threshold. It
    exists so the learned controller has to earn its null decisions rather than
    riding a blanket power cut.
    """

    backoff_range_m: float = 600.0
    backoff_power_frac: float = 0.5

    def __init__(self, backoff_range_m: float = 600.0, backoff_power_frac: float = 0.5) -> None:
        object.__setattr__(self, "backoff_range_m", backoff_range_m)
        object.__setattr__(self, "backoff_power_frac", backoff_power_frac)

    def act(self, state: LiveState, setup: RadioSetup) -> BeamAction:
        beam = greedy_beam(state, setup)
        if state.flock_range_m <= self.backoff_range_m:
            power_mw = setup.tx_power_mw * self.backoff_power_frac
            return BeamAction(
                beam=beam,
                weights=beam.weights,
                null_active=False,
                power_mw=power_mw,
                decision=NullDecision.NOT_APPLICABLE,
                detail=(
                    f"static back-off: flock in range ({state.flock_range_m:.0f} m <= "
                    f"{self.backoff_range_m:.0f} m), transmit at {self.backoff_power_frac:.0%} "
                    f"of full power ({power_mw:.0f} vs {setup.tx_power_mw:.0f} mW)"
                ),
            )
        return BeamAction(
            beam=beam,
            weights=beam.weights,
            null_active=False,
            power_mw=setup.tx_power_mw,
            decision=NullDecision.NOT_APPLICABLE,
            detail="static back-off: flock out of range "
            f"({state.flock_range_m:.0f} m), full power",
        )


class SectorShutdownPolicy:
    """Transmit off while the flock is in range; full power otherwise.

    The most aggressive precautionary baseline: the sector is shut down
    completely whenever the flock is within a configured range. It exists as a
    lower bound on exposure (exactly zero while in range, since no power is
    radiated) and as an upper bound on throughput loss, so the learned
    controller has a worst-case mitigation to beat on the exposure axis.

    ``null_active`` stays ``False``: switching the carrier off is not a null,
    and a consumer must not be told a null is on the air when nothing is.
    """

    shutdown_range_m: float = 600.0

    def __init__(self, shutdown_range_m: float = 600.0) -> None:
        object.__setattr__(self, "shutdown_range_m", shutdown_range_m)

    def act(self, state: LiveState, setup: RadioSetup) -> BeamAction:
        beam = greedy_beam(state, setup)
        if state.flock_range_m <= self.shutdown_range_m:
            return BeamAction(
                beam=beam,
                weights=np.zeros_like(beam.weights),
                null_active=False,
                power_mw=0.0,
                decision=NullDecision.NOT_APPLICABLE,
                detail=(
                    f"sector shutdown: flock in range ({state.flock_range_m:.0f} m <= "
                    f"{self.shutdown_range_m:.0f} m), transmitter off, 0 mW radiated"
                ),
            )
        return BeamAction(
            beam=beam,
            weights=beam.weights,
            null_active=False,
            power_mw=setup.tx_power_mw,
            decision=NullDecision.NOT_APPLICABLE,
            detail="sector shutdown: flock out of range "
            f"({state.flock_range_m:.0f} m), full power",
        )


@dataclass(frozen=True)
class AnalyticNullingPolicy:
    """Guarded analytic nulling: serve users, null the exposure peak when viable.

    The null is placed on the live peak-density cell (see the module docstring
    for why not the centroid). Guards are evaluated in a fixed order so the
    reported :class:`NullDecision` names the first binding constraint: range,
    serving axis, front/back mirror, exposure threshold, served-rate cost, and
    finally the minimum exposure reduction the null must buy.
    """

    null_max_range_m: float = 500.0
    serving_guard_deg: float = 5.0
    mirror_guard_deg: float = 5.0
    min_exposure_reduction_db: float = 1.0
    max_rate_loss_frac: float = 0.15

    def act(self, state: LiveState, setup: RadioSetup) -> BeamAction:
        """Decide whether to place the analytic null on this step."""
        beam = greedy_beam(state, setup)

        if state.flock_range_m > self.null_max_range_m:
            return self._decline(
                beam,
                setup,
                NullDecision.OUT_OF_RANGE,
                f"flock at {state.flock_range_m:.0f} m is beyond the "
                f"{self.null_max_range_m:.0f} m nulling range",
            )

        weights_before = beam.weights
        densities_before = cell_densities_w_m2(weights_before, state.cells, setup)
        p95_before = s_p95(densities_before)
        peak = state.cells[int(np.argmax(densities_before))]

        serving_sep = angular_separation_deg(
            peak.elevation_deg,
            peak.bearing_deg,
            beam.elevation_deg,
            beam.azimuth_deg,
        )
        if serving_sep <= self.serving_guard_deg:
            return self._decline(
                beam,
                setup,
                NullDecision.SERVING_AXIS,
                f"peak cell at {peak.bearing_deg:.1f} deg sits {serving_sep:.1f} deg from "
                f"serving beam {beam.index} at {beam.azimuth_deg:.1f} deg "
                f"(guard {self.serving_guard_deg:.1f} deg); the null would annihilate the beam",
            )

        mirror_bearing_deg = mirror_azimuth_deg(beam.azimuth_deg)
        mirror_sep = angular_separation_deg(
            peak.elevation_deg,
            peak.bearing_deg,
            beam.elevation_deg,
            mirror_bearing_deg,
        )
        if mirror_sep <= self.mirror_guard_deg:
            return self._decline(
                beam,
                setup,
                NullDecision.MIRROR_BEARING,
                f"peak cell at {peak.bearing_deg:.1f} deg is {mirror_sep:.1f} deg from the "
                f"front/back mirror of beam {beam.index} ({mirror_bearing_deg:.1f} deg); "
                f"the array cannot null it without killing the serving beam",
            )

        if p95_before < setup.s_thr_w_m2:
            return self._decline(
                beam,
                setup,
                NullDecision.BELOW_THRESHOLD,
                f"footprint p95 {p95_before:.3f} W/m\u00b2 is under the "
                f"{setup.s_thr_w_m2:.3f} W/m\u00b2 threshold; nothing to mitigate",
            )

        weights_after = transmit_weights(
            beam,
            setup,
            null_elevation_deg=peak.elevation_deg,
            null_azimuth_deg=peak.bearing_deg,
        )
        serving_gain_after = composite_gain_dbi(
            weights_after,
            beam.elevation_deg,
            beam.azimuth_deg,
            setup.n_rows,
            setup.n_cols,
            setup.d_lambda,
        )
        if not math.isfinite(serving_gain_after):
            return self._decline(
                beam,
                setup,
                NullDecision.NOT_HELPFUL,
                "the projection annihilated the serving beam; falling back to un-nulled weights",
            )

        rate_before = sum_served_throughput_mbps(link_budgets(weights_before, state, setup))
        rate_after = sum_served_throughput_mbps(link_budgets(weights_after, state, setup))
        if rate_before > 0.0:
            rate_loss_frac = (rate_before - rate_after) / rate_before
            if rate_loss_frac > self.max_rate_loss_frac:
                return self._decline(
                    beam,
                    setup,
                    NullDecision.SERVING_COST,
                    f"the null would lose {rate_loss_frac:.0%} of the served sum rate "
                    f"({rate_before:.0f} to {rate_after:.0f} Mbps, budget "
                    f"{self.max_rate_loss_frac:.0%})",
                )

        p95_after = exposure_p95_w_m2(weights_after, state.cells, setup)
        if p95_after >= p95_before * 10.0 ** (-self.min_exposure_reduction_db / 10.0):
            return self._decline(
                beam,
                setup,
                NullDecision.NOT_HELPFUL,
                f"the null only moves footprint p95 from {p95_before:.3f} to "
                f"{p95_after:.3f} W/m\u00b2 (need {self.min_exposure_reduction_db:.1f} dB)",
            )

        reduction_db = 10.0 * math.log10(p95_before / p95_after) if p95_after > 0.0 else math.inf
        return BeamAction(
            beam=beam,
            weights=weights_after,
            null_active=True,
            power_mw=setup.tx_power_mw,
            decision=NullDecision.APPLIED,
            detail=(
                f"nulled the peak cell at {peak.bearing_deg:.1f} deg, "
                f"{serving_sep:.1f} deg off beam {beam.index}: p95 "
                f"{p95_before:.3f} to {p95_after:.3f} W/m\u00b2 "
                f"({reduction_db:.1f} dB), sum rate {rate_before:.0f} to "
                f"{rate_after:.0f} Mbps"
            ),
        )

    def _decline(
        self,
        beam: Beam,
        setup: RadioSetup,
        decision: NullDecision,
        reason: str,
    ) -> BeamAction:
        """Return a no-null action carrying ``decision`` and its explanation."""
        return BeamAction(
            beam=beam,
            weights=beam.weights,
            null_active=False,
            power_mw=setup.tx_power_mw,
            decision=decision,
            detail=reason,
        )
