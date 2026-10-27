"""Demo simulator: canned scenarios, seeded motion, and the frames they produce.

This package owns the rule-based demo behind the FastAPI service. Each YAML
file under ``configs/demo`` describes one deterministic scene -- array and
radio geometry, flock entry, user deployment, reward shaping, and which
:mod:`flywaydqn.agents.baselines_demo` policy runs it. :class:`DemoRunner`
loads a scenario, steps its mobility models, asks the policy for a beam
decision, and emits the wire :class:`~flywaydqn.service.schemas.StateFrame`
the dashboard renders.

Determinism: every scenario carries a seed; the request's ``seed`` and
``weight`` override the file's defaults. All randomness flows through one
``numpy.random.Generator`` per session, and flock drift is jitter-free, so the
same request always produces the same frames.

Scenario keys are YAML file stems and must match
``[A-Za-z0-9][A-Za-z0-9_-]*``; anything else is rejected before touching the
filesystem, so a key can never escape ``configs/demo``.

Units follow :mod:`flywaydqn.service.schemas`: meters, degrees, seconds, mW,
Mbps, W/m^2.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Final, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from flywaydqn.agents.baselines_demo import (
    DEFAULT_INTERFERENCE_DBM,
    NO_SERVICE_SINR_DB,
    AnalyticNullingPolicy,
    BeamPolicy,
    CellGeometry,
    LiveState,
    RadioSetup,
    UnconstrainedPolicy,
    UserLink,
    cell_densities_w_m2,
    exposure_p95_w_m2,
    link_budgets,
    sum_served_throughput_mbps,
)
from flywaydqn.agents.policies import build_policy, policy_name_for
from flywaydqn.env.reward import make_reward_fn
from flywaydqn.exposure.metrics import exposure_metrics_from_density
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
from flywaydqn.rf.array import DEFAULT_D_LAMBDA, composite_gain_dbi
from flywaydqn.rf.codebook import DEFAULT_AZIMUTH_SPAN_DEG, build_dft_codebook, normalize_azimuth
from flywaydqn.rf.sinr import DEFAULT_SPECTRAL_EFFICIENCY_CAP, max_throughput_reference_mbps
from flywaydqn.service.schemas import (
    BaselineMetrics,
    BeamState,
    MapPoint,
    ScenarioInfo,
    ScenarioRequest,
    StateFrame,
    StepMetrics,
)
from flywaydqn.service.schemas import (
    FlockState as WireFlockState,
)
from flywaydqn.service.schemas import (
    UserState as WireUserState,
)
from flywaydqn.units import dbm_to_mw, thermal_noise_power_mw

# Environment variable that overrides where scenario YAMLs are loaded from.
CONFIG_DIR_ENV_VAR: Final[str] = "FLYWAYDQN_DEMO_CONFIGS"

# A scenario key is a bare file stem: no separators, no dots, no traversal.
_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

__all__ = [
    "CONFIG_DIR_ENV_VAR",
    "AnalyticNullingPolicy",
    "ArrayConfig",
    "CodebookConfig",
    "DemoRunner",
    "ExposureConfig",
    "FlockConfig",
    "PolicyConfig",
    "RadioConfig",
    "RewardConfig",
    "ScenarioConfig",
    "ScenarioRequest",
    "UnconstrainedPolicy",
    "UsersConfig",
    "demo_config_dir",
    "demo_runner_factory",
    "list_demo_scenarios",
    "load_scenario",
    "scenario_path",
]


class ArrayConfig(BaseModel):
    """UPA panel geometry (see :mod:`flywaydqn.rf.array`)."""

    model_config = ConfigDict(extra="forbid")

    n_rows: int = Field(8, ge=1)
    n_cols: int = Field(8, ge=1)
    d_lambda: float = Field(DEFAULT_D_LAMBDA, gt=0.0)
    elevation_deg: float = 0.0


class CodebookConfig(BaseModel):
    """DFT beam codebook layout (see :mod:`flywaydqn.rf.codebook`)."""

    model_config = ConfigDict(extra="forbid")

    n_beams: int = Field(16, ge=1)
    azimuth_span_deg: tuple[float, float] = DEFAULT_AZIMUTH_SPAN_DEG


class RadioConfig(BaseModel):
    """Carrier, power, and receiver configuration for one scenario."""

    model_config = ConfigDict(extra="forbid")

    fc_ghz: float = Field(3.5, gt=0.0)
    bandwidth_mhz: float = Field(100.0, gt=0.0)
    tx_power_dbm: float = 53.0
    noise_figure_db: float = Field(7.0, ge=0.0)
    interference_dbm: float = DEFAULT_INTERFERENCE_DBM
    spectral_efficiency_cap: float = Field(DEFAULT_SPECTRAL_EFFICIENCY_CAP, gt=0.0)
    h_bs_m: float = Field(25.0, gt=0.0)
    h_ut_m: float = Field(1.5, gt=0.0)

    @model_validator(mode="after")
    def _heights_are_sane(self) -> RadioConfig:
        if self.h_bs_m <= self.h_ut_m:
            raise ValueError("h_bs_m must exceed h_ut_m")
        return self


class ExposureConfig(BaseModel):
    """Footprint sampling and the exposure threshold anchor."""

    model_config = ConfigDict(extra="forbid")

    s_thr_w_m2: float = Field(0.1, gt=0.0)
    n_cells: int = Field(64, ge=1)


class RewardConfig(BaseModel):
    """Reward shaping, matching :func:`flywaydqn.env.reward.make_reward_fn`."""

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(1.0, gt=0.0)
    gamma: float = Field(0.02, ge=0.0)
    kappa: float = Field(5.0, gt=0.0)
    reward_clip: float = Field(10.0, gt=0.0)


class FlockConfig(BaseModel):
    """Flock entry geometry and drift for one scenario."""

    model_config = ConfigDict(extra="forbid")

    bearing_deg: float = 0.0
    range_m: float = Field(..., gt=0.0)
    speed_ms: float = Field(DEFAULT_FLOCK_SPEED_MPS, ge=0.0)
    heading_deg: float = 45.0
    altitude_min_m: float = Field(DEFAULT_ALTITUDE_MIN_M, ge=0.0)
    altitude_max_m: float = Field(DEFAULT_ALTITUDE_MAX_M, ge=0.0)
    extent_m: float = Field(DEFAULT_EXTENT_M, gt=0.0)
    n_individuals: int = Field(DEFAULT_N_INDIVIDUALS, ge=1)
    recycle_range_m: float = Field(900.0, gt=0.0)

    @model_validator(mode="after")
    def _altitude_band_is_sane(self) -> FlockConfig:
        if self.altitude_max_m < self.altitude_min_m:
            raise ValueError("altitude_max_m must be >= altitude_min_m")
        return self


class UsersConfig(BaseModel):
    """User deployment; an optional bearing cluster keeps a scenario on rails.

    Without ``bearing_center_deg`` users are drawn over the full circle by
    :func:`flywaydqn.mobility.users.generate_users`. With it, bearings are
    drawn uniformly within ``center +- half_width`` degrees, which is how the
    mirror-bearing scenario pins the serving beam.
    """

    model_config = ConfigDict(extra="forbid")

    n_users: int = Field(6, ge=0)
    min_range_m: float = Field(50.0, gt=0.0)
    max_range_m: float = Field(400.0, gt=0.0)
    vehicular_fraction: float = Field(0.3, ge=0.0, le=1.0)
    area_m: float = Field(DEFAULT_AREA_M, gt=0.0)
    bearing_center_deg: float | None = None
    bearing_half_width_deg: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _cluster_fields_agree(self) -> UsersConfig:
        if (self.bearing_center_deg is None) != (self.bearing_half_width_deg is None):
            raise ValueError("bearing_center_deg and bearing_half_width_deg must be set together")
        if not self.min_range_m < self.max_range_m:
            raise ValueError("require min_range_m < max_range_m")
        if self.max_range_m > self.area_m:
            raise ValueError("max_range_m must fit inside the area_m waypoint box")
        return self


class PolicyConfig(BaseModel):
    """Which baseline runs the scenario, and with which guards."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["analytic-nulling", "unconstrained", "static-backoff", "sector-shutdown"] = (
        "analytic-nulling"
    )
    null_max_range_m: float = Field(500.0, gt=0.0)
    serving_guard_deg: float = Field(5.0, ge=0.0)
    mirror_guard_deg: float = Field(5.0, ge=0.0)
    min_exposure_reduction_db: float = Field(1.0, ge=0.0)
    max_rate_loss_frac: float = Field(0.15, ge=0.0, le=1.0)


class ScenarioConfig(BaseModel):
    """One demo scenario, as loaded from ``configs/demo/<key>.yaml``."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    dt_seconds: float = Field(0.1, gt=0.0)
    seed: int = Field(1, ge=0)
    weight: float = Field(1.0, ge=0.0, le=5.0)
    array: ArrayConfig = Field(default_factory=ArrayConfig)
    codebook: CodebookConfig = Field(default_factory=CodebookConfig)
    radio: RadioConfig = Field(default_factory=RadioConfig)
    exposure: ExposureConfig = Field(default_factory=ExposureConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    flock: FlockConfig
    users: UsersConfig = Field(default_factory=UsersConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)


def demo_config_dir() -> Path:
    """Return the directory demo scenario YAMLs are loaded from.

    Resolution order: the ``FLYWAYDQN_DEMO_CONFIGS`` environment variable, the
    repository's ``configs/demo`` when the package runs from source, then
    ``configs/demo`` under the current working directory.
    """
    override = os.environ.get(CONFIG_DIR_ENV_VAR)
    if override:
        return Path(override)
    from_source = Path(__file__).resolve().parents[3] / "configs" / "demo"
    if from_source.is_dir():
        return from_source
    return Path.cwd() / "configs" / "demo"


def scenario_path(key: str) -> Path:
    """Return the YAML path for a scenario key, validating the key first.

    Raises
    ------
    ValueError
        If ``key`` is not a bare ``[A-Za-z0-9][A-Za-z0-9_-]*`` stem.
    FileNotFoundError
        If no such scenario exists under :func:`demo_config_dir`.
    """
    if not _KEY_PATTERN.match(key):
        raise ValueError(
            f"invalid scenario key {key!r}: keys are bare file stems "
            f"matching {_KEY_PATTERN.pattern}"
        )
    directory = demo_config_dir()
    path = directory / f"{key}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no demo scenario {key!r} under {directory}")
    return path


def load_scenario(key: str) -> ScenarioConfig:
    """Load and validate the scenario named ``key``."""
    return _load_scenario_file(scenario_path(key))


def list_demo_scenarios() -> tuple[ScenarioInfo, ...]:
    """Return one :class:`ScenarioInfo` per scenario YAML, sorted by key."""
    directory = demo_config_dir()
    infos: list[ScenarioInfo] = []
    for path in sorted(directory.glob("*.yaml")):
        config = _load_scenario_file(path)
        infos.append(
            ScenarioInfo(key=path.stem, label=config.label, description=config.description)
        )
    return tuple(infos)


def _load_scenario_file(path: Path) -> ScenarioConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ScenarioConfig.model_validate(raw)


def _wire_sinr_db(sinr_db: float) -> float:
    """Return a finite SINR for the wire, flooring a link that is not served.

    A shut-down sector or an annihilated beam leaves the linear SINR at exactly
    0, whose dB value is ``-inf`` and which has no JSON spelling. Such a link
    reports :data:`~flywaydqn.agents.baselines_demo.NO_SERVICE_SINR_DB` instead,
    which is a floor, never a plausible rate.
    """
    if not math.isfinite(sinr_db):
        return NO_SERVICE_SINR_DB
    return sinr_db


class DemoRunner:
    """Deterministic frame producer for one demo scenario.

    Satisfies the service's ``FrameRunner`` protocol structurally:
    ``initial_frame()`` renders step 0 and ``step()`` advances the flock and
    the users before rendering the next frame.
    """

    def __init__(
        self,
        request: ScenarioRequest,
        *,
        config: ScenarioConfig | None = None,
        policy: BeamPolicy | None = None,
    ) -> None:
        """Build the runner for ``request``.

        Parameters
        ----------
        request :
            Session request; its ``seed`` and ``weight`` override the file.
        config :
            Pre-loaded scenario, bypassing the YAML lookup (used by tests).
        policy :
            Policy override, bypassing the scenario's policy name (used by
            tests). When given, ``config.policy`` is ignored.
        """
        self._config = config if config is not None else load_scenario(request.key)
        self._request = request
        self._weight = request.weight
        self._dt = self._config.dt_seconds
        self._rng = np.random.default_rng(request.seed)
        self._setup = self._build_setup()
        self._reward_fn = make_reward_fn(
            alpha=self._config.reward.alpha,
            gamma=self._config.reward.gamma,
            kappa=self._config.reward.kappa,
            s_thr=self._config.exposure.s_thr_w_m2,
            max_tx_power_mw=self._setup.tx_power_mw,
            max_throughput_ref_mbps=max(
                1.0,
                max_throughput_reference_mbps(
                    self._config.users.n_users,
                    self._config.radio.bandwidth_mhz,
                    self._config.radio.spectral_efficiency_cap,
                ),
            ),
            reward_clip=self._config.reward.reward_clip,
        )
        self._policy = policy if policy is not None else self._build_policy()
        self._policy_name = (
            self._request.policy or self._config.policy.name
            if policy is None
            else policy_name_for(policy)
        )
        # The no-mitigation reference every frame is scored against: the same
        # state, the same cells, the same users, but the scenario's own greedy
        # full-power beam with no mitigation. Evaluating it here is what makes
        # the frame's ``baseline`` block a true same-step before/after pair
        # rather than a second stream that has to be time-aligned.
        self._baseline_policy: BeamPolicy = UnconstrainedPolicy()
        self._baseline_name = "unconstrained"
        self._users = self._spawn_users()
        self._flock = self._spawn_flock()
        self._step = 0

    def initial_frame(self) -> StateFrame:
        """Return the frame for ``step == 0`` without advancing the scene."""
        return self._frame()

    def step(self) -> StateFrame:
        """Advance the flock and the users one step, then render the frame."""
        self._step += 1
        self._flock = generate_flock_step(self._flock, dt_seconds=self._dt, rng=None)
        if self._flock.polar_from(0.0, 0.0)[1] > self._config.flock.recycle_range_m:
            self._flock = self._spawn_flock()
        area = self._config.users.area_m
        self._users = [
            generate_user_step_random_waypoint(
                user,
                min_x_m=-area,
                max_x_m=area,
                min_y_m=-area,
                max_y_m=area,
                dt_seconds=self._dt,
                rng=self._rng,
            )
            for user in self._users
        ]
        return self._frame()

    def _build_setup(self) -> RadioSetup:
        array = self._config.array
        codebook_cfg = self._config.codebook
        radio = self._config.radio
        beams = build_dft_codebook(
            codebook_cfg.n_beams,
            array.n_rows,
            array.n_cols,
            array.d_lambda,
            elevation_deg=array.elevation_deg,
            azimuth_span_deg=codebook_cfg.azimuth_span_deg,
        )
        return RadioSetup(
            codebook=tuple(beams),
            n_rows=array.n_rows,
            n_cols=array.n_cols,
            d_lambda=array.d_lambda,
            tx_power_mw=dbm_to_mw(radio.tx_power_dbm),
            bandwidth_mhz=radio.bandwidth_mhz,
            fc_ghz=radio.fc_ghz,
            noise_power_mw=thermal_noise_power_mw(radio.bandwidth_mhz * 1e6, radio.noise_figure_db),
            interference_mw=dbm_to_mw(radio.interference_dbm),
            s_thr_w_m2=self._config.exposure.s_thr_w_m2,
            spectral_efficiency_cap=radio.spectral_efficiency_cap,
            h_bs_m=radio.h_bs_m,
            h_ut_m=radio.h_ut_m,
        )

    def _build_policy(self) -> BeamPolicy:
        """Build the controller named by the request, else the scenario's own."""
        name = self._request.policy or self._config.policy.name
        return build_policy(name, self._config.policy)

    def _spawn_flock(self) -> FlockState:
        flock_cfg = self._config.flock
        flock = generate_flock(
            self._rng,
            bearing_deg=flock_cfg.bearing_deg,
            range_m=flock_cfg.range_m,
            speed_mps=flock_cfg.speed_ms,
            altitude_min_m=flock_cfg.altitude_min_m,
            altitude_max_m=flock_cfg.altitude_max_m,
            extent_m=flock_cfg.extent_m,
            n_individuals=flock_cfg.n_individuals,
        )
        # The entry heading is a scenario parameter, not a random draw.
        return replace(flock, centroid_heading_deg=flock_cfg.heading_deg)

    def _spawn_users(self) -> list[UserState]:
        users_cfg = self._config.users
        center = users_cfg.bearing_center_deg
        half_width = users_cfg.bearing_half_width_deg
        if center is None or half_width is None:
            return generate_users(
                self._rng,
                users_cfg.n_users,
                min_range_m=users_cfg.min_range_m,
                max_range_m=users_cfg.max_range_m,
                vehicular_fraction=users_cfg.vehicular_fraction,
                area_m=users_cfg.area_m,
            )

        low_sq = users_cfg.min_range_m**2
        high_sq = users_cfg.max_range_m**2
        users: list[UserState] = []
        for _ in range(users_cfg.n_users):
            radius_m = math.sqrt(float(self._rng.uniform(low_sq, high_sq)))
            bearing_rad = math.radians(
                float(self._rng.uniform(center - half_width, center + half_width))
            )
            vehicular = bool(self._rng.uniform() < users_cfg.vehicular_fraction)
            speed_range = VEHICULAR_SPEED_MPS if vehicular else PEDESTRIAN_SPEED_MPS
            users.append(
                UserState(
                    x_m=radius_m * math.cos(bearing_rad),
                    y_m=radius_m * math.sin(bearing_rad),
                    heading_deg=float(self._rng.uniform(0.0, 360.0)),
                    speed_mps=float(self._rng.uniform(*speed_range)),
                    vehicular=vehicular,
                )
            )
        return users

    def _footprint_cells(self) -> tuple[CellGeometry, ...]:
        """Lay the deterministic footprint spiral onto the flock altitude band."""
        positions = flock_footprint_cells(self._flock, self._config.exposure.n_cells, rng=None)
        dz_m = self._flock.altitude_mean_m - self._setup.h_bs_m
        cells: list[CellGeometry] = []
        for x_m, y_m in positions:
            ground_m = math.hypot(x_m, y_m)
            cells.append(
                CellGeometry(
                    bearing_deg=normalize_azimuth(math.degrees(math.atan2(y_m, x_m))),
                    elevation_deg=math.degrees(math.atan2(dz_m, ground_m)),
                    slant_range_m=math.hypot(ground_m, dz_m),
                )
            )
        return tuple(cells)

    def _user_geometry(self) -> tuple[tuple[UserLink, ...], list[tuple[float, float]]]:
        """Return policy-facing links and wire positions for the live users."""
        dz_m = self._setup.h_ut_m - self._setup.h_bs_m
        links: list[UserLink] = []
        positions: list[tuple[float, float]] = []
        for user in self._users:
            bearing_deg, ground_m = user.polar_from(0.0, 0.0)
            links.append(
                UserLink(
                    bearing_deg=normalize_azimuth(bearing_deg),
                    elevation_deg=math.degrees(math.atan2(dz_m, ground_m)),
                    range_m=math.hypot(ground_m, dz_m),
                )
            )
            positions.append((user.x_m, user.y_m))
        return tuple(links), positions

    def _log10_weight(self) -> float:
        """Preference weight on the log10 scale the reward expects (Decision D2)."""
        if self._weight <= 0.0:
            return float("-inf")
        return math.log10(self._weight)

    def _frame(self) -> StateFrame:
        """Compute one frame from the live scene state."""
        config = self._config
        bearing_deg, range_m = self._flock.polar_from(0.0, 0.0)
        flock_bearing_deg = normalize_azimuth(bearing_deg)
        ground_m = math.hypot(self._flock.centroid_x_m, self._flock.centroid_y_m)
        dz_m = self._flock.altitude_mean_m - self._setup.h_bs_m
        flock_elevation_deg = math.degrees(math.atan2(dz_m, ground_m))

        cells = self._footprint_cells()
        user_links, user_positions = self._user_geometry()
        state = LiveState(
            step=self._step,
            time_s=self._step * self._dt,
            flock_bearing_deg=flock_bearing_deg,
            flock_elevation_deg=flock_elevation_deg,
            flock_range_m=range_m,
            users=user_links,
            cells=cells,
            flock_altitude_mean_m=self._flock.altitude_mean_m,
            flock_extent_m=self._flock.extent_m,
            flock_heading_deg=self._flock.centroid_heading_deg,
            flock_speed_ms=self._flock.speed_mps,
            log10_w=self._log10_weight(),
        )

        started_ns = time.perf_counter_ns()
        action = self._policy.act(state, self._setup)
        inference_latency_ms = (time.perf_counter_ns() - started_ns) / 1.0e6

        # What the policy actually transmits. Every downstream quantity -- the
        # footprint density, the exposure metrics, the served links and the
        # reward -- is scored on this power, so a back-off or a shut-down is a
        # real change in the simulation and on the wire, not just a label.
        power_mw = action.power_mw
        weights = action.weights
        transmitting = power_mw > 0.0 and bool(np.any(weights != 0.0))
        # A carrier that is off cannot be radioing a null, whatever the policy
        # called its action.
        null_active = action.null_active and transmitting
        detail = action.detail
        if transmitting:
            serving_gain_dbi = composite_gain_dbi(
                weights,
                action.beam.elevation_deg,
                action.beam.azimuth_deg,
                self._setup.n_rows,
                self._setup.n_cols,
                self._setup.d_lambda,
            )
        else:
            # Nothing is radiated. Report the beam the policy would have served
            # so the frame carries the geometry alongside ``power_mw == 0``
            # instead of a non-finite gain.
            serving_gain_dbi = composite_gain_dbi(
                action.beam.weights,
                action.beam.elevation_deg,
                action.beam.azimuth_deg,
                self._setup.n_rows,
                self._setup.n_cols,
                self._setup.d_lambda,
            )
        if transmitting and not math.isfinite(serving_gain_dbi):
            # Safety net: never put a non-finite gain or a dead beam on the wire.
            weights = action.beam.weights
            power_mw = action.power_mw
            null_active = False
            serving_gain_dbi = composite_gain_dbi(
                weights,
                action.beam.elevation_deg,
                action.beam.azimuth_deg,
                self._setup.n_rows,
                self._setup.n_cols,
                self._setup.d_lambda,
            )
            detail = (
                f"{action.detail} The nulled weights were discarded: they annihilated "
                "the serving beam, so the un-nulled beam was transmitted instead."
            )

        densities = cell_densities_w_m2(weights, cells, self._setup, power_mw=power_mw)
        exposure = exposure_metrics_from_density(
            densities, config.exposure.s_thr_w_m2, dt_seconds=self._dt
        )
        # Measured, never asserted: the exposure reduction this step's
        # transmitted weights bought against the same scenario's un-nulled beam
        # over the same footprint cells. Exactly 0.0 when no null is on the air;
        # it can come out slightly negative, because a projection null
        # redistributes power into directions that previously sat in a null.
        transmitted_p95 = exposure["s_p95"]
        # Measured against the un-nulled beam at the SAME transmit power, so the
        # depth reports what the null bought and never what a back-off bought.
        un_nulled_p95 = exposure_p95_w_m2(
            action.beam.weights, cells, self._setup, power_mw=power_mw
        )
        null_depth_db = (
            10.0 * math.log10(un_nulled_p95 / transmitted_p95)
            if null_active and un_nulled_p95 > 0.0 and transmitted_p95 > 0.0
            else 0.0
        )
        budgets = link_budgets(weights, state, self._setup, power_mw=power_mw)
        sum_mbps = sum_served_throughput_mbps(budgets)
        result = self._reward_fn(
            sum_throughput_mbps=sum_mbps,
            tx_power_mw=power_mw,
            s_p95_w_per_m2=exposure["s_p95"],
            observation={"log10_w": self._log10_weight()},
        )

        # The same step without mitigation, so every frame carries its own
        # before/after pair for a judge to read.
        reference = self._baseline_policy.act(state, self._setup)
        reference_densities = cell_densities_w_m2(
            reference.weights, cells, self._setup, power_mw=reference.power_mw
        )
        reference_exposure = exposure_metrics_from_density(
            reference_densities, config.exposure.s_thr_w_m2, dt_seconds=self._dt
        )
        reference_mbps = sum_served_throughput_mbps(
            link_budgets(reference.weights, state, self._setup, power_mw=reference.power_mw)
        )
        baseline = BaselineMetrics(
            policy_name=self._baseline_name,
            power_mw=reference.power_mw,
            sum_throughput_mbps=reference_mbps,
            p95_density_wm2=reference_exposure["s_p95"],
            max_density_wm2=reference_exposure["s_max"],
            frac_over=reference_exposure["frac_over"],
            dose_wsm2=reference_exposure["dose"],
        )

        users = [
            WireUserState(
                position=MapPoint(x=x_m, y=y_m),
                bearing_deg=link.bearing_deg,
                range_m=link.range_m,
                sinr=_wire_sinr_db(budget.sinr_db),
                throughput_mbps=budget.throughput_mbps,
            )
            for (x_m, y_m), link, budget in zip(user_positions, user_links, budgets, strict=True)
        ]

        power_frac = power_mw / self._setup.tx_power_mw if self._setup.tx_power_mw > 0.0 else 0.0
        explanation = (
            f"Step {self._step}: beam {action.beam.index} at {action.beam.azimuth_deg:.1f} deg, "
            f"{'null active' if null_active else 'null off'} ({action.decision.value}); "
            f"{detail}. "
            f"Flock {range_m:.0f} m at {flock_bearing_deg:.1f} deg; "
            f"peak density {exposure['s_max']:.3f} W/m\u00b2, "
            f"p95 {exposure['s_p95']:.3f} W/m\u00b2, "
            f"{exposure['frac_over']:.0%} of cells over "
            f"{config.exposure.s_thr_w_m2:.3f} W/m\u00b2; "
            f"sum rate {sum_mbps:.0f} Mbps at {power_frac:.0%} power "
            f"(unmitigated at this step: p95 {baseline.p95_density_wm2:.3f} W/m\u00b2, "
            f"{baseline.sum_throughput_mbps:.0f} Mbps), reward {result['reward']:.2f}."
        )

        return StateFrame(
            step=self._step,
            time_s=self._step * self._dt,
            dt_s=self._dt,
            sector=MapPoint(x=0.0, y=0.0),
            beam=BeamState(
                index=action.beam.index,
                azimuth_deg=action.beam.azimuth_deg,
                zenith_deg=action.beam.elevation_deg,
                gain_dbi=serving_gain_dbi,
            ),
            power_mw=power_mw,
            null_active=null_active,
            null_depth_db=null_depth_db,
            inference_latency_ms=inference_latency_ms,
            policy_name=self._policy_name,
            flock=WireFlockState(
                centroid=MapPoint(x=self._flock.centroid_x_m, y=self._flock.centroid_y_m),
                heading_deg=self._flock.centroid_heading_deg,
                speed_ms=self._flock.speed_mps,
                altitude_min_m=self._flock.altitude_min_m,
                altitude_max_m=self._flock.altitude_max_m,
                extent_m=self._flock.extent_m,
                n_cels=config.exposure.n_cells,
            ),
            users=users,
            baseline=baseline,
            metrics=StepMetrics(
                sum_throughput_mbps=sum_mbps,
                p95_density_wm2=exposure["s_p95"],
                max_density_wm2=exposure["s_max"],
                frac_over=exposure["frac_over"],
                dose_wsm2=exposure["dose"],
                s_thr_wm2=config.exposure.s_thr_w_m2,
                exposure_penalty=result["terms"]["exposure"],
                throughput_term=result["terms"]["throughput"],
                power_term=result["terms"]["power"],
                reward=result["reward"],
            ),
            explanation=explanation,
            weight=self._weight,
        )


def demo_runner_factory(request: ScenarioRequest) -> DemoRunner:
    """Build the seeded demo runner the service uses when no policy is plugged in."""
    return DemoRunner(request)
