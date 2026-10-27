"""Gymnasium environment: ``BeamEnv`` (spec Section 4.3, Phase 3).

The environment wraps the *same* primitives the demo twin uses -- the TR 38.901
path loss, the sector UPA and DFT codebook, projection nulling, the flock and
user mobility models, the footprint exposure metrics and
:func:`flywaydqn.env.reward.make_reward_fn` -- so a policy trained here and a
policy streamed by the service are scored against identical physics. It does not
re-implement any of them.

MDP (spec Section 4.3)
---------------------
- **Observation** (``Box``, 12 sector features followed by 4 features for each of
  the ``users_observed`` served UEs, all scaled into ``[-1, 1]``): current beam
  index, current power level, previous sum throughput and previous p95 exposure,
  the preference weight on its log10 scale, the flock summary (bearing, range,
  altitude, extent, heading, density ratio), and per observed UE the SINR, rate,
  bearing and range under the *transmitted* weights.
- **Action** (``Discrete``, flattened ``n_beams * n_power_levels * 2``): a codebook
  beam, a transmit-power level, and a null on/off flag (spec Decision D4 keeps the
  flattened form, default 16 x 4 x 2 = 128).
- **Transition**: the flock and users take one mobility step, the footprint and
  user geometry are recomputed, and the radio is re-scored under the new action.
- **Episode**: fixed horizon ``episode_steps`` with a fresh randomized scenario per
  ``reset`` (flock entry bearing, range and speed, user layout, and the sampled
  preference weight ``log10(w)``).

Determinism: every random draw flows through one seeded ``numpy`` generator, so
``reset(seed=s)`` followed by the same action sequence reproduces an episode
exactly.

Units
-----
- angles: degrees (azimuth from panel boresight, elevation above the horizon)
- distances: meters (m)
- power: milliwatts (mW)
- power density: watts per square meter (W/m^2)
- throughput: megabits per second (Mbps)
- time: seconds (s)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import yaml
from gymnasium import spaces
from pydantic import BaseModel, ConfigDict, Field

from flywaydqn.agents.baselines_demo import (
    CellGeometry,
    LiveState,
    RadioSetup,
    UserLink,
    cell_densities_w_m2,
    link_budgets,
    peak_cell_geometry,
    sum_served_throughput_mbps,
    transmit_weights,
)
from flywaydqn.env.reward import RewardTerms, make_reward_fn
from flywaydqn.exposure.density import s_p95
from flywaydqn.exposure.metrics import ExposureMetrics, exposure_metrics_from_density
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
from flywaydqn.rf.array import DEFAULT_D_LAMBDA, composite_gain_dbi
from flywaydqn.rf.codebook import DEFAULT_AZIMUTH_SPAN_DEG, build_dft_codebook, normalize_azimuth
from flywaydqn.rf.sinr import DEFAULT_SPECTRAL_EFFICIENCY_CAP, max_throughput_reference_mbps
from flywaydqn.units import dbm_to_mw, thermal_noise_power_mw

# Environment variable that overrides where the env YAML is loaded from.
CONFIG_DIR_ENV_VAR: Final[str] = "FLYWAYDQN_ENV_CONFIG"

# Sector-level observation block: beam, power, previous throughput, previous
# exposure, preference weight, then the six flock features.
SECTOR_FEATURES: Final[int] = 12
# Per observed UE: SINR, throughput, bearing, range.
USER_FEATURES: Final[int] = 4
# Scale factors that map raw features into roughly [-1, 1] before clipping.
_SINR_SCALE_DB: Final[float] = 40.0
_BEARING_SCALE_DEG: Final[float] = 180.0
_HEADING_SCALE_DEG: Final[float] = 360.0
_RANGE_SCALE_M: Final[float] = 1000.0
_ALTITUDE_SCALE_M: Final[float] = 100.0
_EXTENT_SCALE_M: Final[float] = 100.0

# Registered so the environment can be built with ``gymnasium.make`` beside its
# spec: ``gymnasium.utils.env_checker`` and the training entry point both expect
# an environment that has one.
ENV_ID: Final[str] = "FlywayDQN/BeamEnv-v0"

gym.register(id=ENV_ID, entry_point="flywaydqn.env.beam_env:BeamEnv")

__all__ = [
    "CONFIG_DIR_ENV_VAR",
    "ENV_ID",
    "SECTOR_FEATURES",
    "USER_FEATURES",
    "BeamEnv",
    "BeamEnvConfig",
    "DecodedAction",
    "EpisodeConfig",
    "ExposureConfig",
    "RadioConfig",
    "RewardConfig",
    "env_config_path",
    "load_env_config",
]


class RewardConfig(BaseModel):
    """Reward weights and shaping (spec Section 4.4, Decision D2/D3)."""

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(1.0, gt=0.0)
    gamma: float = Field(0.02, ge=0.0)
    log10_w_min: float = -1.0
    log10_w_max: float = 2.0
    kappa: float = Field(5.0, gt=0.0)
    reward_clip: float = Field(10.0, gt=0.0)


class EpisodeConfig(BaseModel):
    """Per-episode randomized scenario ranges (spec Section 4.3)."""

    model_config = ConfigDict(extra="forbid")

    n_users: int = Field(8, ge=0)
    user_min_range_m: float = Field(50.0, gt=0.0)
    user_max_range_m: float = Field(400.0, gt=0.0)
    user_area_m: float = Field(500.0, gt=0.0)
    vehicular_fraction: float = Field(0.3, ge=0.0, le=1.0)
    flock_n_individuals: int = Field(40, ge=1)
    flock_min_range_m: float = Field(250.0, gt=0.0)
    flock_max_range_m: float = Field(700.0, gt=0.0)
    flock_speed_ms: float = Field(12.0, gt=0.0)
    flock_speed_spread: float = Field(0.2, ge=0.0, le=1.0)
    flock_altitude_min_m: float = Field(20.0, ge=0.0)
    flock_altitude_max_m: float = Field(80.0, ge=0.0)
    flock_extent_m: float = Field(30.0, gt=0.0)
    flock_recycle_range_m: float = Field(900.0, gt=0.0)


class RadioConfig(BaseModel):
    """Radio parameters shared by every step of an episode."""

    model_config = ConfigDict(extra="forbid")

    bandwidth_mhz: float = Field(100.0, gt=0.0)
    tx_power_dbm: float = 53.0
    noise_figure_db: float = 7.0
    interference_dbm: float = -80.0
    spectral_efficiency_cap: float = Field(DEFAULT_SPECTRAL_EFFICIENCY_CAP, gt=0.0)
    h_bs_m: float = Field(25.0, gt=0.0)
    h_ut_m: float = Field(1.5, ge=0.0)


class ExposureConfig(BaseModel):
    """Exposure threshold and footprint sampling (spec Section 4.2)."""

    model_config = ConfigDict(extra="forbid")

    # Default anchor: ICNIRP 2020 general-public reference level for 2-300 GHz.
    # Configurable, and never asserted as an avian-harm threshold (spec 2.2).
    s_thr_w_m2: float = Field(10.0, gt=0.0)
    n_cells: int = Field(64, ge=1)


class ObservationConfig(BaseModel):
    """How many served UEs the observation summarises."""

    model_config = ConfigDict(extra="forbid")

    users_observed: int = Field(8, ge=0)


class BeamEnvConfig(BaseModel):
    """Everything one :class:`BeamEnv` needs, mirroring ``configs/env/env.yaml``."""

    model_config = ConfigDict(extra="forbid")

    carrier_hz: float = Field(3.5e9, gt=0.0)
    scenario: str = "UMa"
    array_rows: int = Field(8, ge=1)
    array_cols: int = Field(8, ge=1)
    element_spacing_lambda: float = Field(DEFAULT_D_LAMBDA, gt=0.0)
    n_beams_azimuth: int = Field(16, ge=1)
    n_beams_elevation: int = Field(1, ge=1)
    azimuth_span_deg: tuple[float, float] = DEFAULT_AZIMUTH_SPAN_DEG
    power_levels_dbm: tuple[float, ...] = (44.0, 47.0, 50.0, 53.0)
    dt_seconds: float = Field(0.1, gt=0.0)
    episode_steps: int = Field(600, ge=1)
    seed: int = Field(1, ge=0)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    episode: EpisodeConfig = Field(default_factory=EpisodeConfig)
    radio: RadioConfig = Field(default_factory=RadioConfig)
    exposure: ExposureConfig = Field(default_factory=ExposureConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)


def env_config_path() -> Path:
    """Return the env YAML path: the override variable, then the repository copy."""
    override = os.environ.get(CONFIG_DIR_ENV_VAR)
    if override:
        return Path(override)
    from_source = Path(__file__).resolve().parents[3] / "configs" / "env" / "env.yaml"
    if from_source.is_file():
        return from_source
    return Path.cwd() / "configs" / "env" / "env.yaml"


def load_env_config(path: Path | None = None) -> BeamEnvConfig:
    """Load and validate the environment config, defaulting to ``configs/env/env.yaml``."""
    target = path if path is not None else env_config_path()
    with target.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return BeamEnvConfig.model_validate(raw)


@dataclass(frozen=True)
class DecodedAction:
    """The three factors a flattened action index stands for."""

    beam_index: int
    power_index: int
    null_active: bool
    power_mw: float


class BeamEnv(gym.Env[npt.NDArray[np.float64], np.int64]):
    """Preference-conditioned beam/power/null environment (spec Section 4.3).

    Parameters
    ----------
    config : BeamEnvConfig | None
        Environment configuration. ``None`` loads ``configs/env/env.yaml``.
    seed : int | None
        Overrides ``config.seed`` for the episode generator.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, config: BeamEnvConfig | None = None, seed: int | None = None) -> None:
        super().__init__()
        self._config = config if config is not None else load_env_config()
        self._seed = self._config.seed if seed is None else seed
        self._rng = np.random.default_rng(self._seed)
        self._setup = self._build_setup()

        self._n_beams = len(self._setup.codebook)
        self._n_levels = len(self._config.power_levels_dbm)
        self.n_actions = self._n_beams * self._n_levels * 2
        self._max_throughput_ref_mbps = max(
            1.0,
            max_throughput_reference_mbps(
                self._config.episode.n_users,
                self._config.radio.bandwidth_mhz,
                self._config.radio.spectral_efficiency_cap,
            ),
        )
        self._reward_fn = make_reward_fn(
            alpha=self._config.reward.alpha,
            gamma=self._config.reward.gamma,
            kappa=self._config.reward.kappa,
            s_thr=self._config.exposure.s_thr_w_m2,
            max_tx_power_mw=max(dbm_to_mw(level) for level in self._config.power_levels_dbm),
            max_throughput_ref_mbps=self._max_throughput_ref_mbps,
            reward_clip=self._config.reward.reward_clip,
        )

        self._obs_users = self._config.observation.users_observed
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(SECTOR_FEATURES + USER_FEATURES * self._obs_users,),
            dtype=np.float64,
        )

        # Episode state, filled by reset().
        self._step = 0
        self._log10_w = 0.0
        self._flock = self._spawn_flock()
        self._users = self._spawn_users()
        self._beam_index = 0
        self._power_index = self._n_levels - 1
        self._power_mw = dbm_to_mw(self._config.power_levels_dbm[self._power_index])
        self._weights = self._setup.codebook[0].weights
        self._prev_throughput_mbps = 0.0
        self._prev_s_p95 = 0.0

    # -- spaces and configuration ------------------------------------------------

    @property
    def config(self) -> BeamEnvConfig:
        """The validated configuration this environment was built with."""
        return self._config

    def decode_action(self, action: np.int64) -> DecodedAction:
        """Decode a flattened action index into beam, power level and null flag.

        Layout: ``beam_index * (n_levels * 2) + power_index * 2 + null_flag``.

        Raises
        ------
        ValueError
            If ``action`` is outside ``action_space``.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"action {action!r} is outside action_space {self.action_space}")
        beam_index, remainder = divmod(int(action), self._n_levels * 2)
        power_index, null_flag = divmod(remainder, 2)
        return DecodedAction(
            beam_index=beam_index,
            power_index=power_index,
            null_active=bool(null_flag),
            power_mw=dbm_to_mw(self._config.power_levels_dbm[power_index]),
        )

    # -- gym API -----------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        """Start a fresh randomized episode and return its first observation."""
        del options  # no reset options are defined
        super().reset(seed=seed)
        self._rng = np.random.default_rng(self._seed if seed is None else seed)
        self._step = 0
        self._log10_w = float(
            self._rng.uniform(self._config.reward.log10_w_min, self._config.reward.log10_w_max)
        )
        self._flock = self._spawn_flock()
        self._users = self._spawn_users()
        self._beam_index = 0
        self._power_index = self._n_levels - 1
        self._power_mw = dbm_to_mw(self._config.power_levels_dbm[self._power_index])
        self._weights = self._setup.codebook[0].weights
        self._prev_throughput_mbps = 0.0
        self._prev_s_p95 = 0.0
        observation = self._observation()
        return observation, self._info()

    def step(
        self, action: np.int64
    ) -> tuple[npt.NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        """Advance one step under ``action`` and return the Gymnasium 5-tuple."""
        decoded = self.decode_action(action)

        # Mobility first: the radio is scored against the geometry the action
        # will actually serve (spec 4.3 transition).
        self._users = [self._step_user(user) for user in self._users]
        self._flock = self._step_flock(self._flock)
        self._step += 1

        cells = self._footprint_cells(self._flock)
        state = LiveState(
            step=self._step,
            time_s=self._step * self._config.dt_seconds,
            flock_bearing_deg=self._flock_bearing_deg(),
            flock_elevation_deg=self._flock_elevation_deg(),
            flock_range_m=math.hypot(self._flock.centroid_x_m, self._flock.centroid_y_m),
            users=self._user_links(),
            cells=cells,
            flock_altitude_mean_m=self._flock.altitude_mean_m,
            flock_extent_m=self._flock.extent_m,
            flock_heading_deg=self._flock.centroid_heading_deg,
            flock_speed_ms=self._flock.speed_mps,
            log10_w=self._log10_w,
        )

        beam = self._setup.codebook[decoded.beam_index]
        weights = beam.weights
        null_active = False
        if decoded.null_active:
            peak = peak_cell_geometry(beam.weights, cells, self._setup)
            candidate = transmit_weights(
                beam,
                self._setup,
                null_elevation_deg=peak.elevation_deg,
                null_azimuth_deg=peak.bearing_deg,
            )
            # A projection null can annihilate the serving beam (spec finding 2);
            # never transmit a non-finite gain.
            if math.isfinite(
                composite_gain_dbi(
                    candidate,
                    beam.elevation_deg,
                    beam.azimuth_deg,
                    self._setup.n_rows,
                    self._setup.n_cols,
                    self._setup.d_lambda,
                )
            ):
                weights = candidate
                null_active = True

        step_setup = replace(self._setup, tx_power_mw=decoded.power_mw)
        densities = cell_densities_w_m2(weights, cells, step_setup)
        exposure = exposure_metrics_from_density(
            densities, self._config.exposure.s_thr_w_m2, dt_seconds=self._config.dt_seconds
        )
        budgets = link_budgets(weights, state, step_setup)
        sum_mbps = sum_served_throughput_mbps(budgets)
        result = self._reward_fn(
            sum_throughput_mbps=sum_mbps,
            tx_power_mw=decoded.power_mw,
            s_p95_w_per_m2=exposure["s_p95"],
            observation={"log10_w": self._log10_w},
        )

        self._weights = weights
        self._beam_index = decoded.beam_index
        self._power_index = decoded.power_index
        self._power_mw = decoded.power_mw
        self._prev_throughput_mbps = sum_mbps
        self._prev_s_p95 = exposure["s_p95"]

        terminated = False
        truncated = self._step >= self._config.episode_steps
        observation = self._observation()
        info = self._info(
            null_active=null_active,
            exposure=exposure,
            sum_throughput_mbps=sum_mbps,
            reward_terms=result["terms"],
            user_throughput_mbps=[budget.throughput_mbps for budget in budgets],
            user_sinr_db=[budget.sinr_db for budget in budgets],
        )
        return observation, float(result["reward"]), terminated, truncated, info

    # -- rule-based policy bridge -------------------------------------------

    def policy_state(self) -> tuple[LiveState, RadioSetup]:
        """Return the live geometry and radio setup for scoring a rule-based policy.

        Exposed so an evaluation script can run :class:`BeamPolicy` baselines on
        exactly the geometry the learned controller sees, instead of rebuilding
        the scene.

        The returned :class:`RadioSetup` is the *scene's* static setup, so its
        ``tx_power_mw`` is the sector's configured full power for every step --
        the same value :class:`~flywaydqn.demo.runner.DemoRunner` hands its
        policy. It is deliberately **not** the power transmitted on the previous
        step: a policy that backs off reads ``tx_power_mw`` as full power and
        multiplies it by a fixed fraction, so passing the transient power back
        would compound the reduction every step (a \u00d70.5 back-off would walk
        200 W \u2192 100 W \u2192 50 W \u2192 25 W and then pin at the lowest
        level) and make the baseline's measured row describe a policy nobody
        configured.

        Note that this surface snaps a requested power onto the environment's
        ``power_levels_dbm`` grid (see :meth:`action_index_for`), and that grid
        has no zero-power action. A policy that shuts the carrier off cannot be
        expressed here and must be measured on the demo runner, which applies
        ``BeamAction.power_mw`` exactly.
        """
        cells = self._footprint_cells(self._flock)
        return self._live_state(cells), self._setup

    def action_index_for(self, beam_index: int, power_mw: float, null_active: bool) -> np.int64:
        """Map a rule-based decision onto the nearest flattened action index.

        The power level closest to ``power_mw`` on the log scale is chosen, so a
        baseline that transmits at a level the env does not offer is evaluated at
        the nearest one rather than silently at full power.
        """
        levels_mw = np.asarray(
            [dbm_to_mw(level) for level in self._config.power_levels_dbm], dtype=np.float64
        )
        power_index = int(np.argmin(np.abs(np.log10(levels_mw) - math.log10(max(power_mw, 1e-12)))))
        return np.int64(beam_index * (self._n_levels * 2) + power_index * 2 + int(null_active))

    # -- internal helpers --------------------------------------------------------

    def _build_setup(self) -> RadioSetup:
        """Assemble the static radio geometry for this configuration."""
        config = self._config
        codebook = build_dft_codebook(
            n_beams=config.n_beams_azimuth,
            n_rows=config.array_rows,
            n_cols=config.array_cols,
            d_lambda=config.element_spacing_lambda,
            elevation_deg=0.0,
            azimuth_span_deg=config.azimuth_span_deg,
        )
        radio = config.radio
        return RadioSetup(
            codebook=tuple(codebook),
            n_rows=config.array_rows,
            n_cols=config.array_cols,
            d_lambda=config.element_spacing_lambda,
            tx_power_mw=dbm_to_mw(radio.tx_power_dbm),
            bandwidth_mhz=radio.bandwidth_mhz,
            fc_ghz=config.carrier_hz / 1e9,
            noise_power_mw=thermal_noise_power_mw(radio.bandwidth_mhz * 1e6, radio.noise_figure_db),
            interference_mw=dbm_to_mw(radio.interference_dbm),
            s_thr_w_m2=config.exposure.s_thr_w_m2,
            spectral_efficiency_cap=radio.spectral_efficiency_cap,
            h_bs_m=radio.h_bs_m,
            h_ut_m=radio.h_ut_m,
        )

    def _spawn_flock(self) -> FlockState:
        """Sample one inbound flock crossing for a new episode."""
        episode = self._config.episode
        span_low, span_high = self._config.azimuth_span_deg
        bearing_deg = float(self._rng.uniform(span_low, span_high))
        range_m = float(self._rng.uniform(episode.flock_min_range_m, episode.flock_max_range_m))
        speed_scale = 1.0 + float(
            self._rng.uniform(-episode.flock_speed_spread, episode.flock_speed_spread)
        )
        flock = generate_flock(
            self._rng,
            bearing_deg=bearing_deg,
            range_m=range_m,
            speed_mps=episode.flock_speed_ms * speed_scale,
            altitude_min_m=episode.flock_altitude_min_m,
            altitude_max_m=episode.flock_altitude_max_m,
            extent_m=episode.flock_extent_m,
            n_individuals=episode.flock_n_individuals,
        )
        # Fly inbound through the sector with a little natural wander.
        heading_deg = normalize_azimuth(bearing_deg + 180.0 + float(self._rng.uniform(-20.0, 20.0)))
        return replace(flock, centroid_heading_deg=heading_deg)

    def _spawn_users(self) -> list[UserState]:
        """Sample the user layout for a new episode."""
        episode = self._config.episode
        return generate_users(
            self._rng,
            episode.n_users,
            min_range_m=episode.user_min_range_m,
            max_range_m=episode.user_max_range_m,
            vehicular_fraction=episode.vehicular_fraction,
            area_m=episode.user_area_m,
        )

    def _step_user(self, user: UserState) -> UserState:
        """Advance one user by a bounded random-waypoint step."""
        area = self._config.episode.user_area_m
        return generate_user_step_random_waypoint(
            user,
            min_x_m=-area,
            max_x_m=area,
            min_y_m=-area,
            max_y_m=area,
            dt_seconds=self._config.dt_seconds,
            rng=self._rng,
        )

    def _step_flock(self, flock: FlockState) -> FlockState:
        """Advance the flock, respawning it inbound once it leaves the sector."""
        stepped = generate_flock_step(flock, dt_seconds=self._config.dt_seconds, rng=self._rng)
        if math.hypot(stepped.centroid_x_m, stepped.centroid_y_m) > (
            self._config.episode.flock_recycle_range_m
        ):
            rescale = self._respawn_flock()
            return rescale
        return stepped

    def _respawn_flock(self) -> FlockState:
        """Re-enter the sector from the configured entry range, same species count."""
        episode = self._config.episode
        bearing_deg = float(self._rng.uniform(*self._config.azimuth_span_deg))
        flock = generate_flock(
            self._rng,
            bearing_deg=bearing_deg,
            range_m=float(self._rng.uniform(episode.flock_min_range_m, episode.flock_max_range_m)),
            speed_mps=episode.flock_speed_ms,
            altitude_min_m=episode.flock_altitude_min_m,
            altitude_max_m=episode.flock_altitude_max_m,
            extent_m=episode.flock_extent_m,
            n_individuals=episode.flock_n_individuals,
        )
        return replace(
            flock,
            centroid_heading_deg=normalize_azimuth(bearing_deg + 180.0),
        )

    def _flock_above_antenna_m(self) -> float:
        """Mean height of the flock above the antenna, in meters."""
        return self._flock.altitude_mean_m - self._setup.h_bs_m

    def _flock_bearing_deg(self) -> float:
        """Flock centroid bearing from panel boresight, in degrees."""
        return normalize_azimuth(
            math.degrees(math.atan2(self._flock.centroid_y_m, self._flock.centroid_x_m))
        )

    def _flock_elevation_deg(self) -> float:
        """Flock centroid elevation above the horizon, in degrees."""
        ground_m = math.hypot(self._flock.centroid_x_m, self._flock.centroid_y_m)
        return math.degrees(math.atan2(self._flock_above_antenna_m(), ground_m))

    def _footprint_cells(self, flock: FlockState) -> tuple[CellGeometry, ...]:
        """Lay the deterministic footprint spiral over the flock altitude band."""
        positions = flock_footprint_cells(flock, self._config.exposure.n_cells, rng=None)
        dz_m = self._flock_above_antenna_m()
        cells: list[CellGeometry] = []
        for x_m, y_m in positions:
            ground_m = math.hypot(x_m, y_m)
            cell_elevation_deg = math.degrees(math.atan2(dz_m, ground_m))
            cells.append(
                CellGeometry(
                    bearing_deg=normalize_azimuth(math.degrees(math.atan2(y_m, x_m))),
                    elevation_deg=cell_elevation_deg,
                    slant_range_m=math.hypot(ground_m, dz_m),
                )
            )
        return tuple(cells)

    def _user_links(self) -> tuple[UserLink, ...]:
        """Return the current user geometry seen from the antenna."""
        links: list[UserLink] = []
        for user in self._users:
            ground_m = math.hypot(user.x_m, user.y_m)
            dz_m = self._config.radio.h_ut_m - self._config.radio.h_bs_m
            links.append(
                UserLink(
                    bearing_deg=normalize_azimuth(math.degrees(math.atan2(user.y_m, user.x_m))),
                    elevation_deg=math.degrees(math.atan2(dz_m, ground_m)),
                    range_m=math.hypot(ground_m, dz_m),
                )
            )
        return tuple(links)

    def _observation(self) -> npt.NDArray[np.float64]:
        """Build the scaled observation vector for the current state."""
        cells = self._footprint_cells(self._flock)
        return self.observation_from(
            self._live_state(cells),
            beam_index=self._beam_index,
            power_index=self._power_index,
            power_mw=self._power_mw,
            weights=self._weights,
            prev_throughput_mbps=self._prev_throughput_mbps,
            log10_w=self._log10_w,
        )

    def observation_from(
        self,
        state: LiveState,
        *,
        beam_index: int,
        power_index: int,
        power_mw: float,
        weights: npt.NDArray[np.complex128],
        prev_throughput_mbps: float,
        log10_w: float,
    ) -> npt.NDArray[np.float64]:
        """Assemble the scaled observation from an externally supplied step.

        The single feature-construction code path: training reaches it through
        :meth:`_observation` with the env's own episode state, and a policy
        served on the demo scene reaches it with the runner's :class:`LiveState`
        plus the last action it applied. Same scales, same ordering, same clip,
        so the checkpoint always scores the 44 numbers it was trained on.
        """
        config = self._config
        densities = cell_densities_w_m2(weights, state.cells, self._setup)
        p95 = s_p95(densities) if densities.size else 0.0

        span = max(1e-9, config.reward.log10_w_max - config.reward.log10_w_min)
        weight_scaled = 2.0 * (log10_w - config.reward.log10_w_min) / span - 1.0

        features: list[float] = [
            beam_index / max(1, self._n_beams - 1),
            power_index / max(1, self._n_levels - 1),
            prev_throughput_mbps / self._max_throughput_ref_mbps,
            p95 / config.exposure.s_thr_w_m2,
            weight_scaled,
            state.flock_bearing_deg / _BEARING_SCALE_DEG,
            state.flock_range_m / _RANGE_SCALE_M,
            state.flock_altitude_mean_m / _ALTITUDE_SCALE_M,
            state.flock_extent_m / _EXTENT_SCALE_M,
            state.flock_heading_deg / _HEADING_SCALE_DEG,
            p95 / config.exposure.s_thr_w_m2,
            state.flock_speed_ms / max(1.0, config.episode.flock_speed_ms * 2.0),
        ]

        budgets = link_budgets(
            weights,
            state,
            replace(self._setup, tx_power_mw=power_mw),
        )
        ordered = sorted(budgets, key=lambda budget: budget.throughput_mbps, reverse=True)
        for index in range(self._obs_users):
            if index < len(ordered):
                budget = ordered[index]
                features.extend(
                    [
                        (budget.sinr_db if math.isfinite(budget.sinr_db) else 0.0) / _SINR_SCALE_DB,
                        budget.throughput_mbps / self._max_throughput_ref_mbps,
                        budget.bearing_deg / _BEARING_SCALE_DEG,
                        budget.range_m / _RANGE_SCALE_M,
                    ]
                )
            else:
                features.extend([0.0] * USER_FEATURES)

        return np.clip(np.asarray(features, dtype=np.float64), -1.0, 1.0)

    def _live_state(self, cells: tuple[CellGeometry, ...]) -> LiveState:
        """Assemble a :class:`LiveState` for the current geometry."""
        return LiveState(
            step=self._step,
            time_s=self._step * self._config.dt_seconds,
            flock_bearing_deg=self._flock_bearing_deg(),
            flock_elevation_deg=self._flock_elevation_deg(),
            flock_range_m=math.hypot(self._flock.centroid_x_m, self._flock.centroid_y_m),
            users=self._user_links(),
            cells=cells,
            flock_altitude_mean_m=self._flock.altitude_mean_m,
            flock_extent_m=self._flock.extent_m,
            flock_heading_deg=self._flock.centroid_heading_deg,
            flock_speed_ms=self._flock.speed_mps,
            log10_w=self._log10_w,
        )

    def _info(
        self,
        *,
        null_active: bool = False,
        exposure: ExposureMetrics | None = None,
        sum_throughput_mbps: float = 0.0,
        reward_terms: RewardTerms | None = None,
        user_throughput_mbps: list[float] | None = None,
        user_sinr_db: list[float] | None = None,
    ) -> dict[str, Any]:
        """Return the info dict, recomputing what a bare ``reset`` has not measured yet."""
        metrics: ExposureMetrics
        if exposure is None:
            cells = self._footprint_cells(self._flock)
            densities = cell_densities_w_m2(self._weights, cells, self._setup)
            metrics = exposure_metrics_from_density(
                densities, self._config.exposure.s_thr_w_m2, dt_seconds=self._config.dt_seconds
            )
        else:
            metrics = exposure
        terms: RewardTerms
        if reward_terms is None:
            result = self._reward_fn(
                sum_throughput_mbps=sum_throughput_mbps,
                tx_power_mw=self._power_mw,
                s_p95_w_per_m2=metrics["s_p95"],
                observation={"log10_w": self._log10_w},
            )
            terms = result["terms"]
        else:
            terms = reward_terms
        return {
            "step": self._step,
            "time_s": self._step * self._config.dt_seconds,
            "log10_w": self._log10_w,
            "weight": 10.0**self._log10_w,
            "beam_index": self._beam_index,
            "power_index": self._power_index,
            "power_mw": self._power_mw,
            "null_active": null_active,
            "sum_throughput_mbps": sum_throughput_mbps,
            "user_throughput_mbps": user_throughput_mbps or [],
            "user_sinr_db": user_sinr_db or [],
            "flock_bearing_deg": self._flock_bearing_deg(),
            "flock_range_m": math.hypot(self._flock.centroid_x_m, self._flock.centroid_y_m),
            "flock_altitude_mean_m": self._flock.altitude_mean_m,
            "flock_heading_deg": self._flock.centroid_heading_deg,
            "exposure": metrics,
            "reward_terms": terms,
        }
