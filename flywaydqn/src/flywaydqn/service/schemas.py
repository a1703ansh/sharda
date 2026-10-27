"""Wire schemas for the FastAPI demo service.

Pydantic v2 models mirroring ``ui/src/types/index.ts`` field for field:
``StateFrame``, ``ScenarioRequest``, ``SessionCreated`` and their nested types.
The two files are the contract for the WebSocket stream and ``POST /sessions``
and must stay in sync.

JSON spelling: Python attributes are snake_case (PEP 8); the wire spelling is
the camelCase one the UI reads, produced by ``to_camel``. ``populate_by_name``
accepts either spelling on input, and ``extra="forbid"`` turns a contract
drift into a 422 instead of a silently dropped field.

Units
-----
- positions, ranges, extents: meters (m)
- angles: degrees (deg)
- time: seconds (s)
- transmit power: milliwatts (mW)
- throughput: megabits per second (Mbit/s)
- power density: watts per square meter (W/m^2)
- dose: watt-seconds per square meter (W*s/m^2)
"""

from __future__ import annotations

from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

# Version of the wire contract as a whole. Frames carry no per-message version;
# this is what ``GET /healthz`` reports and what contract tests pin against.
#
# 2: frames carry the scenario's own exposure threshold, the measured null depth
#    and the measured policy-decision latency, so a consumer never has to guess
#    or hardcode any of them; sessions may select a policy by name.
# 3: frames carry the same-step unmitigated reference (``baseline``), so a
#    consumer can show a before/after without running or aligning a second
#    session; a session's stream pacing can be retuned while it runs.
SCHEMA_VERSION: Final[str] = "3"


class WireModel(BaseModel):
    """Base for every wire model: camelCase on the wire, both spellings on input."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class MapPoint(WireModel):
    """A point in sector-local coordinates, in meters."""

    x: float
    y: float


class BeamState(WireModel):
    """Serving beam for one step.

    Angles follow the array convention in ``flywaydqn.rf.array``: azimuth from
    panel boresight, zenith from the horizon.
    """

    index: int
    azimuth_deg: float
    zenith_deg: float
    gain_dbi: float


class FlockState(WireModel):
    """Flock footprint for one step (spec Section 3)."""

    centroid: MapPoint
    heading_deg: float
    speed_ms: float
    altitude_min_m: float
    altitude_max_m: float
    extent_m: float
    n_cels: int


class UserState(WireModel):
    """One scheduled UE link for one step. ``sinr`` is in dB.

    ``sinr`` is floored at ``-30.0`` dB when the link is not served at all -- a
    shut-down sector or an annihilated beam, where the linear SINR is exactly 0
    -- so the value stays finite on the wire. It is never raised: an unserved
    link reports the floor, not a plausible rate.
    """

    position: MapPoint
    bearing_deg: float
    range_m: float
    sinr: float
    throughput_mbps: float


class BaselineMetrics(WireModel):
    """One step scored without mitigation, from the same live geometry.

    This is what the scenario's own no-mitigation reference controller
    (``unconstrained``: greedy serving beam, full transmit power, no null) yields
    at this exact step: the same flock, the same footprint cells, the same
    users, the same threshold. It is measured, not extrapolated, so a consumer
    can print a before/after pair -- how much exposure the selected policy
    actually removed and what it cost in served throughput -- for every policy
    and every scenario without running a second session or aligning two streams.
    """

    policy_name: str
    power_mw: float
    sum_throughput_mbps: float
    p95_density_wm2: float
    max_density_wm2: float
    frac_over: float
    dose_wsm2: float


class StepMetrics(WireModel):
    """Per-step exposure metrics and reward terms (spec Sections 4.2, 4.4).

    ``s_thr_wm2`` is the scenario's own threshold, i.e. the denominator of the
    exposure penalty. It is on the wire so a consumer can draw the threshold it
    is actually being scored against instead of assuming one.
    """

    sum_throughput_mbps: float
    p95_density_wm2: float
    max_density_wm2: float
    frac_over: float
    dose_wsm2: float
    s_thr_wm2: float
    exposure_penalty: float
    throughput_term: float
    power_term: float
    reward: float


class StateFrame(WireModel):
    """One simulation step: the unit of the WebSocket stream.

    ``dt_s`` is the intended wall-clock spacing between consecutive frames, so
    a consumer must not assume a rate the stream has not measured.

    ``null_depth_db`` is measured, not asserted: it is the reduction in
    footprint p95 power density the transmitted weights achieved against the
    same scenario's un-nulled beam at the same step
    (``10 * log10(p95_un-nulled / p95_transmitted)``). It is exactly ``0.0``
    whenever no null was transmitted.

    ``inference_latency_ms`` is the server-measured wall-clock time of this
    step's policy decision.

    ``policy_name`` names the controller that produced this step, so a consumer
    can display what actually ran rather than what it asked for.

    ``power_mw`` is the power the policy actually transmitted this step, so a
    back-off or a shut-down shows up on the wire instead of only in the policy.
    ``baseline`` is the same step under the scenario's no-mitigation reference.
    """

    step: int
    time_s: float
    dt_s: float
    sector: MapPoint
    beam: BeamState
    power_mw: float
    null_active: bool
    null_depth_db: float
    inference_latency_ms: float
    policy_name: str
    flock: FlockState
    users: list[UserState]
    metrics: StepMetrics
    baseline: BaselineMetrics
    explanation: str | None = None
    weight: float


class ScenarioRequest(WireModel):
    """Body of ``POST /sessions``.

    ``weight`` is the trade-off ``w = beta / alpha`` the UI slider exposes,
    bounded to the 0 to 5.0 range documented in ``ui/src/types/index.ts``.

    ``policy`` selects the controller by name (see ``GET /policies``). ``None``
    keeps the scenario file's own policy, which is what a judge sees by default.
    """

    key: str = Field(min_length=1)
    seed: int = Field(ge=0)
    weight: float = Field(ge=0.0, le=5.0)
    policy: str | None = None


class SessionCreated(WireModel):
    """Response of ``POST /sessions``: where to open the stream."""

    id: str
    frame: StateFrame
    stream_path: str


class PaceRequest(WireModel):
    """Body of ``POST /sessions/{id}/pace``: wall-clock pacing for a stream.

    Pacing is presentation only. It scales how fast the service pushes frames
    over the socket; ``dtS`` stays the simulated timestep, so the physics, the
    exposure dose and the reward of a frame are identical at every pace. It
    exists so a live demo can reach a transient -- a flock crossing the sector,
    which is a few seconds of simulated time -- without the presenter waiting
    for it in real time.
    """

    pace: float = Field(gt=0.0, le=50.0)


class ScenarioInfo(WireModel):
    """One entry of ``GET /scenarios``."""

    key: str
    label: str
    description: str


class PolicyInfo(WireModel):
    """One entry of ``GET /policies``: a selectable controller by name."""

    name: str
    label: str
    description: str


class TrainingMetaInfo(WireModel):
    """What the checkpoint behind the ``dqn`` controller records about its run.

    Read straight from the checkpoint's ``meta.json`` by ``GET
    /policies/dqn/meta``: run identity, the recorded spaces, the hyperparameters
    and the metrics the trainer logged. Nothing here is derived or guessed; if
    the checkpoint is absent the route says so instead of inventing a run.
    """

    run_name: str
    seed: int
    total_steps: int
    observation_size: int
    n_actions: int
    hyperparameters: dict[str, Any]
    train_metrics: dict[str, Any]


class HealthResponse(WireModel):
    """Payload of ``GET /healthz``."""

    status: str
    schema_version: str


__all__ = [
    "SCHEMA_VERSION",
    "BaselineMetrics",
    "BeamState",
    "FlockState",
    "HealthResponse",
    "MapPoint",
    "PaceRequest",
    "PolicyInfo",
    "ScenarioInfo",
    "ScenarioRequest",
    "SessionCreated",
    "StateFrame",
    "StepMetrics",
    "TrainingMetaInfo",
    "UserState",
    "WireModel",
]
