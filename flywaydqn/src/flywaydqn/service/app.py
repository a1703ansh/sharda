"""FastAPI demo service: scenarios, sessions, and the frame WebSocket stream.

The service is deliberately thin. It owns session lifecycle and transport; the
physics and policy plug in behind :data:`RunnerFactory`, so training and
evaluation code never lands in a route handler.

Endpoints
---------
``GET /healthz``
    Liveness plus the wire schema version the UI contract is pinned against.
``GET /scenarios``
    The demo scenario keys ``POST /sessions`` accepts.
``GET /policies``
    The controller names ``POST /sessions`` accepts, so a client populates its
    engine selector from the server instead of hardcoding the list.
``POST /sessions``
    Create a session and return its id, initial frame, and stream path.
``POST /sessions/{id}/pace``
    Retune how fast that session's frames are pushed, for a live demo. Pacing
    never changes the simulated timestep, so the physics are untouched.
``WS /sessions/{id}/stream``
    Push ``StateFrame`` JSON at the frame's own ``dt_s`` cadence, scaled by the
    session's pace.

Run from the repo root::

    uvicorn flywaydqn.service.app:app --factory --host 127.0.0.1 --port 8000

or ``flywaydqn-serve`` (``make serve``).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Final, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from flywaydqn.agents.policies import (
    UnknownPolicyError,
    available_policies,
    available_policy_names,
)
from flywaydqn.demo import demo_runner_factory, list_demo_scenarios
from flywaydqn.service.schemas import (
    SCHEMA_VERSION,
    HealthResponse,
    PaceRequest,
    PolicyInfo,
    ScenarioInfo,
    ScenarioRequest,
    SessionCreated,
    StateFrame,
    TrainingMetaInfo,
)
from flywaydqn.service.ws import ConnectionManager

logger = logging.getLogger(__name__)

# Local Vite dev server, the only origin the dashboard is served from.
_UI_ORIGINS: Final[list[str]] = ["http://localhost:5173"]


class FrameRunner(Protocol):
    """Produces the frames of one session, one step at a time.

    The service never inspects how a frame is computed, so a trained policy, a
    replay, or the deterministic demo below can all sit behind it.
    """

    def initial_frame(self) -> StateFrame:
        """Frame for ``step == 0``, the one returned by ``POST /sessions``."""
        ...

    def step(self) -> StateFrame:
        """Advance one step and return the resulting frame."""
        ...


RunnerFactory = Callable[[ScenarioRequest], FrameRunner]
"""Builds the frame producer for a session from its scenario request."""


@dataclass
class _Session:
    """One created session: its request, its runner, and its id."""

    id: str
    request: ScenarioRequest
    runner: FrameRunner
    # Wall-clock pacing multiplier: 1.0 pushes one frame every ``dt_s`` seconds.
    # Presentation only -- each frame's own ``dt_s`` stays the simulated step.
    pace: float = 1.0


def _register_trained_dqn() -> None:
    """Expose the trained checkpoint as the ``dqn`` controller (WP-E).

    Thin delegate: the registration itself lives beside the policy that serves
    it (:func:`flywaydqn.agents.dqn_policy.register_trained_controller`), so
    the controllers this service offers and the controllers
    ``experiments/run_baselines.py`` measures are one list, not two. Heavy
    imports stay lazy, so nothing torch-shaped loads until a session selects
    the controller.
    """
    from flywaydqn.agents.dqn_policy import register_trained_controller

    register_trained_controller()


def create_app(runner_factory: RunnerFactory | None = None) -> FastAPI:
    """Build the demo service.

    Parameters
    ----------
    runner_factory :
        Builds the frame producer for a new session from its request. Defaults
        to :func:`flywaydqn.demo.demo_runner_factory`, which loads a canned
        scenario from ``configs/demo`` and runs its rule-based baseline, so
        the service needs no checkpoint on disk. A trained policy plugs in
        here, not in the route handlers.

    Returns
    -------
    FastAPI
        App with CORS for the local UI dev server and one stream loop per
        active session.
    """
    factory: RunnerFactory = runner_factory if runner_factory is not None else demo_runner_factory
    manager = ConnectionManager()
    sessions: dict[str, _Session] = {}
    producers: dict[str, asyncio.Task[None]] = {}

    # Service startup: offer the trained checkpoint as the ``dqn`` controller.
    # Idempotent (the registry is a dict), so rebuilding the app is safe.
    _register_trained_dqn()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        tasks = list(producers.values())
        producers.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stream_loop(session: _Session) -> None:
        """Step ``session``'s runner and broadcast until its subscribers leave."""
        try:
            while True:
                started = time.perf_counter()
                frame = session.runner.step()
                await manager.broadcast(session.id, frame)
                # The pace scales the wall-clock period only. The frame keeps
                # its own ``dt_s``, so a demo runs the same physics faster
                # without changing what any frame reports. Subtracting the
                # step's own compute time makes pace 1 a true real-time 10 fps;
                # without it every frame would land ``compute + dt/pace`` apart.
                target_period = frame.dt_s / max(session.pace, 1e-3)
                elapsed = time.perf_counter() - started
                await asyncio.sleep(max(0.0, target_period - elapsed))
        except Exception:
            logger.exception("stream loop for session %s stopped", session.id)

    def ensure_producer(session: _Session) -> None:
        """Start the stream loop once per session, whoever connects first."""
        task = producers.get(session.id)
        if task is None or task.done():
            producers[session.id] = asyncio.create_task(stream_loop(session))

    app = FastAPI(title="FlywayDQN demo service", version=SCHEMA_VERSION, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_UI_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        """Liveness probe plus the wire schema version the UI is built against."""
        return HealthResponse(status="ok", schema_version=SCHEMA_VERSION)

    @app.get("/scenarios", response_model=list[ScenarioInfo])
    async def list_scenarios() -> list[ScenarioInfo]:
        """List the scenario keys accepted by ``POST /sessions``."""
        return list(list_demo_scenarios())

    @app.get("/policies", response_model=list[PolicyInfo])
    async def list_policies() -> list[PolicyInfo]:
        """List the controller names accepted by ``POST /sessions``."""
        return [
            PolicyInfo(name=spec.name, label=spec.label, description=spec.description)
            for spec in available_policies()
        ]

    @app.get("/policies/dqn/meta", response_model=TrainingMetaInfo)
    async def dqn_training_meta() -> TrainingMetaInfo:
        """Run metadata of the checkpoint behind the ``dqn`` controller (WP-E).

        Straight from the checkpoint's ``meta.json``: run identity, recorded
        spaces, hyperparameters and logged metrics. 404 when no checkpoint is
        present rather than a plausible substitute.
        """
        from flywaydqn.agents.dqn_policy import default_checkpoint_dir
        from flywaydqn.agents.policy_io import CheckpointError, load_meta

        try:
            meta = load_meta(default_checkpoint_dir())
        except CheckpointError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return TrainingMetaInfo(
            run_name=meta.run_name,
            seed=meta.seed,
            total_steps=meta.total_steps,
            observation_size=meta.observation_size,
            n_actions=meta.n_actions,
            hyperparameters=meta.hyperparameters,
            train_metrics=meta.train_metrics,
        )

    @app.post("/sessions", response_model=SessionCreated, status_code=201)
    def create_session(request: ScenarioRequest) -> SessionCreated:
        """Create a session and return its id, initial frame, and stream path."""
        if not any(info.key == request.key for info in list_demo_scenarios()):
            raise HTTPException(status_code=404, detail=f"unknown scenario key: {request.key}")
        if request.policy is not None and request.policy not in available_policy_names():
            # Reject before a runner exists, so a bad name reports the valid set
            # instead of surfacing as a stream that dies on its first step.
            raise HTTPException(
                status_code=404,
                detail=str(UnknownPolicyError(request.policy, available_policy_names())),
            )

        session = _Session(id=uuid.uuid4().hex, request=request, runner=factory(request))
        sessions[session.id] = session
        return SessionCreated(
            id=session.id,
            frame=session.runner.initial_frame(),
            stream_path=f"/sessions/{session.id}/stream",
        )

    @app.post("/sessions/{session_id}/pace", status_code=204)
    async def set_pace(session_id: str, request: PaceRequest) -> Response:
        """Retune a live session's wall-clock pacing (presentation only)."""
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"unknown session id: {session_id}")
        session.pace = request.pace
        return Response(status_code=204)

    @app.websocket("/sessions/{session_id}/stream")
    async def stream_session(session_id: str, websocket: WebSocket) -> None:
        """Stream frames for ``session_id`` until the client disconnects."""
        session = sessions.get(session_id)
        if session is None:
            await websocket.close(code=1008, reason="unknown session")
            return

        await manager.connect(session_id, websocket)
        ensure_producer(session)
        try:
            # The client only consumes: block until it goes away, then retire
            # its subscription (and the stream loop, if it was the last one).
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            manager.disconnect(session_id, websocket)
            if not manager.has_subscribers(session_id):
                task = producers.pop(session_id, None)
                if task is not None:
                    task.cancel()

    return app


def app() -> FastAPI:
    """Zero-arg factory for ``uvicorn flywaydqn.service.app:app --factory``."""
    return create_app()


def main() -> None:
    """Console entry point behind the ``flywaydqn-serve`` script."""
    uvicorn.run("flywaydqn.service.app:app", factory=True, host="127.0.0.1", port=8000)


__all__ = [
    "FrameRunner",
    "RunnerFactory",
    "app",
    "create_app",
    "demo_runner_factory",
    "main",
]
