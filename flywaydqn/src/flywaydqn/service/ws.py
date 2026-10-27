"""WebSocket connection manager for session frame streams.

The manager owns one job: fan a serialized :class:`StateFrame` out to every
subscriber of a session and keep the subscriber set accurate as sockets come
and go. It does not know where frames come from; the app's per-session stream
loop drives it.

Instances live for one app. Every mutation happens without an ``await`` in
between, so a single event loop observes each update atomically and no lock is
needed.
"""

from __future__ import annotations

import logging

from fastapi import WebSocket

from flywaydqn.service.schemas import StateFrame

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Tracks the live WebSocket subscribers of each session."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket) -> int:
        """Accept ``websocket`` and register it as a subscriber of ``session_id``.

        Returns
        -------
        int
            Number of subscribers of the session after registration.
        """
        await websocket.accept()
        self._subscribers.setdefault(session_id, set()).add(websocket)
        return len(self._subscribers[session_id])

    def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        """Unregister ``websocket``; drop the session entry once it is empty.

        Calling this for an unknown session or an already-removed socket is a
        no-op, so cleanup paths never need to check first.
        """
        session_subscribers = self._subscribers.get(session_id)
        if session_subscribers is None:
            return
        session_subscribers.discard(websocket)
        if not session_subscribers:
            del self._subscribers[session_id]

    def has_subscribers(self, session_id: str) -> bool:
        """True while at least one socket is subscribed to ``session_id``."""
        return bool(self._subscribers.get(session_id))

    async def broadcast(self, session_id: str, frame: StateFrame) -> int:
        """Send ``frame`` to every subscriber of ``session_id``.

        The frame is serialized once, so all subscribers of a session receive
        byte-identical JSON. A socket whose send raises (the client went away
        without a clean close) is dropped rather than aborting the stream for
        the remaining subscribers.

        Returns
        -------
        int
            Number of subscribers that received the frame.
        """
        session_subscribers = self._subscribers.get(session_id)
        if not session_subscribers:
            return 0

        payload = frame.model_dump_json(by_alias=True, exclude_none=True)
        delivered = 0
        for websocket in list(session_subscribers):
            try:
                await websocket.send_text(payload)
            except Exception:
                logger.debug("dropping unreachable socket for session %s", session_id)
                self.disconnect(session_id, websocket)
            else:
                delivered += 1
        return delivered


__all__ = ["ConnectionManager"]
