"""Helpers shared by route modules."""
import json
import logging
from typing import Iterable, Iterator

from fastapi import Response
from fastapi.responses import StreamingResponse

from ..errors import INTERNAL_MESSAGE, Unavailable

log = logging.getLogger("flint.api")


def cache(response: Response, seconds: int) -> None:
    """Let the user's own browser reuse a GET briefly. Never shared caches:
    every response here belongs to an authenticated user."""
    response.headers["Cache-Control"] = f"private, max-age={seconds}"


def unwrap(result):
    """Engines report failure as `{"error": "<safe message>"}`. Turn that into
    a real 503 so a failure is never mistaken for, or cached as, data."""
    if isinstance(result, dict) and isinstance(result.get("error"), str):
        raise Unavailable(result["error"])
    return result


def sse(events: Iterable[dict], first: dict | None = None) -> StreamingResponse:
    """Server-Sent Events from a generator of dicts. An exception mid-stream
    becomes an error frame with a fixed message; the cause goes to the log."""
    def frames() -> Iterator[str]:
        if first is not None:
            yield f"data: {json.dumps(first, default=str)}\n\n"
        try:
            for event in events:
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as exc:
            safe = getattr(exc, "user_message", None)
            if safe is None:
                log.exception("stream failed")
            yield f"data: {json.dumps({'type': 'error', 'message': safe or INTERNAL_MESSAGE})}\n\n"

    return StreamingResponse(frames(), media_type="text/event-stream", headers={
        "Cache-Control": "no-store",
        "X-Accel-Buffering": "no",  # proxies must not buffer a live stream
    })
