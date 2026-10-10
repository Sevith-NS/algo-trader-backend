"""ASGI middleware: request ids, body size cap, security headers, log redaction."""
import json
import logging
import re
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .errors import INTERNAL_MESSAGE

log = logging.getLogger("flint.api")

_SECURITY_HEADERS = [
    # JSON and SSE only: nothing here should ever render, frame or run script.
    (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"),
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
]


class RequestContext:
    """Assign a request id, cap the body, and stamp security headers."""

    def __init__(self, app: ASGIApp, max_body_bytes: int):
        self.app = app
        self.max_body = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request_id = uuid.uuid4().hex[:12]
        scope.setdefault("state", {})["request_id"] = request_id

        declared = dict(scope.get("headers", [])).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_body:
            return await _send_json(send, 413, request_id, {"error": "Request body is too large.", "code": "too_large"})

        received = 0
        started = False

        async def capped_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body:
                    raise _BodyTooLarge()
            return message

        async def stamped_send(message: Message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                headers = [h for h in message.get("headers", [])]
                existing = {k.lower() for k, _ in headers}
                headers += [(k, v) for k, v in _SECURITY_HEADERS if k not in existing]
                headers.append((b"x-request-id", request_id.encode()))
                if b"cache-control" not in existing:
                    # User-scoped API: nothing is shareable by default.
                    headers.append((b"cache-control", b"private, no-store"))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, capped_receive, stamped_send)
        except _BodyTooLarge:
            if not started:
                await _send_json(send, 413, request_id,
                                 {"error": "Request body is too large.", "code": "too_large"})
        except Exception:
            # Unexpected failure: log the cause against the request id and give
            # the client only that id. Handled here, inside this middleware, so
            # the 500 still carries the security headers.
            log.exception("unhandled error ref=%s path=%s", request_id, scope.get("path"))
            if not started:
                await _send_json(send, 500, request_id, {"error": INTERNAL_MESSAGE, "code": "internal", "ref": request_id})


class _BodyTooLarge(Exception):
    pass


async def _send_json(send: Send, status: int, request_id: str, payload: dict):
    body = json.dumps(payload).encode()
    await send({"type": "http.response.start", "status": status, "headers": [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        (b"x-request-id", request_id.encode()),
        (b"cache-control", b"no-store"),
        *_SECURITY_HEADERS,
    ]})
    await send({"type": "http.response.body", "body": body})


_SECRET_PATTERNS = [
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9\-._~+/]+=*"),
    re.compile(r"(?i)((?:api[_-]?key|secret|token|password)[\"'=:\s]+)[^\s\"'&,]+"),
    re.compile(r"(postgres(?:ql)?://[^:/\s]+:)[^@\s]+(@)"),
    re.compile(r"\b(sk_(?:live|test)_)[A-Za-z0-9]+"),
]


class RedactSecrets(logging.Filter):
    """Scrub tokens, keys and connection-string passwords from log records."""

    @staticmethod
    def _scrub(value):
        if not isinstance(value, str):
            return value
        for pattern in _SECRET_PATTERNS:
            value = pattern.sub(
                lambda m: m.group(1) + "[redacted]" + (m.group(2) if m.lastindex and m.lastindex >= 2 else ""),
                value)
        return value

    def filter(self, record: logging.LogRecord) -> bool:
        # Scrub in place and keep the shape: formatters such as uvicorn's
        # access logger unpack record.args positionally.
        record.msg = self._scrub(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(self._scrub(a) for a in record.args)
        elif isinstance(record.args, dict):
            record.args = {k: self._scrub(v) for k, v in record.args.items()}
        return True
