"""One error contract for the whole API.

Every failure is JSON shaped `{"error": "<message>", "code": "...", "ref"?, "fields"?}`.
Messages written here are safe to show a user. Unexpected exceptions never
reach the client: they are logged with a reference id, and the client gets
that id plus a fixed message, so support can find the log line without the
response leaking paths, upstream URLs or library internals (audit finding 7).
"""
import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("flint.api")

INTERNAL_MESSAGE = "Something went wrong on our side. Try again in a moment."


class AppError(Exception):
    status = 400
    code = "bad_request"

    def __init__(self, message: str, *, headers: dict | None = None):
        super().__init__(message)
        self.message = message
        self.headers = headers or {}


class Unauthorized(AppError):
    status, code = 401, "unauthorized"


class Forbidden(AppError):
    status, code = 403, "forbidden"


class NotFound(AppError):
    status, code = 404, "not_found"


class Conflict(AppError):
    status, code = 409, "conflict"


class TooLarge(AppError):
    status, code = 413, "too_large"


class RateLimited(AppError):
    status, code = 429, "rate_limited"


class Unavailable(AppError):
    status, code = 503, "unavailable"


def _body(code: str, message: str, ref: str | None = None) -> dict:
    body = {"error": message, "code": code}
    if ref:
        body["ref"] = ref
    return body


def install(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_req: Request, exc: AppError):
        return JSONResponse(_body(exc.code, exc.message), status_code=exc.status, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_req: Request, exc: RequestValidationError):
        # Field paths and reasons only. FastAPI's default echoes the rejected
        # input back, which can reflect whatever the caller sent.
        fields = [
            {"field": ".".join(str(p) for p in e.get("loc", ()) if p not in ("body", "query", "path")),
             "message": e.get("msg", "Invalid value")}
            for e in exc.errors()
        ]
        body = _body("invalid_request", "Some fields are invalid.")
        body["fields"] = fields
        return JSONResponse(body, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_req: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        message = {404: "Not found.", 405: "Method not allowed."}.get(exc.status_code, "Request failed.")
        return JSONResponse(_body(code, message), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(req: Request, exc: Exception):
        ref = getattr(req.state, "request_id", None) or uuid.uuid4().hex[:12]
        log.exception("unhandled error ref=%s path=%s", ref, req.url.path)
        return JSONResponse(_body("internal", INTERNAL_MESSAGE, ref), status_code=500)
