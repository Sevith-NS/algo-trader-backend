"""Flint API: FastAPI app factory.

The browser never calls this service directly. The Next.js server forwards
requests with the user's Clerk session token, so there is no CORS policy at
all: a browser on another origin gets no permission to read responses.

Run locally:  uvicorn flint_api.main:app --port 5000
"""
import logging

from fastapi import Depends, FastAPI

from . import errors
from .middleware import RedactSecrets, RequestContext
from .security import Principal, current_user
from .settings import settings


def _configure_logging() -> None:
    redact = RedactSecrets()
    for name in ("", "flint.api", "uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        if not any(isinstance(f, RedactSecrets) for f in logger.filters):
            logger.addFilter(redact)
    for handler in logging.getLogger().handlers:
        handler.addFilter(redact)


def create_app() -> FastAPI:
    _configure_logging()
    app = FastAPI(
        title="Flint API",
        version="2.0.0",
        # No public schema or docs UI: the API surface is not advertised.
        openapi_url=None, docs_url=None, redoc_url=None,
    )
    errors.install(app)
    app.add_middleware(RequestContext, max_body_bytes=settings.max_body_bytes)

    @app.get("/api/health", include_in_schema=False)
    def health():
        return {"status": "ok"}

    @app.get("/api/session")
    def session(user: Principal = Depends(current_user)):
        """Echo the verified identity. Used by the frontend to confirm the token path works."""
        return {"user_id": user.user_id}

    from .routers import register
    register(app)
    return app


app = create_app()
