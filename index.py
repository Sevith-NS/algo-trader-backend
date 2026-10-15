"""Vercel entrypoint. Vercel looks for an `app` in a root index.py; the API lives in flint_api."""
from flint_api.main import app  # noqa: F401
