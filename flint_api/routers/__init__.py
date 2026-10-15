"""Route modules. Each exposes `router`; `register` mounts them all under /api."""
from fastapi import FastAPI

_MODULES: list[str] = ["market", "symbol", "news", "portfolio", "backtest", "ai",
                     "me", "ledger", "watchlists", "journal", "brokers", "alerts"]


def register(app: FastAPI) -> None:
    import importlib
    for name in _MODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        app.include_router(module.router, prefix="/api")
