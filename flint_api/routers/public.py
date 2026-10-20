"""The one unauthenticated data route: headline market figures for the public
landing page, fetched by the Next.js server (not the browser) once a minute.

Abuse can't amplify upstream calls: the snapshot is built at most once per
TTL per process, and every caller gets that same copy. Only index levels and
one FX rate leave this route, each with its source and time; nothing here is
per-user, and no stock-level data is published.
"""
import threading
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Response

import data_source
import pulse_engine

router = APIRouter(prefix="/public", tags=["public"])

TTL = 60
_lock = threading.Lock()
_snapshot: tuple[float, dict] | None = None

_INDEX_KEYS = ("symbol", "name", "price", "change", "change_pct", "as_of")


def _market(market: str) -> dict | None:
    p = pulse_engine.get_pulse(market)
    if not isinstance(p, dict) or p.get("error"):
        return None
    breadth = p.get("breadth") or None
    return {"label": p.get("label"), "currency": p.get("currency"),
            "session": (p.get("session") or {}).get("state"),
            "indices": [{k: i.get(k) for k in _INDEX_KEYS} for i in p.get("indices") or []
                        if isinstance(i.get("price"), (int, float))],
            "breadth": {k: breadth.get(k) for k in ("universe", "advancing", "declining", "unchanged", "partial")}
                       if breadth else None,
            "source": p.get("source")}


def _usdinr() -> dict | None:
    q = data_source.get_quote("USDINR=X")
    rate = (q or {}).get("regularMarketPrice")
    if not isinstance(rate, (int, float)) or rate <= 0 or q.get("error"):
        return None
    ts, prev = q.get("regularMarketTime"), q.get("regularMarketPreviousClose") or q.get("previousClose")
    change_pct = round((rate - prev) / prev * 100, 2) if isinstance(prev, (int, float)) and prev > 0 else None
    return {"pair": "USD/INR", "rate": rate, "change_pct": change_pct, "source": "Yahoo Finance (delayed)",
            "as_of": datetime.fromtimestamp(ts, timezone.utc).isoformat() if isinstance(ts, (int, float)) else None}


def _build() -> dict:
    def safe(fn, *args):
        try:
            return fn(*args)
        except Exception:
            return None
    return {"india": safe(_market, "in"), "us": safe(_market, "us"), "fx": safe(_usdinr),
            "generated_at": datetime.now(timezone.utc).isoformat()}


@router.get("/landing")
def landing(response: Response):
    global _snapshot
    with _lock:
        if _snapshot is None or time.monotonic() - _snapshot[0] > TTL:
            _snapshot = (time.monotonic(), _build())
        data = _snapshot[1]
    response.headers["Cache-Control"] = f"public, max-age={TTL}"
    return data
