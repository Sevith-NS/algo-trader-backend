"""
Per-client rate limiting — token bucket, no external dependencies.

System-design notes:
  - Algorithm: token bucket per (client, scope). Buckets refill continuously,
    so short bursts are allowed up to `capacity` while sustained throughput is
    bounded by `refill_per_sec`. This matches a UI that fires 5 parallel calls
    on page load but should not be able to hammer the upstream all day.
  - Scopes: routes are classed by cost. `standard` covers cached market-data
    lookups; `expensive` covers fan-out endpoints (deep analysis, batch
    holdings intel, LLM calls) where one request triggers many upstream calls.
  - Responses follow RFC 6585: HTTP 429 with Retry-After, plus the de-facto
    X-RateLimit-Limit / -Remaining / -Reset headers so clients can back off
    intelligently.
  - State is in-process (this is a single-process dev/paper-trading server).
    Behind a multi-worker deployment this would move to Redis; the interface
    stays the same.
  - Stale buckets are pruned opportunistically to bound memory.
"""
import functools
import os
import threading
import time

from flask import jsonify, request

# Only trust X-Forwarded-For behind a known proxy — the header is client-
# supplied, so honoring it unconditionally lets any direct caller mint a
# fresh bucket per request (full limiter bypass) or drain a victim's bucket.
_TRUST_PROXY = os.environ.get("TRUST_PROXY") == "1"

# scope -> (capacity, refill tokens/sec)
SCOPES = {
    # 120 burst, 60/min sustained: screener page load fires ~6 calls at once
    "standard": (120.0, 1.0),
    # 12 burst, 6/min sustained: each of these fans out to many upstream calls
    "expensive": (12.0, 0.1),
}

_lock = threading.Lock()
_buckets: dict = {}  # (client, scope) -> {"tokens": float, "ts": float}
_last_prune = {"ts": 0.0}
_PRUNE_EVERY = 300  # seconds
_STALE_AFTER = 900  # bucket untouched this long -> drop


def _client_id() -> str:
    if _TRUST_PROXY:
        fwd = request.headers.get("X-Forwarded-For")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _prune(now: float):
    if now - _last_prune["ts"] < _PRUNE_EVERY:
        return
    _last_prune["ts"] = now
    stale = [k for k, b in _buckets.items() if now - b["ts"] > _STALE_AFTER]
    for k in stale:
        _buckets.pop(k, None)


def _take(client: str, scope: str) -> tuple[bool, float, float]:
    """Try to take one token. Returns (allowed, remaining, retry_after_sec)."""
    capacity, refill = SCOPES[scope]
    now = time.monotonic()
    with _lock:
        _prune(now)
        bucket = _buckets.get((client, scope))
        if bucket is None:
            bucket = {"tokens": capacity, "ts": now}
            _buckets[(client, scope)] = bucket
        # Continuous refill since last touch
        bucket["tokens"] = min(capacity, bucket["tokens"] + (now - bucket["ts"]) * refill)
        bucket["ts"] = now
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True, bucket["tokens"], 0.0
        return False, 0.0, (1.0 - bucket["tokens"]) / refill


def rate_limit(scope: str = "standard"):
    """Decorator: enforce the scope's token bucket for the calling client."""
    if scope not in SCOPES:
        raise ValueError(f"Unknown rate-limit scope: {scope}")

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            allowed, remaining, retry_after = _take(_client_id(), scope)
            capacity, refill = SCOPES[scope]
            if not allowed:
                resp = jsonify({
                    "error": "Rate limit exceeded — slow down.",
                    "retry_after_seconds": round(retry_after, 1),
                })
                resp.status_code = 429
                resp.headers["Retry-After"] = str(max(1, int(retry_after + 0.999)))
                resp.headers["X-RateLimit-Limit"] = str(int(capacity))
                resp.headers["X-RateLimit-Remaining"] = "0"
                resp.headers["X-RateLimit-Reset"] = str(int(time.time() + retry_after))
                return resp
            resp = fn(*args, **kwargs)
            # Attach quota headers to successful responses too
            try:
                resp = jsonify(resp) if isinstance(resp, (dict, list)) else resp
                if hasattr(resp, "headers"):
                    resp.headers["X-RateLimit-Limit"] = str(int(capacity))
                    resp.headers["X-RateLimit-Remaining"] = str(int(remaining))
            except Exception:
                pass
            return resp
        return wrapper
    return decorator
