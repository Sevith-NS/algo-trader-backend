"""Per-user token buckets.

Buckets are keyed by the verified Clerk user id, so a caller cannot mint
fresh buckets by spoofing X-Forwarded-For (audit needs-validation #2). The
bucket table is an LRU with a hard cap, so its size never depends on a quiet
period an attacker can withhold.

State is per process. Run one worker, or divide budgets with WEB_CONCURRENCY.
"""
import os
import threading
import time
from collections import OrderedDict

from fastapi import Depends

from .errors import RateLimited
from .security import Principal, current_user

_WORKERS = max(1, int(os.environ.get("WEB_CONCURRENCY", "1")))

# scope -> (burst capacity, refill tokens per second)
_BASE_SCOPES = {
    "standard": (120.0, 1.0),     # cached market data; a page fires ~6 at once
    "expensive": (12.0, 0.1),     # fan-out work and single LLM calls
    "agents": (3.0, 0.017),       # one desk run is ~14 LLM turns
    "backtest": (6.0, 0.033),     # sweeps are hundreds of simulations
    "write": (60.0, 0.5),         # user data mutations
}
SCOPES = {name: (max(1.0, cap / _WORKERS), refill / _WORKERS)
          for name, (cap, refill) in _BASE_SCOPES.items()}

_MAX_BUCKETS = 20_000
_lock = threading.Lock()
_buckets: "OrderedDict[tuple[str, str], list[float]]" = OrderedDict()


def _take(key: str, scope: str) -> tuple[bool, float, float]:
    capacity, refill = SCOPES[scope]
    now = time.monotonic()
    with _lock:
        bucket = _buckets.get((key, scope))
        if bucket is None:
            bucket = [capacity, now]
            _buckets[(key, scope)] = bucket
            if len(_buckets) > _MAX_BUCKETS:
                _buckets.popitem(last=False)
        else:
            _buckets.move_to_end((key, scope))
        tokens = min(capacity, bucket[0] + (now - bucket[1]) * refill)
        bucket[1] = now
        if tokens >= 1.0:
            bucket[0] = tokens - 1.0
            return True, bucket[0], 0.0
        bucket[0] = tokens
        return False, 0.0, (1.0 - tokens) / refill


def limit(scope: str = "standard"):
    """Route dependency: authenticate, then spend one token from `scope`."""
    if scope not in SCOPES:
        raise ValueError(f"Unknown rate-limit scope: {scope}")

    def dependency(user: Principal = Depends(current_user)) -> Principal:
        allowed, _remaining, retry_after = _take(user.user_id, scope)
        if not allowed:
            wait = max(1, int(retry_after + 0.999))
            raise RateLimited(f"Too many requests. Try again in {wait} seconds.",
                              headers={"Retry-After": str(wait)})
        return user

    return dependency


def reset_for_tests() -> None:
    with _lock:
        _buckets.clear()
