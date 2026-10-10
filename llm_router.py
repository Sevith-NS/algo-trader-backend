"""
Fallback LLM transport via OmniRoute, engaged only when Gemini is out of quota.

Flint talks to Gemini directly for everything: it is the cheapest path, the
prompts are tuned for it, and it is the only provider the rest of the backend
knows about. The one thing it cannot do is survive its own rate limit — a
trading-desk run is ~14 model calls, which will exhaust a free-tier per-minute
quota mid-debate and leave half the seats empty.

OmniRoute (https://github.com/diegosouzapw/OmniRoute) is an OpenAI-compatible
local gateway that fans out across many providers. This module wires it in as a
STRICT FALLBACK, on purpose:

  - Direct Gemini is always tried first. Nothing routes through OmniRoute while
    Gemini is answering.
  - A 429 / RESOURCE_EXHAUSTED benches Gemini for a cooldown window (honouring
    the retryDelay Gemini itself returns), and only then does traffic move over.
  - When the cooldown lapses the next call goes back to Gemini, so the fallback
    never quietly becomes the default.
  - If OmniRoute is not installed or not running, every function here degrades
    to "unavailable" and the caller behaves exactly as it did before.

Availability is probed lazily and cached, so an absent gateway costs one failed
connection per minute rather than one per model call.
"""
import json
import os
import re
import threading
import time

import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Local gateway. OmniRoute serves an OpenAI-compatible surface on :20128.
BASE_URL = os.environ.get("OMNIROUTE_BASE_URL", "http://127.0.0.1:20128/v1").rstrip("/")
API_KEY = os.environ.get("OMNIROUTE_API_KEY", "")
# "auto" is OmniRoute's own scoring combo — it picks a live provider per call.
MODEL = os.environ.get("OMNIROUTE_MODEL", "auto")
# Hard off-switch. Absent or "1" leaves the fallback armed but dormant.
ENABLED = os.environ.get("OMNIROUTE_ENABLED", "1") != "0"

REQUEST_TIMEOUT = float(os.environ.get("OMNIROUTE_TIMEOUT", "180"))

# How long Gemini stays benched after a quota error when it does not say.
_DEFAULT_COOLDOWN = 90.0
_MAX_COOLDOWN = 3600.0

# Availability probe cache: an absent gateway must not cost a TCP connect per call.
_PROBE_TTL_OK = 300.0
_PROBE_TTL_FAIL = 60.0

_lock = threading.Lock()
_probe = {"ts": 0.0, "ok": False}
# Monotonic counters. run_desk snapshots these either side of a run to report
# which provider actually answered, without passing a collector through every
# agent function (the analysts and risk seats run on pool threads).
_stats = {"fallback_calls": 0, "fallback_failures": 0, "bench_events": 0}
# Per-scope bench state, so separate callers can hold separate views of the
# same upstream if they ever run on different keys.
_benched: dict = {}

_RETRY_DELAY_RE = re.compile(r"['\"]?retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s")


class RouterUnavailable(RuntimeError):
    """OmniRoute is disabled, not installed, or not answering."""


# ------------------------------------------------------------- quota state ---

def parse_retry_delay(exc) -> float | None:
    """Seconds Gemini asked us to wait, from its own 429 payload.

    Gemini returns {"retryDelay": "52s"} inside the error details; honouring it
    beats guessing, and it is the difference between a 60s per-minute window and
    a multi-hour daily exhaustion.
    """
    match = _RETRY_DELAY_RE.search(str(exc))
    if not match:
        return None
    try:
        return max(0.0, min(_MAX_COOLDOWN, float(match.group(1))))
    except ValueError:
        return None


def bench_primary(scope: str, exc=None, seconds: float | None = None) -> float:
    """Mark the primary provider out of quota. Returns the epoch it comes back."""
    wait = seconds if seconds is not None else parse_retry_delay(exc)
    if wait is None:
        wait = _DEFAULT_COOLDOWN
    until = time.time() + max(1.0, min(_MAX_COOLDOWN, wait))
    with _lock:
        # Never shorten an existing bench: a later per-minute 429 must not
        # release a provider that a daily exhaustion parked for an hour.
        _benched[scope] = max(_benched.get(scope, 0.0), until)
        _stats["bench_events"] += 1
        return _benched[scope]


def primary_benched(scope: str) -> bool:
    with _lock:
        until = _benched.get(scope, 0.0)
    return time.time() < until


def bench_remaining(scope: str) -> float:
    with _lock:
        until = _benched.get(scope, 0.0)
    return max(0.0, until - time.time())


def clear_bench(scope: str):
    with _lock:
        _benched.pop(scope, None)


def counters() -> dict:
    with _lock:
        return dict(_stats)


# ------------------------------------------------------------ availability ---

def _headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def is_available(force: bool = False) -> bool:
    """Is the local gateway up? Cached — a missing gateway is the common case."""
    if not ENABLED:
        return False
    now = time.time()
    with _lock:
        age = now - _probe["ts"]
        ttl = _PROBE_TTL_OK if _probe["ok"] else _PROBE_TTL_FAIL
        if not force and _probe["ts"] and age < ttl:
            return _probe["ok"]

    ok = False
    try:
        resp = requests.get(f"{BASE_URL}/models", headers=_headers(), timeout=4)
        # Any answered request proves the gateway is listening; a 401 means it
        # is up and wants a key, which is a configuration problem, not absence.
        ok = resp.status_code < 500
    except requests.RequestException:
        ok = False

    with _lock:
        _probe["ts"] = time.time()
        _probe["ok"] = ok
    return ok


# ------------------------------------------------------------------- call ---

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strip_fence(text: str) -> str:
    """Unwrap a ```json fence.

    OmniRoute fans out across many models and not all of them honour
    response_format, so the fence Gemini's JSON mode never produces does show up
    here. Cheaper to strip than to lose the whole seat.
    """
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text


def _decode(resp) -> dict:
    """Read one completion, whether the gateway honoured stream:false or not.

    Belt and braces: `stream: False` is set on every request, but OmniRoute
    fans out across many upstreams and a build that streams anyway would
    otherwise fail JSON parsing on the first `data:` frame. Reassembling the
    deltas costs a few lines and removes a whole class of silent failure.
    """
    text = resp.text.lstrip()
    if not text.startswith("data:"):
        return resp.json()

    content, payload = [], {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue  # ": x-omniroute-*" comment lines carry telemetry, not content
        chunk = line[5:].strip()
        if not chunk or chunk == "[DONE]":
            continue
        try:
            payload = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        for choice in payload.get("choices") or []:
            piece = (choice.get("delta") or {}).get("content")
            if isinstance(piece, str):
                content.append(piece)
    return {"choices": [{"message": {"content": "".join(content)}}]}


# The gateway's own upstreams have quotas too, and OmniRoute rotates providers
# between attempts — so a 429 from the gateway is usually worth another try,
# unlike a 429 from a single-provider API.
_CALL_ATTEMPTS = 3
_CALL_BACKOFF = 3.0
_GATEWAY_RETRY_CODES = (408, 429, 500, 502, 503, 504)


class FallbackFailed(RuntimeError):
    """The gateway was reachable but could not produce an answer.

    Distinct from RouterUnavailable (gateway absent) so callers can tell
    "no fallback installed" apart from "the fallback itself failed", and never
    misattribute the latter to the primary provider.
    """


def call_json(prompt: str, model: str | None = None, timeout: float | None = None) -> dict:
    """One JSON-mode turn through OmniRoute's OpenAI-compatible endpoint.

    Raises RouterUnavailable when the gateway is off or unreachable, so callers
    can fall back to their existing behaviour. Raises FallbackFailed when the
    gateway answered but could not deliver, after exhausting its own retries.
    """
    if not ENABLED:
        raise RouterUnavailable("OmniRoute fallback is disabled (OMNIROUTE_ENABLED=0).")
    if not is_available():
        raise RouterUnavailable(
            f"OmniRoute is not answering at {BASE_URL}. Start it with `omniroute start`.")

    payload = {
        "model": model or MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        # OmniRoute streams by default even when the client did not ask, which
        # yields SSE frames where an OpenAI client expects one JSON object.
        # The desk wants a whole answer per seat, so opt out explicitly.
        "stream": False,
    }
    delay = _CALL_BACKOFF
    last = "unknown error"
    for attempt in range(_CALL_ATTEMPTS):
        try:
            resp = requests.post(
                f"{BASE_URL}/chat/completions", json=payload,
                headers=_headers(), timeout=timeout or REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            # A gateway that just went away should be re-probed, not trusted.
            with _lock:
                _probe["ts"] = 0.0
                _stats["fallback_failures"] += 1
            raise RouterUnavailable(f"OmniRoute request failed: {exc}") from exc

        if resp.status_code >= 400:
            last = f"HTTP {resp.status_code}: {resp.text[:160]}"
            if resp.status_code in _GATEWAY_RETRY_CODES and attempt < _CALL_ATTEMPTS - 1:
                time.sleep(delay)
                delay *= 2
                continue
            break

        try:
            body = _decode(resp)
            choices = body.get("choices") or []
            if not choices:
                raise ValueError("no choices in the response")
            content = (choices[0].get("message") or {}).get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty message content")
            parsed = json.loads(_strip_fence(content))
            if not isinstance(parsed, dict):
                raise ValueError("non-object JSON payload")
        except (ValueError, json.JSONDecodeError) as exc:
            # A different upstream on the next attempt may well format correctly.
            last = f"unusable response: {exc}"
            if attempt < _CALL_ATTEMPTS - 1:
                time.sleep(delay)
                delay *= 2
                continue
            break

        with _lock:
            _stats["fallback_calls"] += 1
        return parsed

    with _lock:
        _stats["fallback_failures"] += 1
    raise FallbackFailed(f"OmniRoute could not answer ({last})")


def status() -> dict:
    """Diagnostic snapshot for the /api/agents/routing route."""
    return {
        "enabled": ENABLED,
        "base_url": BASE_URL,
        "model": MODEL,
        "has_key": bool(API_KEY),
        "available": is_available(),
        "benched": {scope: round(bench_remaining(scope), 1) for scope in list(_benched)},
        "counters": counters(),
    }
