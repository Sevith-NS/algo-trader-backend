"""
Resilient market-data layer.

yfinance's internal session/crumb handshake gets flagged behind some corporate
proxies ("Too Many Requests" even when the network is fine), so the primary
source here is Yahoo's public chart/spark JSON API called directly with plain
`requests` — the same endpoints yfinance wraps — used politely:

  - Global throttle: minimum spacing between outbound calls
  - In-memory TTL cache + JSON disk cache (survives restarts)
  - Stale-if-error: when upstream fails, serve last good data instead of erroring
  - Batched spark endpoint (one request for N symbols) for the market overview
  - yfinance kept as a secondary fallback

No auth bypass or scraping tricks — standard public JSON endpoints, browser UA,
low request volume.
"""
import os
import json
import time
import threading
from contextlib import contextmanager

import pandas as pd
import requests
import yfinance as yf

CACHE_FILE = os.path.join(os.path.dirname(__file__), ".market_cache.json")

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_SPARK_URL = "https://query1.finance.yahoo.com/v8/finance/spark"

FRESH_TTL = {
    "history": 30 * 60,   # 30 min
    "quote": 60,          # 1 min
    "overview": 5 * 60,   # 5 min
    "fundamentals": 24 * 3600,  # statements only change on results day
}
STALE_TTL = 7 * 24 * 3600  # serve stale data up to 7 days old rather than nothing

_MIN_CALL_SPACING = 1.5  # seconds between outbound calls
_last_call = {"ts": 0.0}
_throttle_lock = threading.Lock()
_cache_lock = threading.Lock()

# Corporate proxies that MITM TLS (self-signed cert in chain) break verification
# for ALL outbound HTTPS. On first SSLError we fall back to unverified TLS and
# remember. Acceptable trade-off here: public market data only, nothing sensitive
# sent, and the proxy already terminates TLS anyway.
_ssl_state = {"verify": True}


def http_get(url: str, params: dict | None = None, timeout: int = 15,
             headers: dict | None = None):
    """requests.get with browser UA + automatic corporate-MITM fallback.

    `headers` merges over the default UA — NSE's archive host needs a Referer
    and a fuller Accept than the bare UA we send Yahoo.
    """
    hdrs = {**_UA, **(headers or {})}
    try:
        return requests.get(url, params=params, headers=hdrs, timeout=timeout,
                            verify=_ssl_state["verify"])
    except requests.exceptions.SSLError as e:
        # Downgrade ONLY for the corporate-MITM signature (self-signed cert in
        # chain) or an explicit opt-in — a transient handshake error or captive
        # portal must not silently disable TLS verification for the process.
        msg = str(e).lower()
        mitm = "self signed certificate" in msg or "self-signed certificate" in msg
        if _ssl_state["verify"] and (mitm or os.environ.get("ALLOW_INSECURE_TLS") == "1"):
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            _ssl_state["verify"] = False
            print("[data_source] TLS interception detected (corporate proxy) — "
                  "continuing without cert verification for public market data.")
            return requests.get(url, params=params, headers=hdrs, timeout=timeout,
                                verify=False)
        raise

_mem_cache: dict = {}


def _load_disk() -> dict:
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


_disk_cache = _load_disk()


def _save_disk():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(_disk_cache, f)
    except Exception:
        pass


def _cache_get(key: str, fresh_ttl: float):
    """Returns (data, is_fresh) or (None, False)."""
    now = time.time()
    with _cache_lock:
        entry = _mem_cache.get(key) or _disk_cache.get(key)
        if not entry:
            return None, False
        age = now - entry["ts"]
        if age > STALE_TTL:
            return None, False
        return entry["data"], age <= fresh_ttl


def _cache_put(key: str, data):
    entry = {"ts": time.time(), "data": data}
    with _cache_lock:
        _mem_cache[key] = entry
        _disk_cache[key] = entry
        _save_disk()


def _throttled(fn):
    """Space out outbound calls so we never burst."""
    with _throttle_lock:
        wait = _MIN_CALL_SPACING - (time.time() - _last_call["ts"])
        if wait > 0:
            time.sleep(wait)
        _last_call["ts"] = time.time()
    return fn()


# ------------------------------------------------- system-design hardening ---
# Circuit breaker: every upstream Yahoo call flows through _get_json. After
# N consecutive failures the circuit opens and calls fail fast for a cooldown
# instead of stacking 15s timeouts — callers then fall back to stale cache
# (which they already know how to do). Half-open after cooldown: one probe
# is let through; success closes the circuit.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN = 60.0  # seconds
_breaker = {"failures": 0, "opened_at": 0.0, "probing": False}
_breaker_lock = threading.Lock()


class CircuitOpenError(RuntimeError):
    """Upstream circuit is open — serve cached/stale data instead."""


def _breaker_allow() -> bool:
    with _breaker_lock:
        if _breaker["failures"] < _BREAKER_THRESHOLD:
            return True
        if _breaker["probing"]:
            return False  # a probe is already in flight — keep rejecting
        if time.time() - _breaker["opened_at"] >= _BREAKER_COOLDOWN:
            _breaker["probing"] = True  # half-open: admit exactly ONE probe
            return True
        return False


def _breaker_record(ok: bool):
    with _breaker_lock:
        _breaker["probing"] = False
        if ok:
            _breaker["failures"] = 0
        else:
            _breaker["failures"] += 1
            if _breaker["failures"] >= _BREAKER_THRESHOLD:
                _breaker["opened_at"] = time.time()


# Single-flight (request coalescing): when N threads want the same uncached
# key (screener fires 5 endpoints for one symbol; holdings-intel fans out),
# only the first hits the network — the rest wait on the key's lock, then
# re-read the now-warm cache. Classic cache-stampede protection.
# Keys derive from user input (symbols), so idle locks are pruned to bound
# memory; a pruned-and-recreated lock can at worst double-fetch a key idle
# for 15+ minutes, which is a perf hiccup, not a correctness issue.
_flight_locks: dict = {}
_flight_last: dict = {}
_flight_guard = threading.Lock()
_FLIGHT_STALE = 900  # seconds
_FLIGHT_PRUNE_ABOVE = 256  # only bother pruning past this many keys


@contextmanager
def _singleflight(key: str):
    now = time.monotonic()
    with _flight_guard:
        if len(_flight_locks) > _FLIGHT_PRUNE_ABOVE:
            for k in [k for k, ts in _flight_last.items()
                      if now - ts > _FLIGHT_STALE and not _flight_locks[k].locked()]:
                _flight_locks.pop(k, None)
                _flight_last.pop(k, None)
        lock = _flight_locks.setdefault(key, threading.Lock())
        _flight_last[key] = now
    with lock:
        yield


def _get_json(url: str, params: dict | None = None) -> dict:
    if not _breaker_allow():
        raise CircuitOpenError("upstream circuit open — cooling down")
    try:
        resp = _throttled(lambda: http_get(url, params))
    except Exception:
        _breaker_record(False)  # network-level failure: upstream unreachable
        raise
    # 4xx (except 429) means the upstream is ALIVE and rejecting this request —
    # a user's typo symbol must never open the circuit for everyone.
    if 400 <= resp.status_code < 500 and resp.status_code != 429:
        _breaker_record(True)
        resp.raise_for_status()
    try:
        resp.raise_for_status()  # 429 / 5xx
        data = resp.json()
    except Exception:
        _breaker_record(False)
        raise
    _breaker_record(True)
    return data


def get_json_cached(key: str, url: str, params: dict | None = None,
                    fresh_ttl: float = 3600) -> dict | None:
    """Throttled + cached GET reusing the same stale-if-error policy as quotes.

    Returns the last good payload when upstream fails, or None if we have never
    successfully fetched this key. Callers must handle None.
    """
    cached, fresh = _cache_get(key, fresh_ttl)
    if cached is not None and fresh:
        return cached
    with _singleflight(key):
        # another thread may have fetched while we waited on the lock
        cached, fresh = _cache_get(key, fresh_ttl)
        if cached is not None and fresh:
            return cached
        try:
            data = _get_json(url, params)
            _cache_put(key, data)
            return data
        except Exception:
            return cached


# ------------------------------------------------------- chart API parsing ---

def _fetch_chart(symbol: str, range_: str = "1y", interval: str = "1d"):
    """Returns (records, meta) from Yahoo's public v8 chart API."""
    data = _get_json(_CHART_URL.format(symbol=symbol),
                     {"range": range_, "interval": interval})
    result = data["chart"]["result"][0]
    meta = result.get("meta", {})
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]

    records = []
    opens, highs = quote.get("open") or [], quote.get("high") or []
    lows, closes = quote.get("low") or [], quote.get("close") or []
    vols = quote.get("volume") or []
    for i, ts in enumerate(timestamps):
        try:
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        except IndexError:
            continue
        if None in (o, h, l, c):
            continue
        records.append({
            "t": pd.Timestamp(ts, unit="s", tz="UTC").isoformat(),
            "o": float(o), "h": float(h), "l": float(l), "c": float(c),
            "v": float(vols[i] or 0) if i < len(vols) else 0.0,
        })
    return records, meta


# ---------------------------------------------------------------- history ---

def get_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Daily OHLCV DataFrame. Cached; serves stale data when upstream fails."""
    key = f"hist:{symbol}:{period}"
    cached, fresh = _cache_get(key, FRESH_TTL["history"])
    if cached is not None and fresh:
        return _records_to_df(cached)

    with _singleflight(key):
        cached, fresh = _cache_get(key, FRESH_TTL["history"])
        if cached is not None and fresh:
            return _records_to_df(cached)

        # Primary: public chart API
        try:
            records, _ = _fetch_chart(symbol, range_=period)
            if records:
                _cache_put(key, records)
                return _records_to_df(records)
        except Exception:
            pass

        # Secondary: yfinance
        try:
            df = _throttled(lambda: yf.Ticker(symbol).history(period=period))
            if df is not None and not df.empty:
                _cache_put(key, _df_to_records(df))
                return df
        except Exception:
            pass

        if cached is not None:
            return _records_to_df(cached)
        return pd.DataFrame()


# Yahoo's chart API caps intraday history per granularity, and exceeding the cap
# is a hard 422 rather than a truncated response. These ceilings are measured
# against the live endpoint, not guessed:
#
#     1m  -> 7 days      (1mo returns 422)
#     2m  -> 60 days
#     5m  -> 1 month     (3mo returns 422)
#     15m -> 1 month     (3mo returns 422)
#     30m -> 1 month
#     60m -> 2 years     (5y returns 422)
#     1d+ -> full history
#
# Ordered shortest-to-longest so _clamp_range can walk down to a legal value.
_RANGE_ORDER = ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]

MAX_RANGE_FOR_INTERVAL = {
    "1m": "7d", "2m": "1mo", "5m": "1mo", "15m": "1mo", "30m": "1mo",
    "60m": "2y", "90m": "2y", "1h": "2y",
    "1d": "max", "1wk": "max", "1mo": "max",
}

VALID_INTERVALS = tuple(MAX_RANGE_FOR_INTERVAL)


def clamp_range(interval: str, range_: str) -> str:
    """Largest legal range for an interval, so a caller can ask for more history
    than Yahoo allows and get the maximum instead of a 422."""
    cap = MAX_RANGE_FOR_INTERVAL.get(interval, "max")
    if cap == "max" or range_ == cap:
        return range_
    try:
        return range_ if _RANGE_ORDER.index(range_) <= _RANGE_ORDER.index(cap) else cap
    except ValueError:
        return cap


def get_ohlc(symbol: str, range_: str = "1y", interval: str = "1d") -> list:
    """Cached OHLCV records at an arbitrary interval.

    Separate from get_history (which is daily-only and returns a DataFrame)
    because the chart endpoint needs intraday granularity and the raw records.
    Intraday bars go stale fast, so the TTL scales with the bar size.
    """
    interval = interval if interval in MAX_RANGE_FOR_INTERVAL else "1d"
    range_ = clamp_range(interval, range_)
    key = f"ohlc:{symbol}:{range_}:{interval}"
    # A 1m bar is worthless 30 min later; a daily bar is fine for half an hour.
    ttl = 60 if interval in ("1m", "2m") else (
        300 if interval in ("5m", "15m", "30m") else FRESH_TTL["history"])

    cached, fresh = _cache_get(key, ttl)
    if cached is not None and fresh:
        return cached

    with _singleflight(key):
        cached, fresh = _cache_get(key, ttl)
        if cached is not None and fresh:
            return cached
        try:
            records, _meta = _fetch_chart(symbol, range_=range_, interval=interval)
            if records:
                _cache_put(key, records)
                return records
        except Exception:
            pass
        return cached if cached is not None else []


def _df_to_records(df: pd.DataFrame) -> list:
    out = []
    for date, row in df.iterrows():
        out.append({
            "t": date.isoformat(),
            "o": float(row["Open"]), "h": float(row["High"]),
            "l": float(row["Low"]), "c": float(row["Close"]),
            "v": float(row.get("Volume", 0) or 0),
        })
    return out


def _records_to_df(records: list) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame([
        {"Open": r["o"], "High": r["h"], "Low": r["l"], "Close": r["c"], "Volume": r["v"]}
        for r in records
    ], index=pd.to_datetime([r["t"] for r in records]))
    df.index.name = "Date"
    return df


def get_close_series(symbol: str, period: str = "1y") -> pd.Series:
    """Tz-naive, date-normalized Close series — safe to align across symbols."""
    df = get_history(symbol, period)
    if df.empty:
        return pd.Series(dtype=float)
    close = df["Close"].copy()
    idx = pd.to_datetime(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close.index = idx.normalize()
    return close[~close.index.duplicated(keep="last")]


def get_sector(symbol: str) -> str:
    """Sector for allocation charts. Cached forever — worth one `.info` call once."""
    key = f"sector:{symbol}"
    cached, _ = _cache_get(key, STALE_TTL)
    if cached is not None:
        return cached
    try:
        info = _throttled(lambda: yf.Ticker(symbol).info)
        sector = info.get("sector") or info.get("quoteType") or "Other"
    except Exception:
        return "Other"  # don't cache failures
    _cache_put(key, sector)
    return sector


# ------------------------------------------------------------------ quote ---

def get_quote(symbol: str) -> dict:
    """Quote from the chart API meta + last candles (avoids `.info` entirely)."""
    key = f"quote:{symbol}"
    cached, fresh = _cache_get(key, FRESH_TTL["quote"])
    if cached is not None and fresh:
        return cached
    with _singleflight(key):
        got, fresh = _cache_get(key, FRESH_TTL["quote"])
        if got is not None and fresh:
            return got
        return _fetch_quote(key, symbol, cached)


def _fetch_quote(key: str, symbol: str, cached):
    """Uncached quote fetch — call only while holding the key's flight lock."""
    quote = None
    try:
        records, meta = _fetch_chart(symbol, range_="5d")
        price = meta.get("regularMarketPrice")
        closes = [r["c"] for r in records]
        if price is None and closes:
            price = closes[-1]
        # previous session close: last candle before today's
        prev = None
        if len(closes) >= 2:
            prev = closes[-2] if abs(closes[-1] - (price or 0)) < 1e-9 else closes[-1]
        prev = meta.get("previousClose") or prev or meta.get("chartPreviousClose")
        quote = {
            "symbol": symbol,
            "shortName": meta.get("shortName") or symbol,
            "longName": meta.get("longName") or meta.get("shortName") or symbol,
            "currency": meta.get("currency") or "USD",
            "regularMarketPrice": price,
            "previousClose": prev,
            "regularMarketChangePercent": ((price - prev) / prev * 100) if price and prev else 0,
            "regularMarketVolume": meta.get("regularMarketVolume") or (records[-1]["v"] if records else None),
            "fiftyTwoWeekHigh": meta.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": meta.get("fiftyTwoWeekLow"),
            "marketCap": None,
        }
        # enrich from cached yearly history if available (no extra network call)
        hist_cached, _ = _cache_get(f"hist:{symbol}:1y", STALE_TTL)
        if hist_cached:
            vols = [r["v"] for r in hist_cached[-63:] if r["v"]]
            if vols:
                quote["averageDailyVolume3Month"] = sum(vols) / len(vols)
            if not quote["fiftyTwoWeekHigh"]:
                quote["fiftyTwoWeekHigh"] = max(r["h"] for r in hist_cached)
            if not quote["fiftyTwoWeekLow"]:
                quote["fiftyTwoWeekLow"] = min(r["l"] for r in hist_cached)
    except Exception:
        quote = None

    if quote is None:
        df = get_history(symbol, "1y")
        if df.empty or len(df) < 2:
            if cached is not None:
                return cached
            return {"error": "Market data temporarily unavailable (rate limited)."}
        last, prev = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
        quote = {
            "symbol": symbol,
            "shortName": symbol,
            "longName": symbol,
            "currency": "USD",
            "regularMarketPrice": last,
            "previousClose": prev,
            "regularMarketChangePercent": (last - prev) / prev * 100 if prev else 0,
            "regularMarketVolume": float(df["Volume"].iloc[-1]),
            "averageDailyVolume3Month": float(df["Volume"].tail(63).mean()),
            "fiftyTwoWeekHigh": float(df["High"].max()),
            "fiftyTwoWeekLow": float(df["Low"].min()),
            "marketCap": None,
            "stale": True,
        }

    _cache_put(key, quote)
    return quote


# --------------------------------------------------------------- overview ---

OVERVIEW_SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'GOOGL',
                    'JPM', 'V', 'WMT', 'JNJ', 'PG', 'XOM', 'BAC', 'MA']

# Static approximate caps (USD) used only for heatmap tile sizing when live
# caps are unavailable — sizing is cosmetic, not displayed as data.
_APPROX_CAPS = {
    'AAPL': 3.4e12, 'MSFT': 3.6e12, 'NVDA': 3.5e12, 'AMZN': 2.3e12,
    'GOOGL': 2.4e12, 'META': 1.6e12, 'TSLA': 1.1e12, 'JPM': 8e11,
    'V': 7e11, 'WMT': 8e11, 'JNJ': 4e11, 'PG': 4e11, 'XOM': 5e11,
    'BAC': 3.5e11, 'MA': 5e11,
}


def get_market_overview() -> dict:
    key = "overview"
    cached, fresh = _cache_get(key, FRESH_TTL["overview"])
    if cached is not None and fresh:
        return cached

    results = []
    try:
        # ONE batched spark request for all symbols
        data = _get_json(_SPARK_URL, {
            "symbols": ",".join(OVERVIEW_SYMBOLS),
            "range": "5d",
            "interval": "1d",
        })
        for sym in OVERVIEW_SYMBOLS:
            try:
                series = data.get(sym) or {}
                closes = [c for c in (series.get("close") or []) if c is not None]
                if len(closes) < 2:
                    continue
                curr, prev = float(closes[-1]), float(closes[-2])
                results.append({
                    "symbol": sym,
                    "price": curr,
                    "change": (curr - prev) / prev * 100,
                    "volume": 0,
                    "marketCap": _APPROX_CAPS.get(sym),
                })
            except Exception:
                continue
    except Exception:
        results = []

    if not results:
        if cached is not None:
            return cached
        return {"error": "Market data temporarily unavailable (rate limited)."}

    results = sorted(results, key=lambda x: x["change"], reverse=True)
    overview = {
        "top_gainers": results[:5],
        "top_losers": results[-5:],
        "most_active": sorted(results, key=lambda x: abs(x["change"]), reverse=True)[:5],
        "all_assets": results,
    }
    _cache_put(key, overview)
    return overview


# --------------------------------------------------------------- trending ---

def get_trending() -> list:
    """Trending US tickers with quotes; cached, stale-if-error."""
    key = "trending"
    cached, fresh = _cache_get(key, FRESH_TTL["overview"])
    if cached is not None and fresh:
        return cached

    try:
        data = _get_json("https://query1.finance.yahoo.com/v1/finance/trending/US",
                         {"count": 10})
        results = data.get('finance', {}).get('result', [{}])[0].get('quotes', [])
        symbols = [q.get('symbol') for q in results if q.get('symbol')][:8]
        quotes = []
        for sym in symbols:
            q = get_quote(sym)
            if 'error' not in q:
                quotes.append(q)
        if quotes:
            _cache_put(key, quotes)
            return quotes
    except Exception:
        pass

    return cached if cached is not None else []
