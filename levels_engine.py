"""
Support/resistance proximity scanner — the "what is at a decision point today"
board that sits alongside Discover's Ideas and Screens.

The job: given a universe, find the names sitting AT a level that has actually
been respected before, on the daily and weekly timeframes, and say why. These
are swing-trade watch candidates — a stock pressing into a level that has
turned it back four times is a decision point whether it breaks or rejects, and
either outcome is tradeable with a defined stop.

How it works, and what it costs:

  - Series come from screener_engine._load_universe_series, the SAME batched,
    disk-cached spark payloads Discover and Screens already pull. Scanning
    levels for a universe that has been scanned in the last 30 minutes costs
    ZERO upstream calls. Nothing here fetches per symbol.
  - Levels are built from swing pivots (fractal local extrema), then clustered
    by price proximity. A cluster touched repeatedly, over a long span, and
    recently, scores high; a single stale pivot scores near zero.
  - The daily and weekly passes are independent. Where they agree on a price,
    that is confluence and it is scored and labelled as such — a weekly level
    backed by a daily one is the setup this board exists to surface.
  - Proximity is measured in DAILY SIGMAS, not percent. "Within 2%" means
    something completely different for a utility and for a small-cap miner;
    "within 1.5 daily moves" means the same thing for both.

Known limitation, stated plainly because it changes how the output should be
read: these are CLOSE-based pivots. The spark feed carries closes only, so a
level defended intraday by a wick that closed away from it will not appear.
Close-based levels are the more conservative set — everything found here was
respected on a closing basis, which is the stronger claim — but it is not the
complete set. Rows carry basis: "close" so the UI can say so.

Product doctrine: unlike Ideas, this board DOES rank, because "which names are
at the best levels" is the entire question being asked. It ranks by level
quality and proximity, never by expected return, and it makes no directional
call — a level is a place where something happens, not a prediction of what.
"""
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import screener_engine


# ------------------------------------------------------------------ tuning ---

# Fractal half-width: a pivot high must be the highest close within +/- k bars.
# Daily uses a wider window than weekly simply because it has ~5x the bars.
_PIVOT_K = {"daily": 4, "weekly": 2}

# Two pivots belong to the same level if they sit within this fraction of price.
# Wider on the weekly, where a "level" is a zone rather than a line.
_CLUSTER_TOL_PCT = {"daily": 0.015, "weekly": 0.025}

# A level needs at least this many touches to be a level rather than a dot.
_MIN_TOUCHES = 2

# Recency decay: a touch this many bars ago counts e^-1 as much as one today.
_RECENCY_SCALE = {"daily": 90.0, "weekly": 26.0}

# How close price must sit to the level to be "at" it, in daily sigmas.
_NEAR_SIGMAS = 1.75
# ...and a hard percentage ceiling, so a violently volatile name cannot report
# "at its level" from 20% away just because its sigma is enormous.
_NEAR_MAX_PCT = 7.0

# Confluence: a daily and a weekly level within this fraction of each other are
# the same level seen twice. The bonus doubles as the score normaliser, so the
# published score lands on 0-100 without a clamp flattening the top of the rank.
_CONFLUENCE_TOL_PCT = 0.02
_CONFLUENCE_BONUS = 1.30

_MIN_BARS = {"daily": 60, "weekly": 30}

# Result memoisation, matching screener_engine's endpoint cache policy.
_RESULT_TTL = 300.0
_results: dict = {}
_results_lock = threading.Lock()


# ------------------------------------------------------------------ pivots ---

def _pivot_indices(values: np.ndarray, k: int, kind: str) -> np.ndarray:
    """Indices of fractal local extrema: strictly the extreme of a 2k+1 window.

    Vectorised as a strided comparison rather than a Python loop — this runs
    over ~250 bars x 250 symbols x 2 timeframes on every cold scan.
    """
    n = len(values)
    if n < 2 * k + 1:
        return np.empty(0, dtype=int)
    idx = np.arange(k, n - k)
    window = np.stack([values[idx + off] for off in range(-k, k + 1)], axis=1)
    centre = values[idx]
    if kind == "high":
        keep = centre >= window.max(axis=1)
    else:
        keep = centre <= window.min(axis=1)
    return idx[keep]


def _cluster(prices: np.ndarray, ages: np.ndarray, kinds: list,
             tol_pct: float) -> list:
    """Group nearby pivot prices into levels.

    Single linkage over sorted prices: walk upward and start a new cluster
    whenever the gap to the previous price exceeds the tolerance. Simple, and
    correct for this shape of data — pivots genuinely do fall into bands, and
    anything fancier (kernel density, k-means) would need a bandwidth or a k
    that is exactly the parameter being avoided.
    """
    if len(prices) == 0:
        return []
    order = np.argsort(prices)
    clusters = []
    current = [order[0]]
    for prev, cur in zip(order, order[1:]):
        if abs(prices[cur] - prices[prev]) <= tol_pct * max(prices[prev], 1e-9):
            current.append(cur)
        else:
            clusters.append(current)
            current = [cur]
    clusters.append(current)

    out = []
    for members in clusters:
        members = np.asarray(members)
        p = prices[members]
        a = ages[members]
        # Recent touches define where the level IS now; old ones establish that
        # it exists at all. Weighting the price by recency lets a level drift
        # with its more recent defences instead of being pinned by a stale one.
        w = np.exp(-a / max(a.max(), 1.0))
        out.append({
            "price": float(np.average(p, weights=w)),
            "touches": int(len(members)),
            "newest_age": float(a.min()),
            "oldest_age": float(a.max()),
            "kinds": [kinds[i] for i in members],
        })
    return out


def _is_flip(kinds: list) -> bool:
    """Has this level genuinely acted as BOTH support and resistance?

    Not "does the cluster contain one of each" — over a year, a 1.5% band
    catches a stray opposite pivot on almost every level, which made the flag
    fire on ~90% of rows and turned it into decoration. A real flip needs the
    minority role to be a repeated, material share of the level's history.
    """
    highs = sum(1 for k in kinds if k == "high")
    lows = len(kinds) - highs
    minority = min(highs, lows)
    return minority >= 2 and minority / max(len(kinds), 1) >= 0.25


def _score(level: dict, n_bars: int, recency_scale: float) -> float:
    """Level quality, 0-100. Three independent things make a level matter:

      touches  - how many times it was respected (saturating: the 6th touch
                 adds far less than the 2nd)
      recency  - whether it is still live, or archaeology
      span     - a level defended across many months beats one defended three
                 times in a single week, which is just one event

    Multiplicative, so a level that fails any one of them scores low no matter
    how well it does on the others — which is the intended behaviour.
    """
    touch_term = float(np.tanh((level["touches"] - 1) * 0.55))
    recency_term = float(np.exp(-level["newest_age"] / recency_scale))
    span = level["oldest_age"] - level["newest_age"]
    span_term = float(np.log1p(span) / np.log1p(max(n_bars, 2)))
    # A level that has acted as BOTH support and resistance has flipped
    # polarity, which is the strongest evidence a price matters at all.
    polarity = 1.25 if _is_flip(level["kinds"]) else 1.0
    raw = touch_term * (0.35 + 0.65 * recency_term) * (0.45 + 0.55 * span_term)
    return float(min(100.0, 100.0 * raw * polarity))


def _levels_for(closes: pd.Series, timeframe: str) -> list:
    """Scored support/resistance levels for one series on one timeframe."""
    values = closes.to_numpy(dtype=float)
    n = len(values)
    if n < _MIN_BARS[timeframe]:
        return []
    k = _PIVOT_K[timeframe]
    hi = _pivot_indices(values, k, "high")
    lo = _pivot_indices(values, k, "low")
    if len(hi) + len(lo) == 0:
        return []
    idx = np.concatenate([hi, lo])
    kinds = ["high"] * len(hi) + ["low"] * len(lo)
    prices = values[idx]
    ages = (n - 1 - idx).astype(float)  # bars before the last bar

    levels = []
    for lv in _cluster(prices, ages, kinds, _CLUSTER_TOL_PCT[timeframe]):
        if lv["touches"] < _MIN_TOUCHES:
            continue
        lv["timeframe"] = timeframe
        lv["strength"] = _score(lv, n, _RECENCY_SCALE[timeframe])
        levels.append(lv)
    return levels


def _weekly(closes: pd.Series) -> pd.Series:
    """Weekly closes. W-FRI so a partial current week resolves to its latest
    print rather than being dropped — the live week is exactly the one the
    trader is standing in."""
    return closes.resample("W-FRI").last().dropna()


# ------------------------------------------------------------------ scoring ---

def _describe_age(bars: float, timeframe: str) -> str:
    unit_days = 1 if timeframe == "daily" else 7
    days = bars * unit_days
    if days < 14:
        return f"{int(round(days))}d ago"
    if days < 70:
        return f"{int(round(days / 7))}w ago"
    return f"{int(round(days / 30.44))}mo ago"


def _reason(row: dict) -> str:
    """The evidence line. Every number in it is one the scan actually used —
    a reason the user cannot audit is a reason they cannot trust."""
    lv = row["level"]
    parts = [
        f"{row['side']} {lv['price']:.2f} on the {lv['timeframe']}",
        f"{lv['touches']} touches",
        f"last {_describe_age(lv['newest_age'], lv['timeframe'])}",
        f"{abs(row['distance_pct']):.1f}% away ({row['distance_sigmas']:.1f}σ)",
    ]
    if row["confluence"]:
        parts.append("daily+weekly confluence")
    if row["flipped"]:
        parts.append("polarity flip (was support and resistance)")
    return " · ".join(parts)


def _scan_symbol(closes: pd.Series) -> dict | None:
    """Nearest high-quality level for one symbol, or None if it is not near one."""
    if len(closes) < _MIN_BARS["daily"]:
        return None
    price = float(closes.iloc[-1])
    if not np.isfinite(price) or price <= 0:
        return None

    rets = closes.pct_change().dropna()
    if len(rets) < 20:
        return None
    sigma_pct = float(rets.tail(60).std() * 100)
    if not np.isfinite(sigma_pct) or sigma_pct <= 0:
        return None

    levels = _levels_for(closes, "daily") + _levels_for(_weekly(closes), "weekly")
    if not levels:
        return None

    # Confluence: mark every level that has a partner on the other timeframe.
    for lv in levels:
        lv["confluence"] = any(
            other["timeframe"] != lv["timeframe"]
            and abs(other["price"] - lv["price"]) <= _CONFLUENCE_TOL_PCT * lv["price"]
            for other in levels
        )

    best = None
    for lv in levels:
        distance_pct = (lv["price"] / price - 1) * 100
        distance_sigmas = abs(distance_pct) / sigma_pct
        if distance_sigmas > _NEAR_SIGMAS or abs(distance_pct) > _NEAR_MAX_PCT:
            continue
        # Proximity counts, but a level you are sitting exactly on is not
        # automatically better than a strong one half a sigma away — so
        # proximity scales the level's own quality rather than replacing it.
        proximity = float(np.exp(-distance_sigmas / _NEAR_SIGMAS))
        # Divided by the maximum multiplier so the published score stays on a
        # 0-100 scale without clamping, which would flatten the ordering at
        # the top exactly where the ranking matters most.
        conf = _CONFLUENCE_BONUS if lv["confluence"] else 1.0
        score = lv["strength"] * (0.55 + 0.45 * proximity) * conf / _CONFLUENCE_BONUS
        if best is None or score > best["score"]:
            best = {
                "score": float(score),
                "level": lv,
                "side": "resistance" if lv["price"] >= price else "support",
                "distance_pct": float(distance_pct),
                "distance_sigmas": float(distance_sigmas),
                "confluence": bool(lv["confluence"]),
                "flipped": _is_flip(lv["kinds"]),
            }
    if best is None:
        return None

    best["price"] = price
    best["sigma_pct"] = sigma_pct
    best["reason"] = _reason(best)
    return best


# -------------------------------------------------------------- public API ---

def _result_get(key):
    with _results_lock:
        entry = _results.get(key)
        if entry and time.time() - entry["ts"] <= _RESULT_TTL:
            return entry["data"]
    return None


def _result_put(key, data):
    with _results_lock:
        _results[key] = {"ts": time.time(), "data": data}


def get_levels(universe_id: str = "us_large", limit: int = 40,
               side: str = "all") -> dict:
    """Names sitting at a respected support/resistance level, best first.

    `side` filters to "support" / "resistance"; anything else means both.
    Raises screener_engine.UnknownUniverseError for a bad universe id, which
    the route maps to HTTP 400.
    """
    uni = screener_engine._resolve_universe(universe_id)
    limit = max(1, min(int(limit), 100))
    side = side if side in ("support", "resistance") else "all"

    key = ("levels", universe_id, limit, side)
    cached = _result_get(key)
    if cached is not None:
        return cached

    series, as_of = screener_engine._load_universe_series(universe_id)

    rows = []
    for symbol, closes in series.items():
        try:
            hit = _scan_symbol(closes)
        except Exception:
            continue  # one bad series must never sink the scan
        if hit is None or (side != "all" and hit["side"] != side):
            continue
        lv = hit["level"]
        rows.append({
            "symbol": symbol,
            "name": uni["symbols"].get(symbol, symbol),
            "currency": screener_engine._symbol_currency(uni, symbol),
            "price": round(hit["price"], 2),
            "level": round(lv["price"], 2),
            "timeframe": lv["timeframe"],
            "side": hit["side"],
            "distance_pct": round(hit["distance_pct"], 2),
            "distance_sigmas": round(hit["distance_sigmas"], 2),
            "touches": lv["touches"],
            "last_touch_bars": int(lv["newest_age"]),
            "last_touch_label": _describe_age(lv["newest_age"], lv["timeframe"]),
            "strength": round(lv["strength"], 1),
            "score": round(hit["score"], 1),
            "confluence": hit["confluence"],
            "flipped": hit["flipped"],
            "daily_sigma_pct": round(hit["sigma_pct"], 2),
            "reason": hit["reason"],
            "spark": [round(float(v), 2 if abs(v) >= 1 else 4)
                      for v in closes.tail(30)],
        })

    # Ranked, unlike Ideas — see the module docstring. Symbol breaks ties so
    # the order is stable between identical scans.
    rows.sort(key=lambda r: (-r["score"], r["symbol"]))
    rows = rows[:limit]

    payload = {
        "universe": universe_id,
        "universe_label": uni["label"],
        "currency": uni["currency"],
        "universes": screener_engine._universe_index(),
        "side": side,
        "as_of": as_of,
        **screener_engine._freshness(as_of),
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scanned": len(uni["symbols"]),
        "parsed": len(series),  # 0 == upstream outage; routes send no-store
        "matched": len(rows),
        "basis": "close",  # pivots are close-based — see the module docstring
        "rows": rows,
    }
    if series:  # same outage rule as Discover: never memoise a blank scan
        _result_put(key, payload)
    return payload
