"""
Top movers board for the Indian market: gainers, losers, cap-segment movers,
circuit hits and intraday volatility across every NSE cash-EQ company.

Scope and cost
--------------
The scan universe is every NSE cash-EQ operating company (~2,127) resolved by
nse_segments -- deliberately NOT just the index constituents. The five NSE size
indices cover 752 names, and the stocks that actually hit a 20% circuit are
overwhelmingly the ~1,375 that sit below the Total Market 750 line. On
2026-08-12 both +20% names (ABMINTLLTD, SINGERIND) were outside every index, so
an index-only scan could not surface them however hard it looked.

Every selectable scope -- All NSE, NIFTY Total Market, 50, NEXT 50, 100, 500,
Midcap 150, Smallcap 250, Microcap 250, the cap segments, Unranked, F&O and the
intraday-eligible set -- is a SUBSET of that one universe, so switching tabs
costs zero extra upstream calls.

Yahoo's spark endpoint caps at 20 symbols per request and data_source spaces
outbound calls 1.5s apart, so a cold full scan is ~107 requests / ~3 min. That
is far too slow to block a page load, so:

  - Each 20-symbol chunk is cached independently (30 min fresh, 7 day stale) and
    keyed by its sorted symbol list, so chunk boundaries are stable across calls.
  - get_movers() NEVER blocks on a cold chunk. It computes from whatever chunks
    are already warm and reports `coverage` honestly, then a single background
    warmer thread fills the rest. The client polls and the board fills in.
  - `partial` / `coverage` in the payload let the UI say "231 of 2127 priced"
    instead of silently presenting a top-10 drawn from a tenth of the market.

Two data-quality guards, both learned from real bad prints
---------------------------------------------------------
  1. ETFs and mutual funds are excluded upstream in nse_segments by ISIN range
     (INF, not INE). NSE lists 339 of them under cash series EQ; KOTAKNV20
     printed +84,260% on 2026-08-12 because its NAV series is not split-adjusted
     the way an equity series is. One such row at the top would discredit the
     whole board.
  2. Any |move| above _SANITY_MAX_MOVE_PCT is dropped as a suspected unadjusted
     corporate action, because NSE's widest cash circuit band is 20%. The drop
     is COUNTED and reported, never silent.

Circuit inference
-----------------
NSE does not publish per-stock circuit bands in any feed we have, so a circuit
hit is INFERRED: a close within a hair of a standard 2/5/10/20% band. That
inference is corroborated where intraday data is warm -- a genuinely locked
stock closes exactly at its intraday extreme and often trades only a handful of
5-minute bars (ABMINTLLTD traded 2 bars on 2026-08-12). The payload always says
which basis was used so the UI can avoid claiming more than we know.

Ranking
-------
Unlike screener_engine (which lists alphabetically and deliberately never ranks),
a movers board IS a ranking -- that is the entire feature. It lives in its own
module so screener_engine's "this engine LISTS, it never RANKS" contract stays
true of that engine.
"""
import math
import threading
import time
from datetime import datetime, timezone

import pandas as pd

import data_source
import nse_segments

# ------------------------------------------------------------------ config ---

_SPARK_CHUNK = 20        # Yahoo spark hard-caps at 20 symbols (400 above it)
_CHUNK_TTL = 30 * 60     # a mover board 30 min stale is still a useful board
_SPARK_RANGE = "1mo"     # ~22 bars: enough for prev close, 5d return, sparkline

# Intraday enrichment. 5-minute closes for a single session: enough to measure
# the day's travel and to see a circuit lock, without a per-symbol chart call.
_INTRADAY_RANGE = "1d"
_INTRADAY_INTERVAL = "5m"
_INTRADAY_TTL = 10 * 60  # moves during the session, so a tighter window
# Enrichment is capped: it exists to rank a shortlist, not to re-scan the market.
_INTRADAY_CAP = 80       # 4 chunks

# A "big mover" by default. Retail screens usually start around 5%; the user can
# push it to 20% for the rare explosive names.
DEFAULT_MIN_MOVE = 5.0
MAX_ROWS = 100

# How many names each cap-segment column carries on the Top Movers section.
SEGMENT_ROWS = 8

# NSE cash-market circuit bands we are willing to infer from a close.
#
# NSE also operates a 2% band for a small set of surveillance names, but it is
# deliberately NOT listed here: with any workable tolerance the window around
# 2% swallows a large share of ordinary closes. On 2026-08-12 it flagged
# -1.81%, -1.84%, -1.87%, -1.93% and -2.05% as "circuit hits", none of which
# plausibly were. A band we cannot distinguish from noise is not a signal.
_CIRCUIT_BANDS = (20.0, 10.0, 5.0)
# Circuit prices are derived from the previous close and then rounded to the
# tick size, so a locked stock lands a hair under the nominal band: 19.98%,
# 9.99% and 4.99% are all real prints from a single session. 0.15pp covers that
# rounding while keeping the coincidence window half as wide as 0.25 would.
_BAND_TOLERANCE = 0.15

# NSE's widest cash band is 20%, so anything past this is not a market move --
# it is an unadjusted split, bonus or demerger in the price series.
_SANITY_MAX_MOVE_PCT = 25.0

_warm_lock = threading.Lock()
_warm_state = {"running": False, "started_at": 0.0, "done": 0, "total": 0,
               "phase": None}


# ------------------------------------------------------------------- chunks ---

def _chunks(symbols: list[str]) -> list[list[str]]:
    return [symbols[i:i + _SPARK_CHUNK]
            for i in range(0, len(symbols), _SPARK_CHUNK)]


def _chunk_key(chunk: list[str]) -> str:
    return f"spark:{_SPARK_RANGE}:{','.join(chunk)}"


def _read_chunk(chunk: list[str]):
    """Cached payload for one chunk without touching the network.

    Returns (payload, fresh). Reuses screener_engine's cache namespace shape so
    a universe scan and a movers scan share warmed chunks where they overlap.
    """
    return data_source._cache_get(_chunk_key(chunk), _CHUNK_TTL)


def _fetch_chunk(chunk: list[str]):
    """Network fetch for one chunk, honouring the global throttle and cache."""
    joined = ",".join(chunk)
    return data_source.get_json_cached(
        key=_chunk_key(chunk),
        url=data_source._SPARK_URL,
        params={"symbols": joined, "range": _SPARK_RANGE, "interval": "1d"},
        fresh_ttl=_CHUNK_TTL,
    )


# -------------------------------------------------------- intraday chunks ---

def _intraday_key(chunk: list[str]) -> str:
    return f"spark:{_INTRADAY_RANGE}:{_INTRADAY_INTERVAL}:{','.join(chunk)}"


def _intraday_sym_key(sym: str) -> str:
    return f"intraday:{_INTRADAY_INTERVAL}:{sym}"


def _read_intraday_sym(sym: str):
    return data_source._cache_get(_intraday_sym_key(sym), _INTRADAY_TTL)


def _fetch_intraday_chunk(chunk: list[str]):
    """Fetch one 5-minute chunk and fan it out to PER-SYMBOL cache entries.

    The fan-out is the point. The daily scan chunks a fixed 2,127-name universe,
    but the intraday shortlist is a different, shifting set of ~80 names, so
    chunk-keyed intraday entries were written under one grouping and looked up
    under another -- the keys never matched and every read missed. Keying the
    parsed metrics by symbol makes the cache independent of how the request
    happened to be batched.
    """
    payload = data_source.get_json_cached(
        key=_intraday_key(chunk),
        url=data_source._SPARK_URL,
        params={"symbols": ",".join(chunk), "range": _INTRADAY_RANGE,
                "interval": _INTRADAY_INTERVAL},
        fresh_ttl=_INTRADAY_TTL,
    )
    for sym in chunk:
        node = (payload or {}).get(sym)
        if not node:
            continue
        try:
            m = _intraday_metrics(node)
            if m:
                data_source._cache_put(_intraday_sym_key(sym), m)
        except Exception:
            continue
    return payload


def _intraday_metrics(node: dict) -> dict | None:
    """Session travel and circuit-lock evidence from one 5-minute spark node.

    `bars` is load-bearing, not decoration: a stock locked at its upper circuit
    stops trading, so a two-bar session alongside a close at the day's high is
    strong evidence of a genuine lock rather than a coincidental round number.
    """
    closes = [c for c in (node.get("close") or []) if c is not None]
    prev = node.get("chartPreviousClose")
    if len(closes) < 2 or not prev:
        return None
    try:
        prev = float(prev)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(prev) or prev <= 0:
        return None

    # The session this belongs to, so an intraday range can never be shown
    # against a DIFFERENT day's close once the entry goes stale overnight.
    stamps = [t for t in (node.get("timestamp") or []) if t is not None]
    session = (pd.to_datetime(stamps[-1], unit="s").strftime("%Y-%m-%d")
               if stamps else None)

    lo, hi, last, first = min(closes), max(closes), closes[-1], closes[0]
    # Range is quoted against the previous close so it is comparable with the
    # day's change, and so a gap-and-freeze name is not reported as 0% travel.
    rng_pct = (hi - lo) / prev * 100.0
    eps = max(hi, 1.0) * 1e-6
    return {
        "session": session,
        "bars": len(closes),
        "intraday_open": round(float(first), 2),
        "intraday_low": round(float(lo), 2),
        "intraday_high": round(float(hi), 2),
        "intraday_range_pct": round(rng_pct, 2),
        "close_at_high": abs(last - hi) <= eps,
        "close_at_low": abs(last - lo) <= eps,
    }


def _read_intraday(symbols: list[str], session: str | None) -> dict:
    """Cached-only intraday metrics. Never fetches, so a page load stays fast.

    A STALE entry is still accepted -- after the close, a range measured twenty
    minutes ago is exactly as true as one measured now -- but only when it
    belongs to the same session as the daily close it will be shown beside.
    That check is what stops yesterday's travel being attributed to today.
    """
    out: dict = {}
    for sym in symbols:
        m, _fresh = _read_intraday_sym(sym)
        if not m:
            continue
        if session and m.get("session") and m["session"] != session:
            continue
        out[sym] = m
    return out


# -------------------------------------------------------------- background ---

def _warm(symbols: list[str]):
    """Background filler, in two phases, then exit.

    Phase 1 fills every cold daily chunk -- that is what the board is computed
    from. Phase 2 can only be chosen once phase 1 has run, because the intraday
    shortlist IS a function of the daily scan: enriching before the scan exists
    would mean guessing which names matter.

    Only one warmer runs at a time. A page load that finds the cache cold gets
    an immediate partial answer and this thread behind it; the next poll sees
    more coverage.
    """
    try:
        cold = []
        for chunk in _chunks(symbols):
            payload, fresh = _read_chunk(chunk)
            if payload is None or not fresh:
                cold.append(chunk)

        _warm_state.update({"total": len(cold), "done": 0, "phase": "daily"})
        for chunk in cold:
            try:
                _fetch_chunk(chunk)
            except Exception:
                pass  # one bad chunk must not stop the warm
            _warm_state["done"] += 1

        # ---- phase 2: intraday enrichment for the names that earned it ------
        shortlist = _intraday_shortlist(symbols)
        # Cold-check per symbol, matching how the entries are stored: a chunk is
        # worth fetching if any name in it lacks a fresh entry.
        cold_intraday = [
            chunk for chunk in _chunks(shortlist)
            if not all(_read_intraday_sym(s)[1] for s in chunk)
        ]
        _warm_state.update({"total": len(cold_intraday), "done": 0,
                            "phase": "intraday"})
        for chunk in cold_intraday:
            try:
                _fetch_intraday_chunk(chunk)
            except Exception:
                pass
            _warm_state["done"] += 1
    finally:
        with _warm_lock:
            _warm_state["running"] = False
            _warm_state["phase"] = None


def _intraday_shortlist(symbols: list[str]) -> list[str]:
    """Which symbols deserve a 5-minute fetch: the biggest movers and every
    circuit candidate, capped so enrichment stays a shortlist.

    Sorted so chunk boundaries -- and therefore cache keys -- stay stable
    between calls, the same reason nse_segments.all_symbols() sorts.
    """
    metrics, _as_of, _c, _f = _scan(symbols)
    table = nse_segments.get_table()["symbols"]
    by_yahoo = {f"{e['symbol']}.NS": e for e in table.values()}

    ranked = []
    for sym, m in metrics.items():
        chg = m["change_pct"]
        if abs(chg) > _SANITY_MAX_MOVE_PCT:
            continue
        e = by_yahoo.get(sym)
        if e is None:
            continue
        # A circuit candidate is worth confirming even on a small band, and an
        # MIS-eligible big mover is the intraday section's raw material.
        candidate = _circuit_band(chg) is not None
        priority = 1 if candidate else 0
        if not candidate and not e.get("intraday"):
            continue
        ranked.append((priority, abs(chg), sym))

    ranked.sort(reverse=True)
    return sorted(sym for _p, _a, sym in ranked[:_INTRADAY_CAP])


def _ensure_warm(symbols: list[str]):
    """Start the background warmer if chunks are missing and none is running."""
    with _warm_lock:
        if _warm_state["running"]:
            return
        _warm_state["running"] = True
        _warm_state["started_at"] = time.time()
    t = threading.Thread(target=_warm, args=(symbols,),
                         name="movers-warm", daemon=True)
    t.start()


# ------------------------------------------------------------------ parsing ---

def _series_from_node(node: dict) -> pd.Series | None:
    """Daily close series for one spark node, deduped by calendar day.

    During a live session Yahoo appends a partial "today" print alongside the
    day's close; collapsing same-day duplicates to the LAST value keeps the 1d
    change from comparing today against today.
    """
    stamps = node.get("timestamp") or []
    closes = node.get("close") or []
    pairs = [(t, c) for t, c in zip(stamps, closes)
             if c is not None and t is not None]
    if len(pairs) < 2:
        return None
    idx = pd.to_datetime([p[0] for p in pairs], unit="s")
    s = pd.Series([float(p[1]) for p in pairs], index=idx)
    s = s[~s.index.normalize().duplicated(keep="last")]
    return s if len(s) >= 2 else None


def _row_metrics(s: pd.Series) -> dict | None:
    """1d / 5d change plus a compact sparkline from a close series."""
    last = float(s.iloc[-1])
    prev = float(s.iloc[-2])
    if not (math.isfinite(last) and math.isfinite(prev)) or prev <= 0:
        return None

    ret5 = None
    if len(s) >= 6:
        base = float(s.iloc[-6])
        if math.isfinite(base) and base > 0:
            ret5 = (last / base - 1.0) * 100.0

    return {
        "price": round(last, 2),
        "prev_close": round(prev, 2),
        "change_abs": round(last - prev, 2),
        "change_pct": round((last / prev - 1.0) * 100.0, 2),
        "ret_5d_pct": None if ret5 is None else round(ret5, 2),
        # Sparkline is decoration, not data: rounded hard to keep payload small.
        "spark": [round(float(v), 2) for v in s.iloc[-20:].tolist()],
        "last_bar": s.index[-1].strftime("%Y-%m-%d"),
    }


def _scan(symbols: list[str]) -> tuple[dict, str | None, int, int]:
    """Metrics for every symbol whose chunk is cached (fresh OR stale).

    Returns (metrics_by_yahoo_symbol, as_of, cached_chunks, fresh_chunks).
    Never fetches -- stale chunks are still served (stale-if-error), and the
    fresh count is what decides whether a background refresh is owed.
    """
    metrics: dict = {}
    latest: pd.Timestamp | None = None
    cached_chunks = 0
    fresh_chunks = 0

    for chunk in _chunks(symbols):
        payload, fresh = _read_chunk(chunk)
        if payload is None:
            continue
        cached_chunks += 1
        if fresh:
            fresh_chunks += 1
        for sym in chunk:
            node = payload.get(sym)
            if not node:
                continue
            try:
                s = _series_from_node(node)
                if s is None:
                    continue
                m = _row_metrics(s)
                if m is None:
                    continue
                m["stale_chunk"] = not fresh
                metrics[sym] = m
                if latest is None or s.index[-1] > latest:
                    latest = s.index[-1]
            except Exception:
                continue  # one bad symbol must not sink the scan

    as_of = latest.strftime("%Y-%m-%d") if latest is not None else None
    return metrics, as_of, cached_chunks, fresh_chunks


# ------------------------------------------------------------------ circuits ---

def _circuit_band(change_pct: float) -> float | None:
    """The standard NSE band a close sits at, or None if it sits between bands.

    Checked from the widest band down, so a name at +19.98% is read as a 20%
    lock rather than being mistaken for anything narrower. A close BEYOND a band
    (say +10.80%) is not a hit on that band -- it proves the stock has a wider
    one -- so the window is two-sided.
    """
    a = abs(change_pct)
    for band in _CIRCUIT_BANDS:
        if band - _BAND_TOLERANCE <= a <= band + _BAND_TOLERANCE:
            return band
    return None


def _circuit_confidence(row: dict) -> str:
    """How much the payload is entitled to claim about a circuit hit.

    "confirmed" -- the close is at a band AND the session's own 5-minute series
                   shows the close pinned to the day's extreme in the same
                   direction, which is what a locked book looks like.
    "inferred"  -- the close is at a band and we have intraday data that does
                   NOT corroborate a lock, so the round number may be chance.
    "unverified" -- no intraday data warm yet; band match alone.
    """
    if row.get("bars") is None:
        return "unverified"
    up = row["change_pct"] > 0
    pinned = row.get("close_at_high") if up else row.get("close_at_low")
    return "confirmed" if pinned else "inferred"


# ------------------------------------------------------------- public API ---

def get_movers(scope: str = "allnse",
               min_move_pct: float = DEFAULT_MIN_MOVE,
               limit: int = 20) -> dict:
    """Gainers, losers, cap-segment movers, circuit hits and intraday leaders
    for one scope.

    Raises nse_segments.UnknownScopeError for a bad scope id and
    nse_segments.SegmentsUnavailableError when membership data is missing
    entirely -- the route maps those to 400 / 503.
    """
    limit = max(1, min(int(limit), MAX_ROWS))
    try:
        min_move_pct = abs(float(min_move_pct))
    except (TypeError, ValueError):
        min_move_pct = DEFAULT_MIN_MOVE

    entries = nse_segments.resolve_scope(scope)          # validates the scope
    universe = nse_segments.all_symbols()                # one shared scan
    scope_symbols = {e["yahoo"] for e in entries}
    by_yahoo = {e["yahoo"]: e for e in entries}

    metrics, as_of, cached_chunks, fresh_chunks = _scan(universe)
    # Any chunk that is missing OR merely stale is owed a refresh, so a board
    # left untouched for an hour still catches up instead of freezing.
    if fresh_chunks < len(_chunks(universe)):
        _ensure_warm(universe)

    intraday_by_sym = _read_intraday(sorted(scope_symbols), as_of)

    rows = []
    implausible = []
    for sym in scope_symbols:
        m = metrics.get(sym)
        if m is None:
            continue
        e = by_yahoo[sym]
        # Guard 2 (see module docstring): a >25% cash move is a corporate
        # action in the series, not a session. Recorded, not silently dropped.
        if abs(m["change_pct"]) > _SANITY_MAX_MOVE_PCT:
            implausible.append({"symbol": e["symbol"],
                                "change_pct": m["change_pct"]})
            continue
        row = {
            "symbol": e["symbol"],
            "yahoo": sym,
            "name": e["name"],
            "industry": e["industry"],
            "segment": e["segment"],
            "segment_label": nse_segments.SEGMENT_LABELS.get(e["segment"], "?"),
            "indices": e["indices"],
            "fno": e.get("fno"),
            "intraday_eligible": e.get("intraday"),
            "lot_size": e.get("lot_size"),
            "currency": "INR",
            **m,
        }
        row.update(intraday_by_sym.get(sym) or {})
        row["circuit_band"] = _circuit_band(row["change_pct"])
        rows.append(row)

    gainers = sorted(rows, key=lambda r: r["change_pct"], reverse=True)
    losers = sorted(rows, key=lambda r: r["change_pct"])
    big = [r for r in rows if abs(r["change_pct"]) >= min_move_pct]
    big.sort(key=lambda r: abs(r["change_pct"]), reverse=True)

    advancing = sum(1 for r in rows if r["change_pct"] > 0)
    declining = sum(1 for r in rows if r["change_pct"] < 0)

    priced = len(rows)
    scanned = len(scope_symbols)
    return {
        "scope": scope,
        "scope_label": nse_segments.SCOPES[scope]["label"],
        "scopes": nse_segments.scope_index(),
        "min_move_pct": round(min_move_pct, 2),
        "as_of": as_of,
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # Coverage is reported, never hidden: a top-10 drawn from a tenth of the
        # market is not the same claim as one drawn from all of it.
        "scanned": scanned,
        "priced": priced,
        "coverage_pct": round(priced / scanned * 100, 1) if scanned else 0.0,
        "partial": priced < scanned,
        "warming": _warm_state["running"],
        "warm_progress": {
            "done": _warm_state["done"], "total": _warm_state["total"],
            "phase": _warm_state["phase"],
        } if _warm_state["running"] else None,
        "breadth": {
            "advancing": advancing,
            "declining": declining,
            "unchanged": priced - advancing - declining,
        },
        "gainers": gainers[:limit],
        "losers": losers[:limit],
        "big_movers": big[:MAX_ROWS],
        "big_mover_count": len(big),
        # ---- the three sections -------------------------------------------
        "segments": _segment_sections(rows),
        "circuits": _circuit_sections(rows),
        "intraday": _intraday_section(rows),
        # ---- honesty about what was thrown away ---------------------------
        "excluded_implausible": len(implausible),
        "excluded_examples": sorted(
            implausible, key=lambda r: -abs(r["change_pct"]))[:3],
        "sanity_max_move_pct": _SANITY_MAX_MOVE_PCT,
        "warnings": nse_segments.get_table().get("warnings", []),
    }


def _segment_sections(rows: list[dict]) -> list[dict]:
    """Section 1: top movers within each cap segment.

    Ranked by ABSOLUTE move, because "top movers" in a cap bucket means the
    biggest moves either way -- splitting each bucket into its own gainers and
    losers lists would give five columns of three rows and say less.
    """
    out = []
    for seg in nse_segments.SEGMENT_ORDER:
        bucket = [r for r in rows if r["segment"] == seg]
        if not bucket:
            continue
        bucket.sort(key=lambda r: abs(r["change_pct"]), reverse=True)
        out.append({
            "segment": seg,
            "label": nse_segments.SEGMENT_LABELS.get(seg, seg),
            "priced": len(bucket),
            "advancing": sum(1 for r in bucket if r["change_pct"] > 0),
            "declining": sum(1 for r in bucket if r["change_pct"] < 0),
            "movers": bucket[:SEGMENT_ROWS],
        })
    return out


def _circuit_sections(rows: list[dict]) -> dict:
    """Section 3: names closing at a standard circuit band.

    Sorted by band width first so a 20% lock always outranks a 5% one, then by
    the size of the move.
    """
    hits = [r for r in rows if r.get("circuit_band")]
    for r in hits:
        r["circuit_confidence"] = _circuit_confidence(r)

    # Corroborated hits outrank band-match-only ones at the same band, so the
    # rows we can actually stand behind sit at the top.
    _CONF_RANK = {"confirmed": 2, "unverified": 1, "inferred": 0}

    def rank(r):
        return (r["circuit_band"],
                _CONF_RANK[r["circuit_confidence"]],
                abs(r["change_pct"]))

    upper = sorted([r for r in hits if r["change_pct"] > 0],
                   key=rank, reverse=True)
    lower = sorted([r for r in hits if r["change_pct"] < 0],
                   key=rank, reverse=True)
    return {
        "upper": upper[:MAX_ROWS],
        "lower": lower[:MAX_ROWS],
        "upper_count": len(upper),
        "lower_count": len(lower),
        "bands": list(_CIRCUIT_BANDS),
        "band_tolerance": _BAND_TOLERANCE,
        "confirmed_count": sum(
            1 for r in hits if r["circuit_confidence"] == "confirmed"),
    }


def _intraday_section(rows: list[dict]) -> dict:
    """Section 2: intraday-eligible names ranked by how far they actually
    travelled inside the session.

    Only MIS-eligible names qualify -- an intraday board that lists stocks you
    cannot take an intraday position in is decoration. Names WITH a measured
    5-minute range rank above names without one, so an unenriched row can never
    outrank a measured one on missing data.
    """
    eligible = [r for r in rows if r.get("intraday_eligible")]
    measured = [r for r in eligible if r.get("intraday_range_pct") is not None]
    measured.sort(key=lambda r: r["intraday_range_pct"], reverse=True)
    unmeasured = [r for r in eligible if r.get("intraday_range_pct") is None]
    unmeasured.sort(key=lambda r: abs(r["change_pct"]), reverse=True)

    return {
        "rows": (measured + unmeasured)[:MAX_ROWS],
        "eligible_count": len(eligible),
        "measured_count": len(measured),
        # The UI must not imply a full-universe intraday ranking when only a
        # shortlist carries measured ranges.
        "ranked_by": "intraday_range_pct" if measured else "change_pct",
        "interval": _INTRADAY_INTERVAL,
        "enrich_cap": _INTRADAY_CAP,
    }


def get_scopes() -> dict:
    """Scope picker + segment counts, without pricing anything."""
    table = nse_segments.get_table()
    return {
        "scopes": nse_segments.scope_index(),
        "segment_order": nse_segments.SEGMENT_ORDER,
        "segment_labels": nse_segments.SEGMENT_LABELS,
        "counts": table["counts"],
        "index_counts": table["index_counts"],
        "built_at": table.get("built_at"),
        "warnings": table.get("warnings", []),
    }
