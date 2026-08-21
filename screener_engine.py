"""
Discovery engine: horizon-tagged idea cards + prebuilt runnable screens.

Chartink-style scanning, sized for a paper-trading tool that must stay polite
to a free upstream:

  - Universes are small curated lists (~50 symbols). One batched Yahoo spark
    request per 20 symbols (Yahoo's per-request cap) pulls a full year of
    daily closes for a whole universe in at most three calls (via
    data_source.get_json_cached: throttled, disk-cached, stale-if-error).
    No per-symbol chart calls at scan time.
  - Everything cheap derives from those close series: SMAs, RSI, returns,
    z-score, Bollinger %B, realized vol, 20d/52w extremes. Screens that
    genuinely need volume run a second "confirm" stage against full OHLCV —
    capped at 12 candidates so the cold worst case is ~12 throttled upstream
    calls — and a row whose OHLCV cannot be fetched STAYS in the results
    flagged "volume_confirmed": false (degrade visibly, never silently).
  - Each screen is a declarative list of _Cond objects. The SAME objects
    produce the published "conditions" string and evaluate matches, so the UI
    can never advertise a rule the engine doesn't actually run.
  - Endpoint results are memoised in a small 5-minute TTL cache keyed by
    (endpoint, universe, screen); a browsing session's repeat hits are free.
  - Product doctrine: this engine LISTS, it never RANKS. Cards and matches
    are sorted alphabetically — nothing in the output implies "best first".
"""
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import data_source
import nse_segments
import quant_models


# -------------------------------------------------------------- universes ---
#
# Two kinds live in the same registry:
#
#   static  - a curated {symbol: name} map written out below. Used where no
#             keyless constituent feed exists (US caps, commodities, indices).
#   dynamic - "index": <nse_segments index id>, materialised at scan time from
#             the NSE constituent lists nse_segments already fetches and caches
#             for the Movers board. These pick up rebalances on their own, so
#             the Indian universes are never a stale hand-written list.
#
# `group` drives the option groups in the Discover universe picker.

_STATIC_UNIVERSES = {
    "us_large": {
        "label": "US Large Cap",
        "group": "US",
        "currency": "USD",
        "symbols": {
            "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA",
            "GOOGL": "Alphabet", "AMZN": "Amazon", "META": "Meta Platforms",
            "TSLA": "Tesla", "AVGO": "Broadcom", "BRK-B": "Berkshire Hathaway",
            "JPM": "JPMorgan Chase", "V": "Visa", "UNH": "UnitedHealth",
            "XOM": "Exxon Mobil", "LLY": "Eli Lilly", "WMT": "Walmart",
            "MA": "Mastercard", "JNJ": "Johnson & Johnson",
            "PG": "Procter & Gamble", "HD": "Home Depot", "COST": "Costco",
            "ORCL": "Oracle", "MRK": "Merck", "ABBV": "AbbVie",
            "CRM": "Salesforce", "BAC": "Bank of America", "KO": "Coca-Cola",
            "PEP": "PepsiCo", "AMD": "AMD", "NFLX": "Netflix",
            "TMO": "Thermo Fisher", "ADBE": "Adobe", "CSCO": "Cisco",
            "WFC": "Wells Fargo", "MCD": "McDonald's", "QCOM": "Qualcomm",
            "CAT": "Caterpillar", "IBM": "IBM", "GE": "GE Aerospace",
            "INTC": "Intel", "UBER": "Uber", "DIS": "Disney",
            "PLTR": "Palantir", "TXN": "Texas Instruments", "AMGN": "Amgen",
            "CVX": "Chevron", "GS": "Goldman Sachs", "BA": "Boeing",
            "NKE": "Nike",
        },
    },
    "us_mid": {
        # S&P MidCap 400 representatives. Curated rather than fetched: every
        # keyless US constituent feed either needs a crumb or rate-limits to
        # uselessness, so this list is maintained by hand like us_large.
        "label": "US Mid Cap",
        "group": "US",
        "currency": "USD",
        "symbols": {
            "DKS": "Dick's Sporting Goods", "WSM": "Williams-Sonoma",
            "BURL": "Burlington Stores", "DECK": "Deckers Outdoor",
            "CSL": "Carlisle Companies", "RS": "Reliance Inc",
            "EME": "EMCOR Group", "WWD": "Woodward", "SAIA": "Saia",
            "MANH": "Manhattan Associates", "FIX": "Comfort Systems",
            "EXP": "Eagle Materials", "TOL": "Toll Brothers",
            "OC": "Owens Corning", "AVAV": "AeroVironment",
            "CW": "Curtiss-Wright", "HLI": "Houlihan Lokey",
            "EVR": "Evercore", "JEF": "Jefferies", "SF": "Stifel Financial",
            "PSTG": "Pure Storage", "NTNX": "Nutanix", "CIEN": "Ciena",
            "LSCC": "Lattice Semiconductor", "ONTO": "Onto Innovation",
            "MKSI": "MKS Instruments", "COHR": "Coherent",
            "ALTR": "Altair Engineering", "DT": "Dynatrace",
            "ELS": "Equity Lifestyle", "CUBE": "CubeSmart",
            "EGP": "EastGroup Properties", "STAG": "STAG Industrial",
            "THC": "Tenet Healthcare", "MEDP": "Medpace",
            "CRL": "Charles River Labs", "HALO": "Halozyme",
            "NBIX": "Neurocrine Biosciences", "EXEL": "Exelixis",
            "MUR": "Murphy Oil", "MTDR": "Matador Resources",
            "CHRD": "Chord Energy", "AR": "Antero Resources",
            "GPOR": "Gulfport Energy", "UFPI": "UFP Industries",
            "ATR": "AptarGroup", "AVT": "Avnet", "SLM": "SLM Corp",
            "PRI": "Primerica", "RGA": "Reinsurance Group",
        },
    },
    "us_small": {
        # S&P SmallCap 600 representatives — same curation caveat as us_mid.
        "label": "US Small Cap",
        "group": "US",
        "currency": "USD",
        "symbols": {
            "SPSC": "SPS Commerce", "BMI": "Badger Meter",
            "ITGR": "Integer Holdings", "SXI": "Standex International",
            "MLI": "Mueller Industries", "AWI": "Armstrong World",
            "GFF": "Griffon Corp", "TTMI": "TTM Technologies",
            "POWI": "Power Integrations", "SANM": "Sanmina",
            "PLXS": "Plexus", "FORM": "FormFactor", "DIOD": "Diodes Inc",
            "CALM": "Cal-Maine Foods", "JBSS": "John B. Sanfilippo",
            "ANDE": "Andersons", "HELE": "Helen of Troy",
            "SHOO": "Steven Madden", "BOOT": "Boot Barn",
            "CHEF": "Chefs' Warehouse", "SHAK": "Shake Shack",
            "PLMR": "Palomar Holdings", "SKYW": "SkyWest",
            "MATX": "Matson", "ARCB": "ArcBest", "HUBG": "Hub Group",
            "MGY": "Magnolia Oil & Gas", "CRC": "California Resources",
            "TALO": "Talos Energy", "REX": "REX American Resources",
            "AROC": "Archrock", "HP": "Helmerich & Payne",
            "AXNX": "Axonics", "PRVA": "Privia Health",
            "CORT": "Corcept Therapeutics", "ANIP": "ANI Pharmaceuticals",
            "AMPH": "Amphastar Pharma", "LMAT": "LeMaitre Vascular",
            "UFPT": "UFP Technologies", "CRVL": "CorVel",
            "TRMK": "Trustmark", "WSFS": "WSFS Financial",
            "CATY": "Cathay General", "PFBC": "Preferred Bank",
            "AGO": "Assured Guaranty", "KFY": "Korn Ferry",
            "EXPO": "Exponent", "ICFI": "ICF International",
            "MYRG": "MYR Group", "IESC": "IES Holdings",
        },
    },
    "crypto": {
        "label": "Crypto Majors",
        "group": "Crypto",
        "currency": "USD",
        # 7-day weeks: ~365 daily bars per calendar year, and annualized vol
        # must scale by sqrt(365) — sqrt(252) understates crypto vol ~1.2x.
        "periods_per_year": 365,
        "symbols": {
            "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana",
            "BNB-USD": "BNB", "XRP-USD": "XRP", "ADA-USD": "Cardano",
            "DOGE-USD": "Dogecoin", "AVAX-USD": "Avalanche",
            "LINK-USD": "Chainlink", "DOT-USD": "Polkadot",
            "LTC-USD": "Litecoin", "BCH-USD": "Bitcoin Cash",
        },
    },
    "commodities": {
        "label": "Commodities",
        "group": "Commodities",
        "currency": "USD",
        # Yahoo front-month continuous futures ("=F"), plus a few spot-tracking
        # proxies. requests URL-encodes the "=" in query params and the spark
        # response keys by the raw symbol, so the parse path needs no casing.
        #
        # On XAUUSD / XAGUSD: Yahoo does NOT carry the spot metal FX pairs.
        # XAUUSD=X, XAGUSD=X, XAU=X and GCUSD=X all 404 on both the chart and
        # spark endpoints (verified against the live API), and since Yahoo is
        # this app's only price source, listing them would put two permanently
        # dead rows on the board. The closest instruments Yahoo does serve are
        # carried instead, each labelled as what it actually is:
        #   GC=F / SI=F  - COMEX front-month, what most platforms chart as
        #                  "Gold"/"Silver"; rolls, and carries a small premium
        #                  to spot
        #   PAXG-USD     - 1:1 gold-backed token, trades 24/7 and tracks SPOT
        #                  rather than the futures curve
        #   GLD / SLV    - the liquid ETF trackers (~1/10 oz per share)
        "symbols": {
            "GC=F": "Gold (COMEX front-month)",
            "SI=F": "Silver (COMEX front-month)",
            "PAXG-USD": "Gold spot 24/7 (PAXG)",
            "GLD": "Gold ETF (GLD)", "SLV": "Silver ETF (SLV)",
            "PL=F": "Platinum", "PA=F": "Palladium",
            "HG=F": "Copper", "ALI=F": "Aluminium",
            "CL=F": "WTI Crude", "BZ=F": "Brent Crude",
            "NG=F": "Natural Gas", "RB=F": "RBOB Gasoline",
            "HO=F": "Heating Oil", "ZC=F": "Corn", "ZW=F": "Wheat",
            "ZS=F": "Soybeans", "ZM=F": "Soybean Meal", "ZL=F": "Soybean Oil",
            "KC=F": "Coffee", "SB=F": "Sugar", "CC=F": "Cocoa",
            "CT=F": "Cotton", "LE=F": "Live Cattle", "HE=F": "Lean Hogs",
        },
    },
    "indices": {
        "label": "Global Indices",
        "group": "Global",
        "currency": "USD",
        # Index levels quote in each exchange's home currency — the optional
        # per-universe "currencies" map overrides the default per symbol,
        # resolved through _symbol_currency (the ONE place both discover
        # cards and screen matches consult).
        "currencies": {
            "^NSEI": "INR", "^NSEBANK": "INR", "^BSESN": "INR",
            "^FTSE": "GBP", "^GDAXI": "EUR", "^FCHI": "EUR",
            "^N225": "JPY", "^HSI": "HKD", "^AXJO": "AUD",
        },
        "symbols": {
            "^GSPC": "S&P 500", "^NDX": "Nasdaq 100", "^DJI": "Dow Jones",
            "^RUT": "Russell 2000", "^NSEI": "NIFTY 50",
            "^NSEBANK": "BANK NIFTY", "^BSESN": "SENSEX",
            "^FTSE": "FTSE 100", "^GDAXI": "DAX", "^FCHI": "CAC 40",
            "^N225": "Nikkei 225", "^HSI": "Hang Seng", "^AXJO": "ASX 200",
        },
    },
}


# Indian universes are NOT hand-written. Each names an nse_segments index id;
# membership (and therefore what Discover scans) comes from the NSE constituent
# lists that module already fetches, caches for a day and snapshots to disk for
# the Movers board. A semi-annual rebalance flows through on its own.
#
# Cost note: symbols are scanned 20 at a time (Yahoo's spark cap) at ~1.5s
# between outbound calls, so a COLD smallcap250 scan is ~13 calls / ~20s. Warm
# it is free for 30 minutes. Sizes past ~250 are deliberately not offered here
# — NIFTY 500 and the full NSE tape belong on Movers, which is built to stream
# a partial board while it warms.
_DYNAMIC_UNIVERSES = {
    "nifty50": {"label": "NIFTY 50", "index": "nifty50"},
    "next50": {"label": "NIFTY NEXT 50", "index": "next50"},
    "nifty100": {"label": "NIFTY 100", "index": "nifty100"},
    "midcap150": {"label": "NIFTY MIDCAP 150", "index": "midcap150"},
    "smallcap250": {"label": "NIFTY SMALLCAP 250", "index": "smallcap250"},
}

# Cap segment each dynamic universe reports, for the picker's subtitle. Comes
# from nse_segments' own index -> segment mapping, so it cannot drift from what
# the Movers board calls the same names.
_DYNAMIC_ORDER = list(_DYNAMIC_UNIVERSES)

# Group display order in the picker.
GROUP_ORDER = ["US", "India", "Global", "Crypto", "Commodities"]

_dynamic_cache: dict = {}
_dynamic_lock = threading.Lock()


class UnknownUniverseError(ValueError):
    """Universe id not in the registry — the route maps this to HTTP 400."""


class UnknownScreenError(ValueError):
    """Screen id not in the registry — the route maps this to HTTP 404."""


def _build_dynamic(universe_id: str) -> dict | None:
    """Materialise an NSE index universe, or None when membership is unavailable.

    Returns None rather than raising so a missing NSE feed degrades to "this
    universe is not selectable right now" instead of taking down the whole
    Discover page — the static universes are unaffected by an NSE outage.
    """
    spec = _DYNAMIC_UNIVERSES[universe_id]
    try:
        entries = nse_segments.resolve_scope(spec["index"])
    except (nse_segments.SegmentsUnavailableError,
            nse_segments.UnknownScopeError, Exception):
        return None
    if not entries:
        return None
    return {
        "label": spec["label"],
        "group": "India",
        "currency": "INR",
        # nse_segments already carries the Yahoo symbol form for every member,
        # so Discover keeps using Yahoo as its single price source.
        "symbols": {e["yahoo"]: e["name"] for e in entries},
        "segments": {e["yahoo"]: e.get("segment") for e in entries},
        "dynamic": True,
    }


def _dynamic_universe(universe_id: str) -> dict | None:
    """Cached view of a dynamic universe. nse_segments does its own 24h
    caching and disk snapshotting; this just avoids rebuilding the symbol map
    on every request."""
    with _dynamic_lock:
        if universe_id in _dynamic_cache:
            return _dynamic_cache[universe_id]
    built = _build_dynamic(universe_id)
    with _dynamic_lock:
        # A failed build is cached as None only for this process tick — the
        # next request retries, so a transient NSE blip self-heals.
        if built is not None:
            _dynamic_cache[universe_id] = built
    return built


def all_universe_ids() -> list:
    return list(_STATIC_UNIVERSES) + _DYNAMIC_ORDER


def _resolve_universe(universe_id: str) -> dict:
    uni = _STATIC_UNIVERSES.get(universe_id)
    if uni is not None:
        return uni
    if universe_id in _DYNAMIC_UNIVERSES:
        built = _dynamic_universe(universe_id)
        if built is not None:
            return built
        raise UnknownUniverseError(
            f"Universe '{universe_id}' needs the NSE constituent lists, which "
            f"are unreachable right now. Try a US or global universe.")
    raise UnknownUniverseError(
        f"Unknown universe '{universe_id}'. "
        f"Valid: {', '.join(all_universe_ids())}")


# Back-compat: callers (and tests) that read UNIVERSES[id]["symbols"] directly
# still work for the static ones. Dynamic universes are only reachable through
# _resolve_universe, which is the only path that can build them.
UNIVERSES = _STATIC_UNIVERSES


def _universe_index() -> list:
    """Picker payload: every universe, grouped, with its live symbol count.

    A dynamic universe whose membership cannot be built is listed as
    unavailable rather than hidden — silently dropping NIFTY 50 from the
    dropdown would read as a bug, not as an upstream outage.
    """
    out = []
    for uid, u in _STATIC_UNIVERSES.items():
        out.append({"id": uid, "label": u["label"], "group": u.get("group", "Other"),
                    "count": len(u["symbols"]), "available": True})
    for uid in _DYNAMIC_ORDER:
        built = _dynamic_universe(uid)
        out.append({
            "id": uid,
            "label": _DYNAMIC_UNIVERSES[uid]["label"],
            "group": "India",
            "count": len(built["symbols"]) if built else 0,
            "available": built is not None,
        })
    order = {g: i for i, g in enumerate(GROUP_ORDER)}
    out.sort(key=lambda r: order.get(r["group"], len(GROUP_ORDER)))
    return out


def _symbol_currency(uni: dict, symbol: str) -> str:
    """Per-symbol quote currency. Most universes are single-currency, so the
    universe default is enough — but Global Indices mixes USD/INR/GBP/... in
    one list, so an optional per-universe "currencies" map wins per symbol.
    Both discover cards and screen matches resolve through here."""
    return uni.get("currencies", {}).get(symbol, uni["currency"])


# ----------------------------------------------------- bulk close series ----

_SPARK_CHUNK = 20  # spark hard-caps at 20 symbols per request (400 above it)


def _load_universe_series(universe_id: str):
    """1y of daily closes for every symbol in a universe.

    Returns ({symbol: pd.Series}, as_of) — as_of is the date (YYYY-MM-DD) of
    the newest bar seen anywhere in the data, which is what "how fresh is
    this scan" actually means to the user. Symbols that fail to parse are
    simply skipped; `scanned` in responses stays the attempted universe size
    so a partial upstream outage is visible as scanned > len(results).
    """
    # Sorted so chunk boundaries — and therefore the spark cache keys — stay
    # stable across calls. A dynamic universe reshuffling its dict order would
    # otherwise miss the cache and re-hit the upstream for data we already hold.
    symbols = sorted(_resolve_universe(universe_id)["symbols"])
    series: dict = {}
    latest_ts = None
    for i in range(0, len(symbols), _SPARK_CHUNK):
        chunk = symbols[i:i + _SPARK_CHUNK]
        joined = ",".join(chunk)
        payload = data_source.get_json_cached(
            key=f"spark:1y:{joined}",
            url=data_source._SPARK_URL,
            params={"symbols": joined, "range": "1y", "interval": "1d"},
            fresh_ttl=1800,
        )
        if not payload:
            continue  # never fetched and upstream down — skip whole chunk
        for sym in chunk:
            try:
                node = payload.get(sym) or {}
                stamps = node.get("timestamp") or []
                closes = node.get("close") or []
                pairs = [(t, c) for t, c in zip(stamps, closes) if c is not None]
                if len(pairs) < 2:
                    continue
                idx = pd.to_datetime([p[0] for p in pairs], unit="s")
                s = pd.Series([float(p[1]) for p in pairs], index=idx)
                # During live sessions Yahoo appends a partial "today" print
                # alongside the last close; collapse same-day duplicates to
                # the most recent so 1d metrics never compare today vs today.
                series[sym] = s[~s.index.normalize().duplicated(keep="last")]
                if latest_ts is None or pairs[-1][0] > latest_ts:
                    latest_ts = pairs[-1][0]
            except Exception:
                continue  # one bad symbol must not sink the scan
    if latest_ts is not None:
        as_of = pd.Timestamp(latest_ts, unit="s", tz="UTC").strftime("%Y-%m-%d")
    else:
        # Nothing parsed (total outage, cold cache): keep the contract shape;
        # cards/matches will be empty so the date is cosmetic here.
        as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return series, as_of


# ------------------------------------------------------ per-symbol metrics ---

def _metrics_for(closes: pd.Series, periods_per_year: int = 252) -> dict | None:
    """Everything the tagger and screens need, from closes alone.

    Every key is always present; a metric whose lookback exceeds the available
    history is None, and every condition below fails closed on None — a
    symbol can never match a rule we couldn't actually evaluate.
    `periods_per_year` is 252 for exchange-traded assets, 365 for crypto —
    bar-count windows (20d, 60d) stay bar-based, but anything labeled "1y" or
    annualized must respect the asset's actual calendar.
    """
    n = len(closes)
    if n < 2:
        return None
    last = float(closes.iloc[-1])
    rets = closes.pct_change().dropna()
    span_days = (closes.index[-1] - closes.index[0]).days

    def ret(k):  # % return over the last k bars
        if n <= k:
            return None
        base = float(closes.iloc[-(k + 1)])
        return (last / base - 1) * 100 if base else None

    def sma(p):
        return float(closes.iloc[-p:].mean()) if n >= p else None

    def sma_prev5(p):  # the same SMA five bars ago, for slope/cross checks
        return float(closes.iloc[-(p + 5):-5].mean()) if n >= p + 5 else None

    # "1y return" is a calendar claim, so measure it by calendar: the close
    # at/after (last bar − 365 days). Bar-count shortcuts break on 7-day-week
    # assets (252 crypto bars ≈ 8.3 months). Require a near-complete year of
    # coverage (≥ 300 calendar days) or fail closed.
    ret_1y = None
    if span_days >= 300:
        window = closes[closes.index >= closes.index[-1] - pd.Timedelta(days=365)]
        base = float(window.iloc[0]) if len(window) else None
        ret_1y = (last / base - 1) * 100 if base else None

    sma20, sma50, sma200 = sma(20), sma(50), sma(200)

    rsi = None
    if n >= 15:
        v = quant_models._rsi(closes).iloc[-1]
        if pd.notna(v):
            rsi = float(v)

    std20 = float(closes.iloc[-20:].std()) if n >= 20 else None
    zscore = pct_b = None
    if sma20 is not None and std20 and std20 > 0:
        zscore = (last - sma20) / std20
        lower, upper = sma20 - 2 * std20, sma20 + 2 * std20
        pct_b = (last - lower) / (upper - lower)

    vol_ann = (float(rets.tail(20).std() * np.sqrt(periods_per_year) * 100)
               if len(rets) >= 20 else None)

    # Fresh 20d extremes: today's close vs the PRIOR 20-bar window, so the
    # flag flips only on the breakout bar itself, not every day after it.
    new_20d_high = new_20d_low = dist_20d_low = None
    if n >= 21:
        prior = closes.iloc[-21:-1]
        hi, lo = float(prior.max()), float(prior.min())
        new_20d_high = last >= hi
        new_20d_low = last <= lo
        dist_20d_low = (last / lo - 1) * 100 if lo else None

    # 52-week family: a "52w high" computed over two months of history is a
    # lie, so these need a near-complete calendar year (same bar as ret_1y).
    hi_52w = lo_52w = dist_hi = dist_lo = days_since_hi = None
    if span_days >= 300:
        vals = closes.values
        hi_52w, lo_52w = float(vals.max()), float(vals.min())
        dist_hi = (last / hi_52w - 1) * 100 if hi_52w else None
        dist_lo = (last / lo_52w - 1) * 100 if lo_52w else None
        # most recent touch of the high, not the first one
        days_since_hi = int(n - 1 - np.flatnonzero(vals >= hi_52w - 1e-12)[-1])

    vol_contraction = None  # 5d realized vol as % of 20d realized vol
    if len(rets) >= 20:
        v5, v20 = float(rets.tail(5).std()), float(rets.tail(20).std())
        if v20 > 0:
            vol_contraction = v5 / v20 * 100

    dist_sma20 = (last / sma20 - 1) * 100 if sma20 else None
    dist_sma50 = (last / sma50 - 1) * 100 if sma50 else None
    dist_sma200 = (last / sma200 - 1) * 100 if sma200 else None
    sma20_p5 = sma_prev5(20)
    sma20_slope = ((sma20 / sma20_p5 - 1) * 100
                   if sma20 is not None and sma20_p5 else None)

    return {
        "price": last,
        "change_1d_pct": ret(1),
        "ret_5d_pct": ret(5),
        "ret_20d_pct": ret(20),
        "ret_60d_pct": ret(60),
        "ret_1y_pct": ret_1y,
        "sma20": sma20, "sma50": sma50, "sma200": sma200,
        "sma20_prev5": sma_prev5(20),
        "sma50_prev5": sma_prev5(50),
        "sma200_prev5": sma_prev5(200),
        "rsi_14": rsi,
        "zscore_20d": zscore,
        "pct_b": pct_b,
        "vol_ann_pct": vol_ann,
        "new_20d_high": new_20d_high,
        "new_20d_low": new_20d_low,
        "dist_20d_low_pct": dist_20d_low,
        "high_52w": hi_52w, "low_52w": lo_52w,
        "dist_52w_high_pct": dist_hi,
        "dist_52w_low_pct": dist_lo,
        "days_since_52w_high": days_since_hi,
        "vol_contraction_pct": vol_contraction,
        "dist_sma20_pct": dist_sma20,
        "dist_sma50_pct": dist_sma50,
        "dist_sma200_pct": dist_sma200,
        "sma20_slope_pct": sma20_slope,
        "ext_sma20_pct": dist_sma20,  # extension above SMA20 — same number
        "above_sma200": (last > sma200) if sma200 is not None else None,
        # Internal: the session date stage-1 evaluated, so the volume-confirm
        # stage can refuse to mix bars from different days (never published —
        # no screen lists it as a column).
        "last_bar_date": closes.index[-1].date(),
        "spark": [round(float(v), 2 if abs(v) >= 1 else 4)
                  for v in closes.tail(30)],
    }


def _attach_percentiles(metrics: dict):
    """Cross-sectional flags computed over the scanned universe.

    Percentiles over a couple of symbols are noise, so below 3 usable values
    the flags stay False — screens/tags depending on them fail closed.
    """
    vols = [m["vol_ann_pct"] for m in metrics.values() if m["vol_ann_pct"] is not None]
    vol_cut = float(np.percentile(vols, 70)) if len(vols) >= 3 else None
    r60s = [m["ret_60d_pct"] for m in metrics.values() if m["ret_60d_pct"] is not None]
    r60_cut = float(np.percentile(r60s, 90)) if len(r60s) >= 3 else None
    for m in metrics.values():
        m["vol_top30"] = (vol_cut is not None and m["vol_ann_pct"] is not None
                          and m["vol_ann_pct"] >= vol_cut)
        m["ret60_top_decile"] = (r60_cut is not None and m["ret_60d_pct"] is not None
                                 and m["ret_60d_pct"] >= r60_cut)


def _universe_metrics(universe_id: str):
    series, as_of = _load_universe_series(universe_id)
    ppy = _resolve_universe(universe_id).get("periods_per_year", 252)
    metrics = {}
    for sym, closes in series.items():
        m = _metrics_for(closes, periods_per_year=ppy)
        if m is not None:
            metrics[sym] = m
    _attach_percentiles(metrics)
    return metrics, as_of


# ---------------------------------------------------- horizon classifier ----

# vol_top30 is a pure within-universe percentile, so in a sleepy universe
# (bond ETFs at ~7% ann vol) the "top 30%" flag fires on symbols nobody could
# plausibly day-trade. The relative flag only PROMOTES symbols that also clear
# this absolute day-trading-plausible bar.
_INTRADAY_VOL_FLOOR = 20.0  # % annualized


def _classify(m: dict) -> list:
    """Deterministic horizon tags (Epic H3). Each reason carries the actual
    numbers that fired — a tag the user can't audit is a tag they can't trust.
    """
    tags = []

    # Long term: established uptrend, meaningful 1y gain.
    if (m["sma200"] is not None and m["sma50"] is not None
            and m["ret_1y_pct"] is not None
            and m["price"] > m["sma200"] and m["sma50"] > m["sma200"]
            and m["ret_1y_pct"] >= 12):
        tags.append({
            "id": "long_term", "label": "Long Term",
            "reason": f"{m['ret_1y_pct']:+.0f}% 1y · above 200-DMA · 50>200 DMA",
        })

    # Swing: uptrend intact, RSI cooled off, price resting near a fast MA.
    if (m["sma200"] is not None and m["rsi_14"] is not None
            and m["price"] > m["sma200"] and 35 <= m["rsi_14"] <= 55):
        near = None  # closest MA within 3%
        for key, label in (("dist_sma20_pct", "20"), ("dist_sma50_pct", "50")):
            d = m[key]
            if d is not None and abs(d) <= 3 and (near is None or abs(d) < abs(near[0])):
                near = (d, label)
        if near:
            tags.append({
                "id": "swing", "label": "Swing",
                "reason": (f"pullback in uptrend · RSI {m['rsi_14']:.0f} · "
                           f"{abs(near[0]):.1f}% from {near[1]}-DMA"),
            })

    # Short term: momentum burst or a fresh breakout with healthy RSI.
    parts = []
    if m["ret_5d_pct"] is not None and m["ret_5d_pct"] >= 4:
        parts.append(f"{m['ret_5d_pct']:+.1f}% 5d")
    if (m["new_20d_high"] and m["rsi_14"] is not None
            and 55 <= m["rsi_14"] <= 80):
        parts.append("new 20d high")
    if parts:
        if m["rsi_14"] is not None:
            parts.append(f"RSI {m['rsi_14']:.0f}")
        tags.append({"id": "short_term", "label": "Short Term",
                     "reason": " · ".join(parts)})

    # Intraday: enough realized movement for day-trading to be plausible.
    # (.get: the cross-sectional flag comes from _attach_percentiles, not
    # _metrics_for — absence must read as False, not crash.)
    parts = []
    if (m.get("vol_top30") and m["vol_ann_pct"] is not None
            and m["vol_ann_pct"] >= _INTRADAY_VOL_FLOOR):
        parts.append(f"{m['vol_ann_pct']:.0f}% ann vol (top 30% of universe)")
    if m["change_1d_pct"] is not None and abs(m["change_1d_pct"]) >= 2.5:
        parts.append(f"{m['change_1d_pct']:+.1f}% today")
    if parts:
        tags.append({"id": "intraday", "label": "Intraday",
                     "reason": " · ".join(parts)})

    return tags


# --------------------------------------------------------- screen registry ---

class _Cond:
    """One screen condition: `label` is published verbatim (labels join with
    " · " to form the screen's "conditions" string) and `test` is what the
    engine runs — one object, one source of truth. `ok` traps the TypeError
    a None metric raises in a comparison, so every condition fails closed
    without each lambda needing its own None guards.
    """
    __slots__ = ("label", "test")

    def __init__(self, label, test):
        self.label = label
        self.test = test

    def ok(self, m) -> bool:
        try:
            return bool(self.test(m))
        except (TypeError, KeyError):
            return False


CATEGORIES = ["Breakout", "Trend", "Mean Reversion", "Momentum",
              "Volume & Volatility", "Short-Side"]

_SCREENS = [
    {
        "id": "short_term_breakout", "name": "Short-Term Breakouts",
        "category": "Breakout", "direction": "long", "horizon": "short_term",
        "description": "Fresh 20-day closing highs inside an uptrend, with volume confirmation.",
        "conditions": [
            _Cond("New 20d closing high", lambda m: m["new_20d_high"] is True),
            _Cond("Close > SMA20 > SMA50", lambda m: m["price"] > m["sma20"] > m["sma50"]),
            _Cond("RSI(14) 55–80", lambda m: 55 <= m["rsi_14"] <= 80),
        ],
        "confirm": [
            _Cond("RVOL ≥ 1.25", lambda c: c["rvol"] >= 1.25),
        ],
        "columns": [("rsi_14", "RSI(14)"), ("rvol", "RVOL"),
                    ("dist_sma20_pct", "vs SMA20 %")],
    },
    {
        "id": "near_52w_high", "name": "52-Week High Break",
        "category": "Breakout", "direction": "long", "horizon": "swing",
        "description": "Names pressing within 2% of their 52-week high without being overheated.",
        "conditions": [
            _Cond("Close ≥ 98% of 52-week high", lambda m: m["dist_52w_high_pct"] >= -2.0),
            _Cond("RSI(14) ≤ 80", lambda m: m["rsi_14"] <= 80),
        ],
        "confirm": [],
        "columns": [("dist_52w_high_pct", "vs 52w High %"), ("rsi_14", "RSI(14)")],
    },
    {
        "id": "golden_cross", "name": "Golden Cross (fresh)",
        "category": "Trend", "direction": "long", "horizon": "long_term",
        "description": "The 50-day average just crossed above the 200-day — the classic trend-start signal.",
        "conditions": [
            _Cond("SMA50 crossed above SMA200 within 5 bars",
                  lambda m: m["sma50"] > m["sma200"] and m["sma50_prev5"] <= m["sma200_prev5"]),
            _Cond("Close > SMA50", lambda m: m["price"] > m["sma50"]),
        ],
        "confirm": [],
        "columns": [("sma50", "SMA50"), ("sma200", "SMA200")],
    },
    {
        "id": "uptrend_pullback", "name": "Pullback to the Moving Averages",
        "category": "Trend", "direction": "long", "horizon": "swing",
        "description": "Uptrending names resting back onto their 20/50-day averages with a cooled RSI.",
        "conditions": [
            _Cond("Close > SMA200", lambda m: m["price"] > m["sma200"]),
            _Cond("SMA50 > SMA200", lambda m: m["sma50"] > m["sma200"]),
            _Cond("Within 2% of SMA20 or SMA50",
                  lambda m: ((m["dist_sma20_pct"] is not None and abs(m["dist_sma20_pct"]) <= 2)
                             or (m["dist_sma50_pct"] is not None and abs(m["dist_sma50_pct"]) <= 2))),
            _Cond("RSI(14) 35–55", lambda m: 35 <= m["rsi_14"] <= 55),
        ],
        "confirm": [],
        # Both branches of the OR publish their value — a row matching via
        # SMA20 must never look like it violates the published rule.
        "columns": [("rsi_14", "RSI(14)"), ("dist_sma20_pct", "vs SMA20 %"),
                    ("dist_sma50_pct", "vs SMA50 %")],
    },
    {
        "id": "oversold_uptrend", "name": "Oversold in an Uptrend",
        "category": "Mean Reversion", "direction": "long", "horizon": "swing",
        "description": "RSI washed out below 35 while the long-term uptrend is still intact.",
        "conditions": [
            _Cond("RSI(14) < 35", lambda m: m["rsi_14"] < 35),
            _Cond("Close > SMA200", lambda m: m["price"] > m["sma200"]),
        ],
        "confirm": [],
        "columns": [("rsi_14", "RSI(14)"), ("dist_sma200_pct", "vs SMA200 %"),
                    ("ret_5d_pct", "5D %")],
    },
    {
        "id": "deep_reversion", "name": "Statistical Dip",
        "category": "Mean Reversion", "direction": "long", "horizon": "short_term",
        "description": "Two-sigma dips below the 20-day mean while the short-term trend still points up.",
        "conditions": [
            _Cond("Z-score(20d) < -2", lambda m: m["zscore_20d"] < -2),
            _Cond("Bollinger %B < 0.10", lambda m: m["pct_b"] < 0.1),
            _Cond("SMA20 slope ≥ 0", lambda m: m["sma20"] >= m["sma20_prev5"]),
        ],
        "confirm": [],
        "columns": [("zscore_20d", "Z-Score (20d)"), ("pct_b", "%B"),
                    ("sma20_slope_pct", "SMA20 Slope %")],
    },
    {
        "id": "momentum_leaders", "name": "Momentum Leaders",
        "category": "Momentum", "direction": "long", "horizon": "long_term",
        "description": "Top-decile 60-day performers still trading with healthy, not euphoric, momentum.",
        "conditions": [
            # ret60_top_decile is relative: in a universe where everything is
            # down, the least-bad name is still "top decile" — a leader must
            # also have actually gone up.
            _Cond("60d return in top decile of universe (and positive)",
                  lambda m: m["ret60_top_decile"] and m["ret_60d_pct"] > 0),
            _Cond("RSI(14) 55–78", lambda m: 55 <= m["rsi_14"] <= 78),
        ],
        "confirm": [],
        "columns": [("ret_60d_pct", "60D %"), ("rsi_14", "RSI(14)")],
    },
    {
        "id": "short_term_surge", "name": "5-Day Surge",
        "category": "Momentum", "direction": "long", "horizon": "short_term",
        "description": "Up 5% or more over five sessions and still pushing higher today.",
        "conditions": [
            _Cond("5d return ≥ 5%", lambda m: m["ret_5d_pct"] >= 5),
            _Cond("Up ≥ 1% today", lambda m: m["change_1d_pct"] >= 1),
        ],
        "confirm": [],
        # The "up >= 1% today" value lives in the results table's fixed 1D %
        # column — repeating it here would render the same number twice.
        "columns": [("ret_5d_pct", "5D %"), ("rsi_14", "RSI(14)")],
    },
    {
        "id": "volume_thrust", "name": "Volume Thrust",
        "category": "Volume & Volatility", "direction": "long", "horizon": "intraday",
        "description": "A big up day on at least twice normal volume, closing near the high.",
        "conditions": [
            _Cond("Up ≥ 1.5% today", lambda m: m["change_1d_pct"] >= 1.5),
        ],
        "confirm": [
            _Cond("RVOL ≥ 2.0", lambda c: c["rvol"] >= 2.0),
            _Cond("Close in top 25% of day's range", lambda c: c["range_pos"] >= 0.75),
        ],
        "columns": [("rvol", "RVOL"), ("range_pos", "Range Pos")],
    },
    {
        "id": "high_tight_range", "name": "High & Tight (contraction)",
        "category": "Volume & Volatility", "direction": "long", "horizon": "swing",
        "description": "Recent 52-week highs coiling into a low-volatility contraction.",
        "conditions": [
            _Cond("52-week high within last 10 bars", lambda m: m["days_since_52w_high"] <= 10),
            _Cond("5d realized vol ≤ 60% of 20d vol", lambda m: m["vol_contraction_pct"] <= 60),
        ],
        "confirm": [],
        "columns": [("days_since_52w_high", "Days Since 52w High"),
                    ("vol_contraction_pct", "5d/20d Vol %")],
    },
    {
        "id": "overbought_extension", "name": "Overbought Extension",
        "category": "Short-Side", "direction": "short", "horizon": "short_term",
        "description": "Stretched names — RSI above 78 or price 10%+ above the 20-day average.",
        "conditions": [
            _Cond("RSI(14) > 78 or Close ≥ 10% above SMA20",
                  lambda m: ((m["rsi_14"] is not None and m["rsi_14"] > 78)
                             or (m["ext_sma20_pct"] is not None and m["ext_sma20_pct"] >= 10))),
        ],
        "confirm": [],
        "columns": [("rsi_14", "RSI(14)"), ("ext_sma20_pct", "vs SMA20 %")],
    },
    {
        "id": "breakdown_20d", "name": "20-Day Breakdown",
        "category": "Short-Side", "direction": "short", "horizon": "short_term",
        "description": "Fresh 20-day closing lows under the 50-day average with a weak RSI.",
        "conditions": [
            _Cond("New 20d closing low", lambda m: m["new_20d_low"] is True),
            _Cond("Close < SMA50", lambda m: m["price"] < m["sma50"]),
            _Cond("RSI(14) < 45", lambda m: m["rsi_14"] < 45),
        ],
        "confirm": [],
        "columns": [("rsi_14", "RSI(14)"), ("dist_20d_low_pct", "vs 20d Low %"),
                    ("dist_sma50_pct", "vs SMA50 %")],
    },
]

_SCREEN_INDEX = {s["id"]: s for s in _SCREENS}


def _screen_public(s: dict) -> dict:
    """Published shape of a screen. The conditions string is derived from the
    same _Cond objects the evaluator runs — stage-1 labels first, then the
    volume-confirm labels."""
    return {
        "id": s["id"],
        "name": s["name"],
        "category": s["category"],
        "description": s["description"],
        "conditions": " · ".join(c.label for c in s["conditions"] + s["confirm"]),
        "direction": s["direction"],
        "horizon": s["horizon"],
        "volume_confirmed": bool(s["confirm"]),
    }


# ------------------------------------------------------ volume confirmation ---

_CONFIRM_CAP = 12  # max OHLCV fetches per run — bounds the cold worst case


def _confirm_values(symbol: str, expected_date=None) -> dict | None:
    """Last-bar RVOL + position-in-range from cached OHLCV, or None when the
    confirm data is unavailable (the caller keeps the row and flags it).

    The OHLCV cache (hist:{sym}) and the spark cache stage 1 scanned live
    under different keys with independent fetch times and stale windows, so
    the two "last bars" can be different sessions. Confirming Friday's move
    with Monday's half-formed volume would be a silent lie — on a session
    mismatch we return None and let the row degrade to volume_confirmed:false.
    """
    try:
        df = data_source.get_history(symbol, "1y")
        if df is None or df.empty or len(df) < 21:
            return None
        if expected_date is not None:
            idx = df.index[-1]
            bar_date = (idx.tz_localize(None) if idx.tzinfo is not None else idx).date()
            if bar_date != expected_date:
                return None
        last = df.iloc[-1]
        base_vol = float(df["Volume"].iloc[-21:-1].mean())
        last_vol = float(last["Volume"])
        if base_vol <= 0 or last_vol <= 0:
            return None  # volume feed missing/zero: cannot confirm honestly
        rng = float(last["High"]) - float(last["Low"])
        return {
            "rvol": last_vol / base_vol,
            # a zero-range bar means close == high — top of range by definition
            "range_pos": ((float(last["Close"]) - float(last["Low"])) / rng
                          if rng > 0 else 1.0),
        }
    except Exception:
        return None


# ----------------------------------------------------------- result cache ---
# Endpoint-level memoisation on top of data_source's network caches: the
# expensive part of a scan is metrics over ~50 series, and Discovery pages
# re-request on every tab switch. Key space is tiny (universes x screens),
# so no pruning needed.

_RESULT_TTL = 300.0  # seconds
_results: dict = {}
_results_lock = threading.Lock()


def _result_get(key):
    with _results_lock:
        entry = _results.get(key)
        if entry and time.time() - entry["ts"] <= _RESULT_TTL:
            return entry["data"]
    return None


def _result_put(key, data):
    with _results_lock:
        _results[key] = {"ts": time.time(), "data": data}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _freshness(as_of: str) -> dict:
    """How old the newest bar in a scan actually is.

    data_source serves stale cache for up to 7 days rather than nothing when
    upstream is unreachable, which is the right call — but it means a scan can
    quietly keep returning the same names for days with no visible sign that
    the data stopped moving. This is that sign. Weekends and holidays are not
    subtracted: 3 calendar days old over a weekend is genuinely still Friday's
    close, and pretending otherwise is how the staleness stays invisible.
    """
    try:
        bar = datetime.strptime(as_of, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return {"as_of_age_days": None, "stale": False}
    age = (datetime.now(timezone.utc) - bar).days
    return {
        "as_of_age_days": max(age, 0),
        # Two days covers a normal weekend gap; past that the feed is stuck.
        "stale": age > 2,
    }


# -------------------------------------------------------------- rendering ---

# Rounding per output key: prices 2dp, percentages 1-2dp, ratios 2dp,
# day counts as integers. Anything unlisted gets 2dp.
_VALUE_DP = {
    "rsi_14": 1, "ret_60d_pct": 1, "dist_52w_high_pct": 1, "ext_sma20_pct": 1,
    "vol_contraction_pct": 0, "days_since_52w_high": 0,
}


def _fmt_value(key: str, v):
    """Round for output; None (or non-finite) passes through as None so the
    UI can render an em-dash instead of a fake number."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    v = float(v)
    if not np.isfinite(v):
        return None
    dp = _VALUE_DP.get(key, 2)
    return int(round(v)) if dp == 0 else round(v, dp)


def _round1(v):
    return None if v is None else round(float(v), 1)


# ------------------------------------------------------------- public API ---

def get_discover(universe_id: str = "us_large",
                 include_untagged: bool = False) -> dict:
    """Horizon-tagged idea cards for one universe. Alphabetical, never ranked.

    include_untagged=True (the Markets hub's ?all=1) emits a card for EVERY
    parsed symbol — tags may be [] — so a heatmap/movers view can cover the
    whole universe. Default keeps /discover's tagged-only contract.
    """
    uni = _resolve_universe(universe_id)
    key = ("discover", universe_id, include_untagged)
    cached = _result_get(key)
    if cached is not None:
        return cached

    metrics, as_of = _universe_metrics(universe_id)
    cards = []
    for sym in sorted(uni["symbols"]):
        m = metrics.get(sym)
        if m is None:
            continue
        tags = _classify(m)
        if not tags and not include_untagged:
            continue  # only symbols with at least one horizon tag get a card
        cards.append({
            "symbol": sym,
            "name": uni["symbols"][sym],
            "price": _fmt_value("price", m["price"]),
            "change_1d_pct": _fmt_value("change_1d_pct", m["change_1d_pct"]),
            "currency": _symbol_currency(uni, sym),
            "tags": tags,
            "spark": m["spark"],
            "metrics": {
                "rsi_14": _round1(m["rsi_14"]),
                "ret_5d_pct": _round1(m["ret_5d_pct"]),
                "ret_1y_pct": _round1(m["ret_1y_pct"]),
                "vol_ann_pct": _round1(m["vol_ann_pct"]),
                "above_sma200": m["above_sma200"],
            },
        })

    payload = {
        "universe": universe_id,
        "universe_label": uni["label"],
        "currency": uni["currency"],
        "universes": _universe_index(),
        "as_of": as_of,
        **_freshness(as_of),
        "computed_at": _utc_now(),
        "scanned": len(uni["symbols"]),
        "parsed": len(metrics),  # 0 == upstream outage; routes send no-store
        "cards": cards,
    }
    # A scan that parsed zero series is an upstream outage, not an answer —
    # memoising it would pin an empty page for 5 minutes after recovery.
    if metrics:
        _result_put(key, payload)
    return payload


def list_screens() -> dict:
    """The screen catalogue. Pure registry read — no network, no cache."""
    return {
        "screens": [_screen_public(s) for s in _SCREENS],
        "categories": list(CATEGORIES),
    }


def run_screen(screen_id: str, universe_id: str = "us_large") -> dict:
    """Evaluate one screen over one universe. Matches are alphabetical."""
    screen = _SCREEN_INDEX.get(screen_id)
    if screen is None:
        raise UnknownScreenError(f"Unknown screen '{screen_id}'.")
    uni = _resolve_universe(universe_id)
    key = ("run", screen_id, universe_id)
    cached = _result_get(key)
    if cached is not None:
        return cached

    metrics, as_of = _universe_metrics(universe_id)

    # Stage 1: close-only conditions over the whole universe (no extra I/O).
    stage1 = [(sym, m) for sym, m in metrics.items()
              if all(c.ok(m) for c in screen["conditions"])]

    # Stage 2 (volume_confirm screens only): OHLCV confirmation, capped. The
    # sort here is internal plumbing to decide who gets a confirm slot when
    # candidates exceed the cap — the response itself stays alphabetical.
    rows = []
    if screen["confirm"]:
        stage1.sort(key=lambda r: (r[1]["change_1d_pct"] or 0.0), reverse=True)
        for sym, m in stage1[:_CONFIRM_CAP]:
            conf = _confirm_values(sym, m.get("last_bar_date"))
            if conf is None:
                # OHLCV unavailable: keep the row, degrade visibly
                rows.append((sym, m, {"rvol": None, "range_pos": None}, False))
            elif all(c.ok(conf) for c in screen["confirm"]):
                rows.append((sym, m, conf, True))
            # else: failed the volume condition on real data — dropped
    else:
        rows = [(sym, m, {}, True) for sym, m in stage1]

    matches = []
    for sym, m, conf, confirmed in sorted(rows, key=lambda r: r[0]):
        values = {k: _fmt_value(k, conf[k] if k in conf else m.get(k))
                  for k, _label in screen["columns"]}
        matches.append({
            "symbol": sym,
            "name": uni["symbols"][sym],
            "price": _fmt_value("price", m["price"]),
            "change_1d_pct": _fmt_value("change_1d_pct", m["change_1d_pct"]),
            "currency": _symbol_currency(uni, sym),
            "spark": m["spark"],
            "values": values,
            "volume_confirmed": confirmed,
        })

    payload = {
        "screen": _screen_public(screen),
        "universe": universe_id,
        "universe_label": uni["label"],
        "currency": uni["currency"],
        "as_of": as_of,
        **_freshness(as_of),
        "computed_at": _utc_now(),
        "scanned": len(uni["symbols"]),
        "parsed": len(metrics),  # 0 == upstream outage; routes send no-store
        "matched": len(matches),
        "columns": [{"key": k, "label": label} for k, label in screen["columns"]],
        "matches": matches,
    }
    if metrics:  # same outage rule as get_discover: never memoise a blank scan
        _result_put(key, payload)
    return payload
