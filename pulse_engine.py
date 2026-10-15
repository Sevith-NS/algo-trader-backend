"""Market pulse for India and the US: headline indices, breadth and sectors.

Indices and sectors come from one batched Yahoo spark request per market
(cached 5 minutes, stale-if-error), using each node's full-day change against
the previous close. Breadth is honest about its universe:
  - India: advancers/decliners across NIFTY 500, from the movers scan, with
    its coverage (a partial scan says so).
  - US: there is no constituent scan, so breadth counts the 11 S&P 500 sector
    ETFs advancing vs declining, and is labelled that way.
Session state uses regular exchange hours on weekdays; exchange holidays are
not modelled, and the response says so.
"""
from datetime import datetime, time, timedelta, timezone

import data_source
import movers_engine
import nse_segments

_TTL = 5 * 60

MARKETS = {
    "in": {
        "label": "India",
        "currency": "INR",
        "tz": timezone(timedelta(hours=5, minutes=30)),
        "hours": (time(9, 15), time(15, 30)),
        "indices": [("^NSEI", "NIFTY 50"), ("^BSESN", "SENSEX"), ("^NSEBANK", "NIFTY BANK"),
                    ("^INDIAVIX", "INDIA VIX")],
        "sectors": [("^CNXIT", "IT"), ("NIFTY_FIN_SERVICE.NS", "Financial services"),
                    ("^CNXAUTO", "Auto"), ("^CNXPHARMA", "Pharma"), ("^CNXFMCG", "FMCG"),
                    ("^CNXMETAL", "Metal"), ("^CNXENERGY", "Energy"), ("^CNXREALTY", "Realty"),
                    ("^CNXPSUBANK", "PSU banks"), ("^CNXINFRA", "Infrastructure"), ("^CNXMEDIA", "Media")],
    },
    "us": {
        "label": "United States",
        "currency": "USD",
        # Eastern time; DST is approximated from the US rule (2nd Sun Mar - 1st Sun Nov).
        "tz": None,
        "hours": (time(9, 30), time(16, 0)),
        "indices": [("^GSPC", "S&P 500"), ("^IXIC", "Nasdaq Composite"), ("^DJI", "Dow Jones"),
                    ("^RUT", "Russell 2000"), ("^VIX", "VIX")],
        "sectors": [("XLK", "Technology"), ("XLF", "Financials"), ("XLV", "Health care"),
                    ("XLY", "Consumer discretionary"), ("XLC", "Communication services"),
                    ("XLI", "Industrials"), ("XLP", "Consumer staples"), ("XLE", "Energy"),
                    ("XLU", "Utilities"), ("XLRE", "Real estate"), ("XLB", "Materials")],
    },
}


def _us_eastern(now_utc: datetime) -> timezone:
    year = now_utc.year
    march = datetime(year, 3, 8, tzinfo=timezone.utc)
    dst_start = march + timedelta(days=(6 - march.weekday()) % 7, hours=7)       # 2nd Sunday, 02:00 EST
    nov = datetime(year, 11, 1, tzinfo=timezone.utc)
    dst_end = nov + timedelta(days=(6 - nov.weekday()) % 7, hours=6)              # 1st Sunday, 02:00 EDT
    return timezone(timedelta(hours=-4 if dst_start <= now_utc < dst_end else -5))


def session(market: str, now_utc: datetime | None = None) -> dict:
    now_utc = now_utc or datetime.now(timezone.utc)
    cfg = MARKETS[market]
    tz = cfg["tz"] or _us_eastern(now_utc)
    local = now_utc.astimezone(tz)
    opens, closes = cfg["hours"]
    is_open = local.weekday() < 5 and opens <= local.time() < closes
    return {"state": "open" if is_open else "closed",
            "hours": f"{opens.strftime('%H:%M')}-{closes.strftime('%H:%M')} local, Mon-Fri",
            "note": "Exchange holidays are not applied."}


def _quotes(pairs: list[tuple[str, str]]) -> tuple[list[dict], list[str]]:
    symbols = [s for s, _ in pairs]
    payload = data_source.get_json_cached(
        key=f"pulse:{','.join(symbols)}", url=data_source._SPARK_URL,
        params={"symbols": ",".join(symbols), "range": "1mo", "interval": "1d"}, fresh_ttl=_TTL) or {}
    rows, missing = [], []
    for symbol, name in pairs:
        node = payload.get(symbol) or {}
        price, change, pct = node.get("fulldayPrice"), node.get("fulldayChange"), node.get("fulldayChangePercent")
        series = movers_engine._series_from_node(node) if node else None
        if price is None and series is not None:
            price = float(series.iloc[-1])
        if pct is None and series is not None:
            prev = float(series.iloc[-2])
            change, pct = price - prev, (price / prev - 1) * 100 if prev else None
        if price is None or pct is None:
            missing.append(symbol)
            continue
        stamps = [t for t in (node.get("timestamp") or []) if t]
        rows.append({"symbol": symbol, "name": name, "price": round(float(price), 2),
                     "change": round(float(change), 2) if change is not None else None,
                     "change_pct": round(float(pct), 2),
                     "spark": [round(float(v), 2) for v in series.iloc[-20:].tolist()] if series is not None else [],
                     "as_of": datetime.fromtimestamp(stamps[-1], timezone.utc).isoformat() if stamps else None})
    return rows, missing


def _breadth_in() -> dict | None:
    try:
        m = movers_engine.get_movers("nifty500", limit=1)
    except (nse_segments.SegmentsUnavailableError, nse_segments.UnknownScopeError):
        return None
    if not m.get("priced"):
        return None
    return {"universe": "NIFTY 500 stocks", **m["breadth"], "priced": m["priced"], "scanned": m["scanned"],
            "coverage_pct": m["coverage_pct"], "partial": m["partial"], "as_of": m["as_of"]}


def _breadth_from(rows: list[dict], universe: str) -> dict:
    adv = sum(1 for r in rows if r["change_pct"] > 0)
    dec = sum(1 for r in rows if r["change_pct"] < 0)
    return {"universe": universe, "advancing": adv, "declining": dec, "unchanged": len(rows) - adv - dec,
            "priced": len(rows), "scanned": len(rows), "coverage_pct": 100.0 if rows else 0.0,
            "partial": False, "as_of": None}


def get_pulse(market: str) -> dict:
    cfg = MARKETS[market]
    indices, missing_i = _quotes(cfg["indices"])
    sectors, missing_s = _quotes(cfg["sectors"])
    if not indices and not sectors:
        return {"error": "Market data is temporarily unavailable. Try again shortly."}
    sectors.sort(key=lambda r: r["change_pct"], reverse=True)
    breadth = _breadth_in() if market == "in" else None
    if breadth is None:
        breadth = _breadth_from(sectors, "S&P 500 sector ETFs" if market == "us" else "NSE sector indices")
    return {"market": market, "label": cfg["label"], "currency": cfg["currency"],
            "session": session(market), "indices": indices, "breadth": breadth, "sectors": sectors,
            "missing": missing_i + missing_s, "source": "Yahoo Finance (delayed)",
            "computed_at": datetime.now(timezone.utc).isoformat()}
