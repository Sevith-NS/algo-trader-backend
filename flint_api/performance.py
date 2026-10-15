"""Account performance: a daily value series, time-weighted return (TWR),
money-weighted return (XIRR) and a benchmark over the same dates.

Paper accounts are valued by replaying the fill ledger against daily closes.
Real accounts have no ledger (Kite has no trade history API), so they use the
daily snapshots written on each sync; money in and out is inferred from the
change in cost basis, which understates realised gains on days with sales.
Both limits are returned in `method` so the UI can say so.
"""
from datetime import date, datetime, timedelta, timezone

import pandas as pd

import data_source

PERIODS = ("1m", "3m", "6m", "1y", "ytd", "all")
_DAYS = {"1m": 31, "3m": 92, "6m": 183, "1y": 366}
BENCHMARKS = {"INR": ("^NSEI", "NIFTY 50"), "USD": ("^GSPC", "S&P 500")}
_DEFAULT_BENCHMARK = ("^GSPC", "S&P 500")
_MIN_XIRR_DAYS = 30
# Exchanges that quote history in minor units: (major currency, multiplier).
_MINOR_UNITS = {"GBp": ("GBP", 0.01), "GBX": ("GBP", 0.01), "ZAc": ("ZAR", 0.01), "ILA": ("ILS", 0.01)}


# ------------------------------------------------------------------- maths --

def twr(values: list[float], flows: list[float]) -> float | None:
    """Chain-linked daily return. flows[i] is money added on day i (end of day)."""
    growth = 1.0
    for i in range(1, len(values)):
        prev = values[i - 1]
        if prev <= 0:
            continue
        growth *= (values[i] - flows[i]) / prev
    return (growth - 1.0) * 100 if len(values) >= 2 else None


def xirr(cashflows: list[tuple[date, float]]) -> float | None:
    """Annualised money-weighted return, in percent. Investor's view: money put
    in is negative, value taken out (or held at the end) is positive."""
    if len(cashflows) < 2 or not any(c < 0 for _, c in cashflows) or not any(c > 0 for _, c in cashflows):
        return None
    t0 = cashflows[0][0]

    def npv(rate: float) -> float:
        return sum(c / (1 + rate) ** ((d - t0).days / 365.0) for d, c in cashflows)

    lo, hi = -0.9999, 100.0
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-9:
            break
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return mid * 100


def period_start(period: str, created: date, today: date) -> date:
    if period == "ytd":
        start = date(today.year, 1, 1)
    elif period == "all":
        start = created
    else:
        start = today - timedelta(days=_DAYS[period])
    return max(start, created)


def summarise(dates: list[date], values: list[float], flows: list[float]) -> dict:
    """Return measures for a value series that already starts at the period start."""
    growth, index = 1.0, [100.0]
    for i in range(1, len(values)):
        if values[i - 1] > 0:
            growth *= (values[i] - flows[i]) / values[i - 1]
        index.append(round(growth * 100, 4))
    span = (dates[-1] - dates[0]).days if dates else 0
    money_weighted = None
    if span >= _MIN_XIRR_DAYS:
        cashflows = ([(dates[0], -values[0])]
                     + [(d, -f) for d, f in zip(dates[1:], flows[1:]) if f]
                     + [(dates[-1], values[-1])])
        money_weighted = xirr(cashflows)
    time_weighted = twr(values, flows)
    return {
        "twr_pct": round(time_weighted, 4) if time_weighted is not None else None,
        "xirr_pct": round(money_weighted, 4) if money_weighted is not None else None,
        "xirr_note": None if span >= _MIN_XIRR_DAYS else f"XIRR needs at least {_MIN_XIRR_DAYS} days of history.",
        "series": [{"date": d.isoformat(), "value": round(v, 4), "index": ix}
                   for d, v, ix in zip(dates, values, index)],
    }


# -------------------------------------------------------------------- data --

def _history_period(start: date, today: date) -> str:
    days = (today - start).days
    for period, limit in (("1y", 360), ("2y", 725), ("5y", 1820), ("10y", 3645)):
        if days <= limit:
            return period
    return "max"


def _closes(symbol: str, start: date, today: date) -> pd.Series:
    s = data_source.get_close_series(symbol, _history_period(start, today))
    return s.astype(float) if not s.empty else s


def benchmark(currency: str, dates: list[date], today: date) -> dict | None:
    symbol, name = BENCHMARKS.get(currency, _DEFAULT_BENCHMARK)
    if not dates:
        return None
    s = _closes(symbol, dates[0], today)
    if s.empty:
        return None
    grid = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    aligned = s.reindex(s.index.union(grid)).sort_index().ffill().reindex(grid)
    first = aligned.dropna()
    if first.empty:
        return None
    base = float(first.iloc[0])
    points = [{"date": d.isoformat(), "index": round(float(v) / base * 100, 4)}
              for d, v in zip(dates, aligned.tolist()) if pd.notna(v)]
    return {"symbol": symbol, "name": name, "return_pct": round(points[-1]["index"] - 100, 4) if points else None,
            "series": points}


def paper_values(account: dict, fills: list[dict], start: date, today: date) -> tuple[list[date], list[float]]:
    """Daily account value (cash + holdings at close), from `start` to today."""
    base = account["currency"]
    symbols = sorted({f["symbol"] for f in fills})
    closes: dict[str, pd.Series] = {}
    factors: dict[str, float] = {}
    for sym in symbols:
        closes[sym] = _closes(sym, start, today)
        q = data_source.get_quote(sym) or {}
        factors[sym] = _MINOR_UNITS.get(q.get("currency"), (None, 1.0))[1]
    ccys = sorted({f["currency"] for f in fills if f["currency"] != base})
    fx = {c: _closes(f"{c}{base}=X", start, today) for c in ccys}

    days = pd.bdate_range(start, today)
    grid = pd.DatetimeIndex(days)

    def on_grid(s: pd.Series) -> pd.Series:
        if s.empty:
            return pd.Series(index=grid, dtype=float)
        return s.reindex(s.index.union(grid)).sort_index().ffill().reindex(grid)

    closes = {k: on_grid(v) for k, v in closes.items()}
    fx = {k: on_grid(v) for k, v in fx.items()}

    cash = float(account["starting_cash"])
    qty: dict[str, float] = {}
    last_price: dict[str, float] = {}
    last_fx: dict[str, float] = {}
    ordered = sorted(fills, key=lambda f: f["filled_at"])
    currency_of = {f["symbol"]: f["currency"] for f in ordered}
    i = 0
    out_dates, out_values = [], []
    for ts in grid:
        day_end = datetime.combine(ts.date(), datetime.max.time(), tzinfo=timezone.utc)
        while i < len(ordered) and ordered[i]["filled_at"] <= day_end:
            f = ordered[i]
            gross = float(f["quantity"] * f["price"] * f["fx_rate"])
            fee = float(f["fee"] * f["fx_rate"])
            sign = 1 if f["side"] == "buy" else -1
            cash -= sign * gross + fee
            qty[f["symbol"]] = qty.get(f["symbol"], 0.0) + sign * float(f["quantity"])
            last_price[f["symbol"]] = float(f["price"])
            last_fx[f["currency"]] = float(f["fx_rate"])
            i += 1
        value = cash
        for sym, q in qty.items():
            if q == 0:
                continue
            c = closes[sym].get(ts)
            px = float(c) * factors[sym] if c is not None and pd.notna(c) else last_price.get(sym, 0.0)
            ccy = currency_of[sym]
            rate = 1.0
            if ccy != base:
                r = fx[ccy].get(ts)
                rate = float(r) if r is not None and pd.notna(r) else last_fx.get(ccy, 1.0)
            value += q * px * rate
        out_dates.append(ts.date())
        out_values.append(value)
    return out_dates, out_values
