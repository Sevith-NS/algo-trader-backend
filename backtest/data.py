"""Bars for the backtester, drawn from the platform's own market-data layer.

The desktop tester this is ported from called yfinance directly and kept its
own pickle cache (bt/data.py). Here the bars come from `data_source` instead,
so a backtest shares the throttle, the TTL cache and the stale-if-error
fallback that every other endpoint already relies on, rather than opening a
second independent connection to Yahoo from the same process.

One shape difference has to be absorbed: Yahoo's chart API takes a `range`
string, not a start/end pair. An explicit date window is therefore served by
pulling the smallest legal range that still covers it and slicing locally.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import data_source

from .errors import InsufficientData

# Interval -> roughly how many bars a year holds. Used when the loaded window is
# too short to measure the spacing from the data itself.
BARS_PER_YEAR = {
    "1m": 98_280, "2m": 49_140, "5m": 19_656, "15m": 6_552, "30m": 3_276,
    "60m": 1_638, "1h": 1_638, "90m": 1_092,
    "1d": 252, "1wk": 52, "1mo": 12,
}

# What the Terminal is allowed to ask for, in the order a timeframe picker
# shows them. A subset of data_source.VALID_INTERVALS: 2m and 90m are Yahoo
# curiosities nobody backtests on.
VALID_INTERVALS = ("1m", "5m", "15m", "30m", "60m", "1d", "1wk", "1mo")

# Calendar days each range string covers, for choosing the smallest one that
# reaches back to a requested start date.
_RANGE_DAYS = {
    "1d": 1, "5d": 5, "7d": 7, "1mo": 31, "3mo": 93, "6mo": 186,
    "1y": 366, "2y": 731, "5y": 1_827, "10y": 3_653, "max": 10 ** 6,
}

MIN_BARS = 30


@dataclass
class Bars:
    ticker: str
    interval: str
    df: pd.DataFrame
    #: The range string actually served, after clamping to Yahoo's per-interval
    #: ceiling. Surfaced so the UI can say "you asked for 5y, this is 2y".
    range_: str = "max"
    range_clamped: bool = False

    @property
    def start(self):
        return self.df.index[0]

    @property
    def end(self):
        return self.df.index[-1]

    def __len__(self):
        return len(self.df)


def _range_covering(start: pd.Timestamp | None, requested: str | None) -> str:
    """Smallest range string that reaches back to `start`."""
    if start is None:
        return requested or "max"
    days = (pd.Timestamp.now("UTC").tz_convert(None) - start).days
    for name in data_source._RANGE_ORDER:
        if _RANGE_DAYS[name] >= days:
            return name
    return "max"


def _to_df(records: list) -> pd.DataFrame:
    """Chart records -> an OHLCV frame on a tz-naive index.

    Tz-naive throughout: the engines compare a bar's timestamp against a
    benchmark's and against a user's date window, and a tz-aware index makes
    both of those raise rather than compare.
    """
    if not records:
        return pd.DataFrame()
    idx = pd.to_datetime([r["t"] for r in records], utc=True).tz_convert(None)
    df = pd.DataFrame(
        {
            "Open": [float(r["o"]) for r in records],
            "High": [float(r["h"]) for r in records],
            "Low": [float(r["l"]) for r in records],
            "Close": [float(r["c"]) for r in records],
            "Volume": [float(r.get("v") or 0.0) for r in records],
        },
        index=idx,
    )
    df.index.name = "Date"
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]


def load(symbol: str, interval: str = "1d", start=None, end=None,
         range_: str | None = None) -> Bars:
    """Pull OHLCV for one symbol.

    With no `start`/`end`, serves the largest history the interval allows —
    decades on daily, ~2 years on hourly, a month on 5m. With a date window,
    pulls the smallest range that covers it and trims to the window.
    """
    interval = (interval or "1d").lower()
    if interval not in data_source.MAX_RANGE_FOR_INTERVAL:
        raise InsufficientData(
            f'Unsupported interval "{interval}". '
            f'Valid: {", ".join(VALID_INTERVALS)}')

    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None

    wanted = _range_covering(start_ts, range_)
    served = data_source.clamp_range(interval, wanted)

    # "max" on a daily-or-slower interval has to go through get_ohlc_max: the
    # range-string form quietly returns coarsened bars instead of the full
    # series (see data_source.get_ohlc_max). Everything shorter is fine either
    # way, and the range form keeps the payload to what was asked for.
    if served == "max" and interval in data_source.FULL_HISTORY_INTERVALS:
        records = data_source.get_ohlc_max(symbol, interval=interval)
    else:
        records = data_source.get_ohlc(symbol, range_=served, interval=interval)
    df = _to_df(records)

    if df.empty:
        raise InsufficientData(
            f'No bars returned for "{symbol}" at {interval}. Check the symbol — '
            f'Indian equities need a .NS or .BO suffix.')

    # Trim AFTER loading: the fetch is range-based, so the window is a slice.
    if start_ts is not None:
        df = df[df.index >= start_ts]
    if end_ts is not None:
        # An end date names a whole day, not midnight on it.
        df = df[df.index <= end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]

    if len(df) < MIN_BARS:
        raise InsufficientData(
            f"Only {len(df)} usable bars for {symbol} at {interval} — "
            f"{MIN_BARS} is the minimum. Widen the date range or use a "
            f"coarser interval.")

    df.attrs["interval"] = interval
    df.attrs["ticker"] = symbol

    return Bars(ticker=symbol, interval=interval, df=df,
                range_=served, range_clamped=served != wanted)


def bars_per_year(interval: str, index: pd.DatetimeIndex | None = None) -> float:
    """Annualisation factor, measured from the data where there is enough of it."""
    if index is not None and len(index) > 50:
        span_days = (index[-1] - index[0]).total_seconds() / 86_400.0
        if span_days > 30:
            return len(index) / (span_days / 365.25)
    return BARS_PER_YEAR.get(interval, 252)
