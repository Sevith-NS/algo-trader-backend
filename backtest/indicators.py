"""Indicators, matching TradingView / Pine Script conventions.

Notably: RSI and ATR use Wilder's smoothing (Pine's ta.rma), NOT a simple mean.
Getting this wrong is the single most common reason a Python port disagrees
with the TradingView strategy tester.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    # Pine's ta.ema seeds with an SMA; pandas' adjust=False after a seed matches closely.
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing == Pine's ta.rma."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def stdev(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).std(ddof=0)


def roc(s: pd.Series, n: int) -> pd.Series:
    """Pine's ta.roc - percent change over n bars."""
    prev = s.shift(n)
    return (s - prev) / prev * 100.0


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Pine's ta.rsi."""
    d = close.diff()
    up = rma(d.clip(lower=0.0), n)
    dn = rma((-d).clip(lower=0.0), n)
    rs = up / dn.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(100.0).where(dn.notna(), np.nan)


def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    """Pine's ta.atr (RMA of true range)."""
    return rma(true_range(h, l, c), n)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bbands(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = stdev(close, n)
    return mid - k * sd, mid, mid + k * sd


def adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14):
    """Returns (adx, +di, -di) - Pine's ta.dmi."""
    up_move = h.diff()
    dn_move = -l.diff()
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    tr_n = rma(true_range(h, l, c), n)
    plus_di = 100.0 * rma(pd.Series(plus_dm, index=h.index), n) / tr_n
    minus_di = 100.0 * rma(pd.Series(minus_dm, index=h.index), n) / tr_n
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return rma(dx, n), plus_di, minus_di


def supertrend(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 10, mult: float = 3.0):
    """Returns (supertrend_line, direction) where direction is +1 up / -1 down."""
    a = atr(h, l, c, period)
    hl2 = (h + l) / 2.0
    upper = (hl2 + mult * a).to_numpy()
    lower = (hl2 - mult * a).to_numpy()
    close = c.to_numpy()
    n = len(c)
    fu = np.full(n, np.nan)
    fl = np.full(n, np.nan)
    dirn = np.zeros(n)
    for i in range(1, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue
        fu[i] = upper[i] if (np.isnan(fu[i - 1]) or upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (np.isnan(fl[i - 1]) or lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        prev = dirn[i - 1] if dirn[i - 1] != 0 else 1
        if close[i] > fu[i]:
            dirn[i] = 1
        elif close[i] < fl[i]:
            dirn[i] = -1
        else:
            dirn[i] = prev
    line = np.where(dirn == 1, fl, fu)
    return pd.Series(line, index=c.index), pd.Series(dirn, index=c.index)


def donchian(h: pd.Series, l: pd.Series, n: int = 20):
    return l.rolling(n, min_periods=n).min(), h.rolling(n, min_periods=n).max()


def _series_like(x, like: pd.Series) -> pd.Series:
    """Let a bare number stand in for a series, the way Pine does with
    ta.crossover(rsi, 30). Without this, crossing a fixed level blows up."""
    if isinstance(x, pd.Series):
        return x
    return pd.Series(float(x), index=like.index)


def crossover(a: pd.Series, b) -> pd.Series:
    """Pine's ta.crossover(a, b). b may be a series or a constant."""
    b = _series_like(b, a)
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b) -> pd.Series:
    """Pine's ta.crossunder(a, b). b may be a series or a constant."""
    b = _series_like(b, a)
    return (a < b) & (a.shift(1) >= b.shift(1))


def pivot_high(h: pd.Series, left: int, right: int) -> pd.Series:
    """Pine's ta.pivothigh. The value appears at the CONFIRMATION bar
    (`right` bars after the pivot itself), so using it is not look-ahead."""
    v = h.to_numpy(float)
    n = len(v)
    out = np.full(n, np.nan)
    for i in range(left + right, n):
        p = i - right
        x = v[p]
        if np.isnan(x):
            continue
        if (v[p - left:p] < x).all() and (v[p + 1:i + 1] < x).all():
            out[i] = x
    return pd.Series(out, index=h.index)


def pivot_low(l: pd.Series, left: int, right: int) -> pd.Series:
    """Pine's ta.pivotlow, confirmed `right` bars late."""
    v = l.to_numpy(float)
    n = len(v)
    out = np.full(n, np.nan)
    for i in range(left + right, n):
        p = i - right
        x = v[p]
        if np.isnan(x):
            continue
        if (v[p - left:p] > x).all() and (v[p + 1:i + 1] > x).all():
            out[i] = x
    return pd.Series(out, index=l.index)
