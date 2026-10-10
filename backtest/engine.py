"""Event-driven single-instrument backtester.

Contract with a strategy: it returns a target-position Series in {-1, 0, +1},
computed using only information available at that bar's CLOSE. The engine then
fills at the NEXT bar's OPEN. That one-bar delay is what stops look-ahead bias
from flattering your results - TradingView does the same by default.

Within a bar the order of events is: fill any pending order at the open, then
walk the bar for stop / target / trailing hits using that bar's high and low.
When both a stop and a target sit inside the same bar we assume the stop hit
first. That is pessimistic and deliberate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Trade:
    direction: int          # +1 long, -1 short
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    shares: float
    pnl: float              # currency, net of costs
    ret_pct: float          # return on the capital committed to this trade
    bars_held: int
    exit_reason: str
    mae_pct: float = 0.0    # worst adverse excursion while open
    mfe_pct: float = 0.0    # best favourable excursion while open


@dataclass
class Config:
    initial_capital: float = 100_000.0
    commission_bps: float = 3.0     # per side, on notional
    slippage_bps: float = 2.0       # per side, on price
    size_pct: float = 100.0         # % of equity committed per trade
    allow_long: bool = True
    allow_short: bool = False
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None
    atr_stop_mult: float | None = None   # overrides stop_loss_pct when set
    atr_period: int = 14


@dataclass
class Result:
    config: Config
    equity: pd.Series
    position: pd.Series
    trades: list = field(default_factory=list)
    bars: pd.DataFrame = None
    ticker: str = ""
    interval: str = "1d"
    strategy_name: str = ""

    @property
    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=[f.name for f in Trade.__dataclass_fields__.values()])
        return pd.DataFrame([t.__dict__ for t in self.trades])


def run(bars: pd.DataFrame, target: pd.Series, cfg: Config,
        ticker: str = "", interval: str = "1d", strategy_name: str = "") -> Result:
    df = bars
    n = len(df)
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    idx = df.index

    tgt = target.reindex(idx).fillna(0.0).to_numpy(float)
    tgt = np.clip(np.round(tgt), -1, 1)
    if not cfg.allow_long:
        tgt = np.where(tgt > 0, 0, tgt)
    if not cfg.allow_short:
        tgt = np.where(tgt < 0, 0, tgt)

    atr_arr = None
    if cfg.atr_stop_mult:
        from .indicators import atr as _atr
        atr_arr = _atr(df["High"], df["Low"], df["Close"], cfg.atr_period).to_numpy(float)

    slip = cfg.slippage_bps / 10_000.0
    comm = cfg.commission_bps / 10_000.0
    frac = max(0.0, cfg.size_pct / 100.0)

    cash = cfg.initial_capital
    shares = 0.0
    pos = 0                      # -1 / 0 / +1
    entry_price = np.nan
    entry_i = -1
    entry_equity = cfg.initial_capital
    stop_px = np.nan
    target_px = np.nan
    trail_ref = np.nan           # best price seen since entry
    mae = 0.0
    mfe = 0.0

    equity = np.full(n, cfg.initial_capital, float)
    pos_arr = np.zeros(n, float)
    trades: list[Trade] = []

    def mark(i):
        return cash + shares * c[i]

    def _open_position(i, direction):
        nonlocal cash, shares, pos, entry_price, entry_i, entry_equity
        nonlocal stop_px, target_px, trail_ref, mae, mfe
        eq = cash
        px = o[i] * (1 + slip) if direction > 0 else o[i] * (1 - slip)
        notional = eq * frac
        if notional <= 0 or px <= 0:
            return
        qty = notional / px
        fee = notional * comm
        if fee > cash and direction > 0:
            return
        cash -= fee
        shares = qty * direction
        cash -= qty * px * direction
        pos = direction
        entry_price = px
        entry_i = i
        entry_equity = eq
        mae = 0.0
        mfe = 0.0
        trail_ref = px
        stop_px = np.nan
        target_px = np.nan
        if cfg.atr_stop_mult and atr_arr is not None and not np.isnan(atr_arr[i]):
            d = cfg.atr_stop_mult * atr_arr[i]
            stop_px = px - d if direction > 0 else px + d
        elif cfg.stop_loss_pct:
            s = cfg.stop_loss_pct / 100.0
            stop_px = px * (1 - s) if direction > 0 else px * (1 + s)
        if cfg.take_profit_pct:
            t = cfg.take_profit_pct / 100.0
            target_px = px * (1 + t) if direction > 0 else px * (1 - t)

    def _close_position(i, px_raw, reason):
        nonlocal cash, shares, pos, entry_price, entry_i, stop_px, target_px, trail_ref, mae, mfe
        direction = pos
        px = px_raw * (1 - slip) if direction > 0 else px_raw * (1 + slip)
        px = max(px, 1e-9)
        qty = abs(shares)
        cash += qty * px * direction
        cash -= qty * px * comm
        eq_after = cash
        pnl = eq_after - entry_equity
        committed = entry_equity * frac
        ret_pct = (pnl / committed * 100.0) if committed > 0 else 0.0
        trades.append(Trade(
            direction=direction, entry_time=idx[entry_i], entry_price=float(entry_price),
            exit_time=idx[i], exit_price=float(px), shares=float(qty), pnl=float(pnl),
            ret_pct=float(ret_pct), bars_held=int(i - entry_i), exit_reason=reason,
            mae_pct=float(mae * 100.0), mfe_pct=float(mfe * 100.0),
        ))
        shares = 0.0
        pos = 0
        entry_price = np.nan
        stop_px = np.nan
        target_px = np.nan
        trail_ref = np.nan

    for i in range(1, n):
        want = int(tgt[i - 1])          # signal formed at close of i-1

        # 1. fill pending order at this bar's open
        if want != pos:
            if pos != 0:
                _close_position(i, o[i], "signal")
            if want != 0:
                _open_position(i, want)

        # 2. walk the bar for stop / target / trailing hits
        if pos != 0:
            if pos > 0:
                mae = min(mae, (l[i] - entry_price) / entry_price)
                mfe = max(mfe, (h[i] - entry_price) / entry_price)
            else:
                mae = min(mae, (entry_price - h[i]) / entry_price)
                mfe = max(mfe, (entry_price - l[i]) / entry_price)

            if cfg.trailing_stop_pct:
                t = cfg.trailing_stop_pct / 100.0
                if pos > 0:
                    trail_ref = max(trail_ref, h[i])
                    cand = trail_ref * (1 - t)
                    stop_px = cand if np.isnan(stop_px) else max(stop_px, cand)
                else:
                    trail_ref = min(trail_ref, l[i])
                    cand = trail_ref * (1 + t)
                    stop_px = cand if np.isnan(stop_px) else min(stop_px, cand)

            hit_stop = (not np.isnan(stop_px)) and (l[i] <= stop_px if pos > 0 else h[i] >= stop_px)
            hit_tgt = (not np.isnan(target_px)) and (h[i] >= target_px if pos > 0 else l[i] <= target_px)

            if hit_stop:
                _close_position(i, float(stop_px), "stop")
            elif hit_tgt:
                _close_position(i, float(target_px), "target")

        equity[i] = mark(i)
        pos_arr[i] = pos

    if pos != 0:
        _close_position(n - 1, c[n - 1], "end-of-data")
        equity[n - 1] = cash
        pos_arr[n - 1] = 0

    return Result(
        config=cfg,
        equity=pd.Series(equity, index=idx, name="equity"),
        position=pd.Series(pos_arr, index=idx, name="position"),
        trades=trades, bars=df, ticker=ticker, interval=interval,
        strategy_name=strategy_name,
    )
