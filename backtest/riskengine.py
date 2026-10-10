"""Execution engine for strategies that size by risk and scale out.

bt/engine.py takes a target position in {-1, 0, +1} and commits a fixed slice
of equity. That covers most ideas. It cannot express "risk 1% of equity to a
stop under that swing low, take half off at 1.5R, trail the rest" - the shape
almost every discretionary swing strategy actually has.

This engine takes a Plan instead: per-bar entry flags plus the absolute stop
price that entry implies, and optionally a per-bar trailing-stop candidate.
Position size falls out of the distance to the stop.

    plan = Plan(long=entries, stop_long=swing_low - 1.5 * atr,
                trail_long=chandelier)
    result = simulate(df, plan, Risk(risk_pct=1.0, rr1=2.0), cfg)

Conventions, all deliberately pessimistic:

* Signals are read at a bar's CLOSE and the entry fills at that same close,
  matching Pine's process_orders_on_close=true. The stop and target it implies
  go live on the NEXT bar - daily OHLC cannot resolve intrabar sequencing on
  the signal bar itself.
* A gap through the stop fills at the open, not at the stop price.
* When the stop and the target both sit inside one bar, the stop fills first.
* Commission is charged on both fills, slippage on both prices.
* A new position is only opened while flat.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import engine as E


@dataclass
class Risk:
    """How much to bet and how to get out."""
    risk_pct: float = 1.0          # % of equity between entry and the stop
    max_expo_pct: float = 25.0     # hard cap on position notional, % of equity
    rr1: float = 1.5               # partial target in R; 0 disables the scale-out
    scale_out: float = 0.5         # fraction of the position taken at rr1
    max_hold: int = 0              # bars; 0 = no time stop
    breakeven: bool = True         # pull the stop to entry once rr1 trades
    trail_after_t1: bool = True    # False = trail from the entry bar instead


@dataclass
class Plan:
    """What a strategy hands the engine. Arrays are one value per bar."""
    long: np.ndarray                      # bool: enter long at this bar's close
    short: np.ndarray | None = None
    stop_long: np.ndarray | None = None   # absolute price, for a long entered here
    stop_short: np.ndarray | None = None
    trail_long: np.ndarray | None = None  # per-bar trailing stop candidate
    trail_short: np.ndarray | None = None

    def _arr(self, a, n, fill):
        if a is None:
            return np.full(n, fill)
        return np.asarray(a, float) if fill is np.nan else np.asarray(a, bool)

    def resolve(self, n: int):
        return (self._arr(self.long, n, False), self._arr(self.short, n, False),
                self._arr(self.stop_long, n, np.nan), self._arr(self.stop_short, n, np.nan),
                self._arr(self.trail_long, n, np.nan), self._arr(self.trail_short, n, np.nan))


@dataclass
class _Leg:
    tag: str
    qty: float
    entry_px: float
    entry_i: int
    takes_profit: bool
    mae: float = 0.0
    mfe: float = 0.0


def simulate(df: pd.DataFrame, plan: Plan, risk: Risk, cfg: E.Config,
             ticker: str = "", interval: str = "1d", name: str = "") -> E.Result:
    n = len(df)
    idx = df.index
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    sig_l, sig_s, stop_l, stop_s, trail_l, trail_s = plan.resolve(n)

    slip = cfg.slippage_bps / 10_000.0
    comm = cfg.commission_bps / 10_000.0

    cash = cfg.initial_capital
    shares = 0.0
    side = 0                     # +1 long, -1 short, 0 flat
    legs: list[_Leg] = []
    stop = np.nan
    t1 = np.nan
    t1_hit = False
    entry_i = -1
    avg_px = np.nan

    equity = np.full(n, cfg.initial_capital, float)
    pos_arr = np.zeros(n, float)
    trades: list[E.Trade] = []

    def close_legs(i, raw_px, reason, only_tp=False):
        """Fill the given legs at raw_px (before slippage) and book the trades."""
        nonlocal cash, shares, side, legs, stop, t1, t1_hit, entry_i, avg_px
        px = raw_px * (1 - slip) if side > 0 else raw_px * (1 + slip)
        px = max(px, 1e-9)
        keep: list[_Leg] = []
        for leg in legs:
            if only_tp and not leg.takes_profit:
                keep.append(leg)
                continue
            notional_in = leg.qty * leg.entry_px
            notional_out = leg.qty * px
            gross = (notional_out - notional_in) * side
            fees = (notional_in + notional_out) * comm
            pnl = gross - fees
            cash += notional_out * side - notional_out * comm
            shares -= leg.qty * side
            trades.append(E.Trade(
                direction=side,
                entry_time=idx[leg.entry_i], entry_price=float(leg.entry_px),
                exit_time=idx[i], exit_price=float(px),
                shares=float(leg.qty), pnl=float(pnl),
                ret_pct=float(pnl / notional_in * 100.0) if notional_in else 0.0,
                bars_held=int(i - leg.entry_i), exit_reason=f"{leg.tag}:{reason}",
                mae_pct=float(leg.mae * 100.0), mfe_pct=float(leg.mfe * 100.0),
            ))
        legs = keep
        if not legs:
            side = 0
            shares = 0.0
            stop = t1 = np.nan
            t1_hit = False
            entry_i = -1
            avg_px = np.nan

    for i in range(n):
        # ---- 1. manage an open position with levels set at the last close --
        if side != 0:
            for leg in legs:
                if side > 0:
                    leg.mae = min(leg.mae, (l[i] - leg.entry_px) / leg.entry_px)
                    leg.mfe = max(leg.mfe, (h[i] - leg.entry_px) / leg.entry_px)
                else:
                    leg.mae = min(leg.mae, (leg.entry_px - h[i]) / leg.entry_px)
                    leg.mfe = max(leg.mfe, (leg.entry_px - l[i]) / leg.entry_px)

            reason = "trail" if t1_hit else "stop"
            has_tp = any(x.takes_profit for x in legs)
            if side > 0:
                if o[i] <= stop:
                    close_legs(i, o[i], reason + "-gap")
                elif l[i] <= stop:
                    close_legs(i, stop, reason)
                elif not np.isnan(t1) and has_tp:
                    if o[i] >= t1:
                        close_legs(i, o[i], "target", only_tp=True)
                    elif h[i] >= t1:
                        close_legs(i, t1, "target", only_tp=True)
            else:
                if o[i] >= stop:
                    close_legs(i, o[i], reason + "-gap")
                elif h[i] >= stop:
                    close_legs(i, stop, reason)
                elif not np.isnan(t1) and has_tp:
                    if o[i] <= t1:
                        close_legs(i, o[i], "target", only_tp=True)
                    elif l[i] <= t1:
                        close_legs(i, t1, "target", only_tp=True)

        # ---- 2. re-arm the stop for the next bar ---------------------------
        if side != 0:
            if side > 0:
                if not t1_hit and not np.isnan(t1) and h[i] >= t1:
                    t1_hit = True
                if t1_hit or not risk.trail_after_t1:
                    if risk.breakeven and t1_hit:
                        stop = max(stop, avg_px)
                    if not np.isnan(trail_l[i]):
                        stop = max(stop, trail_l[i])
            else:
                if not t1_hit and not np.isnan(t1) and l[i] <= t1:
                    t1_hit = True
                if t1_hit or not risk.trail_after_t1:
                    if risk.breakeven and t1_hit:
                        stop = min(stop, avg_px)
                    if not np.isnan(trail_s[i]):
                        stop = min(stop, trail_s[i])

            if risk.max_hold > 0 and (i - entry_i) >= risk.max_hold:
                close_legs(i, c[i], "time")

        # ---- 3. entries fill at this close ---------------------------------
        if side == 0 and i < n - 1:
            want = 1 if sig_l[i] else (-1 if sig_s[i] else 0)
            if want != 0:
                eq = cash
                sp = stop_l[i] if want > 0 else stop_s[i]
                px = c[i] * (1 + slip) if want > 0 else c[i] * (1 - slip)
                dist = (c[i] - sp) if want > 0 else (sp - c[i])
                if not np.isnan(dist) and dist > 0:
                    qty = min(math.floor(eq * risk.risk_pct / 100.0 / dist),
                              math.floor(eq * risk.max_expo_pct / 100.0 / px))
                    if qty >= 1:
                        if risk.rr1 > 0:
                            half = max(1.0, round(qty * risk.scale_out))
                            rest = qty - half
                            parts = [(half, True)] + ([(rest, False)] if rest >= 1 else [])
                        else:
                            parts = [(float(qty), False)]
                        for q, tp in parts:
                            cash -= q * px * want + q * px * comm
                            shares += q * want
                            legs.append(_Leg(tag="L1" if tp else "L2", qty=float(q),
                                             entry_px=float(px), entry_i=i,
                                             takes_profit=tp))
                        side = want
                        entry_i = i
                        avg_px = px
                        stop = sp
                        t1 = c[i] + dist * risk.rr1 * want if risk.rr1 > 0 else np.nan

        equity[i] = cash + shares * c[i]
        pos_arr[i] = side

    if side != 0:
        close_legs(n - 1, c[n - 1], "end-of-data")
        equity[n - 1] = cash
        pos_arr[n - 1] = 0

    return E.Result(
        config=cfg,
        equity=pd.Series(equity, index=idx, name="equity"),
        position=pd.Series(pos_arr, index=idx, name="position"),
        trades=trades, bars=df, ticker=ticker, interval=interval,
        strategy_name=name,
    )
