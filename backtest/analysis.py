"""Robustness passes: out-of-sample, walk-forward and Monte Carlo.

A single backtest over all available history answers "would this have worked",
which is the question least worth asking — the parameters were chosen knowing
that history. Everything in this module attacks the same weakness from a
different side:

* out-of-sample   - hold back the tail, fit nothing to it, look once
* walk-forward    - re-optimise on a rolling window, trade the window after it,
                    and stitch those untouched segments into one curve
* Monte Carlo     - keep the trades, shuffle their order, and read the spread
                    of outcomes instead of the one path that happened

Walk-forward is the strongest of the three and the slowest: it re-runs the
whole sweep once per fold.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from . import metrics as M
from .errors import BacktestError


# ------------------------------------------------------------ parameter grid

# Mirrors service.MAX_COMBOS. Checked while lo/hi/step are still scalars, so an
# oversized grid is refused before np.arange or itertools.product allocates it
# (audit finding 3: the old cap ran only after the allocation it guarded).
MAX_POINTS = 2000


def _range_values(lo: float, hi: float, step: float, key: str):
    if not all(math.isfinite(x) for x in (lo, hi, step)):
        raise BacktestError(f'Range for "{key}" must be finite numbers.')
    if step <= 0:
        raise BacktestError(f'Step for "{key}" must be positive.')
    points = math.floor((hi - lo) / step + 0.5) + 1
    if points < 1:
        raise BacktestError(f'Range for "{key}" covers no values.')
    if points > MAX_POINTS:
        raise BacktestError(
            f'Range for "{key}" is {points:,} values; the cap is {MAX_POINTS:,}. Widen the step.')
    return np.arange(lo, hi + step / 2, step)


def parse_grid(spec) -> dict:
    """Accepts the CLI string form or an already-expanded mapping.

    'fast=10:50:10,slow=100:200:50' -> {'fast': [10,20,30,40,50], 'slow': [100,150,200]}
    {'fast': {'min':10,'max':50,'step':10}} and {'fast': [10,20]} also work, so
    the UI can send a structured body without building the string.
    """
    if isinstance(spec, dict):
        grid = {}
        for key, value in spec.items():
            if isinstance(value, (list, tuple)):
                grid[key] = [_num(v) for v in value]
            elif isinstance(value, dict):
                lo, hi = float(value["min"]), float(value["max"])
                step = float(value.get("step") or 1)
                grid[key] = [_num(v) for v in _range_values(lo, hi, step, key)]
            else:
                grid[key] = [_num(value)]
            if not grid[key]:
                raise BacktestError(f'Parameter "{key}" has no values to sweep.')
        return grid

    grid = {}
    for part in str(spec).split(","):
        if "=" not in part:
            raise BacktestError(
                f'Sweep spec "{part}" should look like name=min:max:step.')
        key, rng = part.split("=", 1)
        try:
            bits = [float(x) for x in rng.split(":")]
        except ValueError:
            raise BacktestError(f'Sweep values for "{key}" must be numbers.')
        if len(bits) == 3:
            lo, hi, step = bits
            values = list(_range_values(lo, hi, step, key.strip()))
        else:
            values = bits
        grid[key.strip()] = [_num(v) for v in values]
    return grid


def _num(v):
    f = float(v)
    return int(f) if f.is_integer() else f


def combinations(grid: dict) -> list:
    keys = list(grid)
    total = math.prod(len(grid[k]) for k in keys)
    if total > MAX_POINTS:
        raise BacktestError(
            f"That grid is {total:,} combinations; the cap is {MAX_POINTS:,}. "
            f"Widen the step or sweep fewer parameters.")
    return [dict(zip(keys, combo))
            for combo in itertools.product(*[grid[k] for k in keys])]


# -------------------------------------------------------------- walk-forward

def walk_forward_windows(n_bars: int, folds: int, is_frac: float) -> list:
    """(in-sample slice, out-of-sample slice) index pairs, rolling forward.

    Rolling rather than anchored: an anchored window keeps 1999 in the training
    set forever, and a parameter that only ever worked in 1999 keeps being
    chosen. Each fold here is fitted on the `is_frac` of a window immediately
    before the segment it then trades.
    """
    if not 0.3 <= is_frac <= 0.9:
        raise BacktestError("The in-sample fraction must be between 0.3 and 0.9.")
    if folds < 2:
        raise BacktestError("Walk-forward needs at least 2 folds.")

    # oos * folds + is = n, with is = oos * ratio
    ratio = is_frac / (1.0 - is_frac)
    oos = int(n_bars / (folds + ratio))
    is_bars = int(oos * ratio)
    if oos < 30 or is_bars < 100:
        raise BacktestError(
            f"Not enough history for {folds} walk-forward folds: that leaves "
            f"{is_bars} training and {oos} test bars per fold. Use fewer folds, "
            f"a longer date range or a faster interval.")

    windows = []
    for k in range(folds):
        start = k * oos
        mid = start + is_bars
        end = min(mid + oos, n_bars)
        if end - mid < 20:
            break
        windows.append(((start, mid), (mid, end)))
    return windows


def stitch(curves: list, initial: float) -> pd.Series:
    """Chain out-of-sample equity segments into one continuous curve.

    Each fold restarts at the same nominal capital, so the segments are joined
    on their RETURNS, not their levels — otherwise every fold boundary would
    show a jump back to the starting balance.
    """
    pieces = []
    level = initial
    for curve in curves:
        if curve is None or len(curve) < 2:
            continue
        scaled = curve / float(curve.iloc[0]) * level
        pieces.append(scaled)
        level = float(scaled.iloc[-1])
    if not pieces:
        return pd.Series(dtype=float)
    out = pd.concat(pieces)
    return out[~out.index.duplicated(keep="last")].sort_index()


# --------------------------------------------------------------- Monte Carlo

def trade_returns(result) -> np.ndarray:
    """Per-trade return on the equity that existed when the trade closed.

    Read off the equity curve at each exit rather than from trade P&L, because
    that is the only way position sizing (fixed fraction here, risk-based in
    the Plan/Risk engine) survives into the resample. A trade that risked 1% of
    a doubled account has to resample as 1%, not as its currency P&L.
    """
    trades = result.trades
    if not trades:
        return np.array([])
    equity = result.equity
    marks = [float(equity.iloc[0])]
    for t in trades:
        idx = equity.index.searchsorted(t.exit_time)
        idx = min(max(idx, 0), len(equity) - 1)
        marks.append(float(equity.iloc[idx]))
    marks = np.asarray(marks, float)
    prev, curr = marks[:-1], marks[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, curr / prev - 1.0, 0.0)
    return rets[np.isfinite(rets)]


def monte_carlo(result, stats: dict, paths: int = 1000, ruin_pct: float = 0.5,
                seed: int = 7) -> dict:
    """Resample the trade sequence to get a distribution instead of one path.

    The order trades arrived in is an accident of history: the same edge, dealt
    in a different order, has a different worst drawdown. Bootstrapping with
    replacement keeps each trade's size and outcome and asks what the run of
    luck could have looked like.

    Caveat worth keeping in view: this assumes trades are independent. A
    strategy whose losses cluster in a regime (most trend systems) will have a
    fatter real tail than the resample shows, so read the 5th percentile as
    optimistic, not conservative.
    """
    rets = trade_returns(result)
    if len(rets) < 10:
        return {"paths": 0,
                "note": "Fewer than 10 trades — a resample of this would be noise."}

    rng = np.random.default_rng(seed)
    n = len(rets)
    draws = rng.choice(rets, size=(paths, n), replace=True)

    curves = np.cumprod(1.0 + draws, axis=1)
    finals = curves[:, -1]
    peaks = np.maximum.accumulate(curves, axis=1)
    max_dds = (curves / peaks - 1.0).min(axis=1)

    years = max(float(stats.get("years") or 0.0), 1e-9)
    with np.errstate(invalid="ignore"):
        cagrs = np.where(finals > 0, finals ** (1.0 / years) - 1.0, -1.0)

    def pct(arr, q):
        return float(np.percentile(arr, q))

    return {
        "paths": int(paths),
        "trades_resampled": int(n),
        "final_return": {
            "p5": pct(finals, 5) - 1.0, "p25": pct(finals, 25) - 1.0,
            "median": pct(finals, 50) - 1.0, "p75": pct(finals, 75) - 1.0,
            "p95": pct(finals, 95) - 1.0,
        },
        "cagr": {
            "p5": pct(cagrs, 5), "median": pct(cagrs, 50), "p95": pct(cagrs, 95),
        },
        "max_drawdown": {
            "median": pct(max_dds, 50), "p95": pct(max_dds, 5),  # 5th pct = worst
            "worst": float(max_dds.min()),
        },
        "prob_profit": float((finals > 1.0).mean()),
        # "Ruin" is a drawdown you would not keep trading through, not zero.
        "ruin_threshold": float(ruin_pct),
        "prob_ruin": float((max_dds <= -abs(ruin_pct)).mean()),
        # A fan for the chart: the 5th / 50th / 95th path at each trade number.
        "band": {
            "p5": [float(x) for x in np.percentile(curves, 5, axis=0)],
            "median": [float(x) for x in np.percentile(curves, 50, axis=0)],
            "p95": [float(x) for x in np.percentile(curves, 95, axis=0)],
        },
    }


# ------------------------------------------------------------------ verdicts
# Short, honest read-outs. The desktop tester printed these in the terminal and
# they are the most useful thing it said, so they travel with the JSON.

def oos_verdict(is_stats: dict, oos_stats: dict) -> dict:
    is_cagr, oos_cagr = is_stats["cagr"], oos_stats["cagr"]
    if is_cagr > 0 and oos_cagr < 0:
        return {"level": "bad", "text": "Profitable in-sample, losing out-of-sample. "
                "That is the signature of a curve-fit strategy."}
    if is_cagr > 0 and oos_cagr < is_cagr * 0.4:
        return {"level": "warn", "text": "Out-of-sample return is well under half the "
                "in-sample return. Expect real-world results near the lower figure."}
    if oos_cagr > 0:
        return {"level": "good", "text": "The edge holds up out-of-sample. That is the "
                "single best sign in this whole report."}
    return {"level": "warn", "text": "Negative in both halves — there is no edge here to "
            "over-fit in the first place."}


def sweep_verdict(rows: list) -> dict:
    sharpes = np.array([r["sharpe"] for r in rows if np.isfinite(r["sharpe"])])
    if sharpes.size == 0:
        return {"level": "warn", "text": "No combination produced a usable result."}
    best, median = float(sharpes.max()), float(np.median(sharpes))
    if best > 0 and median < best * 0.3:
        return {"level": "bad", "text": "The best parameters are far above the median. "
                "That peak is almost certainly noise — pick a setting from a broad "
                "plateau, not a lone spike."}
    return {"level": "good", "text": "Performance is stable across the parameter space, "
            "which is what a real edge looks like."}


def walk_forward_verdict(efficiency: float, positive_folds: int, folds: int) -> dict:
    if efficiency <= 0:
        return {"level": "bad", "text": "Every fold lost money once it stopped being "
                "optimised on. The parameters describe the past, not the market."}
    if efficiency < 0.4:
        return {"level": "warn", "text": f"Walk-forward efficiency {efficiency:.0%}: "
                "most of the in-sample return did not survive. Size accordingly."}
    return {"level": "good", "text": f"Walk-forward efficiency {efficiency:.0%} with "
            f"{positive_folds} of {folds} folds profitable — the edge re-optimises "
            "into the future rather than only fitting the past."}
