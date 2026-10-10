"""Request -> backtest -> JSON. The layer the API routes talk to.

Everything the desktop tester's run.py did between parsing argv and printing a
rich table happens here, minus the printing: pick the engine the strategy
registered, build the cost/risk config, run it, and add whichever robustness
passes were asked for.

Sweeps are generators. A 200-combination grid is a minute of work, and a
minute of silence is indistinguishable from a hang, so `optimize` and
`walk_forward` yield progress events the route forwards as SSE.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import analysis as A
from . import data as D
from . import engine as E
from . import metrics as M
from . import riskengine as RE
from . import strategies as S
from . import swingpro as SP
from .errors import BacktestError

# A sweep is bounded work, so it gets a bound. 2000 daily runs over 20 years is
# already a couple of minutes; past that the browser has given up anyway.
MAX_COMBOS = 2000
# Walk-forward runs the whole grid once per fold, so folds multiply the work.
# The ceiling is set by the deployed worker timeout, if any:
# gunicorn counts a streaming response as one long request and will kill the
# worker mid-stream, so the cap has to keep the slowest realistic sweep inside
# it rather than relying on the client to ask for something sensible.
MAX_WALK_FORWARD_RUNS = 4000
MAX_MC_PATHS = 5000
MAX_CURVE_POINTS = 1500
MAX_TRADES_RETURNED = 2000

# Which index a symbol is measured against when the caller does not say. Beta
# against the wrong market is worse than no beta at all.
_INDIA_SUFFIXES = (".NS", ".BO")


def default_benchmark(symbol: str) -> str:
    return "^NSEI" if symbol.upper().endswith(_INDIA_SUFFIXES) else "^GSPC"


# ------------------------------------------------------------------ registry

def list_strategies() -> list:
    """The strategy picker's payload: name, one-line doc, and every parameter."""
    out = []
    for name in sorted(S.REGISTRY):
        spec = S.REGISTRY[name]
        params = []
        for key, value in spec["params"].items():
            params.append({
                "name": key,
                "default": value,
                # The UI renders a checkbox for a 0/1 flag, a number field
                # otherwise; Pine-style booleans arrive as ints, so the type has
                # to be declared rather than inferred from the value.
                "type": ("string" if isinstance(value, str)
                         else "int" if isinstance(value, int) and not isinstance(value, bool)
                         else "float"),
            })
        out.append({
            "name": name,
            "doc": (spec.get("doc") or "").split("\n")[0],
            "engine": spec.get("engine", "target"),
            "params": params,
            "costs": spec.get("costs") or {},
        })
    return out


# ------------------------------------------------------------------- running

def _coerce(value):
    """JSON numbers arrive as float; strategy params are typed by their default."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text) if text.lstrip("-").isdigit() else float(text)
        except ValueError:
            return text
    return value


def build_config(req: dict, strategy: str) -> E.Config:
    """Costs and stops. A strategy that declares its own costs wins the default."""
    own = S.REGISTRY.get(strategy, {}).get("costs", {})

    def num(key, fallback):
        value = req.get(key)
        return float(value) if value is not None else fallback

    short_only = bool(req.get("short_only"))
    return E.Config(
        initial_capital=num("capital", 100_000.0),
        commission_bps=num("commission_bps", own.get("commission_bps", 3.0)),
        slippage_bps=num("slippage_bps", own.get("slippage_bps", 2.0)),
        size_pct=num("size_pct", 100.0),
        allow_long=not short_only,
        allow_short=bool(req.get("allow_short")) or short_only,
        stop_loss_pct=req.get("stop_loss_pct"),
        take_profit_pct=req.get("take_profit_pct"),
        trailing_stop_pct=req.get("trailing_stop_pct"),
        atr_stop_mult=req.get("atr_stop_mult"),
        atr_period=int(req.get("atr_period") or 14),
    )


def run_strategy(df: pd.DataFrame, name: str, overrides: dict, cfg: E.Config,
                 ticker: str, interval: str):
    """One strategy over one frame. Returns (Result, resolved params).

    Most strategies hand back a target-position Series and share engine.py.
    Ones that own their execution — risk sizing, scale-outs, per-leg exits —
    registered an "engine" key and get the whole frame instead.
    """
    spec = S.REGISTRY.get(name)
    if spec is None:
        raise BacktestError(
            f'Unknown strategy "{name}". Available: {", ".join(sorted(S.REGISTRY))}.')
    if spec.get("engine") == "swingpro":
        return SP.run(df, overrides, cfg, ticker, interval, name)
    if spec.get("engine") == "risk":
        _, params = S.resolve(name, overrides)
        plan, risk = spec["fn"](df, **params)
        return RE.simulate(df, plan, risk, cfg, ticker, interval, name), params
    sig, params = S.build(name, df, overrides)
    return E.run(df, sig, cfg, ticker, interval, name), params


# ------------------------------------------------------------- serialisation

def _ts(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _clean(value):
    """JSON has no NaN or Infinity. Anything that is not finite becomes null."""
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return _ts(value)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _curve(series: pd.Series, cap: int = MAX_CURVE_POINTS) -> dict:
    """An equity or drawdown curve, thinned to something a chart can hold.

    Thinned by striding, with the last point always kept: a curve whose final
    value disagreed with the reported final equity would look like a bug in the
    numbers rather than in the sampling.
    """
    if series is None or series.empty:
        return {"points": [], "downsampled": False}
    step = max(1, int(np.ceil(len(series) / cap)))
    sliced = series.iloc[::step]
    if sliced.index[-1] != series.index[-1]:
        sliced = pd.concat([sliced, series.iloc[[-1]]])
    return {
        "points": [{"t": _ts(ts), "v": _clean(float(v))} for ts, v in sliced.items()],
        "downsampled": step > 1,
    }


def _trades(result) -> dict:
    df = result.trades_df
    if df.empty:
        return {"rows": [], "total": 0, "truncated": False}
    rows = []
    for t in result.trades[-MAX_TRADES_RETURNED:]:
        rows.append({
            "direction": int(t.direction),
            "entry_time": _ts(t.entry_time), "entry_price": _clean(t.entry_price),
            "exit_time": _ts(t.exit_time), "exit_price": _clean(t.exit_price),
            "shares": _clean(t.shares), "pnl": _clean(t.pnl),
            "ret_pct": _clean(t.ret_pct), "bars_held": int(t.bars_held),
            "exit_reason": t.exit_reason,
            "mae_pct": _clean(t.mae_pct), "mfe_pct": _clean(t.mfe_pct),
        })
    return {"rows": rows, "total": len(result.trades),
            "truncated": len(result.trades) > MAX_TRADES_RETURNED}


def _monthly(equity: pd.Series) -> list:
    table = M.monthly_table(equity)
    if table.empty:
        return []
    months = [c for c in table.columns if c != "Year"]
    out = []
    for year, row in table.iterrows():
        out.append({
            "year": int(year),
            "months": {m: _clean(row[m]) for m in months},
            "total": _clean(row.get("Year")),
        })
    return out


def _summary(stats: dict) -> dict:
    """compute()'s dict minus the things that are not JSON (the b&h Series)."""
    drop = {"bh_curve", "exit_mix", "start", "end", "dd_start", "dd_end"}
    out = {k: _clean(v) for k, v in stats.items() if k not in drop}
    out.update({
        "start": _ts(stats["start"]), "end": _ts(stats["end"]),
        "dd_start": _ts(stats["dd_start"]), "dd_end": _ts(stats["dd_end"]),
        "exit_mix": {str(k): int(v) for k, v in (stats.get("exit_mix") or {}).items()},
    })
    return out


# ----------------------------------------------------------------- one run

def _benchmark_close(symbol: str, index: pd.DatetimeIndex, interval: str):
    """Benchmark closes carried onto the tested bars, or None if unavailable."""
    if not symbol or symbol.lower() == "none":
        return None
    try:
        bars = D.load(symbol, interval)
    except Exception:
        return None
    close = bars.df["Close"]
    joined = close.reindex(close.index.union(index)).ffill().reindex(index)
    return joined.dropna()


def run(req: dict) -> dict:
    """One backtest, plus whichever robustness passes the request asked for."""
    started = time.time()

    symbol = (req.get("symbol") or "").strip()
    if not symbol:
        raise BacktestError("A symbol is required.")
    strategy = (req.get("strategy") or "ema_cross").strip()
    interval = (req.get("interval") or "1d").strip()

    bars = D.load(symbol, interval,
                  start=req.get("start"), end=req.get("end"),
                  range_=req.get("range"))

    cfg = build_config(req, strategy)
    overrides = {k: _coerce(v) for k, v in (req.get("params") or {}).items()}

    result, params = run_strategy(bars.df, strategy, overrides, cfg,
                                  bars.ticker, bars.interval)
    stats = M.compute(result)

    bench_symbol = req.get("benchmark")
    if bench_symbol is None:
        bench_symbol = default_benchmark(symbol)
    bench_close = _benchmark_close(bench_symbol, result.equity.index, bars.interval)

    payload = {
        "symbol": bars.ticker,
        "strategy": strategy,
        "interval": bars.interval,
        "range": bars.range_,
        "range_clamped": bars.range_clamped,
        "params": _clean(params),
        "config": {
            "initial_capital": cfg.initial_capital,
            "commission_bps": cfg.commission_bps,
            "slippage_bps": cfg.slippage_bps,
            "size_pct": cfg.size_pct,
            "allow_long": cfg.allow_long,
            "allow_short": cfg.allow_short,
            "stop_loss_pct": cfg.stop_loss_pct,
            "take_profit_pct": cfg.take_profit_pct,
            "trailing_stop_pct": cfg.trailing_stop_pct,
            "atr_stop_mult": cfg.atr_stop_mult,
            "atr_period": cfg.atr_period,
        },
        "summary": _summary(stats),
        "extended": _clean(M.extended(result, stats, bench_close)),
        "benchmark_symbol": bench_symbol if bench_close is not None else None,
        "equity": _curve(result.equity),
        "buy_hold": _curve(stats["bh_curve"]),
        "drawdown": _curve(M.drawdown(result.equity)),
        "rolling_sharpe": _curve(
            M.rolling_sharpe(result.equity, M._ann_factor(result.equity.index, bars.interval))),
        "trades": _trades(result),
        "monthly": _monthly(result.equity),
        "bars": len(bars.df),
    }

    oos_frac = req.get("oos_frac")
    if oos_frac:
        payload["oos"] = _oos(bars, strategy, overrides, cfg, float(oos_frac))

    if req.get("monte_carlo"):
        paths = int(req.get("mc_paths") or 1000)
        payload["monte_carlo"] = _clean(A.monte_carlo(
            result, stats,
            paths=max(100, min(paths, MAX_MC_PATHS)),
            ruin_pct=float(req.get("ruin_pct") or 0.5)))

    payload["elapsed_ms"] = int((time.time() - started) * 1000)
    return payload


def _oos(bars, strategy: str, overrides: dict, cfg: E.Config, frac: float) -> dict:
    """The desktop tester's --oos: fit nothing, just split and report both halves."""
    if not 0.1 <= frac <= 0.6:
        raise BacktestError("The out-of-sample fraction must be between 0.1 and 0.6.")
    df = bars.df
    cut = int(len(df) * (1 - frac))
    if cut < 100 or len(df) - cut < 50:
        return {"available": False,
                "reason": "Not enough bars for a meaningful in/out-of-sample split."}

    halves = {}
    for label, slc in (("in_sample", df.iloc[:cut]), ("out_of_sample", df.iloc[cut:])):
        result, _ = run_strategy(slc, strategy, overrides, cfg, bars.ticker, bars.interval)
        stats = M.compute(result)
        halves[label] = {"summary": _summary(stats), "equity": _curve(result.equity)}
        halves[label]["_raw"] = stats

    verdict = A.oos_verdict(halves["in_sample"]["_raw"], halves["out_of_sample"]["_raw"])
    for half in halves.values():
        half.pop("_raw")
    return {"available": True, "fraction": frac, "verdict": verdict, **halves}


# ---------------------------------------------------------------- sweeps (SSE)

def _sweep_row(bars, strategy, base_overrides, combo, cfg, slc=None):
    """One grid point. Returns None when that combination cannot run at all."""
    df = bars.df if slc is None else bars.df.iloc[slc[0]:slc[1]]
    try:
        result, _ = run_strategy(df, strategy, {**base_overrides, **combo}, cfg,
                                 bars.ticker, bars.interval)
        stats = M.compute(result)
    except BacktestError:
        raise
    except Exception:
        # A single dead combination (an EMA longer than the window, say) is not
        # a reason to abandon the sweep.
        return None
    return {
        "params": combo,
        "cagr": _clean(stats["cagr"]), "sharpe": _clean(stats["sharpe"]),
        "sortino": _clean(stats["sortino"]), "max_dd": _clean(stats["max_dd"]),
        "calmar": _clean(stats["calmar"]),
        "profit_factor": _clean(stats["profit_factor"]),
        "trades": int(stats["n_trades"]), "win_rate": _clean(stats["win_rate"]),
        "exposure": _clean(stats["exposure"]),
    }


def _grid_for(req: dict) -> tuple:
    strategy = (req.get("strategy") or "ema_cross").strip()
    grid = A.parse_grid(req.get("grid") or "")
    spec = S.REGISTRY.get(strategy)
    if spec is None:
        raise BacktestError(f'Unknown strategy "{strategy}".')
    unknown = [k for k in grid if k not in spec["params"]]
    if unknown:
        raise BacktestError(
            f'{strategy} has no parameter {unknown[0]!r}. '
            f'Valid: {", ".join(spec["params"]) or "(none)"}.')
    combos = A.combinations(grid)
    if len(combos) > MAX_COMBOS:
        raise BacktestError(
            f"That grid is {len(combos):,} combinations; the cap is {MAX_COMBOS:,}. "
            f"Widen the step or sweep fewer parameters.")
    if not combos:
        raise BacktestError("The sweep grid is empty.")
    return strategy, grid, combos


def optimize(req: dict):
    """Parameter sweep, yielding progress. Ranked by Sharpe, like the CLI."""
    strategy, grid, combos = _grid_for(req)
    symbol = (req.get("symbol") or "").strip()
    if not symbol:
        raise BacktestError("A symbol is required.")
    interval = (req.get("interval") or "1d").strip()

    bars = D.load(symbol, interval, start=req.get("start"), end=req.get("end"),
                  range_=req.get("range"))
    cfg = build_config(req, strategy)
    base = {k: _coerce(v) for k, v in (req.get("params") or {}).items()}
    # A swept parameter cannot also be pinned, or the pin silently wins.
    base = {k: v for k, v in base.items() if k not in grid}

    started = time.time()
    yield {"type": "start", "symbol": bars.ticker, "strategy": strategy,
           "interval": bars.interval, "combinations": len(combos),
           "keys": list(grid), "bars": len(bars.df)}

    rows = []
    for i, combo in enumerate(combos, 1):
        row = _sweep_row(bars, strategy, base, combo, cfg)
        if row is not None:
            rows.append(row)
        yield {"type": "progress", "done": i, "total": len(combos),
               "params": combo, "result": row}

    if not rows:
        yield {"type": "error", "message": "No combination produced a result."}
        return

    rows.sort(key=lambda r: (r["sharpe"] is not None, r["sharpe"] or -1e9), reverse=True)
    yield {"type": "done", "rows": rows, "keys": list(grid),
           "verdict": A.sweep_verdict(rows),
           "elapsed_ms": int((time.time() - started) * 1000)}


def walk_forward(req: dict):
    """Rolling optimise-then-trade folds, yielding one event per fold.

    This is the expensive one: the full grid runs once per fold, so a 50-point
    grid over 5 folds is 250 backtests plus the 5 out-of-sample runs.
    """
    strategy, grid, combos = _grid_for(req)
    symbol = (req.get("symbol") or "").strip()
    if not symbol:
        raise BacktestError("A symbol is required.")
    interval = (req.get("interval") or "1d").strip()
    folds = int(req.get("folds") or 5)
    is_frac = float(req.get("is_frac") or 0.7)

    bars = D.load(symbol, interval, start=req.get("start"), end=req.get("end"),
                  range_=req.get("range"))
    cfg = build_config(req, strategy)
    base = {k: _coerce(v) for k, v in (req.get("params") or {}).items()}
    base = {k: v for k, v in base.items() if k not in grid}

    windows = A.walk_forward_windows(len(bars.df), folds, is_frac)
    runs = len(windows) * (len(combos) + 1)
    if runs > MAX_WALK_FORWARD_RUNS:
        raise BacktestError(
            f"{len(combos):,} combinations across {len(windows)} folds is "
            f"{runs:,} backtests; the cap is {MAX_WALK_FORWARD_RUNS:,}. "
            f"Use fewer folds or a coarser grid.")

    started = time.time()
    yield {"type": "start", "symbol": bars.ticker, "strategy": strategy,
           "interval": bars.interval, "folds": len(windows),
           "combinations": len(combos), "keys": list(grid),
           "total_runs": len(windows) * (len(combos) + 1)}

    fold_rows, oos_curves = [], []
    for k, ((is_a, is_b), (oos_a, oos_b)) in enumerate(windows, 1):
        # --- fit on the in-sample slice -----------------------------------
        best = None
        for j, combo in enumerate(combos, 1):
            row = _sweep_row(bars, strategy, base, combo, cfg, slc=(is_a, is_b))
            if row and row["sharpe"] is not None:
                if best is None or row["sharpe"] > best["sharpe"]:
                    best = row
            yield {"type": "progress", "fold": k, "folds": len(windows),
                   "done": j, "total": len(combos)}
        if best is None:
            yield {"type": "fold", "fold": k, "skipped": True,
                   "reason": "no usable parameters in this window"}
            continue

        # --- trade the window after it, untouched -------------------------
        oos_slice = bars.df.iloc[oos_a:oos_b]
        result, _ = run_strategy(oos_slice, strategy, {**base, **best["params"]},
                                 cfg, bars.ticker, bars.interval)
        oos_stats = M.compute(result)
        oos_curves.append(result.equity)

        row = {
            "fold": k,
            "params": _clean(best["params"]),
            "in_sample": {"start": _ts(bars.df.index[is_a]),
                          "end": _ts(bars.df.index[is_b - 1]),
                          "cagr": best["cagr"], "sharpe": best["sharpe"],
                          "max_dd": best["max_dd"], "trades": best["trades"]},
            "out_of_sample": {"start": _ts(bars.df.index[oos_a]),
                              "end": _ts(bars.df.index[oos_b - 1]),
                              "cagr": _clean(oos_stats["cagr"]),
                              "sharpe": _clean(oos_stats["sharpe"]),
                              "max_dd": _clean(oos_stats["max_dd"]),
                              "trades": int(oos_stats["n_trades"])},
        }
        fold_rows.append(row)
        yield {"type": "fold", **row}

    if not fold_rows:
        yield {"type": "error", "message": "No fold produced a result."}
        return

    stitched = A.stitch(oos_curves, cfg.initial_capital)
    is_cagrs = [r["in_sample"]["cagr"] for r in fold_rows if r["in_sample"]["cagr"] is not None]
    oos_cagrs = [r["out_of_sample"]["cagr"] for r in fold_rows
                 if r["out_of_sample"]["cagr"] is not None]
    mean_is = float(np.mean(is_cagrs)) if is_cagrs else 0.0
    mean_oos = float(np.mean(oos_cagrs)) if oos_cagrs else 0.0
    efficiency = mean_oos / mean_is if mean_is > 0 else 0.0
    positive = sum(1 for c in oos_cagrs if c > 0)

    summary = {}
    if len(stitched) > 2:
        years = max((stitched.index[-1] - stitched.index[0]).days / 365.25, 1e-9)
        final = float(stitched.iloc[-1])
        summary = {
            "final": final,
            "total_return": final / cfg.initial_capital - 1.0,
            "cagr": (final / cfg.initial_capital) ** (1.0 / years) - 1.0 if final > 0 else -1.0,
            "max_dd": float(M.drawdown(stitched).min()),
        }

    yield {"type": "done",
           "folds": fold_rows,
           "stitched_equity": _curve(stitched),
           "stitched_summary": _clean(summary),
           "mean_is_cagr": mean_is, "mean_oos_cagr": mean_oos,
           "efficiency": efficiency,
           "verdict": A.walk_forward_verdict(efficiency, positive, len(fold_rows)),
           "elapsed_ms": int((time.time() - started) * 1000)}
