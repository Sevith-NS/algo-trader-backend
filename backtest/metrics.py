"""Performance statistics computed from an equity curve and a trade list."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ann_factor(index: pd.DatetimeIndex, interval: str) -> float:
    if len(index) > 50:
        span = (index[-1] - index[0]).total_seconds() / 86400.0
        if span > 30:
            return len(index) / (span / 365.25)
    return {"1d": 252, "1wk": 52, "1mo": 12, "1h": 1638, "60m": 1638,
            "30m": 3276, "15m": 6552, "5m": 19656}.get(interval, 252)


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_dd_duration(equity: pd.Series) -> tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
    peak = equity.cummax()
    under = equity < peak
    best, cur, start, best_span = 0, 0, None, (None, None)
    cur_start = None
    for ts, flag in under.items():
        if flag:
            if cur == 0:
                cur_start = ts
            cur += 1
            if cur > best:
                best, best_span = cur, (cur_start, ts)
        else:
            cur = 0
    return best, best_span[0], best_span[1]


def compute(result) -> dict:
    eq = result.equity
    idx = eq.index
    cfg = result.config
    af = _ann_factor(idx, result.interval)

    rets = eq.pct_change().fillna(0.0)
    years = max((idx[-1] - idx[0]).total_seconds() / (86400.0 * 365.25), 1e-9)

    total_ret = eq.iloc[-1] / eq.iloc[0] - 1.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else -1.0

    vol = rets.std(ddof=0) * np.sqrt(af)
    sharpe = (rets.mean() * af) / vol if vol > 0 else 0.0
    downside = rets[rets < 0].std(ddof=0) * np.sqrt(af)
    sortino = (rets.mean() * af) / downside if downside > 0 else 0.0

    dd = drawdown(eq)
    max_dd = dd.min()
    dd_bars, dd_start, dd_end = max_dd_duration(eq)
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    # Buy and hold on the same bars, same starting capital, one round trip of costs.
    close = result.bars["Close"]
    bh_cost = 1.0 - (cfg.commission_bps + cfg.slippage_bps) / 10_000.0 * 2
    bh = cfg.initial_capital * (close / close.iloc[0]) * bh_cost
    bh_total = bh.iloc[-1] / bh.iloc[0] - 1.0
    bh_cagr = (bh.iloc[-1] / bh.iloc[0]) ** (1.0 / years) - 1.0
    bh_dd = drawdown(bh).min()
    bh_rets = bh.pct_change().fillna(0.0)
    bh_vol = bh_rets.std(ddof=0) * np.sqrt(af)
    bh_sharpe = (bh_rets.mean() * af) / bh_vol if bh_vol > 0 else 0.0

    td = result.trades_df
    n_tr = len(td)
    if n_tr:
        wins = td[td.pnl > 0]
        losses = td[td.pnl <= 0]
        gross_win = wins.pnl.sum()
        gross_loss = abs(losses.pnl.sum())
        win_rate = len(wins) / n_tr
        avg_win = wins.pnl.mean() if len(wins) else 0.0
        avg_loss = losses.pnl.mean() if len(losses) else 0.0
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
        expectancy = td.pnl.mean()
        payoff = abs(avg_win / avg_loss) if avg_loss else float("inf")
        best, worst = td.ret_pct.max(), td.ret_pct.min()
        avg_bars = td.bars_held.mean()
        longs = int((td.direction > 0).sum())
        shorts = int((td.direction < 0).sum())
        # longest streaks
        signs = (td.pnl > 0).astype(int).to_numpy()
        def streak(v):
            b = c = 0
            for s in signs:
                c = c + 1 if s == v else 0
                b = max(b, c)
            return b
        win_streak, loss_streak = streak(1), streak(0)
        avg_mae = td.mae_pct.mean()
        exit_mix = td.exit_reason.value_counts().to_dict()
        trades_per_year = n_tr / years
    else:
        wins = losses = None
        win_rate = avg_win = avg_loss = expectancy = 0.0
        profit_factor = payoff = 0.0
        best = worst = avg_bars = 0.0
        longs = shorts = win_streak = loss_streak = 0
        avg_mae = 0.0
        exit_mix = {}
        trades_per_year = 0.0

    exposure = float((result.position != 0).mean())

    return {
        "ticker": result.ticker,
        "strategy": result.strategy_name,
        "interval": result.interval,
        "start": idx[0], "end": idx[-1], "bars": len(idx), "years": years,
        "initial": cfg.initial_capital, "final": float(eq.iloc[-1]),
        "total_return": total_ret, "cagr": cagr,
        "volatility": vol, "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "max_dd": max_dd, "dd_bars": dd_bars, "dd_start": dd_start, "dd_end": dd_end,
        "exposure": exposure,
        "n_trades": n_tr, "trades_per_year": trades_per_year,
        "longs": longs, "shorts": shorts,
        "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
        "profit_factor": profit_factor, "payoff": payoff, "expectancy": expectancy,
        "best_trade": best, "worst_trade": worst, "avg_bars_held": avg_bars,
        "win_streak": win_streak, "loss_streak": loss_streak,
        "avg_mae": avg_mae, "exit_mix": exit_mix,
        "bh_total": bh_total, "bh_cagr": bh_cagr, "bh_max_dd": bh_dd, "bh_sharpe": bh_sharpe,
        "bh_curve": bh,
        "alpha_cagr": cagr - bh_cagr,
    }


def periodic_returns(equity: pd.Series, freq: str = "YE") -> pd.Series:
    return equity.resample(freq).last().pct_change().dropna()


def monthly_table(equity: pd.Series) -> pd.DataFrame:
    m = equity.resample("ME").last().pct_change().dropna()
    if m.empty:
        return pd.DataFrame()
    t = pd.DataFrame({"y": m.index.year, "m": m.index.month, "r": m.to_numpy()})
    piv = t.pivot_table(index="y", columns="m", values="r", aggfunc="first")
    piv.columns = [pd.Timestamp(2000, int(c), 1).strftime("%b") for c in piv.columns]
    yearly = equity.resample("YE").last().pct_change()
    yearly.index = yearly.index.year
    piv["Year"] = yearly.reindex(piv.index)
    return piv


# ---------------------------------------------------------------------------
# Everything below is added for the web backtester. compute() above is the
# desktop tester's, unchanged, so the headline figures stay comparable.
#
# The bias here is toward statistics that change a decision: how deep and how
# LONG the drawdowns were, whether the edge was one good year, where price went
# against you before it worked. Ratios that only reshuffle the same three
# moments are left out.
# ---------------------------------------------------------------------------

def ulcer_index(equity: pd.Series) -> float:
    """Root-mean-square drawdown.

    Max drawdown reports one bad day; the Ulcer index reports how much time was
    spent underwater and how deep, which is the part an investor actually sits
    through.
    """
    dd = drawdown(equity) * 100.0
    return float(np.sqrt((dd ** 2).mean()))


def omega(returns: pd.Series, threshold: float = 0.0) -> float:
    """Gains above the threshold divided by losses below it.

    Uses the whole return distribution rather than its first two moments, so a
    strategy that makes its money in rare large wins is not scored as if those
    wins were normally distributed.
    """
    excess = returns - threshold
    gain = excess[excess > 0].sum()
    loss = -excess[excess < 0].sum()
    if loss <= 0:
        return float("inf") if gain > 0 else 0.0
    return float(gain / loss)


def drawdown_table(equity: pd.Series, top: int = 5) -> list:
    """The N deepest drawdowns, each with how long it took to get back.

    A strategy with one -40% that recovered in three months and one with a -25%
    it never recovered from are not close to the same thing, and a single
    max_dd figure cannot tell them apart.
    """
    peak = equity.cummax()
    under = (equity < peak).to_numpy()
    n = len(equity)

    # (first bar below the old high, last bar of the episode, did it recover).
    # An episode still open at the last bar has not recovered, and reporting it
    # as though it had would hide the one drawdown that is still costing money.
    episodes = []
    start = None
    for i in range(n):
        if under[i] and start is None:
            start = i
        elif not under[i] and start is not None:
            episodes.append((start, i, True))
            start = None
    if start is not None:
        episodes.append((start, n - 1, False))

    rows = []
    for a, b, recovered in episodes:
        window = equity.iloc[a:b + 1]
        trough_pos = int(np.argmin(window.to_numpy()))
        rows.append({
            "start": equity.index[a],
            "trough": window.index[trough_pos],
            "end": equity.index[b] if recovered else None,
            "depth": float(window.iloc[trough_pos] / peak.iloc[a] - 1.0),
            "length_bars": int(b - a + 1),
            # Bars from the low back to the old high — the wait, not the fall.
            "recovery_bars": int(b - trough_pos - a) if recovered else None,
            "recovered": recovered,
        })
    rows.sort(key=lambda r: r["depth"])
    return rows[:top]


def rolling_sharpe(equity: pd.Series, af: float, window_years: float = 1.0) -> pd.Series:
    """Sharpe over a rolling window, annualised.

    The one chart that shows whether an edge persisted or was a single good
    stretch that the full-period figure is still averaging in.
    """
    window = max(20, int(af * window_years))
    if len(equity) < window + 5:
        return pd.Series(dtype=float)
    rets = equity.pct_change().fillna(0.0)
    mean = rets.rolling(window).mean() * af
    vol = rets.rolling(window).std(ddof=0) * np.sqrt(af)
    return (mean / vol.replace(0.0, np.nan)).dropna()


def excursions(trades_df: pd.DataFrame) -> dict:
    """MAE / MFE split by outcome — the stop-placement evidence.

    If winners routinely went 4% against you first, a 2% stop was never going
    to hold them, and that is a fact about the strategy no summary ratio shows.
    """
    if trades_df.empty:
        return {}
    wins = trades_df[trades_df.pnl > 0]
    losses = trades_df[trades_df.pnl <= 0]
    out = {
        "avg_mae_all": float(trades_df.mae_pct.mean()),
        "avg_mfe_all": float(trades_df.mfe_pct.mean()),
        "worst_mae": float(trades_df.mae_pct.min()),
        "best_mfe": float(trades_df.mfe_pct.max()),
    }
    if len(wins):
        out["avg_mae_winners"] = float(wins.mae_pct.mean())
        out["avg_mfe_winners"] = float(wins.mfe_pct.mean())
        # How much of the best open profit was handed back on the way out.
        out["avg_giveback_winners"] = float((wins.mfe_pct - wins.ret_pct).mean())
    if len(losses):
        out["avg_mae_losers"] = float(losses.mae_pct.mean())
        out["avg_mfe_losers"] = float(losses.mfe_pct.mean())
    return out


def benchmark_stats(equity: pd.Series, bench_close: pd.Series, af: float) -> dict:
    """Beta, annualised alpha and correlation against an index.

    Buy & hold on the traded symbol answers "was trading it better than owning
    it". This answers the different question of whether the return was just the
    market showing up, which is the one that decides if the strategy is worth
    running next to an index fund.
    """
    if bench_close is None or len(bench_close) < 30:
        return {}
    # Both curves onto one index; a benchmark that does not overlap is no
    # benchmark, so a thin intersection returns nothing rather than a number.
    joined = pd.concat(
        [equity.rename("s"), bench_close.rename("b")], axis=1, join="inner").dropna()
    if len(joined) < 30:
        return {}
    sr = joined["s"].pct_change().dropna()
    br = joined["b"].pct_change().dropna()
    common = sr.index.intersection(br.index)
    sr, br = sr.loc[common], br.loc[common]
    var = float(br.var(ddof=0))
    if var <= 0:
        return {}
    beta = float(np.cov(sr, br, ddof=0)[0, 1] / var)
    alpha_ann = float((sr.mean() - beta * br.mean()) * af)
    bench_total = float(joined["b"].iloc[-1] / joined["b"].iloc[0] - 1.0)
    return {
        "beta": beta,
        "alpha_annual": alpha_ann,
        "correlation": float(sr.corr(br)) if sr.std() > 0 and br.std() > 0 else 0.0,
        "bench_total_return": bench_total,
        "bench_max_dd": float(drawdown(joined["b"]).min()),
    }


def extended(result, stats: dict, bench_close: pd.Series | None = None) -> dict:
    """The professional stat block, on top of compute()'s headline figures."""
    eq = result.equity
    af = _ann_factor(eq.index, result.interval)
    rets = eq.pct_change().fillna(0.0)
    ui = ulcer_index(eq)
    out = {
        "ulcer_index": ui,
        # Martin ratio: CAGR per unit of time-weighted pain.
        "ulcer_performance_index": float(stats["cagr"] / ui * 100.0) if ui > 0 else 0.0,
        "omega": omega(rets),
        "drawdowns": drawdown_table(eq),
        "excursions": excursions(result.trades_df),
        "annual_returns": {
            str(ts.year): float(v)
            for ts, v in periodic_returns(eq, "YE").items()
        },
    }
    if bench_close is not None:
        out["benchmark"] = benchmark_stats(eq, bench_close, af)
    return out
