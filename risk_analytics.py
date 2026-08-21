"""
Institutional-style portfolio risk analytics for the paper portfolio.

Given the user's positions (symbol, shares, averagePrice) and cash balance,
computes on 1y of daily data:
  - Historical & parametric VaR (95%), CVaR / expected shortfall
  - Annualized Sharpe and Sortino ratios
  - Max drawdown of the weighted portfolio
  - Portfolio beta vs S&P 500
  - Correlation matrix, concentration (HHI), sector allocation
"""
import numpy as np
import pandas as pd

import data_source

RISK_FREE = 0.04
TRADING_DAYS = 252


def get_portfolio_analytics(positions: list, balance: float = 0.0):
    try:
        if not positions:
            return {"error": "No positions to analyze."}

        symbols = [p["symbol"] for p in positions]
        raw = pd.DataFrame({
            s: data_source.get_close_series(s, "1y") for s in symbols + ["^GSPC"]
        }).dropna(how="all").ffill()

        prices = {}
        sectors = {}
        for p in positions:
            sym = p["symbol"]
            if sym in raw.columns and not raw[sym].dropna().empty:
                prices[sym] = float(raw[sym].dropna().iloc[-1])
            else:
                prices[sym] = float(p.get("averagePrice", 0))
            sectors[sym] = data_source.get_sector(sym)

        values = {p["symbol"]: p["shares"] * prices[p["symbol"]] for p in positions}
        invested = sum(values.values())
        total_equity = invested + balance
        if invested <= 0:
            return {"error": "Portfolio has no market value."}

        weights = pd.Series({s: v / invested for s, v in values.items()})

        available = [s for s in weights.index if s in raw.columns]
        rets = raw[available].pct_change().dropna()
        if rets.empty or len(rets) < 30:
            return {"error": "Not enough return history for risk analytics."}

        w = weights[available] / weights[available].sum()
        port_rets = (rets * w).sum(axis=1)

        # VaR / CVaR at 95% on the invested (non-cash) sleeve, in dollars
        var_hist = float(np.percentile(port_rets, 5))
        cvar = float(port_rets[port_rets <= var_hist].mean()) if (port_rets <= var_hist).any() else var_hist
        var_param = float(port_rets.mean() - 1.645 * port_rets.std())

        ann_ret = float(port_rets.mean() * TRADING_DAYS)
        ann_vol = float(port_rets.std() * np.sqrt(TRADING_DAYS))
        sharpe = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else 0.0
        downside = port_rets[port_rets < 0].std() * np.sqrt(TRADING_DAYS)
        sortino = (ann_ret - RISK_FREE) / downside if downside and downside > 0 else 0.0

        cum = (1 + port_rets).cumprod()
        drawdown = cum / cum.cummax() - 1
        max_dd = float(drawdown.min())

        beta = None
        if "^GSPC" in raw.columns:
            mkt = raw["^GSPC"].pct_change().dropna()
            aligned = pd.concat([port_rets, mkt], axis=1).dropna()
            if len(aligned) > 30 and aligned.iloc[:, 1].var() > 0:
                beta = float(np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0][1] / aligned.iloc[:, 1].var())

        corr = rets.corr().round(2)
        corr_matrix = {s: {t: float(corr.loc[s, t]) for t in corr.columns} for s in corr.index}

        hhi = float((weights ** 2).sum())  # 1/n (diversified) .. 1 (single asset)
        sector_alloc = {}
        for sym, val in values.items():
            sector = sectors.get(sym, "Other")
            sector_alloc[sector] = sector_alloc.get(sector, 0) + val / invested

        equity_curve = [
            {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 4)}
            for d, v in cum.tail(180).items()
        ]

        return {
            "total_equity": round(total_equity, 2),
            "invested_value": round(invested, 2),
            "cash": round(balance, 2),
            "risk": {
                "var_95_daily_pct": round(var_hist * 100, 2),
                "var_95_daily_usd": round(abs(var_hist) * invested, 2),
                "var_95_parametric_pct": round(var_param * 100, 2),
                "cvar_95_daily_pct": round(cvar * 100, 2),
                "cvar_95_daily_usd": round(abs(cvar) * invested, 2),
                "annual_volatility_pct": round(ann_vol * 100, 2),
                "max_drawdown_pct": round(max_dd * 100, 2),
                "sharpe_ratio": round(sharpe, 2),
                "sortino_ratio": round(sortino, 2),
                "beta_vs_sp500": round(beta, 2) if beta is not None else None,
                "concentration_hhi": round(hhi, 3),
                "diversification_score": round((1 - hhi) * 100, 0),
            },
            "weights": {s: round(float(v), 4) for s, v in weights.items()},
            "sector_allocation": {k: round(v, 4) for k, v in sector_alloc.items()},
            "correlation_matrix": corr_matrix,
            "equity_curve": equity_curve,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
