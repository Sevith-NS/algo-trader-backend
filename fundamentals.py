"""
Fundamental X-Ray engine — the data-verifiable slice of an equity teardown.

Pulls 4-5 years of annual statements + trailing quarters from Yahoo's public
fundamentals-timeseries API (plain requests via data_source, no crumb needed —
works for both US and NSE '.NS' tickers), then derives what an analyst would
actually compute by hand:

  - Revenue/profit trajectory and margin trend
  - Cash conversion (CFO vs reported PAT) — the classic accounting smell test
  - Balance sheet: leverage, interest coverage, current ratio, dilution
  - ROE level and direction
  - Valuation computed from raw statements (P/E, P/B, EV/EBITDA) and compared
    against the stock's OWN multi-year P/E history, not a generic benchmark

Everything here is deterministic and auditable. What this engine deliberately
does NOT claim to cover (no reliable free API exists): concall transcripts,
MD&A tone, promoter pledge / shareholding pattern deltas. The frontend surfaces
that gap honestly instead of papering over it.
"""
import time

import data_source

_TS_URL = "https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{symbol}"

_ANNUAL_TYPES = [
    "annualTotalRevenue", "annualGrossProfit", "annualOperatingIncome",
    "annualNetIncome", "annualEBITDA", "annualDilutedEPS",
    "annualOperatingCashFlow", "annualFreeCashFlow", "annualCapitalExpenditure",
    "annualTotalDebt", "annualStockholdersEquity", "annualCashAndCashEquivalents",
    "annualCurrentAssets", "annualCurrentLiabilities", "annualAccountsReceivable",
    "annualInventory", "annualInterestExpense", "annualDilutedAverageShares",
]
_QUARTERLY_TYPES = [
    "quarterlyTotalRevenue", "quarterlyNetIncome",
    "quarterlyOperatingIncome", "quarterlyDilutedEPS",
]


def _fetch_timeseries(symbol: str) -> dict:
    """One batched call for all annual + quarterly series. {type: {date: value}}"""
    data = data_source.get_json_cached(
        f"fund:{symbol}",
        _TS_URL.format(symbol=symbol),
        {
            "symbol": symbol,
            "type": ",".join(_ANNUAL_TYPES + _QUARTERLY_TYPES),
            "period1": "946684800",  # 2000-01-01
            "period2": str(int(time.time())),
            "merge": "false",
        },
        fresh_ttl=data_source.FRESH_TTL["fundamentals"],
    )
    out: dict = {}
    if not data:
        return out
    for item in data.get("timeseries", {}).get("result", []):
        for key in item:
            if key in ("meta", "timestamp"):
                continue
            series = {}
            for point in item[key] or []:
                if point and point.get("reportedValue") is not None:
                    series[point["asOfDate"]] = point["reportedValue"].get("raw")
            if series:
                out[key] = series
    return out


def _safe_div(a, b):
    if a is None or b in (None, 0):
        return None
    try:
        return a / b
    except ZeroDivisionError:
        return None


def _cagr(first, last, years):
    if not first or not last or first <= 0 or last <= 0 or years <= 0:
        return None
    return (last / first) ** (1 / years) - 1


def _pct(x, digits=1):
    return round(x * 100, digits) if x is not None else None


def _build_years(ts: dict) -> list:
    """Merge per-type series into one record per fiscal year (last 5)."""
    dates = sorted(ts.get("annualTotalRevenue", {}).keys())[-5:]
    years = []
    for d in dates:
        g = lambda t: ts.get(t, {}).get(d)
        rev, ni, equity = g("annualTotalRevenue"), g("annualNetIncome"), g("annualStockholdersEquity")
        y = {
            "date": d,
            "fiscal_year": d[:4],
            "revenue": rev,
            "gross_profit": g("annualGrossProfit"),
            "operating_income": g("annualOperatingIncome"),
            "net_income": ni,
            "ebitda": g("annualEBITDA"),
            "eps": g("annualDilutedEPS"),
            "ocf": g("annualOperatingCashFlow"),
            "fcf": g("annualFreeCashFlow"),
            "capex": g("annualCapitalExpenditure"),
            "debt": g("annualTotalDebt"),
            "equity": equity,
            "cash": g("annualCashAndCashEquivalents"),
            "current_assets": g("annualCurrentAssets"),
            "current_liabilities": g("annualCurrentLiabilities"),
            "receivables": g("annualAccountsReceivable"),
            "inventory": g("annualInventory"),
            "interest_expense": g("annualInterestExpense"),
            "shares": g("annualDilutedAverageShares"),
        }
        y["gross_margin"] = _safe_div(y["gross_profit"], rev)
        y["operating_margin"] = _safe_div(y["operating_income"], rev)
        y["net_margin"] = _safe_div(ni, rev)
        y["roe"] = _safe_div(ni, equity)
        y["debt_to_equity"] = _safe_div(y["debt"], equity)
        y["cash_conversion"] = _safe_div(y["ocf"], ni) if ni and ni > 0 else None
        y["current_ratio"] = _safe_div(y["current_assets"], y["current_liabilities"])
        y["interest_coverage"] = _safe_div(y["operating_income"], y["interest_expense"])
        years.append(y)
    return years


def _series(years, key):
    return [(y["fiscal_year"], y[key]) for y in years if y[key] is not None]


def _flags(years: list, trends: dict) -> tuple[list, list]:
    """Red / green flags an analyst would circle in the annual report."""
    red, green = [], []
    latest = years[-1]

    def add(bucket, severity, title, detail):
        bucket.append({"severity": severity, "title": title, "detail": detail})

    # --- Cash conversion: are reported profits backed by operating cash? ---
    cc = [y["cash_conversion"] for y in years[-3:] if y["cash_conversion"] is not None]
    if cc:
        avg_cc = sum(cc) / len(cc)
        if avg_cc < 0.7:
            add(red, "high", "Profits not converting to cash",
                f"CFO/PAT averaged {avg_cc:.2f}x over the last {len(cc)} years — "
                "reported earnings are running well ahead of operating cash flow. Classic accounting smell.")
        elif avg_cc >= 1.0:
            add(green, "good", "Cash-backed earnings",
                f"CFO/PAT averaged {avg_cc:.2f}x — every rupee/dollar of reported profit arrived as cash.")

    # --- Receivables growing faster than revenue ---
    rec = _series(years, "receivables")
    rev = _series(years, "revenue")
    if len(rec) >= 3 and len(rev) >= 3:
        rec_cagr = _cagr(rec[0][1], rec[-1][1], len(rec) - 1)
        rev_cagr = _cagr(rev[0][1], rev[-1][1], len(rev) - 1)
        if rec_cagr is not None and rev_cagr is not None and rec_cagr > rev_cagr + 0.10:
            add(red, "medium", "Receivables outpacing sales",
                f"Receivables compounding at {rec_cagr*100:.0f}%/yr vs revenue {rev_cagr*100:.0f}%/yr — "
                "the company may be booking sales it hasn't collected.")

    # --- Leverage ---
    de = latest["debt_to_equity"]
    if de is not None:
        de_hist = _series(years, "debt_to_equity")
        rising = len(de_hist) >= 3 and de - de_hist[0][1] > 0.25
        if de > 2.0:
            add(red, "high", "Heavily leveraged balance sheet",
                f"Debt/Equity at {de:.2f}x. At this leverage the equity is a call option on the business.")
        elif de > 1.0 and rising:
            add(red, "medium", "Rising leverage",
                f"Debt/Equity climbed from {de_hist[0][1]:.2f}x to {de:.2f}x over {len(de_hist)-1} years.")
        elif de < 0.3:
            add(green, "good", "Near debt-free",
                f"Debt/Equity just {de:.2f}x — the balance sheet is not a source of risk.")
        elif len(de_hist) >= 3 and de_hist[0][1] - de > 0.25:
            add(green, "good", "Deleveraging",
                f"Debt/Equity reduced from {de_hist[0][1]:.2f}x to {de:.2f}x.")

    # --- Interest coverage ---
    ic = latest["interest_coverage"]
    if ic is not None and ic > 0:
        if ic < 2.5:
            add(red, "high", "Thin interest coverage",
                f"Operating profit covers interest only {ic:.1f}x. One bad year and debt service eats the P&L.")
        elif ic < 4:
            add(red, "medium", "Modest interest coverage",
                f"EBIT/interest at {ic:.1f}x — acceptable, but leaves little room in a downturn.")

    # --- Free cash flow ---
    fcf = [y["fcf"] for y in years[-3:] if y["fcf"] is not None]
    if len(fcf) >= 2:
        neg = sum(1 for v in fcf if v < 0)
        if neg >= 2:
            add(red, "medium", "Persistently negative free cash flow",
                f"FCF negative in {neg} of the last {len(fcf)} years — heavy capex phase or a business that consumes cash. "
                "Judge whether that capex is building moat or just treading water.")
        elif all(v > 0 for v in fcf) and fcf[-1] > fcf[0]:
            add(green, "good", "Growing free cash flow",
                "FCF positive and expanding across the window.")

    # --- Dilution ---
    sh = _series(years, "shares")
    if len(sh) >= 3:
        sh_cagr = _cagr(sh[0][1], sh[-1][1], len(sh) - 1)
        if sh_cagr is not None and sh_cagr > 0.02:
            add(red, "medium", "Shareholder dilution",
                f"Share count growing {sh_cagr*100:.1f}%/yr — your slice of the pie shrinks every year.")
        elif sh_cagr is not None and sh_cagr < -0.005:
            add(green, "good", "Share count shrinking",
                f"Buybacks retiring {abs(sh_cagr)*100:.1f}% of shares per year.")

    # --- Margin trajectory ---
    nm = _series(years, "net_margin")
    if len(nm) >= 3 and nm[0][1] is not None and nm[-1][1] is not None:
        delta_pp = (nm[-1][1] - nm[0][1]) * 100
        if delta_pp < -2:
            add(red, "medium", "Margin compression",
                f"Net margin eroded {abs(delta_pp):.1f}pp over {len(nm)-1} years "
                f"({nm[0][1]*100:.1f}% → {nm[-1][1]*100:.1f}%). Growth without margin is a treadmill.")
        elif delta_pp > 2:
            add(green, "good", "Margin expansion",
                f"Net margin up {delta_pp:.1f}pp over the window — pricing power or operating leverage at work.")

    # --- ROE ---
    roe = _series(years, "roe")
    if roe:
        latest_roe = roe[-1][1]
        if latest_roe is not None:
            if latest_roe < 0.10:
                add(red, "medium", "Sub-par return on equity",
                    f"ROE at {latest_roe*100:.1f}% — below the bar for compounding wealth; "
                    "check if capital is parked in low-return assets.")
            elif latest_roe > 0.15 and len(roe) >= 3 and min(v for _, v in roe) > 0.13:
                add(green, "good", "Consistently high ROE",
                    f"ROE {latest_roe*100:.1f}%, never below 13% in the window — a genuine compounding machine.")
            if len(roe) >= 3 and roe[0][1] - latest_roe > 0.05:
                add(red, "medium", "ROE decay",
                    f"ROE slid from {roe[0][1]*100:.1f}% to {latest_roe*100:.1f}% — "
                    "incremental capital is earning less than the old capital.")
    return red, green


def _health_score(years: list, trends: dict, red: list) -> dict:
    """0-100 composite: growth / profitability / cash quality / balance sheet, 25 each."""
    latest = years[-1]

    def bucket(value, thresholds):
        """thresholds: list of (min_value, points) descending."""
        if value is None:
            return 12  # unknown ≠ bad; score midpoint
        for t, pts in thresholds:
            if value >= t:
                return pts
        return 0

    growth = bucket(trends.get("revenue_cagr"), [(0.15, 25), (0.10, 20), (0.05, 14), (0.0, 8)])
    profitability = bucket(latest["roe"], [(0.20, 25), (0.15, 21), (0.10, 14), (0.05, 7)])
    cc = [y["cash_conversion"] for y in years[-3:] if y["cash_conversion"] is not None]
    cash_quality = bucket(sum(cc) / len(cc) if cc else None,
                          [(1.1, 25), (0.9, 21), (0.7, 13), (0.5, 6)])
    de = latest["debt_to_equity"]
    balance = 12
    if de is not None:
        balance = 25 if de < 0.3 else 20 if de < 0.7 else 13 if de < 1.5 else 5
    ic = latest["interest_coverage"]
    if ic is not None and 0 < ic < 3:
        balance = min(balance, 8)

    total = growth + profitability + cash_quality + balance
    # High-severity red flags cap the score — a great grower with fake cash isn't an A.
    if any(f["severity"] == "high" for f in red):
        total = min(total, 55)
    grade = ("A+" if total >= 85 else "A" if total >= 75 else "B" if total >= 60
             else "C" if total >= 45 else "D")
    return {
        "score": total, "grade": grade,
        "components": {"growth": growth, "profitability": profitability,
                       "cash_quality": cash_quality, "balance_sheet": balance},
    }


def _valuation(symbol: str, years: list, ts: dict) -> dict:
    """P/E, P/B, EV/EBITDA from raw statements; P/E vs the stock's own history."""
    quote = data_source.get_quote(symbol)
    price = quote.get("regularMarketPrice")
    currency = quote.get("currency") or "USD"
    latest = years[-1]
    shares = latest["shares"]

    # TTM EPS: last 4 quarters of net income / diluted shares; fallback annual EPS
    q_ni = sorted(ts.get("quarterlyNetIncome", {}).items())
    ttm_eps, eps_basis = None, None
    if len(q_ni) >= 4 and shares:
        ttm_eps = sum(v for _, v in q_ni[-4:]) / shares
        eps_basis = "TTM"
    elif latest["eps"]:
        ttm_eps, eps_basis = latest["eps"], f"FY{latest['fiscal_year']}"

    pe = _safe_div(price, ttm_eps) if ttm_eps and ttm_eps > 0 else None
    book_ps = _safe_div(latest["equity"], shares)
    pb = _safe_div(price, book_ps) if book_ps and book_ps > 0 else None

    market_cap = price * shares if price and shares else None
    ev = None
    if market_cap is not None:
        ev = market_cap + (latest["debt"] or 0) - (latest["cash"] or 0)
    ev_ebitda = _safe_div(ev, latest["ebitda"]) if latest["ebitda"] and latest["ebitda"] > 0 else None

    # Historical P/E: price at each fiscal year-end vs that year's EPS
    pe_history = []
    hist = data_source.get_history(symbol, period="5y")
    if not hist.empty:
        closes = hist["Close"]
        idx = closes.index.tz_localize(None) if closes.index.tz is not None else closes.index
        closes = closes.set_axis(idx.normalize())
        import pandas as pd
        for y in years:
            if not y["eps"] or y["eps"] <= 0:
                continue
            target = pd.Timestamp(y["date"])
            window = closes[(closes.index >= target - pd.Timedelta(days=7))
                            & (closes.index <= target + pd.Timedelta(days=7))]
            if not window.empty:
                pe_history.append({"fiscal_year": y["fiscal_year"],
                                   "pe": round(float(window.iloc[-1]) / y["eps"], 1)})

    pe_hist_avg = (round(sum(p["pe"] for p in pe_history) / len(pe_history), 1)
                   if pe_history else None)
    premium_pct = None
    if pe and pe_hist_avg:
        premium_pct = round((pe / pe_hist_avg - 1) * 100, 1)

    fair_price = round(pe_hist_avg * ttm_eps, 2) if pe_hist_avg and ttm_eps and ttm_eps > 0 else None

    if premium_pct is None:
        verdict = "Insufficient history to judge the multiple against itself."
    elif premium_pct > 25:
        verdict = (f"Paying a {premium_pct:.0f}% premium to the stock's own {len(pe_history)}-year "
                   "average P/E — the market is pricing in acceleration. If growth merely stays average, the multiple alone can hurt you.")
    elif premium_pct > 5:
        verdict = f"Modest {premium_pct:.0f}% premium to its own history — not stretched, not a bargain."
    elif premium_pct > -15:
        verdict = "Trading roughly in line with its own historical multiple."
    else:
        verdict = (f"Trading {abs(premium_pct):.0f}% BELOW its own historical average P/E — "
                   "either a genuine mispricing or the market smells earnings decay. Find out which.")

    return {
        "price": price, "currency": currency,
        "market_cap": market_cap,
        "pe": round(pe, 1) if pe else None,
        "eps_ttm": round(ttm_eps, 2) if ttm_eps else None,
        "eps_basis": eps_basis,
        "pb": round(pb, 2) if pb else None,
        "ev_ebitda": round(ev_ebitda, 1) if ev_ebitda else None,
        "pe_history": pe_history,
        "pe_hist_avg": pe_hist_avg,
        "premium_to_own_history_pct": premium_pct,
        "fair_price_at_hist_pe": fair_price,
        "verdict": verdict,
    }


def _quarterly_momentum(ts: dict) -> dict | None:
    """Latest quarter vs same quarter last year — is the engine accelerating?"""
    out = {}
    for key, label in [("quarterlyTotalRevenue", "revenue_yoy"),
                       ("quarterlyNetIncome", "net_income_yoy")]:
        pts = sorted(ts.get(key, {}).items())
        if len(pts) < 5:
            continue
        latest_date, latest_val = pts[-1]
        # same quarter last year = closest point 350-380 days earlier
        import datetime
        ld = datetime.date.fromisoformat(latest_date)
        prior = [v for d, v in pts
                 if 350 <= (ld - datetime.date.fromisoformat(d)).days <= 380]
        if prior and prior[-1] and latest_val is not None:
            out[label] = _pct((latest_val - prior[-1]) / abs(prior[-1]))
            out["latest_quarter"] = latest_date
    return out or None


def get_fundamentals(symbol: str) -> dict:
    try:
        ts = _fetch_timeseries(symbol)
        years = _build_years(ts)
        if len(years) < 2:
            return {"error": "No fundamental statements available for this symbol "
                             "(common for ETFs, indices and some small-caps)."}

        rev = _series(years, "revenue")
        ni = _series(years, "net_income")
        trends = {
            "window_years": len(years) - 1,
            "revenue_cagr": _cagr(rev[0][1], rev[-1][1], len(rev) - 1) if len(rev) >= 2 else None,
            "profit_cagr": _cagr(ni[0][1], ni[-1][1], len(ni) - 1) if len(ni) >= 2 else None,
        }
        red, green = _flags(years, trends)
        health = _health_score(years, trends, red)
        valuation = _valuation(symbol, years, ts)

        return {
            "symbol": symbol,
            "years": [
                {**{k: y[k] for k in (
                    "fiscal_year", "revenue", "net_income", "ocf", "fcf",
                    "debt", "equity", "eps", "shares")},
                 "gross_margin": _pct(y["gross_margin"]),
                 "operating_margin": _pct(y["operating_margin"]),
                 "net_margin": _pct(y["net_margin"]),
                 "roe": _pct(y["roe"]),
                 "debt_to_equity": round(y["debt_to_equity"], 2) if y["debt_to_equity"] is not None else None,
                 "cash_conversion": round(y["cash_conversion"], 2) if y["cash_conversion"] is not None else None,
                 "current_ratio": round(y["current_ratio"], 2) if y["current_ratio"] is not None else None,
                 "interest_coverage": round(y["interest_coverage"], 1) if y["interest_coverage"] is not None else None,
                 } for y in years
            ],
            "trends": {
                "window_years": trends["window_years"],
                "revenue_cagr_pct": _pct(trends["revenue_cagr"]),
                "profit_cagr_pct": _pct(trends["profit_cagr"]),
            },
            "quarterly_momentum": _quarterly_momentum(ts),
            "red_flags": red,
            "green_flags": green,
            "health": health,
            "valuation": valuation,
            "not_covered": [
                "Concall transcript tone", "MD&A language shifts",
                "Promoter pledge / shareholding pattern deltas",
            ],
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
