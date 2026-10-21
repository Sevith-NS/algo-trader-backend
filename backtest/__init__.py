"""Event-driven backtester, ported from the standalone desktop tester.

The engines (`engine`, `riskengine`, `swingpro`), the TradingView-equivalent
indicators and the strategy registry are the desktop code unchanged — the same
fills, the same costs, the same pessimistic stop-before-target convention — so
a result here and a result there agree bar for bar on the same frame.

What is new is the plumbing around them: `data` pulls bars from the platform's
own market-data layer instead of yfinance, `analysis` adds the out-of-sample,
walk-forward and Monte Carlo passes, and `service` turns a request from the
Terminal into JSON.
"""
