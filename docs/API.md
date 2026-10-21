# Polaris API reference

What each backend endpoint does, who can call it, and which outside services sit behind it.
The source of truth is the code in `flint_api/routers/`. This file describes the `flint-os-v2` branch as of 2026-10-19, including the uncommitted broker work.

## How a request reaches the API

```
Browser ──> Next.js BFF  /api/backend/api/<route>   (frontend/src/app/api/backend/[...path]/route.ts)
              │  swaps the Clerk session cookie for a short-lived Clerk token
              │  refuses cross-site writes (CSRF check)
              │  forwards pb_* broker cookies in X-Broker-Sessions
              ▼
FastAPI  http://127.0.0.1:5000/api/<route>   (flint_api/main.py)
```

- **Auth:** every route needs a valid Clerk token (`Authorization: Bearer …`) except `GET /api/health` and `GET /api/public/landing`. The browser never calls the API directly.
- **Errors:** always flat JSON, `{error, code, ref?, fields?}`.
- **Rate limits** are per user, by scope (`flint_api/ratelimit.py`):

| Scope | Burst | Refill | Used for |
|---|---|---|---|
| `standard` | 120 | 1 / s | cached market data, reads |
| `expensive` | 12 | 1 / 10 s | fan-out work, single LLM calls |
| `agents` | 3 | 1 / min | AI desk runs (about 14 LLM turns each) |
| `write` | 60 | 1 / 2 s | anything that changes user data |
| `backtest` | (own bucket) | | backtests and sweeps |

- **Streams:** routes marked SSE stream `text/event-stream` events.
- **Cache:** the cache column is the `Cache-Control` max-age the API sets.

---

## Service

| Method | Path | What it does |
|---|---|---|
| GET | `/api/health` | Liveness check. No auth. |
| GET | `/api/session` | Echoes the verified Clerk identity, so the frontend can confirm the token path works. |

## Public (`public.py`)

| Method | Path | What it does |
|---|---|---|
| GET | `/api/public/landing` | **No auth.** Headline indices for India and the US (with breadth) plus USD/INR, each with source and time, for the landing page. Built at most once a minute per process, and every caller gets the same copy, so traffic can't amplify upstream calls. The Next.js server fetches it (`publicGet` in `frontend/src/lib/server-api.ts`, revalidated every 60 s); the browser never does. |

## Me: profile and account (`me.py`)

| Method | Path | What it does |
|---|---|---|
| GET | `/api/me` | Profile: plan, base currency, `risk_per_trade_pct`, markets, onboarded flag. |
| PATCH | `/api/me` | Update base currency, risk per trade (0–10%), or markets. |
| POST | `/api/me/onboarding` | First-run setup: base currency, markets, risk %, and paper starting cash (creates the paper account). |
| GET | `/api/me/export` | Everything Polaris stores about the user, as one JSON download. |
| DELETE | `/api/me` | Deletes all Polaris data and the Clerk account. The user must type a confirmation. |

## Accounts: paper ledger and real holdings (`ledger.py`)

Paper accounts are rows in Neon. Real accounts are **not** stored: each one is a broker connection held in an encrypted cookie in the user's browser, and its id is the connection id.

| Method | Path | What it does |
|---|---|---|
| GET | `/api/accounts` | Paper accounts from the DB, plus one real account per connected broker in this browser. |
| GET | `/api/accounts/{id}/positions` | Paper only. Replays fills (average cost) into cash, open positions, and live value/P&L. |
| GET | `/api/accounts/{id}/fills` | The fill history (append-only). `limit` 1–200. |
| POST | `/api/accounts/{id}/orders` | Paper market order. The server prices it from a live quote; the client never sends a price. Stale quotes are refused. `client_order_id` makes a retry land once. Real accounts get a 403. |
| GET | `/api/accounts/{id}/holdings` | Real only. Holdings fetched **live from the broker** on every call, with FX into the base currency and totals. `expensive`. |
| GET | `/api/accounts/{id}/performance` | Performance (TWR/XIRR against a benchmark) over a chosen period. `expensive`. |

## Brokers: read-only connections (`brokers.py`, `brokers/*.py`, `broker_jar.py`)

No endpoint can place a real order. Nothing per broker is configured on the server: each user brings their own broker developer app (Zerodha, Upstox, FYERS, Angel One, Dhan) or a read-only token (Groww, INDmoney, Binance). Tokens and app keys are sealed with AES-GCM into httpOnly `pb_*` cookies bound to the Clerk user, kept for as long as the user chose, and never written to the DB or returned by any route. The server only needs `BROKER_TOKEN_KEY`.

| Method | Path | What it does |
|---|---|---|
| GET | `/api/brokers` | This browser's connections (with `kept_until`, `session_only`), plus each broker's fields, help link and `callback_path`, and `max_keep_days`. Deletes expired or unreadable cookies. |
| GET | `/api/brokers/watchlist` | One read-only list per connection: its holdings and today's positions, live. `expensive`. |
| POST | `/api/brokers/{broker}/connect` | OAuth brokers. Body: `redirect_uri` (must be `<authorized party><callback_path>`) and either `app` (the user's app keys) + `keep` / `keep_until`, or `connection_id` to reconnect with saved keys. Sets a 10-minute `pb_state` cookie and returns the broker's login URL. |
| POST | `/api/brokers/{broker}/session` | Called by the callback page with the broker's one-time values (Zerodha: `request_token`). Checks `pb_state`, exchanges for an access token with the user's app keys, and saves the connection cookie. |
| POST | `/api/brokers/{broker}/link` | Key-based brokers (Groww, INDmoney, Binance). Body: `credentials` + `keep` / `keep_until`. Checks the pasted token or key with the broker, then saves it. |
| DELETE | `/api/brokers/{connection_id}` | Forgets the connection and its keys and ends the broker session where the broker supports it. |

**Broker names:** `zerodha`, `upstox`, `groww`, `angel`, `fyers`, `dhan`, `indmoney`, `binance`.

**Keep (like a GitHub token's expiration):** `session` (browser-session cookie, at most 24 hours), `token` (until the broker session ends), `7d`, `30d`, `90d`, or `custom` with a zoned `keep_until` up to 90 days out. The deadline is inside the sealed cookie and checked on every request.

**Zerodha flow:**
1. The user creates a Kite Connect app at developers.kite.trade with Redirect URL `<frontend>/settings/brokers/callback`, and enters its API key and secret in Settings.
2. `connect` returns the Kite login URL for that app.
3. The user logs in, and Kite redirects to the callback page, which calls `session`.
4. The API calls `api.kite.trade/session/token`, signed with the user's secret.
5. Holdings come from `/portfolio/holdings`; positions from `/portfolio/positions`.

The session ends at 06:00 IST. Reconnecting reuses the saved keys, so it's one click.

## Market data (`market.py`)

| Method | Path | What it does | Cache |
|---|---|---|---|
| GET | `/api/search?q=` | Symbol search (Yahoo). | 300 s |
| GET | `/api/quote?symbol=` | One quote. | 30 s |
| GET | `/api/chart?symbol=&interval=&range=` | OHLCV candles. The range is clamped to what the interval allows; the response says so. | 300 s |
| GET | `/api/trending` | Trending symbols. | 120 s |
| GET | `/api/market-overview` | Headline indices overview. | 120 s |
| GET | `/api/markets/pulse?market=in\|us` | Headline indices, breadth and sectors for one market. | 60 s |
| GET | `/api/fx?pair=USDINR` | One FX rate, with its source and time. | |
| GET | `/api/discover?universe=` | Tagged stocks across a universe (screener engine). | 300 s |
| GET | `/api/screens` | Catalogue of predefined screens. | 1 h |
| GET | `/api/screens/run?id=&universe=` | Runs one screen over a universe. | 300 s |
| GET | `/api/levels?universe=&side=` | Support and resistance levels across a universe. | 300 s |
| GET | `/api/movers/scopes` | Available NSE mover scopes (segments). | 1 h |
| GET | `/api/movers?scope=&min_move=&limit=` | Top gainers and losers for a scope. Not cached while still warming. | 60 s |

Scans that parse nothing are sent `no-store`, so an upstream outage isn't cached.

## Per-symbol research (`symbol.py`)

| Method | Path | What it does |
|---|---|---|
| GET | `/api/quant-signals?symbol=` | The quant engine (`quant_models.py`): factor readings, composite score, indicators and solved levels (entry, stop, T1/T2, side, odds). The router strips `signal`, `action`, `conviction` and `position_sizing`; size comes from the user's own risk % in the UI. |
| GET | `/api/ml-insights?symbol=` | ML forecasts (`ml_models.py`). |
| GET | `/api/advanced-signals?symbol=` | ATR and 20-day SMA (`ml_models.py`). BUY/SELL labels and their stop/take-profit are stripped. Not used by the UI. |
| GET | `/api/fundamentals?symbol=` | Fundamentals X-ray. |
| GET | `/api/news/ticker?symbol=` | Headlines for a symbol. |
| GET | `/api/news-intel?symbol=` | Headlines scored for tone and sources. |
| GET | `/api/deep-analysis?symbol=&ai=` | Fundamentals, technical structure and five `readings.pillars` (−1…+1, null if a source is down), with no blended call, conviction or "interesting price". `ai=true` adds a Gemini write-up that explains without recommending; only known sections are kept. `expensive`. |

> **Regulatory note (2026-10-19):** Polaris shows readings, never calls. No buy/sell/hold label, conviction or engine-suggested size leaves these routes or the desk (`tests/test_no_calls.py`, `tests/test_desk_no_calls.py`). `/api/ml-insights` also drops the Prophet 7-day price forecast.

## AI (`ai.py`)

| Method | Path | What it does |
|---|---|---|
| POST | `/api/ai-chat` | Assistant chat. Up to 40 messages, with optional portfolio snapshot (capped at 20 KB) and symbol context. `expensive`. |
| POST | `/api/agents/run` | **SSE.** The research desk (`agents_desk.py`). Events: `context`, 4× `analyst` (readings −1…+1), `debate` (case for strength vs caution), `summary` (balanced, how the plan was computed), `rules` (deterministic sizing from the user's own `risk_per_trade_pct`, no model), `journal`, `done`. No agent outputs a call, conviction, target or size. A note is kept per run. `agents` scope. |
| GET | `/api/agents/routing` | Admin only. Where model calls are being routed. |

## Portfolio analytics (`portfolio.py`)

| Method | Path | What it does |
|---|---|---|
| POST | `/api/portfolio-analytics` | Risk analytics over a list of positions and a cash balance: VaR, drawdown, concentration and similar. |
| POST | `/api/portfolio-optimization` | Optimised weights for 2 or more symbols. `expensive`. |
| POST | `/api/holdings-intel` | Per-holding summary: health grade, red flags, valuation vs history, phase, news stance and readings (no calls). `expensive`. |

## Backtesting (`backtest.py`)

| Method | Path | What it does |
|---|---|---|
| GET | `/api/backtest/strategies` | The strategy registry and its parameters. |
| POST | `/api/backtest/run` | One event-driven backtest, with costs and an optional benchmark. |
| POST | `/api/backtest/optimize` | **SSE.** Parameter grid sweep. |
| POST | `/api/backtest/walk-forward` | **SSE.** Walk-forward validation. |

## Watchlists (`watchlists.py`)

| Method | Path | What it does |
|---|---|---|
| GET | `/api/watchlists` | The user's lists and their items. |
| POST | `/api/watchlists` | Create a list. |
| PATCH | `/api/watchlists/{id}` | Rename a list. |
| DELETE | `/api/watchlists/{id}` | Delete a list. |
| POST | `/api/watchlists/{id}/items` | Add a symbol. |
| DELETE | `/api/watchlists/{id}/items/{symbol}` | Remove a symbol. |

## Alerts (`alerts.py`)

Alerts trigger when price crosses above or below a level, or when the day's move passes a percentage. There's no background worker: alerts are checked when the app lists alerts or polls firings, at most once a minute per user, on delayed quotes. A stale quote never fires an alert, and each alert fires once until it's re-armed.

| Method | Path | What it does |
|---|---|---|
| GET | `/api/alerts` | List alerts (and run the per-minute check). |
| POST | `/api/alerts` | Create an alert. Maximum 50. |
| PATCH | `/api/alerts/{id}` | Edit the level or note, pause, or re-arm. |
| DELETE | `/api/alerts/{id}` | Delete. |
| GET | `/api/alerts/firings` | Recent firings (the bell popover). |
| POST | `/api/alerts/firings/seen` | Mark firings as seen. |

## Journal (`journal.py`)

| Method | Path | What it does |
|---|---|---|
| GET | `/api/journal` | The user's desk notes, newest first: summary, what to watch, analyst readings. Nothing is scored. Older notes keep their legacy fields. |
| DELETE | `/api/journal/{id}` | Soft delete; can be restored for 30 days. |
| POST | `/api/journal/{id}/restore` | Restore a deleted entry. |

## News (`news.py`)

| Method | Path | What it does |
|---|---|---|
| GET | `/api/news` | Global market headlines (Google News RSS and Yahoo; only http(s) links are kept). |

---

## Outside services the API calls

| Service | Used for | Notes |
|---|---|---|
| Yahoo Finance (`query1`/`query2.finance.yahoo.com`) | Quotes, charts, search, FX, fundamentals | Unofficial and free. Quotes can be delayed. |
| NSE (`nseindia.com`, `nsearchives.nseindia.com`) | Index constituents, segments, movers universe | Public pages. Mind NSE's terms on redistribution. |
| Google News RSS, Bing | Headlines | |
| Google Gemini | AI chat, desk seats, deep-analysis write-up | Routed through `llm_router.py`. |
| Clerk (`api.clerk.com`) | Token verification (JWKS), account deletion | |
| Neon Postgres | Users, paper accounts, fills, watchlists, alerts, journal | Row-level security keyed to the Clerk user id. No broker data. |
| Zerodha Kite Connect (`api.kite.trade`) | Session exchange, holdings, positions, logout | Read-only calls only. |
| Upstox, FYERS, Angel One, Dhan, Groww, INDmoney, Binance | Same read-only role | Each is off until its keys are set. |
