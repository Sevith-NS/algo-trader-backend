"""Pine v6 "SWING PRO - INDIA (VP / OB / Flow)" ported to Python.

Everything the Pine version does on bar-state is reproduced here bar by bar:
volume profile (POC / VAH / VAL / HVN), displacement order blocks, clustered
pivot S/R, the benchmark + relative-strength + India-VIX regime gates, the
liquidity / volatility / gap filters, and the two-leg exit (half off at 1.5R,
the rest trailed by a Chandelier stop after breakeven).

It needs its own execution engine rather than bt/engine.py because the strategy
sizes by risk, enters two legs at once and exits them on different rules. The
Result it returns is the same shape, so bt/metrics.py and bt/report.py work on
it unchanged.

Deliberate differences from the Pine script, all of them documented:

* ORDER FLOW. Pine's request.security_lower_tf() gives intraday volume delta.
  Yahoo only serves the last couple of years of hourly bars (measured:
  ~720 daily bars' worth), so `flow_mode` decides:
      auto  - hourly delta where Yahoo has it, close-position proxy before that
      ltf   - hourly delta only; bars without it fail the flow filter
      proxy - ((C-L)-(H-C))/(H-L) everywhere, no intraday fetch
      off   - filter disabled
  Neither is real order flow. Pine cannot see the book and neither can this.
  On the overlap the proxy correlates ~0.68 with the hourly delta.
* FILLS. Pine runs process_orders_on_close=true, so entries fill at the close
  of the signal bar. That is reproduced. Stops and targets placed at that close
  go live on the NEXT bar - Pine would let them fill intrabar on the signal bar
  itself, which is not reachable from daily OHLC.
* When a stop and the 1.5R target both sit inside one bar, the stop is assumed
  to fill first. Pessimistic, and the convention bt/engine.py already uses.
* Pine only builds order blocks when `showOB` is on (a drawing flag gating
  logic - a bug in the original). Here they are always built.
* The turnover / price / VIX filters assume an INR instrument. Set
  -p market=us (see MARKETS) for S&P benchmark, ^VIX, dollar turnover.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data as D
from . import engine as E
from . import indicators as ind
from .errors import BacktestError
from . import riskengine as RE

# ---------------------------------------------------------------- parameters
# One-for-one with the Pine inputs. Override any of them from the CLI with
# -p name=value (booleans take 1/0).
PARAMS: dict = dict(
    # 1 - trend & regime
    ema_fast=21, ema_mid=50, ema_slow=200,
    benchmark="^NSEI", bench_ema=50, use_regime=1,
    use_rs=1, rs_len=60,
    use_vix=1, vix_symbol="^INDIAVIX", vix_max=22.0,
    # 2 - volume profile
    vp_lookback=150, vp_bins=24, vp_update=5, va_pct=0.70, hvn_mult=1.6,
    # 3 - order blocks
    ob_disp_atr=1.0, ob_vol_mult=1.5, ob_search=5, ob_max_age=60, ob_max_count=8,
    # 4 - order-flow proxy
    use_flow=1, flow_mode="auto", ltf="", delta_thr=0.08,
    # 5 - support / resistance
    piv_len=8, sr_tol_atr=0.5, sr_max=12, sr_min_touch=2,
    # 6 - entry / exit
    allow_short=0, zone_buf_atr=0.25, stop_atr=0.6, rr1=1.5,
    trail_len=22, trail_atr=2.5, max_hold=45,
    min_confluence=2,          # Pine hard-codes 2; exposed here for testing
    # 7 - risk & liquidity
    risk_pct=1.0, max_expo_pct=25.0,
    min_turnover=5.0, turnover_unit=1e7, min_price=20.0, skip_circuit=1,
    max_atr_pct=9.0, gap_pct=8.0, gap_cool=3,
    # 8 - which market's conventions to assume (see MARKETS)
    market="in",
)

# Everything that is a local convention rather than a trading idea. Applied
# under your -p overrides, so -p market=us -p vix_max=30 does what it looks like.
#   turnover_unit: what one unit of `min_turnover` is worth. India quotes
#   turnover in crore (1e7), the US in millions of dollars (1e6).
MARKETS: dict = {
    "in": dict(benchmark="^NSEI", vix_symbol="^INDIAVIX", vix_max=22.0,
               min_turnover=5.0, turnover_unit=1e7, min_price=20.0,
               skip_circuit=1),
    "us": dict(benchmark="^GSPC", vix_symbol="^VIX", vix_max=25.0,
               min_turnover=20.0, turnover_unit=1e6, min_price=5.0,
               skip_circuit=0),
}

# Pine's commission_value=0.12 (percent per fill) and slippage=5 ticks.
COSTS = dict(commission_bps=12.0, slippage_bps=5.0)

_LTF_FOR = {"1d": "1h", "1h": "15m", "30m": "5m", "15m": "5m"}


def _norm(overrides: dict | None) -> dict:
    """Defaults, then the market preset, then the caller's own overrides."""
    p = dict(PARAMS)
    overrides = dict(overrides or {})
    market = str(overrides.get("market", p["market"])).lower()
    if market not in MARKETS:
        raise BacktestError(f"market must be one of {', '.join(MARKETS)}, got {market!r}")
    p.update(MARKETS[market])
    p["market"] = market
    for k, v in overrides.items():
        if k not in p:
            raise BacktestError(
                f"unknown swingpro parameter {k!r}. Known: {', '.join(sorted(p))}")
        d = p[k]
        if isinstance(d, str):
            p[k] = str(v)
        elif isinstance(d, int) and not isinstance(d, bool):
            p[k] = int(float(v))
        else:
            p[k] = float(v)
    return p


# ------------------------------------------------------------- external data
def _ext_bars(symbol: str, interval: str):
    """Benchmark / VIX / lower-timeframe bars, or None if they cannot be had.

    Unlike the desktop version there is no process-lifetime memo here: this
    runs in a long-lived server, where one would pin a stale benchmark for as
    long as the worker lives. data_source already caches with a TTL, so calling
    straight through is both cheap and correct.

    A gate whose data is missing fails closed, which is why every caller treats
    None as "that filter blocks" rather than "that filter is off".
    """
    try:
        return D.load(symbol, interval).df
    except Exception:
        return None


def _naive(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize(None) if idx.tz is not None else idx


def _align(src, idx) -> np.ndarray:
    """Carry an external series onto our bar index, last known value only."""
    if src is None or len(src) == 0:
        return np.full(len(idx), np.nan)
    s = pd.Series(np.asarray(src, float), index=_naive(src.index)).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    tgt = _naive(idx)
    out = s.reindex(s.index.union(tgt)).ffill().reindex(tgt)
    return out.to_numpy(float)


def _flow_delta(ticker: str, df: pd.DataFrame, interval: str, p: dict) -> np.ndarray:
    """Per-bar signed volume delta as a fraction of that bar's volume."""
    mode = str(p["flow_mode"]).lower()
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    rng = np.where(h > l, h - l, np.nan)
    proxy = ((c - l) - (h - c)) / rng

    if mode == "off":
        return np.full(len(df), np.nan)
    if mode == "proxy":
        return proxy

    ltf = p["ltf"] or _LTF_FOR.get(interval, "")
    ltf_delta = np.full(len(df), np.nan)
    if ltf and interval == "1d":
        b = _ext_bars(ticker, ltf)
        if b is not None and len(b) > 0:
            bc = b["Close"].to_numpy(float)
            bo = b["Open"].to_numpy(float)
            bv = b["Volume"].to_numpy(float)
            up = np.where(bc >= bo, bv, 0.0)
            dn = np.where(bc < bo, bv, 0.0)
            day = pd.DatetimeIndex(b.index).date
            g = pd.DataFrame({"up": up, "dn": dn}).groupby(day).sum()
            tot = g["up"] + g["dn"]
            d = ((g["up"] - g["dn"]) / tot.where(tot > 0)).to_dict()
            ltf_delta = np.array([d.get(t, np.nan)
                                  for t in pd.DatetimeIndex(df.index).date], float)

    if mode == "ltf":
        return ltf_delta
    return np.where(np.isnan(ltf_delta), proxy, ltf_delta)   # auto


# -------------------------------------------------------------------- state
@dataclass
class Signals:
    long: np.ndarray
    short: np.ndarray
    stop_long: np.ndarray       # zone-anchored stop price for a long entered here
    stop_short: np.ndarray
    atr: np.ndarray
    chand_long: np.ndarray
    chand_short: np.ndarray
    demand: np.ndarray
    supply: np.ndarray
    delta: np.ndarray
    diag: dict = field(default_factory=dict)


def compute_signals(df: pd.DataFrame, p: dict, ticker: str, interval: str) -> Signals:
    n = len(df)
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    v = df["Volume"].to_numpy(float)

    atr = ind.atr(df["High"], df["Low"], df["Close"], 14).to_numpy(float)
    atr_pct = atr / c * 100.0
    vol_sma = ind.sma(df["Volume"], 20).to_numpy(float)
    ema_f = ind.ema(df["Close"], p["ema_fast"]).to_numpy(float)
    ema_m = ind.ema(df["Close"], p["ema_mid"]).to_numpy(float)
    ema_s = ind.ema(df["Close"], p["ema_slow"]).to_numpy(float)
    turnover = ind.sma(df["Close"] * df["Volume"], 20).to_numpy(float) / p["turnover_unit"]

    # --- benchmark / relative strength / VIX ------------------------------
    bench = _ext_bars(p["benchmark"], interval) if (p["use_regime"] or p["use_rs"]) else None
    if bench is not None:
        b_close = _align(bench["Close"], df.index)
        b_ema = _align(ind.ema(bench["Close"], p["bench_ema"]), df.index)
        b_roc = _align(ind.roc(bench["Close"], p["rs_len"]), df.index)
    else:
        b_close = b_ema = b_roc = np.full(n, np.nan)

    regime_up = (b_close > b_ema) if p["use_regime"] else np.ones(n, bool)
    regime_dn = (b_close < b_ema) if p["use_regime"] else np.ones(n, bool)

    rs = ind.roc(df["Close"], p["rs_len"]).to_numpy(float) - np.nan_to_num(b_roc)
    rs_ok_l = (rs > 0) if p["use_rs"] else np.ones(n, bool)
    rs_ok_s = (rs < 0) if p["use_rs"] else np.ones(n, bool)

    vix = np.full(n, np.nan)
    vix_ok = np.ones(n, bool)
    if p["use_vix"]:
        vb = _ext_bars(p["vix_symbol"], interval)
        if vb is not None:
            vix = _align(vb["Close"], df.index)
        vix_ok = np.isnan(vix) | (vix <= p["vix_max"])

    # --- event guard: react to an abnormal gap, stand down while it settles
    prev_c = np.concatenate([[np.nan], c[:-1]])
    with np.errstate(invalid="ignore"):
        gap_now = np.abs(o - prev_c) / prev_c * 100.0 >= p["gap_pct"]
    gap_ok = np.ones(n, bool)
    last_gap = -(10 ** 9)
    for i in range(n):
        if gap_now[i]:
            last_gap = i
        gap_ok[i] = (p["gap_cool"] == 0) or (i - last_gap > p["gap_cool"])

    # --- order-flow proxy --------------------------------------------------
    delta = _flow_delta(ticker, df, interval, p)
    thr = p["delta_thr"]
    if p["use_flow"]:
        flow_l = ~np.isnan(delta) & (delta >= thr)
        flow_s = ~np.isnan(delta) & (delta <= -thr)
    else:
        flow_l = flow_s = np.ones(n, bool)

    # --- rolling extremes --------------------------------------------------
    L = p["vp_lookback"]
    bins = p["vp_bins"]
    tl = p["trail_len"]
    vp_hh = pd.Series(h).rolling(L, min_periods=L).max().to_numpy(float)
    vp_ll = pd.Series(l).rolling(L, min_periods=L).min().to_numpy(float)
    chand_l = pd.Series(h).rolling(tl, min_periods=tl).max().to_numpy(float) - atr * p["trail_atr"]
    chand_s = pd.Series(l).rolling(tl, min_periods=tl).min().to_numpy(float) + atr * p["trail_atr"]

    ph = ind.pivot_high(df["High"], p["piv_len"], p["piv_len"]).to_numpy(float)
    pl = ind.pivot_low(df["Low"], p["piv_len"], p["piv_len"]).to_numpy(float)

    disp_up = (c > o) & ((c - o) > atr * p["ob_disp_atr"]) & (v > vol_sma * p["ob_vol_mult"])
    disp_dn = (c < o) & ((o - c) > atr * p["ob_disp_atr"]) & (v > vol_sma * p["ob_vol_mult"])

    bull_bar = np.zeros(n, bool)
    bear_bar = np.zeros(n, bool)
    bull_bar[1:] = (c[1:] > o[1:]) & (c[1:] > h[:-1]) & (c[1:] > (h[1:] + l[1:]) / 2)
    bear_bar[1:] = (c[1:] < o[1:]) & (c[1:] < l[:-1]) & (c[1:] < (h[1:] + l[1:]) / 2)

    # --- bar-state carriers ------------------------------------------------
    prof = np.zeros(bins)
    vp_lo = np.nan
    vp_bin = np.nan
    lvl_off = np.arange(bins) + 0.5

    zones: list[dict] = []              # live order blocks
    sr_lvl: list[float] = []            # clustered pivot levels
    sr_hit: list[int] = []

    long_sig = np.zeros(n, bool)
    short_sig = np.zeros(n, bool)
    stop_l = np.full(n, np.nan)
    stop_s = np.full(n, np.nan)
    demand = np.zeros(n, int)
    supply = np.zeros(n, int)
    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)

    for i in range(n):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        buf = a * p["zone_buf_atr"]

        # ---------------- volume profile ----------------------------------
        bs = (vp_hh[i] - vp_ll[i]) / bins if not np.isnan(vp_hh[i]) else np.nan
        if i > L + 5 and i % p["vp_update"] == 0 and not np.isnan(bs) and bs > 0:
            prof[:] = 0.0
            lo = vp_ll[i]
            for j in range(i - L + 1, i + 1):
                b_lo = int(min(bins - 1, max(0, math.floor((l[j] - lo) / bs))))
                b_hi = int(min(bins - 1, max(0, math.floor((h[j] - lo) / bs))))
                prof[b_lo:b_hi + 1] += v[j] / (b_hi - b_lo + 1)
            vp_lo = lo
            vp_bin = bs

        hvn_dn = hvn_up = np.nan
        if not np.isnan(vp_bin) and vp_bin > 0 and prof.sum() > 0:
            total = prof.sum()
            poc_i = int(np.argmax(prof))
            target = total * p["va_pct"]
            acc = prof[poc_i]
            up, dn = poc_i + 1, poc_i - 1
            while acc < target and (up < bins or dn >= 0):
                vu = prof[up] if up < bins else -1.0
                vd = prof[dn] if dn >= 0 else -1.0
                if vu >= vd:
                    acc += vu
                    up += 1
                else:
                    acc += vd
                    dn -= 1
            poc[i] = vp_lo + (poc_i + 0.5) * vp_bin
            vah[i] = vp_lo + up * vp_bin
            val[i] = vp_lo + (dn + 1) * vp_bin

            levels = vp_lo + lvl_off * vp_bin
            hot = prof >= (total / bins) * p["hvn_mult"]
            below = levels[hot & (levels <= c[i])]
            above = levels[hot & (levels >= c[i])]
            hvn_dn = below.max() if below.size else np.nan
            hvn_up = above.min() if above.size else np.nan

        # ---------------- order blocks ------------------------------------
        for is_bull, fired in ((True, disp_up[i]),
                               (False, disp_dn[i] and bool(p["allow_short"]))):
            if not fired:
                continue
            src = -1
            for k in range(1, p["ob_search"] + 1):
                if i - k < 0:
                    break
                opposite = (c[i - k] < o[i - k]) if is_bull else (c[i - k] > o[i - k])
                if opposite:
                    src = i - k
                    break
            if src >= 0:
                zones.append(dict(top=h[src], bot=l[src], born=i, bull=is_bull))
                while len(zones) > p["ob_max_count"]:
                    zones.pop(0)

        for z in list(zones):
            dead = (i - z["born"]) > p["ob_max_age"]
            if z["bull"] and c[i] < z["bot"]:
                dead = True
            if (not z["bull"]) and c[i] > z["top"]:
                dead = True
            if dead:
                zones.remove(z)

        in_bull = in_bear = False
        bull_bot = bear_top = np.nan
        for z in zones:
            if z["bull"] and (l[i] <= z["top"] + buf) and (l[i] >= z["bot"] - buf):
                in_bull = True
                bull_bot = z["bot"] if np.isnan(bull_bot) else min(bull_bot, z["bot"])
            if (not z["bull"]) and (h[i] >= z["bot"] - buf) and (h[i] <= z["top"] + buf):
                in_bear = True
                bear_top = z["top"] if np.isnan(bear_top) else max(bear_top, z["top"])

        # ---------------- clustered pivot S/R ------------------------------
        for pv in (ph[i], pl[i]):
            if np.isnan(pv):
                continue
            tol = a * p["sr_tol_atr"]
            found = -1
            for k in range(len(sr_lvl)):
                if abs(sr_lvl[k] - pv) <= tol:
                    found = k
                    break
            if found >= 0:
                m = sr_hit[found]
                sr_lvl[found] = (sr_lvl[found] * m + pv) / (m + 1)
                sr_hit[found] = m + 1
            else:
                sr_lvl.append(float(pv))
                sr_hit.append(1)
            while len(sr_lvl) > p["sr_max"]:
                sr_lvl.pop(0)
                sr_hit.pop(0)

        sr_up = sr_dn = np.nan
        for k in range(len(sr_lvl)):
            if sr_hit[k] < p["sr_min_touch"]:
                continue
            lv = sr_lvl[k]
            if lv > c[i]:
                sr_up = lv if np.isnan(sr_up) else min(sr_up, lv)
            elif lv < c[i]:
                sr_dn = lv if np.isnan(sr_dn) else max(sr_dn, lv)

        sr_sup = (not np.isnan(sr_dn)) and l[i] <= sr_dn + buf and c[i] > sr_dn
        sr_res = (not np.isnan(sr_up)) and h[i] >= sr_up - buf and c[i] < sr_up

        # ---------------- confluence ---------------------------------------
        at_hvn_dn = (not np.isnan(hvn_dn)) and l[i] <= hvn_dn + buf and c[i] > hvn_dn
        at_val = (not np.isnan(val[i])) and l[i] <= val[i] + buf and c[i] > val[i]
        at_hvn_up = (not np.isnan(hvn_up)) and h[i] >= hvn_up - buf and c[i] < hvn_up
        at_vah = (not np.isnan(vah[i])) and h[i] >= vah[i] - buf and c[i] < vah[i]

        dem = int(in_bull) + int(at_hvn_dn) + int(at_val) + int(sr_sup)
        sup = int(in_bear) + int(at_hvn_up) + int(at_vah) + int(sr_res)
        demand[i] = dem
        supply[i] = sup

        # ---------------- filters ------------------------------------------
        circuit_ok = (not p["skip_circuit"]) or (h[i] != l[i])
        liquid_ok = (turnover[i] >= p["min_turnover"]
                     and c[i] >= p["min_price"] and circuit_ok)
        if not (liquid_ok and atr_pct[i] <= p["max_atr_pct"]
                and gap_ok[i] and i > L + 20):
            continue

        trend_up = c[i] > ema_s[i] and ema_f[i] > ema_m[i]
        trend_dn = c[i] < ema_s[i] and ema_f[i] < ema_m[i]

        if (trend_up and regime_up[i] and rs_ok_l[i] and vix_ok[i]
                and dem >= p["min_confluence"] and bull_bar[i] and flow_l[i]):
            hd = l[i] if np.isnan(hvn_dn) else hvn_dn
            anchor = min(l[i], hd) if np.isnan(bull_bot) else min(bull_bot, l[i])
            sp = anchor - a * p["stop_atr"]
            if c[i] - sp > 0:
                long_sig[i] = True
                stop_l[i] = sp

        if (p["allow_short"] and trend_dn and regime_dn[i] and rs_ok_s[i]
                and sup >= p["min_confluence"] and bear_bar[i] and flow_s[i]):
            hu = h[i] if np.isnan(hvn_up) else hvn_up
            anchor = max(h[i], hu) if np.isnan(bear_top) else max(bear_top, h[i])
            sp = anchor + a * p["stop_atr"]
            if sp - c[i] > 0:
                short_sig[i] = True
                stop_s[i] = sp

    return Signals(
        long=long_sig, short=short_sig, stop_long=stop_l, stop_short=stop_s,
        atr=atr, chand_long=chand_l, chand_short=chand_s,
        demand=demand, supply=supply, delta=delta,
        diag=dict(poc=poc, vah=vah, val=val, rs=rs, turnover=turnover,
                  atr_pct=atr_pct, vix=vix),
    )


# ----------------------------------------------------------------- execution
def simulate(df: pd.DataFrame, sig: Signals, p: dict, cfg: E.Config,
             ticker: str, interval: str, name: str) -> E.Result:
    """Hand the signals to the shared risk engine (bt/riskengine.py)."""
    plan = RE.Plan(long=sig.long, short=sig.short,
                   stop_long=sig.stop_long, stop_short=sig.stop_short,
                   trail_long=sig.chand_long, trail_short=sig.chand_short)
    risk = RE.Risk(risk_pct=p["risk_pct"], max_expo_pct=p["max_expo_pct"],
                   rr1=p["rr1"], scale_out=0.5, max_hold=p["max_hold"],
                   breakeven=True, trail_after_t1=True)
    return RE.simulate(df, plan, risk, cfg, ticker, interval, name)


def run(df: pd.DataFrame, overrides: dict | None, cfg: E.Config,
        ticker: str, interval: str, name: str = "swingpro_in"):
    """Entry point used by run.py. Returns (Result, resolved params)."""
    p = _norm(overrides)
    if cfg.allow_short and "allow_short" not in (overrides or {}):
        p["allow_short"] = 1          # --short on the CLI
    sig = compute_signals(df, p, ticker, interval)
    res = simulate(df, sig, p, cfg, ticker, interval, name)
    res.signals = sig
    return res, p
