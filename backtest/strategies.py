"""Strategy definitions.

Every strategy is a function that takes the OHLCV frame plus keyword params and
returns a target-position Series in {-1, 0, +1}, aligned to the frame's index.

RULE: use only information available at that bar's close. The engine adds the
one-bar execution delay for you - do NOT shift signals yourself, or you will
double-count the delay and understate the strategy.

To port a Pine Script strategy, add a function here, decorate it with
@strategy("your_name", param=default, ...) and it appears in the Terminal's
strategy picker automatically. The mapping is nearly one-to-one:

    Pine                          ->  here
    ta.ema(close, 20)             ->  ind.ema(df["Close"], 20)
    ta.rsi(close, 14)             ->  ind.rsi(df["Close"], 14)
    ta.atr(14)                    ->  ind.atr(df["High"], df["Low"], df["Close"], 14)
    ta.crossover(a, b)            ->  ind.crossover(a, b)
    strategy.entry("L", long)     ->  set the target Series to +1 from here on
    strategy.close("L")           ->  set it back to 0
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .errors import BacktestError

REGISTRY: dict[str, dict] = {}


def strategy(name: str, **default_params):
    def deco(fn):
        REGISTRY[name] = {"fn": fn, "params": default_params, "doc": (fn.__doc__ or "").strip()}
        return fn
    return deco


def risk_strategy(name: str, **default_params):
    """Register a strategy that sizes by risk instead of by a fixed slice.

    The function returns (Plan, Risk) from bt/riskengine.py rather than a
    target-position Series: per-bar entry flags plus the stop price each entry
    implies. Size, the scale-out and the trail are the engine's job.
    """
    def deco(fn):
        REGISTRY[name] = {"fn": fn, "params": default_params,
                          "doc": (fn.__doc__ or "").strip(), "engine": "risk"}
        return fn
    return deco


def _state_from_events(index, enter_long=None, exit_long=None,
                       enter_short=None, exit_short=None) -> pd.Series:
    """Turn discrete entry/exit events into a held position, the way Pine does:
    a position stays on until something closes it."""
    n = len(index)
    el = enter_long.to_numpy(bool) if enter_long is not None else np.zeros(n, bool)
    xl = exit_long.to_numpy(bool) if exit_long is not None else np.zeros(n, bool)
    es = enter_short.to_numpy(bool) if enter_short is not None else np.zeros(n, bool)
    xs = exit_short.to_numpy(bool) if exit_short is not None else np.zeros(n, bool)
    out = np.zeros(n, float)
    pos = 0
    for i in range(n):
        if pos == 1 and xl[i]:
            pos = 0
        elif pos == -1 and xs[i]:
            pos = 0
        if pos == 0:
            if el[i]:
                pos = 1
            elif es[i]:
                pos = -1
        elif pos == 1 and es[i]:
            pos = -1
        elif pos == -1 and el[i]:
            pos = 1
        out[i] = pos
    return pd.Series(out, index=index)


# --------------------------------------------------------------------------
# Reference strategies. Replace or add your own.
# --------------------------------------------------------------------------

@strategy("ema_cross", fast=20, slow=50)
def ema_cross(df, fast=20, slow=50):
    """Long when the fast EMA is above the slow EMA, short when below."""
    f, s = ind.ema(df["Close"], int(fast)), ind.ema(df["Close"], int(slow))
    return pd.Series(np.where(f > s, 1.0, np.where(f < s, -1.0, 0.0)), index=df.index).where(s.notna(), 0.0)


@strategy("sma_cross", fast=50, slow=200)
def sma_cross(df, fast=50, slow=200):
    """The golden cross. Long above, flat below."""
    f, s = ind.sma(df["Close"], int(fast)), ind.sma(df["Close"], int(slow))
    return pd.Series(np.where(f > s, 1.0, 0.0), index=df.index).where(s.notna(), 0.0)


@strategy("rsi", period=14, lower=30, upper=70)
def rsi_reversion(df, period=14, lower=30, upper=70):
    """Buy oversold, exit when RSI recovers past the upper band."""
    r = ind.rsi(df["Close"], int(period))
    return _state_from_events(df.index, enter_long=ind.crossover(r, float(lower)),
                              exit_long=ind.crossover(r, float(upper)))


@strategy("macd", fast=12, slow=26, signal=9)
def macd_strategy(df, fast=12, slow=26, signal=9):
    """Long on MACD line crossing above its signal line, flat on the cross down."""
    line, sig, _ = ind.macd(df["Close"], int(fast), int(slow), int(signal))
    return _state_from_events(df.index, enter_long=ind.crossover(line, sig),
                              exit_long=ind.crossunder(line, sig))


@strategy("supertrend", period=10, mult=3.0)
def supertrend_strategy(df, period=10, mult=3.0):
    """Follow the SuperTrend direction flip."""
    _, d = ind.supertrend(df["High"], df["Low"], df["Close"], int(period), float(mult))
    return d.replace(0.0, np.nan).ffill().fillna(0.0)


@strategy("donchian", entry=20, exit=10)
def donchian_breakout(df, entry=20, exit=10):
    """Turtle-style breakout: buy the N-bar high, exit on the M-bar low."""
    _, hi = ind.donchian(df["High"], df["Low"], int(entry))
    lo, _ = ind.donchian(df["High"], df["Low"], int(exit))
    return _state_from_events(
        df.index,
        enter_long=df["Close"] > hi.shift(1),
        exit_long=df["Close"] < lo.shift(1),
    )


@strategy("bbands", period=20, k=2.0)
def bbands_reversion(df, period=20, k=2.0):
    """Buy a close below the lower band, exit back at the middle band."""
    lo, mid, _ = ind.bbands(df["Close"], int(period), float(k))
    return _state_from_events(df.index, enter_long=ind.crossunder(df["Close"], lo),
                              exit_long=ind.crossover(df["Close"], mid))


TF_SECONDS = {
    "1m": 60, "2m": 120, "5m": 300, "15m": 900, "30m": 1800,
    "60m": 3600, "1h": 3600, "90m": 5400,
    "1d": 86_400, "5d": 432_000, "1wk": 604_800, "1mo": 2_592_000, "3mo": 7_776_000,
}


@strategy("ema_ribbon", len1=20, len2=50, len3=100, len4=200,
          min_strength=3, exit_strength=1, strict_below_min=30)
def ema_ribbon(df, len1=20, len2=50, len3=100, len4=200,
               min_strength=3, exit_strength=1, strict_below_min=30):
    """EMA 20/50/100/200 ribbon graded 0-4; demands the full stack below 30m."""
    e1 = ind.ema(df["Close"], int(len1))
    e2 = ind.ema(df["Close"], int(len2))
    e3 = ind.ema(df["Close"], int(len3))
    e4 = ind.ema(df["Close"], int(len4))
    c = df["Close"]

    # One point per alignment that has clicked into place.
    bull = ((c > e1).astype(int) + (e1 > e2).astype(int)
            + (e2 > e3).astype(int) + (e3 > e4).astype(int))
    bear = ((c < e1).astype(int) + (e1 < e2).astype(int)
            + (e2 < e3).astype(int) + (e3 < e4).astype(int))
    strength = (bull - bear).where(e4.notna(), 0).to_numpy()

    # Below the cutoff the ribbon must be perfectly stacked; above it, graded.
    interval = df.attrs.get("interval", "1d")
    tf_sec = TF_SECONDS.get(interval, 86_400)
    req = 4 if tf_sec < int(strict_below_min) * 60 else int(min_strength)
    exit_at = int(exit_strength)

    warm = int(max(len1, len2, len3, len4))
    n = len(df)
    long_sig = np.zeros(n, bool)
    short_sig = np.zeros(n, bool)
    ok = np.arange(n) > warm
    long_sig[ok] = strength[ok] >= req
    short_sig[ok] = strength[ok] <= -req

    # Entry needs a fresh trigger; the exit works off the live strength level.
    out = np.zeros(n, float)
    state = 0
    for i in range(1, n):
        ls, ss = long_sig[i], short_sig[i]
        fresh_l = ls and not long_sig[i - 1]
        fresh_s = ss and not short_sig[i - 1]
        if state == 0:
            if fresh_l:
                state = 1
            elif fresh_s:
                state = -1
        elif state == 1:
            if strength[i] <= exit_at:
                state = 0
            if fresh_s:
                state = -1
        else:
            if strength[i] >= -exit_at:
                state = 0
            if fresh_l:
                state = 1
        out[i] = state
    return pd.Series(out, index=df.index)


@strategy("buyhold")
def buy_and_hold(df):
    """Benchmark: in the market from the first bar to the last."""
    return pd.Series(1.0, index=df.index)


@strategy(
    "swing_pro_in",
    ema_fast=21, ema_mid=50, ema_slow=200, bench_ema=50, rs_len=60,
    use_regime=1, use_rs=1, use_vix=1, vix_max=22.0,
    vp_lookback=150, vp_bins=24, vp_update=5, va_pct=0.70, hvn_mult=1.6,
    ob_disp_atr=1.0, ob_vol_mult=1.5, ob_search=5, ob_max_age=60,
    ob_max_count=8, delta_thr=0.08, piv_len=8, sr_tol_atr=0.5,
    sr_max=12, sr_min_touch=2, zone_buf_atr=0.25, stop_atr=0.6,
    rr1=1.5, trail_len=22, trail_atr=2.5, max_hold=45, min_turn_cr=5.0,
    min_price=20.0, max_atr_pct=9.0, gap_pct=8.0, gap_cool=3,
)
def swing_pro_in(df, **p):
    """Swing Pro IN signal port; daily candle volume is the LTF-flow fallback."""
    close, high, low, open_, volume = (df[c].astype(float) for c in ("Close", "High", "Low", "Open", "Volume"))
    atr = ind.atr(high, low, close, 14)
    ema_fast = ind.ema(close, int(p["ema_fast"]))
    ema_mid = ind.ema(close, int(p["ema_mid"]))
    ema_slow = ind.ema(close, int(p["ema_slow"]))
    vol_sma = ind.sma(volume, 20)
    turn_cr = ind.sma(close * volume, 20) / 10_000_000.0
    atr_pct = atr / close * 100.0

    # Optional columns are populated by callers that can download the external
    # symbols. Missing columns deliberately fail the corresponding gate closed.
    bench_close = df.get("BenchmarkClose", pd.Series(np.nan, index=df.index))
    bench_ema = df.get("BenchmarkEMA", pd.Series(np.nan, index=df.index))
    vix = df.get("IndiaVIX", pd.Series(np.nan, index=df.index))
    rs = ind.roc(close, int(p["rs_len"])) - ind.roc(bench_close, int(p["rs_len"]))

    n = len(df)
    entries = np.zeros(n, bool)
    exits = np.zeros(n, bool)
    position = np.zeros(n, float)
    zones = []
    levels = []
    level_hits = []
    stop = target = trail_ref = np.nan
    held_from = None
    in_position = False
    vp = (np.nan, np.nan, np.nan, np.nan, [])

    def nearest_level(above, at):
        candidates = [level for level, hits in zip(levels, level_hits)
                      if hits >= int(p["sr_min_touch"])
                      and ((above and level > at) or (not above and level < at))]
        return (min(candidates) if above else max(candidates)) if candidates else np.nan

    def profile(start, end):
        window_low = float(low.iloc[start:end + 1].min())
        window_high = float(high.iloc[start:end + 1].max())
        width = (window_high - window_low) / int(p["vp_bins"])
        if not np.isfinite(width) or width <= 0:
            return np.nan, np.nan, np.nan, np.nan, []
        rows = np.zeros(int(p["vp_bins"]), float)
        for bar in range(start, end + 1):
            lo_bin = int(np.clip(np.floor((low.iloc[bar] - window_low) / width), 0, len(rows) - 1))
            hi_bin = int(np.clip(np.floor((high.iloc[bar] - window_low) / width), 0, len(rows) - 1))
            rows[lo_bin:hi_bin + 1] += volume.iloc[bar] / (hi_bin - lo_bin + 1)
        poc = int(rows.argmax())
        wanted = rows.sum() * float(p["va_pct"])
        accumulated, upper, lower = rows[poc], poc + 1, poc - 1
        while accumulated < wanted and (upper < len(rows) or lower >= 0):
            upper_volume = rows[upper] if upper < len(rows) else -1.0
            lower_volume = rows[lower] if lower >= 0 else -1.0
            if upper_volume >= lower_volume:
                accumulated += upper_volume; upper += 1
            else:
                accumulated += lower_volume; lower -= 1
        hvns = [window_low + (i + 0.5) * width for i, value in enumerate(rows)
                if value >= rows.mean() * float(p["hvn_mult"])]
        return (window_low + (poc + 0.5) * width, window_low + upper * width,
                window_low + (lower + 1) * width, width, hvns)

    for i in range(n):
        if i >= int(p["vp_lookback"]) and i % int(p["vp_update"]) == 0:
            vp = profile(i - int(p["vp_lookback"]) + 1, i)

        if i >= 1 and i >= int(p["piv_len"]):
            pivot_i = i - int(p["piv_len"])
            pivot_high = high.iloc[pivot_i] == high.iloc[pivot_i - int(p["piv_len"]):i + 1].max()
            pivot_low = low.iloc[pivot_i] == low.iloc[pivot_i - int(p["piv_len"]):i + 1].min()
            for level in ([high.iloc[pivot_i]] if pivot_high else []) + ([low.iloc[pivot_i]] if pivot_low else []):
                tolerance = float(atr.iloc[i]) * float(p["sr_tol_atr"]) if np.isfinite(atr.iloc[i]) else 0.0
                match = next((j for j, old in enumerate(levels) if abs(old - level) <= tolerance), None)
                if match is None:
                    levels.append(float(level)); level_hits.append(1)
                else:
                    level_hits[match] += 1
                    levels[match] = (levels[match] * (level_hits[match] - 1) + level) / level_hits[match]
                while len(levels) > int(p["sr_max"]):
                    levels.pop(0); level_hits.pop(0)

        if i >= 1 and np.isfinite(atr.iloc[i]) and np.isfinite(vol_sma.iloc[i]):
            displacement_up = close.iloc[i] > open_.iloc[i] and close.iloc[i] - open_.iloc[i] > atr.iloc[i] * float(p["ob_disp_atr"]) and volume.iloc[i] > vol_sma.iloc[i] * float(p["ob_vol_mult"])
            displacement_down = close.iloc[i] < open_.iloc[i] and open_.iloc[i] - close.iloc[i] > atr.iloc[i] * float(p["ob_disp_atr"]) and volume.iloc[i] > vol_sma.iloc[i] * float(p["ob_vol_mult"])
            if displacement_up or displacement_down:
                for back in range(1, min(int(p["ob_search"]), i) + 1):
                    opposite = (close.iloc[i - back] < open_.iloc[i - back]) if displacement_up else (close.iloc[i - back] > open_.iloc[i - back])
                    if opposite:
                        zones.append({"top": high.iloc[i - back], "bot": low.iloc[i - back], "born": i, "bull": displacement_up})
                        break
                zones = [z for z in zones if i - z["born"] <= int(p["ob_max_age"])]
                zones = zones[-int(p["ob_max_count"]):]

        if i == 0 or not np.isfinite(atr.iloc[i]):
            continue
        buffer_ = atr.iloc[i] * float(p["zone_buf_atr"])
        zones = [z for z in zones if not ((z["bull"] and close.iloc[i] < z["bot"]) or (not z["bull"] and close.iloc[i] > z["top"]))]
        bull_zone = [z["bot"] for z in zones if z["bull"] and low.iloc[i] <= z["top"] + buffer_ and low.iloc[i] >= z["bot"] - buffer_]
        bear_zone = [z["top"] for z in zones if not z["bull"] and high.iloc[i] >= z["bot"] - buffer_ and high.iloc[i] <= z["top"] + buffer_]
        poc, vah, val, width, hvns = vp
        hvn_down = max((x for x in hvns if x <= close.iloc[i]), default=np.nan)
        hvn_up = min((x for x in hvns if x >= close.iloc[i]), default=np.nan)
        support, resistance = nearest_level(False, close.iloc[i]), nearest_level(True, close.iloc[i])
        demand = sum([bool(bull_zone), np.isfinite(hvn_down) and low.iloc[i] <= hvn_down + buffer_ and close.iloc[i] > hvn_down,
                      np.isfinite(val) and low.iloc[i] <= val + buffer_ and close.iloc[i] > val,
                      np.isfinite(support) and low.iloc[i] <= support + buffer_ and close.iloc[i] > support])
        supply = sum([bool(bear_zone), np.isfinite(hvn_up) and high.iloc[i] >= hvn_up - buffer_ and close.iloc[i] < hvn_up,
                      np.isfinite(vah) and high.iloc[i] >= vah - buffer_ and close.iloc[i] < vah,
                      np.isfinite(resistance) and high.iloc[i] >= resistance - buffer_ and close.iloc[i] < resistance])
        flow = (volume.iloc[i] if close.iloc[i] >= open_.iloc[i] else -volume.iloc[i]) / volume.iloc[i] if volume.iloc[i] > 0 else np.nan
        gap = abs(open_.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1] * 100.0
        recent_gap = any(abs(open_.iloc[j] - close.iloc[j - 1]) / close.iloc[j - 1] * 100.0 >= float(p["gap_pct"])
                         for j in range(max(1, i - int(p["gap_cool"])), i + 1))
        base = (np.isfinite(turn_cr.iloc[i]) and turn_cr.iloc[i] >= float(p["min_turn_cr"]) and close.iloc[i] >= float(p["min_price"])
                and atr_pct.iloc[i] <= float(p["max_atr_pct"]) and i > int(p["vp_lookback"]) + 20 and not recent_gap)
        trend_up = close.iloc[i] > ema_slow.iloc[i] and ema_fast.iloc[i] > ema_mid.iloc[i]
        regime_up = not int(p["use_regime"]) or (np.isfinite(bench_close.iloc[i]) and bench_close.iloc[i] > bench_ema.iloc[i])
        rs_ok = not int(p["use_rs"]) or (np.isfinite(rs.iloc[i]) and rs.iloc[i] > 0)
        vix_ok = not int(p["use_vix"]) or not np.isfinite(vix.iloc[i]) or vix.iloc[i] <= float(p["vix_max"])
        enter = base and trend_up and regime_up and rs_ok and vix_ok and demand >= 2 and close.iloc[i] > open_.iloc[i] and close.iloc[i] > high.iloc[i - 1] and flow >= float(p["delta_thr"])
        if not in_position and enter:
            entries[i] = True; in_position = True; held_from = i
            anchor = min([low.iloc[i]] + bull_zone + ([hvn_down] if np.isfinite(hvn_down) else []))
            stop = anchor - atr.iloc[i] * float(p["stop_atr"])
            risk = close.iloc[i] - stop
            target = close.iloc[i] + risk * float(p["rr1"])
            trail_ref = close.iloc[i]
        elif in_position:
            trail_ref = max(trail_ref, high.iloc[i])
            stop = max(stop, trail_ref - atr.iloc[i] * float(p["trail_atr"])) if i - held_from >= 1 else stop
            hit = low.iloc[i] <= stop or high.iloc[i] >= target or (int(p["max_hold"]) > 0 and i - held_from >= int(p["max_hold"]))
            if hit:
                exits[i] = True; in_position = False; stop = target = trail_ref = np.nan; held_from = None
        position[i] = 1.0 if in_position else 0.0

    return pd.Series(position, index=df.index)


def get(name: str):
    if name not in REGISTRY:
        raise BacktestError(f"Unknown strategy {name!r}. Available: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[name]


def resolve(name: str, overrides: dict | None = None) -> tuple[dict, dict]:
    """(registry entry, parameters with overrides applied)."""
    spec = get(name)
    params = dict(spec["params"])
    for k, v in (overrides or {}).items():
        if k not in params:
            raise BacktestError(f"{name} has no parameter {k!r}. Valid: {', '.join(params) or '(none)'}")
        params[k] = v
    return spec, params


def build(name: str, df: pd.DataFrame, overrides: dict | None = None) -> tuple[pd.Series, dict]:
    spec, params = resolve(name, overrides)
    sig = spec["fn"](df, **params)
    return sig.reindex(df.index).fillna(0.0), params


# ---------------------------------------------------------------------------
# Strategies that own their execution (risk sizing, scale-outs, per-leg exits)
# cannot be expressed as a {-1,0,+1} target series. They register an "engine"
# key and service.py hands the whole backtest to that module instead.
from . import swingpro as _swingpro          # noqa: E402

REGISTRY["swingpro_in"] = {
    "fn": None,
    "params": _swingpro.PARAMS,
    "doc": "Swing Pro IN - volume profile + order blocks + flow proxy, "
           "risk-sized, half off at 1.5R then Chandelier trail.",
    "engine": "swingpro",
    "costs": _swingpro.COSTS,
}


# ---------------------------------------------------------------------------
# The two worked examples that ship with the desktop tester's mystrategies.py.
# They live in the registry here rather than in a file on disk: there is no
# project root to import from inside a web service, and they are the templates
# both styles are explained by.
# ---------------------------------------------------------------------------

@strategy("ema_pullback", fast=21, slow=200, rsi_len=14, rsi_max=45, dip_within=10)
def ema_pullback(df, fast=21, slow=200, rsi_len=14, rsi_max=45, dip_within=10):
    """Buy the recovery from a pullback in an uptrend; exit back under the fast EMA."""
    close = df["Close"]
    ema_fast = ind.ema(close, int(fast))
    ema_slow = ind.ema(close, int(slow))
    rsi = ind.rsi(close, int(rsi_len))

    uptrend = close > ema_slow
    # The dip and the recovery are separate events - asking for both on the
    # same bar is close to a contradiction, and would give you zero trades.
    dipped = (rsi < float(rsi_max)).rolling(int(dip_within), min_periods=1).max().fillna(0).astype(bool)

    return _state_from_events(
        df.index,
        enter_long=uptrend & dipped & ind.crossover(close, ema_fast),
        exit_long=ind.crossunder(close, ema_fast),
    )


@risk_strategy(
    "atr_breakout",
    entry_len=55,          # breakout lookback
    trend_len=200,         # only trade above this EMA
    stop_lookback=10,      # the swing low the stop hides under
    stop_atr=1.0,          # ... plus this much ATR of room
    rr=2.0,                # take half off here, in R
    trail_atr=3.0,         # Chandelier distance for the runner
    trail_len=22,
    risk_pct=1.0,          # % of equity between entry and stop
    max_expo_pct=25.0,     # hard cap on position size
    max_hold=0,            # bars; 0 = no time stop
)
def atr_breakout(df, entry_len=55, trend_len=200, stop_lookback=10, stop_atr=1.0,
                 rr=2.0, trail_atr=3.0, trail_len=22, risk_pct=1.0,
                 max_expo_pct=25.0, max_hold=0):
    """Donchian breakout in an uptrend; stop under the swing low, half off at 2R."""
    from .riskengine import Plan, Risk

    close, high, low = df["Close"], df["High"], df["Low"]
    atr = ind.atr(high, low, close, 14)

    # The breakout level must exclude today, or the signal is trivially true.
    prior_high = high.rolling(int(entry_len), min_periods=int(entry_len)).max().shift(1)
    trend = ind.ema(close, int(trend_len))

    entry = (close > prior_high) & (close > trend)

    # Where the idea is wrong: under the recent swing low, with ATR of room.
    stop = low.rolling(int(stop_lookback), min_periods=1).min() - atr * float(stop_atr)

    # Chandelier trail for whatever is left after the partial comes off.
    trail = high.rolling(int(trail_len), min_periods=int(trail_len)).max() - atr * float(trail_atr)

    plan = Plan(
        long=entry.fillna(False).to_numpy(bool),
        stop_long=stop.to_numpy(float),
        trail_long=trail.to_numpy(float),
    )
    risk = Risk(
        risk_pct=float(risk_pct),
        max_expo_pct=float(max_expo_pct),
        rr1=float(rr),
        scale_out=0.5,
        max_hold=int(max_hold),
        breakeven=True,
        trail_after_t1=True,
    )
    return plan, risk
