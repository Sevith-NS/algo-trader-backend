"""
Quantitative signal engine inspired by systematic/HFT desk models.

Combines several orthogonal alpha families into a composite score, then derives
executable trade levels (entry, stop, targets) and position sizing:

  1. Mean reversion  - Ornstein-Uhlenbeck z-score with half-life estimation
  2. Momentum/trend  - EMA(12/26) crossover slope + MACD histogram
  3. Microstructure  - price vs rolling VWAP, volume imbalance
  4. Oscillators     - RSI(14), Bollinger %B
  5. Volatility      - ATR(14), realized-vol regime percentile

Trade levels are produced by a CONTINUOUS model (see `_solve_levels`) rather
than fixed rules: no if/else on signal direction, no hardcoded 1.8x ATR stop,
no fixed 1.5R/3R targets. Every level is a smooth function of measured state
(signal strength, volatility regime, mean-reversion half-life, trend
persistence, structure), and the entry is chosen by maximising expected value
over a continuum of candidate limit prices.

  - Targets: ATR projections scaled by conviction and trend persistence, then
    soft-capped where 52-week structure stands in the way.
  - Stop:    volatility-regime-dependent ATR multiple, soft-blended with a
    recency-weighted swing extreme (log-sum-exp, so the level glides rather
    than jumping when one candidate overtakes the other).
  - Entry:   argmax over pullback depth x of
                 EV(x) = P_fill(x) * [P_win(x)*R(x) - (1 - P_win(x))]
             where P_fill is a first-passage (reflection-principle) touch
             probability and P_win is the two-barrier hitting probability of
             the target before the stop under drift implied by the signal.
  - Sizing:  half-Kelly on the trade's OWN P_win and R, capped at 25%.
"""
import numpy as np
import pandas as pd

import data_source


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # Wilder: zero average loss with gains present is RSI 100, not undefined;
    # only a truly flat window (0/0) stays NaN. Numerically twinned with the
    # frontend rsi() in src/lib/indicators.ts — keep the two in lockstep.
    rsi[(loss == 0) & (gain > 0)] = 100.0
    return rsi


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift(1)).abs()
    lc = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _half_life(spread: pd.Series) -> float:
    """OU half-life of mean reversion via lag-1 OLS: dx = a + b*x_{t-1}."""
    x = spread.shift(1).dropna()
    dx = spread.diff().dropna()
    x, dx = x.align(dx, join='inner')
    if len(x) < 30 or x.var() == 0:
        return np.nan
    beta = np.cov(dx, x)[0][1] / x.var()
    if beta >= 0:
        return np.nan  # not mean-reverting
    return float(-np.log(2) / beta)


# ---------------------------------------------------------------------------
# Continuous trade-level model
# ---------------------------------------------------------------------------
#
# Every constant below is a coefficient in a smooth function, not a threshold.
# The design rule: perturb any input infinitesimally and no output may jump.
# That is what makes the plan stable day to day -- the old branch-and-clamp
# version could move the stop 4% on a 0.01 change in the composite score.

_EPS = 1e-9

# Stop distance, in ATRs: k = BASE + VOL*vol_regime + SLACK*(1 - |direction|).
# A high-vol regime needs room; a weak signal needs room (worse entry timing).
_STOP_K_BASE, _STOP_K_VOL, _STOP_K_SLACK = 1.15, 1.25, 0.55
_STOP_SWING_BUFFER = 0.25   # ATRs below the swing extreme
_SOFT_T = 0.22              # log-sum-exp temperature, in ATRs

# Target projection, in ATRs from the fair-value anchor.
_T1_BASE, _T1_CONV, _T1_TREND = 1.45, 1.95, 0.95
_T2_BASE, _T2_TREND = 1.65, 0.85   # T2 = T1 * (BASE + TREND*trend) * mr_damp
_MR_TARGET_DAMP = 0.35             # mean-reverting names give back extension

# Fair-value anchor: how far the entry reference leans off last price toward
# VWAP. Only leans when the signal is weak enough to be patient.
_ANCHOR_MAX_W = 0.55
_HALF_LIFE_SCALE = 12.0            # days; sets how fast mean-reversion weight decays

# Entry search.
_FILL_HORIZON_DAYS = 5.0           # how long we are willing to wait for the limit
_MAX_PULLBACK_ATR = 1.60           # deepest limit we will ever post
_ENTRY_GRID = 241                  # objective is smooth; a dense grid is exact enough
_MAX_CHASE_ATR = 0.30              # furthest we will post THROUGH the market
_MIN_FILL_PROB = 0.25              # a limit you cannot get filled on is not a plan
_TIE_TOL = 1e-9                    # below this, two entries are the same entry
_MISS_CHASE = 0.60                 # of the missed depth, paid back on the chase

# Drift implied by the signal, as a daily Sharpe at |direction| = 1.
_SIGNAL_DAILY_SHARPE = 0.05


def _soft_extreme(a: float, b: float, side: int, temperature: float) -> float:
    """Smooth min (side=+1) / max (side=-1) of two candidate stop levels.

    A hard min() makes the stop jump the instant the swing low crosses the ATR
    level. Log-sum-exp glides between them instead, at the cost of sitting up
    to `temperature * ln 2` beyond the true extreme -- i.e. it errs slightly
    WIDER on the stop, which is the safe direction to err.
    """
    if temperature <= _EPS:
        return min(a, b) if side > 0 else max(a, b)
    # Work in "distance below/above" space so one branch handles both sides.
    x, y = (-a / temperature, -b / temperature) if side > 0 else (a / temperature, b / temperature)
    m = max(x, y)
    lse = m + np.log(np.exp(x - m) + np.exp(y - m))
    return float(-temperature * lse if side > 0 else temperature * lse)


def _recency_weighted_extreme(values: np.ndarray, side: int, atr: float,
                              decay_per_bar: float = 0.035) -> float:
    """Swing low (side=+1) / high (side=-1) that discounts stale extremes.

    A 40-bar-old low is real support, but it is not where you put a stop today.
    Rather than pick a fixed lookback -- a discontinuity waiting to happen when
    a bar rolls out of the window -- every bar is penalised in proportion to
    its age, so old extremes fade out smoothly instead of dropping off a cliff.
    """
    n = len(values)
    if n == 0:
        return float("nan")
    age = np.arange(n - 1, -1, -1, dtype=float)
    penalty = decay_per_bar * atr * age
    # side=+1: looking for a low, so age pushes candidates UP (less attractive).
    adjusted = values + penalty if side > 0 else values - penalty
    return float(adjusted.min() if side > 0 else adjusted.max())


def _touch_probability(distance: float, sigma_daily: float, horizon: float) -> float:
    """P(price trades `distance` against us within `horizon` days).

    Reflection principle for driftless Brownian motion:
        P(min_{t<=H} W_t <= -y) = 2 * Phi(-y / (sigma * sqrt(H)))
    Drift is deliberately omitted here: assuming the market helpfully walks
    into our limit would inflate every fill probability in the same direction
    as our own bias. distance <= 0 means the limit is at or through the market,
    which fills by definition.
    """
    if distance <= _EPS:
        return 1.0
    scale = sigma_daily * np.sqrt(max(horizon, _EPS))
    if scale <= _EPS:
        return 0.0
    # 2 * Phi(-z) via the complementary error function, no scipy dependency.
    from math import erfc
    return float(min(1.0, erfc(distance / (scale * np.sqrt(2.0)))))


def _barrier_probability(up: float, down: float, mu: float, sigma: float) -> float:
    """P(hit +up before -down) for arithmetic BM with drift mu, vol sigma.

        P = (1 - exp(-2*mu*down/sigma^2)) / (1 - exp(-2*mu*(up+down)/sigma^2))

    which collapses to the driftless down/(up+down) as mu -> 0. Untimed (no
    horizon): the standard convention for planning an R:R, and the honest one
    -- a target with no deadline is what a swing plan actually is.
    """
    if up <= _EPS:
        return 1.0
    if down <= _EPS:
        return 0.0
    total = up + down
    if sigma <= _EPS:
        return 1.0 if mu > 0 else 0.0
    theta = 2.0 * mu / (sigma * sigma)
    if abs(theta) * total < 1e-6:  # numerically driftless
        return float(down / total)
    # exp(-theta*x) can overflow for a large adverse drift; expm1 keeps the
    # ratio stable and exact near theta -> 0.
    try:
        num = -np.expm1(-theta * down)
        den = -np.expm1(-theta * total)
        if abs(den) <= _EPS:
            return float(down / total)
        return float(np.clip(num / den, 0.0, 1.0))
    except (OverflowError, FloatingPointError):
        return float(down / total)


def _solve_levels(*, last: float, atr: float, vwap: float, direction: float,
                  vol_regime: float, trend_slope: float, half_life: float,
                  sigma_daily: float, highs: np.ndarray, lows: np.ndarray,
                  high_52w: float, low_52w: float) -> dict:
    """The whole trade plan, as one continuous function of measured state.

    `direction` is the signed conviction in [-1, 1] -- the SAME number drives
    orientation and magnitude, so nothing discontinuous happens as it crosses
    zero. Returns a dict of levels plus the diagnostics behind them.
    """
    strength = abs(direction)
    side = 1 if direction >= 0 else -1
    patience = 1.0 - strength          # weak signal -> wait for a better price

    # -- Mean-reversion weight: short half-life => fade extension, lean on VWAP,
    #    and expect less follow-through out of the target.
    mr = 0.0 if (half_life is None or not np.isfinite(half_life) or half_life <= 0) \
        else float(np.exp(-half_life / _HALF_LIFE_SCALE))

    # -- Fair-value anchor. Blends last price toward VWAP, but only as far as
    #    our patience and the mean-reversion evidence jointly justify.
    anchor_w = _ANCHOR_MAX_W * mr * patience
    fair = last * (1.0 - anchor_w) + vwap * anchor_w

    # -- Trend persistence in the direction of the trade, mapped to [0, 1].
    trend = float(0.5 + 0.5 * np.tanh(side * trend_slope / 2.5))

    # -- Targets: ATR projections off the anchor. Continuous in every input.
    m1 = _T1_BASE + _T1_CONV * strength + _T1_TREND * trend
    m2 = m1 * (_T2_BASE + _T2_TREND * trend) * (1.0 - _MR_TARGET_DAMP * mr)
    target_1 = fair + side * m1 * atr
    target_2 = fair + side * m2 * atr

    # -- Structure drag: a 52-week extreme standing between us and the target
    #    pulls it back. Soft-capped, so the target glides as the wall is
    #    approached instead of snapping onto it. Skipped when price has already
    #    cleared the level -- there is no overhead supply above an all-time high.
    wall = high_52w if side > 0 else low_52w
    if np.isfinite(wall) and side * (wall - fair) > 0:
        target_1 = _soft_extreme(target_1, wall, side, _SOFT_T * atr)
        target_2 = _soft_extreme(target_2, wall, side, _SOFT_T * atr)
        # T2 must still sit beyond T1, or the two levels invert at the wall.
        target_2 = fair + side * max(side * (target_2 - fair), side * (target_1 - fair) * 1.15)

    # -- Stop: volatility-regime ATR multiple soft-blended with a
    #    recency-weighted swing extreme. The ATR leg hangs off the ENTRY, not
    #    off spot, so it travels with the entry as the search below moves it.
    #    (Pinning the stop while the entry moves lets R = reward/risk diverge
    #    as entry -> stop, and the search then chases unbounded R into a
    #    lottery-ticket trade that ordinary noise stops out. Anchoring the stop
    #    to the entry holds risk near k*ATR and keeps the objective bounded.)
    k_stop = _STOP_K_BASE + _STOP_K_VOL * vol_regime + _STOP_K_SLACK * patience
    swing = _recency_weighted_extreme(lows if side > 0 else highs, side, atr)
    # No usable swing history (empty or all-NaN window): the ATR leg stands
    # alone. Blending against a NaN would poison the stop and, through it,
    # every R multiple downstream.
    swing_stop = (swing - side * _STOP_SWING_BUFFER * atr
                  if np.isfinite(swing) else None)

    def stop_for(entry_price: float) -> float:
        atr_stop = entry_price - side * k_stop * atr
        if swing_stop is None:
            return float(atr_stop)
        return _soft_extreme(atr_stop, swing_stop, side, _SOFT_T * atr)

    # -- Signal-implied drift. Shrunk by |direction| so a flat signal plans
    #    against a driftless market rather than against our own optimism.
    mu = direction * _SIGNAL_DAILY_SHARPE * sigma_daily

    def plan_at(x: float):
        """The whole trade at a pullback of x ATRs, or None if not a trade."""
        entry = fair - side * x * atr
        stop = stop_for(entry)
        risk = side * (entry - stop)
        reward = side * (target_1 - entry)
        if risk <= _EPS or reward <= _EPS:
            return None  # entry has crossed its own stop or target
        p_win = _barrier_probability(reward, risk, side * mu, sigma_daily)
        r_mult = reward / risk
        return {
            "x": float(x), "entry": float(entry), "stop": float(stop),
            "risk": float(risk), "r_mult": float(r_mult), "p_win": float(p_win),
            "ev": float(p_win * r_mult - (1.0 - p_win)),
            "p_fill": _touch_probability(x * atr, sigma_daily, _FILL_HORIZON_DAYS),
        }

    # -- Entry: maximise the expected value of POSTING the order.
    #
    #    A resting limit either fills at the better price, or it misses and the
    #    setup gets chased at a worse one. Scoring a miss as zero says "never
    #    wait" for any realistic drift, and worse, turns negative-EV setups into
    #    an instruction to post an unfillable price. So the miss is scored as
    #    what it actually is -- entering late:
    #
    #        score(x) = P_fill(x)*EV(x) + (1 - P_fill(x))*EV(chase(x))
    #
    #    The chase price scales with the depth that was missed: a limit 1.5 ATR
    #    down that never fills means the thing ran, and you pay up by a fraction
    #    of what you were waiting for. Without that term the deep limits look
    #    free and the model tells you to wait for a dip on every single setup.
    def ev_missed(x: float) -> float:
        chased = plan_at(-_MISS_CHASE * abs(x))
        return chased["ev"] if chased else 0.0

    # |x| ascending, so a tie resolves to the entry NEAREST the market rather
    # than to floating-point noise. Ties are not a corner case here: at exactly
    # zero drift p_win*R - (1-p_win) is identically 0 for every entry, stop and
    # target (the martingale result), so the whole objective goes flat and the
    # right answer is "no edge, no reason to wait — take it at market".
    # Built around 0 rather than across the whole span, so "enter at market"
    # is exactly representable — a linspace over an asymmetric range lands
    # near zero but never on it, leaving the no-edge case with a phantom offset.
    step = (_MAX_CHASE_ATR + _MAX_PULLBACK_ATR) / (_ENTRY_GRID - 1)
    grid = sorted(
        np.concatenate([
            np.arange(0.0, _MAX_PULLBACK_ATR + step / 2, step),
            -np.arange(step, _MAX_CHASE_ATR + step / 2, step),
        ]),
        key=abs,
    )
    best = None
    best_score = -np.inf
    for x in grid:
        plan = plan_at(float(x))
        if plan is None or plan["p_fill"] < _MIN_FILL_PROB:
            continue
        score = (plan["p_fill"] * plan["ev"]
                 + (1.0 - plan["p_fill"]) * ev_missed(plan["x"]))
        if score > best_score + _TIE_TOL:
            best_score, best = score, plan

    if best is None:
        # Degenerate geometry (zero ATR, target inside the stop). Fall back to
        # the anchor itself and let the caller's rounding show a flat plan.
        entry = fair
        stop = stop_for(entry)
        risk = max(abs(entry - stop), _EPS)
        best = {"x": 0.0, "entry": float(entry), "stop": float(stop),
                "risk": float(risk), "ev": 0.0, "p_fill": 1.0, "p_win": 0.5,
                "r_mult": float(abs(target_1 - entry) / risk)}

    entry = best["entry"]
    stop = best["stop"]
    risk = best["risk"]
    r1 = best["r_mult"]
    r2 = side * (target_2 - entry) / risk

    return {
        "side": side,
        "entry": entry,
        "stop": stop,
        "target_1": target_1,
        "target_2": target_2,
        "risk_per_share": risk,
        "r_multiple_1": r1,
        "r_multiple_2": float(r2),
        "entry_offset_atr": best["x"],
        "fill_probability": best["p_fill"],
        "win_probability": best["p_win"],
        "expectancy_r": best["ev"],
        "atr_stop_mult": float(k_stop),
        "target_atr_mult": float(m1),
        "anchor_price": float(fair),
        "mean_reversion_weight": float(mr),
        "trend_persistence": float(trend),
    }


def get_quant_signals(symbol: str):
    try:
        df = data_source.get_history(symbol, period="1y")
        if df.empty or len(df) < 60:
            return {"error": "Not enough historical data for quant models."}

        close = df['Close']
        last_close = float(close.iloc[-1])

        # --- Volatility ---
        atr_series = _atr(df)
        atr = float(atr_series.iloc[-1])
        returns = close.pct_change().dropna()
        realized_vol = float(returns.tail(21).std() * np.sqrt(252))
        vol_series = returns.rolling(21).std().dropna() * np.sqrt(252)
        vol_percentile = float((vol_series <= realized_vol).mean() * 100)

        # --- Mean reversion (OU z-score around 20d mean) ---
        sma20 = close.rolling(20).mean()
        # ddof=0 (population std) is the classic Bollinger convention and what
        # the frontend bollinger() in src/lib/indicators.ts computes — pandas'
        # default ddof=1 made the same bands differ between chart and signals.
        std20 = close.rolling(20).std(ddof=0)
        zscore = float((last_close - sma20.iloc[-1]) / std20.iloc[-1]) if std20.iloc[-1] > 0 else 0.0
        half_life = _half_life(close - sma20)

        # --- Momentum / trend ---
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = float(macd_line.iloc[-1] - macd_signal.iloc[-1])
        # Normalize MACD histogram by price for cross-asset comparability
        macd_norm = macd_hist / last_close * 100
        trend_slope = float((ema26.iloc[-1] - ema26.iloc[-6]) / ema26.iloc[-6] * 100)

        # --- Microstructure proxies ---
        typical = (df['High'] + df['Low'] + df['Close']) / 3
        vwap20 = float((typical * df['Volume']).tail(20).sum() / max(df['Volume'].tail(20).sum(), 1))
        vwap_dev = (last_close - vwap20) / vwap20 * 100
        up_vol = df['Volume'].tail(10)[close.diff().tail(10) > 0].sum()
        total_vol = df['Volume'].tail(10).sum()
        volume_imbalance = float(up_vol / total_vol) if total_vol > 0 else 0.5

        # --- Oscillators ---
        rsi = float(_rsi(close).iloc[-1])
        upper_bb = float(sma20.iloc[-1] + 2 * std20.iloc[-1])
        lower_bb = float(sma20.iloc[-1] - 2 * std20.iloc[-1])
        pct_b = (last_close - lower_bb) / (upper_bb - lower_bb) if upper_bb != lower_bb else 0.5

        # --- Composite score: each factor votes in [-1, 1] ---
        factors = {
            "mean_reversion": float(np.clip(-zscore / 2, -1, 1)),
            "momentum": float(np.clip(macd_norm / 1.5 + trend_slope / 3, -1, 1)),
            "rsi": float(np.clip((50 - rsi) / 30, -1, 1)),
            "vwap": float(np.clip(-vwap_dev / 4, -1, 1)),
            "volume_flow": float(np.clip((volume_imbalance - 0.5) * 4, -1, 1)),
        }
        weights = {"mean_reversion": 0.25, "momentum": 0.30, "rsi": 0.15, "vwap": 0.15, "volume_flow": 0.15}
        composite = sum(factors[k] * weights[k] for k in factors)

        if composite >= 0.45:
            signal, action = "STRONG BUY", "buy"
        elif composite >= 0.15:
            signal, action = "BUY", "buy"
        elif composite <= -0.45:
            signal, action = "STRONG SELL", "sell"
        elif composite <= -0.15:
            signal, action = "SELL", "sell"
        else:
            signal, action = "HOLD", "none"
        conviction = float(min(abs(composite) / 0.6, 1.0))

        # --- Executable levels (continuous model — see _solve_levels) ---
        # `direction` carries sign AND magnitude in one number so the plan is
        # continuous through composite == 0; tanh saturates smoothly rather
        # than clipping, which would leave a kink at the clip boundary.
        direction = float(np.tanh(composite / 0.45))
        sigma_daily = float(returns.tail(60).std()) * last_close
        levels = _solve_levels(
            last=last_close,
            atr=atr,
            vwap=vwap20,
            direction=direction,
            vol_regime=float(np.clip(vol_percentile / 100.0, 0.0, 1.0)),
            trend_slope=trend_slope,
            half_life=half_life,
            sigma_daily=sigma_daily,
            highs=df['High'].tail(40).to_numpy(dtype=float),
            lows=df['Low'].tail(40).to_numpy(dtype=float),
            high_52w=float(df['High'].max()),
            low_52w=float(df['Low'].min()),
        )
        entry = levels["entry"]
        stop = levels["stop"]
        target_1 = levels["target_1"]
        target_2 = levels["target_2"]

        # --- Kelly sizing on THIS trade's odds ---
        # The old version sized off the symbol's unconditional daily win rate,
        # which is ~50% for everything and told you nothing about the setup in
        # front of you. Kelly wants the probability and payoff of the actual
        # bet: the barrier probability of reaching T1 before the stop, and the
        # R multiple the solved entry buys. Historical stats are still reported
        # as context, but they no longer drive the size.
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        win_rate = float(len(wins) / len(returns)) if len(returns) else 0.5
        payoff = float(wins.mean() / abs(losses.mean())) if len(losses) and losses.mean() != 0 else 1.0

        p_win = levels["win_probability"]
        r_mult = levels["r_multiple_1"]
        kelly = p_win - (1 - p_win) / r_mult if r_mult > 0 else 0.0
        # Fill probability scales the size: a limit that only fills 30% of the
        # time is a 30%-weighted allocation of the risk budget, not a full one.
        position_fraction = float(np.clip(
            0.5 * kelly * conviction * levels["fill_probability"], 0, 0.25))

        return {
            "symbol": symbol,
            "current_price": last_close,
            "signal": signal,
            "action": action,
            "composite_score": round(float(composite), 3),
            "conviction": round(conviction, 2),
            "factors": {k: round(v, 3) for k, v in factors.items()},
            "levels": {
                "entry": round(float(entry), 2),
                "stop_loss": round(float(stop), 2),
                "target_1": round(float(target_1), 2),
                "target_2": round(float(target_2), 2),
                # Now the ACTUAL solved R multiple, not the 1.5 the old code
                # asserted regardless of where the levels landed.
                "risk_reward": round(float(levels["r_multiple_1"]), 2),
                "r_multiple_1": round(float(levels["r_multiple_1"]), 2),
                "r_multiple_2": round(float(levels["r_multiple_2"]), 2),
                "risk_per_share": round(float(levels["risk_per_share"]), 2),
                "side": "long" if levels["side"] > 0 else "short",
                # Diagnostics: what the continuous solve actually chose and why.
                "entry_offset_atr": round(float(levels["entry_offset_atr"]), 2),
                "fill_probability": round(float(levels["fill_probability"]), 3),
                "win_probability": round(float(levels["win_probability"]), 3),
                "expectancy_r": round(float(levels["expectancy_r"]), 3),
                "atr_stop_mult": round(float(levels["atr_stop_mult"]), 2),
                "target_atr_mult": round(float(levels["target_atr_mult"]), 2),
                "anchor_price": round(float(levels["anchor_price"]), 2),
                "trend_persistence": round(float(levels["trend_persistence"]), 2),
                "mean_reversion_weight": round(float(levels["mean_reversion_weight"]), 2),
            },
            "position_sizing": {
                "kelly_fraction": round(float(kelly), 4),
                "recommended_fraction": round(position_fraction, 4),
                # Historical context only — sizing runs off the trade's own odds.
                "win_rate": round(win_rate, 3),
                "payoff_ratio": round(payoff, 2),
                "trade_win_probability": round(float(p_win), 3),
                "trade_r_multiple": round(float(r_mult), 2),
            },
            "indicators": {
                "rsi_14": round(rsi, 1),
                "macd_histogram": round(macd_hist, 3),
                "zscore_20d": round(zscore, 2),
                "half_life_days": round(half_life, 1) if not np.isnan(half_life) else None,
                "vwap_20d": round(vwap20, 2),
                "vwap_deviation_pct": round(float(vwap_dev), 2),
                "atr_14": round(atr, 2),
                "realized_vol_annual": round(realized_vol, 3),
                "vol_regime_percentile": round(vol_percentile, 0),
                "bollinger_pct_b": round(float(pct_b), 2),
                "volume_imbalance": round(volume_imbalance, 2),
            },
            # Back-compat with the old advanced-signals consumers
            "stop_loss": round(float(stop), 2),
            "take_profit": round(float(target_2), 2),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
