"""
Offline tests for the continuous trade-level solver in quant_models.

No network: everything runs against _solve_levels directly with synthetic
state. The properties asserted here are the ones the old branch-and-clamp
level code could NOT satisfy, and they are what "continuous" has to mean if
it is to be worth anything:

  - levels are ordered and on the correct side of the entry,
  - a hair's change in any input moves a level by a hair (within a side),
  - the closed-form probabilities reduce to their known values,
  - degenerate market state degrades instead of raising.

Run directly (`python test_quant_levels.py`) or via pytest.
"""
import numpy as np

import quant_models as qm


# ------------------------------------------------------------- fixtures ---

_LAST, _ATR = 100.0, 2.0
# Deterministic: a seeded generator keeps the swing extremes fixed so a
# failure is always a code change, never a reroll.
_RNG = np.random.default_rng(7)
_HIGHS = _LAST + _RNG.uniform(0, 3, 40)
_LOWS = _LAST - _RNG.uniform(0, 3, 40)

_BASE = dict(
    last=_LAST, atr=_ATR, vwap=99.2, vol_regime=0.5, trend_slope=1.2,
    half_life=18.0, sigma_daily=1.6, highs=_HIGHS, lows=_LOWS,
    high_52w=118.0, low_52w=76.0,
)

_LEVEL_KEYS = ("entry", "stop", "target_1", "target_2")


def _solve(direction, **overrides):
    return qm._solve_levels(**{**_BASE, "direction": direction, **overrides})


def _max_step(key, directions, **overrides):
    v = np.array([_solve(float(d), **overrides)[key] for d in directions])
    return float(np.abs(np.diff(v)).max())


# --------------------------------------------------------------- tests ---

def test_levels_are_ordered_on_both_sides():
    """stop < entry < T1 < T2 for a long, and mirrored for a short."""
    for d in (-0.9, -0.4, -0.05, 0.0, 0.05, 0.4, 0.9):
        L = _solve(d)
        side = L["side"]
        assert side * (L["entry"] - L["stop"]) > 0, f"stop on wrong side at d={d}"
        assert side * (L["target_1"] - L["entry"]) > 0, f"T1 on wrong side at d={d}"
        assert side * (L["target_2"] - L["target_1"]) > 0, f"T2 inside T1 at d={d}"


def test_levels_are_finite_and_positive():
    for d in (-1.0, -0.5, 0.0, 0.5, 1.0):
        L = _solve(d)
        for k in _LEVEL_KEYS:
            assert np.isfinite(L[k]), f"{k} not finite at d={d}"
            assert L[k] > 0, f"{k} non-positive at d={d}"


def test_continuous_in_signal_within_a_side():
    """The headline property.

    A long plan and a short plan are different trades, so the levels do swap
    places at direction == 0 — that is the trade flipping, not a discontinuity.
    Within a side, nothing may jump: the old code moved the stop by a full ATR
    on a 0.01 change in the composite score, because min() chose a different
    branch. The soft blend has to keep every step far below one ATR.
    """
    for lo, hi in ((1e-4, 1.0), (-1.0, -1e-4)):
        directions = np.linspace(lo, hi, 800)
        for key in _LEVEL_KEYS:
            step = _max_step(key, directions)
            assert step < 0.05 * _ATR, f"{key} jumps {step:.4f} on [{lo}, {hi}]"


def test_continuous_in_volatility_and_structure_inputs():
    """Same guarantee for every other continuous input the solver reads."""
    for key, grid in (
        ("vol_regime", np.linspace(0.0, 1.0, 400)),
        ("half_life", np.linspace(0.5, 60.0, 400)),
        ("trend_slope", np.linspace(-8.0, 8.0, 400)),
        ("vwap", np.linspace(94.0, 106.0, 400)),
    ):
        for level in ("stop", "target_1"):
            v = np.array([_solve(0.55, **{key: float(g)})[level] for g in grid])
            step = float(np.abs(np.diff(v)).max())
            assert step < 0.05 * _ATR, f"{level} jumps {step:.4f} across {key}"


def test_no_edge_means_no_waiting():
    """Zero drift makes expected R identically zero for EVERY entry (the
    martingale result), so the objective goes flat. The tie must resolve to
    the market price, not to whichever grid point floating point favoured."""
    assert _solve(0.0)["entry_offset_atr"] == 0.0


def test_entry_offset_is_invariant_to_edge_SIZE():
    """Every EV term is proportional to the drift, so the drift factors out of
    the argmax: a tiny edge and a huge edge want the SAME limit price. The edge
    size shows up in the stop and target instead (see the next test)."""
    offsets = [_solve(d)["entry_offset_atr"] for d in (1e-3, 0.1, 0.5, 1.0)]
    assert max(offsets) - min(offsets) < 0.02, offsets
    # And it must be a realistic dip, not a token one or an unfillable one.
    assert all(0.1 < o < 1.5 for o in offsets), offsets


def test_conviction_tightens_the_stop_and_extends_the_target():
    ks = [_solve(d)["atr_stop_mult"] for d in (0.15, 0.5, 0.99)]
    ts = [_solve(d)["target_atr_mult"] for d in (0.15, 0.5, 0.99)]
    rs = [_solve(d)["r_multiple_1"] for d in (0.15, 0.5, 0.99)]
    assert ks[0] > ks[1] > ks[2], f"stop should tighten with conviction: {ks}"
    assert ts[0] < ts[1] < ts[2], f"target should extend with conviction: {ts}"
    assert rs[0] < rs[1] < rs[2], f"R should improve with conviction: {rs}"


def test_higher_volatility_posts_a_deeper_limit():
    """More daily range means a deeper pullback is reachable inside the fill
    horizon, so the solver should be willing to wait for a better price."""
    offsets = [_solve(0.5, sigma_daily=s)["entry_offset_atr"] for s in (0.4, 0.8, 1.6, 3.2)]
    assert offsets == sorted(offsets), offsets
    assert offsets[-1] > offsets[0] * 2


def test_structure_wall_drags_targets_without_inverting_them():
    for wall in (104.0, 108.0, 115.0, 160.0):
        L = _solve(0.8, high_52w=wall)
        assert L["target_2"] > L["target_1"], f"targets inverted at wall {wall}"
    # A near wall must actually pull the target in versus a distant one.
    near = _solve(0.8, high_52w=104.0)["target_1"]
    far = _solve(0.8, high_52w=160.0)["target_1"]
    assert near < far, f"wall did not drag the target: {near} vs {far}"


def test_barrier_probability_reduces_to_known_values():
    # Driftless: P(hit +a before -b) = b / (a + b).
    assert abs(qm._barrier_probability(10, 10, 0.0, 2.0) - 0.5) < 1e-9
    assert abs(qm._barrier_probability(20, 10, 0.0, 2.0) - 1 / 3) < 1e-9
    # Drift moves it the right way, and stays a probability.
    assert qm._barrier_probability(20, 10, 0.10, 2.0) > 1 / 3
    assert qm._barrier_probability(20, 10, -0.10, 2.0) < 1 / 3
    for mu in (-5.0, -0.1, 0.0, 0.1, 5.0):
        p = qm._barrier_probability(20, 10, mu, 2.0)
        assert 0.0 <= p <= 1.0, (mu, p)
    # Degenerate barriers must not divide by zero.
    assert qm._barrier_probability(0.0, 10, 0.1, 2.0) == 1.0
    assert qm._barrier_probability(10, 0.0, 0.1, 2.0) == 0.0


def test_touch_probability_is_a_decreasing_probability():
    assert qm._touch_probability(0.0, 1.6, 5.0) == 1.0
    ps = [qm._touch_probability(d, 1.6, 5.0) for d in (0.5, 1.0, 2.0, 5.0, 20.0)]
    assert ps == sorted(ps, reverse=True), ps
    assert all(0.0 <= p <= 1.0 for p in ps)
    assert qm._touch_probability(1.0, 0.0, 5.0) == 0.0  # no vol, no touch


def test_soft_extreme_brackets_the_hard_extreme():
    """Soft-min must sit at or just below the true min (erring wider on a
    stop is the safe direction), and within the documented temperature bound."""
    for a, b in ((10.0, 12.0), (12.0, 10.0), (10.0, 10.0), (5.0, 50.0)):
        t = 0.5
        lo = qm._soft_extreme(a, b, +1, t)
        assert lo <= min(a, b) + 1e-9
        assert lo >= min(a, b) - t * np.log(2) - 1e-9
        hi = qm._soft_extreme(a, b, -1, t)
        assert hi >= max(a, b) - 1e-9
        assert hi <= max(a, b) + t * np.log(2) + 1e-9
    # Zero temperature degenerates to the hard extreme.
    assert qm._soft_extreme(10.0, 12.0, +1, 0.0) == 10.0
    assert qm._soft_extreme(10.0, 12.0, -1, 0.0) == 12.0


def test_recency_weighted_extreme_discounts_stale_lows():
    """An old low must not win over a comparable recent one."""
    lows = np.array([90.0] + [95.0] * 39)      # deep low, 40 bars ago
    recent = np.array([95.0] * 39 + [94.0])    # shallower low, today
    old = qm._recency_weighted_extreme(lows, +1, _ATR)
    new = qm._recency_weighted_extreme(recent, +1, _ATR)
    assert old > 90.0, "stale low was not discounted at all"
    assert new < 94.0 + 1e-9, "today's low should not be discounted"


def test_degenerate_market_state_degrades_instead_of_raising():
    for overrides in (
        {"atr": 0.0},
        {"sigma_daily": 0.0},
        {"half_life": float("nan")},
        {"half_life": -3.0},
        {"high_52w": float("nan"), "low_52w": float("nan")},
        {"vol_regime": 0.0},
        {"vwap": 0.0},
        {"highs": np.array([]), "lows": np.array([])},
    ):
        L = _solve(0.5, **overrides)
        for k in _LEVEL_KEYS:
            assert np.isfinite(L[k]), f"{k} not finite for {overrides}"
        assert 0.0 <= L["win_probability"] <= 1.0, overrides
        assert 0.0 <= L["fill_probability"] <= 1.0, overrides


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
    raise SystemExit(1 if failures else 0)
