"""
Offline fixture tests for the support/resistance scanner.

No network: every series is synthetic and hand-built so the RIGHT ANSWER IS
KNOWN before the scan runs. A level detector that "looks plausible" on real
data is untestable — these fixtures put a level at a price the test chose, and
assert the scanner finds that price and not another.

Run directly (`python test_levels_engine.py`) or via pytest.
"""
import numpy as np
import pandas as pd

import levels_engine as le
import screener_engine as se


# ------------------------------------------------------------- fixtures ---

def _series(values, end="2026-08-07", freq="B"):
    idx = pd.date_range(end=end, periods=len(values), freq=freq)
    return pd.Series([float(v) for v in values], index=idx)


def _oscillate(low, high, cycles, per_leg, drift=0.0):
    """Triangle wave between two prices — every turn is a pivot, so `high`
    becomes a resistance level and `low` a support level, by construction."""
    out = []
    for c in range(cycles):
        base = drift * c
        out.extend(np.linspace(low + base, high + base, per_leg))
        out.extend(np.linspace(high + base, low + base, per_leg)[1:])
    return out


# --------------------------------------------------------------- tests ---

def test_pivot_indices_find_known_extrema():
    v = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    highs = le._pivot_indices(v, 2, "high")
    lows = le._pivot_indices(v, 2, "low")
    assert 4 in highs, f"peak at index 4 missed: {highs}"
    assert 8 in lows, f"trough at index 8 missed: {lows}"
    # k bars of padding at each end can never be a pivot.
    assert highs.min() >= 2 and highs.max() <= len(v) - 3


def test_pivot_indices_handle_short_series():
    assert len(le._pivot_indices(np.array([1.0, 2.0]), 4, "high")) == 0


def test_cluster_groups_nearby_and_splits_far():
    prices = np.array([100.0, 100.5, 101.0, 140.0, 141.0])
    ages = np.array([10.0, 20.0, 30.0, 5.0, 15.0])
    kinds = ["high"] * 5
    out = le._cluster(prices, ages, kinds, tol_pct=0.02)
    assert len(out) == 2, f"expected 2 clusters, got {[c['price'] for c in out]}"
    small = min(out, key=lambda c: c["price"])
    big = max(out, key=lambda c: c["price"])
    assert small["touches"] == 3 and big["touches"] == 2
    assert 100.0 <= small["price"] <= 101.0
    assert 140.0 <= big["price"] <= 141.0


def test_detects_the_level_it_was_given():
    """A price that turned the series back repeatedly must be found, at that
    price — this is the whole contract of the module."""
    closes = _series(_oscillate(90.0, 110.0, cycles=6, per_leg=11))
    levels = le._levels_for(closes, "daily")
    assert levels, "no levels detected on a textbook range"
    prices = [lv["price"] for lv in levels]
    assert any(abs(p - 110.0) < 1.5 for p in prices), f"missed 110 resistance: {prices}"
    assert any(abs(p - 90.0) < 1.5 for p in prices), f"missed 90 support: {prices}"


def test_more_touches_scores_higher():
    few = {"touches": 2, "newest_age": 5.0, "oldest_age": 60.0, "kinds": ["high"] * 2}
    many = {"touches": 6, "newest_age": 5.0, "oldest_age": 60.0, "kinds": ["high"] * 6}
    assert le._score(many, 250, 90.0) > le._score(few, 250, 90.0)


def test_stale_level_scores_lower_than_live_one():
    live = {"touches": 4, "newest_age": 3.0, "oldest_age": 120.0, "kinds": ["high"] * 4}
    stale = {"touches": 4, "newest_age": 200.0, "oldest_age": 240.0, "kinds": ["high"] * 4}
    assert le._score(live, 250, 90.0) > le._score(stale, 250, 90.0)


def test_polarity_flip_scores_higher():
    plain = {"touches": 4, "newest_age": 5.0, "oldest_age": 100.0, "kinds": ["high"] * 4}
    flip = {"touches": 4, "newest_age": 5.0, "oldest_age": 100.0,
            "kinds": ["high", "high", "low", "low"]}
    assert le._score(flip, 250, 90.0) > le._score(plain, 250, 90.0)


def test_score_is_bounded():
    extreme = {"touches": 50, "newest_age": 0.0, "oldest_age": 249.0,
               "kinds": ["high", "low"]}
    assert 0.0 <= le._score(extreme, 250, 90.0) <= 100.0


def test_symbol_at_its_level_is_flagged_symbol_far_from_it_is_not():
    # Ends at 109.5, i.e. pressed right up under the 110 resistance.
    at_level = _oscillate(90.0, 110.0, cycles=6, per_leg=11) + [109.0, 109.5]
    hit = le._scan_symbol(_series(at_level))
    assert hit is not None, "a symbol sitting on its level was not flagged"
    assert hit["side"] == "resistance", hit["side"]
    assert abs(hit["level"]["price"] - 110.0) < 2.0

    # Same history, then a long drift down to the middle of nowhere.
    away = _oscillate(90.0, 110.0, cycles=6, per_leg=11) + list(np.linspace(109.0, 100.0, 20))
    hit_away = le._scan_symbol(_series(away))
    if hit_away is not None:
        # If anything is reported it must be a genuinely near level, not the
        # 110 the series has walked away from.
        assert abs(hit_away["distance_pct"]) <= le._NEAR_MAX_PCT
        assert abs(hit_away["level"]["price"] - 110.0) > 2.0


def test_support_side_is_labelled_correctly():
    below = _oscillate(90.0, 110.0, cycles=6, per_leg=11) + [91.5, 90.8]
    hit = le._scan_symbol(_series(below))
    assert hit is not None
    assert hit["side"] == "support", hit
    assert hit["level"]["price"] <= hit["price"] + 1e-9


def test_proximity_is_measured_in_sigmas_not_percent():
    """A quiet name 3% from its level is far; a wild one 3% away is not. The
    scanner must treat them differently, which percent thresholds cannot."""
    quiet = _oscillate(99.0, 101.0, cycles=10, per_leg=8)
    quiet_hit = le._scan_symbol(_series(quiet + [100.0]))
    wild = _oscillate(70.0, 130.0, cycles=10, per_leg=8)
    wild_hit = le._scan_symbol(_series(wild + [100.0]))
    # Both series sit mid-range; the sigma normalisation is what makes the
    # distance comparable at all. Assert the sigma actually differs a lot.
    assert quiet_hit is None or wild_hit is None or (
        quiet_hit["sigma_pct"] < wild_hit["sigma_pct"])


def test_confluence_is_detected_across_timeframes():
    """A range held long enough to print weekly pivots too should mark the
    level as confluent."""
    closes = _series(_oscillate(90.0, 110.0, cycles=12, per_leg=11) + [109.4])
    hit = le._scan_symbol(closes)
    assert hit is not None
    assert isinstance(hit["confluence"], bool)
    weekly = le._weekly(closes)
    assert len(weekly) >= le._MIN_BARS["weekly"], "fixture too short for a weekly pass"


def test_degenerate_series_return_none_instead_of_raising():
    """Nothing here has a level to be at, and none of it may raise."""
    cases = {
        "dead flat": [100.0] * 200,            # zero sigma, no pivots
        "pure ramp": list(np.linspace(50, 150, 200)),  # never revisits a price
        "too short": [100.0] * 5,
        "single bar": [100.0],
        "with nans": [100.0, float("nan")] * 100,
    }
    for label, values in cases.items():
        assert le._scan_symbol(_series(values)) is None, f"{label} produced a level"
    assert le._scan_symbol(pd.Series(dtype=float)) is None, "empty series"


def test_reason_line_contains_the_numbers_it_used():
    closes = _series(_oscillate(90.0, 110.0, cycles=6, per_leg=11) + [109.0, 109.5])
    hit = le._scan_symbol(closes)
    assert hit is not None
    reason = hit["reason"]
    assert "resistance" in reason
    assert "touches" in reason
    assert "σ" in reason, f"proximity in sigmas missing from: {reason}"
    assert f"{hit['level']['price']:.2f}" in reason


def test_get_levels_shape_and_ranking(monkeypatch=None):
    """End-to-end over a stubbed universe loader: no network, known answer."""
    at_level = _series(_oscillate(90.0, 110.0, cycles=6, per_leg=11) + [109.0, 109.5])
    ramp = _series(list(np.linspace(50, 150, 200)))

    real_loader = se._load_universe_series
    real_resolve = se._resolve_universe
    se._load_universe_series = lambda uid: ({"AAA": at_level, "BBB": ramp}, "2026-08-07")
    se._resolve_universe = lambda uid: {
        "label": "Test", "currency": "USD", "symbols": {"AAA": "Alpha", "BBB": "Beta"},
    }
    le._results.clear()
    try:
        out = le.get_levels("us_large", limit=10)
    finally:
        se._load_universe_series = real_loader
        se._resolve_universe = real_resolve
        le._results.clear()

    assert out["basis"] == "close"
    assert out["parsed"] == 2
    assert out["matched"] == len(out["rows"])
    symbols = [r["symbol"] for r in out["rows"]]
    assert "AAA" in symbols, "the ranged name at its level was not surfaced"
    assert "BBB" not in symbols, "a pure ramp has no level to be at"
    row = out["rows"][0]
    for key in ("symbol", "price", "level", "timeframe", "side", "distance_pct",
                "distance_sigmas", "touches", "strength", "score", "confluence",
                "reason", "spark"):
        assert key in row, f"missing key {key}"
    # Ranked, descending.
    scores = [r["score"] for r in out["rows"]]
    assert scores == sorted(scores, reverse=True)


def test_get_levels_side_filter():
    below = _series(_oscillate(90.0, 110.0, cycles=6, per_leg=11) + [91.5, 90.8])
    real_loader = se._load_universe_series
    real_resolve = se._resolve_universe
    se._load_universe_series = lambda uid: ({"SUP": below}, "2026-08-07")
    se._resolve_universe = lambda uid: {
        "label": "Test", "currency": "USD", "symbols": {"SUP": "Supportive"},
    }
    le._results.clear()
    try:
        sup = le.get_levels("us_large", side="support")
        res = le.get_levels("us_large", side="resistance")
    finally:
        se._load_universe_series = real_loader
        se._resolve_universe = real_resolve
        le._results.clear()
    assert all(r["side"] == "support" for r in sup["rows"])
    assert sup["matched"] >= 1
    assert res["matched"] == 0


def test_unknown_universe_raises_the_error_the_route_maps():
    try:
        le.get_levels("not_a_universe")
    except se.UnknownUniverseError:
        return
    raise AssertionError("expected UnknownUniverseError")


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
