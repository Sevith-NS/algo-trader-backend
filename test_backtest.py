"""Offline fixture tests for the backtester.

No network: every bar series is hand-built so the RIGHT ANSWER IS KNOWN before
the engine runs. A backtester that "looks plausible" on real data is exactly
the thing that cannot be trusted — these fixtures put a stop at a price the
test chose and assert the fill happened there, on the bar it should have.

The properties under test are the ones that separate a backtest from a
fantasy: no look-ahead, costs on both sides, and the pessimistic assumption
when a bar contains both a stop and a target.

Run directly (`python test_backtest.py`) or via pytest.
"""
import numpy as np
import pandas as pd

from backtest import analysis as A
from backtest import data as D
from backtest import engine as E
from backtest import metrics as M
from backtest import strategies as S
from backtest.errors import BacktestError


# ------------------------------------------------------------- fixtures ---

def _bars(rows, start="2024-01-01"):
    """rows: (open, high, low, close) per bar. Volume is constant."""
    idx = pd.bdate_range(start=start, periods=len(rows))
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)
    df["Volume"] = 1_000_000.0
    df.attrs["interval"] = "1d"
    return df


def _flat(n, price=100.0):
    return _bars([(price, price, price, price)] * n)


def _free_config(**kw):
    """Costs off, so a P&L figure is the strategy's and nothing else."""
    base = dict(initial_capital=100_000.0, commission_bps=0.0, slippage_bps=0.0)
    base.update(kw)
    return E.Config(**base)


def _target(index, on_bars):
    s = pd.Series(0.0, index=index)
    s.iloc[on_bars] = 1.0
    return s


# ------------------------------------------------- execution / no look-ahead

def test_signal_fills_at_the_next_bar_open():
    """A signal formed at bar 5's close must fill at bar 6's OPEN, not bar 5.

    This is the whole ballgame. If it ever fills on the signal bar, every
    result the tester produces is inflated by one bar of hindsight.
    """
    df = _bars([(100, 101, 99, 100)] * 4
               + [(100, 101, 99, 100)]      # bar 4
               + [(100, 101, 99, 100)]      # bar 5 - signal forms here
               + [(200, 201, 199, 200)]     # bar 6 - and fills HERE, at 200
               + [(210, 211, 209, 210)] * 3)
    result = E.run(df, _target(df.index, [5]), _free_config())

    assert len(result.trades) == 1, result.trades
    trade = result.trades[0]
    assert trade.entry_time == df.index[6]
    assert trade.entry_price == 200.0, "filled at the signal bar's price, not the next open"
    # tgt[6] is 0, so the exit is placed at the close of 6 and fills at open 7.
    assert trade.exit_time == df.index[7]
    assert trade.exit_price == 210.0


def test_costs_are_charged_on_both_sides():
    """Round trip on a flat price is a loss of exactly two sides of costs."""
    df = _flat(8)
    cfg = E.Config(initial_capital=100_000.0, commission_bps=10.0, slippage_bps=0.0)
    result = E.run(df, _target(df.index, [2]), cfg)

    assert len(result.trades) == 1
    # Buy 100k of stock at 10bps, sell it at 10bps: ~0.2% of notional.
    expected = -100_000.0 * 0.001 * 2
    assert abs(result.trades[0].pnl - expected) < 1.0, result.trades[0].pnl


def test_slippage_moves_the_fill_against_you_in_both_directions():
    df = _flat(8)
    cfg = E.Config(initial_capital=100_000.0, commission_bps=0.0, slippage_bps=50.0)
    result = E.run(df, _target(df.index, [2]), cfg)

    trade = result.trades[0]
    assert trade.entry_price > 100.0, "a buy must fill above the quoted price"
    assert trade.exit_price < 100.0, "a sell must fill below the quoted price"


def test_stop_is_assumed_to_fill_before_the_target():
    """When one bar contains both levels, the pessimistic one wins.

    Daily OHLC cannot say which came first. Assuming the target would be
    assuming the best case on every ambiguous bar, which across a backtest is
    worth more than most strategies' entire edge.
    """
    df = _bars([(100, 100, 100, 100),
                (100, 100, 100, 100),
                (100, 100, 100, 100),
                (100, 115, 90, 100),        # entry bar: hits -10% AND +15%
                (100, 100, 100, 100)])
    cfg = _free_config(stop_loss_pct=5.0, take_profit_pct=10.0)
    result = E.run(df, _target(df.index, [2]), cfg)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop"
    assert abs(trade.exit_price - 95.0) < 1e-9


def test_trailing_stop_ratchets_up_and_never_down():
    # The trail arms from the entry bar's own high, so that bar's low can
    # already take it out. Kept clear of it here so the ratchet is what is
    # being tested rather than same-bar arming.
    df = _bars([(100, 100, 100, 100)] * 3
               + [(115, 120, 114, 120)]     # entry at 115; high 120 -> trail 108
               + [(120, 120, 100, 100)]     # low 100 crosses it
               + [(100, 100, 100, 100)])
    cfg = _free_config(trailing_stop_pct=10.0)
    # Held for two bars only: leaving the target on past the stop would just
    # re-enter on the next open and book a second, irrelevant trade.
    result = E.run(df, _target(df.index, [2, 3]), cfg)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop"
    assert abs(trade.exit_price - 108.0) < 1e-9, "trail did not follow the high"


def test_open_position_is_closed_at_the_last_bar():
    """No trade is left hanging: an open position books at the final close, or
    the equity curve reports money the backtest never actually realised."""
    df = _flat(10)
    result = E.run(df, pd.Series(1.0, index=df.index), _free_config())

    assert result.trades, "a permanently-long strategy produced no trade at all"
    assert result.trades[-1].exit_reason == "end-of-data"
    assert result.position.iloc[-1] == 0


def test_short_entries_are_refused_unless_allowed():
    df = _flat(8)
    target = pd.Series(-1.0, index=df.index)

    assert E.run(df, target, _free_config()).trades == []
    assert E.run(df, target, _free_config(allow_short=True)).trades


# --------------------------------------------------------------- strategies

def test_every_registered_strategy_runs_and_stays_in_range():
    """A target-position strategy may only ever ask for -1, 0 or +1."""
    rng = np.random.default_rng(0)
    n = 400
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    df = _bars([(c, c * 1.01, c * 0.99, c) for c in close])
    df["Volume"] = rng.lognormal(14, 0.3, n)

    for name, spec in S.REGISTRY.items():
        if spec.get("engine"):
            continue  # Plan/Risk and swingpro engines are not target series
        signal, _ = S.build(name, df, None)
        assert len(signal) == len(df), name
        assert set(np.unique(signal.to_numpy())) <= {-1.0, 0.0, 1.0}, name


def test_unknown_strategy_and_parameter_are_request_errors_not_exits():
    """The desktop tester called SystemExit here; inside a request handler that
    takes the worker down instead of returning a 400."""
    for call in (lambda: S.get("not_a_strategy"),
                 lambda: S.resolve("ema_cross", {"nonsense": 1})):
        try:
            call()
        except BacktestError:
            pass
        else:
            raise AssertionError("expected a BacktestError")


# ------------------------------------------------------------------ metrics

def test_drawdown_table_finds_both_episodes_and_ranks_by_depth():
    equity = pd.Series([100, 90, 100, 80, 100.0],
                       index=pd.bdate_range("2024-01-01", periods=5))
    rows = M.drawdown_table(equity)

    assert len(rows) == 2
    assert abs(rows[0]["depth"] - (-0.2)) < 1e-9, "deepest drawdown is not first"
    assert abs(rows[1]["depth"] - (-0.1)) < 1e-9
    assert all(r["recovered"] for r in rows)
    assert rows[0]["recovery_bars"] == 1


def test_an_unrecovered_drawdown_is_reported_as_open():
    equity = pd.Series([100, 120, 60.0],
                       index=pd.bdate_range("2024-01-01", periods=3))
    row = M.drawdown_table(equity)[0]

    assert row["recovered"] is False
    assert row["end"] is None and row["recovery_bars"] is None


def test_ulcer_index_is_zero_for_a_curve_that_never_drops():
    equity = pd.Series(np.arange(100, 200, dtype=float),
                       index=pd.bdate_range("2024-01-01", periods=100))
    assert M.ulcer_index(equity) == 0.0
    assert M.omega(equity.pct_change().fillna(0.0)) == float("inf")


def test_beta_against_itself_is_one():
    idx = pd.bdate_range("2024-01-01", periods=120)
    rng = np.random.default_rng(3)
    curve = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))), index=idx)
    stats = M.benchmark_stats(curve, curve, 252)

    assert abs(stats["beta"] - 1.0) < 1e-9
    assert abs(stats["correlation"] - 1.0) < 1e-9
    assert abs(stats["alpha_annual"]) < 1e-9


# -------------------------------------------------------- robustness passes

def test_grid_parsing_accepts_both_the_string_and_the_structured_form():
    assert A.parse_grid("fast=10:30:10") == {"fast": [10, 20, 30]}
    assert A.parse_grid({"fast": {"min": 10, "max": 30, "step": 10}}) == {"fast": [10, 20, 30]}
    assert A.parse_grid({"fast": [5, 9]}) == {"fast": [5, 9]}
    assert len(A.combinations({"a": [1, 2], "b": [3, 4, 5]})) == 6

    for bad in ("fast=1:5:0", "garbage", {"fast": {"min": 1, "max": 5, "step": -1}}):
        try:
            A.parse_grid(bad)
        except BacktestError:
            pass
        else:
            raise AssertionError(f"expected {bad!r} to be rejected")


def test_walk_forward_windows_roll_forward_without_overlapping_their_tests():
    windows = A.walk_forward_windows(2000, folds=4, is_frac=0.7)

    assert len(windows) == 4
    for (is_a, is_b), (oos_a, oos_b) in windows:
        assert is_b == oos_a, "the test window must start where training ended"
        assert is_a < is_b < oos_b
    # Each fold trades a later, non-overlapping segment than the one before.
    tests = [oos for _, oos in windows]
    for earlier, later in zip(tests, tests[1:]):
        assert later[0] >= earlier[1]


def test_walk_forward_refuses_a_window_too_small_to_mean_anything():
    for kwargs in ({"folds": 40, "is_frac": 0.7}, {"folds": 1, "is_frac": 0.7},
                   {"folds": 4, "is_frac": 0.99}):
        try:
            A.walk_forward_windows(600, **kwargs)
        except BacktestError:
            pass
        else:
            raise AssertionError(f"expected {kwargs} to be rejected")


def test_stitch_joins_segments_on_returns_not_on_levels():
    """Each fold restarts at the same capital, so joining on levels would erase
    every fold's gain at the boundary."""
    a = pd.Series([100.0, 110.0], index=pd.bdate_range("2024-01-01", periods=2))
    b = pd.Series([100.0, 120.0], index=pd.bdate_range("2024-02-01", periods=2))
    out = A.stitch([a, b], initial=100.0)

    assert abs(out.iloc[-1] - 132.0) < 1e-9, out.to_dict()


def test_monte_carlo_is_deterministic_and_brackets_the_median():
    df = _bars([(100 + i, 100 + i + 1, 99 + i, 100 + i) for i in range(300)])
    # Alternate in and out so the run books many small trades to resample.
    target = pd.Series([1.0 if (i // 5) % 2 == 0 else 0.0 for i in range(300)],
                       index=df.index)
    result = E.run(df, target, _free_config())
    stats = M.compute(result)

    first = A.monte_carlo(result, stats, paths=200)
    second = A.monte_carlo(result, stats, paths=200)

    assert first["paths"] == 200 and first["trades_resampled"] >= 10
    assert first["final_return"] == second["final_return"], "seed did not pin the draw"
    fr = first["final_return"]
    assert fr["p5"] <= fr["median"] <= fr["p95"]
    assert 0.0 <= first["prob_profit"] <= 1.0
    assert len(first["band"]["median"]) == first["trades_resampled"]


def test_monte_carlo_declines_to_resample_too_few_trades():
    df = _flat(60)
    result = E.run(df, _target(df.index, [2]), _free_config())
    out = A.monte_carlo(result, M.compute(result), paths=100)

    assert out["paths"] == 0 and "note" in out


# --------------------------------------------------------------- data layer

def test_records_become_a_tz_naive_frame_sorted_and_deduped():
    records = [
        {"t": "2024-01-03T14:30:00+00:00", "o": 3, "h": 3, "l": 3, "c": 3, "v": 1},
        {"t": "2024-01-01T14:30:00+00:00", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1},
        {"t": "2024-01-02T14:30:00+00:00", "o": 2, "h": 2, "l": 2, "c": 2, "v": 1},
        {"t": "2024-01-02T14:30:00+00:00", "o": 9, "h": 9, "l": 9, "c": 9, "v": 1},
        {"t": "2024-01-04T14:30:00+00:00", "o": 0, "h": 0, "l": 0, "c": 0, "v": 1},
    ]
    df = D._to_df(records)

    assert df.index.tz is None, "a tz-aware index breaks every date comparison downstream"
    assert list(df.index) == sorted(df.index)
    assert len(df) == 3, "duplicate kept, or the zero-priced bar was not dropped"
    assert df["Close"].iloc[1] == 9.0, "the later duplicate should win"


def test_range_covering_picks_the_smallest_window_that_reaches_back():
    now = pd.Timestamp.now("UTC").tz_convert(None)

    assert D._range_covering(None, None) == "max"
    assert D._range_covering(None, "5y") == "5y"
    assert D._range_covering(now - pd.Timedelta(days=20), None) == "1mo"
    assert D._range_covering(now - pd.Timedelta(days=400), None) == "2y"
    assert D._range_covering(now - pd.Timedelta(days=40_000), None) == "max"


# ------------------------------------------------------------------ runner

if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a test runner reports, it does not raise
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"ok   {name}")
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
