"""Batch quotes for watchlist rows: one Yahoo spark call per 20 symbols, the same
numbers as a single quote, and no per-symbol pile-up when Yahoo is failing.
No network: Yahoo's answers are built here in its v7 spark shape."""
import pytest

import data_source
from conftest import auth

DAY = 86_400
LAST = 1_793_000_000 - 1_793_000_000 % DAY + 15 * 3600  # an afternoon, UTC


def _result(sym: str, closes=(100.0, 110.0), session_day=0):
    """One symbol as Yahoo's spark sends it. `session_day` moves the quote's
    session past the last candle (a new session with no candle yet)."""
    stamps = [LAST - (len(closes) - 1 - i) * DAY for i in range(len(closes))]
    meta = {"currency": "INR" if sym.endswith(".NS") else "USD", "regularMarketPrice": closes[-1],
            "regularMarketTime": LAST + session_day * DAY, "gmtoffset": 0, "shortName": f"{sym} Inc"}
    return {"symbol": sym, "response": [{"meta": meta, "timestamp": stamps,
                                         "indicators": {"quote": [{"close": list(closes)}]}}]}


@pytest.fixture(autouse=True)
def empty_cache(monkeypatch):
    monkeypatch.setattr(data_source, "_mem_cache", {})
    monkeypatch.setattr(data_source, "_disk_cache", {})
    monkeypatch.setattr(data_source, "_save_disk", lambda snapshot: None)


def _yahoo(monkeypatch, answer=None):
    calls = []

    def fake(url, params=None):
        asked = params["symbols"].split(",")
        calls.append(asked)
        return {"spark": {"result": [_result(s) for s in asked if s != "GONE"] if answer is None else answer}}
    monkeypatch.setattr(data_source, "_get_json", fake)
    return calls


def test_twenty_symbols_to_a_call_and_cached_after(monkeypatch):
    calls = _yahoo(monkeypatch)
    symbols = [f"S{i}" for i in range(45)]
    got = data_source.get_quotes(symbols)
    assert [len(c) for c in calls] == [20, 20, 5]
    q = got["quotes"]["S0"]
    assert (q["regularMarketPrice"], q["previousClose"], q["currency"]) == (110.0, 100.0, "USD")
    assert round(q["regularMarketChangePercent"], 9) == 10.0
    assert got["closes"]["S0"] == [100.0, 110.0]

    data_source.get_quotes(symbols)
    assert len(calls) == 3  # fresh for a minute: no new call
    assert data_source.get_quote("S7") == got["quotes"]["S7"]  # and the single-quote cache is filled


def test_a_new_session_without_a_candle_compares_with_the_last_close(monkeypatch):
    """The single quote's rule: if the last candle isn't today's session, it is the previous close."""
    monkeypatch.setattr(data_source, "_get_json",
                        lambda url, params=None: {"spark": {"result": [_result("AAPL", (90.0, 100.0), session_day=1)]}})
    q = data_source.get_quotes(["AAPL"])["quotes"]["AAPL"]
    assert q["previousClose"] == 100.0


def test_a_symbol_yahoo_leaves_out_goes_the_single_way(monkeypatch):
    _yahoo(monkeypatch)
    single = []
    monkeypatch.setattr(data_source, "get_quote", lambda s: single.append(s) or {"symbol": s, "error": "x"})
    got = data_source.get_quotes(["AAPL", "GONE"])
    assert single == ["GONE"] and got["quotes"]["GONE"]["error"] and got["quotes"]["AAPL"]["regularMarketPrice"]


def test_when_yahoo_fails_old_data_is_marked_stale_and_nothing_piles_up(monkeypatch):
    _yahoo(monkeypatch)
    data_source.get_quotes(["AAPL"])
    monkeypatch.setattr(data_source, "FRESH_TTL", {**data_source.FRESH_TTL, "quote": -1})  # everything is old now

    def down(url, params=None):
        raise data_source.CircuitOpenError("cooling down")
    monkeypatch.setattr(data_source, "_get_json", down)
    monkeypatch.setattr(data_source, "get_quote", lambda s: pytest.fail("no per-symbol calls while Yahoo fails"))
    got = data_source.get_quotes(["AAPL", "MSFT"])
    assert got["quotes"]["AAPL"]["stale"] is True and got["quotes"]["AAPL"]["regularMarketPrice"] == 110.0
    assert "error" in got["quotes"]["MSFT"] and got["closes"]["MSFT"] == []


def test_quotes_route_cleans_and_bounds_the_list(client, monkeypatch):
    asked = []
    monkeypatch.setattr(data_source, "get_quotes", lambda symbols: asked.append(symbols) or {"quotes": {}, "closes": {}})
    assert client.get("/api/quotes?symbols=aapl,,AAPL, tcs.ns", headers=auth()).status_code == 200
    assert asked == [["AAPL", "TCS.NS"]]
    too_many = ",".join(f"S{i}" for i in range(251))
    assert client.get(f"/api/quotes?symbols={too_many}", headers=auth()).status_code == 400
    assert client.get("/api/quotes?symbols=AAPL,a%20b", headers=auth()).status_code == 400
    assert client.get("/api/quotes?symbols=AAPL").status_code == 401
