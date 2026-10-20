"""The public landing snapshot: no auth, one upstream build per TTL, and only
index levels plus one FX rate in the payload."""
from flint_api.routers import public


def _fake_pulse(calls):
    def get_pulse(market):
        calls.append(market)
        return {"label": market, "currency": "INR", "session": {"state": "open"},
                "indices": [{"symbol": "^X", "name": "X", "price": 100.0, "change": 1.0, "change_pct": 1.0,
                             "as_of": "t", "spark": [1, 2]}],
                "breadth": {"universe": "u", "advancing": 3, "declining": 2, "unchanged": 0, "partial": False,
                            "priced": 5},
                "source": "Yahoo Finance (delayed)"}
    return get_pulse


def test_landing_is_public_and_built_once_per_ttl(client, monkeypatch):
    calls = []
    monkeypatch.setattr(public, "_snapshot", None)
    monkeypatch.setattr(public.pulse_engine, "get_pulse", _fake_pulse(calls))
    monkeypatch.setattr(public.data_source, "get_quote",
                        lambda s: {"regularMarketPrice": 84.0, "regularMarketPreviousClose": 83.0,
                                   "regularMarketTime": 1_700_000_000})

    first = client.get("/api/public/landing")
    second = client.get("/api/public/landing")

    assert first.status_code == 200 and second.json() == first.json()
    assert calls == ["in", "us"]  # the second request reused the snapshot
    assert first.headers["cache-control"] == "public, max-age=60"
    body = first.json()
    assert set(body["india"]["indices"][0]) == {"symbol", "name", "price", "change", "change_pct", "as_of"}
    assert body["fx"]["rate"] == 84.0 and body["fx"]["change_pct"] == 1.2


def test_landing_survives_an_upstream_failure(client, monkeypatch):
    def boom(_):
        raise RuntimeError("upstream down")
    monkeypatch.setattr(public, "_snapshot", None)
    monkeypatch.setattr(public.pulse_engine, "get_pulse", boom)
    monkeypatch.setattr(public.data_source, "get_quote", boom)

    r = client.get("/api/public/landing")

    assert r.status_code == 200
    assert r.json()["india"] is None and r.json()["us"] is None and r.json()["fx"] is None
