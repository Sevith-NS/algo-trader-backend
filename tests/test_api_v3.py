"""Polaris v3 backend: brokers (Zerodha, read-only), real accounts, holdings,
performance, FX, market pulse and alerts.

Unit tests run anywhere. The DB tests run end to end against Neon and are
skipped without DATABASE_URL_APP. Kite is never called: its HTTP layer is
replaced with a fake that records what was asked.
"""
import base64
import dataclasses
import hashlib
import json
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from urllib.parse import parse_qs, urlsplit

import pytest

from conftest import auth
from flint_api import db, performance, secretbox
from flint_api.brokers import zerodha as kite
from flint_api.settings import settings

needs_db = pytest.mark.skipif(not settings.database_url_app, reason="no DATABASE_URL_APP")
KEY = base64.b64encode(b"k" * 32).decode()


# The user's own Kite Connect app, entered in Settings; nothing is on the server.
APP = {"api_key": "kitekey", "api_secret": "kitesecret"}
CALLBACK = "https://polaris.example/settings/brokers/callback"


@pytest.fixture()
def configured(monkeypatch):
    """Cookie key and a trusted frontend origin, without touching the real .env."""
    from flint_api.routers import brokers as broker_routes
    s = dataclasses.replace(settings, broker_token_key=KEY, authorized_parties=("https://polaris.example",))
    for module in (secretbox, broker_routes):
        monkeypatch.setattr(module, "settings", s)
    return s


class FakeKite:
    """Stands in for api.kite.trade. `holdings_status` 403 simulates an expired session."""

    def __init__(self):
        self.calls = []
        self.holdings_status = 200
        self.user_id = "AB1234"
        self.holdings = [
            {"tradingsymbol": "RELIANCE", "exchange": "NSE", "isin": "INE002A01018", "quantity": 10,
             "t1_quantity": 2, "average_price": 2500, "last_price": 2900, "close_price": 2850},
            {"tradingsymbol": "INFY-BE", "exchange": "NSE", "isin": "INE009A01021", "quantity": 5,
             "t1_quantity": 0, "average_price": 1400, "last_price": 1500, "close_price": 1500},
            {"tradingsymbol": "GOLDBEES", "exchange": "MCX", "quantity": 1, "average_price": 50},
        ]
        self.positions = [
            {"tradingsymbol": "TCS", "exchange": "NSE", "product": "MIS", "quantity": -3,
             "average_price": 4000, "last_price": 3990, "close_price": 4010},
            {"tradingsymbol": "RELIANCE", "exchange": "NSE", "product": "CNC", "quantity": 2,
             "average_price": 2890, "last_price": 2900, "close_price": 2850},
            {"tradingsymbol": "NIFTY26OCTFUT", "exchange": "NFO", "product": "NRML", "quantity": 50,
             "average_price": 25000, "last_price": 25100, "close_price": 25050},
        ]

    def __call__(self, method, url, headers=None, timeout=None, data=None, params=None):
        path = urlsplit(url).path
        self.calls.append((method, path, data, params, headers))
        if method == "POST" and path == "/session/token":
            return _Resp(200, {"status": "success", "data": {
                "user_id": self.user_id, "user_name": "Test Trader", "access_token": "acc_" + data["request_token"]}})
        if method == "GET" and path == "/portfolio/holdings":
            if self.holdings_status == 403:
                return _Resp(403, {"status": "error", "error_type": "TokenException", "message": "expired"})
            return _Resp(200, {"status": "success", "data": self.holdings})
        if method == "GET" and path == "/portfolio/positions":
            return _Resp(200, {"status": "success", "data": {"net": self.positions, "day": []}})
        if method == "DELETE" and path == "/session/token":
            return _Resp(200, {"status": "success", "data": True})
        return _Resp(404, {"status": "error"})


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


@pytest.fixture()
def fake_kite(monkeypatch):
    fake = FakeKite()
    monkeypatch.setattr(kite.requests, "request", fake)
    return fake


# ======================================================================= unit --

def test_kite_checksum_and_login_url(configured):
    assert kite.checksum(APP, "rt123") == hashlib.sha256(b"kitekeyrt123kitesecret").hexdigest()
    url = urlsplit(kite.login_url("STATE_abc", APP, CALLBACK))
    q = parse_qs(url.query)
    assert url.netloc == "kite.zerodha.com" and q["api_key"] == ["kitekey"] and q["v"] == ["3"]
    assert parse_qs(q["redirect_params"][0]) == {"state": ["STATE_abc"]}
    assert "kitesecret" not in kite.login_url("x", APP, CALLBACK)


def test_kite_session_ends_at_six_ist():
    ist = timezone(timedelta(hours=5, minutes=30))
    before = datetime(2026, 10, 15, 5, 0, tzinfo=ist)
    after = datetime(2026, 10, 15, 9, 0, tzinfo=ist)
    assert kite.session_expiry(before) == datetime(2026, 10, 15, 6, 0, tzinfo=ist)
    assert kite.session_expiry(after) == datetime(2026, 10, 16, 6, 0, tzinfo=ist)


@pytest.mark.parametrize("sym,exch,want", [
    ("RELIANCE", "NSE", "RELIANCE.NS"), ("M&M", "NSE", "M&M.NS"), ("BAJAJ-AUTO", "NSE", "BAJAJ-AUTO.NS"),
    ("INFY-BE", "NSE", "INFY.NS"), ("TCS", "BSE", "TCS.BO"), ("GOLD", "MCX", None), ("bad sym", "NSE", None)])
def test_kite_symbol_mapping(sym, exch, want):
    assert kite.yahoo_symbol(sym, exch) == want


def test_kite_holdings_mapping(configured, fake_kite):
    rows, skipped = kite.holdings("tok", APP)
    by = {r["symbol"]: r for r in rows}
    assert skipped == 1  # MCX
    assert by["RELIANCE.NS"]["quantity"] == 12  # T1 shares count
    assert by["INFY.NS"]["broker_symbol"] == "INFY-BE"
    assert fake_kite.calls[0][4]["Authorization"] == "token kitekey:tok"


def test_kite_expired_token(configured, fake_kite):
    fake_kite.holdings_status = 403
    with pytest.raises(kite.TokenExpired):
        kite.holdings("tok", APP)


def test_secretbox_binds_owner_and_row(configured):
    blob = secretbox.seal("access-123", user_id="user_a", row_id="r1")
    assert b"access-123" not in blob
    assert secretbox.open_(blob, user_id="user_a", row_id="r1") == "access-123"
    assert secretbox.open_(blob, user_id="user_b", row_id="r1") is None
    assert secretbox.open_(blob, user_id="user_a", row_id="r2") is None


def test_secretbox_refuses_bad_key(monkeypatch):
    from flint_api.errors import Unavailable
    monkeypatch.setattr(secretbox, "settings", dataclasses.replace(settings, broker_token_key="c2hvcnQ="))
    with pytest.raises(Unavailable):
        secretbox.seal("x", user_id="u", row_id="r")


def test_twr_ignores_deposits():
    # 100 -> 110 is +10%. Next day ends at 165 including 50 deposited at the
    # close, so the invested money went 110 -> 115 (+4.55%): 15% in all, not 65%.
    assert performance.twr([100, 110, 165], [0, 0, 50]) == pytest.approx((1.10 * 115 / 110 - 1) * 100)


def test_xirr_known_answer():
    flows = [(date(2025, 1, 1), -100.0), (date(2026, 1, 1), 110.0)]
    assert performance.xirr(flows) == pytest.approx(10.0, abs=1e-4)
    assert performance.xirr([(date(2025, 1, 1), 100.0)]) is None


def test_summarise_holds_back_xirr_on_short_history():
    d0 = date(2026, 10, 1)
    out = performance.summarise([d0, d0 + timedelta(days=5)], [100.0, 105.0], [0.0, 0.0])
    assert out["twr_pct"] == pytest.approx(5.0)
    assert out["xirr_pct"] is None and "30 days" in out["xirr_note"]
    assert out["series"][-1]["index"] == pytest.approx(105.0)


def test_period_start_never_before_account():
    today, created = date(2026, 10, 15), date(2026, 9, 1)
    assert performance.period_start("1y", created, today) == created
    assert performance.period_start("1m", created, today) == today - timedelta(days=31)
    assert performance.period_start("ytd", date(2020, 1, 1), today) == date(2026, 1, 1)


def _spark(symbols, change=1.0):
    return {s: {"symbol": s, "fulldayPrice": 100 + i, "fulldayChange": change * (1 if i % 3 else -1),
                "fulldayChangePercent": change * (1 if i % 3 else -1), "timestamp": [1792049960],
                "close": [100 + i]} for i, s in enumerate(symbols)}


def test_pulse_us(monkeypatch):
    import data_source
    import pulse_engine
    monkeypatch.setattr(data_source, "get_json_cached",
                        lambda key, url, params, fresh_ttl: _spark(params["symbols"].split(",")))
    p = pulse_engine.get_pulse("us")
    assert [i["symbol"] for i in p["indices"]][:2] == ["^GSPC", "^IXIC"]
    assert len(p["sectors"]) == 11 and p["sectors"][0]["change_pct"] >= p["sectors"][-1]["change_pct"]
    assert p["breadth"]["universe"] == "S&P 500 sector ETFs"
    assert p["breadth"]["advancing"] + p["breadth"]["declining"] == 11
    assert p["missing"] == []


def test_pulse_prefers_closes_over_spark_fullday(monkeypatch):
    # The XLK case from 14 Oct 2026: spark said -1.62% while the closes (and
    # /api/quote) said -2.98%. Two daily closes must win over the fullday field.
    import data_source
    import pulse_engine
    nodes = {"XLK": {"symbol": "XLK", "fulldayPrice": 190.43, "fulldayChange": -3.14, "fulldayChangePercent": -1.623,
                     "timestamp": [1791921600, 1792008000], "close": [196.28, 190.43]},
             "XLV": {"symbol": "XLV", "fulldayPrice": 100.0, "fulldayChange": 1.0, "fulldayChangePercent": 1.0,
                     "timestamp": [1792008000], "close": [100.0]}}
    monkeypatch.setattr(data_source, "get_json_cached", lambda key, url, params, fresh_ttl: nodes)
    rows, missing = pulse_engine._quotes([("XLK", "Technology"), ("XLV", "Health care")])
    by = {r["symbol"]: r for r in rows}
    assert by["XLK"]["change_pct"] == -2.98 and by["XLK"]["price"] == 190.43
    assert by["XLV"]["change_pct"] == 1.0  # one close only: the fullday field is the fallback
    assert missing == []


@pytest.mark.parametrize("coverage,universe", [(0.6, "NSE sector indices"), (96.0, "NIFTY 500 stocks")])
def test_india_breadth_needs_half_the_index(monkeypatch, coverage, universe):
    import data_source
    import movers_engine
    import pulse_engine
    monkeypatch.setattr(data_source, "get_json_cached",
                        lambda key, url, params, fresh_ttl: _spark(params["symbols"].split(",")))
    priced = round(500 * coverage / 100)
    monkeypatch.setattr(movers_engine, "get_movers", lambda scope, limit: {
        "priced": priced, "scanned": 500, "coverage_pct": coverage, "partial": coverage < 100, "as_of": "2026-10-15",
        "breadth": {"advancing": priced - 1, "declining": 1, "unchanged": 0}})
    assert pulse_engine.get_pulse("in")["breadth"]["universe"] == universe


def test_pulse_reports_missing_and_total_outage(monkeypatch):
    import data_source
    import pulse_engine
    monkeypatch.setattr(data_source, "get_json_cached",
                        lambda key, url, params, fresh_ttl: _spark(params["symbols"].split(",")[1:]))
    assert "^GSPC" in pulse_engine.get_pulse("us")["missing"]
    monkeypatch.setattr(data_source, "get_json_cached", lambda *a, **k: None)
    assert "error" in pulse_engine.get_pulse("us")


def test_session_hours():
    import pulse_engine
    # Thu 15 Oct 2026, 14:00 UTC = 10:00 EDT (US open) and 19:30 IST (India closed).
    t = datetime(2026, 10, 15, 14, 0, tzinfo=timezone.utc)
    assert pulse_engine.session("us", t)["state"] == "open"
    assert pulse_engine.session("in", t)["state"] == "closed"
    # Same wall time in January is 09:00 EST: before the US open.
    assert pulse_engine.session("us", datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc))["state"] == "closed"
    assert pulse_engine.session("in", datetime(2026, 10, 17, 5, 0, tzinfo=timezone.utc))["state"] == "closed"  # Sat


# ===================================================================== routes --

def test_new_routes_require_auth(client):
    for path in ("/api/brokers", "/api/alerts", "/api/alerts/firings", "/api/fx?pair=USDINR",
                 "/api/markets/pulse", f"/api/accounts/{uuid.uuid4()}/holdings",
                 f"/api/accounts/{uuid.uuid4()}/performance"):
        assert client.get(path).status_code == 401, path
    for path in ("/api/brokers/zerodha/connect", "/api/brokers/zerodha/session", "/api/alerts"):
        assert client.post(path, json={}).status_code == 401, path


def test_fx_route(client, monkeypatch):
    import data_source
    monkeypatch.setattr(data_source, "get_quote", lambda s: {"symbol": s, "regularMarketPrice": 85.1,
                                                             "regularMarketTime": 1792049960})
    r = client.get("/api/fx", params={"pair": "usdinr"}, headers=auth())
    assert r.status_code == 200
    body = r.json()
    assert body["pair"] == "USDINR" and body["rate"] == 85.1 and body["as_of"].startswith("2026-")
    assert body["source"] and body["stale"] is False
    assert client.get("/api/fx", params={"pair": "INRINR"}, headers=auth()).status_code == 400
    assert client.get("/api/fx", params={"pair": "US;INR"}, headers=auth()).status_code == 422


def test_pulse_route(client, monkeypatch):
    import pulse_engine
    monkeypatch.setattr(pulse_engine, "get_pulse", lambda m: {"market": m, "indices": []})
    assert client.get("/api/markets/pulse?market=us", headers=auth()).json()["market"] == "us"
    assert client.get("/api/markets/pulse?market=eu", headers=auth()).status_code == 422
    monkeypatch.setattr(pulse_engine, "get_pulse", lambda m: {"error": "Market data is temporarily unavailable."})
    assert client.get("/api/markets/pulse", headers=auth()).status_code == 503


def test_connect_without_cookie_key_is_503(client, configured, monkeypatch):
    from flint_api.routers import brokers as broker_routes
    monkeypatch.setattr(broker_routes, "settings", dataclasses.replace(configured, broker_token_key=""))
    r = client.post("/api/brokers/zerodha/connect", headers=auth(),
                    json={"app": APP, "keep": "30d", "redirect_uri": CALLBACK})
    assert r.status_code == 503


# ========================================================================= DB --

@pytest.fixture()
def users():
    ids = ["user_t" + uuid.uuid4().hex[:16] for _ in range(2)]
    yield ids
    for u in ids:
        with db.user_tx(u) as conn:
            conn.execute("delete from users where id = %s", (u,))
        db.forget_user(u)


class Browser:
    """Plays the frontend BFF: keeps the pb_* cookies the API hands back in
    X-Broker-Cookies and sends them on every call in X-Broker-Sessions."""

    def __init__(self, client, user):
        self.client, self.user, self.cookies = client, user, {}

    def request(self, method, path, **kw):
        headers = auth(self.user)
        if self.cookies:
            headers["x-broker-sessions"] = ";".join(f"{k}={v}" for k, v in self.cookies.items())
        r = self.client.request(method, path, headers=headers, **kw)
        for c in json.loads(r.headers.get("x-broker-cookies", "[]")):
            if c["value"] is None:
                self.cookies.pop(c["name"], None)
            else:
                self.cookies[c["name"]] = c["value"]
        return r

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def post(self, path, **kw):
        return self.request("POST", path, **kw)

    def delete(self, path, **kw):
        return self.request("DELETE", path, **kw)


def _start(b, **body):
    r = b.post("/api/brokers/zerodha/connect", json=body or {"app": APP, "keep": "30d", "redirect_uri": CALLBACK})
    assert r.status_code == 200, r.text
    return parse_qs(parse_qs(urlsplit(r.json()["redirect_url"]).query)["redirect_params"][0])["state"][0]


def _connect(b):
    state = _start(b)
    return state, b.post("/api/brokers/zerodha/session",
                         json={"state": state, "params": {"request_token": "reqtok12345"}})


@needs_db
def test_broker_connect_and_live_holdings_store_nothing(client, users, configured, fake_kite, monkeypatch):
    import data_source
    monkeypatch.setattr(data_source, "get_quote", lambda s: {"symbol": s, "regularMarketPrice": 85.0})
    a, _ = users
    b = Browser(client, a)
    _state, r = _connect(b)
    assert r.status_code == 201, r.text
    conn = r.json()["connection"]
    assert conn["status"] == "connected" and conn["broker_user_id"] == "AB1234" and conn["label"] == "Zerodha"
    # The exchange was signed with the secret, which never left the server otherwise.
    post = next(c for c in fake_kite.calls if c[0] == "POST")
    assert post[2]["checksum"] == hashlib.sha256(b"kitekeyreqtok12345kitesecret").hexdigest()
    # The connection is one sealed cookie; the token isn't readable in it, and the state is used up.
    assert set(b.cookies) == {"pb_" + conn["id"].replace("-", "")}
    assert "acc_reqtok12345" not in next(iter(b.cookies.values()))

    accounts = b.get("/api/accounts").json()["accounts"]
    real = next(x for x in accounts if x["kind"] == "real")
    assert real["id"] == conn["id"] and real["starting_cash"] is None and real["broker"] == "zerodha"
    assert real["sync"]["status"] == "connected"

    h = b.get(f"/api/accounts/{real['id']}/holdings").json()
    rel = next(x for x in h["holdings"] if x["symbol"] == "RELIANCE.NS")
    assert rel["quantity"] == "12" and rel["market_value"] == "34800.0000" and rel["currency"] == "INR"
    assert rel["unrealized_pnl"] == "4800.0000" and rel["day_change_pct"] == "1.75"
    assert h["base_currency"] == "INR" and h["totals"]["market_value"] == "42300.0000" and h["as_of"]

    # The broker watchlist: holdings and today's positions, one entry per symbol.
    lists = b.get("/api/brokers/watchlist").json()["lists"]
    assert len(lists) == 1 and lists[0]["connection"]["label"] == "Zerodha"
    items = {i["symbol"]: i for i in lists[0]["items"]}
    assert set(items) == {"RELIANCE.NS", "INFY.NS", "TCS.NS"}
    assert items["RELIANCE.NS"]["held"] == "12" and items["RELIANCE.NS"]["position"] == "2"
    assert items["TCS.NS"]["held"] is None and items["TCS.NS"]["position"] == "-3"

    # Broker accounts take no orders: they aren't accounts in the database at all.
    r = b.post(f"/api/accounts/{real['id']}/orders", json={
        "symbol": "AAPL", "side": "buy", "quantity": "1", "client_order_id": str(uuid.uuid4())})
    assert r.status_code == 404
    perf = b.get(f"/api/accounts/{real['id']}/performance").json()
    assert perf["kind"] == "real" and perf["series"] == []

    # Nothing broker-related reached the database.
    with db.user_tx(a) as c:
        assert c.execute("select count(*) as n from accounts where kind <> 'paper'").fetchone()["n"] == 0
        tables = {r["t"] for r in c.execute("select table_name as t from information_schema.tables "
                                            "where table_schema = 'public'").fetchall()}
    assert not tables & {"broker_connections", "broker_link_states", "holdings", "broker_positions",
                         "account_snapshots"}


@needs_db
def test_broker_state_is_single_use_per_user_and_must_match(client, users, configured, fake_kite):
    a, other = users
    b = Browser(client, a)
    state, r = _connect(b)
    assert r.status_code == 201
    replay = b.post("/api/brokers/zerodha/session", json={"state": state, "params": {"request_token": "reqtok12345"}})
    assert replay.status_code == 403
    # An echoed state that isn't this browser's is refused, and burns the pending one.
    _start(b)
    r = b.post("/api/brokers/zerodha/session", json={"state": "x" * 43, "params": {"request_token": "reqtok12345"}})
    assert r.status_code == 403 and "pb_state" not in b.cookies
    # Another user presenting A's cookies gets nothing: they are sealed to A.
    stolen = Browser(client, other)
    _start(b)
    stolen.cookies = dict(b.cookies)
    r = stolen.post("/api/brokers/zerodha/session", json={"params": {"request_token": "reqtok12345"}})
    assert r.status_code == 403
    assert stolen.get("/api/brokers").json()["connections"] == []
    assert all(x["kind"] == "paper" for x in stolen.get("/api/accounts").json()["accounts"])


@needs_db
def test_state_cookie_alone_completes_brokers_that_dont_echo_it(client, users, configured, fake_kite):
    b = Browser(client, users[0])
    _start(b)
    r = b.post("/api/brokers/zerodha/session", json={"params": {"request_token": "reqtok12345"}})
    assert r.status_code == 201, r.text
    # Without a pending connect in this browser there is nothing to complete.
    assert Browser(client, users[0]).post("/api/brokers/zerodha/session",
                                          json={"params": {"request_token": "reqtok12345"}}).status_code == 403


@needs_db
def test_expired_session_needs_reconnect_and_disconnect_forgets(client, users, configured, fake_kite):
    b = Browser(client, users[0])
    _, r = _connect(b)
    conn_id = r.json()["connection"]["id"]

    fake_kite.holdings_status = 403
    r = b.get(f"/api/accounts/{conn_id}/holdings")
    assert r.status_code == 409 and r.json()["code"] == "needs_reconnect"
    assert b.get("/api/brokers").json()["connections"][0]["status"] == "needs_reconnect"
    assert b.get("/api/brokers/watchlist").json()["lists"][0]["connection"]["status"] == "needs_reconnect"

    # Reconnecting reuses the saved app keys (none typed again) and keeps the id.
    fake_kite.holdings_status = 200
    state = _start(b, connection_id=conn_id, redirect_uri=CALLBACK)
    r = b.post("/api/brokers/zerodha/session", json={"state": state, "params": {"request_token": "reqtok12345"}})
    assert r.json()["connection"]["id"] == conn_id and r.json()["connection"]["status"] == "connected"

    assert b.delete(f"/api/brokers/{conn_id}").status_code == 204
    assert any(c[0] == "DELETE" and c[1] == "/session/token" for c in fake_kite.calls)  # broker logout
    assert b.cookies == {} and b.get("/api/brokers").json()["connections"] == []
    assert b.delete(f"/api/brokers/{conn_id}").status_code == 404


@needs_db
def test_alerts_fire_once_and_rearm(client, users, monkeypatch):
    import data_source
    from flint_api.routers import alerts
    price = {"v": 100.0, "chg": 1.0}
    monkeypatch.setattr(data_source, "get_quote", lambda s: {"symbol": s, "regularMarketPrice": price["v"],
                                                             "regularMarketChangePercent": price["chg"]})
    a, b = users
    r = client.post("/api/alerts", headers=auth(a), json={"symbol": "aapl", "kind": "price_above", "threshold": "105"})
    assert r.status_code == 201 and r.json()["reference_price"] == "100"
    move = client.post("/api/alerts", headers=auth(a), json={"symbol": "TCS.NS", "kind": "move_pct", "threshold": "3"})
    assert client.post("/api/alerts", headers=auth(a),
                       json={"symbol": "AAPL", "kind": "move_pct", "threshold": "150"}).status_code == 400

    alerts.evaluate(a, force=True)
    assert client.get("/api/alerts/firings", headers=auth(a)).json()["unseen"] == 0

    price.update(v=106.0, chg=-3.5)
    alerts.evaluate(a, force=True)
    alerts.evaluate(a, force=True)  # fires once, not twice
    f = client.get("/api/alerts/firings", headers=auth(a)).json()
    assert f["unseen"] == 2 and {x["symbol"] for x in f["firings"]} == {"AAPL", "TCS.NS"}
    assert {x["status"] for x in client.get("/api/alerts", headers=auth(a)).json()["alerts"]} == {"triggered"}
    assert client.get("/api/alerts/firings", headers=auth(b)).json()["firings"] == []

    assert client.post("/api/alerts/firings/seen", headers=auth(a)).status_code == 204
    assert client.get("/api/alerts/firings", headers=auth(a)).json()["unseen"] == 0

    alert_id = r.json()["id"]
    rearmed = client.patch(f"/api/alerts/{alert_id}", headers=auth(a), json={"status": "active", "threshold": "120"})
    assert rearmed.json()["status"] == "active" and rearmed.json()["triggered_at"] is None
    assert client.patch(f"/api/alerts/{alert_id}", headers=auth(b), json={"status": "paused"}).status_code == 404
    assert client.delete(f"/api/alerts/{move.json()['id']}", headers=auth(a)).status_code == 204


@needs_db
def test_stale_quotes_never_fire(client, users, monkeypatch):
    import data_source
    from flint_api.routers import alerts
    monkeypatch.setattr(data_source, "get_quote",
                        lambda s: {"symbol": s, "regularMarketPrice": 500.0, "stale": True})
    a, _ = users
    client.post("/api/alerts", headers=auth(a), json={"symbol": "AAPL", "kind": "price_above", "threshold": "1"})
    alerts.evaluate(a, force=True)
    assert client.get("/api/alerts/firings", headers=auth(a)).json()["firings"] == []


@needs_db
def test_paper_performance(client, users, monkeypatch):
    import pandas as pd
    import data_source
    monkeypatch.setattr(data_source, "get_quote", lambda s: {"symbol": s, "regularMarketPrice": 100.0,
                                                             "currency": "INR"})
    a, _ = users
    r = client.post("/api/me/onboarding", headers=auth(a), json={
        "base_currency": "INR", "markets": ["india"], "starting_cash": "10000"})
    assert r.status_code == 200
    acct = client.get("/api/accounts", headers=auth(a)).json()["accounts"][0]
    assert acct["kind"] == "paper" and acct["sync"] is None
    r = client.post(f"/api/accounts/{acct['id']}/orders", headers=auth(a), json={
        "symbol": "RELIANCE.NS", "side": "buy", "quantity": "10", "client_order_id": str(uuid.uuid4())})
    assert r.status_code == 201, r.text

    days = pd.bdate_range(date.today() - timedelta(days=40), date.today())
    closes = pd.Series([100.0 + i for i in range(len(days))], index=days)
    monkeypatch.setattr(data_source, "get_close_series", lambda s, p: closes)
    perf = client.get(f"/api/accounts/{acct['id']}/performance?period=1m", headers=auth(a)).json()
    # Created today: one point, so no returns yet, and an honest note.
    assert perf["kind"] == "paper" and perf["series"] == [] and perf["xirr_note"]


@needs_db
def test_export_has_no_broker_data(client, users, configured, fake_kite):
    b = Browser(client, users[0])
    _connect(b)
    data = b.get("/api/me/export").json()
    assert "alerts" in data and "alert_firings" in data
    assert not {"broker_connections", "holdings", "broker_positions", "account_snapshots"} & set(data)
    assert "AB1234" not in json.dumps(data)
