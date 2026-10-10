"""User-data routes end to end: HTTP -> auth -> RLS -> Neon dev branch.

Quotes are stubbed so the ledger maths is deterministic. Skipped without
DATABASE_URL_APP.
"""
import uuid
from decimal import Decimal

import pytest

from conftest import auth
from flint_api import db
from flint_api.settings import settings

pytestmark = pytest.mark.skipif(not settings.database_url_app, reason="no DATABASE_URL_APP")

QUOTES = {
    "AAPL": {"regularMarketPrice": 200.0, "currency": "USD"},
    "RELIANCE.NS": {"regularMarketPrice": 2900.0, "currency": "INR"},
    "VOD.L": {"regularMarketPrice": 7000.0, "currency": "GBp"},
    "USDINR=X": {"regularMarketPrice": 85.0, "currency": "INR"},
    "GBPINR=X": {"regularMarketPrice": 110.0, "currency": "INR"},
    "OLD.NS": {"regularMarketPrice": 10.0, "currency": "INR", "stale": True},
}


@pytest.fixture()
def quotes(monkeypatch):
    import data_source
    monkeypatch.setattr(data_source, "get_quote", lambda s: dict(QUOTES[s], symbol=s) if s in QUOTES else {})


@pytest.fixture()
def users():
    ids = ["user_t" + uuid.uuid4().hex[:16] for _ in range(2)]
    yield ids
    for u in ids:
        with db.user_tx(u) as conn:
            conn.execute("delete from users where id = %s", (u,))
        db.forget_user(u)


def _onboard(client, user, cash="1000000", ccy="INR"):
    r = client.post("/api/me/onboarding", headers=auth(user), json={
        "base_currency": ccy, "markets": ["india", "us"], "starting_cash": cash})
    assert r.status_code == 200, r.text
    return client.get("/api/accounts", headers=auth(user)).json()["accounts"][0]["id"]


def _order(client, user, acct, symbol, side, qty, **extra):
    return client.post(f"/api/accounts/{acct}/orders", headers=auth(user), json={
        "symbol": symbol, "side": side, "quantity": str(qty),
        "client_order_id": str(uuid.uuid4()), **extra})


# ------------------------------------------------------------------ profile --

def test_profile_and_onboarding(client, users):
    a, _ = users
    me = client.get("/api/me", headers=auth(a)).json()
    assert me["plan"] == "free" and me["onboarded"] is False
    _onboard(client, a)
    again = client.post("/api/me/onboarding", headers=auth(a), json={
        "base_currency": "INR", "markets": ["india"], "starting_cash": "5"})
    assert again.status_code == 409
    r = client.patch("/api/me", headers=auth(a), json={"risk_per_trade_pct": "0.5", "markets": ["crypto"]})
    assert r.json()["risk_per_trade_pct"] == "0.50" and r.json()["markets"] == ["crypto"]


def test_client_cannot_set_its_own_plan(client, users):
    a, _ = users
    client.patch("/api/me", headers=auth(a), json={"plan": "pro"})
    assert client.get("/api/me", headers=auth(a)).json()["plan"] == "free"


# ------------------------------------------------------------------- ledger --

def test_buy_sell_with_fx_and_average_cost(client, users, quotes):
    a, _ = users
    acct = _onboard(client, a)
    assert _order(client, a, acct, "AAPL", "buy", 10).status_code == 201       # 10*200*85 = 170000 INR
    assert _order(client, a, acct, "RELIANCE.NS", "buy", 100).status_code == 201  # 290000 INR
    assert _order(client, a, acct, "AAPL", "sell", 4).status_code == 201

    book = client.get(f"/api/accounts/{acct}/positions", headers=auth(a)).json()
    assert Decimal(book["cash"]) == Decimal(1_000_000 - 170_000 - 290_000 + 4 * 200 * 85)
    pos = {p["symbol"]: p for p in book["positions"]}
    assert Decimal(pos["AAPL"]["quantity"]) == 6
    assert Decimal(pos["AAPL"]["avg_cost"]) == Decimal(17000)
    assert Decimal(book["realized_pnl"]) == 0  # sold at cost


def test_minor_unit_quotes_are_normalised(client, users, quotes):
    a, _ = users
    acct = _onboard(client, a)
    r = _order(client, a, acct, "VOD.L", "buy", 10)  # 7000 GBp = 70 GBP; *110 = 7700 INR each
    assert r.status_code == 201 and r.json()["currency"] == "GBP" and Decimal(r.json()["price"]) == 70
    book = client.get(f"/api/accounts/{acct}/positions", headers=auth(a)).json()
    assert Decimal(book["cash"]) == Decimal(1_000_000 - 77_000)


def test_cannot_overspend_or_oversell(client, users, quotes):
    a, _ = users
    acct = _onboard(client, a, cash="1000")
    r = _order(client, a, acct, "AAPL", "buy", 1)  # 17000 INR > 1000
    assert r.status_code == 400 and "Not enough paper cash" in r.json()["error"]
    r = _order(client, a, acct, "RELIANCE.NS", "sell", 1)
    assert r.status_code == 400 and "Short selling" in r.json()["error"]


def test_duplicate_submit_fills_once(client, users, quotes):
    a, _ = users
    acct = _onboard(client, a)
    cid = str(uuid.uuid4())
    body = {"symbol": "RELIANCE.NS", "side": "buy", "quantity": "1", "client_order_id": cid}
    first = client.post(f"/api/accounts/{acct}/orders", headers=auth(a), json=body).json()
    second = client.post(f"/api/accounts/{acct}/orders", headers=auth(a), json=body).json()
    assert second == {"fill_id": first["fill_id"], "duplicate": True}
    assert len(client.get(f"/api/accounts/{acct}/fills", headers=auth(a)).json()["fills"]) == 1


def test_client_supplied_price_is_ignored(client, users, quotes):
    a, _ = users
    acct = _onboard(client, a)
    r = _order(client, a, acct, "RELIANCE.NS", "buy", 1, price="0.01")
    assert r.status_code == 201 and Decimal(r.json()["price"]) == 2900


def test_stale_price_is_refused(client, users, quotes):
    a, _ = users
    acct = _onboard(client, a)
    assert _order(client, a, acct, "OLD.NS", "buy", 1).status_code == 503


def test_other_users_account_is_invisible(client, users, quotes):
    a, b = users
    acct = _onboard(client, a)
    _onboard(client, b)
    for r in (client.get(f"/api/accounts/{acct}/positions", headers=auth(b)),
              client.get(f"/api/accounts/{acct}/fills", headers=auth(b)),
              _order(client, b, acct, "RELIANCE.NS", "buy", 1)):
        assert r.status_code == 404


# --------------------------------------------------------------- watchlists --

def test_watchlists_are_private(client, users):
    a, b = users
    wl = client.post("/api/watchlists", headers=auth(a), json={"name": "Core"}).json()["id"]
    assert client.post(f"/api/watchlists/{wl}/items", headers=auth(a), json={"symbol": "aapl"}).status_code == 201
    assert client.get("/api/watchlists", headers=auth(a)).json()["watchlists"][0]["symbols"] == ["AAPL"]
    assert client.get("/api/watchlists", headers=auth(b)).json()["watchlists"] == []
    assert client.post(f"/api/watchlists/{wl}/items", headers=auth(b), json={"symbol": "TSLA"}).status_code == 404
    assert client.patch(f"/api/watchlists/{wl}", headers=auth(b), json={"name": "x"}).status_code == 404
    assert client.delete(f"/api/watchlists/{wl}", headers=auth(b)).status_code == 404
    assert client.post("/api/watchlists", headers=auth(a), json={"name": "Core"}).status_code == 409


# ------------------------------------------------------------------ journal --

def test_journal_soft_delete_and_restore(client, users):
    from flint_api.stores import DbJournalStore
    a, b = users
    entry = DbJournalStore(a).append({"symbol": "AAPL", "ts": 0, "decision": "APPROVE", "action": "BUY"})
    eid = entry["id"]
    listed = client.get("/api/journal", headers=auth(a)).json()["entries"]
    assert [e["id"] for e in listed] == [eid]

    assert client.delete(f"/api/journal/{eid}", headers=auth(b)).status_code == 404  # not B's
    r = client.delete(f"/api/journal/{eid}", headers=auth(a))
    assert r.status_code == 200 and "restore_until" in r.json()
    assert client.get("/api/journal", headers=auth(a)).json()["entries"] == []
    assert client.post(f"/api/journal/{eid}/restore", headers=auth(a)).status_code == 200
    assert len(client.get("/api/journal", headers=auth(a)).json()["entries"]) == 1


# ---------------------------------------------------------- export / delete --

def test_export_contains_only_own_data(client, users, quotes):
    a, b = users
    acct = _onboard(client, a)
    _order(client, a, acct, "RELIANCE.NS", "buy", 1)
    _onboard(client, b)
    r = client.get("/api/me/export", headers=auth(a))
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    data = r.json()
    assert data["profile"]["id"] == a
    assert {x["user_id"] for x in data["accounts"] + data["fills"]} == {a}


def test_delete_account_requires_matching_email(client, users, monkeypatch):
    import flint_api.routers.me as me
    a, _ = users
    calls = []

    class Resp:
        def __init__(self, code, body=None):
            self.status_code, self._body = code, body or {}

        def json(self):
            return self._body

    def fake_clerk(method, user_id):
        calls.append(method)
        return Resp(200, {"email_addresses": [{"email_address": "Me@Example.com"}]}) if method == "GET" else Resp(200)

    monkeypatch.setattr(me, "_clerk", fake_clerk)
    _onboard(client, a)
    r = client.request("DELETE", "/api/me", headers=auth(a), json={"confirm_email": "someone@else.com"})
    assert r.status_code == 403 and calls == ["GET"]
    r = client.request("DELETE", "/api/me", headers=auth(a), json={"confirm_email": " me@example.com "})
    assert r.status_code == 200 and calls[-1] == "DELETE"
    with db.user_tx(a) as conn:  # recreated empty by user_tx; previous data gone
        assert conn.execute("select count(*) as n from accounts").fetchone()["n"] == 0


def test_failed_identity_delete_keeps_data(client, users, monkeypatch):
    import flint_api.routers.me as me
    a, _ = users

    class Resp:
        def __init__(self, code, body=None):
            self.status_code, self._body = code, body or {}

        def json(self):
            return self._body

    monkeypatch.setattr(me, "_clerk", lambda m, u: Resp(200, {"email_addresses": [{"email_address": "a@b.co"}]})
                        if m == "GET" else Resp(500))
    _onboard(client, a)
    r = client.request("DELETE", "/api/me", headers=auth(a), json={"confirm_email": "a@b.co"})
    assert r.status_code == 400
    assert client.get("/api/accounts", headers=auth(a)).json()["accounts"] != []
