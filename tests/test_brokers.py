"""Read-only broker adapters and the generic broker routes, with each user's
own broker-app keys and chosen expiry. No broker is ever called: each adapter's HTTP layer is replaced with a
fake that records what was asked and answers like the documented API.
"""
import base64
import dataclasses
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlsplit

import pytest

from conftest import auth
from flint_api import secretbox
from flint_api.brokers import ADAPTERS, angel, binance, dhan, fyers, groww, indmoney, upstox, zerodha
from flint_api.brokers.base import TokenExpired, jwt_expiry, merge
from flint_api.errors import AppError
from flint_api.settings import settings

KEY = base64.b64encode(b"k" * 32).decode()


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class Fake:
    """Routes (method, path) to a canned (status, body); records every call."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, method, url, headers=None, timeout=None, **kw):
        path = urlsplit(url).path
        self.calls.append({"method": method, "path": path, "url": url, "headers": headers or {}, **kw})
        status, body = self.routes.get((method, path), (404, {}))
        return _Resp(status, body)


ORIGIN = "https://polaris.example"
# Each user's own broker apps, as typed in Settings. Nothing is on the server.
FY = {"app_id": "APP-100", "app_secret": "fysecret"}
ANGEL = {"api_key": "angelkey"}
DHAN = {"client_id": "1100001", "api_key": "dhkey", "api_secret": "dhsecret"}
UP = {"api_key": "upkey", "api_secret": "upsecret"}
KITE = {"api_key": "kitekey", "api_secret": "kitesecret"}


def cb(broker: str) -> str:
    return ORIGIN + "/settings/brokers/callback" + ("" if broker == "zerodha" else f"/{broker}")


@pytest.fixture()
def configured(monkeypatch):
    from flint_api.routers import brokers as broker_routes
    s = dataclasses.replace(settings, broker_token_key=KEY, authorized_parties=(ORIGIN,))
    for module in (secretbox, broker_routes):
        monkeypatch.setattr(module, "settings", s)
    return s


def _fake(monkeypatch, module, routes):
    fake = Fake(routes)
    monkeypatch.setattr(module.requests, "request", fake)
    return fake


def _jwt(exp: int) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return f"eyJhbGciOiJIUzI1NiJ9.{payload}.sig"


# ======================================================================= base --

def test_registry_lists_all_brokers_in_order():
    assert list(ADAPTERS) == ["zerodha", "upstox", "groww", "angel", "fyers", "dhan", "indmoney", "binance"]
    for a in ADAPTERS.values():
        for name in ("LABEL", "CURRENCY", "FIELDS", "HELP", "HELP_URL", "create_session", "holdings", "positions",
                     "logout"):
            assert hasattr(a, name), (a.NAME, name)
        assert a.FIELDS and a.HELP_URL.startswith("https://")
        # OAuth brokers send the user to a login page; the others take pasted read-only credentials.
        assert hasattr(a, "login_url") if a.AUTH == "oauth" else a.AUTH == "keys"


def test_jwt_expiry_reads_exp_and_falls_back():
    fallback = datetime(2030, 1, 1, tzinfo=timezone.utc)
    assert jwt_expiry(_jwt(1_900_000_000), fallback) == datetime.fromtimestamp(1_900_000_000, timezone.utc)
    assert jwt_expiry("not-a-jwt", fallback) == fallback


def test_merge_weights_average_and_keeps_zero_positions():
    from decimal import Decimal as D
    rows = merge([{"symbol": "A", "quantity": D(2), "avg_price": D(10)},
                  {"symbol": "A", "quantity": D(2), "avg_price": D(20)},
                  {"symbol": "B", "quantity": D(1), "avg_price": D(5)},
                  {"symbol": "B", "quantity": D(-1), "avg_price": D(6)}], "symbol")
    by = {r["symbol"]: r for r in rows}
    assert by["A"]["quantity"] == 4 and by["A"]["avg_price"] == 15
    assert by["B"]["quantity"] == 0 and by["B"]["avg_price"] == 5


def test_oauth_adapters_refuse_missing_app_keys():
    for module in (zerodha, upstox, fyers, angel, dhan):
        with pytest.raises(AppError, match="app keys are missing"):
            module.login_url("state", {}, cb(module.NAME))


# ====================================================================== FYERS --

FYERS_HOLDINGS = {"s": "ok", "code": 200, "holdings": [
    {"symbol": "NSE:IDEA-EQ", "isin": "INE669E01016", "quantity": 3, "costPrice": 6.48, "ltp": 7.9},
    {"symbol": "BSE:TCS-A", "isin": "INE467B01029", "quantity": 1, "costPrice": 4000, "ltp": 4100},
    {"symbol": "MCX:GOLDM-X", "quantity": 1, "costPrice": 1}]}
FYERS_POSITIONS = {"s": "ok", "code": 200, "netPositions": [
    {"symbol": "NSE:SBIN-EQ", "netQty": 10, "netAvg": 800.5, "ltp": 810, "productType": "INTRADAY"},
    {"symbol": "MCX:SILVERMIC20AUGFUT", "netQty": 1, "netAvg": 72256, "ltp": 72717, "productType": "MARGIN"}]}


def test_fyers_login_url_and_session(configured, monkeypatch):
    q = parse_qs(urlsplit(fyers.login_url("STATE_x", FY, cb("fyers"))).query)
    assert q == {"client_id": ["APP-100"], "redirect_uri": [cb("fyers")],
                 "response_type": ["code"], "state": ["STATE_x"]}
    fake = _fake(monkeypatch, fyers, {
        ("POST", "/api/v3/validate-authcode"): (200, {"s": "ok", "code": 200, "access_token": "fy_tok"}),
        ("GET", "/api/v3/profile"): (200, {"s": "ok", "code": 200, "data": {"fy_id": "XY01234", "name": "Ravi"}})})
    session = fyers.create_session({"auth_code": "abc.def"}, FY, cb("fyers"))
    assert session.access_token == "fy_tok" and session.user_id == "XY01234" and session.user_name == "Ravi"
    exchange = fake.calls[0]
    # The secret is only ever sent hashed with the app id.
    assert exchange["json"] == {"grant_type": "authorization_code", "code": "abc.def",
                                "appIdHash": hashlib.sha256(b"APP-100:fysecret").hexdigest()}
    assert "fysecret" not in json.dumps(exchange["json"])
    assert fake.calls[1]["headers"]["Authorization"] == "APP-100:fy_tok"


def test_fyers_requires_auth_code(configured):
    with pytest.raises(AppError):
        fyers.create_session({"s": "ok"}, FY, cb("fyers"))


def test_fyers_holdings_and_positions(configured, monkeypatch):
    _fake(monkeypatch, fyers, {("GET", "/api/v3/holdings"): (200, FYERS_HOLDINGS),
                               ("GET", "/api/v3/positions"): (200, FYERS_POSITIONS)})
    rows, skipped = fyers.holdings("tok", FY)
    by = {r["symbol"]: r for r in rows}
    assert skipped == 1 and set(by) == {"IDEA.NS", "TCS-A.BO"}
    assert by["IDEA.NS"]["broker_symbol"] == "IDEA-EQ" and str(by["IDEA.NS"]["avg_price"]) == "6.48"
    positions, skipped = fyers.positions("tok", FY)
    assert skipped == 1 and positions[0]["symbol"] == "SBIN.NS" and positions[0]["product"] == "INTRADAY"


@pytest.mark.parametrize("status,body", [(401, {}), (200, {"s": "error", "code": -16, "message": "expired"})])
def test_fyers_expired_token(configured, monkeypatch, status, body):
    _fake(monkeypatch, fyers, {("GET", "/api/v3/holdings"): (status, body)})
    with pytest.raises(TokenExpired):
        fyers.holdings("tok", FY)


# ================================================================== Angel One --

def test_angel_login_url_and_session(configured, monkeypatch):
    url = urlsplit(angel.login_url("STATE_y", ANGEL, cb("angel")))
    assert url.netloc == "smartapi.angelone.in" and parse_qs(url.query) == {"api_key": ["angelkey"], "state": ["STATE_y"]}
    token = _jwt(1_900_000_000)
    fake = _fake(monkeypatch, angel, {("GET", "/rest/secure/angelbroking/user/v1/getProfile"):
                                      (200, {"status": True, "message": "SUCCESS", "errorcode": "",
                                             "data": {"clientcode": "A123", "name": "Meera"}})})
    session = angel.create_session({"auth_token": token, "feed_token": "feed"}, ANGEL, cb("angel"))
    assert session.user_id == "A123" and session.access_token == token
    assert session.expires_at == datetime.fromtimestamp(1_900_000_000, timezone.utc)
    headers = fake.calls[0]["headers"]
    assert headers["Authorization"] == f"Bearer {token}" and headers["X-PrivateKey"] == "angelkey"


def test_angel_rejected_token_never_connects(configured, monkeypatch):
    # A token pasted into the callback that Angel One doesn't accept.
    _fake(monkeypatch, angel, {("GET", "/rest/secure/angelbroking/user/v1/getProfile"):
                               (200, {"status": False, "errorcode": "AG8001", "message": "Invalid Token"})})
    with pytest.raises(TokenExpired):
        angel.create_session({"auth_token": "forged.token.value"}, ANGEL, cb("angel"))


def test_angel_holdings_and_positions(configured, monkeypatch):
    _fake(monkeypatch, angel, {
        ("GET", "/rest/secure/angelbroking/portfolio/v1/getHolding"): (200, {"status": True, "data": [
            {"tradingsymbol": "TATASTEEL-EQ", "exchange": "NSE", "isin": "INE081A01020", "quantity": "10",
             "t1quantity": "2", "averageprice": "110.5", "ltp": "120", "close": "118"}]}),
        ("GET", "/rest/secure/angelbroking/order/v1/getPosition"): (200, {"status": True, "data": [
            {"tradingsymbol": "INFY-EQ", "exchange": "NSE", "producttype": "INTRADAY", "netqty": "-5",
             "avgnetprice": "1500.25", "ltp": "1490", "close": "1510"},
            {"tradingsymbol": "NIFTY26OCTFUT", "exchange": "NFO", "producttype": "CARRYFORWARD", "netqty": "50",
             "avgnetprice": "25000"}]})})
    rows, skipped = angel.holdings("tok", ANGEL)
    assert skipped == 0 and rows[0]["symbol"] == "TATASTEEL.NS" and rows[0]["quantity"] == 12
    positions, skipped = angel.positions("tok", ANGEL)
    assert skipped == 1 and positions[0]["symbol"] == "INFY.NS" and positions[0]["quantity"] == -5


def test_angel_logout_sends_client_code(configured, monkeypatch):
    fake = _fake(monkeypatch, angel, {("POST", "/rest/secure/angelbroking/user/v1/logout"):
                                      (200, {"status": True, "data": ""})})
    angel.logout("tok", "A123", ANGEL)
    assert fake.calls[0]["json"] == {"clientcode": "A123"}


# ======================================================================= Dhan --

def _dhan_auth(monkeypatch, client_id="1100001"):
    return _fake(monkeypatch, dhan, {
        ("POST", "/app/generate-consent"): (200, {"consentAppId": "cons-1", "consentAppStatus": "GENERATED",
                                                  "status": "success"}),
        ("GET", "/app/consumeApp-consent"): (200, {"dhanClientId": client_id, "dhanClientName": "Arjun",
                                                   "accessToken": "dh_tok", "expiryTime": "2026-10-19T15:30:00"})})


def test_dhan_consent_login_and_session(configured, monkeypatch):
    fake = _dhan_auth(monkeypatch)
    url = dhan.login_url("STATE_z", DHAN, cb("dhan"))
    assert url == "https://auth.dhan.co/login/consentApp-login?consentAppId=cons-1"
    assert fake.calls[0]["headers"] == {"app_id": "dhkey", "app_secret": "dhsecret"}
    assert fake.calls[0]["params"] == {"client_id": "1100001"}
    session = dhan.create_session({"tokenId": "tok-123"}, DHAN, cb("dhan"))
    assert fake.calls[1]["params"] == {"tokenId": "tok-123"}
    assert fake.calls[1]["headers"] == {"app_id": "dhkey", "app_secret": "dhsecret"}
    assert session.user_id == "1100001" and session.access_token == "dh_tok"
    # expiryTime is IST wall time.
    assert session.expires_at == datetime(2026, 10, 19, 10, 0, tzinfo=timezone.utc)


def test_dhan_holdings_and_positions(configured, monkeypatch):
    _fake(monkeypatch, dhan, {
        ("GET", "/v2/holdings"): (200, [
            {"exchange": "ALL", "tradingSymbol": "HDFC", "isin": "INE001A01036", "totalQty": 1000, "avgCostPrice": 2655},
            {"exchange": "BSE", "tradingSymbol": "ITC", "totalQty": 5, "avgCostPrice": 400}]),
        ("GET", "/v2/positions"): (200, [
            {"tradingSymbol": "TCS", "exchangeSegment": "NSE_EQ", "productType": "CNC", "netQty": 40,
             "buyAvg": 3345.8, "sellAvg": 0},
            {"tradingSymbol": "NIFTY-Oct2026-FUT", "exchangeSegment": "NSE_FNO", "productType": "MARGIN",
             "netQty": 50, "buyAvg": 25000}])})
    rows, skipped = dhan.holdings("tok", DHAN)
    by = {r["symbol"]: r for r in rows}
    assert skipped == 0 and set(by) == {"HDFC.NS", "ITC.BO"} and by["HDFC.NS"]["last_price"] is None
    positions, skipped = dhan.positions("tok", DHAN)
    assert skipped == 1 and positions[0]["symbol"] == "TCS.NS" and str(positions[0]["avg_price"]) == "3345.8"


def test_dhan_expired_token(configured, monkeypatch):
    _fake(monkeypatch, dhan, {("GET", "/v2/holdings"): (401, {"errorType": "Invalid_Authentication",
                                                               "errorCode": "DH-901", "errorMessage": "expired"})})
    with pytest.raises(TokenExpired):
        dhan.holdings("tok", DHAN)


# ===================================================================== routes --

def test_broker_routes_require_auth(client):
    assert client.get("/api/brokers/watchlist").status_code == 401
    for broker in ADAPTERS:
        assert client.post(f"/api/brokers/{broker}/connect", json={}).status_code == 401
        assert client.post(f"/api/brokers/{broker}/session", json={}).status_code == 401


def test_unknown_broker_is_rejected(client):
    assert client.post("/api/brokers/robinhood/connect", headers=auth()).status_code == 422


@pytest.mark.parametrize("broker,app", [("fyers", FY), ("angel", ANGEL), ("dhan", DHAN)])
def test_connect_without_cookie_key_is_503(client, configured, monkeypatch, broker, app):
    from flint_api.routers import brokers as broker_routes
    monkeypatch.setattr(broker_routes, "settings", dataclasses.replace(configured, broker_token_key=""))
    r = client.post(f"/api/brokers/{broker}/connect", headers=auth(),
                    json={"app": app, "keep": "7d", "redirect_uri": cb(broker)})
    assert r.status_code == 503


@pytest.mark.parametrize("params", [{}, {"auth_code": "has spaces"}, {"auth_code": "x" * 5000},
                                    {f"k{i}": "v" for i in range(9)}])
def test_session_params_are_validated(client, params):
    r = client.post("/api/brokers/fyers/session", headers=auth(), json={"state": "s" * 43, "params": params})
    assert r.status_code == 422


# ===================================================================== Upstox --

def test_upstox_login_url_and_session(configured, monkeypatch):
    url = urlsplit(upstox.login_url("STATE_u", UP, cb("upstox")))
    assert url.path == "/v2/login/authorization/dialog"
    assert parse_qs(url.query) == {"client_id": ["upkey"], "redirect_uri": [cb("upstox")],
                                   "response_type": ["code"], "state": ["STATE_u"]}
    fake = _fake(monkeypatch, upstox, {("POST", "/v2/login/authorization/token"): (200, {
        "user_id": "UP123", "user_name": "Kiran", "access_token": "up_tok", "extended_token": "ext"})})
    session = upstox.create_session({"code": "mk404x", "state": "x"}, UP, cb("upstox"))
    assert session.user_id == "UP123" and session.access_token == "up_tok"
    assert fake.calls[0]["data"] == {"code": "mk404x", "client_id": "upkey", "client_secret": "upsecret",
                                     "redirect_uri": cb("upstox"),
                                     "grant_type": "authorization_code"}


def test_upstox_holdings_positions_and_expiry(configured, monkeypatch):
    _fake(monkeypatch, upstox, {
        ("GET", "/v2/portfolio/long-term-holdings"): (200, {"status": "success", "data": [
            {"trading_symbol": "GOLDBEES", "exchange": "NSE", "isin": "INF204KB17I5", "quantity": 36,
             "t1_quantity": 0, "average_price": 45.5, "last_price": 47.2, "close_price": 47.0}]}),
        ("GET", "/v2/portfolio/short-term-positions"): (200, {"status": "success", "data": [
            {"trading_symbol": "SBIN", "exchange": "NSE", "product": "I", "quantity": 0, "average_price": None,
             "buy_price": 801.5, "last_price": 805, "close_price": 799},
            {"trading_symbol": "NIFTY26OCT25000CE", "exchange": "NFO", "product": "D", "quantity": 50,
             "average_price": 120}]})})
    rows, skipped = upstox.holdings("tok", UP)
    assert skipped == 0 and rows[0]["symbol"] == "GOLDBEES.NS" and rows[0]["quantity"] == 36
    positions, skipped = upstox.positions("tok", UP)
    assert skipped == 1 and positions[0]["symbol"] == "SBIN.NS" and str(positions[0]["avg_price"]) == "801.5"
    _fake(monkeypatch, upstox, {("GET", "/v2/portfolio/long-term-holdings"): (401, {"status": "error"})})
    with pytest.raises(TokenExpired):
        upstox.holdings("tok", UP)


# ====================================================================== Groww --

def test_groww_token_checked_then_holdings(configured, monkeypatch):
    fake = _fake(monkeypatch, groww, {
        ("GET", "/v1/user/detail"): (200, {"status": "SUCCESS", "payload": {"ucc": "GW123", "vendor_user_id": "v1"}}),
        ("GET", "/v1/holdings/user"): (200, {"status": "SUCCESS", "payload": {"holdings": [
            {"isin": "INE002A01018", "trading_symbol": "RELIANCE", "quantity": 10, "average_price": 2500}]}}),
        ("GET", "/v1/positions/user"): (200, {"status": "SUCCESS", "payload": {"positions": [
            {"trading_symbol": "TCS", "exchange": "NSE", "product": "MIS", "quantity": 2, "net_price": 4000}]}})})
    session = groww.create_session({"access_token": "gw.tok.en"})
    assert session.user_id == "GW123"
    headers = fake.calls[0]["headers"]
    assert headers["Authorization"] == "Bearer gw.tok.en" and headers["X-API-VERSION"] == "1.0"
    rows, _ = groww.holdings("gw.tok.en", {})
    assert rows[0]["symbol"] == "RELIANCE.NS" and rows[0]["last_price"] is None  # priced from a quote later
    positions, _ = groww.positions("gw.tok.en", {})
    assert positions[0]["symbol"] == "TCS.NS" and fake.calls[-1]["params"] == {"segment": "CASH"}


def test_groww_rejected_token(configured, monkeypatch):
    _fake(monkeypatch, groww, {("GET", "/v1/user/detail"): (401, {"status": "FAILURE"})})
    with pytest.raises(TokenExpired):
        groww.create_session({"access_token": "bad.token.here"})


# =================================================================== INDmoney --

def test_indmoney_token_checked_then_holdings(configured, monkeypatch):
    fake = _fake(monkeypatch, indmoney, {
        ("GET", "/user/profile"): (200, {"status": "success", "data": {"user_id": 4242, "first_name": "Asha",
                                                                         "last_name": "R"}}),
        ("GET", "/portfolio/holdings"): (200, {"status": "success", "data": [
            {"security_id": "18520", "symbol": "CUPID", "isin": "INE509F01029", "total_qty": 3, "avg_price": 217.3}]}),
        ("GET", "/portfolio/positions"): (200, {"status": "success", "data": [
            {"symbol": "INDIAGLYCO", "segment": "EQUITY", "product": "INTRADAY", "exchange": "NSE",
             "net_qty": 0, "avg_price": 1146.85}]})})
    session = indmoney.create_session({"access_token": "ind-token-123"})
    assert session.user_id == "4242" and session.user_name == "Asha R"
    assert fake.calls[0]["headers"]["Authorization"] == "ind-token-123"  # raw, no Bearer
    rows, _ = indmoney.holdings("ind-token-123", {})
    assert rows[0]["symbol"] == "CUPID.NS" and rows[0]["quantity"] == 3
    positions, _ = indmoney.positions("ind-token-123", {})
    assert positions[0]["symbol"] == "INDIAGLYCO.NS" and positions[0]["quantity"] == 0


# ==================================================================== Binance --

READ_ONLY = {"ipRestrict": False, "enableReading": True, "enableWithdrawals": False, "enableInternalTransfer": False,
             "enableMargin": False, "enableFutures": False, "permitsUniversalTransfer": False,
             "enableVanillaOptions": False, "enableSpotAndMarginTrading": False,
             "enablePortfolioMarginTrading": False, "enableFixApiTrade": False}
BN_KEY, BN_SECRET = "A" * 64, "B" * 64


def _binance(monkeypatch, rights=None):
    return _fake(monkeypatch, binance, {
        ("GET", "/sapi/v1/account/apiRestrictions"): (200, {**READ_ONLY, **(rights or {})}),
        ("GET", "/api/v3/account"): (200, {"canTrade": True, "uid": 354937868, "balances": [
            {"asset": "BTC", "free": "0.015", "locked": "0.005"}, {"asset": "USDT", "free": "120.5", "locked": "0"},
            {"asset": "ETH", "free": "0", "locked": "0"}]})})


def test_binance_read_only_key_connects_and_is_signed(configured, monkeypatch):
    fake = _binance(monkeypatch)
    session = binance.create_session({"api_key": BN_KEY, "api_secret": BN_SECRET})
    assert session.user_id == "354937868" and session.access_token == f"{BN_KEY}:{BN_SECRET}"
    call = fake.calls[0]
    assert call["headers"] == {"X-MBX-APIKEY": BN_KEY}
    query, _, signature = call["url"].split("?", 1)[1].rpartition("&signature=")
    assert signature == hmac.new(BN_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    assert BN_SECRET not in call["url"]  # the secret only ever signs


@pytest.mark.parametrize("flag", ["enableSpotAndMarginTrading", "enableWithdrawals", "enableFutures",
                                  "enableInternalTransfer"])
def test_binance_refuses_keys_that_can_do_more_than_read(configured, monkeypatch, flag):
    _binance(monkeypatch, {flag: True})
    with pytest.raises(AppError, match="trade, transfer or withdraw"):
        binance.create_session({"api_key": BN_KEY, "api_secret": BN_SECRET})


def test_binance_balances_in_usd_without_cost(configured, monkeypatch):
    _binance(monkeypatch)
    rows, skipped = binance.holdings(f"{BN_KEY}:{BN_SECRET}", {})
    by = {r["symbol"]: r for r in rows}
    assert set(by) == {"BTC-USD", "USDT-USD"} and skipped == 1
    assert str(by["BTC-USD"]["quantity"]) == "0.020" and by["BTC-USD"]["currency"] == "USD"
    assert by["BTC-USD"]["avg_price"] is None and binance.positions("x", {}) == ([], 0)


# ============================================================ key-paste routes --

class _Browser:
    """Plays the BFF: carries the pb_* cookies between calls."""

    def __init__(self, client):
        self.client, self.cookies = client, {}

    def request(self, method, path, **kw):
        headers = auth()
        if self.cookies:
            headers["x-broker-sessions"] = ";".join(f"{k}={v}" for k, v in self.cookies.items())
        r = self.client.request(method, path, headers=headers, **kw)
        for c in json.loads(r.headers.get("x-broker-cookies", "[]")):
            if c["value"] is None:
                self.cookies.pop(c["name"], None)
            else:
                self.cookies[c["name"]] = c["value"]
        return r


def test_link_keeps_binance_key_only_in_a_sealed_cookie(client, configured, monkeypatch):
    _binance(monkeypatch)
    b = _Browser(client)
    r = b.request("POST", "/api/brokers/binance/link", json={"credentials": {"api_key": BN_KEY, "api_secret": BN_SECRET}})
    assert r.status_code == 201, r.text
    assert r.json()["connection"]["currency"] == "USD" and r.json()["connection"]["auth"] == "keys"
    (value,) = b.cookies.values()
    assert BN_SECRET not in value and BN_KEY not in value
    listed = b.request("GET", "/api/brokers").json()
    assert [c["broker"] for c in listed["connections"]] == ["binance"]
    available = {a["broker"]: a for a in listed["available"]}
    assert available["binance"]["auth"] == "keys" and available["binance"]["fields"][1]["secret"] is True
    assert available["zerodha"]["auth"] == "oauth" and available["binance"]["callback_path"] is None
    assert [f["name"] for f in available["zerodha"]["fields"]] == ["api_key", "api_secret"]
    assert available["zerodha"]["callback_path"] == "/settings/brokers/callback"
    assert available["dhan"]["callback_path"] == "/settings/brokers/callback/dhan"


def test_link_refuses_trading_keys_and_wrong_fields(client, configured, monkeypatch):
    _binance(monkeypatch, {"enableSpotAndMarginTrading": True})
    b = _Browser(client)
    r = b.request("POST", "/api/brokers/binance/link", json={"credentials": {"api_key": BN_KEY, "api_secret": BN_SECRET}})
    assert r.status_code == 400 and b.cookies == {}
    r = b.request("POST", "/api/brokers/binance/link", json={"credentials": {"api_key": BN_KEY}})
    assert r.status_code == 400


def test_auth_kinds_are_not_interchangeable(client, configured):
    assert client.post("/api/brokers/groww/connect", headers=auth(),
                       json={"app": {"api_key": "abcdefgh"}, "redirect_uri": cb("groww")}).status_code == 400
    assert client.post("/api/brokers/zerodha/link", headers=auth(),
                       json={"credentials": {"access_token": "abcdefgh"}}).status_code == 400


# ============================================== user's own app keys and expiry --

def _kite(monkeypatch, user_id="ZK1234"):
    return _fake(monkeypatch, zerodha, {
        ("POST", "/session/token"): (200, {"status": "success", "data": {
            "user_id": user_id, "user_name": "Test Trader", "access_token": "kite_access_tok"}}),
        ("DELETE", "/session/token"): (200, {"status": "success", "data": True})})


def _state(r) -> str:
    q = parse_qs(urlsplit(r.json()["redirect_url"]).query)
    return parse_qs(q["redirect_params"][0])["state"][0]


def _kite_connect(b, **body):
    """Settings form -> Kite login -> callback, as the BFF and browser would do it."""
    body = {"app": KITE, "keep": "30d", "redirect_uri": cb("zerodha"), **body}
    r = b.request("POST", "/api/brokers/zerodha/connect", json=body)
    if r.status_code != 200:
        return r
    return b.request("POST", "/api/brokers/zerodha/session",
                     json={"state": _state(r), "params": {"request_token": "rt99"}})


def _changes(r) -> dict:
    return {c["name"]: c for c in json.loads(r.headers.get("x-broker-cookies", "[]"))}


def test_users_own_kite_app_signs_in_and_never_comes_back_out(client, configured, monkeypatch):
    fake = _kite(monkeypatch)
    b = _Browser(client)
    r = b.request("POST", "/api/brokers/zerodha/connect", json={"app": KITE, "keep": "30d",
                                                                 "redirect_uri": cb("zerodha")})
    assert r.status_code == 200, r.text
    assert parse_qs(urlsplit(r.json()["redirect_url"]).query)["api_key"] == ["kitekey"]
    assert "kitesecret" not in r.text and "kitesecret" not in b.cookies["pb_state"]
    r = b.request("POST", "/api/brokers/zerodha/session", json={"state": _state(r), "params": {"request_token": "rt99"}})
    assert r.status_code == 201, r.text
    # The exchange is signed with the user's own secret, read from the sealed state cookie.
    post = next(c for c in fake.calls if c["method"] == "POST")
    assert post["data"]["checksum"] == hashlib.sha256(b"kitekeyrt99kitesecret").hexdigest()
    # No response carries the token or the app keys, and the cookies don't show them.
    listed = b.request("GET", "/api/brokers")
    for text in (r.text, listed.text, *b.cookies.values()):
        assert "kitesecret" not in text and "kite_access_tok" not in text and "kitekey" not in text


def test_live_calls_use_the_users_own_key(client, configured, monkeypatch):
    _kite(monkeypatch)
    b = _Browser(client)
    _kite_connect(b)
    fake = _fake(monkeypatch, zerodha, {
        ("GET", "/portfolio/holdings"): (200, {"status": "success", "data": []}),
        ("GET", "/portfolio/positions"): (200, {"status": "success", "data": {"net": []}})})
    assert b.request("GET", "/api/brokers/watchlist").status_code == 200
    assert {c["headers"]["Authorization"] for c in fake.calls} == {"token kitekey:kite_access_tok"}


@pytest.mark.parametrize("keep,days", [("7d", 7), ("30d", 30), ("90d", 90)])
def test_keep_days_sets_the_cookie_and_the_sealed_deadline(client, configured, monkeypatch, keep, days):
    _kite(monkeypatch)
    r = _kite_connect(_Browser(client), keep=keep)
    conn = r.json()["connection"]
    cookie = _changes(r)["pb_" + conn["id"].replace("-", "")]
    assert abs(cookie["max_age"] - days * 86400) < 60
    kept = datetime.fromisoformat(conn["kept_until"])
    assert abs((kept - datetime.now(timezone.utc)).total_seconds() - days * 86400) < 60
    assert conn["session_only"] is False


def test_keep_session_is_a_browser_session_cookie(client, configured, monkeypatch):
    _kite(monkeypatch)
    r = _kite_connect(_Browser(client), keep="session")
    conn = r.json()["connection"]
    assert conn["session_only"] is True
    assert _changes(r)["pb_" + conn["id"].replace("-", "")]["max_age"] is None  # no Max-Age: gone on close


def test_keep_token_ends_with_the_broker_session(client, configured, monkeypatch):
    _kite(monkeypatch)
    conn = _kite_connect(_Browser(client), keep="token").json()["connection"]
    assert conn["kept_until"] == conn["token_expires_at"]


@pytest.mark.parametrize("until", [None, "2020-01-01T00:00:00+00:00", "2099-01-01T00:00:00+00:00",
                                   "2026-11-01T00:00:00"])
def test_custom_keep_must_be_a_zoned_date_within_90_days(client, configured, monkeypatch, until):
    _kite(monkeypatch)
    b = _Browser(client)
    r = _kite_connect(b, keep="custom", keep_until=until)
    assert r.status_code == 400 and b.cookies == {}


def test_custom_keep_within_range(client, configured, monkeypatch):
    _kite(monkeypatch)
    until = (datetime.now(timezone.utc) + timedelta(days=12)).replace(microsecond=0)
    conn = _kite_connect(_Browser(client), keep="custom", keep_until=until.isoformat()).json()["connection"]
    assert datetime.fromisoformat(conn["kept_until"]) == until


@pytest.mark.parametrize("uri", ["https://evil.example/settings/brokers/callback",
                                 "https://polaris.example/settings/brokers/callback/upstox",
                                 "https://polaris.example/settings/brokers/callback?x=1",
                                 "https://polaris.example.evil.example/settings/brokers/callback"])
def test_logins_only_return_to_polaris_callback(client, configured, monkeypatch, uri):
    _kite(monkeypatch)
    b = _Browser(client)
    r = _kite_connect(b, redirect_uri=uri)
    assert r.status_code == 400 and b.cookies == {}


def test_app_keys_must_be_complete_and_well_formed(client, configured):
    b = _Browser(client)
    r = b.request("POST", "/api/brokers/zerodha/connect",
                  json={"app": {"api_key": "kitekey"}, "keep": "7d", "redirect_uri": cb("zerodha")})
    assert r.status_code == 400 and "API secret" in r.json()["error"]
    # A malformed secret is refused without being echoed back.
    r = b.request("POST", "/api/brokers/zerodha/connect",
                  json={"app": {"api_key": "kitekey", "api_secret": "bad secret <script>"}, "keep": "7d",
                        "redirect_uri": cb("zerodha")})
    assert r.status_code == 422 and "bad secret" not in r.text and b.cookies == {}


def test_reconnect_reuses_saved_keys_and_keeps_the_deadline(client, configured, monkeypatch):
    fake = _kite(monkeypatch)
    b = _Browser(client)
    first = _kite_connect(b, keep="30d").json()["connection"]
    r = b.request("POST", "/api/brokers/zerodha/connect",
                  json={"connection_id": first["id"], "redirect_uri": cb("zerodha")})
    assert r.status_code == 200 and parse_qs(urlsplit(r.json()["redirect_url"]).query)["api_key"] == ["kitekey"]
    again = b.request("POST", "/api/brokers/zerodha/session",
                      json={"state": _state(r), "params": {"request_token": "rt99"}}).json()["connection"]
    assert again["id"] == first["id"] and again["kept_until"] == first["kept_until"]
    assert sum(c["method"] == "POST" for c in fake.calls) == 2
    # A connection id from nowhere has no keys to reuse.
    r = b.request("POST", "/api/brokers/zerodha/connect",
                  json={"connection_id": "00000000-0000-4000-8000-000000000000", "redirect_uri": cb("zerodha")})
    assert r.status_code == 404


def test_another_users_cookie_neither_opens_nor_lingers(client, configured, monkeypatch):
    _kite(monkeypatch)
    a = _Browser(client)
    _kite_connect(a)
    r = client.get("/api/brokers", headers={
        **auth("user_someone_else"), "x-broker-sessions": ";".join(f"{k}={v}" for k, v in a.cookies.items())})
    assert r.json()["connections"] == []
    assert _changes(r) and all(c["value"] is None for c in _changes(r).values())


def test_past_keep_until_is_deleted_on_the_next_request(configured, monkeypatch):
    import flint_api.broker_jar as bj
    now = datetime.now(timezone.utc)
    jar = bj.Jar("user_a", "")
    c = bj.Connection(id="11111111-1111-4111-8111-111111111111", broker="zerodha", broker_user_id="ZK1",
                      broker_user_name=None, access_token="t", expires_at=now + timedelta(hours=1),
                      connected_at=now, keep_until=now + timedelta(days=1), app=KITE)
    jar.save(c)
    raw = ";".join(f"{x['name']}={x['value']}" for x in json.loads(jar.headers()["x-broker-cookies"]))
    assert [x.id for x in bj.Jar("user_a", raw).connections()] == [c.id]

    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return now + timedelta(days=2)

    monkeypatch.setattr(bj, "datetime", Later)
    later = bj.Jar("user_a", raw)
    assert later.connections() == []
    assert json.loads(later.headers()["x-broker-cookies"]) == [{"name": c.cookie, "value": None, "max_age": 0}]


def test_dhan_refuses_a_login_for_another_client_id(configured, monkeypatch):
    _dhan_auth(monkeypatch, client_id="9999999")
    dhan.login_url("STATE_z", DHAN, cb("dhan"))
    with pytest.raises(AppError, match="different client ID"):
        dhan.create_session({"tokenId": "tok-123"}, DHAN, cb("dhan"))


def test_a_login_too_large_for_one_cookie_is_refused_not_dropped(client, configured, monkeypatch):
    _fake(monkeypatch, angel, {("GET", "/rest/secure/angelbroking/user/v1/getProfile"):
                               (200, {"status": True, "data": {"clientcode": "A123", "name": "Meera"}})})
    b = _Browser(client)
    r = b.request("POST", "/api/brokers/angel/connect", json={"app": ANGEL, "keep": "7d", "redirect_uri": cb("angel")})
    state = parse_qs(urlsplit(r.json()["redirect_url"]).query)["state"][0]
    r = b.request("POST", "/api/brokers/angel/session",
                  json={"state": state, "params": {"auth_token": "a" * 4000}})
    assert r.status_code == 400 and "too large" in r.json()["error"]
    assert b.cookies == {}  # the used state is cleared and nothing half-saved
