"""Dhan (DhanHQ v2), read-only: API-key consent login, holdings, positions.

Each user brings their own DhanHQ API key and secret (generated on web.dhan.co
with Polaris's callback as its Redirect URL; valid 12 months). The server
creates a consent with them for the user's client ID, the user signs in on
Dhan, and Dhan redirects back with a tokenId that the server consumes for the
user's access token. Dhan echoes no state, so the frontend keeps ours for the
round trip.
"""
from datetime import datetime, timedelta, timezone

import requests

from ..errors import AppError, Unavailable
from .base import (IST, Session, TokenExpired, app_key, dec, holding, jwt_expiry, merge, param, position,
                   yahoo_symbol)

NAME, LABEL = "dhan", "Dhan"
CURRENCY, AUTH = "INR", "oauth"
FIELDS = [{"name": "client_id", "label": "Dhan client ID", "secret": False},
          {"name": "api_key", "label": "API key", "secret": False},
          {"name": "api_secret", "label": "API secret", "secret": True}]
HELP_URL = "https://web.dhan.co/"
HELP = ("On web.dhan.co, open My Profile → Access DhanHQ APIs, generate an API key with the Redirect URL "
        "below, then paste your client ID, the API key and the API secret.")
AUTH_URL = "https://auth.dhan.co"
API_URL = "https://api.dhan.co/v2"
_TIMEOUT = 15
# Dhan's code for an invalid or expired access token.
_TOKEN_ERRORS = {"DH-901"}
# exchangeSegment values that are cash equities, with the exchange they map to.
_CASH_SEGMENTS = {"NSE_EQ": "NSE", "BSE_EQ": "BSE"}


def _request(method: str, url: str, headers: dict, **kw):
    try:
        resp = requests.request(method, url, headers=headers, timeout=_TIMEOUT, **kw)
    except requests.RequestException:
        raise Unavailable("Dhan didn't respond. Try again in a moment.")
    try:
        body = resp.json()
    except ValueError:
        body = None
    code = body.get("errorCode") if isinstance(body, dict) else None
    if resp.status_code == 401 or code in _TOKEN_ERRORS:
        raise TokenExpired("Your Dhan session has ended. Reconnect to sync again.")
    if resp.status_code >= 400 or body is None:
        if resp.status_code in (400, 403, 422):
            raise AppError("Dhan rejected the request. Check your client ID, API key and secret, "
                           "then connect again.")
        raise Unavailable("Dhan couldn't complete the request. Try again in a moment.")
    return body


def _app_headers(app: dict) -> dict:
    return {"app_id": app_key(app, "api_key", LABEL), "app_secret": app_key(app, "api_secret", LABEL)}


def login_url(state: str, app: dict, redirect_uri: str) -> str:
    """Creates a consent on Dhan first; Dhan sends the user to the Redirect URL
    saved with the API key. `state` travels through the frontend."""
    consent = _request("POST", f"{AUTH_URL}/app/generate-consent", _app_headers(app),
                       params={"client_id": app_key(app, "client_id", LABEL)})
    consent_id = consent.get("consentAppId") if isinstance(consent, dict) else None
    if not consent_id:
        raise Unavailable("Dhan couldn't complete the request. Try again in a moment.")
    return f"{AUTH_URL}/login/consentApp-login?consentAppId={consent_id}"


def _expiry(value, token: str) -> datetime:
    fallback = jwt_expiry(token, datetime.now(timezone.utc) + timedelta(hours=24))
    try:
        at = datetime.fromisoformat(str(value))
    except ValueError:
        return fallback
    return (at if at.tzinfo else at.replace(tzinfo=IST)).astimezone(timezone.utc)


def create_session(params: dict, app: dict, redirect_uri: str) -> Session:
    token_id = param(params, "tokenId", LABEL)
    data = _request("GET", f"{AUTH_URL}/app/consumeApp-consent", _app_headers(app), params={"tokenId": token_id})
    if not isinstance(data, dict) or not data.get("accessToken") or not data.get("dhanClientId"):
        raise Unavailable("Dhan couldn't complete the request. Try again in a moment.")
    if str(data["dhanClientId"]) != app_key(app, "client_id", LABEL):
        raise AppError("Dhan signed in a different client ID than the one you entered. Connect again.")
    return Session(access_token=data["accessToken"], user_id=str(data["dhanClientId"])[:64],
                   user_name=(data.get("dhanClientName") or "")[:120] or None,
                   expires_at=_expiry(data.get("expiryTime"), data["accessToken"]))


def logout(access_token: str, broker_user_id: str, app: dict) -> None:
    """Dhan documents no logout for consent tokens; the token expires by itself."""


def _api(path: str, access_token: str) -> list:
    body = _request("GET", API_URL + path, {"Content-Type": "application/json", "access-token": access_token})
    return body if isinstance(body, list) else []


def holdings(access_token: str, app: dict) -> tuple[list[dict], int]:
    """Dhan holdings carry no last price; the sync fills it from a quote."""
    rows, skipped = [], 0
    for h in _api("/holdings", access_token):
        # "ALL" means the shares are held across both exchanges; quote on NSE.
        exchange = h.get("exchange") if h.get("exchange") in ("NSE", "BSE") else "NSE"
        symbol = yahoo_symbol(h.get("tradingSymbol") or "", exchange)
        qty, avg = dec(h.get("totalQty")), dec(h.get("avgCostPrice"))
        if symbol is None or qty is None or qty <= 0 or avg is None or avg < 0:
            skipped += 1
            continue
        rows.append(holding(symbol, h["tradingSymbol"], exchange, h.get("isin"), qty, avg, None, None))
    return merge(rows, "symbol"), skipped


def positions(access_token: str, app: dict) -> tuple[list[dict], int]:
    rows, skipped = [], 0
    for p in _api("/positions", access_token):
        exchange = _CASH_SEGMENTS.get(p.get("exchangeSegment"))
        symbol = yahoo_symbol(p.get("tradingSymbol") or "", exchange) if exchange else None
        qty = dec(p.get("netQty"))
        avg = dec(p.get("sellAvg") if qty is not None and qty < 0 else p.get("buyAvg"))
        if symbol is None or qty is None or avg is None or avg < 0:
            skipped += 1
            continue
        rows.append(position(symbol, p["tradingSymbol"], exchange, p.get("productType"), qty, avg, None, None))
    return merge(rows, "symbol", "product"), skipped
