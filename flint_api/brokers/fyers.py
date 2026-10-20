"""FYERS API v3, read-only: login, session exchange, holdings, positions, logout.

Login is an OAuth-style redirect: the user signs in on FYERS, which returns an
auth_code (and our state) to the Redirect URL saved on the user's own FYERS
app. The code is exchanged server side with SHA-256(app_id:app_secret), so
the secret is never sent anywhere.
"""
import hashlib
from datetime import time
from urllib.parse import urlencode

import requests

from ..errors import AppError, Unavailable
from .base import (Session, TokenExpired, app_key, dec, holding, merge, next_ist, param, position,
                   yahoo_symbol)

NAME, LABEL = "fyers", "FYERS"
CURRENCY, AUTH = "INR", "oauth"
FIELDS = [{"name": "app_id", "label": "App ID", "secret": False},
          {"name": "app_secret", "label": "Secret ID", "secret": True}]
HELP_URL = "https://myapi.fyers.in/dashboard"
HELP = ("On myapi.fyers.in, create an app with the Redirect URL below, "
        "then paste its App ID and Secret ID.")
API_URL = "https://api-t1.fyers.in/api/v3"
_TIMEOUT = 15
# "FYERS API access tokens expire daily at 6:30 AM" (IST).
_SESSION_ENDS = time(6, 30)
# Error codes FYERS returns for an expired or invalid token.
_TOKEN_CODES = {-8, -15, -16, -17}


def login_url(state: str, app: dict, redirect_uri: str) -> str:
    return f"{API_URL}/generate-authcode?" + urlencode({
        "client_id": app_key(app, "app_id", LABEL), "redirect_uri": redirect_uri,
        "response_type": "code", "state": state})


def app_id_hash(app: dict) -> str:
    raw = f"{app_key(app, 'app_id', LABEL)}:{app_key(app, 'app_secret', LABEL)}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _call(method: str, path: str, app: dict, *, access_token: str | None = None, **kw) -> dict:
    headers = {"Authorization": f"{app_key(app, 'app_id', LABEL)}:{access_token}"} if access_token else {}
    try:
        resp = requests.request(method, API_URL + path, headers=headers, timeout=_TIMEOUT, **kw)
    except requests.RequestException:
        raise Unavailable("FYERS didn't respond. Try again in a moment.")
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if resp.status_code in (401, 403) or body.get("code") in _TOKEN_CODES:
        raise TokenExpired("Your FYERS session has ended. Reconnect to sync again.")
    if resp.status_code >= 400 or body.get("s") != "ok":
        if resp.status_code in (400, 422):
            raise AppError("FYERS rejected the request. Check the app's App ID and Secret ID, then connect again.")
        raise Unavailable("FYERS couldn't complete the request. Try again in a moment.")
    return body


def create_session(params: dict, app: dict, redirect_uri: str) -> Session:
    code = param(params, "auth_code", LABEL)
    token = _call("POST", "/validate-authcode", app, json={
        "grant_type": "authorization_code", "appIdHash": app_id_hash(app), "code": code}).get("access_token")
    if not token:
        raise Unavailable("FYERS couldn't complete the request. Try again in a moment.")
    profile = _call("GET", "/profile", app, access_token=token).get("data") or {}
    if not profile.get("fy_id"):
        raise Unavailable("FYERS couldn't complete the request. Try again in a moment.")
    return Session(access_token=token, user_id=str(profile["fy_id"])[:64],
                   user_name=(profile.get("name") or "")[:120] or None, expires_at=next_ist(_SESSION_ENDS))


def logout(access_token: str, broker_user_id: str, app: dict) -> None:
    """Best effort: the token expires by itself at 06:30 IST anyway."""
    try:
        _call("POST", "/logout", app, access_token=access_token)
    except AppError:
        pass


def _split(fyers_symbol: str) -> tuple[str, str]:
    """'NSE:SBIN-EQ' -> ('NSE', 'SBIN-EQ')."""
    exchange, _, tradingsymbol = (fyers_symbol or "").partition(":")
    return exchange, tradingsymbol


def holdings(access_token: str, app: dict) -> tuple[list[dict], int]:
    rows, skipped = [], 0
    for h in _call("GET", "/holdings", app, access_token=access_token).get("holdings") or []:
        exchange, tradingsymbol = _split(h.get("symbol"))
        symbol = yahoo_symbol(tradingsymbol, exchange)
        qty, avg = dec(h.get("quantity")), dec(h.get("costPrice"))
        if symbol is None or qty is None or qty <= 0 or avg is None or avg < 0:
            skipped += 1
            continue
        rows.append(holding(symbol, tradingsymbol, exchange, h.get("isin"), qty, avg, dec(h.get("ltp")), None))
    return merge(rows, "symbol"), skipped


def positions(access_token: str, app: dict) -> tuple[list[dict], int]:
    rows, skipped = [], 0
    for p in _call("GET", "/positions", app, access_token=access_token).get("netPositions") or []:
        exchange, tradingsymbol = _split(p.get("symbol"))
        symbol = yahoo_symbol(tradingsymbol, exchange)
        qty, avg = dec(p.get("netQty")), dec(p.get("netAvg"))
        if symbol is None or qty is None or avg is None or avg < 0:
            skipped += 1
            continue
        rows.append(position(symbol, tradingsymbol, exchange, p.get("productType"), qty, avg,
                             dec(p.get("ltp")), None))
    return merge(rows, "symbol", "product"), skipped
