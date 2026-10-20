"""Angel One SmartAPI, read-only: publisher login, profile, holdings, positions,
logout.

The user signs in on Angel One's publisher-login page for their own SmartAPI
app, which redirects back with an auth_token (a JWT). Polaris never asks for
the client's PIN or TOTP. The token arrives through the browser, so it is
only trusted once Angel One accepts it for getProfile, which also tells us
whose account it is.
"""
from datetime import time
from urllib.parse import urlencode

import requests

from ..errors import AppError, Unavailable
from .base import (Session, TokenExpired, app_key, dec, holding, jwt_expiry, merge, next_ist, param, position,
                   yahoo_symbol)

NAME, LABEL = "angel", "Angel One"
CURRENCY, AUTH = "INR", "oauth"
FIELDS = [{"name": "api_key", "label": "API key", "secret": False}]
HELP_URL = "https://smartapi.angelone.in/"
HELP = ("On smartapi.angelone.in, create a Publisher app with the Redirect URL below, "
        "then paste its API key.")
LOGIN_URL = "https://smartapi.angelone.in/publisher-login"
API_URL = "https://apiconnect.angelone.in"
_TIMEOUT = 15
# Invalid token, expired token, missing token, session expired.
_TOKEN_ERRORS = {"AG8001", "AG8002", "AG8003", "AB1010"}


def login_url(state: str, app: dict, redirect_uri: str) -> str:
    """Angel One sends the user to the Redirect URL saved on the app."""
    return f"{LOGIN_URL}?" + urlencode({"api_key": app_key(app, "api_key", LABEL), "state": state})


def _headers(access_token: str, app: dict) -> dict:
    # SmartAPI requires the client IP and MAC headers on every call; the
    # server's own values say nothing about the user, so fixed ones are sent.
    return {"Content-Type": "application/json", "Accept": "application/json",
            "X-UserType": "USER", "X-SourceID": "WEB",
            "X-ClientLocalIP": "127.0.0.1", "X-ClientPublicIP": "127.0.0.1", "X-MACAddress": "00:00:00:00:00:00",
            "X-PrivateKey": app_key(app, "api_key", LABEL), "Authorization": f"Bearer {access_token}"}


def _call(method: str, path: str, access_token: str, app: dict, **kw):
    try:
        resp = requests.request(method, API_URL + path, headers=_headers(access_token, app), timeout=_TIMEOUT, **kw)
    except requests.RequestException:
        raise Unavailable("Angel One didn't respond. Try again in a moment.")
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if resp.status_code in (401, 403) or body.get("errorcode") in _TOKEN_ERRORS:
        raise TokenExpired("Your Angel One session has ended. Reconnect to sync again.")
    if resp.status_code >= 400 or body.get("status") is not True:
        if resp.status_code in (400, 422):
            raise AppError("Angel One rejected the request. Check the app's API key, then connect again.")
        raise Unavailable("Angel One couldn't complete the request. Try again in a moment.")
    return body.get("data")


def create_session(params: dict, app: dict, redirect_uri: str) -> Session:
    token = param(params, "auth_token", LABEL).removeprefix("Bearer ")
    profile = _call("GET", "/rest/secure/angelbroking/user/v1/getProfile", token, app) or {}
    if not profile.get("clientcode"):
        raise Unavailable("Angel One couldn't complete the request. Try again in a moment.")
    return Session(access_token=token, user_id=str(profile["clientcode"])[:64],
                   user_name=(profile.get("name") or "")[:120] or None,
                   expires_at=jwt_expiry(token, next_ist(time(0, 0))))


def logout(access_token: str, broker_user_id: str, app: dict) -> None:
    try:
        _call("POST", "/rest/secure/angelbroking/user/v1/logout", access_token, app,
              json={"clientcode": broker_user_id})
    except AppError:
        pass


def holdings(access_token: str, app: dict) -> tuple[list[dict], int]:
    rows, skipped = [], 0
    for h in _call("GET", "/rest/secure/angelbroking/portfolio/v1/getHolding", access_token, app) or []:
        exchange = h.get("exchange")
        symbol = yahoo_symbol(h.get("tradingsymbol") or "", exchange)
        qty = (dec(h.get("quantity")) or 0) + (dec(h.get("t1quantity")) or 0)
        avg = dec(h.get("averageprice"))
        if symbol is None or avg is None or avg < 0 or qty <= 0:
            skipped += 1
            continue
        rows.append(holding(symbol, h["tradingsymbol"], exchange, h.get("isin"), qty, avg,
                            dec(h.get("ltp")), dec(h.get("close"))))
    return merge(rows, "symbol"), skipped


def positions(access_token: str, app: dict) -> tuple[list[dict], int]:
    rows, skipped = [], 0
    for p in _call("GET", "/rest/secure/angelbroking/order/v1/getPosition", access_token, app) or []:
        exchange = p.get("exchange")
        symbol = yahoo_symbol(p.get("tradingsymbol") or "", exchange)
        qty, avg = dec(p.get("netqty")), dec(p.get("avgnetprice") or p.get("netprice"))
        if symbol is None or qty is None or avg is None or avg < 0:
            skipped += 1
            continue
        rows.append(position(symbol, p["tradingsymbol"], exchange, p.get("producttype"), qty, avg,
                             dec(p.get("ltp")), dec(p.get("close"))))
    return merge(rows, "symbol", "product"), skipped
