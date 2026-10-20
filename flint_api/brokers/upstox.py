"""Upstox API v2, read-only: login, session exchange, holdings, positions, logout.

OAuth authorization code: the user signs in on Upstox, which returns a code
(and our state) to the Redirect URL saved on the user's own Upstox app. The
code is exchanged server side with that app's secret. Tokens last until 03:30
IST the next morning.
"""
from datetime import time
from urllib.parse import urlencode

import requests

from ..errors import AppError, Unavailable
from .base import (Session, TokenExpired, app_key, dec, holding, merge, next_ist, param, position,
                   yahoo_symbol)

NAME, LABEL = "upstox", "Upstox"
CURRENCY, AUTH = "INR", "oauth"
FIELDS = [{"name": "api_key", "label": "API key", "secret": False},
          {"name": "api_secret", "label": "API secret", "secret": True}]
HELP_URL = "https://account.upstox.com/developer/apps"
HELP = ("In Upstox, open Developer Apps, create an app with the Redirect URL below, "
        "then paste its API key and API secret.")
API_URL = "https://api.upstox.com/v2"
_TIMEOUT = 15
# "The access token lasts until 3:30 AM the following day."
_SESSION_ENDS = time(3, 30)


def login_url(state: str, app: dict, redirect_uri: str) -> str:
    return f"{API_URL}/login/authorization/dialog?" + urlencode({
        "client_id": app_key(app, "api_key", LABEL), "redirect_uri": redirect_uri,
        "response_type": "code", "state": state})


def _call(method: str, path: str, *, access_token: str | None = None, **kw):
    headers = {"Accept": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    try:
        resp = requests.request(method, API_URL + path, headers=headers, timeout=_TIMEOUT, **kw)
    except requests.RequestException:
        raise Unavailable("Upstox didn't respond. Try again in a moment.")
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if resp.status_code == 401:
        raise TokenExpired("Your Upstox session has ended. Reconnect to see it again.")
    if resp.status_code >= 400:
        if resp.status_code in (400, 422):
            raise AppError("Upstox rejected the request. Check the app's API key, secret and Redirect URL, "
                           "then connect again.")
        raise Unavailable("Upstox couldn't complete the request. Try again in a moment.")
    return body


def create_session(params: dict, app: dict, redirect_uri: str) -> Session:
    code = param(params, "code", LABEL)
    data = _call("POST", "/login/authorization/token", data={
        "code": code, "client_id": app_key(app, "api_key", LABEL),
        "client_secret": app_key(app, "api_secret", LABEL),
        "redirect_uri": redirect_uri, "grant_type": "authorization_code"})
    if not data.get("access_token") or not data.get("user_id"):
        raise Unavailable("Upstox couldn't complete the request. Try again in a moment.")
    return Session(access_token=data["access_token"], user_id=str(data["user_id"])[:64],
                   user_name=(data.get("user_name") or "")[:120] or None, expires_at=next_ist(_SESSION_ENDS))


def logout(access_token: str, broker_user_id: str, app: dict) -> None:
    try:
        _call("DELETE", "/logout", access_token=access_token)
    except AppError:
        pass


def _symbol(row: dict) -> str:
    return row.get("trading_symbol") or row.get("tradingsymbol") or ""


def holdings(access_token: str, app: dict) -> tuple[list[dict], int]:
    rows, skipped = [], 0
    for h in _call("GET", "/portfolio/long-term-holdings", access_token=access_token).get("data") or []:
        exchange, tradingsymbol = h.get("exchange"), _symbol(h)
        symbol = yahoo_symbol(tradingsymbol, exchange)
        qty = (dec(h.get("quantity")) or 0) + (dec(h.get("t1_quantity")) or 0)
        avg = dec(h.get("average_price"))
        if symbol is None or avg is None or avg < 0 or qty <= 0:
            skipped += 1
            continue
        rows.append(holding(symbol, tradingsymbol, exchange, h.get("isin"), qty, avg,
                            dec(h.get("last_price")), dec(h.get("close_price"))))
    return merge(rows, "symbol"), skipped


def positions(access_token: str, app: dict) -> tuple[list[dict], int]:
    rows, skipped = [], 0
    for p in _call("GET", "/portfolio/short-term-positions", access_token=access_token).get("data") or []:
        exchange, tradingsymbol = p.get("exchange"), _symbol(p)
        symbol = yahoo_symbol(tradingsymbol, exchange)
        qty = dec(p.get("quantity"))
        avg = dec(p.get("average_price")) if p.get("average_price") is not None else dec(p.get("buy_price"))
        if symbol is None or qty is None or avg is None or avg < 0:
            skipped += 1
            continue
        rows.append(position(symbol, tradingsymbol, exchange, p.get("product"), qty, avg,
                             dec(p.get("last_price")), dec(p.get("close_price"))))
    return merge(rows, "symbol", "product"), skipped
