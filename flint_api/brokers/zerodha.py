"""Zerodha Kite Connect v3, read-only: login, session exchange, holdings,
positions, logout.

Each user brings their own Kite Connect app (API key and secret), registered
at developers.kite.trade with Polaris's callback as its Redirect URL. The key
and secret are kept only in the user's sealed cookie.

Polaris never places orders through a broker. The only calls made here read
the user's identity, holdings and positions, and end the session on disconnect.
"""
import hashlib
from datetime import datetime, time
from urllib.parse import urlencode

import requests

from ..errors import AppError, Unavailable
from .base import (Session, TokenExpired, app_key, dec, holding, merge, next_ist, param, position,
                   yahoo_symbol)

NAME, LABEL = "zerodha", "Zerodha"
CURRENCY, AUTH = "INR", "oauth"
FIELDS = [{"name": "api_key", "label": "API key", "secret": False},
          {"name": "api_secret", "label": "API secret", "secret": True}]
HELP_URL = "https://developers.kite.trade/apps"
HELP = ("On developers.kite.trade, create a Kite Connect app with the Redirect URL below, "
        "then paste the app's API key and API secret.")
LOGIN_URL = "https://kite.zerodha.com/connect/login"
API_URL = "https://api.kite.trade"
_TIMEOUT = 15
# Kite sessions end at 06:00 IST the next morning, whenever they started.
_SESSION_ENDS = time(6, 0)

__all__ = ["TokenExpired", "yahoo_symbol"]


def login_url(state: str, app: dict, redirect_uri: str) -> str:
    """Kite sends the user to the Redirect URL saved on the app; redirect_uri
    is only shown to the user."""
    # redirect_params come back verbatim on the redirect to our callback.
    return f"{LOGIN_URL}?" + urlencode({"v": 3, "api_key": app_key(app, "api_key", LABEL),
                                        "redirect_params": urlencode({"state": state})})


def checksum(app: dict, request_token: str) -> str:
    raw = app_key(app, "api_key", LABEL) + request_token + app_key(app, "api_secret", LABEL)
    return hashlib.sha256(raw.encode()).hexdigest()


def session_expiry(now: datetime | None = None) -> datetime:
    return next_ist(_SESSION_ENDS, now)


def _call(method: str, path: str, app: dict, *, access_token: str | None = None, **kw) -> dict:
    headers = {"X-Kite-Version": "3"}
    if access_token:
        headers["Authorization"] = f"token {app_key(app, 'api_key', LABEL)}:{access_token}"
    try:
        resp = requests.request(method, API_URL + path, headers=headers, timeout=_TIMEOUT, **kw)
    except requests.RequestException:
        raise Unavailable("Zerodha didn't respond. Try again in a moment.")
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if resp.status_code == 403 or body.get("error_type") == "TokenException":
        raise TokenExpired("Your Zerodha session has ended. Reconnect to sync again.")
    if resp.status_code >= 400 or body.get("status") != "success":
        if resp.status_code in (400, 422) or body.get("error_type") == "InputException":
            raise AppError("Zerodha rejected the request. Check the app's API key and secret, then connect again.")
        raise Unavailable("Zerodha couldn't complete the request. Try again in a moment.")
    return body.get("data") or {}


def create_session(params: dict, app: dict, redirect_uri: str) -> Session:
    """Exchange the one-time request token for an access token."""
    request_token = param(params, "request_token", LABEL)
    data = _call("POST", "/session/token", app, data={
        "api_key": app_key(app, "api_key", LABEL), "request_token": request_token,
        "checksum": checksum(app, request_token)})
    if not data.get("access_token") or not data.get("user_id"):
        raise Unavailable("Zerodha couldn't complete the request. Try again in a moment.")
    return Session(access_token=data["access_token"], user_id=str(data["user_id"])[:64],
                   user_name=(data.get("user_name") or "")[:120] or None, expires_at=session_expiry())


def logout(access_token: str, broker_user_id: str, app: dict) -> None:
    """Best effort: the session expires by itself at 06:00 IST anyway."""
    try:
        _call("DELETE", "/session/token", app, access_token=access_token,
              params={"api_key": app_key(app, "api_key", LABEL), "access_token": access_token})
    except AppError:
        pass


def holdings(access_token: str, app: dict) -> tuple[list[dict], int]:
    """Equity holdings as rows for the holdings table, plus how many were skipped
    (exchanges other than NSE/BSE, or symbols that don't map)."""
    rows, skipped = [], 0
    for h in _call("GET", "/portfolio/holdings", app, access_token=access_token) or []:
        exchange = h.get("exchange")
        symbol = yahoo_symbol(h.get("tradingsymbol") or "", exchange)
        qty = (dec(h.get("quantity")) or 0) + (dec(h.get("t1_quantity")) or 0)
        avg = dec(h.get("average_price"))
        if symbol is None or avg is None or avg < 0 or qty <= 0:
            skipped += 1
            continue
        rows.append(holding(symbol, h["tradingsymbol"], exchange, h.get("isin"), qty, avg,
                            dec(h.get("last_price")), dec(h.get("close_price"))))
    return merge(rows, "symbol"), skipped


def positions(access_token: str, app: dict) -> tuple[list[dict], int]:
    """Today's net positions in NSE/BSE cash symbols, including ones closed
    today (quantity 0). Derivatives don't map to a quote symbol and are skipped."""
    rows, skipped = [], 0
    data = _call("GET", "/portfolio/positions", app, access_token=access_token) or {}
    for p in data.get("net") or []:
        exchange = p.get("exchange")
        symbol = yahoo_symbol(p.get("tradingsymbol") or "", exchange)
        qty, avg = dec(p.get("quantity")), dec(p.get("average_price"))
        if symbol is None or qty is None or avg is None or avg < 0:
            skipped += 1
            continue
        rows.append(position(symbol, p["tradingsymbol"], exchange, p.get("product"), qty, avg,
                             dec(p.get("last_price")), dec(p.get("close_price"))))
    return merge(rows, "symbol", "product"), skipped
