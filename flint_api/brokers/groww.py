"""Groww Trade API, read-only: holdings and cash positions.

Groww has no login flow for other apps: the user generates an access token
in Groww (Trading APIs settings) and pastes it here. It is checked against
Groww's user detail endpoint before it is kept, and expires at 06:00 IST.
"""
from datetime import time

import requests

from ..errors import Unavailable
from .base import Session, TokenExpired, dec, holding, jwt_expiry, merge, next_ist, param, position, yahoo_symbol

NAME, LABEL = "groww", "Groww"
CURRENCY, AUTH = "INR", "keys"
FIELDS = [{"name": "access_token", "label": "Access token", "secret": True}]
HELP_URL = "https://groww.in/trade-api"
HELP = "In Groww, open Settings → Trading APIs and generate an access token. It expires every day at 6:00 IST."
API_URL = "https://api.groww.in/v1"
_TIMEOUT = 15
_SESSION_ENDS = time(6, 0)



def _call(path: str, access_token: str, **kw):
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}", "X-API-VERSION": "1.0"}
    try:
        resp = requests.request("GET", API_URL + path, headers=headers, timeout=_TIMEOUT, **kw)
    except requests.RequestException:
        raise Unavailable("Groww didn't respond. Try again in a moment.")
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if resp.status_code in (401, 403):
        raise TokenExpired("Groww didn't accept that access token. Generate a new one and connect again.")
    if resp.status_code >= 400 or body.get("status") != "SUCCESS":
        raise Unavailable("Groww couldn't complete the request. Try again in a moment.")
    return body.get("payload") or {}


def create_session(params: dict) -> Session:
    token = param(params, "access_token", LABEL)
    user = _call("/user/detail", token)
    ucc = user.get("ucc") or user.get("vendor_user_id")
    if not ucc:
        raise Unavailable("Groww couldn't complete the request. Try again in a moment.")
    return Session(access_token=token, user_id=str(ucc)[:64], user_name=None,
                   expires_at=jwt_expiry(token, next_ist(_SESSION_ENDS)))


def logout(access_token: str, broker_user_id: str, app: dict) -> None:
    """The token was made by the user in Groww; it expires by itself."""


def holdings(access_token: str, app: dict) -> tuple[list[dict], int]:
    """Groww holdings carry no exchange or price: quoted on NSE, priced from a quote."""
    rows, skipped = [], 0
    for h in _call("/holdings/user", access_token).get("holdings") or []:
        symbol = yahoo_symbol(h.get("trading_symbol") or "", "NSE")
        qty, avg = dec(h.get("quantity")), dec(h.get("average_price"))
        if symbol is None or qty is None or qty <= 0 or avg is None or avg < 0:
            skipped += 1
            continue
        rows.append(holding(symbol, h["trading_symbol"], "NSE", h.get("isin"), qty, avg, None, None))
    return merge(rows, "symbol"), skipped


def positions(access_token: str, app: dict) -> tuple[list[dict], int]:
    rows, skipped = [], 0
    payload = _call("/positions/user", access_token, params={"segment": "CASH"})
    for p in payload.get("positions") or []:
        exchange = p.get("exchange") if p.get("exchange") in ("NSE", "BSE") else "NSE"
        symbol = yahoo_symbol(p.get("trading_symbol") or "", exchange)
        qty, avg = dec(p.get("quantity")), dec(p.get("net_price"))
        if symbol is None or qty is None or avg is None or avg < 0:
            skipped += 1
            continue
        rows.append(position(symbol, p["trading_symbol"], exchange, p.get("product"), qty, avg, None, None))
    return merge(rows, "symbol", "product"), skipped
