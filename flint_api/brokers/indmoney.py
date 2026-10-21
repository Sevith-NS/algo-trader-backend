"""INDmoney (INDstocks API), read-only: holdings and equity positions.

INDstocks has no login flow for other apps: the user generates an access
token on indstocks.com and pastes it here. It is checked against the profile
endpoint before it is kept, and lasts 24 hours.
"""
from datetime import datetime, timedelta, timezone

import requests

from ..errors import Unavailable
from .base import Session, TokenExpired, dec, holding, jwt_expiry, merge, param, position, yahoo_symbol

NAME, LABEL = "indmoney", "INDmoney"
CURRENCY, AUTH = "INR", "keys"
FIELDS = [{"name": "access_token", "label": "Access token", "secret": True}]
HELP_URL = "https://indstocks.com/app/api-trading/access-tokens"
HELP = "On indstocks.com, open API Trading → Access Tokens and generate a token. It lasts 24 hours."
API_URL = "https://api.indstocks.com"
_TIMEOUT = 15



def _call(path: str, access_token: str, **kw):
    # INDstocks takes the raw token, without "Bearer".
    headers = {"Content-Type": "application/json", "Authorization": access_token}
    try:
        resp = requests.request("GET", API_URL + path, headers=headers, timeout=_TIMEOUT, **kw)
    except requests.RequestException:
        raise Unavailable("INDmoney didn't respond. Try again in a moment.")
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if resp.status_code in (401, 403):
        raise TokenExpired("INDmoney didn't accept that access token. Generate a new one and connect again.")
    if resp.status_code >= 400 or body.get("status") != "success":
        raise Unavailable("INDmoney couldn't complete the request. Try again in a moment.")
    return body.get("data")


def create_session(params: dict) -> Session:
    token = param(params, "access_token", LABEL)
    profile = _call("/user/profile", token) or {}
    if not profile.get("user_id"):
        raise Unavailable("INDmoney couldn't complete the request. Try again in a moment.")
    name = " ".join(p for p in (profile.get("first_name"), profile.get("last_name")) if p)
    return Session(access_token=token, user_id=str(profile["user_id"])[:64], user_name=name[:120] or None,
                   expires_at=jwt_expiry(token, datetime.now(timezone.utc) + timedelta(hours=24)))


def logout(access_token: str, broker_user_id: str, app: dict) -> None:
    """The token was made by the user on INDstocks; it expires by itself."""


def holdings(access_token: str, app: dict) -> tuple[list[dict], int]:
    """INDstocks holdings carry no exchange or price: quoted on NSE, priced from a quote."""
    rows, skipped = [], 0
    for h in _call("/portfolio/holdings", access_token) or []:
        symbol = yahoo_symbol(h.get("symbol") or "", "NSE")
        qty, avg = dec(h.get("total_qty")), dec(h.get("avg_price"))
        if symbol is None or qty is None or qty <= 0 or avg is None or avg < 0:
            skipped += 1
            continue
        rows.append(holding(symbol, h["symbol"], "NSE", h.get("isin"), qty, avg, None, None))
    return merge(rows, "symbol"), skipped


def positions(access_token: str, app: dict) -> tuple[list[dict], int]:
    rows, skipped = [], 0
    for p in _call("/portfolio/positions", access_token, params={"segment": "equity"}) or []:
        exchange = p.get("exchange") if p.get("exchange") in ("NSE", "BSE") else "NSE"
        symbol = yahoo_symbol(p.get("symbol") or "", exchange)
        qty, avg = dec(p.get("net_qty")), dec(p.get("avg_price"))
        if p.get("segment", "EQUITY").upper() != "EQUITY" or symbol is None or qty is None or avg is None or avg < 0:
            skipped += 1
            continue
        rows.append(position(symbol, p["symbol"], exchange, p.get("product"), qty, avg, None, None))
    return merge(rows, "symbol", "product"), skipped
