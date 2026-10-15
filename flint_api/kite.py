"""Zerodha Kite Connect v3, read-only: login, session exchange, holdings, logout.

Polaris never places orders through a broker. The only calls made here read
the user's identity and holdings, and end the session on disconnect.
"""
import hashlib
import re
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from urllib.parse import urlencode

import requests

from .errors import AppError, Unavailable
from .settings import settings

LOGIN_URL = "https://kite.zerodha.com/connect/login"
API_URL = "https://api.kite.trade"
_TIMEOUT = 15
_IST = timezone(timedelta(hours=5, minutes=30))
# Kite sessions end at 06:00 IST the next morning, whenever they started.
_SESSION_ENDS = time(6, 0)

_SYMBOL = re.compile(r"^[A-Z0-9^][A-Z0-9.=&^-]{0,23}$")
# NSE series suffixes Kite appends to some cash symbols; Yahoo drops them.
_SERIES_SUFFIXES = ("-BE", "-BZ", "-BL", "-SM", "-ST")
_YAHOO_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}


class TokenExpired(AppError):
    """The broker no longer accepts the stored access token."""
    status, code = 409, "needs_reconnect"


def _require_configured() -> None:
    if not settings.zerodha_configured:
        raise Unavailable("Zerodha isn't set up on this server yet.")


def login_url(state: str) -> str:
    _require_configured()
    # redirect_params come back verbatim on the redirect to our callback.
    return f"{LOGIN_URL}?" + urlencode({"v": 3, "api_key": settings.zerodha_api_key,
                                        "redirect_params": urlencode({"state": state})})


def checksum(request_token: str) -> str:
    raw = settings.zerodha_api_key + request_token + settings.zerodha_api_secret
    return hashlib.sha256(raw.encode()).hexdigest()


def session_expiry(now: datetime | None = None) -> datetime:
    now = (now or datetime.now(timezone.utc)).astimezone(_IST)
    ends = datetime.combine(now.date(), _SESSION_ENDS, tzinfo=_IST)
    if now >= ends:
        ends += timedelta(days=1)
    return ends.astimezone(timezone.utc)


def _call(method: str, path: str, *, access_token: str | None = None, **kw) -> dict:
    headers = {"X-Kite-Version": "3"}
    if access_token:
        headers["Authorization"] = f"token {settings.zerodha_api_key}:{access_token}"
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
            raise AppError("Zerodha rejected the request. Start the connection again.")
        raise Unavailable("Zerodha couldn't complete the request. Try again in a moment.")
    return body.get("data") or {}


def create_session(request_token: str) -> dict:
    """Exchange the one-time request token for an access token."""
    _require_configured()
    data = _call("POST", "/session/token", data={
        "api_key": settings.zerodha_api_key, "request_token": request_token,
        "checksum": checksum(request_token)})
    if not data.get("access_token") or not data.get("user_id"):
        raise Unavailable("Zerodha couldn't complete the request. Try again in a moment.")
    return {"access_token": data["access_token"], "user_id": str(data["user_id"])[:64],
            "user_name": (data.get("user_name") or "")[:120] or None}


def logout(access_token: str) -> None:
    """Best effort: the session expires by itself at 06:00 IST anyway."""
    try:
        _call("DELETE", "/session/token", access_token=access_token,
              params={"api_key": settings.zerodha_api_key, "access_token": access_token})
    except AppError:
        pass


def yahoo_symbol(tradingsymbol: str, exchange: str) -> str | None:
    suffix = _YAHOO_SUFFIX.get(exchange)
    if not suffix or not tradingsymbol:
        return None
    base = tradingsymbol.upper()
    for series in _SERIES_SUFFIXES:
        if base.endswith(series):
            base = base[: -len(series)]
            break
    symbol = base + suffix
    return symbol if _SYMBOL.match(symbol) else None


def _dec(value) -> Decimal | None:
    try:
        d = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return d if d.is_finite() else None


def holdings(access_token: str) -> tuple[list[dict], int]:
    """Equity holdings as rows for the holdings table, plus how many were skipped
    (exchanges other than NSE/BSE, or symbols that don't map)."""
    rows, skipped = [], 0
    for h in _call("GET", "/portfolio/holdings", access_token=access_token) or []:
        exchange = h.get("exchange")
        symbol = yahoo_symbol(h.get("tradingsymbol") or "", exchange)
        qty = (_dec(h.get("quantity")) or 0) + (_dec(h.get("t1_quantity")) or 0)
        avg = _dec(h.get("average_price"))
        if symbol is None or avg is None or avg < 0 or qty <= 0:
            skipped += 1
            continue
        isin = h.get("isin") if isinstance(h.get("isin"), str) and len(h["isin"]) == 12 else None
        rows.append({"symbol": symbol, "broker_symbol": h["tradingsymbol"][:40], "exchange": exchange,
                     "isin": isin, "quantity": qty, "avg_price": avg,
                     "last_price": _dec(h.get("last_price")), "close_price": _dec(h.get("close_price")),
                     "currency": "INR"})
    # Two series of one company (e.g. -BE and EQ) map to one Yahoo symbol.
    merged: dict[str, dict] = {}
    for r in rows:
        if r["symbol"] in merged:
            m = merged[r["symbol"]]
            total = m["quantity"] + r["quantity"]
            m["avg_price"] = (m["avg_price"] * m["quantity"] + r["avg_price"] * r["quantity"]) / total
            m["quantity"] = total
        else:
            merged[r["symbol"]] = r
    return list(merged.values()), skipped
