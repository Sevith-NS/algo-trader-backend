"""Pieces every read-only broker adapter shares: the session shape, expiry
helpers, symbol mapping and row builders.

An adapter module exposes NAME, LABEL, CURRENCY, AUTH, FIELDS, HELP,
HELP_URL, create_session(...) -> Session, holdings(token, app),
positions(token, app) and logout(token, broker_user_id, app). None of them
can place an order. Nothing is configured on the server: every key comes
from the user.

AUTH is how the user connects:
  - "oauth": the broker's own login page, through a developer app the user
    registered at the broker with Polaris's callback as its Redirect URL.
    FIELDS are that app's keys (`app`). The adapter also has
    login_url(state, app, redirect_uri), and create_session(params, app,
    redirect_uri) takes the one-time values from the callback.
  - "keys": the user pastes a read-only token or API key they created at the
    broker. FIELDS name them, create_session(params) checks them with the
    broker, and `app` is empty.
"""
import base64
import json
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation

from ..errors import AppError

IST = timezone(timedelta(hours=5, minutes=30))

_SYMBOL = re.compile(r"^[A-Z0-9^][A-Z0-9.=&^-]{0,23}$")
# NSE series suffixes brokers append to some cash symbols; Yahoo drops them.
_SERIES_SUFFIXES = ("-EQ", "-BE", "-BZ", "-BL", "-SM", "-ST")
_YAHOO_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}


class TokenExpired(AppError):
    """The broker no longer accepts the stored access token."""
    status, code = 409, "needs_reconnect"


@dataclass(frozen=True)
class Session:
    access_token: str
    user_id: str
    user_name: str | None
    expires_at: datetime


def next_ist(at: time, now: datetime | None = None) -> datetime:
    """The next time the IST wall clock reads `at`, in UTC."""
    now = (now or datetime.now(timezone.utc)).astimezone(IST)
    ends = datetime.combine(now.date(), at, tzinfo=IST)
    if now >= ends:
        ends += timedelta(days=1)
    return ends.astimezone(timezone.utc)


def jwt_expiry(token: str, fallback: datetime) -> datetime:
    """The `exp` claim of a broker JWT, read without verifying it. Only used to
    know when to ask for a reconnect; the broker still checks the token itself."""
    try:
        payload = token.split(".")[1]
        exp = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["exp"]
        return datetime.fromtimestamp(int(exp), timezone.utc)
    except (IndexError, KeyError, TypeError, ValueError):
        return fallback


def param(params: dict, name: str, label: str) -> str:
    value = params.get(name)
    if not value:
        raise AppError(f"{label} didn't send back a login. Start the connection again.")
    return value


def app_key(app: dict, name: str, label: str) -> str:
    """One of the user's own broker-app keys, kept with the connection."""
    value = app.get(name)
    if not value:
        raise AppError(f"Your {label} app keys are missing. Connect {label} again.")
    return value


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


def dec(value) -> Decimal | None:
    try:
        d = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return d if d.is_finite() else None


def holding(symbol, broker_symbol, exchange, isin, quantity, avg_price, last_price, close_price,
            currency: str = "INR") -> dict:
    """`avg_price` is None where the broker doesn't know the cost (crypto balances)."""
    isin = isin if isinstance(isin, str) and len(isin) == 12 else None
    return {"symbol": symbol, "broker_symbol": broker_symbol[:40], "exchange": exchange, "isin": isin,
            "quantity": quantity, "avg_price": avg_price, "last_price": last_price, "close_price": close_price,
            "currency": currency}


def position(symbol, broker_symbol, exchange, product, quantity, avg_price, last_price, close_price) -> dict:
    return {"symbol": symbol, "broker_symbol": broker_symbol[:40], "exchange": exchange,
            "product": (product or "NA")[:20], "quantity": quantity, "avg_price": avg_price,
            "last_price": last_price, "close_price": close_price, "currency": "INR"}


def merge(rows: list[dict], *key: str) -> list[dict]:
    """Two series of one company (e.g. -BE and EQ) map to one Yahoo symbol:
    add the quantities and weight the average price."""
    merged: dict[tuple, dict] = {}
    for r in rows:
        k = tuple(r[f] for f in key)
        if k not in merged:
            merged[k] = r
            continue
        m = merged[k]
        total = m["quantity"] + r["quantity"]
        if total:
            m["avg_price"] = (m["avg_price"] * m["quantity"] + r["avg_price"] * r["quantity"]) / total
        m["quantity"] = total
    return list(merged.values())
